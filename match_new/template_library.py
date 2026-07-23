"""在线模板库协调器：持久化、LRU、学习准入和固定容量替换。"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from match_new.template_learning import (
    build_learning_evidence,
    common_area_metrics,
    evaluate_learning_decision,
    template_content_hash,
)
from match_new.template_ranking import insert_template_first, ordered_template_paths, touch_template
from match_new.template_replacement import remove_entry, replacement_required, select_lru_victim
from match_new.utils import ensure_dir, safe_id


TemplateLoader = Callable[[str], dict[str, Any]]
TemplateMatcher = Callable[..., dict[str, Any]]
LOGGER = logging.getLogger(__name__)


class TemplateLibraryManager:
    """管理每个身份的有序活动模板列表。

    初始注册模板被标记为 seed，其中配置数量内的 seed 受保护。当前确认方案中
    初始模板数和保护数均为 20，因此模板库达到 40 张后只会替换 learned 模板。
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        identities: list[dict[str, Any]],
        output_root: str | Path,
        config: dict[str, Any],
    ) -> None:
        self.config = dict(config)
        self.output_root = Path(output_root).expanduser()
        self.max_active_templates = int(self.config.get("max_active_templates", 40))
        self.protected_seed_templates = int(self.config.get("protected_seed_templates", 20))
        self.persist_replace_retries = max(1, int(self.config.get("persist_replace_retries", 20)))
        self.persist_retry_delay_seconds = max(
            0.0,
            float(self.config.get("persist_retry_delay_ms", 100.0)) / 1000.0,
        )
        self.persist_strict = bool(self.config.get("persist_strict", False))
        if self.max_active_templates <= 0:
            raise ValueError("template_management.max_active_templates must be positive")
        if self.protected_seed_templates < 0 or self.protected_seed_templates > self.max_active_templates:
            raise ValueError(
                "template_management.protected_seed_templates must be between 0 and max_active_templates"
            )

        self.library_path = self.output_root / str(self.config.get("library_filename", "template_library.json"))
        self.learned_root = ensure_dir(
            self.output_root / str(self.config.get("learned_templates_dir", "learned_templates"))
        )
        self.retired_root = ensure_dir(
            self.output_root / str(self.config.get("retired_templates_dir", "retired_templates"))
        )
        reset = bool(self.config.get("reset_library_on_start", True))
        if self.library_path.exists() and not reset:
            self.state = self._load_state()
        else:
            self.state = self._build_seed_state(identities)
            self.persist()
        self.sync_identities(identities)

    def _build_seed_state(self, identities: list[dict[str, Any]]) -> dict[str, Any]:
        """从静态 identity_templates JSON 初始化动态模板库。"""

        records: list[dict[str, Any]] = []
        for identity in identities:
            paths = [str(path) for path in identity.get("template_paths", [])]
            image_ids = [str(value) for value in identity.get("template_image_ids", [])]
            entries: list[dict[str, Any]] = []
            for index, path in enumerate(paths):
                image_id = image_ids[index] if index < len(image_ids) else Path(path).stem
                entries.append(
                    {
                        "template_path": path,
                        "image_id": image_id,
                        "source": "seed",
                        "protected": index < self.protected_seed_templates,
                        "created_order": 0,
                        "last_used_order": 0,
                        "successful_match_count": 0,
                        "content_hash": "",
                        "initial_order": index,
                    }
                )
            records.append({"identity_id": str(identity.get("identity_id", "")), "templates": entries})
        return {
            "template_library_format_version": self.FORMAT_VERSION,
            "update_counter": 0,
            "max_active_templates": self.max_active_templates,
            "protected_seed_templates": self.protected_seed_templates,
            "identities": records,
        }

    def _load_state(self) -> dict[str, Any]:
        """读取最新的完整索引；必要时从上次失败留下的文件恢复。"""

        candidates = [
            self.library_path,
            self.library_path.with_suffix(self.library_path.suffix + ".pending"),
            self.library_path.with_suffix(self.library_path.suffix + ".tmp"),
        ]
        candidates.extend(self.library_path.parent.glob(f".{self.library_path.name}.*.tmp"))
        existing = sorted(
            {path for path in candidates if path.exists()},
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        errors: list[str] = []
        for candidate in existing:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("identities"), list):
                errors.append(f"{candidate}: invalid structure")
                continue
            if candidate != self.library_path:
                LOGGER.warning("从模板库恢复文件加载最新状态: %s", candidate)
            return payload
        detail = "; ".join(errors) if errors else "no state file found"
        raise ValueError(f"invalid template library: {self.library_path} ({detail})")

    def _identity_record(self, identity_id: str) -> dict[str, Any]:
        target = str(identity_id)
        for record in self.state.get("identities", []):
            if str(record.get("identity_id", "")) == target:
                return record
        record = {"identity_id": target, "templates": []}
        self.state.setdefault("identities", []).append(record)
        return record

    def _next_order(self) -> int:
        value = int(self.state.get("update_counter", 0)) + 1
        self.state["update_counter"] = value
        return value

    def _replace_with_retry(
        self,
        source: Path,
        target: Path,
        *,
        attempts: int | None = None,
    ) -> tuple[bool, PermissionError | None]:
        """处理 Windows 杀毒、索引器等造成的短暂文件共享冲突。"""

        total_attempts = max(1, attempts or self.persist_replace_retries)
        last_error: PermissionError | None = None
        for attempt in range(total_attempts):
            try:
                os.replace(source, target)
                return True, None
            except PermissionError as exc:
                last_error = exc
                if attempt + 1 >= total_attempts:
                    break
                multiplier = min(2**attempt, 5)
                time.sleep(self.persist_retry_delay_seconds * multiplier)
        return False, last_error

    def persist(self) -> bool:
        """原子更新索引；目标被长期占用时保留可恢复的 pending 状态。"""

        ensure_dir(self.library_path.parent)
        temporary = self.library_path.parent / (
            f".{self.library_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        pending = self.library_path.with_suffix(self.library_path.suffix + ".pending")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")

        replaced, error = self._replace_with_retry(temporary, self.library_path)
        if replaced:
            try:
                pending.unlink(missing_ok=True)
            except OSError:
                pass
            return True

        if self.persist_strict:
            temporary.unlink(missing_ok=True)
            assert error is not None
            raise error

        pending_saved, pending_error = self._replace_with_retry(temporary, pending, attempts=3)
        recovery_path = pending if pending_saved else temporary
        LOGGER.warning(
            "模板库索引暂时无法替换，最新状态已保留在 %s；后续查询将继续重试。错误: %s",
            recovery_path,
            error or pending_error,
        )
        return pending_saved

    def sync_identity(self, identity: dict[str, Any]) -> None:
        """把动态LRU顺序同步到现有identity字典，兼容原匹配入口。"""

        entries = self._identity_record(str(identity.get("identity_id", ""))).get("templates", [])
        identity["template_paths"] = ordered_template_paths(entries)
        identity["template_image_ids"] = [str(entry.get("image_id", "")) for entry in entries]
        identity["num_templates"] = len(entries)

    def sync_identities(self, identities: list[dict[str, Any]]) -> None:
        for identity in identities:
            self.sync_identity(identity)

    def ordered_entries(self, identity_id: str) -> list[dict[str, Any]]:
        return self._identity_record(identity_id).setdefault("templates", [])

    def _copy_learned_template(self, query_path: Path, identity_id: str, image_id: str) -> Path:
        identity_dir = ensure_dir(self.learned_root / safe_id(identity_id))
        order = self._next_order()
        filename = f"learned_{order:08d}__{safe_id(image_id) or 'query'}.npz"
        target = identity_dir / filename
        duplicate_index = 1
        while target.exists():
            target = identity_dir / f"{Path(filename).stem}__{duplicate_index:03d}.npz"
            duplicate_index += 1
        temporary = target.parent / f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
        shutil.copy2(query_path, temporary)
        replaced, error = self._replace_with_retry(temporary, target)
        if not replaced:
            temporary.unlink(missing_ok=True)
            assert error is not None
            raise error
        return target

    def _archive_replaced_template(self, victim: dict[str, Any] | None, identity_id: str) -> str:
        if not victim or str(victim.get("source", "")) != "learned":
            return ""
        source = Path(str(victim.get("template_path", "")))
        if not source.exists():
            return ""
        target_dir = ensure_dir(self.retired_root / safe_id(identity_id))
        target = target_dir / source.name
        duplicate_index = 1
        while target.exists():
            target = target_dir / f"{duplicate_index:03d}__{source.name}"
            duplicate_index += 1
        shutil.move(str(source), str(target))
        return str(target)

    def _unlock_threshold(self, full_config: dict[str, Any]) -> float:
        identification = dict(full_config.get("identification", {}))
        value = identification.get("early_stop_threshold")
        if value in {"", None}:
            value = identification.get("match_score_threshold", 0.55)
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"unlock threshold must be in [0,1], got {threshold}")
        return threshold

    def process_query(
        self,
        query_template: dict[str, Any],
        identity: dict[str, Any],
        full_config: dict[str, Any],
        *,
        template_loader: TemplateLoader,
        matcher: TemplateMatcher,
        descriptor_source: str = "hardnet",
    ) -> dict[str, Any]:
        """按LRU执行一次线上匹配，并在成功后尝试学习query模板。"""

        identity_id = str(identity.get("identity_id", ""))
        entries = self.ordered_entries(identity_id)
        event: dict[str, Any] = {
            "identity_id": identity_id,
            "query_image_id": str(query_template.get("image_id", "")),
            "accepted": False,
            "matched_template_path": "",
            "num_unlock_templates_evaluated": 0,
            "unlock_score": 0.0,
            "learned": False,
            "learned_template_path": "",
            "replaced_template_path": "",
            "retired_template_path": "",
            "learning_reasons": ["unlock_failed"],
        }
        if not entries:
            return event

        started = time.perf_counter()
        unlock_threshold = self._unlock_threshold(full_config)
        attempted_results: dict[str, dict[str, Any]] = {}
        loaded_templates: dict[str, dict[str, Any]] = {}
        matched_path = ""
        for entry in list(entries):
            path = str(entry["template_path"])
            gallery = template_loader(path)
            loaded_templates[path] = gallery
            result = matcher(
                query_template,
                gallery,
                full_config,
                descriptor_source=descriptor_source,
            )
            attempted_results[path] = result
            event["num_unlock_templates_evaluated"] += 1
            score = float(result.get("score", 0.0))
            if score >= unlock_threshold:
                matched_path = path
                event["accepted"] = True
                event["unlock_score"] = score
                event["matched_template_path"] = path
                break
        event["unlock_match_ms"] = (time.perf_counter() - started) * 1000.0
        if not matched_path:
            return event

        # LRU和本次学习结果在返回前合并持久化，避免一次查询连续替换索引两次。
        touch_template(entries, matched_path, self._next_order())
        self.sync_identity(identity)

        learning_started = time.perf_counter()
        query_hash = template_content_hash(query_template)
        duplicate_content = False
        evidences: list[dict[str, Any]] = []
        for entry in list(entries):
            path = str(entry["template_path"])
            gallery = loaded_templates.get(path)
            if gallery is None:
                gallery = template_loader(path)
                loaded_templates[path] = gallery
            content_hash = str(entry.get("content_hash", ""))
            if not content_hash:
                content_hash = template_content_hash(gallery)
                entry["content_hash"] = content_hash
            duplicate_content = duplicate_content or content_hash == query_hash
            result = attempted_results.get(path)
            if result is None:
                result = matcher(
                    query_template,
                    gallery,
                    full_config,
                    descriptor_source=descriptor_source,
                )
                attempted_results[path] = result
            common = common_area_metrics(
                query_template.get("overlap_image"),
                gallery.get("overlap_image"),
                result.get("affine_matrix"),
                self.config,
            )
            evidences.append(build_learning_evidence(entry, result, common, self.config))

        decision = evaluate_learning_decision(
            evidences,
            duplicate_content=duplicate_content,
            config=self.config,
        )
        event.update(
            {
                "learning_reasons": decision["reasons"],
                "num_confirmations": decision["num_confirmations"],
                "num_seed_confirmations": decision["num_seed_confirmations"],
                "num_strict_matches": decision["num_strict_matches"],
                "learning_best_score": decision["best_score"],
                "learning_best_common_area_ratio": decision["best_common_area_ratio"],
                "learning_evidences": evidences,
            }
        )
        if not bool(decision["accepted"]):
            self.persist()  # 同时保存首次计算得到的seed内容哈希。
            event["learning_match_ms"] = (time.perf_counter() - learning_started) * 1000.0
            return event

        query_path = Path(str(query_template.get("template_path", "")))
        if not query_path.exists():
            event["learning_reasons"] = ["query_template_path_missing"]
            self.persist()
            event["learning_match_ms"] = (time.perf_counter() - learning_started) * 1000.0
            return event

        victim: dict[str, Any] | None = None
        if replacement_required(entries, self.max_active_templates):
            victim = select_lru_victim(entries)
            if victim is None:
                event["learning_reasons"] = ["template_library_full_and_all_templates_protected"]
                self.persist()
                event["learning_match_ms"] = (time.perf_counter() - learning_started) * 1000.0
                return event

        learned_path = self._copy_learned_template(
            query_path,
            identity_id,
            str(query_template.get("image_id", query_path.stem)),
        )
        if victim is not None:
            removed = remove_entry(entries, str(victim["template_path"]))
            event["replaced_template_path"] = str(removed.get("template_path", "")) if removed else ""
        entry = {
            "template_path": str(learned_path),
            "image_id": str(query_template.get("image_id", learned_path.stem)),
            "source": "learned",
            "protected": False,
            "created_order": int(self.state.get("update_counter", 0)),
            "last_used_order": 0,
            "successful_match_count": 0,
            "content_hash": query_hash,
        }
        insert_template_first(entries, entry, self._next_order())
        self.sync_identity(identity)
        self.persist()
        try:
            event["retired_template_path"] = self._archive_replaced_template(victim, identity_id)
        except OSError as exc:
            # 索引已安全切换到新模板，归档失败只记录诊断，不回滚已完成的替换。
            event["retirement_error"] = str(exc)
        event["learned"] = True
        event["learned_template_path"] = str(learned_path)
        event["learning_reasons"] = ["accepted"]
        event["num_templates_after_update"] = len(entries)
        event["learning_match_ms"] = (time.perf_counter() - learning_started) * 1000.0
        return event

    def summarize_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """生成模板学习、替换和在线LRU匹配摘要。"""

        accepted = [event for event in events if bool(event.get("accepted", False))]
        learned = [event for event in events if bool(event.get("learned", False))]
        replaced = [event for event in events if bool(event.get("replaced_template_path", ""))]
        return {
            "enabled": True,
            "library_path": str(self.library_path),
            "max_active_templates": self.max_active_templates,
            "protected_seed_templates": self.protected_seed_templates,
            "num_queries_processed": len(events),
            "num_unlock_accepts": len(accepted),
            "num_templates_learned": len(learned),
            "num_templates_replaced": len(replaced),
            "update_counter": int(self.state.get("update_counter", 0)),
        }
