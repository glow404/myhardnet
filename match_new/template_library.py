"""离线评估中的动态模板库管理。

模板学习只更新当前运行的 identity_templates JSON；每次主流程启动时，
run_hardnet_matching.py 会先重新生成该 JSON，因此默认从初始注册模板重置。
"""

from __future__ import annotations

import json
import os
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


TemplateLoader = Callable[[str], dict[str, Any]]
PERSIST_REPLACE_RETRIES = 5
PERSIST_RETRY_DELAY_SECONDS = 0.01


class TemplateLibraryManager:
    """管理每个身份的活动模板、学习准入和固定容量替换。"""

    FORMAT_VERSION = 1

    def __init__(
        self,
        identities: list[dict[str, Any]],
        identity_templates_path: str | Path,
        config: dict[str, Any],
        *,
        initial_template_count: int,
    ) -> None:
        self.config = dict(config)
        self.library_path = Path(identity_templates_path).expanduser().resolve()
        self.max_active_templates = int(self.config.get("max_active_templates", 40))
        self.initial_template_count = int(initial_template_count)
        if self.max_active_templates <= 0:
            raise ValueError("template_management.max_active_templates must be positive")
        if self.initial_template_count < 0:
            raise ValueError("enrollment_images_per_identity must be non-negative")
        if self.initial_template_count > self.max_active_templates:
            raise ValueError(
                "enrollment_images_per_identity must not exceed "
                "template_management.max_active_templates"
            )
        self.persist_replace_retries = PERSIST_REPLACE_RETRIES
        self.persist_retry_delay_seconds = PERSIST_RETRY_DELAY_SECONDS
        if not self.library_path.exists():
            raise FileNotFoundError(f"初始模板库索引不存在: {self.library_path}")

        payload = json.loads(self.library_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("identities"), list):
            raise ValueError(
                f"identity template index has an unsupported format: {self.library_path}"
            )
        self.identity_payload = {
            key: value for key, value in payload.items() if key != "identities"
        }
        self.state = self._build_seed_state(identities)
        self.sync_identities(identities)
        # 初始化时也原子重写一次，确保 JSON 带有动态模板字段，但不继承旧状态。
        self.persist()

    def _build_seed_state(self, identities: list[dict[str, Any]]) -> dict[str, Any]:
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
                        "protected": index < self.initial_template_count,
                        "created_order": 0,
                        "last_used_order": 0,
                        "successful_match_count": 0,
                        "content_hash": "",
                        "initial_order": index,
                    }
                )
            records.append(
                {
                    "identity_id": str(identity.get("identity_id", "")),
                    "templates": entries,
                }
            )
        return {
            "template_library_format_version": self.FORMAT_VERSION,
            "update_counter": 0,
            "max_active_templates": self.max_active_templates,
            "initial_template_count": self.initial_template_count,
            "identities": records,
        }

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
    ) -> None:
        last_error: PermissionError | None = None
        for attempt in range(self.persist_replace_retries):
            try:
                os.replace(source, target)
                return
            except PermissionError as exc:
                last_error = exc
                if attempt + 1 >= self.persist_replace_retries:
                    break
                time.sleep(self.persist_retry_delay_seconds * min(2**attempt, 5))
        assert last_error is not None
        raise last_error

    def persist(self) -> bool:
        """将当前活动模板原子写回唯一的 identity_templates JSON。"""

        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        identities: list[dict[str, Any]] = []
        for record in self.state.get("identities", []):
            entries = list(record.get("templates", []))
            identities.append(
                {
                    "identity_id": str(record.get("identity_id", "")),
                    "template_paths": [str(entry["template_path"]) for entry in entries],
                    "template_image_ids": [
                        str(entry.get("image_id", "")) for entry in entries
                    ],
                    "num_templates": len(entries),
                    "template_entries": entries,
                }
            )
        payload = {
            **self.identity_payload,
            "template_library_format_version": self.FORMAT_VERSION,
            "template_library_update_counter": int(self.state.get("update_counter", 0)),
            "template_library_max_active_templates": self.max_active_templates,
            "template_library_initial_template_count": self.initial_template_count,
            "identities": identities,
        }
        temporary = self.library_path.parent / (
            f".{self.library_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            self._replace_with_retry(temporary, self.library_path)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def sync_identity(self, identity: dict[str, Any]) -> None:
        entries = self._identity_record(str(identity.get("identity_id", ""))).get(
            "templates", []
        )
        identity["template_paths"] = ordered_template_paths(entries)
        identity["template_image_ids"] = [
            str(entry.get("image_id", "")) for entry in entries
        ]
        identity["num_templates"] = len(entries)

    def sync_identities(self, identities: list[dict[str, Any]]) -> None:
        for identity in identities:
            self.sync_identity(identity)

    def ordered_entries(self, identity_id: str) -> list[dict[str, Any]]:
        return self._identity_record(identity_id).setdefault("templates", [])

    def _use_existing_query_template(self, query_path: Path) -> Path:
        if not query_path.exists():
            raise FileNotFoundError(f"query 模板文件不存在: {query_path}")
        return query_path

    def learn_after_match(
        self,
        query_template: dict[str, Any],
        identity: dict[str, Any],
        full_config: dict[str, Any],
        match_result: dict[str, Any],
        *,
        template_loader: TemplateLoader,
        descriptor_source: str = "hardnet",
    ) -> dict[str, Any]:
        """用已完成的本人匹配结果执行学习准入和模板提交。"""

        del descriptor_source
        identity_id = str(identity.get("identity_id", ""))
        entries = self.ordered_entries(identity_id)
        score = float(match_result.get("score", 0.0))
        unlock_threshold = float(dict(full_config.get("identification", {})).get("match_score_threshold", 0.55))
        if not 0.0 <= unlock_threshold <= 1.0:
            raise ValueError(f"match score threshold must be in [0,1], got {unlock_threshold}")
        event: dict[str, Any] = {
            "identity_id": identity_id,
            "query_image_id": str(query_template.get("image_id", "")),
            "accepted": bool(score >= unlock_threshold),
            "matched_template_path": str(match_result.get("best_template_path", "")),
            "unlock_score": score,
            "learned": False,
            "learned_template_path": "",
            "replaced_template_path": "",
            "learning_reasons": ["unlock_failed"],
        }
        if score < unlock_threshold:
            return event

        matched_path = event["matched_template_path"]
        matched_entry = next(
            (
                entry
                for entry in entries
                if str(entry.get("template_path", "")) == matched_path
            ),
            None,
        )
        if matched_entry is None:
            event["learning_reasons"] = ["matched_template_not_in_library"]
            return event

        touch_template(entries, matched_path, self._next_order())
        self.sync_identity(identity)
        query_hash = template_content_hash(query_template)
        duplicate_content = False
        for entry in list(entries):
            gallery = template_loader(str(entry["template_path"]))
            content_hash = str(entry.get("content_hash", ""))
            if not content_hash:
                content_hash = template_content_hash(gallery)
                entry["content_hash"] = content_hash
            duplicate_content = duplicate_content or content_hash == query_hash

        gallery = template_loader(matched_path)
        common = common_area_metrics(
            query_template.get("overlap_image"),
            gallery.get("overlap_image"),
            match_result.get("best_affine_matrix"),
            self.config,
        )
        evidence = build_learning_evidence(
            matched_entry,
            match_result,
            common,
            self.config,
        )
        decision = evaluate_learning_decision(
            [evidence],
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
                "learning_best_common_area_ratio": decision[
                    "best_common_area_ratio"
                ],
                "learning_common_area_ratio": float(
                    common.get("common_area_ratio", 0.0)
                ),
                "learning_evidence": evidence,
            }
        )
        if not bool(decision["accepted"]):
            self.persist()
            return event

        query_path = Path(str(query_template.get("template_path", "")))
        if not query_path.exists():
            event["learning_reasons"] = ["query_template_path_missing"]
            self.persist()
            return event

        victim: dict[str, Any] | None = None
        if replacement_required(entries, self.max_active_templates):
            victim = select_lru_victim(entries)
            if victim is None:
                event["learning_reasons"] = [
                    "template_library_full_and_all_templates_protected"
                ]
                self.persist()
                return event

        learned_path = self._use_existing_query_template(query_path)
        if victim is not None:
            removed = remove_entry(entries, str(victim["template_path"]))
            event["replaced_template_path"] = (
                str(removed.get("template_path", "")) if removed else ""
            )
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
        event["learned"] = True
        event["learned_template_path"] = str(learned_path)
        event["learning_reasons"] = ["accepted"]
        event["num_templates_after_update"] = len(entries)
        return event

    def summarize_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        accepted = [event for event in events if bool(event.get("accepted", False))]
        learned = [event for event in events if bool(event.get("learned", False))]
        replaced = [
            event for event in events if bool(event.get("replaced_template_path", ""))
        ]
        return {
            "enabled": True,
            "library_path": str(self.library_path),
            "max_active_templates": self.max_active_templates,
            "initial_template_count": self.initial_template_count,
            "num_queries_processed": len(events),
            "num_unlock_accepts": len(accepted),
            "num_templates_learned": len(learned),
            "num_templates_replaced": len(replaced),
            "update_counter": int(self.state.get("update_counter", 0)),
        }
