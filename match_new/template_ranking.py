"""动态模板库的 LRU 排序操作。

列表第一个元素始终是最近成功匹配或刚学习得到的模板。失败的匹配不调用本模块，
因此不会改变顺序。保护标记只影响模板替换，不影响 LRU 排序。
"""

from __future__ import annotations

from typing import Any


def touch_template(entries: list[dict[str, Any]], template_path: str, used_order: int) -> dict[str, Any] | None:
    """把成功命中的模板移动到 LRU 首位，并更新其使用序号。"""

    target = str(template_path)
    for index, entry in enumerate(entries):
        if str(entry.get("template_path", "")) != target:
            continue
        touched = entries.pop(index)
        touched["last_used_order"] = int(used_order)
        touched["successful_match_count"] = int(touched.get("successful_match_count", 0)) + 1
        entries.insert(0, touched)
        return touched
    return None


def insert_template_first(
    entries: list[dict[str, Any]],
    entry: dict[str, Any],
    used_order: int,
) -> dict[str, Any]:
    """把新学习模板插入 LRU 首位；若路径已存在则先移除旧记录。"""

    target = str(entry.get("template_path", ""))
    if not target:
        raise ValueError("template entry must contain template_path")
    entries[:] = [item for item in entries if str(item.get("template_path", "")) != target]
    inserted = dict(entry)
    inserted["last_used_order"] = int(used_order)
    inserted.setdefault("successful_match_count", 0)
    entries.insert(0, inserted)
    return inserted


def ordered_template_paths(entries: list[dict[str, Any]]) -> list[str]:
    """返回当前持久化 LRU 顺序中的模板路径。"""

    return [str(entry["template_path"]) for entry in entries]
