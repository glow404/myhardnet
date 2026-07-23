"""固定容量模板库的 LRU 替换策略。"""

from __future__ import annotations

from typing import Any


def select_lru_victim(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从 LRU 末尾选择第一个非保护模板。

    当前设计中初始 seed 全部受保护，所以正常情况下只会返回 learned 模板。
    反向扫描使保护模板即使位于列表尾部也不会被错误删除。
    """

    for entry in reversed(entries):
        if not bool(entry.get("protected", False)):
            return entry
    return None


def replacement_required(entries: list[dict[str, Any]], max_active_templates: int) -> bool:
    """判断新增一张模板前是否必须执行替换。"""

    maximum = int(max_active_templates)
    if maximum <= 0:
        raise ValueError("max_active_templates must be positive")
    return len(entries) >= maximum


def remove_entry(entries: list[dict[str, Any]], template_path: str) -> dict[str, Any] | None:
    """按路径移除模板记录并返回被移除项。"""

    target = str(template_path)
    for index, entry in enumerate(entries):
        if str(entry.get("template_path", "")) == target:
            return entries.pop(index)
    return None
