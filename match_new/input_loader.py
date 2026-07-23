"""Raw fingerprint image discovery and input validation."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .utils import resolve_path


DEFAULT_IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _normalize_extensions(values: Iterable[str] | None) -> set[str]:
    extensions: set[str] = set()
    for value in values or DEFAULT_IMAGE_EXTENSIONS:
        extension = str(value).strip().lower()
        if not extension:
            continue
        extensions.add(extension if extension.startswith(".") else f".{extension}")
    if not extensions:
        raise ValueError("data.image_extensions must contain at least one extension.")
    return extensions


def _is_readable_image(path: Path) -> bool:
    raw = np.fromfile(str(path), dtype=np.uint8)
    return bool(raw.size and cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) is not None)


def _duplicate_safe_image_ids(
    items: list[tuple[Path, Path, str]],
) -> dict[Path, str]:
    """Keep simple stems where possible and hash only duplicate stems."""

    counts = Counter((identity_id, path.stem) for path, _relative, identity_id in items)
    image_ids: dict[Path, str] = {}
    for path, relative, identity_id in items:
        image_id = path.stem
        if counts[(identity_id, image_id)] > 1:
            digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:10]
            image_id = f"{image_id}__{digest}"
        image_ids[path] = image_id
    return image_ids


def scan_image_metadata(
    image_root: str | Path,
    identity_depth: int,
    image_extensions: Iterable[str] | None = None,
    validate_readable: bool = True,
) -> list[dict[str, str]]:
    """Scan raw images and derive identity IDs from leading directories."""

    root = Path(image_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Raw image directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Raw image path is not a directory: {root}")
    if int(identity_depth) <= 0:
        raise ValueError(f"data.identity_depth must be greater than 0, got {identity_depth}.")

    allowed = _normalize_extensions(image_extensions)
    items: list[tuple[Path, Path, str]] = []
    shallow_paths: list[Path] = []
    bad_paths: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        relative = path.relative_to(root)
        directory_parts = relative.parts[:-1]
        if len(directory_parts) < int(identity_depth):
            shallow_paths.append(relative)
            continue
        if validate_readable and not _is_readable_image(path):
            bad_paths.append(relative)
            continue
        identity_id = "/".join(directory_parts[: int(identity_depth)])
        items.append((path.resolve(), relative, identity_id))

    if shallow_paths:
        examples = ", ".join(path.as_posix() for path in shallow_paths[:5])
        raise ValueError(
            f"{len(shallow_paths)} image(s) do not have {identity_depth} identity directory level(s) "
            f"under {root}. Examples: {examples}"
        )
    if bad_paths:
        examples = ", ".join(path.as_posix() for path in bad_paths[:5])
        raise ValueError(f"{len(bad_paths)} unreadable image(s) found under {root}. Examples: {examples}")
    if not items:
        suffixes = ", ".join(sorted(allowed))
        raise RuntimeError(f"No raw images with extensions [{suffixes}] found under {root}.")

    image_ids = _duplicate_safe_image_ids(items)
    rows = [
        {
            "identity_id": identity_id,
            "image_id": image_ids[path],
            "image_path": str(path),
            "split": "",
        }
        for path, _relative, identity_id in items
    ]
    keys = [(row["identity_id"], row["image_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Raw image indexing produced duplicate (identity_id, image_id) keys.")
    return rows


def load_raw_image_metadata(config: dict[str, Any]) -> list[dict[str, str]]:
    """Load matching input exclusively from data.image_root."""

    data_cfg = dict(config.get("data", {}))
    image_root = data_cfg.get("image_root")
    if not image_root:
        raise ValueError("data.image_root is required.")
    return scan_image_metadata(
        resolve_path(config, image_root),
        identity_depth=int(data_cfg.get("identity_depth", 1)),
        image_extensions=data_cfg.get("image_extensions", DEFAULT_IMAGE_EXTENSIONS),
        validate_readable=bool(data_cfg.get("validate_readable", True)),
    )


def validate_identity_image_counts(
    rows: list[dict[str, str]],
    minimum: int,
    *,
    context: str,
) -> None:
    """Require enough images per identity for enrollment and at least one query."""

    groups: dict[str, int] = defaultdict(int)
    for row in rows:
        groups[str(row["identity_id"])] += 1
    insufficient = [(identity_id, count) for identity_id, count in sorted(groups.items()) if count < int(minimum)]
    if not insufficient:
        return
    examples = ", ".join(f"{identity_id}={count}" for identity_id, count in insufficient[:10])
    raise ValueError(
        f"{len(insufficient)} identity/identities have fewer than {minimum} images in {context}. "
        f"At least {minimum - 1} enrollment image(s) and one query image are required. "
        f"Examples: {examples}"
    )
