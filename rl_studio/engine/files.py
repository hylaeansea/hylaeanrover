"""File-tree API over configured roots (models/, runs/, ...).

Directories containing a `model.zip` or `model.onnx` are tagged as
bundles so the frontend can render them as draggable model artifacts.
"""

from __future__ import annotations

import os
from typing import Any

IGNORED = {".DS_Store", "__pycache__", "tb"}
MAX_DEPTH = 6


def _entry(root_label: str, abs_path: str, rel: str, depth: int) -> dict[str, Any]:
    name = os.path.basename(abs_path) or abs_path
    node: dict[str, Any] = {
        "name": name,
        # Path is absolute so launch requests are unambiguous regardless
        # of which cwd each tool runs from.
        "path": abs_path,
        "rel": f"{root_label}/{rel}" if rel else root_label,
    }
    if os.path.isdir(abs_path):
        node["type"] = "dir"
        entries = sorted(os.listdir(abs_path)) if depth < MAX_DEPTH else []
        files = set(entries)
        node["bundle"] = bool({"model.zip", "model.onnx"} & files)
        node["has_onnx"] = "model.onnx" in files
        node["has_zip"] = "model.zip" in files
        node["has_vecnorm"] = "vecnorm.pkl" in files
        node["children"] = [
            _entry(root_label, os.path.join(abs_path, e),
                   f"{rel}/{e}" if rel else e, depth + 1)
            for e in entries
            if e not in IGNORED
        ]
    else:
        node["type"] = "file"
        try:
            node["size"] = os.path.getsize(abs_path)
        except OSError:
            node["size"] = None
    return node


def tree(roots: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """roots: [(label, abs_path)] -> list of tree nodes (missing roots skipped)."""
    return [
        _entry(label, path, "", 0)
        for label, path in roots
        if os.path.exists(path)
    ]
