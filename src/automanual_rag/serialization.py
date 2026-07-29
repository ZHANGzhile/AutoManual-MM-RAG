"""Helpers for writing portable project artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def relativize_project_paths(value: Any, project_root: Path) -> Any:
    """Replace absolute paths inside a JSON-ready value with project paths."""
    if isinstance(value, dict):
        return {
            key: relativize_project_paths(item, project_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            relativize_project_paths(item, project_root)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            relativize_project_paths(item, project_root)
            for item in value
        ]
    if not isinstance(value, str):
        return value

    root = project_root.resolve().as_posix().rstrip("/")
    normalized = value.replace("\\", "/")
    if normalized == root:
        return "."
    prefix = root + "/"
    if normalized.startswith(prefix):
        return normalized[len(prefix):]
    return value
