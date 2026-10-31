"""Small dependency-free helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def safe_module_name(value: Any) -> str:
    """value as a lowercase identifier (also used for robot, atom and file names)."""
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value).strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        return "task"
    return f"task_{name}" if name[0].isdigit() else name


def stable_int(value: str) -> int:
    """A 32-bit integer hash of value that is the same in every process."""
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16)


def default_robot_display_name(robot: str) -> str:
    return robot.replace("_", " ").title()


def json_safe(value: Any) -> Any:
    """value with dataclasses, paths, tuples, sets and numpy scalars turned into JSON types."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value
