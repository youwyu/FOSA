"""Declarative contract exposed by the MuJoCo Playground adapter.

Keep backend vocabulary here rather than in the LLM orchestration layer. A
different simulator can provide the same small capability surface without
changing the simulator-neutral scene contract.
"""

from __future__ import annotations

import dataclasses
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


BACKEND_ID = "mujoco_playground_warp_brax_ppo"
CAPABILITY_DOCUMENT_PATH = Path(__file__).with_name("simulator_capabilities.json")


@lru_cache(maxsize=1)
def simulator_capability_document() -> dict[str, Any]:
    """Load the adapter contract shared with simulator-neutral generators."""

    return json.loads(CAPABILITY_DOCUMENT_PATH.read_text(encoding="utf-8"))


_ADAPTER = simulator_capability_document()
_SCENE = _ADAPTER["scene"]
_MODEL_RANDOMIZATION = _ADAPTER["runtime_model_randomization"]

TRAINING_TERRAIN_SIZE_M = (12.8, 12.8)
TRAINING_TERRAIN_SAMPLES = 65
TRAINING_TERRAIN_RESOLUTION_M = round(2.0 * TRAINING_TERRAIN_SIZE_M[0] / (TRAINING_TERRAIN_SAMPLES - 1), 3)

SCENE_PRIMITIVE_SHAPES = tuple(_SCENE["generated_primitives"])
SCENE_OBJECT_MOTIONS = tuple(_SCENE["object_motions"])
MODEL_RANDOMIZATION_LIMITATIONS = tuple(_ADAPTER["limitations"])
MODEL_RANDOMIZATION_ENTITY_AXES = {
    str(kind): dict(value) for kind, value in _MODEL_RANDOMIZATION["entity_axes"].items()
}


def is_nondegenerate_range(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(isinstance(item, (int, float)) for item in value[:2])
        and value[1] > value[0]
    )


@lru_cache(maxsize=1)
def documented_geom_types() -> tuple[str, ...]:
    """Return physical geom types from the installed MuJoCo API."""

    import mujoco

    limit = int(mujoco.mjtGeom.mjNGEOMTYPES)
    return tuple(
        name.removeprefix("mjGEOM_").lower()
        for name, member in mujoco.mjtGeom.__members__.items()
        if name.startswith("mjGEOM_") and int(member) < limit
    )


@lru_cache(maxsize=1)
def model_randomization_operation_fields() -> tuple[str, ...]:
    """Discover batchable physical inputs from the installed MJX model type."""

    from mujoco import mjx

    prefixes = tuple(str(value["field_prefix"]) for value in MODEL_RANDOMIZATION_ENTITY_AXES.values())
    excluded = set(_MODEL_RANDOMIZATION["excluded_fields"])
    return tuple(
        field.name
        for field in dataclasses.fields(mjx.Model)
        if field.name.startswith(prefixes) and field.type.__module__ == "jax" and field.name not in excluded
    )


def model_randomization_semantics(fields: tuple[str, ...]) -> dict[str, str]:
    return {field: field.replace("jnt_", "joint ").replace("dof_", "joint ").replace("_", " ") for field in fields}


def model_randomization_interface(model: Any | None = None) -> dict[str, Any]:
    fields = model_randomization_operation_fields()
    metadata: dict[str, Any] = {}
    if model is not None:
        import numpy as np

        selected: list[str] = []
        for field in fields:
            value = np.asarray(getattr(model, field))
            entity = next(
                (
                    kind
                    for kind, spec in MODEL_RANDOMIZATION_ENTITY_AXES.items()
                    if field.startswith(str(spec["field_prefix"]))
                    and value.ndim > 0
                    and value.shape[0] > 0
                    and value.shape[0] == getattr(model, spec["count"])
                ),
                None,
            )
            if entity is None or not np.issubdtype(value.dtype, np.floating):
                continue
            selected.append(field)
            metadata[field] = {
                "entity": entity,
                "shape_per_entity": list(value.shape[1:]),
                "dtype": str(value.dtype),
                "recommended_mode": "add" if np.allclose(value, 0.0) else "scale",
            }
        fields = tuple(selected)
    return {
        "fields": list(fields),
        "semantics": model_randomization_semantics(fields),
        "field_metadata": metadata,
        "modes": list(_MODEL_RANDOMIZATION["modes"]),
        "selector_kinds": list(MODEL_RANDOMIZATION_ENTITY_AXES),
        "selector_matchers": list(_MODEL_RANDOMIZATION["selector_matchers"]),
        "limitations": list(MODEL_RANDOMIZATION_LIMITATIONS),
    }


RESET_RANDOMIZATION_INTERFACE = {
    "command": {
        "lin_x": "range [min,max] m/s",
        "lin_y": "range [min,max] m/s; omit for differential-drive robots",
        "yaw": "range [min,max] rad/s",
        "standing_probability": "probability in [0,1]",
    },
    "base_xy": "range [min,max] meters applied independently to initial x and y",
    "base_yaw": "range [min,max] radians",
    "base_velocity_noise": "uniform bound in m/s (and rad/s) applied to the initial base velocity",
    "task_object_pose": {
        "x_offset": "range [min,max] meters relative to robot base",
        "y_offset": "range [min,max] meters relative to robot base",
        "z": "absolute world z range [min,max] meters",
        "yaw": "range [min,max] radians",
    },
    "target": {
        "x_offset": "range [min,max] meters relative to robot base",
        "y_offset": "range [min,max] meters relative to robot base",
    },
    "manipulated_object_pose": {
        "position": {
            "x": "offset range [min,max] meters from the authored object pose",
            "y": "offset range [min,max] meters from the authored object pose",
            "z": "offset range [min,max] meters from the authored object pose",
        },
        "orientation_uniform": "1 samples a uniform SO(3) orientation; 0 keeps the scene-authored orientation",
        "hand_pose_noise": "standard deviation in radians",
    },
    "joint_pose_noise": "standard deviation in radians",
}

BACKEND_RESET_RANDOMIZATION_KEYS = frozenset(RESET_RANDOMIZATION_INTERFACE)


def sanitize_reset_randomization(
    value: object, allowed: object = BACKEND_RESET_RANDOMIZATION_KEYS
) -> dict[str, object]:
    """Keep only reset fields implemented by this adapter."""

    return {
        key: cleaned
        for key, item in value.items()
        if key in allowed
        and (cleaned := _sanitize_reset_value(item, RESET_RANDOMIZATION_INTERFACE[key])) not in ({}, [])
    }


def _sanitize_reset_value(value: object, schema: object) -> object:
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            return {}
        return {
            key: cleaned
            for key, item in value.items()
            if key in schema and (cleaned := _sanitize_reset_value(item, schema[key])) not in ({}, [])
        }
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            lo, hi = sorted((float(value[0]), float(value[1])))
        except (TypeError, ValueError):
            return {}
        return [lo, hi] if hi > lo else {}
    return float(value) if isinstance(value, (int, float)) else {}


def semantic_randomization_requirements(contract: dict[str, object]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Map scene semantics to randomization this backend can execute."""

    scene = contract["scene_requirements"]
    reset = list(contract["required_reset_randomization"])
    model = list(contract["required_model_randomization"])
    if contract["execution"] == "velocity" and scene["terrain"]["must_be_nonflat"]:
        reset.append("base_xy")
    if scene["task_objects"]["variant_randomization_required"]:
        model.append("geom_size")
    return tuple(dict.fromkeys(reset)), tuple(dict.fromkeys(model))


def canonical_reset_randomization_key(value: object) -> str | None:
    """Resolve a top-level reset domain from a nested capability path.

    Contracts require top-level reset sections, while an LLM may naturally
    describe a supported nested field such as
    ``manipulated_object_orientation_uniform``. Both forms resolve to the
    backend-owned ``manipulated_object_pose`` section.
    """

    name = _capability_name(value)
    prefix = "reset_randomization_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name in BACKEND_RESET_RANDOMIZATION_KEYS:
        return name

    matches: list[str] = []
    for root, schema in RESET_RANDOMIZATION_INTERFACE.items():
        if not isinstance(schema, dict):
            continue
        root_aliases = {root}
        if root.endswith("_pose"):
            root_aliases.add(root.removesuffix("_pose"))
        for path in _nested_capability_paths(schema):
            suffix = "_".join(path)
            if any(_capability_path_matches(name, f"{alias}_{suffix}") for alias in root_aliases):
                matches.append(root)
                break
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def canonical_model_randomization_field(value: object) -> str | None:
    """Extract one supported model field from a name or serialized operation."""

    name = _capability_name(value)
    fields = model_randomization_operation_fields()
    if name in fields:
        return name
    matches = [field for field in fields if re.search(rf"(?:^|_){re.escape(field)}(?:_|$)", name)]
    return matches[0] if len(matches) == 1 else None


def _nested_capability_paths(value: dict[str, object], prefix: tuple[str, ...] = ()) -> tuple[tuple[str, ...], ...]:
    paths: list[tuple[str, ...]] = []
    for key, item in value.items():
        path = (*prefix, _capability_name(key))
        paths.append(path)
        if isinstance(item, dict):
            paths.extend(_nested_capability_paths(item, path))
    return tuple(paths)


def _capability_name(value: object) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", name).strip("_")


def _capability_path_matches(name: str, path: str) -> bool:
    """Accept an exact path or the same path annotated with numeric values."""

    if name == path:
        return True
    suffix = name.removeprefix(path + "_") if name.startswith(path + "_") else ""
    return bool(suffix and re.fullmatch(r"\d+(?:_\d+)*", suffix))
