"""Simulator-neutral scene and reset contract.

The LLM describes *what* exists.  A simulator adapter decides how to compile it.
This module is deliberately free of MuJoCo/JAX imports so another backend can
consume the same generated JSON without inheriting MuJoCo conventions.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SHAPES = frozenset({"box", "sphere", "cylinder", "capsule", "ellipsoid"})
MOTIONS = frozenset({"static", "dynamic", "attached", "carried"})
CARRIED_ROLES = frozenset({"payload", "cargo", "carried_object", "carried_payload"})
MANIPULATED_ROLES = frozenset({"manipulated_object", "task_object", "task_owned_object"})


def safe_id(value: Any, default: str = "object") -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or default)).strip("_").lower()
    return text or default


def _floats(value: Any, default: Sequence[float], length: int) -> tuple[float, ...]:
    raw = value if isinstance(value, (list, tuple)) else default
    fallback = list(default) or [0.0]
    parsed: list[float] = []
    for index in range(length):
        fallback_value = float(fallback[min(index, len(fallback) - 1)])
        item = raw[index] if index < len(raw) else fallback_value
        try:
            number = float(item)
        except (TypeError, ValueError):
            number = fallback_value
        parsed.append(number if math.isfinite(number) else fallback_value)
    return tuple(parsed)


def _positive(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed > 0.0 else default


def _range(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lo) or not math.isfinite(hi):
        return None
    return [min(lo, hi), max(lo, hi)]


def normalize_object_physics(value: Any) -> dict[str, Any]:
    """Keep only simulator-neutral object physics with explicit SI units."""

    source = value if isinstance(value, Mapping) else {}
    offset = source.get("center_of_mass_offset_m")
    if not isinstance(offset, (list, tuple)) or len(offset) != 3:
        return {}
    parsed = _floats(offset, (), 3)
    return {"center_of_mass_offset_m": list(parsed)}


def _quaternion(value: Mapping[str, Any], pose: Mapping[str, Any]) -> tuple[float, ...]:
    euler = pose.get("orientation_euler_deg") or value.get("orientation_euler_deg")
    if isinstance(euler, (list, tuple)):
        roll, pitch, yaw = (math.radians(angle) / 2.0 for angle in _floats(euler, (0, 0, 0), 3))
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    quaternion = _floats(
        pose.get("orientation_wxyz") or value.get("orientation_wxyz") or value.get("quat"), (1, 0, 0, 0), 4
    )
    norm = math.sqrt(sum(item * item for item in quaternion))
    return tuple(item / norm for item in quaternion) if norm > 1.0e-8 else (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Primitive:
    """A primitive expressed in ordinary metric dimensions, never half-extents."""

    shape: str
    dimensions_m: tuple[float, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Primitive":
        geometry = value.get("geometry") if isinstance(value.get("geometry"), Mapping) else value
        shape = safe_id(geometry.get("shape") or geometry.get("type") or value.get("collider_shape"), "box")
        shape = shape if shape in SHAPES else "box"

        if shape == "sphere":
            raw_size = geometry.get("size")
            size_radius = raw_size[0] if isinstance(raw_size, (list, tuple)) and raw_size else raw_size
            radius = _positive(geometry.get("radius_m", size_radius), 0.05)
            dimensions = (radius,)
        elif shape in {"cylinder", "capsule"}:
            raw_size = geometry.get("size") if isinstance(geometry.get("size"), (list, tuple)) else ()
            radius = _positive(geometry.get("radius_m", raw_size[0] if raw_size else None), 0.04)
            # A MuJoCo-style size list holds the half-height; canonical JSON stores the full length.
            size_length = 2.0 * _positive(raw_size[1], 0.06) if len(raw_size) > 1 else None
            length = _positive(geometry.get("length_m", size_length), 0.12)
            dimensions = (radius, length)
        else:
            raw = geometry.get("dimensions_m") or geometry.get("size")
            dimensions = tuple(max(1.0e-4, item) for item in _floats(raw, (0.08, 0.08, 0.04), 3))
        return cls(shape=shape, dimensions_m=dimensions)

    def to_dict(self) -> dict[str, Any]:
        if self.shape == "sphere":
            return {"shape": self.shape, "radius_m": self.dimensions_m[0]}
        if self.shape in {"cylinder", "capsule"}:
            return {"shape": self.shape, "radius_m": self.dimensions_m[0], "length_m": self.dimensions_m[1]}
        return {"shape": self.shape, "dimensions_m": list(self.dimensions_m)}

    def half_extents(self) -> tuple[float, float, float]:
        if self.shape == "sphere":
            return (self.dimensions_m[0],) * 3
        if self.shape in {"cylinder", "capsule"}:
            radius, length = self.dimensions_m
            return (radius, radius, 0.5 * length + (radius if self.shape == "capsule" else 0.0))
        return tuple(0.5 * item for item in self.dimensions_m[:3])  # type: ignore[return-value]


@dataclass(frozen=True)
class ScenePart:
    name: str
    primitive: Primitive
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] | None = None
    collidable: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> "ScenePart":
        rgba = _floats(value.get("rgba"), (), 4) if isinstance(value.get("rgba"), (list, tuple)) else None
        return cls(
            name=safe_id(value.get("name"), f"part_{index}"),
            primitive=Primitive.from_mapping(value),
            position_m=_floats(value.get("position_m") or value.get("local_pos") or value.get("pos"), (0, 0, 0), 3),
            rgba=rgba,
            collidable=bool(value.get("collidable", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "geometry": self.primitive.to_dict(),
            "position_m": list(self.position_m),
            "collidable": self.collidable,
        }
        if self.rgba is not None:
            result["rgba"] = list(self.rgba)
        return result


def _attachment(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = value.get("attachment")
    result = copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
    if isinstance(raw, str) and raw.strip():
        result["parent_body"] = raw.strip()
    for key in ("parent_body", "attached_to", "attach_to", "mount_body", "robot_body"):
        if isinstance(value.get(key), str) and str(value[key]).strip():
            result.setdefault("parent_body", str(value[key]).strip())
    for key in ("attached_to", "attach_to", "mount_body", "robot_body"):
        if isinstance(result.get(key), str) and not result.get("parent_body"):
            result["parent_body"] = result[key]
        result.pop(key, None)
    local = result.get("position_m") or result.get("local_pos") or result.get("relative_pos") or result.get("offset")
    if isinstance(local, (list, tuple)):
        result["position_m"] = list(_floats(local, (0, 0, 0), 3))
    for key in ("local_pos", "relative_pos", "offset", "pos"):
        result.pop(key, None)
    return result


def _motion(value: Mapping[str, Any], role: str, attachment: Mapping[str, Any]) -> str:
    explicit = safe_id(value.get("motion") or value.get("dynamics"), "")
    aliases = {"fixed": "static", "free": "dynamic", "freejoint": "dynamic", "mounted": "attached"}
    explicit = aliases.get(explicit, explicit)
    if attachment:
        return "attached"
    if explicit in MOTIONS:
        return explicit
    if role in CARRIED_ROLES:
        return "carried"
    return "dynamic" if bool(value.get("freejoint", False)) else "static"


def _reset_spec(value: Mapping[str, Any], motion: str) -> dict[str, Any]:
    raw = value.get("reset") if isinstance(value.get("reset"), Mapping) else {}
    variants = value.get("variant_randomization") or value.get("randomization")
    pose = variants.get("pose") if isinstance(variants, Mapping) and isinstance(variants.get("pose"), Mapping) else {}
    source = {**dict(pose), **dict(raw)}
    if not source:
        return {}
    frame = safe_id(source.get("frame"), "robot" if motion == "carried" else "initial")
    if frame not in {"world", "initial", "robot"}:
        frame = "initial"
    result: dict[str, Any] = {"frame": frame}
    for src, dst in (
        ("x_offset", "x_offset_m"),
        ("y_offset", "y_offset_m"),
        ("z_offset", "z_offset_m"),
        ("x", "x_m"),
        ("y", "y_m"),
        ("z", "z_m"),
        ("yaw", "yaw_rad"),
    ):
        bounds = _range(source.get(dst, source.get(src)))
        if bounds is not None:
            result[dst] = bounds
    return result


@dataclass(frozen=True)
class SceneObject:
    name: str
    role: str
    motion: str
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    primitive: Primitive | None
    parts: tuple[ScenePart, ...]
    collidable: bool
    collision: str
    mass_kg: float | None
    rgba: tuple[float, float, float, float] | None
    physics: dict[str, Any]
    attachment: dict[str, Any]
    reset: dict[str, Any]
    variants: tuple[dict[str, Any], ...]
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int = 0) -> "SceneObject":
        role = safe_id(value.get("role") or value.get("kind"), "scene_object")
        attachment = _attachment(value)
        motion = _motion(value, role, attachment)
        pose = value.get("pose") if isinstance(value.get("pose"), Mapping) else {}
        position = pose.get("position_m") or value.get("position_m") or value.get("pos") or value.get("world_pos")
        quaternion = _quaternion(value, pose)

        raw_parts = value.get("parts") if isinstance(value.get("parts"), list) else []
        parts = tuple(
            ScenePart.from_mapping(item, idx) for idx, item in enumerate(raw_parts[:16]) if isinstance(item, Mapping)
        )
        geometry = value.get("geometry")
        has_primitive = isinstance(geometry, Mapping) or any(
            key in value for key in ("shape", "type", "size", "dimensions_m")
        )
        primitive = (
            None
            if parts
            else Primitive.from_mapping(geometry if isinstance(geometry, Mapping) else value)
            if has_primitive
            else Primitive("box", (0.08, 0.08, 0.04))
        )
        rgba = _floats(value.get("rgba"), (), 4) if isinstance(value.get("rgba"), (list, tuple)) else None
        mass = _positive(value.get("mass_kg", value.get("mass")), 0.0) or None
        variants = value.get("variants") or value.get("object_variants") or value.get("shape_variants")
        variants = (
            tuple(copy.deepcopy(item) for item in variants[:16] if isinstance(item, Mapping))
            if isinstance(variants, list)
            else ()
        )

        consumed = {
            "name",
            "role",
            "kind",
            "motion",
            "dynamics",
            "freejoint",
            "pose",
            "position_m",
            "pos",
            "world_pos",
            "orientation_wxyz",
            "orientation_euler_deg",
            "quat",
            "geometry",
            "shape",
            "type",
            "collider_shape",
            "size",
            "dimensions_m",
            "parts",
            "collidable",
            "collision",
            "mass",
            "mass_kg",
            "rgba",
            "physics",
            "attachment",
            "parent_body",
            "attached_to",
            "attach_to",
            "mount_body",
            "robot_body",
            "reset",
            "variant_randomization",
            "randomization",
            "variants",
            "object_variants",
            "shape_variants",
        }
        extras = {str(key): copy.deepcopy(item) for key, item in value.items() if key not in consumed}
        return cls(
            name=safe_id(value.get("name"), f"object_{index}"),
            role=role,
            motion=motion,
            position_m=_floats(position, (0, 0, 0), 3),
            orientation_wxyz=quaternion,  # type: ignore[arg-type]
            primitive=primitive,
            parts=parts,
            collidable=bool(value.get("collidable", True)),
            collision=safe_id(value.get("collision"), "proxy"),
            mass_kg=mass,
            rgba=rgba,
            physics=normalize_object_physics(value.get("physics")),
            attachment=attachment,
            reset=_reset_spec(value, motion),
            variants=variants,
            extras=extras,
        )

    @property
    def dynamic(self) -> bool:
        return self.motion in {"dynamic", "carried"}

    @property
    def carried(self) -> bool:
        return self.motion == "carried"

    @property
    def manipulated(self) -> bool:
        return self.dynamic and self.role in MANIPULATED_ROLES

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "motion": self.motion,
            "pose": {"position_m": list(self.position_m), "orientation_wxyz": list(self.orientation_wxyz)},
            "collidable": self.collidable,
            "collision": self.collision,
            **copy.deepcopy(self.extras),
        }
        if self.primitive is not None:
            result["geometry"] = self.primitive.to_dict()
        if self.parts:
            result["parts"] = [part.to_dict() for part in self.parts]
        if self.mass_kg is not None:
            result["mass_kg"] = self.mass_kg
        if self.rgba is not None:
            result["rgba"] = list(self.rgba)
        if self.physics:
            result["physics"] = copy.deepcopy(self.physics)
        if self.attachment:
            result["attachment"] = copy.deepcopy(self.attachment)
        if self.reset:
            result["reset"] = copy.deepcopy(self.reset)
        if self.variants:
            result["variants"] = [SceneObject.from_mapping(item).to_variant_dict() for item in self.variants]
        return result

    def to_variant_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        return {
            key: value
            for key, value in result.items()
            if key not in {"role", "motion", "pose", "collidable", "collision", "attachment", "reset", "variants"}
        }


def normalize_scene_object(value: Mapping[str, Any], index: int = 0) -> dict[str, Any]:
    return SceneObject.from_mapping(value, index).to_dict()


def scene_objects(scene: Mapping[str, Any]) -> tuple[SceneObject, ...]:
    raw = scene.get("objects") if isinstance(scene.get("objects"), list) else []
    return tuple(SceneObject.from_mapping(item, idx) for idx, item in enumerate(raw) if isinstance(item, Mapping))


def normalize_training_scene(scene: Mapping[str, Any]) -> dict[str, Any]:
    result = {str(key): copy.deepcopy(value) for key, value in scene.items() if key != "objects"}
    result["schema"] = "fosa_training_scene_v2"
    objects: list[dict[str, Any]] = []
    source_objects = scene_objects(scene)
    for index, item in enumerate(source_objects):
        # Reserve one slot for every remaining requested object family so a
        # large distributed group cannot crowd later families out entirely.
        reserved = len(source_objects) - index - 1
        limit = max(1, 32 - len(objects) - reserved)
        objects.extend(_expand_distributed_object(item, limit=limit))
        if len(objects) >= 32:
            break
    result["objects"] = objects
    return result


def _expand_distributed_object(obj: SceneObject, *, limit: int) -> list[dict[str, Any]]:
    payload = obj.to_dict()
    try:
        count = int(payload.get("count", 1))
    except (TypeError, ValueError):
        count = 1
    count = max(1, min(limit, count))
    if count == 1 or obj.motion != "static":
        return [payload]

    ranges = payload.get("position_range_m")
    ranges = ranges if isinstance(ranges, Mapping) else {}
    x_bounds = _range(ranges.get("x_m", ranges.get("x"))) or [-2.5, 2.5]
    y_bounds = _range(ranges.get("y_m", ranges.get("y"))) or [-2.5, 2.5]
    z_bounds = _range(ranges.get("z_offset_m", ranges.get("z"))) or [0.0, 0.0]
    distribution = safe_id(payload.get("distribution") or payload.get("layout"), "uniform")
    half = scene_object_half_extents(obj)
    clearance = _nonnegative(payload.get("clearance_from_spawn_m"), 0.9)
    positions = _distributed_positions(count, x_bounds, y_bounds, distribution)
    expanded: list[dict[str, Any]] = []
    for index, (x, y) in enumerate(positions):
        x, y = _move_xy_outside_spawn(x, y, half[0], half[1], clearance, index=index)
        instance = copy.deepcopy(payload)
        instance["name"] = f"{obj.name}_{index:02d}"
        z_offset = z_bounds[0] + (z_bounds[1] - z_bounds[0]) * (index / (count - 1))
        instance["pose"]["position_m"] = [x, y, obj.position_m[2] + half[2] + z_offset]
        for key in ("count", "distribution", "layout", "position_range_m", "position_range", "clearance_from_spawn_m"):
            instance.pop(key, None)
        expanded.append(instance)
    return expanded


def scene_object_half_extents(obj: SceneObject) -> tuple[float, float, float]:
    if obj.primitive is not None:
        return obj.primitive.half_extents()
    extents = [0.0, 0.0, 0.0]
    for part in obj.parts:
        half = part.primitive.half_extents()
        for axis in range(3):
            extents[axis] = max(extents[axis], abs(part.position_m[axis]) + half[axis])
    return tuple(extents)  # type: ignore[return-value]


def _nonnegative(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else default


def _distributed_positions(
    count: int, x_bounds: Sequence[float], y_bounds: Sequence[float], distribution: str
) -> list[tuple[float, float]]:
    if "ring" in distribution:
        radius = max(1.0, min(max(abs(x_bounds[0]), abs(x_bounds[1])), max(abs(y_bounds[0]), abs(y_bounds[1]))))
        return [
            (radius * math.cos(math.tau * index / count), radius * math.sin(math.tau * index / count))
            for index in range(count)
        ]
    if "grid" in distribution:
        columns = max(1, math.ceil(math.sqrt(count)))
        rows = max(1, math.ceil(count / columns))
        positions: list[tuple[float, float]] = []
        for index in range(count):
            row, column = divmod(index, columns)
            x_fraction = 0.5 if columns == 1 else column / (columns - 1)
            if "stagger" in distribution and row % 2:
                x_fraction = min(1.0, x_fraction + 0.5 / max(1, columns - 1))
            y_fraction = 0.5 if rows == 1 else row / (rows - 1)
            positions.append(
                (
                    x_bounds[0] + (x_bounds[1] - x_bounds[0]) * x_fraction,
                    y_bounds[0] + (y_bounds[1] - y_bounds[0]) * y_fraction,
                )
            )
        return positions
    return [
        (
            x_bounds[0] + (x_bounds[1] - x_bounds[0]) * _halton(index + 1, 2),
            y_bounds[0] + (y_bounds[1] - y_bounds[0]) * _halton(index + 1, 3),
        )
        for index in range(count)
    ]


def _halton(index: int, base: int) -> float:
    result, fraction = 0.0, 1.0
    while index:
        fraction /= base
        index, remainder = divmod(index, base)
        result += remainder * fraction
    return result


def _move_xy_outside_spawn(
    x: float, y: float, half_x: float, half_y: float, spawn_radius_m: float, *, index: int = 0
) -> tuple[float, float]:
    required_x = spawn_radius_m + half_x + 0.05
    required_y = spawn_radius_m + half_y + 0.05
    if abs(x) > required_x or abs(y) > required_y:
        return x, y
    move_x = required_x - abs(x)
    move_y = required_y - abs(y)
    if move_x <= move_y:
        sign = -1.0 if x < 0 or (x == 0 and index % 2) else 1.0
        return sign * required_x, y
    sign = -1.0 if y < 0 or (y == 0 and index % 2) else 1.0
    return x, sign * required_y


def repair_static_spawn_overlaps(
    scene: Mapping[str, Any], *, spawn_radius_m: float = 0.9
) -> tuple[dict[str, Any], list[str]]:
    """Normalize distributions and relocate any remaining static spawn obstruction."""

    normalized = normalize_training_scene(scene)
    objects = normalized["objects"]
    manipulation = any(obj.manipulated for obj in scene_objects(normalized))
    repairs: list[str] = []
    for raw in scene.get("objects", []):
        if _nonnegative(raw.get("count"), 1.0) > 1:
            repairs.append(f"distributed_object_expanded:{safe_id(raw.get('name'))}")
    for index, value in enumerate(objects):
        obj = SceneObject.from_mapping(value, index)
        if (
            manipulation
            or not obj.collidable
            or obj.motion != "static"
            or obj.role in {"support_surface", "floor", "ground"}
        ):
            continue
        half = scene_object_half_extents(obj)
        x, y = obj.position_m[:2]
        moved_x, moved_y = _move_xy_outside_spawn(x, y, half[0], half[1], spawn_radius_m, index=index)
        if (moved_x, moved_y) == (x, y):
            continue
        value["pose"]["position_m"] = [moved_x, moved_y, obj.position_m[2]]
        repairs.append(f"static_object_relocated:{obj.name}")
    normalized.pop("scene_quality_issues", None)
    return normalized, list(dict.fromkeys(repairs))


def scene_contract_issues(scene: Mapping[str, Any], *, spawn_radius_m: float = 0.9) -> list[str]:
    """Pure structural checks that every simulator adapter can rely on."""

    issues: list[str] = []
    objects = scene_objects(normalize_training_scene(scene))
    manipulation = any(obj.manipulated for obj in objects)
    names: set[str] = set()
    for obj in objects:
        if obj.name in names:
            issues.append(f"duplicate_object_name:{obj.name}")
        names.add(obj.name)
        if obj.motion == "attached" and not obj.attachment:
            issues.append(f"attached_object_missing_parent:{obj.name}")
        if (
            manipulation
            or not obj.collidable
            or obj.motion != "static"
            or obj.role in {"support_surface", "floor", "ground"}
        ):
            continue
        primitives = (
            [(obj.primitive, (0.0, 0.0, 0.0))]
            if obj.primitive is not None
            else [(part.primitive, part.position_m) for part in obj.parts if part.collidable]
        )
        for primitive, local in primitives:
            half = primitive.half_extents()
            center = tuple(obj.position_m[idx] + local[idx] for idx in range(3))
            overlaps_xy = abs(center[0]) <= spawn_radius_m + half[0] and abs(center[1]) <= spawn_radius_m + half[1]
            occupies_robot_height = center[2] + half[2] > 0.05
            if overlaps_xy and occupies_robot_height:
                issues.append(f"static_object_overlaps_spawn:{obj.name}")
                break
    return list(dict.fromkeys(issues))
