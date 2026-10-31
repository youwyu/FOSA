"""Per-world model randomization compiled from the LLM-authored domain-randomization operations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from fosa.sim import capabilities


@dataclass(frozen=True)
class ModelRandomizationOp:
    fields: tuple[str, ...]
    indices_by_field: dict[str, np.ndarray]
    bounds: tuple[float, float]
    mode: str = "set"
    components: tuple[int, ...] | None = None


ENTITY_COUNTS = {kind: str(spec["count"]) for kind, spec in capabilities.MODEL_RANDOMIZATION_ENTITY_AXES.items()}

_ALL_SELECTOR_MARKERS = frozenset({"all", "*"})
_SIZABLE_GEOM_TYPES = frozenset(
    int(getattr(mujoco.mjtGeom, f"mjGEOM_{name.upper()}"))
    for name in ("box", "sphere", "cylinder", "capsule", "ellipsoid")
)


def make_domain_randomization_fn(env: Any) -> Callable[[mjx.Model, jax.Array], tuple[mjx.Model, mjx.Model]] | None:
    config = env._domain_randomization
    if not config["enabled"]:
        return None
    compiled = {raw["id"]: _compile_operation(env, raw) for raw in config["model"]["operations"]}
    dropped = [name for name, op in compiled.items() if op is None]
    if dropped:  # an LLM selector that matches no simulator entity is skipped, so say so in the training log
        print(
            f"[domain-randomization] {len(dropped)}/{len(compiled)} model operations matched no entity and were skipped: {dropped}",
            flush=True,
        )
    operations = [op for op in compiled.values() if op is not None]
    if not operations:
        return None
    touched_fields = tuple(dict.fromkeys(field for op in operations for field in op.fields))

    def domain_randomize(model: mjx.Model, rng: jax.Array) -> tuple[mjx.Model, mjx.Model]:
        @jax.vmap
        def rand(single_rng: jax.Array):
            values = {field: getattr(model, field) for field in touched_fields}
            rng_local = single_rng
            for op in operations:
                rng_local, key = jax.random.split(rng_local)
                sample = _sample(key, op.bounds)
                for field in op.fields:
                    values[field] = _apply_operation(
                        values[field],
                        getattr(model, field),
                        op.indices_by_field[field],
                        sample,
                        mode=op.mode,
                        components=op.components,
                    )
            return tuple(values[field] for field in touched_fields)

        randomized_values = rand(rng)
        updates = dict(zip(touched_fields, randomized_values, strict=True))
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace(dict.fromkeys(touched_fields, 0))
        return model.tree_replace(updates), in_axes

    return domain_randomize


def drop_uncompilable_model_operations(env: Any, config: dict[str, Any]) -> None:
    """Discard model operations that do not compile against this environment."""

    model = config["model"]
    model["operations"] = [
        operation for operation in model["operations"] if _compile_operation(env, operation) is not None
    ]


def _compile_operation(env: Any, raw: dict[str, Any]) -> ModelRandomizationOp | None:
    mode = raw["mode"]
    components = tuple(raw["components"]) if "components" in raw else None
    selector = raw["selector"]
    indices_by_field: dict[str, np.ndarray] = {}
    for field in _operation_fields(raw):
        value = getattr(env.mjx_model, field)
        if (
            components is not None
            and value.ndim > 1
            and not any(0 <= component < value.shape[1] for component in components)
        ):
            continue
        kind = _selector_kind(selector, field)
        indices = _selector_indices(env, kind, selector)
        if field == "geom_size":
            geom_types = np.asarray(env.mj_model.geom_type)[indices]
            indices = indices[np.isin(geom_types, tuple(_SIZABLE_GEOM_TYPES))]
        if indices.size:
            indices_by_field[field] = indices
    if not indices_by_field:
        return None
    return ModelRandomizationOp(
        fields=tuple(indices_by_field),
        indices_by_field=indices_by_field,
        bounds=tuple(raw["range"]),
        mode=mode,
        components=components,
    )


def _operation_fields(raw: dict[str, Any]) -> tuple[str, ...]:
    return tuple(raw["fields"]) if "fields" in raw else (raw["field"],)


def _selector_kind(selector: dict[str, Any], field: str) -> str:
    kind = selector.get("kind")
    if kind in ENTITY_COUNTS:
        return kind
    return next(
        name
        for name, spec in capabilities.MODEL_RANDOMIZATION_ENTITY_AXES.items()
        if field.startswith(spec["field_prefix"])
    )


def _selector_indices(env: Any, kind: str, selector: dict[str, Any]) -> np.ndarray:
    count = getattr(env.mj_model, ENTITY_COUNTS[kind])
    if count <= 0:
        return np.asarray([], dtype=np.int32)

    explicit = selector.get("indices")
    if explicit is not None:
        indices = np.asarray(sorted(set(explicit)), dtype=np.int32)
        return indices[(0 <= indices) & (indices < count)]

    if _selector_requests_all(selector):
        return np.arange(count, dtype=np.int32)

    names = tuple(
        dict.fromkeys(
            variant
            for name in _literal_strings(selector.get("name"))
            for variant in (name, re.sub(r"_(?:collider|collision|visual|freejoint)$", "", name))
            if variant
        )
    )
    prefixes = _literal_strings(selector.get("prefix"))
    suffixes = _literal_strings(selector.get("suffix"))
    contains = _literal_strings(selector.get("contains_any"))
    regexes = _strings(selector.get("regex"))
    # LLM-authored selectors sometimes put a regex in ``prefix``.
    literal_prefixes = []
    for prefix in prefixes:
        if any(character in prefix for character in ".^$()|+?{}[]\\"):
            regexes = (*regexes, prefix)
        else:
            literal_prefixes.append(prefix)
    prefixes = tuple(literal_prefixes)
    # Scene generators occasionally put regular expressions in singular
    # ``name`` (or emit ``*`` for all). Accept those common spellings while
    # keeping literal names exact.
    if len(names) == 1:
        name = names[0]
        if name == "*":
            return np.arange(count, dtype=np.int32)
        if name.startswith("regex:"):
            regexes = (*regexes, name.removeprefix("regex:"))
            names = ()
        elif "*" in name:
            regexes = (*regexes, "^" + re.escape(name).replace(r"\*", ".*") + "$")
            names = ()
        elif any(character in name for character in ".^$()|+?{}[]\\"):
            regexes = (*regexes, name)
            names = ()
    regexes = tuple(".*" if pattern == "*" else pattern for pattern in regexes)
    if not any((names, prefixes, suffixes, contains, regexes)):
        return np.arange(count, dtype=np.int32)
    compiled_regexes = []
    for pattern in regexes:  # an invalid LLM-written pattern is skipped and matches nothing
        try:
            compiled_regexes.append(re.compile(pattern))
        except re.error:
            continue

    matched: list[int] = []
    payload_body_id = env._compiled.payload_body_id
    for idx in range(count):
        aliases = _selector_aliases(env.mj_model, kind, idx)
        body_id = idx if kind == "body" else _entity_body_id(env.mj_model, kind, idx)
        if payload_body_id is not None and body_id == payload_body_id:
            aliases.update(_entity_aliases("payload", kind))
        if _is_robot_entity(env, kind, idx):
            aliases.update(_entity_aliases(env._robot, kind))
        if names and aliases.intersection(names):
            matched.append(idx)
        elif prefixes and any(alias.startswith(prefix) for alias in aliases for prefix in prefixes):
            matched.append(idx)
        elif suffixes and any(alias.endswith(suffix) for alias in aliases for suffix in suffixes):
            matched.append(idx)
        elif contains and any(
            token in alias or set(token.split("_")).issubset(alias.split("_"))
            for alias in aliases
            for token in contains
        ):
            matched.append(idx)
        elif compiled_regexes and any(pattern.search(alias) for alias in aliases for pattern in compiled_regexes):
            matched.append(idx)
    # A singular semantic name can denote a generated entity family, e.g.
    # static_box_obstacle_{small,medium,large}. Use it as a prefix only when
    # no exact entity matched.
    if not matched and names:
        for idx in range(count):
            aliases = _selector_aliases(env.mj_model, kind, idx)
            if any(alias.startswith(name) for alias in aliases for name in names):
                matched.append(idx)
    return np.asarray(sorted(set(matched)), dtype=np.int32)


def _is_robot_entity(env: Any, kind: str, idx: int) -> bool:
    root_body_id = env._compiled.root_body_id
    body_id = idx if kind == "body" else _entity_body_id(env.mj_model, kind, idx)
    while body_id is not None:
        if body_id == root_body_id:
            return True
        if body_id <= 0:
            break
        body_id = int(env.mj_model.body_parentid[body_id])
    return False


def _selector_aliases(model: Any, kind: str, idx: int) -> set[str]:
    """Return aliases for an entity and the body that semantically owns it."""

    aliases = _entity_aliases(_entity_name(model, kind, idx), kind)
    if kind == "actuator" and (joint_id := _actuator_joint_id(model, idx)) is not None:
        joint_name = _entity_name(model, "joint", joint_id)
        aliases.update(_entity_aliases(joint_name, "joint"))
        aliases.update(_entity_aliases(joint_name, kind))
    body_id = _entity_body_id(model, kind, idx)
    if body_id is not None:
        body_name = _entity_name(model, "body", body_id)
        aliases.update(_entity_aliases(body_name, "body"))
        aliases.update(_entity_aliases(body_name, kind))
    return aliases


def _entity_body_id(model: Any, kind: str, idx: int) -> int | None:
    body_ids = {"geom": "geom_bodyid", "joint": "jnt_bodyid", "site": "site_bodyid"}
    field = body_ids.get(kind)
    entity_idx = idx
    if kind == "dof":
        field = "jnt_bodyid"
        entity_idx = int(model.dof_jntid[idx])
    elif kind == "actuator":
        field = "jnt_bodyid"
        joint_id = _actuator_joint_id(model, idx)
        if joint_id is None:
            return None
        entity_idx = joint_id
    if field is None:
        return None
    return int(getattr(model, field)[entity_idx])


def _actuator_joint_id(model: Any, idx: int) -> int | None:
    joint_transmissions = {int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)}
    if int(model.actuator_trntype[idx]) not in joint_transmissions:
        return None
    return int(model.actuator_trnid[idx, 0])


def _entity_aliases(name: str, kind: str) -> set[str]:
    """Stable contract IDs for adapter-generated MuJoCo entity names."""

    aliases = {name, f"{kind}_{name}", f"{name}_{kind}"}
    simplified = re.sub(r"^scene_", "", name)
    simplified = re.sub(r"_(?:collider|collision|visual|freejoint)$", "", simplified)
    aliases.add(simplified)
    aliases.add(f"{kind}_{simplified}")
    aliases.add(f"{simplified}_{kind}")
    suffixes = {"_".join(simplified.split("_")[idx:]) for idx in range(1, len(simplified.split("_")))}
    aliases.update(suffixes)
    aliases.update(f"_{suffix}" for suffix in suffixes)
    return aliases


def _entity_name(model: Any, kind: str, idx: int) -> str:
    if kind == "dof":
        joint_id = int(model.dof_jntid[idx])
        return str(model.joint(joint_id).name or f"joint_{joint_id}")
    return str(getattr(model, kind)(idx).name or f"{kind}_{idx}")


def _apply_operation(
    value: jax.Array,
    base_value: jax.Array,
    indices: np.ndarray,
    sample: jax.Array,
    *,
    mode: str,
    components: tuple[int, ...] | None,
) -> jax.Array:
    if value.ndim == 1 or components is None:
        current = base_value[indices]
        update = _operation_update(current, sample, mode)
        return value.at[indices].set(update)

    for component in components:
        if component < 0 or component >= value.shape[1]:
            continue
        current = base_value[indices, component]
        update = _operation_update(current, sample, mode)
        value = value.at[indices, component].set(update)
    return value


def _operation_update(current: jax.Array, sample: jax.Array, mode: str) -> jax.Array:
    if mode == "scale":
        return current * sample
    if mode == "add":
        return current + sample
    return jp.full_like(current, sample)


def _sample(rng: jax.Array, bounds: tuple[float, float]) -> jax.Array:
    return jax.random.uniform(rng, (), minval=bounds[0], maxval=bounds[1])


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    return tuple(item for item in value or () if item)


def _literal_strings(value: Any) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            variant
            for item in _strings(value)
            for variant in (item, re.sub(r"[^A-Za-z0-9_]+", "_", item).strip("_"))
            if variant
        )
    )


def _selector_requests_all(selector: dict[str, Any]) -> bool:
    if selector.get("all") is True:
        return True
    names = _strings(selector.get("name"))
    return any(name.strip().lower() in _ALL_SELECTOR_MARKERS for name in names)
