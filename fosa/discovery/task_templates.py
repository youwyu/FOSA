"""Task-family templates: the families of tasks, with their typed and bounded dimensions, that the LLM writes once per
robot from its profile and the simulator manifest (cached under <log root>/task_templates). Every grammar is built
from them, and LLM-CL's prompt lists them."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from fosa import llm
from fosa.compiler import contracts
from fosa.discovery import capability_context
from fosa.discovery.capability_context import BackendCapabilitySpec, RobotEmbodimentSpec, as_list

TEMPLATE_ATTEMPTS = 3
SYSTEM_PROMPT = "You design structured, diverse RL task-family libraries for robot curricula. Output JSON only."
DIMENSION_SCHEMA = {
    "kind": "categorical|ordinal|continuous|integer|boolean",
    "values": "required nonempty list for categorical or ordinal",
    "min_value": "required finite number for continuous or integer",
    "max_value": "required finite number for continuous or integer",
    "default": "required value inside the stated range or values",
    "difficulty_direction": "larger_is_harder|smaller_is_harder|unknown",
    "unit": "physical unit when applicable",
    "description": "one precise sentence",
}
FAMILY_SCHEMA = {
    "family": "short_snake_case_name",
    "description": "what this family explores",
    "primitives": ["one or more registered backend primitives that can execute this family"],
    "required_affordances": ["short snake_case affordance tags grounded in robot_profile"],
    "optional_affordances": ["short snake_case affordance tags grounded in robot_profile"],
    "entity_dimensions": {"dimension_name": DIMENSION_SCHEMA},
    "environment_dimensions": {},
    "goal_dimensions": {},
    "observation_dimensions": {
        "robot_state_history_steps": {"kind": "integer", "min_value": 50, "max_value": 50, "default": 50},
        "object_pose_available": {"kind": "boolean", "values": [True], "default": True},
    },
    "constraints": ["plain language constraints"],
    "success_predicates": ["plain language success criteria"],
    "default_primitive": "optional primitive or null",
    "backend_requirements": ["plain language backend requirements"],
    "real_world_utility": 0.0,
}
RULES = [
    "Generate 8 to 16 task families when possible.",
    "Prefer diverse families over a fixed robotics taxonomy.",
    "Include conventional, edge-case, and cross-embodiment tasks when physically plausible (for example, a dexterous "
    "hand doing free-base locomotion, or a mobile robot doing object interaction), as long as the simulator manifest "
    "supports them.",
    "Do not assume locomotion and manipulation are mutually exclusive, and do not restrict a robot to its "
    "embodiment-typical task families.",
    "Derive required_affordances and optional_affordances from robot_profile; do not use a fixed affordance "
    "vocabulary.",
    "Every family must name at least one registered backend primitive. A primitive is an execution adapter and may "
    "use an LLM-authored task-specific reward; never leave primitives empty and rely on downstream guessing.",
    "All policy observations use robot_state_history_steps=50.",
    "Any family requiring external object state must include object_pose_available=true.",
    "Do not use fixed family names or examples; infer families from robot_profile and backend_capabilities.",
    "Every backend_requirement must be directly supported by simulator_capability_manifest; omit families that need "
    "absent physics, assets, observations, or actuation hooks.",
    "Keep generated dimensions bounded and trainable for the backend.",
    "Every continuous or integer dimension must include finite min_value, max_value, and default; every categorical "
    "or ordinal dimension must include nonempty values and a default from that list. Omit a dimension rather than "
    "returning an unbounded placeholder.",
    "Ground every numeric range in simulator_capability_manifest.robot_spec: speeds and turn rates must stay within "
    "max_linear_velocity and max_angular_velocity, durations within episode_length*ctrl_dt, object sizes within the "
    "end effector's reach. The default must be a clearly achievable value and the range must run from easy to hard, "
    "not only hard: the grammar exists to locate the capability frontier, so it has to contain tasks on both sides "
    "of it.",
    "For every dimension, state whether larger values are harder, smaller values are harder, or the direction is "
    "genuinely non-monotonic/unknown.",
]


@dataclass
class DimensionSpec:
    name: str
    kind: str
    values: list[Any] | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    default: Any | None = None
    difficulty_direction: str | None = None
    description: str | None = None
    unit: str | None = None


@dataclass
class TaskFamilyTemplate:
    family: str
    description: str
    primitives: list[str]
    required_affordances: list[str] = field(default_factory=list)
    optional_affordances: list[str] = field(default_factory=list)
    entity_dimensions: dict[str, DimensionSpec] = field(default_factory=dict)
    environment_dimensions: dict[str, DimensionSpec] = field(default_factory=dict)
    goal_dimensions: dict[str, DimensionSpec] = field(default_factory=dict)
    observation_dimensions: dict[str, DimensionSpec] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    success_predicates: list[str] = field(default_factory=list)
    default_primitive: str | None = None
    backend_requirements: list[str] = field(default_factory=list)
    real_world_utility: float = 0.0


def load_task_template_library(
    args: Any, profile: RobotEmbodimentSpec, backend: BackendCapabilitySpec
) -> dict[str, TaskFamilyTemplate]:
    """The robot's task families, from the cache or else from the LLM (then cached)."""
    name = f"{capability_context.cache_name(profile.robot)}_{capability_context.cache_name(backend.backend)}.json"
    cache_path = llm.model_log_root(args.model) / "task_templates" / name
    with llm.exclusive_file(cache_path):
        cached = capability_context.load_json_object(cache_path)
        if cached:
            return templates_from_raw(cached)
        templates = query_llm_task_templates(args, profile, backend)
        payload = {
            "families": {family: asdict(template) for family, template in templates.items()},
            "_generated_by": "FOSA LLM task-template inference",
            "_source_robot": profile.robot,
            "_source_backend": backend.backend,
        }
        llm.write_json_atomic(cache_path, payload)
        return templates


def query_llm_task_templates(
    args: Any, profile: RobotEmbodimentSpec, backend: BackendCapabilitySpec
) -> dict[str, TaskFamilyTemplate]:
    """The LLM's task families. A reply with missing or unbounded dimensions is sent back for correction, up to
    TEMPLATE_ATTEMPTS times in all."""
    prompt = {
        "request": (
            "Create a diverse typed robot task-family library for empirical RL curriculum search. Return JSON only."
        ),
        "robot_profile": asdict(profile),
        "backend_capabilities": asdict(backend),
        "simulator_capability_manifest": contracts.llm_simulator_capability_manifest(backend.generator_robot_name),
        "schema": {"families": {"family_name": FAMILY_SCHEMA}},
        "rules": RULES,
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(prompt, separators=(",", ":"), sort_keys=True)},
    ]
    for _ in range(TEMPLATE_ATTEMPTS):
        response = llm.chat_json_completion(
            endpoint=os.environ.get("FOSA_LLM_ENDPOINT", llm.DEFAULT_ENDPOINT),
            model=args.model,
            messages=messages,
            max_tokens=8192,
            temperature=0.2,
            api_key=llm.default_api_key(),
            timeout_s=llm.default_timeout_s(),
        )
        templates = templates_from_raw(response)
        issues = dimension_issues(templates)
        if templates and not issues:
            return templates
        messages = [*messages, {"role": "user", "content": "Correct the JSON. " + "; ".join(issues[:20])}]
    raise ValueError("LLM task-template response was incomplete: " + "; ".join(issues[:20]))


def dimension_issues(templates: dict[str, TaskFamilyTemplate]) -> list[str]:
    """The dimensions without a range (numeric) or listed values (categorical) and a default."""
    issues = []
    for family, template in templates.items():
        for dimensions in (
            template.entity_dimensions,
            template.environment_dimensions,
            template.goal_dimensions,
            template.observation_dimensions,
        ):
            for name, dimension in dimensions.items():
                kind = dimension.kind.lower()
                bounded = isinstance(dimension.min_value, int | float) and isinstance(dimension.max_value, int | float)
                if kind in {"continuous", "integer"} and (not bounded or dimension.default is None):
                    issues.append(f"{family}.{name} must have min_value, max_value, and default")
                if kind in {"categorical", "ordinal"} and (
                    not dimension.values or dimension.default not in dimension.values
                ):
                    issues.append(f"{family}.{name} must have values and a listed default")
    return issues


def templates_from_raw(raw: dict[str, Any]) -> dict[str, TaskFamilyTemplate]:
    """The families of an LLM reply or a cache file: a dict or a list of families, under "families" (or
    "task_families"), or the whole object."""
    families = raw.get("families", raw.get("task_families", raw))
    templates = {}
    for name, item in named_items(families, "family", "family").items():
        if isinstance(item, dict):
            template = template_from_dict(str(name), item)
            if template.family and template.description:
                templates[template.family] = template
    return templates


def named_items(raw: Any, key: str, prefix: str) -> dict[Any, Any]:
    """raw as a dict: as it is, or a list of dicts keyed by their `key` field (else by `prefix`_<index>)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {str(item.get(key, f"{prefix}_{idx}")): item for idx, item in enumerate(raw) if isinstance(item, dict)}
    return {}


def template_from_dict(name: str, raw: dict[str, Any]) -> TaskFamilyTemplate:
    default_primitive = raw.get("default_primitive")
    return TaskFamilyTemplate(
        family=str(raw.get("family") or name),
        description=str(raw.get("description", "")),
        primitives=[item for item in as_list(raw.get("primitives")) if item != "rl"],
        required_affordances=as_list(raw.get("required_affordances")),
        optional_affordances=as_list(raw.get("optional_affordances")),
        entity_dimensions=dimensions_from_dict(raw.get("entity_dimensions", {})),
        environment_dimensions=dimensions_from_dict(raw.get("environment_dimensions", {})),
        goal_dimensions=dimensions_from_dict(raw.get("goal_dimensions", {})),
        observation_dimensions=dimensions_from_dict(raw.get("observation_dimensions", {})),
        constraints=as_list(raw.get("constraints")),
        success_predicates=as_list(raw.get("success_predicates")),
        default_primitive=None if default_primitive == "rl" else default_primitive,
        backend_requirements=as_list(raw.get("backend_requirements")),
        real_world_utility=float(raw.get("real_world_utility", 0.0) or 0.0),
    )


def dimensions_from_dict(raw: Any) -> dict[str, DimensionSpec]:
    return {str(name): dimension_from_dict(str(name), value) for name, value in named_items(raw, "name", "dim").items()}


def dimension_from_dict(name: str, raw: Any) -> DimensionSpec:
    if not isinstance(raw, dict):  # a bare value: a dimension fixed at it
        return DimensionSpec(name=name, kind="categorical", values=[raw], default=raw)
    values = raw.get("values", raw.get("options"))
    return DimensionSpec(
        name=str(raw.get("name") or name),
        kind=str(raw.get("kind", "categorical")),
        values=values if values is None or isinstance(values, list) else [values],
        min_value=raw.get("min_value", raw.get("min", raw.get("minimum"))),
        max_value=raw.get("max_value", raw.get("max", raw.get("maximum"))),
        default=raw.get("default"),
        difficulty_direction=raw.get("difficulty_direction", "unknown"),
        description=raw.get("description"),
        unit=raw.get("unit"),
    )


def template_applicability(template: TaskFamilyTemplate, backend: BackendCapabilitySpec) -> tuple[bool, str]:
    """Whether the backend can run the family, and why. The family must bind to a registered execution primitive
    and need no observation modality the backend lacks; the compiler checks each concrete task later."""
    configured_block = backend.unsupported_reason_by_family.get(template.family)
    if configured_block:
        return False, configured_block
    if not template.primitives:
        return False, "family has no registered execution primitive"
    if not set(backend.supported_primitives) & set(template.primitives):
        return False, "backend lacks executable primitive for this family"
    missing_modality = capability_context.missing_observation_modality(
        " ".join([template.description, *template.backend_requirements]), backend.supported_observation_terms
    )
    if missing_modality:
        return False, f"backend policy observations do not provide {missing_modality}"
    return True, "family is bound to a registered backend primitive"


def capability_context_payload(
    profile: RobotEmbodimentSpec, backend: BackendCapabilitySpec, templates: dict[str, TaskFamilyTemplate]
) -> dict[str, Any]:
    """The robot's capability context as the prompts see it: profile, backend capabilities and task families."""
    families = []
    for name, template in templates.items():
        supported, reason = template_applicability(template, backend)
        families.append(
            {
                "family": name,
                "description": template.description,
                "primitives": list(template.primitives),
                "required_affordances": list(template.required_affordances),
                "optional_affordances": list(template.optional_affordances),
                "backend_supported": supported,
                "reason": reason,
                "real_world_utility": template.real_world_utility,
            }
        )
    return {
        "robot_profile": asdict(profile),
        "backend_capabilities": asdict(backend),
        "applicable_task_families": families,
        "backend_supported_families": [item["family"] for item in families if item["backend_supported"]],
        "unsupported_families_with_reasons": {
            item["family"]: item["reason"] for item in families if not item["backend_supported"]
        },
    }
