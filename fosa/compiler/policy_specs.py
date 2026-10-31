"""Policy-side stages: the LLM authors the critic's privileged observation terms, the reward function with its success
condition, and task-specific termination conditions over simulator observables."""

from __future__ import annotations

import json
from typing import Any

from fosa.sim import registry, reward_signals, scene_contract
from fosa import utils
from fosa.compiler import contracts

REWARD_GATES = frozenset({"zero_command", "has_target"})
THRESHOLD_CONDITIONS = frozenset({"greater_than", "less_than", "norm_greater_than", "norm_less_than"})
RANGE_CONDITIONS = frozenset({"outside_range", "inside_range"})


def generate_critic_observation_spec(
    description: str,
    *,
    robot: str,
    task_id: str,
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
    simulation_contract: dict[str, Any],
) -> dict[str, Any]:
    """Privileged critic observation terms chosen from a fixed menu; the actor's observation never changes."""
    fallback = critic_observation_schema_template()
    messages = [
        {
            "role": "system",
            "content": (
                "Choose privileged critic observation terms for PPO as compact JSON. "
                "Use the task contract to decide whether external object state is useful. "
                "Do not change actor or policy observations. Choose only from allowed_terms."
            ),
        },
        {
            "role": "user",
            "content": contracts.prompt_json(
                {
                    "schema": "fosa_critic_observation_v1",
                    "robot": robot,
                    "task_id": task_id,
                    "execution": registry.TASKS[task_id].execution,
                    "description": description,
                    "simulator_capability_manifest": contracts.llm_simulator_capability_manifest(robot, task_id),
                    "simulation_contract": simulation_contract,
                    "required_features": simulation_contract["required_features"],
                    "scene_requirements": simulation_contract["scene_requirements"],
                    "allowed_terms": list(registry.DEFAULT_CRITIC_OBSERVATION_TERMS),
                    "term_meanings": {
                        "policy_state": "the exact policy observation vector; does not alter policy inputs",
                        "qpos": "full generalized positions",
                        "qvel": "full generalized velocities",
                        "ctrl": "current actuator controls",
                        "task_state": "commands, target pose, phase counters, and previous actions when present",
                        "body_pose": "world body positions and orientations",
                        "object_state": "state of task-owned dynamic objects when the contract requires such objects",
                        "contact_state": "per-DOF constraint/contact forces (qfrc_constraint)",
                    },
                    "instructions": [
                        "Use object_state only when simulation_contract.scene_requirements.task_objects.required is true and object state is useful for the critic.",
                        "When the task has no task-owned object state, omit object_state.",
                        "Return JSON only.",
                    ],
                    "return_json": {
                        "schema": "fosa_critic_observation_v1",
                        "source": "llm",
                        "terms": "list of chosen terms from allowed_terms",
                        "notes": "one short reason",
                    },
                    "schema_template": fallback,
                },
            ),
        },
    ]
    reply = contracts.request_chat_json(
        endpoint=endpoint,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=min(max_tokens, 1024),
        api_key=api_key,
    )
    return normalize_critic_observation_spec(reply or fallback)


def critic_observation_schema_template() -> dict[str, Any]:
    return {
        "schema": "fosa_critic_observation_v1",
        "source": "schema_template",
        "terms": [],
        "allowed_terms": list(registry.DEFAULT_CRITIC_OBSERVATION_TERMS),
        "policy_observation": "unchanged",
        "notes": "Schema only; the LLM must choose privileged critic terms.",
    }


def normalize_critic_observation_spec(value: dict[str, Any]) -> dict[str, Any]:
    spec = critic_observation_schema_template()
    spec["source"] = str(value.get("source") or "llm")
    if "notes" in value:
        spec["notes"] = value["notes"]
    terms = value.get("terms")
    if isinstance(terms, list):
        cleaned = [
            str(term).strip() for term in terms if str(term).strip() in registry.DEFAULT_CRITIC_OBSERVATION_TERMS
        ]
        if cleaned:
            spec["terms"] = list(dict.fromkeys(cleaned))
    return spec


def scene_elevation_context(training_scene: dict[str, Any]) -> dict[str, Any]:
    """How far the scene raises the surface the robot spawns on, so height thresholds (base_height) account for it."""
    context: dict[str, Any] = {}
    height = training_scene.get("terrain", {}).get("height")
    if height is not None:
        context["terrain_height_range_m"] = [float(height[0]), float(height[1])]
    tops: list[float] = []
    for obj in scene_contract.scene_objects(training_scene):
        if obj.motion != "static" or not obj.collidable:
            continue
        primitives = (
            [(obj.primitive, (0.0, 0.0, 0.0))]
            if obj.primitive is not None
            else [(part.primitive, part.position_m) for part in obj.parts if part.collidable]
        )
        for primitive, local_pos in primitives:
            half = primitive.half_extents()
            center = tuple(obj.position_m[idx] + local_pos[idx] for idx in range(3))
            if (
                abs(center[0]) <= contracts.SPAWN_RADIUS_M + half[0]
                and abs(center[1]) <= contracts.SPAWN_RADIUS_M + half[1]
            ):
                tops.append(center[2] + half[2])
    if tops:
        context["max_static_object_top_m"] = round(max(tops), 3)
    if context:
        context["note"] = (
            "The robot spawns at its nominal base height above the highest support surface near the "
            "origin and settles under gravity. Absolute-height observables (base_height) include this "
            "surface elevation; set height thresholds/targets relative to the elevated surface and "
            "tolerate reset-settling and normal gait transients on this terrain."
        )
    return context


def generate_task_reward_spec(
    description: str,
    *,
    robot: str,
    task_id: str,
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
    success_predicate: str | None,
    simulation_contract: dict[str, Any],
    domain_randomization: dict[str, Any],
    training_scene: dict[str, Any],
) -> dict[str, Any]:
    """The reward as weighted components over simulator observables, plus the success condition used for evaluation."""
    manifest = contracts.llm_simulator_capability_manifest(robot, task_id)
    observables = reward_signals.reward_signals_for_task(task_id, training_scene)
    messages = [
        {
            "role": "system",
            "content": (
                "Design a PPO reward function for one MuJoCo Playground robot task. "
                "Return JSON only. Do not change actor/policy observations, robot model, or action space. "
                "Use only quantities the simulator can observe or compute from qpos, qvel, ctrl, action, contacts, commands, targets, and task-owned objects. "
                "Do not reference unobservable quantities such as wind speed, human intent, hidden perception labels, or visual appearance."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "robot": robot,
                    "task_id": task_id,
                    "description": description,
                    "symbolic_success_predicate": success_predicate,
                    "simulator_capability_manifest": manifest,
                    "required_features": simulation_contract["required_features"],
                    "domain_randomization": domain_randomization,
                    "scene_elevation": scene_elevation_context(training_scene),
                    "scene_object_signals": reward_signals.scene_object_signal_catalog(training_scene),
                    "simulator_observables": observables,
                    "reward_objectives": list(contracts.REWARD_FUNCTION_OBJECTIVES),
                    "component_schema": {
                        "id": "short metric name",
                        "objective": "exp_tracking|minimize_l2|minimize_abs|maximize_value|penalize_action_rate",
                        "weight": "float coefficient; negative for penalties when objective returns a positive cost",
                        "observable": "one simulator_observable",
                        "target": "optional constant number/list or {'observable': simulator_observable}",
                        "components": "optional integer indices for vector observables",
                        "std": "optional positive width for exp_tracking",
                        "enabled_when": "optional condition such as zero_command or has_target",
                    },
                    "success_condition_schema": {
                        "id": "task_success",
                        "observable": "one simulator_observable",
                        "condition": "greater_than|less_than|outside_range|inside_range|norm_greater_than|norm_less_than",
                        "threshold": "number for scalar/norm comparisons",
                        "range": "[min,max] for range comparisons",
                        "target": "optional constant number/list or {'observable': simulator_observable}",
                        "components": "optional integer indices for vector observables",
                        "hold_steps": "positive integer consecutive control steps, normally 25-100",
                    },
                    "instructions": [
                        "Write the reward as a function made from components over simulator observables, not as names chosen from a fixed reward atom catalog.",
                        "Use command, target, object, contact, and robot-state observables only when they are relevant to the task and available in the simulator_capability_manifest.",
                        "For every relevant scene object, choose its exact object/<name>/<quantity> signal from scene_object_signals; do not assume one primary object or invent an object name.",
                        "Never compare an observable to itself. For hand reorientation use manipulated_object_orientation_error, or compare .../orientation against a literal goal quaternion; for a fixed fixture, use its position or a task-defined constant target. enabled_when supports only zero_command and has_target; do not invent other gate names.",
                        "command is [forward velocity, lateral velocity, yaw rate]. Compare root_linear_velocity_body with command using components=[0,1], and root_angular_velocity_body using components=[2].",
                        "The backend already applies base body-stability regularizers for mobile robots (vertical velocity, roll/pitch rate, orientation, torque, pose, action-rate penalties); author the task-specific reward on top and do not duplicate those stability terms.",
                        "Add exactly one executable success_condition for the task's measurable symbolic_success_predicate. Measure the task-defining achievement, not merely survival, uprightness, or absence of failure; the same observable must appear in a task reward component.",
                        "Before returning, verify that success_condition.observable exactly matches at least one reward component observable.",
                        "When that reward component tracks a dynamic target observable such as command, success_condition must use the same target observable rather than a fixed nominal range.",
                        "For an exp_tracking component with a dynamic target, use success condition norm_less_than with a positive tracking-error threshold.",
                        "When the task names a concrete goal pose, write that pose as a literal target on the .../orientation observable (a quaternion, e.g. a 30 degree turn about z is [0.966, 0, 0, 0.259]) with threshold about 0.1 and hold_steps about 25 -- a named goal is far more learnable than the uniformly random pose the environment samples. Use manipulated_object_orientation_error, the scalar geodesic angle in radians to that sampled pose, only when the task really is to match whatever goal the environment picks.",
                        "Never compare a raw .../orientation quaternion against nothing: its norm is a constant 1, so the reward would have no gradient and the predicate could never fire.",
                        "A position observable likewise needs the target it is measured against: write the concrete goal the task names as a literal 3-vector target with a threshold the robot can hold.",
                        "exp_tracking is exp(-||error||^2 / std^2), so std must match the scale of the error it tracks: a std far below the typical error makes the term identically zero and the policy learns nothing. manipulated_object_orientation_error is an angle in radians spanning 0 to pi, so use std around 0.5 and a success threshold around 0.2-0.4; velocity errors in m/s or rad/s take std around 0.1-0.3; positions in metres take std around 0.02-0.05.",
                        "For a moving robot carrying a payload, use motion or command tracking as success; payload drop is already a safety failure, so payload retention alone must not let a stationary policy succeed. When the reward tracks a dynamic command, reuse the same observable, target, and components instead of inventing an arbitrary traversal distance.",
                        "For a mobile velocity or freebase task whose simulation_contract requires a spatial_constraint, use root_displacement for traversal or target_position_error for an explicit waypoint in both reward and success; instantaneous velocity does not prove completion of authored geometry.",
                        "Set success_condition.hold_steps to require sustained achievement (normally 25-100 control steps). Episode termination already enforces safety, so a conjunctive task should use its primary measurable bottleneck as success.",
                        "Calibrate success tolerances to this robot, using simulator_capability_manifest.robot_spec: commanded speeds must not exceed max_linear_velocity / max_angular_velocity, a tracking threshold should be about 10-20% of the commanded magnitude (never below 0.02 m/s, 0.05 rad/s, 0.02 m or 0.1 rad), and a slow small robot such as a floating hand gets proportionally looser bands than a legged robot. The band has to bracket the task from both sides: an untrained policy -- one that stands still, drifts, or acts randomly -- must FAIL it, and a policy that does the task must be able to hold it. A band the robot can never hold makes the task unlearnable; a band it satisfies before any training makes the task worthless.",
                        "Return JSON only.",
                    ],
                    "return_schema": {
                        "schema": "fosa_task_reward_function_v1",
                        "source": "llm",
                        "reward_function": {
                            "components": [
                                {
                                    "id": "task_metric_name",
                                    "objective": "one reward_objective",
                                    "observable": "one exact simulator_observable",
                                    "target": "optional compatible constant or observable",
                                    "components": "optional component indices",
                                    "std": "positive number for exp_tracking",
                                    "weight": "numeric coefficient",
                                }
                            ]
                        },
                        "success_condition": {
                            "id": "task_success",
                            "observable": "one exact simulator_observable",
                            "condition": "one condition_op",
                            "threshold": "numeric threshold when required",
                            "hold_steps": "consecutive satisfying control steps",
                        },
                        "scale_by_dt": True,
                        "policy_observation": "unchanged",
                        "notes": "short reason",
                    },
                }
            ),
        },
    ]
    reply = contracts.request_chat_json(
        endpoint=endpoint,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=min(max_tokens, 2048),
        api_key=api_key,
    )
    spec = normalize_task_reward_spec(reply, task_id=task_id, training_scene=training_scene)
    if success_predicate:
        spec["success_predicate"] = str(success_predicate)
    return spec


def normalize_task_reward_spec(
    value: dict[str, Any], *, task_id: str, training_scene: dict[str, Any]
) -> dict[str, Any]:
    """Keep the components and success condition that use available observables; report what had to be dropped."""
    reward_function = value.get("reward_function") if isinstance(value.get("reward_function"), dict) else {}
    raw_components = reward_function.get("components") or value.get("components")
    if not isinstance(raw_components, list):
        raw_components = []
    allowed_observables = set(reward_signals.reward_signals_for_task(task_id, training_scene))
    requested_observables: list[str] = []
    for item in raw_components:
        if isinstance(item, dict):
            requested_observables.append(_observable(item))
            if isinstance(item.get("target"), dict):
                requested_observables.append(_observable(item["target"]))
    components = [
        normalize_reward_function_component(item, idx, allowed_observables=allowed_observables)
        for idx, item in enumerate(raw_components)
        if isinstance(item, dict)
    ]
    components = [component for component in components if component]
    raw_success_condition = value.get("success_condition")
    if not isinstance(raw_success_condition, dict):
        raw_success_condition = reward_function.get("success_condition")
    success_condition = normalize_task_success_condition(raw_success_condition, allowed_observables=allowed_observables)
    spec = {
        "schema": "fosa_task_reward_function_v1",
        "source": str(value.get("source") or "llm"),
        "reward_function": {"components": components},
        "scale_by_dt": bool(value.get("scale_by_dt", True)),
        "policy_observation": "unchanged",
        "notes": str(value.get("notes") or ""),
    }
    if success_condition is not None:
        spec["success_condition"] = success_condition
    missing_required_features = [
        f"task_reward_observable_unsupported:{observable}"
        for observable in dict.fromkeys(requested_observables)
        if observable and observable not in allowed_observables
    ]
    missing_required_features.extend(
        f"task_reward_gate_unsupported:{item['enabled_when']}"
        for item in raw_components
        if isinstance(item, dict) and item.get("enabled_when") and item["enabled_when"] not in REWARD_GATES
    )
    if not components:
        missing_required_features.append("task_reward_function")
    if success_condition is None:
        missing_required_features.append("task_success_predicate")
    if missing_required_features:
        spec["missing_required_features"] = missing_required_features
    return spec


def normalize_task_success_condition(value: Any, *, allowed_observables: set[str]) -> dict[str, Any] | None:
    """The executable predicate that decides binary evaluation success, or None when it is not executable."""
    item = _condition(value, "task_success", allowed_observables) if isinstance(value, dict) else None
    if item is None:
        return None
    target = value.get("target")
    if isinstance(target, dict):
        target = _observable_target(target)
        if target["observable"] == item["observable"] or not reward_signals.is_reward_signal(
            target["observable"], allowed_observables
        ):
            return None
        item["target"] = target
    elif isinstance(target, (int, float, list, tuple)):
        item["target"] = target
    _add_components(item, value)
    _normalize_command_components(item)
    hold_steps = contracts.parse_number(value.get("hold_steps") or 25, int)
    item["hold_steps"] = 25 if hold_steps is None else max(1, min(1000, hold_steps))
    return item if _has_operand(item) else None


def normalize_reward_function_component(
    value: dict[str, Any], idx: int, *, allowed_observables: set[str]
) -> dict[str, Any]:
    """One reward term, or {} when its observable, objective, target or gate is unsupported."""
    observable = _observable(value)
    objective = str(value.get("objective") or value.get("op") or "maximize_value").strip()
    if (
        not reward_signals.is_reward_signal(observable, allowed_observables)
        or objective not in contracts.REWARD_FUNCTION_OBJECTIVES
    ):
        return {}
    weight = contracts.parse_number(value.get("weight", 1.0))
    weight = 1.0 if weight is None else weight
    if objective in {"minimize_l2", "minimize_abs", "penalize_action_rate"}:
        weight = -abs(weight)
    component: dict[str, Any] = {
        "id": utils.safe_module_name(str(value.get("id") or value.get("metric") or f"component_{idx}"))[:64],
        "objective": objective,
        "observable": observable,
        "weight": max(-100.0, min(100.0, weight)),
    }
    target = value.get("target")
    if isinstance(target, dict):
        target = _observable_target(target)
        if target["observable"] == observable or not reward_signals.is_reward_signal(
            target["observable"], allowed_observables
        ):
            return {}
        component["target"] = target
    elif isinstance(target, (int, float, list, tuple)):
        component["target"] = target
    _add_components(component, value)
    _normalize_command_components(component)
    std = contracts.parse_number(value.get("std"))
    if std is not None:
        component["std"] = max(1.0e-6, std)
    if value.get("enabled_when"):
        if value["enabled_when"] not in REWARD_GATES:
            return {}
        component["enabled_when"] = value["enabled_when"]
    return component


def _observable(entry: dict[str, Any]) -> str:
    return str(entry.get("observable") or entry.get("signal") or "").strip()


def _observable_target(target: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"observable": _observable(target)}
    if isinstance(target.get("components"), list):
        result["components"] = [int(part) for part in target["components"] if isinstance(part, int)]
    return result


def _add_components(item: dict[str, Any], value: dict[str, Any]) -> None:
    if isinstance(value.get("components"), list):
        components = contracts.integer_components(value["components"])
        if components:
            item["components"] = components


def _condition(value: dict[str, Any], default_id: str, allowed_observables: set[str]) -> dict[str, Any] | None:
    """The id, observable, operator and numeric operands of a success or termination condition, or None when the
    observable or the operator is unsupported."""
    observable = _observable(value)
    condition = str(value.get("condition") or value.get("op") or "").strip()
    if (
        not reward_signals.is_reward_signal(observable, allowed_observables)
        or condition not in contracts.TASK_TERMINATION_CONDITIONS
    ):
        return None
    item: dict[str, Any] = {
        "id": utils.safe_module_name(str(value.get("id") or default_id))[:64],
        "observable": observable,
        "condition": condition,
    }
    threshold = contracts.parse_number(value.get("threshold"))
    if threshold is not None:
        item["threshold"] = threshold
    bounds = value.get("range")
    if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
        lo, hi = contracts.parse_number(bounds[0]), contracts.parse_number(bounds[1])
        if lo is not None and hi is not None:
            item["range"] = [min(lo, hi), max(lo, hi)]
    return item


def _has_operand(item: dict[str, Any]) -> bool:
    """A threshold comparison needs its threshold and a range comparison its range."""
    if item["condition"] in THRESHOLD_CONDITIONS:
        return "threshold" in item
    if item["condition"] in RANGE_CONDITIONS:
        return "range" in item
    return True


def _normalize_command_components(item: dict[str, Any]) -> None:
    """Compare body velocities with the command's matching components: [0, 1] linear, [2] yaw."""
    target = item.get("target")
    if not isinstance(target, dict) or target.get("observable") != "command":
        return
    allowed = {
        "root_linear_velocity_body": [0, 1],
        "root_angular_velocity_body": [2],
    }.get(str(item.get("observable") or ""))
    if allowed is None:
        return
    selected = [index for index in item.get("components", []) if index in allowed]
    item["components"] = selected or allowed
    target.pop("components", None)


def generate_task_termination_spec(
    description: str,
    *,
    robot: str,
    task_id: str,
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
    simulation_contract: dict[str, Any],
    domain_randomization: dict[str, Any],
    training_scene: dict[str, Any],
) -> dict[str, Any]:
    """Task-specific failure terminations; the simulator's safety terminations always apply on top."""
    manifest = contracts.llm_simulator_capability_manifest(robot, task_id)
    observables = reward_signals.reward_signals_for_task(task_id, training_scene)
    messages = [
        {
            "role": "system",
            "content": (
                "Design task-specific episode termination conditions for one MuJoCo Playground robot task. "
                "Return JSON only. Do not change actor/policy observations, robot model, action space, or reward function. "
                "The simulator always keeps non-negotiable safety termination for NaNs, falls, invalid root pose, and dropped task objects. "
                "You only author additional task-semantic failure/unsafe terminations using simulator observables."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "robot": robot,
                    "task_id": task_id,
                    "description": description,
                    "simulator_capability_manifest": manifest,
                    "required_features": simulation_contract["required_features"],
                    "domain_randomization": domain_randomization,
                    "scene_elevation": scene_elevation_context(training_scene),
                    "scene_object_signals": reward_signals.scene_object_signal_catalog(training_scene),
                    "simulator_observables": observables,
                    "condition_ops": list(contracts.TASK_TERMINATION_CONDITIONS),
                    "condition_schema": {
                        "id": "short condition name",
                        "observable": "one simulator_observable",
                        "condition": "greater_than|less_than|outside_range|inside_range|norm_greater_than|norm_less_than",
                        "threshold": "number for scalar/norm comparisons",
                        "range": "[min,max] for range comparisons",
                        "target": "optional constant number/list or {'observable': simulator_observable}; compare observable-target when present",
                        "components": "optional integer indices for vector observables",
                    },
                    "instructions": [
                        "Write only task-specific failure or unsafe-state terminations that are actually observable by the simulator.",
                        "Do not create success/completion/reached-goal terminations; success belongs in reward/metrics because termination has no minimum-duration guard.",
                        "Do not duplicate basic fall/NaN/drop safety termination unless the task needs a stricter semantic threshold.",
                        "Thresholds are evaluated from the first step after reset: leave margin for reset settling and for the tilt/height excursions the authored terrain (scene_elevation) makes normal during the task.",
                        "Terminate only on unrecoverable or unsafe states. Transient performance shortfalls (slow, backward, or imprecise motion an early-training policy will constantly produce) belong in the reward function, not termination — otherwise episodes end before any learning signal accumulates.",
                        "Set grace_steps (control steps, ~0.5-1.5 s worth) so task-specific terminations stay inactive while the robot settles after reset; simulator safety termination is always active.",
                        "Add task-specific termination only when the required task failure condition is represented by available observables; otherwise explain the limitation in notes.",
                        "Return an empty terminate_when list only when no additional task-specific termination is reliably observable.",
                        "Return JSON only.",
                    ],
                    "return_schema": {
                        "schema": "fosa_task_termination_v1",
                        "source": "llm",
                        "terminate_when": [
                            {
                                "id": "task_specific_condition",
                                "observable": "one simulator_observable",
                                "condition": "one condition_op",
                                "threshold": "optional numeric threshold",
                            }
                        ],
                        "grace_steps": "integer control steps during which terminate_when is inactive after reset",
                        "safety_termination": "simulator_default",
                        "policy_observation": "unchanged",
                        "notes": "short reason",
                    },
                }
            ),
        },
    ]
    reply = contracts.request_chat_json(
        endpoint=endpoint,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=min(max_tokens, 2048),
        api_key=api_key,
    )
    return normalize_task_termination_spec(reply, task_id=task_id, training_scene=training_scene)


def normalize_task_termination_spec(
    value: dict[str, Any], *, task_id: str, training_scene: dict[str, Any]
) -> dict[str, Any]:
    """Keep the executable failure conditions and clamp the post-reset grace period."""
    raw_conditions = value.get("terminate_when") or value.get("conditions") or value.get("components")
    if not isinstance(raw_conditions, list):
        raw_conditions = []
    allowed_observables = set(reward_signals.reward_signals_for_task(task_id, training_scene))
    conditions = [
        normalize_task_termination_condition(item, idx, allowed_observables=allowed_observables)
        for idx, item in enumerate(raw_conditions)
        if isinstance(item, dict)
    ]
    grace_steps = contracts.parse_number(value.get("grace_steps") or 0, int)
    return {
        "schema": "fosa_task_termination_v1",
        "source": str(value.get("source") or "llm"),
        "terminate_when": [condition for condition in conditions if condition],
        "grace_steps": 0 if grace_steps is None else max(0, min(200, grace_steps)),
        "safety_termination": "simulator_default",
        "policy_observation": "unchanged",
        "notes": str(value.get("notes") or ""),
    }


def normalize_task_termination_condition(
    value: dict[str, Any], idx: int, *, allowed_observables: set[str]
) -> dict[str, Any]:
    """One termination condition, or {} when it is not executable or reads as success."""
    item = _condition(value, f"termination_{idx}", allowed_observables)
    if item is None:
        return {}
    target = value.get("target")
    if isinstance(target, dict):
        target = _observable_target(target)
        if reward_signals.is_reward_signal(target["observable"], allowed_observables):
            item["target"] = target
    elif isinstance(target, (int, float, list, tuple)):
        item["target"] = target
    _add_components(item, value)
    return item if _has_operand(item) and not _reads_as_success(item) else {}


def _reads_as_success(item: dict[str, Any]) -> bool:
    """Success belongs in the reward, not in termination: a condition named like success, or a stay-inside check not
    named as a failure, is dropped."""
    text = utils.safe_module_name(item["id"])
    if any(
        token in text
        for token in (
            "success",
            "complete",
            "completion",
            "reached",
            "achieved",
            "goal_reached",
        )
    ):
        return True
    return item["condition"] in {"inside_range", "norm_less_than"} and not any(
        token in text for token in ("failure", "unsafe", "violation", "fall", "drop", "off", "out")
    )
