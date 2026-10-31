"""Domain-randomization stage: the LLM authors reset ranges and model operations, each with a task-specific rationale,
and the backend keeps only what it can execute."""

from __future__ import annotations

import copy
from typing import Any

from fosa.sim import assets, capabilities, registry, scene_contract
from fosa import utils
from fosa.compiler import contracts

DOMAIN_RANDOMIZATION_ATTEMPTS = 3
PROMPT_OBJECT_LIMIT = 12  # scene objects shown to the randomization prompt


def required_randomization(task_id: str, contract: dict[str, Any]) -> tuple[list[str], tuple[str, ...]]:
    """Reset keys and model fields the task must randomize: those its executor needs plus those its contract asks for."""
    contract_reset, required_model = capabilities.semantic_randomization_requirements(contract)
    required_reset = list(dict.fromkeys((*registry.required_reset_keys_for_task(task_id), *contract_reset)))
    return required_reset, required_model


def domain_randomization_quality_issues(
    value: dict[str, Any],
    *,
    task_id: str,
    contract: dict[str, Any],
    training_scene: dict[str, Any],
) -> list[str]:
    """Validator issues of a randomization spec; they go back to the LLM to fix."""
    issues: list[str] = []
    source = value["source"].strip().lower()
    if not source or source in contracts.UNAUTHORED_SOURCES:
        issues.append(f"domain_randomization_source:{source}")
    if value.get("domains"):
        issues.append("domain_randomization_uses_abstract_domain_menu")
    reset = value["reset"]
    model = value["model"]
    operations = model["operations"]
    if not reset and not operations:
        issues.append("domain_randomization_empty")
    reset_rationale = value["reset_rationale"]
    for key in reset:
        if not _has_rationale(reset_rationale.get(key)):
            issues.append(f"domain_randomization_rationale_missing:reset.{key}")
    for operation in operations:
        if not _has_rationale(operation.get("task_relevance")):
            operation_id = utils.safe_module_name(operation["id"])
            issues.append(f"domain_randomization_rationale_missing:model.{operation_id}")
    required_reset, required_model = required_randomization(task_id, contract)
    for key in required_reset:
        randomized = _is_randomized(reset.get(key)) or _scene_object_randomizes_pose(training_scene, key)
        if not randomized:
            issues.append(f"domain_randomization_reset_missing:{key}")
        elif not _has_rationale(reset_rationale.get(key)):
            issues.append(f"domain_randomization_rationale_missing:reset.{key}")
    for field in required_model:
        if not _has_model_randomization(model, field):
            issues.append(f"domain_randomization_model_missing:{field}")
    return list(dict.fromkeys(issues))


def _scene_object_randomizes_pose(training_scene: dict[str, Any], key: str) -> bool:
    """An object's own reset ranges in the training scene count as object-pose randomization."""
    if key not in {"manipulated_object_pose", "task_object_pose"}:
        return False
    objects = tuple(obj for obj in scene_contract.scene_objects(training_scene) if obj.motion in {"dynamic", "carried"})
    if key == "manipulated_object_pose":
        objects = tuple(obj for obj in objects if obj.role in scene_contract.MANIPULATED_ROLES)
    return any(any(capabilities.is_nondegenerate_range(value) for value in obj.reset.values()) for obj in objects)


def _is_randomized(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_is_randomized(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return capabilities.is_nondegenerate_range(value)
    return isinstance(value, (int, float)) and float(value) > 0.0


def _has_model_randomization(model: dict[str, Any], field: str) -> bool:
    return any(
        field == operation.get("field") or field in operation.get("fields", []) for operation in model["operations"]
    )


def generate_domain_randomization_spec(
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
    training_scene: dict[str, Any],
) -> dict[str, Any]:
    """Ask, validate and ask again with the issues; after the last round, fill missing required resets conservatively."""
    schema_template = domain_randomization_schema_template()
    spec: dict[str, Any] = {}
    issues: list[str] = []
    for _ in range(DOMAIN_RANDOMIZATION_ATTEMPTS):
        llm_spec = request_llm_domain_randomization_spec(
            description,
            robot=robot,
            task_id=task_id,
            endpoint=endpoint,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            schema_template=schema_template,
            simulation_contract=simulation_contract,
            training_scene=training_scene,
            previous_spec=spec or None,
            validation_issues=issues,
        )
        spec = enforce_domain_randomization_contract(
            normalize_domain_randomization_spec(llm_spec),
            contract=simulation_contract,
        )
        _remove_unjustified_randomization(spec, task_id=task_id, contract=simulation_contract)
        issues = domain_randomization_quality_issues(
            spec,
            task_id=task_id,
            contract=simulation_contract,
            training_scene=training_scene,
        )
        if not issues:
            return spec
    missing_reset = [
        issue.rsplit(":", 1)[-1] for issue in issues if issue.startswith("domain_randomization_reset_missing:")
    ]
    for key in missing_reset:
        spec["reset"][key] = _conservative_reset_range(key, robot)
        spec["reset_rationale"].setdefault(
            key,
            "Simulator-bounded fallback after the LLM omitted a required reset range.",
        )
    return spec


def _has_rationale(value: Any) -> bool:
    return bool(value) if isinstance(value, dict) else bool(str(value or "").strip())


def _remove_unjustified_randomization(spec: dict[str, Any], *, task_id: str, contract: dict[str, Any]) -> None:
    """Drop variation without a rationale unless the task requires it; required variation gets a stock rationale."""
    required_reset, required_model = required_randomization(task_id, contract)
    required_reset = set(required_reset)
    required_model = set(required_model)
    reset = spec["reset"]
    rationale = spec["reset_rationale"]
    for key in list(reset):
        if not _has_rationale(rationale.get(key)):
            if key in required_reset:
                rationale[key] = "Required by the task and simulator reset contract."
            else:
                reset.pop(key)
    for key in required_reset:
        rationale.setdefault(key, "Required by the task and simulator reset contract.")
    kept = []
    for operation in spec["model"]["operations"]:
        fields = set(operation.get("fields", [operation.get("field")]))
        if _has_rationale(operation.get("task_relevance")):
            kept.append(operation)
        elif fields & required_model:
            operation["task_relevance"] = "Required by the task's simulation contract."
            kept.append(operation)
    spec["model"]["operations"] = kept


def _conservative_reset_range(key: str, robot: str) -> Any:
    """A small reset range for a required key, scaled by the robot's speed limits."""
    spec = assets.ROBOT_SPECS[robot]
    linear = max(0.1, 0.5 * spec.max_linear_velocity)
    angular = max(0.1, 0.5 * spec.max_angular_velocity)
    if key == "command":
        ranges = {
            name: ([-linear, linear] if name != "yaw" else [-angular, angular]) for name in spec.command_components
        }
        return {**ranges, "standing_probability": 0.1}
    if key == "target":
        return {
            "x_offset": [0.5 * linear, linear],
            "y_offset": [-0.5 * linear, 0.5 * linear],
        }
    if key == "task_object_pose":
        support_z = spec.payload_support_position[2] if spec.payload_support_position is not None else spec.base_height
        return {
            "x_offset": [-0.1, 0.1],
            "y_offset": [-0.1, 0.1],
            "z": [support_z, support_z + 0.05],
            "yaw": [-0.25, 0.25],
        }
    if key == "manipulated_object_pose":
        return {
            "position": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [-0.01, 0.01],
            },
            "orientation_uniform": 1.0,
            "hand_pose_noise": 0.02,
        }
    return {
        "base_xy": [-0.15, 0.15],
        "base_yaw": [-0.25, 0.25],
        "base_velocity_noise": 0.05 * max(linear, angular),
        "joint_pose_noise": 0.02,
    }[key]


def domain_randomization_schema_template() -> dict[str, Any]:
    return {
        "schema": "fosa_domain_randomization_v1",
        "source": "schema_template",
        "enabled": True,
        "reset": {},
        "reset_rationale": {},
        "model": {"operations": []},
        "notes": "Schema only; the LLM must author task-specific reset ranges and model operations.",
    }


def training_scene_randomization_context(training_scene: dict[str, Any]) -> dict[str, Any]:
    """The parts of the training scene the randomization prompt needs: terrain fields, the first objects, notes."""
    context: dict[str, Any] = {}
    terrain = training_scene.get("terrain")
    if terrain is not None:
        context["terrain"] = {
            key: copy.deepcopy(value)
            for key, value in terrain.items()
            if key
            in {
                "type",
                "terrain_type",
                "algorithm",
                "height",
                "size",
                "nrow",
                "ncol",
                "seed",
            }
        }
    objects = [obj.to_dict() for obj in scene_contract.scene_objects(training_scene)[:PROMPT_OBJECT_LIMIT]]
    if objects:
        context["objects"] = objects
    if "notes" in training_scene:
        context["notes"] = training_scene["notes"][:300]
    return context


def randomization_domain_names(domains: Any) -> list[str]:
    """Names of an abstract domain menu, kept only so the validator can reject it."""
    if not isinstance(domains, list):
        return []
    names: list[str] = []
    for item in domains:
        raw_name = (
            item.get("name") or item.get("domain") or item.get("id") or item.get("type")
            if isinstance(item, dict)
            else item
        )
        name = utils.safe_module_name(str(raw_name or ""))
        if name and name not in names:
            names.append(name)
    return names


def request_llm_domain_randomization_spec(
    description: str,
    *,
    robot: str,
    task_id: str,
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
    schema_template: dict[str, Any],
    simulation_contract: dict[str, Any],
    training_scene: dict[str, Any],
    previous_spec: dict[str, Any] | None,
    validation_issues: list[str],
) -> dict[str, Any]:
    manifest = contracts.llm_simulator_capability_manifest(robot, task_id)
    interface = contracts.backend_randomization_interface()
    interface["reset"] = manifest["reset_randomization"]
    interface["model_operation"] = {
        **interface["model_operation"],
        **manifest["model_randomization"],
    }
    required_reset, required_model = required_randomization(task_id, simulation_contract)
    required_model = list(required_model)
    messages = [
        {
            "role": "system",
            "content": (
                "Design concrete domain-randomization specifics for robot policy training and output compact JSON. "
                "You should choose reset ranges and MJX model randomization operations from the task language, robot name, simulation contract, and any object variant hints in the training scene. "
                "The backend will only validate/clamp your JSON; it will not choose semantic randomization domains for you. "
                "Object pose reset ranges live on each training_scene object and are compiled by the simulator adapter. "
                "Use model.operations here only for physical-property randomization such as mass, inertia, size, friction, and actuator parameters. "
                "Every chosen variation must include a short task-specific rationale; do not add randomness merely to increase the count."
            ),
        },
        {
            "role": "user",
            "content": contracts.prompt_json(
                {
                    "schema": "fosa_domain_randomization_v1",
                    "robot": robot,
                    "task_id": task_id,
                    "description": description,
                    "simulator_capability_manifest": manifest,
                    "simulation_contract": simulation_contract,
                    "training_scene": training_scene_randomization_context(training_scene),
                    "backend_randomization_interface": interface,
                    "required_reset_specs": {key: interface["reset"].get(key) for key in required_reset},
                    "required_model_fields": required_model,
                    "previous_spec": previous_spec,
                    "validation_issues_to_fix": validation_issues,
                    "schema_template": schema_template,
                    "instructions": [
                        "Author reset and model randomization directly from the task language and generated training_scene; do not return a fixed-domain menu.",
                        "Use reset keys from the simulator_capability_manifest reset_randomization set.",
                        "The backend has no default randomization values: any reset key you omit stays fixed at zero (no noise, no offsets, zero-velocity commands). Author every range the task needs explicitly.",
                        "After covering the required fields, add physical variation only when it has a direct task-specific rationale; select entities by exposed names.",
                        "For each field, follow field_metadata.recommended_mode and shape_per_entity; components must be valid indices of the first per-entity dimension.",
                        "Write reset and reset_rationale first. reset must contain every required key with a nondegenerate value matching its exact schema, except manipulated_object_pose/task_object_pose may live on the corresponding training_scene object's reset field. reset_rationale must cover every required key either way.",
                        "If validation_issues_to_fix is nonempty, return a complete corrected specification that fixes every listed issue; do not return a patch.",
                        "You must author one executable operation for every required_model_field.",
                        "Choose numeric ranges that are task-specific but conservative enough for stable learning.",
                        "For each reset key, add reset_rationale[key] explaining the task-relevant robustness axis. For each model operation, add task_relevance. Omit irrelevant variation.",
                        "When task-object variants are required, vary primitive dimensions with geom_size in mode='scale' and select the task object's generated geom name; never target meshes or heightfields.",
                        "For non-flat mobile terrain, use supported base_xy reset ranges to sample different authored terrain elevations; heightfield model arrays are not batch-randomizable on this backend.",
                        "Preserve per-object pose reset semantics from training_scene; do not collapse multiple objects into one global reset entry.",
                        "For attached objects, randomize attachment properties with body_pos operations selected by stable object/body name.",
                        "Use operation selectors that match generated training_scene object names or backend entity names exposed by the interface.",
                        "Return JSON only.",
                    ],
                    "return_json": {
                        "schema": "fosa_domain_randomization_v1",
                        "source": "llm",
                        "enabled": True,
                        "reset": {
                            "backend_reset_key": "task-specific nested numeric ranges using keys from backend_randomization_interface.reset",
                        },
                        "reset_rationale": {
                            "backend_reset_key": "why this variation is meaningful for this exact task"
                        },
                        "model": {
                            "operations": [
                                {
                                    "id": "short_semantic_name",
                                    "field": "one supported model operation field",
                                    "selector": "backend entity selector matching generated scene object or robot/world entity",
                                    "components": "optional component indices",
                                    "mode": "set|scale|add",
                                    "range": "[min,max] with min < max; a fixed value such as [0.7,0.7] is discarded, so express a fixed task value as a narrow range around it",
                                    "task_relevance": "why this physical variation matters for this exact task",
                                }
                            ]
                        },
                        "notes": "one short summary of why these ranges match the task",
                    },
                },
            ),
        },
    ]
    return contracts.request_chat_json(
        endpoint=endpoint,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=min(max_tokens, 2048),
        api_key=api_key,
    )


def normalize_domain_randomization_spec(value: dict[str, Any]) -> dict[str, Any]:
    """The LLM's randomization in the backend schema; unsupported operations are dropped."""
    spec = domain_randomization_schema_template()
    spec["source"] = str(value.get("source") or "llm")
    spec["enabled"] = bool(value.get("enabled", True))
    if isinstance(value.get("domains"), list):
        spec["domains"] = randomization_domain_names(value["domains"])
    elif isinstance(value.get("randomization_domains"), list):
        spec["domains"] = randomization_domain_names(value["randomization_domains"])
    if isinstance(value.get("reset"), dict):
        spec["reset"] = normalize_reset_randomization(value["reset"])
    if isinstance(value.get("reset_rationale"), dict):
        spec["reset_rationale"] = {
            str(key): _normalize_rationale(reason)
            for key, reason in value["reset_rationale"].items()
            if str(reason).strip()
        }
    if isinstance(value.get("model"), dict):
        spec["model"] = normalize_model_randomization(value["model"])
    if "notes" in value:
        spec["notes"] = str(value["notes"])
    spec["task_fidelity"] = "follow_task_description_exactly"
    return spec


def _normalize_rationale(value: Any) -> str | dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(reason)[:300] for key, reason in value.items() if str(reason).strip()}
    return str(value)[:300]


def normalize_model_randomization(value: dict[str, Any]) -> dict[str, Any]:
    raw_operations = value.get("operations")
    if not isinstance(raw_operations, list):
        raw_operations = []
    operations = [normalize_model_operation(item) for item in raw_operations if isinstance(item, dict)]
    return {"operations": [item for item in operations if item]}


def normalize_reset_randomization(value: dict[str, Any]) -> dict[str, Any]:
    """Sanitize reset ranges; a per-axis base_xy [[x0, x1], [y0, y1]] is merged into one [min, max] for both axes."""
    raw = dict(value)
    base_xy = raw.get("base_xy")
    if (
        isinstance(base_xy, list)
        and len(base_xy) == 2
        and all(isinstance(axis, list) and len(axis) == 2 for axis in base_xy)
    ):
        raw["base_xy"] = [
            min(float(axis[0]) for axis in base_xy),
            max(float(axis[1]) for axis in base_xy),
        ]
    return capabilities.sanitize_reset_randomization(raw)


def normalize_model_operation(value: dict[str, Any]) -> dict[str, Any] | None:
    """One executable model operation, or None when its field, range, selector or mode is unsupported."""
    raw_fields = value.get("fields")
    if isinstance(raw_fields, str):
        fields = [raw_fields]
    elif isinstance(raw_fields, list):
        fields = [str(field) for field in raw_fields]
    elif value.get("field") is not None:
        fields = [str(value["field"])]
    else:
        fields = []
    supported_fields = capabilities.model_randomization_operation_fields()
    fields = [field for field in fields if field in supported_fields]
    if not fields:
        return None
    bounds = value.get("range") or value.get("bounds") or value.get("value_range")
    if not capabilities.is_nondegenerate_range(bounds):
        return None
    selector = value.get("selector") or value.get("target") or value.get("where")
    if not isinstance(selector, dict):
        return None
    mode = str(value.get("mode") or "set").lower()
    if mode not in {"set", "scale", "add"}:
        mode = "set"
    if "geom_size" in fields and mode != "scale":
        return None
    op: dict[str, Any] = {
        "id": str(value.get("id") or fields[0]),
        "selector": selector,
        "mode": mode,
        "range": [float(bounds[0]), float(bounds[1])],
    }
    if str(value.get("task_relevance") or "").strip():
        op["task_relevance"] = str(value["task_relevance"])[:300]
    if len(fields) == 1:
        op["field"] = fields[0]
    else:
        op["fields"] = fields
    components = value.get("components", value.get("component"))
    if isinstance(components, int):
        op["components"] = [int(components)]
    elif isinstance(components, list):
        components = contracts.integer_components(components)
        if components:
            op["components"] = components
    return op


def enforce_domain_randomization_contract(spec: dict[str, Any], *, contract: dict[str, Any]) -> dict[str, Any]:
    """Record required model fields the spec does not randomize, and attach the contract it was checked against."""
    _, required_model = capabilities.semantic_randomization_requirements(contract)
    missing = [
        f"domain_randomization_model:{key}"
        for key in required_model
        if not _has_model_randomization(spec["model"], key)
    ]
    if missing:
        spec["missing_required_features"] = missing
    spec["simulation_contract"] = contract
    return spec
