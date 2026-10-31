"""Training-scene stage: the LLM authors the physical scene (terrain and task objects), which is normalized against
the simulation contract; also the validator that every compiled spec must pass before PPO."""

from __future__ import annotations

import copy
import functools
from typing import Any

import numpy as np

from fosa.sim import assets, capabilities, registry, reward_signals, scene_contract
from fosa import utils
from fosa.compiler import contracts, randomization

HAND_STATIC_MAX_EXTENT_M = 0.5
MAX_TERRAIN_HEIGHT_M = 0.30
GENERIC_TERRAIN_TYPES = {"", "heightfield", "terrain", "generated_terrain", "auto"}
# Observables whose success shows that a mobile robot got somewhere.
MOBILE_ACHIEVEMENT_OBSERVABLES = {
    "root_linear_velocity_body",
    "root_angular_velocity_body",
    "root_displacement",
    "target_position_error",
}


def generate_training_scene_spec(
    description: str,
    *,
    robot: str,
    task_id: str,
    render_scene: dict[str, Any],
    simulation_contract: dict[str, Any],
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
) -> dict[str, Any]:
    """The physical training scene: terrain and task objects the contract requires, normalized against it."""
    messages = [
        {
            "role": "system",
            "content": (
                "Compile the task into a simulation-ready MJX training_scene JSON. "
                "You own the train-time terrain, task-owned objects, and object-variant design. "
                "Use the robot name, task language, simulation contract, and backend geometry interface to decide what physical scene elements are needed. "
                "When the task involves an object, propose the object's primitive or composite geometry, placement, mass, contact role, and the variant dimensions or physical properties the domain-randomization stage should randomize. "
                "Output only simulator-ready JSON; validation will reject unsupported geometry or robot capabilities."
            ),
        },
        {
            "role": "user",
            "content": contracts.prompt_json(
                {
                    "schema": "fosa_training_scene_v2",
                    "robot": robot,
                    "task_id": task_id,
                    "execution": registry.TASKS[task_id].execution,
                    "description": description,
                    "simulator_capability_manifest": contracts.llm_simulator_capability_manifest(robot, task_id),
                    "simulation_contract": simulation_contract,
                    "domain_randomization_stage": {
                        "owner": "llm_domain_randomization_compiler",
                        "handoff": "Put object variant hints on task-owned objects; the next LLM call converts them into backend reset ranges and model.operations.",
                    },
                    "render_scene_summary": {
                        "terrain": render_scene["terrain"],
                        "scene_style": render_scene["scene_style"],
                        "labels": render_scene["labels"],
                    },
                    "supported_training_scene": {
                        "terrain": {
                            "type": "heightfield",
                            "satisfies": ["terrain"],
                            "algorithm": "mixed|random_uniform|waves|steps|pyramid",
                            "terrain_type": "specific semantic terrain name from simulation_contract.scene_requirements.terrain",
                            "size": capabilities.TRAINING_TERRAIN_SIZE_M,
                            "resolution": capabilities.TRAINING_TERRAIN_RESOLUTION_M,
                            "nrow": capabilities.TRAINING_TERRAIN_SAMPLES,
                            "ncol": capabilities.TRAINING_TERRAIN_SAMPLES,
                            "height": "[min,max] meters when simulation_contract.scene_requirements.terrain.must_be_nonflat is true",
                        },
                        "objects": [
                            {
                                "name": "backend body name or descriptive object name",
                                "satisfies": ["one or more of: task_objects, context_geometry, spatial_constraint"],
                                "task_relevance": "why this physical asset is required by the exact task",
                                "role": "Use manipulated_object for the primary free body in a hand-manipulation task, task_owned_object for other task-state objects, and scene_object/boundary/path_marker/support_surface for context",
                                "motion": "static|dynamic|attached|carried; carried is a free body transported through physical contact, while attached is rigidly fixed",
                                "pose": {
                                    "position_m": "world [x,y,z] meters",
                                    "orientation_euler_deg": "preferred exact [roll,pitch,yaw] degrees; the compiler converts it to a quaternion",
                                },
                                "geometry": {
                                    "shape": "box|sphere|cylinder|capsule|ellipsoid",
                                    "dimensions_m": "full [x,y,z] dimensions for box/ellipsoid",
                                    "radius_m": "sphere/cylinder/capsule radius",
                                    "length_m": "full cylinder/capsule length",
                                },
                                "rgba": "task-appropriate color [r,g,b,a]",
                                "pos": "world position vector",
                                "attachment": {
                                    "parent_body": "required real robot body name when object is mounted/attached to the robot",
                                    "position_m": "parent-body-frame [x,y,z] placement in meters for mounted objects",
                                    "pos_frame": "parent",
                                },
                                "mass": "optional safe mass in kg",
                                "physics": {
                                    "center_of_mass_offset_m": "optional exact body-frame [x,y,z] offset required by simulation_contract.scene_requirements.task_objects.required_physics",
                                },
                                "collidable": True,
                                "collision": "proxy",
                                "variant_randomization": {
                                    "pose": "deprecated legacy input; prefer the object reset field",
                                    "geometry": "optional size/shape/asymmetry fields the domain-randomization stage should randomize",
                                    "physical_properties": "optional mass/friction/inertia fields the domain-randomization stage should randomize",
                                },
                                "variants": [
                                    {
                                        "name": "semantic variant name",
                                        "geometry": "same explicit metric geometry schema as the parent object",
                                        "rgba": "variant color [r,g,b,a]",
                                        "mass": "optional variant mass in kg",
                                        "parts": "optional variant composite parts using the same part schema",
                                    }
                                ],
                                "parts": [
                                    {
                                        "name": "load_bearing_part",
                                        "geometry": {
                                            "shape": "box",
                                            "dimensions_m": [0.08, 0.05, 0.03],
                                        },
                                        "position_m": [0.0, 0.0, 0.0],
                                        "rgba": "task-appropriate color [r,g,b,a]",
                                        "collidable": True,
                                    },
                                    {
                                        "name": "decorative_detail_part",
                                        "geometry": {"shape": "cylinder", "radius_m": 0.01, "length_m": 0.04},
                                        "position_m": [0.0, 0.0, 0.03],
                                        "rgba": "task-appropriate color [r,g,b,a]",
                                        "collidable": False,
                                    },
                                ],
                                "reset": {
                                    "frame": "initial|robot|world",
                                    "x_offset_m": "optional [min,max] offset",
                                    "y_offset_m": "optional [min,max] offset",
                                    "z_offset_m": "optional [min,max] offset",
                                    "yaw_rad": "optional [min,max] yaw",
                                },
                            }
                        ],
                    },
                    "instructions": [
                        "Design train-time physics geometry, not cinematic decoration.",
                        "Give a task object the structure the task actually depends on, built from parts: a handle to grasp, a lid that can open, a strap that holds a payload, legs that make a load tippy, a rim that catches. A bare primitive is right only when the task is indifferent to the object's shape. Keep the parts inside the object's stated footprint and keep its mass, so the task stays exactly as hard; mark parts that only add appearance collidable=false so contact count stays bounded; use at most six parts.",
                        "If an object matters for the task, design it and its meaningful variants from the task language and robot name instead of relying on a fixed shape.",
                        "Specify size, color, mass, collision role, variant_randomization, and visible object variants for every task-relevant object.",
                        "Every object must include task_relevance; omit physical clutter with no task or contract purpose.",
                        "For task-owned dynamic objects, include variants only when shape or identity is part of the robustness challenge.",
                        "Choose motion from physical semantics: static scenery, dynamic loose bodies, rigidly attached fixtures, or carried cargo. Carried cargo has a free joint and must stay with the robot through support contact and friction; never rigidly attach it.",
                        "When execution=manipulation, author exactly one primary object with role=manipulated_object and motion=dynamic. Its primitive, composite geometry, mass, pose, and variants define the manipulated body.",
                        "Attached objects must include attachment.parent_body from simulator_capability_manifest.mujoco_model.body_names and attachment.position_m in that parent frame.",
                        "Objects pushed, approached, or reached on the ground are dynamic manipulated_object/task_object bodies, never carried_object.",
                        "For carried cargo, use role=carried_object, motion=carried, no attachment, and reset.frame=robot. Place it just above a real support surface while clearing every robot collision envelope listed in simulator_capability_manifest.mujoco_model.collision_geoms, including sensors and their supports.",
                        "When robot_spec.payload_support_position_m is present, use it as the payload support top and keep the payload within robot_spec.payload_max_footprint_m.",
                        "Account for the largest domain_randomization geom_size scale when sizing and placing carried cargo, and author realistic surface friction explicitly.",
                        "Every named object must read as the real thing the task names: compose it from 3-8 primitive parts with realistic proportions and material colours (loading dock = deck slab + bumper strip + corner posts + painted edge; chair = seat + back + four legs; brick wall = courses + jambs + lintel). A single bare primitive standing in for a named real object is not acceptable unless the object really is a box, ball or rod.",
                        "Keep the physics cheap and stable while doing so: mark only the load-bearing or contact-relevant parts collidable, and set collidable=false on purely decorative detail parts. The collidable envelope, not the detail, decides task difficulty.",
                        "Give every part the real object's proportions in metres, checked against simulator_capability_manifest.robot_spec.bounding_box_m, so the scene is to scale for this embodiment.",
                        "Size objects against the measured robot: simulator_capability_manifest.robot_spec.bounding_box_m gives the robot's full [x,y,z] size in meters. Attached or carried cargo must fit its footprint and hand-manipulated objects its grasp span; when the task description requests infeasible dimensions, use the largest feasible size instead.",
                        "Read simulation_contract.scene_requirements as authoritative. If terrain.required is true, output terrain matching its description; if must_be_nonflat is true, include a nonzero visible height range.",
                        "A generic heightfield is insufficient for an exact slope angle, step height, beam width, gap, gate, or clearance. Encode each such metric feature as static primitive/composite geometry with exact metric dimensions and orientation_euler_deg from the task, and include terrain and/or spatial_constraint in satisfies as appropriate.",
                        "For every required scene requirement, explicitly list its key in the satisfies field of the terrain or object that realizes it; one asset may satisfy multiple requirements.",
                        "If context_geometry.required is true, include only low-count geometry that causally affects feasible motion, contact, reward, termination, or reset. Peripheral scenery belongs exclusively in render_scene.",
                        "If spatial_constraint.required is true, include geometry whose roles, names, or tags satisfy its acceptable_roles or matching_terms and physically defines the traversable constraint while leaving enough free space for the robot.",
                        "If task_objects.required is true, include task-owned object geometry matching its description; if attachment_required is true, attach it to a real robot body from simulator_capability_manifest.mujoco_model.body_names.",
                        "Copy every task_objects.required_physics field exactly into the satisfying object's physics; these fields compile into physical model properties rather than visual placement.",
                        "Never put visual-only setting props in training_scene. Use collidable context only when the task requires physical interaction; use a noncollidable marker only when reward or observation consumes it.",
                        "Add context geometry only when the task or simulation contract requires it; prefer the smallest physical scene that preserves task semantics.",
                        "For locomotion, keep collidable objects outside the robot spawn clearance unless they are support surfaces. For manipulation, place required fixtures inside the measured hand workspace without initial penetration.",
                        "Keep the scene MJX/Warp-friendly: at most 32 objects, simple primitive/composite collision geometry.",
                        "Return JSON only.",
                    ],
                    "return_json": {
                        "schema": "fosa_training_scene_v2",
                        "source": "llm_training_scene_compiler",
                        "terrain": "omit unless required by the task/contract",
                        "objects": "the smallest set of task-owned and causally relevant physics objects; render_scene owns recognizable setting decoration",
                        "notes": "one short summary of how the scene matches the task",
                    },
                },
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
    return normalize_training_scene_spec(
        reply or training_scene_schema_template(), simulation_contract=simulation_contract
    )


def training_scene_schema_template() -> dict[str, Any]:
    """The empty scene used when the LLM returns nothing."""
    return {
        "schema": "fosa_training_scene_v2",
        "source": "schema_template",
        "objects": [],
        "notes": "Schema only; the LLM must author task-specific training geometry.",
    }


def normalize_training_scene_spec(
    value: dict[str, Any],
    *,
    simulation_contract: dict[str, Any],
) -> dict[str, Any]:
    """The LLM's scene in the backend schema, with the contract's requirements enforced and checked."""
    spec = {
        "schema": "fosa_training_scene_v2",
        "source": str(value.get("source") or "llm"),
    }
    terrain = value.get("terrain")
    if isinstance(terrain, dict):
        spec["terrain"] = sanitize_training_terrain(terrain)
    objects = value.get("objects")
    if isinstance(objects, list):
        normalized = [scene_contract.normalize_scene_object(item) for item in objects if isinstance(item, dict)]
        spec["objects"] = normalized[: contracts.MAX_TRAINING_OBJECTS]
    if isinstance(value.get("notes"), str):
        spec["notes"] = str(value["notes"])[:500]
    spec = scene_contract.normalize_training_scene(spec)
    return enforce_training_scene_contract(spec, contract=simulation_contract)


def sanitize_training_terrain(value: dict[str, Any]) -> dict[str, Any]:
    """A backend heightfield from the LLM's terrain: fixed size and resolution, a known algorithm, a capped height."""
    height = value.get("height", value.get("height_range_m", value.get("height_range")))
    terrain_type = utils.safe_module_name(
        str(value.get("terrain_type") or value.get("surface") or value.get("type") or "generated_terrain")
    )
    algorithm = utils.safe_module_name(str(value.get("algorithm") or "mixed"))
    if algorithm not in contracts.TERRAIN_ALGORITHMS:
        algorithm = "mixed"
    if terrain_type in GENERIC_TERRAIN_TYPES:
        if algorithm in {"waves", "steps", "pyramid"}:
            terrain_type = f"{algorithm}_terrain"
        elif capabilities.is_nondegenerate_range(height) or isinstance(height, (int, float)):
            terrain_type = f"{algorithm}_rough_heightfield"
    # Keep every LLM-authored field, so the repair LLM sees mis-keyed ones; the backend's fields override.
    terrain = copy.deepcopy(value)
    terrain.update(
        {
            "type": "heightfield",
            "terrain_type": terrain_type,
            "algorithm": algorithm,
            "size": list(capabilities.TRAINING_TERRAIN_SIZE_M),
            "resolution": capabilities.TRAINING_TERRAIN_RESOLUTION_M,
            "nrow": capabilities.TRAINING_TERRAIN_SAMPLES,
            "ncol": capabilities.TRAINING_TERRAIN_SAMPLES,
        }
    )
    # A scalar height or a degenerate [h, h] range means the same height.
    if isinstance(height, (list, tuple)) and len(height) == 1:
        height = height[0]
    if isinstance(height, (list, tuple)) and len(height) >= 2 and height[0] == height[-1]:
        height = height[0]
    if capabilities.is_nondegenerate_range(height):
        lo, hi = sorted((float(height[0]), float(height[1])))
        terrain["height"] = [max(0.0, min(MAX_TERRAIN_HEIGHT_M, lo)), max(0.0, min(MAX_TERRAIN_HEIGHT_M, hi))]
    elif isinstance(height, (int, float)) and float(height) > 0:
        terrain["height"] = [0.0, min(MAX_TERRAIN_HEIGHT_M, float(height))]
    else:
        terrain.pop("height", None)
    if "seed" in value:
        terrain["seed"] = int(value["seed"])
    return terrain


def enforce_training_scene_contract(
    spec: dict[str, Any],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Apply what the contract settles without the LLM, and record what is still missing or wrong."""
    # Decoration the contract never asked for is dropped rather than failing the task over a prop.
    if not _scene_requirement(contract, "context_geometry"):
        spec["objects"] = [
            obj for obj in spec["objects"] if set(map(str, obj.get("satisfies", []))) != {"context_geometry"}
        ]
    terrain_req = _scene_requirement(contract, "terrain")
    if terrain_req.get("must_be_nonflat"):
        terrain = spec.get("terrain", {})
        if not capabilities.is_nondegenerate_range(terrain.get("height")):
            robot_spec = assets.ROBOT_SPECS[contract["robot"]]
            terrain["height"] = [0.0, max(0.05, 0.5 * robot_spec.base_height)]
            terrain["adapter_fallback_fields"] = ["height"]
            spec["terrain"] = terrain
    missing: list[str] = []
    objects = spec["objects"]
    task_object_req = _scene_requirement(contract, "task_objects")
    accepted_roles = set(task_object_req.get("acceptable_roles", []))
    if contract["execution"] == "velocity" and accepted_roles & scene_contract.CARRIED_ROLES:
        payload = next((obj for obj in objects if _training_object_is_task_owned(obj, task_object_req)), None)
        if payload is not None:
            payload["role"] = "carried_object"
            payload["motion"] = "carried"
            payload.pop("attachment", None)
            payload["reset"] = {**payload.get("reset", {}), "frame": "robot"}
    task_objects = [obj for obj in objects if _training_object_is_task_owned(obj, task_object_req)]
    if task_object_req and not task_objects:
        missing.append("task_owned_object")
    if task_object_req.get("attachment_required"):
        if not any(obj["motion"] == "attached" for obj in task_objects):
            missing.append("task_object_attachment")
    if not objects:
        spec.pop("objects")
    if missing:
        spec["missing_required_features"] = missing
    quality_issues = training_scene_quality_issues(spec, contract=contract)
    if quality_issues:
        spec["scene_quality_issues"] = quality_issues
    return spec


def _training_object_is_task_owned(obj: dict[str, Any], requirement: dict[str, Any]) -> bool:
    """Owned by its role or satisfies key, or by a role the contract's task_objects requirement accepts."""
    role = str(obj.get("role") or obj.get("kind") or "").lower()
    explicitly_owned = (
        bool(obj.get("task_owned"))
        or "task_objects" in _satisfaction_keys(obj)
        or role in scene_contract.MANIPULATED_ROLES
        or role in {"dynamic_object", "carried_object", "passive_body_object"}
        or ("task" in role and "object" in role)
    )
    if explicitly_owned:
        return True
    acceptable_roles = set(requirement.get("acceptable_roles", []))
    role_tokens = set(utils.safe_module_name(role).split("_"))
    return any(
        accepted == utils.safe_module_name(role) or set(accepted.split("_")).issubset(role_tokens)
        for accepted in acceptable_roles
    )


def scene_spec_issues(scene_document: dict[str, Any]) -> list[str]:
    """Every validator issue of a compiled spec; a spec trains only when there are none (PPO runs the same check)."""
    training_scene = scene_document["training_scene"]
    contract = scene_document["simulation_contract"]
    task_reward = scene_document["task_reward"]
    task_termination = scene_document["task_termination"]
    issues: list[str] = []
    for section in (training_scene, task_reward):
        issues.extend(section.get("missing_required_features", []))
    contract_source = contract["source"].strip().lower()
    training_source = training_scene["source"].strip().lower()
    if not contract_source:
        issues.append("simulation_contract_source:missing")
    elif contract_source in contracts.UNAUTHORED_SOURCES:
        issues.append(f"simulation_contract_source:{contract_source}")
    if training_source in contracts.UNAUTHORED_SOURCES:
        issues.append(f"training_scene_source:{training_source}")
    if not _task_reward_has_llm_function(task_reward):
        issues.append("task_reward_function_missing_or_non_llm")
    if not _is_llm_authored(task_termination):
        issues.append("task_termination_function_missing_or_non_llm")
    task_id = scene_document["task_id"]
    issues.extend(
        task_policy_observable_issues(
            task_reward,
            task_id=task_id,
            section="reward",
            training_scene=training_scene,
        )
    )
    issues.extend(task_success_quality_issues(task_reward, contract=contract))
    issues.extend(
        task_policy_observable_issues(
            task_termination,
            task_id=task_id,
            section="termination",
            training_scene=training_scene,
        )
    )
    for feature in contract["unsupported_features"]:
        issues.append(f"simulation_contract_unsupported:{feature}")
    issues.extend(training_scene_quality_issues(training_scene, contract=contract))
    issues.extend(
        randomization.domain_randomization_quality_issues(
            scene_document["domain_randomization"],
            task_id=task_id,
            contract=contract,
            training_scene=training_scene,
        )
    )
    return list(dict.fromkeys(issues))


def _is_llm_authored(section: dict[str, Any]) -> bool:
    source = section["source"].strip().lower()
    return source not in {"", "missing"} and source not in contracts.UNAUTHORED_SOURCES


def _task_reward_has_llm_function(value: dict[str, Any]) -> bool:
    return _is_llm_authored(value) and bool(value["reward_function"]["components"])


def task_policy_observable_issues(
    value: dict[str, Any], *, task_id: str, section: str, training_scene: dict[str, Any]
) -> list[str]:
    """Observables the reward or termination uses that this scene does not provide."""
    if section == "reward":
        components = list(value["reward_function"]["components"])
        if "success_condition" in value:
            components.append(value["success_condition"])
    else:
        components = value["terminate_when"]
    allowed = set(reward_signals.reward_signals_for_task(task_id, training_scene))
    used: list[str] = []
    for component in components:
        used.append(component["observable"])
        target = component.get("target")
        if isinstance(target, dict):
            used.append(target["observable"])
    return [
        f"task_{section}_observable_unsupported:{observable}"
        for observable in dict.fromkeys(used)
        if observable not in allowed
    ]


def task_success_quality_issues(value: dict[str, Any], *, contract: dict[str, Any]) -> list[str]:
    """Issues with a success condition that does not measure the task the reward trains."""
    components = value["reward_function"]["components"]
    success = value.get("success_condition")
    if success is None:
        return []
    success_observable = success["observable"]
    reward_observables = {component["observable"] for component in components}
    issues = []
    if success_observable not in reward_observables:
        issues.append(f"task_success_not_task_aligned:{success_observable}")
    dynamic_targets = {
        component["target"]["observable"]
        for component in components
        if component["observable"] == success_observable and isinstance(component.get("target"), dict)
    }
    success_target = success.get("target")
    success_target = success_target["observable"] if isinstance(success_target, dict) else ""
    if dynamic_targets and success_target not in dynamic_targets:
        issues.append(f"task_success_target_mismatch:{success_observable}")
    dynamic_tracking = any(
        component["objective"] == "exp_tracking"
        and component["observable"] == success_observable
        and isinstance(component.get("target"), dict)
        for component in components
    )
    # An untargeted .../orientation is a unit quaternion, so its norm never drops below a small threshold.
    if (
        success_observable.endswith("/orientation")
        and success["condition"] == "norm_less_than"
        and not isinstance(success.get("target"), (dict, list, tuple))
    ):
        issues.append(f"task_success_unsatisfiable:{success_observable}")
    if dynamic_tracking and (
        success["condition"] != "norm_less_than"
        or not isinstance(success.get("threshold"), (int, float))
        or float(success["threshold"]) <= 0.0
    ):
        issues.append(f"task_success_tracking_condition_invalid:{success_observable}")
    scene_requirements = contract["scene_requirements"]
    spatial_required = bool(scene_requirements.get("spatial_constraint", {}).get("required"))
    task_objects = scene_requirements.get("task_objects", {})
    carried_task = bool(task_objects.get("required")) and bool(
        set(task_objects.get("acceptable_roles", [])) & scene_contract.CARRIED_ROLES
    )
    spatial_success = success_observable in MOBILE_ACHIEVEMENT_OBSERVABLES or success_observable.endswith(
        "/distance_to_robot"
    )
    mobile = contract.get("execution") == "velocity"
    if mobile and carried_task and not spatial_success:
        issues.append(f"task_success_does_not_complete_mobile_carry:{success_observable}")
    if mobile and spatial_required and not spatial_success:
        issues.append(f"task_success_does_not_complete_spatial_constraint:{success_observable}")
    if spatial_required and spatial_success:
        condition = success["condition"]
        threshold = success.get("threshold")
        valid_condition = condition in {"less_than", "norm_less_than"}
        if success_observable == "root_displacement":
            valid_condition = condition in {"greater_than", "norm_greater_than"}
        if not isinstance(threshold, (int, float)) or float(threshold) <= 0.0 or not valid_condition:
            issues.append("task_success_spatial_threshold_invalid")
    if success["hold_steps"] < 2:
        issues.append("task_success_not_persistent")
    return issues


@functools.lru_cache(maxsize=None)
def _robot_body_names(robot: str) -> frozenset[str]:
    """Body names an attached object may name as its parent."""
    return frozenset(contracts.simulator_capability_manifest(robot)["mujoco_model"]["bodies"])


def hand_static_object_issues(scene: dict[str, Any]) -> list[str]:
    """A hand works on a tabletop: static obstacles stay tabletop-sized and clear of the object it starts holding."""
    objects = scene_contract.scene_objects(scene)
    held = [
        (np.array(o.position_m), np.array(scene_contract.scene_object_half_extents(o)))
        for o in objects
        if o.manipulated
    ]
    issues = []
    for obj in objects:
        if obj.motion != "static":
            continue
        pos, half = np.array(obj.position_m), np.array(scene_contract.scene_object_half_extents(obj))
        if 2 * half.max() > HAND_STATIC_MAX_EXTENT_M:
            issues.append(f"hand_static_object_too_large:{obj.name}")
        if any(np.all(np.abs(pos - p) < half + h) for p, h in held):
            issues.append(f"hand_static_object_overlaps_manipulated_object:{obj.name}")
    return issues


def training_scene_quality_issues(scene: dict[str, Any], *, contract: dict[str, Any]) -> list[str]:
    """Issues of a training scene against its contract; they go back to the repair LLM."""
    execution = contract["execution"]
    issues: list[str] = []
    terrain_req = _scene_requirement(contract, "terrain")
    terrain = scene.get("terrain", {})
    if terrain_req and not terrain:
        issues.append("task_terrain_missing")
    elif terrain_req and not _satisfies(terrain, "terrain"):
        issues.append("task_terrain_semantics_mismatch")
    if terrain_req.get("must_be_nonflat"):
        # A flat floor satisfies a flat terrain requirement; only elevation needs executable terrain semantics.
        if terrain.get("terrain_type", "") in GENERIC_TERRAIN_TYPES and not _terrain_has_executable_semantics(terrain):
            issues.append("task_terrain_semantics_missing")
        if not capabilities.is_nondegenerate_range(terrain.get("height")):
            issues.append("task_terrain_height_missing")
    objects = scene.get("objects", [])
    context_required = _scene_requirement(contract, "context_geometry").get("required")
    for index, obj in enumerate(objects):
        if not str(obj.get("task_relevance") or "").strip():
            name = utils.safe_module_name(obj["name"])
            issues.append(f"training_scene_asset_relevance_missing:{name}")
        # Context geometry is decoration only when the contract did not ask for it.
        if set(map(str, obj.get("satisfies", []))) == {"context_geometry"} and not context_required:
            issues.append(f"training_scene_visual_only_context:/training_scene/objects/{index}")
    # The hand environment does not compile without a free body; the repair loop adds one.
    if execution == "manipulation" and not any(
        obj["role"] in scene_contract.MANIPULATED_ROLES and obj["motion"] in {"dynamic", "carried"} for obj in objects
    ):
        issues.append("manipulated_object_missing")
    if execution == "manipulation":
        issues.extend(hand_static_object_issues(scene))
    task_object_req = _scene_requirement(contract, "task_objects")
    task_objects = [obj for obj in objects if _training_object_is_task_owned(obj, task_object_req)]
    if task_object_req and not task_objects:
        issues.append("task_owned_object_missing")
    elif task_object_req and not any(_satisfies(obj, "task_objects") for obj in task_objects):
        issues.append("task_object_semantics_mismatch")
    required_physics = task_object_req.get("required_physics", {})
    if required_physics and not any(
        _object_physics_matches(obj, required_physics) for obj in task_objects if _satisfies(obj, "task_objects")
    ):
        issues.append("task_object_physics_mismatch")
    if execution == "manipulation" and not any(
        obj["role"] in scene_contract.MANIPULATED_ROLES and obj["motion"] in {"dynamic", "carried"}
        for obj in task_objects
    ):
        issues.append("manipulated_object_missing_or_static")
    if task_object_req.get("attachment_required"):
        if not any(obj["motion"] == "attached" for obj in task_objects):
            issues.append("task_object_attachment_missing")
    robot_bodies = _robot_body_names(contract["robot"])
    for obj in objects:
        if obj["motion"] != "attached":
            continue
        parent = str(obj.get("attachment", {}).get("parent_body") or "")
        if parent not in robot_bodies:
            issues.append(f"attachment_parent_body_unknown:{utils.safe_module_name(parent or 'missing')}")
    spatial_req = _scene_requirement(contract, "spatial_constraint")
    if spatial_req and not any(_satisfies(obj, "spatial_constraint") for obj in objects):
        issues.append("training_scene_spatial_constraint_missing")
    if task_object_req.get("variant_randomization_required"):
        variable_objects = [obj for obj in task_objects if obj["motion"] != "static"] or task_objects
        for obj in variable_objects:
            if len(obj.get("variants", [])) < 2:
                issues.append("task_object_variants_missing")
    issues.extend(scene_contract.scene_contract_issues(scene))
    return list(dict.fromkeys(issues))


def _object_physics_matches(obj: dict[str, Any], required: dict[str, Any]) -> bool:
    actual = obj.get("physics", {})
    return all(
        key in actual
        and len(actual[key]) == len(expected)
        and all(abs(float(left) - float(right)) <= 1.0e-6 for left, right in zip(actual[key], expected, strict=True))
        for key, expected in required.items()
    )


def _terrain_has_executable_semantics(terrain: dict[str, Any]) -> bool:
    return terrain.get("algorithm") in contracts.TERRAIN_ALGORITHMS and capabilities.is_nondegenerate_range(
        terrain.get("height")
    )


def _scene_requirement(contract: dict[str, Any], key: str) -> dict[str, Any]:
    """The contract's scene requirement, or {} when it is not required."""
    requirement = contract["scene_requirements"].get(key, {})
    return requirement if requirement.get("required") else {}


def _satisfies(item: dict[str, Any], key: str) -> bool:
    return key in _satisfaction_keys(item)


def _satisfaction_keys(item: dict[str, Any]) -> set[str]:
    satisfies = item.get("satisfies")
    if isinstance(satisfies, str):
        satisfies = satisfies.split(",")
    return {utils.safe_module_name(str(value)) for value in satisfies} if isinstance(satisfies, list) else set()


def deferred_render_scene(description: str, *, robot: str, task_id: str) -> dict[str, Any]:
    """The render-scene placeholder every spec carries; specs and later prompts embed it, so its content is frozen."""
    style = "llm_designed"
    return {
        "schema": "fosa_render_scene_v2",
        "source": "deferred",
        "robot": robot,
        "task_id": task_id,
        "description": description.strip(),
        "scene_style": style,
        "visual_mode": "render",
        "seed": utils.stable_int(f"{robot}|{task_id}|{description}|{style}"),
        "terrain": {},
        "scene_layout": {"style": style, "assemblies": []},
        "objects": [],
        "materials": {
            "ground": "llm_specified",
            "accent_palette": [],
        },
        "lighting": {"type": "llm_or_renderer_default"},
        "camera": {"preset": "robot_default", "distance_scale": 1.0},
        "atmosphere": {},
        "labels": [style],
        "physics_features": {},
    }
