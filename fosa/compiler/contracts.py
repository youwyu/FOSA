"""The simulator capability manifest every stage compiles against, the simulation contract (the first stage), and the
LLM request helpers the stages share."""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from typing import Any

from fosa.sim import assets, capabilities, registry, reward_signals, scene_contract
from fosa import llm, utils

REWARD_OBSERVABLE_SEMANTICS = {
    "command": "[forward linear velocity, lateral linear velocity, yaw angular velocity] in m/s, m/s, rad/s",
    "root_displacement": "robot-base XYZ displacement from its reset position",
    "root_linear_velocity_body": "body-frame [vx, vy, vz]; compare with command using components [0,1]",
    "root_angular_velocity_body": "body-frame [wx, wy, wz]; compare with command using component [2]",
}
REWARD_FUNCTION_OBJECTIVES = (
    "exp_tracking",
    "minimize_l2",
    "minimize_abs",
    "maximize_value",
    "penalize_action_rate",
)
TASK_TERMINATION_CONDITIONS = (
    "greater_than",
    "less_than",
    "outside_range",
    "inside_range",
    "norm_greater_than",
    "norm_less_than",
)
SCENE_REQUIREMENT_KEYS = (
    "terrain",
    "task_objects",
    "context_geometry",
    "spatial_constraint",
)
TERRAIN_ALGORITHMS = ("mixed", "random_uniform", "waves", "steps", "pyramid")
MAX_TRAINING_OBJECTS = 32
MAX_PARTS_PER_OBJECT = 16
SPAWN_RADIUS_M = 0.9  # static obstacles stay this far from the robot's spawn point
# A section with one of these sources was not authored by the LLM.
UNAUTHORED_SOURCES = frozenset({"heuristic", "schema_only_fallback", "schema_template", "contract_repaired"})


@lru_cache(maxsize=None)
def _robot_bounding_box_m(robot_name: str) -> dict[str, Any]:
    """World AABB of the robot alone, measured so the LLM can size objects against it."""
    import mujoco
    import numpy as np

    mj_model = mujoco.MjSpec.from_file(str(assets.ROBOT_SPECS[robot_name].xml_path)).compile()
    data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, data)
    ground = {int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD)}
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for gid in range(int(mj_model.ngeom)):
        if int(mj_model.geom_type[gid]) in ground:
            continue
        rot = data.geom_xmat[gid].reshape(3, 3)
        center = data.geom_xpos[gid] + rot @ mj_model.geom_aabb[gid, :3]
        half = np.abs(rot) @ mj_model.geom_aabb[gid, 3:]
        lo, hi = np.minimum(lo, center - half), np.maximum(hi, center + half)
    return {
        "full_size_m": [round(float(value), 4) for value in hi - lo],
        "top_z_m": round(float(hi[2]), 4),
    }


@lru_cache(maxsize=None)
def simulator_capability_manifest(robot: str, task_id: str = "") -> dict[str, Any]:
    """What the backend can simulate for this robot and task, introspected from the robot model and its environment."""
    robot_name = normalize_robot(robot)
    canonical_task = task_id or registry.default_task_for_robot(robot_name)
    import mujoco

    spec = assets.ROBOT_SPECS[robot_name]
    adapter = copy.deepcopy(capabilities.simulator_capability_document())
    adapter["installed_version"] = str(mujoco.__version__)
    adapter["scene"]["documented_geom_types"] = list(capabilities.documented_geom_types())
    robot_tasks = {
        name: {
            "description": task.description,
            "primitive": task.primitive,
            "execution": task.execution,
            "reward_observables": list(reward_signals.reward_signals_for_task(name)),
            "episode_length": task.episode_length,
            "num_envs": task.num_envs,
            "num_timesteps": task.num_timesteps,
        }
        for name, task in sorted(registry.TASKS.items())
        if task.robot == robot_name
    }
    manifest: dict[str, Any] = {
        "schema": "fosa_simulator_capability_manifest_v1",
        "backend": capabilities.BACKEND_ID,
        "simulator_adapter": adapter,
        "robot": robot_name,
        "task_id": canonical_task,
        "task_registry": {
            "default_task": registry.default_task_for_robot(robot_name),
            "selected_task": canonical_task,
            "available_tasks": robot_tasks,
        },
        "scene_backend": {
            "terrain": {
                "type": "heightfield",
                "size_m": capabilities.TRAINING_TERRAIN_SIZE_M,
                "resolution_m": capabilities.TRAINING_TERRAIN_RESOLUTION_M,
                "samples": capabilities.TRAINING_TERRAIN_SAMPLES,
                "algorithms": list(TERRAIN_ALGORITHMS),
            },
            "objects": {
                "primitive_shapes": list(capabilities.SCENE_PRIMITIVE_SHAPES),
                "documented_geom_types": list(adapter["scene"]["documented_geom_types"]),
                "terrain_types": list(adapter["scene"]["terrain_types"]),
                "asset_geometry": list(adapter["scene"]["asset_geometry"]),
                "mesh_formats": list(adapter["scene"]["mesh_formats"]),
                "supports_composite_parts": True,
                "object_physics": copy.deepcopy(adapter["scene"].get("object_physics", {})),
                "object_motion": list(capabilities.SCENE_OBJECT_MOTIONS),
                "geometry_units": "SI meters; boxes/ellipsoids use full dimensions_m, never half-extents",
                "supports_fixed_robot_body_attachment": True,
                "supports_llm_authored_manipulated_object": spec.kind == "hand",
                "manipulated_object_contract": {
                    "role": "manipulated_object",
                    "motion": "dynamic",
                    "geometry": "any supported primitive or composite parts",
                },
                "attachment_fields": {
                    "attachment.parent_body": "real MuJoCo body name from mujoco_model.body_names",
                    "attachment.local_pos": "parent-body-frame [x,y,z] placement in meters",
                    "attachment.pos_frame": "parent for local_pos or world for backward-compatible world pos",
                },
                "supports_visual_mesh_with_proxy_collision": True,
                "max_training_objects": MAX_TRAINING_OBJECTS,
                "max_parts_per_object": MAX_PARTS_PER_OBJECT,
            },
        },
        "robot_spec": {
            "kind": spec.kind,
            "joint_names": list(spec.joint_names),
            "action_scale": spec.action_scale,
            "base_height": spec.base_height,
            "ctrl_dt": spec.ctrl_dt,
            "sim_dt": spec.sim_dt,
            "episode_length": spec.episode_length,
            "command_size": spec.command_size,
            "max_linear_velocity": spec.max_linear_velocity,
            "max_angular_velocity": spec.max_angular_velocity,
            "freebase_root_body": spec.freebase_root_body,
            "payload_support_position_m": list(spec.payload_support_position or ()),
            "payload_max_footprint_m": list(spec.payload_max_footprint or ()),
            "render_camera": {
                "lookat_m": list(spec.render_camera_lookat),
                "distance_m": spec.render_camera_distance,
                "azimuth_deg": spec.render_camera_azimuth,
                "elevation_deg": spec.render_camera_elevation,
            },
            "bounding_box_m": _robot_bounding_box_m(robot_name),
        },
    }

    # A manipulation environment needs an object to compile; a small box stands in for it.
    probe_scene = None
    if registry.TASKS[canonical_task].execution == "manipulation":
        probe_scene = {
            "training_scene": {
                "objects": [
                    {
                        "name": "capability_probe",
                        "role": "manipulated_object",
                        "motion": "dynamic",
                        "geometry": {
                            "shape": "box",
                            "dimensions_m": [0.04, 0.04, 0.04],
                        },
                        "mass_kg": 0.02,
                    }
                ]
            }
        }
    env = registry.make_env(canonical_task, impl="warp", scene_spec=probe_scene)
    env_class = type(env)
    mj_model = env.mj_model
    manifest["reset_randomization"] = {
        key: copy.deepcopy(capabilities.RESET_RANDOMIZATION_INTERFACE[key])
        for key in env_class.RESET_RANDOMIZATION_KEYS
    }
    manifest["model_randomization"] = capabilities.model_randomization_interface(mj_model)
    manifest["actor_interface"] = {
        "env_class": env_class.__name__,
        "action_size": int(spec.command_size) if spec.kind == "wheeled" else int(mj_model.nu),
        "policy_observation": {
            "key": "state",
            "source_method": "_get_obs",
        },
        "qpos_size": int(mj_model.nq),
        "qvel_size": int(mj_model.nv),
        "actuator_count": int(mj_model.nu),
    }
    manifest["mujoco_model"] = {
        "xml_path": str(spec.xml_path),
        "nq": int(mj_model.nq),
        "nv": int(mj_model.nv),
        "nu": int(mj_model.nu),
        "nbody": int(mj_model.nbody),
        "ngeom": int(mj_model.ngeom),
        "njnt": int(mj_model.njnt),
        "root_body": _entity_name(mj_model, "body", 1),
        "bodies": _entity_names(mj_model, "body", int(mj_model.nbody), limit=80),
        "joints": _joint_summaries(mj_model, limit=80),
        "actuators": _actuator_summaries(mj_model, limit=80),
        "geoms": _geom_summaries(mj_model, limit=120),
        "sites": _entity_names(mj_model, "site", int(mj_model.nsite), limit=80),
        "sensors": _entity_names(mj_model, "sensor", int(mj_model.nsensor), limit=80),
    }
    return manifest


def _entity_name(model: Any, kind: str, idx: int) -> str:
    name = getattr(model, kind)(idx).name
    return str(name or f"{kind}_{idx}")


def _entity_names(model: Any, kind: str, count: int, *, limit: int) -> list[str]:
    return [_entity_name(model, kind, idx) for idx in range(min(count, limit))]


def _joint_summaries(model: Any, *, limit: int) -> list[dict[str, Any]]:
    joints: list[dict[str, Any]] = []
    for idx in range(min(int(model.njnt), limit)):
        summary: dict[str, Any] = {
            "name": _entity_name(model, "joint", idx),
            "type": _mujoco_enum_name("mjtJoint", int(model.jnt_type[idx])),
        }
        if bool(model.jnt_limited[idx]):
            summary["range"] = [
                float(model.jnt_range[idx][0]),
                float(model.jnt_range[idx][1]),
            ]
        joints.append(summary)
    return joints


def _actuator_summaries(model: Any, *, limit: int) -> list[dict[str, Any]]:
    return [
        {
            "name": _entity_name(model, "actuator", idx),
            "ctrlrange": [
                float(model.actuator_ctrlrange[idx][0]),
                float(model.actuator_ctrlrange[idx][1]),
            ],
        }
        for idx in range(min(int(model.nu), limit))
    ]


def _geom_summaries(model: Any, *, limit: int) -> list[dict[str, Any]]:
    """Geoms at the model's initial pose."""
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return [
        {
            "name": _entity_name(model, "geom", idx),
            "type": _mujoco_enum_name("mjtGeom", int(model.geom_type[idx])),
            "body": _entity_name(model, "body", int(model.geom_bodyid[idx])),
            "size": [float(item) for item in model.geom_size[idx]],
            "friction": [float(item) for item in model.geom_friction[idx]],
            "position_m": [float(item) for item in data.geom_xpos[idx]],
            "contype": int(model.geom_contype[idx]),
            "conaffinity": int(model.geom_conaffinity[idx]),
        }
        for idx in range(min(int(model.ngeom), limit))
    ]


def _mujoco_enum_name(enum_name: str, value: int) -> str:
    import mujoco

    return str(getattr(mujoco, enum_name)(value).name).replace("mj", "").lower()


def llm_simulator_capability_manifest(robot: str, task_id: str = "") -> dict[str, Any]:
    """The manifest as the LLM sees it: shorter model lists and the reset domains reduced to their field names."""
    manifest = simulator_capability_manifest(robot, task_id)
    task_registry = manifest["task_registry"]
    model = manifest["mujoco_model"]
    return {
        "schema": manifest["schema"],
        "backend": manifest["backend"],
        "simulator_adapter": manifest["simulator_adapter"],
        "robot": manifest["robot"],
        "task_id": manifest["task_id"],
        "task_registry": {
            "default_task": task_registry["default_task"],
            "selected_task": task_registry["selected_task"],
            "available_task_names": list(task_registry["available_tasks"].keys()),
            "available_tasks": task_registry["available_tasks"],
        },
        "robot_spec": manifest["robot_spec"],
        "actor_interface": manifest["actor_interface"],
        "mujoco_model": {
            "xml_path": model["xml_path"],
            "nq": model["nq"],
            "nv": model["nv"],
            "nu": model["nu"],
            "nbody": model["nbody"],
            "ngeom": model["ngeom"],
            "njnt": model["njnt"],
            "root_body": model["root_body"],
            "body_names": model["bodies"][:40],
            "joint_summaries": model["joints"][:40],
            "actuator_names": [item["name"] for item in model["actuators"][:40]],
            "geom_names": [item["name"] for item in model["geoms"][:80]],
            "collision_geoms": [item for item in model["geoms"][:80] if item["contype"] or item["conaffinity"]],
            "site_names": model["sites"][:40],
            "sensor_names": model["sensors"][:40],
        },
        "scene_backend": manifest["scene_backend"],
        "reset_randomization": {
            key: list(item) if isinstance(item, dict) else "range_or_scalar"
            for key, item in manifest["reset_randomization"].items()
        },
        "model_randomization": manifest["model_randomization"],
        "reward_function": {
            "observables": list(reward_signals.reward_signals_for_task(manifest["task_id"])),
            "observable_semantics": dict(REWARD_OBSERVABLE_SEMANTICS),
            "object_signal_pattern": "object/<scene object name>/<quantity>",
            "object_quantities": dict(reward_signals.OBJECT_QUANTITIES),
            "objectives": list(REWARD_FUNCTION_OBJECTIVES),
        },
        "critic_observation": {
            "allowed_terms": list(registry.DEFAULT_CRITIC_OBSERVATION_TERMS),
            "policy_observation": "unchanged",
        },
        "introspection_error": None,
    }


def generate_simulation_contract_spec(
    description: str,
    *,
    robot: str,
    task_id: str,
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
) -> dict[str, Any]:
    """Stage 1: the LLM decides which simulator features, scene elements and randomization the task requires."""
    fallback = simulation_contract_schema_template(robot, task_id)
    manifest = llm_simulator_capability_manifest(robot, task_id)
    messages = [
        {
            "role": "system",
            "content": (
                "Compile the natural-language task into a compact simulator contract JSON. "
                "Use the supplied simulator_capability_manifest, which was introspected from the current MuJoCo/MJX backend and robot model. "
                "You decide which supported physical features, visible scene requirements, reset randomization needs, and model randomization needs are required by the task language. "
                "Do not use a fixed task preset."
            ),
        },
        {
            "role": "user",
            "content": prompt_json(
                {
                    "schema": "fosa_simulation_contract_v1",
                    "robot": robot,
                    "task_id": task_id,
                    "description": description,
                    "simulator_capability_manifest": manifest,
                    "instructions": [
                        "Choose required_features only when they are implied by the task language.",
                        "Choose required_features from capabilities evidenced by simulator_capability_manifest; use concise semantic names grounded in that manifest.",
                        "Write scene_requirements from the task semantics. Downstream validators will use these requirements instead of inferring scene needs from Python keyword lists.",
                        "Use scene_requirements.terrain for any task-required terrain or contact surface beyond the default floor; set must_be_nonflat only when non-flat physical height variation is part of the task.",
                        "Use scene_requirements.task_objects for objects the robot must interact with, manipulate, carry, avoid as a task object, or otherwise make part of the task state.",
                        "When task language specifies an object's center-of-mass offset, copy the exact SI vector into task_objects.required_physics.center_of_mass_offset_m; otherwise leave required_physics empty.",
                        "A beam, walkway, slope, platform, or similar surface that the robot traverses belongs to terrain and/or spatial_constraint, not task_objects, unless the robot carries or manipulates it.",
                        "Set task_objects.attachment_required=true only for an explicitly rigid mount (for example a bolted sensor or fixed tool). A box, crate, or payload transported by surface contact is a carried_object and must keep attachment_required=false so gravity, collision, friction, and drop rewards remain physical.",
                        "Use scene_requirements.context_geometry for visible place/context geometry that should appear in renderuation or training scenes.",
                        "Use scene_requirements.spatial_constraint when the scene must physically define a traversable region, route, clearance, boundary, gate, or similar geometric constraint.",
                        "For each required scene requirement, provide a short description and acceptable_roles or matching_terms that later LLM-authored scene objects can satisfy.",
                        "Choose required_reset_randomization only when reset variation is semantically needed. Return its top-level keys from simulator_capability_manifest.reset_randomization; nested fields such as orientation_uniform belong to their parent key manipulated_object_pose.",
                        "Use task_object_pose only for a dynamic carried object. Static terrain and obstacles vary through base_xy sampling or supported model operations, not task_object_pose.",
                        "For mobile tasks on non-flat terrain, require base_xy when available so resets sample distinct authored terrain elevations; do not invent heightfield model fields listed as backend limitations.",
                        "Choose required_model_randomization only when physical-parameter variation is semantically needed; later stages will compile direct operations.",
                        "Treat task-relevant primitive object dimensions as a robustness variable unless exact dimensions define the task; express that with task_objects.variant_randomization_required and geom_size when supported.",
                        "Use the model_randomization semantic descriptions and limitations literally; never use geom_size for meshes or heightfields.",
                        "Put an item in unsupported_features only if the task explicitly requires something outside simulator_capability_manifest.",
                        "Do not put unused supported capabilities, irrelevant task families, or omitted objects in unsupported_features.",
                        "Return JSON only.",
                    ],
                    "return_json": {
                        "schema": "fosa_simulation_contract_v1",
                        "source": "llm_simulation_contract_compiler",
                        "robot": robot,
                        "task_id": task_id,
                        "required_features": [],
                        "scene_requirements": {
                            "terrain": {
                                "required": False,
                                "description": "",
                                "must_be_nonflat": False,
                                "acceptable_roles": [],
                                "matching_terms": [],
                            },
                            "task_objects": {
                                "required": False,
                                "description": "",
                                "attachment_required": False,
                                "variant_randomization_required": False,
                                "required_physics": {},
                                "acceptable_roles": [],
                                "matching_terms": [],
                            },
                            "context_geometry": {
                                "required": False,
                                "description": "",
                                "acceptable_roles": [],
                                "matching_terms": [],
                            },
                            "spatial_constraint": {
                                "required": False,
                                "description": "",
                                "acceptable_roles": [],
                                "matching_terms": [],
                            },
                        },
                        "required_reset_randomization": [],
                        "required_model_randomization": [],
                        "unsupported_features": [],
                        "policy_observation": "unchanged unless the task requires a new actor observation",
                        "task_fidelity": "one short statement",
                    },
                    "schema_template": fallback,
                },
            ),
        },
    ]
    reply = request_chat_json(
        endpoint=endpoint,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=min(max_tokens, 2048),
        api_key=api_key,
    )
    return normalize_simulation_contract_spec(reply or fallback, fallback=fallback)


def normalize_scene_requirements(value: Any) -> dict[str, dict[str, Any]]:
    source = value if isinstance(value, dict) else {}
    normalized: dict[str, dict[str, Any]] = {}
    for key in SCENE_REQUIREMENT_KEYS:
        raw = source.get(key) if isinstance(source.get(key), dict) else {}
        req = {
            "required": bool(raw.get("required", False)),
            "description": str(raw.get("description") or "")[:500],
            "acceptable_roles": _normalize_requirement_terms(raw.get("acceptable_roles") or raw.get("roles")),
            "matching_terms": _normalize_requirement_terms(raw.get("matching_terms") or raw.get("terms")),
        }
        if key == "terrain":
            req["must_be_nonflat"] = bool(raw.get("must_be_nonflat", False))
        if key == "task_objects":
            req["attachment_required"] = bool(raw.get("attachment_required", False))
            req["variant_randomization_required"] = bool(raw.get("variant_randomization_required", False))
            req["required_physics"] = scene_contract.normalize_object_physics(raw.get("required_physics"))
        normalized[key] = req
    return normalized


def _normalize_requirement_terms(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    terms = [utils.safe_module_name(str(item)) for item in items if item is not None and str(item).strip()]
    return list(dict.fromkeys(term for term in terms if term))


def normalize_simulation_contract_spec(value: dict[str, Any], *, fallback: dict[str, Any]) -> dict[str, Any]:
    """Keep the contract to names the backend supports; anything else the task requires is listed as unsupported."""
    robot = fallback["robot"]
    task_id = fallback["task_id"]
    execution = registry.TASKS[task_id].execution

    def list_of_names(key: str) -> list[str]:
        raw = value.get(key)
        if not isinstance(raw, list):
            raw = []
        names = [utils.safe_module_name(str(item)) for item in raw if str(item)]
        return list(dict.fromkeys(name for name in names if name))

    requested_reset = list_of_names("required_reset_randomization")
    requested_canonical_reset = [capabilities.canonical_reset_randomization_key(name) for name in requested_reset]
    canonical_reset = list(requested_canonical_reset)
    requested_model = list_of_names("required_model_randomization")
    required_features = list_of_names("required_features")
    canonical_model = [capabilities.canonical_model_randomization_field(name) for name in requested_model]
    unsupported: list[str] = []
    for name in list_of_names("unsupported_features"):
        # A supported reset entry listed as unsupported (reset_randomization_key_command_...) is recovered.
        reset_key = _reset_key_from_unsupported_feature(name)
        if reset_key is not None:
            canonical_reset.append(reset_key)
        else:
            unsupported.append(name)
    unsupported.extend(
        f"reset_randomization_{name}"
        for name, canonical in zip(requested_reset, requested_canonical_reset, strict=True)
        if canonical is None
    )
    unsupported.extend(
        f"model_randomization_{name}"
        for name, canonical in zip(requested_model, canonical_model, strict=True)
        if canonical is None
    )
    scene_requirements = normalize_scene_requirements(value.get("scene_requirements"))
    task_object_requirement = scene_requirements["task_objects"]
    description = task_object_requirement["description"].lower()
    carried_roles = set(task_object_requirement["acceptable_roles"])
    if any("carried" in feature or "payload_carry" in feature for feature in required_features):
        carried_roles.add("carried_object")
        task_object_requirement["acceptable_roles"] = list(carried_roles)
    explicit_carry = re.search(
        r"\b(?:carry|carries|carried|carrying|transport|transports|transported|transporting)\b",
        description,
    )
    optional_carry = re.search(
        r"\b(?:potentially|optionally|may|might|could)\s+(?:be\s+)?(?:carry|carried|transported)\b",
        description,
    )
    if execution == "velocity" and explicit_carry and not optional_carry:
        carried_roles.add("carried_object")
        task_object_requirement["acceptable_roles"] = list(carried_roles)
    carries = bool(carried_roles & scene_contract.CARRIED_ROLES)
    if not carries:
        canonical_reset = [name for name in canonical_reset if name != "task_object_pose"]
    if execution != "manipulation" and carries and assets.ROBOT_SPECS[robot].payload_support_position is None:
        unsupported.append("carried_object_without_support")
    interaction_text = " ".join([description, *required_features])
    if execution != "manipulation" and re.search(
        r"\b(?:grasp|grasped|grasping|hammer|strike|lift(?:ing)?\s+(?:an?\s+)?object)\b",
        interaction_text,
    ):
        unsupported.append("object_manipulation")
    if execution == "manipulation" or carries:
        # Manipulated bodies and carried payloads stay free so contact, friction and drops stay physical.
        task_object_requirement["attachment_required"] = False

    normalized = {
        "schema": "fosa_simulation_contract_v1",
        "source": str(value.get("source") or "llm"),
        # Robot, task and execution were fixed before this stage; the LLM cannot change them.
        "robot": robot,
        "task_id": task_id,
        "execution": execution,
        "required_features": required_features,
        "scene_requirements": scene_requirements,
        "required_reset_randomization": [name for name in dict.fromkeys(canonical_reset) if name is not None],
        "required_model_randomization": [name for name in dict.fromkeys(canonical_model) if name is not None],
        "unsupported_features": list(dict.fromkeys(unsupported)),
        "policy_observation": str(value.get("policy_observation") or "unchanged"),
        "task_fidelity": str(value.get("task_fidelity") or fallback["task_fidelity"]),
    }
    reset_requirements, model_requirements = capabilities.semantic_randomization_requirements(normalized)
    normalized["required_reset_randomization"] = list(reset_requirements)
    normalized["required_model_randomization"] = list(model_requirements)
    return normalized


def _reset_key_from_unsupported_feature(name: str) -> str | None:
    """The supported top-level reset key an LLM-serialized feature name refers to, if any."""
    canonical = capabilities.canonical_reset_randomization_key(name)
    if canonical is not None:
        return canonical
    prefix = "reset_randomization_"
    if not name.startswith(prefix):
        return None
    remainder = name[len(prefix) :].removeprefix("key_")
    matches = [
        key for key in capabilities.RESET_RANDOMIZATION_INTERFACE if remainder == key or remainder.startswith(f"{key}_")
    ]
    return max(matches, key=len) if matches else None


def prompt_json(payload: Any) -> str:
    """Token-efficient serialization for LLM prompt content (not for on-disk specs)."""
    return json.dumps(payload, separators=(",", ":"))


def parse_number(value: Any, kind: type = float) -> Any:
    """value converted with kind (float or int), or None when LLM output does not convert."""
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


def integer_components(values: list[Any]) -> list[int]:
    """The entries of an LLM-authored component list that parse as integers, deduplicated in order."""
    parsed = (parse_number(value, int) for value in values)
    return list(dict.fromkeys(value for value in parsed if value is not None))


def request_chat_json(
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    api_key: str | None,
) -> dict[str, Any]:
    """One JSON chat completion with the shared LLM timeout."""
    return llm.chat_json_completion(
        endpoint=endpoint,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
        timeout_s=llm.default_timeout_s(),
    )


def backend_randomization_interface() -> dict[str, Any]:
    """The reset and model-operation fields the randomization stages may use."""
    return {
        "reset": copy.deepcopy(capabilities.RESET_RANDOMIZATION_INTERFACE),
        "model_operation": {
            **capabilities.model_randomization_interface(),
            "selector": {
                "kind": "one model_operation.selector_kinds value",
                "name": "exact MuJoCo entity name",
                "prefix": "match names by prefix",
                "contains_any": ["match", "tokens"],
                "regex": "optional regular expression",
            },
            "range": "[min,max] sampled uniformly per vectorized rollout batch",
        },
        "notes": "These are backend API fields, not fixed task-domain choices. Use only what the task requires.",
    }


def simulation_contract_schema_template(robot: str, task_id: str) -> dict[str, Any]:
    """The empty contract shown to the LLM as a schema; it is also the fallback for an empty reply."""
    return {
        "schema": "fosa_simulation_contract_v1",
        "source": "schema_template",
        "robot": robot,
        "task_id": task_id,
        "execution": registry.TASKS[task_id].execution,
        "required_features": [],
        "scene_requirements": normalize_scene_requirements({}),
        "required_reset_randomization": [],
        "required_model_randomization": [],
        "unsupported_features": [],
        "policy_observation": "unchanged",
        "task_fidelity": "Schema only; the LLM must author task-specific required features.",
    }


def normalize_robot(robot: str | None) -> str:
    """Resolve a robot name against the registry: exact match, unique prefix, or default."""
    if not str(robot or "").strip():
        return registry.default_robot()
    normalized = utils.safe_module_name(str(robot))
    if normalized in registry.DEFAULT_ROBOT_TASKS:
        return normalized
    matches = [name for name in registry.DEFAULT_ROBOT_TASKS if name.startswith(normalized)]
    return matches[0] if len(matches) == 1 else normalized
