"""What a robot can do, as every method's prompts and grammar see it: the robot profile the LLM infers once from the
simulator manifest (cached under <log root>/robot_profiles), the simulator's capabilities as tokens, and the checks
that a text needs no observation or runtime effect the simulator lacks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from fosa import llm
from fosa.compiler import contracts
from fosa.sim import registry

LIST_FIELDS = (
    "locomotion_modes",
    "body_support_surfaces",
    "manipulators",
    "end_effectors",
    "sensors",
    "task_object_support_surfaces",
)
BOOL_FIELDS = (
    "can_support_task_objects",
    "task_object_requires_fixture",
    "can_grasp",
    "can_lift_objects",
    "can_in_hand_manipulate",
    "can_use_tools",
)
OBSERVATION_MODALITIES = {
    "lidar": ("lidar", "point_cloud", "point cloud"),
    "camera": ("camera", "rgb", "depth_image", "visual"),
}
RUNTIME_REQUIREMENTS = {  # runtime effect: (phrases that need it, capability tokens that provide it)
    "external_impulse": (
        (
            "external impulse",
            "external_impulse",
            "base disturbance",
            "continuous disturbance",
            "periodic disturbance",
            "disturbance frequency",
            "disturbance_frequency",
            "lateral velocity disturbance",
            "rotational velocity disturbance",
        ),
        ("external_impulse", "disturbance"),
    ),
    "slip": (
        ("slip detect", "slip_detect", "slip ratio", "slip_ratio", "slip recovery", "slip_recovery"),
        ("slip",),
    ),
    "carried_payload": (("carry", "carried", "payload"), ("supports_carried_payload", "primitive:manipulation")),
}
MANIPULATION_PHRASES = ("grasp", "pick up", "use_hammer", "hammer", "lift ")
TOKEN_PREFIXES = {
    "supported_primitives": "primitive",
    "supported_assets": "asset",
    "supported_reward_terms": "reward",
    "supported_observation_terms": "observation",
}


@dataclass
class RobotEmbodimentSpec:
    robot: str
    robot_display_name: str
    embodiment_type: str
    description: str
    locomotion_modes: list[str] = field(default_factory=list)
    body_support_surfaces: list[str] = field(default_factory=list)
    manipulators: list[str] = field(default_factory=list)
    end_effectors: list[str] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)
    can_support_task_objects: bool = False
    task_object_support_surfaces: list[str] = field(default_factory=list)
    task_object_capacity_kg: float | None = None
    task_object_requires_fixture: bool = False
    can_grasp: bool = False
    can_lift_objects: bool = False
    can_in_hand_manipulate: bool = False
    can_use_tools: bool = False
    notes: str | None = None


@dataclass
class BackendCapabilitySpec:
    backend: str
    robot: str
    generator_robot_name: str
    supported_primitives: list[str] = field(default_factory=list)
    supported_assets: list[str] = field(default_factory=list)
    supported_reward_terms: list[str] = field(default_factory=list)
    supported_observation_terms: list[str] = field(default_factory=list)
    unsupported_reason_by_family: dict[str, str] = field(default_factory=dict)
    notes: str | None = None


def load_json_object(path: Path) -> dict[str, Any]:
    """The JSON object in path; {} if there is none (a cache not yet written)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def as_list(value: Any) -> list[str]:
    """An LLM-written list: a list, or a comma-separated string."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):  # None, or a word such as "unknown"
        return None


def cache_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", value.strip().lower()).strip("_") or "robot"


# ----------------------------------------------------------------------------------------------------------------------
# The robot profile


def profile_from_data(data: dict[str, Any], robot: str, display: str) -> RobotEmbodimentSpec:
    """A robot profile from the LLM's (or the cache's) JSON, each field coerced to its type."""
    robot = str(data.get("robot") or robot)
    display = str(data.get("robot_display_name") or display)

    def text(key: str, default: str | None) -> str | None:
        return default if data.get(key) is None else str(data[key])

    return RobotEmbodimentSpec(
        robot=robot,
        robot_display_name=display,
        embodiment_type=text("embodiment_type", "general_robot"),
        description=text("description", f"General robot embodiment for {display}."),
        task_object_capacity_kg=as_float(data.get("task_object_capacity_kg")),
        notes=text("notes", None),
        **{key: as_list(data.get(key)) for key in LIST_FIELDS},
        **{key: as_bool(data.get(key)) for key in BOOL_FIELDS},
    )


def profile_reply(messages: list[dict[str, str]], args: Any, *, max_tokens: int, temperature: float) -> str:
    return llm.chat_completion(
        endpoint=os.environ.get("FOSA_LLM_ENDPOINT", llm.DEFAULT_ENDPOINT),
        model=args.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=llm.default_api_key(),
        timeout_s=120.0,
    )


def json_object_text(content: str) -> str:
    """The outermost braces of a reply (inside its ``` block, if it has one)."""
    match = re.search(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"LLM reply did not contain a JSON object: {content!r}")
    return content[start : end + 1]


def parse_profile_reply(content: str, args: Any) -> dict[str, Any]:
    """The JSON object of a robot-profile reply. Malformed JSON gets one repair pass by the LLM."""
    candidate = json_object_text(content)
    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError as error:
        print("[capability-context] warning: malformed robot-profile JSON; requesting one repair pass", flush=True)
        if len(candidate) > 20000:
            candidate = candidate[:20000] + "\n}"
        request = {
            "request": "Repair this malformed robot-profile JSON without changing its meaning.",
            "json_error": str(error),
            "malformed_json": candidate,
        }
        system = (
            "You repair malformed JSON. Return one valid JSON object only. "
            "Do not add markdown, comments, prose, or trailing commas."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
        ]
        raw = json.loads(json_object_text(profile_reply(messages, args, max_tokens=2048, temperature=0.0)))
    return raw if isinstance(raw, dict) else {}


def query_llm_robot_profile(args: Any, robot: str, display: str) -> RobotEmbodimentSpec:
    prompt = {
        "request": "Infer one reusable robot embodiment profile from repository evidence. Return JSON only.",
        "schema": {item.name: item.type for item in fields(RobotEmbodimentSpec)},
        "robot": robot,
        "robot_display_name": display,
        "evidence": {
            "robot": robot,
            "robot_display_name": display,
            "simulator_capability_manifest": contracts.llm_simulator_capability_manifest(robot),
        },
        "rules": [
            "Fill every RobotEmbodimentSpec field.",
            "Do not treat locomotion and manipulation as mutually exclusive.",
            "Include every physically plausible affordance supported by evidence, because downstream curriculum "
            "generation should be diverse.",
            "Only set manipulation or locomotion affordance fields when repository evidence supports them.",
            "Prefer short normalized labels grounded in repository evidence.",
            "Put uncertainty, evidence, and any unsupported-but-conceptual ideas in notes.",
        ],
    }
    messages = [
        {
            "role": "system",
            "content": "You infer structured robot embodiment profiles for RL curriculum generation. Output JSON only.",
        },
        {"role": "user", "content": json.dumps(prompt, separators=(",", ":"), sort_keys=True)},
    ]
    raw = parse_profile_reply(profile_reply(messages, args, max_tokens=1200, temperature=0.2), args)
    profile = raw["robot_profile"] if isinstance(raw.get("robot_profile"), dict) else raw
    return profile_from_data({**profile, "robot": robot}, robot, display)


def build_or_load_robot_profile(args: Any) -> RobotEmbodimentSpec:
    """The robot's profile, from the cache or else from the LLM (then cached)."""
    robot, display = args.robot, args.robot_display_name
    cache_path = llm.model_log_root(args.model) / "robot_profiles" / f"{cache_name(robot)}.json"
    with llm.exclusive_file(cache_path):
        cached = load_json_object(cache_path)
        if cached:
            return profile_from_data(cached, robot, display)
        profile = query_llm_robot_profile(args, robot, display)
        payload = {**asdict(profile), "_generated_by": "FOSA LLM robot profile inference", "_source_robot": robot}
        llm.write_json_atomic(cache_path, payload)
        return profile


# ----------------------------------------------------------------------------------------------------------------------
# Simulator capabilities


def backend_capabilities(args: Any, profile: RobotEmbodimentSpec) -> BackendCapabilitySpec:
    """The assets, reward terms and observations the simulator manifest offers the robot (the caller adds the
    primitives, from the task registry)."""
    manifest = contracts.llm_simulator_capability_manifest(profile.robot)
    scene, robot_spec = manifest["scene_backend"], manifest["robot_spec"]
    terrain, objects = scene["terrain"], scene["objects"]
    supports_payload = bool(robot_spec["payload_support_position_m"] or robot_spec["kind"] == "hand")
    assets = {
        f"terrain:{terrain['type']}",
        *(f"terrain_algorithm:{item}" for item in terrain["algorithms"]),
        *(f"primitive_shape:{shape}" for shape in objects["primitive_shapes"]),
        *(key for key, value in objects.items() if key.startswith("supports_") and bool(value)),
        *(f"reset:{key}" for key in manifest["reset_randomization"]),
        *(f"model_randomization:{name}" for name in manifest["model_randomization"]["fields"]),
        *(["supports_carried_payload"] if supports_payload else []),
    }
    reward = manifest["reward_function"]
    rewards = {
        *(f"observable:{item}" for item in reward["observables"]),
        *(f"objective:{item}" for item in reward["objectives"]),
    }
    if not supports_payload:
        rewards = {item for item in rewards if "carried_object" not in item}
    observations = {
        *manifest["critic_observation"]["allowed_terms"],
        "policy_state",
        "robot_state_history",
        *(f"joint:{name}" for name in robot_spec["joint_names"]),
    }
    return BackendCapabilitySpec(
        backend=args.backend,
        robot=profile.robot,
        generator_robot_name=profile.robot,
        supported_assets=sorted(assets),
        supported_reward_terms=sorted(rewards),
        supported_observation_terms=sorted(observations),
    )


def runtime_capability_tokens(capabilities: dict[str, Any]) -> set[str]:
    """The backend capabilities (as asdict(BackendCapabilitySpec)) as tokens such as "asset:...", "primitive:..." and
    "execution:..." (the execution modes of the supported primitives' registered tasks)."""
    tokens = {
        f"{prefix}:{item}" for key, prefix in TOKEN_PREFIXES.items() for item in capabilities[key] if str(item).strip()
    }
    primitives = {str(item) for item in capabilities["supported_primitives"]}
    tokens.update(f"execution:{spec.execution}" for spec in registry.TASKS.values() if spec.primitive in primitives)
    return tokens


def missing_observation_modality(text: str, supported_observations: list[str]) -> str | None:
    """A perception modality (lidar or camera) the text names but the policy observations lack."""
    required = str(text).lower().replace("-", "_")
    available = " ".join(str(item).lower().replace("-", "_") for item in supported_observations)
    for name, aliases in OBSERVATION_MODALITIES.items():
        if any(alias in required for alias in aliases) and not any(alias in available for alias in aliases):
            return name
    return None


def missing_runtime_capability(text: str, capability_tokens: set[str], *, category: str = "") -> str | None:
    """A runtime effect (an external impulse, slip, a carried payload, or object manipulation for a capability or
    task) the text needs but no capability token provides."""
    required = str(text).lower().replace("-", "_")
    available = " ".join(str(item).lower().replace("-", "_") for item in capability_tokens)
    for name, (mentions, evidence) in RUNTIME_REQUIREMENTS.items():
        if any(item in required for item in mentions) and not any(item in available for item in evidence):
            return name
    needs_manipulation = any(item in required for item in MANIPULATION_PHRASES)
    if category in {"capability", "task"} and needs_manipulation and "execution:manipulation" not in available:
        return "object_manipulation"
    return None
