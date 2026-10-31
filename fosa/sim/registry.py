"""Task registry: one task per robot and primitive, with its PPO settings and environment class."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fosa.sim import scene_contract


@dataclass(frozen=True)
class TaskSpec:
    name: str
    robot: str
    primitive: str
    execution: str
    description: str
    num_timesteps: int
    num_envs: int
    episode_length: int
    unroll_length: int
    batch_size: int
    num_minibatches: int
    num_updates_per_batch: int
    learning_rate: float
    entropy_cost: float
    discounting: float
    reward_scaling: float
    clipping_epsilon: float
    max_grad_norm: float
    num_evals: int
    num_eval_envs: int
    policy_hidden_layers: tuple[int, ...]
    value_hidden_layers: tuple[int, ...]
    training_metrics_steps: int


# Task primitives per robot morphology: adding a robot to ROBOT_SPECS auto-registers
# its tasks; the first primitive is the robot's default task.
TASK_PRIMITIVES_BY_KIND = {
    "legged": ("velocity",),
    "humanoid": ("velocity",),
    "wheeled": ("velocity",),
    "hand": ("rotate_z", "reorient"),
}
_PRIMITIVE_DESCRIPTIONS = {
    "velocity": "velocity tracking",
    "rotate_z": "object z-axis rotation",
    "reorient": "object reorientation",
}
_PRIMITIVE_EXECUTION = {"velocity": "velocity", "rotate_z": "manipulation", "reorient": "manipulation"}
_PPO_DEFAULTS = dict(
    num_timesteps=200_000_000,
    num_envs=8192,
    episode_length=1000,
    unroll_length=20,
    batch_size=256,
    num_minibatches=32,
    num_updates_per_batch=4,
    learning_rate=3.0e-4,
    entropy_cost=1.0e-2,
    discounting=0.97,
    reward_scaling=1.0,
    clipping_epsilon=0.2,
    max_grad_norm=1.0,
    num_evals=10,
    num_eval_envs=128,
    policy_hidden_layers=(512, 256, 128),
    value_hidden_layers=(512, 256, 128),
    training_metrics_steps=1_000_000,
)
_PPO_BY_KIND: dict[str, dict[str, Any]] = {
    "wheeled": {"entropy_cost": 5.0e-3},
    "humanoid": {"entropy_cost": 5.0e-3},
    "hand": {"unroll_length": 40},
}
_PPO_BY_PRIMITIVE: dict[str, dict[str, Any]] = {"reorient": {"discounting": 0.99}}


def _build_tasks() -> dict[str, TaskSpec]:
    from fosa.sim.assets import ROBOT_SPECS

    tasks: dict[str, TaskSpec] = {}
    for robot, robot_spec in ROBOT_SPECS.items():
        for primitive in TASK_PRIMITIVES_BY_KIND[robot_spec.kind]:
            name = f"{robot}_{primitive}"
            params = {**_PPO_DEFAULTS, **_PPO_BY_KIND.get(robot_spec.kind, {}), **_PPO_BY_PRIMITIVE.get(primitive, {})}
            tasks[name] = TaskSpec(
                name=name,
                robot=robot,
                primitive=primitive,
                execution=_PRIMITIVE_EXECUTION[primitive],
                description=f"{robot.replace('_', ' ').title()} {_PRIMITIVE_DESCRIPTIONS[primitive]}.",
                **params,
            )
    return tasks


TASKS: dict[str, TaskSpec] = _build_tasks()
DEFAULT_ROBOT_TASKS: dict[str, str] = {}
for _spec in TASKS.values():
    DEFAULT_ROBOT_TASKS.setdefault(_spec.robot, _spec.name)


def default_robot() -> str:
    return next(iter(DEFAULT_ROBOT_TASKS))


def default_task_for_robot(robot: str) -> str:
    return DEFAULT_ROBOT_TASKS[robot]


# Every critic term the environments implement; a generated spec may select a subset.
DEFAULT_CRITIC_OBSERVATION_TERMS = (
    "policy_state",
    "qpos",
    "qvel",
    "ctrl",
    "task_state",
    "body_pose",
    "object_state",
    "contact_state",
)

# Reset keys each execution base needs; the LLM authors their values.
REQUIRED_RESET_KEYS_BY_EXECUTION = {"velocity": ("command",), "manipulation": ("manipulated_object_pose",)}


def required_reset_keys_for_task(task: str) -> tuple[str, ...]:
    return REQUIRED_RESET_KEYS_BY_EXECUTION[TASKS[task].execution]


def environment_class_for_task(task: str) -> type:
    """Select an environment adapter by task semantics and robot morphology."""

    from fosa.sim.assets import ROBOT_SPECS
    from fosa.sim.envs import JackalVelocityEnv, ManipulationEnv, VelocityEnv

    spec = TASKS[task]
    if spec.execution == "manipulation":
        return ManipulationEnv
    if ROBOT_SPECS[spec.robot].kind == "wheeled":
        return JackalVelocityEnv
    return VelocityEnv


def make_env(
    task: str,
    *,
    impl: str = "warp",
    scene_spec: str | Path | dict[str, Any] | None = None,
    contact_capacity: dict[str, Any] | None = None,
) -> object:
    """Build the task's environment from a generated scene spec (a path or its parsed JSON)."""
    spec = TASKS[task]
    payload = _load_scene_spec_payload(scene_spec)
    training_scene = payload.get("training_scene")
    return environment_class_for_task(task)(
        spec.robot,
        impl=impl,
        training_scene=None if training_scene is None else scene_contract.normalize_training_scene(training_scene),
        critic_terms=tuple(payload.get("critic_observation", {}).get("terms", ())) or DEFAULT_CRITIC_OBSERVATION_TERMS,
        domain_randomization=payload.get(
            "domain_randomization", {"enabled": True, "reset": {}, "model": {"operations": []}}
        ),
        task_reward=payload.get("task_reward", {"reward_function": {"components": []}, "scale_by_dt": True}),
        task_termination=payload.get("task_termination", {"terminate_when": [], "grace_steps": 0}),
        contact_capacity=contact_capacity,
    )


def _load_scene_spec_payload(scene_spec: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if scene_spec is None:
        return {}
    if isinstance(scene_spec, dict):
        return scene_spec
    return json.loads(Path(scene_spec).read_text(encoding="utf-8"))
