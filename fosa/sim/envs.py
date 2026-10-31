"""MuJoCo Playground environments for the FOSA robots: velocity tracking and in-hand manipulation."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from fosa.sim import assets, capabilities, reward_signals, scene_contract

WARP_CCD_CAPACITY = 262_144
SIGNAL_CLIP = 100.0  # observations and rewards are clipped to [-SIGNAL_CLIP, SIGNAL_CLIP]
STANDING_COMMAND_NORM = 0.1  # below this command norm the gait clock stops
ZERO_COMMAND_NORM = 0.05  # below this command norm a "zero_command" reward term is enabled
TARGET_REACHED_M = 0.15  # within this distance of the target the command drops to zero
TARGET_SPEED_GAIN = 0.8  # commanded speed per meter of remaining distance to the target
HEADING_GAIN = 0.5  # commanded yaw rate per radian of heading error
OBJECT_DROPPED_Z_M = -0.05  # a manipulated object below this height has fallen off the hand


def _default_config(spec: assets.RobotSpec, impl: str) -> config_dict.ConfigDict:
    naconmax = spec.naconmax or 3 * spec.nconmax
    return config_dict.create(
        ctrl_dt=spec.ctrl_dt,
        sim_dt=spec.sim_dt,
        episode_length=spec.episode_length,
        action_repeat=1,
        action_scale=spec.action_scale,
        impl=impl,
        nconmax=spec.nconmax,
        naconmax=naconmax,
        naccdmax=min(naconmax, WARP_CCD_CAPACITY),
        njmax=spec.njmax,
        noise_level=0.0,
    )


def _make_data(
    mj_model: Any,
    mjx_model: mjx.Model,
    config: config_dict.ConfigDict,
    *,
    qpos: jax.Array,
    qvel: jax.Array,
    ctrl: jax.Array,
) -> mjx.Data:
    data = mjx.make_data(
        mj_model, impl=mjx_model.impl.value, naconmax=config.naconmax, naccdmax=config.naccdmax, njmax=config.njmax
    )
    return data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)


def _uniform(rng: jax.Array, shape: tuple[int, ...], lo: float, hi: float) -> jax.Array:
    return jax.random.uniform(rng, shape, minval=lo, maxval=hi)


def _range2(section: dict[str, Any], key: str) -> tuple[float, float]:
    value = section.get(key)
    return tuple(value) if isinstance(value, list) else (0.0, 0.0)


def _offset_range(section: dict[str, Any], key: str, origin: float) -> tuple[float, float]:
    lo, hi = _range2(section, key)
    return origin + lo, origin + hi


def _float_value_from(section: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = section.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def _safe_metric_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.lower()).strip("_")


def _normalize_reward_components(task_reward: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return _uniquify_component_metrics(
        [{**raw, "metric": _safe_metric_name(raw["id"])} for raw in task_reward["reward_function"]["components"]]
    )


def _uniquify_component_metrics(
    components: list[dict[str, Any]] | tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    """Give every component a distinct metric name: rewards are keyed by metric, so a shared name would drop a term."""
    seen: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    for component in components:
        metric = component["metric"]
        if metric in seen:
            seen[metric] += 1
            metric = f"{metric}_{seen[metric]}"
        else:
            seen[metric] = 1
        result.append({**component, "metric": metric})
    return tuple(result)


# Body-stability regularizers for mobile robots, with weights from unitree_rl_mjlab's velocity task
# (lin_vel_z, torque and pose follow legged_gym). The LLM-authored task reward is added on top; an
# LLM component on the same observable and component indices replaces the base term.
_BASE_STABILITY_COMPONENTS = _normalize_reward_components(
    {
        "reward_function": {
            "components": [
                {
                    "id": "base_lin_vel_z",
                    "objective": "minimize_l2",
                    "observable": "root_linear_velocity_body",
                    "components": [2],
                    "weight": -2.0,
                },
                {
                    "id": "base_ang_vel_xy",
                    "objective": "minimize_l2",
                    "observable": "root_angular_velocity_body",
                    "components": [0, 1],
                    "weight": -0.05,
                },
                {
                    "id": "base_orientation",
                    "objective": "minimize_l2",
                    "observable": "projected_gravity_xy",
                    "weight": -1.0,
                },
                {
                    "id": "base_pose",
                    "objective": "minimize_l2",
                    "observable": "joint_position_deviation",
                    "weight": -0.5,
                },
                {"id": "base_torque", "objective": "minimize_l2", "observable": "joint_torque", "weight": -2.0e-4},
                {
                    "id": "base_action_rate",
                    "objective": "penalize_action_rate",
                    "observable": "action",
                    "weight": -0.05,
                },
                {
                    "id": "base_termination",
                    "objective": "maximize_value",
                    "observable": "termination",
                    "weight": -100.0,
                },
            ]
        }
    }
)
# Wheels rotate continuously: pose and torque regularization make no sense for them.
_BASE_STABILITY_COMPONENTS_WHEELED = tuple(
    component for component in _BASE_STABILITY_COMPONENTS if component["metric"] not in {"base_pose", "base_torque"}
)


def _stack_base_stability(
    components: tuple[dict[str, Any], ...], base_components: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    def key(component: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
        return component["observable"], tuple(component.get("components", ()))

    used = {key(component) for component in components}
    kept_base = tuple(component for component in base_components if key(component) not in used)
    return _uniquify_component_metrics(kept_base + tuple(components))


def _projected_gravity(data: mjx.Data, body_id: int) -> jax.Array:
    xmat = data.xmat[body_id].reshape(3, 3)
    return xmat.T @ jp.array([0.0, 0.0, -1.0])


def _body_up_z(data: mjx.Data, body_id: int) -> jax.Array:
    return data.xmat[body_id].reshape(3, 3)[2, 2]


def _fallen(data: mjx.Data, body_id: int, min_up: float) -> jax.Array:
    """Tipped past the orientation limit or numerically unstable.

    Base height is deliberately not used: on generated uneven terrain the base rises and falls with the ground.
    """
    up = _body_up_z(data, body_id)
    return (up < min_up) | jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))


def _root_lin_vel(data: mjx.Data) -> jax.Array:
    return data.qvel[0:3]


def _root_ang_vel(data: mjx.Data) -> jax.Array:
    return data.qvel[3:6]


def _body_frame_vector(data: mjx.Data, body_id: int, value: jax.Array) -> jax.Array:
    return data.xmat[body_id].reshape(3, 3).T @ value


def _sensor_data_slice(model: Any, sensor_name: str) -> slice:
    sensor_id = int(model.sensor(sensor_name).id)
    start = int(model.sensor_adr[sensor_id])
    return slice(start, start + int(model.sensor_dim[sensor_id]))


def _clip(value: jax.Array) -> jax.Array:
    return jp.clip(value, -SIGNAL_CLIP, SIGNAL_CLIP)


def _compatible_difference(value: jax.Array, target: jax.Array) -> jax.Array:
    """Subtract LLM-selected signals without letting a shape mismatch fail the JIT.

    A scalar on either side broadcasts against the first target entry; two vectors of different
    lengths are compared on their shared prefix.
    """
    value_flat = jp.ravel(jp.asarray(value, dtype=jp.float32))
    target_flat = jp.ravel(jp.asarray(target, dtype=jp.float32))
    if value_flat.size == 0 or target_flat.size == 0:
        return jp.zeros((0,), dtype=jp.float32)
    if target_flat.size == 1 or value_flat.size == 1:
        return value_flat - target_flat[0]
    shared_size = min(value_flat.size, target_flat.size)
    return value_flat[:shared_size] - target_flat[:shared_size]


def _wrap_to_pi(angle: jax.Array) -> jax.Array:
    return (angle + jp.pi) % (2.0 * jp.pi) - jp.pi


def _yaw_from_quat(quat: jax.Array) -> jax.Array:
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return jp.arctan2(siny_cosp, cosy_cosp)


def _quat_from_yaw(yaw: jax.Array) -> jax.Array:
    half_yaw = 0.5 * yaw
    return jp.array([jp.cos(half_yaw), 0.0, 0.0, jp.sin(half_yaw)])


def _quat_multiply(a: jax.Array, b: jax.Array) -> jax.Array:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jp.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _rotate_about_z(value: jax.Array, yaw: jax.Array) -> jax.Array:
    cosine, sine = jp.cos(yaw), jp.sin(yaw)
    return jp.array([cosine * value[0] - sine * value[1], sine * value[0] + cosine * value[1], value[2]])


def _gait_phase(step_count: jax.Array, command: jax.Array, dt: float, period: float) -> jax.Array:
    global_phase = jp.mod(step_count * dt, period) / period
    phase = jp.array([jp.sin(global_phase * jp.pi * 2.0), jp.cos(global_phase * jp.pi * 2.0)])
    return jp.where(jp.linalg.norm(command) < STANDING_COMMAND_NORM, jp.zeros_like(phase), phase)


def _quat_distance(a: jax.Array, b: jax.Array) -> jax.Array:
    dot = jp.clip(jp.abs(jp.dot(a, b)), 0.0, 1.0)
    return 2.0 * jp.arccos(dot)


def _goal_quat(start: jax.Array, rng: jax.Array, angle: float, axis: tuple[float, ...] | None = None) -> jax.Array:
    """A uniformly random goal pose, or, given the program's goal angle, the reset pose turned by that angle about the
    given axis (a random one if none)."""
    if angle <= 0.0:
        return _uniform_quat(rng)
    axis = jp.array(axis) if axis else jax.random.normal(rng, (3,))
    axis = axis / jp.linalg.norm(axis)
    turn = jp.concatenate([jp.array([jp.cos(0.5 * angle)]), jp.sin(0.5 * angle) * axis])
    return _quat_multiply(start, turn)


def _uniform_quat(rng: jax.Array) -> jax.Array:
    u1, u2, u3 = jax.random.uniform(rng, (3,))
    return jp.array(
        [
            jp.sqrt(1.0 - u1) * jp.sin(2.0 * jp.pi * u2),
            jp.sqrt(1.0 - u1) * jp.cos(2.0 * jp.pi * u2),
            jp.sqrt(u1) * jp.sin(2.0 * jp.pi * u3),
            jp.sqrt(u1) * jp.cos(2.0 * jp.pi * u3),
        ]
    )


class BaseFOSAEnv(mjx_env.MjxEnv):
    """Model compilation, critic observation, generated reward/termination, and scene-object resets."""

    RESET_RANDOMIZATION_KEYS: tuple[str, ...] = ()

    def __init__(
        self,
        robot: str,
        *,
        impl: str,
        training_scene: dict[str, Any] | None,
        critic_terms: tuple[str, ...],
        domain_randomization: dict[str, Any],
        task_reward: dict[str, Any],
        task_termination: dict[str, Any],
        contact_capacity: dict[str, Any] | None,
    ):
        self._robot = robot
        self._critic_terms = critic_terms
        self._domain_randomization = {
            **domain_randomization,
            "reset": capabilities.sanitize_reset_randomization(
                domain_randomization["reset"], self.RESET_RANDOMIZATION_KEYS
            ),
        }
        self._reward_components = _normalize_reward_components(task_reward)
        self._reward_scale_by_dt = task_reward["scale_by_dt"]
        self._success_condition = task_reward.get("success_condition")
        self._termination_components = tuple(task_termination["terminate_when"])
        self._termination_grace_steps = task_termination["grace_steps"]
        capacity = contact_capacity or {}
        capacity = {key: capacity[key] for key in ("nconmax", "naconmax", "njmax") if key in capacity}
        self._robot_spec = assets.robot_spec_with_capacity(assets.ROBOT_SPECS[robot], **capacity)
        self._compiled = assets.compile_robot_model(
            robot, impl=impl, training_scene_key=assets.training_scene_key(training_scene), **capacity
        )
        self._dynamic_objects = self._compiled.dynamic_objects
        base_xy = _range2(self._domain_randomization["reset"], "base_xy")
        spawn_extent = max(abs(base_xy[0]), abs(base_xy[1]))
        if spawn_extent:
            # Spawn XY randomization widens the reset footprint; recompute the spawn surface over it
            # so the robot never resets inside a static scene asset.
            self._compiled = dataclasses.replace(
                self._compiled,
                spawn_surface_height=assets.spawn_surface_height(
                    self._compiled.mj_model, spawn_radius=assets.SPAWN_RADIUS_M + spawn_extent
                ),
            )
        self._mj_model = self._compiled.mj_model
        self._mjx_model = self._compiled.mjx_model
        super().__init__(_default_config(self._robot_spec, impl))

    @property
    def xml_path(self) -> str:
        return self._compiled.xml_path

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    def _wrap_obs(self, data: mjx.Data, info: dict[str, Any], policy_obs: jax.Array) -> dict[str, jax.Array]:
        return {"state": policy_obs, "privileged_state": self._get_privileged_obs(data, info, policy_obs)}

    def _get_privileged_obs(self, data: mjx.Data, info: dict[str, Any], policy_obs: jax.Array) -> jax.Array:
        parts: list[jax.Array] = []
        for term in self._critic_terms:
            if term == "policy_state":
                parts.append(policy_obs)
            elif term == "qpos":
                parts.append(jp.ravel(data.qpos))
            elif term == "qvel":
                parts.append(jp.ravel(data.qvel))
            elif term == "ctrl":
                parts.append(jp.ravel(data.ctrl))
            elif term == "task_state":
                parts.append(self._task_state_terms(info))
            elif term == "body_pose":
                parts.extend([jp.ravel(data.xpos), jp.ravel(data.xquat)])
            elif term == "object_state":
                parts.append(self._object_state_terms(data))
            elif term == "contact_state":
                parts.append(jp.ravel(data.qfrc_constraint))
        return _clip(jp.concatenate([part.astype(policy_obs.dtype) for part in parts]))

    def _task_state_terms(self, info: dict[str, Any]) -> jax.Array:
        parts: list[jax.Array] = []
        for key in (
            "command",
            "root_initial_position",
            "target_quat",
            "heading_target",
            "is_standing",
            "step_count",
            "last_action",
            "last_last_action",
            "payload_offset",
            "dynamic_object_positions",
            "target_position",
            "manipulated_object_init_pos",
        ):
            value = info.get(key)
            if value is not None:
                parts.append(jp.ravel(jp.asarray(value, dtype=jp.float32)))
        return jp.concatenate(parts)

    def _object_state_terms(self, data: mjx.Data) -> jax.Array:
        parts: list[jax.Array] = []
        for obj in self._dynamic_objects:
            parts.extend([jp.ravel(data.qpos[jp.asarray(obj.qpos_ids)]), jp.ravel(data.qvel[jp.asarray(obj.qvel_ids)])])
        return jp.concatenate(parts) if parts else jp.zeros((0,), dtype=jp.float32)

    def _root_lin_vel_body(self, data: mjx.Data) -> jax.Array:
        return _body_frame_vector(data, self._compiled.root_body_id, _root_lin_vel(data))

    def _root_ang_vel_body(self, data: mjx.Data) -> jax.Array:
        # Rotated like the linear velocity, although MuJoCo stores free-joint angular velocity in the body frame.
        return _body_frame_vector(data, self._compiled.root_body_id, _root_ang_vel(data))

    def _root_position(self, qpos: jax.Array) -> jax.Array:
        if self._robot_spec.kind != "hand":
            return qpos[:3]
        return jp.zeros((3,), dtype=qpos.dtype)

    def _zero_reward_metrics(self) -> dict[str, jax.Array]:
        metrics = {f"reward/{component['metric']}": jp.zeros(()) for component in self._reward_components}
        if self._success_condition is not None:
            metrics["success"] = jp.zeros(())
        return metrics

    def _initial_state(
        self, data: mjx.Data, obs: jax.Array | dict[str, jax.Array], metrics: dict[str, jax.Array], info: dict[str, Any]
    ) -> mjx_env.State:
        """Create a reset state with every key that ``step`` may update."""
        if self._success_condition is not None:
            info["task_success"] = jp.zeros((), dtype=bool)
            info["task_success_streak"] = jp.zeros((), dtype=jp.int32)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def _record_task_success(
        self, metrics: dict[str, jax.Array], data: mjx.Data, action: jax.Array, info: dict[str, Any], done: jax.Array
    ) -> None:
        """Emit one binary pulse when an episode first holds its success predicate for ``hold_steps``."""
        if self._success_condition is None:
            return
        already_succeeded = info["task_success"]
        predicate_satisfied = self._termination_component(self._success_condition, data, action, info, done)
        predicate_satisfied = predicate_satisfied & ~jp.asarray(done, dtype=bool)
        streak = jp.where(predicate_satisfied, info["task_success_streak"] + 1, jp.zeros((), dtype=jp.int32))
        achieved = streak >= self._success_condition["hold_steps"]
        metrics["success"] = (achieved & ~already_succeeded).astype(jp.float32)
        info["task_success_streak"] = streak
        info["task_success"] = already_succeeded | achieved

    def _sum_rewards(self, rewards: dict[str, jax.Array]) -> jax.Array:
        reward = sum(rewards.values())
        scale = self.dt if self._reward_scale_by_dt else 1.0
        return _clip(reward * scale)

    def _get_reward(
        self, data: mjx.Data, action: jax.Array, info: dict[str, Any], done: jax.Array | None = None
    ) -> dict[str, jax.Array]:
        rewards: dict[str, jax.Array] = {}
        for component in self._reward_components:
            value = self._reward_function_component(component, data, action, info, done)
            rewards[component["metric"]] = component["weight"] * value
        return rewards

    def _reset_termination_armed(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Arm each task termination only if it is false at reset.

        A condition already true at reset (e.g. a goal distance the episode starts far from) would otherwise end
        every episode on its first step.
        """
        if not self._termination_components:
            return jp.zeros((0,), dtype=bool)
        action = jp.zeros(self.action_size)
        fired = [
            self._termination_component(component, data, action, info, jp.asarray(False))
            for component in self._termination_components
        ]
        return ~jp.stack(fired)

    def _combine_done(
        self, safety_done: jax.Array, data: mjx.Data, action: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        armed = info["termination_armed"]
        task_done = jp.asarray(False)
        for idx, component in enumerate(self._termination_components):
            fired = self._termination_component(component, data, action, info, safety_done)
            task_done = task_done | (fired & armed[idx])
        if self._termination_components and self._termination_grace_steps > 0:
            task_done = task_done & (info["step_count"] >= self._termination_grace_steps)
        return safety_done | task_done

    def _termination_component(
        self, component: dict[str, Any], data: mjx.Data, action: jax.Array, info: dict[str, Any], safety_done: jax.Array
    ) -> jax.Array:
        signal = self._reward_error(component, data, action, info, safety_done)
        flat = jp.ravel(jp.asarray(signal, dtype=jp.float32))
        condition = component["condition"]
        if condition == "greater_than":
            return jp.any(flat > component["threshold"])
        if condition == "less_than":
            return jp.any(flat < component["threshold"])
        if condition == "norm_greater_than":
            return jp.linalg.norm(flat) > component["threshold"]
        if condition == "norm_less_than":
            return jp.linalg.norm(flat) < component["threshold"]
        lo, hi = component["range"]
        if condition == "outside_range":
            return jp.any((flat < lo) | (flat > hi))
        return jp.all((flat >= lo) & (flat <= hi))

    def _reward_function_component(
        self, component: dict[str, Any], data: mjx.Data, action: jax.Array, info: dict[str, Any], done: jax.Array | None
    ) -> jax.Array:
        condition = component.get("enabled_when")
        if condition == "zero_command":
            command = info.get("command", jp.zeros((3,), dtype=jp.float32))
            enabled = jp.linalg.norm(command) < ZERO_COMMAND_NORM
        elif condition == "has_target":
            enabled = jp.asarray(info.get("target_position") is not None)
        else:
            enabled = jp.asarray(True)
        objective = component["objective"]
        if objective == "penalize_action_rate":
            value = jp.sum(jp.square(action - info["last_action"]))
            return jp.where(enabled, value, jp.zeros(()))
        signal = self._reward_error(component, data, action, info, done)
        if objective == "exp_tracking":
            std = component.get("std", 0.5)
            value = jp.exp(-jp.sum(jp.square(jp.ravel(signal))) / (std * std))
        elif objective == "minimize_l2":
            value = jp.sum(jp.square(jp.ravel(signal)))
        elif objective == "minimize_abs":
            value = jp.sum(jp.abs(jp.ravel(signal)))
        else:
            flat = jp.ravel(signal)
            value = flat[0] if flat.size == 1 else jp.mean(flat)
        return jp.where(enabled, value, jp.zeros(()))

    def _reward_error(self, component, data, action, info, done):
        name = component["observable"]
        observable = self._select_reward_components(
            self._reward_observable(name, data, action, info, done), component.get("components")
        )
        target = self._resolve_reward_value(component.get("target"), data, action, info, done)
        if target is None:
            return observable
        target = self._select_reward_components(target, component.get("components"))
        if name.endswith("/orientation") and target.size == 4 and not component.get("components"):
            target = jp.where(jp.dot(observable, target) < 0, -target, target)
        return _compatible_difference(observable, target)

    def _resolve_reward_value(
        self, value: Any, data: mjx.Data, action: jax.Array, info: dict[str, Any], done: jax.Array | None
    ) -> jax.Array | None:
        if isinstance(value, dict):
            resolved = self._reward_observable(value["observable"], data, action, info, done)
            return self._select_reward_components(resolved, value.get("components"))
        if isinstance(value, (int, float, list, tuple)):
            return jp.asarray(value, dtype=jp.float32)
        return None

    def _select_reward_components(self, value: jax.Array, components: Any) -> jax.Array:
        arr = jp.ravel(jp.asarray(value, dtype=jp.float32))
        if components:
            # Out-of-range indices are dropped; if none remain, the whole signal is used.
            valid = tuple(idx for idx in components if 0 <= int(idx) < arr.size)
            return arr[jp.array(valid, dtype=jp.int32)] if valid else arr
        return arr

    def _reward_observable(
        self, observable: str, data: mjx.Data, action: jax.Array, info: dict[str, Any], done: jax.Array | None
    ) -> jax.Array:
        object_selector = reward_signals.parse_object_signal(observable)
        if object_selector is not None:
            return self._object_reward_signal(*object_selector, data=data, info=info)
        if observable == "command":
            return info["command"]
        if observable == "root_displacement":
            return self._root_position(data.qpos) - info["root_initial_position"]
        if observable == "root_linear_velocity_body":
            return self._root_lin_vel_body(data)
        if observable == "root_angular_velocity_body":
            return self._root_ang_vel_body(data)
        if observable == "base_height":
            return data.qpos[2]
        if observable == "base_up_z":
            return _body_up_z(data, self._compiled.root_body_id)
        if observable == "projected_gravity_xy":
            return _projected_gravity(data, self._compiled.root_body_id)[:2]
        if observable == "joint_velocity":
            return data.qvel[self._qvel_ids]
        if observable == "joint_position_deviation":
            if not hasattr(self, "_default_pose"):  # wheeled robots have no default joint pose
                return jp.zeros((0,), dtype=jp.float32)
            return data.qpos[self._qpos_ids] - self._default_pose
        if observable == "joint_torque":
            return data.qfrc_actuator[self._qvel_ids]
        if observable == "action":
            return action
        if observable == "last_action":
            return info["last_action"]
        if observable == "target_position_error":
            target = info.get("target_position")
            return target - data.qpos[:2] if target is not None else jp.zeros((2,), dtype=jp.float32)
        if observable == "manipulated_object_orientation_error":
            return _quat_distance(self._manipulated_quat(data), info["target_quat"])
        if observable == "termination":
            return jp.asarray(done, dtype=jp.float32)

    def _object_reward_signal(self, name: str, quantity: str, *, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Resolve a signal for any named object in the authored scene."""
        dynamic_index = next((index for index, obj in enumerate(self._dynamic_objects) if obj.name == name), None)
        if dynamic_index is not None:
            obj = self._dynamic_objects[dynamic_index]
            qpos = data.qpos[jp.asarray(obj.qpos_ids)]
            qvel = data.qvel[jp.asarray(obj.qvel_ids)]
            position, orientation = qpos[:3], qpos[3:7]
            reset_position = info["dynamic_object_positions"][dynamic_index]
        else:
            body_id = self.mj_model.body(f"scene_{scene_contract.safe_id(name)}").id
            position = data.xpos[body_id]
            orientation = data.xquat[body_id]
            reset_position = position
            qvel = jp.zeros((6,), dtype=jp.float32)

        if quantity == "position":
            return position
        if quantity == "displacement":
            return position - reset_position
        if quantity == "height_change":
            return position[2] - reset_position[2]
        if quantity == "linear_velocity":
            return qvel[:3]
        if quantity == "angular_velocity":
            return qvel[3:6]
        if quantity == "orientation":
            return orientation
        if quantity == "distance_to_robot":
            return jp.linalg.norm(position - self._root_position(data.qpos))
        if quantity == "distance_to_palm":
            palm_site_id = self._compiled.palm_site_id
            if palm_site_id is not None:
                return jp.linalg.norm(position - data.site_xpos[palm_site_id])
        return jp.zeros(())

    def _bad_base_state(self, data: mjx.Data) -> jax.Array:
        bad = _fallen(data, self._compiled.root_body_id, self._robot_spec.termination_min_up)
        robot_position = self._root_position(data.qpos)
        for obj in self._dynamic_objects:
            if not obj.carried:
                continue
            object_pos = data.qpos[jp.asarray(obj.qpos_ids[:3])]
            relative_xy = object_pos[:2] - robot_position[:2]
            # Only explicitly carried objects participate in carried-object safety.
            bad = (
                bad
                | (object_pos[2] - robot_position[2] < -self._robot_spec.payload_drop_z)
                | (jp.linalg.norm(relative_xy) > self._robot_spec.payload_max_offset)
            )
        return bad

    def _reset_dynamic_objects(
        self, qpos: jax.Array, qvel: jax.Array, rng: jax.Array, global_carried_reset: dict[str, Any] | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Reset every compiled dynamic body from the simulator-neutral scene contract."""
        if not self._dynamic_objects:
            return qpos, qvel, jp.zeros((0, 3)), jp.zeros((0,))
        object_rngs = jax.random.split(rng, len(self._dynamic_objects))
        positions: list[jax.Array] = []
        carried_summary = jp.zeros((0,))
        for obj, object_rng in zip(self._dynamic_objects, object_rngs, strict=True):
            x_rng, y_rng, z_rng, yaw_rng = jax.random.split(object_rng, 4)
            authored_reset = {**((global_carried_reset or {}) if obj.carried else {}), **obj.reset}
            # A carried body starts on the robot's declared support: keep its yaw variation, but
            # never randomize its position off or into the carrier.
            reset = (
                {"frame": "robot", **{key: authored_reset[key] for key in ("yaw", "yaw_rad") if key in authored_reset}}
                if obj.carried
                else authored_reset
            )
            frame = reset.get("frame", "initial")
            initial = jp.asarray(obj.initial_position)
            if self._robot_spec.kind != "hand":
                robot_position = qpos[:3]
                robot_yaw = _yaw_from_quat(qpos[3:7])
                initial_robot_position = jp.array([0.0, 0.0, self._robot_spec.base_height])
            else:
                robot_position = jp.zeros((3,))
                robot_yaw = jp.zeros(())
                initial_robot_position = jp.zeros((3,))
            if frame == "robot":
                origin = robot_position
                relative = _rotate_about_z(initial - initial_robot_position, robot_yaw)
            elif frame == "world":
                origin = jp.zeros((3,))
                relative = initial
            else:
                origin = initial
                relative = jp.zeros((3,))

            position = origin + relative
            axis_rngs = (x_rng, y_rng, z_rng)
            for axis, (offset_key, absolute_key) in enumerate(
                (("x_offset_m", "x_m"), ("y_offset_m", "y_m"), ("z_offset_m", "z_m"))
            ):
                if offset_key in reset:
                    bounds = _range2(reset, offset_key)
                    position = position.at[axis].add(_uniform(axis_rngs[axis], (), bounds[0], bounds[1]))
                if absolute_key in reset:
                    bounds = _range2(reset, absolute_key)
                    position = position.at[axis].set(_uniform(axis_rngs[axis], (), bounds[0], bounds[1]))

            orientation = jp.asarray(obj.initial_orientation)
            yaw_key = "yaw_rad" if "yaw_rad" in reset else "yaw" if "yaw" in reset else None
            if yaw_key is not None:
                bounds = _range2(reset, yaw_key)
                orientation = _quat_multiply(_quat_from_yaw(_uniform(yaw_rng, (), bounds[0], bounds[1])), orientation)
            if frame == "robot":
                orientation = _quat_multiply(_quat_from_yaw(robot_yaw), orientation)

            qpos_ids = jp.asarray(obj.qpos_ids)
            qvel_ids = jp.asarray(obj.qvel_ids)
            qpos = qpos.at[qpos_ids].set(jp.concatenate([position, orientation]))
            qvel = qvel.at[qvel_ids].set(jp.zeros_like(qvel[qvel_ids]))
            positions.append(position)
            if obj.carried:
                carried_summary = jp.concatenate([position - robot_position, jp.array([0.0])])
        return qpos, qvel, jp.stack(positions), carried_summary


class VelocityEnv(BaseFOSAEnv):
    """Velocity tracking for legged and humanoid robots."""

    RESET_RANDOMIZATION_KEYS = (
        "command",
        "base_xy",
        "base_yaw",
        "base_velocity_noise",
        "task_object_pose",
        "target",
        "joint_pose_noise",
    )

    def __init__(self, robot: str, **kwargs: Any):
        super().__init__(robot, **kwargs)
        self._default_pose = jp.array(self._robot_spec.home_qpos)
        self._qpos_ids = jp.array(self._compiled.qpos_ids)
        self._qvel_ids = jp.array(self._compiled.qvel_ids)
        self._ctrl_low = jp.array(self.mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_high = jp.array(self.mj_model.actuator_ctrlrange[:, 1])
        angular_velocity_sensor = self._robot_spec.angular_velocity_sensor
        self._angular_velocity_sensor_slice = (
            _sensor_data_slice(self.mj_model, angular_velocity_sensor) if angular_velocity_sensor else None
        )
        reset_randomization = self._domain_randomization["reset"]
        command_randomization = reset_randomization.get("command", {})
        self._carried_reset = reset_randomization.get("task_object_pose", {})
        target_randomization = reset_randomization.get("target", {})
        # All values come from the LLM-authored spec; an omitted key means zero randomization.
        self._lin_x_range = _range2(command_randomization, "lin_x")
        self._lin_y_range = _range2(command_randomization, "lin_y")
        self._yaw_range = _range2(command_randomization, "yaw")
        self._standing_probability = max(
            0.0, min(1.0, _float_value_from(command_randomization, "standing_probability"))
        )
        self._base_xy_range = _range2(reset_randomization, "base_xy")
        self._base_yaw_range = _range2(reset_randomization, "base_yaw")
        self._base_velocity_noise = max(0.0, _float_value_from(reset_randomization, "base_velocity_noise"))
        self._joint_pose_noise = max(0.0, _float_value_from(reset_randomization, "joint_pose_noise"))
        self._target_enabled = bool(target_randomization)
        self._target_x_range = _range2(target_randomization, "x_offset")
        self._target_y_range = _range2(target_randomization, "y_offset")
        self._reward_components = _stack_base_stability(self._reward_components, _BASE_STABILITY_COMPONENTS)

    @property
    def action_size(self) -> int:
        return self.mj_model.nu

    def reset(self, rng: jax.Array) -> mjx_env.State:
        (
            rng,
            q_rng,
            v_rng,
            cmd_rng,
            heading_rng,
            stand_rng,
            xy_rng,
            yaw_reset_rng,
            object_rng,
            target_rng,
        ) = jax.random.split(rng, 10)
        qpos = jp.asarray(self.mj_model.qpos0)
        qvel = jp.zeros(self.mjx_model.nv)
        xy = _uniform(xy_rng, (2,), self._base_xy_range[0], self._base_xy_range[1])
        yaw = _uniform(yaw_reset_rng, (), self._base_yaw_range[0], self._base_yaw_range[1])
        spawn_z = self._robot_spec.base_height + self._compiled.spawn_surface_height
        qpos = qpos.at[0:3].set(jp.array([xy[0], xy[1], spawn_z]))
        qpos = qpos.at[3:7].set(_quat_from_yaw(yaw))
        qpos = qpos.at[self._qpos_ids].set(
            self._default_pose + self._joint_pose_noise * jax.random.normal(q_rng, self._default_pose.shape)
        )
        qpos, qvel, dynamic_positions, payload_offset = self._reset_dynamic_objects(
            qpos, qvel, object_rng, self._carried_reset
        )
        qvel = qvel.at[0:6].set(_uniform(v_rng, (6,), -self._base_velocity_noise, self._base_velocity_noise))
        ctrl = jp.clip(self._default_pose, self._ctrl_low, self._ctrl_high)

        data = _make_data(self.mj_model, self.mjx_model, self._config, qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = mjx.forward(self.mjx_model, data)

        cmd = self._sample_command(cmd_rng)
        is_standing = jax.random.bernoulli(stand_rng, self._standing_probability)
        heading_target = _uniform(heading_rng, (), -jp.pi, jp.pi) if self._robot_spec.heading_control else jp.zeros(())
        target_position = jp.zeros((0,), dtype=jp.float32)
        if self._target_enabled:
            target_position = self._sample_target_position(qpos, target_rng)
            cmd = self._target_command(data, target_position, cmd, is_standing)
        elif self._robot_spec.heading_control:
            cmd = self._heading_command(data, cmd, heading_target, is_standing)
        else:
            cmd = jp.where(is_standing, jp.zeros_like(cmd), cmd)

        info = {
            "rng": rng,
            "command": cmd,
            "root_initial_position": self._root_position(data.qpos),
            "heading_target": heading_target,
            "is_standing": is_standing,
            "step_count": jp.zeros((), dtype=jp.int32),
            "last_action": jp.zeros(self.action_size),
            "last_last_action": jp.zeros(self.action_size),
            "dynamic_object_positions": dynamic_positions,
        }
        if payload_offset.size:
            info["payload_offset"] = payload_offset
        if self._target_enabled:
            info["target_position"] = target_position
        info["termination_armed"] = self._reset_termination_armed(data, info)
        metrics = self._zero_reward_metrics()
        obs = self._wrap_obs(data, info, self._get_obs(data, info))
        return self._initial_state(data, obs, metrics, info)

    def _sample_target_position(self, qpos: jax.Array, rng: jax.Array) -> jax.Array:
        x_rng, y_rng = jax.random.split(rng)
        return qpos[:2] + jp.array(
            [
                _uniform(x_rng, (), self._target_x_range[0], self._target_x_range[1]),
                _uniform(y_rng, (), self._target_y_range[0], self._target_y_range[1]),
            ]
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        ctrl = jp.clip(self._default_pose + action * self._config.action_scale, self._ctrl_low, self._ctrl_high)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        info = dict(state.info)
        safety_done = self._bad_base_state(data)
        previous_action = info["last_action"]
        info["step_count"] = info["step_count"] + 1
        if self._target_enabled:
            info["command"] = self._target_command(data, info["target_position"], info["command"], info["is_standing"])
        elif self._robot_spec.heading_control:
            info["command"] = self._heading_command(data, info["command"], info["heading_target"], info["is_standing"])
        else:
            info["command"] = jp.where(info["is_standing"], jp.zeros_like(info["command"]), info["command"])
        info["last_last_action"] = previous_action
        info["last_action"] = action
        obs = self._wrap_obs(data, info, self._get_obs(data, info))
        reward_info = dict(info)
        reward_info["last_action"] = previous_action
        done = self._combine_done(safety_done, data, action, reward_info)
        rewards = self._get_reward(data, action, reward_info, done)
        reward = self._sum_rewards(rewards)
        metrics = dict(state.metrics)
        for key, value in rewards.items():
            metrics[f"reward/{key}"] = value
        self._record_task_success(metrics, data, action, info, done)
        return state.replace(
            data=data, obs=obs, reward=reward, done=done.astype(reward.dtype), metrics=metrics, info=info
        )

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        lin_x_rng, lin_y_rng, yaw_rng = jax.random.split(rng, 3)
        command = jp.array(
            [
                _uniform(lin_x_rng, (), self._lin_x_range[0], self._lin_x_range[1]),
                _uniform(lin_y_rng, (), self._lin_y_range[0], self._lin_y_range[1]),
                _uniform(yaw_rng, (), self._yaw_range[0], self._yaw_range[1]),
            ]
        )
        deadzone = self._robot_spec.command_deadzone
        return jp.where(jp.linalg.norm(command) > deadzone, command, jp.zeros_like(command))

    def _heading_command(
        self, data: mjx.Data, command: jax.Array, heading_target: jax.Array, is_standing: jax.Array
    ) -> jax.Array:
        yaw = _yaw_from_quat(data.qpos[3:7])
        heading_error = _wrap_to_pi(heading_target - yaw)
        yaw_command = jp.clip(HEADING_GAIN * heading_error, -1.0, 1.0)
        command = command.at[2].set(yaw_command)
        return jp.where(is_standing, jp.zeros_like(command), command)

    def _target_command(
        self, data: mjx.Data, target_position: jax.Array, command: jax.Array, is_standing: jax.Array
    ) -> jax.Array:
        delta = target_position - data.qpos[:2]
        yaw = _yaw_from_quat(data.qpos[3:7])
        body_delta = _rotate_about_z(jp.array([delta[0], delta[1], 0.0]), -yaw)[:2]
        desired_xy = jp.clip(
            TARGET_SPEED_GAIN * body_delta,
            jp.array([self._lin_x_range[0], self._lin_y_range[0]]),
            jp.array([self._lin_x_range[1], self._lin_y_range[1]]),
        )
        target_heading = jp.arctan2(delta[1], delta[0])
        if self._lin_x_range[1] < 0:
            target_heading += jp.pi
        yaw_command = jp.clip(HEADING_GAIN * _wrap_to_pi(target_heading - yaw), self._yaw_range[0], self._yaw_range[1])
        target_command = jp.array([desired_xy[0], desired_xy[1], yaw_command])
        close = jp.linalg.norm(delta) < TARGET_REACHED_M
        target_command = jp.where(close, jp.zeros_like(target_command), target_command)
        return jp.where(is_standing, jp.zeros_like(command), target_command)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        angular_velocity = (
            data.sensordata[self._angular_velocity_sensor_slice]
            if self._angular_velocity_sensor_slice is not None
            else self._root_ang_vel_body(data)
        )
        parts = [
            angular_velocity * self._robot_spec.angular_velocity_obs_scale,
            _projected_gravity(data, self._compiled.root_body_id),
            info["command"],
        ]
        if self._robot_spec.gait_phase_period is not None:
            parts.append(_gait_phase(info["step_count"], info["command"], self.dt, self._robot_spec.gait_phase_period))
        parts.extend(
            [
                data.qpos[self._qpos_ids] - self._default_pose,
                data.qvel[self._qvel_ids] * self._robot_spec.joint_velocity_obs_scale,
                info["last_action"],
            ]
        )
        return _clip(jp.concatenate(parts))


class JackalVelocityEnv(BaseFOSAEnv):
    """Velocity tracking for differential-drive wheeled robots."""

    RESET_RANDOMIZATION_KEYS = ("command", "base_xy", "base_yaw", "base_velocity_noise", "task_object_pose", "target")

    def __init__(self, robot: str, **kwargs: Any):
        super().__init__(robot, **kwargs)
        self._qvel_ids = jp.array(self._compiled.qvel_ids)
        self._wheel_ctrl_low = jp.asarray(self.mj_model.actuator_ctrlrange[:, 0])
        self._wheel_ctrl_high = jp.asarray(self.mj_model.actuator_ctrlrange[:, 1])
        reset_randomization = self._domain_randomization["reset"]
        command_randomization = reset_randomization.get("command", {})
        self._carried_reset = reset_randomization.get("task_object_pose", {})
        target_randomization = reset_randomization.get("target", {})
        # All values come from the LLM-authored spec; an omitted key means zero randomization.
        self._lin_x_range = _range2(command_randomization, "lin_x")
        self._yaw_range = _range2(command_randomization, "yaw")
        self._standing_probability = max(
            0.0, min(1.0, _float_value_from(command_randomization, "standing_probability"))
        )
        self._base_xy_range = _range2(reset_randomization, "base_xy")
        self._base_yaw_range = _range2(reset_randomization, "base_yaw")
        self._base_velocity_noise = max(0.0, _float_value_from(reset_randomization, "base_velocity_noise"))
        self._target_enabled = bool(target_randomization)
        self._target_x_range = _range2(target_randomization, "x_offset")
        self._target_y_range = _range2(target_randomization, "y_offset")
        self._reward_components = _stack_base_stability(self._reward_components, _BASE_STABILITY_COMPONENTS_WHEELED)

    @property
    def action_size(self) -> int:
        return 2

    def reset(self, rng: jax.Array) -> mjx_env.State:
        (rng, v_rng, cmd_rng, stand_rng, object_rng, xy_rng, yaw_rng, target_rng) = jax.random.split(rng, 8)
        qpos = jp.asarray(self.mj_model.qpos0)
        qvel = jp.zeros(self.mjx_model.nv)
        xy = _uniform(xy_rng, (2,), self._base_xy_range[0], self._base_xy_range[1])
        yaw = _uniform(yaw_rng, (), self._base_yaw_range[0], self._base_yaw_range[1])
        spawn_z = self._robot_spec.base_height + self._compiled.spawn_surface_height
        qpos = qpos.at[0:3].set(jp.array([xy[0], xy[1], spawn_z]))
        qpos = qpos.at[3:7].set(_quat_from_yaw(yaw))
        qpos, qvel, dynamic_positions, payload_offset = self._reset_dynamic_objects(
            qpos, qvel, object_rng, self._carried_reset
        )
        qvel = qvel.at[0:6].set(_uniform(v_rng, (6,), -self._base_velocity_noise, self._base_velocity_noise))
        ctrl = jp.zeros(self.mj_model.nu)
        data = _make_data(self.mj_model, self.mjx_model, self._config, qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = mjx.forward(self.mjx_model, data)
        is_standing = jax.random.bernoulli(stand_rng, self._standing_probability)
        cmd = self._sample_command(cmd_rng)
        target_position = jp.zeros((0,), dtype=jp.float32)
        if self._target_enabled:
            target_position = self._sample_target_position(qpos, target_rng)
            cmd = self._target_command(data, target_position, cmd, is_standing)
        else:
            cmd = jp.where(is_standing, jp.zeros_like(cmd), cmd)
        info = {
            "rng": rng,
            "command": cmd,
            "root_initial_position": self._root_position(data.qpos),
            "is_standing": is_standing,
            "step_count": jp.zeros((), dtype=jp.int32),
            "last_action": jp.zeros(self.action_size),
            "last_last_action": jp.zeros(self.action_size),
            "dynamic_object_positions": dynamic_positions,
        }
        if payload_offset.size:
            info["payload_offset"] = payload_offset
        if self._target_enabled:
            info["target_position"] = target_position
        info["termination_armed"] = self._reset_termination_armed(data, info)
        metrics = self._zero_reward_metrics()
        obs = self._wrap_obs(data, info, self._get_obs(data, info))
        return self._initial_state(data, obs, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        ctrl = self._wheel_ctrl(action)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        info = dict(state.info)
        previous_action = info["last_action"]
        info["step_count"] = info["step_count"] + 1
        if self._target_enabled:
            info["command"] = self._target_command(data, info["target_position"], info["command"], info["is_standing"])
        info["last_last_action"] = previous_action
        info["last_action"] = action
        obs = self._wrap_obs(data, info, self._get_obs(data, info))
        safety_done = self._bad_base_state(data)
        reward_info = dict(info)
        reward_info["last_action"] = previous_action
        done = self._combine_done(safety_done, data, action, reward_info)
        rewards = self._get_reward(data, action, reward_info, done)
        reward = self._sum_rewards(rewards)
        metrics = dict(state.metrics)
        for key, value in rewards.items():
            metrics[f"reward/{key}"] = value
        self._record_task_success(metrics, data, action, info, done)
        return state.replace(
            data=data, obs=obs, reward=reward, done=done.astype(reward.dtype), metrics=metrics, info=info
        )

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        lin_rng, yaw_rng = jax.random.split(rng)
        return jp.array(
            [
                _uniform(lin_rng, (), self._lin_x_range[0], self._lin_x_range[1]),
                _uniform(yaw_rng, (), self._yaw_range[0], self._yaw_range[1]),
            ]
        )

    def _sample_target_position(self, qpos: jax.Array, rng: jax.Array) -> jax.Array:
        x_rng, y_rng = jax.random.split(rng)
        return qpos[:2] + jp.array(
            [_uniform(x_rng, (), *self._target_x_range), _uniform(y_rng, (), *self._target_y_range)]
        )

    def _target_command(
        self, data: mjx.Data, target_position: jax.Array, command: jax.Array, is_standing: jax.Array
    ) -> jax.Array:
        delta = target_position - data.qpos[:2]
        reverse = self._lin_x_range[1] < 0
        heading_error = _wrap_to_pi(
            jp.arctan2(delta[1], delta[0]) - _yaw_from_quat(data.qpos[3:7]) - (jp.pi if reverse else 0.0)
        )
        speed_range = sorted(abs(value) for value in self._lin_x_range)
        linear = jp.clip(jp.linalg.norm(delta), *speed_range) * (-1 if reverse else 1)
        linear = jp.where(jp.abs(heading_error) < 0.5 * jp.pi, linear, 0.0)
        yaw = jp.clip(heading_error, *self._yaw_range)
        target_command = jp.where(jp.linalg.norm(delta) < TARGET_REACHED_M, jp.zeros(2), jp.array([linear, yaw]))
        return jp.where(is_standing, jp.zeros_like(command), target_command)

    def _wheel_ctrl(self, action: jax.Array) -> jax.Array:
        linear = action[0] * self._robot_spec.max_linear_velocity
        angular = action[1] * self._robot_spec.max_angular_velocity
        left = (linear - 0.5 * angular * self._robot_spec.track_width) / self._robot_spec.wheel_radius
        right = (linear + 0.5 * angular * self._robot_spec.track_width) / self._robot_spec.wheel_radius
        return jp.clip(jp.array([left, right, left, right]), self._wheel_ctrl_low, self._wheel_ctrl_high)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        return _clip(
            jp.concatenate(
                [
                    _root_ang_vel(data) * 0.25,
                    _projected_gravity(data, self._compiled.root_body_id),
                    info["command"],
                    data.qvel[self._qvel_ids] * 0.02,
                    info["last_action"],
                ]
            )
        )


class ManipulationEnv(BaseFOSAEnv):
    """Manipulate one LLM-authored free body with a fixed-base hand."""

    RESET_RANDOMIZATION_KEYS = ("manipulated_object_pose",)

    def __init__(self, robot: str, **kwargs: Any):
        super().__init__(robot, **kwargs)
        self._default_pose = jp.array(self._robot_spec.home_qpos)
        self._default_ctrl = jp.array(self._robot_spec.home_actions or self._robot_spec.home_qpos[: self.mj_model.nu])
        self._qpos_ids = jp.array(self._compiled.qpos_ids)
        self._qvel_ids = jp.array(self._compiled.qvel_ids)
        self._manipulated_qpos_ids = jp.array(self._compiled.manipulated_qpos_ids)
        self._manipulated_qvel_ids = jp.array(self._compiled.manipulated_qvel_ids)
        self._ctrl_low = jp.array(self.mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_high = jp.array(self.mj_model.actuator_ctrlrange[:, 1])
        manipulated_qpos_ids = tuple(int(item) for item in self._compiled.manipulated_qpos_ids)
        self._manipulated_dynamic_index = next(
            idx
            for idx, obj in enumerate(self._dynamic_objects)
            if tuple(int(item) for item in obj.qpos_ids) == manipulated_qpos_ids
        )
        object_randomization = self._domain_randomization["reset"].get("manipulated_object_pose", {})
        position_randomization = object_randomization.get("position", {})
        self._position_randomization = position_randomization
        self._uniform_object_orientation = bool(_float_value_from(object_randomization, "orientation_uniform", 0.0))
        # Set by the compiler from the program's goal angle, never by the LLM; 0 keeps a uniform goal.
        self._goal_rotation_angle = self._domain_randomization.get("goal_rotation_angle", 0.0)
        self._goal_rotation_axis = self._domain_randomization.get("goal_rotation_axis")
        # All values come from the LLM-authored spec; without them the object resets at its
        # model home pose with zero noise.
        object_home = [float(self.mj_model.qpos0[int(i)]) for i in self._compiled.manipulated_qpos_ids[:3]]
        self._hand_pose_noise = max(0.0, _float_value_from(object_randomization, "hand_pose_noise"))
        self._object_x_range = _offset_range(position_randomization, "x", object_home[0])
        self._object_y_range = _offset_range(position_randomization, "y", object_home[1])
        self._object_z_range = _offset_range(position_randomization, "z", object_home[2])

    @property
    def action_size(self) -> int:
        return self.mj_model.nu

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, hand_rng, object_rng, position_rng, orientation_rng, target_rng = jax.random.split(rng, 6)
        qpos = jp.asarray(self.mj_model.qpos0)
        qvel = jp.zeros(self.mjx_model.nv)
        hand_pose = self._default_pose + self._hand_pose_noise * jax.random.normal(hand_rng, self._default_pose.shape)
        qpos = qpos.at[self._qpos_ids].set(hand_pose)
        qpos, qvel, dynamic_positions, _ = self._reset_dynamic_objects(qpos, qvel, object_rng)
        x_rng, y_rng, z_rng = jax.random.split(position_rng, 3)
        initial_position = qpos[self._manipulated_qpos_ids[:3]]
        object_position = jp.array(
            [
                (
                    _uniform(x_rng, (), self._object_x_range[0], self._object_x_range[1])
                    if "x" in self._position_randomization
                    else initial_position[0]
                ),
                (
                    _uniform(y_rng, (), self._object_y_range[0], self._object_y_range[1])
                    if "y" in self._position_randomization
                    else initial_position[1]
                ),
                (
                    _uniform(z_rng, (), self._object_z_range[0], self._object_z_range[1])
                    if "z" in self._position_randomization
                    else initial_position[2]
                ),
            ]
        )
        object_orientation = qpos[self._manipulated_qpos_ids[3:7]]
        if self._uniform_object_orientation:
            object_orientation = _uniform_quat(orientation_rng)
        qpos = qpos.at[self._manipulated_qpos_ids].set(jp.concatenate([object_position, object_orientation]))
        qvel = qvel.at[self._manipulated_qvel_ids].set(jp.zeros_like(qvel[self._manipulated_qvel_ids]))
        dynamic_positions = dynamic_positions.at[self._manipulated_dynamic_index].set(object_position)
        ctrl = jp.clip(self._default_ctrl, self._ctrl_low, self._ctrl_high)
        data = _make_data(self.mj_model, self.mjx_model, self._config, qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = mjx.forward(self.mjx_model, data)
        info = {
            "rng": rng,
            "root_initial_position": self._root_position(data.qpos),
            "target_quat": _goal_quat(
                object_orientation, target_rng, self._goal_rotation_angle, self._goal_rotation_axis
            ),
            "manipulated_object_init_pos": object_position,
            "dynamic_object_positions": dynamic_positions,
            "step_count": jp.zeros((), dtype=jp.int32),
            "last_action": jp.zeros(self.action_size),
            "last_last_action": jp.zeros(self.action_size),
        }
        info["termination_armed"] = self._reset_termination_armed(data, info)
        metrics = self._zero_reward_metrics()
        obs = self._wrap_obs(data, info, self._get_obs(data, info))
        return self._initial_state(data, obs, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        ctrl = jp.clip(self._default_ctrl + action * self._config.action_scale, self._ctrl_low, self._ctrl_high)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        info = dict(state.info)
        previous_action = info["last_action"]
        info["step_count"] = info["step_count"] + 1
        info["last_last_action"] = previous_action
        info["last_action"] = action
        obs = self._wrap_obs(data, info, self._get_obs(data, info))
        safety_done = self._dropped(data)
        reward_info = dict(info)
        reward_info["last_action"] = previous_action
        done = self._combine_done(safety_done, data, action, reward_info)
        rewards = self._get_reward(data, action, reward_info, done)
        reward = self._sum_rewards(rewards)
        metrics = dict(state.metrics)
        for key, value in rewards.items():
            metrics[f"reward/{key}"] = value
        self._record_task_success(metrics, data, action, info, done)
        return state.replace(
            data=data, obs=obs, reward=reward, done=done.astype(reward.dtype), metrics=metrics, info=info
        )

    def _manipulated_pos(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._compiled.manipulated_body_id]

    def _manipulated_quat(self, data: mjx.Data) -> jax.Array:
        return data.xquat[self._compiled.manipulated_body_id]

    def _manipulated_qvel(self, data: mjx.Data) -> jax.Array:
        return data.qvel[self._manipulated_qvel_ids]

    def _palm_pos(self, data: mjx.Data) -> jax.Array:
        return data.site_xpos[self._compiled.palm_site_id]

    def _dropped(self, data: mjx.Data) -> jax.Array:
        return (
            (self._manipulated_pos(data)[2] < OBJECT_DROPPED_Z_M)
            | jp.any(jp.isnan(data.qpos))
            | jp.any(jp.isnan(data.qvel))
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        object_qvel = self._manipulated_qvel(data)
        return _clip(
            jp.concatenate(
                [
                    data.qpos[self._qpos_ids] - self._default_pose,
                    data.qvel[self._qvel_ids] * 0.05,
                    info["last_action"],
                    self._manipulated_pos(data) - self._palm_pos(data),
                    self._manipulated_quat(data),
                    object_qvel,
                    info["target_quat"],
                ]
            )
        )
