"""Scene-grounded signal vocabulary for generated reward programs.

The vocabulary describes simulator interfaces, not task semantics.  Object
signals are generated from the authored scene, so a reward can refer to any
number of named objects without adding another backend special case.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fosa.sim import registry, scene_contract


COMMON_SIGNALS = ("joint_velocity", "joint_position_deviation", "joint_torque", "action", "last_action", "termination")

MOBILE_SIGNALS = (
    "command",
    "root_displacement",
    "root_linear_velocity_body",
    "root_angular_velocity_body",
    "base_height",
    "base_up_z",
    "projected_gravity_xy",
    "target_position_error",
)

OBJECT_QUANTITIES = {
    "position": "world XYZ position",
    "displacement": "XYZ position relative to this episode's reset pose",
    "height_change": "world Z relative to this episode's reset pose",
    "linear_velocity": "world XYZ linear velocity",
    "angular_velocity": "world XYZ angular velocity",
    "orientation": "world orientation quaternion (wxyz)",
    "distance_to_robot": "Euclidean distance to the robot root",
    "distance_to_palm": "Euclidean distance to the hand grasp site",
}


def object_signal(name: str, quantity: str) -> str:
    return f"object/{name}/{quantity}"


def parse_object_signal(value: str) -> tuple[str, str] | None:
    parts = value.split("/")
    if len(parts) != 3 or parts[0] != "object" or not parts[1]:
        return None
    if parts[2] not in OBJECT_QUANTITIES:
        return None
    return parts[1], parts[2]


def core_signals_for_task(task_id: str) -> tuple[str, ...]:
    if registry.TASKS[task_id].execution == "velocity":
        return (*COMMON_SIGNALS, *MOBILE_SIGNALS)
    # A manipulation task's goal pose is only usable as the scalar geodesic error the env
    # already computes; comparing a raw quaternion to target_orientation trains unstably.
    return (*COMMON_SIGNALS, "manipulated_object_orientation_error")


def scene_object_signal_catalog(training_scene: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the signal selectors grounded in the exact authored scene."""

    catalog: list[dict[str, Any]] = []
    for obj in scene_contract.scene_objects(training_scene or {}):
        dynamic = obj.motion in {"dynamic", "carried"}
        quantities = tuple(OBJECT_QUANTITIES) if dynamic else ("position", "distance_to_robot", "distance_to_palm")
        catalog.append(
            {
                "name": obj.name,
                "role": obj.role,
                "motion": obj.motion,
                "signals": [object_signal(obj.name, item) for item in quantities],
            }
        )
    return catalog


def reward_signals_for_task(task_id: str, training_scene: dict[str, Any] | None = None) -> tuple[str, ...]:
    signals = list(core_signals_for_task(task_id))
    for item in scene_object_signal_catalog(training_scene):
        signals.extend(item["signals"])
    return tuple(dict.fromkeys(signals))


def is_reward_signal(signal: str, allowed: Iterable[str] | None = None) -> bool:
    if allowed is not None and signal not in allowed:
        return False
    return bool(
        signal in COMMON_SIGNALS
        or signal in MOBILE_SIGNALS
        or signal == "manipulated_object_orientation_error"
        or parse_object_signal(signal)
    )
