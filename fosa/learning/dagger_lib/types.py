"""Data contracts shared by DAgger components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PATH_FIELDS = ("run_dir", "checkpoint", "scene_spec")


@dataclass(frozen=True)
class TeacherSpec:
    """One trained PPO teacher: its run, checkpoint, generated scene and policy dimensions."""

    key: str
    run_dir: Path
    checkpoint: Path
    scene_spec: Path
    task: str
    robot: str
    description: str
    observation_size: int
    action_size: int
    impl: str

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in _PATH_FIELDS:
            payload[key] = str(payload[key])
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TeacherSpec:
        return cls(**{**payload, **{key: Path(payload[key]) for key in _PATH_FIELDS}})


@dataclass
class RolloutStats:
    samples: int
    steps: int
    mean_reward: float
    completed_episodes: int
    mean_episode_return: float | None
