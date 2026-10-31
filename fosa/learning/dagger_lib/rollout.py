"""Rollout sampling utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def actor_observation(observation: Any, size: int | None = None) -> np.ndarray:
    if isinstance(observation, dict):
        observation = observation["state"]
    result = np.asarray(observation, dtype=np.float32)
    result = result.reshape(result.shape[0], -1)
    if size is not None:
        result = np.pad(result, ((0, 0), (0, size - result.shape[-1])))
    return result


def device_actor_observation(observation: Any, size: int | None = None) -> Any:
    """Keep simulator observations in JAX device memory during GPU rollout."""

    import jax.numpy as jp

    if isinstance(observation, dict):
        observation = observation["state"]
    result = jp.asarray(observation, dtype=jp.float32)
    result = result.reshape(result.shape[0], -1)
    if size is not None:
        result = jp.pad(result, ((0, 0), (0, size - result.shape[-1])))
    return result


def jax_to_torch(value: Any) -> Any:
    """Share a CUDA allocation between JAX and PyTorch without a host copy."""

    import torch

    return torch.utils.dlpack.from_dlpack(value)


def torch_to_jax(value: Any) -> Any:
    """Share a PyTorch CUDA allocation with JAX without a host copy."""

    import jax.dlpack

    return jax.dlpack.from_dlpack(value)


def rollout_sample_schedule(
    rollout_steps: int, num_envs: int, max_samples: int, rng: np.random.Generator
) -> dict[int, int]:
    """Distribute replay samples across a small set of episode time slices."""

    samples_per_step = max(math.ceil(max_samples / rollout_steps), min(num_envs, 64))
    batches = math.ceil(max_samples / samples_per_step)
    edges = np.linspace(0, rollout_steps, batches + 1, dtype=int)
    steps = [int(rng.integers(start, end)) for start, end in zip(edges[:-1], edges[1:], strict=True)]
    remaining = max_samples
    schedule: dict[int, int] = {}
    for step in steps:
        take = min(samples_per_step, remaining)
        schedule[step] = take
        remaining -= take
    return schedule
