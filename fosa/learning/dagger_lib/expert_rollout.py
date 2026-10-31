"""Fully compiled expert rollout for JAX/Warp teachers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fosa.learning.dagger_lib.rollout import device_actor_observation, jax_to_torch, rollout_sample_schedule
from fosa.learning.dagger_lib.types import RolloutStats


def collect_expert_rollout(
    runtime: Any,
    *,
    task_id: int,
    history_horizon: int,
    observation_size: int,
    num_envs: int,
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RolloutStats]:
    """Collect an expert episode in one device-side loop."""

    import jax
    import jax.numpy as jp

    rollout_steps = runtime.episode_length
    rng = np.random.default_rng(seed)
    schedule = rollout_sample_schedule(rollout_steps, num_envs, max_samples, rng)
    steps = tuple(schedule)
    counts = tuple(schedule.values())
    samples_per_step = max(counts)
    slots = np.full(rollout_steps, -1, dtype=np.int32)
    indices = np.zeros((len(steps), samples_per_step), dtype=np.int32)
    for slot, (step, count) in enumerate(zip(steps, counts, strict=True)):
        slots[step] = slot
        indices[slot, :count] = rng.choice(num_envs, size=count, replace=False)

    state = runtime.reset()
    history = jp.repeat(device_actor_observation(state.obs, observation_size)[:, None, :], history_horizon, axis=1)
    sampled_histories = jp.zeros((len(steps), samples_per_step, *history.shape[1:]), dtype=history.dtype)
    sampled_actions = jp.zeros((len(steps), samples_per_step, runtime.action_size), dtype=jp.float32)
    slots = jp.asarray(slots)
    indices = jp.asarray(indices)

    def rollout(
        state: Any,
        key: Any,
        history: Any,
        sampled_histories: Any,
        sampled_actions: Any,
        sample_slots: Any,
        sample_indices: Any,
    ) -> tuple[Any, ...]:
        def step(carry: tuple[Any, ...], step_index: Any) -> tuple[Any, None]:
            (
                state,
                key,
                history,
                reward_sum,
                episode_returns,
                completed_return_sum,
                completed_episodes,
                sampled_histories,
                sampled_actions,
            ) = carry
            key, policy_key = jax.random.split(key)
            action, _ = runtime.teacher_fn(state.obs, policy_key)
            slot = sample_slots[step_index]
            safe_slot = jp.maximum(slot, 0)

            def store(samples: tuple[Any, Any]) -> tuple[Any, Any]:
                histories, actions = samples
                selected = sample_indices[safe_slot]
                return (histories.at[safe_slot].set(history[selected]), actions.at[safe_slot].set(action[selected]))

            sampled_histories, sampled_actions = jax.lax.cond(
                slot >= 0, store, lambda samples: samples, (sampled_histories, sampled_actions)
            )
            state = runtime.step_fn(state, action)
            rewards = state.reward.astype(jp.float32)
            dones = state.done.astype(bool)
            reward_sum += rewards.sum()
            episode_returns += rewards
            completed_return_sum += jp.where(dones, episode_returns, 0.0).sum()
            completed_episodes += dones.sum()
            episode_returns = jp.where(dones, 0.0, episode_returns)
            observation = device_actor_observation(state.obs, observation_size)
            history = jp.concatenate((history[:, 1:], observation[:, None, :]), axis=1)
            history = jp.where(
                dones[:, None, None], jp.repeat(observation[:, None, :], history_horizon, axis=1), history
            )
            return (
                state,
                key,
                history,
                reward_sum,
                episode_returns,
                completed_return_sum,
                completed_episodes,
                sampled_histories,
                sampled_actions,
            ), None

        initial = (
            state,
            key,
            history,
            jp.asarray(0.0, dtype=jp.float32),
            jp.zeros(num_envs, dtype=jp.float32),
            jp.asarray(0.0, dtype=jp.float32),
            jp.asarray(0, dtype=jp.int32),
            sampled_histories,
            sampled_actions,
        )
        return jax.lax.scan(step, initial, jp.arange(rollout_steps))[0]

    result = jax.jit(rollout)(state, runtime.key, history, sampled_histories, sampled_actions, slots, indices)
    (
        _,
        runtime.key,
        _,
        reward_sum,
        _,
        completed_return_sum,
        completed_episodes,
        sampled_histories,
        sampled_actions,
    ) = result
    completed_episodes = int(completed_episodes)
    histories = torch.cat(
        [batch[:count] for batch, count in zip(jax_to_torch(sampled_histories), counts, strict=True)]
    ).cpu()
    actions = torch.cat(
        [batch[:count] for batch, count in zip(jax_to_torch(sampled_actions), counts, strict=True)]
    ).cpu()
    return (
        histories,
        actions,
        torch.full((max_samples,), task_id, dtype=torch.long),
        RolloutStats(
            samples=max_samples,
            steps=rollout_steps,
            mean_reward=float(reward_sum) / max(1, rollout_steps * num_envs),
            completed_episodes=completed_episodes,
            mean_episode_return=(float(completed_return_sum) / completed_episodes if completed_episodes else None),
        ),
    )
