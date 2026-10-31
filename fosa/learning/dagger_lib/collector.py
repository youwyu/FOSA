"""Device-resident DAgger rollout collection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fosa.learning.dagger_lib.model import HistoryPolicy
from fosa.learning.dagger_lib.expert_rollout import collect_expert_rollout
from fosa.learning.dagger_lib.runtime import get_teacher_runtime
from fosa.learning.dagger_lib.rollout import (
    actor_observation,
    device_actor_observation,
    jax_to_torch,
    rollout_sample_schedule,
    torch_to_jax,
)
from fosa.learning.dagger_lib.text import TaskText
from fosa.learning.dagger_lib.types import RolloutStats, TeacherSpec


def collect_task_data(
    spec: TeacherSpec,
    *,
    student: HistoryPolicy,
    text: TaskText,
    device: torch.device,
    task_id: int,
    history_horizon: int,
    num_envs: int,
    max_samples: int,
    beta: float,
    seed: int,
    impl: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RolloutStats]:
    """Roll out a beta-mixture of teacher and student for one episode and sample teacher-labelled histories."""
    runtime = get_teacher_runtime(spec, num_envs=num_envs, seed=seed, impl=impl)
    rollout_steps = runtime.episode_length
    if device.type == "cuda" and beta >= 1.0:
        return collect_expert_rollout(
            runtime,
            task_id=task_id,
            history_horizon=history_horizon,
            observation_size=student.observation_size,
            num_envs=num_envs,
            max_samples=max_samples,
            seed=seed,
        )
    state = runtime.reset()
    gpu_rollout = device.type == "cuda"
    if gpu_rollout:
        import jax.numpy as jp

        observation = device_actor_observation(state.obs, student.observation_size)
        history = jp.repeat(observation[:, None, :], history_horizon, axis=1)
        reward_sum: Any = jp.asarray(0.0, dtype=jp.float32)
        episode_returns: Any = jp.zeros(num_envs, dtype=jp.float32)
        completed_return_sum: Any = jp.asarray(0.0, dtype=jp.float32)
        completed_episodes: Any = jp.asarray(0, dtype=jp.int32)
        text_features = text.features.to(device)
        text_padding = text.padding.to(device)
        torch_rng = torch.Generator(device=device).manual_seed(seed)
    else:
        observation = actor_observation(state.obs, student.observation_size)
        history = np.repeat(observation[:, None, :], history_horizon, axis=1)
        reward_sum = 0.0
        episode_returns = np.zeros(num_envs, dtype=np.float64)
        completed_returns: list[float] = []
    rng = np.random.default_rng(seed)
    sample_schedule = rollout_sample_schedule(rollout_steps, num_envs, max_samples, rng)
    sampled_histories: list[torch.Tensor] = []
    sampled_actions: list[torch.Tensor] = []
    sampled_task_ids: list[torch.Tensor] = []

    for step in range(rollout_steps):
        take = sample_schedule.get(step, 0)
        teacher_action = runtime.teacher_action(state.obs) if beta > 0.0 or take else None
        if take:
            indices = rng.choice(num_envs, size=take, replace=False)
            if gpu_rollout:
                sampled_histories.append(jax_to_torch(history[indices]).cpu())
                sampled_actions.append(jax_to_torch(teacher_action[indices]).cpu())
            else:
                sampled_histories.append(torch.from_numpy(np.asarray(history[indices], dtype=np.float32).copy()))
                sampled_actions.append(torch.from_numpy(np.asarray(teacher_action[indices], dtype=np.float32).copy()))
            sampled_task_ids.append(torch.full((take,), task_id, dtype=torch.long))

        if beta >= 1.0:
            action = teacher_action
        elif gpu_rollout:
            student_action = student.act_tensor(jax_to_torch(history), text_features, text_padding, device)
            if beta <= 0.0:
                action = torch_to_jax(student_action)
            else:
                expert_mask = torch.rand(num_envs, device=device, generator=torch_rng) < beta
                action = torch_to_jax(torch.where(expert_mask[:, None], jax_to_torch(teacher_action), student_action))
        else:
            student_action = student.act(history, text.features, text.padding, device)
            action = (
                student_action
                if beta <= 0.0
                else np.where(
                    (rng.random(num_envs) < beta)[:, None], np.asarray(teacher_action, dtype=np.float32), student_action
                )
            )

        state = runtime.step_fn(state, action)
        if gpu_rollout:
            rewards = state.reward.astype(jp.float32)
            dones = state.done.astype(bool)
            reward_sum += rewards.sum()
            episode_returns += rewards
            completed_return_sum += jp.where(dones, episode_returns, 0.0).sum()
            completed_episodes += dones.sum()
            episode_returns = jp.where(dones, 0.0, episode_returns)
            next_observation = device_actor_observation(state.obs, student.observation_size)
            history = jp.concatenate((history[:, 1:], next_observation[:, None, :]), axis=1)
            history = jp.where(
                dones[:, None, None], jp.repeat(next_observation[:, None, :], history_horizon, axis=1), history
            )
        else:
            rewards = np.asarray(state.reward, dtype=np.float64)
            dones = np.asarray(state.done).astype(bool)
            reward_sum += float(rewards.sum())
            episode_returns += rewards
            completed_returns.extend(episode_returns[dones].tolist())
            episode_returns[dones] = 0.0
            next_observation = actor_observation(state.obs, student.observation_size)
            history = np.concatenate((history[:, 1:], next_observation[:, None, :]), axis=1)
            if dones.any():
                history[dones] = np.repeat(next_observation[dones, None, :], history_horizon, axis=1)

    if gpu_rollout:
        reward_sum = float(reward_sum)
        completed_episodes = int(completed_episodes)
        mean_episode_return = float(completed_return_sum) / completed_episodes if completed_episodes else None
    else:
        completed_episodes = len(completed_returns)
        mean_episode_return = float(np.mean(completed_returns)) if completed_returns else None
    return (
        torch.cat(sampled_histories),
        torch.cat(sampled_actions),
        torch.cat(sampled_task_ids),
        RolloutStats(
            samples=sum(len(batch) for batch in sampled_histories),
            steps=rollout_steps,
            mean_reward=reward_sum / max(1, rollout_steps * num_envs),
            completed_episodes=completed_episodes,
            mean_episode_return=mean_episode_return,
        ),
    )
