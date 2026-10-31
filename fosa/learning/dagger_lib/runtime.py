"""Teacher discovery and the cached simulator runtime that rolls out one PPO teacher."""

from __future__ import annotations

import functools
import gc
import json
import math
from pathlib import Path
from typing import Any

from fosa import llm
from fosa.learning.dagger_lib.types import TeacherSpec


class BraxTeacherRuntime:
    """One generated scene, its PPO teacher, and a vectorized environment."""

    def __init__(self, spec: TeacherSpec, *, num_envs: int, seed: int, impl: str | None = None) -> None:
        from fosa.sim.warp_setup import configure_warp

        configure_warp()  # must run before MuJoCo Warp is first imported
        import jax
        from mujoco_playground import wrapper

        from fosa.learning.ppo import load_checkpoint_policy
        from fosa.sim.assets import training_contact_capacity
        from fosa.sim.domain_randomization import make_domain_randomization_fn
        from fosa.sim.registry import TASKS, make_env

        env = make_env(
            spec.task,
            impl=impl or spec.impl,
            scene_spec=spec.scene_spec,
            contact_capacity=training_contact_capacity(spec.robot, num_envs),
        )
        randomization_fn = make_domain_randomization_fn(env)
        if randomization_fn is not None:
            randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(jax.random.PRNGKey(seed + 1_000_000), num_envs)
            )
        self.episode_length = int(TASKS[spec.task].episode_length)
        self.action_size = int(spec.action_size)
        self.env = wrapper.wrap_for_brax_training(
            env, episode_length=self.episode_length, action_repeat=1, randomization_fn=randomization_fn, full_reset=True
        )
        self.num_envs = int(num_envs)
        self.key = jax.random.PRNGKey(seed)
        self.reset_fn = jax.jit(self.env.reset)
        self.step_fn = jax.jit(self.env.step)
        self.teacher_fn = jax.jit(load_checkpoint_policy(spec.checkpoint))

    def reseed(self, seed: int) -> None:
        import jax

        self.key = jax.random.PRNGKey(seed)

    def reset(self) -> Any:
        import jax

        self.key, reset_key = jax.random.split(self.key)
        return self.reset_fn(jax.random.split(reset_key, self.num_envs))

    def teacher_action(self, observation: Any) -> Any:
        import jax

        self.key, policy_key = jax.random.split(self.key)
        action, _ = self.teacher_fn(observation, policy_key)
        return action


_RUNTIME_CACHE: dict[tuple[str, int, str], BraxTeacherRuntime] = {}


def clear_runtime_cache() -> None:
    used_warp = any(key[2] == "warp" for key in _RUNTIME_CACHE)
    _RUNTIME_CACHE.clear()

    from fosa.sim.assets import compile_robot_model

    compile_robot_model.cache_clear()
    if used_warp:
        from mujoco.mjx.third_party.warp._src.jax.ffi import clear_jax_callable_graph_cache

        clear_jax_callable_graph_cache()

    import jax

    jax.clear_caches()
    gc.collect()
    if used_warp:
        import warp

        warp.synchronize_device()


def get_teacher_runtime(spec: TeacherSpec, *, num_envs: int, seed: int, impl: str | None) -> BraxTeacherRuntime:
    """The runtime for ``spec``, reseeded; building a new one first frees the previous one."""
    selected_impl = impl or spec.impl
    key = (spec.key, int(num_envs), selected_impl)
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        if _RUNTIME_CACHE:
            clear_runtime_cache()
        runtime = BraxTeacherRuntime(spec, num_envs=num_envs, seed=seed, impl=selected_impl)
        _RUNTIME_CACHE[key] = runtime
    else:
        runtime.reseed(seed)
    return runtime


def discover_teachers(root: Path, *, robot: str) -> list[TeacherSpec]:
    """Discover the latest valid policy for each generated scene."""
    by_scene: dict[str, TeacherSpec] = {}
    run_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "config.json").is_file() and (path / "metrics.json").is_file()
    )
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if not math.isfinite(float(metrics["eval/episode_reward"])):
            continue
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        task_spec = config.get("task_spec") or {}
        run_robot = str(task_spec.get("robot") or "")
        if run_robot != robot:
            continue
        scene_spec = resolve_scene_path(config.get("scene_spec"), robot=robot)
        checkpoint = latest_brax_checkpoint(run_dir)
        if checkpoint is None:
            continue
        observation_size, action_size = policy_dimensions(checkpoint)
        scene = json.loads(scene_spec.read_text(encoding="utf-8"))
        description = str(scene.get("task_description") or task_spec.get("description") or run_dir.name)
        key = scene_spec.stem
        candidate = TeacherSpec(
            key=key,
            run_dir=run_dir.resolve(),
            checkpoint=checkpoint.resolve(),
            scene_spec=scene_spec.resolve(),
            task=str(config["task"]),
            robot=run_robot,
            description=description,
            observation_size=observation_size,
            action_size=action_size,
            impl=str(config.get("impl") or "warp"),
        )
        previous = by_scene.get(key)
        candidate_rank = (checkpoint_step(candidate.checkpoint), candidate.run_dir.name)
        previous_rank = (
            (checkpoint_step(previous.checkpoint), previous.run_dir.name) if previous is not None else (-1, "")
        )
        if previous is None or candidate_rank > previous_rank:
            by_scene[key] = candidate
    return sorted(by_scene.values(), key=lambda item: item.key)


def resolve_scene_path(raw_path: Any, *, robot: str) -> Path:
    """The scene spec a teacher was trained on; if the log root has moved, the same file under the current one."""
    path = Path(str(raw_path)).expanduser()
    return path if path.exists() else llm.model_log_root() / robot / "generated_envs" / path.name


def latest_brax_checkpoint(run_dir: Path) -> Path | None:
    return max(
        (path.parent for path in run_dir.glob("checkpoints/*/ppo_network_config.json")),
        key=checkpoint_step,
        default=None,
    )


def checkpoint_step(path: Path) -> int:
    return int(path.name) if path.name.isdigit() else -1


def policy_dimensions(checkpoint: Path) -> tuple[int, int]:
    """Flattened actor observation size and action size of a Brax PPO checkpoint."""
    config = json.loads((checkpoint / "ppo_network_config.json").read_text(encoding="utf-8"))
    return math.prod(config["observation_size"]["state"]["shape"]), int(config["action_size"])
