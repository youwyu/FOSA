"""Train a PPO teacher for one generated task with Brax on MuJoCo Warp (``python -m fosa.learning.ppo``)."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fosa import llm
from fosa.learning import use_scratch_tmpdir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_gpu(args.gpu_ids)
    # Importing the registry configures Warp (via fosa.sim.assets) before JAX is imported below.
    from fosa.sim.registry import TASKS, make_env

    task = TASKS[args.task]
    log_dir = make_log_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)

    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    scene_spec_path = Path(args.scene_spec).expanduser().resolve() if args.scene_spec else None
    scene_spec_path = ensure_scene_spec_ready_for_training(scene_spec_path)
    num_envs = args.num_envs or task.num_envs
    # A node can present fewer GPUs than slurm allocated: a failed device simply drops out of
    # jax.devices(). Brax asserts num_envs % device_count == 0, so trim instead of dying on a
    # node that is otherwise fine; the change is well under a percent.
    device_count = jax.local_device_count()
    if num_envs % device_count:
        print(
            f"[mjp-ppo] {device_count} visible devices: num_envs "
            f"{num_envs} -> {num_envs - num_envs % device_count}",
            flush=True,
        )
        num_envs -= num_envs % device_count
    from fosa.sim.assets import training_contact_capacity

    contact_capacity = training_contact_capacity(task.robot, num_envs)
    env = make_env(args.task, impl="warp", scene_spec=scene_spec_path, contact_capacity=contact_capacity)
    from fosa.sim.domain_randomization import make_domain_randomization_fn

    model_randomization_fn = make_domain_randomization_fn(env)

    train_config = {
        "task": args.task,
        "task_spec": asdict(task),
        "impl": "warp",
        "num_timesteps": task.num_timesteps,
        "num_envs": num_envs,
        "episode_length": task.episode_length,
        "unroll_length": task.unroll_length,
        "batch_size": task.batch_size,
        "num_minibatches": task.num_minibatches,
        "num_updates_per_batch": task.num_updates_per_batch,
        "learning_rate": task.learning_rate,
        "entropy_cost": task.entropy_cost,
        "discounting": task.discounting,
        "reward_scaling": task.reward_scaling,
        "num_evals": task.num_evals,
        "num_eval_envs": task.num_eval_envs,
        "seed": args.seed,
        "gpu_ids": args.gpu_ids,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "policy_hidden_layers": list(task.policy_hidden_layers),
        "value_hidden_layers": list(task.value_hidden_layers),
        "scene_spec": str(scene_spec_path) if scene_spec_path else None,
        "training_contact_capacity": contact_capacity,
        "domain_randomization": env._domain_randomization,
        "model_randomization": model_randomization_fn is not None,
        "full_reset": True,
    }
    (log_dir / "config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=task.policy_hidden_layers,
        value_hidden_layer_sizes=task.value_hidden_layers,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

    def wrap_for_training(
        env_to_wrap: Any, episode_length: int, action_repeat: int, randomization_fn: Any = None
    ) -> Any:
        return wrapper.wrap_for_brax_training(
            env_to_wrap,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=randomization_fn,
            full_reset=True,
        )

    progress_state = {"steps": 0, "time": time.monotonic()}

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        # fosa.discovery parses the eval/episode_reward= and eval/success_rate= fields of these lines.
        now = time.monotonic()
        interval_sps = (num_steps - progress_state["steps"]) / (now - progress_state["time"])
        progress_state.update(steps=num_steps, time=now)
        steps_per_second = float(metrics.get("training/sps", interval_sps))
        reward = metrics.get("eval/episode_reward")
        success_rate = metrics.get("eval/episode_success")
        if reward is not None and not math.isfinite(float(reward)):
            raise RuntimeError(f"non-finite eval reward at step {num_steps}; aborting instead of training on NaN")
        if reward is not None:
            success_text = "" if success_rate is None else f" eval/success_rate={float(success_rate):.6f}"
            print(
                f"{num_steps}: eval/episode_reward={float(reward):.3f} "
                f"steps_per_second={steps_per_second:.0f}{success_text}",
                flush=True,
            )
        elif "episode/sum_reward" in metrics:
            print(
                f"{num_steps}: episode/sum_reward={float(metrics['episode/sum_reward']):.3f} "
                f"steps_per_second={steps_per_second:.0f}",
                flush=True,
            )

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=train_config["num_timesteps"],
        num_envs=train_config["num_envs"],
        episode_length=train_config["episode_length"],
        action_repeat=1,
        unroll_length=train_config["unroll_length"],
        batch_size=train_config["batch_size"],
        num_minibatches=train_config["num_minibatches"],
        num_updates_per_batch=train_config["num_updates_per_batch"],
        learning_rate=train_config["learning_rate"],
        entropy_cost=train_config["entropy_cost"],
        discounting=train_config["discounting"],
        reward_scaling=train_config["reward_scaling"],
        normalize_observations=True,
        clipping_epsilon=task.clipping_epsilon,
        max_grad_norm=task.max_grad_norm,
        network_factory=network_factory,
        seed=train_config["seed"],
        wrap_env_fn=wrap_for_training,
        num_evals=train_config["num_evals"],
        num_eval_envs=train_config["num_eval_envs"],
        deterministic_eval=True,
        run_evals=True,
        log_training_metrics=True,
        training_metrics_steps=task.training_metrics_steps,
        progress_fn=progress,
        save_checkpoint_path=str(log_dir / "checkpoints"),
        randomization_fn=model_randomization_fn,
    )

    # fosa.discovery reads log_dir from this line.
    print(f"[mjp-ppo] task={args.task} impl=warp log_dir={log_dir}", flush=True)
    print(
        f"[mjp-ppo] gpu_ids={args.gpu_ids} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"jax_backend={jax.default_backend()} devices={jax.devices()}",
        flush=True,
    )
    train_start = time.monotonic()
    _, _, metrics = train_fn(environment=env)
    train_seconds = time.monotonic() - train_start
    metrics = jax.device_get(metrics)
    (log_dir / "metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=2), encoding="utf-8")
    print(f"[mjp-ppo] train_seconds={train_seconds:.2f}", flush=True)
    print("[mjp-ppo] done", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0", help="'cpu' or comma-separated CUDA ids.")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--log-root", default=str(llm.model_log_root() / "task_policy"))
    parser.add_argument(
        "--scene-spec",
        default=None,
        help="Generated task spec JSON: training scene, reward, termination, randomization and critic terms.",
    )
    return parser.parse_args(argv)


def load_checkpoint_policy(checkpoint_path: Path) -> Any:
    """Deterministic actor of a Brax PPO checkpoint, as ``policy(obs, rng) -> (action, extras)``."""
    from brax.training import networks as brax_networks
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks

    config = json.loads((checkpoint_path / "ppo_network_config.json").read_text(encoding="utf-8"))
    observation_size = {key: tuple(value["shape"]) for key, value in config["observation_size"].items()}
    network_kwargs = dict(config["network_factory_kwargs"])
    network_kwargs["activation"] = brax_networks.ACTIVATION[network_kwargs["activation"]]
    for key in ("policy_network_kernel_init_fn", "value_network_kernel_init_fn"):
        network_kwargs[key] = brax_networks.KERNEL_INITIALIZER[network_kwargs[key]]
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=config["action_size"],
        preprocess_observations_fn=running_statistics.normalize,
        **network_kwargs,
    )
    params = ppo_checkpoint.load(checkpoint_path)
    policy = ppo_networks.make_inference_fn(ppo_network)(params, deterministic=True)

    import jax.numpy as jp

    def _fit_leaf(value: Any, shape: tuple[int, ...]) -> Any:
        """Zero-pad or truncate an observation's trailing dimensions to the checkpoint's shape."""
        target_size = 1
        for dim in shape:
            target_size *= int(dim)
        value = jp.asarray(value)
        leading_shape = value.shape[: value.ndim - len(shape)]
        flat = jp.reshape(value, (*leading_shape, -1))
        current_size = int(flat.shape[-1])
        if current_size < target_size:
            padding = [(0, 0)] * (flat.ndim - 1) + [(0, target_size - current_size)]
            flat = jp.pad(flat, padding)
        fitted = flat[..., :target_size]
        return jp.reshape(fitted, (*leading_shape, *shape))

    def _policy(obs: dict[str, Any], rng: Any) -> Any:
        return policy({key: _fit_leaf(obs[key], shape) for key, shape in observation_size.items()}, rng)

    return _policy


def ensure_scene_spec_ready_for_training(path: Path | None) -> Path | None:
    if path is None:
        return path
    from fosa.compiler import scenes

    issues = scenes.scene_spec_issues(json.loads(path.read_text(encoding="utf-8")))
    if issues:
        raise RuntimeError(f"generated scene spec is not trainable: {path} issues={issues}")
    return path


def configure_gpu(gpu_ids: str) -> None:
    use_scratch_tmpdir()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if gpu_ids == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif gpu_ids != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    # JAX and Warp use separate pools; return unused blocks so Warp can allocate scratch buffers.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


def make_log_dir(args: argparse.Namespace) -> Path:
    root = Path(args.log_root).expanduser().resolve()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{args.run_name}" if args.run_name else ""
    return root / f"{stamp}{suffix}"


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value if value is None or isinstance(value, (str, int, float)) else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
