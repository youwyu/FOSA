"""Distill Brax PPO task policies into one language-conditioned history policy.

The training loop implements iterative DAgger:

1. collect an expert-only warm-start dataset;
2. train the student with balanced task sampling;
3. roll out a decaying expert/student mixture;
4. label every sampled visited state with its task expert;
5. append the samples to persistent replay and retrain.

RGB is intentionally not required.  The student observes task language and a
short history of actor observations, which provides temporal information while
remaining cheap enough for large parallel simulation batches.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader

from fosa import llm
from fosa.learning import use_scratch_tmpdir

# Environment for JAX, set before the DAgger modules below can import it.
_JAX_CACHE_ROOT_ENV = "FOSA_DAGGER_JAX_CACHE_ROOT"  # read by dagger_lib.distributed for per-GPU caches
use_scratch_tmpdir()
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    cache_root = Path(tempfile.gettempdir()) / f"fosa-jax-{os.getuid()}-{os.getenv('SLURM_JOB_ID', 'local')}"
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_root)
    os.environ[_JAX_CACHE_ROOT_ENV] = str(cache_root)
os.environ.setdefault("JAX_COMPILATION_CACHE_MAX_SIZE", str(32 * 1024**3))

from fosa.learning.dagger_lib.distributed import DistributedContext  # noqa: E402
from fosa.learning.dagger_lib.distributed import configure_runtime as configure_distributed_runtime  # noqa: E402
from fosa.learning.dagger_lib.distributed import maybe_launch_distributed  # noqa: E402
from fosa.learning.dagger_lib.collection_worker import run_collection_worker  # noqa: E402
from fosa.learning.dagger_lib.model import HistoryPolicy  # noqa: E402
from fosa.learning.dagger_lib.progress import CollectionProgress, DaggerProgress  # noqa: E402
from fosa.learning.dagger_lib.replay import ReplayDataset  # noqa: E402
from fosa.learning.dagger_lib.rollout import actor_observation  # noqa: E402
from fosa.learning.dagger_lib.text import TaskText, TaskTextEncoder, load_task_text  # noqa: E402
from fosa.learning.dagger_lib.runtime import discover_teachers, get_teacher_runtime  # noqa: E402
from fosa.learning.dagger_lib.types import TeacherSpec  # noqa: E402


@dataclass(frozen=True)
class DaggerConfig:
    rounds: int = 5
    history_horizon: int = 10
    num_envs: int = 1024
    samples_per_task: int = 4096
    epochs: int = 5
    batch_size: int = 4096
    micro_batch: int = 512  # samples per forward pass; gradients accumulate to the full batch
    hidden_dim: int = 256


CONFIG = DaggerConfig()


def train_student(
    student: nn.Module,
    replay: ReplayDataset,
    train_text: TaskText,
    *,
    context: DistributedContext,
    seed: int,
    dashboard: DaggerProgress,
) -> list[float]:
    """Behaviour-clone the replay with task-balanced sampling; returns the mean loss of each epoch."""
    policy = student_policy(student)
    if not bool(policy.normalizers_ready):
        stats = (value.to(context.device) for value in replay.normalizer_stats())
        policy.set_normalizers(*stats)
    sampler = replay.balanced_sampler(rank=context.rank, world_size=context.world_size, seed=seed)
    loader = DataLoader(
        replay,
        batch_size=min(CONFIG.batch_size, len(replay)),
        sampler=sampler,
        pin_memory=context.device.type == "cuda",
    )
    text_features = train_text.features.to(context.device)
    text_padding = train_text.padding.to(context.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    student.train()
    epoch_losses: list[float] = []
    for epoch in range(CONFIG.epochs):
        sampler.set_epoch(epoch)
        total = 0.0
        batches = 0
        for histories, actions, task_ids in loader:
            histories = histories.to(context.device, non_blocking=True)
            actions = actions.to(context.device, non_blocking=True)
            task_ids = task_ids.to(context.device, non_blocking=True)
            target = (actions - policy.action_mean) / policy.action_std
            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            for part in range(0, len(target), CONFIG.micro_batch):  # the same full-batch mean loss
                rows = slice(part, part + CONFIG.micro_batch)
                piece = (
                    F.smooth_l1_loss(
                        student(histories[rows], text_features, text_padding, task_ids[rows]),
                        target[rows],
                        reduction="sum",
                    )
                    / target.numel()
                )
                piece.backward()
                loss += float(piece.detach())
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total += loss
            batches += 1
        loss = context.mean(total / max(1, batches))
        epoch_losses.append(loss)
        dashboard.update("train", epoch + 1, f"loss={loss:.6f}")
    return epoch_losses


def student_policy(student: nn.Module) -> HistoryPolicy:
    """The policy inside a DistributedDataParallel wrapper (or the policy itself)."""
    return getattr(student, "module", student)


def reward_retention(teacher_reward: float, student_reward: float) -> float:
    """Fraction of the teacher's reward the student keeps, in [0, 1]."""
    if not np.isfinite(student_reward):  # a diverged student retained nothing
        return 0.0
    gap = np.maximum(0.0, teacher_reward - student_reward)
    return float(np.clip(1.0 - gap / max(abs(teacher_reward), 1.0e-6), 0.0, 1.0))


def evaluate_policy(
    spec: TeacherSpec,
    *,
    student: HistoryPolicy | None,
    text: TaskText,
    device: torch.device,
    history_horizon: int,
    num_envs: int,
    seed: int,
) -> dict[str, Any]:
    """Roll out the student (or, with ``student=None``, the teacher) for one episode length per env."""
    runtime = get_teacher_runtime(spec, num_envs=num_envs, seed=seed, impl="warp")
    steps = runtime.episode_length
    state = runtime.reset()
    observation_size = student.observation_size if student else spec.observation_size
    observation = actor_observation(state.obs, observation_size)
    history = np.repeat(observation[:, None, :], history_horizon, axis=1)
    reward_sum = 0.0
    success_count = 0.0
    has_success_metric = False
    episode_success = np.zeros(num_envs, dtype=bool)
    episode_returns = np.zeros(num_envs, dtype=np.float64)
    completed_returns: list[float] = []
    for _ in range(steps):
        if student is None:
            action = np.asarray(runtime.teacher_action(state.obs), dtype=np.float32)
        else:
            action = student.act(history, text.features, text.padding, device)
        state = runtime.step_fn(state, action)
        rewards = np.asarray(state.reward, dtype=np.float64)
        dones = np.asarray(state.done).astype(bool)
        success = state.metrics.get("success")
        if success is not None:
            has_success_metric = True
            episode_success |= np.asarray(success, dtype=bool)
            success_count += int(episode_success[dones].sum())
            episode_success[dones] = False
        reward_sum += float(rewards.sum())
        episode_returns += rewards
        completed_returns.extend(episode_returns[dones].tolist())
        episode_returns[dones] = 0.0
        next_observation = actor_observation(state.obs, observation_size)
        history = np.concatenate((history[:, 1:], next_observation[:, None, :]), axis=1)
        if dones.any():
            history[dones] = np.repeat(next_observation[dones, None, :], history_horizon, axis=1)
    success_rate = success_count / max(1, len(completed_returns)) if has_success_metric else None
    return {
        "mean_step_reward": reward_sum / max(1, steps * num_envs),
        "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else None,
        "completed_episodes": len(completed_returns),
        "success_rate": success_rate,
    }


def beta_for_round(round_index: int) -> float:
    """Share of expert actions in a round's rollouts: 1 for the round-0 warm start, then 0.5 decaying to 0."""
    if round_index == 0:
        return 1.0
    return 0.5 * (CONFIG.rounds - round_index) / (CONFIG.rounds - 1)


def save_student(path: Path, student: nn.Module, round_index: int, replay: ReplayDataset) -> None:
    policy = student_policy(student)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": policy.model_config,
        "model_state_dict": policy.state_dict(),
        "round": round_index,
        "replay_samples": len(replay),
        "task_counts": replay.task_counts(),
    }
    for target in (path, path.parent / "student_latest.pt"):
        # Write-then-rename: a failed write (quota, node loss) must not leave a truncated
        # checkpoint that the next resume cannot load.
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)


def load_student(path: Path, device: torch.device) -> tuple[HistoryPolicy, int]:
    payload = torch.load(path, map_location=device, weights_only=True)
    student = HistoryPolicy(**payload["model_config"]).to(device)
    student.load_state_dict(payload["model_state_dict"])
    return student, int(payload.get("round", -1))


def write_manifest(
    output_dir: Path, teachers: Sequence[TeacherSpec], args: argparse.Namespace, control_dt: float, world_size: int
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(),
        "args": vars(args),
        "config": asdict(CONFIG),
        "world_size": world_size,
        "history_seconds": CONFIG.history_horizon * control_dt,
        "train": [teacher.to_json() for teacher in teachers],
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.resume_dir:
        return Path(args.resume_dir).expanduser().resolve()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (Path(args.output_root) / args.robot / args.method / stamp).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True)
    parser.add_argument(
        "--method", default="fosa", help="Whose task policies to distil: <log root>/task_policy/<robot>/<method>."
    )
    parser.add_argument("--teacher-root", default=None, help="Another task-policy directory instead.")
    parser.add_argument("--num-teachers", type=int, default=256)
    parser.add_argument(
        "--gpu-ids", default="0", help="Comma-separated GPU ids, one rollout/DDP rank each; 'cpu' for CPU."
    )
    parser.add_argument("--output-root", default=str(llm.model_log_root() / "dagger"))
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--seed", type=int, default=1, help="Student seed (network init, data order, rollouts).")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    teacher_root = (
        Path(args.teacher_root).expanduser().resolve()
        if args.teacher_root
        else llm.model_log_root() / "task_policy" / args.robot / args.method
    )
    manifest = Path(args.resume_dir or "") / "manifest.json"
    if args.resume_dir and manifest.exists():
        teachers = [TeacherSpec.from_json(row) for row in json.loads(manifest.read_text())["train"]]
    else:
        teachers = discover_teachers(teacher_root, robot=args.robot)[: args.num_teachers]
    assert len(teachers) == args.num_teachers, f"Student comparison requires {args.num_teachers} completed teachers"
    # Shorter teacher observations are zero-padded to the longest one.
    observation_size = max(teacher.observation_size for teacher in teachers)
    action_size = teachers[0].action_size

    launched = maybe_launch_distributed(raw_argv, gpu_ids=args.gpu_ids, script=Path(__file__))
    if launched is not None:
        return launched

    context = configure_distributed_runtime(args.gpu_ids, args.seed)
    dashboard = DaggerProgress(CONFIG.rounds + 1, enabled=context.primary)
    try:
        return run_training(args, teachers, observation_size, action_size, context, dashboard)
    finally:
        dashboard.close()
        context.close()


def run_training(
    args: argparse.Namespace,
    teachers: Sequence[TeacherSpec],
    observation_size: int,
    action_size: int,
    context: DistributedContext,
    dashboard: DaggerProgress,
) -> int:
    output_value = str(make_output_dir(args)) if context.primary else None
    output_dir = Path(context.broadcast_string(output_value))
    if context.primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    context.barrier()
    replay_dir = output_dir / "replay"
    task_text_path = output_dir / "task_text.pt"
    if context.primary and not (args.resume_dir and task_text_path.exists()):
        descriptions = [teacher.description for teacher in teachers]
        # The first encoder only measures the longest description, so no task text is truncated.
        encoder = TaskTextEncoder(max_length=TaskTextEncoder().token_length(descriptions))
        encoded = encoder.encode(descriptions)
        torch.save(
            {
                "keys": [teacher.key for teacher in teachers],
                "descriptions": [teacher.description for teacher in teachers],
                "features": encoded.features,
                "padding": encoded.padding,
            },
            task_text_path,
        )
    context.barrier()
    train_text = load_task_text(task_text_path)
    text_dim = train_text.features.shape[-1]
    if context.primary:
        from fosa.sim.assets import ROBOT_SPECS

        write_manifest(output_dir, teachers, args, ROBOT_SPECS[args.robot].ctrl_dt, context.world_size)
    context.barrier()

    latest_path = output_dir / "checkpoints" / "student_latest.pt"
    if args.resume_dir and latest_path.exists():
        policy, completed_round = load_student(latest_path, context.device)
    else:
        policy = HistoryPolicy(
            observation_size=observation_size,
            action_size=action_size,
            history_horizon=CONFIG.history_horizon,
            text_dim=text_dim,
            max_text_tokens=int(train_text.features.shape[1]),
            hidden_dim=CONFIG.hidden_dim,
            num_layers=3,
            num_heads=8,
        ).to(context.device)
        completed_round = -1
    replay = ReplayDataset()
    replay.load_shards(replay_dir, max_round=completed_round)
    student = context.wrap(policy)
    dashboard.resume(max(0, completed_round + 1))

    def log(message: str) -> None:
        if context.primary and not dashboard.live:
            print(message, flush=True)

    for round_index in range(completed_round + 1, CONFIG.rounds + 1):
        beta = beta_for_round(round_index)
        dashboard.begin_round(round_index, collect=len(teachers), train=CONFIG.epochs)
        log(f"[dagger] round={round_index}/{CONFIG.rounds} beta={beta:.4f} teachers={len(teachers)}")
        collect_distributed_round(
            teachers,
            policy,
            context,
            replay,
            replay_dir,
            task_text_path,
            args,
            beta=beta,
            round_index=round_index,
            dashboard=dashboard,
        )
        losses = train_student(
            student,
            replay,
            train_text,
            context=context,
            seed=args.seed + round_index * CONFIG.epochs,
            dashboard=dashboard,
        )
        if context.primary:
            save_student(
                output_dir / "checkpoints" / f"student_round_{round_index:03d}.pt", student, round_index, replay
            )
        log(f"[dagger] round={round_index} replay={len(replay)} loss={losses[-1]:.6f}")
        context.barrier()
        dashboard.complete_round(round_index)

    log(f"[dagger] complete output_dir={output_dir}")
    return 0


def collect_distributed_round(
    teachers: Sequence[TeacherSpec],
    student: HistoryPolicy,
    context: DistributedContext,
    replay: ReplayDataset,
    replay_dir: Path,
    task_text_path: Path,
    args: argparse.Namespace,
    *,
    beta: float,
    round_index: int,
    dashboard: DaggerProgress,
) -> None:
    """Every rank claims teachers from a shared directory and writes one replay shard per teacher."""
    claim_dir = replay_dir / f".claims_{round_index:03d}"
    if context.primary:
        (replay_dir / f"round_{round_index:03d}.pt").unlink(missing_ok=True)
        shutil.rmtree(claim_dir, ignore_errors=True)
        claim_dir.mkdir(parents=True)
        for shard in replay_dir.glob(f"round_{round_index:03d}_rank_*_task_*.pt"):
            task_id = int(shard.stem.rsplit("_", 1)[-1])
            (claim_dir / f"{task_id:06d}").mkdir()
            (claim_dir / f"{task_id:06d}.done").touch()
        _save_collection_policy(claim_dir / "student.pt", student)
    context.barrier()
    progress = CollectionProgress(len(teachers), round_index, claim_dir, enabled=context.primary, dashboard=dashboard)

    for task_id, teacher in context.claim(teachers, claim_dir):
        shard = replay_dir / (f"round_{round_index:03d}_rank_{context.rank:03d}_task_{task_id:06d}.pt")
        stats = run_collection_worker(
            claim_dir / f"worker_{context.rank:03d}_{task_id:06d}",
            {
                "teacher": teacher.to_json(),
                "task_id": task_id,
                "history_horizon": CONFIG.history_horizon,
                "num_envs": CONFIG.num_envs,
                "max_samples": CONFIG.samples_per_task,
                "beta": beta,
                "seed": args.seed + 10_000 * round_index + task_id,
                "impl": "warp",
                "policy_path": str(claim_dir / "student.pt"),
                "task_text_path": str(task_text_path),
                "shard_path": str(shard),
            },
        )
        progress.complete(task_id, stats)
    context.barrier()
    progress.close()
    replay.load_round(replay_dir, round_index)
    context.barrier()
    if context.primary:
        shutil.rmtree(claim_dir)


def _save_collection_policy(path: Path, student: nn.Module) -> None:
    policy = student_policy(student)
    torch.save({"model_config": policy.model_config, "model_state_dict": policy.state_dict()}, path)


if __name__ == "__main__":
    main()
