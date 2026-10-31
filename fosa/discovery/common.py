"""Shared by the task-discovery methods: command line, robot setup, history rows, and the PPO task-policy runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from fosa import REPO_ROOT, llm, utils
from fosa.discovery import capability_context, task_templates
from fosa.sim import registry

NUM_ENVS = 4096  # PPO environments per task policy
SEED = 1
INACTIVITY_TIMEOUT_S = 8 * 3600.0  # a PPO run that writes nothing for this long is killed
SUCCESS_THRESHOLD = 0.65
MAX_DEPTH = 10  # atoms + argument refinements per skeleton
MAX_ATOMS_PER_PROGRAM = 8
MAX_HISTORY_ITEMS = 50
MIN_COMPLETE_BINDINGS = 4
DIFFICULTY_LEVELS = 5  # evenly spaced values a numeric task argument can take, min_value to max_value
NOVELTY_PAIR_WEIGHT = 0.5
RECENT_DUPLICATE_WINDOW = 100
DUPLICATE_TEXT_THRESHOLD = 0.88
DUPLICATE_ATOM_JACCARD_THRESHOLD = 0.85
TASKS_PER_GPU = 4  # PPO runs that share a GPU
BAD_GPU_EXIT_CODE = 75  # EX_TEMPFAIL: scripts/slurm.sh requeues the job on another node
FAILED_BATCHES_BEFORE_EXIT = 3


@dataclass
class BackendSpec:
    backend: str
    robot: str
    robot_display_name: str
    generator_robot_name: str
    supported_primitives: list[str]
    seed_task: str


@dataclass(kw_only=True)
class TaskRecord:
    """One history row: a proposed task and, once its task policy is trained, the results of that run."""

    iteration: int
    timestamp: float
    backend: str
    robot: str
    description: str
    primitive: str | None
    module_name: str | None
    module_path: str | None
    status: str
    raw_reward: float | None = None
    reward_budget: float | None = None
    normalized_reward: float | None = None
    normalized_reward_clipped: float | None = None
    success_rate: float | None = None
    success_curve: list[float] | None = None  # success at each PPO evaluation, first to last
    num_eval_rollouts: int | None = None
    log_dir: str | None = None
    gpu_id: str | None = None
    error: str | None = None


@dataclass
class TrainingJob:
    """A PPO subprocess training one task policy."""

    record: Any
    process: subprocess.Popen[bytes]
    started_at: float
    stdout_path: Path
    stderr_path: Path
    timed_out: bool = False


# ----------------------------------------------------------------------------------------------------------------------
# Command line and setup


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--robot", required=True)
    parser.add_argument(
        "--iterations", type=int, default=256, help="Stop once the history holds this many trained tasks."
    )
    parser.add_argument(
        "--gpu-ids", default="0", help=f"'all' or comma-separated GPU ids; {TASKS_PER_GPU} task policies train on each."
    )
    parser.add_argument("--model", default=os.environ.get("FOSA_LLM_MODEL", llm.DEFAULT_MODEL))
    return parser


def configure_gpus(args: argparse.Namespace) -> None:
    """Expose only the requested GPUs, before anything imports a GPU library, and train TASKS_PER_GPU tasks on each."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    requested = args.gpu_ids.strip()
    if requested.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif requested and requested.lower() != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = requested
    args.parallel_tasks = TASKS_PER_GPU * len(scheduler_gpu_ids(args.gpu_ids))


def robot_log_dir(robot: str, model: str | None = None) -> Path:
    return llm.model_log_root(model) / utils.safe_module_name(robot)


def generated_env_dir(args: argparse.Namespace) -> Path:
    return robot_log_dir(args.robot, args.model) / "generated_envs"


def set_run_paths(args: argparse.Namespace, method: str) -> Path:
    """The method's history file and run directory under the robot's log directory, which is returned."""
    log_dir = robot_log_dir(args.robot, args.model)
    args.history = log_dir / f"{method}_history.jsonl"
    args.run_dir = log_dir / f"{method}_runs"
    return log_dir


def setup_robot(args: argparse.Namespace) -> None:
    """Put the robot's backend, LLM-written profile, simulator capabilities and task templates on args."""
    display_name = utils.default_robot_display_name(args.robot)
    backend = BackendSpec(
        backend=f"{args.robot}_mjp",
        robot=args.robot,
        robot_display_name=display_name,
        generator_robot_name=args.robot,
        supported_primitives=[],
        seed_task=f"Train a simple simulator-feasible baseline behavior with the {display_name}.",
    )
    args.backend, args.robot_display_name, args.backend_spec = backend.backend, backend.robot_display_name, backend
    profile = capability_context.build_or_load_robot_profile(args)
    args.robot = backend.robot = profile.robot
    args.robot_display_name = backend.robot_display_name = profile.robot_display_name
    capabilities = capability_context.backend_capabilities(args, profile)
    robot_tasks = {
        spec.primitive for spec in registry.TASKS.values() if spec.robot == capabilities.generator_robot_name
    }
    capabilities.supported_primitives = sorted(robot_tasks)
    templates = task_templates.load_task_template_library(args, profile, capabilities)
    backend.backend = args.backend = capabilities.backend
    backend.generator_robot_name = capabilities.generator_robot_name
    backend.supported_primitives = list(capabilities.supported_primitives)
    args.task_templates = templates
    args.capability_context = task_templates.capability_context_payload(profile, capabilities, templates)


# ----------------------------------------------------------------------------------------------------------------------
# Names and history


def count_trained(history: list[dict[str, Any]]) -> int:
    return sum(row["status"] == "trained" for row in history)


def next_iteration(history: list[dict[str, Any]]) -> int:
    return max((row["iteration"] for row in history), default=-1) + 1


def unique_module_name(value: Any, iteration: int, history: list[dict[str, Any]], *, prefix: str) -> str:
    used = {row["module_name"] for row in history}
    # an LLM proposal often echoes an already-prefixed name from the history; do not stack prefixes
    base_name = re.sub(rf"^({re.escape(prefix)}_\d+_)+", "", utils.safe_module_name(value))
    return unused_name(utils.safe_module_name(f"{prefix}_{iteration:04d}_{base_name}"), used, 64)


def unused_name(name: str, used: set[str], limit: int) -> str:
    """name cut to limit, or with the first free _N suffix appended after the cut (so it survives it)."""
    candidate, suffix = name[:limit], 1
    while candidate in used:
        tag = f"_{suffix}"
        candidate, suffix = name[: limit - len(tag)] + tag, suffix + 1
    return candidate


def difficulty_step(spec: dict[str, Any], value: Any, direction: str) -> float | int | None:
    """The next of DIFFICULTY_LEVELS values above ("high") or below ("low") value; None past the range end."""
    low, high, current = float(spec["min_value"]), float(spec["max_value"]), float(value)
    levels = [low + (high - low) * i / (DIFFICULTY_LEVELS - 1) for i in range(DIFFICULTY_LEVELS)]
    if spec.get("kind") == "integer":
        levels = [float(round(level)) for level in levels]
    ahead = [level for level in levels if (level > current + 1e-9 if direction == "high" else level < current - 1e-9)]
    if not ahead:
        return None
    step = min(ahead) if direction == "high" else max(ahead)
    return int(step) if spec.get("kind") == "integer" else round(step, 6)


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a partial line from an interrupted write
    return rows


def append_history(record: Any, history: list[dict[str, Any]], path: Path) -> None:
    """Append a record to the history file and to the in-memory history."""
    row = utils.json_safe(asdict(record))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")
    history.append(row)


# ----------------------------------------------------------------------------------------------------------------------
# PPO task-policy runs


def training_command(module_name: str, iteration: int, args: argparse.Namespace, method: str) -> list[str]:
    spec_path = generated_env_dir(args) / f"{utils.safe_module_name(module_name)}.json"
    return [
        sys.executable,
        "-m",
        "fosa.learning.ppo",
        "--task",
        json.loads(spec_path.read_text(encoding="utf-8"))["task_id"],
        "--gpu-ids",
        "all",
        "--num-envs",
        str(NUM_ENVS),
        "--run-name",
        f"{method}_{iteration:04d}_{module_name}",
        "--log-root",
        str(llm.model_log_root(args.model) / "task_policy" / utils.safe_module_name(args.robot) / method),
        "--seed",
        str(SEED),
        "--scene-spec",
        str(spec_path),
    ]


def launch_training(record: Any, args: argparse.Namespace, method: str) -> TrainingJob:
    """Start PPO on the record's compiled task, on the record's GPU."""
    run_dir = args.run_dir / f"iter_{record.iteration:04d}_{record.module_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = run_dir / "train.out", run_dir / "train.err"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": record.gpu_id}
    env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    command = training_command(record.module_name, record.iteration, args, method)
    record.status = "training"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, env=env)
    return TrainingJob(record, process, time.time(), stdout_path, stderr_path)


def read_reward_budget(spec_path: Path) -> float:
    return json.loads(spec_path.read_text(encoding="utf-8"))["reward_budget"]


def parse_training_reward(stdout: str) -> float | None:
    eval_rewards = [float(v) for v in re.findall(r"eval/episode_reward=(\S+)", stdout)]
    return eval_rewards[-1] if eval_rewards else None


def parse_log_dir(stdout: str) -> str | None:
    matches = re.findall(r"\[mjp-ppo\].*?\blog_dir=(\S+)", stdout)
    return matches[-1] if matches else None


def parse_success_curve(stdout: str) -> list[float]:
    return [float(value) for value in re.findall(r"eval/success_rate=([0-9.]+)", stdout)]


def read_json(path: Path) -> dict[str, Any]:
    """The JSON object in path, or {} if the run did not write it."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def parse_evaluation_success(log_dir: str | None, stdout: str) -> tuple[float | None, int | None]:
    """Success rate of the final evaluation batch, and its number of rollouts."""
    metrics = read_json(Path(log_dir) / "metrics.json") if log_dir else {}
    config = read_json(Path(log_dir) / "config.json") if log_dir else {}
    value = metrics.get("eval/episode_success")
    if value is not None and math.isfinite(float(value)):
        return min(max(float(value), 0.0), 1.0), config.get("num_eval_envs")
    matches = re.findall(r"eval/success_rate=([-+0-9.eE]+)", stdout)
    if matches and math.isfinite(float(matches[-1])):
        return min(max(float(matches[-1]), 0.0), 1.0), config.get("num_eval_envs")
    return None, config.get("num_eval_envs")


def training_artifacts_complete(log_dir: str | None) -> bool:
    """Whether PPO wrote its final metrics and a checkpoint (it may still fail afterwards, e.g. at shutdown)."""
    if not log_dir:
        return False
    root = Path(log_dir)
    metrics = read_json(root / "metrics.json")
    has_checkpoint = (root / "checkpoints").is_dir() and any((root / "checkpoints").iterdir())
    return has_checkpoint and math.isfinite(float(metrics.get("eval/episode_reward", math.nan)))


def finish_training(job: TrainingJob, *, reward_budget: float, clip_normalized_reward: bool = False) -> None:
    """Read a finished PPO run into its record: status, rewards and success rates."""
    record = job.record
    returncode = job.process.poll()
    stdout = job.stdout_path.read_text(encoding="utf-8", errors="replace")
    raw_reward = parse_training_reward(stdout)
    log_dir = parse_log_dir(stdout)
    normalized = None if raw_reward is None else raw_reward / reward_budget
    clipped = None if normalized is None else min(max(normalized, 0.0), 1.0)
    if training_artifacts_complete(log_dir):
        status, error = (
            "trained",
            None if returncode == 0 else f"process exited after checkpoint with code {returncode}",
        )
    elif job.timed_out:
        status, error = "timeout", f"killed after no activity for {INACTIVITY_TIMEOUT_S}s"
    elif returncode == 0:
        status, error = "train_failed", "training exited without both final metrics and a checkpoint"
    else:
        status, error = "train_failed", f"training exited with code {returncode}"
    record.raw_reward = raw_reward
    record.log_dir = log_dir
    record.normalized_reward = clipped if clip_normalized_reward else normalized
    record.normalized_reward_clipped = clipped
    record.success_rate, record.num_eval_rollouts = parse_evaluation_success(log_dir, stdout)
    record.success_curve = parse_success_curve(stdout)
    record.status, record.error = status, error


def kill_if_inactive(job: TrainingJob) -> None:
    if job.timed_out or job.process.poll() is not None:
        return
    log_dir = parse_log_dir(job.stdout_path.read_text(encoding="utf-8", errors="replace"))
    paths = [job.stdout_path, job.stderr_path] + ([Path(log_dir) / "metrics.json"] if log_dir else [])
    last_activity = max([job.started_at] + [path.stat().st_mtime for path in paths if path.exists()])
    if time.time() - last_activity > INACTIVITY_TIMEOUT_S:
        job.timed_out = True
        job.process.kill()


_failed_batches = 0


def wait_for_training(jobs: list[TrainingJob], finish: Callable[[TrainingJob], None]) -> None:
    """Wait for every job, calling finish on each as it ends. Several batches in a row with no trained task mean a
    broken GPU, not hard tasks: exit with BAD_GPU_EXIT_CODE so the launcher requeues the run elsewhere."""
    global _failed_batches
    pending = list(jobs)
    while pending:
        for job in list(pending):
            kill_if_inactive(job)
            if job.process.poll() is not None:
                finish(job)
                pending.remove(job)
        if pending:
            time.sleep(10.0)
    if not jobs:
        return
    _failed_batches = 0 if any(job.record.status == "trained" for job in jobs) else _failed_batches + 1
    if _failed_batches >= FAILED_BATCHES_BEFORE_EXIT:
        raise SystemExit(BAD_GPU_EXIT_CODE)


def scheduler_gpu_ids(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return visible_gpu_ids() or ["0"]
    return [item.strip() for item in value.split(",") if item.strip()] or ["0"]


def batch_gpu_id(gpu_ids: list[str], slot: int) -> str:
    return gpu_ids[slot % len(gpu_ids)]


def visible_gpu_ids() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True
        )
    except OSError:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()] if result.returncode == 0 else []
