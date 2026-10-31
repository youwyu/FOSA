"""Short-lived simulator worker for bounded-memory DAgger collection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import torch

from fosa.learning.dagger_lib.collector import collect_task_data
from fosa.learning.dagger_lib.model import HistoryPolicy
from fosa.learning.dagger_lib.replay import ReplayDataset
from fosa.learning.dagger_lib.text import TaskText
from fosa.learning.dagger_lib.types import RolloutStats, TeacherSpec


def run_collection_worker(prefix: Path, job: dict[str, object]) -> RolloutStats:
    """Run one teacher in a process whose native compiler arenas can be reclaimed."""

    job_path = prefix.with_suffix(".job.json")
    stats_path = prefix.with_suffix(".stats.json")
    job_path.write_text(json.dumps({**job, "stats_path": str(stats_path)}), encoding="utf-8")
    subprocess.run([sys.executable, "-m", __name__, "--job", str(job_path)], check=True)
    return RolloutStats(**json.loads(stats_path.read_text(encoding="utf-8")))


def run_job(path: Path) -> None:
    job = json.loads(path.read_text(encoding="utf-8"))
    teacher = TeacherSpec.from_json(job["teacher"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_data = torch.load(job["policy_path"], map_location=device, weights_only=True)
    student = HistoryPolicy(**policy_data["model_config"]).to(device).eval()
    student.load_state_dict(policy_data["model_state_dict"])

    text_data = torch.load(job["task_text_path"], map_location="cpu", weights_only=True)
    task_index = text_data["keys"].index(teacher.key)
    text = TaskText(
        text_data["features"][task_index : task_index + 1], text_data["padding"][task_index : task_index + 1]
    )
    histories, actions, task_ids, stats = collect_task_data(
        teacher,
        student=student,
        text=text,
        device=device,
        task_id=int(job["task_id"]),
        history_horizon=int(job["history_horizon"]),
        num_envs=int(job["num_envs"]),
        max_samples=int(job["max_samples"]),
        beta=float(job["beta"]),
        seed=int(job["seed"]),
        impl=job.get("impl"),
    )
    ReplayDataset.save_shard(Path(job["shard_path"]), histories, actions, task_ids)
    Path(job["stats_path"]).write_text(json.dumps(asdict(stats)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    run_job(parser.parse_args().job)


if __name__ == "__main__":
    main()
