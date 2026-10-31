"""Minimal torchrun lifecycle for one-GPU-per-rank DAgger."""

from __future__ import annotations

import os
import random
from datetime import timedelta
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

_PHYSICAL_GPUS_ENV = "FOSA_DAGGER_PHYSICAL_GPUS"
_JAX_CACHE_ROOT_ENV = "FOSA_DAGGER_JAX_CACHE_ROOT"


def parse_gpu_ids(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "cpu":
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _configure_jax_cache(ids: tuple[str, ...], local_rank: int) -> None:
    if cache_root := os.environ.get(_JAX_CACHE_ROOT_ENV):
        device = f"gpu-{ids[local_rank]}" if ids else f"cpu-{local_rank}"
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(Path(cache_root) / device)


def maybe_launch_distributed(argv: Sequence[str], *, gpu_ids: str, script: Path) -> int | None:
    """Relaunch once under torchrun when several physical GPUs are requested."""
    ids = parse_gpu_ids(gpu_ids)
    if len(ids) < 2 or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    env[_PHYSICAL_GPUS_ENV] = ",".join(ids)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(ids)}",
        str(script.resolve()),
        *argv,
    ]
    return subprocess.run(command, env=env, check=False).returncode


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def claim(self, values: Sequence[Any], directory: Path) -> Iterator[tuple[int, Any]]:
        """Yield tasks claimed atomically (by mkdir) from a shared directory, each rank starting at its own offset."""
        for offset in range(len(values)):
            index = (self.rank + offset) % len(values)
            try:
                (directory / f"{index:06d}").mkdir()
            except FileExistsError:
                continue
            yield index, values[index]

    def mean(self, value: float) -> float:
        if not self.enabled:
            return float(value)
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item() / self.world_size)

    def broadcast_string(self, value: str | None) -> str:
        values = [value]
        if self.enabled:
            dist.broadcast_object_list(values, src=0)
        return str(values[0])

    def wrap(self, model: nn.Module) -> nn.Module:
        if not self.enabled:
            return model
        if self.device.type == "cuda":
            return DistributedDataParallel(model, device_ids=[self.device.index])
        return DistributedDataParallel(model)

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def configure_runtime(gpu_ids: str, seed: int) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ids = parse_gpu_ids(os.environ.get(_PHYSICAL_GPUS_ENV, gpu_ids))

    if world_size > 1 and ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ids[local_rank]
    elif ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    _configure_jax_cache(ids, local_rank)

    if ids:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if world_size > 1:
        # Ranks can wait on each other longer than NCCL's default 10 minutes (e.g. during the
        # 256-teacher warm start).
        dist.init_process_group(
            backend="nccl" if ids else "gloo", device_id=device if ids else None, timeout=timedelta(hours=2)
        )

    process_seed = int(seed) + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    return DistributedContext(rank, local_rank, world_size, device)
