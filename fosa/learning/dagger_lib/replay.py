"""Persistent aggregated data and balanced distributed sampling."""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import Dataset, Sampler


class BalancedDistributedSampler(Sampler[int]):
    """Draw one globally task-balanced epoch, sharded without rank overlap."""

    def __init__(self, task_ids: torch.Tensor, *, rank: int = 0, world_size: int = 1, seed: int = 0) -> None:
        _, inverse, counts = torch.unique(task_ids, return_inverse=True, return_counts=True)
        self.weights = counts[inverse].float().reciprocal()
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(len(task_ids) / self.world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total = self.num_samples * self.world_size
        indices = torch.multinomial(self.weights, total, replacement=True, generator=generator)
        return iter(indices[self.rank : total : self.world_size].tolist())

    def __len__(self) -> int:
        return self.num_samples


class ReplayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Append-only global replay backed by immutable per-rank round shards."""

    def __init__(self) -> None:
        self._shards: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._offsets = [0]
        self._task_ids: torch.Tensor | None = None
        self._loaded_shards: set[Path] = set()

    def load_shards(self, replay_dir: Path, max_round: int | None = None) -> None:
        for path in sorted(replay_dir.glob("round_*.pt")):
            match = re.match(r"round_(\d+)", path.name)
            if max_round is not None and match and int(match.group(1)) > max_round:
                continue
            self.load_shard(path)

    def load_round(self, replay_dir: Path, round_index: int) -> None:
        for path in sorted(replay_dir.glob(f"round_{round_index:03d}_rank_*.pt")):
            self.load_shard(path)

    def load_shard(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved in self._loaded_shards:
            return
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        task_ids = payload["task_ids"]
        self._shards.append((payload["histories"].nan_to_num(), payload["actions"].nan_to_num(), task_ids))
        self._offsets.append(self._offsets[-1] + len(task_ids))
        self._task_ids = None
        self._loaded_shards.add(resolved)

    @staticmethod
    def save_shard(path: Path, histories: torch.Tensor, actions: torch.Tensor, task_ids: torch.Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "histories": histories.cpu().to(torch.float16),
                "actions": actions.cpu().float(),
                "task_ids": task_ids.cpu().long(),
            },
            temporary,
        )
        temporary.replace(path)

    def normalizer_stats(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observation_mean, observation_std = self._streaming_stats(
            histories.reshape(-1, histories.shape[-1]) for histories, _actions, _task_ids in self._shards
        )
        action_mean, action_std = self._streaming_stats(actions for _histories, actions, _task_ids in self._shards)
        return observation_mean, observation_std, action_mean, action_std

    @staticmethod
    def _streaming_stats(tensors: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        total = 0
        value_sum: torch.Tensor | None = None
        square_sum: torch.Tensor | None = None
        for tensor in tensors:
            for chunk in tensor.reshape(-1, tensor.shape[-1]).split(65_536):
                values = chunk.double()
                if value_sum is None:
                    value_sum = torch.zeros(values.shape[-1], dtype=torch.float64)
                    square_sum = torch.zeros_like(value_sum)
                value_sum += values.sum(dim=0)
                square_sum += values.square().sum(dim=0)
                total += len(values)
        mean = value_sum / total
        variance = (square_sum / total - mean.square()).clamp_min(0.0)
        return mean.float(), variance.sqrt().float()

    def balanced_sampler(self, *, rank: int = 0, world_size: int = 1, seed: int = 0) -> BalancedDistributedSampler:
        return BalancedDistributedSampler(self.task_ids, rank=rank, world_size=world_size, seed=seed)

    def task_counts(self) -> dict[int, int]:
        task_ids, counts = torch.unique(self.task_ids, return_counts=True)
        return {int(task_id): int(count) for task_id, count in zip(task_ids, counts, strict=True)}

    def __len__(self) -> int:
        return self._offsets[-1]

    @property
    def task_ids(self) -> torch.Tensor:
        if self._task_ids is None:
            parts = [task_ids for _histories, _actions, task_ids in self._shards]
            if not parts:
                self._task_ids = torch.empty(0, dtype=torch.long)
            elif len(parts) == 1:
                self._task_ids = parts[0]
            else:
                self._task_ids = torch.cat(parts)
        return self._task_ids

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(index)
        if index < 0:
            index += len(self)
        shard_index = bisect_right(self._offsets, index) - 1
        local_index = index - self._offsets[shard_index]
        histories, actions, task_ids = self._shards[shard_index]
        return (histories[local_index].float(), actions[local_index], task_ids[local_index])
