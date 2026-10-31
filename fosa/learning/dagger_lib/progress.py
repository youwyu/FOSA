"""Rank-zero DAgger progress with a sparse log fallback."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class DaggerProgress:
    """One overall bar and per-round collection and training bars (rich, on a terminal only)."""

    def __init__(self, total_rounds: int, *, enabled: bool) -> None:
        self.enabled = enabled
        self.live = False
        self._progress: Any = None
        self._overall: Any = None
        self._stages: dict[str, Any] = {}
        if not enabled or not sys.stdout.isatty():
            return
        try:
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
        except ImportError:
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description:<24}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[detail]}"),
            console=Console(file=sys.stdout),
        )
        self._progress.start()
        self._overall = self._progress.add_task("overall", total=total_rounds, detail="rounds")
        self.live = True

    def resume(self, completed_rounds: int) -> None:
        if self.live:
            self._progress.update(self._overall, completed=completed_rounds)

    def begin_round(self, round_index: int, *, collect: int, train: int) -> None:
        if not self.live:
            return
        for task in self._stages.values():
            self._progress.remove_task(task)
        self._stages = {
            "collect": self._progress.add_task(f"round {round_index} · collect", total=collect, detail=""),
            "train": self._progress.add_task(f"round {round_index} · train", total=train, detail=""),
        }

    def update(self, stage: str, completed: int, detail: str = "") -> None:
        if self.live:
            self._progress.update(self._stages[stage], completed=completed, detail=detail, refresh=True)

    def complete_round(self, round_index: int) -> None:
        if self.live:
            self._progress.update(self._overall, completed=round_index + 1)

    def close(self) -> None:
        if self.live:
            self._progress.stop()


class CollectionProgress:
    """Counts finished teachers through ``.done`` files, so rank 0 sees every rank's progress."""

    def __init__(
        self, total: int, round_index: int, directory: Path, *, enabled: bool, dashboard: DaggerProgress
    ) -> None:
        self.total = total
        self.round_index = round_index
        self.directory = directory
        self.enabled = enabled
        self.dashboard = dashboard
        self.reported = 0
        self.last_reward: float | None = None
        self.interval = max(1, (total + 9) // 10)

    def complete(self, task_id: int, stats: Any) -> None:
        (self.directory / f"{task_id:06d}.done").touch()
        if self.enabled:
            completed = sum(1 for _ in self.directory.glob("*.done"))
            self.last_reward = float(stats.mean_reward)
            if self.dashboard.live:
                self.dashboard.update("collect", completed, f"reward={self.last_reward:.3f}")
            elif completed == self.total or completed - self.reported >= self.interval:
                self._report(completed)

    def close(self) -> None:
        if not self.enabled:
            return
        if self.dashboard.live:
            reward = "" if self.last_reward is None else f"reward={self.last_reward:.3f}"
            self.dashboard.update("collect", self.total, reward)
        elif self.reported < self.total:
            self._report(self.total)

    def _report(self, completed: int) -> None:
        reward = "n/a" if self.last_reward is None else f"{self.last_reward:.3f}"
        print(
            f"{completed}: collect/mean_reward={reward} "
            f"collect/tasks={completed}/{self.total} round={self.round_index}",
            flush=True,
        )
        self.reported = completed
