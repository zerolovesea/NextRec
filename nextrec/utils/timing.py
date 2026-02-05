"""
Lightweight profiling utilities for timing pipeline stages.

Date: create on 04/02/2026
Checkpoint: edit on 04/02/2026
Author: Yang Zhou,zyaztec@gmail.com
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable


@dataclass
class StageStat:
    total: float = 0.0
    count: int = 0
    min: float = math.inf
    max: float = 0.0

    def add(self, duration: float) -> None:
        self.total += float(duration)
        self.count += 1
        if duration < self.min:
            self.min = float(duration)
        if duration > self.max:
            self.max = float(duration)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


class StageTimer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.stats = {}

    def add(self, name: str, duration: float) -> None:
        if not self.enabled:
            return
        stat = self.stats.get(name)
        if stat is None:
            stat = StageStat()
            self.stats[name] = stat
        stat.add(duration)

    def merge(self, other: "StageTimer") -> None:
        if not other or not other.stats:
            return
        for name, stat in other.stats.items():
            target = self.stats.get(name)
            if target is None:
                self.stats[name] = StageStat(
                    total=stat.total,
                    count=stat.count,
                    min=stat.min,
                    max=stat.max,
                )
            else:
                target.total += stat.total
                target.count += stat.count
                target.min = min(target.min, stat.min)
                target.max = max(target.max, stat.max)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "total": stat.total,
                "count": stat.count,
                "min": stat.min,
                "max": stat.max,
            }
            for name, stat in self.stats.items()
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Dict[str, float]]) -> "StageTimer":
        timer = cls(enabled=True)
        for name, stat in payload.items():
            timer.stats[name] = StageStat(
                total=float(stat.get("total", 0.0)),
                count=int(stat.get("count", 0)),
                min=float(stat.get("min", math.inf)),
                max=float(stat.get("max", 0.0)),
            )
        return timer

    def dump(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "StageTimer":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def summary_rows(self) -> Iterable[tuple[str, StageStat]]:
        return sorted(
            self.stats.items(),
            key=lambda item: item[1].total,
            reverse=True,
        )
