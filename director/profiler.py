"""Player profiling: turn a stream of game events into a compact summary an LLM can reason about."""

from __future__ import annotations

import statistics
from collections import deque
from typing import Callable


class Profiler:
    """Counters, numeric samples, durations and recent events, plus a game-specific summary hook.

    The game calls event()/count()/observe()/start()/stop() as things happen (cheap, every tick if
    needed); the director calls summary() only when it is about to think.
    """

    def __init__(self, summarise: Callable[["Profiler"], dict] | None = None, max_events: int = 300,
                 max_samples: int = 50) -> None:
        self.summarise = summarise
        self.events: deque = deque(maxlen=max_events)
        self.counters: dict[str, float] = {}
        self.samples: dict[str, deque] = {}
        self.durations: dict[str, deque] = {}
        self._open: dict[str, float] = {}
        self._max_samples = max_samples

    def event(self, kind: str, t: float, **data) -> None:
        self.events.append({"t": round(t, 2), "event": kind, **data})

    def count(self, key: str, n: float = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def observe(self, key: str, value: float) -> None:
        self.samples.setdefault(key, deque(maxlen=self._max_samples)).append(float(value))

    def start(self, key: str, t: float) -> None:
        self._open.setdefault(key, t)

    def stop(self, key: str, t: float, min_duration: float = 0.0) -> None:
        began = self._open.pop(key, None)
        if began is not None and t - began >= min_duration:
            self.durations.setdefault(key, deque(maxlen=self._max_samples)).append(round(t - began, 2))

    def recent(self, n: int = 10, kinds: tuple[str, ...] = ()) -> list[dict]:
        items = [e for e in self.events if not kinds or e["event"] in kinds]
        return items[-n:]

    def summary(self) -> dict:
        def stats(values) -> dict:
            values = list(values)
            return {"n": len(values), "mean": round(statistics.fmean(values), 2) if values else None,
                    "last": values[-1] if values else None}

        out = {
            "counters": dict(self.counters),
            "samples": {k: stats(v) for k, v in self.samples.items()},
            "durations": {k: stats(v) for k, v in self.durations.items()},
        }
        if self.summarise is not None:
            out.update(self.summarise(self))
        return out
