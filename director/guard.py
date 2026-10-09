"""Fairness guard: veto knob settings that would make the game unwinnable.

The game supplies simulate(snapshot, knobs, rollout) -> {"survived": bool, ...}: copy the current
state, apply the candidate knobs, and let a strong reference player try. If no rollout survives the
candidate but some survive the CURRENT knobs, the change itself is what makes the game unwinnable:
it is vetoed, and the director keeps its current knobs and is told why. If the state is doomed either
way (hazards already in flight), the change is not blamed.

A guard that uses an omniscient player (one that can see the future) is a necessary condition for
fairness: if even it cannot survive, no real player can.
"""

from __future__ import annotations

from typing import Any, Callable


class FeasibilityGuard:
    def __init__(self, simulate: Callable[[Any, dict, int], dict], rollouts: int = 3, min_survivors: int = 1):
        self.simulate = simulate
        self.rollouts = rollouts
        self.min_survivors = min_survivors
        self.checks = 0
        self.vetoes = 0

    def _run(self, snapshot: Any, knobs: dict) -> tuple[int, list]:
        results = []
        for i in range(self.rollouts):
            try:
                results.append(self.simulate(snapshot, knobs, i))
            except Exception as exc:  # a broken simulation must not block the game: fail open, report it
                results.append({"survived": True, "error": f"{type(exc).__name__}: {exc}"})
        return sum(bool(r.get("survived")) for r in results), results

    def check(self, snapshot: Any, knobs: dict, current: dict | None = None) -> tuple[bool, dict]:
        survivors, results = self._run(snapshot, knobs)
        report = {"survivors": survivors, "rollouts": len(results), "results": results}
        ok = survivors >= self.min_survivors
        if not ok and current is not None:
            baseline, _ = self._run(snapshot, current)
            report["survivors_with_current_knobs"] = baseline
            ok = baseline < self.min_survivors        # doomed anyway: not the change's fault
            report["doomed_anyway"] = ok
        self.checks += 1
        self.vetoes += not ok
        return ok, report
