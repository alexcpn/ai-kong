"""Typed, bounded, rate-limited knobs: the only way a director may influence a game.

A game declares KnobSpecs. The director (an LLM) proposes values; KnobSet.validate clamps them to
bounds and per-update step limits, enforces cooldowns, and explains every adjustment, so the model
can never push the game outside what its designer allowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

KINDS = ("float", "int", "bool", "enum", "weights")


@dataclass(frozen=True)
class KnobSpec:
    name: str
    kind: str                      # float | int | bool | enum | weights
    default: Any
    description: str = ""
    min: float | None = None       # float / int
    max: float | None = None
    choices: tuple[str, ...] = ()  # enum options, or the keys of a weights knob
    max_step: float | None = None  # largest change per update (per component for weights)
    cooldown_s: float = 0.0        # minimum game time between changes

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.kind in ("enum", "weights") and not self.choices:
            raise ValueError(f"{self.name}: {self.kind} knob needs choices")
        if self.kind in ("float", "int") and (self.min is None or self.max is None):
            raise ValueError(f"{self.name}: numeric knob needs min and max")

    def coerce(self, raw: Any) -> Any:
        """Convert a proposed value to this knob's type (ValueError if impossible)."""
        if self.kind == "bool":
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str) and raw.lower() in ("true", "false"):
                return raw.lower() == "true"
            raise ValueError("expected true/false")
        if self.kind == "enum":
            if raw not in self.choices:
                raise ValueError(f"expected one of {list(self.choices)}")
            return raw
        if self.kind == "weights":
            if not isinstance(raw, dict) or set(raw) != set(self.choices):
                raise ValueError(f"expected weights for exactly {list(self.choices)}")
            values = {k: float(raw[k]) for k in self.choices}
            if any(not math.isfinite(v) or v < 0 for v in values.values()) or sum(values.values()) <= 0:
                raise ValueError("weights must be non-negative and not all zero")
            total = sum(values.values())
            return {k: round(v / total, 4) for k, v in values.items()}
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("not a finite number")
        return int(round(value)) if self.kind == "int" else value

    def limit(self, current: Any, value: Any) -> tuple[Any, str]:
        """Clamp to bounds and max_step. Returns (value, note about any adjustment)."""
        notes = []
        if self.kind in ("float", "int"):
            if value < self.min or value > self.max:
                notes.append(f"clamped to [{self.min}, {self.max}]")
                value = min(self.max, max(self.min, value))
            if self.max_step is not None and abs(value - current) > self.max_step:
                notes.append(f"moved at most {self.max_step} per update")
                value = current + math.copysign(self.max_step, value - current)
            if self.kind == "int":
                value = int(round(value))
            else:
                value = round(value, 4)
        elif self.kind == "weights" and self.max_step is not None:
            stepped = {k: current[k] + max(-self.max_step, min(self.max_step, value[k] - current[k]))
                       for k in self.choices}
            if stepped != value:
                notes.append(f"each weight moved at most {self.max_step} per update")
                total = sum(stepped.values()) or 1.0
                value = {k: round(v / total, 4) for k, v in stepped.items()}
        return value, "; ".join(notes)

    def schema(self) -> dict:
        if self.kind == "bool":
            return {"type": "boolean"}
        if self.kind == "enum":
            return {"type": "string", "enum": list(self.choices)}
        if self.kind == "weights":
            return {"type": "object", "properties": {c: {"type": "number"} for c in self.choices},
                    "required": list(self.choices), "additionalProperties": False}
        return {"type": "integer" if self.kind == "int" else "number"}

    def describe(self, current: Any) -> dict:
        out: dict = {"name": self.name, "kind": self.kind, "current": current, "description": self.description}
        if self.kind in ("float", "int"):
            out["range"] = [self.min, self.max]
        if self.choices:
            out["choices"] = list(self.choices)
        if self.max_step is not None:
            out["max_change_per_update"] = self.max_step
        if self.cooldown_s:
            out["cooldown_s"] = self.cooldown_s
        return out


@dataclass
class KnobSet:
    specs: list[KnobSpec]
    values: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("duplicate knob names")
        self._by_name = {s.name: s for s in self.specs}
        for spec in self.specs:
            self.values.setdefault(spec.name, spec.coerce(spec.default))
        self._changed_at = {s.name: -math.inf for s in self.specs}

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def snapshot(self) -> dict:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.values.items()}

    def validate(self, proposal: dict, t: float) -> tuple[dict, list[str]]:
        """Return (changes to apply, notes). Never raises on bad input; explains instead."""
        changes: dict = {}
        notes: list[str] = []
        if not isinstance(proposal, dict):
            return changes, ["proposal is not an object"]
        for name, raw in proposal.items():
            spec = self._by_name.get(name)
            if spec is None:
                notes.append(f"{name}: unknown knob, ignored")
                continue
            try:
                value = spec.coerce(raw)
            except (TypeError, ValueError) as exc:
                notes.append(f"{name}: rejected ({exc})")
                continue
            current = self.values[name]
            value, note = spec.limit(current, value)
            if value == current:
                continue
            if t - self._changed_at[name] < spec.cooldown_s:
                wait = spec.cooldown_s - (t - self._changed_at[name])
                notes.append(f"{name}: cooling down, can change again in {wait:.1f}s")
                continue
            if note:
                notes.append(f"{name}: {note}")
            changes[name] = value
        return changes, notes

    def apply(self, changes: dict, t: float, reason: str = "") -> None:
        if not changes:
            return
        before = {k: self.values[k] for k in changes}
        self.values.update(changes)
        for name in changes:
            self._changed_at[name] = t
        self.history.append({"t": round(t, 2), "before": before, "after": dict(changes), "reason": reason})
        if len(self.history) > 100:
            del self.history[:50]

    def describe(self) -> list[dict]:
        return [s.describe(self.values[s.name]) for s in self.specs]

    def json_schema(self) -> dict:
        """Strict JSON schema for a full knob proposal (every knob required)."""
        return {"type": "object", "properties": {s.name: s.schema() for s in self.specs},
                "required": [s.name for s in self.specs], "additionalProperties": False}
