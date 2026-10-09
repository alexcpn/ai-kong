"""The director: a slow LLM that reasons about the player and sets a fast game's knobs.

    game (every tick) ── profile + outcomes ──► Director.update()   (every N s or on a trigger)
       ▲                                           │ LLM: {opponent_model, strategy, reasons, taunt, knobs,
       └──── KnobSet.apply ◄── guard ◄── KnobSet.validate      commands?}

The LLM never acts on the game directly. It only proposes knob values, which are validated
(bounds, max step, cooldown), checked by an optional fairness guard, and applied by the game's own
deterministic code. Optional Commands carry structured orders that are not knobs (e.g. "send unit 2
to guard the bridge"); the game validates and applies them on its own thread. In background mode the LLM call runs in a thread, so a real-time game never
waits for it; results are applied on the game's next poll().
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .guard import FeasibilityGuard
from .knobs import KnobSet
from .llm import Layer, LLMClient, LLMError

DIRECTOR_SYSTEM = """You are the DIRECTOR of a real-time game. You never control moment-to-moment action:
a deterministic engine does that, every tick, using the KNOBS you set. You are consulted every few
seconds and whenever something important happens.

Each time:
1. Study the player profile and, above all, the RESULTS of your previous knob settings.
2. Write an opponent_model: what this player does, where they are weak or strong (1-3 sentences).
3. Pick a strategy: a deliberate, multi-step plan, not a reflex. Good directors set up and pay off:
   establish a pattern, let the player adapt to it, then change it at the moment it hurts most;
   exploit habits you can see in the data; keep what worked, drop what failed.
4. Set EVERY knob (unchanged ones at their current value). Changes are limited per update and some
   knobs have cooldowns; a fairness guard vetoes settings that would make the game unwinnable, so
   cunning beats brute force.
5. Give short reasons and an optional taunt.

Fields: opponent_model, strategy (ONE line, <= 80 characters, shown on screen), reasons (1-3
sentences), taunt (<= 60 characters, playful and family-friendly, may be empty), knobs."""


@dataclass
class Commands:
    """Structured orders beside the knobs. apply(value, t) runs on the game's thread (in poll() for
    background mode); it must validate and clamp, and returns notes for the LLM's next update."""
    schema: dict
    apply: Callable[[Any, float], list]
    description: str = ""
    state: Callable[[], Any] | None = None     # what the LLM should know about the current orders


@dataclass
class Decision:
    t: float
    trigger: str
    reply: dict | None = None
    changes: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    guard: dict | None = None
    vetoed: bool = False
    error: str | None = None


class Director:
    def __init__(self, knobs: KnobSet, brief: str, layer: Layer | None = None, client: Any = None,
                 every_s: float = 20.0, guard: FeasibilityGuard | None = None, background: bool = False,
                 max_history: int = 6, system: str = DIRECTOR_SYSTEM, commands: Commands | None = None) -> None:
        self.knobs = knobs
        self.commands = commands
        self.brief = brief
        self.system = system
        self.layer = layer or Layer(name="director")
        self.client = client if client is not None else LLMClient()
        self.every_s = every_s
        self.guard = guard
        self.background = background
        self.max_history = max_history
        self.strategy = ""
        self.opponent_model = ""
        self.taunt = ""
        self.history: list[dict] = []
        self.decisions: list[Decision] = []
        self._last_t: float | None = None
        self._last_outcomes: dict = {}
        self._last_notes: list[str] = []
        self._job: cf.Future | None = None
        self._pool = cf.ThreadPoolExecutor(max_workers=1) if background else None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ scheduling

    def due(self, t: float, trigger: str | None = None) -> str | None:
        """Why the director should think now, or None."""
        if self._job is not None:
            return None
        if self._last_t is None:
            return "start"
        if trigger:
            return trigger
        if t - self._last_t >= self.every_s:
            return "interval"
        return None

    def update(self, t: float, context: dict, outcomes: dict, snapshot: Any = None, trigger: str = "interval") -> None:
        """Think now (sync) or start thinking (background). Outcomes are cumulative counters."""
        self._close_span(t, outcomes)
        self._last_t = t
        prompt = {
            "knobs": self.knobs.describe(),
            "situation": context,
            "your_previous_strategies_and_results": self.history,
            "notes_from_your_last_update": self._last_notes,
            "why_you_are_consulted_now": trigger,
        }
        if self.commands is not None:
            prompt["commands"] = {"how": self.commands.description,
                                  "current": self.commands.state() if self.commands.state else None}
        if self.background:
            self._job = self._pool.submit(self._think, t, trigger, prompt, snapshot)
        else:
            self._apply(self._think(t, trigger, prompt, snapshot))

    def poll(self) -> bool:
        """Background mode: apply a finished decision. Returns True if one was applied."""
        job = self._job
        if job is None or not job.done():
            return False
        self._job = None
        self._apply(job.result())
        return True

    # ------------------------------------------------------------------ internals

    def _close_span(self, t: float, outcomes: dict) -> None:
        if self._last_t is not None and self.strategy:
            delta = {k: round(v - self._last_outcomes.get(k, 0), 3) for k, v in outcomes.items()
                     if isinstance(v, (int, float))}
            self.history.append({"strategy": self.strategy, "knobs": self.knobs.snapshot(),
                                 "seconds": round(t - self._last_t, 1), "results": delta})
            self.history = self.history[-self.max_history:]
        self._last_outcomes = dict(outcomes)

    def schema(self) -> dict:
        props = {"opponent_model": {"type": "string"}, "strategy": {"type": "string"},
                 "reasons": {"type": "string"}, "taunt": {"type": "string"}, "knobs": self.knobs.json_schema()}
        if self.commands is not None:
            props["commands"] = self.commands.schema
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    @staticmethod
    def _validate(reply) -> dict:
        if not isinstance(reply, dict):
            raise LLMError("reply is not an object")
        for key in ("opponent_model", "strategy", "reasons", "taunt"):
            if not isinstance(reply.get(key), str):
                raise LLMError(f"missing {key}")
        if not isinstance(reply.get("knobs"), dict):
            raise LLMError("missing knobs")
        return reply

    def _think(self, t: float, trigger: str, prompt: dict, snapshot: Any) -> Decision:
        decision = Decision(t=t, trigger=trigger)
        try:
            reply = self.client.ask(self.layer, self.system + "\n\nTHE GAME\n" + self.brief, prompt,
                                    self.schema(), self._validate)
        except LLMError as exc:
            decision.error = str(exc)
            decision.notes = ["your last reply could not be used; knobs were left unchanged"]
            return decision
        decision.reply = reply
        changes, notes = self.knobs.validate(reply["knobs"], t)
        if changes and self.guard is not None and snapshot is not None:
            candidate = {**self.knobs.snapshot(), **changes}
            ok, report = self.guard.check(snapshot, candidate, current=self.knobs.snapshot())
            decision.guard = {k: v for k, v in report.items() if k != "results"}
            if not ok:
                decision.vetoed = True
                notes.append(f"fairness guard VETOED your knob changes: a strong reference player survived "
                             f"{report['survivors']}/{report['rollouts']} simulations. Be cunning, not impossible.")
                changes = {}
        decision.changes, decision.notes = changes, notes
        return decision

    def _apply(self, decision: Decision) -> None:
        with self._lock:
            self.decisions.append(decision)
            self._last_notes = decision.notes
            if decision.reply is None:
                return
            reply = decision.reply
            self.opponent_model = reply["opponent_model"].strip()
            self.strategy = " ".join(reply["strategy"].split())[:90] + (" (vetoed)" if decision.vetoed else "")
            taunt = " ".join(reply["taunt"].split())[:60]
            if taunt:
                self.taunt = taunt
            self.knobs.apply(decision.changes, decision.t, reason=reply.get("reasons", "")[:300])
            if self.commands is not None and "commands" in reply:
                try:
                    self._last_notes = list(decision.notes) + list(self.commands.apply(reply["commands"], decision.t))
                except Exception as exc:  # a bad order must never break the game
                    self._last_notes = list(decision.notes) + [f"commands rejected: {str(exc)[:120]}"]

    def summary(self) -> dict:
        return {
            "decisions": len(self.decisions),
            "errors": sum(d.error is not None for d in self.decisions),
            "guard_checks": sum(d.guard is not None for d in self.decisions),
            "vetoes": sum(d.vetoed for d in self.decisions),
            "knob_changes": sum(len(d.changes) for d in self.decisions),
            "strategies": [h["strategy"] for h in self.history] + ([self.strategy] if self.strategy else []),
        }
