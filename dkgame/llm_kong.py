"""Kong driven by LLMs through OpenRouter (or any OpenAI-compatible endpoint), in two layers.

  Strategist  every ~20 game-seconds, and after the player loses a life or clears a level.
              Studies the player's habits and how its earlier strategies went, then writes a
              multi-step plan (the cunning part). Medium reasoning by default.
  Tactician   every Kong decision (3 game-seconds). Turns the current plan plus the live
              situation into concrete throws. Low reasoning by default.

Output is constrained with strict JSON schemas (routed only to hosts that enforce them), validated,
retried once, and on failure replaced by a scripted plan. Every fallback is counted.

Configuration (environment variables, or LLMKong(...) arguments; per layer):
  OPENROUTER_API_KEY                 required (or ~/.config/dk-bench/openrouter.env, see director/llm.py)
  KONG_STRATEGIST_MODEL              default anthropic/claude-haiku-5.5
  KONG_TACTICIAN_MODEL               default anthropic/claude-haiku-5.5
  KONG_STRATEGIST_REASONING          none | low | medium | high   (default medium)
  KONG_TACTICIAN_REASONING           none | low | medium | high   (default low)
  KONG_STRATEGIST_JSON / _TACTICIAN_JSON   schema (strict, default) | object (for models without
                                     structured-output support, e.g. some open models)
  KONG_PROVIDER                      optional: pin one OpenRouter provider, e.g. "anthropic"
  KONG_STRATEGY_EVERY                game seconds between strategist calls (default 20)
  KONG_BASE_URL                      default https://openrouter.ai/api/v1
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # dk-game/, for director

from director.llm import Layer, LLMClient, LLMError  # noqa: E402

from .engine import ROUTES, Kong  # noqa: E402
from .kongs import SniperKong  # noqa: E402
SPEEDS = ["slow", "normal", "fast"]

MECHANICS = """GAME MECHANICS (a terminal Donkey Kong; you are Kong at the top-left)
- Girders are numbered 0 (bottom) up to top_floor. The player starts bottom-left and must reach the goal
  column on the top girder. Ladders join girder b and b+1 at column x.
- You throw barrels. Each appears beside you on the top girder and rolls right. When a rolling barrel
  reaches the top of a ladder it may go down it, depending on its route:
    "always"        takes every ladder down it meets
    "never"         ignores ladders; only drops off girder ends
    "random"        coin flip at each ladder
    "toward_player" goes down a ladder only while the player is on a lower girder
  Each time a barrel lands on a lower girder it reverses direction. Barrels leaving the bottom are gone.
- speed: slow, normal or fast (rules.speeds lists what this level allows). Faster barrels are harder to time.
- Limits per level (rules): max_throws_per_decision, min_throw_gap seconds between throws, max_barrels on
  the board at once (throws beyond that are WASTED). Decisions happen every few seconds; a throw's
  "delay" (0-3 s) schedules it within the coming window.
- The player dies touching a barrel. It can jump over barrels rolling on its own girder, stand aside, wait
  beside a ladder until it is clear, and is safe for a short while after respawning.
- A barrel coming down a ladder hits anyone climbing that ladder. Players are most vulnerable while
  climbing, and least when they can see a barrel coming on their own girder in good time."""

STRATEGIST_SYSTEM = MECHANICS + """

YOUR ROLE: STRATEGIST. You are the cunning mind of Kong. You do not throw barrels yourself; you study
the player and write the plan that your tactician will execute for the next ~20 seconds.

Think like a trickster, not a machine gun:
- Build an opponent model from the habits profile, the deaths, and what happened to your barrels.
- Exploit patterns: if it waits under ladders, time barrels to come down exactly that ladder when it
  commits to climbing; if it jumps early, use slow then fast barrels; if it always uses one ladder,
  make that ladder deadly and leave the others quiet as bait.
- Use multi-step plays: establish a rhythm, let the player relax, then break it. Hold barrels back and
  burst when the player is mid-ladder. Avoid wasting throws (board full).
- Learn from your own history: keep what killed the player, drop what it beat.

Return:
- opponent_model: what you believe about this player (habits, weaknesses), 1-3 sentences
- strategy: ONE short line (<= 80 chars) naming your current play, shown on screen to humans
- instructions: concrete guidance for the tactician (which routes/speeds/timings to use when, what
  to watch for), 2-6 sentences
- taunt: a playful, family-friendly taunt (<= 60 chars) that hints at your scheme without giving it away"""

TACTICIAN_SYSTEM = MECHANICS + """

YOUR ROLE: TACTICIAN. Your strategist has given you a plan. Execute it for the next decision window:
choose up to rules.max_throws_per_decision throws (zero is allowed: holding back can be part of a
trick). Follow the plan, but adapt to what the player is doing right now. Respect the limits.
Optionally update the taunt (<= 60 chars) or leave it empty."""

STRATEGIST_SCHEMA = {
    "type": "object",
    "properties": {
        "opponent_model": {"type": "string"},
        "strategy": {"type": "string"},
        "instructions": {"type": "string"},
        "taunt": {"type": "string"},
    },
    "required": ["opponent_model", "strategy", "instructions", "taunt"],
    "additionalProperties": False,
}
TACTICIAN_SCHEMA = {
    "type": "object",
    "properties": {
        "throws": {"type": "array", "items": {
            "type": "object",
            "properties": {"delay": {"type": "number"}, "speed": {"type": "string", "enum": SPEEDS},
                           "route": {"type": "string", "enum": list(ROUTES)}},
            "required": ["delay", "speed", "route"],
            "additionalProperties": False}},
        "taunt": {"type": "string"},
    },
    "required": ["throws", "taunt"],
    "additionalProperties": False,
}


def _validate_strategy(data) -> dict:
    if not isinstance(data, dict) or not all(isinstance(data.get(k), str) for k in STRATEGIST_SCHEMA["required"]):
        raise LLMError("strategist reply does not match the schema")
    if not data["strategy"].strip() or not data["instructions"].strip():
        raise LLMError("empty strategy")
    return data


def _validate_throws(data) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("throws"), list):
        raise LLMError("tactician reply has no throws list")
    throws = []
    for item in data["throws"]:
        if not isinstance(item, dict):
            raise LLMError("throw is not an object")
        try:
            delay = float(item.get("delay", 0))
        except (TypeError, ValueError):
            raise LLMError("bad delay")
        if item.get("speed") not in SPEEDS or item.get("route") not in ROUTES:
            raise LLMError(f"bad throw {item}")
        throws.append({"delay": delay, "speed": item["speed"], "route": item["route"]})
    taunt = data.get("taunt") if isinstance(data.get("taunt"), str) else ""
    return {"throws": throws, "taunt": taunt}


class LLMKong(Kong):
    name = "llm"

    def __init__(self, strategist: dict | None = None, tactician: dict | None = None, background: bool = False,
                 strategy_every: float | None = None, base_url: str | None = None, timeout: float = 60.0) -> None:
        self.client = LLMClient(base_url=base_url, timeout=timeout, title="DK-Bench LLM Kong")
        self.strategist = Layer.from_env("strategist", "medium", 8000, **(strategist or {}))
        self.tactician = Layer.from_env("tactician", "low", 2000, **(tactician or {}))
        self.strategy_every = strategy_every or float(os.environ.get("KONG_STRATEGY_EVERY", "20"))
        self.background = background
        self.fallback = SniperKong()
        for layer in ("strategist", "tactician"):
            self.client._usage(layer)               # report both layers even if one is never called
        self.transcript: list[dict] = []
        self._lock = threading.Lock()
        self.current: dict | None = None        # the strategist's latest plan
        self.history: list[dict] = []          # past strategies and how they went
        self._last_strategy_t = -1e9
        self._last_lives: int | None = None
        self._last_level: int | None = None
        self._stats_at_strategy: dict = {}
        self._pool = cf.ThreadPoolExecutor(max_workers=2) if background else None
        self._strategy_job: cf.Future | None = None
        self._tactic_job: cf.Future | None = None

    # ----------------------------------------------------------------- Kong API

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.fallback.reset(seed)

    @property
    def usage(self) -> dict:
        return self.client.usage

    @property
    def summary_usage(self) -> dict:
        """Totals across both layers (the shape evaluate.py aggregates)."""
        return self.client.total_usage()

    def plan(self, obs: dict) -> dict:
        if self._strategy_due(obs):
            if self.background:
                if self._strategy_job is None:
                    self._strategy_job = self._pool.submit(self._strategize, obs)
            else:
                self._strategize(obs)
        if self.background:
            self._collect_strategy()
            if self._tactic_job is None:
                self._tactic_job = self._pool.submit(self._tactics, obs)
            return {"throws": []}        # delivered through ready_plan() when the call returns
        return self._tactics(obs)

    def ready_plan(self) -> dict | None:
        """Background mode: hand over a finished tactician plan (polled every tick by the game)."""
        if not self.background:
            return None
        self._collect_strategy()
        job = self._tactic_job
        if job is None or not job.done():
            return None
        self._tactic_job = None
        return job.result()

    # ----------------------------------------------------------------- strategist

    def _strategy_due(self, obs: dict) -> bool:
        lost_life = self._last_lives is not None and obs["lives"] < self._last_lives
        new_level = self._last_level is not None and obs["level"] != self._last_level
        self._last_lives, self._last_level = obs["lives"], obs["level"]
        return self.current is None or lost_life or new_level or obs["t"] - self._last_strategy_t >= self.strategy_every

    def _collect_strategy(self) -> None:
        job = self._strategy_job
        if job is not None and job.done():
            self._strategy_job = None
            job.result()

    def _strategize(self, obs: dict) -> None:
        stats = obs["stats"]
        if self.current is not None:
            delta = {k: stats.get(k, 0) - self._stats_at_strategy.get(k, 0) for k in stats}
            outcomes = [o for o in obs["barrel_outcomes"] if o["thrown_at"] >= self._last_strategy_t]
            self.history.append({
                "strategy": self.current["strategy"], "level": self._last_level,
                "seconds": round(obs["t"] - self._last_strategy_t, 1),
                "player_hits": delta.get("hits_barrel", 0) + delta.get("hits_fireball", 0),
                "barrels_jumped": delta.get("jumped_over", 0), "player_climbs": delta.get("climbs", 0),
                "wasted_throws": sum(1 for o in outcomes if (o["outcome"] or "").startswith("wasted")),
            })
            self.history = self.history[-6:]
        self._last_strategy_t = obs["t"]
        self._stats_at_strategy = dict(stats)
        prompt = {
            "situation": _situation(obs),
            "player_habits": obs["habits"],
            "your_previous_strategies_and_results": self.history,
            "what_happened_to_your_recent_barrels": obs["barrel_outcomes"],
            "recent_events": [e for e in obs["recent_events"] if e["event"] != "taunt"][-10:],
        }
        try:
            plan = self._ask(self.strategist, STRATEGIST_SYSTEM, prompt, STRATEGIST_SCHEMA, _validate_strategy)
        except LLMError as exc:
            self._log("strategist", obs, error=str(exc))
            if self.current is None:
                self.current = {"strategy": "(scripted fallback)", "instructions": "Fast barrels toward the player.",
                                "opponent_model": "", "taunt": ""}
            return
        plan["strategy"] = plan["strategy"].strip()[:90]
        with self._lock:
            self.current = plan
        self._log("strategist", obs, reply=plan)

    # ----------------------------------------------------------------- tactician

    def _tactics(self, obs: dict) -> dict:
        current = self.current or {}
        prompt = {
            "strategist_plan": {k: current.get(k, "") for k in ("strategy", "instructions", "opponent_model")},
            "situation": _situation(obs),
            "what_happened_to_your_recent_barrels": obs["barrel_outcomes"][-6:],
        }
        try:
            plan = self._ask(self.tactician, TACTICIAN_SYSTEM, prompt, TACTICIAN_SCHEMA, _validate_throws)
        except LLMError as exc:
            self._log("tactician", obs, error=str(exc))
            plan = self.fallback.plan(obs)
            plan["taunt"] = ""
        else:
            self._log("tactician", obs, reply=plan)
        if not plan.get("taunt") and current.get("taunt") and current.get("_taunted") is not True:
            plan["taunt"] = current["taunt"]
            current["_taunted"] = True
        plan["strategy"] = current.get("strategy", "")
        return plan

    # ----------------------------------------------------------------- transport

    def _ask(self, layer: Layer, system: str, prompt: dict, schema: dict, validate) -> dict:
        """One validated reply via the shared director transport (strict schema, one retry)."""
        return self.client.ask(layer, system, prompt, schema, validate)

    def _log(self, layer: str, obs: dict, reply: dict | None = None, error: str | None = None) -> None:
        with self._lock:
            self.transcript.append({"t": obs["t"], "level": obs["level"], "layer": layer,
                                    **({"reply": reply} if reply is not None else {"error": error})})


def _situation(obs: dict) -> dict:
    return {k: obs[k] for k in ("t", "level", "lives", "time_left", "player", "top_floor", "goal_x", "ladders",
                                "barrels_on_board", "rules")}
