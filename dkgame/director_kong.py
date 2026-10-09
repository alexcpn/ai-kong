"""Kong rebuilt on the director pattern.

ParametricKong is a deterministic Kong whose behaviour is fully set by a handful of knobs.
DirectorKong lets an LLM director (director/) turn those knobs from a profile of the player,
with a fairness guard that vetoes settings even an omniscient player could not survive.
Plans go through the same {"throws": [...]} interface as every Kong. A separate "voice" call writes
the characters' spoken lines ahead of time (see cast.py); the game picks them instantly.

With a fast inference provider, an optional TACTICIAN layer turns the strategist's plan into the
actual throws at every Kong decision (every couple of seconds). It never blocks the game: if its
reply is late, Kong falls back to the knob-driven throws for that window, and the miss is counted.
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # dk-game/, for director

from director import Director, FeasibilityGuard, KnobSet, KnobSpec, Layer, LLMError  # noqa: E402

from .cast import lines_schema  # noqa: E402
from .engine import ROUTES, TICK, Kong  # noqa: E402
from .llm_kong import MECHANICS  # noqa: E402
from .lookahead import LookaheadPlayer  # noqa: E402

ROUTE_NAMES = ("always", "never", "random", "toward_player")

KNOB_SPECS = [
    KnobSpec("throw_rate", "float", 0.6, min=0.0, max=1.0, max_step=0.4,
             description="fraction of the maximum throw rate this level allows (0 = no barrels)"),
    KnobSpec("speed_mix", "weights", {"slow": 0.35, "normal": 0.55, "fast": 0.10}, choices=("slow", "normal", "fast"),
             max_step=0.5, description="chance of each barrel speed; speeds the level forbids become normal"),
    KnobSpec("route_mix", "weights", {"always": 0.2, "never": 0.2, "random": 0.4, "toward_player": 0.2},
             choices=ROUTE_NAMES, max_step=0.6,
             description="chance of each barrel route (see mechanics: how barrels use ladders)"),
    KnobSpec("burst_prob", "float", 0.1, min=0.0, max=1.0, max_step=0.5,
             description="chance a decision becomes a burst: as many barrels as allowed, minimum spacing"),
    KnobSpec("ambush_on_climb", "bool", False, cooldown_s=6.0,
             description="while true, whenever the player is on a ladder at a decision, add a fast barrel "
                         "that takes every ladder"),
    KnobSpec("rhythm_jitter", "float", 0.2, min=0.0, max=1.0, max_step=0.6,
             description="0 = metronome-regular throws (easy to learn), 1 = irregular timing"),
    KnobSpec("hold_back_s", "float", 0.0, min=0.0, max=6.0, max_step=6.0, cooldown_s=10.0,
             description="right after this update, throw nothing for this many seconds (a lull before a strike)"),
    KnobSpec("kong_x", "float", 0.1, min=0.0, max=1.0, max_step=1.0,
             description="where Kong stands along his girder (0 = left end, 1 = right end; on the top girder he "
                         "stays a few columns short of Pauline). Barrels start where he stands and roll toward "
                         "the player: stand above a ladder they need. Ignored if Kong can't move"),
]
DEFAULT_KNOBS = KnobSet(KNOB_SPECS).snapshot()

BRIEF = MECHANICS + """

YOU DIRECT KONG. Your goal is to stop the player from reaching the goal, by out-thinking them, not by
flooding the board (the fairness guard vetoes impossible settings, and wasted throws do nothing).
A deterministic Kong throws barrels every few seconds according to your knobs.

BARREL SUPPLY. When situation.barrels_left is a number, Kong has only that many barrels left this
level (barrel_budget per level; refilled when the player reaches a new level, NOT when they lose a
life). Once it hits 0 Kong is helpless until the next level, but barrels still unspent when the
level ends are WASTED, and a level usually lasts 30-60 s: a passive Kong loses. Keep up steady
pressure (throw_rate 0.5+), spend extra in bursts and ambushes where the player is weakest (ladders
they favour, the climb to the goal), and plan to be nearly empty as they reach the top.

TEMPER. The player's successes (jumping your barrels, reaching a higher girder, getting to the top
girder) raise situation.kong.anger (0-100, it cools over time). At 100 Kong loses his temper and storms
down to the player's girder for a few seconds, throwing point-blank barrels (from your supply).

WHERE KONG IS (situation.kong). When Kong can move he roams a few girders ABOVE the player
(climbing as they climb, up to the top girder) and throws from there, so barrels arrive sooner. If the
player touches Kong anywhere but on the top girder he is DEFEATED and they clear the level."""

TACTICIAN_SYSTEM = MECHANICS + """

WHERE KONG IS: Kong roams a few girders ABOVE the player (not always at the top) and each barrel starts
where he stands, rolling toward the player. kong_x (0 = left end of his girder, 1 = right end) is where
he walks to next: stand above a ladder the player needs.

YOUR ROLE: TACTICIAN, deciding Kong's next few seconds in real time. Your strategist set the plan and
the knobs; turn them into concrete throws for the coming window, adapting to where the player is RIGHT
NOW (climbing a ladder? waiting at a ladder foot? jumping?). Throws: up to rules.max_throws_per_decision,
each {delay (seconds from now, 0..window_s), speed (allowed by rules.speeds), route}. Zero throws is fine
when holding back is part of the plan. Respect barrels_left: unspent barrels at level end are wasted,
but an empty Kong is helpless. Be quick and decisive."""

TACTICIAN_SCHEMA = {"type": "object", "properties": {
    "throws": {"type": "array", "items": {
        "type": "object",
        "properties": {"delay": {"type": "number"}, "speed": {"type": "string", "enum": ["slow", "normal", "fast"]},
                       "route": {"type": "string", "enum": list(ROUTES)}},
        "required": ["delay", "speed", "route"], "additionalProperties": False}},
    "kong_x": {"type": "number"}},
    "required": ["throws", "kong_x"], "additionalProperties": False}

CAST_SYSTEM = """You write the spoken lines for a terminal Donkey Kong game, ahead of time. The game
picks one instantly when a moment happens, so write for each kind of moment, not for one event.

Speakers:
- kong: the villain. Gruff, theatrical, mocking but family-friendly. He knows his own plan.
- pauline: trapped at the top, cheering the player on. In about a third of her lines she HINTS at
  Kong's current plan (your_plan below), as if she overheard him, e.g. "He's watching the left ladder!"
- player: the hero's own quick reactions, first person, a few words ("Oh no!", "Faster!", "Made it!").

Write 2 lines per speaker per moment, each at most 42 characters, no emoji. Make them fit what is
happening: the player's habits, the level, lives left, Kong's anger and plan, what just happened.
Moments: start, near_miss (a barrel just missed), jumped (cleared a barrel), climbed (reached a higher
girder), near_goal (on the top girder), kong_coming (Kong storms down), lost_life, level_clear,
kong_beaten (the player caught Kong), idle (a quiet moment)."""


def _weighted(rng: random.Random, weights: dict, allowed=None) -> str:
    items = [(k, w) for k, w in weights.items() if allowed is None or k in allowed]
    total = sum(w for _, w in items)
    if total <= 0:
        return "normal" if allowed is not None else items[0][0]
    pick = rng.random() * total
    for k, w in items:
        pick -= w
        if pick <= 0:
            return k
    return items[-1][0]


class ParametricKong(Kong):
    """Deterministic Kong driven entirely by knob values."""

    name = "parametric"

    def __init__(self, knobs=None) -> None:
        self.knobs = knobs if knobs is not None else dict(DEFAULT_KNOBS)   # dict or KnobSet
        self.hold_until = -1.0

    def value(self, name: str):
        return self.knobs[name]

    def _with_position(self, plan: dict) -> dict:
        try:
            plan["move_to"] = self.value("kong_x")
        except KeyError:
            pass
        return plan

    def plan(self, obs: dict) -> dict:
        rules, interval, t = obs["rules"], obs["interval"], obs["t"]
        if t < self.hold_until:
            return self._with_position({"throws": []})
        cap = min(rules["max_throws_per_decision"], int(interval // rules["min_throw_gap"]) + 1)
        if self.rng.random() < self.value("burst_prob"):
            n, spacing = cap, rules["min_throw_gap"]
        else:
            expected = self.value("throw_rate") * cap
            n = int(expected) + (self.rng.random() < expected - int(expected))
            spacing = interval / max(1, n)
        throws = []
        for i in range(n):
            jitter = (self.rng.random() - 0.5) * self.value("rhythm_jitter") * spacing
            throws.append({"delay": round(min(interval, max(0.0, i * spacing + jitter)), 2),
                           "speed": _weighted(self.rng, self.value("speed_mix"), allowed=rules["speeds"]),
                           "route": _weighted(self.rng, self.value("route_mix"))})
        if self.value("ambush_on_climb") and obs["player"]["mode"] == "climb":
            fast = "fast" if "fast" in rules["speeds"] else "normal"
            throws.insert(0, {"delay": 0.0, "speed": fast, "route": "always"})
        return self._with_position({"throws": throws})


class RandomKnobKong(ParametricKong):
    """Control condition: re-rolls every knob at random on the director's schedule."""

    name = "random_knobs"

    def __init__(self, every_s: float = 20.0) -> None:
        super().__init__()
        self.every_s, self._next = every_s, 0.0

    def plan(self, obs: dict) -> dict:
        if obs["t"] >= self._next:
            self._next = obs["t"] + self.every_s
            r = self.rng
            mix = lambda names: {n: r.random() for n in names}  # noqa: E731
            self.knobs = {"throw_rate": r.random(), "speed_mix": mix(("slow", "normal", "fast")),
                          "route_mix": mix(ROUTE_NAMES), "burst_prob": r.random(), "ambush_on_climb": r.random() < 0.5,
                          "rhythm_jitter": r.random(), "hold_back_s": r.choice([0.0, 0.0, 2.0, 4.0]),
                          "kong_x": r.random()}
            self.hold_until = obs["t"] + self.knobs["hold_back_s"]
        return super().plan(obs)


def snapshot_game(game):
    """Deep copy of the game for simulation, without the live Kong (threads, LLM clients)."""
    return copy.deepcopy(game, memo={id(game.kong): None})


def simulate(snapshot, knobs: dict, rollout: int, seconds: float = 8.0) -> dict:
    """Can an omniscient player survive `seconds` of these knobs from this state?"""
    sim = copy.deepcopy(snapshot)
    kong = ParametricKong(dict(knobs))
    kong.reset(9_000 + rollout)
    sim.kong = kong
    sim.rng = random.Random(77_000 + rollout)
    sim.next_decision = min(sim.next_decision, sim.t + TICK)      # let the new knobs act immediately
    lives0, player = sim.lives, LookaheadPlayer(horizon=30, replan_every=3)
    for _ in range(int(seconds / TICK)):
        if sim.over:
            break
        sim.step(player.act(sim))
        if sim.lives < lives0:
            return {"survived": False, "died_after_s": round(sim.t - snapshot.t, 2)}
    return {"survived": True}


class DirectorKong(ParametricKong):
    """ParametricKong whose knobs an LLM director turns from a profile of the player."""

    name = "director"

    def __init__(self, background: bool = False, guard: bool = True, every_s: float | None = None,
                 layer: Layer | None = None, client=None, guard_rollouts: int = 3, tactician: bool = False,
                 tactician_deadline_s: float = 1.2) -> None:
        self.knobset = KnobSet(KNOB_SPECS)
        super().__init__(self.knobset)
        every = every_s or float(os.environ.get("KONG_STRATEGY_EVERY", "20"))
        self.director = Director(
            self.knobset, brief=BRIEF, layer=layer or Layer.from_env("director", "medium", 8000),
            client=client, every_s=every, background=background,
            guard=FeasibilityGuard(simulate, rollouts=guard_rollouts) if guard else None)
        self.background = background
        self.game = None
        self._seen = {"lives": None, "level": None}
        self._applied = 0
        self._announced_taunt = ""
        self.voice = Layer.from_env("voice", "none", 2500)          # writes the characters' lines
        self._voice_pool = cf.ThreadPoolExecutor(max_workers=1)
        self._voice_job: cf.Future | None = None
        self.cast = None                          # the game's dkgame.cast.Cast, set by the game
        self.cast_every = float(os.environ.get("KONG_CAST_EVERY", "30"))
        self._cast_next = 0.0
        self._cast_plan = None                    # strategy the current lines were written for
        self.tactician = Layer.from_env("tactician", "low", 1500) if tactician and background else None
        self._tac_pool = cf.ThreadPoolExecutor(max_workers=1) if self.tactician else None
        self._tac_job: cf.Future | None = None
        self._tac_obs: dict | None = None
        self._tac_deadline = 0.0
        self._tac_settled = True                  # this window already has its throws (reply or fallback)
        self.tactics = {"windows": 0, "on_time": 0, "late": 0}
        self.tactician_deadline_s = tactician_deadline_s

    # ------------------------------------------------------------------ the characters' lines

    def _cast_tick(self) -> None:
        """Hand finished lines to the game's Cast; ask for new ones when due or the plan changed."""
        job = self._voice_job
        if job is not None and job.done():
            self._voice_job = None
            try:
                self.cast.apply(job.result())
            except Exception:      # LLMError or network trouble: the current lines stay
                pass
        if self._voice_job is not None or self.cast is None or self.game is None:
            return
        t = self.game.t
        if t >= self._cast_next or (self.director.strategy and self.director.strategy != self._cast_plan):
            self._cast_next, self._cast_plan = t + self.cast_every, self.director.strategy
            self._voice_job = self._voice_pool.submit(self.director.client.ask, self.voice, CAST_SYSTEM,
                                                      self.cast_prompt(), lines_schema(), self._valid_lines)

    def cast_prompt(self) -> dict:
        g = self.game
        recent = [e["event"] for e in g.events[-12:] if e["event"] not in ("throw", "provoked", "taunt")]
        return {"level": g.level, "lives": g.lives, "time_left": round(g.time_left),
                "player": {"floor": g.player.floor, "top_floor": g.layout.top, "mode": g.player.mode},
                "player_habits": g.habit_summary(), "kong_anger": round(g.anger), "kong_mode": g.kong_body.mode,
                "barrels_left": g.barrels_left, "your_plan": self.director.strategy or "(still sizing them up)",
                "what_kong_thinks_of_them": self.director.opponent_model, "recent_events": recent}

    @staticmethod
    def _valid_lines(reply):
        if not isinstance(reply, dict) or not any(isinstance(reply.get(s), dict) for s in ("kong", "pauline", "player")):
            raise LLMError("bad lines reply")
        return reply

    def attach(self, game) -> None:
        """Give the director's fairness guard access to the live game (for snapshots)."""
        self.game = game

    def _after_apply(self, t: float) -> bool:
        applied = len(self.director.decisions) > self._applied
        if applied:
            self._applied = len(self.director.decisions)
            hold = self.knobset["hold_back_s"]
            if hold > 0 and self.director.decisions[-1].changes:
                self.hold_until = t + hold
        return applied

    def _announce(self, plan: dict) -> dict:
        plan["strategy"] = self.director.strategy
        if self.director.taunt and self.director.taunt != self._announced_taunt:
            plan["taunt"] = self._announced_taunt = self.director.taunt
            if self.cast is not None:              # the strategist's own one-liner joins Kong's quiet lines
                self.cast.apply({"kong": {"idle": [self.director.taunt] + self.cast.lines["kong"]["idle"][:2]}})
        return plan

    def plan(self, obs: dict) -> dict:
        t = obs["t"]
        trigger = None
        if self._seen["lives"] is not None and obs["lives"] < self._seen["lives"]:
            trigger = "the player just lost a life"
        elif self._seen["level"] is not None and obs["level"] != self._seen["level"]:
            trigger = f"the player reached level {obs['level']}"
        self._seen = {"lives": obs["lives"], "level": obs["level"]}
        if self.background and self.director.poll():
            self._after_apply(t)
        reason = self.director.due(t, trigger)
        if reason:
            snapshot = snapshot_game(self.game) if (self.game is not None and self.director.guard) else None
            self.director.update(t, context=self.context(obs), outcomes=self.outcomes(obs), snapshot=snapshot,
                                 trigger=reason)
            self._after_apply(t)
        if self.tactician is None:
            return self._announce(super().plan(obs))
        return self._announce(self._start_tactics(obs))

    # ------------------------------------------------------------------ tactician

    def _start_tactics(self, obs: dict) -> dict:
        """A new decision window: ask the tactician; until it answers, throw nothing (it is quick)."""
        self.tactics["windows"] += 1
        if self._tac_job is not None and not self._tac_job.done():
            self._tac_settled = True               # still busy with the last window: knobs this time
            self.tactics["late"] += 1
            return super().plan(obs)
        self._tac_obs, self._tac_deadline, self._tac_settled = obs, obs["t"] + self.tactician_deadline_s, False
        self._tac_job = self._tac_pool.submit(self.director.client.ask, self.tactician, TACTICIAN_SYSTEM,
                                              self.tactics_prompt(obs), TACTICIAN_SCHEMA, self._valid_tactics)
        return self._with_position({"throws": []})

    def tactics_prompt(self, obs: dict) -> dict:
        g = self.game
        barrels = [{"floor": b.floor, "x": b.x, "dir": b.direction, "falling": b.falling} for b in g.barrels] \
            if g is not None else []
        return {"window_s": obs["interval"], "rules": obs["rules"], "level": obs["level"], "lives": obs["lives"],
                "time_left": obs["time_left"], "barrels_left": obs.get("barrels_left"),
                "player": obs["player"], "kong": obs.get("kong"), "barrels_on_board": barrels,
                "top_floor": obs["top_floor"], "goal_x": obs["goal_x"], "ladders": obs["ladders"],
                "strategist": {"plan": self.director.strategy, "about_the_player": self.director.opponent_model,
                               "knobs": self.knobset.snapshot()},
                "recent_barrels": obs["barrel_outcomes"][-5:]}

    @staticmethod
    def _valid_tactics(reply):
        if not isinstance(reply, dict) or not isinstance(reply.get("throws"), list) \
                or not isinstance(reply.get("kong_x"), (int, float)):
            raise LLMError("bad tactician reply")
        throws = [t for t in reply["throws"] if isinstance(t, dict)]
        return {"throws": throws, "kong_x": min(1.0, max(0.0, float(reply["kong_x"])))}

    def _tactics_ready(self) -> dict | None:
        if self._tac_settled or self._tac_job is None:
            return None
        if self._tac_job.done():
            try:
                reply = self._tac_job.result()
            except Exception:      # LLMError or network trouble: use the knobs for this window
                reply = None
            self._tac_settled = True
            if reply is None:
                self.tactics["late"] += 1
                return super().plan(self._tac_obs)
            self.tactics["on_time"] += 1
            return {"throws": reply["throws"], "move_to": reply["kong_x"]}
        if self.game is not None and self.game.t > self._tac_deadline:
            self._tac_settled = True               # too slow for this window: knobs it is
            self.tactics["late"] += 1
            return super().plan(self._tac_obs)
        return None

    def ready_plan(self):
        """Background mode: new lines, new knobs and tactician throws are applied as they arrive."""
        self._cast_tick()
        if self.background and self.director.poll():
            self._after_apply(self.game.t if self.game is not None else 0.0)
            return self._announce({"throws": []})
        return self._tactics_ready()

    def context(self, obs: dict) -> dict:
        return {
            "t": obs["t"], "level": obs["level"], "lives": obs["lives"], "time_left": obs["time_left"],
            "player": obs["player"], "top_floor": obs["top_floor"], "goal_x": obs["goal_x"],
            "ladders": obs["ladders"], "rules": obs["rules"], "barrels_on_board": obs["barrels_on_board"],
            "barrels_left": obs.get("barrels_left"), "barrel_budget": obs.get("barrel_budget"),
            "kong": obs.get("kong"),
            "player_habits": obs["habits"],
            "what_happened_to_recent_barrels": obs["barrel_outcomes"][-10:],
            "recent_events": [e for e in obs["recent_events"] if e["event"] not in ("taunt", "throw")][-8:],
        }

    @staticmethod
    def outcomes(obs: dict) -> dict:
        s = obs["stats"]
        return {"player_hit_by_barrel": s["hits_barrel"], "player_hit_by_fireball": s["hits_fireball"],
                "player_timed_out": s["hits_timeout"], "barrels_jumped_by_player": s["jumped_over"],
                "floors_climbed_by_player": s["climbs"], "level": obs["level"]}

    def usage(self) -> dict:
        return self.director.client.total_usage()

    def roles(self) -> list[tuple[str, Layer]]:
        out = [("strategist", self.director.layer)]
        if self.tactician is not None:
            out.append(("tactician", self.tactician))
        return out + [("lines", self.voice)]

    def llm_status(self) -> tuple[str, str]:
        """(each role's model@provider with its average reply time, running cost) for the screen,
        e.g. "strategist gpt-oss-120b@cerebras 2.1s · tactician @groq 0.5s 3% late · voice @groq 0.4s"."""
        per_layer = getattr(self.director.client, "usage", {})
        parts, last_model = [], None
        for role, layer in self.roles():
            u = per_layer.get(layer.name) if isinstance(per_layer, dict) else None
            model = layer.model.split("/")[-1]
            where = ("" if model == last_model else model) + (f"@{layer.provider}" if layer.provider else "")
            last_model = model
            text = f"{role} {where or model}"
            if u and u["calls"]:
                text += f" {u['seconds'] / u['calls']:.1f}s"
            if role == "tactician" and self.tactics["windows"]:
                text += f" {round(100 * self.tactics['late'] / self.tactics['windows'])}% late"
            parts.append(text)
        total = self.usage()
        return " · ".join(parts) + f" · {total.get('calls', 0)} calls", f"${total.get('cost_usd', 0.0):.4f}"

    def display(self) -> str:
        """One line for human viewers: the current strategy and the knobs it last changed."""
        line = self.director.strategy or "(thinking...)"
        if self.knobset.history:
            changed = []
            for name, value in self.knobset.history[-1]["after"].items():
                if isinstance(value, dict):
                    top = max(value, key=value.get)
                    changed.append(f"{name}:{top}")
                elif isinstance(value, bool):
                    changed.append(f"{name} {'on' if value else 'off'}")
                else:
                    changed.append(f"{name}={value:g}")
            line += "  [" + ", ".join(changed) + "]"
        return line
