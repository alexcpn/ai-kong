"""Kong rebuilt on the director pattern.

ParametricKong is a deterministic Kong whose behaviour is fully set by a handful of knobs.
DirectorKong lets an LLM director (director/) turn those knobs from a profile of the player,
with a fairness guard that vetoes settings even an omniscient player could not survive.
Plans go through the same {"throws": [...]} interface as every Kong. Player taunts get a separate,
fast "voice" call (no reasoning) so Kong answers in a second or two and may charge down at once.
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # dk-game/, for director

from director import Director, FeasibilityGuard, KnobSet, KnobSpec, Layer, LLMError  # noqa: E402

from .engine import TICK, Kong  # noqa: E402
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

TEMPER. The player can taunt you (situation.player_taunts, newest last); each taunt raises
situation.kong.anger (0-100, it cools over time). At 100, or when you take the bait, Kong storms down
to the player's girder for a few seconds and throws point-blank barrels (from your supply) at them.

WHERE KONG IS (situation.kong). When Kong can move he roams a few girders ABOVE the player
(climbing as they climb, up to the top girder) and throws from there, so barrels arrive sooner. If the
player touches Kong anywhere but on the top girder he is DEFEATED and they clear the level.
A taunt can be a bluff ("I'm taking the left ladder") or a dare meant to make you waste barrels."""

VOICE_SYSTEM = """You are KONG in a terminal Donkey Kong game. The player just taunted you. Answer in
character and decide whether to take the bait.

Reply with JSON: say (<= 60 characters, playful, family-friendly, answering what they said) and charge
(true = lose your temper NOW and storm down the ladders after them). Charging is a gamble: up close your
point-blank barrels are deadly, but if the player touches you while you are down you are defeated and
they clear the level. At anger 100 you charge anyway. Weigh your anger, barrels_left, where the player
is (far below = long trip, near the top = they can reach you), and whether the taunt is bait or a bluff."""

VOICE_SCHEMA = {"type": "object", "properties": {"say": {"type": "string"}, "charge": {"type": "boolean"}},
                "required": ["say", "charge"], "additionalProperties": False}


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
                 layer: Layer | None = None, client=None, guard_rollouts: int = 3) -> None:
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
        self.player_taunts: list[dict] = []
        self.voice = Layer.from_env("voice", "none", 300)
        self._voice_pool = cf.ThreadPoolExecutor(max_workers=1)
        self._voice_job: cf.Future | None = None
        self._pending_taunt: str | None = None

    def provoke(self, text: str, t: float) -> None:
        """The player taunted Kong (the game has already raised his anger): answer fast, maybe charge."""
        self.player_taunts = (self.player_taunts + [{"t": round(t, 1), "said": text}])[-5:]
        self._pending_taunt = text
        self._start_voice()

    def _start_voice(self) -> None:
        if self._pending_taunt is None or self.game is None or (self._voice_job and not self._voice_job.done()):
            return
        g, text = self.game, self._pending_taunt
        self._pending_taunt = None
        prompt = {"player_says": text, "your_anger": round(g.anger), "barrels_left": g.barrels_left,
                  "player": {"floor": g.player.floor, "x": g.player.x, "mode": g.player.mode},
                  "top_floor": g.layout.top, "lives": g.lives, "time_left": round(g.time_left),
                  "you_are": g.kong_body.mode, "your_current_strategy": self.director.strategy,
                  "what_you_know_about_them": self.director.opponent_model,
                  "earlier_taunts": [x["said"] for x in self.player_taunts[:-1]]}
        self._voice_job = self._voice_pool.submit(self.director.client.ask, self.voice, VOICE_SYSTEM, prompt,
                                                  VOICE_SCHEMA, self._valid_voice)

    @staticmethod
    def _valid_voice(reply):
        if not isinstance(reply, dict) or not isinstance(reply.get("say"), str) \
                or not isinstance(reply.get("charge"), bool):
            raise LLMError("bad voice reply")
        return reply

    def _voice_reply(self) -> dict | None:
        job = self._voice_job
        if job is None or not job.done():
            return None
        self._voice_job = None
        try:
            reply = job.result()
        except Exception:      # LLMError or network trouble: the grunt already answered
            reply = None
        self._start_voice()    # a taunt that arrived meanwhile
        if not reply:
            return None
        if reply["charge"] and self.game is not None:
            self.game.provoke(0, charge=True)
        return {"throws": [], "taunt": " ".join(reply["say"].split())[:60]}

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
        return self._announce(super().plan(obs))

    def ready_plan(self):
        """Background mode: apply new knobs (and show the new strategy) as soon as they arrive."""
        voice = self._voice_reply()
        if voice:
            return voice
        if not self.background or not self.director.poll():
            return None
        self._after_apply(self.game.t if self.game is not None else 0.0)
        return self._announce({"throws": []})

    def context(self, obs: dict) -> dict:
        return {
            "t": obs["t"], "level": obs["level"], "lives": obs["lives"], "time_left": obs["time_left"],
            "player": obs["player"], "top_floor": obs["top_floor"], "goal_x": obs["goal_x"],
            "ladders": obs["ladders"], "rules": obs["rules"], "barrels_on_board": obs["barrels_on_board"],
            "barrels_left": obs.get("barrels_left"), "barrel_budget": obs.get("barrel_budget"),
            "player_taunts": self.player_taunts, "kong": obs.get("kong"),
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

    def models(self) -> str:
        """Which models play Kong, e.g. "strategist claude-haiku-5.5 (medium) · voice claude-haiku-5.5"."""
        brain, voice = self.director.layer, self.voice
        short = lambda m: m.split("/")[-1]  # noqa: E731
        reasoning = f" ({brain.reasoning})" if brain.reasoning in ("low", "medium", "high") else ""
        return f"strategist {short(brain.model)}{reasoning} · voice {short(voice.model)}"

    def llm_status(self) -> tuple[str, str]:
        """(models with average reply times and call count, running cost) for the screen."""
        per_layer = getattr(self.director.client, "usage", {})
        parts = []
        for layer in (self.director.layer, self.voice):
            u = per_layer.get(layer.name) if isinstance(per_layer, dict) else None
            wait = f", {u['seconds'] / u['calls']:.1f}s" if u and u["calls"] else ""
            reasoning = f"{layer.reasoning}" if layer.reasoning in ("low", "medium", "high") else ""
            extra = ", ".join(x for x in (reasoning, wait.lstrip(", ")) if x)
            role = "strategist" if layer is self.director.layer else "voice"
            parts.append(f"{role} {layer.model.split('/')[-1]}" + (f" ({extra})" if extra else ""))
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
