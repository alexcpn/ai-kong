"""Donkey Kong arena engine.

A deterministic, lockstep simulation: the world advances one fixed tick (TICK seconds)
per call to Game.step(keys). Kong is a pluggable policy that decides, every few game
seconds, which barrels to throw, how fast, and which way they should route.

Board coordinates: integer columns x (right is +), rows y growing DOWNWARD. Girders are
numbered 0 (bottom) upward; layout.floors[i] is the row a character stands on along
girder i. A ladder (x, b) joins girder b and girder b+1 at column x.

Standard library only.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace

TICK = 0.05
SPEED_FACTORS = {"slow": 1.35, "normal": 1.0, "fast": 0.75}
ROUTES = ("always", "never", "random", "toward_player")


# --------------------------------------------------------------------------- configuration

@dataclass(frozen=True)
class Params:
    gravity: float = 46.0            # rows / s^2
    jump_velocity: float = 13.6      # rows / s at take-off
    hold_gravity_scale: float = 0.80  # gravity multiplier while jump is held on the way up
    max_hold: float = 0.26           # seconds of held-jump boost per jump
    max_fall_speed: float = 28.0
    walk_gap: float = 0.075          # seconds per column while walking
    climb_gap: float = 0.11          # seconds per row while climbing
    barrel_gap: float = 0.15         # seconds per column for a "normal" barrel on level 1
    barrel_fall_time: float = 0.45   # seconds for a barrel to drop between girders
    fireball_gap: float = 0.20       # seconds per column for fireballs on level 2
    invuln_time: float = 2.0         # grace after (re)spawning
    start_lives: int = 3
    level_seconds: float = 60.0      # running out costs a life
    max_levels: int = 5              # clearing this many levels wins the episode
    kong_interval: float = 3.0       # Kong decides this often
    hit_rows: float = 0.55           # vertical distance that counts as touching
    barrel_budget: int | None = None  # barrels Kong may throw per level (None = unlimited)
    barrel_budget_per_level: int = 0  # extra barrels for each level after the first
    anger_decay: float = 2.0          # anger (0-100) Kong loses per second while at the top
    rampage_seconds: float = 9.0      # time an enraged Kong spends on the player's girder before going home
    rampage_max_seconds: float = 25.0  # cap on the whole trip down
    kong_walk_gap: float = 0.08       # seconds per column while Kong is down (the player walks at 0.075)
    kong_climb_gap: float = 0.12      # seconds per row while Kong is on a ladder
    rampage_throw_gap: float = 2.0    # Kong throws on the way down too, aimed at the player
    kong_moves: bool = False          # Kong roams the girders above the player and throws from there
    kong_floors_above: int = 2        # ...staying this many girders above the player (or at the top)
    throw_gap_scale: float = 1.0      # < 1 lets Kong throw more often than the level's base pace
    extra_barrels: int = 0            # more barrels on screen and per decision than the level's base

    def scaled(self, rng: random.Random, jitter: float) -> "Params":
        """A copy with physics/pacing jittered by up to +/- jitter (fraction)."""
        def j(v: float) -> float:
            return round(v * (1 + rng.uniform(-jitter, jitter)), 4)
        return replace(self, gravity=j(self.gravity), jump_velocity=j(self.jump_velocity),
                       walk_gap=j(self.walk_gap), climb_gap=j(self.climb_gap), barrel_gap=j(self.barrel_gap),
                       fireball_gap=j(self.fireball_gap))


@dataclass(frozen=True)
class Layout:
    width: int
    floors: tuple[int, ...]               # standing row per girder, bottom first
    x_min: int
    x_max: int
    ladders: tuple[tuple[int, int], ...]  # (x, bottom_floor)
    goal_x: int
    spawn_x: int                          # where Kong's barrels appear on the top girder

    @property
    def top(self) -> int:
        return len(self.floors) - 1

    def to_dict(self) -> dict:
        return {"width": self.width, "floors": list(self.floors), "x_min": self.x_min, "x_max": self.x_max,
                "ladders": [{"x": x, "bottom_floor": b, "top_floor": b + 1} for x, b in self.ladders],
                "goal": {"floor": self.top, "x": self.goal_x}, "spawn_x": self.spawn_x}


def generate_layout(seed: int, floors: int | None = None, max_ladders: int = 2) -> Layout:
    """Random zig-zag board: 5-6 girders, 1-max_ladders ladders between each pair."""
    rng = random.Random(seed)
    n = floors or rng.choice((5, 5, 6))
    gap = 4
    bottom = 1 + gap * (n - 1)
    rows = tuple(bottom - gap * i for i in range(n))
    x_min, x_max, width = 2, 53, 56
    ladders: list[tuple[int, int]] = []
    prev_xs: list[int] = []
    for b in range(n - 1):
        # alternate sides like the arcade, so the player has to cross each girder
        lo, hi = (x_max - 16, x_max - 3) if b % 2 == 0 else (x_min + 3, x_min + 16)
        count = rng.randint(1, max_ladders)
        xs: list[int] = []
        for k in range(count):
            for _ in range(50):
                x = rng.randint(lo, hi) if k == 0 else rng.randint(x_min + 8, x_max - 8)
                if all(abs(x - o) >= 6 for o in xs + prev_xs):
                    xs.append(x)
                    break
        ladders += [(x, b) for x in xs]
        prev_xs = xs
    top_ladders = [x for x, b in ladders if b == n - 2]
    candidates = [x for x in range(x_min + 14, x_max - 1) if all(abs(x - t) >= 8 for t in top_ladders)]
    goal = rng.choice(candidates) if candidates else x_max - 4
    return Layout(width=width, floors=rows, x_min=x_min, x_max=x_max, ladders=tuple(sorted(ladders)),
                  goal_x=goal, spawn_x=x_min + 5)


CLASSIC_LAYOUT = Layout(width=56, floors=(17, 13, 9, 5, 1), x_min=2, x_max=53,
                        ladders=((48, 0), (8, 1), (48, 2), (8, 3)), goal_x=44, spawn_x=7)


def level_rules(level: int, params: Params) -> dict:
    """What Kong is allowed to do on a level. Difficulty rises with level."""
    return {
        "max_barrels": min(9, 2 + level) + params.extra_barrels,
        "min_throw_gap": round(max(0.7, 2.2 - 0.3 * (level - 1)) * params.throw_gap_scale, 2),
        "speeds": ["slow", "normal"] if level == 1 else ["slow", "normal", "fast"],
        "barrel_gap": round(max(0.08, params.barrel_gap - 0.008 * (level - 1)), 4),
        "fireballs": min(3, level - 1),
        "max_throws_per_decision": min(4, 1 + level // 2) + params.extra_barrels,
    }


# --------------------------------------------------------------------------- entities

@dataclass
class Barrel:
    id: int
    x: int
    floor: int
    y: float
    direction: int
    step_gap: float
    route: str
    step_at: float
    falling: bool = False
    fall_from: float = 0.0
    fall_to: float = 0.0
    fall_started: float = 0.0
    fall_floor: int = 0
    fall_direction: int = 1
    passed_player: bool = False
    thrown_at: float = 0.0
    speed_name: str = "normal"


@dataclass
class Fireball:
    id: int
    x: int
    floor: int
    y: float
    direction: int
    step_gap: float
    step_at: float
    climbing: bool = False
    target_floor: int = 0


@dataclass
class Player:
    x: int
    y: float
    floor: int
    mode: str = "ground"          # ground | air | climb
    vy: float = 0.0
    hold_used: float = 0.0
    climb_from: int = 0
    climb_target: int = 0
    invuln_until: float = 0.0
    next_walk: float = 0.0
    next_climb: float = 0.0


@dataclass
class KongBody:
    x: int
    floor: int
    y: float
    mode: str = "perch"           # perch (top, throwing) | rampage (hunting the player) | return
    climb_to: int | None = None
    step_at: float = 0.0
    until: float = 0.0
    next_throw: float = 0.0
    arrived: bool = False         # reached the player's girder this rampage


@dataclass
class Throw:
    at: float
    speed: str
    route: str
    plan: int = 0


# --------------------------------------------------------------------------- Kong interface

class Kong:
    """Base class. plan(obs) returns {"throws": [{"delay", "speed", "route"}...], "taunt": str}.

    Optional: "strategy" (one line, shown to human viewers only, never to player agents).
    A Kong that thinks slowly can also define ready_plan() -> dict | None; the game polls it
    every tick and applies a plan as soon as one is ready (used for real-time human play).
    """

    name = "kong"

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def plan(self, obs: dict) -> dict:
        raise NotImplementedError


# --------------------------------------------------------------------------- the game

@dataclass
class Result:
    levels_cleared: int
    progress: float          # highest girder reached on the unfinished level / top girder
    won: bool
    lives_left: int
    ticks: int
    reason: str

    @property
    def score(self) -> float:
        """Episode score in [0, max_levels]: whole levels plus partial progress on the last one."""
        return round(self.levels_cleared + (0 if self.won else 0.9 * self.progress), 4)

    def to_dict(self) -> dict:
        return {**asdict(self), "score": self.score}


class Game:
    def __init__(self, layout: Layout, params: Params, kong: Kong, seed: int = 0) -> None:
        self.layout = layout
        self.params = params
        self.kong = kong
        self.rng = random.Random(seed)
        kong.reset(seed * 7919 + 17)
        self.t = 0.0
        self.ticks = 0
        self.level = 1
        self.lives = params.start_lives
        self.levels_cleared = 0
        self.over = False
        self.won = False
        self.end_reason = ""
        self.taunt = ""
        self.events: list[dict] = []
        self.stats = {"jumped_over": 0, "hits_barrel": 0, "hits_fireball": 0, "hits_timeout": 0,
                      "stomps": 0, "climbs": 0, "kong_defeats": 0}
        self.anger = 0.0
        self._kong_beaten = False
        self.kong_errors = 0
        self.kong_strategy = ""
        self.plans = 0
        self.barrel_log: list[dict] = []
        self.habits = {"ladder_waits": [], "jump_leads": [], "ladder_use": {}, "climbs": 0,
                       "climbs_under_threat": 0, "deaths": []}
        self._wait = {"x": None, "t": 0.0}
        self._next_id = 1
        self._prev_keys: frozenset[str] = frozenset()
        self._start_level()

    # ----------------------------------------------------------------- helpers

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def _event(self, kind: str, **data) -> None:
        self.events.append({"t": round(self.t, 2), "event": kind, **data})
        if len(self.events) > 200:
            del self.events[:100]

    @property
    def rules(self) -> dict:
        return level_rules(self.level, self.params)

    @property
    def level_budget(self) -> int | None:
        prm = self.params
        if prm.barrel_budget is None:
            return None
        return prm.barrel_budget + prm.barrel_budget_per_level * (self.level - 1)

    def _start_level(self) -> None:
        lay = self.layout
        self.barrels_left = self.level_budget      # refilled per level, not per life
        self.anger = 0.0
        self.kong_body = KongBody(x=lay.x_min + 1, floor=lay.top, y=float(lay.floors[lay.top]))
        self.kong_aim, self._kong_directed, self._sway = 0.0, False, 0   # aim: 0 = left .. 1 = right
        self.player = Player(x=lay.x_min + 2, y=float(lay.floors[0]), floor=0,
                             invuln_until=self.t + self.params.invuln_time)
        self.highest = 0
        self.time_left = self.params.level_seconds
        self.barrels: list[Barrel] = []
        self.fireballs: list[Fireball] = []
        self.queue: list[Throw] = []
        self.last_throw = -1e9
        self.next_decision = self.t + 0.5
        for _ in range(self.rules["fireballs"]):
            self._spawn_fireball()

    def _respawn(self) -> None:
        """After losing a life: same level, fresh positions and hazards, timer restarted."""
        keep = (self.events, self.barrels_left, self.anger, self.kong_aim, self._kong_directed)
        self._start_level()
        self.events, self.barrels_left, self.anger, self.kong_aim, self._kong_directed = keep

    def _spawn_fireball(self) -> None:
        lay = self.layout
        floor = self.rng.randrange(1, lay.top) if lay.top > 1 else 0
        x = self.rng.randint(lay.x_min + 6, lay.x_max - 2)
        gap = max(0.11, self.params.fireball_gap - 0.012 * (self.level - 2))
        self.fireballs.append(Fireball(id=self._id(), x=x, floor=floor, y=float(lay.floors[floor]),
                                       direction=self.rng.choice((-1, 1)), step_gap=gap, step_at=self.t + 1.5))

    def ladder_up(self, floor: int, x: int) -> bool:
        return (x, floor) in self.layout.ladders

    def ladder_down(self, floor: int, x: int) -> bool:
        return floor > 0 and (x, floor - 1) in self.layout.ladders

    # ----------------------------------------------------------------- Kong

    def kong_observation(self) -> dict:
        p = self.player
        rules = self.rules
        return {
            "t": round(self.t, 2),
            "level": self.level,
            "lives": self.lives,
            "time_left": round(self.time_left, 1),
            "player": {"floor": p.floor, "x": p.x, "mode": p.mode},
            "top_floor": self.layout.top,
            "spawn_x": self.layout.spawn_x,
            "goal_x": self.layout.goal_x,
            "ladders": [{"x": x, "bottom_floor": b} for x, b in self.layout.ladders],
            "barrels_on_board": len(self.barrels) + len(self.queue),
            "barrels_left": self.barrels_left,          # None = unlimited
            "barrel_budget": self.level_budget,
            "kong": {"anger": round(self.anger), "mode": self.kong_body.mode, "floor": self.kong_body.floor,
                     "x": self.kong_body.x},
            "rules": rules,
            "interval": self.params.kong_interval,
            "recent_events": self.events[-12:],
            "stats": dict(self.stats),
            "barrel_outcomes": [dict(e) for e in self.barrel_log[-12:]],
            "habits": self.habit_summary(),
        }

    def habit_summary(self) -> dict:
        h = self.habits
        waits, leads = h["ladder_waits"], h["jump_leads"]
        favourite = sorted(h["ladder_use"].items(), key=lambda kv: -kv[1])[:3]
        return {
            "waits_at_ladder_foot": len(waits),
            "avg_wait_s": round(sum(waits) / len(waits), 1) if waits else None,
            "avg_jump_distance_cols": round(sum(leads) / len(leads), 1) if leads else None,
            "favourite_ladders": [{"x": x, "climbs": n} for x, n in favourite],
            "climbs": h["climbs"],
            "climbed_with_barrel_coming_above": h["climbs_under_threat"],
            "recent_deaths": h["deaths"][-5:],
        }

    def _decide(self) -> None:
        try:
            plan = self.kong.plan(self.kong_observation()) or {}
        except Exception as exc:  # a broken Kong must never stop the game
            self.kong_errors += 1
            plan = {"throws": [{"delay": 0.0, "speed": "normal", "route": "random"}],
                    "taunt": "", "error": str(exc)[:200]}
        self._apply_plan(plan)

    def _poll_kong(self) -> None:
        ready = getattr(self.kong, "ready_plan", None)
        if ready is None:
            return
        try:
            plan = ready()
        except Exception:
            self.kong_errors += 1
            return
        if plan:
            self._apply_plan(plan)

    def _apply_plan(self, plan) -> None:
        self.plans += 1
        rules = self.rules
        throws = plan.get("throws") if isinstance(plan, dict) else None
        if not isinstance(throws, list):
            throws, self.kong_errors = [], self.kong_errors + 1
        for item in throws[: rules["max_throws_per_decision"]]:
            if not isinstance(item, dict):
                continue
            try:
                delay = min(self.params.kong_interval, max(0.0, float(item.get("delay", 0.0))))
            except (TypeError, ValueError):
                delay = 0.0
            speed = item.get("speed") if item.get("speed") in rules["speeds"] else "normal"
            route = item.get("route") if item.get("route") in ROUTES else "random"
            self.queue.append(Throw(at=self.t + delay, speed=speed, route=route, plan=self.plans))
        self.queue.sort(key=lambda q: q.at)
        move_to = plan.get("move_to") if isinstance(plan, dict) else None
        if self.params.kong_moves and isinstance(move_to, (int, float)) and not isinstance(move_to, bool):
            self.kong_aim, self._kong_directed = min(1.0, max(0.0, float(move_to))), True
        taunt = plan.get("taunt") if isinstance(plan, dict) else ""
        if isinstance(taunt, str) and taunt.strip():
            self.taunt = " ".join(taunt.split())[:70]
            self._event("taunt", text=self.taunt)
        strategy = plan.get("strategy") if isinstance(plan, dict) else ""
        if isinstance(strategy, str) and strategy.strip():
            self.kong_strategy = " ".join(strategy.split())[:90]

    def _release_throws(self) -> None:
        rules = self.rules
        if self.kong_body.mode != "perch":
            self.queue.clear()             # Kong is away from his barrel pile
            return
        while self.queue and self.queue[0].at <= self.t:
            if self.barrels_left is not None and self.barrels_left <= 0:
                self.queue.clear()         # out of barrels until the next level
                break
            if len(self.barrels) >= rules["max_barrels"]:
                q = self.queue.pop(0)      # no room: Kong wasted the throw
                self._log_barrel(None, q, "wasted (board full)")
                continue
            if self.t - self.last_throw < rules["min_throw_gap"]:
                self.queue[0].at = self.last_throw + rules["min_throw_gap"]
                self.queue.sort(key=lambda q: q.at)
                break
            lay, k = self.layout, self.kong_body
            if self.params.kong_moves and k.climb_to is not None:
                break                      # can't throw from a ladder; wait until he is on a girder
            q = self.queue.pop(0)
            gap = rules["barrel_gap"] * SPEED_FACTORS[q.speed]
            x, direction, floor = lay.spawn_x, 1, lay.top
            if self.params.kong_moves:                 # from wherever Kong stands, rolled toward the player
                direction = 1 if self.player.x >= k.x else -1
                x, floor = min(lay.x_max, max(lay.x_min, k.x + direction)), k.floor
            barrel = Barrel(id=self._id(), x=x, floor=floor, y=float(lay.floors[floor]),
                            direction=direction, step_gap=gap, route=q.route, step_at=self.t + gap,
                            thrown_at=self.t, speed_name=q.speed)
            self.barrels.append(barrel)
            self._log_barrel(barrel.id, q, None)
            self.last_throw = self.t
            if self.barrels_left is not None:
                self.barrels_left -= 1
            self._event("throw", speed=q.speed, route=q.route)

    def _log_barrel(self, ident, q: Throw, outcome) -> None:
        self.barrel_log.append({"id": ident, "plan": q.plan, "thrown_at": round(self.t, 2), "speed": q.speed,
                                "route": q.route, "outcome": outcome})
        if len(self.barrel_log) > 60:
            del self.barrel_log[:20]

    def _outcome(self, ident: int, outcome: str) -> None:
        for entry in reversed(self.barrel_log):
            if entry["id"] == ident:
                if entry["outcome"] is None:
                    entry["outcome"] = outcome
                return

    # ----------------------------------------------------------------- Kong's temper

    def provoke(self, amount: float, charge: bool = False) -> None:
        """The player taunted Kong. At full anger (or if he takes the bait) he comes down after them."""
        k = self.kong_body
        self.anger = min(100.0, self.anger + amount)
        self._event("provoked", anger=round(self.anger))
        if (charge or self.anger >= 100) and k.mode == "perch":
            k.mode, k.until, k.step_at, k.next_throw = "rampage", self.t + self.params.rampage_max_seconds, self.t, self.t + 0.8
            k.arrived = False
            self.anger = 100.0
            self.taunt = "RAAAARGH!!"
            self._event("kong_rampage")
        elif amount > 0 and k.mode == "perch":
            self.taunt = "GRRRR!" if self.anger >= 50 else "Hmph."

    def kong_range(self, floor: int) -> tuple[int, int]:
        """Where Kong may stand on a girder (on the top one, not within a few columns of Pauline)."""
        lay = self.layout
        lo = lay.x_min + 1
        return (lo, max(lo, lay.goal_x - 5)) if floor == lay.top else (lo, lay.x_max - 1)

    def roam_floor(self) -> int:
        """The girder Kong keeps to when he is not storming: a few above the player, or the top."""
        if not self.params.kong_moves:
            return self.layout.top
        return min(self.layout.top, self.player.floor + self.params.kong_floors_above)

    def _aim_x(self, floor: int) -> int:
        lo, hi = self.kong_range(floor)
        return round(lo + self.kong_aim * (hi - lo))

    def _path_step(self, e, goal_floor: int, climb_gap: float, walk_gap: float) -> bool:
        """One step of Kong toward goal_floor over girders and ladders.
        Returns True when e already stands on goal_floor (the caller then moves it along the girder)."""
        lay = self.layout
        if e.climb_to is not None:
            e.step_at = self.t + climb_gap
            target = lay.floors[e.climb_to]
            e.y += 1.0 if target > e.y else -1.0
            if abs(e.y - target) < 1e-9:
                e.y, e.floor, e.climb_to = float(target), e.climb_to, None
            return False
        e.step_at = self.t + walk_gap
        if e.floor == goal_floor:
            return True
        down = goal_floor < e.floor                      # head for the nearest ladder the right way
        ladders = [x for x, b in lay.ladders if b == (e.floor - 1 if down else e.floor)]
        if ladders:
            lx = min(ladders, key=lambda x: abs(x - e.x))
            if lx == e.x:
                e.climb_to = e.floor - 1 if down else e.floor + 1
            else:
                e.x += 1 if lx > e.x else -1
        return False

    def _pace(self) -> None:
        """Kong not storming: keep to his girder above the player, walk to where his strategy wants
        him, shifting about a little."""
        k, prm = self.kong_body, self.params
        if self.t + 1e-9 < k.step_at:
            return
        if not self._path_step(k, self.roam_floor(), prm.kong_climb_gap, prm.kong_walk_gap):
            return                                       # still changing girders
        k.step_at = self.t + prm.kong_walk_gap * 2.5
        lo, hi = self.kong_range(k.floor)
        aim = min(hi, max(lo, self._aim_x(k.floor) + self._sway))
        if k.x != aim:
            k.x += 1 if aim > k.x else -1
        elif self.rng.random() < 0.25:
            self._sway = self.rng.randint(-3, 3)
            if not self._kong_directed and self.rng.random() < 0.3:   # nobody directs him: roam
                self.kong_aim = self.rng.random()

    def _update_kong(self) -> None:
        k, lay, p, prm = self.kong_body, self.layout, self.player, self.params
        if k.mode == "perch":
            self.anger = max(0.0, self.anger - prm.anger_decay * TICK)
            if prm.kong_moves:
                self._pace()
            return
        if k.mode == "rampage" and self.t >= k.until:
            k.mode = "return"
        if self.t + 1e-9 < k.step_at:
            return
        home_floor = self.roam_floor()
        home = self._aim_x(home_floor)
        goal_floor, goal_x = (home_floor, home) if k.mode == "return" else (p.floor, p.x)
        if k.mode == "rampage" and k.climb_to is None:
            if k.floor == goal_floor and not k.arrived:
                k.arrived, k.until = True, min(k.until, self.t + prm.rampage_seconds)
            if self.t >= k.next_throw and p.mode != "climb":
                k.next_throw = self.t + prm.rampage_throw_gap
                self._kong_throw(1 if p.x > k.x else -1)
        if not self._path_step(k, goal_floor, prm.kong_climb_gap, prm.kong_walk_gap):
            return
        if k.mode == "return":
            if k.x == home:
                k.mode = "perch"
                self.anger = min(self.anger, 40.0)
            else:
                k.x += 1 if home > k.x else -1
            return
        dx = goal_x - k.x                                # same girder: keep a throwing distance
        toward = 1 if dx > 0 else -1
        if abs(dx) > 9:
            k.x += toward
        elif abs(dx) < 5 and lay.x_min + 1 <= k.x - toward <= lay.x_max - 1:
            k.x -= toward

    def _kong_throw(self, direction: int) -> None:
        """A point-blank barrel from wherever Kong is standing."""
        if (self.barrels_left is not None and self.barrels_left <= 0) or len(self.barrels) >= self.rules["max_barrels"]:
            return
        k = self.kong_body
        gap = self.rules["barrel_gap"]
        barrel = Barrel(id=self._id(), x=k.x + direction, floor=k.floor, y=k.y, direction=direction, step_gap=gap,
                        route="toward_player", step_at=self.t + gap, thrown_at=self.t)
        self.barrels.append(barrel)
        self._log_barrel(barrel.id, Throw(at=self.t, speed="normal", route="toward_player"), None)
        if self.barrels_left is not None:
            self.barrels_left -= 1
        self._event("throw", speed="normal", route="point_blank")

    # ----------------------------------------------------------------- player

    def _walk(self, delta: int) -> None:
        p, lay = self.player, self.layout
        if p.mode == "climb":
            if abs(p.y - lay.floors[p.climb_target]) < 1e-9 or abs(p.y - lay.floors[p.climb_from]) < 1e-9:
                self._finish_climb()
            else:
                p.mode, p.vy, p.hold_used = "air", 0.0, 0.0   # step off the ladder and drop
                return
        p.x = max(lay.x_min, min(lay.x_max, p.x + delta))

    def _climb(self, direction: int) -> None:
        p, lay = self.player, self.layout
        if p.mode == "air":
            return
        if p.mode == "ground":
            if direction > 0:
                if p.floor >= lay.top or not self.ladder_up(p.floor, p.x):
                    return
                p.climb_from, p.climb_target = p.floor, p.floor + 1
                self._note_climb(p.x, p.floor)
            else:
                if not self.ladder_down(p.floor, p.x):
                    return
                p.climb_from, p.climb_target = p.floor, p.floor - 1
            p.mode = "climb"
        else:
            going_up = p.climb_target > p.climb_from
            if (direction > 0) != going_up:
                p.climb_from, p.climb_target = p.climb_target, p.climb_from
        target = lay.floors[p.climb_target]
        p.y += 1.0 if target > p.y else -1.0
        if abs(p.y - target) < 1e-9:
            self._finish_climb()

    def _note_climb(self, x: int, floor: int) -> None:
        h = self.habits
        h["climbs"] += 1
        h["ladder_use"][x] = h["ladder_use"].get(x, 0) + 1
        threat = any(b.floor == floor + 1 and not b.falling and abs(b.x - x) <= 8 and (x - b.x) * b.direction >= 0
                     for b in self.barrels)
        h["climbs_under_threat"] += threat

    def _track_waiting(self) -> None:
        """How long the player stands still beside a ladder foot before climbing."""
        p, w = self.player, self._wait
        near = p.mode == "ground" and any(b == p.floor and abs(x - p.x) <= 1 for x, b in self.layout.ladders)
        if near and w["x"] == (p.x, p.floor):
            w["t"] += TICK
            return
        if w["t"] >= 0.5:
            self.habits["ladder_waits"] = (self.habits["ladder_waits"] + [round(w["t"], 1)])[-20:]
        w["x"], w["t"] = ((p.x, p.floor) if near else None), 0.0

    def _finish_climb(self) -> None:
        p = self.player
        p.y = float(self.layout.floors[p.climb_target])
        p.floor, p.mode = p.climb_target, "ground"
        if p.floor > self.highest:
            self.highest = p.floor
            self.stats["climbs"] += 1
            self._event("climbed", floor=p.floor, x=p.x)

    def _jump(self) -> None:
        p = self.player
        if p.mode == "ground":
            leads = [abs(b.x - p.x) for b in self.barrels
                     if not b.falling and b.floor == p.floor and (p.x - b.x) * b.direction >= 0]
            if leads:
                self.habits["jump_leads"] = (self.habits["jump_leads"] + [min(leads)])[-20:]
            p.mode, p.vy, p.hold_used = "air", -self.params.jump_velocity, 0.0

    def _physics(self, jump_held: bool) -> None:
        p, prm, lay = self.player, self.params, self.layout
        if p.mode != "air":
            return
        g = prm.gravity
        if jump_held and p.vy < 0 and p.hold_used < prm.max_hold:
            p.hold_used = min(prm.max_hold, p.hold_used + TICK)
            g *= prm.hold_gravity_scale
        p.vy = min(prm.max_fall_speed, p.vy + g * TICK)
        prev = p.y
        p.y += p.vy * TICK
        ceiling = self._ceiling(prev)
        if ceiling is not None and p.y < ceiling:
            p.y, p.vy = ceiling, 0.0
        if p.vy >= 0:
            landing = None
            for i, row in enumerate(lay.floors):
                if prev - 1e-9 <= row <= p.y + 1e-9 and (landing is None or row < lay.floors[landing]):
                    landing = i
            if landing is not None:
                p.y, p.floor, p.mode, p.vy, p.hold_used = float(lay.floors[landing]), landing, "ground", 0.0, 0.0
                if landing > self.highest:
                    self.highest = landing

    def _ceiling(self, y: float) -> float | None:
        """Lowest row a jumper can reach under the girder directly above y."""
        above = [row for row in self.layout.floors if row < y - 1e-9]
        if not above:
            return None
        return max(above) + 2.0   # girder sits one row below its standing row; stay below it

    # ----------------------------------------------------------------- hazards

    def _update_barrels(self) -> None:
        lay = self.layout
        for b in self.barrels[:]:
            if b.falling:
                progress = (self.t - b.fall_started) / self.params.barrel_fall_time
                if progress >= 1.0:
                    b.y, b.floor, b.direction, b.falling = b.fall_to, b.fall_floor, b.fall_direction, False
                    b.step_at = self.t + b.step_gap
                else:
                    b.y = b.fall_from + (b.fall_to - b.fall_from) * progress
                continue
            if self.t + 1e-9 < b.step_at:
                continue
            b.step_at += b.step_gap
            nx = b.x + b.direction
            if lay.x_min <= nx <= lay.x_max:
                b.x = nx
                if b.floor > 0 and (b.x, b.floor - 1) in lay.ladders and self._descend(b):
                    self._drop(b, reverse=True)
            elif b.floor > 0:
                self._drop(b, reverse=True)
            else:
                self.barrels.remove(b)
                self._outcome(b.id, "rolled off the bottom")
                self._event("barrel_exited", id=b.id)

    def _descend(self, b: Barrel) -> bool:
        if b.route == "always":
            return True
        if b.route == "never":
            return False
        if b.route == "toward_player":
            return self.player.floor < b.floor
        return self.rng.random() < 0.5

    def _drop(self, b: Barrel, reverse: bool) -> None:
        b.falling, b.fall_from, b.fall_started = True, b.y, self.t
        b.fall_floor = b.floor - 1
        b.fall_to = float(self.layout.floors[b.fall_floor])
        b.fall_direction = self.rng.choice((-1, 1)) if b.fall_floor == 0 else (-b.direction if reverse else b.direction)

    def _update_fireballs(self) -> None:
        lay, p = self.layout, self.player
        for f in self.fireballs:
            if self.t + 1e-9 < f.step_at:
                continue
            if f.climbing:
                f.step_at = self.t + 0.1
                target = lay.floors[f.target_floor]
                f.y += 1.0 if target > f.y else -1.0
                if abs(f.y - target) < 1e-9:
                    f.y, f.floor, f.climbing = float(target), f.target_floor, False
                    f.step_at = self.t + f.step_gap
                continue
            f.step_at = self.t + f.step_gap
            if f.floor == p.floor and p.mode != "air" and p.x != f.x:
                f.direction = 1 if p.x > f.x else -1
            if f.floor != p.floor:
                up = (f.x, f.floor) in lay.ladders
                down = f.floor > 0 and (f.x, f.floor - 1) in lay.ladders
                target = None
                if p.floor > f.floor and up:
                    target = f.floor + 1
                elif p.floor < f.floor and down:
                    target = f.floor - 1
                elif (up or down) and self.rng.random() < 0.2:
                    target = f.floor + 1 if up else f.floor - 1
                if target is not None and 0 <= target <= lay.top:
                    f.climbing, f.target_floor = True, target
                    continue
            nx = f.x + f.direction
            if nx < lay.x_min or nx > lay.x_max:
                f.direction *= -1
            else:
                f.x = nx

    # ----------------------------------------------------------------- collisions / progress

    def _collide(self, prev_mode: str, prev_vy: float, prev_y: float) -> None:
        p, prm = self.player, self.params
        k = self.kong_body
        off_top = k.mode != "perch" or k.floor != self.layout.top or k.climb_to is not None
        if off_top and abs(k.x - p.x) <= 1 and k.y - 1.0 - 1e-9 <= p.y <= k.y + 0.5:
            self.stats["kong_defeats"] += 1              # caught him away from his top girder
            self._event("kong_defeated", floor=k.floor, x=k.x)
            self._kong_beaten = True
            return
        for b in self.barrels:
            if b.x == p.x and not b.passed_player and p.mode == "air" and b.y - p.y >= prm.hit_rows \
                    and abs(b.floor - p.floor) == 0:
                b.passed_player = True
                self.stats["jumped_over"] += 1
                self._outcome(b.id, f"jumped over on floor {p.floor} at x={p.x}")
                self._event("jumped_over", id=b.id)
        if self.t < p.invuln_until:
            return
        for b in self.barrels:
            if b.x == p.x and abs(b.y - p.y) < prm.hit_rows:
                self._outcome(b.id, f"HIT the player on floor {p.floor} at x={p.x} ({p.mode})")
                self._hit("barrel")
                return
        for f in self.fireballs:
            if f.x != p.x or abs(f.y - p.y) >= prm.hit_rows:
                continue
            if prev_mode == "air" and prev_vy > 0 and prev_y < f.y - 0.01:
                self.fireballs.remove(f)
                self.stats["stomps"] += 1
                self._event("stomped", id=f.id)
                return
            self._hit("fireball")
            return

    def _hit(self, cause: str) -> None:
        p = self.player
        self.habits["deaths"].append({"level": self.level, "floor": p.floor, "x": p.x, "mode": p.mode, "cause": cause})
        self.habits["deaths"] = self.habits["deaths"][-8:]
        for b in self.barrels:
            self._outcome(b.id, "removed when the player lost a life")
        self.lives -= 1
        self.stats["hits_" + cause] += 1
        self._event("player_hit", cause=cause, floor=self.player.floor, x=self.player.x, lives=self.lives)
        if self.lives <= 0:
            self.over, self.end_reason = True, "out of lives"
            return
        self._respawn()

    def _check_goal(self) -> None:
        p = self.player
        reached = p.mode == "ground" and p.floor == self.layout.top and abs(p.x - self.layout.goal_x) <= 1
        if reached or self._kong_beaten:
            self._kong_beaten = False
            self.levels_cleared += 1
            for b in self.barrels:
                self._outcome(b.id, "removed when the player cleared the level")
            self._event("level_cleared", level=self.level, time_left=round(self.time_left, 1))
            if self.levels_cleared >= self.params.max_levels:
                self.over, self.won, self.end_reason = True, True, "won"
                return
            self.level += 1
            self._start_level()

    # ----------------------------------------------------------------- public API

    def step(self, keys) -> None:
        """Advance one tick with these keys held: left, right, up, down, jump."""
        if self.over:
            return
        keys = frozenset(k for k in keys if k in ("left", "right", "up", "down", "jump"))
        p = self.player
        horiz = ("right" in keys) - ("left" in keys)
        vert = ("up" in keys) - ("down" in keys)
        if horiz:
            if self.t + 1e-9 >= p.next_walk:
                self._walk(horiz)
                p.next_walk = self.t + self.params.walk_gap
        else:
            p.next_walk = self.t
        if vert:
            if self.t + 1e-9 >= p.next_climb:
                self._climb(vert)
                p.next_climb = self.t + self.params.climb_gap
        else:
            p.next_climb = self.t
        if "jump" in keys and "jump" not in self._prev_keys:
            self._jump()
        self._prev_keys = keys

        self.t += TICK
        self.ticks += 1
        prev = (p.mode, p.vy, p.y)
        self._physics("jump" in keys)
        self._track_waiting()
        self._poll_kong()
        if self.t + 1e-9 >= self.next_decision:
            self._decide()
            self.next_decision = self.t + self.params.kong_interval
        self._release_throws()
        self._update_barrels()
        self._update_fireballs()
        self._update_kong()
        self._collide(*prev)
        if self.over:
            return
        self._check_goal()
        if self.over:
            return
        self.time_left = max(0.0, self.time_left - TICK)
        if self.time_left <= 0:
            self._hit("timeout")

    def state(self) -> dict:
        """What the player agent sees each tick (everything visible on screen)."""
        p = self.player
        return {
            "t": round(self.t, 4),
            "level": self.level,
            "lives": self.lives,
            "levels_cleared": self.levels_cleared,
            "time_left": round(self.time_left, 3),
            "player": {"x": p.x, "y": round(p.y, 4), "floor": p.floor, "mode": p.mode, "vy": round(p.vy, 4),
                       "invulnerable": self.t < p.invuln_until,
                       "climb_target": p.climb_target if p.mode == "climb" else None},
            "barrels": [{"id": b.id, "x": b.x, "y": round(b.y, 4), "floor": b.floor, "dir": b.direction,
                         "falling": b.falling, "step_gap": round(b.step_gap, 4),
                         "fall_to_floor": b.fall_floor if b.falling else None} for b in self.barrels],
            "fireballs": [{"id": f.id, "x": f.x, "y": round(f.y, 4), "floor": f.floor, "dir": f.direction,
                           "climbing": f.climbing} for f in self.fireballs],
            "kong": {"taunt": self.taunt, "next_decision_in": round(max(0.0, self.next_decision - self.t), 2),
                     "queued_throws": len(self.queue), "barrels_left": self.barrels_left,
                     "anger": round(self.anger, 1), "mode": self.kong_body.mode, "x": self.kong_body.x,
                     "y": round(self.kong_body.y, 4), "floor": self.kong_body.floor},
            "rules": self.rules,
        }

    def result(self, reason: str | None = None) -> Result:
        progress = 0.0 if self.won else self.highest / max(1, self.layout.top)
        return Result(levels_cleared=self.levels_cleared, progress=round(progress, 4), won=self.won,
                      lives_left=max(0, self.lives), ticks=self.ticks, reason=reason or self.end_reason)


def params_dict(params: Params) -> dict:
    return asdict(params)


@dataclass
class Variant:
    """One evaluation setting: a board, physics and a seed."""
    layout: Layout
    params: Params
    seed: int
    tags: dict = field(default_factory=dict)


def dev_variant(seed: int) -> Variant:
    """The variant family agents can train on: random boards, physics jittered up to 8%."""
    rng = random.Random(seed * 31 + 5)
    return Variant(layout=generate_layout(seed), params=Params().scaled(rng, 0.08), seed=seed)
