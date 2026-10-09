"""Omniscient lookahead player (in-process): simulates candidate moves on copies of the real game.

It sees the future (it copies Kong's random state too), so it is an upper bound, not a fair
competitor. Used by the director's fairness guard ("can even a perfect player survive these
settings?") and as a strong reference player.
"""

from __future__ import annotations

import copy

ACTIONS = ["greedy", (), ("left",), ("right",), ("up",), ("down",), ("jump",), ("jump", "left"),
           ("jump", "right")]
FOLLOW_UPS = ["greedy", "wait"]


def target(game) -> int:
    p, lay = game.player, game.layout
    if p.floor == lay.top:
        return lay.goal_x
    ups = [x for x, b in lay.ladders if b == p.floor]
    return min(ups, key=lambda x: abs(x - p.x))


def greedy(game) -> tuple:
    """Walk to the nearest ladder up, climb it, then walk to the goal."""
    p = game.player
    if p.mode == "climb":
        return ("up",)
    if p.mode == "air":
        return ()
    goal = target(game)
    if goal == p.x and p.floor < game.layout.top:
        return ("up",)
    return ("right",) if goal > p.x else ("left",) if goal < p.x else ()


def value(game, lives0: int, level0: int) -> float:
    if game.lives < lives0 or (game.over and not game.won):
        return -1e6
    if game.level > level0 or game.won:
        return 1e6
    p = game.player
    climbed = (game.layout.floors[p.floor] - p.y) if p.mode == "climb" else 0
    return 1000 * p.floor + 50 * climbed - abs(p.x - target(game))


class LookaheadPlayer:
    def __init__(self, horizon: int = 40, commit: int = 6, replan_every: int = 3) -> None:
        self.horizon, self.commit, self.replan_every = horizon, commit, replan_every
        self._action: tuple = ()
        self._since = 0

    def decide(self, game):
        best, best_v = "greedy", -1e18
        for first in ACTIONS:                   # "greedy" first, so ties favour making progress now
            for then in FOLLOW_UPS:
                sim = copy.deepcopy(game)
                lives0, level0 = sim.lives, sim.level
                for i in range(self.horizon):
                    if i < self.commit:
                        keys = greedy(sim) if first == "greedy" else first
                    else:
                        keys = greedy(sim) if then == "greedy" else ()
                    sim.step(keys)
                    if sim.over or sim.lives < lives0 or sim.level > level0:
                        break
                v = value(sim, lives0, level0)
                if v > best_v:
                    best, best_v = first, v
        return best

    def act(self, game) -> tuple:
        if self._since % self.replan_every == 0:
            choice = self.decide(game)
            self._action = choice
        self._since += 1
        return greedy(game) if self._action == "greedy" else self._action
