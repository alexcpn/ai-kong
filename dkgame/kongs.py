"""Scripted Kongs: from the arcade rhythm to ones that hunt you on ladders."""

from __future__ import annotations

from .engine import Kong


class ClassicKong(Kong):
    """Arcade rhythm: one barrel per decision, random routes."""

    name = "classic"

    def plan(self, obs: dict) -> dict:
        speed = "normal" if obs["level"] < 3 else self.rng.choice(obs["rules"]["speeds"])
        return {"throws": [{"delay": 0.0, "speed": speed, "route": "random"}], "taunt": ""}


class RandomKong(Kong):
    """Anything allowed, uniformly at random."""

    name = "random"

    def plan(self, obs: dict) -> dict:
        rules = obs["rules"]
        n = self.rng.randint(0, rules["max_throws_per_decision"])
        throws = [{"delay": round(self.rng.uniform(0, obs["interval"]), 2),
                   "speed": self.rng.choice(rules["speeds"]),
                   "route": self.rng.choice(["always", "never", "random", "toward_player"])} for _ in range(n)]
        return {"throws": throws, "taunt": self.rng.choice(["", "", "Catch!", "Rolling, rolling..."])}


class AimKong(Kong):
    """Sends barrels down toward the player's girder, faster as the player climbs."""

    name = "aim"

    def plan(self, obs: dict) -> dict:
        rules = obs["rules"]
        high = obs["player"]["floor"] >= obs["top_floor"] - 1
        speed = "fast" if high and "fast" in rules["speeds"] else "normal"
        throws = [{"delay": 0.0, "speed": speed, "route": "toward_player"}]
        if obs["level"] >= 2:
            throws.append({"delay": rules["min_throw_gap"], "speed": "slow" if obs["level"] < 4 else speed,
                           "route": "toward_player"})
        return {"throws": throws, "taunt": "Up here, little man!" if high else ""}


class SniperKong(Kong):
    """Breaks rhythm and punishes climbing: slow/fast pairs, instant fast barrels at climbers."""

    name = "sniper"

    def plan(self, obs: dict) -> dict:
        rules = obs["rules"]
        fast = "fast" if "fast" in rules["speeds"] else "normal"
        if obs["player"]["mode"] == "climb":
            return {"throws": [{"delay": 0.0, "speed": fast, "route": "always"}], "taunt": "Going somewhere?"}
        pair = [{"delay": 0.0, "speed": "slow", "route": "toward_player"},
                {"delay": rules["min_throw_gap"], "speed": fast, "route": "toward_player"}]
        return {"throws": pair[: rules["max_throws_per_decision"]], "taunt": ""}


class TrickKong(Kong):
    """Mixes routes so barrels skip the ladders you expect, with irregular delays and bursts."""

    name = "trick"

    def plan(self, obs: dict) -> dict:
        rules = obs["rules"]
        n = rules["max_throws_per_decision"] if self.rng.random() < 0.3 else 1
        throws = []
        for _ in range(n):
            throws.append({"delay": round(self.rng.choice([0.0, 0.4, 1.1, 1.9, 2.6]), 2),
                           "speed": self.rng.choice(rules["speeds"]),
                           "route": self.rng.choice(["always", "never", "toward_player", "toward_player"])})
        return {"throws": throws, "taunt": ""}


SCRIPTED = {k.name: k for k in (ClassicKong, RandomKong, AimKong, SniperKong, TrickKong)}
