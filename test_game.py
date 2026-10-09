"""Game tests:  python3 -m unittest discover dk-game     (from the repo root)"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dk  # noqa: E402
from dkgame.engine import Game, Kong  # noqa: E402
from dkgame.kongs import SCRIPTED  # noqa: E402
from dkgame.lookahead import greedy  # noqa: E402


class Quiet(Kong):
    name = "quiet"

    def plan(self, obs):
        return {"throws": []}


def play_greedily(game: Game, ticks: int) -> dk.Scorer:
    scorer = dk.Scorer()
    for _ in range(ticks):
        if game.over:
            break
        game.step(greedy(game))
        scorer.update(game)
    return scorer


class ScoringTests(unittest.TestCase):
    def test_climbs_and_level_clear_score(self):
        layout, params = dk.make_variant("classic", 1)
        game = Game(layout, params, Quiet(), seed=1)
        scorer = dk.Scorer()
        floors_before = 0
        while game.levels_cleared == 0 and game.ticks < 4000:
            game.step(greedy(game))
            scorer.update(game)
            floors_before = max(floors_before, game.stats["climbs"])
        self.assertEqual(game.levels_cleared, 1)
        climbs = 100 * floors_before
        self.assertGreater(scorer.score, climbs + 1000)            # level bonus + time bonus on top
        self.assertTrue(scorer.popups)

    def test_reclimbing_does_not_farm_points(self):
        layout, params = dk.make_variant("classic", 1)
        game = Game(layout, params, Quiet(), seed=1)
        scorer = play_greedily(game, 400)
        before = scorer.score
        lay = game.layout
        x = [lx for lx, b in lay.ladders if b == 0][0]
        for _ in range(3):                       # walk back down to girder 0 and climb again
            game.player.x, game.player.y, game.player.floor, game.player.mode = x, float(lay.floors[0]), 0, "ground"
            for _ in range(60):
                game.step(["up"])
                scorer.update(game)
        self.assertEqual(scorer.score, before)

    def test_game_is_endless(self):
        _, params = dk.make_variant("random", 3)
        self.assertGreaterEqual(params.max_levels, 50)


class SetupTests(unittest.TestCase):
    def test_every_board_and_kong_runs(self):
        for board, _ in dk.BOARDS:
            for name in SCRIPTED:
                layout, params = dk.make_variant(board, 5)
                game = Game(layout, params, SCRIPTED[name](), seed=5)
                play_greedily(game, 600)
                self.assertGreater(game.ticks, 0)
                self.assertEqual(game.kong_errors, 0, f"{name} on {board}")
        self.assertEqual(len(dk.make_variant("tall", 2)[0].floors), 6)
        self.assertTrue(all(sum(1 for _, b in dk.make_variant("sparse", 2)[0].ladders if b == f) == 1
                            for f in range(4)))

    def test_high_scores_survive_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["XDG_CACHE_HOME"] = tmp
            try:
                self.assertEqual(dk.load_scores(), {})
                dk.save_scores({"sniper": 1200})
                self.assertEqual(dk.load_scores(), {"sniper": 1200})
                with open(dk.score_path(), "w") as handle:
                    handle.write("{not json")
                self.assertEqual(dk.load_scores(), {})
                with open(dk.score_path(), "w") as handle:
                    json.dump({"aim": "lots", "trick": -5, "classic": 300}, handle)
                self.assertEqual(dk.load_scores(), {"classic": 300})
            finally:
                del os.environ["XDG_CACHE_HOME"]


class Spray(Kong):
    """Throws as often as the rules allow."""
    name = "spray"

    def plan(self, obs):
        return {"throws": [{"delay": 0.0, "speed": "slow", "route": "never"}] * 4}


class BudgetTests(unittest.TestCase):
    def test_kong_runs_out_and_refills_per_level_not_per_life(self):
        from dataclasses import replace
        layout, params = dk.make_variant("classic", 1)
        params = replace(params, barrel_budget=8)
        game = Game(layout, params, Spray(), seed=1)
        game.player.invuln_until = 1e9
        budget = game.barrels_left
        self.assertEqual(game.kong_observation()["barrels_left"], budget)
        for _ in range(4000):
            game.step(())
            if game.barrels_left == 0:
                break
        self.assertEqual(game.barrels_left, 0)
        thrown = sum(1 for e in game.events if e["event"] == "throw")
        for _ in range(400):                       # nothing more once the supply is gone
            game.step(())
        self.assertEqual(sum(1 for e in game.events if e["event"] == "throw"), thrown)
        game._respawn()
        self.assertEqual(game.barrels_left, 0)
        game.level += 1
        game._start_level()
        self.assertEqual(game.barrels_left, budget + params.barrel_budget_per_level)

    def test_engine_default_is_unlimited(self):
        layout, _ = dk.make_variant("classic", 1)
        game = Game(layout, dk.Params(), Spray(), seed=1)
        self.assertIsNone(game.barrels_left)
        for _ in range(200):
            game.step(())
        self.assertIsNone(game.barrels_left)


class TauntTests(unittest.TestCase):
    def make(self, charge):
        from dkgame.director_kong import DEFAULT_KNOBS, DirectorKong

        prompts = []

        class Stub:
            def ask(self, layer, system, prompt, schema, validate):
                prompts.append(prompt)
                if "say" in schema["properties"]:                       # the fast voice call
                    return validate({"say": "Ladders are my thing.", "charge": charge})
                return validate({"opponent_model": "", "strategy": "s", "reasons": "", "taunt": "",
                                 "knobs": dict(DEFAULT_KNOBS)})

            def total_usage(self):
                return {}

        layout, params = dk.make_variant("classic", 2)
        kong = DirectorKong(background=True, client=Stub(), guard=False)
        game = Game(layout, params, kong, seed=2)
        kong.attach(game)
        for _ in range(40):
            game.step(())
        return game, kong, prompts

    def taunt_and_wait(self, game, kong):
        import time
        game.provoke(dk.TAUNT_ANGER)
        self.assertIn(game.taunt, ("Hmph.", "GRRRR!"))                # instant grunt
        kong.provoke(dk.TAUNTS[3], game.t)
        deadline = time.monotonic() + 5
        while game.taunt != "Ladders are my thing." and time.monotonic() < deadline:
            game.step(())
            time.sleep(0.002)
        self.assertEqual(game.taunt, "Ladders are my thing.")

    def test_voice_answers_without_charging(self):
        game, kong, prompts = self.make(charge=False)
        self.taunt_and_wait(game, kong)
        voice = [p for p in prompts if "player_says" in p][-1]
        self.assertEqual(voice["player_says"], dk.TAUNTS[3])
        self.assertEqual(game.kong_body.mode, "perch")

    def test_taking_the_bait_sends_kong_down(self):
        game, kong, _ = self.make(charge=True)
        self.taunt_and_wait(game, kong)
        self.assertEqual(game.kong_body.mode, "rampage")


class TemperTests(unittest.TestCase):
    def game(self):
        layout, params = dk.make_variant("classic", 4)
        game = Game(layout, params, Quiet(), seed=4)
        game.player.invuln_until = 1e9
        return game

    def test_three_quick_taunts_start_a_rampage_and_anger_cools(self):
        game = self.game()
        game.provoke(dk.TAUNT_ANGER)
        for _ in range(40):
            game.step(())
        self.assertLess(game.anger, dk.TAUNT_ANGER)                   # cools while he sits
        game.provoke(dk.TAUNT_ANGER)
        self.assertEqual(game.kong_body.mode, "perch")
        game.provoke(dk.TAUNT_ANGER)
        self.assertEqual(game.kong_body.mode, "rampage")

    def test_rampaging_kong_comes_down_then_goes_home(self):
        game = self.game()
        game.provoke(0, charge=True)
        lowest = game.kong_body.floor
        for _ in range(int(game.params.rampage_seconds / 0.05)):
            game.step(())
            lowest = min(lowest, game.kong_body.floor)
        self.assertLess(lowest, game.layout.top)
        for _ in range(2000):
            game.step(())
            if game.kong_body.mode == "perch":
                break
        self.assertEqual(game.kong_body.mode, "perch")
        self.assertEqual(game.kong_body.floor, game.roam_floor())

    def test_touching_kong_off_his_perch_beats_the_level(self):
        game = self.game()
        scorer = dk.Scorer()
        game.provoke(0, charge=True)
        k = game.kong_body
        k.floor, k.y, k.x = game.player.floor, game.player.y, game.player.x
        game.step(())
        scorer.update(game)
        self.assertEqual(game.stats["kong_defeats"], 1)
        self.assertEqual(game.levels_cleared, 1)
        self.assertEqual(game.kong_body.mode, "perch")                 # fresh level
        self.assertGreaterEqual(scorer.score, dk.POINTS["kong"])

    def test_touching_kong_on_the_top_girder_does_nothing(self):
        game = self.game()
        p, k, top = game.player, game.kong_body, game.layout.top
        p.floor, p.y = top, float(game.layout.floors[top])
        k.floor, k.y, k.x, k.step_at = top, p.y, p.x, 1e9
        game.step(())
        self.assertEqual(game.stats["kong_defeats"], 0)


class KongMovesTests(unittest.TestCase):
    def test_kong_walks_where_told_and_throws_from_there(self):
        layout, params = dk.make_variant("classic", 1)

        class Mover(Spray):
            def plan(self, obs):
                return {**super().plan(obs), "move_to": 1.0}

        game = Game(layout, params, Mover(), seed=1)
        game.player.invuln_until = 1e9
        for _ in range(600):
            game.step(())
        lo, hi = game.kong_range(game.kong_body.floor)
        self.assertGreater(game.kong_body.x, hi - 5)
        self.assertTrue([b for b in game.barrel_log if b["id"] is not None])
        self.assertTrue(any(e["event"] == "throw" for e in game.events))

    def test_kong_keeps_a_few_girders_above_the_player(self):
        from dataclasses import replace
        layout, params = dk.make_variant("classic", 1)
        params = replace(params, level_seconds=999)
        game = Game(layout, params, Quiet(), seed=1)
        game.player.invuln_until = 1e9
        for _ in range(400):
            game.step(())
        self.assertEqual(game.kong_body.floor, params.kong_floors_above)       # player on floor 0
        p = game.player
        p.floor, p.y = 1, float(layout.floors[1])                              # player gets a girder higher
        for _ in range(600):
            game.step(())
        self.assertEqual(game.kong_body.floor, 1 + params.kong_floors_above)
        p.floor, p.y, p.x = 3, float(layout.floors[3]), layout.x_max - 1        # well away from Kong
        for _ in range(600):
            game.step(())
        self.assertEqual(game.kong_body.floor, layout.top)                     # never above the top
        self.assertEqual(game.stats["kong_defeats"], 0)

    def test_barrels_start_from_kongs_girder(self):
        layout, params = dk.make_variant("classic", 1)
        game = Game(layout, params, Spray(), seed=1)
        game.player.invuln_until = 1e9
        floors = set()
        for _ in range(800):
            game.step(())
            for b in game.barrels:
                if b.thrown_at == game.t - 0.05 or abs(b.thrown_at - game.t) < 1e-9:
                    floors.add(b.floor)
        self.assertIn(params.kong_floors_above, floors)

    def test_undirected_kong_roams(self):
        layout, params = dk.make_variant("classic", 1)
        game = Game(layout, params, Quiet(), seed=3)
        xs = set()
        for _ in range(1200):
            game.step(())
            xs.add(game.kong_body.x)
        self.assertGreater(len(xs), 6)


class CollisionTests(unittest.TestCase):
    def test_a_barrel_cannot_slip_past_by_swapping_columns(self):
        from dkgame.engine import Barrel
        layout, params = dk.make_variant("classic", 9)
        game = Game(layout, params, Quiet(), seed=9)
        p = game.player
        p.invuln_until, p.x, p.next_walk = 0.0, 20, 0.0
        barrel = Barrel(id=999, x=21, floor=0, y=p.y, direction=-1, step_gap=0.15, route="never",
                        step_at=game.t + 0.04)                 # steps left this tick, as the player steps right
        game.barrels.append(barrel)
        lives = game.lives
        game.step(["right"])
        self.assertEqual(game.lives, lives - 1)
        self.assertEqual(game.stats["hits_barrel"], 1)

    def test_jumping_over_a_swapping_barrel_still_counts(self):
        from dkgame.engine import Barrel
        layout, params = dk.make_variant("classic", 9)
        game = Game(layout, params, Quiet(), seed=9)
        p = game.player
        p.invuln_until = 0.0
        game.step(["jump"])
        for _ in range(4):
            game.step(["jump"])                                 # high enough to clear a barrel
        p.next_walk = game.t
        barrel = Barrel(id=999, x=p.x + 1, floor=0, y=float(layout.floors[0]), direction=-1, step_gap=0.15,
                        route="never", step_at=game.t + 0.04)
        game.barrels.append(barrel)
        lives = game.lives
        game.step(["right"])
        self.assertEqual(game.lives, lives)
        self.assertEqual(game.stats["jumped_over"], 1)

    def test_no_more_pass_throughs_for_a_search_player(self):
        from dkgame.director_kong import ParametricKong
        from dkgame.lookahead import LookaheadPlayer
        layout, params = dk.make_variant("random", 0)
        game = Game(layout, params, ParametricKong(), seed=0)
        player, slips = LookaheadPlayer(), 0
        for _ in range(500):
            if game.over:
                break
            p = game.player
            before = {b.id: b.x for b in game.barrels}
            px, lives, safe = p.x, game.lives, game.t < p.invuln_until
            game.step(player.act(game))
            if game.lives < lives or safe:
                continue
            q = game.player
            slips += sum(1 for b in game.barrels if b.id in before and abs(b.y - q.y) < params.hit_rows
                         and (before[b.id] - px) * (b.x - q.x) < 0)
        self.assertEqual(slips, 0)


class TacticianTests(unittest.TestCase):
    def make(self, tactics_delay=0.0, fail=False):
        import time
        from dkgame.director_kong import DEFAULT_KNOBS, DirectorKong

        calls = []

        class Stub:
            usage = {}

            def ask(self, layer, system, prompt, schema, validate):
                calls.append((layer.name, layer.provider, prompt))
                if layer.name == "tactician":
                    time.sleep(tactics_delay)
                    if fail:
                        raise RuntimeError("provider down")
                    return validate({"throws": [{"delay": 0.0, "speed": "slow", "route": "always"}], "kong_x": 0.9})
                return validate({"opponent_model": "", "strategy": "s", "reasons": "", "taunt": "",
                                 "knobs": dict(DEFAULT_KNOBS)})

            def total_usage(self):
                return {"calls": len(calls), "cost_usd": 0.0}

        layout, params = dk.make_variant("classic", 4)
        kong = DirectorKong(background=True, client=Stub(), guard=False, tactician=True)
        kong.tactician.provider = "groq"
        game = Game(layout, params, kong, seed=4)
        kong.attach(game)
        game.player.invuln_until = 1e9
        return game, kong, calls

    def run_for(self, game, seconds):
        import time
        for _ in range(int(seconds / 0.05)):
            game.step(())
            time.sleep(0.003)

    def test_fast_tactician_decides_the_throws_and_where_kong_stands(self):
        game, kong, calls = self.make()
        self.run_for(game, 6)
        tactics = [c for c in calls if c[0] == "tactician"]
        self.assertGreaterEqual(len(tactics), 2)
        self.assertEqual(tactics[0][1], "groq")                          # per-layer provider reaches the client
        prompt = tactics[-1][2]
        self.assertIn("strategist", prompt)
        self.assertIn("barrels_on_board", prompt)
        self.assertGreater(kong.tactics["on_time"], 0)
        self.assertEqual(game.kong_aim, 0.9)
        self.assertTrue(any(e.get("route") == "always" for e in game.events if e["event"] == "throw"))

    def test_slow_or_failing_tactician_falls_back_to_the_knobs(self):
        for kw in ({"tactics_delay": 2.5}, {"fail": True}):
            game, kong, _ = self.make(**kw)
            self.run_for(game, 7)
            self.assertGreater(kong.tactics["late"], 0, kw)
            self.assertTrue(any(e["event"] == "throw" for e in game.events), kw)  # Kong never goes quiet
            self.assertIn("late", kong.llm_status()[0])

    def test_tactician_is_off_unless_asked(self):
        from dkgame.director_kong import DirectorKong
        self.assertIsNone(DirectorKong(background=True, client=object(), guard=False).tactician)
        self.assertIsNone(DirectorKong(background=False, client=object(), guard=False, tactician=True).tactician)


if __name__ == "__main__":
    unittest.main()
