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


if __name__ == "__main__":
    unittest.main()
