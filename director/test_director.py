"""Run from the repo root:  python3 -m unittest discover director"""

from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from director import Director, FeasibilityGuard, KnobSet, KnobSpec, Layer, LLMError, Profiler  # noqa: E402


def specs():
    return [
        KnobSpec("rate", "float", 0.5, min=0.0, max=1.0, max_step=0.3),
        KnobSpec("count", "int", 2, min=0, max=5),
        KnobSpec("ambush", "bool", False, cooldown_s=10),
        KnobSpec("mode", "enum", "calm", choices=("calm", "angry")),
        KnobSpec("mix", "weights", {"a": 1, "b": 1}, choices=("a", "b"), max_step=0.25),
    ]


class StubClient:
    """Returns scripted replies (or raises) instead of calling an LLM."""

    def __init__(self, replies, delay=0.0):
        self.replies = list(replies)
        self.prompts = []
        self.delay = delay

    def ask(self, layer, system, prompt, schema, validate):
        self.prompts.append(prompt)
        time.sleep(self.delay)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return validate(reply)


def reply(knobs, strategy="plan", taunt=""):
    return {"opponent_model": "m", "strategy": strategy, "reasons": "r", "taunt": taunt, "knobs": knobs}


FULL = {"rate": 0.5, "count": 2, "ambush": False, "mode": "calm", "mix": {"a": 0.5, "b": 0.5}}


class KnobTests(unittest.TestCase):
    def test_defaults_are_coerced(self):
        k = KnobSet(specs())
        self.assertEqual(k["mix"], {"a": 0.5, "b": 0.5})
        self.assertEqual(k["count"], 2)

    def test_bounds_steps_cooldowns_and_unknowns(self):
        k = KnobSet(specs())
        changes, notes = k.validate({"rate": 5, "count": 9.6, "ambush": True, "mode": "furious", "zap": 1,
                                     "mix": {"a": 1, "b": 0}}, t=0.0)
        self.assertEqual(changes["rate"], 0.8)                 # clamped to 1, then max_step 0.3
        self.assertEqual(changes["count"], 5)                  # clamped
        self.assertTrue(changes["ambush"])
        self.assertNotIn("mode", changes)                      # invalid enum rejected
        self.assertAlmostEqual(changes["mix"]["a"], 0.75 / 1.0, places=2)
        self.assertTrue(any("zap" in n for n in notes) and any("mode" in n for n in notes))
        k.apply(changes, t=0.0)
        changes, notes = k.validate({"ambush": False}, t=5.0)  # cooldown 10 s
        self.assertEqual(changes, {})
        self.assertTrue(any("cooling down" in n for n in notes))
        changes, _ = k.validate({"ambush": False}, t=10.5)
        self.assertEqual(changes, {"ambush": False})

    def test_bad_types_never_raise(self):
        k = KnobSet(specs())
        changes, notes = k.validate({"rate": "fast", "mix": {"a": -1, "b": 0}, "ambush": "maybe"}, t=0)
        self.assertEqual(changes, {})
        self.assertEqual(len(notes), 3)
        self.assertEqual(k.validate("nonsense", t=0), ({}, ["proposal is not an object"]))

    def test_schema_is_strict_and_complete(self):
        schema = KnobSet(specs()).json_schema()
        self.assertEqual(set(schema["required"]), {"rate", "count", "ambush", "mode", "mix"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["mode"]["enum"], ["calm", "angry"])
        self.assertFalse(schema["properties"]["mix"]["additionalProperties"])


class ProfilerTests(unittest.TestCase):
    def test_summary(self):
        p = Profiler(summarise=lambda prof: {"custom": prof.counters.get("deaths", 0) * 10})
        p.count("deaths")
        p.observe("jump_lead", 2)
        p.observe("jump_lead", 4)
        p.start("wait", 1.0)
        p.stop("wait", 2.5)
        p.start("blip", 3.0)
        p.stop("blip", 3.1, min_duration=0.5)
        p.event("hit", 2.0, x=3)
        s = p.summary()
        self.assertEqual(s["counters"]["deaths"], 1)
        self.assertEqual(s["samples"]["jump_lead"]["mean"], 3)
        self.assertEqual(s["durations"]["wait"]["mean"], 1.5)
        self.assertNotIn("blip", s["durations"])
        self.assertEqual(s["custom"], 10)
        self.assertEqual(p.recent(kinds=("hit",))[0]["x"], 3)


class DirectorTests(unittest.TestCase):
    def make(self, replies, **kw):
        return Director(KnobSet(specs()), brief="test game", layer=Layer("director"),
                        client=StubClient(replies, kw.pop("delay", 0.0)), **kw)

    def test_applies_validated_knobs_and_tracks_results(self):
        d = self.make([reply({**FULL, "rate": 0.7}, "first", "ha"), reply({**FULL, "rate": 0.9}, "second")])
        self.assertEqual(d.due(0.0), "start")
        d.update(0.0, context={}, outcomes={"hits": 0})
        self.assertEqual(d.knobs["rate"], 0.7)
        self.assertEqual((d.strategy, d.taunt), ("first", "ha"))
        self.assertIsNone(d.due(10.0))
        self.assertEqual(d.due(10.0, trigger="life lost"), "life lost")
        d.update(20.0, context={}, outcomes={"hits": 2})
        self.assertEqual(d.history[0]["results"], {"hits": 2})
        self.assertEqual(d.history[0]["strategy"], "first")
        self.assertEqual(d.knobs["rate"], 0.9)
        self.assertEqual(d.taunt, "ha")                       # empty taunt keeps the old one

    def test_llm_failure_keeps_knobs_and_reports(self):
        d = self.make([LLMError("boom"), reply({**FULL, "rate": 0.6})])
        d.update(0.0, {}, {})
        self.assertEqual(d.knobs["rate"], 0.5)
        self.assertEqual(d.summary()["errors"], 1)
        d.update(20.0, {}, {})
        self.assertIn("could not be used", str(d.client.prompts[1]["notes_from_your_last_update"]))

    def test_guard_vetoes_and_explains(self):
        guard = FeasibilityGuard(lambda snap, knobs, i: {"survived": knobs["rate"] < 0.75}, rollouts=2)
        d = self.make([reply({**FULL, "rate": 0.8}), reply({**FULL, "rate": 0.7})], guard=guard)
        d.update(0.0, {}, {}, snapshot=object())
        self.assertEqual(d.knobs["rate"], 0.5)
        self.assertTrue(d.strategy.endswith("(vetoed)"))
        d.update(20.0, {}, {}, snapshot=object())
        self.assertIn("VETOED", str(d.client.prompts[1]["notes_from_your_last_update"]))
        self.assertEqual(d.knobs["rate"], 0.7)
        self.assertEqual((guard.checks, guard.vetoes), (2, 1))

    def test_guard_does_not_blame_changes_in_doomed_states(self):
        guard = FeasibilityGuard(lambda snap, knobs, i: {"survived": False}, rollouts=2)
        ok, report = guard.check(object(), {"rate": 0.9}, current={"rate": 0.5})
        self.assertTrue(ok)
        self.assertTrue(report["doomed_anyway"])
        ok, _ = guard.check(object(), {"rate": 0.9})            # no baseline given: strict veto
        self.assertFalse(ok)

    def test_background_mode_never_blocks(self):
        d = self.make([reply({**FULL, "count": 4})], background=True, delay=0.3)
        started = time.monotonic()
        d.update(0.0, {}, {})
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertIsNone(d.due(100.0))                       # busy: no second request
        self.assertFalse(d.poll())
        time.sleep(0.4)
        self.assertTrue(d.poll())
        self.assertEqual(d.knobs["count"], 4)


if __name__ == "__main__":
    unittest.main()
