"""Who says what, and when: Kong, Pauline and the player's character talk during play.

The lines are written ahead of time (by an LLM for AI Kong, see director_kong.py; built-in defaults
otherwise) and the game picks one the moment something happens, so speech never waits on a network
call. Cast.react() turns engine events into at most one line at a time, rate-limited so the speech
row stays readable.
"""

from __future__ import annotations

import random

SPEAKERS = ("kong", "pauline", "player")
NAMES = {"kong": "KONG", "pauline": "PAULINE", "player": "YOU"}
MAX_CHARS = 42

# moment -> (engine events that trigger it, speakers who may react, may it cut in over a line?)
MOMENTS = {
    "start": ((), ("pauline", "kong"), False),
    "near_miss": (("near_miss",), ("player",), True),
    "jumped": (("jumped_over",), ("player", "kong"), False),
    "climbed": (("climbed",), ("player", "pauline", "kong"), False),
    "near_goal": (("near_goal",), ("pauline", "kong"), True),
    "lost_life": (("player_hit",), ("player", "kong", "pauline"), True),
    "level_clear": (("level_cleared",), ("pauline", "player"), True),
    "kong_beaten": (("kong_defeated",), ("pauline", "kong"), True),
    "idle": ((), ("pauline", "kong"), False),
}
EVENT_MOMENT = {event: moment for moment, (events, _, _) in MOMENTS.items() for event in events}

DEFAULT_LINES = {
    "kong": {
        "start": ["Fresh climber! My favourite.", "Welcome to my girders."],
        "jumped": ["Lucky hop.", "Jump all you like."],
        "climbed": ["Higher means further to fall!", "Keep climbing. I'm waiting."],
        "near_goal": ["Not one step closer!", "She stays with me!"],
        "lost_life": ["Down you go!", "Ha! Again!"],
        "kong_beaten": ["Ow! Not fair!", "I'll be back up there..."],
        "idle": ["Barrels don't throw themselves.", "Hmm. Which ladder next?"],
    },
    "pauline": {
        "start": ["Help! Up here!", "You can do it!"],
        "climbed": ["Yes! Keep coming!", "That's it, keep climbing!"],
        "near_goal": ["Almost there!", "Just a little further!"],
        "lost_life": ["Oh no! Get up!", "Try again, I'm still here!"],
        "level_clear": ["My hero!", "You made it!"],
        "kong_beaten": ["You beat him!", "Ha! Take that, Kong!"],
        "idle": ["Don't wait too long!", "The clock is ticking!"],
    },
    "player": {
        "near_miss": ["Whoa! Too close!", "Oh no!"],
        "jumped": ["Hup!", "Over you go!"],
        "climbed": ["Up I go.", "One more girder."],
        "lost_life": ["Ouch!", "Not again..."],
        "level_clear": ["Made it!", "Next!"],
    },
}


def lines_schema() -> dict:
    """Strict JSON schema for a full set of lines: speaker -> moment -> list of strings."""
    def speaker(name: str) -> dict:
        moments = [m for m in MOMENTS if name in MOMENTS[m][1]]
        return {"type": "object", "properties": {m: {"type": "array", "items": {"type": "string"}} for m in moments},
                "required": moments, "additionalProperties": False}
    return {"type": "object", "properties": {s: speaker(s) for s in SPEAKERS},
            "required": list(SPEAKERS), "additionalProperties": False}


class Cast:
    def __init__(self, seed: int = 0, gap_s: float = 1.5, speaker_gap_s: float = 4.0, idle_s: float = 8.0) -> None:
        self.lines = {s: {m: list(v) for m, v in DEFAULT_LINES[s].items()} for s in SPEAKERS}
        self.rng = random.Random(seed)
        self.gap_s, self.speaker_gap_s, self.idle_s = gap_s, speaker_gap_s, idle_s
        self.last_any = -1e9
        self.last_by: dict[str, float] = {}
        self.recent: list[str] = []
        self.started = False
        self.updates = 0

    def apply(self, reply) -> int:
        """Take a set of lines (e.g. from an LLM). Bad entries are skipped; returns how many were kept."""
        kept = 0
        if not isinstance(reply, dict):
            return 0
        for speaker in SPEAKERS:
            moments = reply.get(speaker)
            if not isinstance(moments, dict):
                continue
            for moment, lines in moments.items():
                if moment not in MOMENTS or speaker not in MOMENTS[moment][1] or not isinstance(lines, list):
                    continue
                clean = [" ".join(x.split())[:MAX_CHARS] for x in lines if isinstance(x, str) and x.strip()]
                if clean:
                    self.lines[speaker][moment] = clean
                    kept += len(clean)
        self.updates += bool(kept)
        return kept

    def _say(self, moment: str, t: float) -> tuple[str, str] | None:
        _, speakers, urgent = MOMENTS[moment]
        if not urgent and t - self.last_any < self.gap_s:
            return None
        choices = [s for s in speakers if self.lines[s].get(moment) and t - self.last_by.get(s, -1e9) >= self.speaker_gap_s]
        if not choices:
            if not urgent:
                return None
            choices = [s for s in speakers if self.lines[s].get(moment)]   # urgent: anyone with a line
            if not choices:
                return None
        speaker = choices[0] if self.rng.random() < 0.6 else self.rng.choice(choices)
        pool = self.lines[speaker][moment]
        fresh = [line for line in pool if line not in self.recent] or pool
        text = self.rng.choice(fresh)
        self.recent = (self.recent + [text])[-8:]
        self.last_any, self.last_by[speaker] = t, t
        return speaker, text

    def react(self, events: list[dict], t: float) -> tuple[str, str] | None:
        """New engine events since the last call -> at most one (speaker, line)."""
        if not self.started:
            self.started = True
            return self._say("start", t)
        moments = [EVENT_MOMENT[e["event"]] for e in events if e.get("event") in EVENT_MOMENT]
        moments.sort(key=lambda m: not MOMENTS[m][2])        # urgent moments first
        for moment in moments:
            said = self._say(moment, t)
            if said:
                return said
        if t - self.last_any >= self.idle_s:
            return self._say("idle", t)
        return None
