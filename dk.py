#!/usr/bin/env python3
"""AI Kong: Donkey Kong for the terminal, against a Kong that can think.

  python3 dk.py                            # title menu
  python3 dk.py --kong ai                  # straight into a game vs AI Kong
  python3 dk.py --kong classic --board tall --seed 7   # no LLM, for testing

Opponents: ai (AI Kong, an LLM that profiles how you play and re-plans Kong's tactics; needs an
OpenRouter key, see README.md) and classic (scripted, no LLM: for testing without a key).
Keys: arrows / WASD / hjkl move and climb, Space jumps (hold for extra height), T taunts Kong,
P pauses, Q quits.
"""

from __future__ import annotations

import argparse
import curses
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from director import load_api_key  # noqa: E402
from dkgame.engine import CLASSIC_LAYOUT, TICK, Game, Params, generate_layout  # noqa: E402
from dkgame.kongs import SCRIPTED  # noqa: E402
from dkgame.render import centre, draw_game, needed_size, put, setup_colors  # noqa: E402

OPPONENTS = [("ai", "AI Kong"), ("classic", "Classic (no LLM)")]   # dkgame.kongs has more scripted Kongs
BOARDS = [("random", "Random board"), ("classic", "Classic board"), ("tall", "Tall (6 girders)"),
          ("sparse", "Sparse (1 ladder each)")]
KEYMAP = {curses.KEY_LEFT: "left", curses.KEY_RIGHT: "right", curses.KEY_UP: "up", curses.KEY_DOWN: "down",
          ord("a"): "left", ord("d"): "right", ord("w"): "up", ord("s"): "down",
          ord("h"): "left", ord("l"): "right", ord("k"): "up", ord("j"): "down",
          ord(" "): "jump", ord("\n"): "jump"}
HOLD = {"left": 0.11, "right": 0.11, "up": 0.16, "down": 0.16, "jump": 0.2}   # no key-up events in terminals
TAUNTS = ["Is that all you've got?", "You throw like my grandma!", "Too slow, old ape!",
          "Bet you can't hit me on a ladder.", "I'm taking the left ladder.", "I'm taking the right ladder.",
          "You'll run out of barrels!", "Pauline's leaving with me!"]
TAUNT_COOLDOWN = 4.0    # game seconds between taunts (each is one small, fast LLM call for AI Kong)
TAUNT_ANGER = 40        # three quick taunts and Kong loses his temper
POINTS = {"climb": 100, "jump": 100, "stomp": 200, "level": 1000, "time": 10, "kong": 3000}


# --------------------------------------------------------------------------- scores

def score_path() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "dk-game", "highscores.json")


def load_scores() -> dict:
    try:
        with open(score_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        scores = {k: int(v) for k, v in data.items() if isinstance(v, (int, float)) and v >= 0}
        if "director" in scores:                      # AI Kong was called "director" before
            scores["ai"] = max(scores.get("ai", 0), scores.pop("director"))
        return scores
    except (OSError, ValueError, AttributeError):
        return {}


def save_scores(scores: dict) -> None:
    try:
        os.makedirs(os.path.dirname(score_path()), exist_ok=True)
        with open(score_path(), "w", encoding="utf-8") as handle:
            json.dump(scores, handle)
    except OSError:
        pass


class Scorer:
    """Arcade points from the engine's own statistics (no farming: climbs count once per girder)."""

    def __init__(self) -> None:
        self.score = 0
        self.seen = {"climbs": 0, "jumped_over": 0, "stomps": 0, "levels": 0, "kong": 0}
        self.time_left = 0.0
        self.popups: list[dict] = []

    def update(self, game: Game) -> None:
        s, p = game.stats, game.player
        gained = (POINTS["climb"] * (s["climbs"] - self.seen["climbs"])
                  + POINTS["jump"] * (s["jumped_over"] - self.seen["jumped_over"])
                  + POINTS["stomp"] * (s["stomps"] - self.seen["stomps"])
                  + POINTS["kong"] * (s["kong_defeats"] - self.seen["kong"]))
        cleared = game.levels_cleared - self.seen["levels"]
        if cleared:
            gained += cleared * (POINTS["level"] * game.level + POINTS["time"] * int(self.time_left))
        self.seen = {"climbs": s["climbs"], "jumped_over": s["jumped_over"], "stomps": s["stomps"],
                     "levels": game.levels_cleared, "kong": s["kong_defeats"]}
        self.time_left = game.time_left
        if gained:
            self.score += gained
            self.popups.append({"x": p.x, "y": p.y - 1, "text": f"+{gained}", "until": game.t + 1.0})
        self.popups = [q for q in self.popups if q["until"] > game.t]


# --------------------------------------------------------------------------- setup

def make_variant(board: str, seed: int):
    rng = random.Random(seed * 31 + 5)
    # endless (play for score), with a floatier jump than the engine default: same 2-row peak,
    # ~0.85 s in the air instead of ~0.6 s, so landings are readable at terminal frame rates
    # Kong throws ~40% more often than the engine's base pace, from 45 barrels on level 1 (+8 per level),
    # roaming three girders above the player; 3 s of grace after losing a life.
    # Difficulty levers: throw_gap_scale / extra_barrels / kong_interval (pace), kong_floors_above
    # (reaction time: barrels from further up take longer to arrive), invuln_time.
    params = Params(max_levels=99, gravity=20.0, jump_velocity=9.0, barrel_budget=45, barrel_budget_per_level=8,
                    kong_interval=2.0, throw_gap_scale=0.6, extra_barrels=2, kong_moves=True, kong_floors_above=3,
                    invuln_time=3.0)
    if board == "classic":
        return CLASSIC_LAYOUT, params
    floors = 6 if board == "tall" else None
    ladders = 1 if board == "sparse" else 2
    return generate_layout(seed, floors=floors, max_ladders=ladders), params.scaled(rng, 0.06)


def make_kong(name: str):
    if name == "ai":
        from director import Layer
        from dkgame.director_kong import DirectorKong
        return DirectorKong(background=True, layer=Layer.from_env("director", "medium", 8000))
    return SCRIPTED[name]()


# --------------------------------------------------------------------------- screens

TITLE = [
    "▐█▌  ●   ●   ●",
    "▀█▀            ",
    "",
    "A I   K O N G",
    "Donkey Kong, against a Kong that thinks",
]


def menu(win, choice: dict, scores: dict, has_key: bool) -> dict | None:
    win.nodelay(False)
    row = 2
    while True:
        win.erase()
        top = max(1, win.getmaxyx()[0] // 2 - 10)
        for i, line in enumerate(TITLE):
            centre(win, top + i, line, curses.A_BOLD if i in (0, 1, 3) else 0)
        opp = OPPONENTS[choice["opponent"]]
        brd = BOARDS[choice["board"]]
        rows = [f"Opponent:  <  {opp[1]:^16}  >", f"Board:     <  {brd[1]:^22}  >", "[  PLAY  ]", "[  QUIT  ]"]
        for i, text in enumerate(rows):
            centre(win, top + 7 + i * 2, text, curses.A_REVERSE if i == row else 0)
        best = scores.get(opp[0], 0)
        centre(win, top + 16, f"Best vs {opp[1]}: {best:06d}")
        if opp[0] == "ai":
            note = ("An LLM studies your habits and re-plans Kong's tactics every ~20 s."
                    if has_key else "Needs an OpenRouter key: see README.md (key file not found).")
            centre(win, top + 18, note)
            if has_key:
                centre(win, top + 19, ai_models())
        centre(win, top + 20, "↑↓ choose   ←→ change   ENTER play   Q quit")
        win.refresh()
        key = win.getch()
        if key in (ord("q"), ord("Q"), 27):
            return None
        if key in (curses.KEY_UP, ord("w"), ord("k")):
            row = (row - 1) % 4
        elif key in (curses.KEY_DOWN, ord("s"), ord("j")):
            row = (row + 1) % 4
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT, ord("a"), ord("d"), ord("h"), ord("l")):
            step = 1 if key in (curses.KEY_RIGHT, ord("d"), ord("l")) else -1
            if row == 0:
                choice["opponent"] = (choice["opponent"] + step) % len(OPPONENTS)
            elif row == 1:
                choice["board"] = (choice["board"] + step) % len(BOARDS)
        elif key in (10, 13, curses.KEY_ENTER, ord(" ")):
            if row == 3:
                return None
            if OPPONENTS[choice["opponent"]][0] == "ai" and not has_key:
                continue
            return choice


def ai_models() -> str:
    """The models AI Kong will use (from the environment / .config/config.env)."""
    from director import Layer
    brain, voice = Layer.from_env("director", "medium", 8000), Layer.from_env("voice", "none", 300)
    return f"{brain.model.split('/')[-1]} ({brain.reasoning}) + {voice.model.split('/')[-1]} for taunts"


def play(win, opponent: str, board: str, seed: int, scores: dict) -> str:
    """One game. Returns 'menu', 'retry' or 'quit'."""
    layout, params = make_variant(board, seed)
    kong = make_kong(opponent)
    game = Game(layout, params, kong, seed=seed)
    if hasattr(kong, "attach"):
        kong.attach(game)
    label = dict(OPPONENTS)[opponent]
    scorer, best = Scorer(), scores.get(opponent, 0)
    lay = layout.to_dict()
    win.nodelay(True)
    held: dict[str, float] = {}
    paused, next_tick = False, time.monotonic()
    taunt = {"open": False, "pick": 0, "next": 0.0, "said": "", "until": 0.0}

    def hud(banner: str = "") -> dict:
        plan = kong.display() if hasattr(kong, "display") else ""
        said = taunt["said"] if game.t < taunt["until"] else ""
        mode = game.kong_body.mode
        if taunt["open"]:
            footer = f"TAUNT ←{TAUNTS[taunt['pick']]:^35}→ ENTER/ESC"
        elif mode == "rampage":
            footer = "KONG IS COMING FOR YOU!  Touch him to beat the level"
        elif mode == "return":
            footer = "Kong is climbing back up... catch him!"
        else:
            wait = taunt["next"] - game.t
            bar = "█" * int(game.anger / 10) + "░" * (10 - int(game.anger / 10))
            footer = f"ANGER {bar}   " + ("T taunt Kong" if wait <= 0 else f"taunt in {int(wait) + 1}s")
        llm, cost = kong.llm_status() if hasattr(kong, "llm_status") else ("", "")
        return {"score": scorer.score, "best": max(best, scorer.score), "opponent": label, "plan": plan,
                "banner": banner, "popups": scorer.popups, "said": said, "footer": footer,
                "llm": llm, "cost": cost}

    while not game.over:
        now = time.monotonic()
        while True:
            key = win.getch()
            if key == -1:
                break
            if taunt["open"]:                                   # picker: the game is paused
                if key in (curses.KEY_LEFT, curses.KEY_UP, ord("a"), ord("w"), ord("h"), ord("k")):
                    taunt["pick"] = (taunt["pick"] - 1) % len(TAUNTS)
                elif key in (curses.KEY_RIGHT, curses.KEY_DOWN, ord("d"), ord("s"), ord("l"), ord("j")):
                    taunt["pick"] = (taunt["pick"] + 1) % len(TAUNTS)
                elif key in (10, 13, curses.KEY_ENTER, ord(" "), ord("t"), ord("T")):
                    text = TAUNTS[taunt["pick"]]
                    game.provoke(TAUNT_ANGER)                   # instant: anger, a grunt, maybe a rampage
                    if hasattr(kong, "provoke"):
                        kong.provoke(text, game.t)              # AI Kong answers properly a moment later
                    taunt.update(open=False, said=text, until=game.t + 1.2, next=game.t + TAUNT_COOLDOWN)
                elif key in (27, curses.KEY_BACKSPACE, 127, ord("q"), ord("Q")):
                    taunt["open"] = False
                next_tick = time.monotonic()
                continue
            if key in (ord("q"), ord("Q")):
                game.over = True
                break
            if key in (ord("t"), ord("T")) and game.t >= taunt["next"] and game.kong_body.mode == "perch":
                taunt["open"] = True
                held.clear()
            elif key in (ord("p"), ord("P")):
                paused = not paused
                held.clear()
            elif key == curses.KEY_RESIZE:
                pass
            elif KEYMAP.get(key):
                held[KEYMAP[key]] = now + HOLD[KEYMAP[key]]
        if taunt["open"]:
            draw_game(win, lay, game.state(), hud())
            time.sleep(0.02)
            next_tick = time.monotonic()
            continue
        if paused:
            draw_game(win, lay, game.state(), hud(" PAUSED: press P to resume "))
            time.sleep(0.05)
            next_tick = time.monotonic()
            continue
        if now >= next_tick:
            h, w = win.getmaxyx()
            need_h, need_w = needed_size(lay)
            if h >= need_h and w >= need_w:                    # freeze the game while the window is too small
                game.step([k for k, until in held.items() if until >= now])
                scorer.update(game)
            next_tick += TICK
            if next_tick < now - 0.25:
                next_tick = now
            draw_game(win, lay, game.state(), hud())
        time.sleep(max(0.0, min(0.01, next_tick - time.monotonic())))

    new_best = scorer.score > best
    if new_best:
        scores[opponent] = scorer.score
        save_scores(scores)
    note = "  NEW BEST!" if new_best else ""
    cost = f"  LLM {kong.llm_status()[1]}" if hasattr(kong, "llm_status") else ""
    summary = f" GAME OVER  score {scorer.score:06d}{note}  level {game.level}{cost} "
    draw_game(win, lay, game.state(), {**hud(summary), "best": max(best, scorer.score)})
    put(win, 0, 0, "")
    centre(win, win.getmaxyx()[0] - 1, "R retry   M menu   Q quit", curses.A_BOLD)
    win.refresh()
    win.nodelay(False)
    while True:
        key = win.getch()
        if key in (ord("r"), ord("R")):
            return "retry"
        if key in (ord("m"), ord("M")):
            return "menu"
        if key in (ord("q"), ord("Q"), 27):
            return "quit"


def run(win, args) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    win.keypad(True)
    setup_colors()
    scores, has_key = load_scores(), load_api_key() is not None
    names = [o[0] for o in OPPONENTS]
    boards = [b[0] for b in BOARDS]
    choice = {"opponent": names.index(args.kong) if args.kong else (0 if has_key else 1), "board": boards.index(args.board)}
    skip_menu = args.kong is not None
    while True:
        if not skip_menu:
            picked = menu(win, choice, scores, has_key)
            if picked is None:
                return
        skip_menu = False
        while True:
            seed = args.seed if args.seed is not None else random.randrange(1_000_000)
            outcome = play(win, names[choice["opponent"]], boards[choice["board"]], seed, scores)
            if outcome != "retry":
                break
        if outcome == "quit":
            return


ENV_FILES = [os.path.join(HERE, ".config", "config.env"),          # model / reasoning / endpoint settings
             os.path.join(HERE, ".config", "opneroutere.env")]     # API key (git-ignored)


def load_env_files() -> None:
    """Load .config/*.env into os.environ; variables already set in the shell win."""
    present = [path for path in ENV_FILES if os.path.isfile(path)]
    if not present:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("note: pip install python-dotenv to load " + ", ".join(present), file=sys.stderr)
        return
    for path in present:
        load_dotenv(path, override=False)


def main() -> None:
    load_env_files()
    os.environ.setdefault("ESCDELAY", "25")      # curses waits 1 s after ESC by default
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kong", choices=[o[0] for o in OPPONENTS] + ["director"],
                        help="skip the menu and play this opponent (director = old name for ai)")
    parser.add_argument("--board", choices=[b[0] for b in BOARDS], default="random")
    parser.add_argument("--seed", type=int, default=None, help="same seed = same board")
    args = parser.parse_args()
    if args.kong == "director":
        args.kong = "ai"
    if args.kong == "ai" and load_api_key() is None:
        raise SystemExit("AI Kong needs an OpenRouter key: see README.md")
    try:
        curses.wrapper(lambda win: run(win, args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
