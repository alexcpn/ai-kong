"""Curses drawing for the game: board, sprites, HUD, Kong's taunt and plan."""

from __future__ import annotations

import curses

PAIRS = {"girder": 1, "danger": 2, "ladder": 3, "player": 4, "goal": 5, "ui": 6}


def setup_colors() -> None:
    if not curses.has_colors():
        return
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    for name, colour in (("girder", curses.COLOR_YELLOW), ("danger", curses.COLOR_RED), ("ladder", curses.COLOR_CYAN),
                         ("player", curses.COLOR_GREEN), ("goal", curses.COLOR_MAGENTA), ("ui", curses.COLOR_WHITE)):
        curses.init_pair(PAIRS[name], colour, bg)


def c(name: str) -> int:
    try:
        return curses.color_pair(PAIRS[name])
    except curses.error:
        return 0


def put(win, y: int, x: int, text: str, attr: int = 0) -> None:
    h, w = win.getmaxyx()
    if 0 <= y < h and 0 <= x < w and text:
        try:
            win.addnstr(y, x, text, max(0, w - x - (1 if y == h - 1 else 0)), attr)
        except curses.error:
            pass


def centre(win, y: int, text: str, attr: int = 0) -> None:
    put(win, y, max(0, (win.getmaxyx()[1] - len(text)) // 2), text, attr)


def needed_size(layout: dict) -> tuple[int, int]:
    return layout["floors"][0] + 7, layout["width"] + 2


def draw_game(win, layout: dict, s: dict, hud: dict) -> None:
    """layout = Layout.to_dict(); s = Game.state(); hud = score/best/opponent/plan/banner."""
    win.erase()
    h, w = win.getmaxyx()
    need_h, need_w = needed_size(layout)
    if h < need_h or w < need_w:
        put(win, 1, 1, f"Please enlarge the terminal to at least {need_w} x {need_h}.", c("ui") | curses.A_BOLD)
        put(win, 3, 1, "Q quits.")
        win.refresh()
        return
    left = max(0, (w - layout["width"]) // 2)
    p = s["player"]
    left_n = s.get("kong", {}).get("barrels_left")
    barrels = f"  ●×{left_n:02d}" if left_n is not None else f"  vs {hud['opponent']}"
    put(win, 0, left, f"SCORE {hud['score']:06d}  BEST {hud['best']:06d}  LV {s['level']:02d}  "
                      f"{'♥' * s['lives']}  TIME {int(s['time_left']):02d}{barrels}",
        c("ui") | curses.A_BOLD)
    taunt = s.get("kong", {}).get("taunt") or ""
    if hud.get("said"):
        put(win, 1, left, f'YOU: "{hud["said"]}"'[: layout["width"]], c("player") | curses.A_BOLD)
    elif taunt:
        put(win, 1, left, f'KONG: "{taunt}"'[: layout["width"]], c("danger") | curses.A_BOLD)
    oy = 2
    floors = layout["floors"]
    for i, row in enumerate(floors):
        slope = "\\" if i % 2 == 0 else "/"
        put(win, oy + row + 1, left + layout["x_min"],
            slope + "=" * (layout["x_max"] - layout["x_min"] - 1) + slope, c("girder") | curses.A_BOLD)
    for lad in layout["ladders"]:
        for row in range(floors[lad["top_floor"]] + 1, floors[lad["bottom_floor"]] + 1):
            put(win, oy + row, left + lad["x"], "H", c("ladder") | curses.A_BOLD)
    top = floors[-1]
    kong = s.get("kong", {})
    kx, ky = kong.get("x", layout["x_min"] + 1), int(kong.get("y", top) + 0.5)
    loose = kong.get("mode", "perch") != "perch"
    look = c("danger") | curses.A_BOLD | (curses.A_REVERSE if loose else 0)
    put(win, oy + ky - 1, left + kx - 1, "▐█▌", look)
    put(win, oy + ky, left + kx - 1, "/▀\\" if loose else " ▀ ", look)
    put(win, oy + top - 1, left + layout["goal"]["x"] - 1, " O ", c("goal") | curses.A_BOLD)
    put(win, oy + top, left + layout["goal"]["x"] - 1, "|♀|", c("goal") | curses.A_BOLD)
    for b in s["barrels"]:
        put(win, oy + int(b["y"] + 0.5), left + b["x"], "●", c("danger") | curses.A_BOLD)
    for f in s["fireballs"]:
        put(win, oy + int(f["y"] + 0.5), left + f["x"], "※", c("danger") | curses.A_BOLD)
    for pop in hud.get("popups", []):
        put(win, oy + int(pop["y"]), left + pop["x"], pop["text"], c("girder") | curses.A_BOLD)
    if not (p["invulnerable"] and int(s["t"] * 6) % 2):
        put(win, oy + int(p["y"] + 0.5), left + p["x"], "@", c("player") | curses.A_BOLD)
    bottom = oy + floors[0] + 2
    if hud.get("plan"):
        put(win, bottom, left, f"Kong's plan: {hud['plan']}"[: layout["width"]], c("goal"))
    put(win, bottom + 1, left, "←→ move  ↑↓ climb  SPACE jump  T taunt  P pause  Q quit", c("ui"))
    if hud.get("footer"):
        put(win, bottom + 2, left, hud["footer"][: max(layout["width"], w - left)],
            c("player") | curses.A_BOLD if hud["footer"].startswith("TAUNT") else c("ui"))
    cost = hud.get("cost") or ""
    if cost and len(hud.get("footer") or "") + len(cost) + 2 <= layout["width"]:
        put(win, bottom + 2, left + layout["width"] - len(cost), cost, c("goal") | curses.A_BOLD)
        cost = ""                                         # shown at the right end of the footer row
    if hud.get("llm") and bottom + 3 < h:                # models line, when the terminal has a spare row
        line = hud["llm"] + (f" · {cost}" if cost else "")
        put(win, bottom + 3, left, line[: max(layout["width"], w - left)], c("goal"))
    if hud.get("banner"):
        centre(win, oy + floors[0] // 2, hud["banner"], c("ui") | curses.A_REVERSE | curses.A_BOLD)
    win.refresh()
