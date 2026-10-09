"""Curses drawing for the game: board, sprites, HUD, what the characters say, and Kong's plan.

On a 256-colour terminal (most today) the characters are small pixel-art sprites: each character cell
holds two pixels, the top one as the foreground of "▀" and the bottom one as its background. Colours
are picked from the terminal's standard 256-colour palette (nearest to the RGB values below), so the
user's own palette is never redefined. Terminals with fewer colours get the plain character art.
"""

from __future__ import annotations

import curses
import textwrap

PAIRS = {"girder": 1, "danger": 2, "ladder": 3, "player": 4, "goal": 5, "ui": 6}

RGB = {                                                    # all exact xterm-256 palette colours
    "girder": (215, 95, 95), "girder_dark": (135, 0, 95), "ladder": (95, 215, 255),
    "fur": (175, 95, 0), "fur_dark": (95, 0, 0), "rage": (215, 0, 0), "skin": (215, 175, 135),
    "eye": (255, 255, 255), "eye_angry": (255, 215, 0), "tie": (255, 0, 0),
    "hair": (135, 95, 0), "face": (255, 215, 175), "dress": (255, 95, 175), "dress_dark": (175, 0, 135),
    "cap": (255, 0, 0), "overalls": (0, 95, 255), "boots": (95, 0, 0),
    "wood": (215, 135, 0), "hoop": (95, 0, 0), "fire": (255, 175, 0),
    "gold": (255, 215, 0), "heart": (255, 0, 95),
}

# Sprites: one string per pixel row, top to bottom, two pixel rows per character row; "." is transparent.
# 3 x 6: a big brow from end to end, eyes under it, tan muzzle, then tapering to a point (a V)
KONG = ["DDD", "WDW", "SSS", "FSF", "FFF", ".F."]
KONG_BLINK = ["DDD", "FDF", "SSS", "FSF", "FFF", ".F."]
KONG_KEYS = {"D": "fur_dark", "F": "fur", "S": "skin", "W": "eye", "T": "tie"}
KONG_RAGE_KEYS = {"D": "fur_dark", "F": "rage", "S": "skin", "W": "eye_angry", "T": "tie"}
PAULINE = [".H.", "HKH", "PPP", "QPQ"]                                          # 3 x 4
PAULINE_KEYS = {"H": "hair", "K": "face", "P": "dress", "Q": "dress_dark"}
PLAYER = ["C", "K", "B", "N"]                                                   # 1 x 4
PLAYER_KEYS = {"C": "cap", "K": "face", "B": "overalls", "N": "boots"}

_fancy = False
_colour_index: dict[str, int] = {}
_pair_ids: dict[tuple[int, int], int] = {}


def nearest_256(r: int, g: int, b: int) -> int:
    """The xterm-256 palette entry closest to an RGB colour (6x6x6 cube or the grey ramp)."""
    levels = (0, 95, 135, 175, 215, 255)
    step = lambda v: min(range(6), key=lambda i: abs(levels[i] - v))  # noqa: E731
    ci = (step(r), step(g), step(b))
    cube = (levels[ci[0]], levels[ci[1]], levels[ci[2]])
    grey_i = min(23, max(0, round(((r + g + b) / 3 - 8) / 10)))
    grey = 8 + 10 * grey_i
    d = lambda c: (c[0] - r) ** 2 + (c[1] - g) ** 2 + (c[2] - b) ** 2  # noqa: E731
    return 16 + 36 * ci[0] + 6 * ci[1] + ci[2] if d(cube) <= d((grey, grey, grey)) else 232 + grey_i


def pair(fg: str | None, bg: str | None) -> int:
    """A colour pair for palette names (None = the terminal's default background), made on first use."""
    key = (_colour_index.get(fg, -1) if fg else -1, _colour_index.get(bg, -1) if bg else -1)
    if key not in _pair_ids:
        if len(_pair_ids) + 16 >= getattr(curses, "COLOR_PAIRS", 256):
            return 0
        _pair_ids[key] = 16 + len(_pair_ids)
        try:
            curses.init_pair(_pair_ids[key], *key)
        except curses.error:
            return 0
    try:
        return curses.color_pair(_pair_ids[key])
    except curses.error:
        return 0


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
    global _fancy
    _fancy = curses.COLORS >= 256 and curses.COLOR_PAIRS >= 64 and bg == -1
    if _fancy:
        _colour_index.update({name: nearest_256(*rgb) for name, rgb in RGB.items()})
        _pair_ids.clear()


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


def pixels(win, y: int, x: int, rows: list[str], keys: dict) -> None:
    """Draw a sprite whose top-left cell is (y, x); two pixel rows per character row."""
    for r in range(0, len(rows), 2):
        top, bottom = rows[r], (rows[r + 1] if r + 1 < len(rows) else "")
        for i, t in enumerate(top):
            b = bottom[i] if i < len(bottom) else "."
            tc, bc = keys.get(t), keys.get(b)
            if tc and bc:
                put(win, y + r // 2, x + i, "▀", pair(tc, bc))
            elif tc:
                put(win, y + r // 2, x + i, "▀", pair(tc, None))
            elif bc:
                put(win, y + r // 2, x + i, "▄", pair(bc, None))


def needed_size(layout: dict) -> tuple[int, int]:
    return layout["floors"][0] + 8, layout["width"] + 2


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
    # score in gold and lives as red hearts; each flashes when it changes (hud flags), hearts keep
    # blinking on the last life
    blink = int(s.get("t", 0) * 6) % 2 == 0
    gold = (pair("gold", None) if _fancy else c("girder")) | curses.A_BOLD
    red = (pair("heart", None) if _fancy else c("danger")) | curses.A_BOLD
    ui = c("ui") | curses.A_BOLD
    score_look = gold | (curses.A_REVERSE if hud.get("score_flash") and blink else 0)
    hearts_flash = hud.get("lives_flash") or s["lives"] == 1
    hearts_look = red | (curses.A_REVERSE if hearts_flash and blink else 0)
    x = left
    for text, look in ((f"SCORE {hud['score']:06d}", score_look), (f"  BEST {hud['best']:06d}  LV {s['level']:02d}  ", ui),
                       ("♥" * s["lives"] or "-", hearts_look), (f"  TIME {int(s['time_left']):02d}{barrels}", ui)):
        put(win, 0, x, text, look)
        x += len(text)
    said = hud.get("said")                                # (speaker, line): Kong, Pauline or the player
    if said:
        name = {"kong": "KONG", "pauline": "PAULINE", "player": "YOU"}.get(said[0], said[0].upper())
        colour = {"kong": "danger", "pauline": "goal", "player": "player"}.get(said[0], "ui")
        put(win, 1, left, f'{name}: "{said[1]}"'[: layout["width"]], c(colour) | curses.A_BOLD)
    oy = 3                                                # header, Kong's speech, then room for his head
    floors = layout["floors"]
    span = layout["x_max"] - layout["x_min"] + 1
    for i, row in enumerate(floors):
        if _fancy:                                        # a two-tone steel beam
            put(win, oy + row + 1, left + layout["x_min"], "▀" * span, pair("girder", "girder_dark"))
        else:
            slope = "\\" if i % 2 == 0 else "/"
            put(win, oy + row + 1, left + layout["x_min"], slope + "=" * (span - 2) + slope, c("girder") | curses.A_BOLD)
    for lad in layout["ladders"]:
        for row in range(floors[lad["top_floor"]] + 1, floors[lad["bottom_floor"]] + 1):
            put(win, oy + row, left + lad["x"], "H", (pair("ladder", None) if _fancy else c("ladder")) | curses.A_BOLD)
    top = floors[-1]
    kong = s.get("kong", {})
    kx, ky = kong.get("x", layout["x_min"] + 1), int(kong.get("y", top) + 0.5)
    loose = kong.get("mode", "perch") != "perch"
    # 3 rows tall (girders are only 3 rows apart); the pixel Kong is 3 wide, from x-1 to x+1
    if _fancy:
        blink = not loose and int(s.get("t", 0) * 4) % 13 == 0      # a quick blink every few seconds
        pixels(win, oy + ky - 2, left + kx - 1, KONG_BLINK if blink else KONG, KONG_RAGE_KEYS if loose else KONG_KEYS)
        pixels(win, oy + top - 1, left + layout["goal"]["x"] - 1, PAULINE, PAULINE_KEYS)
    else:
        look = c("danger") | curses.A_BOLD | (curses.A_REVERSE if loose else 0)
        put(win, oy + ky - 2, left + kx - 1, "▄▄▄▄", look)
        put(win, oy + ky - 1, left + kx - 1, " ÒÓ " if loose else " òó ", look)
        put(win, oy + ky, left + kx - 1, "/██\\" if loose else "▐██▌", look)
        put(win, oy + top - 1, left + layout["goal"]["x"] - 1, " O ", c("goal") | curses.A_BOLD)
        put(win, oy + top, left + layout["goal"]["x"] - 1, "|♀|", c("goal") | curses.A_BOLD)
    for b in s["barrels"]:
        if _fancy:                                        # wood and a dark hoop that swap as it rolls
            top_bottom = ("wood", "hoop") if b["x"] % 2 else ("hoop", "wood")
            put(win, oy + int(b["y"] + 0.5), left + b["x"], "▀", pair(*top_bottom))
        else:
            put(win, oy + int(b["y"] + 0.5), left + b["x"], "●", c("danger") | curses.A_BOLD)
    for f in s["fireballs"]:
        put(win, oy + int(f["y"] + 0.5), left + f["x"], "※", (pair("fire", None) if _fancy else c("danger")) | curses.A_BOLD)
    for pop in hud.get("popups", []):
        put(win, oy + int(pop["y"]), left + pop["x"], pop["text"], c("girder") | curses.A_BOLD)
    if not (p["invulnerable"] and int(s["t"] * 6) % 2):
        if _fancy:                                        # cap, face, overalls, boots: 1 x 2 cells
            pixels(win, oy + int(p["y"] + 0.5) - 1, left + p["x"], PLAYER, PLAYER_KEYS)
        else:
            put(win, oy + int(p["y"] + 0.5), left + p["x"], "@", c("player") | curses.A_BOLD)
    bottom = oy + floors[0] + 2
    if hud.get("plan"):                                  # wrapped to the board's width, under the board
        rows = max(1, min(3, h - bottom - 2))            # leave room for the controls and footer rows
        lines = textwrap.wrap(f"Kong's plan: {hud['plan']}", layout["width"], max_lines=rows, placeholder=" …")
        for i, ln in enumerate(lines):
            put(win, bottom + i, left, ln, c("player") | curses.A_BOLD)
        bottom += len(lines) - 1
    put(win, bottom + 1, left, "←→ move  ↑↓ climb  SPACE jump  P pause  Q quit", c("ui"))
    llm, cost = hud.get("llm") or "", hud.get("cost") or ""
    if llm:                                               # which models play Kong, their speed, the cost so far
        width = max(layout["width"], w - left - 1)
        line = f"LLM  {llm}"
        if len(line) + len(cost) + 2 > width:            # too long: the cost moves to the start
            line = f"LLM {cost}  {llm}" if cost else line
            cost = ""
            if len(line) > width and " · " in line:      # and the call count goes, if it still doesn't fit
                line = line.rsplit(" · ", 1)[0]
        put(win, bottom + 2, left, line[:width], c("goal") | curses.A_BOLD)
        if cost:
            put(win, bottom + 2, left + width - len(cost), cost, c("girder") | curses.A_BOLD)
    if hud.get("banner"):
        centre(win, oy + floors[0] // 2, hud["banner"], c("ui") | curses.A_REVERSE | curses.A_BOLD)
    win.refresh()
