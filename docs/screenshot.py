"""Render a game frame to a PNG the way a 256-colour terminal shows it (README screenshot, sprite checks).

    python3 docs/screenshot.py . fancy 6 docs/images/screenshot.png        # repo root; "plain" = 8 colours
    python3 docs/screenshot.py . fancy 6 out.png rage                       # Kong storming
"""
import sys
sys.path.insert(0, sys.argv[1])
import curses
from PIL import Image, ImageDraw, ImageFont
import dk
from dkgame import render
from dkgame.engine import Game
from dkgame.director_kong import ParametricKong
from dkgame.lookahead import greedy

BASIC = {curses.COLOR_YELLOW: (230, 200, 40), curses.COLOR_RED: (220, 50, 50), curses.COLOR_CYAN: (60, 200, 220),
         curses.COLOR_GREEN: (80, 200, 80), curses.COLOR_MAGENTA: (210, 80, 210), curses.COLOR_WHITE: (225, 225, 225)}
BG = (24, 24, 28)
pairs = {}


def idx_rgb(i):
    if i == -1:
        return None
    if i in BASIC:
        return BASIC[i]
    if 16 <= i <= 231:
        i -= 16
        lv = (0, 95, 135, 175, 215, 255)
        return lv[i // 36], lv[(i // 6) % 6], lv[i % 6]
    if i >= 232:
        g = 8 + 10 * (i - 232)
        return g, g, g
    return 200, 200, 200


curses.init_pair = lambda n, fg, bg: pairs.__setitem__(n, (fg, bg))
curses.color_pair = lambda n: n << 8
curses.COLORS, curses.COLOR_PAIRS = 256, 256
for name, col in (("girder", curses.COLOR_YELLOW), ("danger", curses.COLOR_RED), ("ladder", curses.COLOR_CYAN),
                  ("player", curses.COLOR_GREEN), ("goal", curses.COLOR_MAGENTA), ("ui", curses.COLOR_WHITE)):
    curses.init_pair(render.PAIRS[name], col, -1)
render._fancy = sys.argv[2] == "fancy"
render._colour_index.update({k: render.nearest_256(*v) for k, v in render.RGB.items()})


class Win:
    def __init__(s, h, w): s.h, s.w = h, w; s.erase()
    def getmaxyx(s): return s.h, s.w
    def erase(s): s.g = [[(" ", 0)] * s.w for _ in range(s.h)]
    def addnstr(s, y, x, t, n, a=0):
        for i, ch in enumerate(t[:n]):
            if x + i < s.w: s.g[y][x + i] = (ch, a)
    def refresh(s): pass


def to_png(win, path, cw=11, ch=22):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 17)
    img = Image.new("RGB", (win.w * cw, win.h * ch), BG)
    d = ImageDraw.Draw(img)
    for y, row in enumerate(win.g):
        for x, (c, a) in enumerate(row):
            fg, bg = pairs.get((a >> 8) & 0xff, (curses.COLOR_WHITE, -1))
            fgc, bgc = idx_rgb(fg) or (225, 225, 225), idx_rgb(bg) or BG
            if a & curses.A_REVERSE:
                fgc, bgc = bgc, fgc
            x0, y0 = x * cw, y * ch
            d.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], fill=bgc)
            if c == "▀":
                d.rectangle([x0, y0, x0 + cw - 1, y0 + ch // 2 - 1], fill=fgc)
            elif c == "▄":
                d.rectangle([x0, y0 + ch // 2, x0 + cw - 1, y0 + ch - 1], fill=fgc)
            elif c != " ":
                d.text((x0, y0 + 1), c, font=font, fill=fgc)
    img.save(path)


lay, prm = dk.make_variant("classic", 3)
g = Game(lay, prm, ParametricKong(), seed=3)
g.player.invuln_until = 0
for _ in range(int(float(sys.argv[3]) / 0.05)):
    g.step(greedy(g))
if len(sys.argv) > 5 and sys.argv[5] == "flash":
    g.lives = 1
    g.t = round(g.t * 6) / 6                                  # land on the "on" half of the blink
if len(sys.argv) > 5 and sys.argv[5] == "rage":
    g.provoke(0, charge=True)
    for _ in range(30): g.step(())
h, w = render.needed_size(lay.to_dict())
win = Win(h, w + 2)
render.draw_game(win, lay.to_dict(), g.state(), {"score": 1200, "best": 5000, "opponent": "AI Kong",
    "plan": "Ambush the x=48 climb with fast barrels", "banner": "", "popups": [],
    "said": ("pauline", "He's eyeing the left ladder! Hurry!"),
    "llm": "haiku-5.5 · plan 6.5s · lines 13.6s · 9 calls", "cost": "$0.0044",
    "score_flash": len(sys.argv) > 5 and sys.argv[5] == "flash", "lives_flash": False})
to_png(win, sys.argv[4])
print("saved", sys.argv[4])
