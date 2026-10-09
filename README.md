# AI Kong

Donkey Kong, against a Kong that thinks.

A terminal Donkey Kong where Kong can be driven by an LLM that **studies how you play and re-plans
its tactics**, without ever making the game lag. Pure Python standard library: no installs
(`python-dotenv` is optional, to load settings from `.config/`).

## Quick start

```bash
python3 dk.py                                      # menu: pick an opponent and a board

# no LLM, for testing without a key
python3 dk.py --kong classic
python3 dk.py --kong classic --board tall --seed 7 # same seed = same board

# AI Kong (needs an OpenRouter key, see below)
python3 dk.py --kong ai
KONG_DIRECTOR_REASONING=high python3 dk.py --kong ai --board tall
```

Options: `--kong ai|classic`, `--board random|classic|tall|sparse`,
`--seed N`. Without `--kong` you get the menu.

In a game: arrows (or WASD) move, Space jumps, **T** taunts Kong, P pauses, Q quits.

**Default models:** `anthropic/claude-haiku-5.5` for both of AI Kong's LLM calls: the strategist
(medium reasoning, every ~20 s and when you lose a life or clear a level) and the voice that answers
taunts (no reasoning, so it replies in about 2 s). Change them with `KONG_DIRECTOR_MODEL` /
`KONG_DIRECTOR_REASONING` and `KONG_VOICE_MODEL` (shell or `.config/config.env`).

**Cost per session (measured with the defaults):** about $0.0006 per strategy call and $0.0001 per
taunt reply. With 3-5 strategy calls a minute that's roughly $0.002-0.003 per minute of play, so
2-3 cents for a 10-minute session. The running total is shown at the bottom right while you play.

To configure AI Kong from files instead of the shell, put the key in
`.config/opneroutere.env` (git-ignored; copy `.config/opneroutere.env.ex`) and settings in
`.config/config.env`, then `pip install python-dotenv`. `dk.py` loads both at start-up, and
variables already set in your shell take precedence.

Use a terminal of at least 60 × 25 (60 × 29 for the Tall board). Pick an opponent and a board in the menu, then climb to Pauline (`|♀|`)
at the top while Kong throws barrels. From level 2, fireballs (`※`) roam the girders too.

## Controls

| Key | Action |
|---|---|
| ← → / A D / H L | walk (you can steer in the air) |
| ↑ ↓ / W S / K J | climb up / down a ladder (step off mid-ladder to drop) |
| Space / Enter | jump (hold for a little extra height) |
| T | taunt Kong: ←→ pick a line, Enter sends, Esc cancels (once every 4 s) |
| P | pause |
| Q | end the game |

## Opponents

| Kong | Style |
|---|---|
| **AI Kong** | An LLM watches your habits (where you wait, how early you jump, which ladders you use, how you died) and re-plans Kong's tactics every ~20 seconds, after every life you lose, and when you taunt him. Its current plan is shown under the board |
| Classic (no LLM) | Arcade rhythm: one barrel at a time, random routes. For testing without a key or network |

**Kong moves and his barrels are limited**: he comes down from the top and roams about three girders
above you, climbing as you climb (up to the top girder), and throws from wherever he is (AI Kong
picks the spot, e.g. above a ladder you need). Catch him anywhere but the top girder and he's beaten.
He has 45 barrels on level 1, 8 more each level, shown as `●×45` at the top; they refill on a new
level, not when you lose a life.

**Taunts and Kong's temper**: press T and pick a line ("Bet you can't hit me on a ladder.", "I'm
taking the left ladder." ...). Each taunt fills Kong's ANGER meter (it cools while he sits at the top),
and he grunts at once. AI Kong then answers in his own words within a second or two, using a small,
fast LLM call (no reasoning; set `KONG_VOICE_MODEL` to change it), and may **take the bait**. At full
anger (three quick taunts) he loses his temper either way.

An angry Kong storms down to your girder, throwing barrels on the way and from a few columns away
(these come out of his supply). **Touch him and Kong is beaten**: +3000 and the level is cleared.
After a few seconds on your girder he climbs back up, and you can still catch him on the way. Classic Kong has a temper too (grunts only, no LLM).

Boards: **Random** (a new layout each game), **Classic**, **Tall** (6 girders), **Sparse** (a
single ladder between girders). Use `--seed N` to replay the same board.

## Scoring

The game is endless; you play for score. High scores are kept per opponent, in
`~/.cache/dk-game/highscores.json`.

| Event | Points |
|---|---|
| Reach a higher girder (once per girder per level) | 100 |
| Jump over a barrel | 100 |
| Land on a fireball from above (destroys it) | 200 |
| Clear a level | 1000 × level + 10 × seconds left |

You have 3 lives, and each level has a time limit: running out costs a life. Each level is harder.

## AI Kong

It needs an [OpenRouter](https://openrouter.ai) API key. Create the key file once in your own terminal:

```bash
mkdir -p ~/.config/dk-game
printf 'OPENROUTER_API_KEY=%s\n' 'sk-or-...' > ~/.config/dk-game/openrouter.env
chmod 600 ~/.config/dk-game/openrouter.env
```

(or export `OPENROUTER_API_KEY`). The default model is `anthropic/claude-haiku-5.5` with medium
reasoning; that costs roughly $0.002-0.003 per minute of play (see Quick start). To use another
model:

```bash
KONG_DIRECTOR_MODEL=openai/gpt-oss-120b python3 dk.py --kong ai
KONG_DIRECTOR_REASONING=high python3 dk.py --kong ai       # more deliberate plans
```

How it stays fair and lag-free:
- The LLM never moves barrels itself. It sets 8 **knobs** for a deterministic Kong: throw rate,
  speed mix, route mix, burst chance, ladder ambush, rhythm jitter, hold-back lulls and where Kong
  stands.
- Every change is clamped to limits, and changes are rate-limited.
- A **fairness guard** simulates proposed changes with a near-perfect player and vetoes any that
  would make the game unwinnable.
- The LLM runs in the background, so the game never waits for it. Until its first plan arrives,
  Kong plays its default tactics.

How the LLM is used: it is far too slow to steer anything moment to moment, so it acts as a
commander. Every ~20 seconds (and when you lose a life or clear a level) the strategist, with
reasoning, studies your habits and sets Kong's 8 knobs (throwing tactics and where he stands); the
engine carries that out every tick. Taunts go to a second,
fast LLM call with no reasoning, so Kong answers in a second or two. The knob limits and fairness
guard apply to everything the strategist decides.

The screen shows what this costs as you play: the running total at the right of the bottom row, and,
if your terminal has a spare row (26+ rows; ~90 columns shows it in full), which models play Kong, their average
reply time and the number of calls. The menu shows the models before you start, and the game-over
line the total.

Only AI Kong uses the network: it sends OpenRouter a summary of the game state and your play
statistics. Nothing else leaves your machine.

## How it's built

```
dk.py                 the game: menu, play loop, scoring, high scores
dkgame/engine.py      deterministic engine (fixed 20 ticks/s, boards, physics, Kong interface)
dkgame/kongs.py       scripted Kongs
dkgame/director_kong.py   knob-driven Kong + the LLM director, fairness guard simulation
dkgame/llm_kong.py    an earlier two-layer LLM Kong (strategist + tactician); kept for comparison
dkgame/lookahead.py   near-perfect reference player used by the fairness guard
dkgame/render.py      curses drawing
director/             the reusable "AI director" library (see director/README.md)
```

The [`director/`](director/README.md) library is game-agnostic. Use it to put LLM reasoning behind
typed knobs in any real-time game or simulation.

## Tests

```bash
python3 -m unittest discover .        # from dk-game/: game + director library tests
```

This game came out of DK-Bench, an experiment in benchmarking coding agents. That benchmark lives in a
separate repository and embeds this folder; you don't need it to play.
