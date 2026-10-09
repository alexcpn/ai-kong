# Donkey Kong, against a Kong that thinks

A terminal Donkey Kong where Kong can be driven by an LLM that **studies how you play and re-plans
its tactics**, without ever making the game lag. Pure Python standard library: no installs.

```bash
python3 dk.py
```

Use a terminal of at least 60 × 24 (60 × 28 for the Tall board). Pick an opponent and a board in the menu, then climb to Pauline (`|♀|`)
at the top while Kong throws barrels. From level 2, fireballs (`※`) roam the girders too.

## Controls

| Key | Action |
|---|---|
| ← → / A D / H L | walk (you can steer in the air) |
| ↑ ↓ / W S / K J | climb up / down a ladder (step off mid-ladder to drop) |
| Space / Enter | jump (hold for a little extra height) |
| P | pause |
| Q | end the game |

## Opponents

| Kong | Style |
|---|---|
| Classic | Arcade rhythm: one barrel at a time, random routes |
| Random | Anything goes |
| Aim | Sends barrels down toward your girder, faster as you climb |
| Sniper | Slow/fast pairs to break your timing; fires at you the moment you're on a ladder |
| Trickster | Barrels that skip the ladders you expect, irregular bursts |
| **AI Director** | An LLM watches your habits (where you wait, how early you jump, which ladders you use, how you died) and re-plans Kong's tactics every ~20 seconds and after every life you lose. Its current plan is shown under the board |

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

## The AI Director

It needs an [OpenRouter](https://openrouter.ai) API key. Create the key file once in your own terminal:

```bash
mkdir -p ~/.config/dk-game
printf 'OPENROUTER_API_KEY=%s\n' 'sk-or-...' > ~/.config/dk-game/openrouter.env
chmod 600 ~/.config/dk-game/openrouter.env
```

(or export `OPENROUTER_API_KEY`). The default model is `anthropic/claude-haiku-5.5` with medium
reasoning; at about one call every 20 seconds that should cost in the region of a cent per few minutes
of play (an estimate). To use another model:

```bash
KONG_DIRECTOR_MODEL=openai/gpt-oss-120b python3 dk.py --kong director
KONG_DIRECTOR_REASONING=high python3 dk.py --kong director       # more deliberate plans
```

How it stays fair and lag-free:
- The LLM never moves barrels itself. It sets 7 **knobs** for a deterministic Kong: throw rate,
  speed mix, route mix, burst chance, ladder ambush, rhythm jitter and hold-back lulls.
- Every change is clamped to limits, and changes are rate-limited.
- A **fairness guard** simulates proposed changes with a near-perfect player and vetoes any that
  would make the game unwinnable.
- The LLM runs in the background, so the game never waits for it. Until its first plan arrives,
  Kong plays its default tactics.

Only the AI Director uses the network: it sends OpenRouter a summary of the game state and your play
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

This game came out of a benchmark experiment; the benchmark lives in the parent repository and
imports this folder. You don't need any of it to play.
