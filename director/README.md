# director: real LLM reasoning in a real-time game, with zero latency

A small, standard-library-only Python library for a pattern that keeps coming up when you want an
LLM to make a game (or any real-time system) smarter:

> **A slow LLM reasons about the player and sets a handful of typed knobs. A fast, deterministic
> engine reads those knobs every tick.** The LLM never touches game state, never blocks the game
> loop, and can never push the game outside what its designer allowed.

```
engine (every tick) ─ telemetry ─► Profiler ─ profile ─► Director (LLM, every ~20 s or on a trigger)
   ▲                                                        │ {opponent_model, strategy, reasons, taunt, knobs}
   └── KnobSet.apply ◄── FeasibilityGuard ◄── KnobSet.validate ◄┘
       (engine reads knobs)   (still winnable?)   (bounds, max step, cooldowns)
```

It was extracted from the Donkey Kong arena in this repo, where an LLM directs Kong's barrel
throwing (`dkgame/director_kong.py`).

## Why not just call the LLM from the game?

| Naive approach | Problem | What `director` does |
|---|---|---|
| LLM picks actions every frame | Seconds of latency, twitchy and expensive | LLM sets **policy** (knobs) every few seconds; deterministic code acts every tick |
| LLM edits game state | Can cheat, break invariants, crash the game | LLM only proposes values for **declared knobs**; everything is validated |
| Free-form JSON replies | Parse errors, missing fields, invented keys | **Strict JSON schema** generated from the knob specs, validation, one retry, then keep current knobs |
| Stateless prompts | Repeats itself, can't learn | Remembers its **past strategies and what each achieved**, plus a player profile |
| "Make it harder" | Unwinnable spikes | **Fairness guard** simulates the change and vetoes it if a strong player could no longer survive |
| Opaque behaviour | Players can't tell the AI is thinking | Every decision has a one-line **strategy**, reasons and an optional taunt to show on screen |

## Quick start

```python
from director import Director, KnobSet, KnobSpec, Layer, Profiler

knobs = KnobSet([
    KnobSpec("spawn_rate", "float", 0.5, min=0.1, max=1.0, max_step=0.2,
             description="enemy spawn rate, relative to the level maximum"),
    KnobSpec("enemy_mix", "weights", {"grunt": 0.8, "sniper": 0.2}, choices=("grunt", "sniper"), max_step=0.3,
             description="chance of each enemy type"),
    KnobSpec("ambush", "bool", False, cooldown_s=15.0,
             description="spawn enemies behind the player instead of ahead"),
])
profiler = Profiler()
director = Director(
    knobs,
    brief="A top-down shooter. Keep the player challenged: aim for them winning about 60% of fights.",
    layer=Layer("director", model="anthropic/claude-haiku-5.5", reasoning="medium"),
    every_s=20.0,
    background=True,            # real-time game: think in a thread, never block the loop
)

def tick(t, game):
    # telemetry is cheap; record it as things happen
    if game.player_died:
        profiler.count("deaths")
        profiler.event("death", t, x=game.player.x, cause=game.killer)

    director.poll()                                     # apply a finished decision, if any
    reason = director.due(t, trigger="player died" if game.player_died else None)
    if reason:
        director.update(t, context={"level": game.level, "profile": profiler.summary()},
                        outcomes={"deaths": profiler.counters.get("deaths", 0)}, trigger=reason)

    game.spawner.rate = knobs["spawn_rate"]            # the engine reads knobs, every tick
    game.spawner.mix = knobs["enemy_mix"]
    game.hud.subtitle = director.strategy               # optional: show what the AI is planning
```

`outcomes` are cumulative counters. The director records how much each changed under each
strategy, so its next prompt contains, for example, "strategy *flank with snipers*: deaths +2 in 20 s".

## Knobs

`KnobSpec(name, kind, default, description=..., ...)` where `kind` is:

| kind | value | options |
|---|---|---|
| `float` / `int` | number | `min`, `max` (required), `max_step` |
| `bool` | true/false | |
| `enum` | one of `choices` | |
| `weights` | `{choice: weight}`, normalised to sum to 1 | `choices` (required), `max_step` per component |

All kinds accept `cooldown_s`, the minimum game time between changes.
`KnobSet.validate(proposal, t)` returns `(changes, notes)`. It clamps to bounds, limits each change
to `max_step`, enforces cooldowns, and rejects unknown knobs or bad values with an explanation. It
never raises. The notes are fed back to the LLM on its next update, so it learns its limits.
`KnobSet.json_schema()` is the strict schema the LLM must fill (every knob required).

Design tip: make knobs **policies, not actions** ("burst probability", "ambush on/off"), and keep
the game's own rules as the hard limits. The knobs choose *how* to play within those limits.

## Fairness guard

```python
from director import FeasibilityGuard

def simulate(snapshot, knobs, rollout):
    sim = copy.deepcopy(snapshot)
    sim.apply_knobs(knobs)
    for _ in range(seconds_to_ticks(8)):
        sim.step(reference_player.act(sim))
        if sim.player_died:
            return {"survived": False}
    return {"survived": True}

director = Director(knobs, brief, guard=FeasibilityGuard(simulate, rollouts=3), ...)
director.update(t, context, outcomes, snapshot=deep_copy_of_the_game_without_the_director)
```

The guard runs the candidate knobs and, only if no rollout survives, the **current** knobs too. It
vetoes a change only when the change itself makes survival impossible; if the state was doomed
anyway, the change isn't blamed. A guard whose reference player is omniscient (it can see the future)
gives a necessary condition for fairness: if even it can't survive, no human can. Vetoes are
explained to the LLM on its next update ("be cunning, not impossible").

Take the snapshot on the game thread (in background mode the guard runs in the director's thread),
and leave the live director out of the copy (`copy.deepcopy(game, memo={id(game.director): None})`).

## The LLM layer

`Layer(name, model, reasoning, json_mode, max_tokens)` configures one role; `LLMClient` does the transport.

- **Endpoint:** OpenRouter by default (`KONG_BASE_URL` to change), so any model id works, e.g.
  `anthropic/claude-haiku-5.5`, `openai/gpt-oss-120b`, `qwen/qwen3.7-flash`.
- **Output:**
  - `json_mode="schema"` (default) uses strict structured outputs and only routes to providers that
    support them (`provider.require_parameters`).
  - Use `json_mode="object"` for models without structured-output support. Replies are still
    validated and retried.
- **Reasoning:** `reasoning="none" | "low" | "medium" | "high"` is passed as OpenRouter's reasoning
  effort. Medium is a good default for a director that runs every ~20 s. No temperature is sent,
  since some current models reject it.
- **API key:** from `OPENROUTER_API_KEY`, or from `~/.config/dk-game/openrouter.env` containing
  `OPENROUTER_API_KEY=...` (chmod 600). It is never logged.
- **Usage:** `client.usage` (per layer) and `client.total_usage()` report calls, retries, failures,
  tokens and cost.
- **Testing:** `Director(..., client=stub)` accepts any object with
  `ask(layer, system, prompt, schema, validate)`, so tests and offline runs need no network (see
  `test_director.py`).

The director's instructions (`DIRECTOR_SYSTEM`) are written for an **adversary** ("set up and pay
off, exploit habits"). For other goals, such as adaptive difficulty, a tutor or a pacing director,
pass `system=` with your own instructions and state the goal in `brief`.

## Example: Donkey Kong

`dkgame/director_kong.py` puts Kong on this library:
- `ParametricKong` is a deterministic Kong driven by 7 knobs: throw rate, speed mix, route mix,
  burst chance, climb ambush, rhythm jitter and hold-back lulls.
- `DirectorKong` adds the LLM director and a guard backed by `dkgame/lookahead.py`.

```bash
python3 dk.py --kong director                          # play against it (taunts + plan on screen)
python3 ../arena/measure_director.py --episodes 10     # (benchmark repo) director vs random vs static knobs
```

## Measuring whether it is actually smart

A director that writes clever strategies can still have zero effect on play. Always compare it to
**static** and **random** knob controllers against the **same fixed players** on the same seeds
(`arena/measure_director.py`). Only a measurable difference (deaths per minute, win rate) counts.
First finding in DK: the knobs strongly affect an imperfect player, but no knob setting within the
game's rules threatens a near-perfect planner. A director can only be as cunning as its knobs allow.

## Tests

```bash
python3 -m unittest discover director        # from dk-game/
```

## Limits

- One LLM call at a time per director; triggers that arrive while it's thinking are dropped until
  the next one.
- The guard costs simulation time (≈0.75 s per DK rollout). It only runs when knobs actually change.
- In background mode, a guard check still in progress can delay process exit by a few seconds.
