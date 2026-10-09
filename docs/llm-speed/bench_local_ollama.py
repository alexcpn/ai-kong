"""Time the game's taunt-reply and tactician requests on local Ollama models (OpenAI-compatible API).

    python3 docs/llm-speed/bench_local_ollama.py . llama3.2:latest gemma:latest ...
"""
import os, statistics, sys, time
sys.path.insert(0, sys.argv[1])
os.environ.pop("KONG_PROVIDER", None)
import dk
from director import Layer, LLMClient
from dkgame.engine import Game
from dkgame.director_kong import (TACTICIAN_SCHEMA, TACTICIAN_SYSTEM, VOICE_SCHEMA, VOICE_SYSTEM, DirectorKong)
from dkgame.lookahead import greedy

lay, prm = dk.make_variant("random", 5)
g = Game(lay, prm, dk.SCRIPTED["classic"](), seed=5)
kong = DirectorKong(background=True, client=object(), guard=False); kong.game = g
for _ in range(200):
    g.step(greedy(g))
tactics = kong.tactics_prompt(g.kong_observation())
taunt = {"player_says": "Bet you can't hit me on a ladder.", "your_anger": 80, "barrels_left": 30,
         "player": {"floor": 2, "x": 20, "mode": "ground"}, "top_floor": 4, "lives": 2}
client = LLMClient(api_key="ollama", base_url="http://localhost:11434/v1", timeout=180)
for model in sys.argv[2:]:
    for task, system, prompt, schema, check in (("taunt", VOICE_SYSTEM, taunt, VOICE_SCHEMA, DirectorKong._valid_voice),
                                                 ("tactician", TACTICIAN_SYSTEM, tactics, TACTICIAN_SCHEMA, DirectorKong._valid_tactics)):
        for mode in ("schema", "object"):
            layer = Layer(name=f"{model}-{task}", model=model, reasoning="off", json_mode=mode, max_tokens=400)
            times, errors, sample = [], 0, None
            for i in range(6):                         # call 0 includes loading the model
                t = time.monotonic()
                try:
                    sample = client.ask(layer, system, prompt, schema, check)
                    times.append(time.monotonic() - t)
                except Exception as exc:
                    errors += 1; last = str(exc)[:80]
                    if errors >= 3: break
            if len(times) >= 2: break
        if len(times) >= 2:
            warm = times[1:]
            print(f"{model:40} {task:9} json={mode:6} cold {times[0]:5.2f}s  warm median {statistics.median(warm):.2f}s "
                  f"worst {max(warm):.2f}s  bad replies {errors}/6  e.g. {str(sample)[:70]}", flush=True)
        else:
            print(f"{model:40} {task:9} FAILED ({errors} errors): {last}", flush=True)
