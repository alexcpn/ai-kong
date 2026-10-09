import sys, os, time, statistics; sys.path.insert(0, sys.argv[1])
import dk; dk.load_env_files()
from director import Layer, LLMClient
from dkgame.engine import Game
from dkgame.director_kong import TACTICIAN_SYSTEM, TACTICIAN_SCHEMA, DirectorKong
from dkgame.lookahead import greedy
lay, prm = dk.make_variant("random", 5)
g = Game(lay, prm, dk.SCRIPTED["classic"](), seed=5)
kong = DirectorKong(background=True, client=object(), guard=False); kong.game = g
for _ in range(200): g.step(greedy(g))
prompt = kong.tactics_prompt(g.kong_observation())
c = LLMClient(timeout=30)
for model, prov, reasoning in [("openai/gpt-oss-120b","groq","low"), ("openai/gpt-oss-120b","cerebras","low"),
                               ("openai/gpt-oss-20b","groq","low"), ("openai/gpt-oss-120b","groq","none")]:
    layer = Layer(name=f"t-{prov}-{model[-4:]}-{reasoning}", model=model, reasoning=reasoning, provider=prov, max_tokens=1500)
    times = []
    for _ in range(5):
        t = time.monotonic()
        try:
            r = c.ask(layer, TACTICIAN_SYSTEM, prompt, TACTICIAN_SCHEMA, DirectorKong._valid_tactics); times.append(time.monotonic()-t)
        except Exception as e:
            print(model, prov, "ERR", str(e)[:80]); break
    u = c.usage.get(layer.name, {})
    if times:
        print(f"{model} @ {prov} ({reasoning}): median {statistics.median(times):.2f}s  p90 {sorted(times)[-1]:.2f}s  "
              f"out {u['completion_tokens']//u['calls']} tok (reasoning {u['reasoning_tokens']//u['calls']})  ${u['cost_usd']/u['calls']:.5f}  last={r['throws'][:2]}", flush=True)
