"""Do the tactician's answers make sense? Same live-board request, N calls per model: how many throws,
which routes and speeds, where Kong stands, and the median time.

    python3 docs/llm-speed/bench_tactics_quality.py . local:gemma3:1b local:llama3.2:latest groq:openai/gpt-oss-20b
"""
import collections, os, statistics, sys, time
sys.path.insert(0, sys.argv[1])
os.environ.pop("KONG_PROVIDER", None)
import dk
dk.load_env_files()
from director import Layer, LLMClient
from dkgame.engine import Game
from dkgame.director_kong import TACTICIAN_SCHEMA, TACTICIAN_SYSTEM, DirectorKong
from dkgame.lookahead import greedy

N = 8
lay, prm = dk.make_variant("random", 5)
g = Game(lay, prm, dk.SCRIPTED["classic"](), seed=5)
kong = DirectorKong(background=True, client=object(), guard=False); kong.game = g
kong.director.strategy = "Ambush their climbs: fast always-down barrels when they reach a ladder"
for _ in range(200):
    g.step(greedy(g))
prompt = kong.tactics_prompt(g.kong_observation())
print(f"board: player floor {prompt['player']['floor']} x {prompt['player']['x']} ({prompt['player']['mode']}), "
      f"max {prompt['rules']['max_throws_per_decision']} throws, speeds {prompt['rules']['speeds']}")
local = LLMClient(api_key="ollama", base_url="http://localhost:11434/v1", timeout=120)
cloud = LLMClient(timeout=30)
for spec in sys.argv[2:]:
    where, model = spec.split(":", 1)
    client = local if where == "local" else cloud
    layer = Layer(name="tactician", model=model, reasoning="off" if where == "local" else "low",
                  provider=None if where == "local" else where, max_tokens=400 if where == "local" else 1500)
    times, counts, routes, speeds, xs, bad = [], [], collections.Counter(), collections.Counter(), [], 0
    for _ in range(N):
        t = time.monotonic()
        try:
            r = client.ask(layer, TACTICIAN_SYSTEM, prompt, TACTICIAN_SCHEMA, DirectorKong._valid_tactics)
        except Exception as exc:
            bad += 1
            print(f"  {spec}: {str(exc)[:120]}")
            continue
        times.append(time.monotonic() - t)
        counts.append(len(r["throws"])); xs.append(round(r["kong_x"], 2))
        routes.update(x.get("route") for x in r["throws"]); speeds.update(x.get("speed") for x in r["throws"])
    allowed = set(prompt["rules"]["speeds"])
    illegal = sum(n for s, n in speeds.items() if s not in allowed)
    print(f"{spec:28} median {statistics.median(times[1:] or times or [0]):.2f}s | throws/call {counts} | routes {dict(routes)} | "
          f"speeds {dict(speeds)}{' (' + str(illegal) + ' not allowed this level)' if illegal else ''} | kong_x {sorted(set(xs))} | failed {bad}", flush=True)
