import sys, time, statistics; sys.path.insert(0, sys.argv[1])
import dk; dk.load_env_files()
from director import Layer, LLMClient
from dkgame.engine import Game
from dkgame.director_kong import DirectorKong
from dkgame.lookahead import greedy
for model, prov, reasoning in [("anthropic/claude-haiku-5.5", None, "medium"),
                               ("openai/gpt-oss-120b", "groq", "medium"), ("openai/gpt-oss-120b", "cerebras", "medium"),
                               ("openai/gpt-oss-120b", "groq", "low")]:
    times, strategies, errs = [], [], 0
    client = LLMClient(provider=prov, timeout=60)
    for seed in (1, 2, 3):
        lay, prm = dk.make_variant("random", seed)
        kong = DirectorKong(background=False, guard=False, client=client,
                            layer=Layer(name="director", model=model, reasoning=reasoning, max_tokens=8000))
        g = Game(lay, prm, kong, seed=seed); kong.attach(g)
        for _ in range(200): g.step(greedy(g))           # 10 s of play so the prompt has real history
        obs = g.kong_observation(); t = time.monotonic()
        kong.director.update(g.t, kong.context(obs), kong.outcomes(obs), trigger="interval")
        d = kong.director.decisions[-1]
        if d.error: errs += 1; continue
        times.append(time.monotonic() - t); strategies.append(d.reply["strategy"])
    u = client.total_usage()
    label = f"{model} @ {prov or 'default'} ({reasoning})"
    med = f"median {statistics.median(times):.2f}s" if times else "no successes"
    print(f"{label:52} {med}  errors {errs}  ${u['cost_usd']/max(1,u['calls']):.5f}/call  in {u['prompt_tokens']//max(1,u['calls'])} tok", flush=True)
    for s in strategies[:2]: print("      ", s)
