"""Microsoft-Decision-1 as AI Kong's tactician: one call answers every question for the next throw window.

It is a decisions model (calibrated probabilities over fixed options, no text), reached through
OpenRouter's /api/alpha/decisions endpoint, not chat completions.

    python3 docs/llm-speed/bench_decision1.py .
"""
import json, statistics, sys, time, urllib.request
sys.path.insert(0, sys.argv[1])
import dk
dk.load_env_files()
from director import load_api_key
from dkgame.engine import Game
from dkgame.lookahead import greedy

lay, prm = dk.make_variant("random", 5)
g = Game(lay, prm, dk.SCRIPTED["classic"](), seed=5)
for _ in range(200):
    g.step(greedy(g))
p, k = g.player, g.kong_body
ladders = ", ".join(f"x={x} from floor {b} to {b + 1}" for x, b in lay.ladders)
barrels = ", ".join(f"floor {b.floor} x={b.x} rolling {'right' if b.direction > 0 else 'left'}" for b in g.barrels) or "none"
state = (f"Donkey Kong. Girders 0 (bottom) to {lay.top}; ladders: {ladders}. Player: floor {p.floor}, x={p.x}, {p.mode}. "
         f"Kong: floor {k.floor}, x={k.x}. Barrels on the board: {barrels}. Barrels left: {g.barrels_left}. "
         f"Strategist's plan: ambush their climbs with always-down barrels.")
questions = {
    "throw": {"type": "noul", "instructions": "Should Kong throw a barrel in the next two seconds?"},
    "route": {"type": "choice", "instructions": "Which route should the barrel take to hit the player?",
              "criteria": {"always": "goes down every ladder it meets", "never": "ignores ladders",
                           "random": "coin flip at each ladder", "toward_player": "goes down ladders while the player is below"}},
    "speed": {"type": "choice", "instructions": "How fast should the barrel roll?",
              "criteria": {"slow": "easy to time", "normal": "standard"}},
    "stand": {"type": "choice", "instructions": "Where along his girder should Kong stand?",
              "criteria": {"left": "near the left end", "middle": "the middle", "right": "near the right end"}},
}
key = load_api_key()
body = json.dumps({"model": "microsoft/microsoft-decision-1", "state": state, "questions": questions}).encode()
times, answers, cost = [], [], 0.0
for i in range(8):
    req = urllib.request.Request("https://openrouter.ai/api/alpha/decisions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    times.append(time.monotonic() - t)
    a = d["answers"]
    answers.append((round(a["throw"]["noul"], 2), a["route"]["choice"], round(max(a["route"]["probabilities"].values()), 2),
                    a["speed"]["choice"], a["stand"]["choice"]))
    cost += d["usage"]["cost"]
print("state:", state[:220], "...")
print("answers (throw prob, route, route prob, speed, stand):", answers)
warm = times[1:]
print(f"latency: first {times[0]:.2f}s, warm median {statistics.median(warm):.2f}s, worst {max(warm):.2f}s; "
      f"{d['usage']['input_tokens']} input tokens; ${cost / len(times):.6f} per call")
