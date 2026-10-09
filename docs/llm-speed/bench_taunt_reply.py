import sys, time, statistics; sys.path.insert(0, sys.argv[1])
import dk; dk.load_env_files()
from director import Layer, LLMClient, LLMError
from director import LLMError

# The taunt-reply call as it was measured in October 2026 (taunts were later removed from the game).
VOICE_SYSTEM = """You are KONG in a terminal Donkey Kong game. The player just taunted you. Answer in
character and decide whether to take the bait.

Reply with JSON: say (<= 60 characters, playful, family-friendly, answering what they said) and charge
(true = lose your temper NOW and storm down the ladders after them). Charging is a gamble: up close your
point-blank barrels are deadly, but if the player touches you while you are down you are defeated and
they clear the level. At anger 100 you charge anyway. Weigh your anger, barrels_left, where the player
is (far below = long trip, near the top = they can reach you), and whether the taunt is bait or a bluff."""

VOICE_SCHEMA = {"type": "object", "properties": {"say": {"type": "string"}, "charge": {"type": "boolean"}},
                "required": ["say", "charge"], "additionalProperties": False}


def valid_voice(reply):
    if not isinstance(reply, dict) or not isinstance(reply.get("say"), str) or not isinstance(reply.get("charge"), bool):
        raise LLMError("bad voice reply")
    return reply

prompt = {"player_says": "Bet you can't hit me on a ladder.", "your_anger": 80, "barrels_left": 30,
          "player": {"floor": 2, "x": 20, "mode": "ground"}, "top_floor": 4, "lives": 2}
cands = [("anthropic/claude-haiku-5.5", None, "none"),
         ("openai/gpt-oss-120b", "cerebras", "low"), ("openai/gpt-oss-120b", "groq", "low"),
         ("openai/gpt-oss-120b", "baseten", "low"), ("openai/gpt-oss-20b", "groq", "low"),
         ("meta-llama/llama-4-scout", "groq", "none"), ("meta-llama/llama-4-scout", "cerebras", "none")]
for model, prov, reasoning in cands:
    c = LLMClient(provider=prov, timeout=30)
    times, err, say = [], None, ""
    for json_mode in ("schema", "object"):
        layer = Layer(name="voice", model=model, reasoning=reasoning, json_mode=json_mode, max_tokens=400)
        times, err = [], None
        for _ in range(3):
            t = time.monotonic()
            try:
                r = c.ask(layer, VOICE_SYSTEM, prompt, VOICE_SCHEMA, valid_voice); say = r["say"]
                times.append(time.monotonic() - t)
            except Exception as e:
                err = str(e)[:90]; break
        if times: break
    u = c.total_usage()
    label = f"{model} @ {prov or 'default'} ({reasoning}, {json_mode if times else '-'})"
    if times:
        print(f"{label:62} median {statistics.median(times):.2f}s  best {min(times):.2f}s  ${u['cost_usd']/max(1,u['calls']):.5f}/call  \"{say[:40]}\"", flush=True)
    else:
        print(f"{label:62} FAILED: {err}", flush=True)
