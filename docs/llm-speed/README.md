# LLM speed and cost data (October 2026)

Published and measured inference speed and cost behind AI Kong's model choices, kept for the article
draft and so the measurements can be repeated.

- `data-2026-10.json`: all figures, with sources, methods and dates.
- `bench_taunt_reply.py`, `bench_strategy_call.py`, `bench_tactician_call.py`: the scripts that
  produced the "measured" rows. Run from the repo root with the game folder as the argument, e.g.
  `python3 docs/llm-speed/bench_tactician_call.py .` (needs an OpenRouter key; each run costs well
  under a cent).

Our samples are small (3-5 calls per row, one machine, one day): read them as orders of magnitude and
tail-latency warnings, not as rankings.

The article draft built on this data: [`docs/article/llm-real-time-game-2026-10.md`](../article/llm-real-time-game-2026-10.md).
