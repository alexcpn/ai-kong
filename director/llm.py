"""OpenRouter (or any OpenAI-compatible) transport with strict JSON output, validation and retries.

Stdlib only. Never logs the API key. Key lookup order: OPENROUTER_API_KEY, KONG_API_KEY, then
~/.config/dk-game/openrouter.env (a KEY=VALUE file; keep it chmod 600; the older
~/.config/dk-bench/openrouter.env is still read).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

DEFAULT_MODEL = "anthropic/claude-haiku-5.5"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
KEY_FILE = os.path.expanduser("~/.config/dk-game/openrouter.env")
LEGACY_KEY_FILE = os.path.expanduser("~/.config/dk-bench/openrouter.env")


class LLMError(Exception):
    pass


def load_api_key() -> str | None:
    for var in ("OPENROUTER_API_KEY", "KONG_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    for path in (KEY_FILE, LEGACY_KEY_FILE):
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    key, _, value = line.strip().partition("=")
                    if key.strip() in ("OPENROUTER_API_KEY", "KONG_API_KEY") and value.strip():
                        return value.strip().strip("'\"")
        except OSError:
            continue
    return None


@dataclass
class Layer:
    """One LLM role (e.g. strategist): which model, how much reasoning, how to constrain output."""
    name: str
    model: str = DEFAULT_MODEL
    reasoning: str = "medium"      # none (send nothing) | off (ask the model not to think) | low | medium | high
    json_mode: str = "schema"      # schema (strict structured outputs) | object (plain JSON mode)
    max_tokens: int = 8000
    provider: str | None = None    # pin one OpenRouter provider for this layer (else the client's)

    @classmethod
    def from_env(cls, name: str, default_reasoning: str, max_tokens: int, prefix: str = "KONG",
                 **override) -> "Layer":
        env = f"{prefix}_{name.upper()}_"
        return cls(name=name,
                   model=override.get("model") or os.environ.get(env + "MODEL", DEFAULT_MODEL),
                   reasoning=override.get("reasoning") or os.environ.get(env + "REASONING", default_reasoning),
                   json_mode=override.get("json_mode") or os.environ.get(env + "JSON", "schema"),
                   max_tokens=max_tokens,
                   provider=override.get("provider") or os.environ.get(env + "PROVIDER") or None)


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, provider: str | None = None,
                 timeout: float = 60.0, title: str = "DK-Bench director") -> None:
        self.api_key = api_key or load_api_key()
        if not self.api_key:
            raise RuntimeError(f"no API key: set OPENROUTER_API_KEY or create {KEY_FILE}")
        self.base_url = (base_url or os.environ.get("KONG_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.provider = provider if provider is not None else os.environ.get("KONG_PROVIDER")
        self.timeout = timeout
        self.title = title
        self.usage: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _usage(self, layer: str) -> dict:
        return self.usage.setdefault(layer, {"calls": 0, "retries": 0, "failures": 0, "prompt_tokens": 0,
                                             "completion_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0,
                                             "seconds": 0.0})

    def total_usage(self) -> dict:
        keys = ("calls", "retries", "failures", "prompt_tokens", "completion_tokens", "cost_usd")
        with self._lock:
            return {k: sum(u[k] for u in self.usage.values()) for k in keys}

    def ask(self, layer: Layer, system: str, prompt: dict | str, schema: dict,
            validate: Callable[[object], object]) -> object:
        """One validated reply. Retries once on a malformed reply; LLMError if it still fails."""
        user = prompt if isinstance(prompt, str) else json.dumps(prompt, separators=(",", ":"))
        last: Exception | None = None
        for attempt in range(2):
            try:
                return validate(parse_json(self._call(layer, system, user, schema)))
            except (LLMError, ValueError, KeyError, TypeError) as exc:
                last = exc
                with self._lock:
                    self._usage(layer.name)["retries"] += attempt == 0
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                break
        with self._lock:
            self._usage(layer.name)["failures"] += 1
        raise LLMError(f"{type(last).__name__}: {str(last)[:200]}")

    def _call(self, layer: Layer, system: str, user: str, schema: dict) -> str:
        body: dict = {
            "model": layer.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": layer.max_tokens,
            "usage": {"include": True},
        }
        if layer.json_mode == "schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": layer.name, "strict": True, "schema": schema}}
            body["provider"] = {"require_parameters": True}
        else:
            body["response_format"] = {"type": "json_object"}
            body["messages"][0]["content"] += "\n\nReply with ONLY a JSON object matching: " + json.dumps(schema)
        if layer.reasoning in ("low", "medium", "high"):
            body["reasoning"] = {"effort": layer.reasoning, "exclude": True}
        elif layer.reasoning == "off":            # thinking-by-default models (Gemma 4, Qwen 3.5) skip it
            body["reasoning"] = {"effort": "none"}
        provider = layer.provider or self.provider
        if provider:
            body.setdefault("provider", {})["order"] = [provider]
            body["provider"]["allow_fallbacks"] = False
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     "X-Title": self.title})
        started = time.monotonic()
        data = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode())
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise urllib.error.URLError(f"HTTP {exc.code}: {detail}")
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(1.5 * (attempt + 1))
        usage = (data or {}).get("usage") or {}
        with self._lock:
            u = self._usage(layer.name)
            u["calls"] += 1
            u["seconds"] += time.monotonic() - started
            u["prompt_tokens"] += usage.get("prompt_tokens") or 0
            u["completion_tokens"] += usage.get("completion_tokens") or 0
            u["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            u["cost_usd"] += float(usage.get("cost") or 0)
        if not data or "error" in data:
            raise LLMError(f"provider error: {str((data or {}).get('error'))[:200]}")
        return data["choices"][0]["message"].get("content") or ""


def parse_json(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise LLMError("no JSON object in reply")
    return json.loads(text[start:end + 1])
