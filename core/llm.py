"""LLM client. Uses the OpenAI SDK only as an HTTP client for OpenRouter / Ollama (both free)."""
from __future__ import annotations

from typing import Protocol

from . import config

DEFAULT_MODELS = {"openrouter": "openrouter/free", "ollama": "qwen2.5-coder:7b"}


class LLM(Protocol):
    def chat(self, messages: list[dict], temperature: float = 0.0) -> str: ...


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=config.LLM_TIMEOUT_SEC)

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature)
        return (r.choices[0].message.content or "").strip()


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    provider = (provider or config.LLM_PROVIDER).lower()
    model = model or config.LLM_MODEL or DEFAULT_MODELS.get(provider, "")
    if provider == "openrouter":
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is missing. Put it in the .env file "
                               "(free key: https://openrouter.ai/keys) or set LLM_PROVIDER=ollama.")
        return OpenAICompatLLM("https://openrouter.ai/api/v1", config.OPENROUTER_API_KEY, model)
    if provider == "ollama":
        return OpenAICompatLLM(config.OLLAMA_BASE_URL, "ollama", model)
    raise ValueError(f"Unknown LLM provider '{provider}' (use openrouter or ollama)")
