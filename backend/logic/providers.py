"""LLM provider factory — returns an OpenAI-compatible client + model name.

Supported providers:
  openai    — Standard OpenAI API (OPENAI_API_KEY)
  azure     — Azure OpenAI (OPENAI_API_KEY + OPENAI_API_BASE)
  anthropic — Anthropic API (ANTHROPIC_API_KEY), via thin OpenAI-compat adapter
  ollama    — Local Ollama (no key, OLLAMA_BASE_URL)
  custom    — Any OpenAI-compatible endpoint (custom api_key + base_url)

Auto-detection priority (when provider="auto" or unset):
  1. ANTHROPIC_API_KEY → anthropic
  2. OPENAI_API_BASE   → azure
  3. OPENAI_API_KEY    → openai
  4. OLLAMA_BASE_URL   → ollama
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


# ── Preset env-var names for each provider ────────────────────────────────────

PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "api_key_env":    "OPENAI_API_KEY",
        "model_env":      "OPENAI_MODEL",
        "default_model":  "gpt-4o",
        "label":          "OpenAI",
    },
    "azure": {
        "api_key_env":    "OPENAI_API_KEY",
        "base_url_env":   "OPENAI_API_BASE",
        "model_env":      "AZURE_OPENAI_DEPLOYMENT",
        "default_model":  "gpt-4o",
        "label":          "Azure OpenAI",
    },
    "anthropic": {
        "api_key_env":    "ANTHROPIC_API_KEY",
        "model_env":      "ANTHROPIC_MODEL",
        "default_model":  "claude-opus-4-8",
        "label":          "Anthropic",
    },
    "ollama": {
        "base_url_env":   "OLLAMA_BASE_URL",
        "model_env":      "OLLAMA_MODEL",
        "default_model":  "llama3",
        "default_base_url": "http://localhost:11434/v1",
        "label":          "Ollama (local)",
    },
    "custom": {
        "label":          "Custom / OpenAI-compatible",
    },
}


@dataclass
class ProviderConfig:
    """Fully resolved provider configuration passed to the autopilot."""
    provider: str = "openai"   # openai | azure | anthropic | ollama | custom
    api_key: str = ""
    model: str = ""
    base_url: str = ""         # Azure endpoint, Ollama URL, or custom base URL
    api_version: str = "2024-12-01-preview"  # Azure only

    def effective_model(self) -> str:
        if self.model:
            return self.model
        preset = PROVIDER_PRESETS.get(self.provider, {})
        return preset.get("default_model", "gpt-4o")


# ── Auto-detection from environment ───────────────────────────────────────────

def detect_provider_from_env() -> str | None:
    """Return the provider name auto-detected from env vars, or None."""
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    if os.environ.get("OPENAI_API_BASE", "").strip():
        return "azure"
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.environ.get("OLLAMA_BASE_URL", "").strip():
        return "ollama"
    return None


def provider_from_env(provider: str | None = None) -> ProviderConfig:
    """Build a ProviderConfig from environment variables.

    If ``provider`` is None or "auto", the provider is auto-detected.
    """
    resolved = provider or detect_provider_from_env() or "openai"
    preset = PROVIDER_PRESETS.get(resolved, {})

    api_key   = os.environ.get(preset.get("api_key_env", ""), "").strip()
    base_url  = os.environ.get(preset.get("base_url_env", ""), "").strip()
    model     = os.environ.get(preset.get("model_env", ""), "").strip()

    if resolved == "ollama" and not base_url:
        base_url = preset.get("default_base_url", "http://localhost:11434/v1")

    return ProviderConfig(
        provider=resolved,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


def configured_providers() -> list[str]:
    """Return which providers are configured via env vars."""
    result = []
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        result.append("anthropic")
    if os.environ.get("OPENAI_API_BASE", "").strip():
        result.append("azure")
    if os.environ.get("OPENAI_API_KEY", "").strip():
        result.append("openai")
    if os.environ.get("OLLAMA_BASE_URL", "").strip():
        result.append("ollama")
    return result


# ── Client factory ────────────────────────────────────────────────────────────

def build_client(cfg: ProviderConfig):
    """Return (client, model_name) for the given ProviderConfig."""
    provider = cfg.provider
    if provider == "anthropic":
        return _build_anthropic_client(cfg)
    if provider == "azure":
        return _build_azure_client(cfg)
    if provider in ("ollama", "custom"):
        return _build_openai_compat_client(cfg)
    return _build_openai_client(cfg)


def _build_openai_client(cfg: ProviderConfig):
    from openai import OpenAI
    model = cfg.effective_model()
    client = OpenAI(api_key=cfg.api_key)
    return client, model


def _build_azure_client(cfg: ProviderConfig):
    from openai import AzureOpenAI
    base_url = cfg.base_url or os.environ.get("OPENAI_API_BASE", "")
    model = cfg.effective_model()
    # Prefer cfg.api_version (set by provider_from_env or explicitly), then
    # the env var directly (covers the non-auto path where cfg comes from the
    # frontend default), then fall back to the SDK-safe minimum.
    api_version = (
        cfg.api_version
        or os.environ.get("AZURE_OPENAI_API_VERSION", "")
        or "2024-12-01-preview"
    )
    client = AzureOpenAI(
        api_key=cfg.api_key,
        azure_endpoint=base_url,
        api_version=api_version,
    )
    return client, model


def _build_openai_compat_client(cfg: ProviderConfig):
    from openai import OpenAI
    base_url = cfg.base_url
    if cfg.provider == "ollama" and not base_url:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = cfg.effective_model()
    client = OpenAI(api_key=cfg.api_key or "ollama", base_url=base_url)
    return client, model


def _build_anthropic_client(cfg: ProviderConfig):
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "The 'anthropic' package is required for Anthropic provider. "
            "Install it with: pip install anthropic"
        ) from e
    model = cfg.effective_model()
    raw = anthropic.Anthropic(api_key=cfg.api_key)
    return _AnthropicAdapter(raw), model


# ── Anthropic → OpenAI-compatible adapter ─────────────────────────────────────

class _AnthropicAdapter:
    """Thin adapter so Anthropic looks like an OpenAI client to BaseAgent."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = _Chat(client)


class _Chat:
    def __init__(self, client: Any) -> None:
        self.completions = _Completions(client)


class _Completions:
    def __init__(self, client: Any) -> None:
        self._client = client

    def create(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        system_msg, anthropic_messages = _convert_messages(messages)
        anthropic_tools = _convert_tools(tools)

        req: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": 8096,
        }
        if system_msg:
            req["system"] = system_msg
        if anthropic_tools:
            req["tools"] = anthropic_tools

        response = self._client.messages.create(**req)
        return _convert_response(response)


def _convert_messages(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Split OpenAI messages into (system_content, anthropic_messages)."""
    system_msg: str | None = None
    out: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            system_msg = msg.get("content") or ""
            continue

        if role == "tool":
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": str(msg.get("content", "")),
                }],
            })
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content: list[dict] = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", "{}")
                try:
                    parsed_args = json.loads(raw_args)
                except (ValueError, TypeError):
                    parsed_args = {}
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": parsed_args,
                })
            out.append({"role": "assistant", "content": content})
            continue

        out.append({"role": role, "content": msg.get("content", "")})

    return system_msg, out


def _convert_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    result = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool["function"]
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result or None


def _convert_response(response: Any) -> Any:
    """Convert an Anthropic MessageResponse to an OpenAI ChatCompletion-like object."""
    text_content = ""
    tool_calls: list[Any] = []

    for block in response.content:
        if block.type == "text":
            text_content += block.text
        elif block.type == "tool_use":
            tool_calls.append(SimpleNamespace(
                id=block.id,
                type="function",
                function=SimpleNamespace(
                    name=block.name,
                    arguments=json.dumps(block.input),
                ),
                model_dump=lambda b=block: {
                    "id": b.id,
                    "type": "function",
                    "function": {
                        "name": b.name,
                        "arguments": json.dumps(b.input),
                    },
                },
            ))

    finish_reason = "tool_calls" if response.stop_reason == "tool_use" else "stop"

    message = SimpleNamespace(
        role="assistant",
        content=text_content or None,
        tool_calls=tool_calls if tool_calls else None,
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


__all__ = [
    "ProviderConfig",
    "PROVIDER_PRESETS",
    "build_client",
    "configured_providers",
    "detect_provider_from_env",
    "provider_from_env",
]
