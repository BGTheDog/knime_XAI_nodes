"""Minimal xAI Grok API client (OpenAI-compatible chat completions)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODELS = [
    "grok-4.6",
    "grok-4.5",
    "grok-4.3",
    "grok-4",
    "grok-build-0.1",
]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ChatResponse:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


def _openai_client(api_key: str, base_url: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


class GrokClient:
    """Thin wrapper around the OpenAI-compatible xAI HTTP API."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL):
        if not api_key:
            raise ValueError("An xAI API key is required.")
        if not base_url:
            raise ValueError("A base URL is required.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = _openai_client(api_key, self._base_url)

    def list_models(self) -> list[str]:
        models = [item.id for item in self._client.models.list().data]
        grok = sorted({m for m in models if m.startswith("grok-")})
        return grok or sorted(models)

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            function = getattr(call, "function", None)
            if function is None:
                continue
            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=function.name,
                    arguments=function.arguments or "{}",
                )
            )
        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )
