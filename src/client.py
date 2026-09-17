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
DEFAULT_IMAGE_MODELS = [
    "grok-imagine-image-2.0",
    "grok-imagine-image",
]


def _is_image_model(model_id: str) -> bool:
    return model_id.startswith("grok-imagine-image")


def _is_chat_model(model_id: str) -> bool:
    return model_id.startswith("grok-") and not model_id.startswith("grok-imagine-")


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
        grok = sorted({m for m in models if _is_chat_model(m)})
        return grok or list(DEFAULT_MODELS)

    def list_image_models(self) -> list[str]:
        try:
            models = [item.id for item in self._client.models.list().data]
            found = {m for m in models if _is_image_model(m)}
        except Exception:
            found = set()
        return sorted(found | set(DEFAULT_IMAGE_MODELS))

    def generate_image(
        self,
        model: str,
        prompt: str,
        aspect_ratio: str = "auto",
        quality: str = "auto",
        resolution: str = "1k",
    ) -> bytes:
        extra: dict[str, Any] = {}
        if aspect_ratio and aspect_ratio != "auto":
            extra["aspect_ratio"] = aspect_ratio
        if quality and quality != "auto":
            extra["quality"] = quality
        if resolution:
            extra["resolution"] = resolution
        kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }
        if extra:
            kwargs["extra_body"] = extra
        try:
            response = self._client.images.generate(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            response = self._client.images.generate(**kwargs)
        item = response.data[0]
        raw = self._image_bytes(item)
        return _to_png(raw)

    def _image_bytes(self, item: Any) -> bytes:
        import base64
        import urllib.request

        b64 = getattr(item, "b64_json", None)
        if b64:
            return base64.b64decode(b64)
        url = getattr(item, "url", None)
        if not url:
            raise RuntimeError("The xAI image API returned neither image bytes nor a URL.")
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

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


def _to_png(raw: bytes) -> bytes:
    if raw.startswith(b"\x89PNG"):
        return raw
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(raw))
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
