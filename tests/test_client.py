from unittest.mock import MagicMock, patch

from client import ChatResponse, DEFAULT_IMAGE_MODELS, GrokClient, ToolCall


def test_list_models_filters_grok():
    listed = MagicMock()
    listed.data = [MagicMock(id="grok-4.6"), MagicMock(id="other-model"), MagicMock(id="grok-4.5")]
    with patch("client._openai_client") as factory:
        factory.return_value.models.list.return_value = listed
        models = GrokClient("key").list_models()
    assert models == ["grok-4.5", "grok-4.6"]


def test_list_image_models_includes_defaults():
    listed = MagicMock()
    listed.data = [
        MagicMock(id="grok-4.6"),
        MagicMock(id="grok-imagine-image-2.0"),
        MagicMock(id="grok-imagine-video"),
    ]
    with patch("client._openai_client") as factory:
        factory.return_value.models.list.return_value = listed
        models = GrokClient("key").list_image_models()
    assert "grok-imagine-image-2.0" in models
    assert "grok-imagine-image" in models
    assert "grok-4.6" not in models
    assert set(DEFAULT_IMAGE_MODELS).issubset(models)


def test_list_models_excludes_imagine():
    listed = MagicMock()
    listed.data = [
        MagicMock(id="grok-4.6"),
        MagicMock(id="grok-imagine-image-2.0"),
        MagicMock(id="other-model"),
    ]
    with patch("client._openai_client") as factory:
        factory.return_value.models.list.return_value = listed
        models = GrokClient("key").list_models()
    assert models == ["grok-4.6"]


def test_generate_image_decodes_png():
    png = b"\x89PNG\r\n\x1a\n" + b"fake"
    item = MagicMock(b64_json=__import__("base64").b64encode(png).decode(), url=None)
    completion = MagicMock(data=[item])
    with patch("client._openai_client") as factory:
        factory.return_value.images.generate.return_value = completion
        result = GrokClient("key").generate_image("grok-imagine-image-2.0", "a cat")
    assert result.startswith(b"\x89PNG")


def test_chat_parses_tool_calls():
    function = MagicMock()
    function.name = "emit_table"
    function.arguments = '{"port_index": 0}'
    call = MagicMock(id="1", function=function)
    message = MagicMock(content=None, tool_calls=[call])
    choice = MagicMock(message=message, finish_reason="tool_calls")
    completion = MagicMock(choices=[choice])
    with patch("client._openai_client") as factory:
        factory.return_value.chat.completions.create.return_value = completion
        result = GrokClient("key").chat("grok-4.6", [{"role": "user", "content": "hi"}])
    assert isinstance(result, ChatResponse)
    assert result.tool_calls == [ToolCall(id="1", name="emit_table", arguments='{"port_index": 0}')]
