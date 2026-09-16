from unittest.mock import MagicMock, patch

from client import ChatResponse, GrokClient, ToolCall


def test_list_models_filters_grok():
    listed = MagicMock()
    listed.data = [MagicMock(id="grok-4.6"), MagicMock(id="other-model"), MagicMock(id="grok-4.5")]
    with patch("client._openai_client") as factory:
        factory.return_value.models.list.return_value = listed
        models = GrokClient("key").list_models()
    assert models == ["grok-4.5", "grok-4.6"]


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
