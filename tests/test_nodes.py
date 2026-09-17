import json
from unittest.mock import patch

import knime.extension as knext
import pandas as pd
from pandas.testing import assert_frame_equal

from agent import GrokAgent
from auth import GrokAuthenticator
from client import ChatResponse, ToolCall
from image_generator import GrokImageGenerator
from ports import (
    GrokAuthenticationPortObject,
    GrokAuthenticationPortObjectSpec,
    GrokChatModelPortObject,
    GrokChatModelPortObjectSpec,
)
from prompter import GrokLLMPrompter
from selector import GrokChatModelSelector
from tests.conftest import FakeContext


def _auth_spec(name: str = "xai") -> GrokAuthenticationPortObjectSpec:
    return GrokAuthenticationPortObjectSpec(name)


def _chat_spec(name: str = "xai") -> GrokChatModelPortObjectSpec:
    return GrokChatModelPortObjectSpec(
        auth=_auth_spec(name),
        model="grok-4.6",
        temperature=0.2,
        max_tokens=128,
    )


def _prompt_schema() -> knext.Schema:
    return knext.Schema.from_columns([knext.Column(knext.string(), "Prompt")])


class TestAuthenticator:
    def test_configure_ok(self):
        node = GrokAuthenticator()
        node.credentials_settings.credentials_param = "xai"
        spec = node.configure(FakeContext())
        assert spec.credentials == "xai"
        assert spec.base_url.endswith("/v1")

    def test_configure_with_pasted_api_key(self):
        node = GrokAuthenticator()
        node.api_key = "xai-test-key"
        spec = node.configure(FakeContext())
        assert spec.api_key(FakeContext()) == "xai-test-key"

    def test_configure_missing_credentials(self):
        node = GrokAuthenticator()
        try:
            node.configure(FakeContext())
            raise AssertionError("expected InvalidParametersError")
        except knext.InvalidParametersError:
            pass

    def test_execute_skips_validation(self):
        node = GrokAuthenticator()
        node.credentials_settings.credentials_param = "xai"
        node.validate_api_key = False
        port = node.execute(FakeContext())
        assert isinstance(port, GrokAuthenticationPortObject)
        assert port.spec.credentials == "xai"


class TestSelector:
    def test_configure_and_execute(self):
        node = GrokChatModelSelector()
        node.model = "grok-4.6"
        node.temperature = 0.1
        node.max_tokens = 64
        ctx = FakeContext()
        spec = node.configure(ctx, _auth_spec())
        assert spec.model == "grok-4.6"
        port = node.execute(ctx, GrokAuthenticationPortObject(_auth_spec()))
        assert isinstance(port, GrokChatModelPortObject)
        assert port.spec.temperature == 0.1


class TestPrompter:
    def test_configure_adds_response_column(self):
        node = GrokLLMPrompter()
        node.prompt_column = "Prompt"
        node.response_column_name = "Response"
        schema = node.configure(FakeContext(), _chat_spec(), _prompt_schema())
        assert "Response" in schema.column_names

    def test_execute_appends_responses(self):
        node = GrokLLMPrompter()
        node.prompt_column = "Prompt"
        node.response_column_name = "Response"
        table = knext.Table.from_pandas(pd.DataFrame({"Prompt": ["Hello", "There"]}))
        fake = ChatResponse(content="ok")
        with patch("ports.GrokClient") as client_cls:
            client_cls.return_value.chat.return_value = fake
            output = node.execute(
                FakeContext(),
                GrokChatModelPortObject(_chat_spec()),
                table,
            )
        frame = output.to_pandas()
        assert list(frame["Response"]) == ["ok", "ok"]
        assert client_cls.return_value.chat.call_count == 2


class TestAgent:
    def test_configure_with_data_outputs(self):
        node = GrokAgent()
        node.prompt_column = "Prompt"
        conversation_schema, data_schemas = node.configure(
            FakeContext(output_ports=[1, 2]),
            _chat_spec(),
            _prompt_schema(),
            [],
        )
        assert "Role" in conversation_schema.column_names
        assert data_schemas == [None, None]

    def test_execute_emits_table(self):
        node = GrokAgent()
        node.prompt_column = "Prompt"
        node.iteration_limit = 4
        prompt = knext.Table.from_pandas(
            pd.DataFrame({"Prompt": ["Emit one row of fruit data"]})
        )
        emit_call = ToolCall(
            id="call_1",
            name="emit_table",
            arguments=json.dumps(
                {"port_index": 0, "rows": [{"fruit": "apple", "n": 2}]}
            ),
        )
        chats = [
            ChatResponse(content=None, tool_calls=[emit_call]),
            ChatResponse(content="Done."),
        ]

        class FakeGrok:
            def __init__(self, *args, **kwargs):
                pass

            def chat(self, **kwargs):
                return chats.pop(0)

        with patch("ports.GrokClient", FakeGrok):
            conversation, outputs = node.execute(
                FakeContext(output_ports=[1, 1]),
                GrokChatModelPortObject(_chat_spec()),
                prompt,
                [],
            )
        conv = conversation.to_pandas()
        assert "Done." in set(conv["Content"])
        assert len(outputs) == 1
        emitted = outputs[0].to_pandas()
        assert_frame_equal(
            emitted.reset_index(drop=True),
            pd.DataFrame({"fruit": ["apple"], "n": [2]}),
            check_dtype=False,
        )


class TestImageGenerator:
    def test_configure_with_prompt(self):
        node = GrokImageGenerator()
        node.model = "grok-imagine-image-2.0"
        node.prompt = "A lighthouse at dawn"
        spec, table_spec = node.configure(FakeContext(), _auth_spec(), None)
        assert spec.format in (knext.ImageFormat.PNG, knext.ImageFormat.PNG.value, "png")
        assert list(table_spec.column_names) == ["Prompt", "Image"]

    def test_configure_requires_prompt(self):
        node = GrokImageGenerator()
        node.model = "grok-imagine-image-2.0"
        node.prompt = ""
        try:
            node.configure(FakeContext(), _auth_spec(), None)
            raise AssertionError("expected InvalidParametersError")
        except knext.InvalidParametersError:
            pass

    def test_execute_returns_png(self):
        node = GrokImageGenerator()
        node.model = "grok-imagine-image-2.0"
        node.prompt = "A cat"
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        class FakeGrok:
            def __init__(self, *args, **kwargs):
                pass

            def generate_image(self, **kwargs):
                assert kwargs["model"] == "grok-imagine-image-2.0"
                assert kwargs["prompt"] == "A cat"
                return png

        with patch("image_generator.GrokClient", FakeGrok):
            result, table = node.execute(
                FakeContext(),
                GrokAuthenticationPortObject(_auth_spec()),
                None,
            )
        assert result == png
        assert "Image" in table.to_pandas().columns
