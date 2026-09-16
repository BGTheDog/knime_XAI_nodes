"""Grok Agent node with prompt table input and dynamic data ports."""

from __future__ import annotations

import json
import logging
from typing import Any

import knime.extension as knext
import pandas as pd

from client import ChatResponse, GrokClient
from ports import (
    GROK_ICON,
    GrokChatModelPortObject,
    GrokChatModelPortObjectSpec,
    grok_category,
    grok_chat_port_type,
)
from prompter import _first_string_column, _is_string_column

LOGGER = logging.getLogger(__name__)

_INSPECT_TOOL = "inspect_input_tables"
_QUERY_TOOL = "query_input_table"
_EMIT_TOOL = "emit_table"

_BUILTIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": _INSPECT_TOOL,
            "description": (
                "List connected data input tables with port index, column names, "
                "column types, and row counts. Call this before querying or emitting data."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": _QUERY_TOOL,
            "description": (
                "Read rows from a connected data input table. Use port_index from "
                "inspect_input_tables. Optionally choose columns and a maximum number of rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "port_index": {
                        "type": "integer",
                        "description": "0-based index of a data input port.",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column names to keep. Omit to return all columns.",
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": "Maximum rows to return (default 50, max 1000).",
                    },
                },
                "required": ["port_index"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": _EMIT_TOOL,
            "description": (
                "Write a table to a data output port so downstream KNIME nodes can use it. "
                "port_index is 0-based among the Data outputs the user added on this node. "
                "rows is a list of objects whose keys become column names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "port_index": {
                        "type": "integer",
                        "description": "0-based index of a data output port.",
                    },
                    "rows": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Table rows as JSON objects.",
                    },
                },
                "required": ["port_index", "rows"],
                "additionalProperties": False,
            },
        },
    },
]


def _empty_table() -> knext.Table:
    return knext.Table.from_pandas(pd.DataFrame({"_empty": pd.Series(dtype="string")}))


def _conversation_schema() -> knext.Schema:
    return knext.Schema.from_columns(
        [
            knext.Column(knext.string(), "Role"),
            knext.Column(knext.string(), "Content"),
        ]
    )


@knext.node(
    name="Grok Agent",
    node_type=knext.NodeType.PREDICTOR,
    icon_path=GROK_ICON,
    category=grok_category,
    keywords=["Grok", "xAI", "GenAI", "agent", "tools"],
)
@knext.input_port(
    "Grok Chat Model",
    "A Grok chat model from Grok Chat Model Selector.",
    grok_chat_port_type,
)
@knext.input_table(
    "Prompt",
    "Table with a string prompt column. The first non-empty row is used as the user message.",
)
@knext.input_table_group(
    "Data inputs",
    "Optional tables the agent can inspect and query.",
)
@knext.output_table(
    "Conversation",
    "The messages exchanged between Grok and its tools.",
)
@knext.output_table_group(
    "Data outputs",
    "Tables the agent emitted. Add ports with the node's Add output port control, then ask for that data in the prompt.",
)
class GrokAgent:
    """Run a Grok agent with a prompt table and optional extra data tables.

    Add **Data outputs** on the node (⋯ → Add output port) when you want the
    agent to return tables. Tell Grok in the prompt how many tables to emit.
    The agent can inspect **Data inputs** and write rows to those output ports
    via built-in tools.

    This node does not connect to KNIME's LLM Prompter or Agent Prompter.
    Use it with **Grok Authenticator** and **Grok Chat Model Selector**.
    """

    system_message = knext.MultilineStringParameter(
        "System message",
        "Optional extra instructions prepended to the agent system prompt.",
        default_value="",
    )

    prompt_column = knext.ColumnParameter(
        "Prompt column",
        "Column containing the user prompt. The first non-empty value is used.",
        port_index=1,
        column_filter=_is_string_column,
    )

    iteration_limit = knext.IntParameter(
        "Iteration limit",
        "Maximum model/tool rounds before the agent must stop.",
        default_value=12,
        min_value=1,
        max_value=50,
    )

    enable_web_search = knext.BoolParameter(
        "Enable web search",
        "Let Grok use xAI web search as a server-side tool.",
        default_value=False,
        is_advanced=True,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        model_spec: GrokChatModelPortObjectSpec,
        prompt_schema: knext.Schema,
        data_schemas: list[knext.Schema],
    ):
        model_spec.validate_context(ctx)
        if not self.prompt_column:
            self.prompt_column = _first_string_column(prompt_schema)
        elif self.prompt_column not in prompt_schema.column_names:
            raise knext.InvalidParametersError(
                f"Prompt column '{self.prompt_column}' is not in the prompt table."
            )
        data_out_count = ctx.get_connected_output_port_numbers()[1]
        return _conversation_schema(), [None] * data_out_count

    def execute(
        self,
        ctx: knext.ExecutionContext,
        model: GrokChatModelPortObject,
        prompt_table: knext.Table,
        data_tables: list[knext.Table],
    ):
        prompt = self._read_prompt(prompt_table)
        client = model.create_client(ctx)
        data_frames = [table.to_pandas() for table in data_tables]
        data_out_count = ctx.get_connected_output_port_numbers()[1]
        emitted: dict[int, pd.DataFrame] = {}
        messages = [
            {"role": "system", "content": self._system_prompt(len(data_frames), data_out_count)},
            {"role": "user", "content": prompt},
        ]
        tools = list(_BUILTIN_TOOLS)
        if self.enable_web_search:
            tools = [{"type": "web_search"}] + tools

        conversation_rows = [
            {"Role": "system", "Content": messages[0]["content"]},
            {"Role": "user", "Content": prompt},
        ]

        for round_index in range(self.iteration_limit):
            if ctx.is_canceled():
                raise RuntimeError("Execution canceled.")
            ctx.set_progress(
                (round_index + 1) / self.iteration_limit,
                f"Agent round {round_index + 1}",
            )
            response = self._chat(client, model.spec, messages, tools)
            if response.content:
                conversation_rows.append({"Role": "assistant", "Content": response.content})
            if not response.tool_calls:
                if response.content:
                    messages.append({"role": "assistant", "content": response.content})
                break
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in response.tool_calls
                ],
            }
            messages.append(assistant_message)
            conversation_rows.append(
                {
                    "Role": "assistant",
                    "Content": response.content
                    or json.dumps([call.name for call in response.tool_calls]),
                }
            )
            for call in response.tool_calls:
                result = self._run_tool(call.name, call.arguments, data_frames, emitted, data_out_count)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
                conversation_rows.append({"Role": "tool", "Content": result})
        else:
            ctx.set_warning("The agent reached the iteration limit.")

        conversation = knext.Table.from_pandas(pd.DataFrame(conversation_rows))
        outputs = []
        for port_index in range(data_out_count):
            frame = emitted.get(port_index)
            outputs.append(
                knext.Table.from_pandas(frame) if frame is not None else _empty_table()
            )
        return conversation, outputs

    def _read_prompt(self, prompt_table: knext.Table) -> str:
        frame = prompt_table.to_pandas()
        column = self.prompt_column or _first_string_column(prompt_table.schema)
        for value in frame[column].tolist():
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                return text
        raise RuntimeError("The prompt table has no non-empty prompt.")

    def _system_prompt(self, n_inputs: int, n_outputs: int) -> str:
        extra = self.system_message.strip()
        parts = [
            "You are a Grok agent running inside a KNIME workflow.",
            f"There are {n_inputs} data input table(s) and {n_outputs} data output port(s).",
            "Use inspect_input_tables and query_input_table to read input data.",
        ]
        if n_outputs:
            parts.append(
                "When the user asks for tabular results, call emit_table for each requested "
                f"output (port_index 0 through {n_outputs - 1}) before your final answer."
            )
        else:
            parts.append(
                "There are no data output ports. Answer in text. If tabular output is needed, "
                "tell the user to add Data outputs on this node."
            )
        if extra:
            parts.append(extra)
        return "\n".join(parts)

    def _chat(
        self,
        client: GrokClient,
        spec: GrokChatModelPortObjectSpec,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatResponse:
        try:
            return client.chat(
                model=spec.model,
                messages=messages,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
                tools=tools,
            )
        except Exception:
            if self.enable_web_search:
                LOGGER.warning("Chat with web search failed; retrying without it.")
                function_tools = [tool for tool in tools if tool.get("type") == "function"]
                return client.chat(
                    model=spec.model,
                    messages=messages,
                    temperature=spec.temperature,
                    max_tokens=spec.max_tokens,
                    tools=function_tools,
                )
            raise

    def _run_tool(
        self,
        name: str,
        arguments: str,
        data_frames: list[pd.DataFrame],
        emitted: dict[int, pd.DataFrame],
        data_out_count: int,
    ) -> str:
        try:
            payload = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return "Invalid JSON arguments."
        if name == _INSPECT_TOOL:
            return self._inspect(data_frames)
        if name == _QUERY_TOOL:
            return self._query(payload, data_frames)
        if name == _EMIT_TOOL:
            return self._emit(payload, emitted, data_out_count)
        return f"Unknown tool '{name}'."

    def _inspect(self, data_frames: list[pd.DataFrame]) -> str:
        summary = []
        for index, frame in enumerate(data_frames):
            summary.append(
                {
                    "port_index": index,
                    "rows": int(len(frame)),
                    "columns": [
                        {"name": str(col), "dtype": str(frame[col].dtype)}
                        for col in frame.columns
                    ],
                }
            )
        if not summary:
            return "No data input tables are connected."
        return json.dumps(summary)

    def _query(self, payload: dict[str, Any], data_frames: list[pd.DataFrame]) -> str:
        try:
            port_index = int(payload["port_index"])
        except (KeyError, TypeError, ValueError):
            return "port_index is required."
        if port_index < 0 or port_index >= len(data_frames):
            return f"port_index {port_index} is out of range (0..{len(data_frames) - 1})."
        frame = data_frames[port_index]
        columns = payload.get("columns")
        if columns:
            missing = [name for name in columns if name not in frame.columns]
            if missing:
                return f"Unknown columns: {missing}"
            frame = frame[list(columns)]
        max_rows = payload.get("max_rows", 50)
        try:
            max_rows = max(1, min(int(max_rows), 1000))
        except (TypeError, ValueError):
            max_rows = 50
        sample = frame.head(max_rows)
        return sample.to_json(orient="records")

    def _emit(
        self,
        payload: dict[str, Any],
        emitted: dict[int, pd.DataFrame],
        data_out_count: int,
    ) -> str:
        if data_out_count == 0:
            return "No data output ports. Add Data outputs on this node, then retry."
        try:
            port_index = int(payload["port_index"])
        except (KeyError, TypeError, ValueError):
            return "port_index is required."
        if port_index < 0 or port_index >= data_out_count:
            return f"port_index {port_index} is out of range (0..{data_out_count - 1})."
        rows = payload.get("rows")
        if not isinstance(rows, list):
            return "rows must be a list of objects."
        emitted[port_index] = pd.DataFrame(rows)
        return f"Wrote {len(rows)} row(s) to data output {port_index}."
