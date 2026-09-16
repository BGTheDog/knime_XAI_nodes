"""Grok LLM Prompter node."""

from __future__ import annotations

import logging

import knime.extension as knext
import pandas as pd

from ports import (
    GROK_ICON,
    GrokChatModelPortObject,
    GrokChatModelPortObjectSpec,
    grok_category,
    grok_chat_port_type,
)

LOGGER = logging.getLogger(__name__)


def _is_string_column(column: knext.Column) -> bool:
    return column.ktype == knext.string()


def _unique_column(existing: list[str], name: str) -> str:
    if name not in existing:
        return name
    index = 1
    while f"{name} ({index})" in existing:
        index += 1
    return f"{name} ({index})"


def _first_string_column(schema: knext.Schema) -> str:
    for column in schema:
        if _is_string_column(column):
            return column.name
    raise knext.InvalidParametersError("The prompt table must contain a string column.")


@knext.node(
    name="Grok LLM Prompter",
    node_type=knext.NodeType.PREDICTOR,
    icon_path=GROK_ICON,
    category=grok_category,
    keywords=["Grok", "xAI", "GenAI", "LLM", "prompt"],
)
@knext.input_port(
    "Grok Chat Model",
    "A Grok chat model from Grok Chat Model Selector.",
    grok_chat_port_type,
)
@knext.input_table(
    "Prompt Table",
    "A table containing a string column of prompts. Each row is sent independently.",
)
@knext.output_table(
    "Result Table",
    "The input table with a new column of Grok responses.",
)
class GrokLLMPrompter:
    """Prompt a Grok model once per input row.

    Each row is treated as its own conversation. The model does not remember
    previous rows. Connect a **Grok Chat Model** and a table of prompts.
    """

    system_message = knext.MultilineStringParameter(
        "System message",
        "Optional instructions describing how the model should behave. Leave empty to omit.",
        default_value="",
    )

    prompt_column = knext.ColumnParameter(
        "Prompt column",
        "Column containing the user prompt for each row.",
        port_index=1,
        column_filter=_is_string_column,
    )

    response_column_name = knext.StringParameter(
        "Response column name",
        "Name of the column that will hold Grok's responses.",
        default_value="Response",
    )

    fail_on_missing = knext.BoolParameter(
        "Fail on missing prompts",
        "If enabled, execution fails when the prompt column contains missing or empty values. "
        "If disabled, those rows get an empty response.",
        default_value=True,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        model_spec: GrokChatModelPortObjectSpec,
        input_schema: knext.Schema,
    ) -> knext.Schema:
        model_spec.validate_context(ctx)
        if not self.prompt_column:
            self.prompt_column = _first_string_column(input_schema)
        elif self.prompt_column not in input_schema.column_names:
            raise knext.InvalidParametersError(
                f"Prompt column '{self.prompt_column}' is not in the input table."
            )
        if not self.response_column_name:
            raise knext.InvalidParametersError("The response column name must not be empty.")
        output_name = _unique_column(list(input_schema.column_names), self.response_column_name)
        return input_schema.append(knext.Column(knext.string(), output_name))

    def execute(
        self,
        ctx: knext.ExecutionContext,
        model: GrokChatModelPortObject,
        input_table: knext.Table,
    ) -> knext.Table:
        frame = input_table.to_pandas()
        prompt_column = self.prompt_column or _first_string_column(input_table.schema)
        output_name = _unique_column(list(frame.columns), self.response_column_name)
        client = model.create_client(ctx)
        spec = model.spec
        responses: list[str] = []
        total = max(len(frame), 1)

        for index, value in enumerate(frame[prompt_column].tolist()):
            if ctx.is_canceled():
                raise RuntimeError("Execution canceled.")
            ctx.set_progress((index + 1) / total, f"Prompting row {index + 1} of {len(frame)}")
            prompt = "" if pd.isna(value) else str(value)
            if not prompt.strip():
                if self.fail_on_missing:
                    raise RuntimeError(f"Missing or empty prompt in row {index + 1}.")
                responses.append("")
                continue
            messages = []
            if self.system_message and self.system_message.strip():
                messages.append({"role": "system", "content": self.system_message})
            messages.append({"role": "user", "content": prompt})
            result = client.chat(
                model=spec.model,
                messages=messages,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
            )
            responses.append(result.content or "")

        frame[output_name] = responses
        return knext.Table.from_pandas(frame)
