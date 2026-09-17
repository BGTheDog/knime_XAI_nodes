"""Grok Chat Model Selector node."""

from __future__ import annotations

import knime.extension as knext

from client import DEFAULT_MODELS
from ports import (
    GROK_ICON,
    NODE_ATTRIBUTION,
    GrokAuthenticationPortObject,
    GrokAuthenticationPortObjectSpec,
    GrokChatModelPortObject,
    GrokChatModelPortObjectSpec,
    grok_auth_port_type,
    grok_category,
    grok_chat_port_type,
)


def _list_models(ctx: knext.ConfigurationContext):
    specs = ctx.get_input_specs()
    if specs and specs[0]:
        return specs[0].get_model_list(ctx)
    return list(DEFAULT_MODELS)


@knext.node(
    name="Grok Chat Model Selector",
    node_type=knext.NodeType.SOURCE,
    icon_path=GROK_ICON,
    category=grok_category,
    keywords=["Grok", "xAI", "GenAI", "LLM"],
)
@knext.input_port(
    "Grok Authentication",
    "Authentication from the Grok Authenticator.",
    grok_auth_port_type,
)
@knext.output_port(
    "Grok Chat Model",
    "A configured Grok chat model for the Grok LLM Prompter and Grok Agent nodes.",
    grok_chat_port_type,
)
class GrokChatModelSelector:
    f"""Select a Grok chat model.

    After authenticating with **Grok Authenticator**, pick a model and sampling
    settings. The output port connects to **Grok LLM Prompter** or **Grok Agent**.

    {NODE_ATTRIBUTION}
    """

    model = knext.StringParameter(
        "Model",
        "Grok model id. The list is fetched from the xAI API when possible.",
        default_value="grok-4.6",
        choices=_list_models,
    )

    temperature = knext.DoubleParameter(
        "Temperature",
        "Sampling temperature between 0.0 and 2.0. Higher values are more random.",
        default_value=0.7,
        min_value=0.0,
        max_value=2.0,
    )

    max_tokens = knext.IntParameter(
        "Max tokens",
        "Maximum number of tokens to generate in the response.",
        default_value=4096,
        min_value=1,
    )

    def create_spec(
        self, auth: GrokAuthenticationPortObjectSpec
    ) -> GrokChatModelPortObjectSpec:
        return GrokChatModelPortObjectSpec(
            auth=auth,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        auth: GrokAuthenticationPortObjectSpec,
    ) -> GrokChatModelPortObjectSpec:
        auth.validate_context(ctx)
        if not self.model:
            raise knext.InvalidParametersError("Select a Grok model.")
        return self.create_spec(auth)

    def execute(
        self, ctx: knext.ExecutionContext, auth: GrokAuthenticationPortObject
    ) -> GrokChatModelPortObject:
        return GrokChatModelPortObject(self.create_spec(auth.spec))
