"""Grok Authenticator node."""

from __future__ import annotations

import knime.extension as knext

from client import DEFAULT_BASE_URL
from ports import (
    GROK_ICON,
    NODE_ATTRIBUTION,
    CredentialsSettings,
    GrokAuthenticationPortObject,
    GrokAuthenticationPortObjectSpec,
    grok_auth_port_type,
    grok_category,
)


@knext.node(
    name="Grok Authenticator",
    node_type=knext.NodeType.SOURCE,
    icon_path=GROK_ICON,
    category=grok_category,
    keywords=["Grok", "xAI", "GenAI", "LLM"],
)
@knext.output_port(
    "Grok Authentication",
    "Validated authentication for the xAI Grok API.",
    grok_auth_port_type,
)
class GrokAuthenticator:
    f"""Authenticates with the xAI API using an API key.

    Paste an xAI API key from [console.x.ai](https://console.x.ai), or (under
    Advanced settings) select workflow credentials whose *password* field holds
    the key. Connect this node to **Grok Chat Model Selector**.

    {NODE_ATTRIBUTION}
    """

    api_key = knext.StringParameter(
        "xAI API key",
        "Paste your xAI API key. Leave empty if you select credentials under Advanced settings instead.",
        default_value="",
    )

    credentials_settings = CredentialsSettings(
        label="Credentials",
        description="Optional. Credentials whose *password* field contains the xAI API key (username is ignored). "
        "If selected, this overrides the pasted API key.",
    )

    base_url = knext.StringParameter(
        "Base URL",
        "The xAI API base URL.",
        default_value=DEFAULT_BASE_URL,
        is_advanced=True,
    )

    validate_api_key = knext.BoolParameter(
        "Validate API key",
        "If set, the API key is checked during execution by listing available models.",
        True,
        is_advanced=True,
    )

    def create_spec(self) -> GrokAuthenticationPortObjectSpec:
        return GrokAuthenticationPortObjectSpec(
            credentials=self.credentials_settings.credentials_param or "",
            base_url=self.base_url,
            api_key=self.api_key or "",
        )

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> GrokAuthenticationPortObjectSpec:
        spec = self.create_spec()
        spec.validate_context(ctx)
        return spec

    def execute(self, ctx: knext.ExecutionContext) -> GrokAuthenticationPortObject:
        spec = self.create_spec()
        if self.validate_api_key:
            spec.validate_api_key(ctx)
        return GrokAuthenticationPortObject(spec)
