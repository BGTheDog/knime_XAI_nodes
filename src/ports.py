"""Custom Grok port types used only by nodes in this extension."""

from __future__ import annotations

import knime.extension as knext

from client import DEFAULT_BASE_URL, DEFAULT_IMAGE_MODELS, DEFAULT_MODELS, GrokClient

GROK_ICON = "icons/grok.png"
VENDOR_NAME = "Whozinzahs Holdings, LLC"
VENDOR_SITE = "https://Whozinzahs.com"
NODE_ATTRIBUTION = (
    f"Provided by {VENDOR_NAME}. [{VENDOR_SITE.removeprefix('https://')}]({VENDOR_SITE})"
)

grok_category = knext.category(
    path="/",
    level_id="grok",
    name="Grok",
    description="xAI Grok models. Provided by Whozinzahs Holdings, LLC (Whozinzahs.com).",
    icon=GROK_ICON,
)


def _register_port(name: str, object_class: type, spec_class: type) -> knext.PortType:
    port = knext.port_type(name, object_class, spec_class)
    if port is None:
        from knime.extension.nodes import PortType

        port = PortType(
            id=f"{object_class.__module__}.{object_class.__name__}",
            name=name,
            object_class=object_class,
            spec_class=spec_class,
        )
    return port


@knext.parameter_group(label="Credentials", is_advanced=True)
class CredentialsSettings:
    def __init__(self, label: str, description: str):
        self.credentials_param = knext.StringParameter(
            label=label,
            description=description,
            choices=lambda ctx: knext.DialogCreationContext.get_credential_names(ctx),
        )


class GrokAuthenticationPortObjectSpec(knext.PortObjectSpec):
    def __init__(
        self,
        credentials: str = "",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
    ) -> None:
        self._credentials = credentials or ""
        self._base_url = base_url
        self._api_key = api_key or ""

    @property
    def credentials(self) -> str:
        return self._credentials

    @property
    def base_url(self) -> str:
        return self._base_url

    def serialize(self) -> dict:
        return {
            "credentials": self._credentials,
            "base_url": self._base_url,
            "api_key": self._api_key,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "GrokAuthenticationPortObjectSpec":
        return cls(
            data.get("credentials", ""),
            data.get("base_url", DEFAULT_BASE_URL),
            data.get("api_key", ""),
        )

    def validate_context(self, ctx: knext.ConfigurationContext) -> None:
        if not self.base_url:
            raise knext.InvalidParametersError("Please provide a base URL.")
        if self._credentials:
            names = ctx.get_credential_names()
            if self._credentials not in names:
                raise knext.InvalidParametersError(
                    f"The selected credentials '{self._credentials}' do not exist. "
                    "Select credentials that are still available on this workflow, or paste an API key."
                )
            token = ctx.get_credentials(self._credentials)
            if not token.password:
                raise knext.InvalidParametersError(
                    f"The xAI API key in '{self._credentials}' is empty. "
                    "Put the key in the password field of the Credentials Configuration node."
                )
            return
        if not self._api_key.strip():
            raise knext.InvalidParametersError(
                "Enter an xAI API key, or select credentials whose password field contains the key."
            )

    def api_key(self, ctx) -> str:
        if self._credentials:
            return ctx.get_credentials(self._credentials).password
        return self._api_key

    def get_model_list(self, ctx: knext.ConfigurationContext) -> list[str]:
        try:
            return GrokClient(self.api_key(ctx), self.base_url).list_models()
        except Exception:
            return list(DEFAULT_MODELS)

    def get_image_model_list(self, ctx: knext.ConfigurationContext) -> list[str]:
        try:
            return GrokClient(self.api_key(ctx), self.base_url).list_image_models()
        except Exception:
            return list(DEFAULT_IMAGE_MODELS)

    def validate_api_key(self, ctx: knext.ExecutionContext) -> None:
        try:
            GrokClient(self.api_key(ctx), self.base_url).list_models()
        except Exception as exc:
            raise RuntimeError("Could not authenticate with the xAI API.") from exc


class GrokAuthenticationPortObject(knext.PortObject):
    def __init__(self, spec: GrokAuthenticationPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> GrokAuthenticationPortObjectSpec:
        return super().spec

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: GrokAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)


grok_auth_port_type = _register_port(
    "Grok Authentication",
    GrokAuthenticationPortObject,
    GrokAuthenticationPortObjectSpec,
)


class GrokChatModelPortObjectSpec(knext.PortObjectSpec):
    def __init__(
        self,
        auth: GrokAuthenticationPortObjectSpec,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._auth = auth
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def auth(self) -> GrokAuthenticationPortObjectSpec:
        return self._auth

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def validate_context(self, ctx: knext.ConfigurationContext) -> None:
        self.auth.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "auth": self._auth.serialize(),
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "GrokChatModelPortObjectSpec":
        return cls(
            auth=GrokAuthenticationPortObjectSpec.deserialize(data["auth"]),
            model=data["model"],
            temperature=data["temperature"],
            max_tokens=data["max_tokens"],
        )


class GrokChatModelPortObject(knext.PortObject):
    def __init__(self, spec: GrokChatModelPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> GrokChatModelPortObjectSpec:
        return super().spec

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: GrokChatModelPortObjectSpec, storage: bytes):
        return cls(spec)

    def create_client(self, ctx) -> GrokClient:
        return GrokClient(self.spec.auth.api_key(ctx), self.spec.auth.base_url)


grok_chat_port_type = _register_port(
    "Grok Chat Model",
    GrokChatModelPortObject,
    GrokChatModelPortObjectSpec,
)
