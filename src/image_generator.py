"""Grok Imagine image generation node."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import knime.extension as knext
import pandas as pd
from PIL import Image

from client import DEFAULT_IMAGE_MODELS, GrokClient
from ports import (
    GROK_ICON,
    NODE_ATTRIBUTION,
    GrokAuthenticationPortObject,
    GrokAuthenticationPortObjectSpec,
    grok_auth_port_type,
    grok_category,
)
from prompter import _first_string_column, _is_string_column


def _list_image_models(ctx: knext.ConfigurationContext):
    specs = ctx.get_input_specs()
    if specs and specs[0]:
        return specs[0].get_image_model_list(ctx)
    return list(DEFAULT_IMAGE_MODELS)


@knext.node(
    name="Grok Image Generator",
    node_type=knext.NodeType.SOURCE,
    icon_path=GROK_ICON,
    category=grok_category,
    keywords=["Grok", "xAI", "GenAI", "Imagine", "image"],
)
@knext.input_port(
    "Grok Authentication",
    "Authentication from the Grok Authenticator.",
    grok_auth_port_type,
)
@knext.input_table(
    "Prompt Table",
    "Optional table with a prompt column. If connected, the first non-empty value is used.",
    optional=True,
)
@knext.output_image(
    "Generated Image",
    "Standard KNIME image port (PNG). Connect Image to Table, or view it on the executed node.",
)
@knext.output_table(
    "Image Table",
    "One-row table with Prompt and Image columns. Connect Image Writer (Table) to save a PNG file.",
)
class GrokImageGenerator:
    f"""Generate an image with Grok Imagine.

    Connect **Grok Authenticator**. Enter a prompt here, or connect a table with a
    prompt column (the first non-empty row is used).

    Outputs:
    - **Generated Image** — standard KNIME Image port (not a Grok port). Use
      **Image to Table**, or open the executed node's output to preview.
    - **Image Table** — one row with the PNG in an Image column. Use
      **Image Writer (Table)** to write a file.

    Optionally set *Write PNG to file* to save the image during execute.

    Imagine models are selected on this node, not on **Grok Chat Model Selector**.

    {NODE_ATTRIBUTION}
    """

    model = knext.StringParameter(
        "Model",
        "Grok Imagine model. The list always includes grok-imagine-image-2.0 and grok-imagine-image, plus any image models returned by the API.",
        default_value="grok-imagine-image-2.0",
        choices=_list_image_models,
    )

    prompt = knext.MultilineStringParameter(
        "Prompt",
        "Image description. Used when no prompt table is connected, or when that table has no non-empty prompt.",
        default_value="",
    )

    prompt_column = knext.ColumnParameter(
        "Prompt column",
        "Column containing the image prompt when a prompt table is connected.",
        port_index=1,
        column_filter=_is_string_column,
    )

    aspect_ratio = knext.StringParameter(
        "Aspect ratio",
        "Output aspect ratio.",
        default_value="auto",
        choices=["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
    )

    quality = knext.StringParameter(
        "Quality",
        "Image quality. auto currently uses low for generation.",
        default_value="auto",
        choices=["auto", "low", "medium"],
        is_advanced=True,
    )

    resolution = knext.StringParameter(
        "Resolution",
        "Output resolution.",
        default_value="1k",
        choices=["1k", "2k"],
        is_advanced=True,
    )

    output_file = knext.StringParameter(
        "Write PNG to file",
        "Optional absolute path (for example /Users/you/Desktop/grok.png). If set, the PNG is written during execute.",
        default_value="",
        is_advanced=True,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        auth: GrokAuthenticationPortObjectSpec,
        prompt_schema=None,
    ):
        auth.validate_context(ctx)
        if not self.model:
            raise knext.InvalidParametersError("Select a Grok Imagine model.")
        if prompt_schema is not None:
            if not self.prompt_column:
                self.prompt_column = _first_string_column(prompt_schema)
            elif self.prompt_column not in prompt_schema.column_names:
                raise knext.InvalidParametersError(
                    f"Prompt column '{self.prompt_column}' is not in the prompt table."
                )
        elif not (self.prompt or "").strip():
            raise knext.InvalidParametersError(
                "Enter an image prompt, or connect a table with a prompt column."
            )
        return knext.ImagePortObjectSpec(knext.ImageFormat.PNG), _image_table_schema()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        auth: GrokAuthenticationPortObject,
        prompt_table=None,
    ):
        prompt = self._read_prompt(prompt_table)
        if not prompt:
            raise RuntimeError("The image prompt is empty.")
        ctx.set_progress(0.2, "Generating image")
        client = GrokClient(auth.spec.api_key(ctx), auth.spec.base_url)
        png = client.generate_image(
            model=self.model,
            prompt=prompt,
            aspect_ratio=self.aspect_ratio,
            quality=self.quality,
            resolution=self.resolution,
        )
        if (self.output_file or "").strip():
            path = Path(self.output_file.strip()).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png)
        ctx.set_progress(1.0, "Done")
        return png, _image_table(prompt, png)

    def _read_prompt(self, prompt_table) -> str:
        if prompt_table is not None:
            frame = prompt_table.to_pandas()
            column = self.prompt_column
            if not column or column not in frame.columns:
                column = _first_string_column(prompt_table.schema)
            for value in frame[column].tolist():
                if pd.isna(value):
                    continue
                text = str(value).strip()
                if text:
                    return text
        return (self.prompt or "").strip()


def _image_column_type():
    try:
        return knext.logical(Image.Image)
    except TypeError:
        return knext.blob()


def _image_table_schema() -> knext.Schema:
    return knext.Schema.from_columns(
        [
            knext.Column(knext.string(), "Prompt"),
            knext.Column(_image_column_type(), "Image"),
        ]
    )


def _image_table(prompt: str, png: bytes) -> knext.Table:
    image = Image.open(BytesIO(png))
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    return knext.Table.from_pandas(pd.DataFrame({"Prompt": [prompt], "Image": [image]}))
