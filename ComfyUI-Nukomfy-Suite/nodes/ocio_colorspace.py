"""Dual-describer OCIO colorspace conversion node.

Mirror of Nuke ``OCIOColorSpace``: converts an
incoming image from Input Colorspace directly to Output Colorspace
using the active OCIO config (studio-config-v4.0.0 ACES 2.0 by default,
with the fallback chain in _backends).

Both knobs are required and expose the full OCIO colorspace list (no
``auto`` option). When the two values match, the input image is
returned untouched.

Alpha handling: ``unpremult=True`` is passed to OIIO ``colorconvert``,
matching NukomfyRead and NukomfyWrite. This is robust when the input
arrives pre-multiplied (rare in ComfyUI but possible at integration
boundaries).
"""
from __future__ import annotations

import OpenImageIO as oiio

from comfy_api.latest import IO

from . import _backends
from ._backends import _LINEAR_REC709
from ._io_helpers import comfy_to_oiio, oiio_to_comfy


class NukomfyOCIOColorSpace(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        cs_list = _backends.get_available_colorspaces()
        default_cs = _LINEAR_REC709 if _LINEAR_REC709 in cs_list else (
            cs_list[0] if cs_list else ""
        )
        return IO.Schema(
            node_id="NukomfyOCIOColorSpace",
            display_name="Nukomfy OCIO ColorSpace",
            category="Nukomfy/Color",
            description=(
                "Convert an image from one OCIO colorspace to another "
                "using the active OCIO config."
            ),
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input(
                    "input_colorspace",
                    display_name="Input Colorspace",
                    options=cs_list,
                    default=default_cs,
                    tooltip=(
                        "Colorspace the input image is already in.\n\n"
                        "The image is converted from this colorspace to the "
                        "Output Colorspace.\n\n"
                        "When Input and Output match, the image is returned "
                        "unchanged."
                    ),
                ),
                IO.Combo.Input(
                    "output_colorspace",
                    display_name="Output Colorspace",
                    options=cs_list,
                    default=default_cs,
                    tooltip=(
                        "Colorspace the image is converted to.\n\n"
                        "The transform is direct (no intermediate roundtrip)."
                        "\n\n"
                        "Alpha is unpremultiplied around the transform to "
                        "avoid halos on partially transparent pixels."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
            ],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls,
        image,
        input_colorspace: str,
        output_colorspace: str,
    ) -> IO.NodeOutput:
        _backends.validate_colorspace(input_colorspace)
        _backends.validate_colorspace(output_colorspace)

        if input_colorspace == output_colorspace:
            return IO.NodeOutput(image)

        bufs = comfy_to_oiio(image)
        bufs = [
            _colorconvert(b, input_colorspace, output_colorspace) for b in bufs
        ]
        return IO.NodeOutput(oiio_to_comfy(bufs))


def _colorconvert(
    buf: "oiio.ImageBuf", src: str, dst: str
) -> "oiio.ImageBuf":
    result = oiio.ImageBufAlgo.colorconvert(
        buf,
        src,
        dst,
        unpremult=True,
        colorconfig=_backends.get_color_config_uri(),
    )
    err = result.geterror()
    if err:
        raise RuntimeError(
            f"Colorspace conversion failed ({src} -> {dst}): {err}"
        )
    return result
