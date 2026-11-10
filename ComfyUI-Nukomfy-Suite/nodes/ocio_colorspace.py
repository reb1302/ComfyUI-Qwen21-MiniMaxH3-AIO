"""Dual-describer OCIO colorspace conversion node.

Mirror of Nuke ``OCIOColorSpace``: converts an
incoming image from Input Colorspace directly to Output Colorspace
using the active OCIO config (studio-config-v4.0.0 ACES 2.0 by default,
with the fallback chain in _backends).
"""
from __future__ import annotations

try:
    import OpenImageIO as oiio
except ImportError:
    oiio = None  # type: ignore[assignment]

from comfy_api.latest import IO

from . import _backends
from ._backends import _LINEAR_REC709
from ._io_helpers import _FallbackImageBuf, comfy_to_oiio, oiio_to_comfy


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


def _colorconvert(buf, src: str, dst: str):
    if src == dst:
        return buf
    if oiio is not None and _backends.has_ocio_config() and not isinstance(buf, _FallbackImageBuf):
        result = oiio.ImageBufAlgo.colorconvert(
            buf,
            src,
            dst,
            unpremult=True,
            colorconfig=_backends.get_color_config_uri(),
        )
        if not result.geterror():
            return result
    # Analytic fallback when OCIO config is absent or colorspace name differs
    pixels = buf.get_pixels(oiio.TypeFloat) if (oiio is not None and not isinstance(buf, _FallbackImageBuf)) else buf.get_pixels()
    converted = _backends.analytic_colorconvert_numpy(pixels, src, dst)
    if oiio is not None and not isinstance(buf, _FallbackImageBuf):
        spec = oiio.ImageSpec(converted.shape[1], converted.shape[0], converted.shape[2], oiio.TypeFloat)
        out_buf = oiio.ImageBuf(spec)
        out_buf.set_pixels(out_buf.roi, converted)
        return out_buf
    return _FallbackImageBuf(converted)
