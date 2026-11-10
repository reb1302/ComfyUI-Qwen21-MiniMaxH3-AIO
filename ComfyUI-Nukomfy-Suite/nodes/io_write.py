"""Target-describer image/sequence writer.

Behaves like Nuke Write for the common case:
writes one file per frame in the input batch, substituting the padding
token in file_path with frame_start + i for batch index i.

Knob naming, ordering and option values mirror Nuke verbatim
(displayed labels: "Data Type", "Compression", "Zip (1 scanline)",
"DWAA", "RLE", ...). Internal ids stay snake_case so workflow JSON
consumers can rely on stable keys. Format-specific knobs are exposed
via IO.DynamicCombo. The output channel count is derived automatically
from the input tensor: no-alpha formats (jpeg, dpx, hdr) write 3
channels; other formats write 4 if the input has alpha, 3 otherwise.

HDR-on-integer write paths clip silently with no warning, matching
Nuke.

Returns the Comfy ui-dict::

    {"ui": {"nukomfy_written": [{"paths": [...], "count": N}]}}

so the Nukomfy plugin can read server-confirmed frame counts via
/api/history.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import OpenImageIO as oiio

from comfy_api.latest import IO

from . import _backends
from . import _sequence
from ._backends import _LINEAR_REC709
from ._io_helpers import comfy_to_oiio
from .multilayer_pack import NukomfyMultilayer, _channel_names_for_layer

_log = logging.getLogger(__name__)


_EXR_COMPRESSION_MAP = {
    "none": "none",
    "Zip (1 scanline)": "zips",
    "Zip (16 scanlines)": "zip",
    "PIZ Wavelet (32 scanlines)": "piz",
    "RLE": "rle",
    "B44": "b44",
    "B44A": "b44a",
    "DWAA": "dwaa",
    "DWAB": "dwab",
}

_TIFF_COMPRESSION_LABELS = ["Deflate", "LZW", "PackBits", "none"]
_TIFF_COMPRESSION_MAP = {
    "none": "none",
    "PackBits": "packbits",
    "LZW": "lzw",
    "Deflate": "zip",
}

_TGA_COMPRESSION_LABELS = ["RLE", "none"]
_TGA_COMPRESSION_MAP = {"none": "none", "RLE": "rle"}

_NO_ALPHA_FORMATS = frozenset({"jpeg", "dpx", "hdr"})


def _typedesc_for(file_type: str, datatype: str) -> "oiio.TypeDesc":
    if file_type == "exr":
        return oiio.TypeFloat if datatype == "32 bit float" else oiio.TypeHalf
    if file_type == "png":
        return oiio.TypeUInt16 if datatype == "16 bit" else oiio.TypeUInt8
    if file_type == "tiff":
        if datatype == "32 bit float":
            return oiio.TypeFloat
        if datatype == "16 bit":
            return oiio.TypeUInt16
        return oiio.TypeUInt8
    if file_type == "dpx":
        return oiio.TypeUInt8 if datatype == "8 bit" else oiio.TypeUInt16
    if file_type == "jpeg":
        return oiio.TypeUInt8
    if file_type == "hdr":
        return oiio.TypeFloat
    if file_type == "tga":
        return oiio.TypeUInt8
    raise RuntimeError(f"Unsupported file type: {file_type}.")


_DPX_BITS_PER_SAMPLE = {"8 bit": 8, "10 bit": 10, "12 bit": 12, "16 bit": 16}

# Modern subset of DPX transfer characteristics (4 of the 14 Nuke
# exposes). Covers ~99% of VFX/film workflows; the SDTV/broadcast/depth
# legacy values are omitted. "(auto detect)" maps to
# None so OIIO picks the transfer from the image data itself.
_DPX_TRANSFER_LABELS = [
    "(auto detect)",
    "log",
    "linear",
    "printing density",
]
_DPX_TRANSFER_TAG = {
    "(auto detect)": None,
    "log": "Logarithmic",
    "linear": "Linear",
    "printing density": "Printing density",
}


_TT_FILE_PATH = (
    "File to write.\n\n"
    "Examples:\n"
    "• image.####.exr or image.%04d.exr : sequence\n"
    "• image.0008.exr or image.exr : single frame\n\n"
    "Multi-frame renders need a sequence pattern (#### or %04d)."
)
_TT_FILE_TYPE = "Output format. Only the knobs valid for the format are shown."
_TT_FRAME_START = (
    "Frame number written for the first input image."
)
_TT_APPLY_COLOR_TRANSFORM = (
    "When enabled, the input pixels are converted from linear "
    "Rec.709 to Output Transform before being written.\n\n"
    "When disabled, the input pixels are written through unchanged "
    "and Output Transform is ignored."
)
_TT_OUTPUT_TRANSFORM = (
    "Colorspace the file is written in.\n\n"
    "When Apply Color Transform is enabled, pixels are converted "
    "from linear Rec.709 to this colorspace.\n\n"
    "Entries are the full OCIO colorspace list from the active config."
)
_TT_CREATE_DIRS = (
    "Create missing parent directories. When disabled and the parent\n"
    "is missing, the node raises an error."
)
_TT_DATATYPE = "Pixel precision."
_TT_COMPRESSION_EXR = "EXR compression."
_TT_COMPRESSION_TIFF = "TIFF compression."
_TT_COMPRESSION_TGA = "TGA compression."
_TT_QUALITY = "JPEG quality (0.0 to 1.0)."
_TT_SUBSAMPLING = "JPEG chroma sub-sampling."
_TT_TRANSFER = "DPX transfer characteristic written into the file header."
_TT_DW_LEVEL = (
    "DWAA / DWAB compression level. Default 45.\n\n"
    "Lower values keep more quality and produce larger files. "
    "Higher values compress more aggressively.\n\n"
    "Hidden when Compression is not DWAA or DWAB."
)
_TT_EXTRA_LAYERS = (
    "Optional bundle of extra image layers (depth, normal, mask, "
    "motion, …) from Nukomfy Multilayer Pack or the extra_layers "
    "output of Nukomfy Read.\n\n"
    "When connected and File Type is EXR, writes a multilayer EXR "
    "with the main image as the rgba layer (no channel prefix) plus "
    "every extra layer prefixed by its name (e.g. depth.Z, "
    "normal.R, motion.u).\n\n"
    "When File Type is not EXR, this socket is ignored: only the "
    "main image is written, and a warning is logged."
)


class NukomfyWrite(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NukomfyWrite",
            display_name="Nukomfy Write",
            category="Nukomfy/Image",
            description=(
                "Write an image or sequence to disk. Mirror of Nuke "
                "Write for the common case.\n\n"
                "Format-specific knobs are shown "
                "only for the formats that support them."
            ),
            inputs=[
                IO.Image.Input("image"),
                NukomfyMultilayer.Input(
                    "extra_layers",
                    optional=True,
                    tooltip=_TT_EXTRA_LAYERS,
                ),
                IO.String.Input(
                    "file_path",
                    display_name="Output",
                    default="",
                    placeholder="####.exr",
                    tooltip=_TT_FILE_PATH,
                ),
                IO.DynamicCombo.Input(
                    "file_type",
                    display_name="File Type",
                    options=[
                        # Widget IDs are format-prefixed (e.g. exr_datatype,
                        # png_datatype) so each DynamicCombo Option owns a
                        # unique identifier even when the display label
                        # ("Data Type", "Compression") is shared across
                        # formats. Without prefixing, clients that build a
                        # flat widget map by name would collide between
                        # options - the display name stays user-friendly
                        # while the wire name carries the format context.
                        IO.DynamicCombo.Option("exr", [
                            IO.Combo.Input(
                                "exr_datatype",
                                display_name="Data Type",
                                options=["16 bit half", "32 bit float"],
                                default="16 bit half",
                                tooltip=_TT_DATATYPE,
                            ),
                            # Nested DynamicCombo: DW Compression Level is
                            # exposed server-side only inside DWAA / DWAB
                            # options.
                            IO.DynamicCombo.Input(
                                "exr_compression",
                                display_name="Compression",
                                # First option is the frontend default for a
                                # nested DynamicCombo (schema default ignored),
                                # so "Zip (1 scanline)" comes first to match
                                # Nuke's native default compression.
                                options=[
                                    IO.DynamicCombo.Option("Zip (1 scanline)", []),
                                    IO.DynamicCombo.Option("Zip (16 scanlines)", []),
                                    IO.DynamicCombo.Option("PIZ Wavelet (32 scanlines)", []),
                                    IO.DynamicCombo.Option("RLE", []),
                                    IO.DynamicCombo.Option("B44", []),
                                    IO.DynamicCombo.Option("B44A", []),
                                    IO.DynamicCombo.Option("DWAA", [
                                        IO.Int.Input(
                                            "exr_dw_compression_level",
                                            display_name="Compression Level",
                                            default=45,
                                            min=0,
                                            max=500,
                                            display_mode=IO.NumberDisplay.slider,
                                            tooltip=_TT_DW_LEVEL,
                                        ),
                                    ]),
                                    IO.DynamicCombo.Option("DWAB", [
                                        IO.Int.Input(
                                            "exr_dw_compression_level",
                                            display_name="Compression Level",
                                            default=45,
                                            min=0,
                                            max=500,
                                            display_mode=IO.NumberDisplay.slider,
                                            tooltip=_TT_DW_LEVEL,
                                        ),
                                    ]),
                                    IO.DynamicCombo.Option("none", []),
                                ],
                                tooltip=_TT_COMPRESSION_EXR,
                            ),
                        ]),
                        IO.DynamicCombo.Option("dpx", [
                            IO.Combo.Input(
                                "dpx_datatype",
                                display_name="Data Type",
                                options=["8 bit", "10 bit", "12 bit", "16 bit"],
                                default="10 bit",
                                tooltip=_TT_DATATYPE,
                            ),
                            IO.Combo.Input(
                                "dpx_transfer",
                                display_name="Transfer",
                                options=_DPX_TRANSFER_LABELS,
                                default="(auto detect)",
                                tooltip=_TT_TRANSFER,
                            ),
                        ]),
                        IO.DynamicCombo.Option("hdr", []),
                        IO.DynamicCombo.Option("jpeg", [
                            IO.Float.Input(
                                "jpeg_quality",
                                display_name="Quality",
                                default=0.75,
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                display_mode=IO.NumberDisplay.slider,
                                tooltip=_TT_QUALITY,
                            ),
                            IO.Combo.Input(
                                "jpeg_sub_sampling",
                                display_name="Sub-sampling",
                                options=["4:1:1", "4:2:2", "4:4:4"],
                                default="4:1:1",
                                tooltip=_TT_SUBSAMPLING,
                            ),
                        ]),
                        IO.DynamicCombo.Option("png", [
                            IO.Combo.Input(
                                "png_datatype",
                                display_name="Data Type",
                                options=["8 bit", "16 bit"],
                                default="8 bit",
                                tooltip=_TT_DATATYPE,
                            ),
                        ]),
                        IO.DynamicCombo.Option("tga", [
                            IO.Combo.Input(
                                "tga_compression",
                                display_name="Compression",
                                options=_TGA_COMPRESSION_LABELS,
                                default="RLE",
                                tooltip=_TT_COMPRESSION_TGA,
                            ),
                        ]),
                        IO.DynamicCombo.Option("tiff", [
                            IO.Combo.Input(
                                "tiff_datatype",
                                display_name="Data Type",
                                options=["8 bit", "16 bit", "32 bit float"],
                                default="8 bit",
                                tooltip=_TT_DATATYPE,
                            ),
                            IO.Combo.Input(
                                "tiff_compression",
                                display_name="Compression",
                                options=_TIFF_COMPRESSION_LABELS,
                                default="Deflate",
                                tooltip=_TT_COMPRESSION_TIFF,
                            ),
                        ]),
                    ],
                    tooltip=_TT_FILE_TYPE,
                ),
                IO.Int.Input(
                    "frame_start",
                    display_name="Frame Start",
                    default=1,
                    min=-999999,
                    max=999999,
                    tooltip=_TT_FRAME_START,
                ),
                IO.Boolean.Input(
                    "apply_color_transform",
                    display_name="Apply Color Transform",
                    default=False,
                    tooltip=_TT_APPLY_COLOR_TRANSFORM,
                ),
                IO.Combo.Input(
                    "output_transform",
                    display_name="Output Transform",
                    options=_backends.get_available_colorspaces(),
                    default=_LINEAR_REC709,
                    tooltip=_TT_OUTPUT_TRANSFORM,
                ),
                IO.Boolean.Input(
                    "create_directories",
                    display_name="Create Directories",
                    default=True,
                    tooltip=_TT_CREATE_DIRS,
                ),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        image,
        file_path: str,
        file_type: dict,
        frame_start: int,
        apply_color_transform: bool,
        output_transform: str,
        create_directories: bool,
        extra_layers: "dict | None" = None,
    ) -> IO.NodeOutput:
        file_path = (file_path or "").strip()
        if not file_path:
            raise RuntimeError("The Output field is empty. Enter a file path.")
        if image is None:
            raise RuntimeError(
                "No image is connected. Connect an image into the Nukomfy Write node."
            )

        file_format = file_type["file_type"]
        # Per-option sub-input IDs are format-prefixed in the schema
        # (e.g. exr_datatype, png_datatype) so clients see unique
        # widget names across DynamicCombo options. Server reads them
        # back with the same prefix.
        datatype = file_type.get(f"{file_format}_datatype", "8 bit")
        # exr_compression is a nested DynamicCombo (DWAA/DWAB carry the
        # exr_dw_compression_level sub-input); tga/tiff use a flat
        # Combo and arrive as a plain string.
        comp_value = file_type.get(f"{file_format}_compression")
        if isinstance(comp_value, dict):
            compression = comp_value.get(f"{file_format}_compression", "none")
            dw_level = int(comp_value.get(f"{file_format}_dw_compression_level", 45))
        else:
            compression = comp_value or "none"
            dw_level = 45
        quality = float(file_type.get("jpeg_quality", 0.75))
        sub_sampling = file_type.get("jpeg_sub_sampling", "4:1:1")
        transfer = file_type.get("dpx_transfer", "(auto detect)")

        bufs = comfy_to_oiio(image)
        batch_size = len(bufs)
        if batch_size == 0:
            raise RuntimeError(
                f"Nothing to write for '{file_path}': the input has no frames."
            )

        if apply_color_transform:
            bufs = [_apply_output_transform(b, output_transform) for b in bufs]

        paths = _resolve_paths(file_path, frame_start, batch_size)
        typedesc = _typedesc_for(file_format, datatype)
        input_nchans = bufs[0].spec().nchannels
        nchans = _resolve_nchannels(file_format, input_nchans)

        # Fallback path: extra_layers only makes sense for EXR. For every
        # other file_type, log a clear warning and drop the bundle so the
        # main image still writes successfully (mirror Nuke's "format does
        # not support this knob, ignored" behavior).
        warnings: list[str] = []
        if extra_layers and file_format != "exr":
            names = sorted(extra_layers.keys())
            msg = (
                f"extra_layers ignored: only EXR supports multilayer "
                f"(file_type={file_format!r}, dropping {len(names)} layer(s): {names})"
            )
            _log.warning(msg)
            warnings.append(msg)
            extra_layers = None

        if extra_layers:
            # Multilayer EXR: single-part with main rgba (no prefix) +
            # named extra layers prefixed per Nuke channel convention.
            written_paths = _write_multilayer_exr(
                main_bufs=bufs,
                extra_layers=extra_layers,
                paths=paths,
                typedesc=typedesc,
                main_nchans=nchans,
                compression=compression,
                dw_level=dw_level,
                create_directories=create_directories,
            )
        else:
            written_paths = []
            for i, path in enumerate(paths):
                _ensure_parent_dir(path, create_directories)
                write_buf = _channel_adapt(bufs[i], nchans)
                spec = _build_spec(
                    write_buf, typedesc, nchans, file_format,
                    datatype=datatype,
                    compression=compression,
                    quality=quality,
                    sub_sampling=sub_sampling,
                    transfer=transfer,
                    dw_level=dw_level,
                )
                _write_one(write_buf, path, spec)
                written_paths.append(path)

        if not written_paths:
            raise RuntimeError(f"No frames written for '{file_path}'.")

        ui_payload: dict = {
            "nukomfy_written": [
                {"paths": written_paths, "count": len(written_paths)}
            ]
        }
        if warnings:
            ui_payload["nukomfy_warnings"] = warnings
        return IO.NodeOutput(ui=ui_payload)


def _resolve_paths(
    file_path: str, frame_start: int, batch_size: int
) -> list[str]:
    # When the path has a padding token (####, %04d, @@@@) the padding
    # width is taken from the token. When the path is a plain literal
    # and the batch contains multiple frames, the write is refused with
    # a clear message: the user must provide a padding token explicitly.
    # This mirrors Nuke, which raises "write cannot be executed
    # for multiple frames" instead of silently promoting digits.
    if _sequence.has_padding_token(file_path):
        return [
            _sequence.path_for_frame(file_path, frame_start + i)
            for i in range(batch_size)
        ]
    if batch_size > 1:
        raise RuntimeError(
            f"Output '{file_path}' is a single file but the render has "
            f"{batch_size} frames.\n"
            f"Use a frame pattern (#### or %04d) to write a sequence."
        )
    return [file_path]


def _resolve_nchannels(file_type: str, input_nchans: int) -> int:
    """Auto-resolve target channel count.

    No-alpha formats (jpeg/dpx/hdr) always write 3 channels. For other
    formats, mirror the input: 4 channels if the input has alpha, 3 if not.
    Mirrors Nuke Write "all" channels default behavior.
    """
    if file_type in _NO_ALPHA_FORMATS:
        return 3
    return 4 if input_nchans >= 4 else 3


def _channel_adapt(buf: "oiio.ImageBuf", target_nchans: int) -> "oiio.ImageBuf":
    src_nchans = buf.spec().nchannels
    if src_nchans == target_nchans:
        return buf
    if src_nchans == 1:
        rgb = oiio.ImageBufAlgo.channels(buf, (0, 0, 0))
        err = rgb.geterror()
        if err:
            raise RuntimeError(f"Channel adapt failed (1 -> rgb): {err}")
        return _channel_adapt(rgb, target_nchans)
    if target_nchans == 3 and src_nchans >= 3:
        out = oiio.ImageBufAlgo.channels(buf, (0, 1, 2))
        err = out.geterror()
        if err:
            raise RuntimeError(f"Channel adapt failed (drop alpha): {err}")
        return out
    if target_nchans == 4 and src_nchans == 3:
        out = oiio.ImageBufAlgo.channels(buf, (0, 1, 2, 1.0))
        err = out.geterror()
        if err:
            raise RuntimeError(f"Channel adapt failed (append alpha): {err}")
        return out
    raise RuntimeError(
        f"Cannot adapt channels {src_nchans} -> {target_nchans}"
    )


def _build_spec(
    buf: "oiio.ImageBuf",
    typedesc: "oiio.TypeDesc",
    nchans: int,
    file_type: str,
    *,
    datatype: str,
    compression: str,
    quality: float,
    sub_sampling: str,
    transfer: str,
    dw_level: int,
) -> "oiio.ImageSpec":
    src = buf.spec()
    spec = oiio.ImageSpec(src.width, src.height, nchans, typedesc)

    if file_type == "exr":
        # DWAA/DWAB level is passed via the codec:level syntax of the
        # compression attribute itself. The separate openexr:dwaCompressionLevel
        # attribute reflects the level only when READING an existing file; on
        # write, OIIO honors the level only via this codec:N suffix.
        codec = _EXR_COMPRESSION_MAP.get(compression, "zip")
        if codec in ("dwaa", "dwab"):
            spec.attribute("compression", f"{codec}:{dw_level}")
        else:
            spec.attribute("compression", codec)
    elif file_type == "jpeg":
        spec.attribute("CompressionQuality", int(round(quality * 100)))
        spec.attribute("jpeg:subsampling", sub_sampling)
    elif file_type == "tiff":
        codec = _TIFF_COMPRESSION_MAP.get(compression, "zip")
        spec.attribute("compression", codec)
    elif file_type == "dpx":
        bps = _DPX_BITS_PER_SAMPLE.get(datatype, 10)
        spec.attribute("oiio:BitsPerSample", bps)
        tag = _DPX_TRANSFER_TAG.get(transfer)
        if tag is not None:
            spec.attribute("dpx:Transfer", tag)
    elif file_type == "tga":
        codec = _TGA_COMPRESSION_MAP.get(compression, "none")
        spec.attribute("compression", codec)

    return spec


def _ensure_parent_dir(path: str, create_directories: bool) -> None:
    parent = os.path.dirname(path)
    if not parent:
        return
    if create_directories:
        os.makedirs(parent, exist_ok=True)
    elif not os.path.isdir(parent):
        raise RuntimeError(
            f"Parent directory does not exist: '{parent}'. "
            f"Enable Create Directories or create the folder manually."
        )


def _apply_output_transform(
    buf: "oiio.ImageBuf", transform: str
) -> "oiio.ImageBuf":
    """Convert from linear Rec.709 to a user-picked OCIO colorspace.

    ``transform`` is a canonical OCIO colorspace name from the active
    config. When it equals linear Rec.709, the conversion is a no-op
    and the input buffer is returned unchanged.
    """
    if transform == _LINEAR_REC709:
        return buf
    result = oiio.ImageBufAlgo.colorconvert(
        buf,
        _LINEAR_REC709,
        transform,
        unpremult=True,
        colorconfig=_backends.get_color_config_uri(),
    )
    err = result.geterror()
    if err:
        raise RuntimeError(
            f"Colorspace conversion failed (linear Rec.709 -> {transform}): {err}"
        )
    return result


def _write_one(
    buf: "oiio.ImageBuf", path: str, spec: "oiio.ImageSpec"
) -> None:
    out = oiio.ImageOutput.create(path)
    if out is None:
        raise RuntimeError(f"OIIO cannot create writer for '{path}'")
    if not out.open(path, spec):
        err = out.geterror() or "unknown"
        out.close()
        raise RuntimeError(f"Failed to open '{path}' for write: {err}")
    try:
        if not buf.write(out):
            err = buf.geterror() or out.geterror() or "unknown"
            raise RuntimeError(f"Failed to write '{path}': {err}")
    finally:
        out.close()


def _frame_pixels_from_buf(buf: "oiio.ImageBuf") -> "np.ndarray":
    """Read a frame's pixel data from an ImageBuf as a contiguous [H, W, C] array."""
    pixels = buf.get_pixels(oiio.TypeFloat)
    if pixels is None:
        raise RuntimeError(f"OIIO get_pixels failed: {buf.geterror()}")
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    if pixels.ndim == 2:
        pixels = pixels[..., np.newaxis]
    return pixels


def _frame_pixels_from_tensor(tensor, frame_index: int) -> "np.ndarray":
    """Extract frame ``frame_index`` from a torch tensor [B, H, W, C] as [H, W, C] float32."""
    if tensor.shape[0] <= frame_index:
        raise RuntimeError(
            f"Extra layer batch is too short: got {tensor.shape[0]} frames, "
            f"main image needs frame {frame_index}."
        )
    frame = tensor[frame_index].detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(frame, dtype=np.float32)


def _exr_compression_attr(compression: str, dw_level: int) -> str:
    """Map UI compression label to the OIIO compression attribute string.

    DWAA/DWAB use the ``codec:level`` syntax; other codecs use the
    plain codec name. Mirror of the logic in _build_spec.
    """
    codec = _EXR_COMPRESSION_MAP.get(compression, "zip")
    if codec in ("dwaa", "dwab"):
        return f"{codec}:{dw_level}"
    return codec


def _write_multilayer_exr(
    *,
    main_bufs: list,
    extra_layers: dict,
    paths: list[str],
    typedesc: "oiio.TypeDesc",
    main_nchans: int,
    compression: str,
    dw_level: int,
    create_directories: bool,
) -> list[str]:
    """Write a single-part multilayer EXR per frame.

    Channel layout: main rgba/rgb (no prefix) followed by each extra
    layer prefixed by its name (see ``_channel_names_for_layer``).
    All layers must share resolution with the main image.
    """
    batch_size = len(main_bufs)
    if batch_size == 0:
        raise RuntimeError("Main image has no frames for multilayer write.")

    # Pre-build the channel name layout once. It does not change per
    # frame (resolution and channel count are constant across the batch).
    main_spec = main_bufs[0].spec()
    main_w, main_h = main_spec.width, main_spec.height
    main_layer_name = "rgba" if main_nchans == 4 else "rgb"
    channel_names: list[str] = _channel_names_for_layer(main_layer_name, main_nchans)
    layer_order: list[tuple[str, int]] = []  # (name, n_channels) preserving dict order

    for name, tensor in extra_layers.items():
        if tensor.ndim != 4:
            raise RuntimeError(
                f"Extra layer '{name}' has shape {tuple(tensor.shape)}, "
                f"expected [B, H, W, C]."
            )
        lb, lh, lw, lc = tensor.shape
        if (lh, lw) != (main_h, main_w):
            raise RuntimeError(
                f"Extra layer '{name}' resolution {lw}x{lh} does not match "
                f"main image {main_w}x{main_h}. All layers must share width "
                f"and height."
            )
        if lb < batch_size:
            raise RuntimeError(
                f"Extra layer '{name}' has batch size {lb}, smaller than "
                f"main image batch size {batch_size}."
            )
        channel_names.extend(_channel_names_for_layer(name, lc))
        layer_order.append((name, lc))

    # Two layers can map to the same EXR channel: a 'motion' layer emits the
    # forward/backward channels a separate 'forward'/'backward' layer also emits.
    # OIIO would silently rename the collision to anonymous channels and drop a
    # layer's data, so refuse it with a clear error.
    seen: set[str] = set()
    collisions: set[str] = set()
    for cn in channel_names:
        if cn in seen:
            collisions.add(cn)
        seen.add(cn)
    if collisions:
        raise RuntimeError(
            f"Layers collide on EXR channel(s) {sorted(collisions)}. A 'motion' "
            f"layer already writes forward/backward channels, so it cannot share "
            f"a Pack with a separate 'forward' or 'backward' layer. Rename or "
            f"remove one of the colliding layers."
        )

    total_channels = len(channel_names)
    written_paths: list[str] = []

    for i, path in enumerate(paths):
        _ensure_parent_dir(path, create_directories)

        # Assemble combined [H, W, total_channels] buffer for this frame.
        main_frame = _frame_pixels_from_buf(main_bufs[i])  # [H, W, main_nchans]
        if main_frame.shape[-1] != main_nchans:
            # Defensive: _channel_adapt logic only ran in the single-layer
            # branch. For multilayer we drop alpha when main_nchans == 3
            # and the input has 4 channels, or pad with 1.0 alpha when
            # main_nchans == 4 and input has 3.
            main_frame = _adapt_frame_channels(main_frame, main_nchans)

        chunks: list[np.ndarray] = [main_frame]
        for name, lc in layer_order:
            extra_frame = _frame_pixels_from_tensor(extra_layers[name], i)
            if extra_frame.shape[-1] != lc:
                raise RuntimeError(
                    f"Extra layer '{name}' channel count drift: tensor has "
                    f"{extra_frame.shape[-1]} channels, registered {lc}."
                )
            if extra_frame.shape[:2] != (main_h, main_w):
                raise RuntimeError(
                    f"Extra layer '{name}' frame {i} resolution mismatch: "
                    f"{extra_frame.shape[1]}x{extra_frame.shape[0]} vs main "
                    f"{main_w}x{main_h}."
                )
            chunks.append(extra_frame)

        combined = np.concatenate(chunks, axis=-1)
        if combined.shape != (main_h, main_w, total_channels):
            raise RuntimeError(
                f"Combined buffer shape {combined.shape} does not match "
                f"expected ({main_h}, {main_w}, {total_channels})."
            )

        spec = oiio.ImageSpec(main_w, main_h, total_channels, typedesc)
        spec.channelnames = tuple(channel_names)
        spec.attribute("compression", _exr_compression_attr(compression, dw_level))

        out = oiio.ImageOutput.create(path)
        if out is None:
            raise RuntimeError(f"OIIO cannot create writer for '{path}'")
        if not out.open(path, spec):
            err = out.geterror() or "unknown"
            out.close()
            raise RuntimeError(f"Failed to open '{path}' for multilayer write: {err}")
        try:
            if not out.write_image(combined):
                err = out.geterror() or "unknown"
                raise RuntimeError(f"Failed to write multilayer '{path}': {err}")
        finally:
            out.close()
        written_paths.append(path)

    return written_paths


def _adapt_frame_channels(frame: "np.ndarray", target_nchans: int) -> "np.ndarray":
    """Adapt a single-frame [H, W, C] array to target channel count.

    Used only in the multilayer write path. Mirror of _channel_adapt at
    the numpy level. Behavior matches Nuke Write for the common
    cases:
      - 1 ch -> 3 / 4 ch : R = G = B = src; A = 1.0 when target is 4
      - 2 ch -> 3 / 4 ch : R = src.R, G = src.G, B = 0; A = 1.0 when target is 4
      - 3 ch -> 4 ch     : append opaque alpha
      - 4 ch -> 3 ch     : drop alpha
    """
    src = frame.shape[-1]
    if src == target_nchans:
        return frame
    if target_nchans == 3 and src >= 3:
        return np.ascontiguousarray(frame[..., :3])
    if target_nchans == 4 and src == 3:
        alpha = np.ones(frame.shape[:2] + (1,), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate([frame, alpha], axis=-1))
    if src == 1 and target_nchans in (3, 4):
        rgb = np.repeat(frame, 3, axis=-1)
        if target_nchans == 3:
            return np.ascontiguousarray(rgb)
        alpha = np.ones(frame.shape[:2] + (1,), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=-1))
    if src == 2 and target_nchans in (3, 4):
        blue = np.zeros(frame.shape[:2] + (1,), dtype=np.float32)
        rgb = np.concatenate([frame, blue], axis=-1)
        if target_nchans == 3:
            return np.ascontiguousarray(rgb)
        alpha = np.ones(frame.shape[:2] + (1,), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=-1))
    raise RuntimeError(f"Cannot adapt frame channels {src} -> {target_nchans}")
