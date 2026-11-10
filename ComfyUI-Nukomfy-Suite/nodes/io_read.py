"""Source-describer image/sequence reader.

Behaves like Nuke Read for the common case:
literal path -> single frame; pattern path -> sequence; folder path ->
auto-detected sequence inside.

Two read modes (mirror Nuke):
    Input Range  - scan disk, return every frame found that matches
                   file_path (pattern, literal, or folder auto-detect).
    Custom Range - return frames First Frame..Last Frame, applying the
                   Missing Frames policy on gaps. Supports reverse
                   playback when First Frame > Last Frame.

When Read Mode is Input Range, First Frame / Last Frame are grayed
out in the UI (still visible per Nuke). When Apply Color Transform is
disabled, Input Transform is grayed out and ignored.
"""
from __future__ import annotations

import OpenImageIO as oiio

from comfy_api.latest import IO

from . import _backends
from . import _sequence
from ._backends import _LINEAR_REC709
from ._io_helpers import oiio_to_multilayer
from .multilayer_pack import NukomfyMultilayer

# Fallback frame dimensions (width, height, channels) when every requested
# frame is a gap and there is no real reference frame to size from.
_PLACEHOLDER_FRAME = (64, 64, 4)


class NukomfyRead(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NukomfyRead",
            display_name="Nukomfy Read",
            category="Nukomfy/Image",
            description=(
                "Read an image or sequence from disk. Mirror of Nuke "
                "Read for the common case. Supports every format that "
                "OpenImageIO can decode: EXR, JPEG, PNG, TIFF, DPX, HDR, "
                "TGA, BMP, and more.\n\n"
                "Outputs the image as a 3-channel "
                "RGB tensor, the source alpha as a separate 1-channel mask "
                "(zeros if the file has no alpha), and any extra EXR layers "
                "as a multilayer bundle for the Nukomfy Multilayer Unpack "
                "(empty bundle for non-EXR or EXR without extra layers)."
            ),
            inputs=[
                IO.String.Input(
                    "file_path",
                    display_name="Source",
                    default="",
                    placeholder="####.exr or path/to/folder/",
                    tooltip=(
                        "File to read. Any format supported by "
                        "OpenImageIO is accepted (EXR, JPEG, PNG, TIFF, "
                        "DPX, HDR, TGA, BMP, …).\n\n"
                        "Examples:\n"
                        "• image.####.exr or image.%04d.jpg : sequence\n"
                        "• image.0008.png or image.tif : single frame\n"
                        "• /path/folder/ : auto-detect in folder"
                    ),
                ),
                IO.Combo.Input(
                    "read_mode",
                    display_name="Read Mode",
                    options=["Input Range", "Custom Range"],
                    default="Input Range",
                    tooltip=(
                        "How to decide which frames to read.\n\n"
                        "• Input Range: read every frame that matches the File "
                        "pattern on disk. First Frame, Last Frame and Missing "
                        "Frames are ignored.\n\n"
                        "• Custom Range: read the frames between First Frame and "
                        "Last Frame. Any gaps inside that range are filled per the "
                        "Missing Frames policy."
                    ),
                ),
                IO.Int.Input(
                    "first_frame",
                    display_name="First Frame",
                    default=1,
                    min=-999999,
                    max=999999,
                    tooltip=(
                        "First frame number of the sequence to read. Used only "
                        "in Custom Range mode.\n\n"
                        "When First Frame is greater than "
                        "Last Frame, the sequence is read in reverse."
                    ),
                ),
                IO.Int.Input(
                    "last_frame",
                    display_name="Last Frame",
                    default=1,
                    min=-999999,
                    max=999999,
                    tooltip=(
                        "Last frame number of the sequence to read. Used only "
                        "in Custom Range mode."
                    ),
                ),
                IO.Combo.Input(
                    "missing_frames",
                    display_name="Missing Frames",
                    options=["error", "black", "checkerboard", "nearest"],
                    default="error",
                    tooltip=(
                        "How to handle missing frames inside the requested range.\n\n"
                        "• error: stop with a message listing the missing frames.\n"
                        "• black: return a black frame for each gap.\n"
                        "• checkerboard: return a magenta and black pattern for each gap.\n"
                        "• nearest: return the closest existing frame on disk."
                    ),
                ),
                IO.Boolean.Input(
                    "apply_color_transform",
                    display_name="Apply Color Transform",
                    default=False,
                    tooltip=(
                        "When enabled, the file pixels are converted from "
                        "Input Transform to linear Rec.709.\n\n"
                        "When disabled, the file pixels pass through "
                        "unchanged and Input Transform is ignored."
                    ),
                ),
                IO.Combo.Input(
                    "input_transform",
                    display_name="Input Transform",
                    options=_backends.get_available_colorspaces(),
                    default=_LINEAR_REC709,
                    tooltip=(
                        "Colorspace the file is in.\n\n"
                        "When Apply Color Transform is enabled, pixels are "
                        "converted from this colorspace to linear Rec.709.\n\n"
                        "Entries are the full OCIO colorspace list from "
                        "the active config."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="rgb",
                    tooltip=(
                        "RGB image (3 channels). Source alpha, if any, "
                        "is dropped here and exposed separately on the "
                        "alpha output."
                    ),
                ),
                IO.Mask.Output(
                    display_name="alpha",
                    tooltip=(
                        "Source alpha as a 1-channel mask.\n\n"
                        "When the "
                        "file has no alpha (JPEG, HDR, RGB-only PNG, "
                        "EXR 3-channel, grayscale), this is filled with "
                        "zeros.\n\n"
                        "Use a Combine Image With Alpha node "
                        "(or similar) upstream of Nukomfy Write if you "
                        "need to round-trip the source alpha into a "
                        "4-channel output file."
                    ),
                ),
                NukomfyMultilayer.Output(
                    display_name="extra_layers",
                    tooltip=(
                        "Bundle of extra EXR layers (depth, normal, "
                        "position, motion, …) detected by parsing the "
                        "file's channel names. Empty for non-EXR files "
                        "and EXR files without extra layers. Connect to "
                        "Nukomfy Multilayer Unpack to extract individual "
                        "layers by name."
                    ),
                ),
            ],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls,
        file_path: str,
        read_mode: str,
        first_frame: int,
        last_frame: int,
        missing_frames: str,
        apply_color_transform: bool,
        input_transform: str,
    ) -> IO.NodeOutput:
        if not file_path or not file_path.strip():
            raise RuntimeError("The Source field is empty. Enter a file path or folder.")

        kind, normalized = _sequence.classify_path(file_path)

        if read_mode == "Input Range":
            positions = _discover_input_range(kind, normalized, missing_frames)
        else:
            positions = _discover_custom_range(
                kind, normalized, first_frame, last_frame, missing_frames
            )

        if not positions:
            raise RuntimeError(
                f"No frames found for '{file_path}'. "
                f"Check the path and the frame numbering."
            )

        bufs = _read_positions(positions, missing_frames)
        if apply_color_transform:
            bufs = [_apply_colorspace(b, input_transform) for b in bufs]

        image, mask, extras = oiio_to_multilayer(bufs)
        return IO.NodeOutput(image, mask, extras)


def _discover_input_range(
    kind: str, path: str, missing_policy: str
) -> "list[str | None]":
    """Detect first..last bounds from disk, then resolve gaps via policy.

    Differs from Custom Range only in how the (first, last) bounds are
    computed: here they come from min(frameSet()) / max(frameSet()) on
    disk; in Custom Range they come from the First/Last Frame knobs.
    Missing Frames policy then resolves any interior gaps the same way
    for both modes.
    """
    if kind == "literal_file":
        return [path]
    if kind == "pattern":
        seq = _sequence.find_sequence_on_disk(path)
        if seq is None:
            return []
        frames_on_disk = sorted(seq.frameSet())
        if not frames_on_disk:
            return []
        return _resolve_gaps(
            seq, frames_on_disk[0], frames_on_disk[-1], missing_policy
        )
    if kind == "literal_folder":
        seqs = _sequence.find_sequences_in_folder(path)
        if not seqs:
            raise RuntimeError(f"No sequences found in folder '{path}'.")
        if len(seqs) > 1:
            names = ", ".join(_seq_label(s) for s in seqs)
            raise RuntimeError(
                f"Folder '{path}' contains multiple sequences: {names}. "
                f"Point the Source field at one of them."
            )
        seq = seqs[0]
        frames_on_disk = sorted(seq.frameSet())
        if not frames_on_disk:
            return []
        return _resolve_gaps(
            seq, frames_on_disk[0], frames_on_disk[-1], missing_policy
        )
    raise RuntimeError(f"Path '{path}' does not exist.")


def _resolve_gaps(
    seq, first: int, last: int, missing_policy: str
) -> "list[str | None]":
    """Build the frame list from first..last, applying missing_policy on gaps."""
    frames = _sequence.expand_custom_range(first, last)
    available = set(seq.frameSet())

    if missing_policy == "error":
        mfr = _sequence.missing_frame_range(frames, available)
        if mfr:
            raise RuntimeError(
                f"Missing frames in sequence ({first}..{last}): {mfr}. "
                f"Choose a different Missing Frames policy or fill the gaps."
            )

    available_sorted = sorted(available)
    positions: list[str | None] = []
    for f in frames:
        if f in available:
            positions.append(seq.frame(f))
            continue
        if missing_policy in ("black", "checkerboard"):
            positions.append(None)
        elif missing_policy == "nearest":
            if not available_sorted:
                raise RuntimeError(
                    "No frames available for 'nearest' policy."
                )
            nearest = min(available_sorted, key=lambda af: (abs(af - f), af))
            positions.append(seq.frame(nearest))
        else:
            raise RuntimeError(f"Unknown Missing Frames policy: {missing_policy!r}")
    return positions


def _discover_custom_range(
    kind: str,
    path: str,
    first: int,
    last: int,
    missing_policy: str,
) -> "list[str | None]":
    if kind == "literal_file":
        return [path]

    if kind == "pattern":
        seq = _sequence.find_sequence_on_disk(path)
        if seq is None:
            raise RuntimeError(f"No sequence found on disk for '{path}'.")
    elif kind == "literal_folder":
        seqs = _sequence.find_sequences_in_folder(path)
        if not seqs:
            raise RuntimeError(f"No sequences found in folder '{path}'.")
        if len(seqs) > 1:
            names = ", ".join(_seq_label(s) for s in seqs)
            raise RuntimeError(
                f"Folder '{path}' contains multiple sequences: {names}. "
                f"Point the Source field at one of them."
            )
        seq = seqs[0]
    else:
        raise RuntimeError(f"Path '{path}' does not exist.")

    return _resolve_gaps(seq, first, last, missing_policy)


def _read_positions(
    positions: "list[str | None]",
    missing_policy: str,
) -> "list[oiio.ImageBuf]":
    cache: dict[str, oiio.ImageBuf] = {}
    for p in positions:
        if p is not None and p not in cache:
            cache[p] = _read_buf(p)

    # Synthesized gap frames must mirror a real frame's channel layout,
    # names included: oiio_to_multilayer keys AOV parsing off bufs[0], so a
    # leading gap carrying default channel names would drop every extra layer.
    if cache:
        ref_spec = next(iter(cache.values())).spec()
        nx, ny, nc = ref_spec.width, ref_spec.height, ref_spec.nchannels
        channelnames = list(ref_spec.channelnames)
    else:
        nx, ny, nc = _PLACEHOLDER_FRAME
        channelnames = None

    result: list[oiio.ImageBuf] = []
    for p in positions:
        if p is None:
            if missing_policy == "checkerboard":
                result.append(_make_checker_buf(nx, ny, nc, channelnames))
            else:
                spec = oiio.ImageSpec(nx, ny, nc, oiio.TypeFloat)
                if channelnames:
                    spec.channelnames = channelnames
                buf = oiio.ImageBuf(spec)
                oiio.ImageBufAlgo.zero(buf)
                result.append(buf)
        else:
            result.append(cache[p])
    return result


def _make_checker_buf(w: int, h: int, c: int, channelnames=None) -> "oiio.ImageBuf":
    sq = max(8, min(w, h) // 8)
    color1 = (1.0, 0.0, 1.0) + (1.0,) * max(c - 3, 0)
    color2 = (0.0, 0.0, 0.0) + (1.0,) * max(c - 3, 0)
    color1 = color1[:c]
    color2 = color2[:c]
    spec = oiio.ImageSpec(w, h, c, oiio.TypeFloat)
    if channelnames:
        spec.channelnames = channelnames
    buf = oiio.ImageBuf(spec)
    if not oiio.ImageBufAlgo.checker(buf, sq, sq, 1, color1, color2):
        oiio.ImageBufAlgo.zero(buf)
    return buf


def _read_buf(path: str) -> oiio.ImageBuf:
    buf = oiio.ImageBuf(path)
    buf.read()
    err = buf.geterror()
    if err:
        raise RuntimeError(f"Failed to read '{path}': {err}")
    return buf


def _apply_colorspace(buf: oiio.ImageBuf, transform: str) -> oiio.ImageBuf:
    """Convert from the user-picked OCIO colorspace to linear Rec.709.

    ``transform`` is a canonical OCIO colorspace name from the active
    config. When it equals linear Rec.709, the conversion is a no-op
    and the input buffer is returned unchanged.
    """
    if transform == _LINEAR_REC709:
        return buf
    result = oiio.ImageBufAlgo.colorconvert(
        buf,
        transform,
        _LINEAR_REC709,
        unpremult=True,
        colorconfig=_backends.get_color_config_uri(),
    )
    err = result.geterror()
    if err:
        raise RuntimeError(
            f"Colorspace conversion failed ({transform} -> linear Rec.709): {err}"
        )
    return result


def _seq_label(seq) -> str:
    return f"{seq.basename()}{seq.padding()}{seq.extension()}"
