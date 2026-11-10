"""Lightricks LTX-2.5 Creative Lab bridge nodes for Nukomfy Suite.

Provides two companion nodes that sit between ``NukomfyRead`` and
``NukomfyWrite`` in LTX-2.5 Creative Lab workflows:

1. ``NukomfyLTX25BakeIn`` ("Nukomfy LTX-2.5 Sequence -> MP4 Bake"):
   - Receives the Nuke frame sequence ``[N, H, W, 3]`` (and optional mask
     ``[N, H, W]``) from ``NukomfyRead``.
   - Records the original frame count ``N``, original resolution ``(W, H)``,
     and original input ``base_colorspace``.
   - Converts pixels from ``base_colorspace`` to ``sRGB Encoded Rec.709 (sRGB)``
     via OpenImageIO + OCIO (so LTX-2.5's VAE receives proper sRGB [0..1]).
   - Pads the frame count from ``N`` to the nearest valid LTX-2.5 frame count
     ``8k + 1`` by repeating the last frame.
   - Aligns spatial dimensions ``(W, H)`` to a multiple of 64 (for Two-Stage
     Latent Upsampler workflows) or 32 (for Single-Stage workflows), and
     outputs both ``(full_width, full_height)`` and ``(half_width, half_height)``.
   - Bakes the ``8k + 1`` sRGB frames into a single high-bitrate H.264 ``.mp4``
     chunk (``crf 12``, ``yuv420p``) inside ``ComfyUI/input/nukomfy_ltx25/``
     and decodes the MP4 frames back so LTX-2.5 receives genuine MP4-baked
     frames (or ``baked_mp4_name`` can be wired to ``LoadVideo``).

2. ``NukomfyLTX25UnbakeOut`` ("Nukomfy LTX-2.5 MP4 -> Sequence Restore"):
   - Receives the generated sRGB frames ``[8k+1, H_aligned, W_aligned, 3]``
     from ``VAEDecodeTiled`` together with ``ltx25_context``.
   - Optionally bakes the output into an ``.mp4`` chunk and unpacks it back
     into frames.
   - Trims the ``8k + 1`` padded frames back to the exact original Nuke
     frame count ``N`` (``frames[:orig_frames]``).
   - Restores the exact original Nuke resolution ``(orig_width, orig_height)``.
   - Extracts a 1-channel ``restored_alpha_mask`` (luminance in sRGB space,
     ideal for ``alpha-gen-0.9`` IC-LoRA mattes).
   - Converts the RGB sequence from ``sRGB Encoded Rec.709 (sRGB)`` back to
     the exact ``base_colorspace`` of the Nuke input via OpenImageIO + OCIO
     so ``NukomfyWrite`` writes the EXR/DPX/PNG sequence in the original
     base colorspace.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile

import numpy as np
import OpenImageIO as oiio

from comfy_api.latest import IO

from . import _backends
from ._backends import _LINEAR_REC709
from ._io_helpers import _torch, comfy_to_oiio, oiio_to_comfy
from .ocio_colorspace import _colorconvert

_log = logging.getLogger(__name__)

_SRGB_CS_CANDIDATES = (
    "sRGB Encoded Rec.709 (sRGB)",
    "sRGB - Display",
    "Gamma 2.2 Encoded Rec.709",
)


def _get_srgb_colorspace() -> str:
    available = set(_backends.get_available_colorspaces())
    for candidate in _SRGB_CS_CANDIDATES:
        if candidate in available:
            return candidate
    return _LINEAR_REC709


def calc_ltx25_valid_frames(n: int) -> int:
    """LTX-2.5 requires frame count = 8k + 1 (1, 9, 17, 25, 33, 41, 49, 57, ...)."""
    if n <= 1:
        return 1
    return ((n - 1 + 7) // 8) * 8 + 1


def calc_aligned_dims(width: int, height: int, multiple: int = 64) -> tuple[int, int]:
    """Round width and height up to the nearest multiple of `multiple` (32 or 64)."""
    w_aligned = max(multiple, ((int(width) + multiple - 1) // multiple) * multiple)
    h_aligned = max(multiple, ((int(height) + multiple - 1) // multiple) * multiple)
    return w_aligned, h_aligned


def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return None


def _encode_frames_to_mp4(frames_u8: np.ndarray, mp4_path: str, fps: float = 24.0) -> bool:
    """Encode uint8 RGB frames [B, H, W, 3] to H.264 MP4 (crf 12, yuv420p).
    Tries PyAV (`av`) first, then `ffmpeg` CLI. Returns True on success.
    """
    os.makedirs(os.path.dirname(os.path.abspath(mp4_path)), exist_ok=True)
    b, h, w, _ = frames_u8.shape

    # Ensure even dimensions for yuv420p
    w_even = (w // 2) * 2
    h_even = (h // 2) * 2
    if w_even != w or h_even != h:
        frames_u8 = frames_u8[:, :h_even, :w_even, :]

    # 1. Try PyAV (installed in ComfyUI portable)
    try:
        import av
        from fractions import Fraction

        container = av.open(mp4_path, mode="w")
        try:
            rate = Fraction(int(round(fps * 1000)), 1000)
            stream = container.add_stream("libx264", rate=rate)
            stream.width = w_even
            stream.height = h_even
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "12", "preset": "fast"}

            for i in range(b):
                frame = av.VideoFrame.from_ndarray(frames_u8[i], format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        if os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 0:
            return True
    except Exception as exc:
        _log.debug("PyAV encode fallback to ffmpeg: %s", exc)

    # 2. Try ffmpeg CLI via rawvideo stdin pipe
    ffmpeg_bin = _find_ffmpeg()
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w_even}x{h_even}",
            "-pix_fmt", "rgb24",
            "-r", f"{fps:.4f}",
            "-i", "-",
            "-c:v", "libx264",
            "-crf", "12",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            mp4_path,
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=frames_u8.tobytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode == 0 and os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 0:
                return True
        except Exception as exc:
            _log.warning("ffmpeg MP4 bake failed: %s", exc)

    return False


def _decode_mp4_to_frames(mp4_path: str, target_w: int, target_h: int) -> np.ndarray | None:
    """Decode an MP4 file back to float32 RGB [B, H, W, 3] in [0, 1]."""
    if not os.path.isfile(mp4_path):
        return None
    try:
        import av

        container = av.open(mp4_path)
        decoded: list[np.ndarray] = []
        try:
            for frame in container.decode(video=0):
                if frame.width != target_w or frame.height != target_h:
                    frame = frame.reformat(width=target_w, height=target_h, format="rgb24")
                else:
                    frame = frame.to_rgb()
                arr = frame.to_ndarray().astype(np.float32) / 255.0
                decoded.append(arr)
        finally:
            container.close()
        if decoded:
            return np.stack(decoded, axis=0)
    except Exception as exc:
        _log.debug("PyAV decode fallback: %s", exc)
    return None


def _resize_tensor_bhwc(tensor, target_w: int, target_h: int):
    """Resize a [B, H, W, C] float32 tensor using OpenImageIO ImageBufAlgo.resize."""
    torch = _torch()
    b, h, w, c = tensor.shape
    if w == target_w and h == target_h:
        return tensor
    bufs = comfy_to_oiio(tensor)
    roi = oiio.ROI(0, int(target_w), 0, int(target_h), 0, 1, 0, int(c))
    resized_bufs = [oiio.ImageBufAlgo.resize(buf, roi=roi) for buf in bufs]
    return oiio_to_comfy(resized_bufs)


def _resize_mask_bhw(mask, target_w: int, target_h: int):
    """Resize a [B, H, W] float32 mask tensor using OpenImageIO ImageBufAlgo.resize."""
    b, h, w = mask.shape
    if w == target_w and h == target_h:
        return mask
    m4 = mask.unsqueeze(-1)
    resized4 = _resize_tensor_bhwc(m4, target_w, target_h)
    return resized4.squeeze(-1)


@IO.comfytype(io_type="NUKOMFY_LTX25_CTX")
class NukomfyLTX25Context(IO.ComfyTypeIO):
    """Metadata context connecting NukomfyLTX25BakeIn -> NukomfyLTX25UnbakeOut."""
    Type = dict


class NukomfyLTX25BakeIn(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        cs_list = _backends.get_available_colorspaces()
        default_cs = _LINEAR_REC709 if _LINEAR_REC709 in cs_list else (
            cs_list[0] if cs_list else ""
        )
        return IO.Schema(
            node_id="NukomfyLTX25BakeIn",
            display_name="Nukomfy LTX-2.5 Sequence -> MP4 Bake",
            category="Nukomfy/LTX-2.5",
            description=(
                "Preprocesses a Nuke image sequence from NukomfyRead for "
                "Lightricks LTX-2.5 Creative Lab workflows:\n\n"
                "1. Records original frame count N, resolution (W, H), and "
                "Input Base Colorspace.\n"
                "2. Converts from Input Base Colorspace to sRGB Encoded Rec.709.\n"
                "3. Pads frame count to 8k + 1 (repeating the last frame) and "
                "aligns resolution to a multiple of 64 (or 32).\n"
                "4. Bakes the sequence into a single H.264 MP4 chunk (crf 12) "
                "and outputs the sRGB frames, aligned dimensions, and LTX-2.5 "
                "context for NukomfyLTX25UnbakeOut."
            ),
            inputs=[
                IO.Image.Input("image", tooltip="RGB sequence from NukomfyRead."),
                IO.Combo.Input(
                    "base_colorspace",
                    display_name="Input Base Colorspace",
                    options=cs_list,
                    default=default_cs,
                    tooltip=(
                        "Colorspace of the incoming tensor from NukomfyRead.\n\n"
                        "• If NukomfyRead has 'Apply Color Transform' OFF on a "
                        "linear EXR cache, leave at 'Linear Rec.709 (sRGB)' or "
                        "select the plate's native colorspace (e.g. 'ACEScg', "
                        "'ARRI LogC3 (EI800)').\n"
                        "• NukomfyLTX25UnbakeOut will automatically restore the "
                        "output back to this exact Base Colorspace!"
                    ),
                ),
                IO.Float.Input(
                    "fps",
                    display_name="Frame Rate (FPS)",
                    default=24.0,
                    min=1.0,
                    max=120.0,
                    step=0.01,
                    tooltip="Frame rate used when baking the MP4 chunk.",
                ),
                IO.Combo.Input(
                    "align_multiple",
                    display_name="Resolution Alignment",
                    options=["64 (Two-Stage Upscale)", "32 (Single-Stage)"],
                    default="64 (Two-Stage Upscale)",
                    tooltip=(
                        "LTX-2.5 requires dimensions divisible by 32 at every stage.\n"
                        "Use 64 for Two-Stage workflows (so half-res stage 1 is "
                        "also divisible by 32)."
                    ),
                ),
                IO.Boolean.Input(
                    "bake_mp4_chunk",
                    display_name="Bake MP4 Chunk",
                    default=True,
                    tooltip=(
                        "When enabled, bakes the padded sRGB sequence into a "
                        "single H.264 .mp4 file (crf 12, yuv420p) in "
                        "ComfyUI/input/nukomfy_ltx25/ and reads it back."
                    ),
                ),
                IO.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Optional mask/alpha sequence from NukomfyRead.",
                ),
            ],
            outputs=[
                IO.Image.Output(
                    "srgb_images",
                    display_name="srgb_images",
                    tooltip="Padded (8k+1) sRGB image batch ready for LTX-2.5.",
                ),
                IO.Mask.Output(
                    "padded_mask",
                    display_name="padded_mask",
                    tooltip="Padded (8k+1) mask batch matching srgb_images.",
                ),
                NukomfyLTX25Context.Output(
                    "ltx25_context",
                    display_name="ltx25_context",
                    tooltip="Connect to NukomfyLTX25UnbakeOut to restore frame count, size, and base colorspace.",
                ),
                IO.Int.Output(
                    "padded_length",
                    display_name="padded_length",
                    tooltip="Valid LTX-2.5 frame count (8k + 1). Connect to EmptyLTXVLatentVideo.length and LTXVEmptyLatentAudio.frames_number.",
                ),
                IO.Int.Output(
                    "full_width",
                    display_name="full_width",
                    tooltip="Aligned full-res width (multiple of 64/32).",
                ),
                IO.Int.Output(
                    "full_height",
                    display_name="full_height",
                    tooltip="Aligned full-res height (multiple of 64/32).",
                ),
                IO.Int.Output(
                    "half_width",
                    display_name="half_width",
                    tooltip="Stage-1 half-res width (multiple of 32).",
                ),
                IO.Int.Output(
                    "half_height",
                    display_name="half_height",
                    tooltip="Stage-1 half-res height (multiple of 32).",
                ),
                IO.String.Output(
                    "baked_mp4_name",
                    display_name="baked_mp4_name",
                    tooltip="Relative path of the baked .mp4 inside ComfyUI/input/ (can be wired to LoadVideo).",
                ),
            ],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls,
        image,
        base_colorspace: str,
        fps: float,
        align_multiple: str,
        bake_mp4_chunk: bool,
        mask=None,
    ) -> IO.NodeOutput:
        torch = _torch()
        _backends.validate_colorspace(base_colorspace)
        srgb_cs = _get_srgb_colorspace()

        orig_b, orig_h, orig_w, _ = image.shape
        mult = 64 if str(align_multiple).startswith("64") else 32
        full_w, full_h = calc_aligned_dims(orig_w, orig_h, multiple=mult)
        half_w = max(32, (full_w // 2 // 32) * 32)
        half_h = max(32, (full_h // 2 // 32) * 32)
        padded_len = calc_ltx25_valid_frames(orig_b)

        # 1. Convert from base_colorspace -> sRGB Encoded Rec.709 (sRGB)
        if base_colorspace != srgb_cs:
            bufs = comfy_to_oiio(image)
            bufs = [_colorconvert(b, base_colorspace, srgb_cs) for b in bufs]
            srgb_img = oiio_to_comfy(bufs)
        else:
            srgb_img = image.clone()

        srgb_img = torch.clamp(srgb_img, 0.0, 1.0)

        # 2. Align spatial resolution to (full_w, full_h)
        srgb_img = _resize_tensor_bhwc(srgb_img, full_w, full_h)

        # 3. Pad frame count from orig_b -> padded_len (8k + 1) by repeating last frame
        if padded_len > orig_b:
            pad_count = padded_len - orig_b
            last_frame = srgb_img[-1:].repeat(pad_count, 1, 1, 1)
            srgb_img = torch.cat([srgb_img, last_frame], dim=0)

        # 4. Handle optional mask (resize + pad to 8k + 1)
        if mask is None:
            padded_mask = torch.zeros((padded_len, full_h, full_w), dtype=torch.float32)
        else:
            m = mask.detach().cpu().float()
            if m.ndim == 2:
                m = m.unsqueeze(0)
            if m.shape[0] == 1 and orig_b > 1:
                m = m.repeat(orig_b, 1, 1)
            m = _resize_mask_bhw(m, full_w, full_h)
            if padded_len > m.shape[0]:
                pad_m = m[-1:].repeat(padded_len - m.shape[0], 1, 1)
                m = torch.cat([m, pad_m], dim=0)
            elif m.shape[0] > padded_len:
                m = m[:padded_len]
            padded_mask = torch.clamp(m, 0.0, 1.0)

        # 5. Bake into 1 single MP4 chunk in ComfyUI/input/nukomfy_ltx25/
        baked_rel_name = ""
        baked_abs_path = ""
        if bake_mp4_chunk:
            try:
                import folder_paths
                input_dir = folder_paths.get_input_directory()
            except Exception:
                input_dir = os.path.join(tempfile.gettempdir(), "nukomfy_comfy_input")

            sub_dir = os.path.join(input_dir, "nukomfy_ltx25")
            os.makedirs(sub_dir, exist_ok=True)

            arr_u8 = (
                (srgb_img.detach().cpu().numpy() * 255.0)
                .round()
                .clip(0, 255)
                .astype(np.uint8)
            )
            digest = hashlib.sha1(
                arr_u8[0].tobytes() + f"_{orig_b}_{padded_len}_{full_w}x{full_h}_{fps}".encode()
            ).hexdigest()[:12]
            fname = f"nukomfy_in_{digest}.mp4"
            baked_abs_path = os.path.join(sub_dir, fname)
            baked_rel_name = f"nukomfy_ltx25/{fname}"

            if _encode_frames_to_mp4(arr_u8, baked_abs_path, fps=fps):
                decoded = _decode_mp4_to_frames(baked_abs_path, full_w, full_h)
                if decoded is not None and decoded.shape[0] == padded_len:
                    srgb_img = torch.from_numpy(decoded)

        ctx = {
            "orig_frames": int(orig_b),
            "orig_width": int(orig_w),
            "orig_height": int(orig_h),
            "padded_frames": int(padded_len),
            "aligned_width": int(full_w),
            "aligned_height": int(full_h),
            "base_colorspace": str(base_colorspace),
            "srgb_colorspace": str(srgb_cs),
            "fps": float(fps),
            "baked_mp4_path": baked_abs_path,
            "baked_mp4_name": baked_rel_name,
        }

        return IO.NodeOutput(
            srgb_img,
            padded_mask,
            ctx,
            int(padded_len),
            int(full_w),
            int(full_h),
            int(half_w),
            int(half_h),
            baked_rel_name,
        )


class NukomfyLTX25UnbakeOut(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NukomfyLTX25UnbakeOut",
            display_name="Nukomfy LTX-2.5 MP4 -> Sequence Restore",
            category="Nukomfy/LTX-2.5",
            description=(
                "Postprocesses LTX-2.5 Creative Lab output frames before "
                "NukomfyWrite writes the sequence back to Nuke:\n\n"
                "1. Optionally bakes the generated frames to an .mp4 chunk "
                "and unpacks them back into a frame sequence.\n"
                "2. Strips the 8k + 1 tail padding back to the exact original "
                "Nuke frame count N.\n"
                "3. Restores the exact original Nuke resolution (orig_width, "
                "orig_height).\n"
                "4. Extracts a 1-channel alpha matte (restored_alpha_mask) "
                "from the sRGB luminance (for Alpha-Gen IC-LoRA workflows).\n"
                "5. Converts colorspace from sRGB back to the exact Input Base "
                "Colorspace recorded by NukomfyLTX25BakeIn!"
            ),
            inputs=[
                IO.Image.Input("images", tooltip="Generated sRGB frames from VAEDecodeTiled."),
                NukomfyLTX25Context.Input(
                    "ltx25_context",
                    tooltip="Context from NukomfyLTX25BakeIn.",
                ),
                IO.Boolean.Input(
                    "bake_output_mp4",
                    display_name="Bake & Unpack Output MP4",
                    default=True,
                    tooltip=(
                        "Bakes the LTX-2.5 output to an .mp4 file and converts "
                        "it back into the output frame sequence."
                    ),
                ),
                IO.Boolean.Input(
                    "restore_base_colorspace",
                    display_name="Restore Input Base Colorspace",
                    default=True,
                    tooltip=(
                        "Converts the output sequence from sRGB back to the "
                        "exact Input Base Colorspace recorded by "
                        "NukomfyLTX25BakeIn."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    "restored_images",
                    display_name="restored_images",
                    tooltip="Restored image sequence in original frame count, resolution, and Base Colorspace. Connect to NukomfyWrite.image.",
                ),
                IO.Mask.Output(
                    "restored_alpha_mask",
                    display_name="restored_alpha_mask",
                    tooltip="1-channel luminance mask (in sRGB space) trimmed to original frame count and resolution.",
                ),
                IO.String.Output(
                    "output_mp4_path",
                    display_name="output_mp4_path",
                    tooltip="Path to the baked output .mp4 chunk.",
                ),
            ],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls,
        images,
        ltx25_context: dict,
        bake_output_mp4: bool,
        restore_base_colorspace: bool,
    ) -> IO.NodeOutput:
        torch = _torch()
        ctx = ltx25_context or {}
        orig_frames = int(ctx.get("orig_frames", images.shape[0]))
        orig_w = int(ctx.get("orig_width", images.shape[2]))
        orig_h = int(ctx.get("orig_height", images.shape[1]))
        base_cs = str(ctx.get("base_colorspace", _LINEAR_REC709))
        srgb_cs = str(ctx.get("srgb_colorspace", _get_srgb_colorspace()))
        fps = float(ctx.get("fps", 24.0))

        out_tensor = torch.clamp(images.detach().cpu().float(), 0.0, 1.0)

        # 1. Optional output MP4 bake -> unpack to frame sequence
        out_mp4_path = ""
        if bake_output_mp4:
            try:
                import folder_paths
                out_dir = os.path.join(folder_paths.get_output_directory(), "nukomfy_ltx25_mp4")
            except Exception:
                out_dir = os.path.join(tempfile.gettempdir(), "nukomfy_ltx25_mp4")
            os.makedirs(out_dir, exist_ok=True)

            arr_u8 = (
                (out_tensor.numpy() * 255.0)
                .round()
                .clip(0, 255)
                .astype(np.uint8)
            )
            digest = hashlib.sha1(arr_u8[0].tobytes()).hexdigest()[:12]
            out_mp4_path = os.path.join(out_dir, f"nukomfy_out_{digest}.mp4")
            cur_h, cur_w = out_tensor.shape[1], out_tensor.shape[2]
            if _encode_frames_to_mp4(arr_u8, out_mp4_path, fps=fps):
                decoded = _decode_mp4_to_frames(out_mp4_path, cur_w, cur_h)
                if decoded is not None and decoded.shape[0] >= orig_frames:
                    out_tensor = torch.from_numpy(decoded)

        # 2. Trim 8k + 1 padded tail frames back to exact Nuke frame count
        if out_tensor.shape[0] > orig_frames:
            out_tensor = out_tensor[:orig_frames]

        # 3. Restore exact original Nuke resolution (orig_w, orig_h)
        out_tensor = _resize_tensor_bhwc(out_tensor, orig_w, orig_h)

        # 4. Extract 1-channel alpha matte from sRGB luminance (for Alpha-Gen IC-LoRA)
        # Rec.709 luma weights on sRGB output
        alpha_mask = (
            0.2126 * out_tensor[..., 0]
            + 0.7152 * out_tensor[..., 1]
            + 0.0722 * out_tensor[..., 2]
        ).clamp(0.0, 1.0)

        # 5. Restore original Input Base Colorspace via OpenImageIO + OCIO
        if restore_base_colorspace and base_cs != srgb_cs:
            _backends.validate_colorspace(base_cs)
            _backends.validate_colorspace(srgb_cs)
            bufs = comfy_to_oiio(out_tensor)
            bufs = [_colorconvert(b, srgb_cs, base_cs) for b in bufs]
            restored_images = oiio_to_comfy(bufs)
        else:
            restored_images = out_tensor

        return IO.NodeOutput(restored_images, alpha_mask, out_mp4_path)
