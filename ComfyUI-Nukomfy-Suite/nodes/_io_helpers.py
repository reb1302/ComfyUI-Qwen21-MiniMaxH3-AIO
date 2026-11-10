"""Bridge helpers between ComfyUI tensors and OpenImageIO ImageBufs.

ComfyUI convention:
    IMAGE: torch.Tensor float32 in [0, 1], shape [B, H, W, C], C in {1, 3, 4}.
    MASK : torch.Tensor float32 in [0, 1], shape [B, H, W].

OIIO convention:
    ImageBuf with float32 pixels, layout [H, W, C], 1/3/4 channels.

These helpers do not make alpha-premultiplication decisions; they
preserve raw pixel values. Caller nodes (NukomfyRead/Write) own that
policy.
"""
from __future__ import annotations

import numpy as np
import OpenImageIO as oiio


def _torch():
    """Lazy import of torch.

    Torch is always present in the ComfyUI runtime. Lazy-loading it lets
    this module import cleanly on dev machines without a full torch
    install.
    """
    import torch
    return torch


def comfy_to_oiio(image) -> list[oiio.ImageBuf]:
    """Split a ComfyUI batch tensor [B, H, W, C] into a list of OIIO ImageBufs.

    Each ImageBuf carries float32 pixel data with layout [H, W, C]. The
    caller is responsible for tagging colorspace on the buffer afterwards.
    """
    torch = _torch()
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image).__name__}")
    if image.ndim != 4:
        raise ValueError(f"Expected shape [B,H,W,C], got {tuple(image.shape)}")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels, got {image.shape[-1]}")

    arr = image.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    b, h, w, c = arr.shape
    bufs: list[oiio.ImageBuf] = []
    for i in range(b):
        spec = oiio.ImageSpec(w, h, c, oiio.TypeFloat)
        buf = oiio.ImageBuf(spec)
        if not buf.set_pixels(buf.roi, arr[i]):
            raise RuntimeError(f"OIIO set_pixels failed: {buf.geterror()}")
        bufs.append(buf)
    return bufs


def oiio_to_comfy(bufs: list[oiio.ImageBuf]):
    """Stack OIIO ImageBuf list into a ComfyUI batch tensor [B, H, W, C].

    Preserves native channel count: 1ch stays 1ch, 3ch stays 3ch, 4ch
    stays 4ch. Channel count > 4 is truncated to the first 4 (multilayer
    EXR routing happens elsewhere). All buffers must share H, W and C.
    """
    torch = _torch()
    if not bufs:
        raise ValueError("oiio_to_comfy received empty list")

    frames: list[np.ndarray] = []
    for buf in bufs:
        pixels = buf.get_pixels(oiio.TypeFloat)
        if pixels is None:
            raise RuntimeError(f"OIIO get_pixels failed: {buf.geterror()}")
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        if pixels.ndim == 2:
            pixels = pixels[..., np.newaxis]
        if pixels.shape[-1] > 4:
            pixels = pixels[..., :4]
        frames.append(pixels)

    return torch.from_numpy(np.stack(frames, axis=0))


def parse_multilayer_channelnames(channelnames: list[str]) -> dict[str, list[tuple[int, str]]]:
    """Group EXR channel names by layer prefix.

    Each entry maps a layer name to the channel indices that compose it
    (with their suffix). The main layer (channels named R/G/B/A with no
    prefix) is keyed as the empty string "".

    Example:
        ["R", "G", "B", "A", "depth.Z", "crypto00.R", "crypto00.G"] ->
        {
            "":         [(0, "R"), (1, "G"), (2, "B"), (3, "A")],
            "depth":    [(4, "Z")],
            "crypto00": [(5, "R"), (6, "G")],
        }
    """
    groups: dict[str, list[tuple[int, str]]] = {}
    for idx, name in enumerate(channelnames):
        if "." in name:
            prefix, suffix = name.split(".", 1)
            groups.setdefault(prefix, []).append((idx, suffix))
        else:
            groups.setdefault("", []).append((idx, name))
    return groups


_MAIN_CANONICAL = ["R", "G", "B", "A"]
_POSITION_CANONICAL = ["X", "Y", "Z"]
_MOTION_VEC_CANONICAL = ["u", "v"]


def _ordered_channel_indices(layer_key: str, entries: list[tuple[int, str]]) -> list[int]:
    """Return channel indices ordered by Nuke convention for the layer.

    Recognizes the special-name conventions documented in
    multilayer_pack._channel_names_for_layer: depth/z (Z), mask (a),
    position/p (X/Y/Z), forward/backward (u/v). Falls back to canonical
    R/G/B/A order for the main layer and generic prefixes; unknown
    suffixes are appended in their original index order so no data is
    silently dropped.
    """
    key = layer_key.lower()
    by_suffix = {suffix: idx for idx, suffix in entries}
    if key in ("depth", "z") and len(entries) == 1 and "Z" in by_suffix:
        return [by_suffix["Z"]]
    if key == "mask" and len(entries) == 1 and "a" in by_suffix:
        return [by_suffix["a"]]
    if key in ("position", "p"):
        canonical = _POSITION_CANONICAL
    elif key in ("forward", "backward"):
        canonical = _MOTION_VEC_CANONICAL
    else:
        canonical = _MAIN_CANONICAL
    ordered = [by_suffix[s] for s in canonical if s in by_suffix]
    matched = set(canonical)
    for idx, suffix in entries:
        if suffix not in matched:
            ordered.append(idx)
    return ordered


def _synthesize_motion_layer(extra: dict):
    """Aggregate forward/backward layers into a single 'motion' tensor.

    Mirrors the Pack's motion write order
    (forward.u, forward.v, backward.u, backward.v) so that a user who
    typed "motion" as the layer name in the Pack can read it back by
    typing "motion" in the Unpack, regardless of whether forward and
    backward are also exposed separately. Returns the synthesized
    tensor or None when neither forward nor backward is present.
    """
    torch = _torch()
    fwd = None
    bwd = None
    for key, tensor in extra.items():
        lk = key.lower()
        if lk == "forward":
            fwd = tensor
        elif lk == "backward":
            bwd = tensor
    if fwd is None and bwd is None:
        return None
    parts = []
    if fwd is not None:
        parts.append(fwd)
    if bwd is not None:
        parts.append(bwd)
    return torch.cat(parts, dim=-1)


def oiio_to_multilayer(bufs: list[oiio.ImageBuf]):
    """Split an OIIO ImageBuf list into (image, mask, extra_layers).

    Returns:
        image (Tensor [B,H,W,3])      : main R/G/B
        mask  (Tensor [B,H,W])        : main alpha (zeros if no source)
        extra (dict[name -> Tensor])  : extra layers, each [B,H,W,Nc]

    Main-layer routing follows the standard rgba split. Extra layers are
    detected by parsing channelnames and grouping by prefix; each
    layer's channels are reordered to canonical positions for known
    Nuke conventions (see _ordered_channel_indices) so that a Pack/Unpack
    round-trip is bit-identical for the recognized layer names.

    The forward/backward layers are also exposed as a synthesized
    'motion' aggregate, mirroring the Pack motion write order. Users
    who typed "motion" in the Pack can therefore read it back by typing
    "motion" in the Unpack.
    """
    torch = _torch()
    if not bufs:
        raise ValueError("oiio_to_multilayer received empty list")

    first_spec = bufs[0].spec()
    channelnames = list(first_spec.channelnames)
    groups = parse_multilayer_channelnames(channelnames)
    main_indices = _ordered_channel_indices("", groups.get("", []))
    extra_prefixes = [p for p in groups if p != ""]

    image_frames: list[np.ndarray] = []
    mask_frames: list[np.ndarray] = []
    extra_frames: dict[str, list[np.ndarray]] = {p: [] for p in extra_prefixes}

    for buf in bufs:
        pixels = buf.get_pixels(oiio.TypeFloat)
        if pixels is None:
            raise RuntimeError(f"OIIO get_pixels failed: {buf.geterror()}")
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        if pixels.ndim == 2:
            pixels = pixels[..., np.newaxis]
        h, w, _ = pixels.shape

        if len(main_indices) >= 4:
            image = np.ascontiguousarray(pixels[..., main_indices[:3]])
            mask = np.ascontiguousarray(pixels[..., main_indices[3]])
        elif len(main_indices) == 3:
            image = np.ascontiguousarray(pixels[..., main_indices])
            mask = np.zeros((h, w), dtype=np.float32)
        elif len(main_indices) == 1:
            grayscale = pixels[..., main_indices]
            image = np.repeat(grayscale, 3, axis=-1)
            mask = np.zeros((h, w), dtype=np.float32)
        elif len(main_indices) == 2:
            image = np.ascontiguousarray(pixels[..., main_indices])
            mask = np.zeros((h, w), dtype=np.float32)
        else:
            image = np.zeros((h, w, 3), dtype=np.float32)
            mask = np.zeros((h, w), dtype=np.float32)
        image_frames.append(image)
        mask_frames.append(mask)

        for prefix in extra_prefixes:
            indices = _ordered_channel_indices(prefix, groups[prefix])
            layer = np.ascontiguousarray(pixels[..., indices])
            if layer.ndim == 2:
                layer = layer[..., np.newaxis]
            extra_frames[prefix].append(layer)

    image_tensor = torch.from_numpy(np.stack(image_frames, axis=0))
    mask_tensor = torch.from_numpy(np.stack(mask_frames, axis=0))
    extra: dict[str, "torch.Tensor"] = {
        name: torch.from_numpy(np.stack(frames, axis=0))
        for name, frames in extra_frames.items()
    }

    # Do not clobber a real 'motion' layer parsed from the file (any casing;
    # the Unpack lookup is case-insensitive). The forward/backward aggregate
    # is only a convenience for sources that lack a real motion layer.
    motion = _synthesize_motion_layer(extra)
    if motion is not None and not any(k.lower() == "motion" for k in extra):
        extra["motion"] = motion

    return image_tensor, mask_tensor, extra
