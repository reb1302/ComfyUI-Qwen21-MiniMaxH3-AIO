"""Bridge helpers between ComfyUI tensors and OpenImageIO ImageBufs.

ComfyUI convention:
    IMAGE: torch.Tensor float32 in [0, 1], shape [B, H, W, C], C in {1, 3, 4}.
    MASK : torch.Tensor float32 in [0, 1], shape [B, H, W].

OIIO convention:
    ImageBuf with float32 pixels, layout [H, W, C], 1/3/4 channels.

Falls back to a lightweight NumPy-backed ImageBuf wrapper if OpenImageIO
is not installed yet so node import and basic execution still succeed.
"""
from __future__ import annotations

import numpy as np

try:
    import OpenImageIO as oiio
except ImportError:
    oiio = None  # type: ignore[assignment]


class _FallbackSpec:
    def __init__(self, width: int, height: int, nchannels: int, channelnames=None):
        self.width = int(width)
        self.height = int(height)
        self.nchannels = int(nchannels)
        default_names = ["R", "G", "B", "A"][: self.nchannels]
        while len(default_names) < self.nchannels:
            default_names.append(f"C{len(default_names)}")
        self.channelnames = list(channelnames) if channelnames else default_names


class _FallbackImageBuf:
    def __init__(self, pixels: np.ndarray, channelnames=None):
        arr = np.ascontiguousarray(pixels, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        self._pixels = arr
        h, w, c = arr.shape
        self._spec = _FallbackSpec(w, h, c, channelnames=channelnames)

    def spec(self) -> _FallbackSpec:
        return self._spec

    def get_pixels(self, *_args, **_kwargs) -> np.ndarray:
        return self._pixels

    def geterror(self) -> str:
        return ""


def _torch():
    """Lazy import of torch."""
    import torch
    return torch


def comfy_to_oiio(image) -> list:
    """Split a ComfyUI batch tensor [B, H, W, C] into a list of OIIO ImageBufs."""
    torch = _torch()
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image).__name__}")
    if image.ndim != 4:
        raise ValueError(f"Expected shape [B,H,W,C], got {tuple(image.shape)}")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels, got {image.shape[-1]}")

    arr = image.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    b, h, w, c = arr.shape
    bufs: list = []
    for i in range(b):
        if oiio is not None:
            spec = oiio.ImageSpec(w, h, c, oiio.TypeFloat)
            buf = oiio.ImageBuf(spec)
            if not buf.set_pixels(buf.roi, arr[i]):
                raise RuntimeError(f"OIIO set_pixels failed: {buf.geterror()}")
            bufs.append(buf)
        else:
            bufs.append(_FallbackImageBuf(arr[i]))
    return bufs


def oiio_to_comfy(bufs: list):
    """Stack OIIO ImageBuf list into a ComfyUI batch tensor [B, H, W, C]."""
    torch = _torch()
    if not bufs:
        raise ValueError("oiio_to_comfy received empty list")

    frames: list[np.ndarray] = []
    for buf in bufs:
        if oiio is not None and not isinstance(buf, _FallbackImageBuf):
            pixels = buf.get_pixels(oiio.TypeFloat)
        else:
            pixels = buf.get_pixels()
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
    """Group EXR channel names by layer prefix."""
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
    """Return channel indices ordered by Nuke convention for the layer."""
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
    """Aggregate forward/backward layers into a single 'motion' tensor."""
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


def oiio_to_multilayer(bufs: list):
    """Split an OIIO ImageBuf list into (image, mask, extra_layers)."""
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
        if oiio is not None and not isinstance(buf, _FallbackImageBuf):
            pixels = buf.get_pixels(oiio.TypeFloat)
        else:
            pixels = buf.get_pixels()
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

    motion = _synthesize_motion_layer(extra)
    if motion is not None and not any(k.lower() == "motion" for k in extra):
        extra["motion"] = motion

    return image_tensor, mask_tensor, extra
