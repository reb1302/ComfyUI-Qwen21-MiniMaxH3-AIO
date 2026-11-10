"""Pack named extra image layers for EXR multilayer write.

Bundles N extra IMAGE layers (depth, normal, mask, motion, ...) into
a single NUKOMFY_MULTILAYER value. The output is consumed by the
optional ``extra_layers`` socket of NukomfyWrite, which produces a
multilayer EXR compatible with Nuke.

The main rgba/rgb layer is NOT packed here: it travels via the standard
``image`` socket of NukomfyWrite. The Pack rejects ``rgba`` and ``rgb``
as layer names so users don't accidentally route the main beauty pass
through the extra channel.

Pattern: DynamicCombo + count knob.
Each option exposes N pairs of (IMAGE socket, layer_name string).
Frontend reveals only the inputs for the selected count.
"""
from __future__ import annotations

import logging
import re

from comfy_api.latest import IO

_log = logging.getLogger(__name__)


# NUKOMFY_MULTILAYER io_type registered here. Kept as a plain dict for
# wire format; the keys are user-chosen layer names, values are IMAGE
# tensors with shape [B, H, W, C]. All tensors in a single pack share
# B, H, W; per-layer C may differ (e.g. normal=3, depth=1, motion=2).
@IO.comfytype(io_type="NUKOMFY_MULTILAYER")
class NukomfyMultilayer(IO.ComfyTypeIO):
    """Pack of named extra image layers for EXR multilayer write."""
    Type = dict


_TT_LAYER_COUNT = (
    "How many extra layers to pack.\n\n"
    "Choose the number of additional passes (depth, normal, mask, "
    "motion, …) you want to bundle into the multilayer EXR. The "
    "main rgba layer is not packed here: it goes to the image socket "
    "of Nukomfy Write directly."
)
_TT_LAYER_IMAGE = (
    "Image tensor for this extra layer.\n\n"
    "Must share resolution (height, width) with every other layer "
    "in the pack and with the main image passed to Nukomfy Write."
)
_TT_LAYER_NAME = (
    "Layer name for the EXR channel prefix. Recognized names (depth, "
    "mask, position, forward, backward, motion) follow Nuke channel "
    "conventions when the input channel count matches, for example "
    "a 1-channel mask becomes mask.a. Other names or channel counts "
    "fall back to <name>.R/.G/.B[/.A].\n\n"
    "Empty -> layer_<slot>. Reserved: rgba, rgb. "
    "Letters, digits, underscore only."
)


# Reject characters that OpenEXR or common filesystems dislike, plus
# whitespace. Allow only ASCII letters, digits and underscore.
_LAYER_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
_RESERVED_NAMES = frozenset({"rgba", "rgb"})

# Upper bound for the layer_count dropdown. Covers the heaviest VFX
# pipelines (multi-channel cryptomatte + depth + normal + motion +
# diffuse/specular + custom AOVs) with room to spare. Increase here if
# users hit the cap in practice.
_MAX_LAYERS = 16


def _generate_layer_inputs(count: int) -> list:
    """Generate ``count`` pairs of (layer_name string, image socket) inputs.

    Name comes first (top) so the user types the label before wiring
    the socket below it. Names are empty by default so each layer name
    must be filled explicitly, avoiding silent reuse across workflows.
    """
    inputs = []
    for i in range(1, count + 1):
        inputs.append(IO.String.Input(
            f"layer_{i}_name",
            display_name=f"Layer {i} Name",
            default="",
            placeholder=f"layer_{i}",
            tooltip=_TT_LAYER_NAME,
        ))
        # Socket name follows Comfy convention (lowercase snake_case).
        # No "_image" suffix: the blue socket color already conveys IMAGE.
        inputs.append(IO.Image.Input(
            f"layer_{i}",
            tooltip=_TT_LAYER_IMAGE,
        ))
    return inputs


def _validate_layer_name(raw: str, slot_index: int, taken: set[str]) -> str:
    """Strip and validate a user-supplied layer name. Returns the canonical form.

    When the name is empty or whitespace, falls back to ``layer_{i}``
    where ``i`` is the slot index (1-based). The fallback name is still
    checked for duplication against ``taken`` to catch the rare case
    where the user explicitly wrote ``layer_2`` in another slot.

    Raises RuntimeError with a clear, actionable message when invalid.
    """
    if raw is None:
        raw = ""
    name = str(raw).strip()
    if not name:
        # Fallback to slot-indexed default name.
        name = f"layer_{slot_index}"
        if name.lower() in taken:
            raise RuntimeError(
                f"Layer {slot_index} name is empty and its default "
                f"fallback '{name}' is already used by another slot. "
                f"Provide an explicit name."
            )
        return name
    if name.lower() in _RESERVED_NAMES:
        raise RuntimeError(
            f"Layer {slot_index} name '{name}' is reserved. Use the main "
            f"'image' socket of Nukomfy Write for rgba/rgb. The Pack is "
            f"for additional passes only."
        )
    if not _LAYER_NAME_RE.match(name):
        raise RuntimeError(
            f"Layer {slot_index} name '{name}' contains illegal characters. "
            f"Use only letters, digits and underscore."
        )
    # Case-insensitive dedup: the Unpack lookup lowercases names, so 'Foo'
    # and 'foo' must not both be accepted here (one would be unreachable).
    if name.lower() in taken:
        raise RuntimeError(
            f"Layer {slot_index} name '{name}' is duplicated. Each layer "
            f"in a single Pack must have a unique name."
        )
    return name


def _validate_image_shape(image, slot_index: int, ref_bhw: tuple | None) -> tuple:
    """Validate that ``image`` is a 4D torch.Tensor [B, H, W, C].

    Returns (B, H, W). When ``ref_bhw`` is not None, also enforces match.
    """
    # Lazy torch import: matches the convention in _io_helpers.py and lets
    # this module import on dev machines without torch installed.
    import torch
    if not isinstance(image, torch.Tensor):
        raise RuntimeError(
            f"Layer {slot_index} input is not a valid image. "
            f"Connect an image output to it."
        )
    if image.ndim != 4:
        raise RuntimeError(
            f"Layer {slot_index} image has shape {tuple(image.shape)}, "
            f"expected [B, H, W, C]."
        )
    b, h, w, _c = image.shape
    if ref_bhw is not None:
        rb, rh, rw = ref_bhw
        if (b, h, w) != (rb, rh, rw):
            raise RuntimeError(
                f"Layer {slot_index} resolution {w}x{h} does not match "
                f"{rw}x{rh} from the first layer.\n"
                f"All layers in a Pack must share resolution and frame count."
            )
    return (b, h, w)


def _apply_semantic_channel_reduction(name: str, image):
    """Reduce the IMAGE channel count to match Nuke conventions for the layer name.

    Nuke convention: ``depth`` and ``z`` are single-channel passes
    (one ``Z`` per pixel). Many upstream nodes - including NukomfyRead
    on a 1-channel EXR source - promote the tensor to 4ch RGBA with
    R==G==B==Z. To restore the Nuke-friendly ``depth.Z`` channel name
    instead of a 4-channel ``depth.R/.G/.B/.A`` layer, this helper
    reduces a ``depth``/``z`` layer to a single channel.

    Depth data usually lives in R (grayscale R==G==B), but some graphs
    route it to G/B/A with the other channels zeroed by a shuffle.
    Picking R blindly would silently drop the data in those cases, so
    the first channel carrying data is taken instead, scanning
    R -> G -> B -> A. A channel counts as empty only when it is exactly
    all-zero, so a populated R (even noisy) is never overridden; the
    scan advances only past channels a shuffle deliberately blanked.
    Fully black images fall back to R.

    Motion and other layers are passed through unchanged: the channel
    count comes from the upstream tensor and the channel name from
    ``_channel_names_for_layer``.
    """
    key = name.lower()
    if key in ("depth", "z") and image.shape[-1] > 1:
        # Take the first channel carrying data; slice keeps shape [B, H, W, 1].
        for c in range(image.shape[-1]):
            if image[..., c].abs().max() > 0:
                return image[..., c:c + 1]
        return image[..., :1]
    return image


def _build_pack(layer_count: dict) -> dict:
    """Assemble and validate the NUKOMFY_MULTILAYER dict from raw inputs.

    ``layer_count`` is the DynamicCombo runtime payload: the selected
    option key under ``layer_count["layer_count"]`` (e.g. "3 layers")
    plus the per-slot values under ``layer_count["layer_{i}_image"]``
    and ``layer_count["layer_{i}_name"]``.
    """
    key = layer_count.get("layer_count", "")
    try:
        count = int(str(key).split()[0])
    except (ValueError, IndexError):
        raise RuntimeError(
            f"Invalid Layer Count value: {key}. Expected something "
            f"like '1 layer' or '3 layers'."
        )
    if count < 1:
        raise RuntimeError(
            "Layer Count must be at least 1."
        )

    pack: dict = {}
    taken: set[str] = set()
    ref_bhw: tuple | None = None
    for i in range(1, count + 1):
        raw_name = layer_count.get(f"layer_{i}_name", "")
        name = _validate_layer_name(raw_name, i, taken)
        image = layer_count.get(f"layer_{i}")
        if image is None:
            raise RuntimeError(
                f"Layer {i} has no image connected. Connect an "
                f"image output to every layer slot."
            )
        ref_bhw = _validate_image_shape(image, i, ref_bhw)
        image = _apply_semantic_channel_reduction(name, image)
        pack[name] = image
        taken.add(name.lower())
    return pack


class NukomfyMultiLayerPack(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NukomfyMultiLayerPack",
            display_name="Nukomfy Multilayer Pack",
            category="Nukomfy/Channels",
            description=(
                "Pack named extra image layers (depth, normal, mask, "
                "motion, …) into a single bundle.\n\n"
                "Pipe the output "
                "into the extra_layers socket of Nukomfy Write to "
                "produce a multilayer EXR compatible with Nuke.\n\n"
                "The main rgba/rgb layer goes to the Nukomfy Write "
                "image socket directly."
            ),
            inputs=[
                IO.DynamicCombo.Input(
                    "layer_count",
                    display_name="Layer Count",
                    options=[
                        IO.DynamicCombo.Option(
                            f"{n} layer" if n == 1 else f"{n} layers",
                            _generate_layer_inputs(n),
                        )
                        for n in range(1, _MAX_LAYERS + 1)
                    ],
                    tooltip=_TT_LAYER_COUNT,
                ),
            ],
            outputs=[
                NukomfyMultilayer.Output("extra_layers"),
            ],
        )

    @classmethod
    def execute(cls, layer_count: dict) -> IO.NodeOutput:
        pack = _build_pack(layer_count)
        return IO.NodeOutput(pack)


# === Channel naming helper (Nuke EXR convention) ===
#
# Lives here next to the pack node so the multilayer convention is owned
# by one module: NukomfyWrite imports this helper when assembling EXR
# multilayer channelnames. Pure stdlib, no Comfy/OIIO dependency.
def _channel_names_for_layer(layer_name: str, n_channels: int) -> list[str]:
    """Return OpenEXR channel names for a layer, following Nuke convention.

    Cases:
      rgba / rgb           : no prefix (Nuke's main layer)
      depth / z            : <layer>.Z when n_channels == 1
      mask                 : <layer>.a when n_channels == 1
                             (Nuke reserved channel set for mattes)
      position / p         : <layer>.X/.Y/.Z when n_channels == 3
                             (Nuke spatial position layer convention)
      motion (1..4 ch)     : Nuke "motion" channel set, canonical
                             order forward.u, forward.v, backward.u,
                             backward.v. The first N channels are
                             taken for each input width:
                               1 ch -> forward.u
                               2 ch -> forward-only
                               3 ch -> forward + bwd.u
                               4 ch -> full forward + backward
                             Input is packed in RGBA order (R=fwd.u,
                             G=fwd.v, B=bwd.u, A=bwd.v).
      forward / backward   : <layer>.u/.v when n_channels == 2
                             (Nuke 2D motion vector layer convention)
      other 1 ch           : <layer>.R
      other 3 ch           : <layer>.R/.G/.B
      other 4 ch           : <layer>.R/.G/.B/.A

    The case-insensitive lookup applies only to the special names
    (rgba/rgb/depth/z/mask/position/p/motion/forward/backward).
    Generic layer names preserve the caller-supplied casing for the
    channel prefix.
    """
    key = layer_name.lower()
    if key == "rgba":
        if n_channels == 4:
            return ["R", "G", "B", "A"]
        if n_channels == 3:
            return ["R", "G", "B"]
    if key == "rgb":
        if n_channels == 3:
            return ["R", "G", "B"]
    if key in ("depth", "z"):
        if n_channels == 1:
            return [f"{layer_name}.Z"]
    if key == "mask":
        if n_channels == 1:
            return [f"{layer_name}.a"]
    if key in ("position", "p"):
        if n_channels == 3:
            return [f"{layer_name}.X", f"{layer_name}.Y", f"{layer_name}.Z"]
    if key == "motion":
        # Nuke "motion" channel set: forward + backward 2D vectors,
        # canonical order forward.u, forward.v, backward.u, backward.v.
        # The first N channels of that sequence are taken for each
        # input width (1 ch = fwd.u; 2 ch = forward-only;
        # 3 ch = forward + bwd.u; 4 ch = full forward + backward).
        _MOTION_CANONICAL = ["forward.u", "forward.v", "backward.u", "backward.v"]
        if 1 <= n_channels <= 4:
            return _MOTION_CANONICAL[:n_channels]
    if key in ("forward", "backward"):
        if n_channels == 2:
            return [f"{layer_name}.u", f"{layer_name}.v"]
    # Generic fallback by channel count. Nuke reads at most 8 channels per
    # layer, so 1..8 map to R/G/B/A plus zero-padded channelNN beyond the
    # fourth (padded so the alphabetical channel order EXR stores matches the
    # numeric order on read-back). More than 8 cannot round-trip: Nuke drops
    # the excess silently, so refuse it with an actionable message instead.
    if 1 <= n_channels <= 8:
        rgba = ["R", "G", "B", "A"]
        names = [f"{layer_name}.{rgba[i]}" for i in range(min(n_channels, 4))]
        names += [f"{layer_name}.channel{i:02d}" for i in range(4, n_channels)]
        return names
    raise RuntimeError(
        f"Layer '{layer_name}' has {n_channels} channels. Nuke supports at "
        f"most 8 channels per layer. Split it into multiple layers of 8 "
        f"channels or fewer."
    )


# ============================================================
# NukomfyMultiLayerUnpack - inverse of NukomfyMultiLayerPack
# ============================================================

_TT_UNPACK_LAYER_COUNT = (
    "How many extra layers to extract.\n\n"
    "Add one slot per layer you want to read out of the multilayer "
    "bundle. The output socket layer_N exposes the layer matched by "
    "Layer N Name."
)
_TT_UNPACK_LAYER_NAME = (
    "Name of the layer to extract (case-insensitive). Any layer in "
    "the bundle can be addressed by name, for example depth, normal, "
    "position or motion.\n\n"
    "Missing or misnamed layers fall back to 3-channel zeros."
)


def _generate_unpack_inputs(count: int) -> list:
    """Generate ``count`` layer_name string inputs for the Unpack node.

    Unlike the Pack (which has a name + image socket per slot), the
    Unpack only needs the name: the image is the output, exposed as a
    static schema socket layer_N.
    """
    inputs = []
    for i in range(1, count + 1):
        inputs.append(IO.String.Input(
            f"layer_{i}_name",
            display_name=f"Layer {i} Name",
            default="",
            placeholder="depth, normal, position, …",
            tooltip=_TT_UNPACK_LAYER_NAME,
        ))
    return inputs


def _unpack_fallback_shape(bundle: dict) -> tuple:
    """Return [B, H, W, 3] derived from the first tensor in the bundle.

    Fallback to [1, 1, 1, 3] when the bundle has no layers (e.g. a
    non-EXR source connected upstream).
    """
    for tensor in bundle.values():
        try:
            if tensor.ndim == 4:
                b, h, w, _c = tensor.shape
                return (int(b), int(h), int(w), 3)
        except AttributeError:
            continue
    return (1, 1, 1, 3)


def _promote_to_image(tensor):
    """Promote a single-channel tensor to 3-channel by zero-padding G and B.

    The original channel lands on R; G and B are zero. This preserves
    the source data on a slot that a downstream Pack's semantic
    reduction (_apply_semantic_channel_reduction) will recover, while
    avoiding the grayscale "fake" of replicating R=G=B. Tensors with
    2, 3 or 4 channels pass through untouched.
    """
    import torch
    if tensor.ndim == 4 and tensor.shape[-1] == 1:
        zeros = torch.zeros_like(tensor)
        return torch.cat([tensor, zeros, zeros], dim=-1)
    return tensor


def _unpack_bundle(extra_layers, layer_count: dict) -> list:
    """Build the list of output tensors for NukomfyMultiLayerUnpack.

    Returns exactly _MAX_LAYERS tensors so the runtime can match the
    static output sockets. Slots beyond the user-selected layer_count
    are placeholder 1x1x1x3 zeros (downstream nodes typically leave
    them unconnected).

    Single-channel layers (depth.Z, mask.a) are promoted to 3-channel
    on the output socket so Preview Image and standard AI nodes can
    consume them directly. The semantic-reduction step on a downstream
    Pack restores the single-channel shape via _apply_semantic_channel_reduction.
    """
    key = layer_count.get("layer_count", "")
    try:
        count = int(str(key).split()[0])
    except (ValueError, IndexError):
        raise RuntimeError(
            f"Invalid Layer Count value: {key}. Expected something "
            f"like '1 layer' or '3 layers'."
        )
    if count < 1:
        raise RuntimeError(
            "Layer Count must be at least 1."
        )

    import torch
    bundle = extra_layers if isinstance(extra_layers, dict) else {}
    lookup = {k.lower(): v for k, v in bundle.items()}
    fallback_bhwc = _unpack_fallback_shape(bundle)

    tensors: list = []
    for i in range(1, _MAX_LAYERS + 1):
        if i > count:
            tensors.append(torch.zeros(1, 1, 1, 3, dtype=torch.float32))
            continue
        raw_name = layer_count.get(f"layer_{i}_name", "") or ""
        name = str(raw_name).strip()
        if name:
            layer = lookup.get(name.lower())
            if layer is not None:
                tensors.append(_promote_to_image(layer))
                continue
            # A non-empty name matching nothing is almost always a typo; the
            # black fallback is otherwise indistinguishable from a real frame.
            _log.warning(
                "Nukomfy Multilayer Unpack: layer %r not found in bundle; "
                "available: %s. Returning zeros.",
                name, sorted(lookup.keys()),
            )
        tensors.append(torch.zeros(*fallback_bhwc, dtype=torch.float32))
    return tensors


class NukomfyMultiLayerUnpack(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="NukomfyMultiLayerUnpack",
            display_name="Nukomfy Multilayer Unpack",
            category="Nukomfy/Channels",
            description=(
                "Extract named image layers from a multilayer bundle.\n\n"
                "The bundle comes from the extra_layers output of "
                "Nukomfy Read or a Nukomfy Multilayer Pack.\n\n"
                "One layer "
                "per slot, addressed by name. Missing or misnamed "
                "layers fall back to a 3-channel black image."
            ),
            inputs=[
                NukomfyMultilayer.Input(
                    "extra_layers",
                    tooltip=(
                        "Multilayer bundle to unpack. Comes from "
                        "Nukomfy Read (extra_layers output) or Nukomfy "
                        "Multilayer Pack."
                    ),
                ),
                IO.DynamicCombo.Input(
                    "layer_count",
                    display_name="Layer Count",
                    options=[
                        IO.DynamicCombo.Option(
                            f"{n} layer" if n == 1 else f"{n} layers",
                            _generate_unpack_inputs(n),
                        )
                        for n in range(1, _MAX_LAYERS + 1)
                    ],
                    tooltip=_TT_UNPACK_LAYER_COUNT,
                ),
            ],
            outputs=[
                IO.Image.Output(display_name=f"layer_{i}")
                for i in range(1, _MAX_LAYERS + 1)
            ],
        )

    @classmethod
    def execute(cls, extra_layers, layer_count: dict) -> IO.NodeOutput:
        tensors = _unpack_bundle(extra_layers, layer_count)
        return IO.NodeOutput(*tensors)
