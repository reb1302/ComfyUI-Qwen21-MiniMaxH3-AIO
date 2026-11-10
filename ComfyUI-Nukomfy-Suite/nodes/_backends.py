"""OpenImageIO bootstrap and OCIO Studio Config fallback chain.

Imported eagerly by ``nodes/__init__.py`` so the boot banner and color
config selection happen once at ComfyUI startup.

The fallback chain follows the standard VFX practice for OCIO version
compatibility: try the most recent ACES Studio Config first, fall back
to older ones if the active OpenImageIO build does not ship a matching
OCIO release. If OpenImageIO or its built-in OCIO configs are absent,
falls back gracefully to a built-in colorspace table and analytic color
transforms so node registration never fails at import time.
"""
from __future__ import annotations

import logging
import numpy as np

_log = logging.getLogger(__name__)

try:
    import OpenImageIO as oiio
except ImportError:
    oiio = None  # type: ignore[assignment]
    _log.warning(
        "OpenImageIO is not installed yet; Nukomfy Suite will use pure-Python/PyTorch fallback until OpenImageIO is available."
    )


_STUDIO_CONFIGS: tuple[str, ...] = (
    "studio-config-v4.0.0_aces-v2.0_ocio-v2.5",
    "studio-config-v2.2.0_aces-v1.3_ocio-v2.4",
    "studio-config-v2.1.0_aces-v1.3_ocio-v2.3",
    "studio-config-v1.0.0_aces-v1.3_ocio-v2.1",
    "cg-config-v2.2.0_aces-v1.3_ocio-v2.4",
    "cg-config-v2.1.0_aces-v1.3_ocio-v2.3",
    "cg-config-v1.0.0_aces-v1.3_ocio-v2.1",
    "default",
)

# Canonical name of linear Rec.709 in the active OCIO config, used as
# the default Input/Output Transform across Read and Write.
_LINEAR_REC709: str = "Linear Rec.709 (sRGB)"

_FALLBACK_COLORSPACES: list[str] = [
    "Linear Rec.709 (sRGB)",
    "sRGB Encoded Rec.709 (sRGB)",
    "sRGB - Display",
    "Rec.1886 Rec.709 - Display",
    "ACEScg",
    "ACEScct",
    "ACEScc",
    "ACES2065-1",
    "ARRI LogC3 (EI800)",
    "ARRI LogC4",
    "RED Log3G10 RedWideGamutRGB",
    "Sony S-Log3 S-Gamut3.Cine",
    "DaVinci Intermediate WideGamut",
    "Gamma 2.2 Encoded Rec.709",
    "Raw",
]

_color_config: "oiio.ColorConfig | None" = None
_loaded_config_uri: str | None = None


def _load_color_config() -> None:
    global _color_config, _loaded_config_uri
    if oiio is None:
        return
    for uri in _STUDIO_CONFIGS:
        try:
            cfg = oiio.ColorConfig(f"ocio://{uri}")
        except Exception:
            continue
        if cfg.geterror():
            continue
        try:
            names = list(cfg.getColorSpaceNames())
        except Exception:
            names = []
        if not names:
            continue
        _color_config = cfg
        _loaded_config_uri = f"ocio://{uri}"
        return

    # Also try default environment / OIIO ColorConfig()
    try:
        cfg = oiio.ColorConfig()
        if not cfg.geterror() and list(cfg.getColorSpaceNames()):
            _color_config = cfg
            _loaded_config_uri = ""
    except Exception:
        pass


_load_color_config()

if oiio is not None:
    _log.info("OpenImageIO %s", getattr(oiio, "VERSION_STRING", "loaded"))
    if _color_config is not None:
        _log.info("OCIO config loaded: %s", _loaded_config_uri or "default")
    else:
        _log.warning(
            "No built-in ACES Studio Config found in OpenImageIO; using built-in analytic colorspace fallback."
        )


def has_oiio() -> bool:
    return oiio is not None


def has_ocio_config() -> bool:
    return oiio is not None and _color_config is not None


def get_color_config_uri() -> str:
    """Return the full ``ocio://...`` URI of the active config."""
    if _loaded_config_uri is None:
        return ""
    return _loaded_config_uri


def get_available_colorspaces() -> list[str]:
    """Return the sorted list of colorspace names from the active OCIO config."""
    if _color_config is not None:
        try:
            names = sorted(_color_config.getColorSpaceNames())
            if names:
                if _LINEAR_REC709 not in names:
                    names.insert(0, _LINEAR_REC709)
                return names
        except Exception:
            pass
    return list(_FALLBACK_COLORSPACES)


def validate_colorspace(colorspace: str) -> None:
    """Validate that ``colorspace`` is present in the active OCIO config."""
    if not colorspace or _color_config is None:
        return
    try:
        available = set(_color_config.getColorSpaceNames()) | set(_FALLBACK_COLORSPACES)
    except Exception:
        return
    if colorspace not in available:
        _log.warning(
            "Colorspace %r not found in active OCIO config (%s); proceeding with fallback.",
            colorspace,
            _loaded_config_uri,
        )


# Analytic transfer/gamut helpers used when OCIO Studio Config is unavailable
_ACESCG_TO_REC709 = np.array(
    [
        [1.70505099, -0.62179212, -0.08325887],
        [-0.13025642, 1.14080473, -0.01054832],
        [-0.02400336, -0.12896898, 1.15297234],
    ],
    dtype=np.float32,
)
_REC709_TO_ACESCG = np.linalg.inv(_ACESCG_TO_REC709).astype(np.float32)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    pos = np.maximum(x, 0.0)
    return np.where(pos <= 0.04045, pos / 12.92, np.power((pos + 0.055) / 1.055, 2.4)).astype(np.float32)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    pos = np.maximum(x, 0.0)
    return np.where(pos <= 0.0031308, pos * 12.92, 1.055 * np.power(pos, 1.0 / 2.4) - 0.055).astype(np.float32)


def _to_linear_rec709(rgb: np.ndarray, cs: str) -> np.ndarray:
    cs_low = (cs or "").lower()
    if not cs_low or "raw" in cs_low or ("linear" in cs_low and ("709" in cs_low or "srgb" in cs_low)):
        return rgb
    if "srgb" in cs_low:
        return _srgb_to_linear(rgb)
    if "2.2" in cs_low or "1886" in cs_low or "rec.709" in cs_low:
        return np.power(np.maximum(rgb, 0.0), 2.2).astype(np.float32)
    if "acescg" in cs_low:
        return np.einsum("...c,dc->...d", rgb, _ACESCG_TO_REC709)
    if "acescct" in cs_low:
        lin_ap1 = np.where(
            rgb > 0.155251141552511,
            np.power(2.0, rgb * 17.52 - 9.72),
            (rgb - 0.0729055341958355) / 10.5402377416545,
        ).astype(np.float32)
        return np.einsum("...c,dc->...d", lin_ap1, _ACESCG_TO_REC709)
    return rgb


def _from_linear_rec709(rgb: np.ndarray, cs: str) -> np.ndarray:
    cs_low = (cs or "").lower()
    if not cs_low or "raw" in cs_low or ("linear" in cs_low and ("709" in cs_low or "srgb" in cs_low)):
        return rgb
    if "srgb" in cs_low:
        return _linear_to_srgb(rgb)
    if "2.2" in cs_low or "1886" in cs_low or "rec.709" in cs_low:
        return np.power(np.maximum(rgb, 0.0), 1.0 / 2.2).astype(np.float32)
    if "acescg" in cs_low:
        return np.einsum("...c,dc->...d", rgb, _REC709_TO_ACESCG)
    if "acescct" in cs_low:
        ap1 = np.einsum("...c,dc->...d", rgb, _REC709_TO_ACESCG)
        return np.where(
            ap1 <= 0.0078125,
            10.5402377416545 * ap1 + 0.0729055341958355,
            (np.log2(np.maximum(ap1, 1e-10)) + 9.72) / 17.52,
        ).astype(np.float32)
    return rgb


def analytic_colorconvert_numpy(arr: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Convert float32 image array [..., C] from ``src`` to ``dst`` analytically."""
    if src == dst:
        return arr
    out = arr.astype(np.float32, copy=True)
    c = out.shape[-1]
    rgb = out[..., :3] if c >= 3 else out
    lin = _to_linear_rec709(rgb, src)
    converted = _from_linear_rec709(lin, dst)
    if c >= 3:
        out[..., :3] = converted
    else:
        out = converted
    return out
