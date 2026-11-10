"""OpenImageIO bootstrap and OCIO Studio Config fallback chain.

Imported eagerly by ``nodes/__init__.py`` so the boot banner and color
config selection happen once at ComfyUI startup.

The fallback chain follows the standard VFX practice for OCIO version
compatibility: try the most recent ACES Studio Config first, fall back
to older ones if the active OpenImageIO build does not ship a matching
OCIO release. The most recent successfully loaded config wins.
"""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

try:
    import OpenImageIO as oiio
except ImportError as exc:
    raise ImportError(
        "Nukomfy Suite requires OpenImageIO. Install via: pip install OpenImageIO"
    ) from exc


_STUDIO_CONFIGS: tuple[str, ...] = (
    "studio-config-v4.0.0_aces-v2.0_ocio-v2.5",
    "studio-config-v2.2.0_aces-v1.3_ocio-v2.4",
    "studio-config-v2.1.0_aces-v1.3_ocio-v2.3",
    "studio-config-v1.0.0_aces-v1.3_ocio-v2.1",
)

# Canonical name of linear Rec.709 in the active OCIO config, used as
# the default Input/Output Transform across Read and Write. The OCIO
# config loaded at boot is expected to expose this colorspace under
# this name (true for
# studio-config-v4.0.0_aces-v2.0_ocio-v2.5).
_LINEAR_REC709: str = "Linear Rec.709 (sRGB)"

_color_config: "oiio.ColorConfig | None" = None
_loaded_config_uri: str | None = None


def _load_color_config() -> None:
    global _color_config, _loaded_config_uri
    for uri in _STUDIO_CONFIGS:
        try:
            cfg = oiio.ColorConfig(f"ocio://{uri}")
        except Exception:
            continue
        # OIIO returns a ColorConfig instance even on failure; the error
        # surface is geterror() returning a non-empty string.
        if cfg.geterror():
            continue
        _color_config = cfg
        _loaded_config_uri = uri
        return


_load_color_config()

if _color_config is None:
    raise RuntimeError(
        "Nukomfy Suite: no ACES Studio Config could be loaded. "
        "Upgrade OpenImageIO: pip install -U OpenImageIO"
    )


_log.info("OpenImageIO %s", oiio.VERSION_STRING)
_log.info("OCIO config loaded: %s", _loaded_config_uri)


def get_color_config_uri() -> str:
    """Return the full ``ocio://...`` URI of the active config.

    OIIO ImageBufAlgo.colorconvert accepts the colorconfig argument as a
    string URI, not as a ColorConfig instance.
    """
    if _loaded_config_uri is None:
        return ""
    return f"ocio://{_loaded_config_uri}"


def get_available_colorspaces() -> list[str]:
    """Return the sorted list of colorspace names from the active OCIO config."""
    if _color_config is None:
        return []
    return sorted(_color_config.getColorSpaceNames())


def validate_colorspace(colorspace: str) -> None:
    """Validate that ``colorspace`` is present in the active OCIO config.

    Used by NukomfyColorspaceConvert at runtime to surface stale knob
    values when the active OCIO release changes between workflow saves.
    """
    if _color_config is None:
        raise RuntimeError(
            "Nukomfy Suite: no OCIO color config is available. "
            "Check the ComfyUI OCIO setup and restart ComfyUI."
        )
    if colorspace not in _color_config.getColorSpaceNames():
        raise RuntimeError(
            f"Colorspace '{colorspace}' not found in active OCIO config "
            f"({_loaded_config_uri}). In Nukomfy Workflow Editor, "
            f"use 'Sync Parameters' to refresh."
        )
