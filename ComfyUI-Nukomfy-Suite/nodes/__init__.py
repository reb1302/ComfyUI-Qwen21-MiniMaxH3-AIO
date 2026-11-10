"""Graph nodes sub-package for the Nukomfy Suite.

Importing this package triggers the OpenImageIO + OCIO bootstrap in
``_backends`` so the boot banner prints once at ComfyUI startup.

Exposes both the v3 ``ComfyExtension`` entrypoint (canonical pattern
from ``comfy_api.latest``) and the legacy ``NODE_CLASS_MAPPINGS``
dictionary so older ComfyUI runtimes that have not yet picked up the
async entrypoint API can still load the suite.
"""
from __future__ import annotations

from . import _backends  # noqa: F401  - eager import: OCIO fallback chain + banner

from .io_read import NukomfyRead
from .io_write import NukomfyWrite
from .ocio_colorspace import NukomfyOCIOColorSpace
from .multilayer_pack import NukomfyMultiLayerPack, NukomfyMultiLayerUnpack
from .ltx25_bridge import NukomfyLTX25BakeIn, NukomfyLTX25UnbakeOut


_NODES = [
    NukomfyRead,
    NukomfyWrite,
    NukomfyOCIOColorSpace,
    NukomfyMultiLayerPack,
    NukomfyMultiLayerUnpack,
    NukomfyLTX25BakeIn,
    NukomfyLTX25UnbakeOut,
]

for _node_cls in _NODES:
    try:
        _node_cls.GET_SCHEMA()
    except Exception:
        pass


from comfy_api.latest import ComfyExtension


class NukomfySuiteExtension(ComfyExtension):
    async def get_node_list(self):
        return _NODES


async def comfy_entrypoint() -> ComfyExtension:
    return NukomfySuiteExtension()


NODE_CLASS_MAPPINGS = {
    "NukomfyRead": NukomfyRead,
    "NukomfyWrite": NukomfyWrite,
    "NukomfyOCIOColorSpace": NukomfyOCIOColorSpace,
    "NukomfyMultiLayerPack": NukomfyMultiLayerPack,
    "NukomfyMultiLayerUnpack": NukomfyMultiLayerUnpack,
    "NukomfyLTX25BakeIn": NukomfyLTX25BakeIn,
    "NukomfyLTX25UnbakeOut": NukomfyLTX25UnbakeOut,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukomfyRead": "Nukomfy Read",
    "NukomfyWrite": "Nukomfy Write",
    "NukomfyOCIOColorSpace": "Nukomfy OCIO ColorSpace",
    "NukomfyMultiLayerPack": "Nukomfy Multilayer Pack",
    "NukomfyMultiLayerUnpack": "Nukomfy Multilayer Unpack",
    "NukomfyLTX25BakeIn": "Nukomfy LTX-2.5 Sequence -> MP4 Bake",
    "NukomfyLTX25UnbakeOut": "Nukomfy LTX-2.5 MP4 -> Sequence Restore",
}


__all__ = [
    "comfy_entrypoint",
    "NukomfySuiteExtension",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
