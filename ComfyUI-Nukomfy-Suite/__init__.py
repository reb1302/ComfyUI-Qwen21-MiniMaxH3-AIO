"""Companion custom node for the Nukomfy Nuke plugin.

Two responsibilities:
- Server-side admin/reboot/availability/activity routes under /nukomfy/*.
- Five graph nodes (NukomfyRead, NukomfyWrite, NukomfyOCIOColorSpace,
  NukomfyMultiLayerPack, NukomfyMultiLayerUnpack) for Nuke-style IO
  inside ComfyUI workflows.

License: Apache-2.0
"""
from __future__ import annotations

import logging

from .version import __version__

# Own the console format so every getLogger(__name__) in the package inherits
# it. Set up before the server import: job-history boot logs fire during it.
_log = logging.getLogger(__name__)
if not _log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        fmt='[Nukomfy Suite] %(levelname)s: %(message)s'))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)
    _log.propagate = False

from . import server  # noqa: F401,E402  - registers HTTP routes on import (no heavy deps)
from .server import node_status  # noqa: E402

# The manager (HTTP routes + sidebar) has no third-party dependencies and
# must stay available even when the graph nodes cannot import. The nodes
# need OpenImageIO + fileseq; guarding their import here keeps WEB_DIRECTORY
# set and the sidebar served, and records the failure so the manager can
# show a banner instead of ComfyUI dropping the whole custom node (UI
# included) on a missing dependency.
WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import (
        comfy_entrypoint as _comfy_entrypoint,
        NODE_CLASS_MAPPINGS as _NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception as exc:
    _missing = getattr(exc, "name", None)
    node_status.set_import_error(str(exc), _missing)
    _log.error(
        "graph nodes failed to import (missing module: %s); manager stays "
        "available. Install dependencies (see README) and restart",
        _missing or "unknown", exc_info=True,
    )
else:
    comfy_entrypoint = _comfy_entrypoint
    NODE_CLASS_MAPPINGS = dict(_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS = dict(_NODE_DISPLAY_NAME_MAPPINGS)
    _log.info("loaded (%s)", __version__)

__all__ = [
    "comfy_entrypoint",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "__version__",
]
