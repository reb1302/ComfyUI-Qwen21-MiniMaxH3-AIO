"""Records whether the Suite graph nodes imported successfully.

The manager (HTTP routes + sidebar) has no third-party dependencies and
loads unconditionally. The graph nodes need OpenImageIO + fileseq; the
top-level package guards their import and, on failure, stores the error
here so the sidebar ping can surface a banner. This module keeps no heavy
imports: it must stay importable in every situation the manager is.
"""
from __future__ import annotations

_import_error = None
_missing_module = None


def set_import_error(message, missing_module=None):
    """Record that the graph nodes failed to import."""
    global _import_error, _missing_module
    _import_error = str(message) if message else None
    _missing_module = missing_module or None


def get_import_error():
    """Return the import error string, or None when the nodes loaded."""
    return _import_error


def get_missing_module():
    """Return the missing module name when known, else None."""
    return _missing_module
