"""Activity log configuration (retention policy).

Single-key JSON file describing the time-based pruning policy for
``activity.jsonl``. Mirrors the atomic-write pattern used by ``auth.py``
and ``state.py``: write to a temp file, then ``os.replace`` into place
with a best-effort ``chmod 0o600``.

Schema::

    {"max_days_to_keep": int}

``max_days_to_keep`` is the threshold in days. ``0`` disables the
time-based prune (size-based rotation via ``RotatingFileHandler`` still
applies, capping disk usage at ~60 MB). Negative values are rejected.

The runtime value is held in an in-memory cache; ``load_config`` returns
the cached dict and never re-reads the file at runtime once bootstrapped.
Externally-modified files are rejected by the SHA-256 fingerprint check
(boot) and the mtime/size check (runtime); see ``integrity.py``. The
tamper-tracking scaffolding lives in ``TamperTracked``.
"""
from __future__ import annotations

import json
import logging

from . import integrity, paths
from .tamper_tracked import TamperTracked

_log = logging.getLogger(__name__)

_DEFAULT_MAX_DAYS = 90
_DEFAULT_CONFIG = {"max_days_to_keep": _DEFAULT_MAX_DAYS}
_TRACKED_KEY = "activity_config.json"


def _normalise(data: dict) -> dict:
    raw = data.get("max_days_to_keep", _DEFAULT_MAX_DAYS)
    try:
        days = int(raw)
    except (TypeError, ValueError):
        days = _DEFAULT_MAX_DAYS
    if days < 0:
        days = _DEFAULT_MAX_DAYS
    return {"max_days_to_keep": days}


_tracker = TamperTracked(
    path_fn=paths.activity_config_path,
    tracked_key=_TRACKED_KEY,
    default=_DEFAULT_CONFIG,
    fingerprint_fn=integrity.public_fingerprint,
    record_public_fn=integrity.record_public,
    normalise=_normalise,
    logger=_log,
)


def load_config() -> dict:
    """Return the cached config; default on first-call IO or parse error."""
    return _tracker.load()


def pending_tamper() -> bool:
    """True if a runtime or boot-time tamper has been flagged.

    Also runs a fresh mtime/size check so callers polling this from a
    no-auth surface (``/ping``) see runtime modifications even when no
    other code path has called ``load_config`` recently.
    """
    return _tracker.pending_tamper()


def save_config(new_value: int, *, password: str) -> dict:
    """Persist ``max_days_to_keep`` atomically. Raises ValueError on input.

    Accepts any value that ``int(...)`` can resolve to a non-negative
    integer. Returns the persisted dict for echoing to the caller.

    ``password`` is the verified admin password; it is used to re-seal
    the file via ``integrity.record`` and then discarded. The SHA-256
    public fingerprint is refreshed too so the boot-time check accepts
    the new on-disk content.
    """
    try:
        days = int(new_value)
    except (TypeError, ValueError):
        raise ValueError("max_days_to_keep must be an integer")
    if days < 0:
        raise ValueError("max_days_to_keep must be >= 0")
    data = {"max_days_to_keep": days}
    path = paths.activity_config_path()
    payload = json.dumps(data, indent=2).encode("utf-8")
    paths.atomic_write_bytes(path, payload)
    integrity.record(password, _TRACKED_KEY)
    integrity.record_public(_TRACKED_KEY, payload)
    _tracker.mark_saved(data, path)
    return data


def reset_to_default(*, password: str) -> dict:
    """Overwrite activity_config.json with the default retention policy."""
    return save_config(_DEFAULT_MAX_DAYS, password=password)


def restore_to_cached(*, password: str) -> dict:
    """Re-write the on-disk file from the in-memory cache.

    Called by the integrity check when external tampering is detected:
    the cache holds the last value persisted by an authorised save, so
    re-emitting it restores the user's most recent legitimate setting
    rather than silently downgrading to defaults.

    Falls back to ``reset_to_default`` if the cache is somehow empty
    (e.g. ``load_config`` has never been called this process).
    """
    snapshot = _tracker.peek_cached()
    if snapshot is None:
        return reset_to_default(password=password)
    days = snapshot.get("max_days_to_keep", _DEFAULT_MAX_DAYS)
    return save_config(days, password=password)
