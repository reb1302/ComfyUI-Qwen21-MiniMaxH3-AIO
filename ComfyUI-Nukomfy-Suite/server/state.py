"""Persistent availability state for the Nukomfy Manager.

Tracks a single flag per host (available / unavailable) plus audit-only
metadata (changed_at, changed_by). The flag is consumed by the Nukomfy
plugin as a cooperative soft-lock: it gates submissions and surfaces a
reboot warning. It is honored by the client and is not an access-control
boundary on ComfyUI itself. Mirrors the save_auth / load_auth pattern in auth.py:
atomic write via temp file + os.replace, best-effort chmod 0o600.

The runtime value is held in an in-memory cache; ``load_state`` returns
the cached dict and never re-reads the file at runtime once bootstrapped.
Externally-modified files are rejected by the SHA-256 fingerprint check
(boot) and the mtime/size check (runtime); see ``integrity.py``. The
tamper-tracking scaffolding lives in ``TamperTracked``.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging

from . import integrity, paths
from .tamper_tracked import TamperTracked

_log = logging.getLogger(__name__)

_VALID_VALUES = ("available", "unavailable")
_DEFAULT_STATE = {
    "availability": "available",
    "changed_at": "",
    "changed_by": "",
}
_TRACKED_KEY = "availability.json"


def _normalise(data: dict) -> dict:
    if data.get("availability") not in _VALID_VALUES:
        return dict(_DEFAULT_STATE)
    return {
        "availability": data["availability"],
        "changed_at": str(data.get("changed_at", "") or ""),
        "changed_by": str(data.get("changed_by", "") or ""),
    }


_tracker = TamperTracked(
    path_fn=paths.availability_json_path,
    tracked_key=_TRACKED_KEY,
    default=_DEFAULT_STATE,
    fingerprint_fn=integrity.public_fingerprint,
    record_public_fn=integrity.record_public,
    normalise=_normalise,
    logger=_log,
)


def load_state() -> dict:
    """Return the cached availability state; default on first-call error."""
    return _tracker.load()


def pending_tamper() -> bool:
    """True if a runtime or boot-time tamper has been flagged.

    Also runs a fresh mtime/size check so callers polling this from a
    no-auth surface (``/ping``) see runtime modifications even when no
    other code path has called ``load_state`` recently.
    """
    return _tracker.pending_tamper()


def save_state(new_value: str, *, changed_by: str, password: str) -> dict:
    """Atomically persist a new availability value with audit metadata.

    Raises ValueError if new_value is not one of the valid values.
    Returns the persisted dict so the caller can echo it in the response.

    ``password`` is the plaintext admin password verified by the route
    handler. It is forwarded to ``integrity.record`` to re-seal the file
    after the write, then discarded. The SHA-256 public fingerprint is
    refreshed too so the boot-time check accepts the new on-disk content.
    """
    if new_value not in _VALID_VALUES:
        raise ValueError("invalid availability value: {!r}".format(new_value))
    data = {
        "availability": new_value,
        "changed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "changed_by": changed_by or "unknown",
    }
    path = paths.availability_json_path()
    payload = json.dumps(data, indent=2).encode("utf-8")
    paths.atomic_write_bytes(path, payload)
    integrity.record(password, _TRACKED_KEY)
    integrity.record_public(_TRACKED_KEY, payload)
    _tracker.mark_saved(data, path)
    return data


def reset_to_default(*, password: str) -> dict:
    """Overwrite availability.json with the default ``available`` state.

    Invoked by the route handler when ``integrity.verify_all`` flags
    availability as tampered. The new file is sealed under the current
    password so subsequent checks pass cleanly.
    """
    return save_state("available", changed_by="integrity_reset",
                      password=password)


def restore_to_cached(*, password: str) -> dict:
    """Re-write the on-disk file from the in-memory cache.

    Called by the integrity check when external tampering is detected:
    the cache holds the last value persisted by an authorised save, so
    re-emitting it restores the user's most recent legitimate setting
    rather than silently downgrading to defaults. ``changed_by`` is
    overridden to ``integrity_reset`` to make the recovery auditable;
    ``changed_at`` is regenerated to the current UTC time.

    Falls back to ``reset_to_default`` if the cache is somehow empty.
    """
    snapshot = _tracker.peek_cached()
    if snapshot is None:
        return reset_to_default(password=password)
    value = snapshot.get("availability") or "available"
    return save_state(value, changed_by="integrity_reset",
                      password=password)


def availability() -> str:
    """Convenience accessor for the bare availability string."""
    return load_state().get("availability") or "available"
