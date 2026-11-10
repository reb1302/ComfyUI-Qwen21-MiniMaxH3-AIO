"""Job history retention configuration.

Persistent JSON file controlling the lifecycle of ``job_history.db``:
whether the persistent history feature is enabled, and how aggressively
old entries are pruned. Mirrors the atomic-write pattern used by
``activity_config.py``, ``state.py`` and ``auth.py``: write to a temp
file, then ``os.replace`` into place with a best-effort ``chmod 0o600``.

Schema::

    {
        "enabled": bool,                # default True
        "max_jobs_to_keep": int,        # 0..5000 (0 = unlimited, capped by hard limit)
        "max_days_to_keep": int         # 0..3650 (0 = no time-based prune)
    }

The runtime value is held in an in-memory cache; ``load_config`` returns
the cached dict and never re-reads the file at runtime once bootstrapped.
The cache is populated at first call from disk and refreshed only by
authorised writes via ``save_config``. Externally-modified files are
rejected by the SHA-256 fingerprint check (boot) and the mtime/size
check (runtime); see ``integrity.py`` for the two-layer model. The
tamper-tracking scaffolding lives in ``TamperTracked``.

``save_config`` is strict: unknown keys or out-of-range values raise
``ValueError`` so an HTTP handler can surface the error to the user.
"""
from __future__ import annotations

import json
import logging

from . import integrity, paths
from .tamper_tracked import TamperTracked

_log = logging.getLogger(__name__)

_DEFAULT_CONFIG = {
    "enabled": True,
    "max_jobs_to_keep": 500,
    "max_days_to_keep": 90,
}

_KNOWN_KEYS = frozenset(_DEFAULT_CONFIG.keys())
_TRACKED_KEY = "job_history_config.json"

_MAX_JOBS_CEILING = 5000
_MAX_DAYS_CEILING = 3650


def _coerce_bool(value, default):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
    return default


def _coerce_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _normalise(data: dict) -> dict:
    enabled = _coerce_bool(data.get("enabled"), _DEFAULT_CONFIG["enabled"])
    max_jobs = _clamp(_coerce_int(
        data.get("max_jobs_to_keep"),
        _DEFAULT_CONFIG["max_jobs_to_keep"]), 0, _MAX_JOBS_CEILING)
    max_days = _clamp(_coerce_int(
        data.get("max_days_to_keep"),
        _DEFAULT_CONFIG["max_days_to_keep"]), 0, _MAX_DAYS_CEILING)
    return {
        "enabled": enabled,
        "max_jobs_to_keep": max_jobs,
        "max_days_to_keep": max_days,
    }


_tracker = TamperTracked(
    path_fn=paths.history_config_path,
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


def _check_int(name, value, lo, hi):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("{} must be an integer".format(name))
    if value < lo or value > hi:
        raise ValueError(
            "{} must be in range {}..{}".format(name, lo, hi))


def save_config(*, password: str, **updates) -> dict:
    """Persist a partial config update atomically.

    ``updates`` accepts any subset of the three known keys. Unknown keys
    or out-of-range values raise ``ValueError`` before any write happens.
    Returns the full persisted dict for echoing to the caller.

    ``password`` is the verified admin password; it is used to re-seal
    the file via ``integrity.record`` and then discarded. The SHA-256
    public fingerprint is refreshed too so the boot-time check accepts
    the new on-disk content.
    """
    unknown = set(updates.keys()) - _KNOWN_KEYS
    if unknown:
        raise ValueError(
            "unknown key(s): {}".format(", ".join(sorted(unknown))))

    merged = load_config()
    for key, value in updates.items():
        if key == "enabled":
            if not isinstance(value, bool):
                raise ValueError("enabled must be a boolean")
            merged[key] = value
        else:
            try:
                merged[key] = int(value)
            except (TypeError, ValueError):
                raise ValueError("{} must be an integer".format(key))

    _check_int("max_jobs_to_keep", merged["max_jobs_to_keep"],
               0, _MAX_JOBS_CEILING)
    _check_int("max_days_to_keep", merged["max_days_to_keep"],
               0, _MAX_DAYS_CEILING)

    path = paths.history_config_path()
    payload = json.dumps(merged, indent=2).encode("utf-8")
    paths.atomic_write_bytes(path, payload)
    integrity.record(password, _TRACKED_KEY)
    integrity.record_public(_TRACKED_KEY, payload)
    _tracker.mark_saved(merged, path)
    return merged


def reset_to_default(*, password: str) -> dict:
    """Overwrite job_history_config.json with the default policy."""
    return save_config(password=password, **_DEFAULT_CONFIG)


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
    return save_config(password=password, **snapshot)
