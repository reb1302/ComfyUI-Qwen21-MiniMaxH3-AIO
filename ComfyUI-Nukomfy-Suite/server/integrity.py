"""Tamper detection for the manager's on-disk state files.

Two complementary layers:

1. HMAC-SHA256 keyed by the admin password (``integrity.json``).
   Recomputed on every authorised write, verified on every authorised
   admin action. A client with filesystem access cannot forge a valid
   signature without knowing the password, so direct file edits are
   detected on the next admin login.

2. SHA-256 public fingerprint (``integrity_public.json``).
   Recomputed on every authorised write, verified on every cache
   bootstrap (boot time and first lazy load per module). Lets the
   in-memory layer reject externally-modified content before any admin
   password is available. Defends against accidental edits and prevents
   tampered values from being silently used between a manual edit and
   the next admin login.

Tracked files (others are out of scope; see docstrings
in ``activity.py`` and ``job_history_db.py`` for the rationale):

    - ``auth.json``
    - ``availability.json``
    - ``activity_config.json``
    - ``job_history_config.json``
"""
from __future__ import annotations

import hashlib
import hmac
import json

from . import paths


def _tracked_paths() -> dict[str, str]:
    return {
        "auth.json": paths.auth_json_path(),
        "availability.json": paths.availability_json_path(),
        "activity_config.json": paths.activity_config_path(),
        "job_history_config.json": paths.history_config_path(),
    }


def is_tracked(key: str) -> bool:
    return key in _tracked_paths()


def _password_key(password: str) -> bytes:
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    return password.encode("utf-8")


def _compute_hmac(password: str, file_bytes: bytes) -> str:
    return hmac.new(
        _password_key(password), file_bytes, hashlib.sha256
    ).hexdigest()


def _read_file_bytes(abs_path: str) -> bytes | None:
    try:
        with open(abs_path, "rb") as f:
            return f.read()
    except OSError:
        return None


def _load_store() -> dict:
    path = paths.integrity_store_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    sigs = data.get("signatures")
    return sigs if isinstance(sigs, dict) else {}


def _save_store(signatures: dict) -> None:
    path = paths.integrity_store_path()
    payload = json.dumps(
        {"signatures": signatures}, indent=2, sort_keys=True).encode("utf-8")
    paths.atomic_write_bytes(path, payload)


def record(password: str, key: str) -> None:
    """Recompute and persist the HMAC for one tracked key.

    Silently no-op when the key is unknown or the file is missing.
    IO failures while persisting the store are swallowed so the
    primary write (auth/state/config) is never compromised by a
    secondary failure on integrity.json.
    """
    tracked = _tracked_paths()
    abs_path = tracked.get(key)
    if abs_path is None:
        return
    data = _read_file_bytes(abs_path)
    if data is None:
        return
    try:
        sig = _compute_hmac(password, data)
        store = _load_store()
        store[key] = sig
        _save_store(store)
    except (OSError, ValueError):
        pass


def record_all(password: str) -> None:
    """Recompute the HMAC for every tracked file present on disk.

    Used after a password change: the secret has rotated so every
    previously-stored signature is invalid and must be regenerated.
    Missing files are simply skipped (they will be sealed on first
    write).
    """
    tracked = _tracked_paths()
    try:
        store = _load_store()
        for key, abs_path in tracked.items():
            data = _read_file_bytes(abs_path)
            if data is None:
                store.pop(key, None)
                continue
            store[key] = _compute_hmac(password, data)
        _save_store(store)
    except (OSError, ValueError):
        pass


def verify_all(password: str) -> list[str]:
    """Return the sorted list of tracked keys whose signature is invalid.

    Detection rules (HMAC layer):

        - File present, signature present, HMAC matches -> clean.
        - File present, no signature recorded -> tampered (the store
          was wiped or never bootstrapped after the password was set).
        - File absent, signature recorded -> tampered (the attacker
          deleted the file).
        - File absent, no signature recorded -> clean (legitimate
          first-run state, e.g. a config that was never persisted).

    The HMAC result is augmented with the in-memory "pending tamper"
    flags maintained by each tracked module: cache bootstraps and
    runtime mtime checks may detect a modification before any admin
    password reaches the server. Any module reporting
    ``pending_tamper()`` is added to the returned list.
    """
    tracked = _tracked_paths()
    store = _load_store()
    tampered: list[str] = []
    for key, abs_path in tracked.items():
        on_disk = _read_file_bytes(abs_path)
        recorded = store.get(key)
        if on_disk is None and recorded is None:
            continue
        if on_disk is None or recorded is None:
            tampered.append(key)
            continue
        try:
            actual = _compute_hmac(password, on_disk)
        except ValueError:
            tampered.append(key)
            continue
        if not hmac.compare_digest(actual, str(recorded)):
            tampered.append(key)

    # Local import to avoid a circular dependency: the four modules
    # below import ``integrity`` at module level.
    from . import activity_config, auth, job_history_config, state
    pending = {
        "auth.json": auth.pending_tamper(),
        "availability.json": state.pending_tamper(),
        "activity_config.json": activity_config.pending_tamper(),
        "job_history_config.json": job_history_config.pending_tamper(),
    }
    for key, flagged in pending.items():
        if flagged and key not in tampered:
            tampered.append(key)

    tampered.sort()
    return tampered


def _load_public_store() -> dict:
    path = paths.integrity_public_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    fps = data.get("fingerprints")
    return fps if isinstance(fps, dict) else {}


def _save_public_store(fingerprints: dict) -> None:
    path = paths.integrity_public_path()
    payload = json.dumps(
        {"fingerprints": fingerprints}, indent=2, sort_keys=True).encode("utf-8")
    paths.atomic_write_bytes(path, payload)


def record_public(key: str, file_bytes: bytes) -> None:
    """Persist the SHA-256 fingerprint for one tracked key.

    Called by every authorised write alongside ``record(password, key)``.
    It exists so the in-memory bootstrap layer can reject
    externally-modified files before the next admin login, even across a
    server restart.

    Silently no-op when the key is unknown. IO failures are swallowed
    so the primary write is never compromised by a secondary failure.
    """
    if not is_tracked(key):
        return
    try:
        store = _load_public_store()
        store[key] = hashlib.sha256(file_bytes).hexdigest()
        _save_public_store(store)
    except (OSError, ValueError):
        pass


def public_fingerprint(key: str) -> str | None:
    """Return the persisted SHA-256 fingerprint for ``key``, or None."""
    if not is_tracked(key):
        return None
    return _load_public_store().get(key)
