"""Filesystem paths used by the Nukomfy Manager custom node.

All persistent state (auth.json, availability.json, activity.jsonl,
activity_config.json) lives under the ComfyUI user directory so it
survives custom-node upgrades.
"""
from __future__ import annotations

import os
import time

_PACKAGE_DIR = "nukomfy_manager"

try:
    import folder_paths  # type: ignore[import-not-found]
    _BASE_USER_DIR = folder_paths.get_user_directory()
except Exception:
    # Fallback for standalone test runs outside ComfyUI: assume the custom
    # node lives at ComfyUI/custom_nodes/ComfyUI-Nukomfy-Suite/server/paths.py
    # so the user dir is three levels up.
    _BASE_USER_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "user")
    )


def get_config_dir() -> str:
    """Return the absolute path to the manager's per-user config directory.

    Resolves to <ComfyUI user dir>/default/nukomfy_manager/ and creates the
    directory if missing.
    """
    path = os.path.join(_BASE_USER_DIR, "default", _PACKAGE_DIR)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        # Read-only or permission-denied filesystem: return the path anyway so
        # readers degrade to "file missing" (no-auth default) instead of
        # raising out of read-only endpoints like /ping. Writers still error.
        pass
    return path


def auth_json_path() -> str:
    return os.path.join(get_config_dir(), "auth.json")


def availability_json_path() -> str:
    return os.path.join(get_config_dir(), "availability.json")


def activity_log_path() -> str:
    return os.path.join(get_config_dir(), "activity.jsonl")


def activity_config_path() -> str:
    return os.path.join(get_config_dir(), "activity_config.json")


def history_db_path() -> str:
    return os.path.join(get_config_dir(), "job_history.db")


def history_config_path() -> str:
    return os.path.join(get_config_dir(), "job_history_config.json")


def integrity_store_path() -> str:
    return os.path.join(get_config_dir(), "integrity.json")


def integrity_public_path() -> str:
    return os.path.join(get_config_dir(), "integrity_public.json")


def _owner_only_opener(path: str, flags: int) -> int:
    # Create at 0o600 so the bytes are not briefly world-readable before the
    # chmod. umask only clears bits (0o600 or tighter); ignored on Windows.
    return os.open(path, flags, 0o600)


def atomic_write_bytes(path: str, payload: bytes,
                       retries: int = 5, backoff_step: float = 0.05) -> None:
    """Atomically write bytes to a path via a temp file and os.replace.

    The temp file is created owner-only (0o600); the trailing chmod
    re-tightens a pre-existing temp that O_TRUNC left with looser bits.
    Both are best-effort and no-op on Windows (file ACLs, not POSIX modes).
    Creates the parent directory if missing.

    On Windows the destination can be transiently locked by an AV scanner,
    OneDrive/Dropbox sync or the search indexer; the replace is retried
    with linear backoff (0.05s, 0.10s, ...) before giving up. A permanent
    failure still raises - every caller is a config-save path that reports
    the OSError.
    """
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "wb", opener=_owner_only_opener) as f:
        f.write(payload)
    retries = max(1, retries)  # always attempt the replace at least once
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(backoff_step * (attempt + 1))
                continue
            raise
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
