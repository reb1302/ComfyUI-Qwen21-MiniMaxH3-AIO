"""Password hashing, verification and storage.

Uses hashlib.scrypt from the stdlib (no external dependencies) with a random
32-byte salt. Verification is timing-safe via hmac.compare_digest.

scrypt is the strongest password-hashing primitive available in the Python
standard library: memory-hard, resistant to GPU/ASIC attacks. Argon2 is
arguably better but requires an external dependency (argon2-cffi); not used
to keep this package zero-dependency.

Rate limiting is per-IP and isolated: a brute-force attempt from one client
does not block other clients on the same network.

The on-disk ``auth.json`` is mirrored in an in-memory cache so external
edits (intentional or accidental) cannot silently swap the password
record between admin operations. The SHA-256 fingerprint check at boot
and the mtime/size check at runtime reject externally-modified files;
see ``integrity.py``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from . import integrity, paths
from .tamper_tracked import TamperTracked

_log = logging.getLogger(__name__)

# scrypt parameters. n=2^15 doubles the OWASP minimum and lands around
# ~50 ms per verification on modern CPUs, which is fine for sparse auth
# operations on a LAN. Bump n further if hardware gets faster.
_SCRYPT_N = 2 ** 15  # 32768
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
# OpenSSL's default scrypt memory cap is ~32 MiB, which N=2^15 * r=8 just
# brushes against (128*N*r ~= 32 MiB). Raise the explicit ceiling to leave
# headroom for the algorithm's working set and future parameter bumps.
_SCRYPT_MAXMEM = 128 * 1024 * 1024  # 128 MiB

MIN_PASSWORD_LENGTH = 4

# Rate limit policy (per-IP, isolated):
#   - up to _MAX_FAILS wrong attempts allowed in _WINDOW seconds before block
#   - on threshold hit, the offending IP is blocked for _BLOCK_SECONDS
#   - other IPs are unaffected
_WINDOW = 60.0
_MAX_FAILS = 10
_BLOCK_SECONDS = 60.0

_failures: dict[str, deque] = defaultdict(lambda: deque(maxlen=_MAX_FAILS * 2))
_blocked: dict[str, float] = {}

_TRACKED_KEY = "auth.json"

# default=None: "no record" is a legitimate cached value, so the class
# gates re-reads on its _bootstrapped flag, not on cached-is-None. No
# normalise: the scrypt record is cached verbatim.
_tracker = TamperTracked(
    path_fn=paths.auth_json_path,
    tracked_key=_TRACKED_KEY,
    default=None,
    fingerprint_fn=integrity.public_fingerprint,
    record_public_fn=integrity.record_public,
    logger=_log,
)


def hash_password(password: str) -> dict:
    """Compute a scrypt hash record for the given password.

    Returns a dict ready to be JSON-serialised.
    """
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    salt = secrets.token_bytes(32)
    params = {"n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P, "dklen": _SCRYPT_DKLEN}
    key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        maxmem=_SCRYPT_MAXMEM,
        **params,
    )
    return {
        "algorithm": "scrypt",
        "salt": salt.hex(),
        "hash": key.hex(),
        "params": params,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def verify_password(password: str, stored: dict | None) -> bool:
    """Compare a candidate password against a stored hash record."""
    if not isinstance(password, str) or not stored or not isinstance(stored, dict):
        return False
    if stored.get("algorithm") != "scrypt":
        return False
    try:
        salt = bytes.fromhex(stored["salt"])
        expected = bytes.fromhex(stored["hash"])
        params = stored["params"]
        key = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(params["n"]),
            r=int(params["r"]),
            p=int(params["p"]),
            dklen=int(params["dklen"]),
            maxmem=_SCRYPT_MAXMEM,
        )
    except (KeyError, ValueError, TypeError, OverflowError, MemoryError):
        # A crafted params record (e.g. an absurd dklen -> OverflowError,
        # or n past the maxmem cap) must fail closed, not 500 the gate.
        return False
    return hmac.compare_digest(key, expected)


def load_auth() -> dict | None:
    """Return the cached auth record, or None if absent / tampered."""
    return _tracker.load()


def pending_tamper() -> bool:
    """True if a runtime or boot-time tamper has been flagged.

    Also runs a fresh mtime/size check so callers polling this from a
    no-auth surface (``/ping``) see runtime modifications even when no
    other code path has called ``load_auth`` recently.
    """
    return _tracker.pending_tamper()


def save_auth(auth_record: dict, *, password: str) -> None:
    """Persist auth.json atomically with restrictive permissions (best-effort).

    The new ``password`` keyword is the plaintext that was just hashed
    into ``auth_record``. It is forwarded to ``integrity.record_all`` so every
    tracked state file is re-sealed under the current secret, then
    discarded. Callers must not stash it. The SHA-256 public fingerprint
    for ``auth.json`` is refreshed too so the boot-time check accepts
    the new on-disk content.
    """
    path = paths.auth_json_path()
    payload = json.dumps(auth_record, indent=2).encode("utf-8")
    paths.atomic_write_bytes(path, payload)
    integrity.record_all(password)
    integrity.record_public(_TRACKED_KEY, payload)
    _tracker.mark_saved(auth_record, path)


def has_password() -> bool:
    """True iff an admin password has been configured."""
    return load_auth() is not None


def check_rate_limit(ip: str) -> tuple[bool, float, str | None]:
    """Return (allowed, retry_after_seconds, event) for THIS ip only.

    allowed=False means the IP is currently blocked. retry_after_seconds is
    the number of seconds until the block expires. event is "block_cleared"
    if this call lifted a previously-active block (cooldown expired); None
    otherwise. Callers can surface the lift as a distinct audit entry.
    """
    now = time.monotonic()
    unblock_ts = _blocked.get(ip)
    if unblock_ts is not None:
        if unblock_ts > now:
            return False, unblock_ts - now, None
        # Block expired: clear it so future failures start fresh.
        del _blocked[ip]
        _failures.pop(ip, None)
        return True, 0.0, "block_cleared"
    return True, 0.0, None


def record_failure(ip: str) -> str | None:
    """Register a failed verification attempt for rate-limit accounting.

    Only affects the calling IP's bucket; never global. Returns
    "block_started" if this failure crossed the threshold and the IP is now
    blocked; None otherwise.
    """
    now = time.monotonic()
    attempts = _failures[ip]
    attempts.append(now)
    # Purge expired entries (anything older than the window).
    while attempts and attempts[0] < now - _WINDOW:
        attempts.popleft()
    if len(attempts) >= _MAX_FAILS:
        _blocked[ip] = now + _BLOCK_SECONDS
        attempts.clear()
        return "block_started"
    return None


def record_success(ip: str) -> None:
    """Reset the failure bucket for an IP after a successful auth."""
    _failures.pop(ip, None)
