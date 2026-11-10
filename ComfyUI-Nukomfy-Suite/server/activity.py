"""Activity log of privileged operations.

JSONL append-only file under the manager config directory, rotated by size
via RotatingFileHandler (10 MB per file, 5 backups => ~60 MB cap on disk).
Time-based pruning is handled separately by ``cleanup_old(max_days)``.

Each entry has the shape::

    {"ts": iso8601, "ip": str, "user": str, "action": str,
     "success": bool, "reason": str}

Failures during logging are swallowed: an audit miss is preferable to an
admin action that errors out because the disk is full.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
from logging.handlers import RotatingFileHandler

from . import paths

_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5
_MAX_TAIL_ENTRIES = 10000  # hard upper bound on entries returned by tail()
_LOGGER_NAME = "nukomfy.activity"

_logger = logging.getLogger(_LOGGER_NAME)
_log = logging.getLogger(__name__)  # console logger; _logger above is the JSONL file logger
_activity_lock = threading.RLock()


class _ActivityJsonFormatter(logging.Formatter):
    """Emit ``record.activity_data`` as a single compact JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        data = getattr(record, "activity_data", None)
        if not isinstance(data, dict):
            return ""
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def setup_activity_logger() -> None:
    """Idempotently attach a rotating file handler to the activity logger.

    Safe to call from import time or lazily on first event. The logger is
    isolated (``propagate=False``) so ComfyUI's root logger does not echo
    JSONL lines onto the console.
    """
    if getattr(_logger, "_nukomfy_activity_configured", False):
        return
    try:
        path = paths.activity_log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(_ActivityJsonFormatter())
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
        _logger._nukomfy_activity_configured = True  # type: ignore[attr-defined]
    except Exception:
        # Best-effort: leave the logger unconfigured so log_event() short-
        # circuits gracefully.
        _log.exception("Failed to set up activity logger (non-fatal)")


def log_event(
    action: str,
    *,
    success: bool,
    user: str = "",
    ip: str = "",
    reason: str = "",
) -> None:
    """Append a single JSONL line to activity.jsonl.

    Always swallows errors: the activity log is never allowed to break the
    privileged operation it is recording.
    """
    try:
        # Serialise under the same lock cleanup_old() holds while it tears
        # down and re-attaches the rotating handler, so a write can't land
        # in that brief window and (on Windows) block the os.replace or lose
        # the line. Non-hot path; the lock cost is negligible.
        with _activity_lock:
            setup_activity_logger()
            entry = {
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "ip": str(ip or ""),
                "user": str(user or ""),
                "action": str(action or ""),
                "success": bool(success),
                "reason": str(reason or ""),
            }
            _logger.info("", extra={"activity_data": entry})
    except Exception:
        pass


def tail(n: int = 200) -> list[dict]:
    """Return the last ``n`` parsed entries from the main log file.

    Newest entry last (chronological order). Only the active rotation
    target is inspected; ``.1``..``.5`` archives are skipped because they
    are kept for forensic recovery and would mix older history into a
    "recent activity" view. Malformed lines are dropped silently.
    """
    try:
        setup_activity_logger()
        path = paths.activity_log_path()
        if not os.path.exists(path):
            return []
        try:
            limit = max(1, min(int(n), _MAX_TAIL_ENTRIES))
        except (TypeError, ValueError):
            limit = 200
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        out: list[dict] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(entry, dict):
                out.append(entry)
        return out
    except Exception:
        return []


def cleanup_old(max_days: int) -> int:
    """Drop entries older than ``max_days`` from the activity log set.

    Reads the active log and every rotated backup, keeps the entries with
    ``ts >= now - max_days``, rewrites the main file with the survivors
    (sorted ascending by ts), and deletes the rotated backups. Returns the
    number of entries dropped. ``max_days <= 0`` disables the time cap and
    is a no-op.

    Concurrency: serialised under ``_activity_lock``, which ``log_event()``
    also holds while it writes, so a concurrent log line waits out the
    teardown/rewrite/re-attach window instead of racing the handler swap.
    """
    if not isinstance(max_days, int) or max_days <= 0:
        return 0
    with _activity_lock:
        try:
            base = paths.activity_log_path()
            cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=max_days)
            survivors: list[tuple[str, dict]] = []
            dropped = 0
            files = [base] + [base + "." + str(i) for i in range(1, _BACKUP_COUNT + 1)]
            for f in files:
                if not os.path.exists(f):
                    continue
                try:
                    with open(f, "r", encoding="utf-8", errors="replace") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                            except (ValueError, json.JSONDecodeError):
                                dropped += 1
                                continue
                            if not isinstance(entry, dict):
                                dropped += 1
                                continue
                            ts_raw = entry.get("ts", "")
                            try:
                                ts = _dt.datetime.fromisoformat(ts_raw)
                            except (TypeError, ValueError):
                                # No valid timestamp -> treat as old/junk.
                                dropped += 1
                                continue
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=_dt.timezone.utc)
                            if ts >= cutoff:
                                survivors.append((ts_raw, entry))
                            else:
                                dropped += 1
                except OSError:
                    continue

            # Sort survivors chronologically. Stable on equal ts.
            survivors.sort(key=lambda pair: pair[0])

            # Close existing handlers so the file is writable on Windows.
            for h in list(_logger.handlers):
                try:
                    h.close()
                except Exception:
                    pass
                _logger.removeHandler(h)
            _logger._nukomfy_activity_configured = False  # type: ignore[attr-defined]

            tmp = base + ".tmp"
            with open(tmp, "w", encoding="utf-8") as out:
                for _, entry in survivors:
                    out.write(json.dumps(entry, separators=(",", ":"), ensure_ascii=False))
                    out.write("\n")
            os.replace(tmp, base)

            # Wipe rotated backups; the consolidated file is the new
            # baseline for size-based rotation going forward.
            for i in range(1, _BACKUP_COUNT + 1):
                p = base + "." + str(i)
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass

            # Re-attach the handler for subsequent writes.
            setup_activity_logger()
            return dropped
        except Exception:
            _log.exception("Failed to clean up activity (non-fatal)")
            # Try to recover a usable logger even on partial failure.
            try:
                setup_activity_logger()
            except Exception:
                pass
            return 0
