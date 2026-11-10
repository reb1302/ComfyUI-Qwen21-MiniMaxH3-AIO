"""Shared tamper-tracking scaffolding for the manager's state files.

A single in-memory cache backed by a JSON file on disk, implementing the
two-layer integrity model used across the manager (see ``integrity.py``):

  - Boot / first lazy load: a SHA-256 public fingerprint rejects
    externally modified content before any admin password is available.
    A missing fingerprint trusts the current content and records it
    (trust-on-first-use).
  - Runtime: an mtime/size baseline flags later external edits, deletes,
    or re-creations.

``state.py``, ``auth.py``, ``activity_config.py`` and
``job_history_config.py`` each instantiate one tracker and expose thin
public wrappers. The integrity functions and the path resolver are
injected, so this module depends only on the standard library: no import
of ``integrity``/``paths``, hence no import cycle.

The auth divergence is absorbed uniformly:
  - ``default=None`` makes "no record" a legitimate cached value, so a
    ``_bootstrapped`` flag (not ``cached is not None``) gates re-reads.
    For the dict-default trackers the flag is equivalent: their cache is
    never legitimately ``None`` once loaded.
  - ``normalise=None`` stores the parsed dict verbatim (auth keeps its
    scrypt record untouched); the others pass their ``_normalise``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading


class TamperTracked:
    def __init__(self, *, path_fn, tracked_key, default, fingerprint_fn,
                 record_public_fn, normalise=None, logger=None):
        self._path_fn = path_fn
        self._tracked_key = tracked_key
        self._default = default
        self._fingerprint_fn = fingerprint_fn
        self._record_public_fn = record_public_fn
        self._normalise = normalise or (lambda d: d)
        self._log = logger or logging.getLogger(__name__)

        self._lock = threading.RLock()
        self._cached: dict | None = None
        self._bootstrapped = False
        self._baseline_mtime: int | None = None
        self._baseline_size: int | None = None
        self._tamper = False

    def _copy_default(self) -> dict | None:
        return dict(self._default) if self._default is not None else None

    def _snapshot_baseline(self, path: str) -> None:
        try:
            st = os.stat(path)
            self._baseline_mtime = st.st_mtime_ns
            self._baseline_size = st.st_size
        except OSError:
            self._baseline_mtime = None
            self._baseline_size = None

    def _load_disk(self) -> None:
        if self._bootstrapped:
            return
        path = self._path_fn()
        # A missing file raises OSError here, the same outcome as an
        # explicit exists() check: cache the default and bootstrap.
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            self._cached = self._copy_default()
            self._bootstrapped = True
            self._snapshot_baseline(path)
            return
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._cached = self._copy_default()
            self._bootstrapped = True
            self._snapshot_baseline(path)
            return
        if not isinstance(data, dict):
            self._cached = self._copy_default()
            self._bootstrapped = True
            self._snapshot_baseline(path)
            return

        expected = self._fingerprint_fn(self._tracked_key)
        actual = hashlib.sha256(raw).hexdigest()
        if expected is None:
            self._record_public_fn(self._tracked_key, raw)
            self._cached = self._normalise(data)
            self._bootstrapped = True
            self._snapshot_baseline(path)
            return
        if actual != expected:
            self._log.warning(
                "external modification detected at boot: %s",
                self._tracked_key)
            self._cached = self._copy_default()
            self._tamper = True
            self._bootstrapped = True
            self._snapshot_baseline(path)
            return

        self._cached = self._normalise(data)
        self._bootstrapped = True
        self._snapshot_baseline(path)

    def _check_tamper(self) -> None:
        if self._tamper:
            return
        path = self._path_fn()
        try:
            st = os.stat(path)
        except FileNotFoundError:
            if (self._baseline_mtime is not None
                    or self._baseline_size is not None):
                self._log.warning(
                    "tracked file deleted externally: %s",
                    self._tracked_key)
                self._tamper = True
            return
        except OSError:
            return
        if self._baseline_mtime is None and self._baseline_size is None:
            self._log.warning(
                "tracked file appeared externally: %s",
                self._tracked_key)
            self._tamper = True
            return
        if (st.st_mtime_ns != self._baseline_mtime
                or st.st_size != self._baseline_size):
            self._log.warning(
                "external modification detected: %s",
                self._tracked_key)
            self._tamper = True

    def load(self) -> dict | None:
        """Cached value (a copy), or ``None`` for a None-default tracker."""
        with self._lock:
            self._load_disk()
            self._check_tamper()
            return dict(self._cached) if self._cached is not None else None

    def pending_tamper(self) -> bool:
        """True if a runtime or boot-time tamper has been flagged.

        Runs a fresh mtime/size check so a no-auth caller (``/ping``) sees
        runtime modifications even when no other path loaded recently.
        """
        with self._lock:
            self._load_disk()
            self._check_tamper()
            return self._tamper

    def peek_cached(self) -> dict | None:
        """Snapshot of the live cache, with no disk read or tamper check.

        Mirrors ``restore_to_cached`` reading the cache under the lock; may
        be ``None`` if no authorised value has been loaded or saved yet.
        """
        with self._lock:
            return dict(self._cached) if self._cached is not None else None

    def mark_saved(self, value: dict, path: str) -> None:
        """Adopt ``value`` as the cache after an authorised write to ``path``.

        Re-baselines mtime/size and clears the tamper flag; bootstraps so a
        later ``load`` does not re-read disk over the just-saved value.
        """
        with self._lock:
            self._cached = dict(value)
            self._bootstrapped = True
            self._snapshot_baseline(path)
            self._tamper = False
