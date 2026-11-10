"""Server-side modules for the Nukomfy Suite custom node."""
from __future__ import annotations

import logging

from . import routes  # noqa: F401  - registers HTTP routes on import

_log = logging.getLogger(__name__)


def _boot_cleanup_activity() -> None:
    """Prune time-expired activity entries once at startup.

    Reads the configured retention threshold and rewrites the log file.
    Errors are best-effort: a failed cleanup must not break ComfyUI boot.
    """
    try:
        from . import activity, activity_config
        cfg = activity_config.load_config()
        days = int(cfg.get("max_days_to_keep", 0) or 0)
        if days > 0:
            activity.cleanup_old(days)
    except Exception:
        _log.exception("Failed to clean up activity at boot (non-fatal)")


def _boot_prune_history() -> None:
    """Apply retention to job_history.db once at startup.

    No-op when the feature is disabled. Errors are best-effort: a failed
    prune must not break ComfyUI boot.
    """
    try:
        from . import job_history_config, job_history_db as db
        cfg = job_history_config.load_config()
        if not cfg.get("enabled", True):
            return
        max_days = int(cfg.get("max_days_to_keep") or 0)
        max_jobs = int(cfg.get("max_jobs_to_keep") or 0)
        if max_days > 0 or max_jobs > 0:
            db.prune(max_days, max_jobs)
    except Exception:
        _log.exception("Failed to prune history at boot (non-fatal)")


def _boot_start_job_history() -> None:
    """Start the job-history backend at boot.

    Installs the prompt submit hook plus the 1 Hz polling thread that
    persists job lifecycle to job_history.db. Failures are best-effort:
    a broken history backend must not block ComfyUI boot.
    """
    try:
        from . import job_history
        job_history.start()
    except Exception:
        _log.exception("Failed to start job history at boot (non-fatal)")


def _boot_start_progress_relay() -> None:
    """Start the live progress broadcast relay at boot.

    Lets every connected client see the running job's progress, not only
    the submitter. Best-effort: a failure must not block ComfyUI boot.
    """
    try:
        from . import progress_relay
        progress_relay.start()
    except Exception:
        _log.exception("Failed to start progress relay at boot (non-fatal)")


def _boot_bootstrap_caches() -> None:
    """Force-populate the in-memory caches of all tracked state modules.

    The activity-config and job-history-config caches are seeded by the
    boot helpers below, but ``auth.json`` and ``availability.json`` have
    no other boot consumer. Calling their loaders here snapshots the
    mtime/size baseline and runs the SHA-256 fingerprint check while
    nothing else is touching the files yet.
    """
    try:
        from . import auth, state
        auth.load_auth()
        state.load_state()
    except Exception:
        _log.exception("Failed to bootstrap cache at boot (non-fatal)")


_boot_bootstrap_caches()
_boot_cleanup_activity()
_boot_prune_history()
_boot_start_job_history()
_boot_start_progress_relay()
