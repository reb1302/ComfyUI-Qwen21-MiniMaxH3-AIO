"""HTTP route handlers for the Nukomfy Manager custom node.

Discovery endpoints (ping, availability, activity config) are reachable
without auth. Admin endpoints require an admin password (see
server/auth.py).
"""
from __future__ import annotations

import getpass
import json
import logging
import os
import time

from aiohttp import web

_log = logging.getLogger(__name__)
_BOOT_EPOCH = time.time()

# Grace period before killing the process on reboot, so the HTTP response
# can flush to the client first.
_RESTART_GRACE_SECONDS = 0.5
# Maximum rows returnable per paginated request (tail / history).
_MAX_PAGE_SIZE = 1000
_MAX_SEARCH_LEN = 200  # cap on the history search term length

try:
    from server import PromptServer
except ImportError:  # pragma: no cover - module imported outside ComfyUI
    PromptServer = None  # type: ignore[assignment]

from . import activity, auth, integrity
from ..version import __version__


async def _read_json_body(request: web.Request) -> dict:
    """Best-effort JSON body parse. Returns {} on parse failure."""
    try:
        data = await request.json()
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _client_ip(request: web.Request) -> str:
    """Extract the requesting client IP. Falls back to 'unknown' if absent."""
    peer = request.transport.get_extra_info("peername") if request.transport else None
    if peer and len(peer) >= 1:
        return str(peer[0])
    return request.headers.get("X-Forwarded-For", "unknown").split(",")[0].strip()


def _resolve_user(request: web.Request) -> str:
    """Best-effort identification of the actor behind a request.

    Honours the client-declared ``X-Nukomfy-User`` header when present
    and not the sentinel string "unknown". Otherwise falls back to the
    OS login session of the ComfyUI process. Returns "unknown" when no
    name can be resolved.
    """
    declared = request.headers.get("X-Nukomfy-User", "").strip()
    if declared and declared.lower() != "unknown":
        return declared
    try:
        return os.getlogin()
    except OSError:
        try:
            return getpass.getuser()
        except Exception:
            return "unknown"


def _log_activity(
    action: str,
    *,
    success: bool,
    user: str = "",
    ip: str = "",
    reason: str = "",
) -> None:
    """Forward an event to the persistent activity log.

    Failures inside ``activity.log_event`` are already swallowed at that
    layer; this wrapper exists for naming clarity at call sites.
    """
    try:
        activity.log_event(action, success=success, user=user, ip=ip, reason=reason)
    except Exception:
        pass


def _check_integrity_with_password(password: str, user: str, ip: str) -> list[str]:
    """Run ``integrity.verify_all`` and auto-restore recoverable tampering.

    Returns the list of tampered keys (pre-restore) so the caller can
    echo it back to the WebUI as a warning. Auto-recoverable files
    (``availability.json``, the two config JSONs) are restored from the
    in-memory cache, which holds the last value persisted by an
    authorised save. If the cache is empty (e.g. boot-time SHA-256
    mismatch already populated it with defaults), the restore falls back
    to defaults transparently. ``auth.json`` is never auto-reset: the
    operator must intervene manually.

    Each detected mismatch is appended to the activity log under the
    ``file_changed_externally`` action with the offending filename and
    the source of the restored value.
    """
    from . import activity_config, job_history_config, state

    try:
        tampered = integrity.verify_all(password)
    except Exception:
        return []
    if not tampered:
        return []

    for key in tampered:
        restored = False
        restore_error = ""
        try:
            if key == "availability.json":
                state.restore_to_cached(password=password)
                restored = True
            elif key == "activity_config.json":
                activity_config.restore_to_cached(password=password)
                restored = True
            elif key == "job_history_config.json":
                job_history_config.restore_to_cached(password=password)
                restored = True
            # auth.json: not auto-reset; that would discard the configured
            # password and lock the operator out. Surface only.
        except Exception as e:
            restore_error = str(e)

        reason = "file={}".format(key)
        if restored:
            reason += "; restore=cache"
        elif restore_error:
            reason += "; restore_failed={}".format(restore_error)
        _log_activity(
            "file_changed_externally", success=True, user=user, ip=ip,
            reason=reason)

    return tampered


def _set_aborting(prompt_id: str) -> bool:
    """Mark the running job *prompt_id* as aborting, in-process.

    Writes ``nfy_aborting: True`` into the running item's ``extra_data``
    (index 3 of the queue tuple) under the prompt-queue mutex, so the flag
    rides ComfyUI's own ``GET /api/queue`` response - no DB, no extra
    request, independent of the history-persistence toggle. The job keeps
    running until it honours the interrupt flag; ``task_done`` drops the
    item (and the flag) when it terminates.

    The whole tuple is swapped atomically - a fresh tuple carrying a fresh
    ``extra_data`` dict replaces the slot - rather than mutated in place,
    so a ``/api/queue`` serialization already in flight keeps reading the
    prior immutable tuple. Returns True if the prompt was found running and
    marked, False otherwise.
    """
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return False
    pq = getattr(PromptServer.instance, "prompt_queue", None)
    if pq is None:
        return False
    try:
        with pq.mutex:
            for slot, item in list(pq.currently_running.items()):
                if len(item) > 3 and item[1] == prompt_id:
                    extra = item[3] if isinstance(item[3], dict) else {}
                    new_extra = {**extra, "nfy_aborting": True}
                    pq.currently_running[slot] = (
                        item[:3] + (new_extra,) + item[4:])
                    return True
    except Exception:
        # Unexpected change in ComfyUI's queue internals: degrade to a
        # no-op (abort reported as not-marked) rather than 500 the route.
        _log.warning("nfy_aborting mark failed for %s", prompt_id,
                     exc_info=True)
    return False


def _require_admin(action, user, ip, password):
    """Admin gate: per-IP rate limit + password check.

    Returns a rejecting ``web.Response``, or ``None`` when the caller may
    proceed. The caller is then responsible for ``auth.record_success(ip)``
    (its exact placement varies per handler). ``action`` is the activity-log
    action name recorded on rate-limit / wrong-password failures.
    """
    allowed, retry_after, evt = auth.check_rate_limit(ip)
    if evt == "block_cleared":
        _log_activity("password_block_cleared", success=True, user=user, ip=ip)
    if not allowed:
        _log_activity(action, success=False, user=user, ip=ip,
                      reason="rate_limited")
        return web.json_response(
            {"ok": False, "reason": "rate_limited",
             "retry_after_seconds": int(retry_after)},
            status=429,
        )

    if not auth.has_password():
        return web.json_response(
            {"ok": False, "reason": "no_password_set"},
            status=400,
        )

    if not isinstance(password, str) or not auth.verify_password(password, auth.load_auth()):
        block_evt = auth.record_failure(ip)
        if block_evt == "block_started":
            _log_activity("password_block_started", success=True, user=user, ip=ip)
        _log_activity(action, success=False, user=user, ip=ip,
                      reason="wrong_password")
        return web.json_response(
            {"ok": False, "reason": "wrong_password"},
            status=401,
        )

    return None


def register_routes() -> None:
    """Attach handlers to the running PromptServer instance.

    Safe to call multiple times: aiohttp will raise on duplicate routes, so we
    guard against the case where ComfyUI re-imports the module.
    """
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    routes = PromptServer.instance.routes

    @routes.get("/nukomfy/manager/ping")
    async def _ping(request: web.Request) -> web.Response:
        from . import (activity_config, job_history_config,
                       job_history_db, node_status, state)
        stored = auth.load_auth()
        password_changed_at = ""
        if isinstance(stored, dict):
            password_changed_at = str(stored.get("created_at", "") or "")
        history_enabled = True
        try:
            history_enabled = bool(
                job_history_config.load_config().get("enabled", True))
        except Exception:
            pass
        # Surface the in-memory "pending tamper" flags so the sidebar
        # banner can light up on a plain ping refresh, without waiting
        # for an admin password to trigger the HMAC check. The list is
        # informational only: it does not reveal what changed, just
        # which file is currently flagged as out-of-sync with the cache.
        pending_tamper_files: list[str] = []
        try:
            if auth.pending_tamper():
                pending_tamper_files.append("auth.json")
            if state.pending_tamper():
                pending_tamper_files.append("availability.json")
            if activity_config.pending_tamper():
                pending_tamper_files.append("activity_config.json")
            if job_history_config.pending_tamper():
                pending_tamper_files.append("job_history_config.json")
            pending_tamper_files.sort()
        except Exception:
            pending_tamper_files = []
        return web.json_response({
            "version": __version__,
            "has_password": stored is not None,
            "password_changed_at": password_changed_at,
            "availability": state.availability(),
            "has_persistent_history": True,
            "history_schema_version": job_history_db._SCHEMA_VERSION,
            "history_enabled": history_enabled,
            "boot_epoch": _BOOT_EPOCH,
            "pending_tamper_files": pending_tamper_files,
            "node_import_error": node_status.get_import_error(),
            "node_missing_module": node_status.get_missing_module(),
        })

    @routes.get("/nukomfy/availability")
    async def _availability(request: web.Request) -> web.Response:
        from . import state
        return web.json_response(state.load_state())

    @routes.get("/nukomfy/progress")
    async def _progress(request: web.Request) -> web.Response:
        """Live progress snapshot {prompt_id: {fraction, tooltip}}.

        Read-only, no auth: returns the same values already broadcast to
        every WS listener via the progress relay - strictly less than the
        public /api/queue. Lets a client reseed bars it missed (restart,
        second viewer, queue rebuild).
        """
        from . import progress_relay
        return web.json_response(progress_relay.current_progress())

    @routes.post("/nukomfy/admin/set_password")
    async def _set_password(request: web.Request) -> web.Response:
        body = await _read_json_body(request)
        new_password = body.get("new")
        current_password = body.get("current")
        user = _resolve_user(request)
        ip = _client_ip(request)

        if not isinstance(new_password, str) or len(new_password) < auth.MIN_PASSWORD_LENGTH:
            _log_activity("set_password", success=False, user=user, ip=ip,
                          reason="weak_password")
            return web.json_response(
                {"ok": False, "reason": "weak_password",
                 "min_length": auth.MIN_PASSWORD_LENGTH},
                status=400,
            )

        tampered_pre: list[str] = []
        if auth.has_password():
            stored = auth.load_auth()
            if not isinstance(current_password, str) or not auth.verify_password(current_password, stored):
                _log_activity("set_password", success=False, user=user, ip=ip,
                              reason="wrong_current")
                return web.json_response(
                    {"ok": False, "reason": "wrong_current"},
                    status=401,
                )
            # Snapshot integrity under the OLD password before re-sealing.
            # Auto-reset is skipped here: the new password is about to
            # rewrite the signatures anyway, and resetting auth-keyed
            # state mid-change would mix secrets.
            try:
                tampered_pre = integrity.verify_all(current_password)
            except Exception:
                tampered_pre = []
            for key in tampered_pre:
                _log_activity("file_changed_externally", success=True,
                              user=user, ip=ip,
                              reason="file={}; during=set_password".format(key))

        try:
            auth.save_auth(auth.hash_password(new_password),
                           password=new_password)
        except (OSError, ValueError) as e:
            _log_activity("set_password", success=False, user=user, ip=ip,
                          reason=str(e))
            return web.json_response(
                {"ok": False, "reason": "internal_error"},
                status=500,
            )

        _log_activity("set_password", success=True, user=user, ip=ip)
        return web.json_response({"ok": True, "tampered_files": tampered_pre})

    @routes.get("/nukomfy/admin/rate_limit_status")
    async def _rate_limit_status(request: web.Request) -> web.Response:
        ip = _client_ip(request)
        user = _resolve_user(request)
        allowed, retry_after, evt = auth.check_rate_limit(ip)
        if evt == "block_cleared":
            _log_activity("password_block_cleared", success=True, user=user, ip=ip)
        return web.json_response({
            "rate_limited": not allowed,
            "retry_after_seconds": int(retry_after) if not allowed else 0,
        })

    @routes.post("/nukomfy/admin/auth")
    async def _auth_endpoint(request: web.Request) -> web.Response:
        body = await _read_json_body(request)
        password = body.get("password")
        user = _resolve_user(request)
        ip = _client_ip(request)

        allowed, retry_after, evt = auth.check_rate_limit(ip)
        if evt == "block_cleared":
            _log_activity("password_block_cleared", success=True, user=user, ip=ip)
        if not allowed:
            _log_activity("auth", success=False, user=user, ip=ip,
                          reason="rate_limited")
            return web.json_response(
                {"ok": False, "reason": "rate_limited",
                 "retry_after_seconds": int(retry_after)},
                status=429,
            )

        if not auth.has_password():
            return web.json_response(
                {"ok": False, "reason": "no_password_set"},
                status=400,
            )

        if not isinstance(password, str):
            block_evt = auth.record_failure(ip)
            if block_evt == "block_started":
                _log_activity("password_block_started", success=True, user=user, ip=ip)
            _log_activity("auth", success=False, user=user, ip=ip,
                          reason="missing_password")
            return web.json_response(
                {"ok": False, "reason": "wrong_password"},
                status=401,
            )

        if auth.verify_password(password, auth.load_auth()):
            auth.record_success(ip)
            _log_activity("auth", success=True, user=user, ip=ip)
            tampered = _check_integrity_with_password(password, user, ip)
            return web.json_response({"ok": True, "tampered_files": tampered})

        block_evt = auth.record_failure(ip)
        if block_evt == "block_started":
            _log_activity("password_block_started", success=True, user=user, ip=ip)
        _log_activity("auth", success=False, user=user, ip=ip,
                      reason="wrong_password")
        return web.json_response(
            {"ok": False, "reason": "wrong_password"},
            status=401,
        )

    @routes.post("/nukomfy/admin/integrity/restore")
    async def _integrity_restore(request: web.Request) -> web.Response:
        """Run integrity check + auto-restore. Requires admin password.

        Lets the sidebar "Restore" button drive the same flow as the
        activity-log screen's own "open" path without forcing the user
        through that screen. Behaviour identical to the integrity
        verification embedded in ``/admin/auth`` and
        ``/activity/tail``: recoverable files are restored from the
        in-memory cache via ``restore_to_cached``, ``auth.json`` is
        only surfaced.
        """
        body = await _read_json_body(request)
        password = body.get("password")
        user = _resolve_user(request)
        ip = _client_ip(request)

        reject = _require_admin("integrity_restore", user, ip, password)
        if reject is not None:
            return reject

        auth.record_success(ip)
        tampered = _check_integrity_with_password(password, user, ip)
        _log_activity("integrity_restore", success=True, user=user, ip=ip,
                      reason="files={}".format(",".join(tampered) or "none"))
        return web.json_response(
            {"ok": True, "tampered_files": tampered}
        )

    @routes.post("/nukomfy/admin/reboot")
    async def _reboot(request: web.Request) -> web.Response:
        import asyncio
        from . import reboot as reboot_module

        body = await _read_json_body(request)
        password = body.get("password")
        user = _resolve_user(request)
        ip = _client_ip(request)

        reject = _require_admin("reboot", user, ip, password)
        if reject is not None:
            return reject

        auth.record_success(ip)
        _log_activity("reboot", success=True, user=user, ip=ip)
        tampered = _check_integrity_with_password(password, user, ip)

        async def _delayed_restart() -> None:
            await asyncio.sleep(_RESTART_GRACE_SECONDS)
            reboot_module.restart_comfyui()

        asyncio.create_task(_delayed_restart())
        return web.json_response({"ok": True, "tampered_files": tampered})

    @routes.post("/nukomfy/admin/force_abort")
    async def _force_abort(request: web.Request) -> web.Response:
        import nodes  # ComfyUI core; provides interrupt_processing()

        body = await _read_json_body(request)
        password = body.get("password")
        prompt_id = body.get("prompt_id")
        # Plugin-side human-readable id (7-char Base62) carried through
        # extra_data at submit time; logged in place of the long ComfyUI
        # UUID when present. Falls back to prompt_id for raw API callers.
        nfy_job_id = (body.get("nfy_job_id") or "").strip()
        affected_user = body.get("affected_user") or "unknown"
        user = _resolve_user(request)
        ip = _client_ip(request)

        reject = _require_admin("force_abort", user, ip, password)
        if reject is not None:
            return reject

        if not isinstance(prompt_id, str) or not prompt_id:
            _log_activity("force_abort", success=False, user=user, ip=ip,
                          reason="missing_prompt_id")
            return web.json_response(
                {"ok": False, "reason": "missing_prompt_id"},
                status=400,
            )

        auth.record_success(ip)

        # Mark the running job aborting (every client then sees "Aborting…"
        # via /api/queue) and interrupt it. Silent no-op if prompt_id is not
        # the running prompt - same semantics as native /api/interrupt.
        target_running = _set_aborting(prompt_id)
        if target_running:
            nodes.interrupt_processing()

        _log_activity("force_abort", success=True, user=user, ip=ip,
                      reason="job_id={}; affected_user={}".format(
                          nfy_job_id or prompt_id, affected_user))
        tampered = _check_integrity_with_password(password, user, ip)
        return web.json_response(
            {"ok": True, "action_performed": target_running,
             "tampered_files": tampered}
        )

    @routes.post("/nukomfy/abort")
    async def _abort(request: web.Request) -> web.Response:
        """Abort the running job: mark it aborting, then interrupt it.

        Non-privileged: this is the path the normal Abort button uses.
        The password-gated /nukomfy/admin/force_abort path sets the same
        flag and additionally runs the manager's integrity check.

        Returns 200 {"ok": true} when the prompt was running and is now
        marked aborting; 409 {"ok": false, "reason": "not_running"} when it
        is not the running prompt (already finished, or never started), so
        the client can tell that apart from an unreachable machine.
        """
        import nodes  # ComfyUI core; provides interrupt_processing()

        body = await _read_json_body(request)
        prompt_id = body.get("prompt_id")
        user = _resolve_user(request)
        ip = _client_ip(request)

        if not isinstance(prompt_id, str) or not prompt_id:
            return web.json_response(
                {"ok": False, "reason": "missing_prompt_id"}, status=400)

        if not _set_aborting(prompt_id):
            _log_activity("abort", success=False, user=user, ip=ip,
                          reason="not_running")
            return web.json_response(
                {"ok": False, "reason": "not_running"}, status=409)

        nodes.interrupt_processing()
        _log_activity("abort", success=True, user=user, ip=ip)
        return web.json_response({"ok": True})

    @routes.post("/nukomfy/admin/force_remove")
    async def _force_remove(request: web.Request) -> web.Response:
        body = await _read_json_body(request)
        password = body.get("password")
        prompt_id = body.get("prompt_id")
        nfy_job_id = (body.get("nfy_job_id") or "").strip()
        affected_user = body.get("affected_user") or "unknown"
        user = _resolve_user(request)
        ip = _client_ip(request)

        reject = _require_admin("force_remove", user, ip, password)
        if reject is not None:
            return reject

        if not isinstance(prompt_id, str) or not prompt_id:
            _log_activity("force_remove", success=False, user=user, ip=ip,
                          reason="missing_prompt_id")
            return web.json_response(
                {"ok": False, "reason": "missing_prompt_id"},
                status=400,
            )

        auth.record_success(ip)

        # Silent no-op if prompt_id is not in the pending queue (same
        # semantics as ComfyUI's native /api/queue?delete=[...]).
        # Guard the prompt-server deref like _set_aborting: on a very early
        # boot / teardown window instance can be None - return a clean 500
        # instead of an uncaught AttributeError traceback.
        pq = None
        if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
            pq = getattr(PromptServer.instance, "prompt_queue", None)
        if pq is None:
            _log_activity("force_remove", success=False, user=user, ip=ip,
                          reason="internal_error: prompt server unavailable")
            return web.json_response(
                {"ok": False, "reason": "internal_error"},
                status=500,
            )
        try:
            pq.delete_queue_item(lambda a: len(a) >= 2 and a[1] == prompt_id)
        except Exception as e:
            _log_activity("force_remove", success=False, user=user, ip=ip,
                          reason="internal_error: {}".format(e))
            return web.json_response(
                {"ok": False, "reason": "internal_error"},
                status=500,
            )

        _log_activity("force_remove", success=True, user=user, ip=ip,
                      reason="job_id={}; affected_user={}".format(
                          nfy_job_id or prompt_id, affected_user))
        tampered = _check_integrity_with_password(password, user, ip)
        return web.json_response({"ok": True, "tampered_files": tampered})

    @routes.post("/nukomfy/availability/set")
    async def _availability_set(request: web.Request) -> web.Response:
        from . import state

        body = await _read_json_body(request)
        password = body.get("password")
        new_value = body.get("availability")
        user = _resolve_user(request)
        ip = _client_ip(request)

        reject = _require_admin("availability_set", user, ip, password)
        if reject is not None:
            return reject

        if new_value not in ("available", "unavailable"):
            _log_activity("availability_set", success=False, user=user, ip=ip,
                          reason="invalid_value")
            return web.json_response(
                {"ok": False, "reason": "invalid_value"},
                status=400,
            )

        auth.record_success(ip)
        tampered = _check_integrity_with_password(password, user, ip)

        try:
            saved = state.save_state(new_value, changed_by=user,
                                     password=password)
        except (OSError, ValueError) as e:
            _log_activity("availability_set", success=False, user=user, ip=ip,
                          reason="internal_error: {}".format(e))
            return web.json_response(
                {"ok": False, "reason": "internal_error"},
                status=500,
            )

        _log_activity("availability_set", success=True, user=user, ip=ip,
                      reason="value={}".format(new_value))
        return web.json_response({"ok": True, "state": saved,
                                  "tampered_files": tampered})

    @routes.post("/nukomfy/activity/tail")
    async def _activity_tail(request: web.Request) -> web.Response:
        body = await _read_json_body(request)
        password = body.get("password")
        raw_n = body.get("n", 200)
        user = _resolve_user(request)
        ip = _client_ip(request)

        try:
            n = int(raw_n)
        except (TypeError, ValueError):
            n = 200
        if n < 1:
            n = 1
        elif n > _MAX_PAGE_SIZE:
            n = _MAX_PAGE_SIZE

        reject = _require_admin("activity_tail", user, ip, password)
        if reject is not None:
            return reject

        auth.record_success(ip)
        tampered = _check_integrity_with_password(password, user, ip)
        # Reading the log is not itself recorded: read access is outside
        # the audit trail.
        entries = activity.tail(n)
        return web.json_response(
            {"ok": True, "entries": entries, "tampered_files": tampered}
        )

    @routes.get("/nukomfy/history/{nfy_job_id}")
    async def _history_by_nfy_id(request: web.Request) -> web.Response:
        """Nukomfy short-id alias for ComfyUI's ``GET /api/history/{prompt_id}``.

        Looks up ``nfy_job_id`` (Base62 short id Nukomfy assigns at submit
        time and stores in ``extra_data``) in the ComfyUI runtime history
        and returns **exactly** the same payload ComfyUI's native endpoint
        would return if queried by the matching ``prompt_id``. Empty
        object ``{}`` if the id is unknown, mirroring ComfyUI's native
        behaviour for missing / not-yet-completed prompts.

        Read-only, no auth: surfaces only data that ComfyUI's own
        ``/api/history`` already exposes publicly.
        """
        target = (request.match_info.get("nfy_job_id") or "").strip()
        if not target:
            return web.json_response({})
        try:
            history = PromptServer.instance.prompt_queue.get_history()
        except Exception:
            return web.json_response({})
        for pid, entry in history.items():
            if not isinstance(entry, dict):
                continue
            prompt_arr = entry.get("prompt", [])
            if len(prompt_arr) < 4:
                continue
            edata = prompt_arr[3]
            if isinstance(edata, dict) and str(edata.get("nfy_job_id") or "") == target:
                return web.json_response({pid: entry})
        return web.json_response({})

    @routes.get("/nukomfy/activity/config")
    async def _activity_config_get(request: web.Request) -> web.Response:
        from . import activity_config
        return web.json_response(activity_config.load_config())

    @routes.post("/nukomfy/activity/config/set")
    async def _activity_config_set(request: web.Request) -> web.Response:
        import asyncio

        from . import activity_config

        body = await _read_json_body(request)
        password = body.get("password")
        max_days_raw = body.get("max_days_to_keep")
        user = _resolve_user(request)
        ip = _client_ip(request)

        reject = _require_admin("activity_config_set", user, ip, password)
        if reject is not None:
            return reject

        auth.record_success(ip)
        tampered = _check_integrity_with_password(password, user, ip)

        try:
            saved = activity_config.save_config(max_days_raw, password=password)
        except ValueError:
            _log_activity("activity_config_set", success=False, user=user, ip=ip,
                          reason="invalid_value")
            return web.json_response(
                {"ok": False, "reason": "invalid_value",
                 "tampered_files": tampered},
                status=400,
            )
        except OSError as e:
            _log_activity("activity_config_set", success=False, user=user, ip=ip,
                          reason="internal_error: {}".format(e))
            return web.json_response(
                {"ok": False, "reason": "internal_error",
                 "tampered_files": tampered},
                status=500,
            )

        max_days = saved.get("max_days_to_keep", 0)
        _log_activity("activity_config_set", success=True, user=user, ip=ip,
                      reason="max_days={}".format(max_days))

        # Apply the new retention immediately. Fire-and-forget so the
        # response isn't blocked by potentially large log rewrites.
        async def _async_cleanup() -> None:
            activity.cleanup_old(int(max_days))
        asyncio.create_task(_async_cleanup())

        return web.json_response({"ok": True, "config": saved,
                                  "tampered_files": tampered})

    @routes.get("/nukomfy/jobs/history")
    async def _history_list(request: web.Request) -> web.Response:
        """Paginated list of persisted job history entries.

        Query params: ``limit`` (default 100, clamped 1..1000),
        ``offset`` (default 0, clamped >=0), ``since`` (ISO-8601 string;
        lexicographic compare against ``nfy_sent_at``). Non-numeric
        ``limit``/``offset`` fall back to defaults. Unparseable ``since``
        yields an empty result (DB compare returns no rows). ``status``
        (comma-separated ``nfy_status_str`` values) and ``q`` (case-
        insensitive substring over the identity columns) are optional
        filters; absent ones leave the query unfiltered.

        Read-only, no auth: history is consultable even when the feature
        is disabled via config (write path off, read path always on).
        """
        from . import job_history_db

        raw_limit = request.query.get("limit")
        try:
            limit = int(raw_limit) if raw_limit is not None else 100
        except (TypeError, ValueError):
            limit = 100
        if limit < 1:
            limit = 1
        elif limit > _MAX_PAGE_SIZE:
            limit = _MAX_PAGE_SIZE

        raw_offset = request.query.get("offset")
        try:
            offset = int(raw_offset) if raw_offset is not None else 0
        except (TypeError, ValueError):
            offset = 0
        if offset < 0:
            offset = 0

        since = request.query.get("since")
        if since is not None:
            since = since.strip() or None

        raw_status = request.query.get("status")
        statuses = None
        if raw_status:
            statuses = [s.strip() for s in raw_status.split(",") if s.strip()]

        raw_q = request.query.get("q")
        query = raw_q.strip()[:_MAX_SEARCH_LEN] if raw_q else None
        if not query:
            query = None

        entries = job_history_db.get_history(
            limit=limit, offset=offset,
            only_terminal=False, since_iso=since,
            statuses=statuses, query=query,
        ) or []
        total = job_history_db.count_history(
            only_terminal=False, since_iso=since,
            statuses=statuses, query=query,
        )
        return web.json_response({
            "ok": True,
            "entries": entries,
            "total_count": total,
        })

    @routes.get("/nukomfy/jobs/history/{prompt_id}")
    async def _history_by_prompt_id(request: web.Request) -> web.Response:
        """Single persisted entry by exact ``prompt_id``.

        Returns ``200 {"ok": true, "entry": {...}}`` when found, or
        ``404 {"ok": false, "reason": "not_found"}`` otherwise. Whitespace
        or empty path components map to 404.
        """
        from . import job_history_db

        prompt_id = (request.match_info.get("prompt_id") or "").strip()
        if not prompt_id:
            return web.json_response(
                {"ok": False, "reason": "not_found"}, status=404)

        entry = job_history_db.get_by_prompt_id(prompt_id)
        if entry is None:
            return web.json_response(
                {"ok": False, "reason": "not_found"}, status=404)
        return web.json_response({"ok": True, "entry": entry})

    @routes.get("/nukomfy/jobs/history/{prompt_id}/workflow_api")
    async def _history_workflow_api_get(request: web.Request) -> web.Response:
        """Read the workflow_api JSON snapshot for a persisted entry.

        The blob is excluded from list/single responses to keep them
        compact; this dedicated endpoint loads it on demand. Returns 404
        when the prompt_id is unknown or no snapshot was recorded.
        """
        from . import job_history_db

        prompt_id = (request.match_info.get("prompt_id") or "").strip()
        if not prompt_id:
            return web.json_response(
                {"ok": False, "reason": "not_found"}, status=404)

        payload = job_history_db.load_workflow_api(prompt_id)
        if payload is None:
            return web.json_response(
                {"ok": False, "reason": "not_found"}, status=404)
        return web.json_response(
            {"ok": True, "prompt_id": prompt_id, "workflow_api": payload})

    @routes.post("/nukomfy/jobs/history/clear")
    async def _history_clear(request: web.Request) -> web.Response:
        """Drop every row from ``job_history.db``. Requires admin password.

        The table itself stays in place; subsequent submissions repopulate
        it from scratch. When the feature is paused (``enabled=false``)
        no further rows will be added until it's resumed.
        """
        from . import job_history_db

        body = await _read_json_body(request)
        password = body.get("password")
        user = _resolve_user(request)
        ip = _client_ip(request)

        reject = _require_admin("history_clear", user, ip, password)
        if reject is not None:
            return reject

        try:
            job_history_db.clear_history()
        except Exception as e:
            _log_activity("history_clear", success=False, user=user, ip=ip,
                          reason="internal_error: {}".format(e))
            return web.json_response(
                {"ok": False, "reason": "internal_error"},
                status=500,
            )

        auth.record_success(ip)
        _log_activity("history_clear", success=True, user=user, ip=ip)
        tampered = _check_integrity_with_password(password, user, ip)
        return web.json_response({"ok": True, "tampered_files": tampered})

    @routes.get("/nukomfy/jobs/history/config")
    async def _history_config_get(request: web.Request) -> web.Response:
        """Read the persisted retention config (no auth)."""
        from . import job_history_config
        return web.json_response(job_history_config.load_config())

    @routes.post("/nukomfy/jobs/history/config/set")
    async def _history_config_set(request: web.Request) -> web.Response:
        """Update the retention config. Requires admin password.

        Accepts a subset of ``{enabled, max_jobs_to_keep,
        max_days_to_keep}``. Unknown keys or out-of-range values result
        in HTTP 400 ``invalid_value``. The ``enabled`` flag takes effect
        on next ComfyUI restart; the two numeric values are runtime-live.
        """
        from . import job_history_config

        body = await _read_json_body(request)
        password = body.get("password")
        user = _resolve_user(request)
        ip = _client_ip(request)

        reject = _require_admin("history_config_set", user, ip, password)
        if reject is not None:
            return reject

        updates = {k: v for k, v in body.items() if k != "password"}

        auth.record_success(ip)
        tampered = _check_integrity_with_password(password, user, ip)

        try:
            saved = job_history_config.save_config(password=password, **updates)
        except ValueError:
            _log_activity("history_config_set", success=False, user=user, ip=ip,
                          reason="invalid_value")
            return web.json_response(
                {"ok": False, "reason": "invalid_value",
                 "tampered_files": tampered},
                status=400,
            )
        except OSError as e:
            _log_activity("history_config_set", success=False, user=user, ip=ip,
                          reason="internal_error: {}".format(e))
            return web.json_response(
                {"ok": False, "reason": "internal_error",
                 "tampered_files": tampered},
                status=500,
            )

        reason_parts = [
            "{}={}".format(k, saved.get(k)) for k in updates.keys()
        ]
        _log_activity("history_config_set", success=True, user=user, ip=ip,
                      reason="; ".join(reason_parts) or "noop")
        return web.json_response({"ok": True, "config": saved,
                                  "tampered_files": tampered})


try:
    register_routes()
except Exception:  # pragma: no cover
    _log.exception("Failed to register routes")
