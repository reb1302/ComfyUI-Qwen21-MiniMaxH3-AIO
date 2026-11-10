"""Event hooks + write path for the persistent job history.

Glues ComfyUI's prompt lifecycle to the persistent DB declared in
``job_history_db.py``. Two paths feed the DB:

1. **Submit hook** (``PromptServer.add_on_prompt_handler``): fires when
   ComfyUI accepts a POST /prompt, *before* the prompt is validated and
   queued. We read ``extra_data`` and emit ``record_submit`` with zero
   latency and full workflow snapshot access. The row starts as
   ``submitted``: a POST being accepted is not proof the prompt is
   runnable (validation can still reject it for a missing model, bad
   value, ...), so the durable state stays provisional until the polling
   thread observes the prompt live in the runtime.

2. **Polling thread** (1 Hz): diffs ``currently_running``, ``pq.queue``
   and ``pq.get_history()`` against the previous snapshot to emit:
   - ``queued`` (``submitted`` -> ``pending``, once observed live: the
     prompt passed validation and entered the queue)
   - ``started`` (queue -> running)
   - ``terminal`` (entry materialises in history)
   - ``cancelled`` (pid disappears from running without a history entry;
     force_abort cleanup). Pids removed from the queue while still
     pending are deleted instead - they never reached execution, and
     ComfyUI's own /api/jobs has never tracked queue-cancel events
     either, so the persistent layer mirrors that.
   - ``reaped`` (a ``submitted`` row never observed live after a short
     grace: validation rejected the prompt, so the provisional row is
     deleted and never surfaces in any view).

Only Nukomfy-submitted prompts are tracked (gated on ``extra_data
.nfy_job_id``). Foreign prompts (e.g. native WebUI submissions) are
ignored, the DB stays clean.

Boot reconciliation freezes DB rows still in-flight after a restart to
``failed`` (reason ``server_restart``) - ComfyUI does not track jobs lost
to a crash, so the Nukomfy client owns the lost-detection semantics via
its own client-side timeout. Idempotent.

Every DB call is best-effort: a failure logs but never propagates to
ComfyUI.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Set, Tuple

from . import job_history_db as db

_log = logging.getLogger(__name__)


_POLL_INTERVAL_S = 1.0

# Pid -> consecutive cycles seen "missing" before we declare it cancelled.
# Lets us tolerate the 1-frame window where a pid has left running but not
# yet materialised in get_history().
_RUNNING_MISSING_GRACE = 3

# Pid -> consecutive cycles a 'submitted' row stays unobserved before we
# conclude the prompt was rejected at validation and reap it. A valid prompt
# is enqueued synchronously, so it shows up on the next poll; this only bounds
# how long an unproven submit lingers before cleanup.
_SUBMITTED_MISSING_GRACE = 3

_started = False
_polling_thread: Optional[threading.Thread] = None
_polling_stop = threading.Event()


# ---------------------------------------------------------------------------
# extra_data -> record_submit kwargs
# ---------------------------------------------------------------------------

def _extract_record_kwargs(prompt_id: str,
                           extra_data: Dict[str, Any],
                           workflow_api: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Build the kwargs dict for ``db.record_submit`` from extra_data.

    Returns None if extra_data isn't from a Nukomfy submit (no nfy_job_id),
    in which case the caller MUST skip.
    """
    if not isinstance(extra_data, dict):
        return None
    nfy_job_id = extra_data.get("nfy_job_id")
    if not nfy_job_id:
        return None

    meta = extra_data.get("nfy_workflow_meta") or {}
    env = extra_data.get("nfy_environment") or {}
    server_snapshot = extra_data.get("nfy_server_snapshot") or {}

    return dict(
        prompt_id=prompt_id,
        sent_at=extra_data.get("nfy_sent_at"),
        machine_name=extra_data.get("nfy_machine_name", ""),
        machine_url=extra_data.get("nfy_machine_url", ""),
        workflow_name=extra_data.get("nfy_workflow_name", ""),
        frame_range=extra_data.get("nfy_frame_range") or [],
        nk_file=extra_data.get("nfy_nk_file", ""),
        node_name=extra_data.get("nfy_node_name", ""),
        output_paths=extra_data.get("nfy_output_paths") or [],
        batch_count=int(extra_data.get("nfy_batch_count") or 1),
        batch_index=int(extra_data.get("nfy_batch_index") or 1),
        seeds_used=extra_data.get("nfy_seeds_used"),
        input_ranges=extra_data.get("nfy_input_ranges"),
        nfy_job_id=nfy_job_id,
        nfy_submitted_by=extra_data.get("nfy_submitted_by", ""),
        nfy_submitter_host=extra_data.get("nfy_submitter_host", ""),
        read_color=0,
        workflow_uuid=meta.get("workflow_uuid"),
        workflow_author=meta.get("workflow_author"),
        workflow_version=meta.get("workflow_version"),
        params_spec=meta.get("params_spec"),
        write_templates=meta.get("write_templates"),
        os_submitter=env.get("os_submitter"),
        nuke_version=env.get("nuke_version"),
        nukomfy_version=env.get("nukomfy_version"),
        input_cache=env.get("input_cache"),
        system_stats=server_snapshot.get("system_stats"),
        server_version=server_snapshot.get("server_version"),
        workflow_api_payload=workflow_api,
    )


# ---------------------------------------------------------------------------
# Submit hook (PromptServer.add_on_prompt_handler)
# ---------------------------------------------------------------------------

def _on_prompt(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Record the submit before ComfyUI enqueues the prompt."""
    try:
        if not isinstance(json_data, dict):
            return json_data
        prompt_id = json_data.get("prompt_id")
        extra_data = json_data.get("extra_data") or {}
        workflow_api = json_data.get("prompt")
        if not prompt_id:
            return json_data
        kwargs = _extract_record_kwargs(prompt_id, extra_data, workflow_api)
        if kwargs is None:
            return json_data
        # Idempotent: re-submit of the same prompt_id (unlikely) would hit
        # the UNIQUE constraint. Skip if we already have the row.
        if db.get_by_prompt_id(prompt_id) is not None:
            return json_data
        db.record_submit(**kwargs)
    except Exception:
        _log.exception("Failed to run submit hook (non-fatal)")
    return json_data


def _install_submit_hook() -> bool:
    try:
        from server import PromptServer  # type: ignore  # ComfyUI core
    except Exception:
        return False
    inst = getattr(PromptServer, "instance", None)
    if inst is None:
        return False
    fn = getattr(inst, "add_on_prompt_handler", None)
    if not callable(fn):
        return False
    try:
        fn(_on_prompt)
        return True
    except Exception:
        _log.exception("Failed to install submit hook")
        return False


# ---------------------------------------------------------------------------
# Polling thread (1 Hz)
# ---------------------------------------------------------------------------

def _snapshot() -> Optional[Tuple[
        Dict[str, Dict[str, Any]],   # running:  pid -> extra_data
        Dict[str, Dict[str, Any]],   # pending:  pid -> extra_data
        Dict[str, str]]]:            # terminal: pid -> raw status_str
    """Read the live queue state. Returns None if PromptServer unavailable."""
    try:
        from server import PromptServer  # type: ignore
        pq = PromptServer.instance.prompt_queue
    except Exception:
        return None

    running: Dict[str, Dict[str, Any]] = {}
    pending: Dict[str, Dict[str, Any]] = {}
    terminal: Dict[str, str] = {}

    try:
        for v in pq.currently_running.values():
            if isinstance(v, (list, tuple)) and len(v) >= 4:
                pid = str(v[1])
                extra = v[3] if isinstance(v[3], dict) else {}
                running[pid] = extra
    except Exception:
        pass

    try:
        for entry in getattr(pq, "queue", []) or []:
            if isinstance(entry, (list, tuple)) and len(entry) >= 4:
                pid = str(entry[1])
                extra = entry[3] if isinstance(entry[3], dict) else {}
                pending[pid] = extra
    except Exception:
        pass

    try:
        hist = pq.get_history()
        for pid, h in (hist or {}).items():
            if not isinstance(h, dict):
                continue
            status = h.get("status")
            if isinstance(status, dict):
                terminal[str(pid)] = str(status.get("status_str", "success"))
            else:
                terminal[str(pid)] = "success"
    except Exception:
        pass

    return running, pending, terminal


def _is_nukomfy_pid(extra: Dict[str, Any]) -> bool:
    return bool(isinstance(extra, dict) and extra.get("nfy_job_id"))


def _parse_messages(messages: Any) -> Tuple[Optional[int], Optional[int], bool,
                                            Optional[Dict[str, Any]]]:
    """Extract timing + interruption flag + error payload from status.messages.

    Mirrors ComfyUI core's `normalize_history_item` in `comfy_execution/jobs.py`,
    which is what populates the `execution_start_time`/`execution_end_time`/
    `status` fields on the `/api/jobs/{id}` endpoint - the same endpoint the
    Nukomfy client reads in `comfy_api.py:_parse_job_detail`. Building the
    same fields here means client and server compute identical values from
    the same source of truth.

    Returns ``(start_ms, end_ms, was_interrupted, error_payload)``.
    """
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    was_interrupted = False
    error_payload: Optional[Dict[str, Any]] = None
    if not isinstance(messages, list):
        return (start_ms, end_ms, was_interrupted, error_payload)
    end_events = ("execution_success", "execution_error", "execution_interrupted")
    for entry in messages:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        name, payload = entry[0], entry[1]
        if not isinstance(payload, dict):
            continue
        ts = payload.get("timestamp")
        if name == "execution_start" and start_ms is None:
            if isinstance(ts, (int, float)):
                start_ms = int(ts)
        elif name in end_events:
            if isinstance(ts, (int, float)):
                end_ms = int(ts)  # last one wins (some workflows emit multiple)
            if name == "execution_error":
                error_payload = payload
            elif name == "execution_interrupted":
                was_interrupted = True
    return (start_ms, end_ms, was_interrupted, error_payload)


def _duration_seconds(start_ms: Optional[int], end_ms: Optional[int]) -> float:
    """Same formula as the client: ``max(0, end - start)`` in seconds."""
    if start_ms is None or end_ms is None:
        return 0.0
    delta = end_ms - start_ms
    return delta / 1000.0 if delta > 0 else 0.0


def _resolve_status(raw_status_str: str, was_interrupted: bool) -> str:
    """Map ComfyUI's status_str to our DB enum.

    Mirrors the client's mapping in ``comfy_api.py:_parse_job_detail``
    (which itself reads the value pre-normalised by ComfyUI core's
    `normalize_history_item`).
    """
    if raw_status_str == "success":
        return "completed"
    if raw_status_str == "interrupted":
        return "cancelled"
    if raw_status_str == "error":
        return "cancelled" if was_interrupted else "failed"
    return "failed"


def _history_entry_terminal(pq, pid: str) -> Tuple[float, Dict[str, Any]]:
    """Pull duration + execution_error + messages + outputs for a terminal pid.

    Returns ``(duration_seconds, terminal_payload)`` where ``terminal_payload``
    matches the keyword args of ``db.persist_terminal_state``.
    """
    try:
        h = pq.get_history().get(pid)
    except Exception:
        h = None
    if not isinstance(h, dict):
        return (0.0, {})
    status = h.get("status") if isinstance(h.get("status"), dict) else {}
    raw_status_str = status.get("status_str", "success")
    messages = status.get("messages") if isinstance(status, dict) else None
    outputs = h.get("outputs") if isinstance(h.get("outputs"), dict) else {}

    start_ms, end_ms, was_interrupted, error_payload = _parse_messages(messages)
    duration = _duration_seconds(start_ms, end_ms)
    db_status = _resolve_status(raw_status_str, was_interrupted)

    return duration, dict(
        status_str=db_status,
        duration=duration,
        execution_error=error_payload,
        outputs_count=len(outputs) if isinstance(outputs, dict) else None,
        messages=messages if isinstance(messages, list) else None,
        outputs=outputs if isinstance(outputs, dict) else None,
    )


def _written_frame_count(outputs) -> int:
    """Total frames written across all NukomfyWrite nodes, from the
    `nukomfy_written` ui payload. 0 when absent."""
    total = 0
    if isinstance(outputs, dict):
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for rec in node_out.get("nukomfy_written") or []:
                if not isinstance(rec, dict):
                    continue
                count = rec.get("count")
                if isinstance(count, int):
                    total += count
                else:
                    paths_list = rec.get("paths")
                    total += len(paths_list) if isinstance(paths_list, list) else 0
    return total


def _resolve_range_from_count(pid: str, outputs) -> None:
    """Resolve the Sequence output range end from the server-confirmed
    write count and persist it, so the history view shows the full range
    without any client-side disk glob (mirrors the client fallback).

    No-op unless exactly one range entry is unresolved (end is None),
    keeping multi-output jobs unambiguous. io_write writes frames
    contiguously from frame_start, so end = start + count - 1.
    """
    try:
        count = _written_frame_count(outputs)
        if count <= 0:
            return
        entry = db.get_by_prompt_id(pid)
        if not isinstance(entry, dict) or entry.get("nfy_range_resolved"):
            return
        fr = entry.get("nfy_frame_range")
        if not isinstance(fr, list):
            return
        pending = [it for it in fr if isinstance(it, dict)
                   and it.get("end") is None and isinstance(it.get("start"), int)]
        if len(pending) != 1:
            return
        pending[0]["end"] = pending[0]["start"] + count - 1
        db.update_entry(pid, nfy_frame_range=fr, nfy_range_resolved=True)
    except Exception:
        _log.exception("Failed to resolve output range (pid=%s)", pid)


def _poll_once(snap, prev_running, prev_pending, prev_terminal,
               running_grace, submitted_grace):
    """Process one polling cycle against ``snap`` and emit the DB transitions.

    Split out of the loop so the transition logic (promote, reap, run,
    terminal, queue-cancel) can be unit-tested with synthetic snapshots.
    Mutates the two grace dicts in place; returns the new
    ``(prev_running, prev_pending, prev_terminal)`` for the next cycle.
    """
    running_map, pending_map, terminal_map = snap
    running_pids = set(running_map.keys())
    pending_pids = set(pending_map.keys())
    terminal_pids = set(terminal_map.keys())

    # ----- submitted -> pending: the first time a prompt is observed live
    # in the queue it has passed validation, so promote the provisional
    # row. No-op unless the row is still 'submitted'.
    for pid in pending_pids - prev_pending:
        if not _is_nukomfy_pid(pending_map.get(pid) or {}):
            continue
        try:
            db.mark_queued(pid)
        except Exception:
            _log.exception("Failed to mark job queued (pid=%s)", pid)

    # ----- reap 'submitted' rows the runtime never acknowledged: a prompt
    # rejected by validate_prompt (missing model, bad value, ...) is
    # recorded by the hook but never enqueued, so it is never seen live
    # here. A valid prompt is enqueued synchronously and shows up on the
    # next poll, so a few cycles of grace cleanly separate the two.
    # Mirrors the running_grace idiom below.
    seen = running_pids | pending_pids | terminal_pids
    submitted_now: Set[str] = set()
    for row in db.get_incomplete():
        if row.get("nfy_status_str") != "submitted":
            continue
        pid = row.get("prompt_id")
        if not pid or pid in seen:
            continue
        submitted_now.add(pid)
        submitted_grace[pid] = submitted_grace.get(pid, 0) + 1
    for pid in [p for p, n in submitted_grace.items()
                if n >= _SUBMITTED_MISSING_GRACE]:
        submitted_grace.pop(pid, None)
        try:
            db.delete_submitted(pid)
        except Exception:
            _log.exception("Failed to reap submitted orphan (pid=%s)", pid)
    for pid in list(submitted_grace):
        if pid not in submitted_now:
            submitted_grace.pop(pid, None)

    # ----- pending -> running transition: mark the entry running so
    # the sidebar history view reflects the live state immediately.
    # Gated on the Nukomfy-pid check so foreign WebUI submissions
    # (which never recorded a submit row in our DB) are no-ops.
    for pid in running_pids - prev_running:
        if not _is_nukomfy_pid(running_map.get(pid) or {}):
            continue
        try:
            db.mark_running(pid)
        except Exception:
            _log.exception("Failed to mark job running (pid=%s)", pid)

    # ----- new terminal entries (success / error)
    for pid in terminal_pids - prev_terminal:
        # We trust the polling DB write only for pids we know are Nukomfy
        # (i.e. recorded by the submit hook OR present in the live snapshot
        # we just took). If the row was never recorded by the hook, the
        # update is a no-op - that's fine.
        try:
            from server import PromptServer  # type: ignore
            pq = PromptServer.instance.prompt_queue
        except Exception:
            continue
        _, payload = _history_entry_terminal(pq, pid)
        try:
            db.persist_terminal_state(pid, **payload)
        except Exception:
            _log.exception("Failed to persist terminal state (pid=%s)", pid)
        else:
            _resolve_range_from_count(pid, payload.get("outputs"))
        running_grace.pop(pid, None)

    # ----- pending pids that disappeared without ever running:
    # delete the submit-time row. ComfyUI's /api/jobs never tracked
    # queue-cancel events, so the persistent layer must not surface
    # them either - otherwise the client History sub-table would show
    # rows that never had a counterpart in the live runtime. The
    # `cancelled` status is reserved for the running->missing path
    # below (force_abort cleanup).
    for pid in prev_pending - pending_pids:
        if pid in running_pids or pid in terminal_pids:
            continue
        try:
            db.delete_entry(pid)
        except Exception:
            _log.exception("Failed to delete cancelled job from queue (pid=%s)", pid)

    # ----- running pids that disappeared without a history entry:
    # give them _RUNNING_MISSING_GRACE cycles before declaring cancelled.
    for pid in prev_running - running_pids:
        if pid in terminal_pids:
            running_grace.pop(pid, None)
            continue
        running_grace[pid] = running_grace.get(pid, 0) + 1

    # Clear grace for pids that came back to running (e.g. requeue).
    for pid in running_pids:
        running_grace.pop(pid, None)

    # Promote grace-expired pids to cancelled.
    ripe = [pid for pid, n in running_grace.items() if n >= _RUNNING_MISSING_GRACE]
    for pid in ripe:
        running_grace.pop(pid, None)
        if pid in terminal_pids:
            continue
        try:
            db.persist_terminal_state(
                pid, status_str="cancelled", duration=0.0,
                execution_error={"reason": "running_disappeared"})
        except Exception:
            _log.exception("Failed to persist vanished running job (pid=%s)", pid)

    return running_pids, pending_pids, terminal_pids


def _polling_loop() -> None:
    """Background diff loop. Quiet on idle, writes only on transitions."""
    prev_running: Set[str] = set()
    prev_pending: Set[str] = set()
    prev_terminal: Set[str] = set()
    # pid -> cycles missing since we last saw it in running, used to give
    # the history entry one extra cycle to materialise before we call it cancelled.
    running_grace: Dict[str, int] = {}
    # pid -> consecutive cycles a 'submitted' row has gone unobserved in the
    # live runtime; past the grace it is reaped (validation rejected it).
    submitted_grace: Dict[str, int] = {}

    while not _polling_stop.is_set():
        try:
            snap = _snapshot()
        except Exception:
            _log.exception("Failed to take queue snapshot")
            snap = None
        if snap is None:
            time.sleep(_POLL_INTERVAL_S)
            continue
        try:
            prev_running, prev_pending, prev_terminal = _poll_once(
                snap, prev_running, prev_pending, prev_terminal,
                running_grace, submitted_grace)
        except Exception:
            # One bad cycle must never kill the poll thread: it is the
            # real-time reaper the whole non-terminal-cleanup strategy
            # leans on (retention no longer evicts non-terminal rows), so
            # its death would let those rows accumulate until restart.
            # Keep the previous snapshot and retry on the next cycle.
            _log.exception("Job-history poll cycle failed (non-fatal)")
        time.sleep(_POLL_INTERVAL_S)


def _start_polling_thread() -> None:
    global _polling_thread
    if _polling_thread is not None and _polling_thread.is_alive():
        return
    _polling_stop.clear()
    _polling_thread = threading.Thread(
        target=_polling_loop,
        name="nukomfy-history-poll",
        daemon=True,
    )
    _polling_thread.start()


# ---------------------------------------------------------------------------
# Boot reconciliation
# ---------------------------------------------------------------------------

def boot_reconcile() -> int:
    """Reconcile DB entries still in-flight after a restart.

    Every non-terminal pid that the live runtime no longer reports is
    persisted as ``failed`` with execution_error ``{"reason": "server_restart"}``.
    Pending queue entries are included: a server crash that loses them is
    a real failure from the user's point of view, distinct from the polling
    loop's ``pending -> missing -> delete_entry`` path (which models an
    explicit queue-cancel where the user removed the row themselves).

    Returns the number of rows reconciled.
    """
    try:
        incomplete = db.get_incomplete()
    except Exception:
        _log.exception("Failed to load incomplete jobs during boot reconcile")
        return 0
    if not incomplete:
        return 0

    snap = _snapshot()
    if snap is None:
        return 0
    running, pending, terminal = snap
    known = set(running.keys()) | set(pending.keys()) | set(terminal.keys())

    reconciled = 0
    for entry in incomplete:
        pid = entry.get("prompt_id")
        if not pid or pid in known:
            continue
        try:
            db.persist_terminal_state(
                pid, status_str="failed", duration=0.0,
                execution_error={"reason": "server_restart"})
            reconciled += 1
        except Exception:
            _log.exception("Failed to reconcile job at boot (pid=%s)", pid)
    return reconciled


# ---------------------------------------------------------------------------
# Top-level start
# ---------------------------------------------------------------------------

def start() -> None:
    """Install hook + start polling thread + run boot reconciliation.

    Idempotent: safe to call multiple times. Reads the persisted config
    and short-circuits if the feature has been disabled.
    """
    global _started
    if _started:
        return

    cfg: Dict[str, Any] = {}
    try:
        from . import job_history_config
        cfg = job_history_config.load_config()
    except Exception:
        _log.exception("Failed to load history config; proceeding with defaults")

    if not cfg.get("enabled", True):
        _log.info("job history disabled by config")
        _started = True
        return

    hook_installed = _install_submit_hook()
    reconciled = boot_reconcile()
    _start_polling_thread()
    _started = True
    _log.info(
        "job history started (hook=%s, reconciled=%d)",
        hook_installed, reconciled)
