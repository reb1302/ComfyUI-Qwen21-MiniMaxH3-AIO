"""Server-side persistent job history (SQLite).

Structural mirror of the client-side job-history schema: same ``jobs``
table, columns, types and indices. The ``nfy_status_str`` CHECK set and
``_SCHEMA_VERSION`` intentionally differ - the server persists the full
job lifecycle (``submitted``/``pending``/``running``) because it polls
the live queue, while the client stores terminal states only.

The server-side DB exists only on machines where the Nukomfy Manager
custom node is installed. It captures the same fields the client
already persists, so the History sub-table of the Render Manager can
survive a ComfyUI restart and show entries for all users.

Path: ``<ComfyUI user>/default/nukomfy_manager/job_history.db``.

Concurrency: WAL mode, ``busy_timeout=5000``, single in-process RLock.
ComfyUI's prompt server is async/single-process; the background poll
thread in ``job_history.py`` may write concurrently with the HTTP read
endpoints.

NO migration code. If the existing DB on disk reports a different
``user_version``, write path is disabled (log + silent), but the custom
node keeps working. Operator deletes ``job_history.db`` (plus
``.db-wal``/``.db-shm``) to regenerate from the next submit onwards.
"""
from __future__ import annotations

import datetime
import gzip
import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from . import paths

_log = logging.getLogger(__name__)


_MAX_ENTRIES_HARD = 5000

_TERMINAL_STATUSES = ("completed", "failed", "cancelled")

_UPDATABLE_COLUMNS = {
    "nfy_frame_range",
    "nfy_output_paths",
    "nfy_input_ranges",
    "nfy_seeds_used",
    "nfy_read_color",
    "nfy_range_resolved",
    "nfy_sent_at",
}

_JSON_COL_MAP = {
    "nfy_frame_range":       "nfy_frame_range_json",
    "nfy_output_paths":      "nfy_output_paths_json",
    "nfy_input_ranges":      "nfy_input_ranges_json",
    "nfy_seeds_used":        "nfy_seeds_used_json",
    "nfy_execution_error":   "nfy_execution_error_json",
    "nfy_messages":          "nfy_messages_json",
    "nfy_preview_output":    "nfy_preview_output_json",
    "nfy_outputs":           "nfy_outputs_json",
    "nfy_workflow_meta":     "nfy_workflow_meta_json",
    "nfy_environment":       "nfy_environment_json",
    "nfy_server_snapshot":   "nfy_server_snapshot_json",
}

_SCHEMA_VERSION = 5

_GZIP_THRESHOLD = 20 * 1024

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    -- Identity / timing (native, queryable)
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id                TEXT    NOT NULL UNIQUE,
    nfy_job_id               TEXT    NOT NULL UNIQUE,
    nfy_submitted_by         TEXT    NOT NULL DEFAULT '',
    nfy_submitter_host       TEXT    NOT NULL DEFAULT '',
    nfy_machine_name         TEXT    NOT NULL,
    nfy_machine_url          TEXT    NOT NULL,
    nfy_workflow_name        TEXT    NOT NULL,
    nfy_nk_file              TEXT    NOT NULL DEFAULT '',
    nfy_node_name            TEXT    NOT NULL DEFAULT '',
    nfy_sent_at              TEXT    NOT NULL,
    nfy_batch_count          INTEGER NOT NULL DEFAULT 1,
    nfy_batch_index          INTEGER NOT NULL DEFAULT 1,
    nfy_read_color           INTEGER NOT NULL DEFAULT 0,

    -- Render config payload (lists/dicts as JSON)
    nfy_frame_range_json     TEXT    NOT NULL DEFAULT '[]',
    nfy_output_paths_json    TEXT    NOT NULL DEFAULT '[]',
    nfy_input_ranges_json    TEXT,
    nfy_seeds_used_json      TEXT,

    -- Diagnostic metadata (grouped JSON blobs, easier to extend)
    -- nfy_workflow_meta_json: workflow_uuid, workflow_author, workflow_version, params_spec
    -- nfy_environment_json:   os_submitter, nuke_version, nukomfy_version, input_cache
    -- nfy_server_snapshot_json: system_stats {os, comfyui_version, python_version, gpu, vram_total, ram_total}, server_version
    nfy_workflow_meta_json   TEXT,
    nfy_environment_json     TEXT,
    nfy_server_snapshot_json TEXT,

    -- Terminal state (set by persist_terminal_state)
    nfy_terminal_persisted   INTEGER NOT NULL DEFAULT 0
        CHECK (nfy_terminal_persisted IN (0, 1)),
    nfy_status_str           TEXT
        CHECK (nfy_status_str IS NULL OR nfy_status_str IN
               ('submitted', 'pending', 'running',
                'completed', 'failed', 'cancelled')),
    nfy_duration             REAL,
    nfy_outputs_count        INTEGER,
    nfy_execution_error_json TEXT,
    nfy_messages_json        TEXT,
    nfy_preview_output_json  TEXT,
    nfy_outputs_json         TEXT,

    -- Refresh-disk freeze flag
    nfy_range_resolved       INTEGER NOT NULL DEFAULT 0
        CHECK (nfy_range_resolved IN (0, 1)),

    -- Workflow API JSON snapshot of the payload posted to /prompt.
    -- Loaded on-demand via load_workflow_api(prompt_id). Excluded
    -- from SELECT * to keep history rebuilds light-weight.
    nfy_workflow_api            BLOB,
    nfy_workflow_api_compressed INTEGER NOT NULL DEFAULT 0
        CHECK (nfy_workflow_api_compressed IN (0, 1))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_prompt_id ON jobs(prompt_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_nfy_job_id ON jobs(nfy_job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_terminal_sent
    ON jobs(nfy_terminal_persisted, nfy_sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_sent_at ON jobs(nfy_sent_at DESC);
"""

_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None
_write_disabled = False


def db_path() -> str:
    return paths.history_db_path()


def _connect() -> Optional[sqlite3.Connection]:
    """Return the singleton sqlite3.Connection, or None if disabled.

    Fresh DB: creates the canonical schema and stamps user_version.
    Existing DB with mismatching user_version: logs an error, disables
    the write path, returns None.
    """
    global _conn, _write_disabled
    if _write_disabled:
        return None
    if _conn is not None:
        return _conn
    db_file = paths.history_db_path()
    try:
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        conn = sqlite3.connect(db_file, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current == 0:
            conn.executescript(_SCHEMA)
            conn.execute("PRAGMA user_version=%d" % _SCHEMA_VERSION)
            conn.commit()
        elif current != _SCHEMA_VERSION:
            conn.close()
            _write_disabled = True
            _log.error(
                "Job history DB at %s reports schema v%d but this build "
                "expects v%d. History write path disabled. Delete the file "
                "(plus .db-wal/.db-shm) and restart ComfyUI to regenerate.",
                db_file, current, _SCHEMA_VERSION,
            )
            return None
    except sqlite3.Error as e:
        _write_disabled = True
        _log.error("Failed to open job history DB: %s", e)
        return None
    _conn = conn
    return _conn


_ROW_COLUMNS = (
    "id", "prompt_id",
    "nfy_job_id", "nfy_submitted_by", "nfy_submitter_host",
    "nfy_machine_name", "nfy_machine_url",
    "nfy_workflow_name", "nfy_nk_file", "nfy_node_name",
    "nfy_sent_at",
    "nfy_batch_count", "nfy_batch_index", "nfy_read_color",
    "nfy_frame_range_json", "nfy_output_paths_json",
    "nfy_input_ranges_json", "nfy_seeds_used_json",
    "nfy_workflow_meta_json", "nfy_environment_json",
    "nfy_server_snapshot_json",
    "nfy_terminal_persisted", "nfy_status_str", "nfy_duration",
    "nfy_outputs_count",
    "nfy_execution_error_json", "nfy_messages_json",
    "nfy_preview_output_json", "nfy_outputs_json",
    "nfy_range_resolved",
)

_SELECT_COLS_SQL = ", ".join(_ROW_COLUMNS)


def _safe_json_loads(text: Optional[str], default: Any) -> Any:
    if text is None:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _row_to_dict(row: Optional[Tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    r = dict(zip(_ROW_COLUMNS, row))
    out: Dict[str, Any] = {
        "prompt_id": r["prompt_id"],
        "nfy_job_id": r["nfy_job_id"] or "",
        "nfy_submitted_by": r["nfy_submitted_by"] or "",
        "nfy_submitter_host": r["nfy_submitter_host"] or "",
        "nfy_machine_name": r["nfy_machine_name"],
        "nfy_machine_url": r["nfy_machine_url"],
        "nfy_workflow_name": r["nfy_workflow_name"],
        "nfy_sent_at": r["nfy_sent_at"],
        "nfy_frame_range": _safe_json_loads(r["nfy_frame_range_json"], []),
        "nfy_nk_file": r["nfy_nk_file"] or "",
        "nfy_node_name": r["nfy_node_name"] or "",
        "nfy_output_paths": _safe_json_loads(r["nfy_output_paths_json"], []),
        "nfy_batch_count": r["nfy_batch_count"],
        "nfy_batch_index": r["nfy_batch_index"],
        "nfy_read_color": r["nfy_read_color"] or 0,
    }
    if r["nfy_input_ranges_json"] is not None:
        out["nfy_input_ranges"] = _safe_json_loads(r["nfy_input_ranges_json"], [])
    if r["nfy_seeds_used_json"] is not None:
        out["nfy_seeds_used"] = _safe_json_loads(r["nfy_seeds_used_json"], [])
    if r.get("nfy_workflow_meta_json") is not None:
        meta = _safe_json_loads(r["nfy_workflow_meta_json"], None)
        if meta:
            out["nfy_workflow_meta"] = meta
    if r.get("nfy_environment_json") is not None:
        env = _safe_json_loads(r["nfy_environment_json"], None)
        if env:
            out["nfy_environment"] = env
    if r.get("nfy_server_snapshot_json") is not None:
        snap = _safe_json_loads(r["nfy_server_snapshot_json"], None)
        if snap:
            out["nfy_server_snapshot"] = snap
    # Always expose lifecycle status: the polling thread writes 'pending'
    # at submit time and 'running' on start, so non-terminal rows now
    # carry meaningful state and the sidebar (and client) need to see it.
    out["nfy_terminal_persisted"] = bool(r["nfy_terminal_persisted"])
    if r["nfy_status_str"] is not None:
        out["nfy_status_str"] = r["nfy_status_str"]
    if r["nfy_terminal_persisted"]:
        if r["nfy_duration"] is not None:
            out["nfy_duration"] = r["nfy_duration"]
        if r["nfy_execution_error_json"] is not None:
            out["nfy_execution_error"] = _safe_json_loads(
                r["nfy_execution_error_json"], None)
        if r["nfy_outputs_count"] is not None:
            out["nfy_outputs_count"] = r["nfy_outputs_count"]
        if r["nfy_messages_json"] is not None:
            out["nfy_messages"] = _safe_json_loads(r["nfy_messages_json"], [])
        if r["nfy_preview_output_json"] is not None:
            out["nfy_preview_output"] = _safe_json_loads(
                r["nfy_preview_output_json"], None)
        if r["nfy_outputs_json"] is not None:
            out["nfy_outputs"] = _safe_json_loads(r["nfy_outputs_json"], None)
    if r["nfy_range_resolved"]:
        out["nfy_range_resolved"] = True
    return out


def _build_workflow_meta(workflow_uuid, workflow_author, workflow_version,
                         params_spec=None, write_templates=None):
    m: Dict[str, Any] = {}
    if workflow_uuid:
        m["workflow_uuid"] = workflow_uuid
    if workflow_author:
        m["workflow_author"] = workflow_author
    if workflow_version:
        m["workflow_version"] = workflow_version
    if params_spec:
        m["params_spec"] = params_spec
    if write_templates:
        m["write_templates"] = write_templates
    return m or None


def _build_environment(os_submitter, nuke_version, nukomfy_version, input_cache):
    e: Dict[str, Any] = {}
    if os_submitter:
        e["os_submitter"] = os_submitter
    if nuke_version:
        e["nuke_version"] = nuke_version
    if nukomfy_version:
        e["nukomfy_version"] = nukomfy_version
    if input_cache:
        e["input_cache"] = input_cache
    return e or None


def _build_server_snapshot(system_stats, server_version):
    s: Dict[str, Any] = {}
    if system_stats:
        s["system_stats"] = system_stats
    if server_version:
        s["server_version"] = server_version
    return s or None


def _serialise_workflow_api(payload: Any) -> Tuple[Optional[bytes], int]:
    if payload is None:
        return (None, 0)
    try:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        _log.exception("Failed to serialise workflow API")
        return (None, 0)
    if len(raw) > _GZIP_THRESHOLD:
        return (gzip.compress(raw), 1)
    return (raw, 0)


def _retention_caps() -> Tuple[int, int]:
    """Return ``(max_days, max_jobs)`` from the retention config.

    Returns ``(0, 0)`` if the config module is unavailable or the values
    are missing/invalid - both dimensions interpret 0 as "disabled".
    """
    try:
        from . import job_history_config
        cfg = job_history_config.load_config()
        return (int(cfg.get("max_days_to_keep", 0) or 0),
                int(cfg.get("max_jobs_to_keep", 0) or 0))
    except Exception:
        return (0, 0)


def _trim_to_cap(conn: sqlite3.Connection) -> None:
    """Enforce hard cap + soft job cap. Time cap is applied by prune().

    Only terminal-persisted rows are ever evicted. A job still in flight
    (submitted/pending/running, persisted=0) that ages past the cap must
    survive so its terminal state can be frozen when it finishes -
    deleting it here would lose that job's outcome permanently.
    """
    _, max_jobs = _retention_caps()
    cap = max_jobs if max_jobs > 0 else _MAX_ENTRIES_HARD
    cap = min(max(cap, 1), _MAX_ENTRIES_HARD)
    conn.execute(
        "DELETE FROM jobs WHERE nfy_terminal_persisted=1 AND id NOT IN ("
        "SELECT id FROM jobs ORDER BY id DESC LIMIT ?)",
        (cap,))


# ---------------------------------------------------------------------------
# Public API - mirrors the client-side job-history surface where applicable
# ---------------------------------------------------------------------------

def record_submit(prompt_id, machine_name, machine_url, workflow_name,
                  frame_range, nk_file, node_name, output_paths, batch_count,
                  seeds_used=None, input_ranges=None,
                  batch_index=1, nfy_job_id=None,
                  nfy_submitted_by=None, nfy_submitter_host=None,
                  read_color=0, sent_at=None,
                  system_stats=None, server_version=None,
                  os_submitter=None, nuke_version=None, nukomfy_version=None,
                  input_cache=None,
                  workflow_author=None, workflow_uuid=None,
                  workflow_version=None, params_spec=None,
                  write_templates=None,
                  workflow_api_payload=None) -> None:
    """Append a new submit entry, mirroring the client-side schema.

    Same parameter list, same semantics. Best-effort: a DB failure does
    not propagate to the caller - the submit hook must never break
    ComfyUI execution.
    """
    workflow_meta = _build_workflow_meta(
        workflow_uuid, workflow_author, workflow_version, params_spec,
        write_templates)
    environment = _build_environment(
        os_submitter, nuke_version, nukomfy_version, input_cache)
    server_snapshot = _build_server_snapshot(system_stats, server_version)
    wf_blob, wf_flag = _serialise_workflow_api(workflow_api_payload)
    sql = (
        "INSERT INTO jobs ("
        "prompt_id, nfy_job_id, nfy_submitted_by, nfy_submitter_host, "
        "nfy_machine_name, nfy_machine_url, nfy_workflow_name, "
        "nfy_nk_file, nfy_node_name, nfy_sent_at, "
        "nfy_batch_count, nfy_batch_index, nfy_read_color, "
        "nfy_frame_range_json, nfy_output_paths_json, "
        "nfy_input_ranges_json, nfy_seeds_used_json, "
        "nfy_workflow_meta_json, nfy_environment_json, "
        "nfy_server_snapshot_json, "
        "nfy_workflow_api, nfy_workflow_api_compressed, "
        "nfy_status_str"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?)"
    )
    params = (
        prompt_id,
        nfy_job_id or "",
        nfy_submitted_by or "",
        nfy_submitter_host or "",
        machine_name,
        machine_url,
        workflow_name,
        nk_file or "",
        node_name or "",
        sent_at or datetime.datetime.now().isoformat(timespec="microseconds"),
        int(batch_count or 1),
        int(batch_index or 1),
        int(read_color or 0),
        json.dumps(frame_range or []),
        json.dumps(output_paths or []),
        json.dumps(input_ranges) if input_ranges else None,
        json.dumps(seeds_used) if seeds_used else None,
        json.dumps(workflow_meta) if workflow_meta else None,
        json.dumps(environment) if environment else None,
        json.dumps(server_snapshot) if server_snapshot else None,
        wf_blob,
        wf_flag,
        # Provisional: the POST was accepted but not yet validated/queued. The
        # polling loop promotes this to 'pending' once it observes the prompt
        # live, or reaps the row if validation rejected the prompt.
        "submitted",
    )
    with _lock:
        conn = _connect()
        if conn is None:
            return
        try:
            with conn:
                conn.execute(sql, params)
                _trim_to_cap(conn)
        except sqlite3.Error:
            _log.exception("Failed to record submit (prompt_id=%s)", prompt_id)


def save_workflow_api(prompt_id: str, payload: Any) -> None:
    if not prompt_id:
        return
    blob, flag = _serialise_workflow_api(payload)
    with _lock:
        conn = _connect()
        if conn is None:
            return
        try:
            with conn:
                conn.execute(
                    "UPDATE jobs SET nfy_workflow_api=?, "
                    "nfy_workflow_api_compressed=? WHERE prompt_id=?",
                    (blob, flag, prompt_id))
        except sqlite3.Error:
            _log.exception("Failed to save workflow API (prompt_id=%s)", prompt_id)


def load_workflow_api(prompt_id: str) -> Any:
    if not prompt_id:
        return None
    with _lock:
        conn = _connect()
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT nfy_workflow_api, nfy_workflow_api_compressed "
                "FROM jobs WHERE prompt_id=?",
                (prompt_id,)).fetchone()
        except sqlite3.Error:
            _log.exception("Failed to load workflow API (prompt_id=%s)", prompt_id)
            return None
    if not row or row[0] is None:
        return None
    blob, flag = row[0], row[1]
    try:
        raw = gzip.decompress(blob) if flag else blob
        return json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        _log.exception("Failed to decode workflow API (prompt_id=%s)", prompt_id)
        return None


# Columns the free-text history search scans (case-insensitive substring).
# The human-meaningful identity fields the plugin/WebUI search box targets;
# machine name is omitted because the viewer is already per-machine.
_HISTORY_SEARCH_COLS = (
    "nfy_workflow_name", "nfy_submitted_by", "nfy_node_name",
    "nfy_nk_file", "nfy_job_id", "prompt_id",
)

# Statuses accepted by the status filter (mirror of the schema CHECK).
_HISTORY_FILTER_STATUSES = frozenset(
    ("pending", "running", "completed", "failed", "cancelled"))


def _history_where(only_terminal: bool,
                   since_iso: Optional[str],
                   statuses: Optional[List[str]],
                   query: Optional[str]) -> Tuple[List[str], List[Any]]:
    """Build the shared WHERE for get_history / count_history.

    One place so the row query and the COUNT(*) pager stay in lockstep -
    a divergence would make the page count disagree with the rows shown.
    """
    where: List[str] = []
    params: List[Any] = []
    # 'submitted' is a provisional pre-queue state (POST accepted but not yet
    # observed live); it must never surface in any user-facing list/count -
    # the polling loop promotes it to 'pending' or reaps it.
    where.append("(nfy_status_str IS NULL OR nfy_status_str != 'submitted')")
    if only_terminal:
        where.append("nfy_terminal_persisted=1")
    if since_iso:
        where.append("nfy_sent_at >= ?")
        params.append(since_iso)
    if statuses:
        valid = [s for s in statuses if s in _HISTORY_FILTER_STATUSES]
        if valid:
            where.append(
                "nfy_status_str IN (%s)" % ",".join("?" * len(valid)))
            params.extend(valid)
    if query:
        # Escape LIKE wildcards so a literal _ or % (common in nk file
        # paths) matches itself instead of acting as a pattern.
        esc = (query.replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_"))
        like = "%" + esc + "%"
        clause = " OR ".join(
            col + " LIKE ? ESCAPE '\\'" for col in _HISTORY_SEARCH_COLS)
        where.append("(" + clause + ")")
        params.extend([like] * len(_HISTORY_SEARCH_COLS))
    return where, params


def get_history(limit: Optional[int] = None,
                offset: int = 0,
                only_terminal: bool = False,
                since_iso: Optional[str] = None,
                statuses: Optional[List[str]] = None,
                query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return history entries, most recent first.

    ``since_iso`` filters entries with ``nfy_sent_at >= since_iso`` (lexicographic
    on ISO-8601 strings is correct ordering). ``statuses`` keeps only the listed
    ``nfy_status_str`` values; ``query`` is a case-insensitive substring matched
    across the identity columns. Both no-op when falsy, so existing callers get
    the unchanged query.
    """
    where, params = _history_where(only_terminal, since_iso, statuses, query)
    sql = "SELECT " + _SELECT_COLS_SQL + " FROM jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    # Primary: state rank - running on top, then pending, then terminals.
    # Secondary: client submit time (nfy_sent_at, the value the Sent column
    # shows), ascending for active states (oldest pending - next to run -
    # under the running one) and descending for terminals (most-recent
    # first), so every view agrees on ordering. Tiebreak: id, keeping the
    # ascending/descending split via negation for same-instant batches.
    sql += (
        " ORDER BY "
        "CASE nfy_status_str "
        "WHEN 'running' THEN 0 "
        "WHEN 'pending' THEN 1 "
        "ELSE 2 END, "
        "CASE WHEN nfy_status_str IN ('running', 'pending') "
        "THEN nfy_sent_at END ASC, "
        "CASE WHEN nfy_status_str NOT IN ('running', 'pending') "
        "THEN nfy_sent_at END DESC, "
        "CASE "
        "WHEN nfy_status_str IN ('running', 'pending') THEN id "
        "ELSE -id "
        "END ASC"
    )
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
    elif offset:
        sql += " LIMIT -1 OFFSET ?"
        params.append(int(offset))
    with _lock:
        conn = _connect()
        if conn is None:
            return []
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            _log.exception("Failed to get history")
            return []
    return [_row_to_dict(r) for r in rows if r is not None]


def count_history(only_terminal: bool = False,
                  since_iso: Optional[str] = None,
                  statuses: Optional[List[str]] = None,
                  query: Optional[str] = None) -> int:
    """Total rows matching the same WHERE clauses as get_history.

    Used by the WebUI pager to compute total page count. SQL COUNT(*)
    only - no row materialization, cost is independent of table size.
    """
    where, params = _history_where(only_terminal, since_iso, statuses, query)
    sql = "SELECT COUNT(*) FROM jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    with _lock:
        conn = _connect()
        if conn is None:
            return 0
        try:
            row = conn.execute(sql, params).fetchone()
        except sqlite3.Error:
            _log.exception("Failed to count history")
            return 0
    if not row:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def get_by_prompt_id(prompt_id: str) -> Optional[Dict[str, Any]]:
    if not prompt_id:
        return None
    with _lock:
        conn = _connect()
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT " + _SELECT_COLS_SQL + " FROM jobs WHERE prompt_id=?",
                (prompt_id,)).fetchone()
        except sqlite3.Error:
            _log.exception("Failed to get job (prompt_id=%s)", prompt_id)
            return None
    return _row_to_dict(row)


def get_incomplete() -> List[Dict[str, Any]]:
    """Non-terminal entries flat list. Used by boot reconciliation and sweep."""
    with _lock:
        conn = _connect()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                "SELECT " + _SELECT_COLS_SQL
                + " FROM jobs WHERE nfy_terminal_persisted=0 "
                "ORDER BY nfy_sent_at DESC, id DESC").fetchall()
        except sqlite3.Error:
            _log.exception("Failed to get incomplete jobs")
            return []
    return [_row_to_dict(r) for r in rows if r is not None]


def update_entry(prompt_id: str, **fields: Any) -> None:
    """Update whitelisted fields of an existing entry by prompt_id."""
    if not fields:
        return
    bad = set(fields.keys()) - _UPDATABLE_COLUMNS
    if bad:
        raise ValueError(
            "update_entry: unknown/forbidden fields %s. Allowed: %s" %
            (sorted(bad), sorted(_UPDATABLE_COLUMNS)))
    sets: List[str] = []
    params: List[Any] = []
    for k, v in fields.items():
        if k in _JSON_COL_MAP:
            sets.append("%s=?" % _JSON_COL_MAP[k])
            params.append(json.dumps(v) if v is not None else None)
        elif k == "nfy_range_resolved":
            sets.append("nfy_range_resolved=?")
            params.append(1 if v else 0)
        elif k == "nfy_read_color":
            sets.append("nfy_read_color=?")
            params.append(int(v or 0))
        else:
            sets.append("%s=?" % k)
            params.append(v)
    params.append(prompt_id)
    sql = "UPDATE jobs SET " + ", ".join(sets) + " WHERE prompt_id=?"
    with _lock:
        conn = _connect()
        if conn is None:
            return
        try:
            with conn:
                conn.execute(sql, params)
        except sqlite3.Error:
            _log.exception("Failed to update entry (prompt_id=%s)", prompt_id)


def mark_running(prompt_id: str) -> None:
    """Update an in-flight entry's status to ``running``.

    Called by the polling thread on the pending->running transition so
    the sidebar history view shows the right pill in real time. No-op
    when the row is already terminal-persisted (defensive against a
    late running event for a job that already completed).
    """
    if not prompt_id:
        return
    with _lock:
        conn = _connect()
        if conn is None:
            return
        try:
            with conn:
                conn.execute(
                    "UPDATE jobs SET nfy_status_str=? "
                    "WHERE prompt_id=? AND nfy_terminal_persisted=0",
                    ("running", prompt_id))
        except sqlite3.Error:
            _log.exception("Failed to mark job running (prompt_id=%s)", prompt_id)


def mark_queued(prompt_id: str) -> None:
    """Promote a pre-queue ``submitted`` entry to ``pending``.

    Called by the polling thread the first time a prompt is observed live in
    the runtime, i.e. it passed validation and was enqueued. No-op unless the
    row is still ``submitted`` - a prompt rejected before enqueue is never
    seen live, so it stays ``submitted`` and is reaped instead.
    """
    if not prompt_id:
        return
    with _lock:
        conn = _connect()
        if conn is None:
            return
        try:
            with conn:
                conn.execute(
                    "UPDATE jobs SET nfy_status_str='pending' "
                    "WHERE prompt_id=? AND nfy_status_str='submitted'",
                    (prompt_id,))
        except sqlite3.Error:
            _log.exception("Failed to mark job queued (prompt_id=%s)", prompt_id)


def delete_entry(prompt_id: str) -> None:
    """Remove a single entry by prompt_id."""
    if not prompt_id:
        return
    with _lock:
        conn = _connect()
        if conn is None:
            return
        try:
            with conn:
                conn.execute("DELETE FROM jobs WHERE prompt_id=?", (prompt_id,))
        except sqlite3.Error:
            _log.exception("Failed to delete entry (prompt_id=%s)", prompt_id)


def delete_submitted(prompt_id: str) -> None:
    """Reap a ``submitted`` entry that never entered the runtime.

    A prompt rejected by ``validate_prompt`` (missing model, bad value, ...)
    is recorded by the submit hook but never enqueued, so the polling loop
    never observes it live. The status guard makes this incapable of touching
    a row that ever advanced to ``pending``/``running``/terminal.
    """
    if not prompt_id:
        return
    with _lock:
        conn = _connect()
        if conn is None:
            return
        try:
            with conn:
                conn.execute(
                    "DELETE FROM jobs WHERE prompt_id=? "
                    "AND nfy_status_str='submitted' AND nfy_terminal_persisted=0",
                    (prompt_id,))
        except sqlite3.Error:
            _log.exception("Failed to reap submitted orphan (prompt_id=%s)", prompt_id)


def is_terminal_persisted(prompt_id: str) -> bool:
    if not prompt_id:
        return False
    with _lock:
        conn = _connect()
        if conn is None:
            return False
        try:
            row = conn.execute(
                "SELECT nfy_terminal_persisted FROM jobs WHERE prompt_id=?",
                (prompt_id,)).fetchone()
        except sqlite3.Error:
            _log.exception("Failed to check terminal-persisted state (prompt_id=%s)", prompt_id)
            return False
    return bool(row[0]) if row else False


def persist_terminal_state(prompt_id, status_str, duration,
                           execution_error=None, outputs_count=None,
                           messages=None, preview_output=None,
                           outputs=None) -> None:
    """Freeze a history entry with terminal server data. Idempotent."""
    if not prompt_id or status_str not in _TERMINAL_STATUSES:
        return
    with _lock:
        conn = _connect()
        if conn is None:
            return
        try:
            with conn:
                row = conn.execute(
                    "SELECT nfy_terminal_persisted FROM jobs WHERE prompt_id=?",
                    (prompt_id,)).fetchone()
                if not row or row[0]:
                    return
                sets = ["nfy_status_str=?", "nfy_duration=?",
                        "nfy_terminal_persisted=1"]
                params: List[Any] = [status_str, duration]
                if execution_error is not None:
                    sets.append("nfy_execution_error_json=?")
                    params.append(json.dumps(execution_error))
                if outputs_count is not None:
                    sets.append("nfy_outputs_count=?")
                    params.append(outputs_count)
                if messages is not None:
                    sets.append("nfy_messages_json=?")
                    params.append(json.dumps(messages))
                if preview_output is not None:
                    sets.append("nfy_preview_output_json=?")
                    params.append(json.dumps(preview_output))
                if outputs is not None:
                    sets.append("nfy_outputs_json=?")
                    params.append(json.dumps(outputs))
                params.append(prompt_id)
                conn.execute(
                    "UPDATE jobs SET " + ", ".join(sets)
                    + " WHERE prompt_id=?", params)
        except sqlite3.Error:
            _log.exception("Failed to persist terminal state (prompt_id=%s)", prompt_id)


def clear_history() -> None:
    with _lock:
        conn = _connect()
        if conn is None:
            return
        try:
            with conn:
                conn.execute("DELETE FROM jobs")
        except sqlite3.Error:
            _log.exception("Failed to clear history")


def clear_terminal() -> None:
    with _lock:
        conn = _connect()
        if conn is None:
            return
        try:
            with conn:
                conn.execute("DELETE FROM jobs WHERE nfy_terminal_persisted=1")
        except sqlite3.Error:
            _log.exception("Failed to clear terminal state")


def sweep_aged_out(max_days: int) -> int:
    """Drop non-terminal entries older than ``max_days`` from the DB.

    Server knows the state of every job at all times, so an entry stuck
    non-terminal past the threshold can only be a casualty of a missed
    transition (e.g. polling thread stopped). Delete the row outright
    rather than freezing it as a synthetic ``unknown`` - the Nukomfy
    client owns lost-detection semantics via its own client-side timeout.

    ``max_days <= 0`` disables the sweep (no-op, returns 0). Returns the
    number of rows deleted.
    """
    try:
        days = int(max_days)
    except (TypeError, ValueError):
        return 0
    if days <= 0:
        return 0
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat(
        timespec="microseconds")
    deleted = 0
    # Iterate get_incomplete (all non-terminal rows) rather than get_history,
    # which hides the provisional 'submitted' state - a stuck 'submitted' row
    # must still be sweepable if the polling reaper never ran.
    for e in get_incomplete():
        if e.get("nfy_terminal_persisted"):
            continue
        pid = e.get("prompt_id")
        if not pid:
            continue
        sent_raw = e.get("nfy_sent_at", "")
        if not sent_raw or sent_raw >= cutoff:
            continue
        try:
            delete_entry(pid)
            deleted += 1
        except Exception:
            _log.exception("Failed to delete aged-out entry (pid=%s)", pid)
    return deleted


def prune(max_days: int = 0, max_jobs: int = 0) -> int:
    """Apply retention. Returns number of rows deleted.

    ``max_days``: drop entries with ``nfy_sent_at`` older than the cutoff.
    ``max_jobs``: keep only the N most recent entries (hard cap still applies).
    ``0`` disables either dimension.

    Only terminal-persisted rows are pruned. A non-terminal row (a job
    still in flight) is never deleted here - dropping it would lose its
    terminal state when it later completes. Non-terminal rows are retired
    by the lifecycle reaper in job_history.py (rejected submits reaped,
    cancel on queue-vanish, terminal freeze on finish), not by retention.
    """
    deleted = 0
    with _lock:
        conn = _connect()
        if conn is None:
            return 0
        try:
            with conn:
                if max_days and max_days > 0:
                    cutoff = (
                        datetime.datetime.now()
                        - datetime.timedelta(days=int(max_days))
                    ).isoformat(timespec="microseconds")
                    cur = conn.execute(
                        "DELETE FROM jobs WHERE nfy_terminal_persisted=1 "
                        "AND nfy_sent_at < ?", (cutoff,))
                    deleted += cur.rowcount or 0
                jobs_cap = max_jobs if max_jobs > 0 else _MAX_ENTRIES_HARD
                jobs_cap = min(max(jobs_cap, 1), _MAX_ENTRIES_HARD)
                cur = conn.execute(
                    "DELETE FROM jobs WHERE nfy_terminal_persisted=1 AND id NOT IN ("
                    "SELECT id FROM jobs ORDER BY id DESC LIMIT ?)",
                    (jobs_cap,))
                deleted += cur.rowcount or 0
            # Truncate the WAL sidecar on each retention pass so the
            # .db-wal file does not linger after a hard ComfyUI kill.
            # Auto-checkpoint self-bounds it during normal writes; this
            # just keeps the on-disk footprint tidy. Best-effort.
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
        except sqlite3.Error:
            _log.exception("Failed to prune history")
    return deleted
