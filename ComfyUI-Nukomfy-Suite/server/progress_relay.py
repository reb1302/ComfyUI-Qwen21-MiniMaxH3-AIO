"""Live progress broadcast relay.

ComfyUI sends per-node progress (`progress_state`) only to the client that
submitted the job: the WebUIProgressHandler targets `server.client_id` (see
comfy_execution/progress.py). That isolation means a second viewer - another
workstation, or a second app instance - never sees a running job's progress.

This relay lifts that limit for Nukomfy clients. It wraps the server's
`send_sync` once at startup, observes the native event stream, computes one
weighted completion fraction for the running job, and re-emits it as a
`nukomfy.progress` broadcast (sid=None) that reaches every connected client.
Job lifecycle transitions are mirrored the same way as `nukomfy.lifecycle`.

The weighting matches the Nukomfy client bar: granular nodes (KSampler family,
detected by an `inputs.steps` int) carry their step count as weight; cached
nodes count as done; the emitted fraction is ratcheted so it never regresses.
Computing it server-side keeps every client a thin display with no per-job
state of its own.

Best-effort throughout: a failure in the relay must never disturb the native
event the host is delivering.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Set

_log = logging.getLogger(__name__)

# Input names whose positive-int value marks a granular node that drives one
# ProgressBar spanning N internal steps. The KSampler family carries `steps` on
# the sampler node itself. The decomposed SamplerCustom family does not - its
# step count lives on an upstream scheduler (a SIGMAS producer) - and is
# recovered separately by `_reattribute_sigmas_weights`.
_GRANULAR_STEP_INPUTS = ("steps",)

# Sampler classes that drive a step ProgressBar. A running node of one of these
# is promoted to its observed step weight even when no static `steps` could be
# found (e.g. the count is wired through a switch/int node, so it is unreadable
# from the graph), without promoting arbitrary custom nodes that merely report a
# large max.
_PROGRESS_SAMPLER_TYPES = frozenset((
    "KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced",
))

# Broadcast pacing. Sends are coalesced: emit only when the fraction moved at
# least _MIN_DELTA, or once _MIN_INTERVAL_S elapsed since the last send (the
# first update of a job always sends). This never changes a computed value -
# the ratchet updates on every event; a skipped send just means the next one
# carries the current value - it only caps how many websocket messages fan out
# to every connected client, which matters as the viewer count grows. The
# interval also bounds staleness: the bar is never more than _MIN_INTERVAL_S
# behind while events keep arriving. Lifecycle events are never coalesced.
_MIN_DELTA = 0.005
_MIN_INTERVAL_S = 0.25

_PROGRESS_EVENT = "nukomfy.progress"
_LIFECYCLE_EVENT = "nukomfy.lifecycle"

_installed = False
_original_send_sync = None  # bound PromptServer.send_sync captured at install
_relay_error_logged = False
_lock = threading.Lock()

# prompt_id -> per-job computation state. One job runs at a time per process,
# so this holds a single live entry; stale entries are pruned when a new job
# appears (guards against a missed terminal event).
_state: Dict[str, Dict[str, Any]] = {}
# prompt_id -> set(cached node ids), captured from `execution_cached`.
_cached: Dict[str, Set[str]] = {}


def _is_link(value: Any) -> bool:
    """True if `value` is a node connection `[node_id, slot]`, not a literal.

    Connections serialize as a 2-list whose head is the source node id (a string
    key into the prompt). Literal widget lists (e.g. a float sigma list) have a
    non-string head, so they are excluded.
    """
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[0], str))


def _count_sigma_values(text: str) -> int:
    """Count comma-separated sigma boundaries in a ManualSigmas literal."""
    return sum(1 for tok in text.split(",") if tok.strip())


def _sigmas_step_count(prompt: Dict[str, Any], nid: str, _depth: int = 0):
    """Resolve how many steps the sigmas reaching node `nid` represent.

    Walks back from a SIGMAS-producing node to the source that defines the step
    count: a scheduler exposing an int `steps`, or a ManualSigmas literal whose
    value list has N boundaries (N-1 steps). Follows sigma-op chains (SplitSigmas,
    SetFirstSigma, ...) through their own `sigmas` input; `_depth` bounds cycles.
    Returns `(steps, scheduler_id)`, where `scheduler_id` is the node holding the
    static `steps` weight to drop (None for a ManualSigmas literal).
    """
    if _depth > 8:
        return (None, None)
    entry = prompt.get(nid) if isinstance(prompt, dict) else None
    if not isinstance(entry, dict):
        return (None, None)
    inputs = entry.get("inputs") or {}
    v = inputs.get("steps")
    if isinstance(v, int) and not isinstance(v, bool) and v > 1:
        return (v, nid)
    sig = inputs.get("sigmas")
    if isinstance(sig, str):
        n = _count_sigma_values(sig)
        if n >= 2:
            return (n - 1, None)
    if _is_link(sig):
        return _sigmas_step_count(prompt, sig[0], _depth + 1)
    return (None, None)


def _reattribute_sigmas_weights(prompt: Dict[str, Any], weights: Dict[str, int]) -> None:
    """Move a scheduler's step weight onto the SamplerCustom* that consumes it.

    `_extract_weights` weights whatever node declares `steps`. For the KSampler
    family that node runs the progress bar (correct). For the decomposed family
    the declaring node is an instant scheduler, while the N-step bar runs on the
    downstream SamplerCustom* - which would otherwise carry weight 1 and let the
    bar race ahead. Here each SamplerCustom* (identified by taking both a SIGMAS
    link and a `latent_image`, which excludes the sigma-op nodes) inherits the
    step count traced back through its `sigmas` input, and the scheduler's phantom
    weight is dropped. The KSampler family has no `sigmas` input, so it is never
    touched and its weighting stays identical.
    """
    drop: Set[str] = set()
    for nid, entry in (prompt or {}).items():
        if not isinstance(entry, dict):
            continue
        inputs = entry.get("inputs") or {}
        if "latent_image" not in inputs:
            continue
        link = inputs.get("sigmas")
        if not _is_link(link):
            continue
        steps, sched = _sigmas_step_count(prompt, link[0])
        if not steps:
            continue
        weights[nid] = max(weights.get(nid, 0), steps)
        if sched is not None:
            drop.add(sched)
    for s in drop:
        weights.pop(s, None)


def _extract_weights(prompt: Dict[str, Any]) -> Dict[str, int]:
    """Return {node_id: step_count} for nodes that drive a granular step bar.

    Nodes declaring an int `steps` are weighted directly (the KSampler family);
    the decomposed SamplerCustom family is then corrected by
    `_reattribute_sigmas_weights` (see there).
    """
    out: Dict[str, int] = {}
    for nid, entry in (prompt or {}).items():
        if not isinstance(entry, dict):
            continue
        inputs = entry.get("inputs") or {}
        for key in _GRANULAR_STEP_INPUTS:
            v = inputs.get(key)
            if isinstance(v, bool):
                continue
            if isinstance(v, int) and v > 1:
                out[nid] = v
                break
    _reattribute_sigmas_weights(prompt, out)
    return out


def _current_prompt() -> Dict[str, Any]:
    """Return the running job's API prompt dict from the progress registry."""
    try:
        from comfy_execution.progress import get_progress_state
        dp = getattr(get_progress_state(), "dynprompt", None)
        prompt = getattr(dp, "original_prompt", None)
        return prompt if isinstance(prompt, dict) else {}
    except Exception:
        return {}


def _ensure_state(prompt_id: str) -> Dict[str, Any]:
    """Return (creating if needed) the per-job state for prompt_id.

    A new prompt_id means a new job: only one runs at a time, so any leftover
    entries are stale (their terminal event never arrived) and are dropped.
    Prompt-derived fields (weights/types/total) are filled later by
    `_load_prompt`: `execution_start` fires just before the core resets the
    progress registry, so reading the graph here would capture the previous
    (or empty) prompt.
    """
    st = _state.get(prompt_id)
    if st is not None:
        return st
    _state.clear()
    for k in [k for k in _cached if k != prompt_id]:
        _cached.pop(k, None)
    st = {
        "weights": {},
        "types": {},
        "total": 0,
        "prompt_loaded": False,
        "finished": set(),   # non-cached node ids observed finished
        "last_frac": 0.0,    # ratchet floor (computed; updated every event)
        "last_tooltip": "",  # most recent tooltip (updated every event, pre-throttle)
        "sent_frac": -1.0,   # last broadcast fraction (-1 forces first send)
        "sent_t": 0.0,       # last broadcast time (monotonic)
    }
    _state[prompt_id] = st
    return st


def _load_prompt(st: Dict[str, Any]) -> None:
    """Fill weights/types/total from the running job's graph. No-op until the
    registry holds the prompt (retried on each progress event until it does)."""
    prompt = _current_prompt()
    if not prompt:
        return
    st["weights"] = _extract_weights(prompt)
    st["types"] = {nid: (e.get("class_type", "") if isinstance(e, dict) else "")
                   for nid, e in prompt.items()}
    st["total"] = len(prompt)
    st["prompt_loaded"] = True


def _should_send(st: Dict[str, Any], frac: float) -> bool:
    """Coalesce progress sends. True on the first update, on a >= _MIN_DELTA
    move, or once _MIN_INTERVAL_S has elapsed. Gates the broadcast cadence
    only, never the computed value."""
    now = time.monotonic()
    if (st["sent_frac"] < 0
            or frac - st["sent_frac"] >= _MIN_DELTA
            or now - st["sent_t"] >= _MIN_INTERVAL_S):
        st["sent_frac"] = frac
        st["sent_t"] = now
        return True
    return False


def _broadcast(event: str, data: Dict[str, Any]) -> None:
    if _original_send_sync is not None:
        _original_send_sync(event, data, None)


def _pick_active(running, weights):
    """Index in `running` of the genuinely-active node.

    ComfyUI keeps a lazy/subgraph wrapper (e.g. a downstream Switch awaiting
    its branch) `running` for the whole time its input computes, next to the
    real worker; a naive `running[0]`/`running[-1]` would pick the idle
    wrapper. Rank by (static weight, observed max) so the stepping sampler
    (weight N, or max>1) beats the idle weight-1 wrapper; ties break to the
    last-started node (the deepest active one).
    """
    return max(
        range(len(running)),
        key=lambda i: (weights.get(running[i][0], 1),
                       float(running[i][1].get("max") or 1), i),
    )


def _relay_progress(data: Dict[str, Any]) -> None:
    prompt_id = data.get("prompt_id")
    nodes = data.get("nodes") or {}
    if not prompt_id or not nodes:
        return
    running = [(nid, n) for nid, n in nodes.items()
               if isinstance(n, dict) and n.get("state") == "running"]
    if not running:
        return  # transient between-nodes state; hold the last value
    with _lock:
        st = _ensure_state(prompt_id)
        if not st["prompt_loaded"]:
            _load_prompt(st)
        cached_set = _cached.get(prompt_id) or set()
        weights = st["weights"]
        finished = st["finished"]
        total = st["total"]

        # The active node drives the fraction and the tooltip below.
        running_nid, rn = running[_pick_active(running, weights)]
        val_v = float(rn.get("value") or 0)
        max_v = float(rn.get("max") or 1)
        running_frac = val_v / max_v if max_v else 0.0
        running_frac = max(0.0, min(1.0, running_frac))

        # A running node reporting max>1 is granular; grow its weight to the
        # observed max (never shrink). For statically recognized nodes this just
        # refines the estimate. For the known sampler classes we also promote a
        # node carrying no static weight - it covers samplers whose step count is
        # wired through a switch/int node and is unreadable from the graph. Any
        # other unrecognized node is left at weight 1 so a custom node that
        # reports a large max without being the dominant work cannot stall the
        # bar. The ratchet below keeps the bar monotonic when total_weight grows.
        if (max_v > 1 and int(max_v) > weights.get(running_nid, 1)
                and (running_nid in weights
                     or st["types"].get(running_nid) in _PROGRESS_SAMPLER_TYPES)):
            weights[running_nid] = int(max_v)

        for obs_nid, obs_n in nodes.items():
            if obs_nid in cached_set:
                continue
            if isinstance(obs_n, dict) and obs_n.get("state") == "finished":
                finished.add(obs_nid)

        n_cached = len(cached_set)
        known_weight_sum = sum(weights.values())
        unknown_nc = max(0, total - n_cached - len(weights))
        total_weight = max(1, n_cached + known_weight_sum + unknown_nc)

        finished_excl = finished - {running_nid}
        done_weight = n_cached + sum(weights.get(nid, 1) for nid in finished_excl)
        running_weight = weights.get(running_nid, 1)
        new_frac = (done_weight + running_frac * running_weight) / float(total_weight)
        new_frac = max(0.0, min(1.0, new_frac))
        emit_frac = max(new_frac, st["last_frac"])
        st["last_frac"] = emit_frac

        node_total = max(1, total)
        node_done = n_cached + len(finished_excl)
        idx = max(1, min(node_done + 1, node_total))
        # When the picked node is a parked wrapper (no granular work of its
        # own, and overtaken by a later-started node that already finished,
        # e.g. a lazy Switch awaiting its branch), label the tooltip with the
        # most-recently-reached finished node instead of the idle wrapper. A
        # genuine lone worker has nothing finished after it, so it is never
        # relabeled; an unresolved (runtime-expanded) finished node is
        # skipped, so the wrapper name stands.
        display_nid = running_nid
        if max_v <= 1 and weights.get(running_nid, 1) <= 1:
            order = {nid: i for i, nid in enumerate(nodes)}
            here = order.get(running_nid, -1)
            done_after = [f for f in finished
                          if order.get(f, -1) > here and f in st["types"]]
            if done_after:
                display_nid = max(done_after, key=lambda f: order[f])
        class_type = st["types"].get(display_nid, display_nid)
        if max_v > 1:
            tooltip = "{} - step {}/{} ({}/{})".format(
                class_type, int(val_v), int(max_v), idx, node_total)
        else:
            tooltip = "{} ({}/{})".format(class_type, idx, node_total)
        st["last_tooltip"] = tooltip

        send = _should_send(st, emit_frac)

    if send:
        _broadcast(_PROGRESS_EVENT, {
            "prompt_id": prompt_id,
            "fraction": round(emit_frac, 5),
            "tooltip": tooltip,
            "state": "running",
        })


def _relay_terminal(prompt_id: Optional[str], event: str) -> None:
    if prompt_id:
        _broadcast(_LIFECYCLE_EVENT, {"prompt_id": prompt_id, "event": event})
    with _lock:
        if prompt_id:
            _state.pop(prompt_id, None)
            _cached.pop(prompt_id, None)


def current_progress() -> Dict[str, Dict[str, Any]]:
    """Snapshot {prompt_id: {fraction, tooltip}} for every tracked job.

    Read under `_lock` so a concurrent relay update can't yield a torn
    dict. `last_frac` is ratcheted on every event (independent of the
    send throttle), so the snapshot is always current. Backs the
    read-only GET /nukomfy/progress, letting a client that missed the WS
    stream (restart, second viewer, queue rebuild) reseed its bars.
    """
    with _lock:
        return {
            pid: {
                "fraction": round(st.get("last_frac", 0.0), 5),
                "tooltip": st.get("last_tooltip", ""),
            }
            for pid, st in _state.items()
        }


def _maybe_relay(event: str, data: Any) -> None:
    if not isinstance(data, dict):
        return
    if event == "progress_state":
        _relay_progress(data)
    elif event == "execution_cached":
        pid = data.get("prompt_id")
        if pid:
            with _lock:
                _cached[pid] = set(data.get("nodes") or [])
    elif event == "execution_start":
        pid = data.get("prompt_id")
        if pid:
            with _lock:
                _ensure_state(pid)
            _broadcast(_LIFECYCLE_EVENT, {"prompt_id": pid, "event": "start"})
    elif event == "execution_success":
        _relay_terminal(data.get("prompt_id"), "success")
    elif event == "execution_error":
        _relay_terminal(data.get("prompt_id"), "error")
    elif event == "execution_interrupted":
        _relay_terminal(data.get("prompt_id"), "interrupted")


def start() -> None:
    """Install the send_sync wrapper. Idempotent; best-effort."""
    global _installed, _original_send_sync
    if _installed:
        return
    try:
        from server import PromptServer  # type: ignore  # ComfyUI core
    except Exception:
        return
    inst = getattr(PromptServer, "instance", None)
    if inst is None:
        return
    original = getattr(inst, "send_sync", None)
    if not callable(original):
        return
    _original_send_sync = original

    def _send_sync(event, data, sid=None):
        # Native delivery first, always, untouched.
        result = original(event, data, sid)
        try:
            _maybe_relay(event, data)
        except Exception:
            global _relay_error_logged
            if not _relay_error_logged:
                _relay_error_logged = True
                _log.exception("Failed to relay progress (further errors suppressed)")
        return result

    inst.send_sync = _send_sync
    _installed = True
    _log.info("progress relay installed")
