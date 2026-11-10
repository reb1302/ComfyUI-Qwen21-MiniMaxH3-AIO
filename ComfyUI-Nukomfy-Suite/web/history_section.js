// Sidebar section + modals for the persistent job history feature.
// Visually parallel to the Availability section: the same status row,
// the same iOS toggle. Privileged actions (toggle / save settings /
// restart) always open a dedicated password confirmation popup so the
// sidebar surface stays clean and the popup title makes it unmistakable
// which manager is asking for credentials.

import {
    nukomfyPing,
    nukomfyAuth,
    nukomfyJobsHistory,
    nukomfyLiveQueue,
    nukomfyJobsHistoryWorkflowApi,
    nukomfyJobsHistoryConfig,
    nukomfyJobsHistoryConfigSet,
    nukomfyJobsHistoryClear,
    nukomfyReboot,
} from "./api.js";
import { openModal, openPasswordConfirmModal, mapAuthError } from "./modal.js";
import { faithfulStringify } from "./json_faithful.js";

// -------------------------------------------------------------------------
// DOM helpers (mirror of sidebar_tab.js helpers; duplicated here so the
// module stays standalone-importable without circular deps).
// -------------------------------------------------------------------------
function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") node.className = v;
        else if (k === "style") node.style.cssText = v;
        else if (k.startsWith("on") && typeof v === "function") {
            node.addEventListener(k.slice(2).toLowerCase(), v);
        } else if (v !== false && v !== null && v !== undefined) {
            node.setAttribute(k, v);
        }
    }
    for (const c of [].concat(children)) {
        if (c == null) continue;
        node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
}

function formatLocalTimestamp(iso) {
    if (!iso) return "";
    try {
        const d = new Date(iso);
        if (Number.isNaN(d.getTime())) return iso;
        const pad = (n) => String(n).padStart(2, "0");
        return (
            d.getFullYear() +
            "-" + pad(d.getMonth() + 1) +
            "-" + pad(d.getDate()) +
            " " + pad(d.getHours()) +
            ":" + pad(d.getMinutes()) +
            ":" + pad(d.getSeconds())
        );
    } catch (e) {
        return iso;
    }
}

function makeBanner() {
    const div = el("div", { class: "nukomfy-mgr-banner" });
    div.style.display = "none";
    return div;
}
function showBanner(banner, kind, message) {
    banner.className = `nukomfy-mgr-banner nukomfy-mgr-banner--${kind}`;
    banner.textContent = message;
    banner.style.display = "block";
}
function hideBanner(banner) {
    banner.style.display = "none";
}

// -------------------------------------------------------------------------
// Status pill helpers - label, colour and glyph for each job status
// match the Nukomfy Nuke plugin exactly.
// Unicode glyphs because the ComfyUI WebUI doesn't bundle Material Icons.
// -------------------------------------------------------------------------
const STATUS_MAP = {
    pending:   { glyph: "●", color: "#dedede", label: "Queued" },
    running:   { glyph: "●", color: "#6abf6e", label: "Running" },
    aborting:  { glyph: "●", color: "#777777", label: "Aborting…" },
    completed: { glyph: "✔", color: "#3fa3f0", label: "Completed" },
    failed:    { glyph: "✖", color: "#e53935", label: "Failed" },
    cancelled: { glyph: "⊘", color: "#a0a0a0", label: "Cancelled" },
    unknown:   { glyph: "?", color: "#888888", label: "Unknown" },
};

// The polling thread writes nfy_status_str at every lifecycle
// transition (pending at submit, running on start, terminal at end),
// so the raw column is authoritative. The empty-string fallback is a
// defensive guard for an entry that somehow carries no status.
//
// `abortingPids` is refreshed from the live /api/queue on every reload()
// (the nfy_aborting flag rides extra_data and is NOT in the persisted rows).
// A running job in that set shows as "Aborting…", matching the Nuke row.
let abortingPids = new Set();

function _abortingPidsFromQueue(queue) {
    const pids = new Set();
    const running = (queue && Array.isArray(queue.queue_running))
        ? queue.queue_running : [];
    for (const item of running) {
        // item = [number, prompt_id, prompt, extra_data, ...]; the abort flag
        // rides extra_data (index 3).
        if (Array.isArray(item) && item.length > 3 && item[1]) {
            const extra = item[3];
            if (extra && typeof extra === "object"
                && extra.nfy_aborting === true) {
                pids.add(item[1]);
            }
        }
    }
    return pids;
}

function effectiveStatus(entry) {
    const raw = (entry && entry.nfy_status_str) ? entry.nfy_status_str : "pending";
    if (raw === "running" && entry && abortingPids.has(entry.prompt_id)) {
        return "aborting";
    }
    return raw;
}

function statusInfo(statusStr) {
    return STATUS_MAP[statusStr] || STATUS_MAP.unknown;
}

function renderStatusPill(statusStr) {
    const info = statusInfo(statusStr);
    const span = el("span", { class: "nukomfy-mgr-history-status" });
    span.style.color = info.color;
    span.appendChild(el("span", { class: "nukomfy-mgr-history-status-glyph" }, info.glyph));
    span.appendChild(document.createTextNode(" " + info.label));
    return span;
}

// -------------------------------------------------------------------------
// Formatting helpers - output matches the Nukomfy Nuke plugin.
// -------------------------------------------------------------------------
function formatDuration(seconds) {
    // Truncate (not round) and drop seconds once hours appear, so the
    // plugin and the Suite show an identical duration.
    if (!seconds || seconds <= 0) return "";
    if (seconds < 60) return `${Math.trunc(seconds)}s`;
    let m = Math.trunc(seconds / 60);
    const s = Math.trunc(seconds % 60);
    if (m < 60) return `${m}m ${s}s`;
    const h = Math.trunc(m / 60);
    m = m % 60;
    return `${h}h ${m}m`;
}

function formatSeeds(seedsObj) {
    if (!seedsObj || typeof seedsObj !== "object") return "";
    const entries = Object.entries(seedsObj);
    if (entries.length === 0) return "";
    if (entries.length === 1) return `Seed: ${entries[0][1]}`;
    return "Seeds: " + entries.map(([k, v]) => `${k}: ${v}`).join(" | ");
}

function formatFrameRange(rangesList) {
    if (!Array.isArray(rangesList) || rangesList.length === 0) return "";
    const fmt = (r) => {
        // != null catches both null and undefined (an unresolved end is
        // null in JSON); 0 is a valid frame and passes through.
        const start = r.start != null ? r.start : "?";
        const end = r.end != null ? r.end : "?";
        return `${start}-${end}`;
    };
    if (rangesList.length === 1) return fmt(rangesList[0]);
    return rangesList.map((r) => `${r.name || "range"}: ${fmt(r)}`).join(" | ");
}

function cleanOutputLabel(name) {
    // Strip a trailing "File"/"file" suffix (case-sensitive, no leading
    // space required), then trailing "_ -" chars - so output labels match
    // the plugin exactly (e.g. "depthFile" -> "depth", "mask_File" -> "mask").
    if (!name) return "";
    let s = String(name);
    if (s.endsWith("File")) s = s.slice(0, -4);
    else if (s.endsWith("file")) s = s.slice(0, -4);
    return s.replace(/[_ -]+$/, "");
}

function formatOutputPaths(pathsList) {
    if (!Array.isArray(pathsList) || pathsList.length === 0) return "";
    if (pathsList.length === 1) {
        const p = pathsList[0];
        return typeof p === "string" ? p : (p.path || "");
    }
    return pathsList.map((p, idx) => {
        if (typeof p === "string") return `output ${idx + 1}: ${p}`;
        const label = cleanOutputLabel(p.name) || `output ${idx + 1}`;
        return `${label}: ${p.path || ""}`;
    }).join(" | ");
}

function shortPid(pid) {
    if (!pid) return "";
    return String(pid).length > 12 ? String(pid).slice(0, 8) + "…" : pid;
}

// Strip the verbose `(tags/..., compiler info)` suffix from sys.version,
// keeping only the leading semver chunk so the displayed version matches
// the Nuke plugin. The raw sys.version is e.g.
// `3.13.11 (tags/v3.13.11:6278944, ...) [MSC ...]`.
function _trimPythonVersion(value) {
    if (!value || typeof value !== "string") return value;
    const parts = value.split(/\s+/);
    return parts[0] || value;
}

function _formatGB(bytes) {
    if (!bytes || typeof bytes !== "number") return "";
    const gb = bytes / (1024 ** 3);
    return `${gb.toFixed(1)} GB`;
}

// -------------------------------------------------------------------------
// History list modal
// -------------------------------------------------------------------------
function _td(text, options) {
    const opts = options || {};
    const className = opts.class || "";
    const value = text === null || text === undefined ? "" : String(text);
    const td = el("td", {
        class: className,
        title: opts.title !== undefined ? opts.title : value,
    }, value);
    return td;
}

function _buildHistoryTable(entries, onRowClick) {
    const COLS = [
        { label: "Status",   w: "100px" },
        { label: "Job ID",   w: "90px"  },
        { label: "Sent",     w: "150px" },
        { label: "User",     w: "110px" },
        { label: "Workflow", w: "180px" },
        { label: "Node",     w: "150px" },
        { label: "NK File",  w: "auto"  },
        { label: "Duration", w: "90px"  },
    ];
    const table = el("table", { class: "nukomfy-mgr-history-table" });
    const colgroup = el("colgroup");
    for (const c of COLS) colgroup.appendChild(el("col", { style: `width:${c.w}` }));
    table.appendChild(colgroup);
    table.appendChild(el("thead", {}, [
        el("tr", {}, COLS.map((c) => el("th", {}, c.label))),
    ]));
    const tbody = el("tbody");
    for (const entry of entries) {
        const tr = el("tr", { class: "nukomfy-mgr-history-row" });
        // Status pill is rich content - keep the title attribute on the cell
        // so hovering shows the same label as the pill text.
        const statusInfo_ = statusInfo(effectiveStatus(entry));
        const statusTd = el("td", { title: statusInfo_.label },
            [renderStatusPill(effectiveStatus(entry))]);
        tr.appendChild(statusTd);
        tr.appendChild(_td(shortPid(entry.nfy_job_id) || "External",
            { class: "nukomfy-mgr-mono" }));
        tr.appendChild(_td(formatLocalTimestamp(entry.nfy_sent_at) || "-"));
        tr.appendChild(_td(entry.nfy_submitted_by || "-"));
        tr.appendChild(_td(entry.nfy_workflow_name || "-"));
        tr.appendChild(_td(entry.nfy_node_name || "-"));
        tr.appendChild(_td(entry.nfy_nk_file || "-",
            { class: "nukomfy-mgr-history-nk", title: entry.nfy_nk_file || "Unknown" }));
        tr.appendChild(_td(formatDuration(entry.nfy_duration) || "-",
            { class: "nukomfy-mgr-history-duration" }));
        tr.addEventListener("click", () => onRowClick(entry));
        tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    return table;
}

// Status filter options (value -> label), mirrors the plugin's filter and
// the server-side `status` param. localStorage remembers the selection.
const _STATUS_FILTER_OPTIONS = [
    ["completed", "Completed"],
    ["failed", "Failed"],
    ["cancelled", "Cancelled"],
    ["running", "Running"],
    ["pending", "Queued"],
];
const _STATUS_FILTER_STORE_KEY = "nukomfy.history.statusFilter";

function _loadStatusFilter() {
    try {
        const arr = JSON.parse(localStorage.getItem(_STATUS_FILTER_STORE_KEY) || "[]");
        const valid = _STATUS_FILTER_OPTIONS.map((o) => o[0]);
        return Array.isArray(arr) ? arr.filter((s) => valid.includes(s)) : [];
    } catch (e) {
        return [];
    }
}
function _saveStatusFilter(statuses) {
    try {
        localStorage.setItem(_STATUS_FILTER_STORE_KEY, JSON.stringify(statuses));
    } catch (e) { /* private mode / quota - non-fatal */ }
}

function openHistoryListModal({ hasPassword = false, onChanged } = {}) {
    let allEntries = [];
    let limitChoice = 50;
    let selectedStatuses = _loadStatusFilter();
    let searchTimer = null;
    let currentPage = 1;
    let totalCount = 0;
    let reqSeq = 0;

    const searchInput = el("input", {
        type: "text",
        class: "nukomfy-mgr-history-search",
        placeholder: "Search workflow, user, node, job id…",
    });
    const searchClear = el("button", {
        class: "nukomfy-mgr-history-search-clear",
        title: "Clear search",
    }, "×");
    searchClear.style.display = "none";
    const searchWrap = el("div", { class: "nukomfy-mgr-history-search-wrap" },
        [searchInput, searchClear]);
    function updateClearX() {
        searchClear.style.display = searchInput.value ? "" : "none";
    }
    const counter = el("span", { class: "nukomfy-mgr-history-counter" }, "0 entries");
    const refreshBtn = el("button", { class: "nukomfy-mgr-btn" }, "Refresh");
    const perPageLabel = el("span",
        { class: "nukomfy-mgr-history-perpage" }, "Per page:");
    const limitSelect = el("select", { class: "nukomfy-mgr-history-limit" });
    for (const n of [50, 100, 200]) {
        const opt = el("option", { value: String(n) }, String(n));
        if (n === limitChoice) opt.selected = true;
        limitSelect.appendChild(opt);
    }

    // Status filter ("Filter" dropdown): server-side via the status param,
    // selection persisted in localStorage. Unicode caret (Suite has no
    // Material Icons). A flat popup of checkboxes opens below the button.
    const filterBtn = el("button", {
        class: "nukomfy-mgr-btn nukomfy-mgr-history-filter",
    });
    const filterPopup = el("div", { class: "nukomfy-mgr-history-filter-popup" });
    filterPopup.style.display = "none";
    const filterChecks = {};
    for (const [value, label] of _STATUS_FILTER_OPTIONS) {
        const cb = el("input", { type: "checkbox" });
        cb.checked = selectedStatuses.includes(value);
        cb.addEventListener("change", () => {
            selectedStatuses = _STATUS_FILTER_OPTIONS
                .map((o) => o[0]).filter((s) => filterChecks[s].checked);
            _saveStatusFilter(selectedStatuses);
            updateFilterLabel();
            currentPage = 1;
            reload();
        });
        filterChecks[value] = cb;
        filterPopup.appendChild(
            el("label", { class: "nukomfy-mgr-history-filter-item" }, [cb, label]));
    }
    const filterWrap = el("div", { class: "nukomfy-mgr-history-filter-wrap" },
        [filterBtn, filterPopup]);
    function updateFilterLabel() {
        const n = selectedStatuses.length;
        filterBtn.textContent = n ? `Filter (${n}) ▾` : "Filter ▾";
    }
    function _onFilterDocClick(ev) {
        if (!filterWrap.contains(ev.target)) {
            filterPopup.style.display = "none";
            document.removeEventListener("click", _onFilterDocClick);
        }
    }
    filterBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (filterPopup.style.display === "none") {
            filterPopup.style.display = "";
            setTimeout(() => document.addEventListener("click", _onFilterDocClick), 0);
        } else {
            filterPopup.style.display = "none";
            document.removeEventListener("click", _onFilterDocClick);
        }
    });
    updateFilterLabel();

    // Pager: «  ‹  Page X of Y  ›  »
    const pagerFirst = el("button", { class: "nukomfy-mgr-btn" }, "«");
    const pagerPrev  = el("button", { class: "nukomfy-mgr-btn" }, "‹");
    const pagerLabel = el("span", { class: "nukomfy-mgr-history-pager-label" }, "Page 1 of 1");
    const pagerNext  = el("button", { class: "nukomfy-mgr-btn" }, "›");
    const pagerLast  = el("button", { class: "nukomfy-mgr-btn" }, "»");
    const pagerWrap  = el("span", { class: "nukomfy-mgr-history-pager" },
        [pagerFirst, pagerPrev, pagerLabel, pagerNext, pagerLast]);

    const tableHost = el("div", { class: "nukomfy-mgr-history-table-host" });

    function rerenderTable() {
        // Search + status filter run server-side, so allEntries is already the
        // filtered page - render it as-is.
        tableHost.innerHTML = "";
        if (allEntries.length === 0) {
            const filtering = !!searchInput.value.trim() || selectedStatuses.length > 0;
            tableHost.appendChild(el("div", { class: "nukomfy-mgr-history-empty" },
                filtering
                    ? "No entries match the current filter."
                    : "No history entries yet."));
        } else {
            tableHost.appendChild(_buildHistoryTable(allEntries, openDetailModal));
        }
        if (totalCount === 0) {
            counter.textContent = "0 entries";
        } else {
            const from = (currentPage - 1) * limitChoice + 1;
            const to = from + allEntries.length - 1;
            counter.textContent = `Showing ${from}-${to} of ${totalCount}`;
        }
    }

    function updatePagerUI(totalPages) {
        // Pager stays visible (buttons greyed when not applicable) so the
        // toolbar layout doesn't shift when a filter shrinks the result set.
        pagerLabel.textContent = `Page ${currentPage} of ${totalPages}`;
        const atFirst = currentPage <= 1;
        const atLast  = currentPage >= totalPages;
        pagerFirst.disabled = atFirst;
        pagerPrev.disabled  = atFirst;
        pagerNext.disabled  = atLast;
        pagerLast.disabled  = atLast;
    }

    async function reload() {
        // Sequence guard: only the most recent reload writes state and
        // renders. Without it a slow response (e.g. page 1) can resolve after
        // a newer one (page 2) and overwrite it, desyncing the counter and
        // rows under fast interaction (search + pager/filter overlap).
        const seq = ++reqSeq;
        tableHost.innerHTML = "";
        tableHost.appendChild(el("div", { class: "nukomfy-mgr-history-loading" }, "Loading…"));
        const offset = (currentPage - 1) * limitChoice;
        const [data, queue] = await Promise.all([
            nukomfyJobsHistory(limitChoice, offset, {
                statuses: selectedStatuses.length ? selectedStatuses : null,
                query: searchInput.value.trim() || null,
            }),
            nukomfyLiveQueue(),
        ]);
        if (seq !== reqSeq) return;   // superseded by a newer reload
        abortingPids = _abortingPidsFromQueue(queue);
        allEntries = (data && Array.isArray(data.entries)) ? data.entries : [];
        totalCount = (data && typeof data.total_count === "number")
            ? data.total_count
            : allEntries.length;
        // Clamp page if total shrank under us (e.g. retention purge between
        // reloads, or Delete history clicked from another tab).
        const totalPages = Math.max(1, Math.ceil(totalCount / limitChoice));
        if (currentPage > totalPages) {
            currentPage = totalPages;
            return reload();
        }
        rerenderTable();
        updatePagerUI(totalPages);
    }

    function goToPage(target) {
        const totalPages = Math.max(1, Math.ceil(totalCount / limitChoice));
        const clamped = Math.min(Math.max(1, target), totalPages);
        if (clamped === currentPage) return;
        currentPage = clamped;
        reload();
    }

    searchInput.addEventListener("input", () => {
        updateClearX();
        clearTimeout(searchTimer);
        searchTimer = setTimeout(() => { currentPage = 1; reload(); }, 250);
    });
    searchClear.addEventListener("click", () => {
        searchInput.value = "";
        updateClearX();
        searchInput.focus();
        currentPage = 1;
        reload();
    });
    refreshBtn.addEventListener("click", reload);
    limitSelect.addEventListener("change", () => {
        limitChoice = parseInt(limitSelect.value, 10) || 50;
        currentPage = 1;
        reload();
    });
    pagerFirst.addEventListener("click", () => goToPage(1));
    pagerPrev.addEventListener("click",  () => goToPage(currentPage - 1));
    pagerNext.addEventListener("click",  () => goToPage(currentPage + 1));
    pagerLast.addEventListener("click",  () => {
        const totalPages = Math.max(1, Math.ceil(totalCount / limitChoice));
        goToPage(totalPages);
    });

    const toolbar = el("div", { class: "nukomfy-mgr-history-toolbar" }, [
        searchWrap, filterWrap, perPageLabel, limitSelect, refreshBtn,
        pagerWrap, counter,
    ]);

    const deleteBtn = el("button", {
        class: "nukomfy-mgr-btn nukomfy-mgr-btn--danger-primary nukomfy-mgr-btn--sm",
    }, "Delete history");
    if (!hasPassword) {
        deleteBtn.disabled = true;
        deleteBtn.title = "Set an admin password first";
    }
    deleteBtn.addEventListener("click", () => {
        if (deleteBtn.disabled) return;
        openPasswordConfirmModal({
            title: "Delete job history",
            description:
                "Permanently removes every entry from the local job log. " +
                "This cannot be undone. If history persistence is enabled, " +
                "new submissions will repopulate the log from now on.",
            actionLabel: "Delete",
            onConfirm: async (pwd, ctx) => {
                const res = await nukomfyJobsHistoryClear(pwd);
                if (res.status === 200 && res.body && res.body.ok) {
                    ctx.close();
                    currentPage = 1;
                    reload();
                    if (typeof onChanged === "function") onChanged();
                    return;
                }
                ctx.showError(mapAuthError(res));
            },
        });
    });
    const footer = el("div", { class: "nukomfy-mgr-history-modal-footer" }, [
        deleteBtn,
    ]);

    const body = el("div", { class: "nukomfy-mgr-history-modal-body" },
        [toolbar, tableHost, footer]);
    openModal({ title: "Job history", content: body,
        width: 1100, height: 700,
        onClose: () => {
            // Drop the pending debounce and the document-level click listener
            // so neither fires after the modal is gone.
            clearTimeout(searchTimer);
            document.removeEventListener("click", _onFilterDocClick);
        } });
    reload();
}

// -------------------------------------------------------------------------
// Detail job modal - three tabs (Detail / Log / Workflow API).
// -------------------------------------------------------------------------
function _safeJson(value, fallback) {
    if (value === null || value === undefined) return fallback;
    if (typeof value === "object") return value;
    try {
        const parsed = JSON.parse(value);
        return parsed === null ? fallback : parsed;
    } catch (e) {
        return fallback;
    }
}

// The client plugin marks "hidden URL" machines by sending the URL
// already obfuscated (prefix "nfyo:") inside extra_data. The server
// has no key to decode and, by policy, should not even try - it
// honours the client's "hide URL" intent by showing a placeholder.
function _resolveMachineDisplay(url) {
    if (typeof url === "string" && url.startsWith("nfyo:")) {
        return { display: "(Hidden)", isHidden: true };
    }
    return { display: url || "", isHidden: false };
}


function _kvRow(label, value, mono = false) {
    // value may be a DOM node (styled content, e.g. an amber warning glyph)
    // or a plain string/number; both are accepted.
    const isNode = value instanceof Node;
    const hasValue = isNode
        || (value !== null && value !== undefined && value !== "");
    const child = !hasValue ? "-" : (isNode ? value : String(value));
    const title = isNode ? (value.textContent || "") : String(child);
    return el("div", { class: "nukomfy-mgr-detail-field-row" }, [
        el("span", { class: "nukomfy-mgr-detail-field-label" }, label),
        el("span", {
            class: "nukomfy-mgr-detail-field-value"
                + (mono ? " nukomfy-mgr-mono" : "")
                + (hasValue ? "" : " nukomfy-mgr-detail-field-value--empty"),
            title: title,
        }, child),
    ]);
}

function _makeSection(title) {
    const caret = el("span", { class: "nukomfy-mgr-detail-section-caret" }, "▼ ");
    const titleEl = el("div", { class: "nukomfy-mgr-detail-section-title" }, [caret, title]);
    const body = el("div", { class: "nukomfy-mgr-detail-section-body" });
    let open = true;
    titleEl.addEventListener("click", () => {
        open = !open;
        body.style.display = open ? "block" : "none";
        caret.textContent = open ? "▼ " : "▶ ";
    });
    const wrap = el("div", { class: "nukomfy-mgr-detail-section" }, [titleEl, body]);
    return { wrap, body };
}

function _appendRows(container, rows) {
    for (const r of rows) if (r) container.appendChild(r);
}

function _padSectionIfEmpty(body) {
    if (body.childNodes.length === 0) {
        body.appendChild(el("div", { class: "nukomfy-mgr-detail-empty" },
            "No data available."));
    }
}

function renderDetailTab(entry, wfApiPromise) {
    const wrap = el("div", { class: "nukomfy-mgr-detail-tab" });
    const meta = _safeJson(entry.nfy_workflow_meta, {});
    const env = _safeJson(entry.nfy_environment, {});
    const snap = _safeJson(entry.nfy_server_snapshot, {});
    const frames = _safeJson(entry.nfy_frame_range, []);
    const outputs = _safeJson(entry.nfy_output_paths, []);

    // Header compact (9 fields)
    const header = el("div", { class: "nukomfy-mgr-detail-header" });
    _appendRows(header, [
        _kvRow("Job ID", entry.nfy_job_id, true),
        _kvRow("Prompt ID", entry.prompt_id, true),
    ]);
    // Detail header status: coloured label WITHOUT the glyph, matching the
    // Nuke plugin's Detail (the colour conveys state). renderStatusPill, with
    // the glyph, stays for the compact history list.
    const _statusInfo = statusInfo(effectiveStatus(entry));
    const _statusVal = el("span", { class: "nukomfy-mgr-detail-field-value" },
                          _statusInfo.label);
    _statusVal.style.color = _statusInfo.color;
    header.appendChild(el("div", { class: "nukomfy-mgr-detail-field-row" }, [
        el("span", { class: "nukomfy-mgr-detail-field-label" }, "Status"),
        _statusVal,
    ]));
    _appendRows(header, [
        _kvRow("Submitted", formatLocalTimestamp(entry.nfy_sent_at)),
        _kvRow("Duration", formatDuration(entry.nfy_duration)),
        _kvRow("Submitted by", entry.nfy_submitted_by),
        (() => {
            const m = _resolveMachineDisplay(entry.nfy_machine_url);
            const text = entry.nfy_machine_name
                ? (m.isHidden
                    ? entry.nfy_machine_name
                    : `${entry.nfy_machine_name} (${m.display})`)
                : m.display;
            return _kvRow("Machine", text);
        })(),
        _kvRow("Output range", formatFrameRange(frames)),
        _kvRow("Output", formatOutputPaths(outputs)),
    ]);
    wrap.appendChild(header);

    // Submission section
    {
        const s = _makeSection("Submission");
        _appendRows(s.body, [
            _kvRow("Submitter", entry.nfy_submitted_by),
            _kvRow("Submitter host", entry.nfy_submitter_host),
            _kvRow("OS", env.os_submitter),
            _kvRow("Nuke version", env.nuke_version),
            _kvRow("Nukomfy version", env.nukomfy_version),
            _kvRow("Sent", formatLocalTimestamp(entry.nfy_sent_at)),
        ]);
        wrap.appendChild(s.wrap);
    }
    // Machine snapshot
    {
        const s = _makeSection("Machine snapshot at submit");
        const stats = snap.system_stats || {};
        const pyv = _trimPythonVersion(stats.python_version);
        const vram = stats.vram_total ? _formatGB(stats.vram_total) : null;
        const ram = stats.ram_total ? _formatGB(stats.ram_total) : null;
        _appendRows(s.body, [
            _kvRow("Name", entry.nfy_machine_name),
            _kvRow("URL", _resolveMachineDisplay(entry.nfy_machine_url).display, true),
            _kvRow("OS", stats.os),
            _kvRow("ComfyUI version", snap.server_version || stats.comfyui_version),
            _kvRow("Python version", pyv),
            _kvRow("GPU", stats.gpu),
            _kvRow("VRAM", vram),
            _kvRow("RAM", ram),
        ]);
        wrap.appendChild(s.wrap);
    }
    // Workflow section
    {
        const s = _makeSection("Workflow");
        _appendRows(s.body, [
            _kvRow("Name", entry.nfy_workflow_name),
            _kvRow("Version", meta.workflow_version),
            _kvRow("Author", meta.workflow_author),
            _kvRow("UUID", meta.workflow_uuid, true),
        ]);
        wrap.appendChild(s.wrap);
    }
    // Render configuration - labels match the plugin exactly
    // (singular/plural), ranges and paths go multi-line when there's more
    // than one entry, output paths are suppressed for cancelled/failed jobs.
    {
        const s = _makeSection("Render configuration");
        const seeds = _safeJson(entry.nfy_seeds_used, {});
        const inputRanges = _safeJson(entry.nfy_input_ranges, []);
        const outputRanges = _safeJson(entry.nfy_output_ranges, null) || frames;
        const inputCache = env.input_cache;
        const writeTemplates = meta.write_templates;

        _appendRows(s.body, [
            _kvRow("NK File", entry.nfy_nk_file),
            _kvRow("Node", entry.nfy_node_name),
            _kvRow("Batch",
                `${entry.nfy_batch_index || 1}/${entry.nfy_batch_count || 1}`),
        ]);
        _appendRangeRows(s.body, outputRanges, "Output Range", "Output Ranges");
        _appendRangeRows(s.body, inputRanges, "Input Range", "Input Ranges");
        const seedRow = _renderSeedsRow(seeds);
        s.body.appendChild(seedRow);
        _appendListRows(s.body, _normaliseInputCache(inputCache),
            "Input cache", "Input caches");
        _appendListRows(s.body, _normaliseWriteTemplates(writeTemplates),
            "Write template", "Write templates");
        // Output paths row(s): always shown, including cancelled/failed
        // jobs - the path is a static job metadatum (where output was
        // expected), useful to locate partial output even after a failure.
        _appendOutputPathsRows(s.body, outputs);
        wrap.appendChild(s.wrap);
    }
    // Submitted parameters - matches the Nukomfy Nuke plugin. Loads the
    // workflow_api blob asynchronously and fills the section once
    // available; until then a Loading placeholder is shown.
    {
        const s = _makeSection("Submitted parameters");
        s.body.appendChild(el("div", { class: "nukomfy-mgr-detail-empty" },
                              "Loading…"));
        if (wfApiPromise && typeof wfApiPromise.then === "function") {
            wfApiPromise.then((wfApi) => {
                s.body.innerHTML = "";
                s.body.appendChild(_renderParamsSpec(meta, wfApi));
            });
        }
        wrap.appendChild(s.wrap);
    }
    return wrap;
}

// Helpers for Submitted parameters rendering, matching the Nukomfy
// Nuke plugin.
function _paramValueFor(wfApi, nodeId, name) {
    if (!wfApi || typeof wfApi !== "object") return null;
    const node = wfApi[String(nodeId)];
    if (!node || typeof node !== "object") return null;
    const inputs = node.inputs;
    if (!inputs || typeof inputs !== "object") return null;
    return name in inputs ? inputs[name] : null;
}

function _paramClassFor(wfApi, nodeId) {
    if (!wfApi || typeof wfApi !== "object") return "?";
    const node = wfApi[String(nodeId)];
    if (node && typeof node === "object" && node.class_type) {
        return String(node.class_type);
    }
    return "?";
}

function _paramValueNode(value) {
    // Mirrors the Python branching: link reference -> "<link>"; null /
    // undefined -> dash placeholder; empty string -> "(empty)" italic
    // grey; otherwise the full stringified value. No length cap - a long
    // value (e.g. a positive prompt) wraps rather than being cut; the
    // Detail view must never hide submitted data behind a truncation.
    if (Array.isArray(value)) {
        return el("span", { class: "nukomfy-mgr-mono" }, "<link>");
    }
    if (value === null || value === undefined) {
        return el("span", { class: "nukomfy-mgr-detail-field-value--empty" }, "-");
    }
    if (typeof value === "string" && value === "") {
        return el("span", { class: "nukomfy-mgr-detail-param-empty" }, "(empty)");
    }
    return el("span", {}, String(value));
}

function _buildParamRow(p, wfApi) {
    const nodeId = String(p.node_id || "");
    const name = p.name || "";
    if (!nodeId || !name) return null;
    const cls = _paramClassFor(wfApi, nodeId);
    const value = _paramValueFor(wfApi, nodeId, name);
    // A param absent from the workflow API payload was never sent (e.g. a
    // V3 sub of a non-active option, stripped before submit) - skip it
    // instead of listing a bare dash. Empty string is kept (the user chose
    // to send nothing) and renders as "(empty)".
    if (value === null || value === undefined) return null;
    const label = p.label || "";
    const labelSuffix = (label && label !== name) ? ` (${label})` : "";

    const row = el("div", { class: "nukomfy-mgr-detail-param-row" });
    row.appendChild(el("span", { class: "nukomfy-mgr-detail-param-class" },
        `(${cls}) [ID: ${nodeId}]`));
    row.appendChild(document.createTextNode(" "));
    row.appendChild(el("span", { class: "nukomfy-mgr-detail-param-name" },
        name + labelSuffix));
    row.appendChild(document.createTextNode(" = "));
    row.appendChild(_paramValueNode(value));
    return row;
}

// Multi-entry row helpers used by the Render configuration section.
// All follow the same shape: when 1 entry exists the row is an inline
// "<label>: <value>"; when 2+ exist the row becomes a "<label_plural>:"
// header followed by indented "<name>: <value>" sub-rows.

function _indentedRow(name, value) {
    // Accept a DOM node (styled content) or a plain string, like _kvRow.
    const isNode = value instanceof Node;
    const child = isNode
        ? value
        : String(value === null || value === undefined ? "" : value);
    const title = isNode ? (value.textContent || "") : child;
    return el("div", { class: "nukomfy-mgr-detail-indented-row" }, [
        el("span", { class: "nukomfy-mgr-detail-field-name" }, name + ": "),
        el("span", { class: "nukomfy-mgr-detail-field-value", title: title },
            child),
    ]);
}

function _headerRow(text) {
    return el("div", { class: "nukomfy-mgr-detail-subheader" }, text);
}

function _appendRangeRows(container, ranges, labelSingular, labelPlural) {
    const valid = Array.isArray(ranges)
        ? ranges.filter((r) => r && typeof r === "object")
        : [];
    if (valid.length === 0) {
        container.appendChild(_kvRow(labelSingular, null));
        return;
    }
    function fmt(r) {
        const start = (r.start !== null && r.start !== undefined) ? r.start : "?";
        const end = (r.end !== null && r.end !== undefined) ? r.end : "?";
        return `${start}-${end}`;
    }
    if (valid.length === 1) {
        container.appendChild(_kvRow(labelSingular, fmt(valid[0])));
        return;
    }
    container.appendChild(_headerRow(labelPlural + ":"));
    for (const r of valid) {
        container.appendChild(_indentedRow(r.name || "?", fmt(r)));
    }
}

function _renderSeedsRow(seedsObj) {
    if (!seedsObj || typeof seedsObj !== "object") {
        return _kvRow("Seed", null);
    }
    const entries = Object.entries(seedsObj);
    if (entries.length === 0) return _kvRow("Seed", null);
    if (entries.length === 1) {
        return _kvRow("Seed", entries[0][1]);
    }
    return _kvRow("Seeds",
        entries.map(([k, v]) => `${k}: ${v}`).join(" | "));
}

function _normaliseInputCache(value) {
    if (!value) return [];
    if (typeof value === "string") return [value];
    if (Array.isArray(value)) return value.filter(Boolean);
    return [];
}

function _normaliseWriteTemplates(value) {
    if (!Array.isArray(value)) return [];
    return value.filter((e) => e && typeof e === "object");
}

function _formatWriteTemplate(entry) {
    let tpl = String((entry.template || "")).trim() || "?";
    if (tpl.toLowerCase().endsWith(".nk")) tpl = tpl.slice(0, -3);
    const scope = String(entry.scope || "").toLowerCase() || "?";
    const base = `${tpl} (${scope})`;
    if (entry.missing) {
        // The configured template was not found at gizmo build time and a
        // default Write node was used instead. Amber glyph matches the Nuke
        // Render Manager Detail; the message text keeps the default colour.
        return el("span", {}, [
            // Amber warning colour matching the Nuke Render Manager Detail.
            el("span", { style: "color: #f99500;" }, "⚠"),
            ` Error: ${base} not found. Used a default Write node.`,
        ]);
    }
    return base;
}

function _appendListRows(container, items, labelSingular, labelPlural) {
    if (!items || items.length === 0) {
        container.appendChild(_kvRow(labelSingular, null));
        return;
    }
    if (items.length === 1) {
        const v = items[0];
        const str = (typeof v === "object")
            ? _formatWriteTemplate(v)
            : String(v);
        container.appendChild(_kvRow(labelSingular, str));
        return;
    }
    container.appendChild(_headerRow(labelPlural + ":"));
    for (const v of items) {
        if (typeof v === "object" && v) {
            const input = String(v.input || "").trim() || "input";
            container.appendChild(_indentedRow(input, _formatWriteTemplate(v)));
        } else {
            const path = String(v);
            const parts = path.replace(/\\/g, "/").replace(/\/+$/, "").split("/");
            const label = parts.length >= 2 ? parts[parts.length - 2] : "input";
            container.appendChild(_indentedRow(label, path));
        }
    }
}

function _appendOutputPathsRows(container, paths) {
    if (!Array.isArray(paths) || paths.length === 0) {
        container.appendChild(_kvRow("Output Path", null));
        return;
    }
    if (paths.length === 1) {
        const p = paths[0];
        const str = typeof p === "string" ? p : (p && p.path) || "";
        container.appendChild(_kvRow("Output Path", str));
        return;
    }
    container.appendChild(_headerRow("Output Paths:"));
    for (let i = 0; i < paths.length; i++) {
        const p = paths[i];
        const path = typeof p === "string" ? p : (p && p.path) || "";
        const rawName = (typeof p === "object" && p && p.name) || "";
        const label = cleanOutputLabel(rawName) || `output ${i + 1}`;
        container.appendChild(_indentedRow(label, path));
    }
}

function _renderParamsSpec(meta, wfApi) {
    const wrap = el("div");
    const spec = (meta && Array.isArray(meta.params_spec)) ? meta.params_spec : [];

    const visible = [];
    const hidden = [];
    for (const p of spec) {
        if (!p || typeof p !== "object") continue;
        const row = _buildParamRow(p, wfApi);
        if (!row) continue;
        if (p.enabled) visible.push(row);
        else hidden.push(row);
    }

    function appendGroup(title, rows, marginTop) {
        const header = el("div", {
            class: "nukomfy-mgr-detail-param-group-title",
            style: marginTop ? `margin-top: ${marginTop}px;` : "",
        }, title);
        wrap.appendChild(header);
        if (rows.length === 0) {
            wrap.appendChild(el("div", {
                class: "nukomfy-mgr-detail-field-value--empty",
            }, "-"));
        } else {
            for (const r of rows) wrap.appendChild(r);
        }
    }
    appendGroup("Visible", visible, 0);
    appendGroup("Hidden", hidden, 8);

    return wrap;
}

function _logTimestamp(payload) {
    let ts = payload && payload.timestamp;
    if (!ts) return "";
    // ComfyUI emits both seconds and milliseconds depending on the
    // event; the heuristic matches the Nukomfy Nuke plugin.
    if (ts > 1e12) ts = ts / 1000;
    const d = new Date(ts * 1000);
    if (Number.isNaN(d.getTime())) return "";
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} `
        + `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

// Map an execution event onto (Unicode glyph, colour). The Material Icon
// + colour table matches the Nukomfy Nuke plugin; the WebUI doesn't ship
// Material Icons, so we use Unicode replacements that read close enough in
// a monospace dark theme.
const _LOG_GLYPHS = {
    execution_start:       { glyph: "▶",  color: "#6abf6e" },
    execution_cached:      { glyph: "↻",  color: "#888888" },
    executing:             { glyph: "●",  color: "#dedede" },
    execution_success:     { glyph: "✔",  color: "#6abf6e" },
    execution_error:       { glyph: "✖",  color: "#e53935" },
    execution_interrupted: { glyph: "⚠",  color: "#f99500" },
};

// Plain-language lines for terminal reasons that left ComfyUI with no
// captured execution log (a server restart/crash mid-render, or a running
// job dropped from the queue). Matches the Nukomfy Nuke plugin.
const _LOG_FOR_REASON = {
    server_restart:
        "The server was restarted while this job was running, "
        + "so it did not finish.",
    running_disappeared:
        "This job stopped responding and was dropped from the queue.",
};

function _logGlyphHtml(eventName) {
    const info = _LOG_GLYPHS[eventName];
    if (!info) return "";
    return `<span class="nukomfy-mgr-detail-log-glyph" `
        + `style="color:${info.color};">${info.glyph}</span> `;
}

function renderLogTab(entry) {
    const wrap = el("div", { class: "nukomfy-mgr-detail-tab nukomfy-mgr-detail-log" });
    const messages = _safeJson(entry.nfy_messages, []);
    const error = _safeJson(entry.nfy_execution_error, null);

    function appendLine(html) {
        const div = el("div", { class: "nukomfy-mgr-detail-log-line" });
        div.innerHTML = html;
        wrap.appendChild(div);
    }
    function escapeHtml(s) {
        // Escape quotes too, so the helper stays safe if a value is ever
        // interpolated into an HTML attribute, not only text content.
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    if (Array.isArray(messages) && messages.length > 0) {
        for (const m of messages) {
            if (!Array.isArray(m) || m.length < 2) continue;
            const [name, payload] = m;
            const data = (payload && typeof payload === "object") ? payload : {};
            const ts = _logTimestamp(data) || "?";
            const g = _logGlyphHtml(name);
            // Timestamp-first layout (syslog/journalctl convention).
            const tsHtml = `<span class="nukomfy-mgr-detail-log-ts">`
                + `${escapeHtml(ts)}</span>`;
            if (name === "execution_start") {
                appendLine(`${g}${tsHtml}  Started`);
            } else if (name === "execution_cached") {
                const nodes = Array.isArray(data.nodes) ? data.nodes : [];
                appendLine(`${g}${tsHtml}  Cached: ${nodes.length} nodes`);
            } else if (name === "executing") {
                const node = data.display_node || data.node || "?";
                appendLine(`${g}${tsHtml}  Executing node ${escapeHtml(node)}`);
            } else if (name === "execution_success") {
                appendLine(`${g}${tsHtml}  Completed`);
            } else if (name === "execution_interrupted") {
                appendLine(`${g}${tsHtml}  Interrupted`);
            } else if (name === "execution_error") {
                const errBlock = el("div", { class: "nukomfy-mgr-detail-log-error" });
                const header = el("div", { class: "nukomfy-mgr-detail-log-error-msg" });
                header.innerHTML = `${g}${tsHtml}  <b>Error</b>`;
                errBlock.appendChild(header);
                if (data.exception_message) {
                    errBlock.appendChild(el(
                        "div", { class: "nukomfy-mgr-detail-log-error-msg" },
                        String(data.exception_message)));
                }
                if (data.node_id || data.node_type) {
                    const meta = el("div",
                        { class: "nukomfy-mgr-detail-log-error-type" });
                    meta.innerHTML =
                        `<i>At node: ${escapeHtml(data.node_type || "?")} `
                        + `(${escapeHtml(data.node_id || "?")})</i>`;
                    errBlock.appendChild(meta);
                }
                if (data.traceback) {
                    const tb = Array.isArray(data.traceback)
                        ? data.traceback.join("\n")
                        : String(data.traceback);
                    errBlock.appendChild(el(
                        "pre", { class: "nukomfy-mgr-detail-log-trace" },
                        tb.trim()));
                }
                wrap.appendChild(errBlock);
            }
        }
    }

    // Top-level execution_error (separate from the message stream): only
    // render if the stream didn't already include it inline.
    const errorPainted = wrap.querySelector(".nukomfy-mgr-detail-log-error");
    if (!errorPainted && error && typeof error === "object") {
        const reasonText = _LOG_FOR_REASON[error.reason];
        if (reasonText) {
            // Known terminal reason with no captured log (server restart,
            // dropped job): one plain-language line instead of an empty red
            // panel / "No data available.".
            appendLine(
                `${_logGlyphHtml("execution_error")}${escapeHtml(reasonText)}`);
        } else {
            const block = el("div", { class: "nukomfy-mgr-detail-log-error" });
            if (error.exception_type) {
                block.appendChild(el("div", { class: "nukomfy-mgr-detail-log-error-type" },
                                      String(error.exception_type)));
            }
            if (error.exception_message) {
                block.appendChild(el("div", { class: "nukomfy-mgr-detail-log-error-msg" },
                                      String(error.exception_message)));
            }
            if (error.traceback) {
                const tb = Array.isArray(error.traceback)
                    ? error.traceback.join("\n")
                    : String(error.traceback);
                block.appendChild(el("pre", { class: "nukomfy-mgr-detail-log-trace" }, tb));
            }
            if (block.childNodes.length > 0) {
                wrap.appendChild(block);
            }
        }
    }

    if (wrap.children.length === 0) {
        wrap.appendChild(el("div", { class: "nukomfy-mgr-detail-empty" },
            "No data available."));
    }
    return wrap;
}

function renderWorkflowApiTab(promptId, wfApiPromise) {
    // Copy / Save live in the shared modal footer (acting on the active tab),
    // so this tab only renders the JSON. faithfulStringify keeps 64-bit
    // seeds and float formatting exact in the dump.
    const wrap = el("div", {
        class: "nukomfy-mgr-detail-tab nukomfy-mgr-detail-tab--wfapi",
    });
    const status = el("div", { class: "nukomfy-mgr-detail-empty" }, "Loading…");
    const pre = el("pre", { class: "nukomfy-mgr-detail-json" });
    pre.style.display = "none";
    wrap.appendChild(status);
    wrap.appendChild(pre);
    wfApiPromise.then((wfApi) => {
        if (wfApi && typeof wfApi === "object") {
            pre.textContent = faithfulStringify(wfApi, 2);
            pre.style.display = "block";
            status.style.display = "none";
        } else {
            status.textContent = "Workflow API snapshot not available for this entry.";
        }
    });
    return wrap;
}

// Faithful plain-text extraction for Copy / Save. Walks raw text nodes (not
// innerText, which collapses runs of spaces) so a multi-line prompt keeps its
// exact whitespace; a newline is appended after each block element so rows
// stay on separate lines. Suite status/log icons are plain Unicode glyphs, so
// they survive verbatim - no material->utf step is needed (unlike the plugin).
function _domText(node) {
    let out = "";
    node.childNodes.forEach((ch) => {
        if (ch.nodeType === 3) {
            out += ch.nodeValue;
        } else if (ch.nodeType === 1) {
            // Blank line before each param group ("Visible" / "Hidden") so the
            // copied split matches the plugin's Detail (leading one is trimmed).
            if (ch.classList
                    && ch.classList.contains("nukomfy-mgr-detail-param-group-title")) {
                out += "\n";
            }
            out += _domText(ch);
            // _kvRow draws the label/value separator with CSS only (no text
            // node), so inject ": " after the label - otherwise the copy
            // reads "NK FileP:/..." instead of "NK File: P:/...". Indented
            // rows (.field-name) and param rows (" = ") already carry their
            // own separator, so only .field-label needs this.
            if (ch.classList
                    && ch.classList.contains("nukomfy-mgr-detail-field-label")) {
                out += ": ";
            }
            const tag = ch.tagName;
            if (tag === "DIV" || tag === "P" || tag === "BR" || tag === "TR") {
                out += "\n";
            }
        }
    });
    return out;
}

// Detail tab -> structured text ("--- Section ---" + body), mirroring the Nuke
// plugin's Detail Copy/Save. Skips collapsed/empty sections.
function _detailToText(detailNode) {
    const out = [];
    const header = detailNode.querySelector(".nukomfy-mgr-detail-header");
    if (header) {
        const h = _domText(header).trim();
        if (h) out.push(h, "");
    }
    detailNode.querySelectorAll(".nukomfy-mgr-detail-section").forEach((sec) => {
        const titleEl = sec.querySelector(".nukomfy-mgr-detail-section-title");
        const body = sec.querySelector(".nukomfy-mgr-detail-section-body");
        if (!body) return;
        const bodyText = _domText(body).trim();
        if (!bodyText) return;
        let title = titleEl ? titleEl.textContent : "";
        title = title.replace(/^[\s▶▼►▾]+/, "").trim();
        out.push("--- " + title + " ---", bodyText, "");
    });
    return out.join("\n").trim();
}

function openDetailModal(entry) {
    const promptId = entry.prompt_id || "";
    // Fetch the workflow_api blob in parallel with the modal opening so
    // both the Detail tab (Submitted parameters) and the Workflow API
    // tab can consume it without a second round-trip.
    let wfPayload = null;
    const wfApiPromise = nukomfyJobsHistoryWorkflowApi(promptId).then((res) => {
        if (res.status === 200 && res.body && res.body.ok
                && res.body.workflow_api) {
            wfPayload = res.body.workflow_api;
            return res.body.workflow_api;
        }
        return null;
    });
    const tabs = [
        { key: "detail", label: "Detail",
          render: () => renderDetailTab(entry, wfApiPromise) },
        { key: "log", label: "Log", render: () => renderLogTab(entry) },
        { key: "wf", label: "Workflow API",
          render: () => renderWorkflowApiTab(promptId, wfApiPromise) },
    ];
    const tabBar = el("div", { class: "nukomfy-mgr-detail-tab-strip" });
    const tabBody = el("div", { class: "nukomfy-mgr-detail-tab-body" });
    const rendered = {};
    let activeKey = "detail";
    function showTab(key) {
        for (const child of tabBar.children) {
            child.classList.toggle("nukomfy-mgr-detail-tab--active",
                                   child.dataset.key === key);
        }
        if (!rendered[key]) {
            const t = tabs.find((x) => x.key === key);
            rendered[key] = t.render();
        }
        tabBody.innerHTML = "";
        tabBody.appendChild(rendered[key]);
        activeKey = key;
    }
    for (const t of tabs) {
        const btn = el("button", {
            class: "nukomfy-mgr-detail-tab-btn",
            "data-key": t.key,
        }, t.label);
        btn.addEventListener("click", () => showTab(t.key));
        tabBar.appendChild(btn);
    }
    // Footer: Copy / Save / Close act on the ACTIVE tab, mirroring the Nuke
    // plugin's Detail dialog. Text is built faithfully - whitespace preserved
    // (raw text nodes, not innerText) and Suite icons are already portable
    // Unicode, so no material->utf step is needed. Detail/Log save as .txt,
    // Workflow API as .json (faithful big-int / float dump).
    const jobRef = entry.nfy_job_id || promptId.slice(0, 8) || "job";
    function activeTabPayload() {
        if (activeKey === "wf") {
            return { text: wfPayload ? faithfulStringify(wfPayload, 2) : "",
                     ext: "json", base: "nukomfy_workflow_" + jobRef };
        }
        const node = rendered[activeKey];
        if (activeKey === "log") {
            return { text: node ? _domText(node).trim() : "",
                     ext: "txt", base: "nukomfy_log_" + jobRef };
        }
        return { text: node ? _detailToText(node) : "",
                 ext: "txt", base: "nukomfy_detail_" + jobRef };
    }
    const copyBtn = el("button", { class: "nukomfy-mgr-btn" }, "Copy");
    const saveBtn = el("button", { class: "nukomfy-mgr-btn" }, "Save…");
    const closeBtn = el("button", { class: "nukomfy-mgr-btn" }, "Close");
    copyBtn.addEventListener("click", async () => {
        const p = activeTabPayload();
        if (!p.text) return;
        try {
            await navigator.clipboard.writeText(p.text);
            copyBtn.textContent = "Copied!";
            setTimeout(() => { copyBtn.textContent = "Copy"; }, 1500);
        } catch (e) { copyBtn.textContent = "Copy failed"; }
    });
    saveBtn.addEventListener("click", () => {
        const p = activeTabPayload();
        if (!p.text) return;
        const fname = p.base.replace(/[^A-Za-z0-9_.-]/g, "_") + "." + p.ext;
        const type = p.ext === "json" ? "application/json" : "text/plain";
        const url = URL.createObjectURL(new Blob([p.text], { type }));
        const a = el("a", { href: url, download: fname });
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
    let modalHandle = null;
    closeBtn.addEventListener("click", () => { if (modalHandle) modalHandle.close(); });
    const footer = el("div", { class: "nukomfy-mgr-detail-footer" },
                      [copyBtn, saveBtn, closeBtn]);
    const content = el("div", { class: "nukomfy-mgr-detail-modal-body" },
                       [tabBar, tabBody, footer]);
    modalHandle = openModal({
        title: `Job: ${shortPid(entry.nfy_job_id) || shortPid(promptId)}`,
        content, width: 1000, height: 750,
    });
    showTab("detail");
}

// -------------------------------------------------------------------------
// Restart polling - watch ping().boot_epoch to know when the manager has
// genuinely restarted after a POST /admin/reboot. ``os.execv`` keeps the
// PID stable, so a monotonically-increasing boot timestamp is the only
// reliable way to tell new-process apart from old-process responses.
// -------------------------------------------------------------------------
function _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function _waitForServerRestart(oldEpoch, maxSeconds = 60) {
    const startedAt = Date.now();
    let sawDown = false;
    // Brief grace so the server has time to kick off the kill.
    await _sleep(1500);
    while ((Date.now() - startedAt) / 1000 < maxSeconds) {
        const ping = await nukomfyPing();
        if (!ping) {
            sawDown = true;
        } else {
            const newEpoch = ping.boot_epoch;
            const isNewProcess =
                (typeof newEpoch === "number" && typeof oldEpoch === "number"
                    && newEpoch > oldEpoch) || sawDown;
            if (isNewProcess) return true;
        }
        await _sleep(1500);
    }
    return false;
}

// -------------------------------------------------------------------------
// Main entry point: renders the Job history sidebar section.
// -------------------------------------------------------------------------
const RANGES = {
    max_jobs_to_keep: { min: 0, max: 5000 },
    max_days_to_keep: { min: 0, max: 3650 },
};

function _validateSettings(values) {
    for (const [k, v] of Object.entries(values)) {
        const r = RANGES[k];
        if (!r) continue;
        if (!Number.isFinite(v) || v < r.min || v > r.max) {
            const label = { max_jobs_to_keep: "Maximum entries to keep",
                            max_days_to_keep: "Maximum days to keep" }[k] || k;
            return `${label} must be between ${r.min} and ${r.max}.`;
        }
    }
    return null;
}

// Static hints rendered next to each numeric setting. They describe the
// accepted range and the meaning of the 0 sentinel so the user doesn't
// have to guess.
const HINTS = {
    max_jobs_to_keep: "0 = unlimited; max 5000",
    max_days_to_keep: "0 = never expire; max 3650",
};

// Short-lived cache for an admin password that was just verified by the
// user. Used to skip the password prompt on the restart that immediately
// follows a successful toggle. 60 s window - long enough for the user to
// click "Restart now", short enough to limit the exposure if a tab is
// abandoned.
let _cachedAdminPassword = null;
let _cachedAdminPasswordExpiresAt = 0;

function _cacheAdminPassword(pwd) {
    _cachedAdminPassword = pwd;
    _cachedAdminPasswordExpiresAt = Date.now() + 60_000;
}
function _consumeCachedAdminPassword() {
    if (_cachedAdminPassword && Date.now() < _cachedAdminPasswordExpiresAt) {
        return _cachedAdminPassword;
    }
    _cachedAdminPassword = null;
    _cachedAdminPasswordExpiresAt = 0;
    return null;
}
function _clearCachedAdminPassword() {
    _cachedAdminPassword = null;
    _cachedAdminPasswordExpiresAt = 0;
}

// -------------------------------------------------------------------------
// Standalone restart entry point, exported so callers outside this module
// (e.g. the sidebar's dedicated "Restart ComfyUI" button) can reuse the
// password prompt + boot-epoch polling flow without relying on
// renderHistorySection closure state. Always opens the password modal -
// the cached-password short-circuit is reserved for the in-section
// "Restart now" banner that immediately follows a verified toggle.
// -------------------------------------------------------------------------
export function confirmAndRunRestart({
    title = "Restart ComfyUI",
    description =
        "Restart the ComfyUI process. The interface will reconnect " +
        "automatically once the restart completes.",
    actionLabel = "Restart now",
    onComplete,
} = {}) {
    openPasswordConfirmModal({
        title,
        description,
        actionLabel,
        onConfirm: async (pwd, modalCtx) => {
            // Capture the current boot_epoch BEFORE the reboot POST so
            // _waitForServerRestart can detect the new process. Falls
            // back to the sawDown heuristic if the ping fails.
            let oldEpoch = null;
            try {
                const ping = await nukomfyPing();
                if (ping && typeof ping.boot_epoch === "number") {
                    oldEpoch = ping.boot_epoch;
                }
            } catch (e) { /* sawDown fallback handles it */ }
            const res = await nukomfyReboot(pwd);
            if (res.status === 200 && res.body && res.body.ok) {
                modalCtx.setBusy("Restart in progress…");
                _clearCachedAdminPassword();
                const ok = await _waitForServerRestart(oldEpoch);
                modalCtx.close();
                if (typeof onComplete === "function") {
                    try { onComplete(ok); } catch (e) { /* swallow */ }
                }
                return;
            }
            modalCtx.showError(mapAuthError(res));
        },
    });
}

export function renderHistorySection(container, info, onChanged) {
    container.innerHTML = "";

    const hasFeature = info && info.has_persistent_history === true;
    const hasPassword = !!(info && info.has_password);
    const enabled = !!(info && info.history_enabled);
    const bootEpoch = info && typeof info.boot_epoch === "number"
        ? info.boot_epoch : null;

    container.appendChild(
        el("h4", { class: "nukomfy-mgr-section-title" }, "Job history persistence"));
    container.appendChild(el("p", { class: "nukomfy-mgr-section-description" },
        "Persistent log of jobs submitted to this machine. " +
        "When disabled, new submissions are no longer saved but " +
        "existing entries stay readable."));

    if (!hasFeature) {
        container.appendChild(el("p", { class: "nukomfy-mgr-placeholder" },
            "Not available in this manager build."));
        return;
    }

    // Toggle (iOS-style). ON = Enabled.
    const toggleClasses = ["nukomfy-toggle"];
    if (enabled) toggleClasses.push("nukomfy-toggle--on");
    if (!hasPassword) toggleClasses.push("nukomfy-toggle--disabled");
    const toggle = el("div", {
        class: toggleClasses.join(" "),
        role: "switch",
        "aria-checked": enabled ? "true" : "false",
        "aria-label": "Job history persistence",
        tabindex: hasPassword ? "0" : "-1",
        title: hasPassword
            ? "Enable / disable history persistence"
            : "Set an admin password first",
        style: "cursor: " + (hasPassword ? "pointer" : "not-allowed") + ";",
    }, [el("span", { class: "nukomfy-toggle__knob" })]);
    // State text (left) and switch (right) share one line; no action label.
    container.appendChild(el("div", {
        class: "nukomfy-mgr-row", style: "margin: 6px 0;",
    }, [
        el("span", { class: "nukomfy-mgr-status-label" }, [
            "History persistence: ",
            el("strong", {
                class: enabled
                    ? "nukomfy-mgr-status-value nukomfy-mgr-status-value--ok"
                    : "nukomfy-mgr-status-value nukomfy-mgr-status-value--err",
            }, enabled ? "Enabled" : "Disabled"),
        ]),
        toggle,
    ]));

    // Restart banner (hidden by default; shown after a successful toggle
    // change because the `enabled` flag is read at boot time).
    const restartBanner = el("div", { class: "nukomfy-mgr-restart-banner" });
    restartBanner.style.display = "none";
    container.appendChild(restartBanner);

    async function runRestartWithModal(pwd, modalCtx) {
        const res = await nukomfyReboot(pwd);
        if (res.status === 200 && res.body && res.body.ok) {
            modalCtx.setBusy("Restart in progress…");
            restartBanner.style.display = "none";
            _clearCachedAdminPassword();
            const ok = await _waitForServerRestart(bootEpoch);
            modalCtx.close();
            if (ok && typeof onChanged === "function") onChanged();
            return true;
        }
        modalCtx.showError(mapAuthError(res));
        return false;
    }

    function promptRestart() {
        const cached = _consumeCachedAdminPassword();
        if (cached) {
            // Skip the redundant password prompt - the user just verified
            // it for the toggle. Open a busy-style modal directly.
            const busy = openModal({
                title: "Nukomfy Manager - Restart ComfyUI",
                content: el("div", { class: "nukomfy-mgr-pw-modal-busy" }, [
                    el("p", { class: "nukomfy-mgr-placeholder" },
                        "Restart in progress…"),
                ]),
                width: 480,
                height: "auto",
            });
            busy.dialog.style.height = "auto";
            const ctx = {
                close: () => busy.close(),
                showError: (msg) => {
                    // If the cached password somehow no longer works, fall
                    // back to the normal prompt so the user can re-enter it.
                    busy.close();
                    _clearCachedAdminPassword();
                    openPasswordConfirmModal({
                        title: "Restart ComfyUI",
                        description: msg + " Please confirm the password again.",
                        actionLabel: "Restart now",
                        onConfirm: (pwd, ctx2) => runRestartWithModal(pwd, ctx2),
                    });
                },
                setBusy: () => { /* already busy */ },
            };
            runRestartWithModal(cached, ctx);
            return;
        }
        openPasswordConfirmModal({
            title: "Restart ComfyUI",
            description:
                "Restart the ComfyUI process so the new history setting " +
                "can take effect. The interface will reconnect automatically " +
                "once the restart completes.",
            actionLabel: "Restart now",
            onConfirm: (pwd, ctx) => runRestartWithModal(pwd, ctx),
        });
    }

    function showRestartBanner() {
        restartBanner.innerHTML = "";
        restartBanner.style.display = "flex";
        restartBanner.appendChild(el("span", {},
            "Restart ComfyUI to apply the new setting."));
        const nowBtn = el("button", {
            class: "nukomfy-mgr-btn nukomfy-mgr-btn--primary",
        }, "Restart now");
        const laterBtn = el("button", { class: "nukomfy-mgr-btn" }, "Later");
        nowBtn.addEventListener("click", promptRestart);
        laterBtn.addEventListener("click", () => {
            restartBanner.style.display = "none";
            // The user opted out of restart-now: drop the cached
            // password so a later click goes through the full prompt.
            _clearCachedAdminPassword();
        });
        restartBanner.appendChild(nowBtn);
        restartBanner.appendChild(laterBtn);
    }

    function promptToggle() {
        const desired = !toggle.classList.contains("nukomfy-toggle--on");
        const verb = desired ? "Enable" : "Disable";
        const onOff = desired ? "on" : "off";
        openPasswordConfirmModal({
            title: `${verb} history persistence`,
            description: `Toggle history persistence ${onOff}.`,
            actionLabel: verb,
            onConfirm: async (pwd, ctx) => {
                const res = await nukomfyJobsHistoryConfigSet(pwd,
                    { enabled: desired });
                if (res.status === 200 && res.body && res.body.ok) {
                    _cacheAdminPassword(pwd);
                    toggle.classList.toggle("nukomfy-toggle--on", desired);
                    ctx.close();
                    showRestartBanner();
                    return;
                }
                ctx.showError(mapAuthError(res));
            },
        });
    }

    toggle.addEventListener("click", () => {
        if (!hasPassword) return;
        promptToggle();
    });
    toggle.addEventListener("keydown", (ev) => {
        if (!hasPassword) return;
        if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            promptToggle();
        }
    });

    // Settings button - opens an auth gate, then a modal form with the
    // numeric retention values. Editing is admin-only: the same
    // password is reused for the Save POST that follows.
    const settingsBtn = el("button", { class: "nukomfy-mgr-btn" }, "Settings");
    if (!hasPassword) {
        settingsBtn.disabled = true;
        settingsBtn.title = "Set an admin password first";
    }
    container.appendChild(settingsBtn);
    settingsBtn.addEventListener("click", () => {
        if (settingsBtn.disabled) return;
        openPasswordConfirmModal({
            title: "Edit job history settings",
            description:
                "Admin password required to view and edit retention limits " +
                "for the local job log.",
            actionLabel: "Continue",
            onConfirm: async (pwd, ctx) => {
                const res = await nukomfyAuth(pwd);
                if (res.status === 200 && res.body && res.body.ok) {
                    ctx.close();
                    openSettingsModal(pwd);
                    return;
                }
                ctx.showError(mapAuthError(res));
            },
        });
    });

    function openSettingsModal(password) {
        const banner = makeBanner();
        const inputs = {
            max_jobs_to_keep: el("input", {
                type: "number", min: "0", max: "5000",
                class: "nukomfy-mgr-history-num", value: "500",
            }),
            max_days_to_keep: el("input", {
                type: "number", min: "0", max: "3650",
                class: "nukomfy-mgr-history-num", value: "90",
            }),
        };

        function buildRow(key, label) {
            return el("div", { class: "nukomfy-mgr-history-settings-row" }, [
                el("label", { class: "nukomfy-mgr-history-settings-label" }, label),
                inputs[key],
                el("span", {
                    class: "nukomfy-mgr-history-settings-hint",
                }, HINTS[key]),
            ]);
        }

        const cancelBtn = el("button", { class: "nukomfy-mgr-btn" }, "Cancel");
        const saveBtn = el("button", {
            class: "nukomfy-mgr-btn nukomfy-mgr-btn--primary",
        }, "Save");
        const body = el("div", { class: "nukomfy-mgr-pw-modal-body" }, [
            el("p", { class: "nukomfy-mgr-pw-modal-description" },
                "Numeric values take effect immediately, no restart needed."),
            banner,
            buildRow("max_jobs_to_keep", "Maximum entries to keep"),
            buildRow("max_days_to_keep", "Maximum days to keep"),
            el("div", { class: "nukomfy-mgr-actions" }, [cancelBtn, saveBtn]),
        ]);

        const modal = openModal({
            title: "Nukomfy Manager - Job history settings",
            content: body,
            width: 480,
            height: "auto",
        });
        modal.dialog.style.height = "auto";

        // Pre-populate with the live values.
        nukomfyJobsHistoryConfig().then((cfg) => {
            if (!cfg) return;
            for (const key of Object.keys(inputs)) {
                if (typeof cfg[key] === "number" || typeof cfg[key] === "string") {
                    inputs[key].value = String(cfg[key]);
                }
            }
        });

        cancelBtn.addEventListener("click", () => modal.close());
        saveBtn.addEventListener("click", async () => {
            const values = {
                max_jobs_to_keep: parseInt(inputs.max_jobs_to_keep.value, 10),
                max_days_to_keep: parseInt(inputs.max_days_to_keep.value, 10),
            };
            for (const k of Object.keys(values)) {
                if (Number.isNaN(values[k])) values[k] = RANGES[k].min;
            }
            const err = _validateSettings(values);
            if (err) {
                showBanner(banner, "err", err);
                return;
            }
            hideBanner(banner);
            saveBtn.setAttribute("disabled", "true");
            const res = await nukomfyJobsHistoryConfigSet(password, values);
            if (res.status === 200 && res.body && res.body.ok) {
                modal.close();
                return;
            }
            showBanner(banner, "err", mapAuthError(res));
            saveBtn.removeAttribute("disabled");
        });
    }

    // View history button - open without password (reads are not gated;
    // the Nukomfy plugin polls the same endpoint anonymously). The
    // destructive "Delete history" action lives inside the modal footer
    // so the sidebar surface stays focused on configuration.
    const viewBtn = el("button", { class: "nukomfy-mgr-btn" }, "View history");
    container.appendChild(viewBtn);
    viewBtn.addEventListener("click", () => {
        openHistoryListModal({ hasPassword, onChanged });
    });
}
