// Minimal fetch wrappers around the /nukomfy/* HTTP surface.

import { api } from "../../scripts/api.js";
import { parseFaithful } from "./json_faithful.js";

async function _json(r) {
    try { return await r.json(); }
    catch (e) { return null; }
}

// Integrity tampering surfaces via two channels:
//   - Admin endpoints reply with "tampered_files" after recoverable files
//     have been restored from the cache (restored: true).
//   - /ping replies with "pending_tamper_files" populated by the in-memory
//     layer (mtime/size + SHA-256 fingerprint), so the banner can appear
//     on a plain refresh without an admin password (restored: false).
// Both are forwarded to any UI surface via the same CustomEvent.
function _emitTamperedEvent(files, restored) {
    try {
        if (Array.isArray(files)) {
            window.dispatchEvent(new CustomEvent("nukomfy-mgr-tampered", {
                detail: { files, restored },
            }));
        }
    } catch (e) { /* swallow */ }
}

async function _wrapPost(url, payload) {
    try {
        const r = await api.fetchApi(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const body = await _json(r);
        if (body) _emitTamperedEvent(body.tampered_files, true);
        return { status: r.status, body };
    } catch (e) {
        console.warn(`[nukomfy.manager] ${url} failed:`, e);
        return { status: 0, body: null };
    }
}

export async function nukomfyPing() {
    try {
        const r = await api.fetchApi("/nukomfy/manager/ping");
        if (!r.ok) return null;
        const body = await r.json();
        if (body) _emitTamperedEvent(body.pending_tamper_files, false);
        return body;
    } catch (e) {
        console.warn("[nukomfy.manager] ping failed:", e);
        return null;
    }
}

export async function nukomfyAvailability() {
    try {
        const r = await api.fetchApi("/nukomfy/availability");
        if (!r.ok) return null;
        return await r.json();
    } catch (e) {
        console.warn("[nukomfy.manager] availability fetch failed:", e);
        return null;
    }
}

export async function nukomfyAuth(password) {
    return _wrapPost("/nukomfy/admin/auth", { password });
}

export async function nukomfySetPassword(newPassword, currentPassword = null) {
    const body = { new: newPassword };
    if (currentPassword !== null) body.current = currentPassword;
    return _wrapPost("/nukomfy/admin/set_password", body);
}

export async function nukomfySetAvailability(newValue, password) {
    return _wrapPost("/nukomfy/availability/set",
        { password, availability: newValue });
}

export async function nukomfyActivityTail(password, limit = 200) {
    return _wrapPost("/nukomfy/activity/tail", { password, n: limit });
}

export async function nukomfyRestoreIntegrity(password) {
    return _wrapPost("/nukomfy/admin/integrity/restore", { password });
}

export async function nukomfyActivityConfig() {
    try {
        const r = await api.fetchApi("/nukomfy/activity/config");
        if (!r.ok) return null;
        return await r.json();
    } catch (e) {
        console.warn("[nukomfy.manager] activity config fetch failed:", e);
        return null;
    }
}

export async function nukomfyActivityConfigSet(password, maxDays) {
    return _wrapPost("/nukomfy/activity/config/set",
        { password, max_days_to_keep: maxDays });
}

export async function nukomfyJobsHistory(limit = 50, offset = 0,
                                         { statuses = null, query = null, since = null } = {}) {
    const params = new URLSearchParams();
    params.set("limit", String(limit));
    if (offset) params.set("offset", String(offset));
    if (since) params.set("since", since);
    if (statuses && statuses.length) params.set("status", statuses.join(","));
    if (query) params.set("q", query);
    try {
        const r = await api.fetchApi(`/nukomfy/jobs/history?${params.toString()}`);
        if (!r.ok) return null;
        // Faithful integer parse: 64-bit seeds in nfy_seeds_used must keep
        // their exact digits (JSON.parse rounds beyond 2^53). integersOnly
        // so float fields (durations) stay real numbers for arithmetic.
        return parseFaithful(await r.text(), { integersOnly: true });
    } catch (e) {
        console.warn("[nukomfy.manager] jobs history fetch failed:", e);
        return null;
    }
}

// Raw core queue (/api/queue): the live nfy_aborting flag rides extra_data
// and is NOT persisted in the history rows, so the history viewer fetches the
// queue too to mark running jobs as "Aborting…". ComfyUI's api.getQueue()
// strips extra_data, hence the raw fetch + parse here.
export async function nukomfyLiveQueue() {
    try {
        const r = await api.fetchApi("/queue");
        if (!r.ok) return null;
        return await r.json();
    } catch (e) {
        console.warn("[nukomfy.manager] live queue fetch failed:", e);
        return null;
    }
}

export async function nukomfyJobsHistorySingle(promptId) {
    try {
        const r = await api.fetchApi(
            `/nukomfy/jobs/history/${encodeURIComponent(promptId)}`);
        return { status: r.status, body: await _json(r) };
    } catch (e) {
        console.warn("[nukomfy.manager] jobs history single failed:", e);
        return { status: 0, body: null };
    }
}

export async function nukomfyJobsHistoryWorkflowApi(promptId) {
    try {
        const r = await api.fetchApi(
            `/nukomfy/jobs/history/${encodeURIComponent(promptId)}/workflow_api`);
        // Faithful parse: the workflow_api payload must keep every numeric
        // literal exactly (64-bit seeds, floats like 1.0) for display / copy /
        // save - JSON.parse would round / normalise them.
        const text = await r.text();
        return { status: r.status, body: text ? parseFaithful(text) : null };
    } catch (e) {
        console.warn("[nukomfy.manager] jobs history workflow_api failed:", e);
        return { status: 0, body: null };
    }
}

export async function nukomfyJobsHistoryConfig() {
    try {
        const r = await api.fetchApi("/nukomfy/jobs/history/config");
        if (!r.ok) return null;
        return await r.json();
    } catch (e) {
        console.warn("[nukomfy.manager] jobs history config fetch failed:", e);
        return null;
    }
}

export async function nukomfyJobsHistoryConfigSet(password, updates) {
    return _wrapPost("/nukomfy/jobs/history/config/set",
        { password, ...updates });
}

export async function nukomfyJobsHistoryClear(password) {
    return _wrapPost("/nukomfy/jobs/history/clear", { password });
}

export async function nukomfyReboot(password) {
    return _wrapPost("/nukomfy/admin/reboot", { password });
}
