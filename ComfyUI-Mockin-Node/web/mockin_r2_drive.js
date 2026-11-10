import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DEFAULT_API_BASE = "https://api.mockin.app/api/room";
const DEFAULT_ROOM_CODE = "9999";
const DEFAULT_ROOM_ID = "R-6ZZP";
const BRIDGE_ROOM_ID = "R-LAPTOP-BRIDGE";
const LAPTOP_BRIDGE_URL = "http://127.0.0.1:18189";

const INVALID_ROOM_IDS = new Set([
    "",
    "9999",
    BRIDGE_ROOM_ID,
    "read_file_or_note",
    "sync_pull_to_local",
    "sync_push_from_local",
    "copy_on_r2",
    "move_on_r2",
    "mkdir_on_r2",
    "list_fs",
    "manifest_folder",
    "sync_download_to_local",
    "sync_upload_to_r2",
    "copy_folder_on_r2",
    "move_or_rename_on_r2",
    "delete_on_r2",
    "raw_api_request",
    "ComfyUI/models/loras",
    "ComfyUI/output",
    "ComfyUI/input",
]);

function normalizeR2Params(apiBase, code, roomId) {
    let cleanBase = String(apiBase || "").trim();
    if (!cleanBase.startsWith("http://") && !cleanBase.startsWith("https://")) {
        cleanBase = DEFAULT_API_BASE;
    }
    cleanBase = cleanBase.replace(/\/+$/, "");

    let cleanCode = String(code || "").trim();
    if (!cleanCode || cleanCode === DEFAULT_ROOM_ID || INVALID_ROOM_IDS.has(cleanCode) || cleanCode.startsWith("http")) {
        cleanCode = DEFAULT_ROOM_CODE;
    }

    let cleanRoom = String(roomId || "").trim();
    if (!cleanRoom || INVALID_ROOM_IDS.has(cleanRoom) || cleanRoom.startsWith("http") || cleanRoom.includes("/")) {
        cleanRoom = DEFAULT_ROOM_ID;
    }

    return { apiBase: cleanBase, code: cleanCode, roomId: cleanRoom };
}

function sanitizeNodeWidgets(node) {
    if (!node?.widgets) return;
    const getW = (name) => node.widgets.find((w) => w.name === name);
    const wBase = getW("api_base");
    const wCode = getW("room_code");
    const wRoom = getW("room_id");
    const norm = normalizeR2Params(wBase?.value, wCode?.value, wRoom?.value);
    if (wBase && wBase.value !== norm.apiBase) wBase.value = norm.apiBase;
    if (wCode && wCode.value !== norm.code) wCode.value = norm.code;
    if (wRoom && wRoom.value !== norm.roomId) wRoom.value = norm.roomId;
}

const ICONS = {
    cube: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>`,
    upload: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>`,
    newFolder: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/><line x1="12" y1="11" x2="12" y2="17"/><line x1="9" y1="14" x2="15" y2="14"/></svg>`,
    sync: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>`,
    dots: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/></svg>`,
    dotsH: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/></svg>`,
    folderFill: `<svg width="44" height="36" viewBox="0 0 48 38" fill="none"><path d="M4 6C4 3.79086 5.79086 2 8 2H18.3431C19.404 2 20.4214 2.42143 21.1716 3.17157L24.8284 6.82843C25.5786 7.57857 26.596 8 27.6569 8H40C42.2091 8 44 9.79086 44 12V32C44 34.2091 42.2091 36 40 36H8C5.79086 36 4 34.2091 4 32V6Z" fill="#525b66"/><path d="M4 13C4 11.3431 5.34315 10 7 10H41C42.6569 10 44 11.3431 44 13V32C44 34.2091 42.2091 36 40 36H8C5.79086 36 4 34.2091 4 32V13Z" fill="#687382"/></svg>`,
    folderOutline: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#71717a" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>`,
    fileDoc: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`,
    chevDown: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`,
};

const ENDPOINT_PRESETS = [
    { id: "list", label: "GET /list", method: "GET", path: "/list", query: {}, body: null },
    { id: "create", label: "POST /create", method: "POST", path: "/create", query: {}, body: { room_id: "{room_id}", title: "Room {room_id}" } },
    { id: "room", label: "GET /{room_id}", method: "GET", path: "/{room_id}", query: {}, body: null },
    { id: "notes", label: "POST /{room_id}/notes", method: "POST", path: "/{room_id}/notes", query: {}, body: { title: "Ghi chú", text: "Quick note" } },
    { id: "text", label: "POST /{room_id}/text", method: "POST", path: "/{room_id}/text", query: {}, body: { text: "Quick note" } },
    { id: "fs", label: "GET /{room_id}/fs", method: "GET", path: "/{room_id}/fs", query: { path: "", recursive: "false" }, body: null },
    { id: "mkdir", label: "POST /{room_id}/fs/mkdir", method: "POST", path: "/{room_id}/fs/mkdir", query: {}, body: { path: "models/loras" } },
    { id: "copy", label: "POST /{room_id}/fs/copy", method: "POST", path: "/{room_id}/fs/copy", query: {}, body: { src_path: "models", dest_path: "backup/models", dest_room_id: "{room_id}" } },
    { id: "move", label: "POST /{room_id}/fs/move", method: "POST", path: "/{room_id}/fs/move", query: {}, body: { src_path: "a.txt", dest_path: "b.txt" } },
    { id: "del", label: "DELETE /{room_id}/fs", method: "DELETE", path: "/{room_id}/fs", query: { path: "backup" }, body: null },
    { id: "manifest", label: "GET /{room_id}/fs/manifest", method: "GET", path: "/{room_id}/fs/manifest", query: { path: "" }, body: null },
];

function formatBytes(bytes) {
    const n = Number(bytes) || 0;
    if (n <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
    return (n / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1) + " " + units[i];
}

function hideAllNativeWidgets(node) {
    if (!node?.widgets) return;
    if (!NODE_KIND_MAP[node?.comfyClass || node?.type]) return;
    for (const w of node.widgets) {
        if (w.name?.startsWith("mockin_r2_")) continue;
        w.hidden = true;
        w.options = { ...(w.options || {}), hidden: true };
        w.computeSize = () => [0, -4];
        w.draw = () => {};
    }
}

function lockDomWidgetFullWidth(domWidget, node, minH = 410, minW = 600) {
    domWidget.serialize = false;
    if (domWidget.options) {
        domWidget.options.serialize = false;
        domWidget.options.getMinHeight = () => minH;
        delete domWidget.options.getMaxHeight;
        delete domWidget.options.getHeight;
    }
    try {
        delete domWidget.computeSize;
    } catch (_) {
        domWidget.computeSize = undefined;
    }
    domWidget.computeLayoutSize = () => ({
        minHeight: minH,
        maxHeight: undefined,
        minWidth: minW,
    });
    try {
        Object.defineProperty(domWidget, "width", {
            get: () => undefined,
            set: () => {},
            configurable: true,
        });
    } catch (_) {}
}

function extractRoomNotesList(roomData) {
    if (roomData && Array.isArray(roomData.notes) && roomData.notes.length > 0) {
        return roomData.notes.map((n, idx) => ({
            id: n.id || `note_${idx}`,
            title: n.title || `Note ${idx + 1}`,
            text: typeof n.text === "string" ? n.text : "",
            updated_at: n.updated_at || 0,
        }));
    }
    const rawText = String(roomData?.text || "");
    const str = rawText.trim();
    if (str.startsWith("{") && str.includes('"_mockin_notes"')) {
        try {
            const parsed = JSON.parse(str);
            if (Array.isArray(parsed._mockin_notes) && parsed._mockin_notes.length > 0) {
                return parsed._mockin_notes.map((n, idx) => ({
                    id: n.id || `note_${idx}`,
                    title: n.title || `Note ${idx + 1}`,
                    text: typeof n.text === "string" ? n.text : "",
                }));
            }
        } catch (_) {}
    }
    return [{ id: "note_init", title: "Ghi chú", text: rawText }];
}

function stopCanvasEvents(el) {
    ["keydown", "keyup", "keypress", "mousedown", "wheel", "pointerdown", "contextmenu"].forEach((evt) => {
        el.addEventListener(evt, (e) => e.stopPropagation());
    });
}

async function extractDroppedFiles(dataTransfer) {
    const out = [];
    const items = dataTransfer.items ? Array.from(dataTransfer.items) : [];
    const readEntry = async (entry, prefix = "") => {
        if (!entry) return;
        if (entry.isFile) {
            await new Promise((resolve) => {
                entry.file((f) => {
                    out.push({ file: f, relPath: prefix ? `${prefix}/${f.name}` : f.name });
                    resolve();
                }, () => resolve());
            });
        } else if (entry.isDirectory) {
            const reader = entry.createReader();
            const entries = await new Promise((resolve) => {
                reader.readEntries((res) => resolve(res || []), () => resolve([]));
            });
            for (const child of entries) {
                await readEntry(child, prefix ? `${prefix}/${entry.name}` : entry.name);
            }
        }
    };

    const webkitEntries = items.map((it) => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null)).filter(Boolean);
    if (webkitEntries.length > 0) {
        for (const entry of webkitEntries) {
            await readEntry(entry, "");
        }
        return out;
    }

    const rawFiles = Array.from(dataTransfer.files || []);
    for (const f of rawFiles) {
        out.push({ file: f, relPath: (f.webkitRelativePath || f.name).replace(/\\/g, "/") });
    }
    return out;
}

async function directR2Request(apiBase, code, roomId, method, endpointPath, query = {}, body = null) {
    const norm = normalizeR2Params(apiBase, code, roomId);
    const ep = String(endpointPath || "/list").replace(/\{room_id\}/g, encodeURIComponent(norm.roomId));
    const cleanEp = ep.startsWith("/") ? ep : `/${ep}`;

    // Always prefer backend proxy first so H200 IPv4 connection handles it reliably without browser CORS/IPv6 issues
    try {
        const r = await api.fetchApi("/mockin/r2/proxy", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                api_base: norm.apiBase,
                code: norm.code,
                room_id: norm.roomId,
                method,
                endpoint_path: cleanEp,
                query,
                body,
                save_config: true,
            }),
        });
        if (r.ok) {
            const j = await r.json();
            if (j && j.ok) return j;
        }
    } catch (_) {}

    const qs = new URLSearchParams();
    Object.entries(query || {}).forEach(([k, v]) => {
        if (v !== undefined && v !== null && String(v) !== "") qs.set(k, String(v));
    });
    const url = `${norm.apiBase}${cleanEp}${qs.toString() ? "?" + qs.toString() : ""}`;
    const t0 = performance.now();

    try {
        const headers = { "X-Room-Code": norm.code };
        const opts = { method: method.toUpperCase(), headers };
        if (body !== null && method.toUpperCase() !== "GET") {
            headers["Content-Type"] = "application/json";
            opts.body = JSON.stringify(body);
        }
        const resp = await fetch(url, opts);
        const elapsed = Math.round(performance.now() - t0);
        const data = await resp.json().catch(() => ({}));
        if (data && Array.isArray(data.rooms)) {
            data.rooms = data.rooms.filter((rm) => rm.room_id !== BRIDGE_ROOM_ID);
        }
        return { ok: resp.ok, status: resp.status, elapsed_ms: elapsed, data };
    } catch (err) {
        return { ok: false, status: 0, error: String(err), data: {} };
    }
}

async function directR2FileRead(apiBase, code, roomId, remotePath, saveLocal = true, targetPreset = "ComfyUI/models/loras") {
    const norm = normalizeR2Params(apiBase, code, roomId);
    try {
        const resp = await api.fetchApi("/mockin/r2/read_file", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                api_base: norm.apiBase,
                code: norm.code,
                room_id: norm.roomId,
                remote_path: remotePath,
                save_local: saveLocal,
                target_preset: targetPreset,
            }),
        });
        if (resp.ok) {
            return await resp.json();
        }
    } catch (_) {}

    try {
        const streamUrl = `/mockin/r2/stream_file?room_id=${encodeURIComponent(norm.roomId)}&path=${encodeURIComponent(remotePath)}&code=${encodeURIComponent(norm.code)}`;
        const r = await api.fetchApi(streamUrl);
        if (!r.ok) return { ok: false, error: `HTTP ${r.status}` };
        const blob = await r.blob();
        const ext = remotePath.split(".").pop()?.toLowerCase() || "";
        if (["png", "jpg", "jpeg", "webp", "gif"].includes(ext)) {
            const dataUrl = await new Promise((res) => {
                const fr = new FileReader();
                fr.onload = () => res(fr.result || "");
                fr.readAsDataURL(blob);
            });
            return { ok: true, size: blob.size, preview_type: "image", data_url: dataUrl, text_content: "" };
        }
        const text = await blob.text();
        return { ok: true, size: blob.size, preview_type: "text", text_content: text, data_url: "" };
    } catch (e) {
        return { ok: false, error: String(e) };
    }
}

async function uploadBrowserFileEntries(apiBase, code, roomId, baseDir, fileEntries, onProgress) {
    if (!fileEntries || fileEntries.length === 0) return { ok: false, count: 0 };
    const norm = normalizeR2Params(apiBase, code, roomId);
    const cleanBase = (baseDir || "").trim().replace(/^\/+|\/+$/g, "");
    const metaList = fileEntries.map((item) => ({
        path: item.relPath,
        size: item.file.size,
        content_type: item.file.type || "application/octet-stream",
    }));

    const presignData = await directR2Request(norm.apiBase, norm.code, norm.roomId, "POST", `/${encodeURIComponent(norm.roomId)}/files/presign`, {}, {
        base_dir: cleanBase,
        files: metaList,
    });
    const uploads = presignData?.data?.uploads || [];
    if (!presignData.ok || uploads.length === 0) {
        return { ok: false, error: presignData.error || "Presign failed", count: 0 };
    }

    const confirmed = [];
    for (let i = 0; i < fileEntries.length; i++) {
        const { file } = fileEntries[i];
        const uInfo = uploads[i];
        if (!uInfo?.upload_url) continue;
        if (onProgress) onProgress(`Uploading ${i + 1}/${fileEntries.length}: ${uInfo.name || file.name}`);
        let uploadedOk = false;
        try {
            const putResp = await fetch(uInfo.upload_url, {
                method: "PUT",
                headers: { "Content-Type": uInfo.content_type || "application/octet-stream" },
                body: file,
            });
            uploadedOk = putResp.ok;
        } catch (_) {
            uploadedOk = false;
        }
        if (!uploadedOk) {
            try {
                const fallbackResp = await api.fetchApi("/mockin/r2/upload_blob", {
                    method: "POST",
                    headers: {
                        "X-Upload-Url": uInfo.upload_url,
                        "X-Upload-Content-Type": uInfo.content_type || "application/octet-stream",
                    },
                    body: file,
                });
                const fbData = await fallbackResp.json().catch(() => ({}));
                uploadedOk = Boolean(fbData.ok);
            } catch (_) {}
        }
        if (uploadedOk) confirmed.push(uInfo);
    }

    if (confirmed.length > 0) {
        await directR2Request(norm.apiBase, norm.code, norm.roomId, "POST", `/${encodeURIComponent(norm.roomId)}/files/confirm`, {}, {
            base_dir: "",
            files: confirmed,
        });
    }
    return { ok: confirmed.length > 0, count: confirmed.length };
}

// ============================================================================
// LAPTOP SYNC BRIDGE & PROJECT STATE
// ============================================================================
const ACTIVE_PROJECT = {
    mode: "laptop", // "laptop" | "device" | "server"
    label: "Mockin_Sync (Laptop)",
    laptopPath: "",
    serverPath: "",
    dirHandle: null,
};

const _projectListeners = new Set();
function notifyProjectChanged() {
    _projectListeners.forEach((fn) => {
        try {
            fn();
        } catch (_) {}
    });
}

async function callLaptopBridge(action, payload = {}) {
    // 1. Try direct localhost Laptop Bridge first (fastest when browser allows loopback)
    try {
        const ctrl = new AbortController();
        const timeoutMs = action === "get_status" ? 1000 : 180000;
        const timer = setTimeout(() => ctrl.abort(), timeoutMs);
        const resp = await fetch(`${LAPTOP_BRIDGE_URL}/${action}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action, ...payload }),
            signal: ctrl.signal,
        });
        clearTimeout(timer);
        if (resp.ok) {
            const data = await resp.json();
            if (data) return data;
        }
    } catch (_) {}

    // 2. Fallback: Relay through H200 /mockin/r2/laptop_cmd -> R-LAPTOP-BRIDGE -> Laptop daemon
    try {
        const r = await api.fetchApi("/mockin/r2/laptop_cmd", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action, ...payload }),
        });
        if (r.ok) {
            return await r.json();
        }
    } catch (_) {}

    return { ok: false, error: "Laptop bridge unreachable" };
}

let _laptopInitStarted = false;
async function ensureLaptopProjectInitialized() {
    if (_laptopInitStarted) return;
    _laptopInitStarted = true;
    const st = await callLaptopBridge("get_status");
    if (st && st.ok && st.laptop_path) {
        ACTIVE_PROJECT.mode = "laptop";
        ACTIVE_PROJECT.laptopPath = st.laptop_path;
        ACTIVE_PROJECT.label = `${st.label || st.laptop_path.split(/[\\/]/).pop()} (Laptop)`;
        notifyProjectChanged();
    }
}

async function downloadSingleFileViaTunnel(apiBase, code, roomId, remotePath, fileName, onStatus) {
    const norm = normalizeR2Params(apiBase, code, roomId);
    if (onStatus) onStatus(`Downloading ${fileName || remotePath}...`);
    try {
        const streamUrl = `/mockin/r2/stream_file?api_base=${encodeURIComponent(norm.apiBase)}&code=${encodeURIComponent(norm.code)}&room_id=${encodeURIComponent(norm.roomId)}&path=${encodeURIComponent(remotePath)}`;
        const resp = await api.fetchApi(streamUrl);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const blob = await resp.blob();
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = blobUrl;
        a.download = fileName || remotePath.split("/").pop() || "file";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(blobUrl), 15000);
        if (onStatus) onStatus(`Saved ${fileName || remotePath}`);
        return true;
    } catch (e) {
        if (onStatus) onStatus(`Error: ${e.message || e}`);
        return false;
    }
}

async function syncOrSaveToLaptop(apiBase, code, roomId, remotePath, mode = "auto", skipExisting = true, onStatus = null) {
    const norm = normalizeR2Params(apiBase, code, roomId);
    if (onStatus) onStatus("Syncing to Laptop...");

    // 1. Primary: 1:1 Direct Folder Sync via Laptop Bridge (no zip, preserves exact folder structure!)
    const bridgeRes = await callLaptopBridge("sync_pull", {
        api_base: norm.apiBase,
        code: norm.code,
        room_id: norm.roomId,
        remote_path: remotePath || "",
        mode,
        skip_existing: skipExisting,
        laptop_path: ACTIVE_PROJECT.laptopPath || "",
    });
    if (bridgeRes && bridgeRes.ok) {
        if (bridgeRes.laptop_path) {
            ACTIVE_PROJECT.laptopPath = bridgeRes.laptop_path;
            ACTIVE_PROJECT.label = `${bridgeRes.label || bridgeRes.laptop_path.split(/[\\/]/).pop()} (Laptop)`;
        }
        const dlCount = bridgeRes.downloaded_count ?? 0;
        const skCount = bridgeRes.skipped_count ?? 0;
        if (onStatus) {
            onStatus(
                mode === "file"
                    ? `Saved -> ${ACTIVE_PROJECT.label}`
                    : `Synced -> Laptop (${dlCount} new, ${skCount} ok)`
            );
        }
        return bridgeRes;
    }

    // 2. Secondary: If user picked a browser FileSystemDirectoryHandle on localhost/https
    if (ACTIVE_PROJECT.mode === "device" && ACTIVE_PROJECT.dirHandle) {
        const mfRes = await directR2Request(norm.apiBase, norm.code, norm.roomId, "GET", `/${encodeURIComponent(norm.roomId)}/fs/manifest`, {
            path: remotePath || "",
        });
        const files = mfRes?.data?.files || [];
        let count = 0;
        for (const f of files) {
            const rel = (f.relative_path || f.name || f.path).replace(/^\/+/, "");
            const parts = rel.split("/").filter(Boolean);
            const fName = parts.pop();
            let curr = ACTIVE_PROJECT.dirHandle;
            for (const p of parts) {
                curr = await curr.getDirectoryHandle(p, { create: true });
            }
            if (onStatus) onStatus(`Syncing ${count + 1}/${files.length}: ${fName}`);
            const streamUrl = `/mockin/r2/stream_file?api_base=${encodeURIComponent(norm.apiBase)}&code=${encodeURIComponent(norm.code)}&room_id=${encodeURIComponent(norm.roomId)}&path=${encodeURIComponent(f.path)}`;
            const r = await api.fetchApi(streamUrl);
            const blob = await r.blob();
            const fh = await curr.getFileHandle(fName, { create: true });
            const wr = await fh.createWritable();
            await wr.write(blob);
            await wr.close();
            count++;
        }
        if (onStatus) onStatus(`Synced ${count} files -> ${ACTIVE_PROJECT.dirHandle.name}`);
        return { ok: true, downloaded_count: count, skipped_count: 0 };
    }

    // 3. Fallback: Same-origin tunnel stream download (never hits external r2.cloudflarestorage.com in browser)
    if (mode === "file" && remotePath) {
        const ok = await downloadSingleFileViaTunnel(norm.apiBase, norm.code, norm.roomId, remotePath, remotePath.split("/").pop(), onStatus);
        return { ok, downloaded_count: ok ? 1 : 0, skipped_count: 0 };
    }

    const mfRes = await directR2Request(norm.apiBase, norm.code, norm.roomId, "GET", `/${encodeURIComponent(norm.roomId)}/fs/manifest`, {
        path: remotePath || "",
    });
    const files = mfRes?.data?.files || [];
    for (let i = 0; i < files.length; i++) {
        const f = files[i];
        await downloadSingleFileViaTunnel(norm.apiBase, norm.code, norm.roomId, f.path, f.name || f.path.split("/").pop(), (m) => {
            if (onStatus) onStatus(`[${i + 1}/${files.length}] ${m}`);
        });
    }
    if (onStatus) onStatus(`Downloaded ${files.length} files`);
    return { ok: true, downloaded_count: files.length, skipped_count: 0 };
}

async function pickFolderOnLaptopFlow(onStatus) {
    if (onStatus) onStatus("Opening Laptop Folder Picker...");
    const res = await callLaptopBridge("pick_folder");
    if (res && res.ok && res.laptop_path) {
        ACTIVE_PROJECT.mode = "laptop";
        ACTIVE_PROJECT.laptopPath = res.laptop_path;
        ACTIVE_PROJECT.dirHandle = null;
        ACTIVE_PROJECT.label = `${res.label || res.laptop_path.split(/[\\/]/).pop()} (Laptop)`;
        if (onStatus) onStatus(`Project: ${res.laptop_path}`);
        notifyProjectChanged();
        return true;
    }
    if (res && res.cancelled) {
        if (onStatus) onStatus("Ready");
        return false;
    }
    if (window.showDirectoryPicker) {
        try {
            const dh = await window.showDirectoryPicker({ mode: "readwrite" });
            ACTIVE_PROJECT.mode = "device";
            ACTIVE_PROJECT.dirHandle = dh;
            ACTIVE_PROJECT.label = `${dh.name} (Laptop)`;
            if (onStatus) onStatus(`Project: ${dh.name}`);
            notifyProjectChanged();
            return true;
        } catch (_) {
            if (onStatus) onStatus("Ready");
            return false;
        }
    }
    const manual = prompt("Nhập đường dẫn thư mục trên Laptop (Máy con):", ACTIVE_PROJECT.laptopPath || "C:\\Users\\EXT-013\\Downloads\\Mockin_Sync");
    if (manual && manual.trim()) {
        const setRes = await callLaptopBridge("set_folder", { laptop_path: manual.trim() });
        ACTIVE_PROJECT.mode = "laptop";
        ACTIVE_PROJECT.laptopPath = setRes?.laptop_path || manual.trim();
        ACTIVE_PROJECT.label = `${ACTIVE_PROJECT.laptopPath.split(/[\\/]/).pop()} (Laptop)`;
        if (onStatus) onStatus(`Project: ${ACTIVE_PROJECT.laptopPath}`);
        notifyProjectChanged();
        return true;
    }
    if (onStatus) onStatus("Ready");
    return false;
}

// ============================================================================
// 1. MAIN NODE: MockinR2CloudDriveNode ("Mockin R2 Cloud Drive & Endpoint Explorer")
// ============================================================================
function setupMainDriveNode(node) {
    if (node.__mockinR2MainSetup) return;
    node.__mockinR2MainSetup = true;

    const getW = (name) => node.widgets?.find((w) => w.name === name);
    const setW = (name, val) => {
        const w = getW(name);
        if (w) w.value = val;
    };

    sanitizeNodeWidgets(node);
    hideAllNativeWidgets(node);
    ensureLaptopProjectInitialized();

    let activeTab = "files";
    let currentPath = "";
    let rooms = [];
    let folders = [];
    let files = [];
    let notesList = [{ id: "note_init", title: "Ghi chú", text: "" }];
    let activeNoteIdx = 0;
    let initialNotesLoaded = false;
    let noteSaveTimer = null;
    let noteStatusEl = null;
    let selectedItem = null;
    let statusMsg = "Ready";
    let statusOk = true;
    let showSettings = false;
    let contextMenuEl = null;

    // API Reviewer state
    let apiPreset = ENDPOINT_PRESETS[0].id;
    let apiMethod = "GET";
    let apiPath = "/list";
    let apiBody = "";
    let apiOutput = "";

    const cfg = () => {
        sanitizeNodeWidgets(node);
        const norm = normalizeR2Params(getW("api_base")?.value, getW("room_code")?.value, getW("room_id")?.value);
        return {
            apiBase: norm.apiBase,
            code: norm.code,
            roomId: norm.roomId,
            remotePath: getW("remote_path")?.value || "",
            destPath: getW("dest_or_target_path")?.value || "",
            preset: getW("target_local_preset")?.value || "ComfyUI/models/loras",
        };
    };

    async function proxyReq(method, endpointPath, query = {}, body = null) {
        const c = cfg();
        const res = await directR2Request(c.apiBase, c.code, c.roomId, method, endpointPath, query, body);
        statusOk = Boolean(res.ok);
        statusMsg = res.ok ? `${res.status || 200} · ${res.elapsed_ms || 0}ms` : "Error";
        return res;
    }

    async function saveNoteTabToR2(noteObj, reRender = false) {
        const c = cfg();
        const activeText = notesList[activeNoteIdx]?.text || "";
        setW("selected_content", activeText);
        if (noteStatusEl) {
            noteStatusEl.textContent = "Syncing...";
            noteStatusEl.style.color = "#38bdf8";
        }
        const payload = {
            title: noteObj.title || "Ghi chú",
            text: noteObj.text || "",
        };
        if (noteObj.id && !String(noteObj.id).startsWith("local_")) {
            payload.id = noteObj.id;
        }
        const res = await proxyReq("POST", `/${c.roomId}/notes`, {}, payload);
        if (res.ok && res.data) {
            if (res.data.note?.id) {
                noteObj.id = res.data.note.id;
            }
            if (Array.isArray(res.data.notes) && res.data.notes.length > 0 && reRender) {
                const targetId = res.data.note?.id || noteObj.id;
                notesList = extractRoomNotesList(res.data);
                const foundIdx = notesList.findIndex((n) => n.id === targetId);
                if (foundIdx >= 0) activeNoteIdx = foundIdx;
            }
        }
        if (noteStatusEl) {
            noteStatusEl.textContent = res.ok ? "Auto-saved & Synced ✓" : "Sync failed";
            noteStatusEl.style.color = res.ok ? "#4ade80" : "#f87171";
        }
        if (reRender) render();
    }

    function scheduleAutoSaveNote(immediate = false) {
        const curNote = notesList[activeNoteIdx];
        if (!curNote) return;
        setW("selected_content", curNote.text || "");
        if (noteSaveTimer) clearTimeout(noteSaveTimer);
        if (immediate) {
            saveNoteTabToR2(curNote, false);
        } else {
            if (noteStatusEl) {
                noteStatusEl.textContent = "Saving...";
                noteStatusEl.style.color = "#a1a1aa";
            }
            noteSaveTimer = setTimeout(() => saveNoteTabToR2(curNote, false), 350);
        }
    }

    async function loadRoomsAndFS(targetPath = null) {
        if (targetPath !== null) currentPath = targetPath;
        const c = cfg();
        const [rList, rFs] = await Promise.all([
            proxyReq("GET", "/list"),
            proxyReq("GET", `/${c.roomId}/fs`, { path: currentPath, recursive: "false" }),
        ]);
        if (rList.ok && Array.isArray(rList.data?.rooms)) {
            rooms = rList.data.rooms.filter((rm) => rm.room_id && rm.room_id !== BRIDGE_ROOM_ID);
        }
        if (rFs.ok && rFs.data) {
            folders = rFs.data.folders || [];
            files = rFs.data.files || [];
            const prevActiveId = notesList[activeNoteIdx]?.id;
            notesList = extractRoomNotesList(rFs.data);
            if (!initialNotesLoaded) {
                initialNotesLoaded = true;
                const nonEmptyIdx = notesList.findIndex((n) => (n.text || "").trim().length > 0);
                activeNoteIdx = nonEmptyIdx >= 0 ? nonEmptyIdx : 0;
            } else if (prevActiveId) {
                const idx = notesList.findIndex((n) => n.id === prevActiveId);
                activeNoteIdx = idx >= 0 ? idx : Math.min(activeNoteIdx, notesList.length - 1);
            } else {
                activeNoteIdx = Math.min(activeNoteIdx, notesList.length - 1);
            }
            if (!selectedItem) {
                setW("selected_content", notesList[activeNoteIdx]?.text || "");
            }
        }
        render();
    }

    async function inspectFile(fileObj) {
        const c = cfg();
        selectedItem = {
            isFolder: false,
            name: fileObj.name || fileObj.path.split("/").pop(),
            path: fileObj.path,
            size: fileObj.size || 0,
            type: fileObj.path.split(".").pop()?.toLowerCase() || "file",
            previewText: "Loading...",
            dataUrl: "",
            localPath: "",
        };
        setW("remote_path", fileObj.path);
        render();

        const data = await directR2FileRead(c.apiBase, c.code, c.roomId, fileObj.path, true, c.preset);
        if (data.ok) {
            selectedItem.previewText = data.text_content || "";
            selectedItem.dataUrl = data.data_url || "";
            selectedItem.localPath = data.local_path || "";
            selectedItem.type = data.preview_type || selectedItem.type;
            setW("selected_content", data.text_content || data.local_path || "");
            statusMsg = `Loaded ${selectedItem.name}`;
            statusOk = true;
        } else {
            selectedItem.previewText = data.error || "Cannot preview";
        }
        render();
    }

    function closeContextMenu() {
        if (contextMenuEl) {
            contextMenuEl.remove();
            contextMenuEl = null;
        }
    }

    function openItemMenu(anchorEl, item, isFolder) {
        closeContextMenu();
        const c = cfg();
        const menu = document.createElement("div");
        menu.style.cssText = `
            position: fixed;
            z-index: 99999;
            background: #1c1c21;
            border: 1px solid #2e2e36;
            border-radius: 6px;
            padding: 4px;
            min-width: 148px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.65);
            display: flex;
            flex-direction: column;
            gap: 2px;
            font-family: system-ui, -apple-system, sans-serif;
            font-size: 11px;
        `;
        const rect = anchorEl.getBoundingClientRect();
        menu.style.left = `${Math.min(window.innerWidth - 160, rect.left)}px`;
        menu.style.top = `${rect.bottom + 4}px`;

        const addMenuRow = (label, color, fn) => {
            const btn = document.createElement("button");
            btn.textContent = label;
            btn.style.cssText = `
                background: transparent;
                color: ${color};
                border: none;
                border-radius: 4px;
                padding: 5px 8px;
                text-align: left;
                cursor: pointer;
                font-size: 11px;
            `;
            btn.onmouseenter = () => (btn.style.background = "#27272f");
            btn.onmouseleave = () => (btn.style.background = "transparent");
            btn.onclick = async (e) => {
                e.stopPropagation();
                closeContextMenu();
                await fn();
            };
            menu.appendChild(btn);
        };

        if (isFolder) {
            addMenuRow("Open", "#f4f4f5", () => loadRoomsAndFS(item.path));
            addMenuRow("Sync to Laptop (1:1)", "#38bdf8", async () => {
                await syncOrSaveToLaptop(c.apiBase, c.code, c.roomId, item.path, "folder", true, (m) => {
                    statusOk = true;
                    statusMsg = m;
                    render();
                });
            });
            addMenuRow("Save to Laptop", "#f4f4f5", async () => {
                await syncOrSaveToLaptop(c.apiBase, c.code, c.roomId, item.path, "folder", false, (m) => {
                    statusOk = true;
                    statusMsg = m;
                    render();
                });
            });
            addMenuRow("Sync to H200 Server", "#a1a1aa", async () => {
                statusMsg = `Syncing ${item.name} -> H200...`;
                render();
                const r = await api.fetchApi("/mockin/r2/download_sync", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        api_base: c.apiBase,
                        code: c.code,
                        room_id: c.roomId,
                        remote_path: item.path,
                        mode: "folder",
                        skip_existing: true,
                        target_preset: c.preset,
                    }),
                });
                const d = await r.json();
                statusOk = Boolean(d.ok);
                statusMsg = d.ok ? `H200: ${d.downloaded_count} synced` : "Sync error";
                render();
            });
        } else {
            addMenuRow("Preview / Read", "#38bdf8", () => inspectFile(item));
            addMenuRow("Save to Laptop", "#f4f4f5", async () => {
                await syncOrSaveToLaptop(c.apiBase, c.code, c.roomId, item.path, "file", false, (m) => {
                    statusOk = true;
                    statusMsg = m;
                    render();
                });
            });
            addMenuRow("Browser Download", "#a1a1aa", async () => {
                await downloadSingleFileViaTunnel(c.apiBase, c.code, c.roomId, item.path, item.name || "file", (m) => {
                    statusOk = true;
                    statusMsg = m;
                    render();
                });
            });
        }

        addMenuRow("Copy on R2...", "#f4f4f5", async () => {
            const dest = prompt("Copy to R2 path:", `${item.path}_copy`);
            if (!dest) return;
            await proxyReq("POST", `/${c.roomId}/fs/copy`, {}, {
                src_path: item.path,
                dest_path: dest.trim(),
                dest_room_id: c.roomId,
            });
            await loadRoomsAndFS(currentPath);
        });

        addMenuRow("Rename / Move...", "#f4f4f5", async () => {
            const dest = prompt("Move / rename to:", item.path);
            if (!dest || dest === item.path) return;
            await proxyReq("POST", `/${c.roomId}/fs/move`, {}, {
                src_path: item.path,
                dest_path: dest.trim(),
            });
            await loadRoomsAndFS(currentPath);
        });

        addMenuRow("Delete", "#f87171", async () => {
            if (!confirm(`Delete '${item.path}'?`)) return;
            await proxyReq("DELETE", `/${c.roomId}/fs`, { path: item.path });
            if (selectedItem?.path === item.path) selectedItem = null;
            await loadRoomsAndFS(currentPath);
        });

        document.body.appendChild(menu);
        contextMenuEl = menu;
        setTimeout(() => {
            window.addEventListener("click", closeContextMenu, { once: true });
        }, 10);
    }

    const root = document.createElement("div");
    root.style.cssText = `
        background: #141417;
        border: 1px solid #27272c;
        border-radius: 10px;
        color: #f4f4f5;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
        font-size: 11.5px;
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100%;
        min-height: 410px;
        box-sizing: border-box;
        overflow: hidden;
        user-select: none;
    `;
    stopCanvasEvents(root);

    const hiddenFileInput = document.createElement("input");
    hiddenFileInput.type = "file";
    hiddenFileInput.multiple = true;
    hiddenFileInput.style.display = "none";
    hiddenFileInput.onchange = async () => {
        const c = cfg();
        const entries = Array.from(hiddenFileInput.files || []).map((f) => ({ file: f, relPath: f.name }));
        await uploadBrowserFileEntries(c.apiBase, c.code, c.roomId, currentPath, entries, (m) => {
            statusMsg = m;
            render();
        });
        hiddenFileInput.value = "";
        await loadRoomsAndFS(currentPath);
    };

    function render() {
        const c = cfg();
        root.innerHTML = "";
        root.appendChild(hiddenFileInput);

        // 1. HEADER BAR
        const header = document.createElement("div");
        header.style.cssText = `
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-bottom: 1px solid #222228;
            background: #16161a;
            flex-shrink: 0;
        `;

        // Room Pill Selector
        const roomPill = document.createElement("div");
        roomPill.style.cssText = `
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 6px;
            padding: 3px 8px;
            flex-shrink: 0;
        `;
        roomPill.innerHTML = ICONS.cube;

        const roomSel = document.createElement("select");
        roomSel.style.cssText = `
            background: transparent;
            color: #38bdf8;
            border: none;
            outline: none;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
        `;
        const roomIds = Array.from(
            new Set([c.roomId, ...rooms.map((r) => r.room_id).filter((id) => id && id !== BRIDGE_ROOM_ID)])
        );
        roomIds.forEach((rid) => {
            const opt = document.createElement("option");
            opt.value = rid;
            opt.textContent = rid;
            opt.style.background = "#18181c";
            opt.style.color = "#f4f4f5";
            if (rid === c.roomId) opt.selected = true;
            roomSel.appendChild(opt);
        });
        roomSel.onchange = () => {
            setW("room_id", roomSel.value);
            selectedItem = null;
            initialNotesLoaded = false;
            loadRoomsAndFS("");
        };
        roomPill.appendChild(roomSel);
        header.appendChild(roomPill);

        // Middle Header: Note Tabs (+ [1] [2] [3]...) when on Note tab, or Breadcrumbs when on Files/API
        const midBar = document.createElement("div");
        midBar.style.cssText = "display:flex; align-items:center; gap:5px; overflow-x:auto; white-space:nowrap; flex:1; min-width:0; scrollbar-width:none;";

        if (activeTab === "note") {
            const addNoteBtn = document.createElement("button");
            addNoteBtn.textContent = "+";
            addNoteBtn.title = "New Note Tab";
            addNoteBtn.style.cssText = `
                background: #232329;
                color: #38bdf8;
                border: 1px solid #32323c;
                border-radius: 6px;
                width: 24px;
                height: 24px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                flex-shrink: 0;
            `;
            addNoteBtn.onclick = async () => {
                const res = await proxyReq("POST", `/${c.roomId}/notes`, {}, {
                    title: `Ghi chú ${notesList.length + 1}`,
                    text: "",
                });
                if (res.ok && Array.isArray(res.data?.notes)) {
                    notesList = extractRoomNotesList(res.data);
                    const newId = res.data.note?.id;
                    const idx = newId ? notesList.findIndex((n) => n.id === newId) : 0;
                    activeNoteIdx = idx >= 0 ? idx : 0;
                } else {
                    notesList.push({ id: `local_${Date.now()}`, title: `Ghi chú ${notesList.length + 1}`, text: "" });
                    activeNoteIdx = notesList.length - 1;
                }
                setW("selected_content", notesList[activeNoteIdx]?.text || "");
                render();
            };
            midBar.appendChild(addNoteBtn);

            notesList.forEach((nt, idx) => {
                const isAct = idx === activeNoteIdx;
                const pill = document.createElement("div");
                pill.title = "Click to switch · Double-click to rename";
                pill.style.cssText = `
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    background: ${isAct ? "#25252e" : "#18181d"};
                    border: 1px solid ${isAct ? "#38bdf8" : "#27272f"};
                    color: ${isAct ? "#f4f4f5" : "#8e8e99"};
                    border-radius: 6px;
                    padding: 3px 8px;
                    font-size: 11px;
                    font-weight: ${isAct ? "600" : "400"};
                    cursor: pointer;
                    flex-shrink: 0;
                `;
                const snippet = (nt.text || "").trim().split("\n")[0].slice(0, 14);
                const lbl = document.createElement("span");
                lbl.textContent = nt.title && nt.title !== "Ghi chú" ? nt.title : snippet || nt.title || `Note ${idx + 1}`;
                pill.onclick = () => {
                    activeNoteIdx = idx;
                    setW("selected_content", notesList[activeNoteIdx]?.text || "");
                    render();
                };
                pill.ondblclick = async (e) => {
                    e.stopPropagation();
                    const newTitle = prompt("Note title:", nt.title || `Note ${idx + 1}`);
                    if (newTitle === null || !newTitle.trim()) return;
                    nt.title = newTitle.trim();
                    await saveNoteTabToR2(nt, true);
                };

                const delX = document.createElement("span");
                delX.textContent = "×";
                delX.title = "Delete note";
                delX.style.cssText = "color:#71717a; font-size:12px; font-weight:700; padding:0 2px; border-radius:3px;";
                delX.onmouseenter = () => (delX.style.color = "#f87171");
                delX.onmouseleave = () => (delX.style.color = "#71717a");
                delX.onclick = async (e) => {
                    e.stopPropagation();
                    if (nt.id && !String(nt.id).startsWith("local_")) {
                        const res = await proxyReq("DELETE", `/${c.roomId}/notes/${encodeURIComponent(nt.id)}`);
                        if (res.ok && Array.isArray(res.data?.notes)) {
                            notesList = extractRoomNotesList(res.data);
                            activeNoteIdx = Math.max(0, Math.min(activeNoteIdx, notesList.length - 1));
                            setW("selected_content", notesList[activeNoteIdx]?.text || "");
                            render();
                            return;
                        }
                    }
                    if (notesList.length <= 1) {
                        notesList[0].text = "";
                        activeNoteIdx = 0;
                    } else {
                        notesList.splice(idx, 1);
                        if (activeNoteIdx >= notesList.length) activeNoteIdx = notesList.length - 1;
                    }
                    setW("selected_content", notesList[activeNoteIdx]?.text || "");
                    render();
                };

                pill.appendChild(lbl);
                pill.appendChild(delX);
                midBar.appendChild(pill);
            });
        } else {
            const rootCrumb = document.createElement("span");
            rootCrumb.textContent = "/";
            rootCrumb.style.cssText = `cursor:pointer; padding:2px 4px; border-radius:4px; color:${!currentPath ? "#f4f4f5" : "#71717a"};`;
            rootCrumb.onclick = () => loadRoomsAndFS("");
            midBar.appendChild(rootCrumb);

            const parts = currentPath.split("/").filter(Boolean);
            parts.forEach((p, idx) => {
                const sep = document.createElement("span");
                sep.textContent = "/";
                sep.style.color = "#3f3f46";
                if (idx > 0) midBar.appendChild(sep);

                const sp = document.createElement("span");
                sp.textContent = p;
                const isLast = idx === parts.length - 1;
                sp.style.cssText = `cursor:pointer; padding:2px 4px; border-radius:4px; color:${isLast ? "#f4f4f5" : "#71717a"}; font-weight:${isLast ? "600" : "400"};`;
                const target = parts.slice(0, idx + 1).join("/");
                sp.onclick = () => loadRoomsAndFS(target);
                midBar.appendChild(sp);
            });
        }
        header.appendChild(midBar);

        // Segmented Tabs (Files | Note | API)
        const seg = document.createElement("div");
        seg.style.cssText = "display:flex; background:#101013; border:1px solid #25252b; border-radius:7px; padding:2px; gap:2px; flex-shrink:0;";
        [
            { id: "files", label: "Files" },
            { id: "note", label: "Note" },
            { id: "api", label: "API" },
        ].forEach((t) => {
            const btn = document.createElement("button");
            const active = activeTab === t.id;
            btn.textContent = t.label;
            btn.style.cssText = `
                background: ${active ? "#232329" : "transparent"};
                color: ${active ? "#f4f4f5" : "#71717a"};
                border: none;
                border-radius: 5px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 500;
                cursor: pointer;
            `;
            btn.onclick = () => {
                activeTab = t.id;
                render();
            };
            seg.appendChild(btn);
        });
        header.appendChild(seg);

        // Right Toolbar Icon Buttons (Upload, New Folder, Sync, Settings)
        const toolWrap = document.createElement("div");
        toolWrap.style.cssText = "display:flex; align-items:center; gap:6px; margin-left:4px; flex-shrink:0;";

        const makeToolBtn = (svg, label, onClick) => {
            const b = document.createElement("button");
            b.style.cssText = `
                background: transparent;
                color: #a1a1aa;
                border: none;
                border-radius: 5px;
                padding: 2px 5px;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 1px;
                cursor: pointer;
                font-size: 9px;
            `;
            b.innerHTML = `${svg}<span>${label}</span>`;
            b.onmouseenter = () => (b.style.color = "#f4f4f5");
            b.onmouseleave = () => (b.style.color = "#a1a1aa");
            b.onclick = onClick;
            return b;
        };

        toolWrap.appendChild(makeToolBtn(ICONS.upload, "Upload", () => hiddenFileInput.click()));
        toolWrap.appendChild(
            makeToolBtn(ICONS.newFolder, "New Folder", async () => {
                const prefix = currentPath ? `${currentPath}/` : "";
                const name = prompt("New folder path:", `${prefix}folder`);
                if (!name?.trim()) return;
                await proxyReq("POST", `/${c.roomId}/fs/mkdir`, {}, { path: name.trim() });
                await loadRoomsAndFS(currentPath);
            })
        );
        toolWrap.appendChild(makeToolBtn(ICONS.sync, "Refresh", () => loadRoomsAndFS(currentPath)));

        const moreBtn = document.createElement("button");
        moreBtn.innerHTML = ICONS.dots;
        moreBtn.style.cssText = "background:transparent; color:#a1a1aa; border:none; padding:4px; cursor:pointer; border-radius:4px;";
        moreBtn.onclick = () => {
            showSettings = !showSettings;
            render();
        };
        toolWrap.appendChild(moreBtn);

        header.appendChild(toolWrap);
        root.appendChild(header);

        // Optional compact settings drawer (for API Base, Passcode, Create Room)
        if (showSettings) {
            const cfgBar = document.createElement("div");
            cfgBar.style.cssText = "display:flex; gap:6px; align-items:center; padding:6px 12px; background:#19191f; border-bottom:1px solid #24242b; font-size:10.5px; flex-shrink:0;";

            const baseIn = document.createElement("input");
            baseIn.value = c.apiBase;
            baseIn.placeholder = "API Base";
            baseIn.style.cssText = "flex:1; background:#111115; color:#d4d4d8; border:1px solid #2b2b33; border-radius:4px; padding:3px 6px; font-size:10.5px;";
            baseIn.onchange = () => setW("api_base", baseIn.value.trim());

            const codeIn = document.createElement("input");
            codeIn.value = c.code;
            codeIn.placeholder = "Code";
            codeIn.style.cssText = "width:54px; background:#111115; color:#d4d4d8; border:1px solid #2b2b33; border-radius:4px; padding:3px 6px; font-size:10.5px; text-align:center;";
            codeIn.onchange = () => setW("room_code", codeIn.value.trim());

            const newRmBtn = document.createElement("button");
            newRmBtn.textContent = "+ Room";
            newRmBtn.style.cssText = "background:#27272f; color:#f4f4f5; border:1px solid #363640; border-radius:4px; padding:3px 8px; cursor:pointer; font-size:10.5px;";
            newRmBtn.onclick = async () => {
                const id = prompt("Room ID (blank = auto):", "");
                if (id === null) return;
                const r = await proxyReq("POST", "/create", {}, id.trim() ? { room_id: id.trim() } : {});
                if (r.ok && r.data?.room_id) {
                    setW("room_id", r.data.room_id);
                    await loadRoomsAndFS("");
                }
            };

            cfgBar.appendChild(baseIn);
            cfgBar.appendChild(codeIn);
            cfgBar.appendChild(newRmBtn);
            root.appendChild(cfgBar);
        }

        // 2. MAIN WORKSPACE AREA
        const body = document.createElement("div");
        body.style.cssText = "display:flex; flex:1; min-height:0; overflow:hidden;";

        if (activeTab === "files") {
            // Left Pane: Grid + Dropzone
            const leftPane = document.createElement("div");
            leftPane.style.cssText = "flex:1; display:flex; flex-direction:column; padding:10px 12px; gap:10px; overflow-y:auto; min-width:0;";

            // Grid of Folders & Files
            const grid = document.createElement("div");
            grid.style.cssText = "display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap:8px;";

            // Render Folders
            folders.forEach((fd) => {
                const fCard = document.createElement("div");
                const isSel = selectedItem?.isFolder && selectedItem?.path === fd.path;
                fCard.style.cssText = `
                    background: ${isSel ? "#22222a" : "#18181c"};
                    border: 1px solid ${isSel ? "#38bdf8" : "#25252c"};
                    border-radius: 8px;
                    padding: 8px 10px;
                    display: flex;
                    align-items: center;
                    gap: 9px;
                    cursor: pointer;
                    position: relative;
                    transition: background 0.12s;
                `;
                fCard.onmouseenter = () => {
                    if (!isSel) fCard.style.background = "#1e1e24";
                };
                fCard.onmouseleave = () => {
                    if (!isSel) fCard.style.background = "#18181c";
                };
                fCard.onclick = () => {
                    selectedItem = {
                        isFolder: true,
                        name: fd.name || fd.path.split("/").pop(),
                        path: fd.path,
                        size: fd.total_size || 0,
                        fileCount: fd.file_count || 0,
                        type: "folder",
                    };
                    setW("remote_path", fd.path);
                    render();
                };
                fCard.ondblclick = () => loadRoomsAndFS(fd.path);

                const iconBox = document.createElement("div");
                iconBox.innerHTML = ICONS.folderFill;
                iconBox.style.cssText = "flex-shrink:0; display:flex; align-items:center;";

                const infoBox = document.createElement("div");
                infoBox.style.cssText = "flex:1; min-width:0; display:flex; flex-direction:column; gap:2px;";
                const titleEl = document.createElement("div");
                titleEl.textContent = fd.name || fd.path.split("/").pop();
                titleEl.style.cssText = "color:#f4f4f5; font-weight:500; font-size:11.5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;";
                const subEl = document.createElement("div");
                subEl.textContent = `${fd.file_count || 0} items · ${formatBytes(fd.total_size || 0)}`;
                subEl.style.cssText = "color:#71717a; font-size:10px;";
                infoBox.appendChild(titleEl);
                infoBox.appendChild(subEl);

                const menuBtn = document.createElement("button");
                menuBtn.innerHTML = ICONS.dotsH;
                menuBtn.style.cssText = "background:transparent; color:#71717a; border:none; cursor:pointer; padding:2px;";
                menuBtn.onclick = (e) => {
                    e.stopPropagation();
                    openItemMenu(menuBtn, fd, true);
                };

                fCard.appendChild(iconBox);
                fCard.appendChild(infoBox);
                fCard.appendChild(menuBtn);
                grid.appendChild(fCard);
            });

            // Render Files
            files.forEach((fl) => {
                const card = document.createElement("div");
                const isSel = !selectedItem?.isFolder && selectedItem?.path === fl.path;
                card.style.cssText = `
                    background: ${isSel ? "#22222a" : "#1c1c21"};
                    border: 1px solid ${isSel ? "#38bdf8" : "#27272f"};
                    border-radius: 8px;
                    padding: 8px 10px;
                    display: flex;
                    align-items: center;
                    gap: 9px;
                    cursor: pointer;
                    position: relative;
                `;
                card.onmouseenter = () => {
                    if (!isSel) card.style.background = "#222228";
                };
                card.onmouseleave = () => {
                    if (!isSel) card.style.background = "#1c1c21";
                };
                card.onclick = () => inspectFile(fl);

                const thumb = document.createElement("div");
                thumb.style.cssText = `
                    width: 36px;
                    height: 36px;
                    border-radius: 6px;
                    background: #121216;
                    border: 1px solid #27272f;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-shrink: 0;
                    overflow: hidden;
                `;
                const ext = (fl.name || fl.path).split(".").pop()?.toLowerCase() || "";
                if (["png", "jpg", "jpeg", "webp", "gif"].includes(ext)) {
                    const img = document.createElement("img");
                    img.src = `/api/mockin/r2/stream_file?room_id=${encodeURIComponent(c.roomId)}&path=${encodeURIComponent(fl.path)}&code=${encodeURIComponent(c.code)}&inline=1`;
                    img.onerror = () => {
                        img.onerror = null;
                        img.src = `${c.apiBase}/${encodeURIComponent(c.roomId)}/fs/download?path=${encodeURIComponent(fl.path)}&code=${encodeURIComponent(c.code)}`;
                    };
                    img.style.cssText = "width:100%; height:100%; object-fit:cover;";
                    thumb.appendChild(img);
                } else {
                    thumb.innerHTML = ICONS.fileDoc;
                }

                const meta = document.createElement("div");
                meta.style.cssText = "flex:1; min-width:0; display:flex; flex-direction:column; gap:3px;";
                const nameEl = document.createElement("div");
                nameEl.textContent = fl.name || fl.path.split("/").pop();
                nameEl.style.cssText = "color:#f4f4f5; font-weight:500; font-size:11px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;";

                const badgeRow = document.createElement("div");
                badgeRow.style.cssText = "display:flex; align-items:center; gap:4px;";
                const szPill = document.createElement("span");
                szPill.textContent = formatBytes(fl.size || 0);
                szPill.style.cssText = "background:#25252d; color:#a1a1aa; border-radius:4px; padding:1px 5px; font-size:9.5px;";
                badgeRow.appendChild(szPill);

                meta.appendChild(nameEl);
                meta.appendChild(badgeRow);

                const menuBtn = document.createElement("button");
                menuBtn.innerHTML = ICONS.dotsH;
                menuBtn.style.cssText = "background:transparent; color:#71717a; border:none; cursor:pointer; padding:2px; align-self:flex-start;";
                menuBtn.onclick = (e) => {
                    e.stopPropagation();
                    openItemMenu(menuBtn, fl, false);
                };

                card.appendChild(thumb);
                card.appendChild(meta);
                card.appendChild(menuBtn);
                grid.appendChild(card);
            });

            leftPane.appendChild(grid);

            // Drag & Drop Zone
            const dropZone = document.createElement("div");
            dropZone.style.cssText = `
                margin-top: auto;
                min-height: 92px;
                border: 1px dashed #2e2e38;
                border-radius: 8px;
                background: #16161a;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 6px;
                color: #71717a;
                cursor: pointer;
                transition: border-color 0.15s, background 0.15s;
            `;
            dropZone.innerHTML = `${ICONS.folderOutline}<span style="font-size:11px;">Drop files or folders here</span>`;
            dropZone.onclick = () => hiddenFileInput.click();
            dropZone.ondragover = (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.style.borderColor = "#38bdf8";
                dropZone.style.background = "rgba(56, 189, 248, 0.05)";
            };
            dropZone.ondragleave = (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.style.borderColor = "#2e2e38";
                dropZone.style.background = "#16161a";
            };
            dropZone.ondrop = async (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.style.borderColor = "#2e2e38";
                dropZone.style.background = "#16161a";
                const entries = await extractDroppedFiles(e.dataTransfer);
                if (entries.length === 0) return;
                await uploadBrowserFileEntries(c.apiBase, c.code, c.roomId, currentPath, entries, (m) => {
                    statusMsg = m;
                    render();
                });
                await loadRoomsAndFS(currentPath);
            };
            leftPane.appendChild(dropZone);
            body.appendChild(leftPane);

            // Right Pane: Preview & Metadata Inspector
            const rightPane = document.createElement("div");
            rightPane.style.cssText = `
                width: 190px;
                border-left: 1px solid #222228;
                background: #151519;
                padding: 10px 12px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                flex-shrink: 0;
            `;

            const prevHeader = document.createElement("div");
            prevHeader.style.cssText = "display:flex; justify-content:space-between; align-items:center; color:#d4d4d8; font-weight:600; font-size:11px;";
            prevHeader.innerHTML = `<span>Preview</span>${ICONS.chevDown}`;
            rightPane.appendChild(prevHeader);

            const prevViewport = document.createElement("div");
            prevViewport.style.cssText = `
                height: 112px;
                background: #101013;
                border: 1px solid #24242b;
                border-radius: 7px;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                padding: 4px;
            `;
            if (selectedItem?.dataUrl) {
                const img = document.createElement("img");
                img.src = selectedItem.dataUrl;
                img.style.cssText = "max-width:100%; max-height:100%; object-fit:contain; border-radius:4px;";
                prevViewport.appendChild(img);
            } else if (selectedItem && !selectedItem.isFolder && selectedItem.previewText) {
                const pre = document.createElement("div");
                pre.textContent = selectedItem.previewText.slice(0, 300);
                pre.style.cssText = "font-family:monospace; font-size:9.5px; color:#a1a1aa; width:100%; height:100%; overflow:auto; white-space:pre-wrap; word-break:break-all;";
                prevViewport.appendChild(pre);
            } else if (selectedItem?.isFolder) {
                prevViewport.innerHTML = ICONS.folderFill;
            } else {
                prevViewport.innerHTML = `<span style="color:#52525b; font-size:10px;">Select an item</span>`;
            }
            rightPane.appendChild(prevViewport);

            // Metadata key-value rows
            const metaTable = document.createElement("div");
            metaTable.style.cssText = "display:flex; flex-direction:column; gap:5px; font-size:10.5px; margin-top:2px;";
            const mkRow = (k, v, valColor = "#d4d4d8") => {
                const r = document.createElement("div");
                r.style.cssText = "display:flex; justify-content:space-between; gap:6px;";
                r.innerHTML = `<span style="color:#71717a;">${k}</span><span style="color:${valColor}; font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${v}</span>`;
                return r;
            };
            metaTable.appendChild(mkRow("Status", statusMsg, statusOk ? "#4ade80" : "#f87171"));
            metaTable.appendChild(mkRow("Name", selectedItem?.name || currentPath || "/"));
            metaTable.appendChild(mkRow("Type", selectedItem?.type || "folder"));
            metaTable.appendChild(mkRow("Size", formatBytes(selectedItem?.size || 0)));
            rightPane.appendChild(metaTable);

            if (selectedItem) {
                const actRow = document.createElement("div");
                actRow.style.cssText = "margin-top:auto; display:grid; grid-template-columns: 1fr 1fr; gap:5px;";

                const cpBtn = document.createElement("button");
                cpBtn.textContent = "Copy R2";
                cpBtn.style.cssText = "background:#222229; color:#d4d4d8; border:1px solid #2e2e38; border-radius:5px; padding:4px; font-size:10.5px; cursor:pointer;";
                cpBtn.onclick = async () => {
                    const dest = prompt("Copy to R2 path:", `${selectedItem.path}_copy`);
                    if (!dest) return;
                    await proxyReq("POST", `/${c.roomId}/fs/copy`, {}, {
                        src_path: selectedItem.path,
                        dest_path: dest.trim(),
                        dest_room_id: c.roomId,
                    });
                    await loadRoomsAndFS(currentPath);
                };

                const delBtn = document.createElement("button");
                delBtn.textContent = "Delete";
                delBtn.style.cssText = "background:#222229; color:#f87171; border:1px solid #2e2e38; border-radius:5px; padding:4px; font-size:10.5px; cursor:pointer;";
                delBtn.onclick = async () => {
                    if (!confirm(`Delete '${selectedItem.path}'?`)) return;
                    await proxyReq("DELETE", `/${c.roomId}/fs`, { path: selectedItem.path });
                    selectedItem = null;
                    await loadRoomsAndFS(currentPath);
                };

                actRow.appendChild(cpBtn);
                actRow.appendChild(delBtn);
                rightPane.appendChild(actRow);
            }

            body.appendChild(rightPane);
        } else if (activeTab === "note") {
            const notePane = document.createElement("div");
            notePane.style.cssText = "flex:1; display:flex; flex-direction:column; padding:10px 12px; gap:8px; min-width:0;";

            const curNote = notesList[activeNoteIdx] || { id: "note_init", title: "Ghi chú", text: "" };

            const area = document.createElement("textarea");
            area.value = curNote.text || "";
            area.placeholder = "Paste or type here — auto-saves & syncs to R2 immediately...";
            area.style.cssText = "flex:1; width:100%; box-sizing:border-box; background:#101013; color:#f4f4f5; border:1px solid #25252c; border-radius:7px; padding:10px; font-family:monospace; font-size:12px; line-height:1.45; resize:none; outline:none;";
            area.oninput = () => {
                if (!notesList[activeNoteIdx]) {
                    notesList[activeNoteIdx] = { id: `local_${Date.now()}`, title: `Ghi chú ${activeNoteIdx + 1}`, text: "" };
                }
                notesList[activeNoteIdx].text = area.value;
                scheduleAutoSaveNote(false);
            };
            area.onpaste = () => {
                setTimeout(() => {
                    if (!notesList[activeNoteIdx]) {
                        notesList[activeNoteIdx] = { id: `local_${Date.now()}`, title: `Ghi chú ${activeNoteIdx + 1}`, text: "" };
                    }
                    notesList[activeNoteIdx].text = area.value;
                    scheduleAutoSaveNote(true);
                }, 20);
            };

            const btnBar = document.createElement("div");
            btnBar.style.cssText = "display:flex; align-items:center; justify-content:space-between; gap:6px;";

            noteStatusEl = document.createElement("span");
            noteStatusEl.textContent = "Auto-sync ready";
            noteStatusEl.style.cssText = "color:#71717a; font-size:10.5px;";

            const rightBtns = document.createElement("div");
            rightBtns.style.cssText = "display:flex; gap:6px;";

            const clearBtn = document.createElement("button");
            clearBtn.textContent = "Clear";
            clearBtn.style.cssText = "background:#1e1e24; color:#a1a1aa; border:1px solid #2e2e38; border-radius:6px; padding:5px 10px; font-size:11px; cursor:pointer;";
            clearBtn.onclick = () => {
                if (notesList[activeNoteIdx]) {
                    notesList[activeNoteIdx].text = "";
                    area.value = "";
                    saveNoteTabToR2(notesList[activeNoteIdx], true);
                }
            };

            const copyBtn = document.createElement("button");
            copyBtn.textContent = "Copy";
            copyBtn.style.cssText = "background:#38bdf8; color:#09090b; font-weight:600; border:none; border-radius:6px; padding:5px 14px; font-size:11px; cursor:pointer;";
            copyBtn.onclick = async () => {
                try {
                    await navigator.clipboard.writeText(area.value || "");
                    copyBtn.textContent = "Copied ✓";
                    setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
                } catch (_) {}
            };

            rightBtns.appendChild(clearBtn);
            rightBtns.appendChild(copyBtn);
            btnBar.appendChild(noteStatusEl);
            btnBar.appendChild(rightBtns);
            notePane.appendChild(area);
            notePane.appendChild(btnBar);
            body.appendChild(notePane);
        } else if (activeTab === "api") {
            const apiPane = document.createElement("div");
            apiPane.style.cssText = "flex:1; display:flex; flex-direction:column; padding:10px 12px; gap:7px; min-width:0;";

            const topRow = document.createElement("div");
            topRow.style.cssText = "display:flex; gap:6px;";

            const pSel = document.createElement("select");
            pSel.style.cssText = "background:#101013; color:#d4d4d8; border:1px solid #27272f; border-radius:6px; padding:4px 8px; font-size:11px;";
            ENDPOINT_PRESETS.forEach((p) => {
                const o = document.createElement("option");
                o.value = p.id;
                o.textContent = p.label;
                if (p.id === apiPreset) o.selected = true;
                pSel.appendChild(o);
            });
            pSel.onchange = () => {
                const p = ENDPOINT_PRESETS.find((x) => x.id === pSel.value);
                if (!p) return;
                apiPreset = p.id;
                apiMethod = p.method;
                apiPath = p.path.replace("{room_id}", c.roomId);
                apiBody = p.body ? JSON.stringify(p.body).replace(/\{room_id\}/g, c.roomId) : "";
                render();
            };

            const mIn = document.createElement("input");
            mIn.value = apiMethod;
            mIn.style.cssText = "width:54px; background:#101013; color:#38bdf8; font-weight:600; border:1px solid #27272f; border-radius:6px; padding:4px 6px; font-size:11px; text-align:center;";
            mIn.oninput = () => (apiMethod = mIn.value.trim().toUpperCase());

            const epIn = document.createElement("input");
            epIn.value = apiPath;
            epIn.style.cssText = "flex:1; background:#101013; color:#f4f4f5; border:1px solid #27272f; border-radius:6px; padding:4px 8px; font-family:monospace; font-size:11px;";
            epIn.oninput = () => (apiPath = epIn.value.trim());

            const sendBtn = document.createElement("button");
            sendBtn.textContent = "Send";
            sendBtn.style.cssText = "background:#38bdf8; color:#09090b; font-weight:600; border:none; border-radius:6px; padding:4px 12px; font-size:11px; cursor:pointer;";
            sendBtn.onclick = async () => {
                let parsedBody = null;
                if (apiMethod !== "GET" && apiBody.trim()) {
                    try {
                        parsedBody = JSON.parse(apiBody);
                    } catch (e) {
                        apiOutput = `Invalid JSON body: ${e}`;
                        render();
                        return;
                    }
                }
                const res = await proxyReq(apiMethod, apiPath, {}, parsedBody);
                apiOutput = JSON.stringify(res.data || res, null, 2);
                setW("selected_content", apiOutput);
                render();
            };

            topRow.appendChild(pSel);
            topRow.appendChild(mIn);
            topRow.appendChild(epIn);
            topRow.appendChild(sendBtn);

            const bodyIn = document.createElement("input");
            bodyIn.value = apiBody;
            bodyIn.placeholder = 'Request JSON body (e.g. {"path": "models/loras"})';
            bodyIn.style.cssText = "background:#101013; color:#a1a1aa; border:1px solid #25252c; border-radius:6px; padding:5px 8px; font-family:monospace; font-size:10.5px;";
            bodyIn.oninput = () => (apiBody = bodyIn.value);

            const outBox = document.createElement("textarea");
            outBox.readOnly = true;
            outBox.value = apiOutput || "// Response JSON";
            outBox.style.cssText = "flex:1; background:#101013; color:#a1a1aa; border:1px solid #25252c; border-radius:6px; padding:6px 8px; font-family:monospace; font-size:10.5px; resize:none; outline:none;";

            apiPane.appendChild(topRow);
            apiPane.appendChild(bodyIn);
            apiPane.appendChild(outBox);
            body.appendChild(apiPane);
        }

        root.appendChild(body);

        // 3. FOOTER BAR (Set Project + Sync)
        const footer = document.createElement("div");
        footer.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 7px 12px;
            border-top: 1px solid #222228;
            background: #16161a;
            flex-shrink: 0;
        `;

        const projWrap = document.createElement("div");
        projWrap.title = ACTIVE_PROJECT.laptopPath
            ? `Click to open Laptop folder: ${ACTIVE_PROJECT.laptopPath}`
            : "Click Set Project to choose folder";
        projWrap.style.cssText = "display:flex; align-items:center; gap:6px; font-size:11px; color:#71717a; overflow:hidden; white-space:nowrap; cursor:pointer;";
        projWrap.innerHTML = `<span>Project:</span><span style="color:#e4e4e7; font-weight:500;">${ACTIVE_PROJECT.label}</span>`;
        projWrap.onclick = () => {
            if (ACTIVE_PROJECT.mode === "laptop") {
                callLaptopBridge("open_explorer");
            }
        };
        footer.appendChild(projWrap);

        const footBtns = document.createElement("div");
        footBtns.style.cssText = "display:flex; align-items:center; gap:6px;";

        const setProjBtn = document.createElement("button");
        setProjBtn.textContent = "Set Project";
        setProjBtn.title = "Choose local folder on Laptop (Máy con) or ComfyUI Server";
        setProjBtn.style.cssText = `
            background: #1e1e24;
            color: #d4d4d8;
            border: 1px solid #2e2e38;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 500;
            cursor: pointer;
        `;
        setProjBtn.onclick = (e) => {
            e.stopPropagation();
            closeContextMenu();
            const menu = document.createElement("div");
            menu.style.cssText = `
                position: fixed;
                z-index: 99999;
                background: #1c1c21;
                border: 1px solid #2e2e36;
                border-radius: 6px;
                padding: 4px;
                min-width: 210px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.65);
                display: flex;
                flex-direction: column;
                gap: 2px;
                font-family: system-ui, -apple-system, sans-serif;
                font-size: 11px;
            `;
            const rect = setProjBtn.getBoundingClientRect();
            menu.style.left = `${Math.max(10, rect.right - 210)}px`;
            menu.style.top = `${Math.max(10, rect.top - 150)}px`;

            const addOpt = (label, color, fn) => {
                const b = document.createElement("button");
                b.textContent = label;
                b.style.cssText = `background:transparent; color:${color}; border:none; border-radius:4px; padding:6px 8px; text-align:left; cursor:pointer; font-size:11px;`;
                b.onmouseenter = () => (b.style.background = "#27272f");
                b.onmouseleave = () => (b.style.background = "transparent");
                b.onclick = async (ev) => {
                    ev.stopPropagation();
                    closeContextMenu();
                    await fn();
                };
                menu.appendChild(b);
            };

            addOpt("📁 Pick Folder on This Device (Laptop)", "#38bdf8", async () => {
                await pickFolderOnLaptopFlow((m) => {
                    statusOk = true;
                    statusMsg = m;
                    render();
                });
            });

            addOpt("⌨️ Enter Laptop Folder Path...", "#f4f4f5", async () => {
                const p = prompt("Laptop Folder Path:", ACTIVE_PROJECT.laptopPath || "C:\\Users\\EXT-013\\Downloads\\Mockin_Sync");
                if (!p || !p.trim()) return;
                const st = await callLaptopBridge("set_folder", { laptop_path: p.trim() });
                ACTIVE_PROJECT.mode = "laptop";
                ACTIVE_PROJECT.laptopPath = st?.laptop_path || p.trim();
                ACTIVE_PROJECT.label = `${ACTIVE_PROJECT.laptopPath.split(/[\\/]/).pop()} (Laptop)`;
                statusOk = true;
                statusMsg = `Laptop: ${ACTIVE_PROJECT.laptopPath}`;
                render();
            });

            addOpt("📂 Open Laptop Folder in Explorer", "#f4f4f5", async () => {
                await callLaptopBridge("open_explorer");
            });

            addOpt("⬆️ Push Laptop Folder -> R2", "#4ade80", async () => {
                statusMsg = "Pushing Laptop -> R2...";
                render();
                const upRes = await callLaptopBridge("sync_push", {
                    api_base: c.apiBase,
                    code: c.code,
                    room_id: c.roomId,
                    remote_base_dir: currentPath || "",
                    laptop_path: ACTIVE_PROJECT.laptopPath || "",
                });
                statusOk = Boolean(upRes?.ok);
                statusMsg = upRes?.ok ? `Pushed ${upRes.uploaded_count} files -> R2` : upRes?.error || "Push error";
                await loadRoomsAndFS(currentPath);
            });

            addOpt("🖥️ ComfyUI Root (H200 Server 1:1)", "#a1a1aa", async () => {
                ACTIVE_PROJECT.mode = "server";
                ACTIVE_PROJECT.dirHandle = null;
                ACTIVE_PROJECT.serverPath = "";
                ACTIVE_PROJECT.label = "ComfyUI (H200 1:1)";
                setW("target_local_preset", "ComfyUI/");
                setW("dest_or_target_path", "");
                statusMsg = "Project: ComfyUI (H200 1:1)";
                render();
            });

            document.body.appendChild(menu);
            contextMenuEl = menu;
            setTimeout(() => window.addEventListener("click", closeContextMenu, { once: true }), 10);
        };

        const syncBtn = document.createElement("button");
        syncBtn.textContent = "Sync";
        syncBtn.title = "Sync current/selected path 1:1 with active Project (Laptop or H200)";
        syncBtn.style.cssText = `
            background: #38bdf8;
            color: #09090b;
            border: none;
            border-radius: 6px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
        `;
        syncBtn.onclick = async () => {
            if (activeTab === "note") {
                await loadRoomsAndFS(currentPath);
                if (noteStatusEl) {
                    noteStatusEl.textContent = "Synced from R2 ✓";
                    noteStatusEl.style.color = "#4ade80";
                }
                return;
            }
            const target = selectedItem ? selectedItem.path : currentPath;
            const mode = selectedItem && !selectedItem.isFolder ? "file" : "folder";

            if (ACTIVE_PROJECT.mode !== "server") {
                const res = await syncOrSaveToLaptop(c.apiBase, c.code, c.roomId, target, mode, true, (m) => {
                    statusOk = true;
                    statusMsg = m;
                    render();
                });
                if (res) {
                    setW("selected_content", JSON.stringify(res, null, 2));
                }
                return;
            }

            statusMsg = "Syncing -> H200...";
            render();
            try {
                const r = await api.fetchApi("/mockin/r2/download_sync", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        api_base: c.apiBase,
                        code: c.code,
                        room_id: c.roomId,
                        remote_path: target,
                        mode,
                        skip_existing: true,
                        preserve_subpath: true,
                        target_preset: "ComfyUI/",
                        custom_local_dir: ACTIVE_PROJECT.serverPath || "",
                    }),
                });
                const d = await r.json();
                if (d && d.ok) {
                    statusOk = true;
                    statusMsg = `H200 Synced (${d.downloaded_count} new, ${d.skipped_count || 0} ok)`;
                    setW("selected_content", JSON.stringify(d, null, 2));
                    render();
                    return;
                }
            } catch (_) {}
            statusOk = false;
            statusMsg = "H200 Sync failed";
            render();
        };

        footBtns.appendChild(setProjBtn);
        footBtns.appendChild(syncBtn);
        footer.appendChild(footBtns);
        root.appendChild(footer);
    }

    _projectListeners.add(render);
    render();

    const domWidget = node.addDOMWidget("mockin_r2_drive_minimal", "custom", root, {
        serialize: false,
        hideOnZoom: false,
        margin: 10,
        getMinHeight: () => 410,
    });
    lockDomWidgetFullWidth(domWidget, node, 410, 600);

    const clampMainSize = (n) => {
        const curW = n.size?.[0] || 640;
        const curH = n.size?.[1] || 505;
        if (curW < 620 || curH < 495 || curH > 720) {
            n.setSize([Math.max(640, Math.min(1200, curW)), 505]);
        }
    };
    clampMainSize(node);

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
        sanitizeNodeWidgets(this);
        hideAllNativeWidgets(this);
        clampMainSize(this);
        setTimeout(() => loadRoomsAndFS(currentPath), 60);
        return r;
    };

    setTimeout(() => {
        clampMainSize(node);
        loadRoomsAndFS("");
    }, 120);
}

// ============================================================================
// 2. COMPANION NODES
// ============================================================================
function setupCompactCompanionNode(node, nodeKind) {
    if (node.__mockinR2CompSetup) return;
    node.__mockinR2CompSetup = true;

    const getW = (name) => node.widgets?.find((w) => w.name === name);
    const setW = (name, val) => {
        const w = getW(name);
        if (w) w.value = val;
    };

    sanitizeNodeWidgets(node);
    hideAllNativeWidgets(node);
    ensureLaptopProjectInitialized();

    let statusText = "Idle";
    let statusDot = "#71717a";

    const card = document.createElement("div");
    card.style.cssText = `
        background: #141417;
        border: 1px solid #27272c;
        border-radius: 9px;
        padding: 10px 12px;
        color: #f4f4f5;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
        font-size: 11px;
        display: flex;
        flex-direction: column;
        gap: 7px;
        box-sizing: border-box;
        width: 100%;
        height: 100%;
    `;
    stopCanvasEvents(card);

    const mkInputRow = (label, val, placeholder, onChange) => {
        const wrap = document.createElement("div");
        wrap.style.cssText = "display:flex; flex-direction:column; gap:3px;";
        const lbl = document.createElement("div");
        lbl.style.cssText = "display:flex; justify-content:space-between; color:#a1a1aa; font-size:10.5px;";
        lbl.innerHTML = `<span>${label}</span>`;
        const inp = document.createElement("input");
        inp.value = val || "";
        inp.placeholder = placeholder;
        inp.style.cssText = `
            background: #101013;
            color: #e4e4e7;
            border: 1px solid #26262e;
            border-radius: 6px;
            padding: 5px 8px;
            font-size: 11px;
            outline: none;
            width: 100%;
            box-sizing: border-box;
        `;
        inp.oninput = () => onChange(inp.value);
        wrap.appendChild(lbl);
        wrap.appendChild(inp);
        return wrap;
    };

    function renderCompanion() {
        sanitizeNodeWidgets(node);
        card.innerHTML = "";
        const norm = normalizeR2Params(getW("api_base")?.value, getW("room_code")?.value, getW("room_id")?.value);
        const roomId = norm.roomId;
        const apiBase = norm.apiBase;
        const code = norm.code;

        if (nodeKind === "pull") {
            const rem = getW("remote_path")?.value || "models/loras";
            card.appendChild(mkInputRow("Input", rem, "models/loras", (v) => setW("remote_path", v)));

            const ftr = document.createElement("div");
            ftr.style.cssText = "display:flex; align-items:center; justify-content:space-between; margin-top:auto;";
            ftr.innerHTML = `<span style="color:#71717a; font-size:10.5px;">Status: <span style="color:${statusDot};">●</span> ${statusText}</span>`;

            const btns = document.createElement("div");
            btns.style.cssText = "display:flex; gap:4px;";
            const bSet = document.createElement("button");
            bSet.textContent = "Set Project";
            bSet.style.cssText = "background:#1e1e24; color:#d4d4d8; border:1px solid #2e2e38; border-radius:5px; padding:3px 7px; font-size:10px; cursor:pointer;";
            bSet.onclick = async () => {
                await pickFolderOnLaptopFlow((m) => {
                    statusText = ACTIVE_PROJECT.label;
                    statusDot = "#38bdf8";
                    renderCompanion();
                });
            };

            const bPull = document.createElement("button");
            bPull.textContent = "Sync";
            bPull.style.cssText = "background:#38bdf8; color:#09090b; font-weight:600; border:none; border-radius:5px; padding:3px 9px; font-size:10px; cursor:pointer;";
            bPull.onclick = async () => {
                statusText = "Syncing...";
                statusDot = "#38bdf8";
                renderCompanion();
                const targetRem = getW("remote_path")?.value || "";
                if (ACTIVE_PROJECT.mode !== "server") {
                    const res = await syncOrSaveToLaptop(apiBase, code, roomId, targetRem, "auto", true, (m) => {
                        statusText = m;
                        statusDot = "#4ade80";
                        renderCompanion();
                    });
                    if (!res?.ok) {
                        statusDot = "#f87171";
                        statusText = "Sync error";
                        renderCompanion();
                    }
                    return;
                }
                try {
                    const r = await api.fetchApi("/mockin/r2/download_sync", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            api_base: apiBase,
                            code,
                            room_id: roomId,
                            remote_path: targetRem,
                            target_preset: "ComfyUI/",
                            custom_local_dir: ACTIVE_PROJECT.serverPath || "",
                            preserve_subpath: true,
                            mode: "auto",
                            skip_existing: true,
                        }),
                    });
                    const d = await r.json();
                    statusDot = d.ok ? "#4ade80" : "#f87171";
                    statusText = d.ok ? `${d.downloaded_count} synced` : "Error";
                } catch (_) {
                    statusDot = "#f87171";
                    statusText = "Error";
                }
                renderCompanion();
            };
            btns.appendChild(bSet);
            btns.appendChild(bPull);
            ftr.appendChild(btns);
            card.appendChild(ftr);
        } else if (nodeKind === "push") {
            const remBase = getW("remote_base_dir")?.value || "comfy_uploads";
            const locSub = getW("local_file_or_folder_path")?.value || "";
            card.appendChild(mkInputRow("Remote destination", remBase, "comfy_uploads", (v) => setW("remote_base_dir", v)));
            card.appendChild(mkInputRow("Local subfolder / file", locSub, "ComfyUI/output", (v) => setW("local_file_or_folder_path", v)));

            const ftr = document.createElement("div");
            ftr.style.cssText = "display:flex; align-items:center; justify-content:space-between; margin-top:auto;";
            ftr.innerHTML = `<span style="color:#71717a; font-size:10.5px;">Status: <span style="color:${statusDot};">●</span> ${statusText}</span>`;

            const bPush = document.createElement("button");
            bPush.textContent = "Push";
            bPush.style.cssText = "background:#38bdf8; color:#09090b; font-weight:600; border:none; border-radius:5px; padding:3px 10px; font-size:10px; cursor:pointer;";
            bPush.onclick = async () => {
                statusText = "Pushing...";
                statusDot = "#38bdf8";
                renderCompanion();
                const locVal = (getW("local_file_or_folder_path")?.value || "").trim();
                const remVal = (getW("remote_base_dir")?.value || "comfy_uploads").trim();
                // If path looks like a Windows Laptop path (e.g. C:\...) or blank with Laptop active project, try Laptop push first
                if (locVal.includes(":\\") || (!locVal && ACTIVE_PROJECT.mode === "laptop")) {
                    const lpRes = await callLaptopBridge("sync_push", {
                        api_base: apiBase,
                        code,
                        room_id: roomId,
                        remote_base_dir: remVal,
                        laptop_path: locVal || ACTIVE_PROJECT.laptopPath,
                    });
                    if (lpRes && lpRes.ok) {
                        statusDot = "#4ade80";
                        statusText = `${lpRes.uploaded_count} pushed (Laptop)`;
                        renderCompanion();
                        return;
                    }
                }
                const r = await api.fetchApi("/mockin/r2/upload_local", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        api_base: apiBase,
                        code,
                        room_id: roomId,
                        source_preset: getW("source_local_preset")?.value || "ComfyUI/output",
                        local_path: locVal,
                        remote_base_dir: remVal,
                    }),
                });
                const d = await r.json();
                statusDot = d.ok ? "#4ade80" : "#f87171";
                statusText = d.ok ? `${d.uploaded_count} pushed` : "Error";
                renderCompanion();
            };
            ftr.appendChild(bPush);
            card.appendChild(ftr);
        } else if (nodeKind === "copy") {
            const src = getW("src_path")?.value || "";
            const dst = getW("dest_path")?.value || "";
            card.appendChild(mkInputRow("Source folder / file", src, "models/loras", (v) => setW("src_path", v)));
            card.appendChild(mkInputRow("Destination path", dst, "backup/loras", (v) => setW("dest_path", v)));

            const ftr = document.createElement("div");
            ftr.style.cssText = "display:flex; align-items:center; justify-content:space-between; margin-top:auto;";
            ftr.innerHTML = `<span style="color:#71717a; font-size:10.5px;">Status: <span style="color:${statusDot};">●</span> ${statusText}</span>`;

            const bRun = document.createElement("button");
            bRun.textContent = "Copy";
            bRun.style.cssText = "background:#38bdf8; color:#09090b; font-weight:600; border:none; border-radius:5px; padding:3px 10px; font-size:10px; cursor:pointer;";
            bRun.onclick = async () => {
                statusText = "Copying...";
                statusDot = "#38bdf8";
                renderCompanion();
                const d = await directR2Request(
                    apiBase,
                    code,
                    roomId,
                    "POST",
                    `/${encodeURIComponent(roomId)}/fs/copy`,
                    {},
                    {
                        src_path: getW("src_path")?.value || "",
                        dest_path: getW("dest_path")?.value || "",
                        dest_room_id: roomId,
                    }
                );
                statusDot = d.ok ? "#4ade80" : "#f87171";
                statusText = d.ok ? `Copied (${d.data?.copied_count ?? 1})` : "Error";
                renderCompanion();
            };
            ftr.appendChild(bRun);
            card.appendChild(ftr);
        } else if (nodeKind === "read") {
            const remFile = getW("remote_file_path")?.value || "";
            card.appendChild(mkInputRow("Remote file (blank = Note)", remFile, "workflow.json", (v) => setW("remote_file_path", v)));

            const ftr = document.createElement("div");
            ftr.style.cssText = "display:flex; align-items:center; justify-content:space-between; margin-top:auto;";
            ftr.innerHTML = `<span style="color:#71717a; font-size:10.5px;">Status: <span style="color:${statusDot};">●</span> ${statusText}</span>`;

            const bRead = document.createElement("button");
            bRead.textContent = "Read";
            bRead.style.cssText = "background:#38bdf8; color:#09090b; font-weight:600; border:none; border-radius:5px; padding:3px 10px; font-size:10px; cursor:pointer;";
            bRead.onclick = async () => {
                statusText = "Reading...";
                statusDot = "#38bdf8";
                renderCompanion();
                const path = (getW("remote_file_path")?.value || "").trim();
                if (!path) {
                    const d = await directR2Request(apiBase, code, roomId, "GET", `/${encodeURIComponent(roomId)}`, {});
                    const notes = extractRoomNotesList(d.data || {});
                    const activeNote = notes.find((n) => (n.text || "").trim().length > 0) || notes[0];
                    statusDot = d.ok ? "#4ade80" : "#f87171";
                    statusText = d.ok ? `Note (${(activeNote?.text || "").length} chars)` : "Error";
                } else {
                    const d = await directR2FileRead(apiBase, code, roomId, path, true);
                    statusDot = d.ok ? "#4ade80" : "#f87171";
                    statusText = d.ok ? formatBytes(d.size) : "Error";
                }
                renderCompanion();
            };
            ftr.appendChild(bRead);
            card.appendChild(ftr);
        }
    }

    renderCompanion();

    const baseH = nodeKind === "read" ? 105 : 152;
    const domW = node.addDOMWidget(`mockin_r2_${nodeKind}_ui`, "custom", card, {
        serialize: false,
        hideOnZoom: false,
        margin: 10,
        getMinHeight: () => baseH,
    });
    lockDomWidgetFullWidth(domW, node, baseH, 240);

    const clampCompSize = (n) => {
        const curW = n.size?.[0] || 260;
        const curH = n.size?.[1] || baseH + 65;
        if (curW < 240 || curH < baseH + 45 || curH > 340) {
            n.setSize([Math.max(260, Math.min(600, curW)), baseH + 65]);
        }
    };
    clampCompSize(node);

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
        sanitizeNodeWidgets(this);
        hideAllNativeWidgets(this);
        clampCompSize(this);
        renderCompanion();
        return r;
    };
}

const NODE_KIND_MAP = {
    MockinR2CloudDriveNode: "main",
    MockinR2SyncDownloadNode: "pull",
    MockinR2SyncUploadNode: "push",
    MockinR2FolderOpsNode: "copy",
    MockinR2ReadFileNode: "read",
};

function attachMockinNodeUI(node) {
    const kind = NODE_KIND_MAP[node?.comfyClass || node?.type];
    if (!kind) return;
    if (kind === "main") {
        setupMainDriveNode(node);
    } else {
        setupCompactCompanionNode(node, kind);
    }
}

app.registerExtension({
    name: "Mockin.R2CloudDrive",
    async nodeCreated(node) {
        attachMockinNodeUI(node);
    },
    async loadedGraphNode(node) {
        const kind = NODE_KIND_MAP[node?.comfyClass || node?.type];
        if (!kind) return;
        attachMockinNodeUI(node);
        sanitizeNodeWidgets(node);
        hideAllNativeWidgets(node);
        if (kind === "main" && (node.size?.[1] > 720 || node.size?.[1] < 495)) {
            node.setSize([Math.max(640, node.size?.[0] || 640), 505]);
        }
    },
});
