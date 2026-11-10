// Sidebar tab for the Nukomfy Manager panel.
// Uses the official app.extensionManager.registerSidebarTab API (ComfyUI v1.2.4+).

import {
    nukomfyPing,
    nukomfySetPassword,
    nukomfySetAvailability,
    nukomfyAvailability,
    nukomfyActivityTail,
    nukomfyActivityConfig,
    nukomfyActivityConfigSet,
    nukomfyRestoreIntegrity,
} from "./api.js";
import { renderHistorySection, confirmAndRunRestart } from "./history_section.js";
import { openModal, openPasswordConfirmModal, mapAuthError } from "./modal.js";

const MIN_PASSWORD_LENGTH = 4;

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

function makePasswordField(labelText, placeholder) {
    const input = el("input", {
        type: "password",
        placeholder: placeholder || "",
        autocomplete: "new-password",
    });
    const eye = el("span", {
        class: "nukomfy-mgr-form__eye",
        title: "Show / hide password",
    }, "\u{1F441}");
    eye.addEventListener("click", () => {
        if (input.type === "password") {
            input.type = "text";
            eye.textContent = "\u{1F576}";
        } else {
            input.type = "password";
            eye.textContent = "\u{1F441}";
        }
    });
    const wrap = el("div", { class: "nukomfy-mgr-form__password-row" }, [input, eye]);
    return { wrap, input, label: el("label", {}, labelText) };
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

// Red banner shown at the top of the panel when the graph nodes failed to
// import (missing OpenImageIO / fileseq). The manager itself keeps working;
// the banner points to the README for the install steps. Reuses the
// integrity-banner styling for a consistent look.
function _renderNodesBanner(banner, info) {
    const err = info && info.node_import_error;
    if (!err) {
        banner.style.display = "none";
        banner.innerHTML = "";
        return;
    }
    banner.innerHTML = "";
    const missing = (info && info.node_missing_module) || "";
    banner.appendChild(el(
        "div",
        { class: "nukomfy-mgr-integrity-banner__header" },
        [el("strong",
            { class: "nukomfy-mgr-integrity-banner__title" },
            "Graph nodes unavailable")],
    ));
    const body = el("div", { class: "nukomfy-mgr-integrity-banner__body" });
    body.appendChild(el("p", {}, missing
        ? `The Nukomfy graph nodes could not load (missing "${missing}"). `
            + "The manager still works."
        : "The Nukomfy graph nodes could not load. The manager still works."));
    const p2 = el("p", {},
        "Install the Python dependencies and restart ComfyUI. See the ");
    p2.appendChild(el("a", {
        href: "https://github.com/francescolorussi/ComfyUI-Nukomfy-Suite",
        target: "_blank",
        rel: "noopener noreferrer",
        class: "nukomfy-mgr-deps-banner__link",
    }, "README"));
    p2.appendChild(document.createTextNode(" for instructions."));
    body.appendChild(p2);
    banner.appendChild(body);
    banner.style.display = "block";
}

function openSetPasswordModal(hasPassword, onSaved) {
    const banner = makeBanner();
    let currentField = null;
    if (hasPassword) {
        currentField = makePasswordField(
            "Current password", "Current password");
    }
    const newField = makePasswordField(
        "New password",
        `At least ${MIN_PASSWORD_LENGTH} characters`);
    const confirmField = makePasswordField(
        "Confirm new password", "Repeat new password");

    const cancelBtn = el("button", { class: "nukomfy-mgr-btn" }, "Cancel");
    const saveBtn = el("button", {
        class: "nukomfy-mgr-btn nukomfy-mgr-btn--primary",
        disabled: "true",
    }, "Save");

    const description = hasPassword
        ? "The current password is required to confirm the change."
        : "Set an admin password for this machine.";

    const body = el("div", { class: "nukomfy-mgr-pw-modal-body" }, [
        el("p", { class: "nukomfy-mgr-pw-modal-description" }, description),
        banner,
        currentField ? currentField.label : null,
        currentField ? currentField.wrap : null,
        newField.label,
        newField.wrap,
        confirmField.label,
        confirmField.wrap,
        el("div", { class: "nukomfy-mgr-actions" }, [cancelBtn, saveBtn]),
    ]);

    const fullTitle = hasPassword
        ? "Nukomfy Manager - Change admin password"
        : "Nukomfy Manager - Set admin password";
    const modal = openModal({
        title: fullTitle,
        content: body,
        width: 480,
        height: "auto",
    });
    modal.dialog.style.height = "auto";

    function validate() {
        const newV = newField.input.value;
        const confirmV = confirmField.input.value;
        const currentV = currentField ? currentField.input.value : "";
        const valid =
            newV.length >= MIN_PASSWORD_LENGTH &&
            newV === confirmV &&
            (!currentField || currentV.length > 0);
        if (valid) saveBtn.removeAttribute("disabled");
        else saveBtn.setAttribute("disabled", "true");
    }

    [newField.input, confirmField.input].forEach(
        (i) => i.addEventListener("input", validate));
    if (currentField) currentField.input.addEventListener("input", validate);

    cancelBtn.addEventListener("click", () => modal.close());

    saveBtn.addEventListener("click", async () => {
        hideBanner(banner);
        saveBtn.setAttribute("disabled", "true");
        const newV = newField.input.value;
        const currentV = currentField ? currentField.input.value : null;
        const { status, body: respBody } = await nukomfySetPassword(newV, currentV);
        if (status === 200 && respBody && respBody.ok) {
            modal.close();
            if (typeof onSaved === "function") onSaved();
            return;
        }
        if (status === 401 && respBody && respBody.reason === "wrong_current") {
            showBanner(banner, "err", "Current password is wrong.");
            if (currentField) {
                currentField.input.value = "";
                currentField.input.focus();
            }
        } else if (status === 400 && respBody && respBody.reason === "weak_password") {
            showBanner(banner, "err",
                `Password too short (min ${MIN_PASSWORD_LENGTH}).`);
        } else if (status === 0) {
            showBanner(banner, "err", "Cannot reach server.");
        } else {
            showBanner(banner, "err", "Could not save password. Please try again.");
        }
        saveBtn.removeAttribute("disabled");
    });

    setTimeout(() => {
        (currentField ? currentField.input : newField.input).focus();
    }, 0);
}

function renderPasswordForm(container, hasPassword, passwordChangedAt, onSaved) {
    container.innerHTML = "";

    container.appendChild(
        el("h4", { class: "nukomfy-mgr-section-title" }, "Admin password"));
    container.appendChild(el("p", { class: "nukomfy-mgr-section-description" },
        "Required for privileged actions on this machine: reboot, " +
        "change Availability, toggle history recording, force-abort or " +
        "force-remove jobs submitted by other users."));

    container.appendChild(el("div", { class: "nukomfy-mgr-status-row" }, [
        el("span", { class: "nukomfy-mgr-status-label" }, [
            "Admin password: ",
            el("strong", {
                class: hasPassword
                    ? "nukomfy-mgr-status-value nukomfy-mgr-status-value--ok"
                    : "nukomfy-mgr-status-value nukomfy-mgr-status-value--err",
            }, hasPassword ? "Set" : "Not set"),
        ]),
    ]));

    if (hasPassword && passwordChangedAt) {
        container.appendChild(el("p", {
            class: "nukomfy-mgr-placeholder",
            style: "font-size: 11px; margin: 2px 0 6px 0;",
        }, "Last changed: " + formatLocalTimestamp(passwordChangedAt)));
    }

    const btn = el("button", { class: "nukomfy-mgr-btn" },
        hasPassword ? "Change admin password" : "Set admin password");
    container.appendChild(btn);
    btn.addEventListener("click", () => {
        openSetPasswordModal(hasPassword, onSaved);
    });
}

// Convert a server-side ISO 8601 UTC timestamp (e.g. "2026-01-01T00:00:00+00:00")
// to a local-time string in YYYY-MM-DD HH:MM:SS form (24h, with seconds).
// Falls back to the raw input if parsing fails so the value is never
// silently dropped.
function formatLocalTimestamp(isoStr) {
    if (!isoStr) return "";
    try {
        const d = new Date(isoStr);
        if (isNaN(d.getTime())) return isoStr;
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
        return isoStr;
    }
}

function renderAvailabilitySection(container, hasPassword, state, onChanged) {
    container.innerHTML = "";

    const current = (state && state.availability) || "available";
    const isUnavailable = current === "unavailable";

    // Section heading + description.
    container.appendChild(
        el(
            "h4",
            { class: "nukomfy-mgr-section-title" },
            "Machine availability",
        ),
    );
    container.appendChild(
        el(
            "p",
            { class: "nukomfy-mgr-section-description" },
            "Mark this machine as Unavailable so the Nukomfy plugin " +
                "warns and blocks new submissions from other users. " +
                "Running jobs are not affected.",
        ),
    );

    // Toggle (iOS-style). ON = Available (the healthy default); OFF = Unavailable.
    const toggleClasses = ["nukomfy-toggle", "nukomfy-mgr-availability-toggle"];
    if (!isUnavailable) toggleClasses.push("nukomfy-toggle--on");
    if (!hasPassword) toggleClasses.push("nukomfy-toggle--disabled");
    const toggle = el(
        "div",
        {
            class: toggleClasses.join(" "),
            role: "switch",
            "aria-checked": isUnavailable ? "false" : "true",
            "aria-label": "Machine availability",
            tabindex: hasPassword ? "0" : "-1",
            title: hasPassword
                ? "Click to toggle Availability"
                : "Set an admin password first",
            style: "cursor: " + (hasPassword ? "pointer" : "not-allowed") + ";",
        },
        [el("span", { class: "nukomfy-mgr-toggle__knob nukomfy-toggle__knob" })],
    );

    // State text (left) and switch (right) share one line, so the toggle sits
    // directly against the status it flips; no separate action label needed.
    const statusRow = el(
        "div",
        { class: "nukomfy-mgr-row", style: "margin: 6px 0;" },
        [
            el(
                "span",
                { class: "nukomfy-mgr-status-label" },
                [
                    "Availability: ",
                    el(
                        "strong",
                        {
                            class: isUnavailable
                                ? "nukomfy-mgr-status-value nukomfy-mgr-status-value--unavailable"
                                : "nukomfy-mgr-status-value nukomfy-mgr-status-value--ok",
                        },
                        isUnavailable ? "Unavailable" : "Available",
                    ),
                ],
            ),
            toggle,
        ],
    );
    container.appendChild(statusRow);

    const targetValue = isUnavailable ? "available" : "unavailable";

    function promptToggle() {
        if (!hasPassword) return;
        const goingUnavailable = targetValue === "unavailable";
        openPasswordConfirmModal({
            title: goingUnavailable
                ? "Mark machine as Unavailable"
                : "Mark machine as Available",
            description: goingUnavailable
                ? "Toggle this machine Unavailable."
                : "Toggle this machine Available.",
            actionLabel: goingUnavailable ? "Mark Unavailable" : "Mark Available",
            onConfirm: async (pwd, ctx) => {
                const res = await nukomfySetAvailability(targetValue, pwd);
                if (res.status === 200 && res.body && res.body.ok) {
                    ctx.close();
                    onChanged();
                    return;
                }
                ctx.showError(mapAuthError(res));
            },
        });
    }

    toggle.addEventListener("click", promptToggle);
    toggle.addEventListener("keydown", (ev) => {
        if (!hasPassword) return;
        if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            promptToggle();
        }
    });
}

function _activityEntryKind(entry) {
    const action = entry.action || "";
    if (action.startsWith("password_block_")) return "warn";
    return entry.success === true ? "success" : "fail";
}

function _renderActivityEntry(entry) {
    const action = entry.action || "";
    const kind = _activityEntryKind(entry);
    const user = entry.user || "";
    const ip = entry.ip || "";
    const who = user && ip ? user + "@" + ip : (user || ip || "-");
    return el(
        "div",
        { class: "nukomfy-mgr-activity-entry " + kind },
        [
            el("span", { class: "ts" }, formatLocalTimestamp(entry.ts || "")),
            el("span", { class: "action" }, action),
            el("span", { class: "user" }, who),
            el("span", { class: "reason" }, entry.reason || ""),
        ],
    );
}

function _matchActivityEntry(entry, query) {
    if (!query) return true;
    const q = query.toLowerCase();
    return (
        (entry.action || "").toLowerCase().includes(q) ||
        (entry.user || "").toLowerCase().includes(q) ||
        (entry.ip || "").toLowerCase().includes(q) ||
        (entry.reason || "").toLowerCase().includes(q)
    );
}

async function renderActivitySettings(container, password, onSaved) {
    container.innerHTML = "";
    container.appendChild(
        el("h5", { class: "nukomfy-mgr-subsection-title" }, "Settings"),
    );
    container.appendChild(
        el(
            "p",
            { class: "nukomfy-mgr-subsection-description" },
            "Older entries are automatically deleted after this many days. " +
                "0 = never expire. The log file is also capped at about " +
                "60 MB total: when it reaches that size, the oldest entries " +
                "are dropped to make room.",
        ),
    );

    let initial = 90;
    try {
        const cfg = await nukomfyActivityConfig();
        if (cfg && typeof cfg.max_days_to_keep === "number") {
            initial = cfg.max_days_to_keep;
        }
    } catch (e) { /* default to 90 */ }

    const row = el("div", { class: "nukomfy-mgr-config-row" });
    const label = el(
        "span",
        { class: "nukomfy-mgr-status-label" },
        "Max days to keep:",
    );
    const input = el("input", {
        type: "number",
        min: "0",
        step: "1",
        value: String(initial),
    });
    const saveBtn = el(
        "button",
        { class: "nukomfy-mgr-btn nukomfy-mgr-btn--primary" },
        "Save",
    );
    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(saveBtn);
    container.appendChild(row);

    const banner = makeBanner();
    container.appendChild(banner);

    saveBtn.addEventListener("click", async () => {
        hideBanner(banner);
        saveBtn.setAttribute("disabled", "true");
        const days = parseInt(input.value, 10);
        if (isNaN(days) || days < 0) {
            showBanner(banner, "err", "Enter a non-negative integer.");
            saveBtn.removeAttribute("disabled");
            return;
        }
        const { status, body } = await nukomfyActivityConfigSet(password, days);
        if (status === 200 && body && body.ok) {
            showBanner(banner, "ok", "Saved.");
            if (onSaved) onSaved();
        } else if (status === 401) {
            showBanner(banner, "err", "Wrong admin password.");
        } else if (status === 429) {
            const retry = (body && body.retry_after_seconds) || 0;
            showBanner(
                banner,
                "err",
                "Too many failed attempts. Try again in " + retry + "s.",
            );
        } else if (status === 400) {
            showBanner(banner, "err", "Invalid value.");
        } else if (status === 0) {
            showBanner(banner, "err", "Cannot reach server.");
        } else {
            showBanner(banner, "err", "Could not save settings.");
        }
        saveBtn.removeAttribute("disabled");
    });
}

function _openActivityListModal(initialPassword, initialEntries) {
    let password = initialPassword;
    let entries = initialEntries;

    const searchInput = el("input", {
        type: "search",
        class: "nukomfy-mgr-activity-search",
        placeholder: "Filter (action, user, IP, reason)…",
    });
    const refreshBtn = el("button", { class: "nukomfy-mgr-btn" }, "Refresh");
    const counter = el("span", { class: "nukomfy-mgr-activity-counter" });
    const toolbar = el("div", { class: "nukomfy-mgr-activity-toolbar" }, [
        searchInput, refreshBtn, counter,
    ]);

    const list = el("div", { class: "nukomfy-mgr-activity-list" });
    const settings = el("div");

    function rebuild(filter) {
        list.innerHTML = "";
        // tail() returns oldest-first; show newest at the top.
        const ordered = entries.slice().reverse();
        const shown = ordered.filter((e) => _matchActivityEntry(e, filter));
        if (!shown.length) {
            list.appendChild(el(
                "div",
                { class: "nukomfy-mgr-activity-empty" },
                ordered.length
                    ? "No entries match the current filter."
                    : "No activity recorded yet.",
            ));
        }
        for (const entry of shown) list.appendChild(_renderActivityEntry(entry));
        counter.textContent = filter
            ? `${shown.length} of ${ordered.length} shown`
            : `${ordered.length} entries`;
    }

    const body = el("div", { class: "nukomfy-mgr-activity-modal-body" }, [
        toolbar,
        list,
        el("hr", {
            style: "border: 0; border-top: 1px solid #2a2a2a; margin: 10px 0;",
        }),
        settings,
    ]);

    const modal = openModal({
        title: "Nukomfy Manager - Activity log",
        content: body,
        width: 1100,
        height: 700,
    });

    rebuild("");
    searchInput.addEventListener("input", () => rebuild(searchInput.value));

    refreshBtn.addEventListener("click", async () => {
        refreshBtn.setAttribute("disabled", "true");
        const { status, body: resp } = await nukomfyActivityTail(password, 200);
        refreshBtn.removeAttribute("disabled");
        if (status === 200 && resp && resp.ok) {
            entries = resp.entries || [];
            rebuild(searchInput.value);
            return;
        }
        // Authentication broke between loads (e.g. password rotated, or a
        // fresh rate-limit block). Close + re-open the auth gate.
        modal.close();
        openActivityPanel();
    });

    renderActivitySettings(settings, password);
}

function openActivityPanel() {
    openPasswordConfirmModal({
        title: "View activity log",
        description: "View the audit log of privileged operations on this machine.",
        actionLabel: "Load entries",
        onConfirm: async (pwd, ctx) => {
            const res = await nukomfyActivityTail(pwd, 200);
            if (res.status === 200 && res.body && res.body.ok) {
                ctx.close();
                _openActivityListModal(pwd, res.body.entries || []);
                return;
            }
            ctx.showError(mapAuthError(res));
        },
    });
}

async function buildPanel(host, app) {
    host.innerHTML = "";
    const root = el("div", { class: "nukomfy-mgr-panel" });
    host.appendChild(root);

    const versionTag = el("span", { class: "nukomfy-mgr-tag" }, "");
    const header = el("header", { class: "nukomfy-mgr-panel__header" }, [
        el("h3", {}, "Nukomfy Manager"),
        versionTag,
    ]);
    root.appendChild(header);

    // Missing-dependencies banner. Lit up by the /ping response when the
    // graph nodes failed to import; the manager stays usable regardless.
    const nodesBanner = el("div",
        { class: "nukomfy-mgr-integrity-banner nukomfy-mgr-deps-banner" });
    nodesBanner.style.display = "none";
    root.appendChild(nodesBanner);

    // Integrity warning banner. Populated by the global
    // "nukomfy-mgr-tampered" event broadcast from api.js whenever an
    // admin endpoint reports tampered files in its response. Hidden
    // until the first non-empty list arrives.
    const integrityBanner = el("div",
        { class: "nukomfy-mgr-integrity-banner" });
    integrityBanner.style.display = "none";
    root.appendChild(integrityBanner);

    function _renderIntegrityBanner(files, restored) {
        integrityBanner.innerHTML = "";
        if (!files || !files.length) {
            integrityBanner.style.display = "none";
            return;
        }
        // Post-restore the recoverable files are already back to the
        // cached value. Only auth.json can linger here because it
        // requires a manual password change to be re-sealed.
        const sourceList = restored
            ? files.filter((f) => f === "auth.json")
            : files.slice();
        if (!sourceList.length) {
            integrityBanner.style.display = "none";
            return;
        }
        const list = sourceList.sort();
        const includesAuth = list.indexOf("auth.json") >= 0;
        const recoverable = list.filter((f) => f !== "auth.json");
        const header = el("div",
            { class: "nukomfy-mgr-integrity-banner__header" }, [
            el("strong",
                { class: "nukomfy-mgr-integrity-banner__title" },
                "Manager files changed outside this app"),
        ]);
        integrityBanner.appendChild(header);
        const body = el("div", { class: "nukomfy-mgr-integrity-banner__body" });
        body.appendChild(el("p", {},
            "These files were modified directly on disk:"));
        const ul = el("ul", { class: "nukomfy-mgr-integrity-banner__list" });
        for (const f of list) {
            ul.appendChild(el("li", {}, f));
        }
        body.appendChild(ul);
        let restoreBtn = null;
        if (recoverable.length) {
            let restoreLabel;
            if (recoverable.length === 1) {
                restoreLabel = recoverable[0];
            } else if (recoverable.length === 2) {
                restoreLabel = recoverable.join(" and ");
            } else {
                restoreLabel = recoverable.slice(0, -1).join(", ")
                    + ", and " + recoverable[recoverable.length - 1];
            }
            body.appendChild(el("p", {},
                "Click Restore now to revert " + restoreLabel
                + ". Admin password required."));
            restoreBtn = el("button", {
                type: "button",
                class: "nukomfy-mgr-btn nukomfy-mgr-btn--danger-primary",
                onClick: () => {
                    openPasswordConfirmModal({
                        title: "Restore manager files",
                        description:
                            "Revert " + restoreLabel
                            + " to the last value saved through the "
                            + "manager. Confirm with the admin password.",
                        actionLabel: "Restore",
                        onConfirm: async (password, ctx) => {
                            ctx.clearError();
                            const res = await nukomfyRestoreIntegrity(
                                password);
                            if (res && res.body && res.body.ok) {
                                ctx.close();
                            } else {
                                ctx.showError(mapAuthError(res));
                            }
                        },
                    });
                },
            }, "Restore now");
        }
        if (includesAuth) {
            body.appendChild(el("p", {},
                "auth.json holds the admin password and cannot be "
                + "restored automatically. To clear it: change the "
                + "password below, or delete auth.json on disk to "
                + "reset it."));
        }
        if (restoreBtn) {
            body.appendChild(restoreBtn);
        }
        integrityBanner.appendChild(body);
        integrityBanner.style.display = "block";
    }

    const _onTampered = (ev) => {
        const files = (ev.detail && ev.detail.files) || [];
        const restored = !!(ev.detail && ev.detail.restored);
        _renderIntegrityBanner(files, restored);
        // Only trigger a full sidebar refresh after a restore (an
        // admin action just changed server state). The pending-tamper
        // broadcast comes from nukomfyPing itself; calling refresh()
        // here would loop because refresh() re-invokes nukomfyPing.
        if (files.length && restored) {
            try { refresh(); } catch (e) { /* refresh not ready */ }
        }
    };
    window.addEventListener("nukomfy-mgr-tampered", _onTampered);

    const formSection = el("section", { class: "nukomfy-mgr-section" });
    root.appendChild(formSection);

    root.appendChild(el("hr", { style: "border: 0; border-top: 1px solid #333; margin: 14px 0;" }));

    const availabilitySection = el("section", { class: "nukomfy-mgr-section" });
    root.appendChild(availabilitySection);

    root.appendChild(el("hr", { style: "border: 0; border-top: 1px solid #333; margin: 14px 0;" }));

    const historySection = el("section", { class: "nukomfy-mgr-section" });
    root.appendChild(historySection);

    root.appendChild(el("hr", { style: "border: 0; border-top: 1px solid #333; margin: 14px 0;" }));

    const activitySection = el("section", { class: "nukomfy-mgr-section" });
    root.appendChild(activitySection);
    activitySection.appendChild(
        el("h4", { class: "nukomfy-mgr-section-title" }, "Activity log"),
    );
    activitySection.appendChild(
        el(
            "p",
            { class: "nukomfy-mgr-section-description" },
            "Record of privileged operations on this machine: " +
                "admin sign-in, password changes, reboot, " +
                "force abort/remove, availability toggles, and " +
                "password rate-limit blocks.",
        ),
    );
    const activityBtn = el(
        "button",
        { class: "nukomfy-mgr-btn" },
        "View activity log",
    );
    activitySection.appendChild(activityBtn);
    activityBtn.addEventListener("click", () => {
        if (activityBtn.disabled) return;
        openActivityPanel();
    });

    root.appendChild(el("hr", { style: "border: 0; border-top: 1px solid #333; margin: 14px 0;" }));

    const restartSection = el("section", { class: "nukomfy-mgr-section" });
    root.appendChild(restartSection);
    restartSection.appendChild(
        el("h4", { class: "nukomfy-mgr-section-title" }, "Restart ComfyUI"),
    );
    restartSection.appendChild(
        el(
            "p",
            { class: "nukomfy-mgr-section-description" },
            "Restart the ComfyUI process. The interface will reconnect " +
                "automatically once the restart completes.",
        ),
    );
    const restartBtn = el(
        "button",
        { class: "nukomfy-mgr-btn nukomfy-mgr-btn--danger-primary" },
        "Restart ComfyUI",
    );
    restartSection.appendChild(restartBtn);
    restartBtn.addEventListener("click", () => {
        if (restartBtn.disabled) return;
        confirmAndRunRestart({
            onComplete: (ok) => { if (ok) refresh(); },
        });
    });

    async function refresh() {
        const [info, availState] = await Promise.all([
            nukomfyPing(),
            nukomfyAvailability(),
        ]);
        _renderNodesBanner(nodesBanner, info);
        const versionStr = (info && info.version) || "";
        versionTag.textContent = versionStr ? `v${versionStr}` : "";
        const hasPwd = !!(info && info.has_password);
        const pwdChangedAt = (info && info.password_changed_at) || "";
        renderPasswordForm(formSection, hasPwd, pwdChangedAt, refresh);
        renderAvailabilitySection(
            availabilitySection,
            hasPwd,
            availState || { availability: "available" },
            refresh,
        );
        if (hasPwd) {
            activityBtn.disabled = false;
            activityBtn.removeAttribute("title");
            restartBtn.disabled = false;
            restartBtn.removeAttribute("title");
        } else {
            activityBtn.disabled = true;
            activityBtn.title = "Set an admin password first";
            restartBtn.disabled = true;
            restartBtn.title = "Set an admin password first";
        }
        renderHistorySection(historySection, info || {}, refresh);
    }
    await refresh();
}

export function setupSidebarTab(app) {
    if (!app?.extensionManager?.registerSidebarTab) {
        console.warn("[nukomfy.manager] sidebar tab API not available");
        return;
    }
    app.extensionManager.registerSidebarTab({
        id: "nukomfy-manager",
        icon: "nukomfy-mgr-tab-icon",
        title: "Nukomfy",
        tooltip: "Nukomfy Manager",
        type: "custom",
        render: (host) => { buildPanel(host, app); },
    });
}
