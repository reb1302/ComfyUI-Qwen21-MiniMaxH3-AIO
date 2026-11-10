// Shared modal infrastructure for the Nukomfy Manager sidebar.
// Two surfaces:
//   - openModal(opts): generic overlay dialog (used by the history list +
//     job detail viewers).
//   - openPasswordConfirmModal(opts): purpose-built password confirmation
//     popup used for every privileged action (toggle availability, toggle
//     history, save settings, restart). The title is always prefixed with
//     "Nukomfy Manager -" so it's unmistakable in stacked-modal contexts.
//
// Both honour ESC and click-outside-to-close. Multiple open modals are
// stacked: ESC closes the topmost only.

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

const _modalStack = [];

export function openModal({ title, content, width, height, onClose }) {
    const overlay = el("div", { class: "nukomfy-mgr-modal-overlay" });
    const dialog = el("div", { class: "nukomfy-mgr-modal-dialog" });
    if (width) dialog.style.width = typeof width === "number" ? `${width}px` : width;
    if (height) dialog.style.height = typeof height === "number" ? `${height}px` : height;

    const closeBtn = el("button", {
        class: "nukomfy-mgr-modal-close",
        title: "Close (Esc)",
    }, "×");
    const header = el("div", { class: "nukomfy-mgr-modal-header" }, [
        el("h3", { class: "nukomfy-mgr-modal-title" }, title || ""),
        closeBtn,
    ]);
    const body = el("div", { class: "nukomfy-mgr-modal-body" }, content || "");
    dialog.appendChild(header);
    dialog.appendChild(body);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    let closed = false;
    const handle = {
        overlay, dialog, body,
        close() {
            if (closed) return;
            closed = true;
            const idx = _modalStack.indexOf(handle);
            if (idx >= 0) _modalStack.splice(idx, 1);
            try { document.body.removeChild(overlay); } catch (e) { /* gone */ }
            if (typeof onClose === "function") {
                try { onClose(); } catch (e) { /* swallow */ }
            }
        },
    };
    closeBtn.addEventListener("click", () => handle.close());

    // Close on click-outside only when both mousedown AND mouseup land on
    // the overlay itself. This prevents resize drags (mousedown on the
    // dialog's resize handle, mouseup that strays onto the overlay) from
    // being mis-interpreted as a dismiss click.
    let downOnOverlay = false;
    overlay.addEventListener("mousedown", (ev) => {
        downOnOverlay = ev.target === overlay;
    });
    overlay.addEventListener("mouseup", (ev) => {
        if (downOnOverlay && ev.target === overlay) handle.close();
        downOnOverlay = false;
    });

    _modalStack.push(handle);
    return handle;
}

// Single global ESC handler - closes the topmost open modal only.
document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && _modalStack.length > 0) {
        _modalStack[_modalStack.length - 1].close();
        ev.stopPropagation();
    }
});

// Map a {status, body} response into a human-readable error string for
// the standard rate-limit / wrong-password / unreachable cases.
export function mapAuthError(res) {
    if (!res) return "Cannot reach server.";
    const { status, body } = res;
    if (status === 401) return "Wrong admin password.";
    if (status === 429) {
        const retry = (body && body.retry_after_seconds) || 0;
        return `Too many failed attempts. Try again in ${retry}s.`;
    }
    if (status === 400 && body && body.reason === "no_password_set") {
        return "Set an admin password first.";
    }
    if (status === 400 && body && body.reason === "invalid_value") {
        return "Server rejected the values.";
    }
    if (status === 0) return "Cannot reach server.";
    return "Action failed. Try again.";
}

// Password-confirmation modal. The title is always namespaced with
// "Nukomfy Manager -" so users can tell where the prompt comes from
// even when other extensions display their own dialogs.
//
// onConfirm receives (password, ctx) where ctx exposes:
//   ctx.close()                  - dismiss the modal
//   ctx.showError(message)       - surface an inline error banner
//   ctx.setBusy(text)            - replace the form with a spinner row
//                                  (caller is responsible for calling
//                                  close() once the long-running action
//                                  completes)
//
// Helpful conventions:
//   - On success, the caller is expected to ctx.close() explicitly.
//   - On error, the caller calls ctx.showError(mapAuthError(res)) or a
//     custom string. The confirm button re-enables automatically.
export function openPasswordConfirmModal({
    title,
    description,
    actionLabel,
    onConfirm,
    onCancel,
}) {
    const banner = el("div", { class: "nukomfy-mgr-banner" });
    banner.style.display = "none";

    const pwdInput = el("input", {
        type: "password",
        class: "nukomfy-mgr-pw-modal-input",
        placeholder: "Confirm with admin password",
        autocomplete: "current-password",
    });

    const cancelBtn = el("button", { class: "nukomfy-mgr-btn" }, "Cancel");
    const confirmBtn = el("button", {
        class: "nukomfy-mgr-btn nukomfy-mgr-btn--primary",
        disabled: "true",
    }, actionLabel || "Confirm");

    const formBody = el("div", { class: "nukomfy-mgr-pw-modal-body" }, [
        description
            ? el("p", { class: "nukomfy-mgr-pw-modal-description" }, description)
            : null,
        el("label", { class: "nukomfy-mgr-pw-modal-label" }, "Admin password"),
        pwdInput,
        banner,
        el("div", { class: "nukomfy-mgr-actions" }, [cancelBtn, confirmBtn]),
    ]);

    const fullTitle = title
        ? `Nukomfy Manager - ${title}`
        : "Nukomfy Manager";
    let didSucceed = false;
    const modal = openModal({
        title: fullTitle,
        content: formBody,
        width: 480,
        height: "auto",
        onClose: () => {
            if (!didSucceed && typeof onCancel === "function") {
                try { onCancel(); } catch (e) { /* swallow */ }
            }
        },
    });
    // Lock the dialog to its content height so the description + form
    // never feel cramped while remaining compact for short prompts.
    modal.dialog.style.height = "auto";

    const ctx = {
        close: () => {
            didSucceed = true;
            modal.close();
        },
        showError: (msg) => {
            banner.className = "nukomfy-mgr-banner nukomfy-mgr-banner--err";
            banner.textContent = msg || "Action failed.";
            banner.style.display = "block";
            confirmBtn.removeAttribute("disabled");
            pwdInput.focus();
            pwdInput.select();
        },
        clearError: () => {
            banner.style.display = "none";
            banner.textContent = "";
        },
        setBusy: (text) => {
            formBody.innerHTML = "";
            const spinner = el("div", { class: "nukomfy-mgr-pw-modal-busy" }, [
                el("p", { class: "nukomfy-mgr-placeholder" }, text || "Working…"),
            ]);
            formBody.appendChild(spinner);
        },
    };

    pwdInput.addEventListener("input", () => {
        if (pwdInput.value) confirmBtn.removeAttribute("disabled");
        else confirmBtn.setAttribute("disabled", "true");
    });
    pwdInput.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" && !confirmBtn.hasAttribute("disabled")) {
            ev.preventDefault();
            confirmBtn.click();
        }
    });
    cancelBtn.addEventListener("click", () => modal.close());

    confirmBtn.addEventListener("click", async () => {
        if (!pwdInput.value) return;
        ctx.clearError();
        confirmBtn.setAttribute("disabled", "true");
        const password = pwdInput.value;
        try {
            await onConfirm(password, ctx);
        } catch (e) {
            console.warn("[nukomfy.manager] password modal handler threw:", e);
            ctx.showError("Unexpected error. Check the console.");
        }
    });

    // Autofocus once the dialog is in the DOM.
    setTimeout(() => pwdInput.focus(), 0);

    return modal;
}
