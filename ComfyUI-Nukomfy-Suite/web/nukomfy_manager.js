// Nukomfy Manager - extension entrypoint.
// Registers the sidebar tab and probes the server.

import { app } from "../../scripts/app.js";
import { setupSidebarTab } from "./sidebar_tab.js";
import { nukomfyPing } from "./api.js";

function injectStyles() {
    if (document.head.querySelector('link[data-nukomfy-mgr]')) return;
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/extensions/ComfyUI-Nukomfy-Suite/styles.css";
    link.dataset.nukomfyMgr = "1";
    document.head.appendChild(link);
}

app.registerExtension({
    name: "nukomfy.manager",
    async setup() {
        injectStyles();

        try { setupSidebarTab(app); }
        catch (e) { console.warn("[nukomfy.manager] sidebar tab failed:", e); }

        nukomfyPing().then((info) => {
            if (info) console.log("[nukomfy.manager] server reachable:", info);
            else console.warn("[nukomfy.manager] server ping returned no payload");
        });
    },
});
