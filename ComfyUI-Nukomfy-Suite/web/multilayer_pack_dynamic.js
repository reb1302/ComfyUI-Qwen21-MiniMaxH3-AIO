// Display labels for NukomfyMultiLayerPack.
//
// The ComfyUI frontend ignores schema `display_name` for nested
// IO.DynamicCombo inputs (same issue documented in io_raw_data_toggle.js
// for NukomfyWrite). This extension forces the user-facing labels by
// setting widget.label after node creation and on configure / count
// change events.
//
// Mapping:
//   layer_count       -> "Layer Count"
//   layer_N           -> "layer_N"        (lowercase, Comfy socket convention)
//   layer_N_name      -> "Layer N Name"   (Title Case, widget convention)

import { app } from "/scripts/app.js";


function baseName(widgetName) {
    // Nested DynamicCombo widgets may appear with or without the
    // "layer_count." prefix depending on ComfyUI frontend version.
    return widgetName.includes(".") ? widgetName.split(".").pop() : widgetName;
}


function relabel(obj) {
    // Works for both widgets (node.widgets[]) and sockets (node.inputs[]).
    // Both expose .name (the schema id) and .label (the canvas display).
    if (!obj || !obj.name) return;
    const base = baseName(obj.name);
    if (base === "layer_count") {
        obj.label = "Layer Count";
        return;
    }
    // Image sockets (layer_1, layer_2, ...) keep lowercase snake_case
    // display to match Comfy convention (image, extra_layers, latent, ...).
    let m = base.match(/^layer_(\d+)$/);
    if (m) {
        obj.label = `layer_${m[1]}`;
        return;
    }
    m = base.match(/^layer_(\d+)_name$/);
    if (m) {
        obj.label = `Layer ${m[1]} Name`;
        return;
    }
}


function applyLabels(node) {
    // Widget labels (text/combo widgets shown below sockets).
    for (const w of node.widgets || []) {
        relabel(w);
    }
    // Input socket labels (image sockets shown at the top of the node).
    for (const inp of node.inputs || []) {
        relabel(inp);
    }
    node.setDirtyCanvas(true, false);
}


function chainCallback(widget, fn) {
    if (!widget) return;
    const original = widget.callback;
    widget.callback = function (value) {
        fn(value);
        if (original) return original.apply(this, arguments);
    };
}


function findWidget(node, name) {
    if (!node.widgets) return null;
    return node.widgets.find((w) => w.name === name)
        || node.widgets.find((w) => baseName(w.name) === name);
}


function hookLayerNameSnapshot(node) {
    // Persistent per-slot snapshot. Hooks each layer_<i>_name widget
    // so user edits are saved to node._userTypedLayerNames; this lets
    // restore() replay them after IO.DynamicCombo rebuilds the option
    // widget set on Layer Count change.
    if (!node._userTypedLayerNames) node._userTypedLayerNames = {};
    for (const w of node.widgets || []) {
        const m = baseName(w.name || "").match(/^layer_(\d+)_name$/);
        if (m && !w._nukomfyLayerNameHooked) {
            w._nukomfyLayerNameHooked = true;
            const slot = m[1];
            if (w.value) node._userTypedLayerNames[slot] = w.value;
            const orig = w.callback;
            w.callback = function (value) {
                node._userTypedLayerNames[slot] = value;
                if (orig) return orig.apply(this, arguments);
            };
        }
    }
}


// Match width with NukomfyRead / NukomfyWrite so the IO + helper nodes
// share the same footprint in the canvas.
const NODE_WIDTH = 285;


app.registerExtension({
    name: "Nukomfy.MultilayerPackDynamic",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== "NukomfyMultiLayerPack") return;

        // Pin the LiteGraph-computed width to NODE_WIDTH even when the
        // nested IO.DynamicCombo expands and adds widgets at runtime
        // (changing Layer Count from N to N+1 adds a layer_name +
        // layer_image pair). In legacy LiteGraph this is the only way
        // to prevent the auto-grow. In Vue (Nodes 2.0) this prototype
        // hook is ignored by the Vue node renderer, which reads the
        // width from its own layout store; the override is harmless
        // there.
        const originalComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (out) {
            const size = originalComputeSize
                ? originalComputeSize.call(this, out)
                : [200, 200];
            size[0] = NODE_WIDTH;
            return size;
        };

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = originalCreated ? originalCreated.apply(this, arguments) : undefined;
            const node = this;
            hookLayerNameSnapshot(node);
            applyLabels(node);

            // Apply NODE_WIDTH after the layout settles. Two nested
            // requestAnimationFrame calls ensure both the LiteGraph
            // backing model and the Vue node component (Nodes 2.0)
            // have completed their initial layout pass before we call
            // setSize. setTimeout(0) is too early in Vue mode.
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    // Skip the canvas-tuned NODE_WIDTH in Vue (Nodes 2.0):
                    // the Vue node renderer uses wider padding and a
                    // larger default font, so the 285px tuned for the
                    // canvas LiteGraph rendering truncates labels there.
                    // Letting the Vue layout pick the natural width
                    // produces readable labels by default.
                    if (LiteGraph.vueNodesMode) return;
                    if (!node._userResized && !node._serializedSize) {
                        node._programmaticResize = true;
                        node.setSize([NODE_WIDTH, node.size[1]]);
                        node._programmaticResize = false;
                    }
                });
            });
            // Attach the onResize listener AFTER the initial layout has
            // stabilized. In Vue mode the mount triggers a spurious
            // onResize that would otherwise flip _userResized and skip
            // the setSize above.
            setTimeout(() => {
                const originalResize = node.onResize;
                node.onResize = function () {
                    if (!node._programmaticResize) {
                        node._userResized = true;
                    }
                    if (originalResize) return originalResize.apply(this, arguments);
                };
            }, 100);

            // Re-apply labels + restore layer_N_name values whenever
            // Layer Count changes. IO.DynamicCombo rebuilds the option
            // widgets on each change, wiping any text the user typed in
            // layer_<i>_name. Snapshot before the framework rebuilds,
            // then restore on the next frame.
            chainCallback(findWidget(node, "layer_count"), () => {
                const restore = () => {
                    if (!node._userTypedLayerNames) return;
                    for (const w of node.widgets || []) {
                        const m = baseName(w.name || "").match(/^layer_(\d+)_name$/);
                        if (m && node._userTypedLayerNames[m[1]] && !w.value) {
                            w.value = node._userTypedLayerNames[m[1]];
                        }
                    }
                };
                const scheduleRestore = (ms) => setTimeout(() => {
                    hookLayerNameSnapshot(node);
                    restore();
                    applyLabels(node);
                }, ms);
                scheduleRestore(0);
                scheduleRestore(50);
                scheduleRestore(200);
                // Always recompute height on Layer Count change. Sync
                // setSize first (instant in legacy LiteGraph), then
                // re-apply on two RAFs for Vue layout settling.
                const fit = () => {
                    const sz = node.computeSize();
                    // Vue (Nodes 2.0) keeps its natural width; only
                    // legacy LiteGraph needs NODE_WIDTH pinning.
                    const targetWidth = LiteGraph.vueNodesMode ? node.size[0] : NODE_WIDTH;
                    node._programmaticResize = true;
                    node.setSize([targetWidth, sz[1]]);
                    node._programmaticResize = false;
                };
                fit();
                if (app.graph) app.graph.setDirtyCanvas(true, true);
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        restore();
                        fit();
                    });
                });
            });
            return ret;
        };

        // Workflow load + undo/redo. Also mark the node as carrying a
        // serialized size to gate the RAF default-width pass.
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = originalConfigure ? originalConfigure.apply(this, arguments) : undefined;
            this._serializedSize = true;
            const node = this;
            setTimeout(() => {
                hookLayerNameSnapshot(node);
                applyLabels(node);
            }, 0);
            return ret;
        };
    },
});
