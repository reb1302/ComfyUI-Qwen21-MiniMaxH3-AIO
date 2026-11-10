// Display labels, sizing and dynamic output sockets for
// NukomfyMultiLayerUnpack.
//
// Mirror of multilayer_pack_dynamic.js for labels + sizing. In
// addition, the Unpack trims its visible output sockets to match
// the user-selected Layer Count. The Python schema declares
// _MAX_LAYERS (16) static outputs (necessary so the v3 execute()
// return arity stays static), but the schema-driven Vue (Nodes 2.0)
// layout would otherwise size the node for 16 sockets even when the
// user only uses one. Trimming nodeData.output in
// beforeRegisterNodeDef tells Vue to render a single output by
// default; syncOutputs() in onNodeCreated grows/shrinks the live
// node.outputs list to match Layer Count, mirroring how the Pack
// grows its inputs via the official DynamicCombo schema.
//
// Mapping:
//   extra_layers      -> "extra_layers"  (lowercase, Comfy socket convention)
//   layer_count       -> "Layer Count"
//   layer_N_name      -> "Layer N Name"  (widget convention)
//   layer_N (output)  -> "layer_N"       (lowercase, socket convention)

import { app } from "/scripts/app.js";


function baseName(widgetName) {
    return widgetName.includes(".") ? widgetName.split(".").pop() : widgetName;
}


function relabel(obj) {
    if (!obj || !obj.name) return;
    const base = baseName(obj.name);
    if (base === "layer_count") {
        obj.label = "Layer Count";
        return;
    }
    if (base === "extra_layers") {
        obj.label = "extra_layers";
        return;
    }
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
    for (const w of node.widgets || []) {
        relabel(w);
    }
    for (const inp of node.inputs || []) {
        relabel(inp);
    }
    for (const out of node.outputs || []) {
        relabel(out);
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


function parseLayerCount(value) {
    // Layer Count widget value looks like "1 layer" or "3 layers".
    if (!value) return 1;
    const m = String(value).match(/^(\d+)/);
    return m ? parseInt(m[1], 10) : 1;
}


function hookLayerNameSnapshot(node) {
    // Persistent per-slot snapshot store. Each layer_<i>_name widget
    // gets a small hook on its callback so any user edit is saved into
    // node._userTypedLayerNames[<i>]. The IO.DynamicCombo rebuild on
    // Layer Count change wipes the widgets themselves, but the store
    // survives and is then replayed by restore().
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


function syncOutputs(node, count) {
    while (node.outputs.length > count) {
        node.removeOutput(node.outputs.length - 1);
    }
    while (node.outputs.length < count) {
        const idx = node.outputs.length + 1;
        node.addOutput(`layer_${idx}`, "IMAGE");
    }
    for (const out of node.outputs) {
        relabel(out);
    }
    node.setDirtyCanvas(true, true);
}


// Match width with the other Suite nodes so they share a footprint.
const NODE_WIDTH = 285;


app.registerExtension({
    name: "Nukomfy.MultilayerUnpackDynamic",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== "NukomfyMultiLayerUnpack") return;

        // Tell the schema-driven layout (Vue Nodes 2.0 reads nodeData
        // here) that the node has a single output by default. The
        // Python execute() still returns _MAX_LAYERS tensors; the JS
        // syncOutputs() in onNodeCreated grows/shrinks the runtime
        // node.outputs list to match Layer Count.
        if (Array.isArray(nodeData.output) && nodeData.output.length > 1) {
            nodeData.output = nodeData.output.slice(0, 1);
            if (Array.isArray(nodeData.output_name)) {
                nodeData.output_name = nodeData.output_name.slice(0, 1);
            }
            if (Array.isArray(nodeData.output_is_list)) {
                nodeData.output_is_list = nodeData.output_is_list.slice(0, 1);
            }
        }

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

            const widget = findWidget(node, "layer_count");
            syncOutputs(node, parseLayerCount(widget && widget.value));
            hookLayerNameSnapshot(node);
            applyLabels(node);

            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    if (LiteGraph.vueNodesMode) return;
                    if (!node._userResized && !node._serializedSize) {
                        node._programmaticResize = true;
                        node.setSize([NODE_WIDTH, node.size[1]]);
                        node._programmaticResize = false;
                    }
                });
            });
            setTimeout(() => {
                const originalResize = node.onResize;
                node.onResize = function () {
                    // Only flag userResized for genuine drag events, not
                    // programmatic setSize calls triggered by Layer Count
                    // changes. Otherwise the first syncOutputs() at mount
                    // fires onResize and blocks every subsequent fit.
                    if (!node._programmaticResize) {
                        node._userResized = true;
                    }
                    if (originalResize) return originalResize.apply(this, arguments);
                };
            }, 100);

            chainCallback(widget, (value) => {
                // Persistent restore: hookLayerNameSnapshot(node) keeps
                // node._userTypedLayerNames up to date every time the
                // user edits a slot; restore() reapplies those values
                // to the freshly rebuilt widgets after the framework
                // swaps the option set.
                const restore = () => {
                    if (!node._userTypedLayerNames) return;
                    for (const w of node.widgets || []) {
                        const m = baseName(w.name || "").match(/^layer_(\d+)_name$/);
                        if (m && node._userTypedLayerNames[m[1]] && !w.value) {
                            w.value = node._userTypedLayerNames[m[1]];
                        }
                    }
                };
                syncOutputs(node, parseLayerCount(value));
                const scheduleRestore = (ms) => setTimeout(() => {
                    hookLayerNameSnapshot(node);
                    restore();
                    applyLabels(node);
                }, ms);
                scheduleRestore(0);
                scheduleRestore(50);
                scheduleRestore(200);
                // Always recompute height on Layer Count change. Apply
                // setSize synchronously first (instant fit in legacy
                // LiteGraph), then re-apply after two animation frames
                // for Vue (Nodes 2.0) which settles its layout async.
                const fit = () => {
                    const sz = node.computeSize();
                    // In Vue (Nodes 2.0) preserve the renderer's natural
                    // width (wider padding/font than legacy LiteGraph);
                    // only legacy needs NODE_WIDTH pinning.
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

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = originalConfigure ? originalConfigure.apply(this, arguments) : undefined;
            this._serializedSize = true;
            const node = this;
            setTimeout(() => {
                const w = findWidget(node, "layer_count");
                syncOutputs(node, parseLayerCount(w && w.value));
                hookLayerNameSnapshot(node);
                applyLabels(node);
            }, 0);
            return ret;
        };
    },
});
