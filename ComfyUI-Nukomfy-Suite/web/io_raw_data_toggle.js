// Conditional knob state for NukomfyWrite:
//   - Gray out Output Transform when Apply Color Transform is OFF.
//   - Hide DW Compression Level when Compression is not DWAA / DWAB.
//   - Force display labels on nested DynamicCombo widgets (the ComfyUI
//     frontend ignores schema `display_name` for nested IO.DynamicCombo
//     inputs; setting `widget.label` here keeps the Nuke-style labels).
//
// IO.DynamicCombo handles file_type-driven visibility of format-specific
// knobs server-side.

import { app } from "/scripts/app.js";

// Snake_case widget id -> Nuke-style display label. Widget ids are
// format-prefixed (e.g. exr_datatype, png_datatype) so each DynamicCombo
// option owns a unique identifier; the label shown to the user stays
// short ("Data Type") regardless of the prefix.
const NESTED_LABELS = {
    "exr_datatype": "Data Type",
    "dpx_datatype": "Data Type",
    "png_datatype": "Data Type",
    "tiff_datatype": "Data Type",
    "exr_compression": "Compression",
    "tga_compression": "Compression",
    "tiff_compression": "Compression",
    "exr_dw_compression_level": "Compression Level",
    "jpeg_quality": "Quality",
    "jpeg_sub_sampling": "Sub-sampling",
    "dpx_transfer": "Transfer",
};

// Widget ids that act as compression masters per file format. Used to
// re-attach the applyState chainCallback whenever the DynamicCombo
// swaps which compression widget is mounted.
const COMPRESSION_WIDGET_IDS = ["exr_compression", "tga_compression", "tiff_compression"];


function baseName(widgetName) {
    // Nested DynamicCombo widgets may appear with or without the
    // "file_type." prefix depending on ComfyUI frontend version.
    return widgetName.includes(".") ? widgetName.split(".").pop() : widgetName;
}


function findWidget(node, name) {
    if (!node.widgets) return null;
    return node.widgets.find((w) => w.name === name)
        || node.widgets.find((w) => baseName(w.name) === name);
}


function applyLabels(node) {
    for (const w of node.widgets || []) {
        const label = NESTED_LABELS[baseName(w.name)];
        // On BaseWidget (Vue Nodes 2.0) `label` is a setter onto the reactive
        // store entry, so this drives the rendered label; on legacy widgets it
        // sets the plain property the canvas renderer reads. Either way one
        // assignment is enough.
        if (label) w.label = label;
    }
}


// Attach the label/state refresh to the CURRENT compression widget. The
// DynamicCombo replaces this widget object whenever file_type or the
// compression option changes AND on reload (onConfigure re-applies the saved
// value, remounting it), so a hook bound once in onNodeCreated is lost - which
// is why switching compression after loading a saved workflow left the nested
// sub with its raw id. The flag prevents wrapping the same widget object twice.
function hookCompressionWidgets(node) {
    for (const wname of COMPRESSION_WIDGET_IDS) {
        const w = findWidget(node, wname);
        if (w && !w.__nfyLabelHooked) {
            w.__nfyLabelHooked = true;
            // The DynamicCombo assigns its value - which mounts the nested sub
            // (e.g. exr_dw_compression_level) - BEFORE it fires this callback,
            // so the sub already exists on node.widgets here. Relabel it
            // synchronously; no timer / mount-race guessing. Re-pin the width
            // too: the remount can auto-grow the node in legacy canvas.
            chainCallback(w, () => {
                applyState(node);
                repinWidth(node);
            });
        }
    }
}


function applyState(node) {
    applyLabels(node);

    const applyTransform = findWidget(node, "apply_color_transform");
    const outputTransform = findWidget(node, "output_transform");
    if (applyTransform && outputTransform) {
        // Output Transform is enabled only when the toggle is checked.
        outputTransform.disabled = !applyTransform.value;
    }

    // DW Compression Level visibility is handled server-side via the
    // nested IO.DynamicCombo on the compression knob (DWAA/DWAB options
    // expose dw_compression_level; other codecs don't).
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


// Shared with io_read_dynamic.js. Width matched to the minimum the user
// can manually shrink the Write node to while keeping DW Compression Level
// fully readable (file_type=exr + compression=DWAA). Matches NukomfyRead.
const NODE_WIDTH = 285;


// Re-pin the node to NODE_WIDTH after the IO.DynamicCombo remounts the
// format / compression option widgets: changing file_type or the EXR
// compression codec swaps the widget set and lets legacy LiteGraph auto-grow
// the node past NODE_WIDTH. Legacy canvas only - in Vue (Nodes 2.0) the
// renderer owns the width and gives the natural, readable one, so this is a
// no-op there. The _programmaticResize guard keeps this resize from being
// mistaken for a user drag by the onResize wrapper. Since the computeSize
// override already forces NODE_WIDTH, only the height it returns is used here.
function pinWidth(node) {
    if (LiteGraph.vueNodesMode) return;
    const sz = node.computeSize();
    node._programmaticResize = true;
    node.setSize([NODE_WIDTH, sz[1]]);
    node._programmaticResize = false;
}


// The remount can settle across frames, so re-pin synchronously and again
// after two animation frames (mirrors multilayer_pack_dynamic.js).
function repinWidth(node) {
    pinWidth(node);
    requestAnimationFrame(() => requestAnimationFrame(() => pinWidth(node)));
}


app.registerExtension({
    name: "Nukomfy.WriteDynamic",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== "NukomfyWrite") return;

        // Pin the LiteGraph-computed width to NODE_WIDTH even when the
        // nested IO.DynamicCombo expands and adds widgets at runtime
        // (e.g. DW Compression Level appears for DWAA / DWAB). In legacy
        // LiteGraph this is the only way to prevent the auto-grow.
        // In Vue (Nodes 2.0) this prototype hook is ignored by the Vue
        // node renderer, which reads the width from its own layout store
        // updated via setSize calls; the override is harmless there.
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
            applyState(node);

            // Node-width pinning is a legacy-canvas concern only: in Vue Nodes
            // 2.0 the renderer owns the width (and gives the natural width we
            // want), so setSize below is a no-op there and its guard machinery
            // is moot - the whole async width dance must not even run. In
            // legacy the layout size is only known after the render pass, so
            // the deferral is intrinsic: the two RAFs wait for layout to
            // settle, and the delayed onResize attach avoids treating the
            // initial mount resize as a user drag. Same shape as
            // io_read_dynamic.js, which is only reached in legacy mode.
            if (!LiteGraph.vueNodesMode) {
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
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
                        if (!node._programmaticResize) {
                            node._userResized = true;
                        }
                        if (originalResize) return originalResize.apply(this, arguments);
                    };
                }, 100);
            }

            chainCallback(findWidget(node, "apply_color_transform"), () => applyState(node));
            hookCompressionWidgets(node);
            chainCallback(findWidget(node, "file_type"), () => {
                // file_type assigns its value (remounting the whole option
                // group, so the compression widget is a new object) before
                // firing this callback - re-hook and relabel synchronously.
                hookCompressionWidgets(node);
                applyState(node);
                repinWidth(node);
            });
            return ret;
        };

        // onConfigure fires on workflow load AND on undo/redo. Without this
        // hook the labels + DW Compression Level hidden state would be
        // lost on Ctrl+Z and the user would see the raw widget ids
        // (e.g. exr_datatype / exr_compression / exr_dw_compression_level)
        // instead of the friendly labels. Also mark the node as carrying
        // a serialized size to gate the RAF default-width pass in
        // onNodeCreated.
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = originalConfigure ? originalConfigure.apply(this, arguments) : undefined;
            this._serializedSize = true;
            // configure() applies widgets_values (mounting the saved option's
            // nested subs) before calling onConfigure, so the subs exist now:
            // relabel them and re-hook the compression widget (a new object
            // after reload, otherwise switching compression later never
            // refreshes the nested sub label) - all synchronous.
            hookCompressionWidgets(this);
            applyState(this);
            return ret;
        };
    },
});
