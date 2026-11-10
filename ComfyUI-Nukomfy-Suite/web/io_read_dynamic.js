// Gray-out (disable, not hide) conditional knobs on NukomfyRead:
//   - First Frame / Last Frame are still visible but disabled when
//     Read Mode is "Input Range".
//   - Input Transform is still visible but disabled when Apply Color
//     Transform is OFF.
// Knobs are kept visible (not hidden) so the user can see what would
// apply if the upstream toggle were flipped.

import { app } from "/scripts/app.js";

// Missing Frames is NOT in this list: gaps inside a detected sequence are
// resolved by the policy regardless of mode (Input Range or Custom Range).
const READ_MODE_DEPENDENT = ["first_frame", "last_frame"];

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function setDisabled(widget, disabled) {
    if (!widget) return;
    widget.disabled = !!disabled;
}

function applyDisabling(node) {
    const readMode = findWidget(node, "read_mode");
    const applyTransform = findWidget(node, "apply_color_transform");
    const inputTransform = findWidget(node, "input_transform");

    const isInputRange = readMode ? readMode.value === "Input Range" : false;
    for (const name of READ_MODE_DEPENDENT) {
        setDisabled(findWidget(node, name), isInputRange);
    }

    if (applyTransform && inputTransform) {
        // Input Transform is enabled only when the toggle is checked.
        setDisabled(inputTransform, !applyTransform.value);
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



// Matched to NukomfyWrite so the two IO nodes share the same footprint.
const NODE_WIDTH = 285;


app.registerExtension({
    name: "Nukomfy.ReadDynamic",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== "NukomfyRead") return;

        // Pin the LiteGraph-computed width to NODE_WIDTH. NukomfyRead has
        // no IO.DynamicCombo so its widget set is static, but applying
        // the same prototype override as NukomfyWrite keeps the two
        // nodes' rendered widths perfectly aligned in legacy LiteGraph
        // (without it the LiteGraph computeSize default produces a
        // 1-2px discrepancy vs. NukomfyWrite). In Vue (Nodes 2.0) this
        // hook is ignored by the Vue node renderer; the override is
        // harmless there.
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
            applyDisabling(node);

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
            // the setSize above. The _programmaticResize guard mirrors
            // io_raw_data_toggle.js / multilayer_pack_dynamic.js so a
            // scripted resize is not mistaken for a user drag; NukomfyRead
            // has no IO.DynamicCombo, so there is no per-change re-pin to add.
            setTimeout(() => {
                const originalResize = node.onResize;
                node.onResize = function () {
                    if (!node._programmaticResize) {
                        node._userResized = true;
                    }
                    if (originalResize) return originalResize.apply(this, arguments);
                };
            }, 100);

            chainCallback(findWidget(node, "read_mode"), () => applyDisabling(node));
            chainCallback(findWidget(node, "apply_color_transform"), () => applyDisabling(node));
            return ret;
        };

        // Re-apply disabled state on workflow load AND undo/redo.
        // Also mark the node as carrying a serialized size: this gates
        // the RAF default-width pass in onNodeCreated so saved workflow
        // sizes are not overwritten.
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = originalConfigure ? originalConfigure.apply(this, arguments) : undefined;
            this._serializedSize = true;
            setTimeout(() => applyDisabling(this), 0);
            return ret;
        };
    },
});
