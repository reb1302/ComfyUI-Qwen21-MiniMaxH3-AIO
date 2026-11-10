// Fixed initial width for NukomfyOCIOColorSpace. Sized to fit OCIO
// colorspace names like "sRGB Encoded Rec.709 (sRGB)" or "Linear Rec.709
// (sRGB)" without truncation. Wider than NukomfyRead / NukomfyWrite (285)
// because the combo values are longer.
//
// Pattern: deferred setSize via requestAnimationFrame x2 (Vue Nodes 2.0
// compatible) plus a delayed onResize listener attach so the spurious
// resize fired during Vue mount does not flip _userResized.

import { app } from "/scripts/app.js";

const NODE_WIDTH = 340;


app.registerExtension({
    name: "Nukomfy.OCIOColorSpaceDynamic",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== "NukomfyOCIOColorSpace") return;

        // Pin the LiteGraph-computed width to NODE_WIDTH. Same rationale
        // as NukomfyRead / NukomfyWrite: keeps the LiteGraph rendered
        // width pixel-aligned across the Suite. In Vue (Nodes 2.0) the
        // override is ignored but harmless.
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

            // Apply NODE_WIDTH after the layout settles. Two nested
            // requestAnimationFrame calls ensure both the LiteGraph
            // backing model and the Vue node component (Nodes 2.0)
            // have completed their initial layout pass before we call
            // setSize.
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    // Skip the canvas-tuned NODE_WIDTH in Vue (Nodes 2.0):
                    // the Vue node renderer uses wider padding and a
                    // larger default font, so the 340px tuned for the
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
            // io_read_dynamic.js / multilayer_pack_dynamic.js so a scripted
            // resize is not mistaken for a user drag; NukomfyOCIOColorSpace
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

            return ret;
        };

        // Mark this node as carrying a serialized size when loaded from
        // a workflow, so the RAF default-width pass in onNodeCreated
        // does not overwrite the saved width.
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const ret = originalConfigure ? originalConfigure.apply(this, arguments) : undefined;
            this._serializedSize = true;
            return ret;
        };
    },
});
