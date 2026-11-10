// Lossless JSON parse / stringify for the job history viewer.
//
// Native JSON.parse rounds integers beyond 2^53 (ComfyUI seeds are 64-bit,
// so almost every random seed is affected) and normalises float literals
// (1.0 -> 1). The Detail view must show / copy / save the exact value that
// was submitted, so we keep the original numeric literal.
//
// We use the JSON.parse source-text reviver (the third "context" argument,
// ES2024 / Chromium 114+): for any number whose double round-trip differs
// from its source text we keep the source verbatim, wrapped in RawNumber so
// a matching faithfulStringify can re-emit it unquoted. On engines without
// source access we fall back to a plain parse (lossy) rather than failing.

export class RawNumber {
    constructor(literal) { this.literal = String(literal); }
    toString() { return this.literal; }
}

// integersOnly: preserve only big integer literals (seeds). Used for the
// history list, whose float fields (durations) must stay real numbers so
// arithmetic / formatting keeps working. Omit it for the workflow_api
// payload, where every numeric literal (incl. 1.0, 5e-7) must be faithful.
export function parseFaithful(text, { integersOnly = false } = {}) {
    try {
        return JSON.parse(text, function (key, value, context) {
            if (typeof value === "number"
                    && context && typeof context.source === "string"
                    && context.source !== String(value)) {
                if (integersOnly && !/^-?\d+$/.test(context.source)) {
                    return value;
                }
                return new RawNumber(context.source);
            }
            return value;
        });
    } catch (e) {
        // No source access (older engine) or malformed body: degrade to a
        // plain parse so the viewer still works, then null on hard failure.
        try { return JSON.parse(text); } catch (e2) { return null; }
    }
}

const _ESC = {
    "\"": "\\\"", "\\": "\\\\", "\b": "\\b", "\f": "\\f",
    "\n": "\\n", "\r": "\\r", "\t": "\\t",
};

function _quote(s) {
    s = String(s);
    let out = "\"";
    for (let i = 0; i < s.length; i++) {
        const c = s[i];
        const mapped = _ESC[c];
        if (mapped !== undefined) {
            out += mapped;
        } else if (s.charCodeAt(i) < 0x20) {
            out += "\\u" + s.charCodeAt(i).toString(16).padStart(4, "0");
        } else {
            out += c;
        }
    }
    return out + "\"";
}

// JSON.stringify-equivalent that emits RawNumber values as their raw literal
// (unquoted), so the pretty-printed dump and the saved .json keep exact
// 64-bit seeds and float formatting. `indent` is a space count.
export function faithfulStringify(value, indent) {
    const pad = typeof indent === "number" ? " ".repeat(indent) : (indent || "");
    function rec(v, cur) {
        if (v instanceof RawNumber) return v.literal;
        if (v === null) return "null";
        const t = typeof v;
        if (t === "number" || t === "boolean") return String(v);
        if (t === "string") return _quote(v);
        const next = cur + pad;
        if (Array.isArray(v)) {
            if (v.length === 0) return "[]";
            return "[\n" + v.map((e) => next + rec(e, next)).join(",\n")
                + "\n" + cur + "]";
        }
        if (t === "object") {
            const keys = Object.keys(v);
            if (keys.length === 0) return "{}";
            return "{\n" + keys.map(
                (k) => next + _quote(k) + ": " + rec(v[k], next)).join(",\n")
                + "\n" + cur + "}";
        }
        return "null"; // undefined / function: not valid JSON, drop to null
    }
    return rec(value, "");
}
