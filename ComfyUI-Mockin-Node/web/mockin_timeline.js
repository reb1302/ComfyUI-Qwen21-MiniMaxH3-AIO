import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const UNCENSORED_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition";

function getCleanDefaultState() {
    return {
        duration: 5.0,
        style: "",
        camera_style: "",
        audio_style: "",
        objects: [
            { tag: "<image1>", desc: "" },
            { tag: "<image2>", desc: "" },
            { tag: "<audio1>", desc: "" },
        ],
        keyframes: [
            {
                start: 0.0,
                end: 2.2,
                type: "CUT 1",
                tags: "<image1>, <audio1>",
                camera: "",
                sound: "",
                action: "",
            },
            {
                start: 2.2,
                end: 2.8,
                type: "TRANSITION",
                tags: "",
                camera: "",
                sound: "",
                action: "",
            },
            {
                start: 2.8,
                "end": 5.0,
                type: "CUT 2",
                tags: "<image2>, <audio1>",
                camera: "",
                sound: "",
                action: "",
            },
        ],
    };
}

function round1(val) {
    return Math.round((parseFloat(val) || 0) * 10) / 10;
}

app.registerExtension({
    name: "Mockin.TimelineDirector",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MockinTimelineNode") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            const wTimelineData = node.widgets?.find((w) => w.name === "timeline_data");
            const wFinalPrompt = node.widgets?.find((w) => w.name === "final_prompt");
            const wModel = node.widgets?.find((w) => w.name === "model");
            const wApiKey = node.widgets?.find((w) => w.name === "api_key");

            if (wModel && (!wModel.value || wModel.value.includes("deepseek"))) {
                wModel.value = UNCENSORED_MODEL;
            }

            if (wTimelineData) {
                wTimelineData.type = "hidden";
                wTimelineData.computeSize = () => [0, -4];
            }

            let state = getCleanDefaultState();

            function cleanOldSampleData(st) {
                if (!st || typeof st !== "object") return;
                if (st.style && (st.style.includes("Bold comic-book") || st.style.includes("heavy linework"))) {
                    st.style = "";
                }
                if (typeof st.camera_style !== "string") st.camera_style = "";
                if (typeof st.audio_style !== "string") st.audio_style = "";
                if (Array.isArray(st.objects)) {
                    st.objects.forEach((o) => {
                        if (
                            o.desc &&
                            (o.desc.includes("Cậu bé siêu nhân") ||
                                o.desc.includes("Quái vật kaiju") ||
                                o.desc.includes("GET READY TO MEET") ||
                                o.desc.includes("Nhân vật / Đối tượng"))
                        ) {
                            o.desc = "";
                        }
                    });
                }
                if (Array.isArray(st.keyframes)) {
                    st.keyframes.forEach((k) => {
                        if (typeof k.camera !== "string") k.camera = "";
                        if (typeof k.sound !== "string") k.sound = "";
                        if (
                            k.action &&
                            (k.action.includes("GET READY TO") ||
                                k.action.includes("cậu bé chống hông") ||
                                k.action.includes("quái vật khổng lồ") ||
                                k.action.includes("Lia máy cực mạnh (violent WHIP PAN) khỏi sân thượng"))
                        ) {
                            k.action = "";
                        }
                    });
                }
            }

            // Reconcile timeline math: ripple-chain start/end or auto-fit total duration
            function reconcileTimelineMath(changedIdx = -1, field = "") {
                const dur = Math.max(1.0, round1(state.duration || 5.0));
                state.duration = dur;
                if (!Array.isArray(state.keyframes) || state.keyframes.length === 0) return;

                // Ensure first starts at 0 if not explicitly edited
                if (changedIdx !== 0 || field !== "start") {
                    state.keyframes[0].start = 0.0;
                }

                for (let i = 0; i < state.keyframes.length; i++) {
                    const kf = state.keyframes[i];
                    kf.start = Math.max(0.0, round1(kf.start));
                    if (i > 0 && changedIdx !== i) {
                        // Ripple chain: start of cut[i] follows end of cut[i-1]
                        kf.start = state.keyframes[i - 1].end;
                    }
                    kf.end = round1(kf.end);
                    if (kf.end <= kf.start) {
                        kf.end = round1(kf.start + (kf.type.includes("TRANS") ? 0.5 : 1.5));
                    }
                    // If user edited 'end' of segment i, ripple-shift start of segment i+1
                    if (i === changedIdx && field === "end" && i + 1 < state.keyframes.length) {
                        state.keyframes[i + 1].start = kf.end;
                    }
                    // If user edited 'start' of segment i (and i > 0), snap previous end to it
                    if (i === changedIdx && field === "start" && i > 0) {
                        state.keyframes[i - 1].end = kf.start;
                    }
                }

                // If last keyframe end exceeds duration, update total duration to match
                const lastEnd = state.keyframes[state.keyframes.length - 1].end;
                if (lastEnd > state.duration) {
                    state.duration = lastEnd;
                }
            }

            // Evenly distribute / scale all keyframes to fit state.duration exactly
            function scaleKeyframesToDuration(newDuration) {
                const targetDur = Math.max(1.0, round1(newDuration));
                const kfs = state.keyframes || [];
                if (kfs.length === 0) {
                    state.duration = targetDur;
                    return;
                }
                const currentEnd = round1(kfs[kfs.length - 1].end) || state.duration || 5.0;
                const ratio = currentEnd > 0 ? targetDur / currentEnd : 1.0;
                let cursor = 0.0;
                kfs.forEach((kf, idx) => {
                    const origLen = Math.max(0.2, (parseFloat(kf.end) || 0) - (parseFloat(kf.start) || 0));
                    kf.start = cursor;
                    if (idx === kfs.length - 1) {
                        kf.end = targetDur;
                    } else {
                        const scaledLen = Math.max(0.2, round1(origLen * ratio));
                        kf.end = Math.min(round1(targetDur - (kfs.length - 1 - idx) * 0.2), round1(cursor + scaledLen));
                    }
                    cursor = kf.end;
                });
                state.duration = targetDur;
            }

            try {
                if (wTimelineData && wTimelineData.value) {
                    const parsed = JSON.parse(wTimelineData.value);
                    if (parsed && typeof parsed === "object") {
                        cleanOldSampleData(parsed);
                        state = parsed;
                        reconcileTimelineMath();
                    }
                }
            } catch (e) {}

            if (wFinalPrompt && wFinalPrompt.value && wFinalPrompt.value.includes("Bold comic-book ink style")) {
                wFinalPrompt.value = "";
            }

            const origOnConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                if (origOnConfigure) origOnConfigure.apply(this, arguments);
                if (wModel && (!wModel.value || wModel.value.includes("deepseek"))) {
                    wModel.value = UNCENSORED_MODEL;
                }
                if (wFinalPrompt && wFinalPrompt.value && wFinalPrompt.value.includes("Bold comic-book ink style")) {
                    wFinalPrompt.value = "";
                    if (wFinalPrompt.inputEl) wFinalPrompt.inputEl.value = "";
                }
                try {
                    if (wTimelineData && wTimelineData.value) {
                        const parsed = JSON.parse(wTimelineData.value);
                        if (parsed && typeof parsed === "object") {
                            cleanOldSampleData(parsed);
                            state = parsed;
                            reconcileTimelineMath();
                            renderUI();
                        }
                    }
                } catch (e) {}
            };

            function syncState() {
                if (wTimelineData) {
                    wTimelineData.value = JSON.stringify(state);
                }
            }

            const container = document.createElement("div");
            container.style.cssText = `
                background: #181a20;
                border: 1px solid #2e3440;
                border-radius: 8px;
                padding: 10px;
                color: #e5e9f0;
                font-family: system-ui, -apple-system, sans-serif;
                font-size: 12px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                box-sizing: border-box;
                width: 100%;
                max-height: 560px;
                overflow-y: auto;
            `;

            ["keydown", "keyup", "keypress", "mousedown", "wheel", "pointerdown"].forEach((evt) => {
                container.addEventListener(evt, (e) => e.stopPropagation());
            });

            function renderUI() {
                syncState();
                container.innerHTML = "";

                // 1. Header: Duration + Auto-Fit + Clear Button
                const topRow = document.createElement("div");
                topRow.style.cssText = "display:flex; gap:6px; align-items:center; flex-wrap:wrap;";

                const durWrap = document.createElement("div");
                durWrap.style.cssText = "display:flex; align-items:center; gap:4px; background:#232731; padding:4px 8px; border-radius:6px; white-space:nowrap;";
                durWrap.innerHTML = `<span style="color:#88c0d0;font-weight:700;">⏱ Tổng Shot:</span>`;
                const durInput = document.createElement("input");
                durInput.type = "number";
                durInput.step = "0.5";
                durInput.min = "1";
                durInput.max = "15";
                durInput.value = state.duration;
                durInput.style.cssText = "width:48px; background:#111318; color:#ebcb8b; font-weight:700; border:1px solid #3b4252; border-radius:4px; padding:2px 4px; text-align:center;";
                durInput.onchange = () => {
                    const newDur = Math.max(1, Math.min(15, parseFloat(durInput.value) || 5.0));
                    scaleKeyframesToDuration(newDur);
                    renderUI();
                };
                durWrap.appendChild(durInput);
                const framesSpan = document.createElement("span");
                framesSpan.style.cssText = "color:#81a1c1; font-size:11px; font-weight:600;";
                let fc = Math.max(5, Math.round(state.duration * 24));
                while (fc % 17 !== 5) fc++;
                framesSpan.textContent = `s (${fc} frames @ 24fps)`;
                durWrap.appendChild(framesSpan);

                const fitBtn = document.createElement("button");
                fitBtn.textContent = "⚖️ Khớp Giây Chuẩn";
                fitBtn.title = "Tự động nối liền các mốc giây (không hở, không chồng chéo) khớp đúng Tổng giây";
                fitBtn.style.cssText = "background:#2e3440; color:#88c0d0; border:1px solid #434c5e; border-radius:6px; padding:4px 8px; cursor:pointer; font-size:11px; font-weight:700; white-space:nowrap;";
                fitBtn.onclick = () => {
                    reconcileTimelineMath();
                    scaleKeyframesToDuration(state.duration);
                    renderUI();
                };

                const clearBtn = document.createElement("button");
                clearBtn.textContent = "🗑️ Xóa Trắng";
                clearBtn.title = "Xóa trắng toàn bộ để nhập cảnh mới";
                clearBtn.style.cssText = "margin-left:auto; background:#4c2326; color:#ff9999; border:1px solid #7a373b; border-radius:6px; padding:4px 8px; cursor:pointer; font-size:11px; font-weight:700; white-space:nowrap;";
                clearBtn.onclick = () => {
                    state = getCleanDefaultState();
                    if (wFinalPrompt) {
                        wFinalPrompt.value = "";
                        if (wFinalPrompt.inputEl) wFinalPrompt.inputEl.value = "";
                    }
                    renderUI();
                };

                topRow.appendChild(durWrap);
                topRow.appendChild(fitBtn);
                topRow.appendChild(clearBtn);
                container.appendChild(topRow);

                // 1B. Director Harness Controls (Visual Look + Camera Master + Audio/Music Master)
                const dirBox = document.createElement("div");
                dirBox.style.cssText = "display:grid; grid-template-columns: 1fr 1fr 1fr; gap:5px; background:#20242d; padding:6px; border-radius:6px;";

                const mkDirInput = (label, icon, key, placeholder) => {
                    const wrap = document.createElement("div");
                    wrap.style.cssText = "display:flex; flex-direction:column; gap:2px;";
                    const lbl = document.createElement("span");
                    lbl.style.cssText = "font-size:10px; font-weight:700; color:#81a1c1;";
                    lbl.textContent = `${icon} ${label}`;
                    const inp = document.createElement("input");
                    inp.type = "text";
                    inp.value = state[key] || "";
                    inp.placeholder = placeholder;
                    inp.style.cssText = "background:#111318; color:#eceff4; border:1px solid #3b4252; border-radius:4px; padding:4px 6px; font-size:11px;";
                    inp.oninput = () => {
                        state[key] = inp.value;
                        syncState();
                    };
                    wrap.appendChild(lbl);
                    wrap.appendChild(inp);
                    return wrap;
                };

                dirBox.appendChild(mkDirInput("1. Visual & Ánh sáng", "🎨", "style", "VD: Realistic 4K, neon cyberpunk, warm bedroom..."));
                dirBox.appendChild(mkDirInput("2. Camera & Ống kính", "🎥", "camera_style", "VD: 35mm anamorphic, f/1.4 xóa phông, handheld..."));
                dirBox.appendChild(mkDirInput("3. Nhạc & Sound Design", "🎵", "audio_style", "VD: Khớp chặt beat <audio1>, bass mạnh, tiếng thở..."));
                container.appendChild(dirBox);

                // 2. Proportional Time-Ruler & Visual Timeline Bar (True-to-scale positioning!)
                const rulerWrap = document.createElement("div");
                rulerWrap.style.cssText = "display:flex; flex-direction:column; gap:2px;";

                const rulerTicks = document.createElement("div");
                rulerTicks.style.cssText = "position:relative; height:14px; font-size:9px; color:#88c0d0; font-family:monospace;";
                const totalDur = Math.max(0.5, parseFloat(state.duration) || 5.0);
                const stepSec = totalDur <= 6 ? 0.5 : 1.0;
                for (let t = 0; t <= totalDur + 0.001; t = round1(t + stepSec)) {
                    const pct = Math.min(100, (t / totalDur) * 100);
                    const tick = document.createElement("span");
                    tick.style.cssText = `position:absolute; left:${pct}%; transform:translateX(${pct > 95 ? "-100%" : pct < 5 ? "0%" : "-50%"});`;
                    tick.textContent = `${t}s`;
                    rulerTicks.appendChild(tick);
                }
                rulerWrap.appendChild(rulerTicks);

                const barContainer = document.createElement("div");
                barContainer.style.cssText = "position:relative; height:26px; background:#111318; border:1px solid #3b4252; border-radius:5px; overflow:hidden;";
                const colors = ["#5e81ac", "#b48ead", "#a3be8c", "#ebcb8b", "#d08770", "#bf616a"];
                state.keyframes.forEach((kf, idx) => {
                    const s = Math.max(0, parseFloat(kf.start) || 0);
                    const e = Math.max(s + 0.05, parseFloat(kf.end) || s + 0.5);
                    const leftPct = Math.min(100, Math.max(0, (s / totalDur) * 100));
                    const widthPct = Math.min(100 - leftPct, Math.max(2, ((e - s) / totalDur) * 100));
                    const seg = document.createElement("div");
                    seg.style.cssText = `
                        position:absolute;
                        left:${leftPct}%;
                        width:${widthPct}%;
                        top:0;
                        bottom:0;
                        background:${kf.type.includes("TRANS") ? "#d08770" : colors[idx % colors.length]};
                        border-right:1px solid #181a20;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:10px;
                        font-weight:700;
                        color:#fff;
                        overflow:hidden;
                        white-space:nowrap;
                        text-overflow:ellipsis;
                        padding:0 4px;
                        box-sizing:border-box;
                    `;
                    seg.textContent = `${kf.type} [${s}s–${e}s] ${kf.tags || ""}`;
                    seg.title = `${kf.type} [${s}s - ${e}s] (${round1(e - s)}s): ${kf.action}`;
                    barContainer.appendChild(seg);
                });
                rulerWrap.appendChild(barContainer);
                container.appendChild(rulerWrap);

                // 3. Objects / Tags Section (<image1>, <audio1>, <video1>)
                const objBox = document.createElement("div");
                objBox.style.cssText = "background:#20242d; border-radius:6px; padding:6px 8px; display:flex; flex-direction:column; gap:5px;";

                const objHeader = document.createElement("div");
                objHeader.style.cssText = "display:flex; justify-content:space-between; align-items:center;";
                objHeader.innerHTML = `<span style="font-weight:700; color:#a3be8c;">🏷️ Đối Tượng Tham Chiếu (&lt;image1&gt; ➔ &lt;Picture 1&gt;, &lt;audio1&gt; ➔ &lt;Audio 1&gt;)</span>`;

                const addObjBtns = document.createElement("div");
                addObjBtns.style.cssText = "display:flex; gap:4px;";
                const makeAddTagBtn = (prefix, label, color) => {
                    const b = document.createElement("button");
                    b.textContent = label;
                    b.style.cssText = `background:${color}; color:#fff; border:none; border-radius:4px; padding:2px 7px; cursor:pointer; font-size:11px; font-weight:600;`;
                    b.onclick = () => {
                        const count = state.objects.filter((o) => o.tag.toLowerCase().includes(prefix)).length + 1;
                        state.objects.push({ tag: `<${prefix}${count}>`, desc: "" });
                        renderUI();
                    };
                    return b;
                };
                addObjBtns.appendChild(makeAddTagBtn("image", "+ <image>", "#3b6998"));
                addObjBtns.appendChild(makeAddTagBtn("audio", "+ <audio>", "#684982"));
                addObjBtns.appendChild(makeAddTagBtn("video", "+ <video>", "#3d7a5a"));
                objHeader.appendChild(addObjBtns);
                objBox.appendChild(objHeader);

                state.objects.forEach((obj, idx) => {
                    const row = document.createElement("div");
                    row.style.cssText = "display:flex; gap:6px; align-items:center;";

                    const tagBadge = document.createElement("input");
                    tagBadge.type = "text";
                    tagBadge.value = obj.tag;
                    tagBadge.style.cssText = "width:76px; background:#14171d; color:#ebcb8b; font-weight:700; border:1px solid #3b4252; border-radius:4px; padding:3px 5px; text-align:center; font-size:11px;";
                    tagBadge.onchange = () => {
                        obj.tag = tagBadge.value.trim();
                        renderUI();
                    };

                    const descIn = document.createElement("input");
                    descIn.type = "text";
                    descIn.placeholder = "Mô tả nhân vật / đối tượng / nhạc của tag này (hoặc để trống)...";
                    descIn.value = obj.desc || "";
                    descIn.style.cssText = "flex:1; background:#14171d; color:#eceff4; border:1px solid #3b4252; border-radius:4px; padding:3px 6px; font-size:11px;";
                    descIn.oninput = () => {
                        obj.desc = descIn.value;
                        syncState();
                    };

                    const delBtn = document.createElement("button");
                    delBtn.textContent = "✕";
                    delBtn.style.cssText = "background:#4c2326; color:#ff8080; border:none; border-radius:4px; padding:2px 6px; cursor:pointer;";
                    delBtn.onclick = () => {
                        state.objects.splice(idx, 1);
                        renderUI();
                    };

                    row.appendChild(tagBadge);
                    row.appendChild(descIn);
                    row.appendChild(delBtn);
                    objBox.appendChild(row);
                });
                container.appendChild(objBox);

                // 4. Keyframes / Timeline Cuts Section
                const kfBox = document.createElement("div");
                kfBox.style.cssText = "background:#20242d; border-radius:6px; padding:6px 8px; display:flex; flex-direction:column; gap:6px;";

                const kfHeader = document.createElement("div");
                kfHeader.style.cssText = "display:flex; justify-content:space-between; align-items:center;";
                kfHeader.innerHTML = `<span style="font-weight:700; color:#88c0d0;">🎬 Timeline Phân Cảnh Chính Xác (Tự động nối mốc giây & Đạo diễn Camera / Âm thanh)</span>`;

                const kfBtns = document.createElement("div");
                kfBtns.style.cssText = "display:flex; gap:4px;";

                const addCutBtn = document.createElement("button");
                addCutBtn.textContent = "+ Thêm CUT";
                addCutBtn.style.cssText = "background:#2e6f95; color:#fff; border:none; border-radius:4px; padding:2px 8px; cursor:pointer; font-size:11px; font-weight:600;";
                addCutBtn.onclick = () => {
                    const lastEnd = state.keyframes.length ? round1(state.keyframes[state.keyframes.length - 1].end) : 0.0;
                    const nextEnd = round1(lastEnd + 2.0);
                    const cutNum = state.keyframes.filter((k) => k.type.startsWith("CUT")).length + 1;
                    state.keyframes.push({
                        start: lastEnd,
                        end: nextEnd,
                        type: `CUT ${cutNum}`,
                        tags: state.objects[0]?.tag || "<image1>",
                        camera: "",
                        sound: "",
                        action: "",
                    });
                    if (nextEnd > state.duration) state.duration = nextEnd;
                    renderUI();
                };

                const addTransBtn = document.createElement("button");
                addTransBtn.textContent = "+ Thêm TRANSITION";
                addTransBtn.style.cssText = "background:#8f563b; color:#fff; border:none; border-radius:4px; padding:2px 8px; cursor:pointer; font-size:11px; font-weight:600;";
                addTransBtn.onclick = () => {
                    const lastEnd = state.keyframes.length ? round1(state.keyframes[state.keyframes.length - 1].end) : 0.0;
                    const nextEnd = round1(lastEnd + 0.5);
                    state.keyframes.push({
                        start: lastEnd,
                        end: nextEnd,
                        type: "TRANSITION",
                        tags: "",
                        camera: "",
                        sound: "",
                        action: "",
                    });
                    if (nextEnd > state.duration) state.duration = nextEnd;
                    renderUI();
                };

                kfBtns.appendChild(addCutBtn);
                kfBtns.appendChild(addTransBtn);
                kfHeader.appendChild(kfBtns);
                kfBox.appendChild(kfHeader);

                state.keyframes.forEach((kf, idx) => {
                    const card = document.createElement("div");
                    card.style.cssText = "background:#14171d; border:1px solid #323949; border-radius:5px; padding:6px; display:flex; flex-direction:column; gap:5px;";

                    const row1 = document.createElement("div");
                    row1.style.cssText = "display:flex; gap:5px; align-items:center; flex-wrap:wrap;";

                    const typeIn = document.createElement("input");
                    typeIn.type = "text";
                    typeIn.value = kf.type;
                    typeIn.style.cssText = "width:78px; background:#232731; color:#88c0d0; font-weight:700; border:1px solid #3b4252; border-radius:4px; padding:2px 5px; font-size:11px;";
                    typeIn.onchange = () => {
                        kf.type = typeIn.value;
                        renderUI();
                    };

                    const startIn = document.createElement("input");
                    startIn.type = "number";
                    startIn.step = "0.1";
                    startIn.value = kf.start;
                    startIn.style.cssText = "width:46px; background:#232731; color:#ebcb8b; font-weight:700; border:1px solid #3b4252; border-radius:4px; padding:2px 4px; font-size:11px; text-align:center;";
                    startIn.onchange = () => {
                        kf.start = round1(startIn.value);
                        reconcileTimelineMath(idx, "start");
                        renderUI();
                    };

                    const endIn = document.createElement("input");
                    endIn.type = "number";
                    endIn.step = "0.1";
                    endIn.value = kf.end;
                    endIn.style.cssText = "width:46px; background:#232731; color:#ebcb8b; font-weight:700; border:1px solid #3b4252; border-radius:4px; padding:2px 4px; font-size:11px; text-align:center;";
                    endIn.onchange = () => {
                        kf.end = round1(endIn.value);
                        reconcileTimelineMath(idx, "end");
                        renderUI();
                    };

                    const segDur = round1((parseFloat(kf.end) || 0) - (parseFloat(kf.start) || 0));
                    const fStart = Math.round((parseFloat(kf.start) || 0) * 24);
                    const fEnd = Math.round((parseFloat(kf.end) || 0) * 24);
                    const durBadge = document.createElement("span");
                    durBadge.style.cssText = "color:#a3be8c; font-size:10px; font-family:monospace; background:#1d242b; padding:2px 5px; border-radius:4px;";
                    durBadge.textContent = `=${segDur}s (${fStart}f–${fEnd}f)`;

                    row1.appendChild(typeIn);
                    row1.appendChild(startIn);
                    const tSpan2 = document.createElement("span");
                    tSpan2.textContent = "s ➔";
                    tSpan2.style.color = "#999";
                    row1.appendChild(tSpan2);
                    row1.appendChild(endIn);
                    row1.appendChild(durBadge);

                    state.objects.forEach((obj) => {
                        const active = (kf.tags || "").includes(obj.tag);
                        const chip = document.createElement("button");
                        chip.textContent = obj.tag;
                        chip.style.cssText = `
                            background:${active ? "#ebcb8b" : "#232731"};
                            color:${active ? "#111" : "#aaa"};
                            border:1px solid ${active ? "#ebcb8b" : "#3b4252"};
                            border-radius:10px;
                            padding:1px 6px;
                            font-size:10px;
                            font-weight:700;
                            cursor:pointer;
                        `;
                        chip.onclick = () => {
                            let list = (kf.tags || "")
                                .split(",")
                                .map((s) => s.trim())
                                .filter(Boolean);
                            if (list.includes(obj.tag)) {
                                list = list.filter((x) => x !== obj.tag);
                            } else {
                                list.push(obj.tag);
                            }
                            kf.tags = list.join(", ");
                            renderUI();
                        };
                        row1.appendChild(chip);
                    });

                    const delKf = document.createElement("button");
                    delKf.textContent = "✕";
                    delKf.style.cssText = "margin-left:auto; background:#4c2326; color:#ff8080; border:none; border-radius:4px; padding:1px 6px; cursor:pointer;";
                    delKf.onclick = () => {
                        state.keyframes.splice(idx, 1);
                        reconcileTimelineMath();
                        renderUI();
                    };
                    row1.appendChild(delKf);

                    // Per-cut Camera + Sound mini-row
                    const rowCamSound = document.createElement("div");
                    rowCamSound.style.cssText = "display:grid; grid-template-columns: 1fr 1fr; gap:5px;";

                    const camIn = document.createElement("input");
                    camIn.type = "text";
                    camIn.value = kf.camera || "";
                    camIn.placeholder = "🎥 Góc máy/Lens (Để trống = AI tự đạo diễn góc máy)...";
                    camIn.style.cssText = "background:#0f1116; color:#88c0d0; border:1px solid #2e3440; border-radius:4px; padding:3px 6px; font-size:11px;";
                    camIn.oninput = () => {
                        kf.camera = camIn.value;
                        syncState();
                    };

                    const sndIn = document.createElement("input");
                    sndIn.type = "text";
                    sndIn.value = kf.sound || "";
                    sndIn.placeholder = "🔊 Âm thanh/Nhạc ở mốc này (Để trống = AI tự phối)...";
                    sndIn.style.cssText = "background:#0f1116; color:#b48ead; border:1px solid #2e3440; border-radius:4px; padding:3px 6px; font-size:11px;";
                    sndIn.oninput = () => {
                        kf.sound = sndIn.value;
                        syncState();
                    };

                    rowCamSound.appendChild(camIn);
                    rowCamSound.appendChild(sndIn);

                    const actionIn = document.createElement("textarea");
                    actionIn.rows = 2;
                    actionIn.placeholder = `🎬 Diễn biến tại [${kf.start}s - ${kf.end}s]: Nhân vật làm gì, biểu cảm, chuyển động...`;
                    actionIn.value = kf.action || "";
                    actionIn.style.cssText = "width:100%; box-sizing:border-box; background:#0f1116; color:#eceff4; border:1px solid #2e3440; border-radius:4px; padding:4px 6px; font-size:11px; resize:vertical;";
                    actionIn.oninput = () => {
                        kf.action = actionIn.value;
                        syncState();
                    };

                    card.appendChild(row1);
                    card.appendChild(rowCamSound);
                    card.appendChild(actionIn);
                    kfBox.appendChild(card);
                });
                container.appendChild(kfBox);

                // 5. Generate with Venice Dolphin 24B Uncensored Director Button
                const genBtn = document.createElement("button");
                genBtn.textContent = "🎬 ĐẠO DIỄN FULL PROMPT (KHỚP MỐC GIÂY + CAMERA + ÁNH SÁNG + ÂM NHẠC)";
                genBtn.style.cssText = `
                    background: linear-gradient(90deg, #b91c1c, #ea580c);
                    color: #ffffff;
                    font-weight: 800;
                    font-size: 12px;
                    padding: 8px 12px;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    text-align: center;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                `;
                genBtn.onclick = async () => {
                    cleanOldSampleData(state);
                    reconcileTimelineMath();
                    syncState();
                    const origText = genBtn.textContent;
                    if (wModel && (!wModel.value || wModel.value.includes("deepseek"))) {
                        wModel.value = UNCENSORED_MODEL;
                    }
                    const chosenModel = wModel ? wModel.value : UNCENSORED_MODEL;
                    genBtn.textContent = `⏳ Đang đạo diễn kịch bản (${chosenModel})...`;
                    genBtn.disabled = true;
                    genBtn.style.opacity = "0.7";
                    try {
                        const resp = await api.fetchApi("/mockin/generate", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                timeline_state: state,
                                model: chosenModel,
                                api_key: wApiKey ? wApiKey.value : "",
                            }),
                        });
                        const data = await resp.json();
                        if (!data.ok) {
                            alert("Lỗi gọi AI:\n" + data.error);
                        } else {
                            if (data.reconciled_state) {
                                state = data.reconciled_state;
                                renderUI();
                            }
                            if (wFinalPrompt) {
                                wFinalPrompt.value = data.prompt;
                                if (wFinalPrompt.inputEl) {
                                    wFinalPrompt.inputEl.value = data.prompt;
                                }
                                app.graph.setDirtyCanvas(true, true);
                            }
                        }
                    } catch (err) {
                        alert("Lỗi kết nối tới server Mockin Node: " + err);
                    } finally {
                        genBtn.textContent = origText;
                        genBtn.disabled = false;
                        genBtn.style.opacity = "1.0";
                    }
                };
                container.appendChild(genBtn);
            }

            renderUI();

            const domWidget = node.addDOMWidget("mockin_timeline_ui", "div", container, {
                serialize: false,
                hideOnZoom: false,
            });
            domWidget.computeSize = (width) => [width, 550];

            const idx = node.widgets.indexOf(domWidget);
            if (idx > 0) {
                node.widgets.splice(idx, 1);
                node.widgets.unshift(domWidget);
            }

            node.setSize([660, 840]);
            return r;
        };
    },
    async setup() {
        function fixGraphModelPaths() {
            if (!app.graph || !app.graph._nodes) return;
            let changed = false;
            for (const n of app.graph._nodes) {
                if (!n.widgets) continue;
                for (const w of n.widgets) {
                    if (typeof w.value === "string" && (w.value.startsWith("Downloads/") || w.value.startsWith("minimax/") || w.value.startsWith("Downloads\\") || w.value.startsWith("minimax\\"))) {
                        let cleaned = w.value
                            .replace(/^Downloads[\/\\]/, "")
                            .replace(/^minimax[\/\\]/, "");
                        if (cleaned !== w.value) {
                            w.value = cleaned;
                            if (Array.isArray(n.widgets_values)) {
                                const idx = n.widgets.indexOf(w);
                                if (idx >= 0 && idx < n.widgets_values.length) {
                                    n.widgets_values[idx] = cleaned;
                                }
                            }
                            changed = true;
                        }
                    }
                }
            }
            if (changed) {
                app.graph.setDirtyCanvas(true, true);
            }
        }
        setTimeout(fixGraphModelPaths, 500);
        setTimeout(fixGraphModelPaths, 2000);
        const origLoadGraphData = app.loadGraphData;
        if (origLoadGraphData) {
            app.loadGraphData = async function () {
                const res = await origLoadGraphData.apply(this, arguments);
                setTimeout(fixGraphModelPaths, 200);
                return res;
            };
        }
    },
});
