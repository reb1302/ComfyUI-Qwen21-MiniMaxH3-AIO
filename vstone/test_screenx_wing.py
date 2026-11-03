"""
Kiểm thử helper ScreenX bằng video tổng hợp có camera biết trước (không cần GPU, không tải model).
Chạy trên H200 bằng venv của ComfyUI:

    python vstone/test_screenx_wing.py [đường_dẫn_tới_node_helper_vstone.py]
"""
import __future__
import ast
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

NEEDED = (
    "COLOR_CHOICES", "OUTER_BG_CHOICES", "MINIMAX_SAFE_DI_W", "MINIMAX_SAFE_DI_H", "SCREENX_MOTION_MODELS",
    "SCREENX_GRID_PX", "SCREENX_ANALYSIS_HEIGHT",
)
NEEDED_FUNCS = (
    "_parse_fill_rgb", "_parse_outer_bg_rgb", "create_solid_pad_tensor", "vstone_outpaint_pad_tensor", "_resize_bhwc",
    "_align_minimax_h3_frames", "_build_screenx_wing_trapezoid_mask", "vstone_screenx_combine3_tensor",
)


def load_helpers(path):
    """Chỉ nạp phần ScreenX để không kích hoạt daemon sync và bản vá nóng khi chạy test."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    body = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in NEEDED:
            body.append(node)
        elif name and (name in NEEDED_FUNCS or name.startswith("_screenx") or name.startswith("stoneV_ScreenX_") and name != "stoneV_ScreenX_Combine3"):
            body.append(node)
    ns = {"torch": torch, "F": F, "math": math, "Any": object, "__name__": "screenx_test"}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True), ns)
    return ns


def make_world(seed=7):
    rng = random.Random(seed)
    waves = []
    for _ in range(40):
        length = rng.uniform(14.0, 160.0)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        waves.append((math.cos(angle) * 2.0 * math.pi / length, math.sin(angle) * 2.0 * math.pi / length, rng.uniform(0.0, 6.28), rng.randrange(3), rng.uniform(0.02, 0.07)))
    return waves


def render_world(waves, xs, ys):
    h, w = xs.shape
    out = torch.full((h, w, 3), 0.5, dtype=torch.float32)
    for fx, fy, phase, channel, amp in waves:
        out[:, :, channel] = out[:, :, channel] + amp * torch.sin(fx * xs + fy * ys + phase)
    return out.clamp(0.0, 1.0)


def render_camera(waves, matrix, real_x, real_y, H, W):
    """Tô màu cho toạ độ thật ở frame t: đưa về toạ độ plate bằng nghịch đảo ma trận camera."""
    half_h = H / 2.0
    grid_y, grid_x = torch.meshgrid(real_y, real_x, indexing="ij")
    ux, uy = ((grid_x - W / 2.0) / half_h).reshape(-1), ((grid_y - half_h) / half_h).reshape(-1)
    pts = torch.stack([ux, uy, torch.ones_like(ux)], dim=0)
    plate = torch.linalg.inv(matrix) @ pts
    px = (plate[0] / plate[2] * half_h + W / 2.0).reshape(grid_x.shape)
    py = (plate[1] / plate[2] * half_h + half_h).reshape(grid_x.shape)
    return render_world(waves, px, py)


def camera_matrix(tx_px, ty_px, zoom, rot, H):
    half_h = H / 2.0
    c, s = math.cos(rot) * zoom, math.sin(rot) * zoom
    return torch.tensor([[c, -s, tx_px / half_h], [s, c, ty_px / half_h], [0.0, 0.0, 1.0]], dtype=torch.float32)


def render_center(waves, matrices, H, W, subject=False, noise=0.0, seed=3):
    rng = random.Random(seed)
    real_x = torch.arange(W, dtype=torch.float32) + 0.5
    real_y = torch.arange(H, dtype=torch.float32) + 0.5
    frames = []
    for t in range(matrices.shape[0]):
        frame = render_camera(waves, matrices[t], real_x, real_y, H, W)
        if subject:
            x0 = int(W * 0.25 + t * 3)
            frame[H // 3 : H // 3 + H // 3, x0 : x0 + W // 5, :] = 0.15
        if noise > 0.0:
            ripple = rng.uniform(0.0, 6.28)
            grid_y, grid_x = torch.meshgrid(real_y, real_x, indexing="ij")
            frame = (frame + noise * torch.sin(grid_x * 1.7 + grid_y * 2.3 + ripple).unsqueeze(-1)).clamp(0.0, 1.0)
        frames.append(frame)
    return torch.stack(frames, dim=0)


def make_plates(ns, waves, center, settings):
    """Plate đúng định dạng node 1A/1B, vẽ từ chính thế giới tổng hợp thay cho Qwen."""
    H, W = center.shape[1], center.shape[2]
    plates = {}
    identity = torch.eye(3, dtype=torch.float32)
    for side in ("left", "right"):
        is_left = side == "left"
        _, mask, plate_w, plate_h = ns["vstone_outpaint_pad_tensor"](
            center[:1],
            add_left_ratio=settings["wing_width_ratio"] if is_left else 0.0,
            add_right_ratio=0.0 if is_left else settings["wing_width_ratio"],
            keep_orig_ratio=settings["plate_keep_ratio"],
            outer_height_ratio=settings["outer_height_ratio"],
        )
        kept = mask[0] < 0.5
        kept_x = kept.any(dim=0).nonzero().reshape(-1)
        kept_y = kept.any(dim=1).nonzero().reshape(-1)
        real_start = 0 if is_left else W - kept_x.shape[0]
        real_x = torch.arange(plate_w, dtype=torch.float32) + 0.5 - int(kept_x[0]) + real_start
        real_y = torch.arange(plate_h, dtype=torch.float32) + 0.5 - int(kept_y[0])
        plates[side] = render_camera(waves, identity, real_x, real_y, H, W).unsqueeze(0)
    return plates


def default_settings(ns, **overrides):
    settings = {
        "wing_width_ratio": 1.0, "outer_height_ratio": 1.3, "plate_keep_ratio": 0.5, "window_keep_ratio": 0.5, "plate_frame": 0,
        "motion_model": "similarity", "motion_smooth_frames": 0, "static_threshold_px": 0.5, "seam_match_px": 32,
    }
    settings.update(overrides)
    return ns["stoneV_ScreenX_Settings"]().build(**settings)[0]


def outer_edge_error(estimated, truth, H, W):
    """Sai lệch (px thật) tại cạnh ngoài cùng của cánh, nơi sai số ngoại suy lớn nhất."""
    half_h = H / 2.0
    edge = 1.5 * W / half_h
    pts = torch.tensor([[-edge, edge, -edge, edge], [-1.3, -1.3, 1.3, 1.3], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    a = estimated @ pts
    b = truth @ pts
    return float(((a[:, :2, :] / a[:, 2:3, :] - b[:, :2, :] / b[:, 2:3, :]).abs()).max()) * half_h


def check(name, ok, detail):
    print(("PASS " if ok else "FAIL ") + name + " | " + detail)
    return ok


def test_layout(ns):
    results = []
    settings = default_settings(ns)
    ns_w, ns_h = ns["MINIMAX_SAFE_DI_W"], ns["MINIMAX_SAFE_DI_H"]
    ns["MINIMAX_SAFE_DI_W"], ns["MINIMAX_SAFE_DI_H"] = 1280, 704
    for H, W in ((1080, 1920), (720, 1280), (1608, 3840), (2160, 3840)):
        for side in ("left", "right"):
            lay = ns["_screenx_window_layout"](H, W, 60, side, settings)
            on_grid = all(lay[k] % 32 == 0 for k in ("width", "height", "seam_x", "keep_y0", "keep_y1", "keep_x0", "keep_x1", "wing_x0", "wing_x1"))
            bounded = lay["width"] <= 1280 and lay["height"] <= 704
            results.append(check(f"layout {W}x{H} {side}", on_grid and bounded and lay["length"] % 17 == 5, f"{lay['width']}x{lay['height']} wing={lay['wing_x1'] - lay['wing_x0']} seam={lay['seam_x']} length={lay['length']}"))
    ns["MINIMAX_SAFE_DI_W"], ns["MINIMAX_SAFE_DI_H"] = ns_w, ns_h
    return results


def run_shot(ns, name, waves, cameras, H, W, side="left", max_edge_error=3.0, max_guide_ratio=0.45, plate_frame=0, **render_kw):
    results = []
    settings = default_settings(ns, plate_frame=plate_frame)
    center = render_center(waves, cameras, H, W, **render_kw)
    plates = make_plates(ns, waves, center, settings)
    canvas = ns["stoneV_ScreenX_Plate_Canvas"]().build(center, settings, plates["left"], plates["right"])[0]
    motion, report = ns["stoneV_ScreenX_Center_Motion"]().estimate(center, settings, side)
    edge_err = outer_edge_error(motion["matrices"], cameras, H, W)
    results.append(check(f"{name}: motion", edge_err <= max_edge_error, f"sai lệch cạnh ngoài {edge_err:.2f}px (cho phép {max_edge_error}px)"))

    guide, mask, ref, width, height, length, layout, _ = ns["stoneV_ScreenX_Wing_Prepare"]().prepare(center, canvas, motion, settings)
    T = center.shape[0]
    results.append(check(f"{name}: shapes", tuple(guide.shape) == (T, height, width, 3) and tuple(mask.shape) == (1, height, width) and ref.shape[0] == 1 and length >= T, f"guide {tuple(guide.shape)} mask {tuple(mask.shape)} length {length}"))

    y0, y1, x0, x1 = layout["keep_y0"], layout["keep_y1"], layout["keep_x0"], layout["keep_x1"]
    results.append(check(f"{name}: mask", float(mask[:, y0:y1, x0:x1].max()) == 0.0 and float(mask[:, :, layout["wing_x0"] : layout["wing_x1"]].min()) == 1.0, "0 ở dải giữa, 1 ở cánh"))

    wx0, wx1 = layout["wing_x0"], layout["wing_x1"]
    real_x = (torch.arange(width, dtype=torch.float32) + 0.5 - layout["seam_x"]) / layout["scale_x"] + layout["seam_real_x"]
    real_y = (torch.arange(height, dtype=torch.float32) + 0.5 - y0) / layout["scale_y"]
    truth_last = render_camera(waves, cameras[-1], real_x, real_y, H, W)[:, wx0:wx1]
    guide_err = float((guide[-1, :, wx0:wx1] - truth_last).abs().mean())
    frozen_err = float((guide[plate_frame, :, wx0:wx1] - truth_last).abs().mean())
    ratio = guide_err / max(frozen_err, 1e-6)
    results.append(check(f"{name}: plate_frame không thiếu dữ liệu", layout["holes"][plate_frame] == 0.0, f"thiếu {layout['holes'][plate_frame]:.1f}% ở frame {plate_frame}"))
    results.append(check(f"{name}: guide bám camera", ratio <= max_guide_ratio or frozen_err < 0.01, f"sai số guide {guide_err:.4f}, plate đứng yên {frozen_err:.4f}, tỉ lệ {ratio:.2f}"))

    # Bản gen giả: lệch màu toàn khung và dư frame như sau VAEDecode, để kiểm tra Finish.
    extra = length + 4 - T
    generated = torch.cat([guide, guide[-1:].repeat(extra, 1, 1, 1)], dim=0)
    generated = (generated + 0.06).clamp(0.0, 1.0)
    wing = ns["stoneV_ScreenX_Wing_Finish"]().finish(generated, guide, layout)[0]
    expected_h = int(round((layout["canvas_h"] - H) * layout["scale_y"])) + (y1 - y0)
    shape_ok = wing.shape[0] == T and wing.shape[2] == wx1 - wx0 and abs(wing.shape[1] - expected_h) <= 2
    results.append(check(f"{name}: finish shape", shape_ok, f"wing {tuple(wing.shape)}, cao dự kiến {expected_h}"))

    seam = layout["seam_x"]
    top = y0 - max(0, layout["keep_y0"] - int(round(layout["pad_top"] * layout["scale_y"])))
    rows = wing[:, top : top + (y1 - y0)]
    wing_edge = rows[:, :, -1 if side == "left" else 0]
    strip_edge = guide[:, y0:y1, seam if side == "left" else seam - 1]
    raw_edge = generated[:T, y0:y1, seam - 1 if side == "left" else seam]
    after = float((wing_edge - strip_edge).abs().mean())
    before = float((raw_edge - strip_edge).abs().mean())
    results.append(check(f"{name}: khớp màu mép nối", after < before * 0.6, f"lệch mép {before:.4f} -> {after:.4f}"))
    return results, center, wing, layout


def main():
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "node_helper_vstone.py")
    ns = load_helpers(sys.argv[1] if len(sys.argv) > 1 else default_path)
    # Thu nhỏ giới hạn khung để test chạy nhanh trên CPU; hình học không phụ thuộc giới hạn này.
    ns["MINIMAX_SAFE_DI_W"], ns["MINIMAX_SAFE_DI_H"] = 640, 352

    H, W, T = 216, 384, 22
    waves = make_world()
    results = test_layout(ns)

    pan = torch.stack([camera_matrix(-2.5 * t, 0.6 * t, 1.0, 0.0, H) for t in range(T)], dim=0)
    res, _, _, _ = run_shot(ns, "pan sang phải (cánh trái)", waves, pan, H, W, side="left")
    results += res
    res, center, wing_r, layout_r = run_shot(ns, "pan sang phải (cánh phải)", waves, pan, H, W, side="right")
    results += res

    zoom = torch.stack([camera_matrix(-1.0 * t, 0.0, 1.0 + 0.003 * t, 0.0006 * t, H) for t in range(T)], dim=0)
    res, _, _, _ = run_shot(ns, "zoom + pan + xoay", waves, zoom, H, W, side="left", max_edge_error=4.0)
    results += res

    whip = torch.stack([camera_matrix(-14.0 * t, 0.0, 1.0, 0.0, H) for t in range(T)], dim=0)
    res, _, _, _ = run_shot(ns, "pan nhanh 14px/frame", waves, whip, H, W, side="left", max_edge_error=6.0, max_guide_ratio=0.8)
    results += res

    mid = T // 2
    mid_inverse = torch.linalg.inv(pan[mid])
    pan_from_mid = torch.stack([pan[t] @ mid_inverse for t in range(T)], dim=0)
    res, _, _, _ = run_shot(ns, "plate vẽ ở frame giữa", waves, pan_from_mid, H, W, side="right", plate_frame=mid)
    results += res

    static = torch.stack([camera_matrix(0.0, 0.0, 1.0, 0.0, H) for _ in range(T)], dim=0)
    settings = default_settings(ns)
    still_center = render_center(waves, static, H, W, subject=True, noise=0.008)
    matrices, report = ns["_screenx_estimate_motion"](still_center, "left", settings)
    drift = outer_edge_error(matrices, static, H, W)
    results.append(check("camera tĩnh + vật chuyển động", drift <= 1.0, f"lệch {drift:.2f}px | {report}"))

    res, _, wing_l, _ = run_shot(ns, "ghép preview", waves, pan, H, W, side="left")
    combined = ns["vstone_screenx_combine3_tensor"](center, wing_l, wing_r, wing_width_ratio=1.0, outer_height_ratio=1.3)[0]
    results.append(check("Combine3 nhận 2 cánh", combined.shape[0] == T and combined.shape[2] == W * 3, f"combined {tuple(combined.shape)}"))

    passed = sum(1 for r in results if r)
    print(f"\n{passed}/{len(results)} kiểm tra đạt")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
