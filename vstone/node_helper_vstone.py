"""
VStone Node Helper & Cloud Sync Hub (node_helper_vstone.py)
--------------------------------------------------------------------------------
Tích hợp 2 chức năng cốt lõi:
1. Bộ Node VStone Outpaint Pad (Tạo vùng bù Trái / Phải, tô màu Xám #808080 / Đỏ #FF0000 / Xanh #00FF00 / Hex, sắc nét 100% không blur).
2. Đường ống đồng bộ tự động 2 chiều (VStone Cloud Sync Hub & Auto-Sync Daemon 30s):
   - Tự động kéo Custom Node & Workflow JSON mới nhất từ Laptop (GitHub `reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO`)
     về thẳng máy H200 Hàn Quốc và nạp nóng (Hot-Reload) vào RAM ComfyUI mà KHÔNG cần khởi động lại Server!
   - Tải trực tiếp LoRA / Checkpoint / Upscaler tốc độ cao (aria2c 16 luồng) ngay từ giao diện ComfyUI.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = Any  # type: ignore
    F = None  # type: ignore


GITHUB_REPO = "reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO"
GITHUB_BRANCH = "main"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}"
GITHUB_API_TREE = f"https://api.github.com/repos/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1"

COLOR_CHOICES = [
    "gray (#808080 - Tô Xám chuẩn Outpaint LoRA)",
    "red (#FF0000 - Tô Đỏ)",
    "green (#00FF00 - Tô Xanh Lá)",
    "blue (#0000FF - Tô Xanh Dương)",
    "white (#FFFFFF - Tô Trắng)",
    "black (#000000 - Tô Đen)",
    "custom_hex (Dùng mã màu Hex bên dưới)",
]

# Danh mục LoRA & Model chuẩn để bấm tải 1-click ngay trên máy H200 Hàn Quốc
PRESET_DOWNLOAD_CATALOG: dict[str, tuple[str, str, str]] = {
    "none (Không tải thêm model)": ("", "", ""),
    "Qwen 2.1 Outpaint LoRA v2 (Xám #808080 - 152MB)": (
        "loras",
        "qwen-image-2.1-outpaint-v2.safetensors",
        "https://huggingface.co/ausboss/Qwen-Image-2.1-Outpaint-LoRA/resolve/main/qwen-image-2.1-outpaint-v2.safetensors",
    ),
    "Qwen 2.1 Outpaint LoRA v1 (Zoom-out lớn - 152MB)": (
        "loras",
        "qwen-image-2.1-outpaint.safetensors",
        "https://huggingface.co/ausboss/Qwen-Image-2.1-Outpaint-LoRA/resolve/main/qwen-image-2.1-outpaint.safetensors",
    ),
    "Qwen 2.1 360 Panorama Outpaint LoRA (Xanh #00FF00 - 152MB)": (
        "loras",
        "qwen_image_2_1_360_panorama_outpaint_lora_v0_1.safetensors",
        "https://huggingface.co/nomadoor/qwen-image-2.1-360-panorama-outpaint-lora/resolve/main/qwen_image_2_1_360_panorama_outpaint_lora_v0_1.safetensors",
    ),
    "Qwen 2.1 Fun ControlNet Union Inpaint (2.68GB)": (
        "model_patches",
        "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors",
    ),
    "MiniMax-H3 3D Latent Upscaler bf16 (691MB)": (
        "latent_upscale_models",
        "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
        "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
    ),
    "MiniMax-H3 VBVR Motion Consistency LoRA": (
        "loras",
        "H3-Licon-VBVR-I2V.safetensors",
        "https://huggingface.co/LiconStudio/Minimax-H3-VBVR/resolve/main/H3-Licon-VBVR-I2V.safetensors",
    ),
    "MiniMax-H3 Fun ControlNet Union 2.0 int8": (
        "model_patches",
        "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors",
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors",
    ),
    "Ostris Qwen Edit Green Screen Inpaint LoRA (563MB)": (
        "loras",
        "qwen_image_edit_inpainting.safetensors",
        "https://huggingface.co/ostris/qwen_image_edit_inpainting/resolve/main/qwen_image_edit_inpainting.safetensors",
    ),
}


# ==============================================================================
# 1. CLOUD SYNC & HOT-RELOAD ENGINE (Laptop <-> H200 Korea)
# ==============================================================================
_LAST_SYNCED_SHA: str = ""


def _http_get_bytes(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "VStone-ComfyUI-Sync/1.0", "Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _get_comfy_root() -> str | None:
    try:
        import folder_paths
        return folder_paths.base_path
    except Exception:
        return None


def sync_from_github_and_hot_reload(force: bool = False) -> str:
    """
    Kéo toàn bộ code `vstone/` và `workflows/` mới nhất từ GitHub `reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO`
    về máy H200 và nạp nóng (Hot-Reload) các node mới vào RAM của ComfyUI.
    """
    global _LAST_SYNCED_SHA
    comfy_root = _get_comfy_root()
    if not comfy_root:
        return "[VStone Sync] Không tìm thấy thư mục gốc ComfyUI."

    logs: list[str] = []
    try:
        tree_raw = _http_get_bytes(f"{GITHUB_API_TREE}&_t={int(time.time())}", timeout=15)
        tree_data = json.loads(tree_raw.decode("utf-8"))
        sha = tree_data.get("sha", "")
        if not force and sha and sha == _LAST_SYNCED_SHA:
            return f"[VStone Sync] Đã ở phiên bản mới nhất ({sha[:7]})."

        items = tree_data.get("tree", [])
        custom_nodes_dir = os.path.join(comfy_root, "custom_nodes")
        vstone_dir = os.path.join(custom_nodes_dir, "vstone")
        aio_vstone_dir = os.path.join(custom_nodes_dir, "ComfyUI_Qwen21_MiniMaxH3_AIO", "vstone")
        workflows_dir = os.path.join(comfy_root, "user", "default", "workflows")

        os.makedirs(vstone_dir, exist_ok=True)
        os.makedirs(workflows_dir, exist_ok=True)

        updated_py_code: str | None = None
        wf_count = 0

        for item in items:
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")

            # 1. Đồng bộ các file trong vstone/
            if path in ("vstone/node_helper_vstone.py", "vstone/__init__.py"):
                raw_url = f"{GITHUB_RAW_BASE}/{path}?_t={int(time.time())}"
                content = _http_get_bytes(raw_url, timeout=20)
                fname = os.path.basename(path)

                # Ghi vào custom_nodes/vstone/
                dst1 = os.path.join(vstone_dir, fname)
                with open(dst1, "wb") as f:
                    f.write(content)

                # Nếu là node_helper_vstone.py, ghi cả vào custom_nodes/node_helper_vstone.py và AIO (nếu có)
                if fname == "node_helper_vstone.py":
                    updated_py_code = content.decode("utf-8", errors="replace")
                    dst_root = os.path.join(custom_nodes_dir, "node_helper_vstone.py")
                    try:
                        with open(dst_root, "wb") as f:
                            f.write(content)
                    except Exception:
                        pass
                    if os.path.isdir(aio_vstone_dir):
                        try:
                            with open(os.path.join(aio_vstone_dir, fname), "wb") as f:
                                f.write(content)
                        except Exception:
                            pass
                logs.append(f"Updated {path}")

            # 2. Đồng bộ mọi file Workflow .json trong workflows/
            elif path.startswith("workflows/") and path.endswith(".json"):
                rel_wf = path[len("workflows/"):]
                dst_wf = os.path.join(workflows_dir, rel_wf)
                os.makedirs(os.path.dirname(dst_wf), exist_ok=True)
                raw_url = f"{GITHUB_RAW_BASE}/{path}?_t={int(time.time())}"
                content = _http_get_bytes(raw_url, timeout=20)
                with open(dst_wf, "wb") as f:
                    f.write(content)
                wf_count += 1

        # 3. Hot-Reload trực tiếp các Node mới vào RAM của ComfyUI (không cần tắt bật lại Server!)
        if updated_py_code:
            try:
                import nodes
                ns: dict[str, Any] = {"__name__": "vstone_hot_reload", "_DISABLE_DAEMON_ON_EXEC": True}
                exec(compile(updated_py_code, "node_helper_vstone.py", "exec"), ns)
                new_classes = ns.get("NODE_CLASS_MAPPINGS", {})
                new_names = ns.get("NODE_DISPLAY_NAME_MAPPINGS", {})
                if new_classes:
                    nodes.NODE_CLASS_MAPPINGS.update(new_classes)
                    nodes.NODE_DISPLAY_NAME_MAPPINGS.update(new_names)
                    logs.append(f"Hot-reloaded {len(new_classes)} VStone nodes into ComfyUI RAM")
            except Exception as e:
                logs.append(f"Hot-reload warning: {e}")

        _LAST_SYNCED_SHA = sha
        msg = f"[VStone Cloud Sync] Đồng bộ thành công commit {sha[:7]} ({wf_count} workflows)! Nhấn phím R trên trình duyệt để làm mới."
        print(msg)
        return msg + "\n" + "\n".join(logs)
    except Exception as e:
        err_msg = f"[VStone Cloud Sync] Lỗi kết nối GitHub: {e}"
        return err_msg


def download_file_to_comfy_models(folder_type: str, filename: str, url: str) -> str:
    """Tải LoRA / Model tốc độ cao bằng aria2c (hoặc curl/urllib) vào đúng thư mục models của ComfyUI."""
    comfy_root = _get_comfy_root()
    if not comfy_root:
        return "[VStone Downloader] Không tìm thấy thư mục ComfyUI."

    target_dir = os.path.join(comfy_root, "models", folder_type)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024 * 1024:
        return f"[VStone Downloader] File đã tồn tại sẵn: {target_path} ({os.path.getsize(target_path) // (1024*1024)} MB)"

    print(f"[VStone Downloader] Đang tải {filename} vào {target_dir} từ {url} ...")
    aria2_bin = shutil.which("aria2c")
    if aria2_bin:
        cmd = [
            aria2_bin, "--disable-ipv6=true", "-x", "16", "-s", "16", "-k", "1M",
            "--allow-overwrite=true", "-d", target_dir, "-o", filename, url
        ]
        res = subprocess.run(cmd, check=False)
        if res.returncode == 0 and os.path.exists(target_path):
            return f"[VStone Downloader] Tải thành công bằng aria2c: {target_path}"

    curl_bin = shutil.which("curl")
    if curl_bin:
        cmd = [curl_bin, "-4", "-L", url, "-o", target_path]
        res = subprocess.run(cmd, check=False)
        if res.returncode == 0 and os.path.exists(target_path):
            return f"[VStone Downloader] Tải thành công bằng curl: {target_path}"

    urllib.request.urlretrieve(url, target_path)
    return f"[VStone Downloader] Tải thành công: {target_path}"


def _start_background_sync_daemon(interval_sec: int = 30):
    """Chạy ngầm kiểm tra GitHub mỗi 30 giây khi đang chạy trong ComfyUI trên máy H200."""
    if globals().get("_DISABLE_DAEMON_ON_EXEC"):
        return
    if _get_comfy_root() is None:
        return
    if getattr(sys_mod := __import__("sys"), "_vstone_sync_daemon_started", False):
        return
    setattr(sys_mod, "_vstone_sync_daemon_started", True)

    def _loop():
        time.sleep(3)
        sync_from_github_and_hot_reload(force=False)
        while True:
            time.sleep(interval_sec)
            try:
                sync_from_github_and_hot_reload(force=False)
            except Exception:
                pass

    t = threading.Thread(target=_loop, name="VStoneAutoSyncDaemon", daemon=True)
    t.start()


try:
    _start_background_sync_daemon(30)
except Exception:
    pass


# ==============================================================================
# 2. CORE HELPER FUNCTIONS (VStone Outpaint Pad)
# ==============================================================================
def _parse_fill_rgb(fill_color: str, custom_hex: str = "#808080") -> tuple[float, float, float]:
    """Chuyển lựa chọn màu tô sang bộ 3 giá trị RGB trong khoảng [0.0, 1.0]."""
    fc = (fill_color or "").lower()
    if fc.startswith("gray") or "#808080" in fc:
        return (128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0)
    if fc.startswith("red") or "#ff0000" in fc:
        return (1.0, 0.0, 0.0)
    if fc.startswith("green") or "#00ff00" in fc:
        return (0.0, 1.0, 0.0)
    if fc.startswith("blue") or "#0000ff" in fc:
        return (0.0, 0.0, 1.0)
    if fc.startswith("white") or "#ffffff" in fc:
        return (1.0, 1.0, 1.0)
    if fc.startswith("black") or "#000000" in fc:
        return (0.0, 0.0, 0.0)

    hx = (custom_hex or "#808080").strip().lstrip("#")
    if len(hx) == 3:
        hx = "".join(c * 2 for c in hx)
    if len(hx) == 6:
        try:
            r = int(hx[0:2], 16) / 255.0
            g = int(hx[2:4], 16) / 255.0
            b = int(hx[4:6], 16) / 255.0
            return (r, g, b)
        except ValueError:
            pass
    return (128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0)


def create_solid_pad_tensor(
    batch: int,
    height: int,
    width: int,
    channels: int,
    rgb: tuple[float, float, float],
    device: Any,
    dtype: Any,
) -> torch.Tensor:
    """Tạo vùng ảnh bù tô màu đặc sắc nét 100%, không làm mờ."""
    if height <= 0 or width <= 0:
        return torch.empty((batch, max(0, height), max(0, width), channels), dtype=dtype, device=device)

    pad = torch.zeros((batch, height, width, channels), dtype=dtype, device=device)
    pad[:, :, :, 0] = rgb[0]
    if channels >= 2:
        pad[:, :, :, 1] = rgb[1]
    if channels >= 3:
        pad[:, :, :, 2] = rgb[2]
    if channels >= 4:
        pad[:, :, :, 3] = 1.0
    return pad


def vstone_outpaint_pad_tensor(
    image: torch.Tensor,
    add_left_ratio: float = 0.0,
    add_right_ratio: float = 0.0,
    extra_left_px: int = 0,
    extra_right_px: int = 0,
    extra_top_px: int = 0,
    extra_bottom_px: int = 0,
    fill_color: str = COLOR_CHOICES[0],
    custom_hex: str = "#808080",
    feather_mask_px: int = 0,
    base_image_for_width: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Nhận 1 tấm ảnh [B, H, W, C]:
    - Giữ nguyên 100% từng pixel của ảnh gốc (không resize, không blur).
    - Mở rộng tạo vùng bù bên Trái / bên Phải (theo tỷ lệ chiều rộng W hoặc số pixel) và tô màu đã chọn.
    - Trả về (padded_image [B, new_H, new_W, 3], outpaint_mask [B, new_H, new_W], new_W, new_H).
    """
    if image.ndim != 4:
        raise ValueError(f"Expected 4D IMAGE tensor [B, H, W, C], got shape {tuple(image.shape)}")

    orig_rgb = image[:, :, :, :3]
    B, H, W, C = orig_rgb.shape
    device = orig_rgb.device
    dtype = orig_rgb.dtype

    unit_W = int(base_image_for_width.shape[2]) if (base_image_for_width is not None and base_image_for_width.ndim == 4) else W

    pad_left = max(0, int(round(unit_W * float(add_left_ratio))) + int(extra_left_px))
    pad_right = max(0, int(round(unit_W * float(add_right_ratio))) + int(extra_right_px))
    pad_top = max(0, int(extra_top_px))
    pad_bottom = max(0, int(extra_bottom_px))

    rgb = _parse_fill_rgb(fill_color, custom_hex)

    left_block = create_solid_pad_tensor(B, H, pad_left, C, rgb, device, dtype)
    right_block = create_solid_pad_tensor(B, H, pad_right, C, rgb, device, dtype)
    row_img = torch.cat([left_block, orig_rgb, right_block], dim=2)
    new_W = pad_left + W + pad_right

    top_block = create_solid_pad_tensor(B, pad_top, new_W, C, rgb, device, dtype)
    bottom_block = create_solid_pad_tensor(B, pad_bottom, new_W, C, rgb, device, dtype)
    padded_image = torch.cat([top_block, row_img, bottom_block], dim=1)
    new_H = pad_top + H + pad_bottom

    mask = torch.ones((B, new_H, new_W), dtype=dtype, device=device)
    mask[:, pad_top : pad_top + H, pad_left : pad_left + W] = 0.0

    if feather_mask_px > 0:
        f = int(feather_mask_px)
        if pad_left > 0 and f > 0:
            f_len = min(f, W)
            t = torch.linspace(0.0, 1.0, steps=f_len, dtype=dtype, device=device)
            ramp = 0.5 * (1.0 + torch.cos(math.pi * t))
            mask[:, pad_top : pad_top + H, pad_left : pad_left + f_len] = torch.maximum(
                mask[:, pad_top : pad_top + H, pad_left : pad_left + f_len],
                ramp.view(1, 1, f_len),
            )
        if pad_right > 0 and f > 0:
            f_len = min(f, W)
            t = torch.linspace(0.0, 1.0, steps=f_len, dtype=dtype, device=device)
            ramp = 0.5 * (1.0 - torch.cos(math.pi * t))
            mask[:, pad_top : pad_top + H, pad_left + W - f_len : pad_left + W] = torch.maximum(
                mask[:, pad_top : pad_top + H, pad_left + W - f_len : pad_left + W],
                ramp.view(1, 1, f_len),
            )

    return padded_image, mask, new_W, new_H


def blend_restore_original_tensor(
    reference_or_original_image: torch.Tensor,
    generated_extended_image: torch.Tensor,
    outpaint_mask: torch.Tensor,
) -> torch.Tensor:
    """Ghép khôi phục 100% độ nét của ảnh gốc vào ảnh đã mở rộng."""
    if reference_or_original_image.shape[-1] == 4:
        reference_or_original_image = reference_or_original_image[:, :, :, :3]
    if generated_extended_image.shape[-1] == 4:
        generated_extended_image = generated_extended_image[:, :, :, :3]

    B, ref_H, ref_W, ref_C = reference_or_original_image.shape
    _, gen_H, gen_W, gen_C = generated_extended_image.shape

    if gen_H != ref_H:
        scale = float(ref_H) / float(gen_H)
        target_W = int(round(gen_W * scale))
        gen_nchw = generated_extended_image.movedim(-1, 1)
        gen_resized = F.interpolate(gen_nchw, size=(ref_H, target_W), mode="bicubic", align_corners=False).movedim(1, -1).clamp(0.0, 1.0)
    else:
        gen_resized = generated_extended_image
        target_W = gen_W

    if outpaint_mask.shape[-2:] != (ref_H, target_W):
        m = outpaint_mask.unsqueeze(1) if outpaint_mask.ndim == 3 else outpaint_mask.unsqueeze(0).unsqueeze(0)
        m = F.interpolate(m.float(), size=(ref_H, target_W), mode="bilinear", align_corners=False).squeeze(1)
    else:
        m = outpaint_mask

    out = gen_resized.clone()

    if abs(ref_W - target_W) <= 32:
        if ref_W != target_W:
            ref_resized = F.interpolate(reference_or_original_image.movedim(-1, 1), size=(ref_H, target_W), mode="bicubic", align_corners=False).movedim(1, -1)
        else:
            ref_resized = reference_or_original_image
        alpha = m.unsqueeze(-1).to(out.device, out.dtype)
        ref_slice = ref_resized[:, :, :, :gen_C].to(out.device, out.dtype)
        out[:, :, :, :gen_C] = ref_slice * (1.0 - alpha) + out[:, :, :, :gen_C] * alpha
        return out

    copy_W = min(ref_W, target_W)
    left_mask_val = float(m[:, :, : max(1, copy_W // 4)].mean().item())
    right_mask_val = float(m[:, :, -max(1, copy_W // 4) :].mean().item())

    if left_mask_val <= right_mask_val and left_mask_val < 0.25:
        start_x = 0
    elif right_mask_val < left_mask_val and right_mask_val < 0.25:
        start_x = target_W - copy_W
    else:
        start_x = (target_W - copy_W) // 2

    end_x = start_x + copy_W
    alpha = m[:, :, start_x:end_x].unsqueeze(-1).to(out.device, out.dtype)
    orig_slice = reference_or_original_image[:, :, :copy_W, :gen_C].to(out.device, out.dtype)
    out[:, :, start_x:end_x, :gen_C] = orig_slice * (1.0 - alpha) + out[:, :, start_x:end_x, :gen_C] * alpha
    return out


# ==============================================================================
# NODE 1: VStone Outpaint (Tạo vùng bù Trái / Phải & Chọn màu tô Xám / Đỏ / Xanh)
# ==============================================================================
class VStone_Outpaint_Pad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "add_left_widths": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Số lần chiều rộng cộng thêm qua bên TRÁI (1.0 = +1 lần chiều rộng)."
                }),
                "add_right_widths": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Số lần chiều rộng cộng thêm qua bên PHẢI (1.0 = +1 lần chiều rộng)."
                }),
                "fill_color": (COLOR_CHOICES, {
                    "default": COLOR_CHOICES[0],
                }),
                "custom_hex": ("STRING", {
                    "default": "#808080",
                }),
                "extra_left_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "extra_right_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "extra_top_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "extra_bottom_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "feather_mask_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 4}),
            },
            "optional": {
                "base_image_for_width": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "outpaint_pad"
    CATEGORY = "vstone"

    def outpaint_pad(
        self,
        image: torch.Tensor,
        add_left_widths: float = 0.0,
        add_right_widths: float = 1.0,
        fill_color: str = COLOR_CHOICES[0],
        custom_hex: str = "#808080",
        extra_left_px: int = 0,
        extra_right_px: int = 0,
        extra_top_px: int = 0,
        extra_bottom_px: int = 0,
        feather_mask_px: int = 0,
        base_image_for_width: torch.Tensor | None = None,
    ):
        padded_img, mask, new_w, new_h = vstone_outpaint_pad_tensor(
            image=image,
            add_left_ratio=add_left_widths,
            add_right_ratio=add_right_widths,
            extra_left_px=extra_left_px,
            extra_right_px=extra_right_px,
            extra_top_px=extra_top_px,
            extra_bottom_px=extra_bottom_px,
            fill_color=fill_color,
            custom_hex=custom_hex,
            feather_mask_px=feather_mask_px,
            base_image_for_width=base_image_for_width,
        )
        return (padded_img, mask, new_w, new_h)


# ==============================================================================
# NODE 2: VStone +Qua Trái 1 Lần Chiều Rộng (Chọn Màu Tô Xám / Đỏ / Xanh)
# ==============================================================================
class VStone_ExtendLeft_1xWidth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_multiplier_left": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "fill_color": (COLOR_CHOICES, {"default": COLOR_CHOICES[0]}),
                "custom_hex": ("STRING", {"default": "#808080"}),
                "extra_left_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "feather_mask_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 4}),
            },
            "optional": {
                "base_image_for_width": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "extend_left"
    CATEGORY = "vstone"

    def extend_left(
        self,
        image: torch.Tensor,
        width_multiplier_left: float = 1.0,
        fill_color: str = COLOR_CHOICES[0],
        custom_hex: str = "#808080",
        extra_left_px: int = 0,
        feather_mask_px: int = 0,
        base_image_for_width: torch.Tensor | None = None,
        **kwargs,
    ):
        padded_img, mask, new_w, new_h = vstone_outpaint_pad_tensor(
            image=image,
            add_left_ratio=width_multiplier_left,
            add_right_ratio=0.0,
            extra_left_px=extra_left_px,
            fill_color=fill_color,
            custom_hex=custom_hex,
            feather_mask_px=feather_mask_px,
            base_image_for_width=base_image_for_width,
        )
        return (padded_img, mask, new_w, new_h)


# ==============================================================================
# NODE 3: VStone +Qua Phải 1 Lần Chiều Rộng (Chọn Màu Tô Xám / Đỏ / Xanh)
# ==============================================================================
class VStone_ExtendRight_1xWidth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_multiplier_right": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "fill_color": (COLOR_CHOICES, {"default": COLOR_CHOICES[0]}),
                "custom_hex": ("STRING", {"default": "#808080"}),
                "extra_right_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "feather_mask_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 4}),
            },
            "optional": {
                "base_image_for_width": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "extend_right"
    CATEGORY = "vstone"

    def extend_right(
        self,
        image: torch.Tensor,
        width_multiplier_right: float = 1.0,
        fill_color: str = COLOR_CHOICES[0],
        custom_hex: str = "#808080",
        extra_right_px: int = 0,
        feather_mask_px: int = 0,
        base_image_for_width: torch.Tensor | None = None,
        **kwargs,
    ):
        padded_img, mask, new_w, new_h = vstone_outpaint_pad_tensor(
            image=image,
            add_left_ratio=0.0,
            add_right_ratio=width_multiplier_right,
            extra_right_px=extra_right_px,
            fill_color=fill_color,
            custom_hex=custom_hex,
            feather_mask_px=feather_mask_px,
            base_image_for_width=base_image_for_width,
        )
        return (padded_img, mask, new_w, new_h)


# ==============================================================================
# NODE 4: Ghép khôi phục ảnh gốc (Seamless Restore)
# ==============================================================================
class VStone_SeamlessRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_extended_image": ("IMAGE",),
                "outpaint_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore_seamless"
    CATEGORY = "vstone"

    def restore_seamless(
        self,
        original_image: torch.Tensor,
        generated_extended_image: torch.Tensor,
        outpaint_mask: torch.Tensor,
    ):
        blended = blend_restore_original_tensor(
            reference_or_original_image=original_image,
            generated_extended_image=generated_extended_image,
            outpaint_mask=outpaint_mask,
        )
        return (blended,)


# ==============================================================================
# NODE 5: VStone Cloud Sync & Model Downloader (Cầu nối Laptop <-> H200 Hàn Quốc)
# ==============================================================================
class VStone_CloudSyncHub:
    """
    Node điều khiển đồng bộ trực tiếp giữa Laptop (Antigravity) và Server H200 Hàn Quốc:
    - Tự kéo Custom Node & Workflow JSON mới nhất từ GitHub và nạp nóng vào RAM ComfyUI.
    - Tải trực tiếp các LoRA / Upscaler chuẩn về đúng thư mục `ComfyUI/models/` trên H200.
    Chỉ cần đặt node này ra màn hình và bấm Queue Prompt!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": ([
                    "1. Pull Latest Nodes & Workflows from Laptop (Cập nhật Code + Workflow ngay)",
                    "2. Download Selected Model / LoRA (Tải Model/LoRA đã chọn bên dưới)",
                    "3. Pull Code + Download Model (Cập nhật tất cả)",
                ], {
                    "default": "1. Pull Latest Nodes & Workflows from Laptop (Cập nhật Code + Workflow ngay)"
                }),
                "preset_model": (list(PRESET_DOWNLOAD_CATALOG.keys()), {
                    "default": "Qwen 2.1 Outpaint LoRA v2 (Xám #808080 - 152MB)"
                }),
                "custom_url": ("STRING", {
                    "default": "",
                    "tooltip": "Tùy chọn: Dán link HuggingFace bất kỳ vào đây nếu muốn tải file ngoài danh sách."
                }),
                "custom_target_folder": ([
                    "loras",
                    "diffusion_models",
                    "vae",
                    "text_encoders",
                    "model_patches",
                    "controlnet",
                    "latent_upscale_models",
                    "upscale_models",
                    "embeddings",
                ], {
                    "default": "loras"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_report",)
    FUNCTION = "run_sync"
    OUTPUT_NODE = True
    CATEGORY = "vstone"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run_sync(
        self,
        action: str,
        preset_model: str,
        custom_url: str = "",
        custom_target_folder: str = "loras",
    ):
        reports: list[str] = []

        if action.startswith("1.") or action.startswith("3."):
            reports.append(sync_from_github_and_hot_reload(force=True))

        if action.startswith("2.") or action.startswith("3."):
            if preset_model in PRESET_DOWNLOAD_CATALOG:
                folder, fname, url = PRESET_DOWNLOAD_CATALOG[preset_model]
                if folder and fname and url:
                    reports.append(download_file_to_comfy_models(folder, fname, url))
            if custom_url and custom_url.strip().startswith("http"):
                u = custom_url.strip()
                fn = u.split("?")[0].rstrip("/").split("/")[-1]
                if fn:
                    reports.append(download_file_to_comfy_models(custom_target_folder, fn, u))

        final_report = "\n".join(reports) if reports else "Done."
        print(final_report)
        return {"ui": {"text": [final_report]}, "result": (final_report,)}


# Giữ đầy đủ tên class cũ làm alias tương thích ngược
VStone_ScreenX_ExtendCanvas = VStone_Outpaint_Pad
VStone_ImageExtendRight = VStone_ExtendRight_1xWidth
VStone_SeamlessRestoreLeft = VStone_SeamlessRestore


VSTONE_NODE_CLASS_MAPPINGS = {
    "VStone_Outpaint_Pad": VStone_Outpaint_Pad,
    "VStone_ExtendLeft_1xWidth": VStone_ExtendLeft_1xWidth,
    "VStone_ExtendRight_1xWidth": VStone_ExtendRight_1xWidth,
    "VStone_SeamlessRestore": VStone_SeamlessRestore,
    "VStone_CloudSyncHub": VStone_CloudSyncHub,
    "VStone_ScreenX_ExtendCanvas": VStone_ScreenX_ExtendCanvas,
    "VStone_ImageExtendRight": VStone_ImageExtendRight,
    "VStone_SeamlessRestoreLeft": VStone_SeamlessRestoreLeft,
}

VSTONE_NODE_DISPLAY_NAME_MAPPINGS = {
    "VStone_Outpaint_Pad": "VStone: Outpaint Tạo Vùng Bù (Chọn Màu Xám / Đỏ / Xanh)",
    "VStone_ExtendLeft_1xWidth": "VStone: Outpaint +Qua Trái 1 Lần Chiều Rộng (Chọn Màu Tô)",
    "VStone_ExtendRight_1xWidth": "VStone: Outpaint +Qua Phải 1 Lần Chiều Rộng (Chọn Màu Tô)",
    "VStone_SeamlessRestore": "VStone: Ghép Khôi Phục Ảnh Gốc (Seamless Restore)",
    "VStone_CloudSyncHub": "VStone: Cloud Sync & Model Downloader (Laptop <-> H200)",
    "VStone_ScreenX_ExtendCanvas": "VStone: Screen X Mở Rộng Khung Ngang (Trái + Phải)",
    "VStone_ImageExtendRight": "VStone: Extend Image Right (1 Frame Outpaint Pad)",
    "VStone_SeamlessRestoreLeft": "VStone: Seamless Restore Left (Blend Original)",
}

NODE_CLASS_MAPPINGS = VSTONE_NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = VSTONE_NODE_DISPLAY_NAME_MAPPINGS
