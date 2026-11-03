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
    "ALL Laptop LoRAs + Upscalers (Tự tải trọn bộ LoRA + 4x-UltraSharp + DMAD + 3D Upscaler)": ("__ALL_LAPTOP_LORAS__", "", ""),
    "ALL Laptop LoRAs (Tự tải High quality + Outpaint v2 + 360)": ("__ALL_LAPTOP_LORAS__", "", ""),
    "4x-UltraSharp.safetensors (AI Pixel Upscaler 4K-16K cho H200 & RTX - 64MB)": (
        "upscale_models",
        "4x-UltraSharp.safetensors",
        "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.safetensors",
    ),
    "Kijai DMAD 4-Step Turbo LoRA (LoRA số 1 Seed Hunter v2.5.1 cho FL2VA & REF2VA - 349MB)": (
        "loras",
        "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors",
        "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors",
    ),
    "MiniMax-H3 3D Latent Upscaler fp16 (Chuẩn Seed Hunter v2.5.1 - 658MB)": (
        "latent_upscale_models",
        "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
        "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
    ),
    "MiniMax-H3 3D Latent Upscaler bf16 (691MB)": (
        "latent_upscale_models",
        "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
        "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
    ),
    "Silveroxide Dareties REF2VA Turbo LoRA (Seed Hunter R2V - 675MB)": (
        "loras",
        "minimax_h3_ref2v_turbo_4step_v0.1_v4_step600_dareties_fro0975.safetensors",
        "https://huggingface.co/silveroxides/MiniMax-H3_tests/resolve/main/ref2v_dareties/minimax_h3_ref2v_turbo_4step_v0.1_v4_step600_dareties_fro0975.safetensors",
    ),
    "Kijai ELM Longlive 4-Step LoRA (Kéo dài Video Continuation - 1.82GB)": (
        "loras",
        "minimax_h3_ELM_longlive_plug_4step_lora_bf16.safetensors",
        "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/minimax_h3_ELM_longlive_plug_4step_lora_bf16.safetensors",
    ),
    "Minimax-h3 Singularity REF2VA Pruned v1.3 int8 (Checkpoint R2V tốt nhất Seed Hunter)": (
        "diffusion_models",
        "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
        "https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/resolve/main/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
    ),
    "High quality.safetensors (LoRA từ Laptop - 76MB)": (
        "loras",
        "High quality.safetensors",
        "https://github.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/releases/download/v1.0-loras/High_quality.safetensors",
    ),
    "Qwen 2.1 Outpaint LoRA v2 (Xám #808080 - 152MB)": (
        "loras",
        "qwen-image-2.1-outpaint-v2.safetensors",
        "https://github.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/releases/download/v1.0-loras/qwen-image-2.1-outpaint-v2.safetensors",
    ),
    "Qwen 2.1 360 Panorama Outpaint LoRA (Xanh #00FF00 - 152MB)": (
        "loras",
        "qwen_image_2_1_360_panorama_outpaint_lora_v0_1.safetensors",
        "https://github.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/releases/download/v1.0-loras/qwen_image_2_1_360_panorama_outpaint_lora_v0_1.safetensors",
    ),
    "Qwen 2.1 Outpaint LoRA v1 (Zoom-out lớn & Không Overfit - 152MB)": (
        "loras",
        "qwen-image-2.1-outpaint.safetensors",
        "https://huggingface.co/ausboss/Qwen-Image-2.1-Outpaint-LoRA/resolve/main/qwen-image-2.1-outpaint.safetensors",
    ),
    "Qwen 2.1 Consistency LoRA (Khóa vị trí & ánh sáng - 152MB)": (
        "loras",
        "qwen-image-2.1-consistency.safetensors",
        "https://huggingface.co/ausboss/Qwen-Image-2.1-Consistency-LoRA/resolve/main/qwen-image-2.1-consistency.safetensors",
    ),
    "Qwen 2.1 GH Outpaint 扩图 LoRA (Mở rộng khung hình GH - 152MB)": (
        "loras",
        "Qwen2.1_扩图_GH_3750.safetensors",
        "https://huggingface.co/KENIC-1/Turbo-Qwen-Image-2.1-Turbo-Inpaint-LORA/resolve/main/Qwen2.1_%E6%89%A9%E5%9B%BE_GH_3750.safetensors",
    ),
    "Qwen 2.1 GH Zero-Shift Consistency (Chống lệch khung GH - 152MB)": (
        "loras",
        "Qwen2.1_一致性无偏移_GH_V1.0.safetensors",
        "https://huggingface.co/KENIC-1/Turbo-Qwen-Image-2.1-Turbo-Inpaint-LORA/resolve/main/Qwen2.1_%E4%B8%80%E8%87%B4%E6%80%A7%E6%97%A0%E5%81%8F%E7%A7%BB_GH_V1.0.safetensors",
    ),
    "Qwen 2.1 Texture Fix VAE bf16 (Khử bệt mờ, tăng nét - 254MB)": (
        "vae",
        "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
        "https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1/resolve/main/texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
    ),
    "Qwen 2.1 Fix LoRA v1.0 (Khôi phục chi tiết & hạt nét)": (
        "loras",
        "qwen-image-2.1-fix-1.0-comfy.safetensors",
        "https://huggingface.co/e-n-v-y/Qwen-Image-2.1-Fix/resolve/main/qwen-image-2.1-fix-1.0-comfy.safetensors",
    ),
    "Qwen 2.1 Fun ControlNet Union Inpaint (2.68GB)": (
        "model_patches",
        "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors",
        "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors",
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
    "none (Không tải thêm model)": ("", "", ""),
}

# Danh sách LoRA & Upscaler được tự động đồng bộ sang ComfyUI/models/ trên H200
DEFAULT_AUTO_SYNC_LORAS: list[dict[str, str]] = [
    {
        "folder": "loras",
        "filename": "High quality.safetensors",
        "url": "https://github.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/releases/download/v1.0-loras/High_quality.safetensors",
    },
    {
        "folder": "loras",
        "filename": "qwen-image-2.1-outpaint-v2.safetensors",
        "url": "https://github.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/releases/download/v1.0-loras/qwen-image-2.1-outpaint-v2.safetensors",
    },
    {
        "folder": "loras",
        "filename": "qwen_image_2_1_360_panorama_outpaint_lora_v0_1.safetensors",
        "url": "https://github.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/releases/download/v1.0-loras/qwen_image_2_1_360_panorama_outpaint_lora_v0_1.safetensors",
    },
    {
        "folder": "upscale_models",
        "filename": "4x-UltraSharp.safetensors",
        "url": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.safetensors",
    },
    {
        "folder": "loras",
        "filename": "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors",
        "url": "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors",
    },
    {
        "folder": "latent_upscale_models",
        "filename": "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
        "url": "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
    },
]


# ==============================================================================
# 1. CLOUD SYNC & HOT-RELOAD ENGINE (Laptop <-> H200 Korea - ZERO API RATE LIMIT)
# ==============================================================================
import hashlib

_LAST_SYNCED_HASH: str = ""

DEFAULT_SYNC_WORKFLOWS = [
    "workflows/upscale_pro/image_rtx_vsr_pro_ultrasharp_16k.json",
    "workflows/upscale_pro/video_rtx_vsr_pro_ultrasharp_4k.json",
    "workflows/minimax_h3/minimaxH3SEEDHUNTERLatent_v251.json",
    "workflows/minimax_h3/image_minimax_h3_t2i.json",
    "workflows/minimax_h3/video_minimax_h3_1pass_v2v_upscale_ref.json",
    "workflows/minimax_h3/video_minimax_h3_stoneV_outpaint_ref.json",
    "workflows/minimax_h3/video_minimax_h3_stoneV_gen_video.json",
    "workflows/minimax_h3/video_minimax_h3_fun_controlnet_union.json",
    "workflows/minimax_h3/video_minimax_h3_i2v.json",
    "workflows/minimax_h3/video_minimax_h3_i2v_continuation.json",
    "workflows/minimax_h3/video_minimax_h3_multiframe_reference.json",
    "workflows/minimax_h3/video_minimax_h3_r2v.json",
    "workflows/minimax_h3/video_minimax_h3_t2v.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_background_removal.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_image_edit.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_t2i.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_turbo_image_edit.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_turbo_t2i.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_vstone_extend_right.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_stoneV_outpaint_lora_best.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_stoneV_screenx_plates.json",
]


def _http_get_bytes(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "VStone-ComfyUI-Sync/2.0", "Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _get_comfy_root() -> str | None:
    try:
        import folder_paths
        return folder_paths.base_path
    except Exception:
        return None


def _get_latest_commit_sha() -> str:
    """
    Lấy mã Commit SHA mới nhất trực tiếp qua giao thức `git ls-remote` (0.2 giây):
    - KHÔNG đi qua `api.github.com` -> KHÔNG BAO GIỜ bị Rate Limit 60 lần/giờ.
    - KHÔNG bị dính Cache 5 phút của `raw.githubusercontent.com/main`.
    """
    git_bin = shutil.which("git")
    if git_bin:
        try:
            res = subprocess.run(
                [git_bin, "ls-remote", f"https://github.com/{GITHUB_REPO}.git", f"refs/heads/{GITHUB_BRANCH}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                sha = res.stdout.strip().split()[0]
                if len(sha) >= 7:
                    return sha
        except Exception:
            pass
    return GITHUB_BRANCH


def sync_from_github_and_hot_reload(force: bool = False) -> str:
    """
    Đồng bộ tức thì (0 giây trễ Cache, 0 lỗi Rate Limit):
    1. Lấy mã commit thực tế bằng `git ls-remote`.
    2. Tải trực tiếp từ `raw.githubusercontent.com/<COMMIT_SHA>/...` (bỏ qua hoàn toàn cache 5 phút của nhánh main).
    3. Ghi đè vào CẢ 3 vị trí (`custom_nodes/vstone/`, `custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone/`, `custom_nodes/`)
       để bản cũ không bao giờ đè lên bản mới.
    4. Nạp nóng (Hot-Reload) toàn bộ Node mới vào `nodes.NODE_CLASS_MAPPINGS` trong RAM ComfyUI.
    """
    global _LAST_SYNCED_HASH
    comfy_root = _get_comfy_root()
    if not comfy_root:
        return "[VStone Sync] Không tìm thấy thư mục gốc ComfyUI."

    logs: list[str] = []
    try:
        commit_ref = _get_latest_commit_sha()
        raw_ref_base = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{commit_ref}"
        ts = int(time.time())

        helper_url = f"{raw_ref_base}/vstone/node_helper_vstone.py?_t={ts}"
        helper_bytes = _http_get_bytes(helper_url, timeout=20)
        new_hash = commit_ref[:7] if commit_ref != GITHUB_BRANCH else hashlib.sha256(helper_bytes).hexdigest()[:7]

        if not force and new_hash and new_hash == _LAST_SYNCED_HASH:
            return f"[VStone Sync] Đã ở bản mới nhất ({new_hash})."

        custom_nodes_dir = os.path.join(comfy_root, "custom_nodes")
        vstone_dir = os.path.join(custom_nodes_dir, "vstone")
        aio_vstone_dir = os.path.join(custom_nodes_dir, "ComfyUI_Qwen21_MiniMaxH3_AIO", "vstone")
        sibling_repo_vstone = os.path.join(os.path.dirname(comfy_root), "ComfyUI-Qwen21-MiniMaxH3-AIO", "vstone")
        workflows_dir = os.path.join(comfy_root, "user", "default", "workflows")

        os.makedirs(vstone_dir, exist_ok=True)
        os.makedirs(workflows_dir, exist_ok=True)

        # 1. Ghi node_helper_vstone.py vào TẤT CẢ các thư mục custom_nodes để không bị bản cũ đè
        target_helper_paths = [
            os.path.join(vstone_dir, "node_helper_vstone.py"),
            os.path.join(custom_nodes_dir, "node_helper_vstone.py"),
        ]
        if os.path.isdir(os.path.join(custom_nodes_dir, "ComfyUI_Qwen21_MiniMaxH3_AIO")):
            os.makedirs(aio_vstone_dir, exist_ok=True)
            target_helper_paths.append(os.path.join(aio_vstone_dir, "node_helper_vstone.py"))
        if os.path.isdir(sibling_repo_vstone):
            target_helper_paths.append(os.path.join(sibling_repo_vstone, "node_helper_vstone.py"))

        for dst in target_helper_paths:
            try:
                with open(dst, "wb") as f:
                    f.write(helper_bytes)
                logs.append(f"Updated: {dst}")
            except Exception as e:
                logs.append(f"Skip {dst}: {e}")

        # 2. Ghi __init__.py
        try:
            init_bytes = _http_get_bytes(f"{raw_ref_base}/vstone/__init__.py?_t={ts}", timeout=15)
            with open(os.path.join(vstone_dir, "__init__.py"), "wb") as f:
                f.write(init_bytes)
            if os.path.isdir(aio_vstone_dir):
                with open(os.path.join(aio_vstone_dir, "__init__.py"), "wb") as f:
                    f.write(init_bytes)
        except Exception:
            pass

        # 3. Đồng bộ danh sách Workflows & LoRAs theo commit mới nhất
        wf_list = list(DEFAULT_SYNC_WORKFLOWS)
        lora_list = list(DEFAULT_AUTO_SYNC_LORAS)
        try:
            manifest_bytes = _http_get_bytes(f"{raw_ref_base}/sync_manifest.json?_t={ts}", timeout=10)
            manifest_data = json.loads(manifest_bytes.decode("utf-8"))
            if isinstance(manifest_data.get("workflows"), list):
                wf_list = manifest_data["workflows"]
            if isinstance(manifest_data.get("loras"), list):
                lora_list = manifest_data["loras"]
        except Exception:
            pass

        wf_count = 0
        for path in wf_list:
            if not (path.startswith("workflows/") and path.endswith(".json")):
                continue
            rel_wf = path[len("workflows/"):]
            dst_wf = os.path.join(workflows_dir, rel_wf)
            try:
                os.makedirs(os.path.dirname(dst_wf), exist_ok=True)
                wf_bytes = _http_get_bytes(f"{raw_ref_base}/{path}?_t={ts}", timeout=15)
                with open(dst_wf, "wb") as f:
                    f.write(wf_bytes)
                wf_count += 1
            except Exception:
                pass

        # 3b. Hot-Reload trực tiếp các Node mới vào RAM của ComfyUI TRƯỚC khi tải file nặng
        updated_py_code = helper_bytes.decode("utf-8", errors="replace")
        try:
            import nodes
            ns: dict[str, Any] = {"__name__": "vstone_hot_reload", "_DISABLE_DAEMON_ON_EXEC": True}
            exec(compile(updated_py_code, "node_helper_vstone.py", "exec"), ns)
            new_classes = ns.get("NODE_CLASS_MAPPINGS", {})
            new_names = ns.get("NODE_DISPLAY_NAME_MAPPINGS", {})
            if new_classes:
                for _k, _cls in new_classes.items():
                    try:
                        if not getattr(_cls, "RELATIVE_PYTHON_MODULE", None):
                            _cls.RELATIVE_PYTHON_MODULE = "custom_nodes.vstone"
                    except Exception:
                        pass
                nodes.NODE_CLASS_MAPPINGS.update(new_classes)
                nodes.NODE_DISPLAY_NAME_MAPPINGS.update(new_names)
                logs.append(f"Hot-reloaded {len(new_classes)} VStone nodes into ComfyUI RAM: {list(new_classes.keys())}")
            if callable(ns.get("apply_comfy_runtime_fixes")):
                patch_msg = ns["apply_comfy_runtime_fixes"]()
                if patch_msg:
                    logs.append(patch_msg)
        except Exception as e:
            logs.append(f"Hot-reload warning: {e}")

        # 4. Tự động tải các LoRA / Upscaler về ComfyUI/models/ nếu chưa có
        for l_item in lora_list:
            folder = l_item.get("folder", "loras")
            fname = l_item.get("filename", "")
            url = l_item.get("url", "")
            if folder and fname and url:
                try:
                    dl_res = download_file_to_comfy_models(folder, fname, url)
                    logs.append(dl_res)
                except Exception as e:
                    logs.append(f"LoRA sync skip {fname}: {e}")

        _LAST_SYNCED_HASH = new_hash
        msg = f"[VStone Cloud Sync] Đồng bộ thành công commit {new_hash} ({wf_count} workflows)! Nhấn phím R hoặc F5 trên trình duyệt để làm mới."
        print(msg)
        return msg + "\n" + "\n".join(logs)
    except Exception as e:
        err_msg = f"[VStone Cloud Sync] Lỗi kết nối: {e}"
        return err_msg


def _get_target_node_classes(node_name: str, module_name: str | None = None) -> list[Any]:
    """Lấy tất cả các bản thể class của 1 node trong cả `nodes.NODE_CLASS_MAPPINGS` và `comfy_extras`."""
    classes: list[Any] = []
    try:
        import nodes
        c = getattr(nodes, node_name, None)
        if c is not None and c not in classes:
            classes.append(c)
        c_map = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(node_name)
        if c_map is not None and c_map not in classes:
            classes.append(c_map)
    except Exception:
        pass
    if module_name:
        try:
            mod = __import__(module_name, fromlist=[node_name])
            c_mod = getattr(mod, node_name, None)
            if c_mod is not None and c_mod not in classes:
                classes.append(c_mod)
        except Exception:
            pass
    return classes


VSTONE_RUNTIME_PATCH_VER = 4
MINIMAX_SAFE_MAX_FRAMES = 73  # 17 * 4 + 5 (~3.04s @ 24fps)
MINIMAX_SAFE_DI_W = 1280
MINIMAX_SAFE_DI_H = 704


def _vstone_free_vram(unload_models: bool = False) -> None:
    """Giải phóng bộ nhớ đệm CUDA và (tuỳ chọn) gỡ các model cũ khỏi VRAM để tránh tràn VRAM trên H200."""
    try:
        import gc
        import comfy.model_management
        if unload_models:
            comfy.model_management.unload_all_models()
        gc.collect()
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_unwrapped_orig(cls, attr_name: str, is_classmethod: bool = True):
    """Lấy hàm gốc chưa bị bọc để có thể nâng cấp bản vá nóng (v1 -> v2) ngay trong RAM mà không cần restart ComfyUI."""
    store_key = f"_vstone_orig_{attr_name}"
    if hasattr(cls, store_key):
        return getattr(cls, store_key)
    cur = getattr(cls, attr_name)
    raw_fn = cur.__func__ if (is_classmethod and hasattr(cur, "__func__")) else cur
    if getattr(cls, "_vstone_patched", False) and getattr(raw_fn, "__closure__", None):
        for cell in raw_fn.__closure__:
            try:
                contents = cell.cell_contents
                if callable(contents):
                    raw_fn = contents
                    break
            except Exception:
                pass
    setattr(cls, store_key, raw_fn)
    return raw_fn


def _chunked_cpu_resize(images: torch.Tensor, width: int, height: int, method: str = "lanczos", crop: str = "center", chunk_size: int = 16) -> torch.Tensor:
    """Phóng to/thu nhỏ tensor video [N, H, W, C] theo từng cụm 16 frame trên CPU để không làm vọt VRAM/RAM."""
    import comfy.utils
    if images is None or not hasattr(images, "shape") or images.ndim != 4:
        return images
    n, h, w, _ = images.shape
    if w == width and h == height:
        return images
    out_chunks = []
    for i in range(0, n, chunk_size):
        sub = images[i : i + chunk_size, ..., :3].to("cpu", dtype=torch.float32).movedim(-1, 1).contiguous()
        scaled = comfy.utils.common_upscale(sub, int(width), int(height), method, crop)
        out_chunks.append(scaled.movedim(1, -1).contiguous())
    return torch.cat(out_chunks, dim=0)


def apply_comfy_runtime_fixes() -> str:
    """
    Vá nóng (Hot-Patch v2 - H200 VRAM Guard) trực tiếp trong RAM của ComfyUI:
    1. `ImageScale`: Chấp nhận `' center'` + chia nhỏ theo cụm 16 frame trên CPU để không ngốn VRAM khi upscale video.
    2. `GetVideoComponents`: Tự động hạ FPS > 30 về ~24fps, giới hạn tối đa 73 frames (chuẩn 17k+5 của MiniMax-H3), tự sinh/cắt âm thanh Stereo 32kHz khớp thời lượng video.
    3. `VAEEncodeAudio`: Tự động tạo âm thanh tĩnh Stereo 32kHz khi `audio is None`, giới hạn thời lượng hợp lý để tránh tràn VRAM.
    4. `VAEEncode` & `VAEDecode`: Tự động đưa độ phân giải vào vùng an toàn chuẩn DiT (`1280x704`) khi encode bằng MiniMax VideoVAE, xả sạch model cũ trước khi chạy `ViT3DDecoder`, và tự động phóng to hậu kỳ lên Full HD (`1920x1088`).
    5. `LTXVConcatAVLatent`: Tự động căn chỉnh độ dài `audio_latent` khớp chuẩn `round(frame_count / 24 * 40)` với `video_latent`.
    6. `MiniMaxH3ReferenceToVideo` & `KSampler`: Giữ nguyên ảnh và video tham chiếu, tự động xả `Qwen3-VL-32B` (22GB) và `Qwen-Image` khỏi VRAM trước khi chạy Sampler.
    """
    patched: list[str] = []

    # 1. Patch ImageScale (fix ' center' leading space + chunked CPU video scaling)
    try:
        for img_scale_cls in _get_target_node_classes("ImageScale"):
            if hasattr(img_scale_cls, "crop_methods") and isinstance(img_scale_cls.crop_methods, list):
                for v in (" center", "center ", " disabled", "disabled "):
                    if v not in img_scale_cls.crop_methods:
                        img_scale_cls.crop_methods.append(v)
            if getattr(img_scale_cls, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                orig_upscale = _get_unwrapped_orig(img_scale_cls, "upscale", is_classmethod=False)

                def _make_safe_upscale(_orig):
                    def _safe_upscale(self, image, upscale_method, width, height, crop="center"):
                        if isinstance(crop, str):
                            crop = crop.strip() or "center"
                        if crop not in ("disabled", "center"):
                            crop = "center"
                        if image is not None and hasattr(image, "shape") and image.ndim == 4 and image.shape[0] > 8:
                            return (_chunked_cpu_resize(image, int(width), int(height), upscale_method, crop, chunk_size=16),)
                        return _orig(self, image, upscale_method, width, height, crop)
                    return _safe_upscale

                @classmethod  # type: ignore
                def _validate_image_scale(cls, **kwargs):
                    return True

                img_scale_cls.upscale = _make_safe_upscale(orig_upscale)
                img_scale_cls.VALIDATE_INPUTS = _validate_image_scale
                setattr(img_scale_cls, "_vstone_patched", True)
                setattr(img_scale_cls, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
                if "ImageScale" not in patched:
                    patched.append("ImageScale")
    except Exception:
        pass

    # 2. Patch GetVideoComponents (cap frames to 73 / 17k+5 & synthesize/trim stereo 32kHz audio)
    try:
        for gvc_cls in _get_target_node_classes("GetVideoComponents", "comfy_extras.nodes_video"):
            if getattr(gvc_cls, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                orig_gvc_func = _get_unwrapped_orig(gvc_cls, "execute", is_classmethod=True)

                def _make_safe_gvc(_orig_fn):
                    @classmethod  # type: ignore
                    def _safe_gvc_execute(cls, video):
                        res = _orig_fn(cls, video)
                        try:
                            args = list(res.args) if hasattr(res, "args") else list(res)
                            if len(args) >= 3:
                                imgs, aud, fps_val = args[0], args[1], float(args[2] or 24.0)
                                if imgs is not None and hasattr(imgs, "shape") and imgs.ndim == 4:
                                    if fps_val > 30.0 and imgs.shape[0] > 24:
                                        step = max(1, int(round(fps_val / 24.0)))
                                        if step > 1:
                                            imgs = imgs[::step].contiguous()
                                            fps_val = fps_val / float(step)
                                            args[2] = fps_val
                                    if imgs.shape[0] > MINIMAX_SAFE_MAX_FRAMES:
                                        imgs = imgs[:MINIMAX_SAFE_MAX_FRAMES].contiguous()
                                    if imgs.shape[0] < 5:
                                        rep = (5 + imgs.shape[0] - 1) // max(1, imgs.shape[0])
                                        imgs = imgs.repeat(rep, 1, 1, 1)[:5].contiguous()
                                    else:
                                        valid_n = imgs.shape[0]
                                        while valid_n > 5 and valid_n % 17 != 5:
                                            valid_n -= 1
                                        if valid_n != imgs.shape[0]:
                                            imgs = imgs[:valid_n].contiguous()
                                    args[0] = imgs

                                n_frames = int(imgs.shape[0]) if (imgs is not None and hasattr(imgs, "shape")) else 24
                                dur_sec = max(0.25, float(n_frames) / 24.0)
                                sr = 32000
                                n_samples = max(1600, int(round(dur_sec * sr)))
                                if aud is None or not isinstance(aud, dict) or aud.get("waveform") is None:
                                    aud = {
                                        "waveform": torch.zeros((1, 2, n_samples), dtype=torch.float32),
                                        "sample_rate": sr,
                                    }
                                else:
                                    wf = aud["waveform"]
                                    cur_sr = int(aud.get("sample_rate", sr))
                                    target_samples = max(1600, int(round(dur_sec * cur_sr)))
                                    if wf.ndim == 2:
                                        wf = wf.unsqueeze(0)
                                    if wf.shape[1] == 1:
                                        wf = wf.repeat(1, 2, 1)
                                    elif wf.shape[1] > 2:
                                        wf = wf[:, :2, :]
                                    if wf.shape[-1] == 0:
                                        wf = torch.zeros((1, 2, target_samples), dtype=torch.float32)
                                    elif wf.shape[-1] > target_samples:
                                        wf = wf[..., :target_samples]
                                    elif wf.shape[-1] < target_samples:
                                        wf = F.pad(wf, (0, target_samples - wf.shape[-1]))
                                    aud = {
                                        "waveform": wf.contiguous(),
                                        "sample_rate": cur_sr,
                                    }
                                args[1] = aud
                                if hasattr(res, "args"):
                                    res.args = tuple(args)
                                else:
                                    return tuple(args)
                        except Exception:
                            pass
                        return res
                    return _safe_gvc_execute

                gvc_cls.execute = _make_safe_gvc(orig_gvc_func)
                setattr(gvc_cls, "_vstone_patched", True)
                setattr(gvc_cls, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
                if "GetVideoComponents" not in patched:
                    patched.append("GetVideoComponents")
    except Exception:
        pass

    # 3. Patch VAEEncodeAudio (fix ValueError: input audio is None on silent videos & cap audio length)
    try:
        for vae_aud_cls in _get_target_node_classes("VAEEncodeAudio", "comfy_extras.nodes_audio"):
            if getattr(vae_aud_cls, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                orig_aud_func = _get_unwrapped_orig(vae_aud_cls, "execute", is_classmethod=True)

                def _make_safe_vae_aud(_orig_fn):
                    @classmethod  # type: ignore
                    def _safe_vae_encode_audio_execute(cls, vae, audio=None):
                        vae_sr = int(getattr(vae, "audio_sample_rate", 32000))
                        default_samples = int(round((MINIMAX_SAFE_MAX_FRAMES / 24.0) * vae_sr))
                        if audio is None or not isinstance(audio, dict) or audio.get("waveform") is None:
                            audio = {
                                "waveform": torch.zeros((1, 2, default_samples), dtype=torch.float32),
                                "sample_rate": vae_sr,
                            }
                        else:
                            wf = audio["waveform"]
                            cur_sr = int(audio.get("sample_rate", vae_sr))
                            max_cur = cur_sr * 6
                            if wf.ndim == 2:
                                wf = wf.unsqueeze(0)
                            if wf.shape[1] == 1:
                                wf = wf.repeat(1, 2, 1)
                            elif wf.shape[1] > 2:
                                wf = wf[:, :2, :]
                            if wf.shape[-1] > max_cur:
                                wf = wf[..., :max_cur]
                            elif wf.shape[-1] < 1600:
                                wf = F.pad(wf, (0, 1600 - wf.shape[-1]))
                            audio = {
                                "waveform": wf.contiguous(),
                                "sample_rate": cur_sr,
                            }
                        return _orig_fn(cls, vae, audio)
                    return _safe_vae_encode_audio_execute

                safe_fn = _make_safe_vae_aud(orig_aud_func)
                vae_aud_cls.execute = safe_fn
                vae_aud_cls.encode = safe_fn
                setattr(vae_aud_cls, "_vstone_patched", True)
                setattr(vae_aud_cls, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
                if "VAEEncodeAudio" not in patched:
                    patched.append("VAEEncodeAudio")
    except Exception:
        pass

    # 4. Patch VAEEncode & VAEDecode (auto-cap MiniMax-H3 video resolution to 1280x704 during DiT & restore 1920x1088 after decode)
    try:
        for vae_enc_cls in _get_target_node_classes("VAEEncode"):
            if getattr(vae_enc_cls, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                orig_enc = _get_unwrapped_orig(vae_enc_cls, "encode", is_classmethod=False)

                def _make_safe_vae_enc(_orig):
                    def _safe_vae_enc(self, vae, pixels):
                        try:
                            is_minimax_vae = (
                                getattr(vae, "latent_channels", 0) == 24
                                and getattr(vae, "latent_dim", 0) == 3
                            )
                            if is_minimax_vae and pixels is not None and hasattr(pixels, "shape") and pixels.ndim == 4 and pixels.shape[0] > 1:
                                _vstone_free_vram(unload_models=False)
                                if pixels.shape[0] > MINIMAX_SAFE_MAX_FRAMES:
                                    pixels = pixels[:MINIMAX_SAFE_MAX_FRAMES].contiguous()
                                h, w = int(pixels.shape[1]), int(pixels.shape[2])
                                if w * h > 1344 * 768:
                                    setattr(vae, "_vstone_upscale_target", (w, h))
                                    ratio = float(w) / max(1.0, float(h))
                                    if ratio >= 1.0:
                                        tw, th = MINIMAX_SAFE_DI_W, MINIMAX_SAFE_DI_H
                                    else:
                                        tw, th = MINIMAX_SAFE_DI_H, MINIMAX_SAFE_DI_W
                                    pixels = _chunked_cpu_resize(pixels, tw, th, "lanczos", "center", chunk_size=16)
                                else:
                                    setattr(vae, "_vstone_upscale_target", None)
                        except Exception:
                            pass
                        res = _orig(self, vae, pixels)
                        _vstone_free_vram(unload_models=False)
                        return res
                    return _safe_vae_enc

                vae_enc_cls.encode = _make_safe_vae_enc(orig_enc)
                setattr(vae_enc_cls, "_vstone_patched", True)
                setattr(vae_enc_cls, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
                if "VAEEncode" not in patched:
                    patched.append("VAEEncode")
    except Exception:
        pass

    try:
        for vae_dec_cls in _get_target_node_classes("VAEDecode"):
            if getattr(vae_dec_cls, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                orig_dec = _get_unwrapped_orig(vae_dec_cls, "decode", is_classmethod=False)

                def _make_safe_vae_dec(_orig):
                    def _safe_vae_dec(self, vae, samples):
                        is_minimax_vae = (
                            getattr(vae, "latent_channels", 0) == 24
                            and getattr(vae, "latent_dim", 0) == 3
                        )
                        if is_minimax_vae:
                            # Giải phóng DiT (28GB) khỏi VRAM trước khi chạy ViT3DDecoder
                            _vstone_free_vram(unload_models=True)
                        res = _orig(self, vae, samples)
                        try:
                            if is_minimax_vae:
                                _vstone_free_vram(unload_models=False)
                                target_wh = getattr(vae, "_vstone_upscale_target", None)
                                if target_wh is not None and isinstance(res, tuple) and len(res) >= 1:
                                    imgs = res[0]
                                    tw, th = int(target_wh[0]), int(target_wh[1])
                                    if imgs is not None and hasattr(imgs, "shape") and imgs.ndim == 4:
                                        if int(imgs.shape[2]) != tw or int(imgs.shape[1]) != th:
                                            imgs = _chunked_cpu_resize(imgs, tw, th, "lanczos", "center", chunk_size=16)
                                            res = (imgs,) + tuple(res[1:])
                                    setattr(vae, "_vstone_upscale_target", None)
                        except Exception:
                            pass
                        return res
                    return _safe_vae_dec

                vae_dec_cls.decode = _make_safe_vae_dec(orig_dec)
                setattr(vae_dec_cls, "_vstone_patched", True)
                setattr(vae_dec_cls, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
                if "VAEDecode" not in patched:
                    patched.append("VAEDecode")
    except Exception:
        pass

    # 5. Patch LTXVConcatAVLatent (auto-fit audio_latent length to MiniMax-H3 video_latent)
    try:
        for concat_cls in _get_target_node_classes("LTXVConcatAVLatent", "comfy_extras.nodes_lt"):
            if getattr(concat_cls, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                orig_concat_func = _get_unwrapped_orig(concat_cls, "execute", is_classmethod=True)

                def _make_safe_concat(_orig_fn):
                    @classmethod  # type: ignore
                    def _safe_concat_av_execute(cls, video_latent, audio_latent):
                        try:
                            v_samp = video_latent.get("samples") if isinstance(video_latent, dict) else None
                            a_samp = audio_latent.get("samples") if isinstance(audio_latent, dict) else None
                            if (
                                v_samp is not None
                                and a_samp is not None
                                and not getattr(v_samp, "is_nested", False)
                                and v_samp.ndim == 5
                                and v_samp.shape[1] == 24
                                and a_samp.ndim == 4
                            ):
                                latent_t = int(v_samp.shape[2])
                                frame_per_tok = (1, 4, 4, 4, 4)
                                frame_count = sum(frame_per_tok[k % 5] for k in range(latent_t))
                                target_audio_t = max(1, int(round((frame_count / 24.0) * 40.0)))
                                if a_samp.shape[2] == 1:
                                    a_samp = a_samp.repeat(1, 1, 2, 1)
                                elif a_samp.shape[2] > 2:
                                    a_samp = a_samp[:, :, :2, :]
                                cur_t = int(a_samp.shape[-1])
                                if cur_t > target_audio_t:
                                    a_samp = a_samp[..., :target_audio_t].contiguous()
                                elif cur_t < target_audio_t:
                                    pad_t = target_audio_t - cur_t
                                    last_frame = a_samp[..., -1:] if cur_t > 0 else torch.zeros_like(a_samp[..., :1])
                                    a_samp = torch.cat([a_samp, last_frame.repeat(1, 1, 1, pad_t)], dim=-1).contiguous()
                                audio_latent = dict(audio_latent)
                                audio_latent["samples"] = a_samp
                        except Exception:
                            pass
                        return _orig_fn(cls, video_latent, audio_latent)
                    return _safe_concat_av_execute

                concat_cls.execute = _make_safe_concat(orig_concat_func)
                setattr(concat_cls, "_vstone_patched", True)
                setattr(concat_cls, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
                if "LTXVConcatAVLatent" not in patched:
                    patched.append("LTXVConcatAVLatent")
    except Exception:
        pass

    # 6. Patch MiniMaxH3ReferenceToVideo (unload Qwen3-VL-32B after conditioning, references pass through unchanged)
    try:
        for r2v_cls in _get_target_node_classes("MiniMaxH3ReferenceToVideo", "comfy_extras.nodes_minimax_h3"):
            if getattr(r2v_cls, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                orig_r2v_func = _get_unwrapped_orig(r2v_cls, "execute", is_classmethod=True)

                def _make_safe_r2v(_orig_fn):
                    @classmethod  # type: ignore
                    def _safe_r2v_execute(
                        cls,
                        clip,
                        prompt,
                        width,
                        height,
                        length,
                        ref_image_size="match",
                        vae=None,
                        audio_vae=None,
                        ref_images=None,
                        ref_videos=None,
                        ref_video_audios=None,
                        ref_audios=None,
                        **kwargs,
                    ):
                        _vstone_free_vram(unload_models=True)

                        if ref_videos is None:
                            ref_videos = {}
                        else:
                            ref_videos = dict(ref_videos)
                        if ref_video_audios is None:
                            ref_video_audios = {}
                        else:
                            ref_video_audios = dict(ref_video_audios)

                        for k, v in list(kwargs.items()):
                            if k.startswith("ref_videos.") or k.startswith("ref_video_"):
                                short_k = k.split(".")[-1]
                                ref_videos[short_k] = v
                            elif k.startswith("ref_video_audios.") or k.startswith("ref_video_audio_"):
                                short_k = k.split(".")[-1]
                                ref_video_audios[short_k] = v

                        res = _orig_fn(
                            cls,
                            clip=clip,
                            prompt=prompt,
                            width=width,
                            height=height,
                            length=length,
                            ref_image_size=ref_image_size,
                            vae=vae,
                            audio_vae=audio_vae,
                            ref_images=ref_images,
                            ref_videos=ref_videos,
                            ref_video_audios=ref_video_audios,
                            ref_audios=ref_audios,
                        )
                        # Giải phóng ngay Qwen3-VL-32B (22GB) khỏi VRAM trước khi KSampler nạp MiniMax-H3 DiT
                        _vstone_free_vram(unload_models=True)
                        return res
                    return _safe_r2v_execute

                r2v_cls.execute = _make_safe_r2v(orig_r2v_func)
                setattr(r2v_cls, "_vstone_patched", True)
                setattr(r2v_cls, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
                if "MiniMaxH3ReferenceToVideo" not in patched:
                    patched.append("MiniMaxH3ReferenceToVideo")
    except Exception:
        pass

    # 7. Patch KSampler (flush VRAM before and after sampling so Qwen/VAE and DiT never collide in VRAM)
    try:
        for ks_cls in _get_target_node_classes("KSampler"):
            if getattr(ks_cls, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                orig_sample = _get_unwrapped_orig(ks_cls, "sample", is_classmethod=False)

                def _make_safe_ks(_orig):
                    def _safe_ks_sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
                        _vstone_free_vram(unload_models=True)
                        res = _orig(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
                        _vstone_free_vram(unload_models=False)
                        return res
                    return _safe_ks_sample

                ks_cls.sample = _make_safe_ks(orig_sample)
                setattr(ks_cls, "_vstone_patched", True)
                setattr(ks_cls, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
                if "KSampler" not in patched:
                    patched.append("KSampler")
    except Exception:
        pass

    # 8. Patch LoraLoader & LoraLoaderModelOnly (tự động ánh xạ tên file & tự tải LoRA như Kijai DMAD nếu trên H200 chưa có)
    try:
        import folder_paths
        for lora_cls_name in ("LoraLoader", "LoraLoaderModelOnly"):
            for l_cls in _get_target_node_classes(lora_cls_name):
                if getattr(l_cls, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                    orig_load = _get_unwrapped_orig(l_cls, "load_lora", is_classmethod=False)

                    def _make_safe_lora(_orig):
                        def _safe_load_lora(self, model, clip, lora_name, strength_model, strength_clip):
                            clean_name = str(lora_name).replace("\\", "/").split("/")[-1]
                            if folder_paths.get_full_path("loras", lora_name) is None:
                                if folder_paths.get_full_path("loras", clean_name) is not None:
                                    lora_name = clean_name
                                else:
                                    for _k, (fld, fn, url) in PRESET_DOWNLOAD_CATALOG.items():
                                        if fld == "loras" and fn == clean_name and url:
                                            download_file_to_comfy_models("loras", clean_name, url)
                                            lora_name = clean_name
                                            break
                            return _orig(self, model, clip, lora_name, strength_model, strength_clip)
                        return _safe_load_lora

                    l_cls.load_lora = _make_safe_lora(orig_load)
                    l_cls.VALIDATE_INPUTS = classmethod(lambda cls, **kwargs: True)
                    setattr(l_cls, "_vstone_patched", True)
                    setattr(l_cls, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
                    if lora_cls_name not in patched:
                        patched.append(lora_cls_name)
    except Exception:
        pass

    if patched:
        msg = f"[VStone Runtime Fix v{VSTONE_RUNTIME_PATCH_VER}] Đã vá nóng & bật H200 VRAM Guard trong RAM: {', '.join(patched)}"
        print(msg)
        return msg
    return f"[VStone Runtime Fix v{VSTONE_RUNTIME_PATCH_VER}] H200 VRAM Guard đã hoạt động sẵn."


def _clear_comfy_model_cache() -> None:
    try:
        import folder_paths
        if hasattr(folder_paths, "filename_list_cache"):
            folder_paths.filename_list_cache.clear()
    except Exception:
        pass


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
            _clear_comfy_model_cache()
            return f"[VStone Downloader] Tải thành công bằng aria2c: {target_path}"

    curl_bin = shutil.which("curl")
    if curl_bin:
        cmd = [curl_bin, "-4", "-L", url, "-o", target_path]
        res = subprocess.run(cmd, check=False)
        if res.returncode == 0 and os.path.exists(target_path):
            _clear_comfy_model_cache()
            return f"[VStone Downloader] Tải thành công bằng curl: {target_path}"

    urllib.request.urlretrieve(url, target_path)
    _clear_comfy_model_cache()
    return f"[VStone Downloader] Tải thành công: {target_path}"


def _start_background_sync_daemon(interval_sec: int = 30):
    """Chạy ngầm kiểm tra GitHub mỗi 30 giây khi đang chạy trong ComfyUI trên máy H200."""
    try:
        apply_comfy_runtime_fixes()
    except Exception:
        pass
    if globals().get("_DISABLE_DAEMON_ON_EXEC"):
        return
    if _get_comfy_root() is None:
        return
    if getattr(sys_mod := __import__("sys"), "_vstone_sync_daemon_started", False):
        return
    setattr(sys_mod, "_vstone_sync_daemon_started", True)

    def _loop():
        time.sleep(3)
        try:
            apply_comfy_runtime_fixes()
        except Exception:
            pass
        sync_from_github_and_hot_reload(force=False)
        while True:
            time.sleep(interval_sec)
            try:
                apply_comfy_runtime_fixes()
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


OUTER_BG_CHOICES = [
    "black (#000000 - Nền ngoài hình thang màu Đen)",
    "white (#FFFFFF - Nền ngoài hình thang màu Trắng)",
    "fill_full_wing (Tô tràn màu vùng bù cả khung chữ nhật cánh)",
]


def _parse_outer_bg_rgb(outer_bg_mode: str, pad_rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    mode = (outer_bg_mode or "").lower()
    if mode.startswith("white"):
        return (1.0, 1.0, 1.0)
    if mode.startswith("fill_full_wing"):
        return pad_rgb
    return (0.0, 0.0, 0.0)


def _build_screenx_wing_trapezoid_mask(
    new_H: int,
    wing_W: int,
    pad_top: int,
    inner_H: int,
    pad_bottom: int,
    side: str,
    device: Any,
    dtype: Any,
) -> torch.Tensor:
    """
    Tạo mask hình thang cân cho cánh Trái ('left') hoặc cánh Phải ('right') chuẩn ScreenX:
    - Cạnh sát mép tấm giữa (1): cao đúng bằng `inner_H` (từ `pad_top` đến `pad_top + inner_H`).
    - Cạnh biên ngoài cùng: bung rộng toàn chiều cao `new_H` (từ `0` đến `new_H`, tức `outer_height_ratio * inner_H`).
    Trả về tensor 2D [new_H, wing_W] giá trị 1.0 trong hình thang, 0.0 ngoài hình thang.
    """
    if wing_W <= 0 or new_H <= 0:
        return torch.zeros((max(0, new_H), max(0, wing_W)), dtype=dtype, device=device)

    y = torch.arange(new_H, device=device, dtype=torch.float32).unsqueeze(1)
    if wing_W == 1:
        t = torch.ones((1, 1), device=device, dtype=torch.float32)
    elif side == "left":
        # x = 0 là cạnh biên ngoài (t = 1.0), x = wing_W - 1 là cạnh sát tấm giữa 1 (t = 0.0)
        t = torch.linspace(1.0, 0.0, steps=wing_W, device=device, dtype=torch.float32).unsqueeze(0)
    else:
        # x = 0 là cạnh sát tấm giữa 1 (t = 0.0), x = wing_W - 1 là cạnh biên ngoài (t = 1.0)
        t = torch.linspace(0.0, 1.0, steps=wing_W, device=device, dtype=torch.float32).unsqueeze(0)

    y_top = float(pad_top) * (1.0 - t)
    y_bot = float(pad_top + inner_H) + float(pad_bottom) * t
    inside = (y >= (y_top - 0.5)) & (y < (y_bot + 0.5))
    return inside.to(dtype=dtype)


def vstone_outpaint_pad_tensor(
    image: torch.Tensor,
    add_left_ratio: float = 0.0,
    add_right_ratio: float = 0.0,
    keep_orig_ratio: float = 0.5,
    outer_height_ratio: float = 1.3,
    extra_left_px: int = 0,
    extra_right_px: int = 0,
    extra_top_px: int = 0,
    extra_bottom_px: int = 0,
    fill_color: str = COLOR_CHOICES[0],
    custom_hex: str = "#808080",
    feather_mask_px: int = 0,
    base_image_for_width: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Nhận 1 tấm ảnh gốc [B, H, W, C] (khung xanh lá):
    - Để giảm diện tích thừa khi paint với Qwen (kết hợp đưa ảnh gốc nguyên vẹn vào `image2` làm ref thứ 2),
      chỉ cắt lấy phần ảnh gốc nằm sát phía cần vẽ theo `keep_orig_ratio` (mặc định `0.5` = lấy đúng 1/2 tấm gốc):
      + Nếu mở rộng bên TRÁI: chỉ lấy nửa bên TRÁI của tấm gốc (`0.5 * W`) ghép với vùng xám bên trái (`1.0 * W`).
      + Nếu mở rộng bên PHẢI: chỉ lấy nửa bên PHẢI của tấm gốc (`0.5 * W`) ghép với vùng xám bên phải (`1.0 * W`).
    - Tô màu Xám #808080 (hoặc Đỏ / Xanh) phủ KÍN toàn bộ vùng bao quanh (cả phần mở rộng ngang lẫn 2 dải trên/dưới
      theo `outer_height_ratio = 1.3`).
    - Node cuối `VStone_ScreenX_Combine3` sẽ tự động ghép trả lại vào tấm gốc đầy đủ và cắt hình thang ScreenX!
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

    # Cắt lấy phần ảnh gốc sát bên cần vẽ (mặc định keep_orig_ratio = 0.5 -> lấy nửa tấm đúng bên cần vẽ)
    keep_ratio = max(0.1, min(1.0, float(keep_orig_ratio)))
    if pad_left > 0 and pad_right == 0 and keep_ratio < 0.999:
        keep_W = max(1, min(W, int(round(W * keep_ratio))))
        kept_orig = orig_rgb[:, :, :keep_W, :]
    elif pad_right > 0 and pad_left == 0 and keep_ratio < 0.999:
        keep_W = max(1, min(W, int(round(W * keep_ratio))))
        kept_orig = orig_rgb[:, :, W - keep_W :, :]
    else:
        keep_W = W
        kept_orig = orig_rgb

    # Tính chiều cao tổng theo outer_height_ratio và làm tròn chuẩn bội số 32 cho Qwen-Image-2.1
    effective_ratio = max(1.0, float(outer_height_ratio))
    raw_outer_H = max(H, int(round(H * effective_ratio))) + max(0, int(extra_top_px)) + max(0, int(extra_bottom_px))
    target_new_H = max(H, max(32, int(round(raw_outer_H / 32.0)) * 32))
    if target_new_H < H:
        target_new_H = ((H + 31) // 32) * 32
    total_extra_H = max(0, target_new_H - H)
    pad_top = total_extra_H // 2
    pad_bottom = total_extra_H - pad_top

    # Làm tròn chiều rộng tổng new_W theo chuẩn bội số 32 cho Qwen-Image-2.1
    raw_new_W = pad_left + keep_W + pad_right
    target_new_W = max(keep_W, max(32, int(round(raw_new_W / 32.0)) * 32))
    if target_new_W < keep_W:
        target_new_W = ((keep_W + 31) // 32) * 32
    diff_W = target_new_W - raw_new_W
    if diff_W != 0:
        if pad_left > 0 and pad_right == 0:
            pad_left = max(0, pad_left + diff_W)
        elif pad_right > 0 and pad_left == 0:
            pad_right = max(0, pad_right + diff_W)
        else:
            add_l = diff_W // 2
            pad_left = max(0, pad_left + add_l)
            pad_right = max(0, pad_right + (diff_W - add_l))

    new_W = pad_left + keep_W + pad_right
    new_H = pad_top + H + pad_bottom

    pad_rgb = _parse_fill_rgb(fill_color, custom_hex)

    # Tô kín toàn bộ khung chữ nhật [new_H, new_W] bằng màu vùng bù (Xám #808080 / Đỏ / Xanh)
    padded_image = create_solid_pad_tensor(B, new_H, new_W, C, pad_rgb, device, dtype)
    mask = torch.ones((B, new_H, new_W), dtype=dtype, device=device)

    # Đặt phần ảnh gốc được giữ lại (`kept_orig`, giữ nguyên 100% độ nét gốc, mask = 0.0)
    padded_image[:, pad_top : pad_top + H, pad_left : pad_left + keep_W, :] = kept_orig
    mask[:, pad_top : pad_top + H, pad_left : pad_left + keep_W] = 0.0

    if feather_mask_px > 0:
        f = int(feather_mask_px)
        if pad_left > 0 and f > 0:
            f_len = min(f, keep_W)
            t = torch.linspace(0.0, 1.0, steps=f_len, dtype=dtype, device=device)
            ramp = 0.5 * (1.0 + torch.cos(math.pi * t))
            mask[:, pad_top : pad_top + H, pad_left : pad_left + f_len] = torch.maximum(
                mask[:, pad_top : pad_top + H, pad_left : pad_left + f_len],
                ramp.view(1, 1, f_len),
            )
        if pad_right > 0 and f > 0:
            f_len = min(f, keep_W)
            t = torch.linspace(0.0, 1.0, steps=f_len, dtype=dtype, device=device)
            ramp = 0.5 * (1.0 - torch.cos(math.pi * t))
            mask[:, pad_top : pad_top + H, pad_left + keep_W - f_len : pad_left + keep_W] = torch.maximum(
                mask[:, pad_top : pad_top + H, pad_left + keep_W - f_len : pad_left + keep_W],
                ramp.view(1, 1, f_len),
            )

    return padded_image, mask, new_W, new_H


def _resize_bhwc(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if img.shape[1] == target_h and img.shape[2] == target_w:
        return img
    nchw = img.movedim(-1, 1)
    resized = F.interpolate(nchw, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def vstone_screenx_combine3_tensor(
    center_image: torch.Tensor,
    left_image: torch.Tensor | None = None,
    right_image: torch.Tensor | None = None,
    keep_orig_ratio: float = 0.5,
    outer_height_ratio: float = 1.3,
    wing_width_ratio: float = 1.0,
    blend_width_px: int = 32,
    center_priority: float = 0.75,
    apply_trapezoid_clip: bool = True,
    outer_bg_color: str = OUTER_BG_CHOICES[0],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ghép 3 tấm chuẩn ScreenX [2 Trái | 1 Giữa Gốc | 3 Phải]:
    - Tấm 1 (center_image): Giữ nguyên 100% ảnh gốc đầy đủ (W x H) ở trung tâm.
    - Tấm 2 (left_image) & Tấm 3 (right_image): Tự động nhận diện dù đầu vào là:
      + Ảnh 1.5W (chế độ lấy 1/2 tấm gốc `keep_orig_ratio = 0.5`: `[1.0W Wing | 0.5W Half-Orig]`),
      + Ảnh 2.0W (chế độ lấy cả tấm gốc `keep_orig_ratio = 1.0`: `[1.0W Wing | 1.0W Full-Orig]`),
      + Hoặc ảnh 1.0W (chỉ riêng cánh).
    - Hòa trộn nhẹ (blend) ở 2 đường giáp mí (2 <-> 1 và 1 <-> 3) ưu tiên giữ tấm giữa 1 (`center_priority`).
    - Cắt vát thành đúng khung hình thang ScreenX (cạnh sát tấm 1 = 1.0xH, cạnh biên ngoài bung 1.3xH).
    """
    center_rgb = center_image[:, :, :, :3]
    B, H, W, C = center_rgb.shape
    device = center_rgb.device
    dtype = center_rgb.dtype

    wing_W = max(1, int(round(W * float(wing_width_ratio))))
    keep_W = max(1, min(W, int(round(W * max(0.1, min(1.0, float(keep_orig_ratio)))))))
    eff_ratio = max(1.0, float(outer_height_ratio))
    new_H = max(H, int(round(H * eff_ratio)))
    flare_extra = new_H - H
    pad_top = flare_extra // 2
    pad_bottom = flare_extra - pad_top
    total_W = wing_W + W + wing_W

    bg_rgb = _parse_outer_bg_rgb(outer_bg_color, (0.0, 0.0, 0.0))
    combined = create_solid_pad_tensor(B, new_H, total_W, C, bg_rgb, device, dtype)

    # --- 1. Trích xuất Cánh Trái (2) và phần giao thoa với Tấm Giữa (1) ---
    left_wing = create_solid_pad_tensor(B, new_H, wing_W, C, bg_rgb, device, dtype)
    left_center_overlap: torch.Tensor | None = None

    if left_image is not None and left_image.ndim == 4:
        l_rgb = left_image[:, :, :, :3].to(device=device, dtype=dtype)
        if l_rgb.shape[0] != B:
            l_rgb = l_rgb[:1].expand(B, -1, -1, -1) if l_rgb.shape[0] == 1 else l_rgb[:B]
        _, l_H, l_W, _ = l_rgb.shape
        norm_l_W = float(l_W) * (float(new_H) / float(max(l_H, 1)))

        if norm_l_W >= (2.0 * wing_W + 0.5 * W):
            # Trường hợp ảnh 3 màn (3W)
            l_full = _resize_bhwc(l_rgb, new_H, total_W)
            left_wing = l_full[:, :, :wing_W, :]
            left_center_overlap = l_full[:, pad_top : pad_top + H, wing_W : wing_W + W, :]
        elif norm_l_W >= (wing_W + 0.15 * W):
            # Tự động phân biệt ảnh (wing_W + keep_W, ví dụ 1.5W) hay ảnh (wing_W + W, 2.0W)
            if abs(norm_l_W - (wing_W + keep_W)) <= abs(norm_l_W - (wing_W + W)):
                target_ext_W = wing_W + keep_W
            else:
                target_ext_W = wing_W + W
            l_ext = _resize_bhwc(l_rgb, new_H, target_ext_W)
            left_wing = l_ext[:, :, :wing_W, :]
            left_center_overlap = l_ext[:, pad_top : pad_top + H, wing_W:, :]
        else:
            # Trường hợp ảnh 1W chỉ chứa riêng Cánh Trái 2
            left_wing = _resize_bhwc(l_rgb, new_H, wing_W)

    # --- 2. Trích xuất Cánh Phải (3) và phần giao thoa với Tấm Giữa (1) ---
    right_wing = create_solid_pad_tensor(B, new_H, wing_W, C, bg_rgb, device, dtype)
    right_center_overlap: torch.Tensor | None = None

    if right_image is not None and right_image.ndim == 4:
        r_rgb = right_image[:, :, :, :3].to(device=device, dtype=dtype)
        if r_rgb.shape[0] != B:
            r_rgb = r_rgb[:1].expand(B, -1, -1, -1) if r_rgb.shape[0] == 1 else r_rgb[:B]
        _, r_H, r_W, _ = r_rgb.shape
        norm_r_W = float(r_W) * (float(new_H) / float(max(r_H, 1)))

        if norm_r_W >= (2.0 * wing_W + 0.5 * W):
            # Trường hợp ảnh 3 màn (3W)
            r_full = _resize_bhwc(r_rgb, new_H, total_W)
            right_wing = r_full[:, :, wing_W + W :, :]
            right_center_overlap = r_full[:, pad_top : pad_top + H, wing_W : wing_W + W, :]
        elif norm_r_W >= (wing_W + 0.15 * W):
            # Tự động phân biệt ảnh (keep_W + wing_W, ví dụ 1.5W) hay ảnh (W + wing_W, 2.0W)
            if abs(norm_r_W - (keep_W + wing_W)) <= abs(norm_r_W - (W + wing_W)):
                target_ext_W = keep_W + wing_W
            else:
                target_ext_W = W + wing_W
            ov_W = target_ext_W - wing_W
            r_ext = _resize_bhwc(r_rgb, new_H, target_ext_W)
            right_wing = r_ext[:, :, ov_W:, :]
            right_center_overlap = r_ext[:, pad_top : pad_top + H, :ov_W, :]
        else:
            # Trường hợp ảnh 1W chỉ chứa riêng Cánh Phải 3
            right_wing = _resize_bhwc(r_rgb, new_H, wing_W)

    # --- 3. Hòa trộn nhẹ ở mép nối (2 <-> 1 và 1 <-> 3) ưu tiên giữ Tấm Giữa (1) ---
    center_blended = center_rgb.clone()
    max_ov_l = left_center_overlap.shape[2] if left_center_overlap is not None else (W // 2)
    max_ov_r = right_center_overlap.shape[2] if right_center_overlap is not None else (W // 2)
    b_px = max(0, min(int(blend_width_px), max_ov_l, max_ov_r, W // 2, wing_W // 2))
    prio = max(0.50, min(0.98, float(center_priority)))

    if b_px > 0:
        # Đường cong trọng số w_center(u) tăng cực nhanh từ mép (u=0) vào trong tấm giữa (u=1)
        # giúp xóa hoàn toàn đường cắt mà vẫn ưu tiên giữ tối đa chi tiết gốc của tấm 1.
        u = torch.linspace(0.0, 1.0, steps=b_px, device=device, dtype=dtype)
        smooth_u = 3.0 * (u ** 2) - 2.0 * (u ** 3)
        gamma = max(0.20, 1.25 - prio)
        w_start = prio * 0.55  # Ngay tại sát mép nối, tấm gốc 1 vẫn giữ ~40-54% trọng số
        w_center_left = (w_start + (1.0 - w_start) * (smooth_u ** gamma)).view(1, 1, b_px, 1)

        if left_center_overlap is not None:
            # Blend nhẹ phần giao thoa vào mép trái của tấm 1 + khớp màu sát mép cánh 2
            ai_slice_l = left_center_overlap[:, :, :b_px, :]
            orig_slice_l = center_blended[:, :, :b_px, :]
            center_blended[:, :, :b_px, :] = orig_slice_l * w_center_left + ai_slice_l * (1.0 - w_center_left)

            # Khử độ lệch sáng nhẹ ở sát mép trong của Cánh Trái (2)
            edge_diff_l = (center_blended[:, :, 0:1, :] - left_wing[:, pad_top : pad_top + H, -1:, :]) * prio
            wing_decay_l = (1.0 - smooth_u).flip(0).view(1, 1, b_px, 1) * 0.5
            left_wing[:, pad_top : pad_top + H, -b_px:, :] = (
                left_wing[:, pad_top : pad_top + H, -b_px:, :] + edge_diff_l * wing_decay_l
            ).clamp(0.0, 1.0)
        elif left_image is not None:
            # Nếu đầu vào chỉ là cánh 1W rời: tự động hòa trộn ánh sáng sát mép với cột đầu tiên của tấm 1
            edge_diff_l = (center_blended[:, :, 0:1, :] - left_wing[:, pad_top : pad_top + H, -1:, :]) * prio
            wing_decay_l = (1.0 - smooth_u).flip(0).view(1, 1, b_px, 1) * 0.75
            left_wing[:, pad_top : pad_top + H, -b_px:, :] = (
                left_wing[:, pad_top : pad_top + H, -b_px:, :] + edge_diff_l * wing_decay_l
            ).clamp(0.0, 1.0)

        w_center_right = w_center_left.flip(2)
        if right_center_overlap is not None:
            ai_slice_r = right_center_overlap[:, :, -b_px:, :]
            orig_slice_r = center_blended[:, :, -b_px:, :]
            center_blended[:, :, -b_px:, :] = orig_slice_r * w_center_right + ai_slice_r * (1.0 - w_center_right)

            edge_diff_r = (center_blended[:, :, -1:, :] - right_wing[:, pad_top : pad_top + H, 0:1, :]) * prio
            wing_decay_r = (1.0 - smooth_u).view(1, 1, b_px, 1) * 0.5
            right_wing[:, pad_top : pad_top + H, :b_px, :] = (
                right_wing[:, pad_top : pad_top + H, :b_px, :] + edge_diff_r * wing_decay_r
            ).clamp(0.0, 1.0)
        elif right_image is not None:
            edge_diff_r = (center_blended[:, :, -1:, :] - right_wing[:, pad_top : pad_top + H, 0:1, :]) * prio
            wing_decay_r = (1.0 - smooth_u).view(1, 1, b_px, 1) * 0.75
            right_wing[:, pad_top : pad_top + H, :b_px, :] = (
                right_wing[:, pad_top : pad_top + H, :b_px, :] + edge_diff_r * wing_decay_r
            ).clamp(0.0, 1.0)

    # --- 4. Áp dụng Mask Hình Thang ScreenX (1.0 * H -> outer_height_ratio * H) ---
    trap_l = _build_screenx_wing_trapezoid_mask(new_H, wing_W, pad_top, H, pad_bottom, "left", device, dtype)
    trap_r = _build_screenx_wing_trapezoid_mask(new_H, wing_W, pad_top, H, pad_bottom, "right", device, dtype)

    screenx_mask = torch.zeros((B, new_H, total_W), dtype=dtype, device=device)
    screenx_mask[:, :, :wing_W] = trap_l.unsqueeze(0).expand(B, -1, -1)
    screenx_mask[:, pad_top : pad_top + H, wing_W : wing_W + W] = 1.0
    screenx_mask[:, :, wing_W + W :] = trap_r.unsqueeze(0).expand(B, -1, -1)

    if apply_trapezoid_clip and not (outer_bg_color or "").lower().startswith("fill_full_wing"):
        bg_pixel = torch.tensor(bg_rgb[:C], dtype=dtype, device=device).view(1, 1, 1, C)
        tl_4d = trap_l.unsqueeze(0).unsqueeze(-1)
        tr_4d = trap_r.unsqueeze(0).unsqueeze(-1)
        left_wing = left_wing * tl_4d + bg_pixel * (1.0 - tl_4d)
        right_wing = right_wing * tr_4d + bg_pixel * (1.0 - tr_4d)

    combined[:, :, :wing_W, :] = left_wing
    combined[:, pad_top : pad_top + H, wing_W : wing_W + W, :] = center_blended
    combined[:, :, wing_W + W :, :] = right_wing

    return combined, left_wing, center_rgb, right_wing, screenx_mask


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


def _align_minimax_h3_frames(n: int) -> int:
    """Làm tròn số frame theo lưới 17k + 5 chuẩn của MiniMax-H3 Video VAE (5, 22, 39, 56, 73, 90, 107, 124...)."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


# ==============================================================================
# NODE 1: stoneV Outpaint (Tạo vùng bù tổng hợp)
# ==============================================================================
class stoneV_Outpaint_Pad:
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
                "keep_orig_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Tỉ lệ phần ảnh gốc giữ lại sát bên cần vẽ (0.5 = chỉ lấy 1/2 tấm gốc sát bên cần vẽ để tiết kiệm diện tích paint, 1.0 = giữ nguyên cả tấm)."
                }),
                "outer_height_ratio": ("FLOAT", {
                    "default": 1.3,
                    "min": 1.0,
                    "max": 2.5,
                    "step": 0.05,
                    "tooltip": "Tỉ số bung rộng chiều cao tổng chuẩn ScreenX (1.3 = cao 1.3xH, tô xám kín cả phần mở rộng lẫn 2 dải trên/dưới)."
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
    CATEGORY = "stoneV"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def outpaint_pad(
        self,
        image: torch.Tensor,
        add_left_widths: float = 0.0,
        add_right_widths: float = 1.0,
        keep_orig_ratio: float = 0.5,
        outer_height_ratio: float = 1.3,
        fill_color: str = COLOR_CHOICES[0],
        custom_hex: str = "#808080",
        extra_left_px: int = 0,
        extra_right_px: int = 0,
        extra_top_px: int = 0,
        extra_bottom_px: int = 0,
        feather_mask_px: int = 0,
        base_image_for_width: torch.Tensor | None = None,
        **kwargs,
    ):
        padded_img, mask, new_w, new_h = vstone_outpaint_pad_tensor(
            image=image,
            add_left_ratio=add_left_widths,
            add_right_ratio=add_right_widths,
            keep_orig_ratio=keep_orig_ratio,
            outer_height_ratio=outer_height_ratio,
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
# NODE 1A: stoneV Mở Rộng Trái (Lấy 1/2 Tấm Gốc Bên Trái + Vùng Xám 1.3x)
# ==============================================================================
class stoneV_ExtendLeft_1xWidth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_multiplier_left": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "keep_orig_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "0.5 = Chỉ lấy nửa bên TRÁI của tấm gốc (0.5xW) ghép với vùng xám cần vẽ (1.0xW) -> Tổng rộng 1.5xW nhẹ và nét hơn."
                }),
                "outer_height_ratio": ("FLOAT", {
                    "default": 1.3,
                    "min": 1.0,
                    "max": 2.5,
                    "step": 0.05,
                    "tooltip": "Tỉ số bung chiều cao tổng (1.3 = cao 1.3xH, tô kín vùng xám cả bên trái lẫn 2 dải trên/dưới)."
                }),
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
    CATEGORY = "stoneV"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def extend_left(
        self,
        image: torch.Tensor,
        width_multiplier_left: float = 1.0,
        keep_orig_ratio: float = 0.5,
        outer_height_ratio: float = 1.3,
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
            keep_orig_ratio=keep_orig_ratio,
            outer_height_ratio=outer_height_ratio,
            extra_left_px=extra_left_px,
            fill_color=fill_color,
            custom_hex=custom_hex,
            feather_mask_px=feather_mask_px,
            base_image_for_width=base_image_for_width,
        )
        return (padded_img, mask, new_w, new_h)


# ==============================================================================
# NODE 1B: stoneV Mở Rộng Phải (Lấy 1/2 Tấm Gốc Bên Phải + Vùng Xám 1.3x)
# ==============================================================================
class stoneV_ExtendRight_1xWidth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_multiplier_right": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "keep_orig_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "0.5 = Chỉ lấy nửa bên PHẢI của tấm gốc (0.5xW) ghép với vùng xám cần vẽ (1.0xW) -> Tổng rộng 1.5xW nhẹ và nét hơn."
                }),
                "outer_height_ratio": ("FLOAT", {
                    "default": 1.3,
                    "min": 1.0,
                    "max": 2.5,
                    "step": 0.05,
                    "tooltip": "Tỉ số bung chiều cao tổng (1.3 = cao 1.3xH, tô kín vùng xám cả bên phải lẫn 2 dải trên/dưới)."
                }),
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
    CATEGORY = "stoneV"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def extend_right(
        self,
        image: torch.Tensor,
        width_multiplier_right: float = 1.0,
        keep_orig_ratio: float = 0.5,
        outer_height_ratio: float = 1.3,
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
            keep_orig_ratio=keep_orig_ratio,
            outer_height_ratio=outer_height_ratio,
            extra_right_px=extra_right_px,
            fill_color=fill_color,
            custom_hex=custom_hex,
            feather_mask_px=feather_mask_px,
            base_image_for_width=base_image_for_width,
        )
        return (padded_img, mask, new_w, new_h)


# ==============================================================================
# NODE 2: stoneV ScreenX Combine 3 Tấm (2 Trái + 1 Giữa Gốc + 3 Phải)
# ==============================================================================
class stoneV_ScreenX_Combine3:
    """
    Ghép 3 tấm ảnh chuẩn ScreenX [2 Trái | 1 Giữa Gốc | 3 Phải]:
    - Giữ nguyên tấm giữa (1) gốc đầy đủ (W x H).
    - Tự động nhận diện ảnh nhánh Trái (2) và Phải (3) dù ở chế độ lấy 1/2 tấm gốc (1.5W) hay cả tấm (2.0W).
    - Blend nhẹ 2 mép nối ưu tiên giữ tấm giữa (`center_priority`) và cắt khung hình thang ScreenX 1.3x.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "center_image_1": ("IMAGE",),
                "left_image_2": ("IMAGE",),
                "right_image_3": ("IMAGE",),
                "keep_orig_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Tỉ lệ phần ảnh gốc đã giữ lại ở bước 1A/1B (mặc định 0.5 = nửa tấm gốc)."
                }),
                "outer_height_ratio": ("FLOAT", {
                    "default": 1.3,
                    "min": 1.0,
                    "max": 2.5,
                    "step": 0.05,
                    "tooltip": "Tỉ số bung rộng cạnh biên ngoài chuẩn ScreenX (1.3 = cạnh sát tấm 1 bằng 1.0xH, cạnh biên ngoài bung 1.3xH)."
                }),
                "wing_width_ratio": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Tỉ lệ chiều rộng mỗi cánh (2, 3) so với chiều rộng tấm giữa (1)."
                }),
                "blend_width_px": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 512,
                    "step": 4,
                    "tooltip": "Độ rộng vùng blend nhẹ ở 2 mép nối (2 <-> 1 và 1 <-> 3)."
                }),
                "center_priority": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.50,
                    "max": 0.98,
                    "step": 0.05,
                    "tooltip": "Mức độ ưu tiên giữ tấm giữa (1) khi blend mép (càng cao càng ưu tiên giữ nguyên chi tiết tấm 1)."
                }),
                "apply_trapezoid_clip": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cắt chuẩn khung hình thang 2 - 1 - 3 theo tỉ số bung 1.3x."
                }),
                "outer_bg_color": (OUTER_BG_CHOICES, {
                    "default": OUTER_BG_CHOICES[0],
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("combined_screenx", "left_wing_2", "center_1", "right_wing_3", "screenx_mask")
    FUNCTION = "combine_screenx"
    CATEGORY = "stoneV"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def combine_screenx(
        self,
        center_image_1: torch.Tensor,
        left_image_2: torch.Tensor,
        right_image_3: torch.Tensor,
        keep_orig_ratio: float = 0.5,
        outer_height_ratio: float = 1.3,
        wing_width_ratio: float = 1.0,
        blend_width_px: int = 32,
        center_priority: float = 0.75,
        apply_trapezoid_clip: bool = True,
        outer_bg_color: str = OUTER_BG_CHOICES[0],
        **kwargs,
    ):
        combined, left_wing, center_orig, right_wing, sx_mask = vstone_screenx_combine3_tensor(
            center_image=center_image_1,
            left_image=left_image_2,
            right_image=right_image_3,
            keep_orig_ratio=keep_orig_ratio,
            outer_height_ratio=outer_height_ratio,
            wing_width_ratio=wing_width_ratio,
            blend_width_px=blend_width_px,
            center_priority=center_priority,
            apply_trapezoid_clip=apply_trapezoid_clip,
            outer_bg_color=outer_bg_color,
        )
        return (combined, left_wing, center_orig, right_wing, sx_mask)


# ==============================================================================
# NODE 3: stoneV Cloud Sync & Model Downloader (Cầu nối Laptop <-> H200 Hàn Quốc)
# ==============================================================================
class stoneV_CloudSyncHub:
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
                    "3. Pull Code + Download Model (Cập nhật tất cả)",
                    "1. Pull Latest Nodes & Workflows from Laptop (Cập nhật Code + Workflow ngay)",
                    "2. Download Selected Model / LoRA (Tải Model/LoRA đã chọn bên dưới)",
                ], {
                    "default": "3. Pull Code + Download Model (Cập nhật tất cả)"
                }),
                "preset_model": (list(PRESET_DOWNLOAD_CATALOG.keys()), {
                    "default": "ALL Laptop LoRAs (Tự tải High quality + Outpaint v2 + 360)"
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
    CATEGORY = "stoneV"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

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
                if folder == "__ALL_LAPTOP_LORAS__":
                    for item in DEFAULT_AUTO_SYNC_LORAS:
                        reports.append(
                            download_file_to_comfy_models(item["folder"], item["filename"], item["url"])
                        )
                elif folder and fname and url:
                    reports.append(download_file_to_comfy_models(folder, fname, url))
            if custom_url and custom_url.strip().startswith("http"):
                u = custom_url.strip()
                fn = u.split("?")[0].rstrip("/").split("/")[-1]
                if fn:
                    reports.append(download_file_to_comfy_models(custom_target_folder, fn, u))

        final_report = "\n".join(reports) if reports else "Done."
        print(final_report)
        return {"ui": {"text": [final_report]}, "result": (final_report,)}


# ==============================================================================
# NODE 4: stoneV Seamless Restore & Video Outpaint Prepare
# ==============================================================================
class stoneV_SeamlessRestore:
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
    RETURN_NAMES = ("final_extended_image",)
    FUNCTION = "restore"
    CATEGORY = "stoneV"

    def restore(self, original_image: torch.Tensor, generated_extended_image: torch.Tensor, outpaint_mask: torch.Tensor):
        return (blend_restore_original_tensor(original_image, generated_extended_image, outpaint_mask),)


class stoneV_Video_Outpaint_Prepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expanded_image": ("IMAGE",),
                "narrow_video": ("IMAGE",),
                "add_left_widths": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "add_right_widths": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "outer_height_ratio": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 2.5, "step": 0.05}),
                "target_megapixels": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 4.0, "step": 0.02}),
                "fill_color": (COLOR_CHOICES, {"default": COLOR_CHOICES[0]}),
                "feather_mask_px": ("INT", {"default": 16, "min": 0, "max": 256, "step": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("expanded_ref_image", "padded_source_video", "outpaint_mask", "width", "height", "length")
    FUNCTION = "prepare_video_outpaint"
    CATEGORY = "stoneV"

    def prepare_video_outpaint(
        self,
        expanded_image: torch.Tensor,
        narrow_video: torch.Tensor,
        add_left_widths: float = 1.0,
        add_right_widths: float = 1.0,
        outer_height_ratio: float = 1.3,
        target_megapixels: float = 0.98,
        fill_color: str = COLOR_CHOICES[0],
        feather_mask_px: int = 16,
    ):
        padded_vid, mask, raw_w, raw_h = vstone_outpaint_pad_tensor(
            image=narrow_video,
            add_left_ratio=add_left_widths,
            add_right_ratio=add_right_widths,
            keep_orig_ratio=1.0,
            outer_height_ratio=outer_height_ratio,
            fill_color=fill_color,
            feather_mask_px=feather_mask_px,
        )
        if target_megapixels > 0.05:
            aspect = float(raw_w) / float(max(1, raw_h))
            target_pixels = float(target_megapixels) * 1_000_000.0
            out_h = max(256, int(round(math.sqrt(target_pixels / aspect) / 32.0)) * 32)
            out_w = max(256, int(round((out_h * aspect) / 32.0)) * 32)
        else:
            out_w = max(256, int(round(raw_w / 32.0)) * 32)
            out_h = max(256, int(round(raw_h / 32.0)) * 32)

        if padded_vid.shape[1] != out_h or padded_vid.shape[2] != out_w:
            padded_vid = _resize_bhwc(padded_vid, out_h, out_w)
            mask = F.interpolate(mask.unsqueeze(1), size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(1).clamp(0.0, 1.0)

        exp_rgb = expanded_image[:, :, :, :3].to(device=padded_vid.device, dtype=padded_vid.dtype)
        if exp_rgb.shape[1] != out_h or exp_rgb.shape[2] != out_w:
            exp_rgb = _resize_bhwc(exp_rgb, out_h, out_w)

        length = _align_minimax_h3_frames(int(narrow_video.shape[0]))
        return (exp_rgb, padded_vid, mask, out_w, out_h, length)


# ==============================================================================
# RTX Video Super Resolution Pro & AI 4x-UltraSharp 16K Upscaler
# ==============================================================================
_ULTRASHARP_CACHE: dict[str, Any] = {}
_VERIFIED_VSR_MAX_EDGE = 15360
_MAX_OUTPUT_EDGE = 16384


def _apply_cas_sharpen_bhwc(images: torch.Tensor, strength: float) -> torch.Tensor:
    """Contrast-Adaptive Sharpening on RGB [B, H, W, C] without halo artifacts."""
    if strength <= 0.001:
        return images
    s = min(1.0, max(0.0, float(strength)))
    out_batches: list[torch.Tensor] = []
    chunk_sz = 8
    for i in range(0, images.shape[0], chunk_sz):
        batch = images[i : i + chunk_sz].movedim(-1, 1)
        blurred = F.avg_pool2d(F.pad(batch, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        high_freq = batch - blurred
        local_contrast = high_freq.abs().mean(dim=1, keepdim=True)
        adaptive_w = (1.0 - (local_contrast * 3.0).clamp(0.0, 0.75)) * (s * 1.15)
        sharpened = (batch + high_freq * adaptive_w).clamp(0.0, 1.0)
        out_batches.append(sharpened.movedim(1, -1))
    return torch.cat(out_batches, dim=0)


def _apply_temporal_deflicker_bhwc(images: torch.Tensor, strength: float) -> torch.Tensor:
    """Temporal de-flicker across adjacent video frames (B >= 3)."""
    if strength <= 0.001 or images.shape[0] < 3:
        return images
    alpha = min(0.5, max(0.0, float(strength)))
    res = images.clone()
    prev_f = images[:-2]
    curr_f = images[1:-1]
    next_f = images[2:]
    avg_neighbor = 0.5 * (prev_f + next_f)
    diff = avg_neighbor - curr_f
    motion_gate = 1.0 - (diff.abs().mean(dim=-1, keepdim=True) * 8.0).clamp(0.0, 1.0)
    res[1:-1] = (curr_f + diff * (alpha * motion_gate)).clamp(0.0, 1.0)
    return res


def _assert_rtx_vsr_channel_integrity(input_frame: torch.Tensor, output_frame: torch.Tensor) -> None:
    """Guard against NVIDIA RTX VSR RGB channel collapse at extreme resolutions."""
    in_sy = max(1, input_frame.shape[1] // 512)
    in_sx = max(1, input_frame.shape[2] // 512)
    out_sy = max(1, output_frame.shape[1] // 512)
    out_sx = max(1, output_frame.shape[2] // 512)
    in_s = input_frame[:, ::in_sy, ::in_sx]
    out_s = output_frame[:, ::out_sy, ::out_sx]
    if not torch.isfinite(out_s).all():
        raise RuntimeError("NVIDIA RTX VSR returned NaN or infinite pixel values")
    src_means = in_s.float().mean(dim=(1, 2))
    out_means = out_s.float().mean(dim=(1, 2))
    for ch, name in enumerate(("red", "green", "blue")):
        if bool((src_means[ch] > 0.02) and (out_means[ch] < 0.003)):
            others = torch.cat((out_means[:ch], out_means[ch + 1 :])).max()
            if bool(others > 0.02):
                raise RuntimeError(f"RTX VSR channel collapse detected on {name} channel.")


def _load_or_download_ultrasharp_model(model_name: str = "4x-UltraSharp.safetensors"):
    """Auto-download 4x-UltraSharp.safetensors if missing and load via Spandrel + ComfyUI ModelPatcher."""
    if model_name in _ULTRASHARP_CACHE:
        return _ULTRASHARP_CACHE[model_name]

    import folder_paths
    import comfy.utils
    import comfy.model_management as mm
    import comfy.model_patcher
    from spandrel import ModelLoader

    if "upscale_models" not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path("upscale_models", os.path.join(folder_paths.models_dir, "upscale_models"))

    model_path = folder_paths.get_full_path("upscale_models", model_name)
    if not model_path or not os.path.exists(model_path):
        url = "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.safetensors"
        download_file_to_comfy_models("upscale_models", model_name, url)
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        if not model_path:
            comfy_root = _get_comfy_root() or ""
            model_path = os.path.join(comfy_root, "models", "upscale_models", model_name)

    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).eval()
    out.patcher = comfy.model_patcher.CoreModelPatcher(
        out.model,
        load_device=mm.get_torch_device(),
        offload_device=mm.unet_offload_device(),
    )
    _ULTRASHARP_CACHE[model_name] = out
    return out


class RTXVideoSuperResolution:
    """
    NVIDIA RTX Video Super Resolution Pro (4K/16K) + AI 4x-UltraSharp Fallback:
    - Uses NVIDIA RTX VSR SDK (nvvfx) with 16K Hybrid guard (15360px) and RGB channel integrity check on RTX GPUs.
    - Automatically falls back to tiled 4x-UltraSharp.safetensors + Temporal De-flicker + CAS on Datacenter GPUs (H200/A100).
    """

    RELATIVE_PYTHON_MODULE = "custom_nodes.vstone"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "engine_mode": ([
                    "auto",
                    "ai_ultrasharp_4x",
                    "rtx_vsr_pro",
                    "lanczos_cas_fast",
                ], {
                    "default": "auto"
                }),
                "resize_mode": ([
                    "scale by multiplier",
                    "target dimensions",
                    "print size (mm + DPI)",
                ], {
                    "default": "scale by multiplier"
                }),
                "scale": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.05,
                }),
                "quality": (["ULTRA", "HIGH", "MEDIUM", "LOW"], {"default": "ULTRA"}),
            },
            "optional": {
                "target_width": ("INT", {"default": 3840, "min": 64, "max": 16384, "step": 8}),
                "target_height": ("INT", {"default": 2160, "min": 64, "max": 16384, "step": 8}),
                "keep_aspect_ratio": ("BOOLEAN", {"default": True}),
                "sharpen_cas": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "temporal_deflicker": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                }),
                "width_mm": ("FLOAT", {"default": 500.0, "min": 1.0, "max": 5000.0, "step": 1.0}),
                "height_mm": ("FLOAT", {"default": 500.0, "min": 1.0, "max": 5000.0, "step": 1.0}),
                "dpi": ("INT", {"default": 300, "min": 72, "max": 1200, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("upscaled_images", "calculated_scale", "output_width", "output_height")
    FUNCTION = "upscale_pro"
    CATEGORY = "image/upscaling"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def upscale_pro(
        self,
        images: torch.Tensor,
        engine_mode: str = "auto",
        resize_mode: str | dict = "scale by multiplier",
        scale: float = 2.0,
        quality: str = "ULTRA",
        target_width: int = 3840,
        target_height: int = 2160,
        keep_aspect_ratio: bool = True,
        sharpen_cas: float = 0.25,
        temporal_deflicker: float = 0.15,
        width_mm: float = 500.0,
        height_mm: float = 500.0,
        dpi: int = 300,
        **kwargs,
    ):
        import gc

        # Backward compatibility if resize_mode and quality were swapped in older workflow JSONs
        if isinstance(resize_mode, str) and resize_mode.upper() in ("ULTRA", "HIGH", "MEDIUM", "LOW"):
            old_q = resize_mode.upper()
            if isinstance(quality, str) and ("x" in quality.lower() or "uhd" in quality.lower() or "target" in quality.lower()):
                resize_mode = "target dimensions"
            else:
                resize_mode = "scale by multiplier"
            quality = old_q

        if "resize_type" in kwargs and isinstance(kwargs["resize_type"], dict):
            rt = kwargs["resize_type"]
            resize_mode = str(rt.get("resize_type", resize_mode))
            scale = float(rt.get("scale", scale))
            target_width = int(rt.get("width", target_width))
            target_height = int(rt.get("height", target_height))
            keep_aspect_ratio = bool(rt.get("keep_aspect_ratio", keep_aspect_ratio))
            width_mm = float(rt.get("width_mm", width_mm))
            height_mm = float(rt.get("height_mm", height_mm))
            dpi = int(rt.get("dpi", dpi))
        elif isinstance(resize_mode, dict):
            rt = resize_mode
            resize_mode = str(rt.get("resize_type", "scale by multiplier"))
            scale = float(rt.get("scale", scale))
            target_width = int(rt.get("width", target_width))
            target_height = int(rt.get("height", target_height))
            keep_aspect_ratio = bool(rt.get("keep_aspect_ratio", keep_aspect_ratio))

        rgb_images = images[..., :3]
        B, in_H, in_W, C = rgb_images.shape

        rm = str(resize_mode).lower()
        if "target" in rm:
            req_w, req_h = int(target_width), int(target_height)
            if keep_aspect_ratio:
                fit_s = min(req_w / max(1, in_W), req_h / max(1, in_H))
                req_w = max(8, int(math.floor(in_W * fit_s / 8.0)) * 8)
                req_h = max(8, int(math.floor(in_H * fit_s / 8.0)) * 8)
        elif "print" in rm:
            box_w = max(8, int(round(width_mm / 25.4 * dpi)))
            box_h = max(8, int(round(height_mm / 25.4 * dpi)))
            if keep_aspect_ratio:
                fit_s = min(box_w / max(1, in_W), box_h / max(1, in_H))
                req_w = max(8, int(math.floor(in_W * fit_s / 8.0)) * 8)
                req_h = max(8, int(math.floor(in_H * fit_s / 8.0)) * 8)
            else:
                req_w, req_h = box_w, box_h
        else:
            req_w = int(round(in_W * float(scale)))
            req_h = int(round(in_H * float(scale)))

        out_W = min(_MAX_OUTPUT_EDGE, max(8, int(round(req_w / 8.0)) * 8))
        out_H = min(_MAX_OUTPUT_EDGE, max(8, int(round(req_h / 8.0)) * 8))
        calc_scale = max(out_W / max(1, in_W), out_H / max(1, in_H))

        eng = (engine_mode or "auto").lower()
        used_nvvfx = False
        output_cpu: torch.Tensor | None = None

        # 1. Try NVIDIA RTX VSR Pro hardware SDK if auto or rtx_vsr_pro
        if eng.startswith("auto") or eng.startswith("rtx_vsr"):
            try:
                import nvvfx
                if max(out_W, out_H) <= _VERIFIED_VSR_MAX_EDGE:
                    vsr_W, vsr_H = out_W, out_H
                    use_hybrid = False
                else:
                    red = _VERIFIED_VSR_MAX_EDGE / float(max(out_W, out_H))
                    vsr_W = max(in_W, max(8, int(math.floor(out_W * red / 8.0)) * 8))
                    vsr_H = max(in_H, max(8, int(math.floor(out_H * red / 8.0)) * 8))
                    use_hybrid = True

                q_map = {
                    "LOW": nvvfx.effects.QualityLevel.LOW,
                    "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
                    "HIGH": nvvfx.effects.QualityLevel.HIGH,
                    "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
                }
                sel_q = q_map.get(str(quality).upper(), nvvfx.effects.QualityLevel.ULTRA)
                output_cpu = torch.empty((B, out_H, out_W, 3), device="cpu", dtype=rgb_images.dtype)

                with nvvfx.VideoSuperRes(sel_q) as sr:
                    sr.output_width = vsr_W
                    sr.output_height = vsr_H
                    sr.load()
                    for idx in range(B):
                        in_f = rgb_images[idx].movedim(-1, 0).to(device="cuda", dtype=torch.float32).contiguous()
                        res = sr.run(in_f)
                        vsr_f = torch.from_dlpack(res.image).clone()
                        del res
                        _assert_rtx_vsr_channel_integrity(in_f, vsr_f)
                        if use_hybrid:
                            vsr_f = F.interpolate(
                                vsr_f.unsqueeze(0),
                                size=(out_H, out_W),
                                mode="bicubic",
                                align_corners=False,
                                antialias=True,
                            ).squeeze(0)
                        final_f = vsr_f.clamp_(0.0, 1.0).movedim(0, -1)
                        output_cpu[idx].copy_(final_f.to(device="cpu", dtype=rgb_images.dtype))
                        del in_f, vsr_f, final_f
                used_nvvfx = True
                print(f"[RTX VSR Pro] Upscaled {B} frame(s) via NVIDIA RTX Tensor Cores -> {out_W}x{out_H}")
            except Exception as e:
                output_cpu = None
                if eng.startswith("rtx_vsr"):
                    print(f"[RTX VSR Pro] NVIDIA VSR SDK unavailable on this GPU ({e}), falling back to AI 4x-UltraSharp.")

        # 2. Run AI Upscale Model 4x-UltraSharp.safetensors (works on H200 Linux & all GPUs)
        if not used_nvvfx and not eng.startswith("lanczos"):
            try:
                import comfy.utils
                import comfy.model_management as mm

                up_desc = _load_or_download_ultrasharp_model("4x-UltraSharp.safetensors")
                dev = up_desc.patcher.load_device
                mem_req = (512 * 512 * 3) * rgb_images.element_size() * max(up_desc.scale, 1.0) * 384.0
                mm.load_models_gpu([up_desc.patcher], memory_required=mem_req, force_full_load=True)

                output_cpu = torch.empty((B, out_H, out_W, 3), device="cpu", dtype=rgb_images.dtype)
                batch_step = 4
                for start_i in range(0, B, batch_step):
                    end_i = min(B, start_i + batch_step)
                    sub_in = rgb_images[start_i:end_i].movedim(-1, -3).to(dev)
                    s_tiled = comfy.utils.tiled_scale(
                        sub_in,
                        lambda a: up_desc(a.float()),
                        tile_x=512,
                        tile_y=512,
                        overlap=32,
                        upscale_amount=up_desc.scale,
                        output_device=mm.intermediate_device(),
                    )
                    del sub_in
                    if s_tiled.shape[-2] != out_H or s_tiled.shape[-1] != out_W:
                        s_tiled = F.interpolate(
                            s_tiled.to(dev, dtype=torch.float32),
                            size=(out_H, out_W),
                            mode="bicubic",
                            align_corners=False,
                            antialias=True,
                        ).cpu()
                    s_bhwc = s_tiled.clamp(0.0, 1.0).movedim(-3, -1).to(dtype=rgb_images.dtype)
                    output_cpu[start_i:end_i].copy_(s_bhwc)
                    del s_tiled, s_bhwc
                print(f"[AI UltraSharp 16K] Upscaled {B} frame(s) via 4x-UltraSharp -> {out_W}x{out_H}")
            except Exception as e:
                print(f"[AI UltraSharp] Fallback to Antialiased Bicubic + CAS ({e})")
                output_cpu = None

        # 3. Fallback Antialiased Bicubic
        if output_cpu is None:
            output_cpu = torch.empty((B, out_H, out_W, 3), device="cpu", dtype=rgb_images.dtype)
            for idx in range(B):
                f_in = rgb_images[idx : idx + 1].movedim(-1, 1).to(dtype=torch.float32)
                f_out = F.interpolate(f_in, size=(out_H, out_W), mode="bicubic", align_corners=False, antialias=True)
                output_cpu[idx : idx + 1].copy_(f_out.clamp(0.0, 1.0).movedim(1, -1).to(dtype=rgb_images.dtype))

        # 4. Post-process Temporal De-flicker & Contrast-Adaptive Sharpening (CAS)
        if temporal_deflicker > 0.001 and B >= 3:
            output_cpu = _apply_temporal_deflicker_bhwc(output_cpu, temporal_deflicker)
        if sharpen_cas > 0.001:
            output_cpu = _apply_cas_sharpen_bhwc(output_cpu, sharpen_cas)

        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (output_cpu, float(calc_scale), int(out_W), int(out_H))


stoneV_RTX_VSR_Pro_16K = RTXVideoSuperResolution
VStone_RTX_VSR_Pro_16K = RTXVideoSuperResolution


# ==============================================================================
# Official Minimax H3 Latent Upscaler (3D) (LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
# ==============================================================================
try:
    import torch.nn as nn
except Exception:
    class _DummyNN:
        Module = object
        ModuleList = list
    nn = _DummyNN()  # type: ignore

_MMH3_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
_MMH3_VAE_DOWNSAMPLE = 16

_MMH3_LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
_MMH3_LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]
_MMH3_3D_MODEL_CACHE: dict[str, Any] = {}


def _mmh3_scan_models() -> list[str]:
    names: list[str] = []
    try:
        import folder_paths
        if _MMH3_LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(
                _MMH3_LATENT_UPSCALE_FOLDER,
                os.path.join(folder_paths.models_dir, _MMH3_LATENT_UPSCALE_FOLDER),
            )
        names = [
            name for name in folder_paths.get_filename_list(_MMH3_LATENT_UPSCALE_FOLDER)
            if os.path.splitext(name)[1].lower() in (".pth", ".safetensors")
        ]
    except Exception:
        pass
    for default_m in (
        "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
        "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
    ):
        if default_m not in names:
            names.append(default_m)
    return names


class _ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int, emb_channels: int, dropout: float = 0.0, out_channels: int | None = None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = nn.GroupNorm(32, self.out_channels)
        out_conv = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        for p in out_conv.parameters():
            p.detach().zero_()
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            out_conv,
        )
        self.skip = nn.Conv3d(channels, self.out_channels, 1) if self.out_channels != channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class _TemporalConv3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.GroupNorm(32, channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm(x))
        return x + self.pwconv(self.dwconv(h))


class _LatentResizer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        in_blocks: int = 12,
        out_blocks: int = 12,
        channels: int = 512,
        dropout: float = 0.1,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(_ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(_TemporalConv3D(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(_ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(_TemporalConv3D(channels, temporal_kernel))
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def _forward_seg(self, x: torch.Tensor, scale: float, size: tuple[int, int, int]) -> torch.Tensor:
        scale_emb = torch.tensor([scale - 1.0], dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)
        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, _ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            if isinstance(b, _ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)
        return self.conv_out(F.silu(self.norm_out(x)))

    def forward(self, x: torch.Tensor, scale: float, target_size: tuple[int, int, int], enable_chunking: bool = True) -> torch.Tensor:
        import gc
        if target_size == x.shape[-3:]:
            return x
        B, C, T, H, W = x.shape
        overlap = 5
        for b in self.in_blocks:
            if isinstance(b, _TemporalConv3D):
                overlap = int(b.dwconv.weight.shape[2])
                break
        chunk = 32
        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, target_size)

        print(f"[MinimaxH3-3D] temporal chunking: T={T} chunks={(T + chunk - 1) // chunk} overlap={overlap}")
        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        out_full = torch.zeros(B, C, T, target_size[-2], target_size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        start = 0
        while start < T:
            seg_start = start
            seg_end = min(T, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(T, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(T + 2 * overlap, out_end + overlap)

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_out = self._forward_seg(seg, scale, (hi - lo, target_size[-2], target_size[-1]))

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                bl = seg_start - out_start
                weight[:bl] = torch.arange(1, bl + 1, device=x.device, dtype=x.dtype) / (bl + 1)
            if out_end > seg_end:
                bl = out_end - seg_end
                weight[-bl:] = torch.arange(bl, 0, -1, device=x.device, dtype=x.dtype) / (bl + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)
            start += chunk
            del seg, seg_out, valid_out
            if start % (chunk * 4) == 0:
                gc.collect()

        return out_full / weight_full.clamp(min=1e-8)


def _load_or_download_mmh3_3d_upscaler(model_name: str, device: torch.device, precision: str) -> _LatentResizer3D:
    import re
    import folder_paths

    if _MMH3_LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(
            _MMH3_LATENT_UPSCALE_FOLDER,
            os.path.join(folder_paths.models_dir, _MMH3_LATENT_UPSCALE_FOLDER),
        )

    cache_key = f"{model_name}::{device.type}::{precision}"
    if cache_key in _MMH3_3D_MODEL_CACHE:
        return _MMH3_3D_MODEL_CACHE[cache_key].to(device, non_blocking=True)

    clean_name = os.path.basename(str(model_name).replace("\\", "/"))
    path = folder_paths.get_full_path(_MMH3_LATENT_UPSCALE_FOLDER, model_name) or folder_paths.get_full_path(_MMH3_LATENT_UPSCALE_FOLDER, clean_name)
    if not path or not os.path.exists(path):
        if "bf16" in clean_name:
            dl_name = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
            url = "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
        else:
            dl_name = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
            url = "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
        download_file_to_comfy_models(_MMH3_LATENT_UPSCALE_FOLDER, dl_name, url)
        path = folder_paths.get_full_path(_MMH3_LATENT_UPSCALE_FOLDER, dl_name)
        if not path:
            comfy_root = _get_comfy_root() or ""
            path = os.path.join(comfy_root, "models", _MMH3_LATENT_UPSCALE_FOLDER, dl_name)

    from safetensors.torch import load_file
    sd = load_file(path, device="cpu")
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler.") :]: v for k, v in sd.items() if k.startswith("upscaler.")}

    in_ch = sd["conv_in.weight"].shape[1] if "conv_in.weight" in sd else 24
    ch = sd["conv_in.weight"].shape[0] if "conv_in.weight" in sd else 512
    in_ids = {int(m.group(1)) for k in sd if (m := re.match(r"in_blocks\.(\d+)\.in_layers\.", k))}
    out_ids = {int(m.group(1)) for k in sd if (m := re.match(r"out_blocks\.(\d+)\.in_layers\.", k))}
    has_temp = any("dwconv.weight" in k for k in sd)
    tk = 5
    for k in sd:
        if k.endswith("dwconv.weight"):
            tk = int(sd[k].shape[2])
            break

    model = _LatentResizer3D(
        in_channels=in_ch,
        in_blocks=len(in_ids) if in_ids else 12,
        out_blocks=len(out_ids) if out_ids else 12,
        channels=ch,
        dropout=0.1,
        temporal_every=2 if has_temp else 0,
        temporal_kernel=tk,
    )
    model.load_state_dict(sd, strict=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float16)
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    _MMH3_3D_MODEL_CACHE[cache_key] = model
    print(f"[MinimaxH3-3D] Loaded upscale model: {clean_name} ({precision} on {device})")
    return model


def _execute_mmh3_latent_upscaler_3d(
    latent: dict,
    model_name: str,
    mode: Any,
    align: int = 32,
    enable_temporal_chunking: bool = True,
    force_unload: bool = True,
    device: str = "cuda",
    precision: str = "fp16",
    megapixels: float = 2.0,
) -> dict:
    import gc

    if isinstance(mode, dict):
        selected_mode = str(mode.get("mode", "megapixels"))
        scale_val = float(mode.get("scale", 2.0))
        width_val = float(mode.get("width", 1280))
        height_val = float(mode.get("height", 704))
        mp_val = float(mode.get("megapixels", megapixels))
    else:
        selected_mode = str(mode)
        scale_val = float(megapixels) if selected_mode == "scale by multiplier" else 2.0
        width_val = 1280.0
        height_val = 704.0
        mp_val = float(megapixels)

    src = latent["samples"]
    orig_dtype = src.dtype
    was_4d = (src.dim() == 4)

    dev = torch.device("cuda" if (device in ("cuda", "rocm") and torch.cuda.is_available()) else "cpu")
    compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float16)

    s = src.to(device=dev, dtype=compute_dtype, copy=True)
    if was_4d:
        s = s.unsqueeze(2)

    b, c, t, h_in, w_in = s.shape
    downsample = _MMH3_VAE_DOWNSAMPLE

    if selected_mode == "scale by multiplier":
        w_pixel_target = w_in * downsample * scale_val
        h_pixel_target = h_in * downsample * scale_val
        effective_scale = scale_val
    elif selected_mode == "target dimensions":
        w_pixel_target = width_val
        h_pixel_target = height_val
        effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
    elif selected_mode == "megapixels":
        target_pixels = mp_val * 1024.0 * 1024.0
        aspect_ratio = w_in / h_in
        h_pixel_target = (target_pixels / aspect_ratio) ** 0.5
        w_pixel_target = h_pixel_target * aspect_ratio
        effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
    else:
        raise ValueError(f"Unsupported mode: {selected_mode}")

    alignment = max(1, int(align))
    w_pixel_aligned = round(w_pixel_target / alignment) * alignment
    h_pixel_aligned = round(h_pixel_target / alignment) * alignment
    w_pixel_final = round(w_pixel_aligned / downsample) * downsample
    h_pixel_final = round(h_pixel_aligned / downsample) * downsample

    w_out = max(1, int(w_pixel_final // downsample))
    h_out = max(1, int(h_pixel_final // downsample))

    if effective_scale < 1.0 and (w_out < w_in or h_out < h_in):
        raise ValueError("This model only supports upscaling (effective scale >= 1.0).")

    if w_out == w_in and h_out == h_in:
        return latent

    print(
        f"[MinimaxH3-3D] Latent {w_in}x{h_in} -> {w_out}x{h_out} | "
        f"Pixels {w_out * downsample}x{h_out * downsample} | scale={effective_scale:.3f}"
    )

    model = _load_or_download_mmh3_3d_upscaler(model_name, dev, precision)
    norm_mean = torch.tensor(_MMH3_LATENTS_MEAN, dtype=compute_dtype, device=dev).view(1, -1, 1, 1, 1)
    norm_std = torch.tensor(_MMH3_LATENTS_STD, dtype=compute_dtype, device=dev).view(1, -1, 1, 1, 1)

    with torch.inference_mode():
        s_norm = (s - norm_mean) / norm_std
        del s
        out = model(
            s_norm,
            scale=effective_scale,
            target_size=(t, h_out, w_out),
            enable_chunking=enable_temporal_chunking,
        )
        del s_norm
        out = out * norm_std + norm_mean

    if was_4d:
        out = out.squeeze(2)

    out = out.to(device="cpu", dtype=orig_dtype, non_blocking=True)

    if dev.type == "cuda":
        if force_unload:
            model.to("cpu", non_blocking=True)
            print("[MinimaxH3-3D] Model offloaded to CPU. VRAM released.")
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except Exception:
            torch.cuda.empty_cache()
        gc.collect()

    return {"samples": out}


class MinimaxH3LatentUpscaler3D:
    """Minimax H3 latent upscaler with Temporal Chunking and pixel-space alignment."""

    RELATIVE_PYTHON_MODULE = "custom_nodes.vstone"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model_name": (_mmh3_scan_models(), {"default": "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"}),
                "mode": (["megapixels", "scale by multiplier", "target dimensions"], {"default": "megapixels"}),
                "megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "align": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "enable_temporal_chunking": ("BOOLEAN", {"default": True}),
                "force_unload": ("BOOLEAN", {"default": False}),
                "device": (["cuda", "rocm", "cpu"], {"default": "cuda"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute_v1"
    CATEGORY = "video/MinimaxH3"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def execute_v1(
        self,
        latent: dict,
        model_name: str = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
        mode: Any = "megapixels",
        megapixels: float = 2.0,
        align: int = 32,
        enable_temporal_chunking: bool = True,
        force_unload: bool = False,
        device: str = "cuda",
        precision: str = "fp16",
        **kwargs,
    ):
        out_dict = _execute_mmh3_latent_upscaler_3d(
            latent=latent,
            model_name=model_name,
            mode=mode,
            align=align,
            enable_temporal_chunking=enable_temporal_chunking,
            force_unload=force_unload,
            device=device,
            precision=precision,
            megapixels=megapixels,
        )
        return (out_dict,)



# ==============================================================================
# NODE 6: stoneV MiniMax-H3 Gen Video (1 Node Duy Nhất)
# ==============================================================================
H3_GEN_BASE_URL = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
H3_GEN_UNETS = {
    "int8": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "bf16": "minimax_h3_ref2va_pruned_bf16.safetensors",
}
H3_GEN_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_GEN_VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
H3_GEN_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
H3_GEN_TURBO_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
H3_GEN_CONTROL_PATCH = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
H3_GEN_MAX_FRAMES = 362  # Giới hạn trên của dải frame model được train
_H3_GEN_LOADED: dict[str, Any] = {}


def _h3_gen_load(key: str, folder: str, filename: str, loader: Any) -> Any:
    """Tự tải file model nếu H200 chưa có, và giữ lại kết quả để đổi seed không phải nạp lại."""
    if key not in _H3_GEN_LOADED:
        download_file_to_comfy_models(folder, filename, f"{H3_GEN_BASE_URL}/{folder}/{filename}")
        _H3_GEN_LOADED[key] = loader(filename)
    return _H3_GEN_LOADED[key]


def _h3_gen_resize(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if images.shape[1] == height and images.shape[2] == width:
        return images
    chunks = []
    for i in range(0, images.shape[0], 16):
        chunk = F.interpolate(images[i : i + 16].movedim(-1, 1), size=(height, width), mode="bicubic", align_corners=False, antialias=True)
        chunks.append(chunk.movedim(1, -1).clamp(0.0, 1.0))
    return torch.cat(chunks, dim=0)


class stoneV_H3_Gen_Video:
    """
    1 node gen video bằng MiniMax-H3: tự tải và nạp model, xuất thẳng VIDEO.
    - Chỉ có prompt: text to video.
    - Nối `image`: ảnh tham chiếu <Picture 1>, đồng thời neo vào frame đầu.
    - Nối `video` và đặt expand: giữ nguyên video gốc, model vẽ thêm vùng mở rộng theo video đó.
    - Nối `control_video`: điều khiển bằng video đã xử lý (Canny, Depth, Pose...).
    """

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "tooltip": "20 khi tắt turbo_lora, 4 khi bật."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "turbo_lora": ("BOOLEAN", {"default": False}),
                "precision": (list(H3_GEN_UNETS.keys()), {"default": "int8", "tooltip": "bf16 nặng 37GB, tự tải ở lần dùng đầu."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "0 = tự chọn theo tỉ lệ khung, tối đa 1 MP như model khuyến nghị."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                "frames": ("INT", {"default": 0, "min": 0, "max": H3_GEN_MAX_FRAMES, "tooltip": "0 = theo video nối vào, không có video thì 124 frame (5 giây)."}),
                "expand_left": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Mở rộng sang trái, tính theo chiều rộng video."}),
                "expand_right": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "expand_height": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.5, "step": 0.05, "tooltip": "Tổng chiều cao so với video."}),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "video": ("VIDEO",),
                "image": ("IMAGE",),
                "control_video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("video", "images")
    FUNCTION = "generate"
    CATEGORY = "stoneV"

    def generate(
        self,
        prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        turbo_lora: bool,
        precision: str,
        width: int,
        height: int,
        frames: int,
        expand_left: float,
        expand_right: float,
        expand_height: float,
        control_strength: float,
        video: Any = None,
        image: torch.Tensor | None = None,
        control_video: torch.Tensor | None = None,
    ):
        import nodes
        from comfy_extras.nodes_minimax_h3 import adapt_canvas

        registry = nodes.NODE_CLASS_MAPPINGS
        source_frames, source_audio, fps = None, None, 24.0
        if video is not None:
            components = video.get_components()
            source_frames, source_audio, fps = components.images[:, :, :, :3], components.audio, float(components.frame_rate)
            limit = frames if frames > 0 else H3_GEN_MAX_FRAMES
            source_frames = source_frames[:limit]

        canvas, canvas_mask = source_frames, None
        if source_frames is not None and (expand_left > 0.0 or expand_right > 0.0 or expand_height > 1.0):
            canvas, canvas_mask, _, _ = vstone_outpaint_pad_tensor(
                source_frames,
                add_left_ratio=expand_left,
                add_right_ratio=expand_right,
                outer_height_ratio=expand_height,
            )

        if width > 0 and height > 0:
            width, height = width // 32 * 32, height // 32 * 32
        elif canvas is not None:
            width, height = adapt_canvas(canvas.shape[2], canvas.shape[1])
        elif image is not None:
            width, height = adapt_canvas(image.shape[2], image.shape[1])
        else:
            width, height = 1344, 768

        frame_count = source_frames.shape[0] if source_frames is not None else (frames if frames > 0 else 124)
        length = _align_minimax_h3_frames(frame_count)

        source_video, mask, keep_alpha = None, None, None
        if canvas_mask is not None:
            source_video = _h3_gen_resize(canvas, height, width)
            scaled_mask = F.interpolate(canvas_mask[:1].unsqueeze(1), size=(height, width), mode="area")
            # Cạnh mask phải nằm trên lưới 32px của model, nếu không sẽ thành đường viền quanh video gốc.
            cells = F.adaptive_max_pool2d(scaled_mask, (height // 32, width // 32))
            mask = F.interpolate(cells, size=(height, width), mode="nearest").squeeze(1)
            soft = F.avg_pool2d(F.pad(scaled_mask, (16, 16, 16, 16), mode="replicate"), 33, stride=1)
            keep_alpha = (soft * 2.0).clamp(0.0, 1.0).movedim(1, -1)
        elif source_frames is not None and control_video is None:
            control_video = _h3_gen_resize(source_frames, height, width)

        clip = _h3_gen_load("clip", "text_encoders", H3_GEN_CLIP, lambda name: nodes.CLIPLoader().load_clip(name, "minimax", "default")[0])
        vae = _h3_gen_load("video_vae", "vae", H3_GEN_VIDEO_VAE, lambda name: nodes.VAELoader().load_vae(name)[0])
        audio_vae = _h3_gen_load("audio_vae", "vae", H3_GEN_AUDIO_VAE, lambda name: nodes.VAELoader().load_vae(name)[0])
        model = _h3_gen_load(f"unet_{precision}", "diffusion_models", H3_GEN_UNETS[precision], lambda name: nodes.UNETLoader().load_unet(name, "default")[0])
        if turbo_lora:
            model = _h3_gen_load(
                f"turbo_{precision}",
                "loras",
                H3_GEN_TURBO_LORA,
                lambda name: nodes.LoraLoaderModelOnly().load_lora_model_only(model, name, 1.0)[0],
            )
        model = registry["MiniMaxH3SigmaShift"].execute(model, shift_video=12.0, shift_audio=3.0)[0]

        # Gọi hàm gốc chưa qua VRAM Guard để giữ đủ độ phân giải, số frame và ảnh tham chiếu qua VAE.
        r2v = registry["MiniMaxH3ReferenceToVideo"]
        r2v_args = {
            "clip": clip,
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
            "ref_image_size": "match",
            "vae": vae,
            "audio_vae": audio_vae,
            "ref_images": {"ref_image_0": image} if image is not None else None,
        }
        r2v_original = getattr(r2v, "_vstone_orig_execute", None)
        conditioned = r2v_original(r2v, **r2v_args) if r2v_original is not None else r2v.execute(**r2v_args)
        positive, latent = conditioned[0], conditioned[1]
        if image is not None:
            positive = registry["MiniMaxH3AddGuide"].execute(positive=positive, latent=latent, frame_idx=0, vae=vae, image=image[:1])[0]

        if mask is not None or control_video is not None:
            patch = _h3_gen_load("control_patch", "model_patches", H3_GEN_CONTROL_PATCH, lambda name: registry["ModelPatchLoader"]().load_model_patch(name)[0])
            model = registry["MiniMaxH3FunControlNetApply"].execute(
                model=model,
                model_patch=patch,
                vae=vae,
                strength=control_strength,
                start_percent=0.0,
                end_percent=1.0,
                control_video=control_video,
                mask=mask,
                source_video=source_video,
            )[0]

        print(f"[stoneV H3] {width}x{height}, {length} frames, steps={steps}, cfg={cfg}, {sampler_name}/{scheduler}, seed={seed}")
        negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
        sampled = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0)[0]
        images = nodes.VAEDecode().decode(vae, sampled)[0][:frame_count]
        audio = source_audio if source_audio is not None else registry["VAEDecodeAudio"].execute(audio_vae, sampled)[0]

        if source_video is not None:
            original = source_video[: images.shape[0]].to(images.device, images.dtype)
            images = original * (1.0 - keep_alpha.to(images.device, images.dtype)) + images * keep_alpha.to(images.device, images.dtype)

        output = registry["CreateVideo"].execute(images=images, fps=fps, audio=audio)[0]
        return (output, images)


class stoneV_Split3_Pick:
    """Chia ảnh (hoặc từng frame video) thành 3 phần bằng nhau theo chiều ngang, lấy phần trái hoặc phải."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "side": (["left", "right"], {"default": "left"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "split"
    CATEGORY = "stoneV"

    def split(self, image: torch.Tensor, side: str):
        width = image.shape[2]
        part = width // 3
        return (image[:, :, :part] if side == "left" else image[:, :, width - part :],)


# Backward compatibility aliases
VStone_Outpaint_Pad = stoneV_Outpaint_Pad
VStone_ExtendLeft_1xWidth = stoneV_ExtendLeft_1xWidth
VStone_ExtendRight_1xWidth = stoneV_ExtendRight_1xWidth
VStone_ScreenX_Combine3 = stoneV_ScreenX_Combine3
VStone_CloudSyncHub = stoneV_CloudSyncHub
VStone_SeamlessRestore = stoneV_SeamlessRestore
VStone_Video_Outpaint_Prepare = stoneV_Video_Outpaint_Prepare
stoneV_MinimaxH3_LatentUpscaler3D = MinimaxH3LatentUpscaler3D
VStone_MinimaxH3_LatentUpscaler3D = MinimaxH3LatentUpscaler3D

VSTONE_NODE_CLASS_MAPPINGS = {
    "RTXVideoSuperResolution": RTXVideoSuperResolution,
    "stoneV_RTX_VSR_Pro_16K": RTXVideoSuperResolution,
    "VStone_RTX_VSR_Pro_16K": RTXVideoSuperResolution,
    "MinimaxH3LatentUpscaler3D": MinimaxH3LatentUpscaler3D,
    "stoneV_MinimaxH3_LatentUpscaler3D": MinimaxH3LatentUpscaler3D,
    "VStone_MinimaxH3_LatentUpscaler3D": MinimaxH3LatentUpscaler3D,
    "stoneV_H3_Gen_Video": stoneV_H3_Gen_Video,
    "stoneV_Split3_Pick": stoneV_Split3_Pick,
    "stoneV_ExtendLeft_1xWidth": stoneV_ExtendLeft_1xWidth,
    "stoneV_ExtendRight_1xWidth": stoneV_ExtendRight_1xWidth,
    "stoneV_ScreenX_Combine3": stoneV_ScreenX_Combine3,
    "stoneV_CloudSyncHub": stoneV_CloudSyncHub,
    "stoneV_Outpaint_Pad": stoneV_Outpaint_Pad,
    "stoneV_SeamlessRestore": stoneV_SeamlessRestore,
    "stoneV_Video_Outpaint_Prepare": stoneV_Video_Outpaint_Prepare,
    "VStone_ExtendLeft_1xWidth": stoneV_ExtendLeft_1xWidth,
    "VStone_ExtendRight_1xWidth": stoneV_ExtendRight_1xWidth,
    "VStone_ScreenX_Combine3": stoneV_ScreenX_Combine3,
    "VStone_CloudSyncHub": stoneV_CloudSyncHub,
    "VStone_Outpaint_Pad": stoneV_Outpaint_Pad,
    "VStone_SeamlessRestore": stoneV_SeamlessRestore,
    "VStone_Video_Outpaint_Prepare": stoneV_Video_Outpaint_Prepare,
}

VSTONE_NODE_DISPLAY_NAME_MAPPINGS = {
    "RTXVideoSuperResolution": "RTX Video Super Resolution Pro (4K/16K)",
    "stoneV_RTX_VSR_Pro_16K": "stoneV RTX VSR Pro & UltraSharp 16K",
    "VStone_RTX_VSR_Pro_16K": "stoneV RTX VSR Pro & UltraSharp 16K",
    "MinimaxH3LatentUpscaler3D": "Minimax H3 Latent Upscaler (3D)",
    "stoneV_MinimaxH3_LatentUpscaler3D": "stoneV Minimax H3 Latent Upscaler (3D)",
    "VStone_MinimaxH3_LatentUpscaler3D": "stoneV Minimax H3 Latent Upscaler (3D)",
    "stoneV_H3_Gen_Video": "stoneV: MiniMax-H3 Gen Video",
    "stoneV_Split3_Pick": "stoneV: Split Image in 3 (Pick Left / Right)",
    "stoneV_ExtendLeft_1xWidth": "stoneV 1A: Extend Left (Half-Crop + 1.3x Pad)",
    "stoneV_ExtendRight_1xWidth": "stoneV 1B: Extend Right (Half-Crop + 1.3x Pad)",
    "stoneV_ScreenX_Combine3": "stoneV 2: ScreenX Combine 3 Plates (Left + Center + Right)",
    "stoneV_CloudSyncHub": "stoneV 3: Cloud Sync & Model Downloader",
    "stoneV_Outpaint_Pad": "stoneV: Outpaint Pad (Custom Left / Right)",
    "stoneV_SeamlessRestore": "stoneV: Seamless Restore Original (Image / Video)",
    "stoneV_Video_Outpaint_Prepare": "stoneV: Video Outpaint Prepare",
    "VStone_ExtendLeft_1xWidth": "stoneV 1A: Extend Left (Half-Crop + 1.3x Pad)",
    "VStone_ExtendRight_1xWidth": "stoneV 1B: Extend Right (Half-Crop + 1.3x Pad)",
    "VStone_ScreenX_Combine3": "stoneV 2: ScreenX Combine 3 Plates (Left + Center + Right)",
    "VStone_CloudSyncHub": "stoneV 3: Cloud Sync & Model Downloader",
    "VStone_Outpaint_Pad": "stoneV: Outpaint Pad (Custom Left / Right)",
    "VStone_SeamlessRestore": "stoneV: Seamless Restore Original (Image / Video)",
    "VStone_Video_Outpaint_Prepare": "stoneV: Video Outpaint Prepare",
}

for _node_cls in VSTONE_NODE_CLASS_MAPPINGS.values():
    try:
        if not getattr(_node_cls, "RELATIVE_PYTHON_MODULE", None):
            _node_cls.RELATIVE_PYTHON_MODULE = "custom_nodes.vstone"
    except Exception:
        pass

NODE_CLASS_MAPPINGS = VSTONE_NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = VSTONE_NODE_DISPLAY_NAME_MAPPINGS
