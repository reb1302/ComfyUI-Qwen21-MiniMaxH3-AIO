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

import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
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

# Danh mục LoRA & Model chuẩn để bấm tải 1-click ngay trên máy H200 Hàn Quốc (Mirrors Ungated HTTP 200 - Không cần Token/API Key)
LTX25_OUTPAINT_SUITE_MODELS: list[dict[str, str]] = [
    {
        "folder": "loras",
        "filename": "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
        "url": "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
    },
    {
        "folder": "text_encoders",
        "filename": "gemma4_e2b_it_bf16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e2b_it_bf16.safetensors",
    },
    {
        "folder": "diffusion_models",
        "filename": "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "url": "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    },
    {
        "folder": "latent_upscale_models",
        "filename": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "url": "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    },
    {
        "folder": "vae",
        "filename": "ltx-2.5-video-vae-bf16.safetensors",
        "url": "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-video-vae-bf16.safetensors",
    },
    {
        "folder": "vae",
        "filename": "ltx-2.5-audio-vae-bf16.safetensors",
        "url": "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-audio-vae-bf16.safetensors",
    },
    {
        "folder": "text_encoders",
        "filename": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "url": "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    },
]

LTX25_URL_BY_FILENAME: dict[str, tuple[str, str]] = {
    item["filename"]: (item["folder"], item["url"]) for item in LTX25_OUTPAINT_SUITE_MODELS
}

PRESET_DOWNLOAD_CATALOG: dict[str, tuple[str, str, str]] = {
    "ALL Laptop LoRAs + Upscalers (Tự tải trọn bộ LoRA + 4x-UltraSharp + DMAD + 3D Upscaler)": ("__ALL_LAPTOP_LORAS__", "", ""),
    "ALL Laptop LoRAs (Tự tải High quality + Outpaint v2 + 360)": ("__ALL_LAPTOP_LORAS__", "", ""),
    "ALL LTX-2.5 Outpaint/Inpaint Suite (Tải trọn bộ Model LTX-2.5 Outpaint Ungated - Không cần API Key)": ("__ALL_LTX25_OUTPAINT__", "", ""),
    "LTX-2.5 IC-LoRA In-Outpainting 0.9 (ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors)": (
        "loras",
        "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
        "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
    ),
    "LTX-2.3 IC-LoRA Outpaint Black-Sentinel (ltx-2.3-22b-ic-lora-outpaint.safetensors)": (
        "loras",
        "ltx-2.3-22b-ic-lora-outpaint.safetensors",
        "https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/resolve/main/ltx-2.3-22b-ic-lora-outpaint.safetensors",
    ),
    "LTX-2.5 22B Distilled Transformer BF16 (ltx-2.5-22b-distilled-transformer-bf16.safetensors)": (
        "diffusion_models",
        "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    ),
    "LTX-2.5 Latent Spatial Upscaler x2 BF16 (ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors)": (
        "latent_upscale_models",
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    ),
    "LTX-2.5 Video VAE BF16 (ltx-2.5-video-vae-bf16.safetensors)": (
        "vae",
        "ltx-2.5-video-vae-bf16.safetensors",
        "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-video-vae-bf16.safetensors",
    ),
    "LTX-2.5 Audio VAE BF16 (ltx-2.5-audio-vae-bf16.safetensors)": (
        "vae",
        "ltx-2.5-audio-vae-bf16.safetensors",
        "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-audio-vae-bf16.safetensors",
    ),
    "LTX-2.5 Gemma4-12B Text Encoder BF16 (gemma4-12b-with-proj-ltx-2.5-bf16.safetensors)": (
        "text_encoders",
        "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    ),
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
    "MiniMax-H3 3D Latent Upscaler Asirus BF16-MAXQUALITY v5 (658MB)": (
        "latent_upscale_models",
        "minimax_h3_bf16_CONSERVATIVE_v5.safetensors",
        "https://huggingface.co/Asirus/Minimax-H3-Latent-Upscaler-BF16-MAXQUALITY/resolve/main/minimax_h3_bf16_CONSERVATIVE_v5.safetensors",
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
        "filename": "minimax_h3_bf16_CONSERVATIVE_v5.safetensors",
        "url": "https://huggingface.co/Asirus/Minimax-H3-Latent-Upscaler-BF16-MAXQUALITY/resolve/main/minimax_h3_bf16_CONSERVATIVE_v5.safetensors",
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
    "workflows/ltx_2.5/LTX25_Cai_Dat_1_Click.json",
    "workflows/ltx_2.5/LTX25_Outpaint_Official.json",
    "workflows/LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json",
    "workflows/ltx_2.5/LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json",
    "workflows/upscale_pro/image_rtx_vsr_pro_ultrasharp_16k.json",
    "workflows/upscale_pro/video_rtx_vsr_pro_ultrasharp_4k.json",
    "workflows/minimax_h3/minimaxH3SEEDHUNTERLatent_v251.json",
    "workflows/minimax_h3/huy_minhmax_screenx.json",
    "workflows/minimax_h3/image_minimax_h3_t2i.json",
    "workflows/minimax_h3/video_minimax_h3_1pass_v2v_upscale_ref.json",
    "workflows/minimax_h3/video_minimax_h3_stoneV_outpaint_ref.json",
    "workflows/minimax_h3/video_minimax_h3_fun_controlnet_union.json",
    "workflows/minimax_h3/video_minimax_h3_i2v.json",
    "workflows/minimax_h3/video_minimax_h3_i2v_continuation.json",
    "workflows/minimax_h3/video_minimax_h3_i2v_latent_upscaler.json",
    "workflows/minimax_h3/video_minimax_h3_multiframe_reference.json",
    "workflows/minimax_h3/video_minimax_h3_r2v.json",
    "workflows/minimax_h3/video_minimax_h3_r2v_image_video.json",
    "workflows/minimax_h3/video_minimax_h3_r2v_latent_upscaler.json",
    "workflows/minimax_h3/video_minimax_h3_t2v.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_background_removal.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_image_edit.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_t2i.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_turbo_image_edit.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_turbo_t2i.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_vstone_extend_right.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_stoneV_outpaint_lora_best.json",
    "workflows/qwen_image_2.1/image_qwen_image_2_1_panorama_360_outpaint.json",
]


def _ensure_ltx_video_custom_node(comfy_root: str | None = None) -> str:
    """Tự động kiểm tra và cài đặt Lightricks/ComfyUI-LTXVideo vào custom_nodes nếu chưa có."""
    root = comfy_root or _get_comfy_root()
    if not root:
        return "[LTX-2.5] Không tìm thấy thư mục ComfyUI."
    ltx_dir = os.path.join(root, "custom_nodes", "ComfyUI-LTXVideo")
    git_bin = shutil.which("git")
    if os.path.isdir(ltx_dir):
        if git_bin and os.path.isdir(os.path.join(ltx_dir, ".git")):
            try:
                subprocess.run(
                    [git_bin, "-C", ltx_dir, "pull", "--ff-only"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            except Exception:
                pass
        return f"[LTX-2.5] Custom Node ComfyUI-LTXVideo đã sẵn sàng tại: {ltx_dir}"
    if not git_bin:
        return "[LTX-2.5] Không tìm thấy git để clone ComfyUI-LTXVideo."
    try:
        os.makedirs(os.path.join(root, "custom_nodes"), exist_ok=True)
        res = subprocess.run(
            [git_bin, "clone", "https://github.com/Lightricks/ComfyUI-LTXVideo.git", ltx_dir],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        req_txt = os.path.join(ltx_dir, "requirements.txt")
        if os.path.isfile(req_txt):
            import sys as _sys_p
            subprocess.run(
                [_sys_p.executable, "-m", "pip", "install", "-r", req_txt],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        if res.returncode == 0 and os.path.isdir(ltx_dir):
            return f"[LTX-2.5] Đã tự động cài đặt ComfyUI-LTXVideo tại: {ltx_dir}"
        return f"[LTX-2.5] Cảnh báo khi clone ComfyUI-LTXVideo: {res.stderr.strip()}"
    except Exception as e:
        return f"[LTX-2.5] Lỗi cài ComfyUI-LTXVideo: {e}"


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


def _atomic_write_bytes(dst: str, data: bytes, min_bytes: int = 50) -> bool:
    """Ghi file an toàn tuyệt đối (không bao giờ tạo file 0KB hoặc đụng độ giữa nhiều luồng/symlink)."""
    if not data or len(data) < min_bytes:
        return False
    try:
        if os.path.isfile(dst) and os.path.getsize(dst) == len(data):
            with open(dst, "rb") as rf:
                if rf.read() == data:
                    return False
    except Exception:
        pass
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp_dst = f"{dst}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp_dst, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if os.path.getsize(tmp_dst) == len(data):
            os.replace(tmp_dst, dst)
            return True
    finally:
        if os.path.exists(tmp_dst):
            try:
                os.remove(tmp_dst)
            except Exception:
                pass
    return False


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
    import sys as _sys_mod
    sync_lock = getattr(_sys_mod, "_vstone_global_sync_lock", None)
    if sync_lock is None:
        sync_lock = threading.Lock()
        setattr(_sys_mod, "_vstone_global_sync_lock", sync_lock)

    if not sync_lock.acquire(blocking=force):
        return "[VStone Sync] Một luồng đồng bộ khác đang chạy, bỏ qua để tránh xung đột."

    comfy_root = _get_comfy_root()
    if not comfy_root:
        sync_lock.release()
        return "[VStone Sync] Không tìm thấy thư mục gốc ComfyUI."

    logs: list[str] = []
    try:
        commit_ref = _get_latest_commit_sha()
        raw_ref_base = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{commit_ref}"
        ts = int(time.time())

        helper_url = f"{raw_ref_base}/vstone/node_helper_vstone.py?_t={ts}"
        helper_bytes = _http_get_bytes(helper_url, timeout=20)
        if not helper_bytes or len(helper_bytes) < 1000:
            return "[VStone Sync] File tải về rỗng hoặc không hợp lệ (<1KB), bỏ qua để bảo vệ file."
        new_hash = commit_ref[:7] if commit_ref != GITHUB_BRANCH else hashlib.sha256(helper_bytes).hexdigest()[:7]

        last_global_hash = getattr(_sys_mod, "_vstone_last_synced_hash", "") or _LAST_SYNCED_HASH
        if not force and new_hash and new_hash == last_global_hash:
            return f"[VStone Sync] Đã ở bản mới nhất ({new_hash})."

        custom_nodes_dir = os.path.join(comfy_root, "custom_nodes")
        vstone_dir = os.path.join(custom_nodes_dir, "vstone")
        aio_dir = os.path.join(custom_nodes_dir, "ComfyUI_Qwen21_MiniMaxH3_AIO")
        aio_vstone_dir = os.path.join(aio_dir, "vstone")
        sibling_repo_vstone = os.path.join(os.path.dirname(comfy_root), "ComfyUI-Qwen21-MiniMaxH3-AIO", "vstone")
        workflows_dir = os.path.join(comfy_root, "user", "default", "workflows")
        vstone_wf_dir = os.path.join(vstone_dir, "workflows")

        os.makedirs(vstone_dir, exist_ok=True)
        os.makedirs(workflows_dir, exist_ok=True)
        os.makedirs(vstone_wf_dir, exist_ok=True)

        # 1. Ghi node_helper_vstone.py vào TẤT CẢ các thư mục custom_nodes (loại bỏ trùng lặp symlink)
        target_helper_paths = [
            os.path.join(vstone_dir, "node_helper_vstone.py"),
        ]
        loose_helper = os.path.join(custom_nodes_dir, "node_helper_vstone.py")
        if os.path.exists(loose_helper):
            target_helper_paths.append(loose_helper)
        extras_vstone = os.path.join(comfy_root, "comfy_extras", "nodes_vstone.py")
        if os.path.exists(extras_vstone):
            target_helper_paths.append(extras_vstone)
        if os.path.isdir(aio_dir):
            os.makedirs(aio_vstone_dir, exist_ok=True)
            target_helper_paths.append(os.path.join(aio_vstone_dir, "node_helper_vstone.py"))
        if os.path.isdir(sibling_repo_vstone):
            target_helper_paths.append(os.path.join(sibling_repo_vstone, "node_helper_vstone.py"))

        seen_realpaths: set[str] = set()
        for dst in target_helper_paths:
            try:
                rp = os.path.realpath(dst)
                if rp in seen_realpaths:
                    continue
                seen_realpaths.add(rp)
                if _atomic_write_bytes(dst, helper_bytes, min_bytes=1000):
                    logs.append(f"Updated: {dst}")
            except Exception as e:
                logs.append(f"Skip {dst}: {e}")

        # 2. Ghi __init__.py an toàn bằng _atomic_write_bytes
        try:
            init_bytes = _http_get_bytes(f"{raw_ref_base}/vstone/__init__.py?_t={ts}", timeout=15)
            if init_bytes and len(init_bytes) > 50:
                init_targets = [os.path.join(vstone_dir, "__init__.py")]
                if os.path.isdir(aio_vstone_dir):
                    init_targets.append(os.path.join(aio_vstone_dir, "__init__.py"))
                if os.path.isdir(sibling_repo_vstone):
                    init_targets.append(os.path.join(sibling_repo_vstone, "__init__.py"))
                seen_init_rp: set[str] = set()
                for init_dst in init_targets:
                    irp = os.path.realpath(init_dst)
                    if irp not in seen_init_rp:
                        seen_init_rp.add(irp)
                        _atomic_write_bytes(init_dst, init_bytes, min_bytes=50)
        except Exception:
            pass

        # 3. Đồng bộ danh sách Workflows & LoRAs theo commit mới nhất
        wf_list = list(DEFAULT_SYNC_WORKFLOWS)
        lora_list = list(DEFAULT_AUTO_SYNC_LORAS)
        try:
            manifest_bytes = _http_get_bytes(f"{raw_ref_base}/sync_manifest.json?_t={ts}", timeout=10)
            manifest_data = json.loads(manifest_bytes.decode("utf-8"))
            if isinstance(manifest_data.get("workflows"), list):
                for _w in manifest_data["workflows"]:
                    if _w not in wf_list:
                        wf_list.append(_w)
            if isinstance(manifest_data.get("loras"), list):
                lora_list = manifest_data["loras"]
        except Exception:
            pass

        wf_count = 0
        for path in wf_list:
            if not (path.startswith("workflows/") and path.endswith(".json")):
                continue
            rel_wf = path[len("workflows/"):]
            try:
                wf_bytes = _http_get_bytes(f"{raw_ref_base}/{path}?_t={ts}", timeout=15)
                if not wf_bytes or len(wf_bytes) < 50:
                    continue
                target_wf_dirs = [workflows_dir, vstone_wf_dir]
                if os.path.isdir(aio_dir):
                    target_wf_dirs.append(os.path.join(aio_dir, "workflows"))
                seen_wf_rp: set[str] = set()
                for base_wf_d in target_wf_dirs:
                    dst_wf = os.path.join(base_wf_d, rel_wf)
                    wrp = os.path.realpath(dst_wf)
                    if wrp not in seen_wf_rp:
                        seen_wf_rp.add(wrp)
                        _atomic_write_bytes(dst_wf, wf_bytes, min_bytes=50)
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

        # 4. Tự động tải ngầm các LoRA / Model về ComfyUI/models/ (không khóa luồng Sync)
        def _bg_download_models(items_to_dl):
            for l_item in items_to_dl:
                folder = l_item.get("folder", "loras")
                fname = l_item.get("filename", "")
                url = l_item.get("url", "")
                if folder and fname and url:
                    try:
                        download_file_to_comfy_models(folder, fname, url)
                    except Exception as e:
                        print(f"[VStone Downloader] Skip {fname}: {e}")

        if lora_list:
            threading.Thread(target=_bg_download_models, args=(list(lora_list),), name="_bg_download_models", daemon=True).start()
            logs.append(f"Started background check/download for {len(lora_list)} models/LoRAs.")

        _LAST_SYNCED_HASH = new_hash
        setattr(_sys_mod, "_vstone_last_synced_hash", new_hash)
        msg = f"[VStone Cloud Sync] Đồng bộ thành công commit {new_hash} ({wf_count} workflows)! Nhấn phím R hoặc F5 trên trình duyệt để làm mới."
        print(msg)
        return msg + "\n" + "\n".join(logs)
    except Exception as e:
        err_msg = f"[VStone Cloud Sync] Lỗi kết nối: {e}"
        return err_msg
    finally:
        try:
            sync_lock.release()
        except Exception:
            pass


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


VSTONE_RUNTIME_PATCH_VER = 5
MINIMAX_SAFE_MAX_FRAMES = 73  # 17 * 4 + 5 (~3.04s @ 24fps)
MINIMAX_SAFE_DI_W = 1280
MINIMAX_SAFE_DI_H = 704
# Lưu trên sys để mọi lần hot-reload dùng chung 1 lock, không tải 2 lần vào cùng file .part
_DOWNLOAD_LOCK = sys.__dict__.setdefault("_vstone_download_lock", threading.Lock())


def _vstone_free_vram(unload_models: bool = False) -> None:
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


def _unwrap_to_true_orig(cls, attr_name: str):
    """Unwrap any _vstone_orig_* or _safe_* closure wrappers back to the original ComfyUI function."""
    store_key = f"_vstone_orig_{attr_name}"
    fn = cls.__dict__.get(store_key) or getattr(cls, store_key, None) or cls.__dict__.get(attr_name) or getattr(cls, attr_name, None)
    seen: set[int] = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        if isinstance(fn, classmethod):
            fn = fn.__func__
            continue
        if hasattr(fn, "__func__"):
            fn = fn.__func__
            continue
        fn_name = getattr(fn, "__name__", "") or ""
        if fn_name.startswith("_safe_") and getattr(fn, "__closure__", None):
            inner = None
            for cell in fn.__closure__:
                try:
                    c = cell.cell_contents
                    if callable(c) or isinstance(c, classmethod):
                        inner = c
                        break
                except Exception:
                    pass
            if inner is not None:
                fn = inner
                continue
        break
    return fn


def _chunked_cpu_resize(images: torch.Tensor, width: int, height: int, method: str = "lanczos", crop: str = "center", chunk_size: int = 16) -> torch.Tensor:
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


def _recover_comfy_queue_and_worker() -> list[str]:
    """
    Ensure ComfyUI's prompt_worker thread and PromptExecutor are healthy:
    1. Guard PromptExecutor.execute so exceptions inside cache.set_prompt never kill prompt_worker.
    2. Clear stale NODE_CLASS_CONTAINS_UNIQUE_ID cache entries.
    3. Rebind PromptServer.instance.client_id to an active WebSocket if the user refreshed the browser (F5).
    4. If prompt_worker thread died earlier, clear orphaned currently_running items and respawn an immortal prompt_worker.
    """
    notes: list[str] = []
    try:
        import sys
        import traceback
        import nodes
        import execution
        import server

        try:
            import comfy_execution.caching as _ce_caching
            if hasattr(_ce_caching, "NODE_CLASS_CONTAINS_UNIQUE_ID"):
                _ce_caching.NODE_CLASS_CONTAINS_UNIQUE_ID.clear()
        except Exception:
            pass

        # 1. Guard PromptExecutor.execute against unhandled exceptions
        if not getattr(execution.PromptExecutor, "_vstone_exec_guarded", False):
            orig_pe_exec = execution.PromptExecutor.execute

            def _safe_pe_execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
                try:
                    return orig_pe_exec(self, prompt, prompt_id, extra_data, execute_outputs)
                except BaseException as ex:
                    print(f"[VStone Queue Guard] Caught exception in PromptExecutor.execute ({prompt_id}): {ex}")
                    traceback.print_exc()
                    self.success = False
                    self.history_result = {"outputs": {}, "meta": {}}
                    try:
                        self.add_message(
                            "execution_error",
                            {
                                "prompt_id": prompt_id,
                                "node_id": "0",
                                "node_type": "PromptExecutor",
                                "executed": [],
                                "exception_message": str(ex),
                                "exception_type": type(ex).__name__,
                                "traceback": traceback.format_tb(ex.__traceback__),
                                "current_inputs": {},
                                "current_outputs": [],
                            },
                            broadcast=True,
                        )
                    except Exception:
                        pass

            execution.PromptExecutor.execute = _safe_pe_execute
            setattr(execution.PromptExecutor, "_vstone_exec_guarded", True)
            notes.append("Guarded PromptExecutor.execute")

        ps = getattr(server.PromptServer, "instance", None)
        if ps is None:
            return notes

        # 2. If browser tab refreshed (F5), point ps.client_id to the active socket so progress bars work
        try:
            sockets = getattr(ps, "sockets", None)
            if isinstance(sockets, dict) and len(sockets) > 0:
                if getattr(ps, "client_id", None) not in sockets:
                    ps.client_id = list(sockets.keys())[-1]
        except Exception:
            pass

        # 3. Check if prompt_worker thread is alive
        worker_thread = None
        for t in threading.enumerate():
            if not t.is_alive():
                continue
            t_name = getattr(t, "name", "") or ""
            t_target = getattr(t, "_target", None)
            t_target_name = getattr(t_target, "__name__", "") if t_target is not None else ""
            if (
                getattr(t, "_vstone_is_prompt_worker", False)
                or "prompt_worker" in t_name
                or t_target_name in ("prompt_worker", "_vstone_immortal_prompt_worker")
            ):
                worker_thread = t
                break

        q = getattr(ps, "prompt_queue", None)
        am = getattr(ps, "asset_manager", None)

        if worker_thread is None and q is not None:
            # prompt_worker died! Clean up orphaned currently_running items and respawn worker
            with q.mutex:
                if q.currently_running:
                    dead_ids = list(q.currently_running.keys())
                    for did in dead_ids:
                        q.currently_running.pop(did, None)
                    ps.queue_updated()
                    notes.append(f"Cleared orphaned running queue items: {dead_ids}")
            try:
                if getattr(ps, "client_id", None) is not None:
                    ps.send_sync("executing", {"node": None, "prompt_id": None}, ps.client_id)
            except Exception:
                pass

            main_mod = sys.modules.get("__main__")
            orig_pw = getattr(main_mod, "prompt_worker", None) if main_mod is not None else None

            def _vstone_immortal_prompt_worker():
                print("[VStone Queue Guard] Respawned immortal prompt_worker thread!")
                while True:
                    try:
                        if callable(orig_pw):
                            orig_pw(q, ps, am)
                        else:
                            e = execution.PromptExecutor(ps, asset_manager=am)
                            while True:
                                queue_item = q.get(timeout=1000.0)
                                if queue_item is not None:
                                    item, item_id = queue_item
                                    prompt_id = item[1]
                                    ps.last_prompt_id = prompt_id
                                    extra_data = item[3].copy()
                                    for k in item[5]:
                                        extra_data[k] = item[5][k]
                                    e.execute(item[2], prompt_id, extra_data, item[4])
                                    remove_sensitive = lambda p: p[:5] + p[6:]
                                    q.task_done(
                                        item_id,
                                        e.history_result,
                                        status=execution.PromptQueue.ExecutionStatus(
                                            status_str="success" if e.success else "error",
                                            completed=e.success,
                                            messages=e.status_messages,
                                        ),
                                        process_item=remove_sensitive,
                                    )
                                    if getattr(ps, "client_id", None) is not None:
                                        ps.send_sync("executing", {"node": None, "prompt_id": prompt_id}, ps.client_id)
                    except BaseException as ex:
                        print(f"[VStone Queue Guard] prompt_worker exception recovered: {ex}")
                        traceback.print_exc()
                        try:
                            with q.mutex:
                                for did in list(q.currently_running.keys()):
                                    q.currently_running.pop(did, None)
                                ps.queue_updated()
                        except Exception:
                            pass
                        time.sleep(1.0)

            new_wt = threading.Thread(target=_vstone_immortal_prompt_worker, name="prompt_worker", daemon=True)
            setattr(new_wt, "_vstone_is_prompt_worker", True)
            new_wt.start()
            notes.append("Respawned dead prompt_worker thread")
        elif worker_thread is not None and q is not None:
            # One-time recovery on v5 upgrade if an old job was stuck in currently_running
            if getattr(sys, "_vstone_queue_recovery_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                setattr(sys, "_vstone_queue_recovery_ver", VSTONE_RUNTIME_PATCH_VER)
                with q.mutex:
                    running_count = len(q.currently_running)
                    pending_count = len(q.queue)
                if running_count > 0:
                    try:
                        frames = sys._current_frames()
                        wf = frames.get(worker_thread.ident)
                        if wf is not None:
                            stack_summary = " -> ".join(
                                f"{os.path.basename(f.filename)}:{f.lineno}({f.name})"
                                for f in traceback.extract_stack(wf)[-5:]
                            )
                            print(f"[VStone Queue Guard] Active prompt_worker stack (running={running_count}, pending={pending_count}): {stack_summary}")
                        nodes.interrupt_processing(True)
                        notes.append(f"Interrupted stuck running job (running={running_count}, pending={pending_count})")
                    except Exception:
                        pass
    except Exception as e:
        notes.append(f"Queue check error: {e}")
    return notes


def apply_comfy_runtime_fixes() -> str:
    """
    Restore all built-in ComfyUI nodes to their original unpatched functions and
    ensure the ComfyUI prompt queue & prompt_worker thread are healthy and unblocked.
    """
    restored: list[str] = []

    # 1. Restore V1 instance methods if previously monkey-patched
    v1_targets = [
        ("ImageScale", None, "upscale"),
        ("VAEEncode", None, "encode"),
        ("VAEDecode", None, "decode"),
        ("KSampler", None, "sample"),
        ("LoraLoader", None, "load_lora"),
    ]
    for cls_name, mod_name, attr_name in v1_targets:
        try:
            for c in _get_target_node_classes(cls_name, mod_name):
                if getattr(c, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                    if getattr(c, "_vstone_patched", False) or f"_vstone_orig_{attr_name}" in c.__dict__:
                        orig_fn = _unwrap_to_true_orig(c, attr_name)
                        if callable(orig_fn):
                            setattr(c, attr_name, orig_fn)
                        if "VALIDATE_INPUTS" in c.__dict__:
                            try:
                                delattr(c, "VALIDATE_INPUTS")
                            except Exception:
                                pass
                        setattr(c, "_vstone_patched", False)
                        restored.append(cls_name)
                    setattr(c, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
        except Exception:
            pass

    # Restore LoraLoaderModelOnly (inherits load_lora from LoraLoader)
    try:
        for c in _get_target_node_classes("LoraLoaderModelOnly", None):
            if getattr(c, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                if "load_lora" in c.__dict__:
                    try:
                        delattr(c, "load_lora")
                    except Exception:
                        pass
                if "VALIDATE_INPUTS" in c.__dict__:
                    try:
                        delattr(c, "VALIDATE_INPUTS")
                    except Exception:
                        pass
                setattr(c, "_vstone_patched", False)
                setattr(c, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
    except Exception:
        pass

    # 2. Restore V3 ComfyNode classmethods if previously monkey-patched
    v3_targets = [
        ("GetVideoComponents", "comfy_extras.nodes_video", "execute"),
        ("VAEEncodeAudio", "comfy_extras.nodes_audio", "execute"),
        ("LTXVConcatAVLatent", "comfy_extras.nodes_lt", "execute"),
        ("MiniMaxH3ReferenceToVideo", "comfy_extras.nodes_minimax_h3", "execute"),
    ]
    for cls_name, mod_name, attr_name in v3_targets:
        try:
            for c in _get_target_node_classes(cls_name, mod_name):
                if getattr(c, "_vstone_patch_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                    if getattr(c, "_vstone_patched", False) or f"_vstone_orig_{attr_name}" in c.__dict__:
                        orig_fn = _unwrap_to_true_orig(c, attr_name)
                        if callable(orig_fn):
                            cm = orig_fn if isinstance(orig_fn, classmethod) else classmethod(orig_fn)
                            setattr(c, attr_name, cm)
                            if cls_name == "VAEEncodeAudio":
                                setattr(c, "encode", cm)
                        setattr(c, "_vstone_patched", False)
                        restored.append(cls_name)
                    setattr(c, "_vstone_patch_ver", VSTONE_RUNTIME_PATCH_VER)
        except Exception:
            pass

    # 3. Hook folder_paths so native UNETLoader/CLIPLoader/VAELoader/LoraLoaderModelOnly recognize & auto-download LTX-2.5 models
    try:
        import folder_paths
        if not getattr(folder_paths, "_vstone_ltx25_hooked", False):
            orig_get_filename_list = folder_paths.get_filename_list
            orig_get_full_path = folder_paths.get_full_path

            _ltx_virtual_files = {
                "diffusion_models": ["ltx-2.5-22b-distilled-transformer-bf16.safetensors"],
                "unet": ["ltx-2.5-22b-distilled-transformer-bf16.safetensors"],
                "loras": ["ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"],
                "text_encoders": ["gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"],
                "clip": ["gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"],
                "vae": [
                    "ltx-2.5-video-vae-bf16.safetensors",
                    "ltx-2.5-audio-vae-bf16.safetensors",
                ],
                "latent_upscale_models": ["ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"],
            }

            def _vstone_get_filename_list(folder_name: str):
                res = list(orig_get_filename_list(folder_name))
                for vf in _ltx_virtual_files.get(folder_name, []):
                    if vf not in res:
                        res.append(vf)
                return res

            def _vstone_get_full_path(folder_name: str, filename: str):
                p = orig_get_full_path(folder_name, filename)
                if p is not None and os.path.exists(p):
                    return p
                clean_name = os.path.basename(filename) if filename else ""
                if clean_name in LTX25_URL_BY_FILENAME:
                    target_folder, url = LTX25_URL_BY_FILENAME[clean_name]
                    print(f"[VStone Auto-Downloader] Missing {clean_name}, downloading from ungated mirror...")
                    download_file_to_comfy_models(target_folder, clean_name, url)
                    p = orig_get_full_path(folder_name, filename)
                    if p is not None and os.path.exists(p):
                        return p
                    comfy_root = _get_comfy_root()
                    if comfy_root:
                        fallback_p = os.path.join(comfy_root, "models", target_folder, clean_name)
                        if os.path.exists(fallback_p):
                            return fallback_p
                return p

            folder_paths.get_filename_list = _vstone_get_filename_list
            folder_paths.get_full_path = _vstone_get_full_path
            setattr(folder_paths, "_vstone_ltx25_hooked", True)
    except Exception:
        pass

    # 4. Check and recover ComfyUI prompt_worker & queue
    q_notes = _recover_comfy_queue_and_worker()
    all_notes = []
    if restored:
        all_notes.append(f"Restored native nodes: {', '.join(restored)}")
    if q_notes:
        all_notes.extend(q_notes)

    if all_notes:
        msg = f"[VStone Runtime v{VSTONE_RUNTIME_PATCH_VER}] " + " | ".join(all_notes)
        print(msg)
        return msg
    return f"[VStone Runtime v{VSTONE_RUNTIME_PATCH_VER}] Native ComfyUI nodes & queue healthy."


def _clear_comfy_model_cache() -> None:
    try:
        import folder_paths
        if hasattr(folder_paths, "filename_list_cache"):
            folder_paths.filename_list_cache.clear()
    except Exception:
        pass


def download_file_to_comfy_models(folder_type: str, filename: str, url: str) -> str:
    """Download model/LoRA safely using a lock and temporary .part file with timeouts."""
    comfy_root = _get_comfy_root()
    if not comfy_root:
        return "[VStone Downloader] ComfyUI root not found."

    target_dir = os.path.join(comfy_root, "models", folder_type)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024 * 1024:
        return f"[VStone Downloader] Exists: {target_path} ({os.path.getsize(target_path) // (1024*1024)} MB)"

    with _DOWNLOAD_LOCK:
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1024 * 1024:
            return f"[VStone Downloader] Exists: {target_path} ({os.path.getsize(target_path) // (1024*1024)} MB)"

        tmp_name = f"{filename}.part"
        tmp_path = os.path.join(target_dir, tmp_name)
        print(f"[VStone Downloader] Downloading {filename} to {target_dir} from {url} ...")

        aria2_bin = shutil.which("aria2c")
        if aria2_bin:
            cmd = [
                aria2_bin, "--check-certificate=false", "--disable-ipv6=true",
                "--file-allocation=none", "-c", "-x", "16", "-s", "16", "-k", "1M",
                "--allow-overwrite=true", "-d", target_dir, "-o", tmp_name, url
            ]
            res = subprocess.run(cmd, check=False, timeout=3600)
            if res.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
                os.replace(tmp_path, target_path)
                _clear_comfy_model_cache()
                return f"[VStone Downloader] Downloaded via aria2c: {target_path}"

        curl_bin = shutil.which("curl")
        if curl_bin:
            cmd = [curl_bin, "-4", "-L", "-C", "-", "--max-time", "3600", url, "-o", tmp_path]
            res = subprocess.run(cmd, check=False, timeout=3610)
            if res.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
                os.replace(tmp_path, target_path)
                _clear_comfy_model_cache()
                return f"[VStone Downloader] Downloaded via curl: {target_path}"

        req = urllib.request.Request(url, headers={"User-Agent": "VStone-ComfyUI-Sync/2.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
            os.replace(tmp_path, target_path)
        _clear_comfy_model_cache()
        return f"[VStone Downloader] Downloaded: {target_path}"


# ------------------------------------------------------------------------------
# Cài LTX-2.5 bằng 1 lần Queue trên stoneV_CloudSyncHub (H200 không gõ lệnh được)
# ------------------------------------------------------------------------------
LTX25_CUSTOM_NODE_PACKS: list[dict[str, Any]] = [
    {
        "dir": "ComfyUI-LTXVideo",
        "repo": "Lightricks/ComfyUI-LTXVideo",
        "branch": "master",
        "node_ids": [
            "GemmaAPITextEncode",
            "LTXAddVideoICLoRAGuideAdvanced",
            "LTXICLoRALoaderModelOnly",
            "LTXVImgToVideoConditionOnly",
            "LTXVInpaintPreprocess",
            "LTXVLaplacianPyramidBlend",
            "LTXVSetAudioRefTokens",
        ],
    },
    {
        "dir": "ComfyUI-KJNodes",
        "repo": "kijai/ComfyUI-KJNodes",
        "branch": "main",
        "node_ids": ["ImagePadForOutpaintTargetSize"],
    },
]
LTX25_RESTARTED_ENV = "VSTONE_LTX25_RESTARTED"


def _missing_node_ids(node_ids: list[str]) -> list[str]:
    import nodes
    return [node_id for node_id in node_ids if node_id not in nodes.NODE_CLASS_MAPPINGS]


def _run_install_cmd(cmd: list[str], timeout: int) -> tuple[bool, str]:
    """Chạy lệnh cài đặt, trả về (thành công, phần cuối của log) thay vì ném lỗi."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    return res.returncode == 0, (res.stderr or res.stdout or "").strip()[-300:]


def _fetch_node_pack(pack: dict[str, Any], pack_dir: str) -> tuple[bool, str]:
    """Clone hoặc pull pack. Máy không có git, hoặc thư mục không phải repo git, thì tải zip từ GitHub."""
    git_bin = shutil.which("git")
    if git_bin and os.path.isdir(os.path.join(pack_dir, ".git")):
        return _run_install_cmd([git_bin, "-C", pack_dir, "pull", "--ff-only"], 300)
    if git_bin and not os.path.exists(pack_dir):
        clone_url = f"https://github.com/{pack['repo']}.git"
        return _run_install_cmd([git_bin, "clone", "--depth", "1", clone_url, pack_dir], 900)
    zip_url = f"https://github.com/{pack['repo']}/archive/refs/heads/{pack['branch']}.zip"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(_http_get_bytes(zip_url, timeout=600))) as archive:
                archive.extractall(tmp)
            shutil.copytree(os.path.join(tmp, os.listdir(tmp)[0]), pack_dir, dirs_exist_ok=True)
        return True, ""
    except Exception as e:
        return False, str(e)


def _pip_install_requirements(pack_dir: str) -> list[str]:
    """Cài requirements.txt. Một gói lỗi làm hỏng cả file, nên khi lỗi thì cài lại từng gói và trả về các gói hỏng."""
    req_path = os.path.join(pack_dir, "requirements.txt")
    if not os.path.isfile(req_path):
        return []
    pip_install = [sys.executable, "-m", "pip", "install"]
    if _run_install_cmd(pip_install + ["-r", req_path], 1800)[0]:
        return []
    failed: list[str] = []
    with open(req_path, encoding="utf-8") as req_file:
        for line in req_file:
            spec = line.split("#")[0].strip()
            if spec and not spec.startswith("-"):
                ok, tail = _run_install_cmd(pip_install + [spec], 900)
                if not ok:
                    failed.append(f"{spec}: {tail}")
    return failed


def _restart_comfyui_after(delay_sec: float) -> None:
    """Khởi động lại ComfyUI với đúng tham số đang chạy (giữ nguyên port), cùng cách ComfyUI-Manager dùng."""
    def _restart() -> None:
        time.sleep(delay_sec)
        print("[VStone] Khởi động lại ComfyUI để nạp pack LTX-2.5...", flush=True)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_restart, name="VStoneRestart", daemon=True).start()


def _wait_for_background_downloads(unique_id: Any) -> None:
    """Luồng tải nền của Cloud Sync có thể đang chạy code cũ với lock riêng, tải song song sẽ hỏng file .part nên phải chờ."""
    announced = False
    while any("_bg_download_models" in thread.name for thread in threading.enumerate()):
        if not announced:
            _vstone_ui_status(unique_id, "Bước 2/2: Cloud Sync đang tải model nền, chờ xong rồi kiểm tra tiếp...")
            announced = True
        time.sleep(15)


def install_ltx25_one_click(unique_id: Any = None) -> str:
    """Cài pack và model LTX-2.5 còn thiếu, khởi động lại nếu cần. Dòng đầu của báo cáo là kết luận."""
    root = _get_comfy_root()
    if not root:
        return "❌ Không tìm thấy thư mục ComfyUI."

    details: list[str] = []
    custom_nodes = os.path.join(root, "custom_nodes")
    os.makedirs(custom_nodes, exist_ok=True)
    installed_dirs = {name.lower(): name for name in os.listdir(custom_nodes)}

    fetch_failed = False
    for pack in LTX25_CUSTOM_NODE_PACKS:
        if not _missing_node_ids(pack["node_ids"]):
            details.append(f"Pack {pack['dir']}: đã nạp đủ node.")
            continue
        _vstone_ui_status(unique_id, f"Bước 1/2: cài pack {pack['dir']}...")
        pack_dir = os.path.join(custom_nodes, installed_dirs.get(pack["dir"].lower(), pack["dir"]))
        ok, tail = _fetch_node_pack(pack, pack_dir)
        details.append(f"Pack {pack['dir']}: " + ("đã tải về." if ok else f"LỖI khi tải: {tail}"))
        fetch_failed = fetch_failed or not os.path.isdir(pack_dir)
        for failure in _pip_install_requirements(pack_dir):
            details.append(f"Pack {pack['dir']}: LỖI pip {failure}")

    _wait_for_background_downloads(unique_id)

    models_missing: list[str] = []
    total = len(LTX25_OUTPAINT_SUITE_MODELS)
    for index, item in enumerate(LTX25_OUTPAINT_SUITE_MODELS, start=1):
        filename = item["filename"]
        _vstone_ui_status(
            unique_id,
            f"Bước 2/2: model {index}/{total}: {filename}\nFile lớn có thể mất hàng chục phút, node vẫn đang chạy.",
        )
        try:
            download_file_to_comfy_models(item["folder"], filename, item["url"])
        except Exception as e:
            details.append(f"Model {filename}: LỖI {e}")
        path = os.path.join(root, "models", item["folder"], filename)
        if not os.path.isfile(path) or os.path.getsize(path) <= 1024 * 1024:
            models_missing.append(filename)
            if os.path.isfile(path):
                os.remove(path)  # trang lỗi HTML bị lưu thành .safetensors, không phải model

    if models_missing:
        details.append("Model chưa có: " + ", ".join(models_missing) + ". Bấm Queue lần nữa để tải tiếp.")

    missing_ids = _missing_node_ids([node_id for pack in LTX25_CUSTOM_NODE_PACKS for node_id in pack["node_ids"]])
    if missing_ids and not fetch_failed and os.environ.get(LTX25_RESTARTED_ENV) != "1":
        os.environ[LTX25_RESTARTED_ENV] = "1"
        _restart_comfyui_after(5.0)
        summary = "🔄 ComfyUI đang khởi động lại để nạp pack. Chờ 1-2 phút, F5, rồi bấm Queue lần nữa để xem kết quả."
    elif missing_ids:
        details.append("Thiếu node: " + ", ".join(missing_ids) + ". Xem log khởi động, dòng IMPORT FAILED.")
        summary = "❌ Chưa cài xong pack LTX-2.5."
    elif models_missing:
        summary = "❌ Pack đã đủ, còn thiếu model."
    else:
        summary = "✅ LTX-2.5 đã đủ pack và model. Mở workflow ltx_2.5/LTX25_Outpaint_Official."
    return "\n".join([summary] + details)


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
                    "0. Cài LTX-2.5 đầy đủ (Pack + Model, tự khởi động lại)",
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
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
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
        unique_id: Any = None,
    ):
        if action.startswith("0."):
            report = install_ltx25_one_click(unique_id)
            _vstone_ui_status(unique_id, report)
            return {"ui": {"text": [report]}, "result": (report,)}

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
                elif folder == "__ALL_LTX25_OUTPAINT__":
                    reports.append(_ensure_ltx_video_custom_node())
                    for item in LTX25_OUTPAINT_SUITE_MODELS:
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
        _vstone_ui_status(unique_id, f"✅ Đã nạp VStone v3.0 (2D Master + 3 Wings)\n{final_report.splitlines()[0]}")
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
# NODE 5: stoneV ScreenX Cut (Cắt ảnh ScreenX thành 1 bên để gen video từng cánh)
# ==============================================================================
SCREENX_CUT_MODES = {
    "cánh + 50% ảnh giữa": 0.5,
    "cánh + 100% ảnh giữa": 1.0,
    "chỉ cánh": 0.0,
    "cánh + % tự chọn": None,
}


def _h3_canvas_for_image(image_w: int, image_h: int, target_megapixels: float) -> tuple[int, int]:
    """Khung gen MiniMax-H3 theo tỉ lệ ảnh: bội số 32, không phóng to ảnh nhỏ (0.98 MP với 16:9 ra đúng 1344x768)."""
    target_pixels = float(target_megapixels) * 1024.0 * 1024.0
    scale = min(1.0, math.sqrt(target_pixels / float(max(1, image_w * image_h))))
    out_w = max(32, int(round(image_w * scale / 32.0)) * 32)
    out_h = max(32, int(round(image_h * scale / 32.0)) * 32)
    return out_w, out_h


class stoneV_ScreenX_Cut:
    """
    Nhận ảnh ScreenX đầy đủ [Cánh Trái | Giữa | Cánh Phải], cắt theo chiều dọc lấy đúng 1 bên, giữ nguyên chiều cao:
    - "cánh + 50% ảnh giữa": cánh + nửa ảnh giữa nằm sát cánh (mặc định, wing_width_ratio = 1.0 thì đúng bằng 1/2 ảnh vào).
    - "cánh + 100% ảnh giữa": cánh + cả ảnh giữa.
    - "chỉ cánh": riêng cánh.
    - "cánh + % tự chọn": cánh + `custom_center_percent` % ảnh giữa nằm sát cánh.
    `width`, `height` là khung gen cho node MiniMax H3 Reference to Video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "side": (["left", "right"], {
                    "default": "left",
                    "tooltip": "left = lấy bên Cánh Trái, right = lấy bên Cánh Phải.",
                }),
                "mode": (list(SCREENX_CUT_MODES), {
                    "default": "cánh + 50% ảnh giữa",
                }),
                "custom_center_percent": ("FLOAT", {
                    "default": 12.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Chỉ dùng ở chế độ \"cánh + % tự chọn\": số % chiều rộng ảnh giữa lấy kèm cánh.",
                }),
                "wing_width_ratio": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Tỉ lệ chiều rộng mỗi cánh so với ảnh giữa, giống node ScreenX Combine 3 Plates.",
                }),
                "target_megapixels": ("FLOAT", {
                    "default": 0.98,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.02,
                    "tooltip": "Diện tích khung gen. 0.98 là mức chuẩn của model (1344x768 với 16:9). Chỉ ảnh hưởng width, height, không resize ảnh ra.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "cut"
    CATEGORY = "stoneV"

    def cut(
        self,
        image: torch.Tensor,
        side: str = "left",
        mode: str = "cánh + 50% ảnh giữa",
        custom_center_percent: float = 12.0,
        wing_width_ratio: float = 1.0,
        target_megapixels: float = 0.98,
    ):
        rgb = image[:, :, :, :3]
        _, height, total_w, _ = rgb.shape
        center_w = float(total_w) / (1.0 + 2.0 * float(wing_width_ratio))
        wing_w = center_w * float(wing_width_ratio)

        center_part = SCREENX_CUT_MODES[mode]
        if center_part is None:
            center_part = float(custom_center_percent) / 100.0

        cut_w = max(1, min(total_w, int(round(wing_w + center_w * center_part))))
        cut = rgb[:, :, :cut_w, :] if side == "left" else rgb[:, :, total_w - cut_w :, :]
        out_w, out_h = _h3_canvas_for_image(cut_w, height, target_megapixels)
        return (cut, out_w, out_h)


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
        "minimax_h3_bf16_CONSERVATIVE_v5.safetensors",
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
        in_norm, in_act, in_conv = self.in_layers[0], self.in_layers[1], self.in_layers[2]
        h = in_act(in_norm(x.to(in_norm.weight.dtype)))
        h = in_conv(h.to(in_conv.weight.dtype))
        emb_dtype = self.emb_layers[1].weight.dtype
        emb_out = self.emb_layers(emb.to(emb_dtype))
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        out_norm_dtype = self.out_norm.weight.dtype
        h = self.out_norm(h.to(out_norm_dtype)) * (1 + scale.to(out_norm_dtype)) + shift.to(out_norm_dtype)
        out_act, out_drop, out_conv = self.out_layers[0], self.out_layers[1], self.out_layers[2]
        h = out_drop(out_act(h))
        h = out_conv(h.to(out_conv.weight.dtype))
        skip_x = self.skip(x.to(self.skip.weight.dtype)) if isinstance(self.skip, nn.Conv3d) else x
        return skip_x.to(h.dtype) + h


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
        h = F.silu(self.norm(x.to(self.norm.weight.dtype)))
        h = self.dwconv(h.to(self.dwconv.weight.dtype))
        h = self.pwconv(h.to(self.pwconv.weight.dtype))
        return x.to(h.dtype) + h


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
        in_dtype = x.dtype
        embed_dtype = self.embed[0].weight.dtype
        scale_emb = torch.tensor([scale - 1.0], dtype=embed_dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)
        x = self.conv_in(x.to(self.conv_in.weight.dtype))
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
        h = F.silu(self.norm_out(x.to(self.norm_out.weight.dtype)))
        out = self.conv_out(h.to(self.conv_out.weight.dtype))
        return out.to(in_dtype)

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
        if "conservative" in clean_name.lower() or "v5" in clean_name.lower():
            dl_name = "minimax_h3_bf16_CONSERVATIVE_v5.safetensors"
            url = "https://huggingface.co/Asirus/Minimax-H3-Latent-Upscaler-BF16-MAXQUALITY/resolve/main/minimax_h3_bf16_CONSERVATIVE_v5.safetensors"
        elif "bf16" in clean_name.lower():
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
    sd_dtypes = {v.dtype for v in sd.values() if isinstance(v, torch.Tensor)}
    is_mixed_bf16_fp16 = (len(sd_dtypes) > 1 and precision == "bf16")
    model.load_state_dict(sd, strict=True)
    if is_mixed_bf16_fp16:
        for name, param in model.named_parameters():
            if name in sd:
                param.data = sd[name].to(device=device)
        for name, buf in model.named_buffers():
            if name in sd:
                buf.data = sd[name].to(device=device)
        model = model.to(device=device).eval().requires_grad_(False)
        print(f"[MinimaxH3-3D] Loaded mixed-precision upscale model: {clean_name} (BF16+FP16 fallback on {device})")
    else:
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float16)
        model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        print(f"[MinimaxH3-3D] Loaded upscale model: {clean_name} ({precision} on {device})")
    _MMH3_3D_MODEL_CACHE[cache_key] = model
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

    raw_samples = latent["samples"]
    audio_stream = None
    if getattr(raw_samples, "is_nested", False):
        streams = raw_samples.unbind()
        src = streams[0]
        if len(streams) > 1:
            audio_stream = streams[1]
    else:
        src = raw_samples

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
    elif selected_mode == "megapixels" or "megapixels" in selected_mode:
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

    w_out = max(w_in, int(w_pixel_final // downsample))
    h_out = max(h_in, int(h_pixel_final // downsample))
    effective_scale = max(1.0, float(effective_scale))

    def _pack_av_output(video_tensor: torch.Tensor) -> dict:
        if was_4d:
            return {"samples": video_tensor}
        frame_per_tok = (1, 4, 4, 4, 4)
        frame_count = sum(frame_per_tok[k % 5] for k in range(t))
        target_audio_t = max(1, int(round((frame_count / 24.0) * 40.0)))
        if audio_stream is not None and audio_stream.ndim == 4:
            aud = audio_stream.to(device="cpu", dtype=orig_dtype)
            if aud.shape[-1] > target_audio_t:
                aud = aud[..., :target_audio_t].contiguous()
            elif aud.shape[-1] < target_audio_t:
                pad_len = target_audio_t - aud.shape[-1]
                aud = torch.cat([aud, torch.zeros((aud.shape[0], aud.shape[1], aud.shape[2], pad_len), dtype=orig_dtype, device="cpu")], dim=-1)
        else:
            aud = torch.zeros((b, 32, 2, target_audio_t), dtype=orig_dtype, device="cpu")
        try:
            import comfy.nested_tensor
            return {"samples": comfy.nested_tensor.NestedTensor((video_tensor, aud))}
        except Exception:
            return {"samples": video_tensor}

    if w_out == w_in and h_out == h_in:
        out_same = s.squeeze(2) if was_4d else s
        out_same = out_same.to(device="cpu", dtype=orig_dtype, non_blocking=True)
        return _pack_av_output(out_same)

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

    return _pack_av_output(out)


class MinimaxH3LatentUpscaler3D:
    """Minimax H3 latent upscaler with Temporal Chunking and pixel-space alignment."""

    RELATIVE_PYTHON_MODULE = "custom_nodes.vstone"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model_name": (_mmh3_scan_models(), {"default": "minimax_h3_bf16_CONSERVATIVE_v5.safetensors"}),
                "mode": (["megapixels", "scale by multiplier", "target dimensions"], {"default": "megapixels"}),
                "megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "align": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "enable_temporal_chunking": ("BOOLEAN", {"default": True}),
                "force_unload": ("BOOLEAN", {"default": False}),
                "device": (["cuda", "rocm", "cpu"], {"default": "cuda"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "bf16"}),
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
        model_name: str = "minimax_h3_bf16_CONSERVATIVE_v5.safetensors",
        mode: Any = "megapixels",
        megapixels: float = 2.0,
        align: int = 32,
        enable_temporal_chunking: bool = True,
        force_unload: bool = False,
        device: str = "cuda",
        precision: str = "bf16",
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
# NODE 6: stoneV Panorama 360 Multi-Ref Outpaint Loop (Qwen-Image-2.1)
# ==============================================================================
def _vstone_ui_status(unique_id: Any, text: str) -> None:
    """Hiển thị trực tiếp dòng trạng thái + phiên bản ngay bên dưới Node trên màn hình trình duyệt ComfyUI."""
    print(text)
    if not unique_id:
        return
    try:
        import server
        if hasattr(server, "PromptServer") and getattr(server.PromptServer, "instance", None) is not None:
            server.PromptServer.instance.send_progress_text(text, str(unique_id))
    except Exception:
        pass


def _cosine_alpha_1d(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 1:
        return torch.ones((1,), device=device, dtype=dtype)
    t = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=torch.float32)
    return (0.5 * (1.0 - torch.cos(math.pi * t))).to(dtype=dtype)


def _feather_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Làm mềm biên (feather) cho mask [B, H, W] (0.0 = vùng ảnh gốc, 1.0 = vùng AI vẽ mới).
    Chỉ làm mềm lấn vào bên trong vùng ảnh gốc (giữ nguyên 1.0 ở vùng bù xám/xanh bên ngoài
    để không bao giờ bị lem màu nền xám vào đường giáp mí).
    """
    r = max(1, int(radius))
    m4 = mask.unsqueeze(1).to(dtype=torch.float32)
    k = 2 * r + 1
    blurred = F.avg_pool2d(F.pad(m4, (r, r, r, r), mode="replicate"), kernel_size=k, stride=1)
    blurred = F.avg_pool2d(F.pad(blurred, (r, r, r, r), mode="replicate"), kernel_size=k, stride=1).squeeze(1)
    return torch.maximum(mask, blurred).clamp(0.0, 1.0)


def _match_color_stats(src: torch.Tensor, ref: torch.Tensor, strength: float = 0.75) -> torch.Tensor:
    """Đồng bộ ánh sáng/nhiệt độ màu (RGB Mean & Std) của `src` theo `ref` để khi ghép nối không bị lệch tông màu."""
    s_mean = src.mean(dim=(1, 2), keepdim=True)
    s_std = src.std(dim=(1, 2), keepdim=True).clamp(min=1e-4)
    r_mean = ref.mean(dim=(1, 2), keepdim=True)
    r_std = ref.std(dim=(1, 2), keepdim=True).clamp(min=1e-4)
    matched = (src - s_mean) * (r_std / s_std).clamp(0.7, 1.4) + r_mean
    return (src * (1.0 - strength) + matched * strength).clamp(0.0, 1.0)


def _find_best_horizontal_offset(
    left_strip: torch.Tensor,
    right_img: torch.Tensor,
    base_start_r: int,
    strip_w: int,
    max_shift: int = 24,
) -> int:
    """
    Tìm độ lệch ngang `delta_x` (trong khoảng `[-max_shift, +max_shift]`) giúp vùng chồng lấp của `right_img`
    khớp vị trí pixel chính xác nhất (L1 error nhỏ nhất) với `left_strip` của ảnh gốc.
    """
    w_r = int(right_img.shape[2])
    if strip_w < 16 or max_shift <= 0:
        return 0
    # Lấy dải giữa theo chiều dọc để so khớp nhanh và ổn định
    h = int(left_strip.shape[1])
    y0, y1 = int(h * 0.15), max(int(h * 0.15) + 16, int(h * 0.85))
    target = left_strip[:, y0:y1, :, :]

    best_dx = 0
    best_err = float("inf")
    for dx in range(-max_shift, max_shift + 1, 2):
        r0 = base_start_r + dx
        r1 = r0 + strip_w
        if r0 < 0 or r1 > w_r:
            continue
        cand = right_img[:, y0:y1, r0:r1, :]
        # Ưu tiên dx = 0 nếu sai số tương đương (regularization nhẹ)
        err = float(torch.mean(torch.abs(target - cand)).item()) + 0.00015 * abs(dx)
        if err < best_err:
            best_err = err
            best_dx = dx

    # Tinh chỉnh từng pixel quanh best_dx
    for dx in (best_dx - 1, best_dx + 1):
        r0 = base_start_r + dx
        r1 = r0 + strip_w
        if 0 <= r0 and r1 <= w_r:
            cand = right_img[:, y0:y1, r0:r1, :]
            err = float(torch.mean(torch.abs(target - cand)).item()) + 0.00015 * abs(dx)
            if err < best_err:
                best_err = err
                best_dx = dx
    return best_dx


def _blend_2band_dp_seam(left_seam: torch.Tensor, right_seam: torch.Tensor) -> torch.Tensor:
    """
    Hòa trộn 2 dải chồng lấp `left_seam` và `right_seam` [1, H, BW, 3] bằng kỹ thuật 2-Band (Tần số thấp + Tần số cao):
    - Tần số thấp (ánh sáng, màu nền): trải mượt bằng Cosine toàn dải `BW`.
    - Tần số cao (chi tiết sắc nét, cạnh kiến trúc): cắt theo đường Minimum-Error DP Seam 2D để không bị bóng ma/nhòe nét.
    """
    _, h, bw, _ = left_seam.shape
    if bw < 8:
        alpha = _cosine_alpha_1d(bw, left_seam.device, left_seam.dtype).view(1, 1, bw, 1)
        return (left_seam * (1.0 - alpha) + right_seam * alpha).clamp(0.0, 1.0)

    alpha_low = _cosine_alpha_1d(bw, left_seam.device, left_seam.dtype).view(1, 1, bw, 1)

    # Tách tần số thấp bằng AvgPool2d (kernel ~15x15)
    k_sz = min(15, max(3, (min(h, bw) // 4) | 1))
    pad = k_sz // 2
    l_nchw = left_seam.permute(0, 3, 1, 2)
    r_nchw = right_seam.permute(0, 3, 1, 2)
    l_low = F.avg_pool2d(F.pad(l_nchw, (pad, pad, pad, pad), mode="reflect"), kernel_size=k_sz, stride=1).permute(0, 2, 3, 1)
    r_low = F.avg_pool2d(F.pad(r_nchw, (pad, pad, pad, pad), mode="reflect"), kernel_size=k_sz, stride=1).permute(0, 2, 3, 1)
    l_high = left_seam - l_low
    r_high = right_seam - r_low

    # Tìm đường cắt tối ưu (Minimum-Error Seam) trên bản đồ sai số thu nhỏ chiều dọc (tối đa 128 hàng) để chạy cực nhanh trên CPU
    diff_map = torch.mean(torch.abs(left_seam[0] - right_seam[0]), dim=-1)  # [H, BW]
    ds_h = min(h, 128)
    diff_small = F.interpolate(diff_map.unsqueeze(0).unsqueeze(0), size=(ds_h, bw), mode="bilinear", align_corners=False)[0, 0]
    # Khuyến khích đường cắt nằm trong vùng trung tâm của dải BW
    x_grid = torch.linspace(-1.0, 1.0, steps=bw, device=diff_small.device, dtype=diff_small.dtype)
    border_penalty = (x_grid * x_grid) * 0.05
    cost = diff_small + border_penalty.unsqueeze(0)

    dp = cost.clone()
    back = torch.zeros((ds_h, bw), dtype=torch.long, device=dp.device)
    for y in range(1, ds_h):
        prev = dp[y - 1]
        p_left = torch.cat([prev[:1] + 1e4, prev[:-1]], dim=0)
        p_right = torch.cat([prev[1:], prev[-1:] + 1e4], dim=0)
        stacked = torch.stack([p_left, prev, p_right], dim=0)  # [3, BW]
        min_vals, min_idx = torch.min(stacked, dim=0)
        dp[y] = cost[y] + min_vals
        back[y] = min_idx - 1  # -1, 0, +1

    seam_x_small = torch.zeros((ds_h,), dtype=torch.float32, device=dp.device)
    cur_x = int(torch.argmin(dp[-1]).item())
    seam_x_small[-1] = float(cur_x)
    for y in range(ds_h - 1, 0, -1):
        cur_x = max(0, min(bw - 1, cur_x + int(back[y, cur_x].item())))
        seam_x_small[y - 1] = float(cur_x)

    seam_x = F.interpolate(seam_x_small.view(1, 1, ds_h), size=h, mode="linear", align_corners=False).view(1, h, 1, 1)
    col_idx = torch.arange(bw, device=left_seam.device, dtype=left_seam.dtype).view(1, 1, bw, 1)
    # Làm mềm cạnh đường cắt tần số cao trong khoảng ~6px để không gắt mà vẫn giữ nguyên chi tiết
    feather_hi = max(3.0, min(8.0, float(bw) * 0.06))
    alpha_high = torch.sigmoid((col_idx - seam_x) / (feather_hi * 0.35))

    blended_low = l_low * (1.0 - alpha_low) + r_low * alpha_low
    blended_high = l_high * (1.0 - alpha_high) + r_high * alpha_high
    return (blended_low + blended_high).clamp(0.0, 1.0)


def _stitch_horizontal_cosine(
    left_img: torch.Tensor,
    right_img: torch.Tensor,
    overlap_w: int,
    blend_w: int,
    anchor_side: str = "left",
) -> torch.Tensor:
    """
    Ghép ngang 2 ảnh `left_img` [1, H, W_L, 3] và `right_img` [1, H, W_R, 3]
    trong đó `overlap_w` cột đầu của `right_img` trùng không gian với `overlap_w` cột cuối của `left_img`:
    - QUAN TRỌNG: Đặt vị trí nối ở PHÍA NGOÀI CỦA VÙNG MỒI NEO (nơi đã được khóa cứng `denoise_mask = 0.0`
      và cách xa đường cắt xám `#808080`), giữ nguyên 100% vùng chuyển tiếp liền mạch mà model đã vẽ qua đường cắt xám!
    - Tự động căn chỉnh vị trí khớp pixel (`_find_best_horizontal_offset`) + cân bằng màu + ghép 2-Band DP Seam.
    """
    h = int(left_img.shape[1])
    if int(right_img.shape[1]) != h:
        right_img = _chunked_cpu_resize(right_img, int(right_img.shape[2]), h, "lanczos", "disabled", chunk_size=1)
    w_l = int(left_img.shape[2])
    w_r = int(right_img.shape[2])
    overlap_w = max(0, min(int(overlap_w), w_l, w_r))
    if overlap_w == 0:
        return torch.cat([left_img, right_img], dim=2)

    # Đặt dải hòa trộn ở NỬA NGOÀI CỦA MỒI NEO (cách xa đường biên xám nơi model vừa vẽ chuyển tiếp):
    # - Khi anchor_side == "left" (mở rộng sang Phải): right_img là ảnh mới Gen, biên xám cũ nằm tại k = overlap_w.
    #   Vùng mồi neo gốc chưa bị biến đổi nằm ở đầu trái của right_img (k gần 0).
    # - Khi anchor_side == "right" (mở rộng sang Trái): left_img là ảnh mới Gen, biên xám cũ nằm tại k = 0 (w_l - overlap_w).
    #   Vùng mồi neo gốc chưa bị biến đổi nằm ở đầu phải của left_img (k gần overlap_w).
    guard = min(24, max(4, overlap_w // 16))
    locked_zone_w = max(16, int(overlap_w * 0.42))
    bw = max(min(int(blend_w) * 2, locked_zone_w - guard), int(locked_zone_w * 0.75))
    bw = max(8, min(bw, overlap_w - guard * 2))

    if anchor_side == "left":
        # Mở rộng sang Phải: nối tại vùng đầu mồi neo k in [guard, guard + bw] (cách xa biên xám k = overlap_w),
        # giữ nguyên 100% phần chuyển tiếp của right_img (gen_win) đi qua k = overlap_w!
        k_start = guard
        k_end = k_start + bw
        seam_start_l = (w_l - overlap_w) + k_start
        seam_end_l = (w_l - overlap_w) + k_end

        left_ov_locked = left_img[:, :, seam_start_l:seam_end_l, :]
        right_ov_locked = right_img[:, :, k_start:k_end, :]
        right_img = _match_color_stats(right_img, left_ov_locked, strength=0.65)

        dx = _find_best_horizontal_offset(left_ov_locked, right_img, base_start_r=k_start, strip_w=bw, max_shift=min(24, guard))
        r_start = max(0, min(w_r - bw, k_start + dx))
        r_end = r_start + bw

        left_pure = left_img[:, :, :seam_start_l, :]
        left_seam = left_img[:, :, seam_start_l:seam_end_l, :]
        right_seam = right_img[:, :, r_start:r_end, :]
        right_pure = right_img[:, :, r_end:, :]
    else:
        # Mở rộng sang Trái: nối tại vùng cuối mồi neo k in [overlap_w - guard - bw, overlap_w - guard] (cách xa biên xám k = 0),
        # giữ nguyên 100% phần chuyển tiếp của left_img (gen_win) đi qua biên xám w_l - overlap_w!
        k_end = overlap_w - guard
        k_start = k_end - bw
        seam_start_l = (w_l - overlap_w) + k_start
        seam_end_l = (w_l - overlap_w) + k_end

        right_ov_locked = right_img[:, :, k_start:k_end, :]
        left_img = _match_color_stats(left_img, right_ov_locked, strength=0.65)

        dx = _find_best_horizontal_offset(
            left_img[:, :, seam_start_l:seam_end_l, :], right_img, base_start_r=k_start, strip_w=bw, max_shift=min(24, guard)
        )
        r_start = max(0, min(w_r - bw, k_start + dx))
        r_end = r_start + bw

        left_pure = left_img[:, :, :seam_start_l, :]
        left_seam = left_img[:, :, seam_start_l:seam_end_l, :]
        right_seam = right_img[:, :, r_start:r_end, :]
        right_pure = right_img[:, :, r_end:, :]

    blended_seam = _blend_2band_dp_seam(left_seam, right_seam)
    return torch.cat([left_pure, blended_seam, right_pure], dim=2).clamp(0.0, 1.0)


def _prepare_ref_for_qwen21(img: torch.Tensor, target_res: int = 768) -> torch.Tensor:
    """Chuẩn hóa ảnh Reference về diện tích ~768x768 (bội số của 32) để nhẹ VRAM khi nạp 3-5 tấm Ref."""
    rgb = img[:1, :, :, :3].to("cpu", dtype=torch.float32)
    h, w = int(rgb.shape[1]), int(rgb.shape[2])
    ratio = float(w) / max(1.0, float(h))
    tw = max(64, int(round(math.sqrt(target_res * target_res * ratio) / 32.0)) * 32)
    th = max(64, int(round(math.sqrt(target_res * target_res / ratio) / 32.0)) * 32)
    return _chunked_cpu_resize(rgb, tw, th, "lanczos", "disabled", chunk_size=1)


def _synthesize_master_wide_from_2d_refs(
    model,
    clip,
    vae,
    all_refs: list[torch.Tensor],
    target_w: int,
    target_h: int,
    user_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    max_window_mp: float = 1.4,
    keep_model_patches: bool = False,
) -> torch.Tensor:
    """
    Giai đoạn 1: Đưa toàn bộ 1-5 tấm ảnh Reference qua VAE 2D (`reference_latents` 2D đầy đủ)
    để Qwen-Image-2.1 hiểu đúng 100% môi trường 2D của các tấm Ref và tổng hợp thành
    1 Tấm Phong Cảnh Panorama Dài Thống Nhất (`target_w x target_h`, tỉ lệ 2.5:1 -> 4.0:1).
    """
    import uuid
    import nodes
    import comfy.model_management as mm

    target_pixels = max(0.5, float(max_window_mp)) * 1_000_000.0
    cur_pixels = float(target_w * target_h)
    if cur_pixels > target_pixels * 1.05:
        scale = math.sqrt(target_pixels / cur_pixels)
        gen_w = max(64, int(round((target_w * scale) / 32.0)) * 32)
        gen_h = max(64, int(round((target_h * scale) / 32.0)) * 32)
    else:
        gen_w = max(64, int(round(target_w / 32.0)) * 32)
        gen_h = max(64, int(round(target_h / 32.0)) * 32)

    images_dict: dict[str, torch.Tensor] = {}
    for idx, r_img in enumerate(all_refs[:5], start=1):
        if r_img is not None:
            images_dict[f"image_{idx}"] = _prepare_ref_for_qwen21(r_img, target_res=768)

    ref_tags = ", ".join(f"<image{i}>" for i in range(1, len(images_dict) + 1))
    other_tags = ", ".join(f"<image{i}>" for i in range(2, len(images_dict) + 1)) or "<image1>"
    synth_prompt = (
        f"Task: Unified Ultra-Wide Panoramic Landscape Synthesis from Reference Context ({ref_tags}). "
        f"Analyze the spatial context, architectural style, terrain, vegetation, background structures, "
        f"surface materials, sky, and lighting atmosphere across {ref_tags}. "
        f"Synthesize and harmonize all environmental elements from {ref_tags} into one single unbroken, "
        f"ultra-wide panoramic landscape scene: anchor the primary environmental theme on <image1> and "
        f"seamlessly integrate the surrounding contextual elements from {other_tags} across the wide horizontal frame, "
        f"maintaining a single continuous horizon line, coherent wide-angle perspective, unified lighting, "
        f"and natural spatial transitions without split borders, frames, or collage seams. "
        + (user_prompt or "").strip()
    ).strip()

    qwen_enc_classes = _get_target_node_classes("TextEncodeQwenImage21", "comfy_extras.nodes_qwen")
    if not qwen_enc_classes:
        raise RuntimeError("Không tìm thấy node TextEncodeQwenImage21 trong ComfyUI.")

    enc_out = qwen_enc_classes[0].execute(
        clip=clip,
        prompt=synth_prompt,
        negative_prompt=negative_prompt or "",
        vae=vae,
        resolution=0,
        images=images_dict,
    )
    enc_args = list(enc_out.args) if hasattr(enc_out, "args") else list(enc_out)
    positive, negative = enc_args[0], enc_args[1]

    # Tạo khung Latent dài chuẩn tỉ lệ Panorama (gen_w x gen_h)
    wide_latent = torch.zeros([1, 64, gen_h // 16, gen_w // 16], device=mm.intermediate_device())
    latent_dict = {"samples": wide_latent}

    # Nếu chưa qua _prepare_panorama_hq_model (keep_model_patches=False), tự gỡ Outpaint LoRA cũ
    synth_model = model
    if not keep_model_patches:
        try:
            if hasattr(model, "clone") and getattr(model, "patches", None):
                synth_model = model.clone()
                synth_model.patches = {}
                synth_model.patches_uuid = uuid.uuid4()
        except Exception:
            synth_model = model

    _vstone_free_vram(unload_models=False)
    sampled = nodes.common_ksampler(
        model=synth_model,
        seed=int(seed),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent=latent_dict,
        denoise=1.0,
    )[0]

    decoded = vae.decode(sampled["samples"])
    if decoded.ndim == 5:
        decoded = decoded[0]
    decoded = decoded[:1, :, :, :3].to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
    if int(decoded.shape[2]) != target_w or int(decoded.shape[1]) != target_h:
        decoded = _chunked_cpu_resize(decoded, target_w, target_h, "lanczos", "disabled", chunk_size=1)
    _vstone_free_vram(unload_models=False)
    return decoded


def _boost_outpaint_lora_strength(model, target_strength: float = 1.0):
    """
    Đảm bảo ở bước Outpaint vùng xám `#808080`, LoRA `qwen-image-2.1-outpaint-v2` luôn chạy đủ `strength = 1.0`
    để khóa cứng phần ảnh mồi neo `<image1>` (kể cả khi trên giao diện người dùng đang để Strength = 0.5).
    """
    import uuid
    try:
        if hasattr(model, "clone") and getattr(model, "patches", None):
            needs_boost = False
            for plist in model.patches.values():
                for p in plist:
                    if isinstance(p, tuple) and len(p) >= 2 and 0.05 < float(p[0]) < (target_strength - 0.05):
                        needs_boost = True
                        break
                if needs_boost:
                    break
            if needs_boost:
                boosted = model.clone()
                new_patches = {}
                for k, plist in boosted.patches.items():
                    new_list = []
                    for p in plist:
                        if isinstance(p, tuple) and len(p) >= 2 and 0.05 < float(p[0]) < (target_strength - 0.05):
                            new_list.append((float(target_strength),) + p[1:])
                        else:
                            new_list.append(p)
                    new_patches[k] = new_list
                boosted.patches = new_patches
                boosted.patches_uuid = uuid.uuid4()
                return boosted
    except Exception:
        pass
    return model


def _build_spatially_aligned_ref_canvas(
    work_canvas: torch.Tensor,
    target_ref: torch.Tensor,
) -> torch.Tensor:
    """
    Tạo `<image2>` có CÙNG KÍCH THƯỚC 2D (`win_h x win_w`) và CÙNG HỆ TỌA ĐỘ 2D RoPE `(hh, ww)` với `work_canvas` (`<image1>`):
    - Tại vùng mồi neo (nơi `work_canvas` là ảnh thật): `<image2>` giữ nguyên 100% pixel mồi neo của `work_canvas`
      để KHÔNG xảy ra xung đột 2D RoPE làm biến dạng phần mồi neo khi ghép nối.
    - Tại vùng xám `#808080` (nơi cần vẽ nối tiếp): `<image2>` đặt tấm ảnh `target_ref` (đã đồng bộ ánh sáng/màu sắc
      theo vùng mồi neo) đúng vào vị trí vùng xám để Qwen-Image-2.1 tái tạo bối cảnh của `target_ref` nối mượt vào mồi neo.
    """
    _, win_h, win_w, _ = work_canvas.shape
    # Phát hiện các cột vùng xám/xanh đồng nhất (std theo chiều dọc ~ 0)
    col_std = work_canvas[0].std(dim=(0, 2))  # [win_w]
    is_blank_col = col_std < 1e-3

    if not bool(is_blank_col.any()) or bool(is_blank_col.all()):
        return _chunked_cpu_resize(target_ref[:1, :, :, :3], win_w, win_h, "lanczos", "disabled", chunk_size=1)

    anchor_cols = work_canvas[:, :, ~is_blank_col, :]
    ref_rgb = target_ref[:1, :, :, :3].to("cpu", dtype=torch.float32)
    ref_matched = _match_color_stats(ref_rgb, anchor_cols, strength=0.70)

    blank_indices = torch.nonzero(is_blank_col, as_tuple=False).squeeze(-1)
    b_start = int(blank_indices[0].item())
    b_end = int(blank_indices[-1].item()) + 1
    b_width = max(32, b_end - b_start)

    ref_resized = _chunked_cpu_resize(ref_matched, b_width, win_h, "lanczos", "disabled", chunk_size=1)
    spatial_ref = work_canvas.clone()
    spatial_ref[:, :, b_start:b_end, :] = ref_resized[:, :, : (b_end - b_start), :]

    # Làm mềm nhẹ 32px tại ranh giới giữa mồi neo và ref trong `<image2>`
    feather_w = min(32, b_width // 4)
    if feather_w >= 4:
        if b_start > 0 and b_start + feather_w <= win_w:
            alpha = _cosine_alpha_1d(feather_w, spatial_ref.device, spatial_ref.dtype).view(1, 1, feather_w, 1)
            edge_anchor = work_canvas[:, :, max(0, b_start - feather_w) : b_start, :]
            if int(edge_anchor.shape[2]) == feather_w:
                spatial_ref[:, :, b_start : b_start + feather_w, :] = (
                    edge_anchor * (1.0 - alpha) + spatial_ref[:, :, b_start : b_start + feather_w, :] * alpha
                )
        if b_end < win_w and b_end - feather_w >= 0:
            alpha = _cosine_alpha_1d(feather_w, spatial_ref.device, spatial_ref.dtype).view(1, 1, feather_w, 1)
            edge_anchor = work_canvas[:, :, b_end : min(win_w, b_end + feather_w), :]
            if int(edge_anchor.shape[2]) == feather_w:
                spatial_ref[:, :, b_end - feather_w : b_end, :] = (
                    spatial_ref[:, :, b_end - feather_w : b_end, :] * (1.0 - alpha) + edge_anchor * alpha
                )
    return spatial_ref.clamp(0.0, 1.0)


def _run_qwen21_outpaint_pass(
    model,
    clip,
    vae,
    canvas_img: torch.Tensor,
    ref_images_list: list[torch.Tensor],
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    max_window_mp: float = 1.4,
) -> torch.Tensor:
    """
    Chạy 1 bước Qwen-Image-2.1 Outpaint trên khung `canvas_img`:
    - Tự động căn biên vùng xám `#808080` khít bội số 32px ở độ phân giải lấy mẫu để không bị nhiễu biên VAE/DiT.
    - Tự động tạo `<image2>` khớp 100% tọa độ 2D RoPE `(hh, ww)` với `<image1>` (giữ nguyên mồi neo ở vùng neo, đặt ảnh Ref ở vùng xám).
    - Tự động đẩy Outpaint LoRA lên `strength = 1.0` để phần mồi neo khớp tuyệt đối khi ghép nối.
    """
    import nodes

    orig_h, orig_w = int(canvas_img.shape[1]), int(canvas_img.shape[2])
    target_pixels = max(0.4, float(max_window_mp)) * 1_000_000.0
    cur_pixels = float(orig_w * orig_h)
    if cur_pixels > target_pixels * 1.05:
        scale = math.sqrt(target_pixels / cur_pixels)
        win_w = max(64, int(round((orig_w * scale) / 64.0)) * 64)
        win_h = max(64, int(round((orig_h * scale) / 64.0)) * 64)
    else:
        win_w = max(64, int(round(orig_w / 64.0)) * 64)
        win_h = max(64, int(round(orig_h / 64.0)) * 64)

    # Dùng bilinear (không dùng lanczos trên biên xám để tránh quầng ringing đen/trắng tại mép ghép)
    work_canvas = _chunked_cpu_resize(canvas_img[:1, :, :, :3], win_w, win_h, "bilinear", "disabled", chunk_size=1)

    # Giữ nguyên chính xác tỉ lệ vị trí ranh giới vùng xám trên work_canvas (không làm tròn lệch cột)
    orig_col_std = canvas_img[0].std(dim=(0, 2))
    orig_blank = orig_col_std < 1e-3
    has_blank = bool(orig_blank.any()) and not bool(orig_blank.all())
    b_s_win, b_e_win = 0, win_w
    if has_blank:
        b_idx = torch.nonzero(orig_blank, as_tuple=False).squeeze(-1)
        b_s_orig = int(b_idx[0].item())
        b_e_orig = int(b_idx[-1].item()) + 1
        fill_rgb = canvas_img[0, orig_h // 2, b_s_orig, :3].view(1, 1, 1, 3)
        b_s_win = max(0, min(win_w - 16, int(round(b_s_orig * win_w / float(orig_w)))))
        b_e_win = max(b_s_win + 16, min(win_w, int(round(b_e_orig * win_w / float(orig_w)))))
        work_canvas[:, :, b_s_win:b_e_win, :] = fill_rgb

    images_dict: dict[str, torch.Tensor] = {"image_1": work_canvas}
    valid_refs = [r for r in ref_images_list if r is not None]
    if valid_refs:
        images_dict["image_2"] = _build_spatially_aligned_ref_canvas(work_canvas, valid_refs[0])

    qwen_enc_classes = _get_target_node_classes("TextEncodeQwenImage21", "comfy_extras.nodes_qwen")
    if not qwen_enc_classes:
        raise RuntimeError("Không tìm thấy node TextEncodeQwenImage21 trong ComfyUI.")
    enc_out = qwen_enc_classes[0].execute(
        clip=clip,
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        vae=vae,
        resolution=0,
        images=images_dict,
    )
    enc_args = list(enc_out.args) if hasattr(enc_out, "args") else list(enc_out)
    positive, negative, latent_dict = enc_args[0], enc_args[1], enc_args[2]

    # Khóa cứng 42% phía ngoài của vùng mồi neo trong không gian Latent (noise_mask = 0.0)
    # để phần mồi neo trong ảnh Gen KHÔNG BAO GIỜ bị trượt/dịch chuyển vị trí so với ảnh gốc,
    # đồng thời để 100% tự do (noise_mask = 1.0) tại 18% sát mép cắt xám + toàn bộ vùng xám để model tự nối liền mạch.
    if has_blank:
        try:
            ref_lats = positive[0][1].get("reference_latents", None)
            if ref_lats and len(ref_lats) > 0:
                canvas_lat = ref_lats[0]
                target_shape = latent_dict["samples"].shape
                if canvas_lat.ndim == 5 and len(target_shape) == 4 and canvas_lat.shape[2] == 1:
                    canvas_lat = canvas_lat.squeeze(2)
                if tuple(canvas_lat.shape) == tuple(target_shape):
                    latent_dict = latent_dict.copy()
                    latent_dict["samples"] = canvas_lat.clone().to(
                        device=latent_dict["samples"].device, dtype=latent_dict["samples"].dtype
                    )
                    mask_1d = torch.ones((win_w,), dtype=torch.float32)
                    # Khóa nửa ngoài mồi neo bên Trái (nếu có)
                    if b_s_win >= 64:
                        anc_l = b_s_win
                        lock_l = max(16, int(anc_l * 0.42))
                        ramp_end_l = max(lock_l + 16, int(anc_l * 0.82))
                        mask_1d[:lock_l] = 0.0
                        rw_l = ramp_end_l - lock_l
                        mask_1d[lock_l:ramp_end_l] = _cosine_alpha_1d(rw_l, mask_1d.device, mask_1d.dtype)
                    # Khóa nửa ngoài mồi neo bên Phải (nếu có)
                    if (win_w - b_e_win) >= 64:
                        anc_r = win_w - b_e_win
                        ramp_start_r = b_e_win + max(16, int(anc_r * 0.18))
                        lock_r_start = max(ramp_start_r + 16, b_e_win + int(anc_r * 0.58))
                        rw_r = lock_r_start - ramp_start_r
                        mask_1d[ramp_start_r:lock_r_start] = 1.0 - _cosine_alpha_1d(rw_r, mask_1d.device, mask_1d.dtype)
                        mask_1d[lock_r_start:] = 0.0
                    latent_dict["noise_mask"] = mask_1d.view(1, 1, 1, win_w).expand(1, 1, win_h, win_w).contiguous()
        except Exception:
            pass

    outpaint_model = _boost_outpaint_lora_strength(model, target_strength=1.0)

    _vstone_free_vram(unload_models=False)
    sampled = nodes.common_ksampler(
        model=outpaint_model,
        seed=int(seed),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent=latent_dict,
        denoise=1.0,
    )[0]

    decoded = vae.decode(sampled["samples"])
    if decoded.ndim == 5:
        decoded = decoded[0]
    decoded = decoded[:1, :, :, :3].to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
    if int(decoded.shape[2]) != orig_w or int(decoded.shape[1]) != orig_h:
        decoded = _chunked_cpu_resize(decoded, orig_w, orig_h, "lanczos", "disabled", chunk_size=1)
    _vstone_free_vram(unload_models=False)
    return decoded


def _prepare_panorama_hq_model(model, lora_hq_strength: float = 0.65):
    """
    Chuẩn bị Model cho quy trình Vẽ Panorama Tổng Hòa (Pha 1) & Upscale Hi-Pass (Pha 2):
    - Tự động gỡ bỏ Outpaint LoRA (vốn chỉ dành cho vẽ hộp xám #808080, nếu để lại sẽ làm hỏng bố cục tổng hòa).
    - Tự động tìm và nạp LoRA `High quality.safetensors` (nếu có trong `models/loras`) ở mức `lora_hq_strength`
      để tăng độ sắc nét, chiều sâu không gian và chi tiết phong cảnh Panorama.
    """
    import uuid
    import nodes
    import folder_paths

    clean_model = model
    try:
        if hasattr(model, "clone") and getattr(model, "patches", None):
            clean_model = model.clone()
            clean_model.patches = {}
            clean_model.patches_uuid = uuid.uuid4()
    except Exception:
        clean_model = model

    if float(lora_hq_strength) <= 0.01:
        return clean_model

    try:
        available_loras = folder_paths.get_filename_list("loras") or []
        hq_name = None
        for cand in ("High quality.safetensors", "High_quality.safetensors"):
            if cand in available_loras:
                hq_name = cand
                break
        if hq_name is None:
            for item in available_loras:
                if "high" in item.lower() and "quality" in item.lower():
                    hq_name = item
                    break
        if hq_name is not None:
            clean_model = nodes.LoraLoaderModelOnly().load_lora_model_only(
                clean_model, hq_name, float(lora_hq_strength)
            )[0]
    except Exception:
        pass
    return clean_model


def _upscale_with_ultrasharp_or_lanczos(img: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    """
    Phóng to ảnh Panorama Pha 1 lên kích thước Pha 2 `(target_w, target_h)`:
    - Ưu tiên dùng model `4x-UltraSharp.safetensors` trong `models/upscale_models` (nếu đã tải về máy) để giữ cạnh sắc nét.
    - Nếu chưa có thì dùng Lanczos kết hợp Unsharp Mask tăng chi tiết tần số cao trước khi vào bước Hi-Pass Denoise thấp.
    """
    import folder_paths

    rgb = img[:1, :, :, :3].to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
    up_img = None
    try:
        up_list = folder_paths.get_filename_list("upscale_models") or []
        up_name = None
        for cand in ("4x-UltraSharp.safetensors", "4x-UltraSharp.pth", "4x_NMKD-Siax_200k.pth"):
            if cand in up_list:
                up_name = cand
                break
        if up_name is None and up_list:
            for item in up_list:
                if "ultrasharp" in item.lower() or "4x" in item.lower():
                    up_name = item
                    break
        if up_name is not None:
            loader_cls = _get_target_node_classes("UpscaleModelLoader", "comfy_extras.nodes_upscale_model")
            upscaler_cls = _get_target_node_classes("ImageUpscaleWithModel", "comfy_extras.nodes_upscale_model")
            if loader_cls and upscaler_cls:
                ldr = loader_cls[0]()
                up_model_out = ldr.execute(up_name) if hasattr(ldr, "execute") else ldr.load_model(up_name)
                up_model = up_model_out[0]
                ups = upscaler_cls[0]()
                up_res = ups.execute(up_model, rgb) if hasattr(ups, "execute") else ups.upscale(up_model, rgb)
                up_img = up_res[0][:1, :, :, :3].to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
                del up_model
                _vstone_free_vram(unload_models=False)
    except Exception:
        up_img = None

    if up_img is None:
        up_img = _chunked_cpu_resize(rgb, target_w, target_h, "lanczos", "disabled", chunk_size=1)
    elif int(up_img.shape[2]) != target_w or int(up_img.shape[1]) != target_h:
        up_img = _chunked_cpu_resize(up_img, target_w, target_h, "lanczos", "disabled", chunk_size=1)

    # Nhấn nhẹ tần số cao (Unsharp Mask 15%) để mồi chi tiết cho Pha 2 Hi-Pass
    nchw = up_img.permute(0, 3, 1, 2)
    blurred = F.avg_pool2d(F.pad(nchw, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1).permute(0, 2, 3, 1)
    sharpened = (up_img + (up_img - blurred) * 0.18).clamp(0.0, 1.0)
    return sharpened


def _phase2_tiled_hipass_upscale(
    hq_model,
    clip,
    vae,
    phase1_pano: torch.Tensor,
    all_refs: list[torch.Tensor],
    up_w: int,
    up_h: int,
    hi_pass_denoise: float,
    hi_pass_steps: int,
    tile_size_px: int,
    tile_overlap_px: int,
    user_prompt: str,
    negative_prompt: str,
    seed: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    uid=None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Pha 2: Phóng to tấm Panorama tổng thể từ Pha 1 lên độ phân giải 5K/6K `(up_w x up_h)`,
    sau đó chia thành các ô nhỏ chồng lấp (`tile_size_px`, `tile_overlap_px`) và chạy Hi-Pass KSampler
    với `denoise` thấp (`0.25 - 0.38`) trực tiếp trên Latent của từng ô (`vae.encode(tile)`):
    - KHÔNG cắt nối / KHÔNG tạo hộp xám `#808080`.
    - Bố cục, đường chân trời và ánh sáng của Pha 1 được giữ nguyên 100%, không bao giờ sinh đường seam.
    - Bơm chi tiết bề mặt sắc nét chuẩn Hi-Pass 5K/6K cho toàn bộ bức Panorama.
    """
    import nodes

    _vstone_ui_status(
        uid,
        f"🔍 [Pha 2/2 - Hi-Pass Upscale] Đang phóng to Panorama lên {up_w}x{up_h}px (4x-UltraSharp / Lanczos)...",
    )
    upscaled = _upscale_with_ultrasharp_or_lanczos(phase1_pano, up_w, up_h)
    if float(hi_pass_denoise) <= 0.02:
        return upscaled, [upscaled]

    # Tính danh sách tọa độ X của các ô Tile chồng lấp trải dọc chiều ngang up_w
    t_w = max(768, min(up_w, int(round(float(tile_size_px) / 64.0)) * 64))
    ov_w = max(192, min(t_w // 2, int(round(float(tile_overlap_px) / 64.0)) * 64))
    stride_w = max(256, t_w - ov_w)

    tile_starts: list[int] = [0]
    while tile_starts[-1] + t_w < up_w:
        next_x = tile_starts[-1] + stride_w
        if next_x + t_w >= up_w:
            tile_starts.append(max(0, up_w - t_w))
            break
        tile_starts.append(next_x)

    qwen_enc_classes = _get_target_node_classes("TextEncodeQwenImage21", "comfy_extras.nodes_qwen")
    if not qwen_enc_classes:
        raise RuntimeError("Không tìm thấy node TextEncodeQwenImage21 trong ComfyUI.")

    hipass_prompt = (
        "High-resolution panoramic landscape refinement and detail enhancement of <image1>. "
        "Preserve the exact composition, horizon line, geometry, perspective, and lighting of <image1> "
        "while enhancing ultra-sharp micro-textures, crisp architectural details, natural foliage, "
        "surface materials, and atmospheric clarity. " + (user_prompt or "").strip()
    ).strip()

    # Giới hạn diện tích lấy mẫu mỗi ô Tile ~1.55 MP để chạy cực nhanh và nhẹ VRAM rồi trả lại đúng kích thước Tile
    max_tile_mp = 1.55
    tile_pixels = float(t_w * up_h)
    if tile_pixels > max_tile_mp * 1_000_000.0 * 1.05:
        t_scale = math.sqrt((max_tile_mp * 1_000_000.0) / tile_pixels)
        work_tw = max(64, int(round((t_w * t_scale) / 64.0)) * 64)
        work_th = max(64, int(round((up_h * t_scale) / 64.0)) * 64)
    else:
        work_tw, work_th = t_w, up_h

    # Trong ComfyUI, KSampler.set_steps chia `int(steps / denoise)`, nên truyền `eff_steps = max(5, round(hi_pass_steps * denoise))`
    # để số bước chạy thực tế trên mỗi ô Tile đúng bằng `~6 - 10` bước thuộc phần đuôi lịch trình Hi-Pass!
    den = max(0.05, min(0.55, float(hi_pass_denoise)))
    eff_steps = max(5, int(round(float(hi_pass_steps) * den)))

    accum_img = torch.zeros_like(upscaled)
    accum_weight = torch.zeros((1, 1, up_w, 1), dtype=torch.float32)
    tile_previews: list[torch.Tensor] = []

    num_tiles = len(tile_starts)
    for idx, x0 in enumerate(tile_starts):
        x1 = x0 + t_w
        _vstone_ui_status(
            uid,
            f"✨ [Pha 2/2 - Hi-Pass Upscale] Đang tinh chỉnh Ô Tile {idx + 1}/{num_tiles} [{x0}:{x1}] (denoise={den:.2f})...",
        )
        raw_tile = upscaled[:, :, x0:x1, :]
        if work_tw != t_w or work_th != up_h:
            work_tile = _chunked_cpu_resize(raw_tile, work_tw, work_th, "lanczos", "disabled", chunk_size=1)
        else:
            work_tile = raw_tile

        images_dict: dict[str, torch.Tensor] = {"image_1": work_tile}
        enc_out = qwen_enc_classes[0].execute(
            clip=clip,
            prompt=hipass_prompt,
            negative_prompt=negative_prompt or "",
            vae=vae,
            resolution=0,
            images=images_dict,
        )
        enc_args = list(enc_out.args) if hasattr(enc_out, "args") else list(enc_out)
        positive, negative, latent_dict = enc_args[0], enc_args[1], enc_args[2]

        # Lấy trực tiếp Latent thật của work_tile từ reference_latents[0] để chạy Hi-Pass Denoise thấp (KHÔNG dùng zeros!)
        ref_lats = positive[0][1].get("reference_latents", None)
        tile_lat = ref_lats[0] if (ref_lats and len(ref_lats) > 0) else vae.encode(work_tile[:, :, :, :3])
        if tile_lat is not None:
            target_shape = latent_dict["samples"].shape
            if tile_lat.ndim == 5 and len(target_shape) == 4 and tile_lat.shape[2] == 1:
                tile_lat = tile_lat.squeeze(2)
            elif tile_lat.ndim == 4 and len(target_shape) == 5 and target_shape[2] == 1:
                tile_lat = tile_lat.unsqueeze(2)
            if tuple(tile_lat.shape) == tuple(target_shape):
                latent_dict = latent_dict.copy()
                latent_dict["samples"] = tile_lat.clone().to(
                    device=latent_dict["samples"].device, dtype=latent_dict["samples"].dtype
                )
                # Khóa mềm biên Trái/Phải của ô Tile bằng noise_mask để mép nối giữa các ô khớp màu và chi tiết 100%
                mask_1d = torch.ones((work_tw,), dtype=torch.float32)
                border_feather = max(32, min(work_tw // 4, int(round(ov_w * work_tw / float(t_w) * 0.5))))
                if idx > 0 and border_feather >= 8:
                    mask_1d[:border_feather] = _cosine_alpha_1d(border_feather, mask_1d.device, mask_1d.dtype)
                if idx < (num_tiles - 1) and border_feather >= 8:
                    mask_1d[-border_feather:] = 1.0 - _cosine_alpha_1d(border_feather, mask_1d.device, mask_1d.dtype)
                latent_dict["noise_mask"] = mask_1d.view(1, 1, 1, work_tw).expand(1, 1, work_th, work_tw).contiguous()

        _vstone_free_vram(unload_models=False)
        sampled = nodes.common_ksampler(
            model=hq_model,
            seed=int(seed) + idx * 37,
            steps=int(eff_steps),
            cfg=float(cfg),
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent_dict,
            denoise=float(den),
        )[0]

        decoded_tile = vae.decode(sampled["samples"])
        if decoded_tile.ndim == 5:
            decoded_tile = decoded_tile[0]
        decoded_tile = decoded_tile[:1, :, :, :3].to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
        if int(decoded_tile.shape[2]) != t_w or int(decoded_tile.shape[1]) != up_h:
            decoded_tile = _chunked_cpu_resize(decoded_tile, t_w, up_h, "lanczos", "disabled", chunk_size=1)

        # Đồng bộ nhẹ màu/độ sáng của ô Tile sau Hi-Pass với raw_tile gốc để 100% không đổi tông màu
        decoded_tile = _match_color_stats(decoded_tile, raw_tile, strength=0.60)
        tile_previews.append(decoded_tile)

        # Tạo cửa sổ trọng số Cosine 1D mượt ở 2 biên chồng lấp để cộng dồn không vết nối
        w_1d = torch.ones((t_w,), dtype=torch.float32)
        if idx > 0 and ov_w > 0:
            w_1d[:ov_w] = _cosine_alpha_1d(ov_w, w_1d.device, w_1d.dtype).clamp(min=1e-3)
        if idx < (num_tiles - 1) and ov_w > 0:
            w_1d[-ov_w:] = (1.0 - _cosine_alpha_1d(ov_w, w_1d.device, w_1d.dtype)).clamp(min=1e-3)
        w_view = w_1d.view(1, 1, t_w, 1)

        accum_img[:, :, x0:x1, :] += decoded_tile * w_view
        accum_weight[:, :, x0:x1, :] += w_view
        _vstone_free_vram(unload_models=False)

    final_pano = (accum_img / accum_weight.clamp(min=1e-5)).clamp(0.0, 1.0)
    return final_pano, tile_previews


class stoneV_Panorama360_AutoLoop:
    """
    Node All-in-One Vẽ Phong Cảnh Panorama 2 Pha (KHÔNG CẮT NỐI - Zero Outpaint Seam):
    - Pha 1 (Tổng Hòa Ngữ Cảnh Từ 1-5 Ảnh Ref): Đưa toàn bộ `image_1` + `ref_image_2..5` qua 2D VAE & Qwen3-VL
      cùng LoRA `High quality.safetensors` để vẽ 1 Tấm Ảnh Dài Phong Cảnh Panorama thống nhất ngay từ đầu
      theo tỉ lệ chuẩn Panorama (`3.0:1`, `3.5:1`, `4.0:1`, hoặc `2.5:1`) — hoàn toàn KHÔNG cắt nối hay đệm vùng xám!
    - Pha 2 (Upscale Hi-Pass Bằng Cách Cắt Nhỏ Chạy Denoise Thấp): Phóng to bức Panorama Pha 1 lên 4K/5K/6K
      (`4x-UltraSharp` / Lanczos), chia thành các ô Tile chồng lấp rộng (`512px`) và chạy Hi-Pass `KSampler`
      với `denoise` thấp (`0.28 - 0.35`) để bơm chi tiết sắc nét mà giữ nguyên 100% bố cục liền mạch của Pha 1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "Create a breathtaking, coherent ultra-wide panoramic landscape background. "
                        "Synthesize and harmonize the environmental context, architecture, terrain, vegetation, sky, "
                        "and lighting from all reference images into one single unbroken wide-angle panorama "
                        "with a continuous horizon line and natural depth."
                    )
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "tiled images, split panels, collage, picture frames, borders, gray background, mismatched perspective, "
                        "broken horizon, disconnected walls or floor, blurry, distorted geometry, watermark, text"
                    )
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 24, "min": 4, "max": 60, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "sampler_name": (["euler", "res_multistep", "dpmpp_2m", "uni_pc"], {"default": "euler"}),
                "scheduler": (["simple", "beta", "normal", "sgm_uniform"], {"default": "simple"}),
            },
            "optional": {
                "panorama_ratio": ([
                    "1. Ratio 3.0:1 - Chuẩn Panorama Phong Cảnh (Pha 1: 2304x768 -> Pha 2: 4608x1536)",
                    "2. Ratio 3.5:1 - Siêu Rộng Ultra-Wide (Pha 1: 2560x704 -> Pha 2: 5120x1408)",
                    "3. Ratio 4.0:1 - Toàn Cảnh Cực Dài (Pha 1: 2688x640 -> Pha 2: 5376x1280)",
                    "4. Ratio 2.5:1 - Điện Ảnh Rộng 21:9 (Pha 1: 2176x832 -> Pha 2: 4352x1664)",
                ], {
                    "default": "1. Ratio 3.0:1 - Chuẩn Panorama Phong Cảnh (Pha 1: 2304x768 -> Pha 2: 4608x1536)"
                }),
                "upscale_factor": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.25,
                    "tooltip": "Hệ số phóng to ở Pha 2 Hi-Pass Upscale (2.0x = từ 2304x768 lên 4608x1536; 2.5x = lên 5760x1920)."
                }),
                "hi_pass_denoise": ("FLOAT", {
                    "default": 0.32,
                    "min": 0.00,
                    "max": 0.55,
                    "step": 0.02,
                    "tooltip": "Mức Denoise thấp ở Pha 2 khi cắt ô nhỏ chạy Hi-Pass (0.28 - 0.35 giúp bơm chi tiết 5K/6K cực nét mà KHÔNG làm lệch bố cục hay sinh đường seam)."
                }),
                "hi_pass_steps": ("INT", {
                    "default": 18,
                    "min": 6,
                    "max": 40,
                    "step": 1,
                    "tooltip": "Tổng số bước lịch trình (Schedule Steps) cho Pha 2 Hi-Pass (với denoise=0.32 sẽ chạy ~6 bước lấy mẫu thực tế cho mỗi ô Tile)."
                }),
                "tile_size_px": ("INT", {
                    "default": 1536,
                    "min": 1024,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Chiều rộng mỗi ô cắt nhỏ (Tile) ở Pha 2 Hi-Pass Upscale (Mặc định 1536px)."
                }),
                "tile_overlap_px": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 768,
                    "step": 64,
                    "tooltip": "Độ rộng vùng chồng lấp Cosine giữa các ô Tile ở Pha 2 (Mặc định 512px xóa sạch 100% ranh giới giữa các ô)."
                }),
                "lora_hq_strength": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.00,
                    "max": 1.20,
                    "step": 0.05,
                    "tooltip": "Tự động gỡ Outpaint LoRA cũ và nạp LoRA 'High quality.safetensors' ở cường độ này để hỗ trợ vẽ phong cảnh Panorama sắc nét."
                }),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "ref_image_5": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("panorama_image", "step_windows_preview", "panorama_width", "panorama_height")
    FUNCTION = "generate_panorama"
    CATEGORY = "stoneV"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def _resolve_base_dimensions(self, panorama_ratio: str) -> tuple[int, int]:
        r_str = str(panorama_ratio or "")
        if "3.5:1" in r_str or r_str.startswith("2."):
            return 2560, 704
        if "4.0:1" in r_str or r_str.startswith("3."):
            return 2688, 640
        if "2.5:1" in r_str or r_str.startswith("4."):
            return 2176, 832
        # Mặc định 3.0:1 (Chuẩn Panorama Phong Cảnh)
        return 2304, 768

    def generate_panorama(
        self,
        model,
        clip,
        vae,
        image_1: torch.Tensor,
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = 42,
        steps: int = 24,
        cfg: float = 1.0,
        sampler_name: str = "euler",
        scheduler: str = "simple",
        panorama_ratio: str = "1. Ratio 3.0:1 - Chuẩn Panorama Phong Cảnh (Pha 1: 2304x768 -> Pha 2: 4608x1536)",
        upscale_factor: float = 2.0,
        hi_pass_denoise: float = 0.32,
        hi_pass_steps: int = 18,
        tile_size_px: int = 1536,
        tile_overlap_px: int = 512,
        lora_hq_strength: float = 0.65,
        ref_image_2: torch.Tensor | None = None,
        ref_image_3: torch.Tensor | None = None,
        ref_image_4: torch.Tensor | None = None,
        ref_image_5: torch.Tensor | None = None,
        **kwargs,
    ):
        uid = kwargs.get("unique_id", None)
        base_img = image_1[:1, :, :, :3].to("cpu", dtype=torch.float32).clamp(0.0, 1.0)

        stale_prefixes = (
            "Gray #808080", "Green #00FF00", "1. Sequential", "2. Rotate",
            "3. First", "Both Sides", "Right Only", "Left Only",
        )
        if not isinstance(prompt, str) or any(prompt.startswith(p) for p in stale_prefixes):
            prompt = ""
        if not (prompt or "").strip() or (prompt or "").strip().startswith("Seamless ultra-wide"):
            prompt = (
                "Create a breathtaking, coherent ultra-wide panoramic landscape background. "
                "Synthesize and harmonize the environmental context, architecture, terrain, vegetation, sky, "
                "and lighting from all reference images into one single unbroken wide-angle panorama "
                "with a continuous horizon line and natural depth."
            )
        if not isinstance(negative_prompt, str) or any(negative_prompt.startswith(p) for p in stale_prefixes):
            negative_prompt = (
                "tiled images, split panels, collage, picture frames, borders, gray background, mismatched perspective, "
                "broken horizon, disconnected walls or floor, blurry, distorted geometry, watermark, text"
            )
        try:
            seed = int(seed)
            if seed < 0:
                seed = 42
        except Exception:
            seed = 42
        try:
            steps = int(steps)
            if steps < 8 or steps > 60:
                steps = 24
        except Exception:
            steps = 24
        try:
            cfg = max(1.0, float(cfg))
        except Exception:
            cfg = 1.0
        if sampler_name not in ("euler", "res_multistep", "dpmpp_2m", "uni_pc"):
            sampler_name = "euler"
        if scheduler not in ("simple", "beta", "normal", "sgm_uniform"):
            scheduler = "simple"
        try:
            upscale_factor = float(upscale_factor)
            if upscale_factor < 1.0 or upscale_factor > 3.0:
                upscale_factor = 2.0
        except Exception:
            upscale_factor = 2.0
        try:
            hi_pass_denoise = float(hi_pass_denoise)
            if hi_pass_denoise < 0.0 or hi_pass_denoise > 0.65:
                hi_pass_denoise = 0.32
        except Exception:
            hi_pass_denoise = 0.32
        try:
            hi_pass_steps = int(hi_pass_steps)
            if hi_pass_steps < 6 or hi_pass_steps > 50:
                hi_pass_steps = 18
        except Exception:
            hi_pass_steps = 18
        try:
            tile_size_px = int(tile_size_px)
            if tile_size_px < 768 or tile_size_px > 2560:
                tile_size_px = 1536
        except Exception:
            tile_size_px = 1536
        try:
            tile_overlap_px = int(tile_overlap_px)
            if tile_overlap_px < 192 or tile_overlap_px > 1024:
                tile_overlap_px = 512
        except Exception:
            tile_overlap_px = 512
        try:
            lora_hq_strength = float(lora_hq_strength)
            if lora_hq_strength < 0.0 or lora_hq_strength > 1.5:
                lora_hq_strength = 0.65
        except Exception:
            lora_hq_strength = 0.65

        # Thu thập và đồng bộ ánh sáng/nhiệt độ màu của toàn bộ 1-5 ảnh ngữ cảnh Reference
        all_2d_refs: list[torch.Tensor] = [base_img]
        for r in (ref_image_2, ref_image_3, ref_image_4, ref_image_5):
            if r is not None and hasattr(r, "shape") and r.ndim == 4:
                r_clean = r[:1, :, :, :3].to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
                all_2d_refs.append(_match_color_stats(r_clean, base_img, strength=0.45))

        base_w, base_h = self._resolve_base_dimensions(panorama_ratio)
        eff_scale = max(1.0, min(3.0, float(upscale_factor)))
        up_w = max(base_w, int(round((base_w * eff_scale) / 64.0)) * 64)
        up_h = max(base_h, int(round((base_h * eff_scale) / 64.0)) * 64)

        # Tự động chuẩn bị Model với LoRA High quality.safetensors (loại bỏ Outpaint LoRA cũ nếu có)
        hq_model = _prepare_panorama_hq_model(model, lora_hq_strength=float(lora_hq_strength))

        # ======================================================================
        # PHA 1: TỔNG HÒA NGỮ CẢNH 1-5 ẢNH REF VẼ 1 TẤM PANORAMA DÀI TỈ LỆ CHUẨN (KHÔNG CẮT NỐI)
        # ======================================================================
        _vstone_ui_status(
            uid,
            f"🌅 [Pha 1/2 - Tổng Hòa Panorama] Đang vẽ 1 tấm Panorama liền mạch ({base_w}x{base_h}px) từ {len(all_2d_refs)} ảnh Ref (Không cắt nối)...",
        )
        phase1_pano = _synthesize_master_wide_from_2d_refs(
            model=hq_model,
            clip=clip,
            vae=vae,
            all_refs=all_2d_refs,
            target_w=base_w,
            target_h=base_h,
            user_prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=sampler_name,
            scheduler=scheduler,
            max_window_mp=2.1,
            keep_model_patches=True,
        )

        # ======================================================================
        # PHA 2: UPSCALE + HI-PASS TILED REFINEMENT (CẮT Ô NHỎ CHẠY DENOISE THẤP, KHÔNG LỘ SEAM)
        # ======================================================================
        if eff_scale <= 1.02 and float(hi_pass_denoise) <= 0.02:
            final_h, final_w = int(phase1_pano.shape[1]), int(phase1_pano.shape[2])
            _vstone_ui_status(uid, f"✅ [Hoàn Tất Pha 1] Panorama: {final_w}x{final_h}px")
            return (phase1_pano.clamp(0.0, 1.0), phase1_pano.clamp(0.0, 1.0), final_w, final_h)

        final_pano, _ = _phase2_tiled_hipass_upscale(
            hq_model=hq_model,
            clip=clip,
            vae=vae,
            phase1_pano=phase1_pano,
            all_refs=all_2d_refs,
            up_w=up_w,
            up_h=up_h,
            hi_pass_denoise=float(hi_pass_denoise),
            hi_pass_steps=int(hi_pass_steps),
            tile_size_px=int(tile_size_px),
            tile_overlap_px=int(tile_overlap_px),
            user_prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed) + 500,
            cfg=float(cfg),
            sampler_name=sampler_name,
            scheduler=scheduler,
            uid=uid,
        )

        final_h, final_w = int(final_pano.shape[1]), int(final_pano.shape[2])
        _vstone_ui_status(
            uid,
            f"✅ [Hoàn Tất 2 Pha - Zero Seam] Pha 1 ({base_w}x{base_h}) -> Pha 2 Hi-Pass ({final_w}x{final_h}px, denoise={float(hi_pass_denoise):.2f})",
        )
        # Đầu ra 1: Ảnh Panorama 5K/6K sau Pha 2 Hi-Pass; Đầu ra 2 (Preview): Ảnh Panorama gốc Pha 1 để đối chiếu
        return (final_pano.clamp(0.0, 1.0), phase1_pano.clamp(0.0, 1.0), final_w, final_h)


class stoneV_Panorama_Stitch2:
    """
    Node ghép nối thủ công 2 tấm Panorama chồng lấp (Left + Right) bằng bộ hòa trộn Cosine mượt,
    dành cho người dùng muốn tự nối từng bước trên sơ đồ node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_panorama": ("IMAGE",),
                "right_extended_window": ("IMAGE",),
                "keep_anchor_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "step_extend_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "blend_width_px": ("INT", {"default": 48, "min": 4, "max": 512, "step": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("stitched_panorama", "width", "height")
    FUNCTION = "stitch"
    CATEGORY = "stoneV"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def stitch(
        self,
        left_panorama: torch.Tensor,
        right_extended_window: torch.Tensor,
        keep_anchor_ratio: float = 0.5,
        step_extend_ratio: float = 1.0,
        blend_width_px: int = 48,
    ):
        win_w = int(right_extended_window.shape[2])
        overlap_ratio = float(keep_anchor_ratio) / max(0.05, float(keep_anchor_ratio) + float(step_extend_ratio))
        overlap_w = int(round(win_w * overlap_ratio))
        out = _stitch_horizontal_cosine(
            left_panorama[:1, :, :, :3].to("cpu", dtype=torch.float32),
            right_extended_window[:1, :, :, :3].to("cpu", dtype=torch.float32),
            overlap_w=overlap_w,
            blend_w=int(blend_width_px),
        )
        return (out, int(out.shape[2]), int(out.shape[1]))


# ==============================================================================
# NODE 9: LTX-2.5 Outpaint / Inpaint Native VStone Integration (Zero Missing Nodes, Zero API Key)
# ==============================================================================
def _ensure_ltx25_model_path(folder_type: str, filename: str) -> str:
    import folder_paths
    try:
        existing = folder_paths.get_full_path(folder_type, filename)
        if existing and os.path.exists(existing) and os.path.getsize(existing) > 1024 * 1024:
            return existing
    except Exception:
        pass
    comfy_root = _get_comfy_root()
    if comfy_root:
        candidate = os.path.join(comfy_root, "models", folder_type, filename)
        if os.path.exists(candidate) and os.path.getsize(candidate) > 1024 * 1024:
            return candidate
    if filename in LTX25_URL_BY_FILENAME:
        f_type, url = LTX25_URL_BY_FILENAME[filename]
        download_file_to_comfy_models(f_type, filename, url)
        try:
            existing = folder_paths.get_full_path(folder_type, filename)
            if existing and os.path.exists(existing):
                return existing
        except Exception:
            pass
        if comfy_root:
            candidate = os.path.join(comfy_root, "models", folder_type, filename)
            if os.path.exists(candidate):
                return candidate
    return folder_paths.get_full_path_or_raise(folder_type, filename)


class stoneV_LTX25_Outpaint_Loader:
    """
    Trình nạp & tự động tải trọn bộ 5 Model LTX-2.5 Outpaint/Inpaint trên H200 (Không cần API Key, Không báo đỏ thiếu file):
    - Tự động kiểm tra và tải bằng aria2c 16 luồng từ Mirror Ungated HTTP 200 nếu máy H200 chưa có model.
    - Trả ra cả `model_iclora_stage1` (có gắn IC-LoRA) và `model_base_stage2` (cho Stage 2 High-Res), cùng `latent_downscale_factor`, `clip`, `video_vae`, `audio_vae`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (
                    ["ltx-2.5-22b-distilled-transformer-bf16.safetensors"],
                    {"default": "ltx-2.5-22b-distilled-transformer-bf16.safetensors"},
                ),
                "ic_lora_name": (
                    [
                        "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
                        "ltx-2.3-22b-ic-lora-outpaint.safetensors",
                    ],
                    {"default": "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"},
                ),
                "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "clip_name": (
                    ["gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"],
                    {"default": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"},
                ),
                "video_vae_name": (
                    ["ltx-2.5-video-vae-bf16.safetensors"],
                    {"default": "ltx-2.5-video-vae-bf16.safetensors"},
                ),
                "audio_vae_name": (
                    ["ltx-2.5-audio-vae-bf16.safetensors"],
                    {"default": "ltx-2.5-audio-vae-bf16.safetensors"},
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MODEL", "MODEL", "FLOAT", "CLIP", "VAE", "VAE")
    RETURN_NAMES = (
        "model_iclora_stage1",
        "model_base_stage2",
        "latent_downscale_factor",
        "clip",
        "video_vae",
        "audio_vae",
    )
    FUNCTION = "load_all"
    CATEGORY = "stoneV"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def load_all(
        self,
        unet_name: str,
        ic_lora_name: str,
        ic_lora_strength: float,
        clip_name: str,
        video_vae_name: str,
        audio_vae_name: str,
        unique_id: Any = None,
    ):
        import folder_paths
        import comfy.sd
        import comfy.utils

        uid = str(unique_id) if unique_id is not None else None
        _vstone_ui_status(uid, "⏳ [LTX-2.5 Loader] Đang kiểm tra / tự động tải bộ 5 Model LTX-2.5 Outpaint (Ungated)...")

        lora_path = _ensure_ltx25_model_path("loras", ic_lora_name)
        v_vae_path = _ensure_ltx25_model_path("vae", video_vae_name)
        a_vae_path = _ensure_ltx25_model_path("vae", audio_vae_name)
        clip_path = _ensure_ltx25_model_path("text_encoders", clip_name)
        unet_path = _ensure_ltx25_model_path("diffusion_models", unet_name)

        _vstone_ui_status(uid, "⏳ [LTX-2.5 Loader] Đang nạp Transformer 22B + IC-LoRA + Gemma4-12B + Video/Audio VAE vào H200...")
        model_base = comfy.sd.load_diffusion_model(unet_path)

        lora_sd, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        latent_downscale_factor = 1.0
        if metadata:
            try:
                if "reference_downscale_factor" in metadata:
                    latent_downscale_factor = float(metadata["reference_downscale_factor"])
                else:
                    for k, v in metadata.items():
                        if k.endswith("reference_downscale_factor"):
                            latent_downscale_factor = float(v)
                            break
            except Exception:
                latent_downscale_factor = 1.0

        if float(ic_lora_strength) != 0.0:
            model_iclora, _ = comfy.sd.load_lora_for_models(model_base, None, lora_sd, float(ic_lora_strength), 0.0)
        else:
            model_iclora = model_base

        clip_type = getattr(comfy.sd.CLIPType, "LTXV", comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )
        video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(v_vae_path))
        audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(a_vae_path))

        _vstone_ui_status(uid, f"✅ [LTX-2.5 Loader] Đã nạp hoàn tất (downscale_factor={latent_downscale_factor})")
        return (model_iclora, model_base, float(latent_downscale_factor), clip, video_vae, audio_vae)


class ImagePadForOutpaintMasked:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"
    CATEGORY = "image"

    def expand_image(self, image, left, top, right, bottom, feathering, mask=None):
        if mask is not None and torch.allclose(mask, torch.zeros_like(mask)):
            mask = None
        B, H, W, C = image.size()
        new_image = torch.ones((B, H + top + bottom, W + left + right, C), dtype=torch.float32, device=image.device) * 0.5
        new_image[:, top : top + H, left : left + W, :] = image

        if mask is None:
            new_mask = torch.ones((B, H + top + bottom, W + left + right), dtype=torch.float32, device=image.device)
            t = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)
        else:
            mask = F.pad(mask, (left, right, top, bottom), mode="constant", value=0)
            mask = 1.0 - mask
            t = torch.zeros_like(mask)

        if feathering > 0 and feathering * 2 < H and feathering * 2 < W:
            ys = torch.arange(H, device=image.device, dtype=torch.float32).view(H, 1)
            xs = torch.arange(W, device=image.device, dtype=torch.float32).view(1, W)
            dt = ys if top != 0 else torch.full_like(ys, float(H))
            db = (H - ys) if bottom != 0 else torch.full_like(ys, float(H))
            dl = xs if left != 0 else torch.full_like(xs, float(W))
            dr = (W - xs) if right != 0 else torch.full_like(xs, float(W))
            d = torch.minimum(torch.minimum(dt, db), torch.minimum(dl, dr))
            v = ((float(feathering) - d).clamp(min=0.0) / float(feathering)) ** 2
            if mask is None:
                t[:, :, :] = v.unsqueeze(0)
            else:
                t[:, top : top + H, left : left + W] = v.unsqueeze(0)

        if mask is None:
            new_mask[:, top : top + H, left : left + W] = t
            return (new_image, new_mask)
        return (new_image, mask)


class ImagePadForOutpaintTargetSize:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 1920, "min": 0, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 1088, "min": 0, "max": 8192, "step": 8}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "upscale_method": (cls.upscale_methods, {"default": "nearest-exact"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"
    CATEGORY = "image"

    def expand_image(self, image, target_width, target_height, feathering, upscale_method="nearest-exact", mask=None):
        import comfy.utils
        B, H, W, C = image.size()
        new_height = H
        new_width = W
        scaling_factor = min(float(target_width) / float(W), float(target_height) / float(H))
        if scaling_factor < 1.0:
            new_width = max(1, int(W * scaling_factor))
            new_height = max(1, int(H * scaling_factor))
            image_scaled = comfy.utils.common_upscale(
                image.movedim(-1, 1), new_width, new_height, upscale_method, "disabled"
            ).movedim(1, -1)
        else:
            image_scaled = image

        if mask is not None:
            mask_scaled = F.interpolate(mask.unsqueeze(0), size=(new_height, new_width), mode="nearest").squeeze(0)
        else:
            mask_scaled = None

        pad_top = max(0, (int(target_height) - new_height) // 2)
        pad_bottom = max(0, int(target_height) - new_height - pad_top)
        pad_left = max(0, (int(target_width) - new_width) // 2)
        pad_right = max(0, int(target_width) - new_width - pad_left)
        return ImagePadForOutpaintMasked().expand_image(
            image_scaled, pad_left, pad_top, pad_right, pad_bottom, feathering, mask_scaled
        )


class LTXICLoRALoaderModelOnly:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            loras = list(folder_paths.get_filename_list("loras"))
        except Exception:
            loras = []
        for default_ltx_lora in (
            "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
            "ltx-2.3-22b-ic-lora-outpaint.safetensors",
        ):
            if default_ltx_lora not in loras:
                loras.insert(0, default_ltx_lora)
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (loras, {"default": "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "FLOAT")
    RETURN_NAMES = ("model", "latent_downscale_factor")
    FUNCTION = "load_ic_lora"
    CATEGORY = "Lightricks/IC-LoRA"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def load_ic_lora(self, model, lora_name: str, strength_model: float = 1.0):
        import comfy.sd
        import comfy.utils
        lora_path = _ensure_ltx25_model_path("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        latent_downscale_factor = 1.0
        if metadata:
            try:
                if "reference_downscale_factor" in metadata:
                    latent_downscale_factor = float(metadata["reference_downscale_factor"])
                else:
                    for k, v in metadata.items():
                        if k.endswith("reference_downscale_factor"):
                            latent_downscale_factor = float(v)
                            break
            except Exception:
                latent_downscale_factor = 1.0
        if float(strength_model) == 0.0:
            return (model, latent_downscale_factor)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, float(strength_model), 0.0)
        return (model_lora, latent_downscale_factor)


class LTXVDilateLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "horizontal_scale": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "vertical_scale": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "dilate_latent"
    CATEGORY = "latent/video"

    def dilate_latent(self, latent: dict, horizontal_scale: int = 1, vertical_scale: int = 1):
        if horizontal_scale == 1 and vertical_scale == 1:
            return (latent,)
        samples = latent["samples"]
        mask = latent.get("noise_mask", None)
        dilated_shape = samples.shape[:3] + (
            samples.shape[3] * int(vertical_scale),
            samples.shape[4] * int(horizontal_scale),
        )
        dilated_samples = torch.zeros(dilated_shape, device=samples.device, dtype=samples.dtype)
        dilated_samples[..., :: int(vertical_scale), :: int(horizontal_scale)] = samples
        dilated_mask_shape = (
            dilated_samples.shape[0],
            1,
            dilated_samples.shape[2],
            dilated_samples.shape[3],
            dilated_samples.shape[4],
        )
        dilated_mask = torch.full(dilated_mask_shape, -1.0, device=samples.device, dtype=samples.dtype)
        dilated_mask[..., :: int(vertical_scale), :: int(horizontal_scale)] = mask if mask is not None else 1.0
        return ({"samples": dilated_samples, "noise_mask": dilated_mask},)


def _ltx_normalize_mask(mask):
    if mask is None:
        return None
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:
        return mask.unsqueeze(0).unsqueeze(0)
    return mask


def _ltx_append_guide_attention_entry(conditioning, pre_filter_count, latent_shape, attention_strength=1.0, attention_mask=None):
    import node_helpers
    existing_entries = []
    for t in conditioning:
        entries = t[1].get("guide_attention_entries", None)
        if entries is not None:
            existing_entries = list(entries)
            break
    existing_entries.append(
        {
            "pre_filter_count": pre_filter_count,
            "strength": attention_strength,
            "pixel_mask": attention_mask,
            "latent_shape": latent_shape,
        }
    )
    return node_helpers.conditioning_set_values(conditioning, {"guide_attention_entries": existing_entries})


class LTXAddVideoICLoRAGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "latent_downscale_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 1.0}),
                "crop": (["disabled", "center"], {"default": "disabled"}),
                "use_tiled_encode": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 512, "step": 32}),
                "tile_overlap": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
            },
            "optional": {
                "use_streaming": ("BOOLEAN", {"default": False}),
                "tile_frames": ("INT", {"default": 97, "min": 0, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "add_guide"
    CATEGORY = "Lightricks/IC-LoRA"

    def add_guide(
        self,
        positive,
        negative,
        vae,
        latent,
        image,
        frame_idx=0,
        strength=1.0,
        latent_downscale_factor=1.0,
        crop="disabled",
        use_tiled_encode=False,
        tile_size=256,
        tile_overlap=64,
        attention_strength=1.0,
        attention_mask=None,
        use_streaming=False,
        tile_frames=97,
    ):
        import comfy.utils
        import comfy_extras.nodes_lt as nodes_lt

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = nodes_lt.get_noise_mask(latent)
        _, _, latent_length, latent_height, latent_width = latent_image.shape

        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        num_frames_to_keep = ((image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        causal_fix = int(frame_idx) == 0 or num_frames_to_keep == 1
        if not causal_fix:
            image = torch.cat([image[:1], image], dim=0)

        keep_frames = ((image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        images_kept = image[:keep_frames]
        target_width = int(latent_width * width_scale_factor / float(latent_downscale_factor))
        target_height = int(latent_height * height_scale_factor / float(latent_downscale_factor))
        pixels = comfy.utils.common_upscale(
            images_kept.movedim(-1, 1), target_width, target_height, "bilinear", crop=crop
        ).movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        if use_tiled_encode:
            guide_latent = vae.encode_tiled(encode_pixels, tile_x=int(tile_size), tile_y=int(tile_size), overlap=int(tile_overlap))
        else:
            guide_latent = vae.encode(encode_pixels)

        if not causal_fix:
            guide_latent = guide_latent[:, :, 1:, :, :]
            encode_pixels = encode_pixels[1:]

        guide_orig_shape = list(guide_latent.shape[2:])
        guide_mask = None
        if float(latent_downscale_factor) > 1.0:
            dilated = LTXVDilateLatent().dilate_latent(
                {"samples": guide_latent},
                horizontal_scale=int(latent_downscale_factor),
                vertical_scale=int(latent_downscale_factor),
            )[0]
            guide_mask = dilated["noise_mask"]
            guide_latent = dilated["samples"]

        iclora_tokens_added = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
        if int(frame_idx) >= 0:
            frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(
                positive, latent_length, len(encode_pixels), int(frame_idx), scale_factors
            )

        positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
            positive,
            negative,
            int(frame_idx),
            latent_image,
            noise_mask,
            guide_latent,
            float(strength),
            scale_factors,
            guide_mask=guide_mask,
            latent_downscale_factor=float(latent_downscale_factor),
            causal_fix=causal_fix,
        )

        norm_mask = _ltx_normalize_mask(attention_mask)
        positive = _ltx_append_guide_attention_entry(
            positive, iclora_tokens_added, guide_orig_shape, attention_strength=float(attention_strength), attention_mask=norm_mask
        )
        negative = _ltx_append_guide_attention_entry(
            negative, iclora_tokens_added, guide_orig_shape, attention_strength=float(attention_strength), attention_mask=norm_mask
        )
        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask})


class LTXAddVideoICLoRAGuideAdvanced(LTXAddVideoICLoRAGuide):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "latent_downscale_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 1.0}),
                "crop": (["disabled", "center"], {"default": "disabled"}),
                "use_tiled_encode": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 512, "step": 32}),
                "tile_overlap": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
                "attention_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "attention_mask": ("MASK",),
                "use_streaming": ("BOOLEAN", {"default": False}),
                "tile_frames": ("INT", {"default": 97, "min": 0, "max": 1000, "step": 1}),
            },
        }


class LTXVImgToVideoConditionOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "bypass": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "conditioning/video_models"

    def generate(self, vae, image, latent, strength=1.0, bypass=False):
        import comfy.utils
        if bypass:
            return (latent,)
        samples = latent["samples"].clone()
        _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
        _, _, _, latent_height, latent_width = samples.shape
        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor
        if image.shape[1] != height or image.shape[2] != width:
            pixels = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        else:
            pixels = image
        t = vae.encode(pixels[:, :, :, :3])
        samples[:, :, : t.shape[2]] = t
        conditioning_latent_frames_mask = torch.ones(
            (1, 1, samples.shape[2], 1, 1), dtype=torch.float32, device=samples.device
        )
        conditioning_latent_frames_mask[:, :, : t.shape[2]] = 1.0 - float(strength)
        return ({"samples": samples, "noise_mask": conditioning_latent_frames_mask},)


class LTXVInpaintPreprocess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preprocess"
    CATEGORY = "Lightricks/image_processing"

    def preprocess(self, images: torch.Tensor, mask: torch.Tensor):
        if mask.ndim == 4:
            mask = mask[:, :, :, 0]
        if mask.shape[0] == 1 and images.shape[0] > 1:
            mask = mask.expand(images.shape[0], -1, -1)
        min_frames = min(mask.shape[0], images.shape[0])
        mask = mask[:min_frames]
        images = images[:min_frames]
        mask_4d = mask.unsqueeze(-1).to(images.device, dtype=images.dtype)
        bg_color = torch.tensor((102.0 / 255.0, 1.0, 0.0), device=images.device, dtype=images.dtype).view(1, 1, 1, 3)
        result = images * (1.0 - mask_4d) + bg_color * mask_4d
        return (result,)


class LTXVDilateVideoMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spatial_radius": ("INT", {"default": 1, "min": 0, "max": 300}),
                "temporal_radius": ("INT", {"default": 0, "min": 0, "max": 10}),
            },
            "optional": {
                "mask": ("MASK",),
                "image_as_mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "dilate"
    CATEGORY = "Lightricks/mask_operations"

    def dilate(self, spatial_radius: int = 1, temporal_radius: int = 0, mask=None, image_as_mask=None):
        if mask is None and image_as_mask is None:
            raise ValueError("Either 'mask' or 'image_as_mask' must be provided.")
        if mask is None:
            mask = image_as_mask.mean(dim=-1)
        if mask.ndim == 4:
            mask = mask[:, :, :, 0]
        s_kernel = int(spatial_radius) * 2 + 1
        t_kernel = int(temporal_radius) * 2 + 1
        if s_kernel > 1:
            mask = F.max_pool2d(mask.unsqueeze(1), kernel_size=s_kernel, stride=1, padding=int(spatial_radius)).squeeze(1)
        if t_kernel > 1:
            B, H, W = mask.shape
            mask = mask.permute(1, 2, 0).reshape(H * W, 1, B)
            mask = F.max_pool1d(mask, kernel_size=t_kernel, stride=1, padding=int(temporal_radius))
            mask = mask.reshape(H, W, B).permute(2, 0, 1)
        return ((mask > 0.5).float(),)


def _ltx_gauss_blur_2d(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    k1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=x.dtype) / 16.0
    k2d = (k1d[:, None] * k1d[None, :]).view(1, 1, 5, 5).expand(c, 1, 5, 5).contiguous()
    x_pad = F.pad(x, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(x_pad, k2d, groups=c)


def _ltx_pyr_down(x: torch.Tensor) -> torch.Tensor:
    return _ltx_gauss_blur_2d(x)[:, :, ::2, ::2]


def _ltx_pyr_up(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    up = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
    return _ltx_gauss_blur_2d(up)


class LTXVLaplacianPyramidBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mask": ("MASK",),
                "trim_to_shortest": ("BOOLEAN", {"default": True}),
                "mask_low_res_dilation": ("INT", {"default": 5, "min": 0, "max": 15}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend"
    CATEGORY = "Lightricks/utility"

    def blend(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        mask: torch.Tensor,
        trim_to_shortest: bool = True,
        mask_low_res_dilation: int = 5,
    ):
        if mask.ndim == 4:
            mask = mask[:, :, :, 0]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if trim_to_shortest:
            shortest = min(image_a.shape[0], image_b.shape[0], mask.shape[0])
            image_a = image_a[:shortest]
            image_b = image_b[:shortest]
            mask = mask[:shortest]

        original = image_a.permute(0, 3, 1, 2).float()
        target = image_b.permute(0, 3, 1, 2).float()
        mask_4d = mask.unsqueeze(1).float()

        if int(mask_low_res_dilation) > 0:
            orig_hw = mask_4d.shape[-2:]
            h, w = orig_hw
            scale = 64.0 / max(1, max(h, w))
            rh, rw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
            m_low = F.interpolate(mask_4d, size=(rh, rw), mode="bilinear", align_corners=False)
            rad = int(mask_low_res_dilation)
            m_low = F.max_pool2d(m_low, kernel_size=rad * 2 + 1, stride=1, padding=rad)
            mask_4d = F.interpolate(m_low, size=orig_hw, mode="bilinear", align_corners=False)

        B, _, H, W = original.shape
        levels = max(1, min(6, int(math.log2(max(2, min(H, W)))) - 1))
        results = []
        for start in range(0, B, 8):
            end = min(start + 8, B)
            g1 = [original[start:end]]
            g2 = [target[start:end]]
            gm = [mask_4d[start:end].to(original.device)]
            for _ in range(levels):
                if min(g1[-1].shape[-2], g1[-1].shape[-1]) < 4:
                    break
                g1.append(_ltx_pyr_down(g1[-1]))
                g2.append(_ltx_pyr_down(g2[-1]))
                gm.append(_ltx_pyr_down(gm[-1]))

            l1 = [g1[i] - _ltx_pyr_up(g1[i + 1], g1[i].shape[-2:]) for i in range(len(g1) - 1)] + [g1[-1]]
            l2 = [g2[i] - _ltx_pyr_up(g2[i + 1], g2[i].shape[-2:]) for i in range(len(g2) - 1)] + [g2[-1]]

            out = l1[-1] * gm[-1] + l2[-1] * (1.0 - gm[-1])
            for i in range(len(l1) - 2, -1, -1):
                res_i = l1[i] * gm[i] + l2[i] * (1.0 - gm[i])
                out = _ltx_pyr_up(out, l1[i].shape[-2:]) + res_i
            results.append(out.clamp(0.0, 1.0).cpu())

        return (torch.cat(results, dim=0).permute(0, 2, 3, 1),)


class LTXVSetAudioRefTokens:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "audio_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "frozen_audio")
    FUNCTION = "set_tokens"
    CATEGORY = "Lightricks/IC-LoRA"

    def set_tokens(self, positive, negative, audio_latent):
        import node_helpers
        latent = audio_latent["samples"]
        b, c, t, f = latent.shape
        ref_tokens = latent.permute(0, 2, 1, 3).reshape(b, t, c * f)
        ref_audio = {"tokens": ref_tokens}
        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref_audio})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref_audio})
        frozen = audio_latent.copy()
        frozen["noise_mask"] = torch.zeros((b, 1, t, 1), dtype=torch.float32, device=latent.device)
        return (positive, negative, frozen)


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
VStone_Panorama360_AutoLoop = stoneV_Panorama360_AutoLoop
VStone_Panorama_Stitch2 = stoneV_Panorama_Stitch2

VSTONE_NODE_CLASS_MAPPINGS = {
    "RTXVideoSuperResolution": RTXVideoSuperResolution,
    "stoneV_RTX_VSR_Pro_16K": RTXVideoSuperResolution,
    "VStone_RTX_VSR_Pro_16K": RTXVideoSuperResolution,
    "MinimaxH3LatentUpscaler3D": MinimaxH3LatentUpscaler3D,
    "stoneV_MinimaxH3_LatentUpscaler3D": MinimaxH3LatentUpscaler3D,
    "VStone_MinimaxH3_LatentUpscaler3D": MinimaxH3LatentUpscaler3D,
    "stoneV_ExtendLeft_1xWidth": stoneV_ExtendLeft_1xWidth,
    "stoneV_ExtendRight_1xWidth": stoneV_ExtendRight_1xWidth,
    "stoneV_ScreenX_Combine3": stoneV_ScreenX_Combine3,
    "stoneV_CloudSyncHub": stoneV_CloudSyncHub,
    "stoneV_Outpaint_Pad": stoneV_Outpaint_Pad,
    "stoneV_SeamlessRestore": stoneV_SeamlessRestore,
    "stoneV_Video_Outpaint_Prepare": stoneV_Video_Outpaint_Prepare,
    "stoneV_ScreenX_Cut": stoneV_ScreenX_Cut,
    "stoneV_Panorama360_AutoLoop": stoneV_Panorama360_AutoLoop,
    "stoneV_Panorama_Stitch2": stoneV_Panorama_Stitch2,
    "VStone_ExtendLeft_1xWidth": stoneV_ExtendLeft_1xWidth,
    "VStone_ExtendRight_1xWidth": stoneV_ExtendRight_1xWidth,
    "VStone_ScreenX_Combine3": stoneV_ScreenX_Combine3,
    "VStone_CloudSyncHub": stoneV_CloudSyncHub,
    "VStone_Outpaint_Pad": stoneV_Outpaint_Pad,
    "VStone_SeamlessRestore": stoneV_SeamlessRestore,
    "VStone_Video_Outpaint_Prepare": stoneV_Video_Outpaint_Prepare,
    "VStone_Panorama360_AutoLoop": stoneV_Panorama360_AutoLoop,
    "VStone_Panorama_Stitch2": stoneV_Panorama_Stitch2,
    "stoneV_LTX25_Outpaint_Loader": stoneV_LTX25_Outpaint_Loader,
    "ImagePadForOutpaintMasked": ImagePadForOutpaintMasked,
    "ImagePadForOutpaintTargetSize": ImagePadForOutpaintTargetSize,
    "LTXICLoRALoaderModelOnly": LTXICLoRALoaderModelOnly,
    "LTXVDilateLatent": LTXVDilateLatent,
    "LTXAddVideoICLoRAGuide": LTXAddVideoICLoRAGuide,
    "LTXAddVideoICLoRAGuideAdvanced": LTXAddVideoICLoRAGuideAdvanced,
    "LTXVImgToVideoConditionOnly": LTXVImgToVideoConditionOnly,
    "LTXVInpaintPreprocess": LTXVInpaintPreprocess,
    "LTXVDilateVideoMask": LTXVDilateVideoMask,
    "LTXVLaplacianPyramidBlend": LTXVLaplacianPyramidBlend,
    "LTXVSetAudioRefTokens": LTXVSetAudioRefTokens,
}

VSTONE_NODE_DISPLAY_NAME_MAPPINGS = {
    "stoneV_LTX25_Outpaint_Loader": "stoneV LTX-2.5 Outpaint All-in-One Model Loader (Auto-Download)",
    "ImagePadForOutpaintMasked": "Pad Image For Outpaint Masked",
    "ImagePadForOutpaintTargetSize": "Pad Image For Outpaint Target Size",
    "LTXICLoRALoaderModelOnly": "🅛🅣🅧 IC-LoRA Loader Model Only",
    "LTXVDilateLatent": "🅛🅣🅧 Dilate Latent",
    "LTXAddVideoICLoRAGuide": "🅛🅣🅧 Add Video IC-LoRA Guide",
    "LTXAddVideoICLoRAGuideAdvanced": "🅛🅣🅧 Add Video IC-LoRA Guide Advanced",
    "LTXVImgToVideoConditionOnly": "🅛🅣🅧 Img To Video Condition Only",
    "LTXVInpaintPreprocess": "🅛🅣🅧 Inpaint Preprocess (#66FF00)",
    "LTXVDilateVideoMask": "🅛🅣🅧 Dilate Video Mask",
    "LTXVLaplacianPyramidBlend": "🅛🅣🅧 Laplacian Pyramid Blend",
    "LTXVSetAudioRefTokens": "🅛🅣🅧 Set Audio Ref Tokens",
    "RTXVideoSuperResolution": "RTX Video Super Resolution Pro (4K/16K)",
    "stoneV_RTX_VSR_Pro_16K": "stoneV RTX VSR Pro & UltraSharp 16K",
    "VStone_RTX_VSR_Pro_16K": "stoneV RTX VSR Pro & UltraSharp 16K",
    "MinimaxH3LatentUpscaler3D": "Minimax H3 Latent Upscaler (3D)",
    "stoneV_MinimaxH3_LatentUpscaler3D": "stoneV Minimax H3 Latent Upscaler (3D)",
    "VStone_MinimaxH3_LatentUpscaler3D": "stoneV Minimax H3 Latent Upscaler (3D)",
    "stoneV_ExtendLeft_1xWidth": "stoneV 1A: Extend Left (Half-Crop + 1.3x Pad)",
    "stoneV_ExtendRight_1xWidth": "stoneV 1B: Extend Right (Half-Crop + 1.3x Pad)",
    "stoneV_ScreenX_Combine3": "stoneV 2: ScreenX Combine 3 Plates (Left + Center + Right)",
    "stoneV_CloudSyncHub": "stoneV 3: Cloud Sync & Model Downloader",
    "stoneV_Outpaint_Pad": "stoneV: Outpaint Pad (Custom Left / Right)",
    "stoneV_SeamlessRestore": "stoneV: Seamless Restore Original (Image / Video)",
    "stoneV_Video_Outpaint_Prepare": "stoneV: Video Outpaint Prepare",
    "stoneV_ScreenX_Cut": "stoneV: ScreenX Cut (Cánh + Ảnh Giữa)",
    "stoneV_Panorama360_AutoLoop": "stoneV Panorama 360: Auto-Loop Multi-Ref & Height Expand",
    "stoneV_Panorama_Stitch2": "stoneV Panorama: Stitch & Feather 2 Plates",
    "VStone_ExtendLeft_1xWidth": "stoneV 1A: Extend Left (Half-Crop + 1.3x Pad)",
    "VStone_ExtendRight_1xWidth": "stoneV 1B: Extend Right (Half-Crop + 1.3x Pad)",
    "VStone_ScreenX_Combine3": "stoneV 2: ScreenX Combine 3 Plates (Left + Center + Right)",
    "VStone_CloudSyncHub": "stoneV 3: Cloud Sync & Model Downloader",
    "VStone_Outpaint_Pad": "stoneV: Outpaint Pad (Custom Left / Right)",
    "VStone_SeamlessRestore": "stoneV: Seamless Restore Original (Image / Video)",
    "VStone_Video_Outpaint_Prepare": "stoneV: Video Outpaint Prepare",
    "VStone_Panorama360_AutoLoop": "stoneV Panorama 360: Auto-Loop Multi-Ref & Height Expand",
    "VStone_Panorama_Stitch2": "stoneV Panorama: Stitch & Feather 2 Plates",
}

for _node_cls in VSTONE_NODE_CLASS_MAPPINGS.values():
    try:
        if not getattr(_node_cls, "RELATIVE_PYTHON_MODULE", None):
            _node_cls.RELATIVE_PYTHON_MODULE = "custom_nodes.vstone"
    except Exception:
        pass

NODE_CLASS_MAPPINGS = VSTONE_NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = VSTONE_NODE_DISPLAY_NAME_MAPPINGS
