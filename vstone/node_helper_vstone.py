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

import importlib.util
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

WAN21_VACE_OUTPAINT_MODELS: list[dict[str, str]] = [
    {
        "folder": "text_encoders",
        "filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    },
    {
        "folder": "vae",
        "filename": "wan_2.1_vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
    },
    {
        "folder": "loras",
        "filename": "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
    },
    {
        "folder": "diffusion_models",
        "filename": "wan2.1_vace_14B_fp8_e4m3fn.safetensors",
        "url": "https://huggingface.co/Kamikaze-88/Wan2.1-VACE-14B-fp8/resolve/main/wan2.1_vace_14B_fp8_e4m3fn.safetensors",
    },
    {
        "folder": "diffusion_models",
        "filename": "wan2.1_vace_14B_fp16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors",
    },
    {
        "folder": "text_encoders",
        "filename": "umt5_xxl_fp16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors",
    },
    {
        "folder": "diffusion_models",
        "filename": "wan2.1_vace_1.3B_fp16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors",
    },
    {
        "folder": "loras",
        "filename": "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
    },
]

LTX25_URL_BY_FILENAME: dict[str, tuple[str, str]] = {
    item["filename"]: (item["folder"], item["url"])
    for item in (LTX25_OUTPAINT_SUITE_MODELS + WAN21_VACE_OUTPAINT_MODELS)
}

PRESET_DOWNLOAD_CATALOG: dict[str, tuple[str, str, str]] = {
    "ALL Wan 2.1 VACE Outpaint Suite (Tải trọn bộ Wan 2.1 VACE 14B + UMT5 + VAE + CausVid LoRA)": ("__ALL_WAN21_VACE__", "", ""),
    "Wan 2.1 UMT5-XXL FP8 Scaled Text Encoder (umt5_xxl_fp8_e4m3fn_scaled.safetensors - 6.7GB)": (
        "text_encoders",
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    "Wan 2.1 VACE 14B Full Merged FP8 (wan2.1_vace_14B_fp8_e4m3fn.safetensors - 16.1GB)": (
        "diffusion_models",
        "wan2.1_vace_14B_fp8_e4m3fn.safetensors",
        "https://huggingface.co/Kamikaze-88/Wan2.1-VACE-14B-fp8/resolve/main/wan2.1_vace_14B_fp8_e4m3fn.safetensors",
    ),
    "Wan 2.1 VACE 14B Full Merged FP16 Official (wan2.1_vace_14B_fp16.safetensors - 32.3GB)": (
        "diffusion_models",
        "wan2.1_vace_14B_fp16.safetensors",
        "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors",
    ),
    "Wan 2.1 VAE Official (wan_2.1_vae.safetensors - 254MB)": (
        "vae",
        "wan_2.1_vae.safetensors",
        "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
    ),
    "Wan 2.1 CausVid 14B T2V LoRA Rank32 (Wan21_CausVid_14B_T2V_lora_rank32.safetensors - 291MB)": (
        "loras",
        "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
    ),
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
    "workflows/video_wan_vace_outpainting.json",
    "workflows/wan_vace/video_wan_vace_outpainting.json",
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
    "workflows/03_Netflix_VOID_And_VFX_Roto_Inpaint/2B_Netflix_VOID_Click_Box_Select_SAM31.json",
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
        import urllib.parse as _urlparse
        for path in wf_list:
            if not (path.startswith("workflows/") and path.endswith(".json")):
                continue
            rel_wf = path[len("workflows/"):]
            try:
                quoted_path = _urlparse.quote(path, safe="/")
                wf_bytes = _http_get_bytes(f"{raw_ref_base}/{quoted_path}?_t={ts}", timeout=15)
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

        # 3a. Đồng bộ bản vá patches/nodes_void.py -> ComfyUI/comfy_extras/nodes_void.py (sửa lỗi center-crop của Netflix VOID)
        try:
            void_patch_bytes = _http_get_bytes(f"{raw_ref_base}/patches/nodes_void.py?_t={ts}", timeout=15)
            if void_patch_bytes and len(void_patch_bytes) > 500:
                void_dst = os.path.join(comfy_root, "comfy_extras", "nodes_void.py")
                if _atomic_write_bytes(void_dst, void_patch_bytes, min_bytes=500):
                    logs.append(f"Patched Netflix VOID: {void_dst}")
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


# 11: Auto-update & hot-load all 28 Portable Workflow Pack custom node repos & pip deps on H200
VSTONE_RUNTIME_PATCH_VER = 11
MINIMAX_SAFE_MAX_FRAMES = 73  # 17 * 4 + 5 (~3.04s @ 24fps)
MINIMAX_SAFE_DI_W = 1280
MINIMAX_SAFE_DI_H = 704
# Lưu trên sys để mọi lần hot-reload dùng chung 1 lock, không tải 2 lần vào cùng file .part
_DOWNLOAD_LOCK = sys.__dict__.setdefault("_vstone_download_lock", threading.Lock())


def _read_safetensors_header(path: str) -> tuple[dict[str, Any] | None, int, int]:
    """Read the JSON header of a .safetensors file in <1ms and compute (header, expected_file_size, actual_file_size)."""
    import struct
    try:
        if not path or not os.path.isfile(path):
            return None, 0, 0
        actual_size = os.path.getsize(path)
        if actual_size < 16:
            return None, 0, actual_size
        with open(path, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) < 8:
                return None, 0, actual_size
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len <= 0 or header_len > 100 * 1024 * 1024 or actual_size < 8 + header_len:
                return None, 8 + int(header_len), actual_size
            hdr = json.loads(f.read(header_len).decode("utf-8"))
        if not isinstance(hdr, dict):
            return None, 0, actual_size
        max_end = 0
        for k, v in hdr.items():
            if k != "__metadata__" and isinstance(v, dict):
                offsets = v.get("data_offsets")
                if isinstance(offsets, list) and len(offsets) == 2:
                    end_off = int(offsets[1])
                    if end_off > max_end:
                        max_end = end_off
        expected_size = 8 + int(header_len) + max_end
        return hdr, expected_size, actual_size
    except Exception:
        return None, 0, 0


def _is_valid_safetensors_file(path: str) -> bool:
    """Verify that a .safetensors file has a valid JSON header and its size on disk matches 8 + header_len + max_offset."""
    if not path or not os.path.isfile(path):
        return False
    if not path.lower().endswith(".safetensors"):
        return os.path.getsize(path) > 1024 * 1024
    hdr, expected_size, actual_size = _read_safetensors_header(path)
    return hdr is not None and expected_size > 8 and actual_size == expected_size


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

    # 3. Hook folder_paths & comfy.sd so native UNETLoader/CLIPLoader/VAELoader/LoraLoaderModelOnly
    #    recognize, validate, auto-download, and merge LTX-2.5 & Wan 2.1 VACE models seamlessly.
    try:
        import folder_paths
        orig_get_filename_list = getattr(folder_paths, "_vstone_orig_get_filename_list", None) or folder_paths.get_filename_list
        orig_get_full_path = getattr(folder_paths, "_vstone_orig_get_full_path", None) or folder_paths.get_full_path
        setattr(folder_paths, "_vstone_orig_get_filename_list", orig_get_filename_list)
        setattr(folder_paths, "_vstone_orig_get_full_path", orig_get_full_path)

        if getattr(folder_paths, "_vstone_hook_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
            _ltx_virtual_files = {
                "diffusion_models": [
                    "wan2.1_vace_14B_fp16.safetensors",
                    "wan2.1_vace_14B_fp8_e4m3fn.safetensors",
                    "wan2.1_vace_1.3B_fp16.safetensors",
                    "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                ],
                "unet": [
                    "wan2.1_vace_14B_fp16.safetensors",
                    "wan2.1_vace_14B_fp8_e4m3fn.safetensors",
                    "wan2.1_vace_1.3B_fp16.safetensors",
                    "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                ],
                "loras": [
                    "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
                    "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
                    "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
                ],
                "text_encoders": [
                    "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "umt5_xxl_fp16.safetensors",
                    "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                ],
                "clip": [
                    "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "umt5_xxl_fp16.safetensors",
                    "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                ],
                "vae": [
                    "wan_2.1_vae.safetensors",
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
                for item in list(res):
                    if "/" in item:
                        bs_item = item.replace("/", "\\")
                        if bs_item not in res:
                            res.append(bs_item)
                return res

            def _vstone_get_full_path(folder_name: str, filename: str):
                p = orig_get_full_path(folder_name, filename)
                if p is None and filename and "\\" in filename:
                    p = orig_get_full_path(folder_name, filename.replace("\\", "/"))
                if p is not None and os.path.exists(p):
                    if _is_valid_safetensors_file(p):
                        return p
                    print(f"[VStone Auto-Downloader] Detected corrupted/incomplete file at {p}, removing and re-downloading...")
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                clean_name = os.path.basename(filename) if filename else ""
                if clean_name in LTX25_URL_BY_FILENAME:
                    target_folder, url = LTX25_URL_BY_FILENAME[clean_name]
                    print(f"[VStone Auto-Downloader] Fetching {clean_name} from direct mirror...")
                    download_file_to_comfy_models(target_folder, clean_name, url)
                    p = orig_get_full_path(folder_name, filename)
                    if p is not None and os.path.exists(p) and _is_valid_safetensors_file(p):
                        return p
                    comfy_root = _get_comfy_root()
                    if comfy_root:
                        fallback_p = os.path.join(comfy_root, "models", target_folder, clean_name)
                        if os.path.exists(fallback_p) and _is_valid_safetensors_file(fallback_p):
                            return fallback_p
                return p

            folder_paths.get_filename_list = _vstone_get_filename_list
            folder_paths.get_full_path = _vstone_get_full_path
            setattr(folder_paths, "_vstone_ltx25_hooked", True)
            setattr(folder_paths, "_vstone_hook_ver", VSTONE_RUNTIME_PATCH_VER)
    except Exception:
        pass

    # 3b. Hook comfy.sd.load_clip & comfy.sd.load_diffusion_model to auto-heal corrupted UMT5
    #     and auto-merge Kijai's split Wan2_1-VACE_module_14B_*.safetensors in UNETLoader!
    try:
        import comfy.sd
        import comfy.utils
        import folder_paths

        orig_load_clip = getattr(comfy.sd, "_vstone_orig_load_clip", None) or comfy.sd.load_clip
        setattr(comfy.sd, "_vstone_orig_load_clip", orig_load_clip)

        if getattr(comfy.sd, "_vstone_sd_hook_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
            def _vstone_load_clip(ckpt_paths, embedding_directory=None, clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options={}, disable_dynamic=False):
                fixed_paths = []
                for p in ckpt_paths:
                    bname = os.path.basename(p).lower() if p else ""
                    need_umt5_fallback = False
                    if p and not _is_valid_safetensors_file(p):
                        print(f"[VStone CLIP Guard] Corrupted text encoder detected: {p}")
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                        cname = os.path.basename(p)
                        if cname in LTX25_URL_BY_FILENAME:
                            tfolder, turl = LTX25_URL_BY_FILENAME[cname]
                            download_file_to_comfy_models(tfolder, cname, turl)
                            comfy_root = _get_comfy_root()
                            re_p = os.path.join(comfy_root, "models", tfolder, cname) if comfy_root else p
                            if os.path.exists(re_p) and _is_valid_safetensors_file(re_p):
                                fixed_paths.append(re_p)
                                continue
                        need_umt5_fallback = "umt5" in bname or clip_type == getattr(comfy.sd.CLIPType, "WAN", None)
                    elif bname.startswith("umt5-xxl-enc"):
                        # Kijai's umt5-xxl-enc-*.safetensors uses blocks.0.attn.* keys which native CLIPLoader
                        # mis-detects as SD1ClipModel (logging missing text_model.encoder.layers.* keys).
                        print(f"[VStone CLIP Guard] Converting Kijai '{os.path.basename(p)}' selection to Comfy-Org native 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'...")
                        need_umt5_fallback = True

                    if need_umt5_fallback:
                        umt5_name = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
                        tfolder, turl = LTX25_URL_BY_FILENAME[umt5_name]
                        download_file_to_comfy_models(tfolder, umt5_name, turl)
                        comfy_root = _get_comfy_root()
                        if comfy_root:
                            umt5_p = os.path.join(comfy_root, "models", tfolder, umt5_name)
                            if os.path.exists(umt5_p) and _is_valid_safetensors_file(umt5_p):
                                fixed_paths.append(umt5_p)
                                continue
                    fixed_paths.append(p)
                return orig_load_clip(fixed_paths, embedding_directory=embedding_directory, clip_type=clip_type, model_options=model_options, disable_dynamic=disable_dynamic)

            orig_load_diffusion_model = getattr(comfy.sd, "_vstone_orig_load_diffusion_model", None) or comfy.sd.load_diffusion_model
            setattr(comfy.sd, "_vstone_orig_load_diffusion_model", orig_load_diffusion_model)

            def _vstone_load_diffusion_model(unet_path, model_options={}, disable_dynamic=False):
                if unet_path and not _is_valid_safetensors_file(unet_path):
                    cname = os.path.basename(unet_path)
                    print(f"[VStone UNET Guard] Corrupted diffusion model detected at {unet_path}, removing...")
                    try:
                        os.remove(unet_path)
                    except Exception:
                        pass
                    if cname in LTX25_URL_BY_FILENAME:
                        tfolder, turl = LTX25_URL_BY_FILENAME[cname]
                        download_file_to_comfy_models(tfolder, cname, turl)

                hdr, _, _ = _read_safetensors_header(unet_path)
                # Detect Kijai's split VACE module (e.g. Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors or bf16)
                # which has vace_patch_embedding.weight (236 tensors) but lacks the 1,095 base T2V tensors (head.modulation).
                if isinstance(hdr, dict) and "vace_patch_embedding.weight" in hdr and "head.modulation" not in hdr:
                    vace_shape = hdr.get("vace_patch_embedding.weight", {}).get("shape", [])
                    target_dim = vace_shape[0] if isinstance(vace_shape, list) and len(vace_shape) > 0 else 5120
                    prefer_fp8 = "fp8" in os.path.basename(unet_path).lower()

                    diff_dirs = []
                    try:
                        diff_dirs.extend(folder_paths.get_folder_paths("diffusion_models"))
                    except Exception:
                        pass
                    comfy_root = _get_comfy_root()
                    if comfy_root:
                        d_default = os.path.join(comfy_root, "models", "diffusion_models")
                        if d_default not in diff_dirs:
                            diff_dirs.append(d_default)

                    full_vace_candidates: list[str] = []
                    base_t2v_candidates: list[str] = []
                    for d_root in diff_dirs:
                        if not os.path.isdir(d_root):
                            continue
                        for dirpath, _, filenames in os.walk(d_root):
                            for fn in filenames:
                                if not fn.lower().endswith(".safetensors"):
                                    continue
                                full_p = os.path.join(dirpath, fn)
                                if os.path.abspath(full_p) == os.path.abspath(unet_path):
                                    continue
                                c_hdr, c_exp, c_act = _read_safetensors_header(full_p)
                                if not isinstance(c_hdr, dict) or c_exp <= 8 or c_act != c_exp:
                                    continue
                                pe_shape = c_hdr.get("patch_embedding.weight", {}).get("shape")
                                if "head.modulation" in c_hdr and pe_shape == [target_dim, 16, 1, 2, 2]:
                                    if "vace_patch_embedding.weight" in c_hdr:
                                        full_vace_candidates.append(full_p)
                                    else:
                                        base_t2v_candidates.append(full_p)

                    # Sort candidates so matching precision (fp8 vs bf16/fp16) comes first
                    base_t2v_candidates.sort(key=lambda p: (0 if ("fp8" in os.path.basename(p).lower()) == prefer_fp8 else 1, p))
                    full_vace_candidates.sort(key=lambda p: (0 if ("fp8" in os.path.basename(p).lower()) == prefer_fp8 else 1, p))

                    if base_t2v_candidates:
                        base_path = base_t2v_candidates[0]
                        print(f"[VStone WanVACE Adapter] Merging local 14B T2V base '{os.path.basename(base_path)}' + VACE module '{os.path.basename(unet_path)}' in RAM...")
                        base_sd, base_meta = comfy.utils.load_torch_file(base_path, return_metadata=True)
                        vace_sd, _ = comfy.utils.load_torch_file(unet_path, return_metadata=True)
                        merged_sd = dict(base_sd)
                        base_linear_dtype = base_sd["blocks.0.self_attn.q.weight"].dtype if "blocks.0.self_attn.q.weight" in base_sd else None
                        base_pe_dtype = base_sd["patch_embedding.weight"].dtype if "patch_embedding.weight" in base_sd else None
                        for k, v in vace_sd.items():
                            if k.startswith("model_type."):
                                continue
                            if hasattr(v, "dtype"):
                                if k == "vace_patch_embedding.weight" and base_pe_dtype is not None and v.dtype != base_pe_dtype:
                                    v = v.to(dtype=base_pe_dtype)
                                elif (
                                    base_linear_dtype is not None
                                    and k.startswith("vace_blocks.")
                                    and k.endswith(".weight")
                                    and "norm" not in k
                                    and v.dtype != base_linear_dtype
                                ):
                                    v = v.to(dtype=base_linear_dtype)
                            merged_sd[k] = v
                        model_opts = dict(model_options) if isinstance(model_options, dict) else {}
                        if base_linear_dtype == getattr(torch, "float8_e4m3fn", None) and "dtype" not in model_opts:
                            model_opts["dtype"] = torch.float8_e4m3fn
                        model = comfy.sd.load_diffusion_model_state_dict(
                            merged_sd, model_options=model_opts, metadata=base_meta, disable_dynamic=disable_dynamic
                        )
                        if model is not None:
                            model.cached_patcher_init = (_vstone_load_diffusion_model, (unet_path, model_options))
                            return model

                    if full_vace_candidates:
                        chosen_full = full_vace_candidates[0]
                        print(f"[VStone WanVACE Adapter] Redirecting split VACE module '{os.path.basename(unet_path)}' to full merged VACE model '{os.path.basename(chosen_full)}'...")
                        return orig_load_diffusion_model(chosen_full, model_options=model_options, disable_dynamic=disable_dynamic)

                    # Neither local T2V base nor full VACE model is on disk yet: auto-download full merged VACE model!
                    dl_name = "wan2.1_vace_1.3B_fp16.safetensors" if target_dim == 1536 else "wan2.1_vace_14B_fp8_e4m3fn.safetensors"
                    tfolder, turl = LTX25_URL_BY_FILENAME[dl_name]
                    print(f"[VStone WanVACE Adapter] Split module '{os.path.basename(unet_path)}' has no local base T2V model. Auto-downloading full merged '{dl_name}'...")
                    download_file_to_comfy_models(tfolder, dl_name, turl)
                    if comfy_root:
                        dl_path = os.path.join(comfy_root, "models", tfolder, dl_name)
                        if os.path.exists(dl_path) and _is_valid_safetensors_file(dl_path):
                            return orig_load_diffusion_model(dl_path, model_options=model_options, disable_dynamic=disable_dynamic)

                return orig_load_diffusion_model(unet_path, model_options=model_options, disable_dynamic=disable_dynamic)

            comfy.sd.load_clip = _vstone_load_clip
            comfy.sd.load_diffusion_model = _vstone_load_diffusion_model
            setattr(comfy.sd, "_vstone_sd_hook_ver", VSTONE_RUNTIME_PATCH_VER)
    except Exception as e:
        print(f"[VStone Runtime] Warning hooking comfy.sd: {e}")

    # 3c. Hook PointsEditor (KJNodes), SAM3_Detect & SAM3_VideoTrack for seamless Click + Bounding Box selection:
    #     - PointsEditor: allow empty coordinates ("[]") without raising ValueError (so first run loads Frame 1 and box-only selection works),
    #       auto-scale coordinates/boxes to bg_image dimensions, and store drawn boxes in sys._vstone_last_points_editor_boxes.
    #     - SAM3_Detect: accept both BOUNDING_BOX and BBOX on bboxes input, auto-convert PointsEditor tuples/boxes,
    #       and combine point_inputs + box_inputs in sam3_model.forward_segment when both are provided.
    #     - SAM3_VideoTrack: fast-return empty track when initial_mask is all-zeros and conditioning is empty (instant Frame 1 preview load).
    try:
        import io as _py_io
        import base64 as _b64
        from PIL import Image as _PILImage
        import numpy as _np
        import nodes

        pe_cls = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("PointsEditor")
        if pe_cls is not None and getattr(pe_cls, "_vstone_pe_hook_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
            orig_pointdata = getattr(pe_cls, "_vstone_orig_pointdata", None) or getattr(pe_cls, "pointdata", None)
            if orig_pointdata is not None:
                setattr(pe_cls, "_vstone_orig_pointdata", orig_pointdata)

                def _vstone_pointdata(self, points_store, bbox_store, width, height, coordinates, neg_coordinates, normalize, bboxes, bbox_format="xyxy", bg_image=None):
                    try:
                        raw_pos = json.loads(coordinates) if coordinates else []
                    except Exception:
                        raw_pos = []
                    try:
                        raw_neg = json.loads(neg_coordinates) if neg_coordinates else []
                    except Exception:
                        raw_neg = []
                    try:
                        raw_bboxes = json.loads(bboxes) if bboxes else []
                    except Exception:
                        raw_bboxes = []

                    w_in = max(1, int(width) if width else 512)
                    h_in = max(1, int(height) if height else 512)
                    if bg_image is not None and hasattr(bg_image, "shape") and bg_image.ndim == 4:
                        out_h, out_w = int(bg_image.shape[1]), int(bg_image.shape[2])
                    else:
                        out_h, out_w = h_in, w_in

                    sx = float(out_w) / float(w_in)
                    sy = float(out_h) / float(h_in)

                    pos_coordinates = []
                    if isinstance(raw_pos, list):
                        for coord in raw_pos:
                            if isinstance(coord, dict) and coord.get("x") is not None and coord.get("y") is not None:
                                cx = float(coord["x"])
                                cy = float(coord["y"])
                                if normalize:
                                    pos_coordinates.append({"x": cx / w_in, "y": cy / h_in})
                                else:
                                    pos_coordinates.append({"x": int(round(cx * sx)), "y": int(round(cy * sy))})

                    neg_list = []
                    if isinstance(raw_neg, list):
                        for coord in raw_neg:
                            if isinstance(coord, dict) and coord.get("x") is not None and coord.get("y") is not None:
                                cx = float(coord["x"])
                                cy = float(coord["y"])
                                if normalize:
                                    neg_list.append({"x": cx / w_in, "y": cy / h_in})
                                else:
                                    neg_list.append({"x": int(round(cx * sx)), "y": int(round(cy * sy))})

                    mask = _np.zeros((out_h, out_w), dtype=_np.uint8)
                    valid_bboxes = []
                    sam3_bbox_dicts = []
                    if isinstance(raw_bboxes, list):
                        for b in raw_bboxes:
                            if isinstance(b, dict) and all(b.get(k) is not None for k in ("startX", "startY", "endX", "endY")):
                                x_min = int(round(min(float(b["startX"]), float(b["endX"])) * sx))
                                y_min = int(round(min(float(b["startY"]), float(b["endY"])) * sy))
                                x_max = int(round(max(float(b["startX"]), float(b["endX"])) * sx))
                                y_max = int(round(max(float(b["startY"]), float(b["endY"])) * sy))
                                x_min = max(0, min(out_w, x_min))
                                x_max = max(0, min(out_w, x_max))
                                y_min = max(0, min(out_h, y_min))
                                y_max = max(0, min(out_h, y_max))
                                if x_max > x_min and y_max > y_min:
                                    valid_bboxes.append((x_min, y_min, x_max, y_max))
                                    sam3_bbox_dicts.append({
                                        "x": float(x_min),
                                        "y": float(y_min),
                                        "width": float(x_max - x_min),
                                        "height": float(y_max - y_min),
                                    })
                                    mask[y_min:y_max, x_min:x_max] = 1

                    setattr(sys, "_vstone_last_points_editor_boxes", sam3_bbox_dicts)

                    if bbox_format == "xywh":
                        out_bboxes = [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in valid_bboxes]
                    else:
                        out_bboxes = list(valid_bboxes)

                    mask_tensor = torch.from_numpy(mask).unsqueeze(0).float().cpu()
                    if bg_image is not None and len(valid_bboxes) > 0:
                        x1, y1, x2, y2 = valid_bboxes[0]
                        cropped_image = bg_image[:, y1:y2, x1:x2, :]
                    elif bg_image is not None:
                        cropped_image = bg_image
                    else:
                        cropped_image = torch.zeros(1, out_h, out_w, 3, dtype=torch.float32)

                    res_tuple = (json.dumps(pos_coordinates), json.dumps(neg_list), out_bboxes, mask_tensor, cropped_image)
                    # SAM3_Detect chỉ mượn các khung trên khi nhận đúng các điểm của lần chạy PointsEditor này.
                    setattr(sys, "_vstone_last_points_editor_coords", (res_tuple[0], res_tuple[1]))
                    if bg_image is None:
                        return res_tuple
                    frame0_np = (bg_image[0, :, :, :3].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(_np.uint8)
                    pil_img = _PILImage.fromarray(frame0_np)
                    buffered = _py_io.BytesIO()
                    pil_img.save(buffered, format="JPEG", quality=80)
                    img_base64 = _b64.b64encode(buffered.getvalue()).decode("utf-8")
                    return {
                        "ui": {"bg_image": [img_base64]},
                        "result": res_tuple,
                    }

                pe_cls.pointdata = _vstone_pointdata
                setattr(pe_cls, "_vstone_pe_hook_ver", VSTONE_RUNTIME_PATCH_VER)
    except Exception as e:
        print(f"[VStone Runtime] Warning hooking PointsEditor: {e}")

    try:
        from comfy_api.latest import io as _comfy_io
        import comfy.model_management
        import comfy.utils
        import torch.nn.functional as _F

        for sd_cls in _get_target_node_classes("SAM3_Detect", "comfy_extras.nodes_sam3"):
            if getattr(sd_cls, "_vstone_sam3_hook_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                orig_def_schema = _unwrap_to_true_orig(sd_cls, "define_schema")
                orig_sd_exec = _unwrap_to_true_orig(sd_cls, "execute")
                if orig_def_schema is not None:
                    setattr(sd_cls, "_vstone_orig_define_schema", orig_def_schema)

                    def _safe_sam3_define_schema(cls, _orig=orig_def_schema):
                        raw_fn = _orig.__func__ if isinstance(_orig, classmethod) else _orig
                        schema = raw_fn(cls)
                        try:
                            new_inputs = []
                            for inp in schema.inputs:
                                if getattr(inp, "id", None) == "bboxes":
                                    new_inputs.append(
                                        _comfy_io.MultiType.Input(
                                            _comfy_io.BoundingBox.Input(
                                                "bboxes",
                                                display_name="bboxes",
                                                force_input=True,
                                                optional=True,
                                                tooltip="Bounding boxes to segment within (BOUNDING_BOX or KJNodes BBOX)",
                                            ),
                                            [_comfy_io.BoundingBox, _comfy_io.BBOX],
                                        )
                                    )
                                else:
                                    new_inputs.append(inp)
                            schema.inputs = new_inputs
                        except Exception:
                            pass
                        return schema

                    sd_cls.define_schema = classmethod(_safe_sam3_define_schema)
                    try:
                        sd_cls.GET_SCHEMA()
                    except Exception:
                        pass

                if orig_sd_exec is not None:
                    setattr(sd_cls, "_vstone_orig_execute", orig_sd_exec)

                    def _safe_sam3_detect_execute(cls, model, image, conditioning=None, bboxes=None, positive_coords=None, negative_coords=None, threshold=0.5, refine_iterations=2, individual_masks=False, _orig=orig_sd_exec):
                        import comfy_extras.nodes_sam3 as _ns3
                        B, H, W, C = image.shape

                        def _norm_box_list(raw_b):
                            if raw_b is None:
                                return []
                            if isinstance(raw_b, str):
                                try:
                                    raw_b = json.loads(raw_b)
                                except Exception:
                                    return []
                            if isinstance(raw_b, dict):
                                raw_b = [raw_b]
                            if not isinstance(raw_b, (list, tuple)):
                                return []
                            out = []
                            for item in raw_b:
                                if isinstance(item, dict):
                                    if "x" in item and "y" in item and "width" in item and "height" in item:
                                        if float(item["width"]) > 0 and float(item["height"]) > 0:
                                            out.append({
                                                "x": float(item["x"]),
                                                "y": float(item["y"]),
                                                "width": float(item["width"]),
                                                "height": float(item["height"]),
                                            })
                                    elif all(item.get(k) is not None for k in ("startX", "startY", "endX", "endY")):
                                        x1 = min(float(item["startX"]), float(item["endX"]))
                                        y1 = min(float(item["startY"]), float(item["endY"]))
                                        x2 = max(float(item["startX"]), float(item["endX"]))
                                        y2 = max(float(item["startY"]), float(item["endY"]))
                                        if x2 > x1 and y2 > y1:
                                            out.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})
                                elif isinstance(item, (list, tuple)) and len(item) == 4:
                                    pe_boxes = getattr(sys, "_vstone_last_points_editor_boxes", None)
                                    if isinstance(pe_boxes, list) and len(pe_boxes) == len(raw_b):
                                        return list(pe_boxes)
                                    x1, y1, v3, v4 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                                    if v3 > x1 and v4 > y1:
                                        out.append({"x": x1, "y": y1, "width": v3 - x1, "height": v4 - y1})
                                    elif v3 > 0 and v4 > 0:
                                        out.append({"x": x1, "y": y1, "width": v3, "height": v4})
                            return out

                        per_frame_boxes = None
                        if isinstance(bboxes, list) and len(bboxes) > 0 and isinstance(bboxes[0], list) and (len(bboxes[0]) == 0 or isinstance(bboxes[0][0], dict)):
                            norm_pf = [_norm_box_list(fb) for fb in bboxes]
                            if any(len(fb) > 0 for fb in norm_pf):
                                per_frame_boxes = norm_pf
                                while len(per_frame_boxes) < B:
                                    per_frame_boxes.append(per_frame_boxes[-1])
                        else:
                            flat_boxes = _norm_box_list(bboxes)
                            if not flat_boxes:
                                pe_fallback = getattr(sys, "_vstone_last_points_editor_boxes", None)
                                # Chỉ khi các điểm đến từ chính PointsEditor vừa chạy; điểm từ nơi khác (trang concept, workflow khác)
                                # không được nhận khung cũ của PointsEditor, nếu không vùng chọn bị khoá vào khung của ảnh khác.
                                pe_coords = getattr(sys, "_vstone_last_points_editor_coords", None)
                                from_points_editor = pe_coords is not None and (positive_coords is not None or negative_coords is not None) \
                                    and (positive_coords is None or positive_coords == pe_coords[0]) \
                                    and (negative_coords is None or negative_coords == pe_coords[1])
                                if isinstance(pe_fallback, list) and len(pe_fallback) > 0 and from_points_editor:
                                    flat_boxes = list(pe_fallback)
                            if flat_boxes:
                                per_frame_boxes = [flat_boxes] * B

                        try:
                            pos_pts = json.loads(positive_coords) if positive_coords else []
                        except Exception:
                            pos_pts = []
                        try:
                            neg_pts = json.loads(negative_coords) if negative_coords else []
                        except Exception:
                            neg_pts = []
                        has_points = len(pos_pts) > 0 or len(neg_pts) > 0
                        has_cond = conditioning is not None and len(conditioning) > 0

                        if not has_points and per_frame_boxes is None and not has_cond:
                            idev = comfy.model_management.intermediate_device()
                            empty_m = torch.zeros((0, H, W) if individual_masks else (B, H, W), device=idev)
                            return _comfy_io.NodeOutput(empty_m, [[] for _ in range(B)])

                        def _boxes_to_tensor(box_list):
                            coords = []
                            for d in box_list:
                                cx = (d["x"] + d["width"] / 2.0) / W
                                cy = (d["y"] + d["height"] / 2.0) / H
                                coords.append([cx, cy, d["width"] / W, d["height"] / H])
                            return torch.tensor([coords], dtype=torch.float32)

                        pf_tensors = None
                        if per_frame_boxes is not None:
                            pf_tensors = [_boxes_to_tensor(fb) if fb else None for fb in per_frame_boxes]

                        image_in = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
                        comfy.model_management.load_model_gpu(model)
                        device = comfy.model_management.get_torch_device()
                        dtype = model.model.get_dtype()
                        sam3_model = model.model.diffusion_model

                        point_inputs = None
                        if has_points:
                            all_coords = [[p["x"] / W * 1008.0, p["y"] / H * 1008.0] for p in pos_pts] + \
                                         [[p["x"] / W * 1008.0, p["y"] / H * 1008.0] for p in neg_pts]
                            all_labels = [1] * len(pos_pts) + [0] * len(neg_pts)
                            point_inputs = {
                                "point_coords": torch.tensor([all_coords], dtype=dtype, device=device),
                                "point_labels": torch.tensor([all_labels], dtype=torch.int32, device=device),
                            }

                        cond_list = _ns3._extract_text_prompts(conditioning, device, dtype) if has_cond else []
                        has_text = len(cond_list) > 0

                        all_bbox_dicts = []
                        all_masks = []
                        pbar = comfy.utils.ProgressBar(B)

                        for b in range(B):
                            frame = image_in[b:b+1].to(device=device, dtype=dtype)
                            b_boxes = None
                            if pf_tensors is not None and pf_tensors[b] is not None:
                                b_boxes = pf_tensors[b].to(device=device, dtype=dtype)

                            frame_bbox_dicts = []
                            frame_masks = []

                            if not has_text and b_boxes is not None and point_inputs is not None:
                                for box_cxcywh in b_boxes[0]:
                                    cx, cy, bw, bh = box_cxcywh.tolist()
                                    sam_box = torch.tensor([[[(cx - bw / 2.0) * 1008.0, (cy - bh / 2.0) * 1008.0],
                                                             [(cx + bw / 2.0) * 1008.0, (cy + bh / 2.0) * 1008.0]]],
                                                           device=device, dtype=dtype)
                                    mask_logit = sam3_model.forward_segment(frame, point_inputs=point_inputs, box_inputs=sam_box)
                                    for _ in range(max(0, refine_iterations - 1)):
                                        mask_logit = sam3_model.forward_segment(frame, point_inputs=point_inputs, box_inputs=sam_box, mask_inputs=mask_logit)
                                    mask = _F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
                                    frame_masks.append((mask[0] > 0).float())
                            else:
                                if point_inputs is not None:
                                    mask_logit = sam3_model.forward_segment(frame, point_inputs=point_inputs)
                                    for _ in range(max(0, refine_iterations - 1)):
                                        mask_logit = sam3_model.forward_segment(frame, point_inputs=point_inputs, mask_inputs=mask_logit)
                                    mask = _F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
                                    frame_masks.append((mask[0] > 0).float())

                                if b_boxes is not None and not has_text:
                                    for box_cxcywh in b_boxes[0]:
                                        cx, cy, bw, bh = box_cxcywh.tolist()
                                        sam_box = torch.tensor([[[(cx - bw / 2.0) * 1008.0, (cy - bh / 2.0) * 1008.0],
                                                                 [(cx + bw / 2.0) * 1008.0, (cy + bh / 2.0) * 1008.0]]],
                                                               device=device, dtype=dtype)
                                        mask_logit = sam3_model.forward_segment(frame, box_inputs=sam_box)
                                        for _ in range(max(0, refine_iterations - 1)):
                                            mask_logit = sam3_model.forward_segment(frame, box_inputs=sam_box, mask_inputs=mask_logit)
                                        mask = _F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
                                        frame_masks.append((mask[0] > 0).float())

                            for text_embeddings, text_mask, max_det in cond_list:
                                results = sam3_model(
                                    frame, text_embeddings=text_embeddings, text_mask=text_mask,
                                    boxes=b_boxes, threshold=threshold, orig_size=(H, W)
                                )
                                pred_boxes = results["boxes"][0]
                                scores = results["scores"][0]
                                masks = results["masks"][0]

                                probs = scores.sigmoid()
                                keep = probs > threshold
                                kept_boxes = pred_boxes[keep].cpu()
                                kept_scores = probs[keep].cpu()
                                kept_masks = masks[keep]

                                order = kept_scores.argsort(descending=True)[:max_det]
                                kept_boxes = kept_boxes[order]
                                kept_scores = kept_scores[order]
                                kept_masks = kept_masks[order]

                                for box, score in zip(kept_boxes, kept_scores):
                                    frame_bbox_dicts.append({
                                        "x": float(box[0]), "y": float(box[1]),
                                        "width": float(box[2] - box[0]), "height": float(box[3] - box[1]),
                                        "score": float(score),
                                    })
                                for m, box in zip(kept_masks, kept_boxes):
                                    frame_masks.append(_ns3._refine_mask(
                                        sam3_model, image[b], m, box, H, W, device, dtype, refine_iterations
                                    ))

                            all_bbox_dicts.append(frame_bbox_dicts)
                            if len(frame_masks) > 0:
                                combined = torch.cat(frame_masks, dim=0)
                                if individual_masks:
                                    all_masks.append(combined)
                                else:
                                    all_masks.append((combined > 0).any(dim=0).float())
                            else:
                                if individual_masks:
                                    all_masks.append(torch.zeros(0, H, W, device=comfy.model_management.intermediate_device()))
                                else:
                                    all_masks.append(torch.zeros(H, W, device=comfy.model_management.intermediate_device()))
                            pbar.update(1)

                        idev = comfy.model_management.intermediate_device()
                        all_masks = [m.to(idev) for m in all_masks]
                        mask_out = torch.cat(all_masks, dim=0) if individual_masks else torch.stack(all_masks)
                        if mask_out.numel() == 0:
                            mask_out = torch.zeros(1, H, W, device=idev)
                        return _comfy_io.NodeOutput(mask_out, all_bbox_dicts)

                    sd_cls.execute = classmethod(_safe_sam3_detect_execute)
                setattr(sd_cls, "_vstone_sam3_hook_ver", VSTONE_RUNTIME_PATCH_VER)

        for vt_cls in _get_target_node_classes("SAM3_VideoTrack", "comfy_extras.nodes_sam3"):
            if getattr(vt_cls, "_vstone_sam3_vt_hook_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
                orig_vt_exec = _unwrap_to_true_orig(vt_cls, "execute")
                if orig_vt_exec is not None:
                    setattr(vt_cls, "_vstone_orig_execute", orig_vt_exec)

                    def _safe_sam3_videotrack_execute(cls, images, model, initial_mask=None, conditioning=None, detection_threshold=0.5, max_objects=0, detect_interval=1, _orig=orig_vt_exec):
                        has_cond = conditioning is not None and len(conditioning) > 0
                        has_mask = initial_mask is not None and hasattr(initial_mask, "numel") and initial_mask.numel() > 0 and bool((initial_mask > 0).any())
                        if not has_cond and not has_mask:
                            N, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
                            return _comfy_io.NodeOutput({"packed_masks": None, "n_frames": N, "orig_size": (H, W), "scores": []})
                        raw_fn = _orig.__func__ if isinstance(_orig, classmethod) else _orig
                        return raw_fn(cls, images, model, initial_mask=initial_mask, conditioning=conditioning, detection_threshold=detection_threshold, max_objects=max_objects, detect_interval=detect_interval)

                    vt_cls.execute = classmethod(_safe_sam3_videotrack_execute)
                setattr(vt_cls, "_vstone_sam3_vt_hook_ver", VSTONE_RUNTIME_PATCH_VER)
    except Exception as e:
        print(f"[VStone Runtime] Warning hooking SAM3: {e}")

    # 4. Auto-install & hot-load Portable Workflow Pack custom nodes in background (once per patch version)
    try:
        if getattr(sys, "_vstone_portable_pack_bg_ver", 0) < VSTONE_RUNTIME_PATCH_VER:
            if not any("_bg_portable_pack_installer" in t.name for t in threading.enumerate()):
                setattr(sys, "_vstone_portable_pack_bg_ver", VSTONE_RUNTIME_PATCH_VER)

                def _bg_install_packs():
                    try:
                        res_msg = install_portable_workflow_pack(unique_id=None, allow_restart=False)
                        print(f"[VStone Auto-Pack] {res_msg.splitlines()[0] if res_msg else 'Done'}", flush=True)
                    except Exception as _bg_err:
                        print(f"[VStone Auto-Pack] Warning: {_bg_err}", flush=True)

                threading.Thread(target=_bg_install_packs, name="_bg_portable_pack_installer", daemon=True).start()
                all_notes_bg = "Started background update & hot-load for 28 Custom Node Packs"
            else:
                all_notes_bg = ""
        else:
            all_notes_bg = ""
    except Exception:
        all_notes_bg = ""

    # 5. Check and recover ComfyUI prompt_worker & queue
    q_notes = _recover_comfy_queue_and_worker()
    all_notes = []
    if restored:
        all_notes.append(f"Restored native nodes: {', '.join(restored)}")
    if all_notes_bg:
        all_notes.append(all_notes_bg)
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
    """Download model/LoRA safely using a lock, .safetensors integrity verification, and temporary .part file."""
    comfy_root = _get_comfy_root()
    if not comfy_root:
        return "[VStone Downloader] ComfyUI root not found."

    if "huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/" in url:
        url = url.replace(
            "huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/",
            "huggingface.co/Comfy-Org/Wan-2.1/",
        )

    target_dir = os.path.join(comfy_root, "models", folder_type)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    if os.path.exists(target_path):
        if _is_valid_safetensors_file(target_path):
            return f"[VStone Downloader] Exists & verified: {target_path} ({os.path.getsize(target_path) // (1024*1024)} MB)"
        print(f"[VStone Downloader] Corrupted/incomplete file found at {target_path}, removing to re-download...")
        try:
            os.remove(target_path)
        except Exception:
            pass

    with _DOWNLOAD_LOCK:
        if os.path.exists(target_path):
            if _is_valid_safetensors_file(target_path):
                return f"[VStone Downloader] Exists & verified: {target_path} ({os.path.getsize(target_path) // (1024*1024)} MB)"
            try:
                os.remove(target_path)
            except Exception:
                pass

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
                if _is_valid_safetensors_file(target_path):
                    _clear_comfy_model_cache()
                    return f"[VStone Downloader] Downloaded & verified via aria2c: {target_path}"
                print(f"[VStone Downloader] aria2c file failed safetensors verification, retrying via curl: {target_path}")
                try:
                    os.remove(target_path)
                except Exception:
                    pass

        curl_bin = shutil.which("curl")
        if curl_bin:
            cmd = [curl_bin, "-4", "-L", "-C", "-", "--max-time", "3600", url, "-o", tmp_path]
            res = subprocess.run(cmd, check=False, timeout=3610)
            if res.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1024:
                os.replace(tmp_path, target_path)
                if _is_valid_safetensors_file(target_path):
                    _clear_comfy_model_cache()
                    return f"[VStone Downloader] Downloaded & verified via curl: {target_path}"
                try:
                    os.remove(target_path)
                except Exception:
                    pass

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
LTX25_RESTARTED_ENV = "VSTONE_LTX25_PACK_RESTARTED"


def _missing_node_ids(node_ids: list[str]) -> list[str]:
    """Chỉ tính node của pack gốc. Bản chép cũ trong helper còn nằm trong RAM cho tới lần khởi động lại sau."""
    import nodes
    missing: list[str] = []
    for node_id in node_ids:
        node_cls = nodes.NODE_CLASS_MAPPINGS.get(node_id)
        origin = f"{getattr(node_cls, '__module__', '')} {getattr(node_cls, 'RELATIVE_PYTHON_MODULE', '')}".lower()
        if node_cls is None or "vstone" in origin:
            missing.append(node_id)
    return missing


def _run_install_cmd(cmd: list[str], timeout: int) -> tuple[bool, str]:
    """Chạy lệnh cài đặt, trả về (thành công, phần cuối của log) thay vì ném lỗi."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    return res.returncode == 0, (res.stderr or res.stdout or "").strip()[-300:]


def _module_missing(name: str) -> bool:
    """find_spec ném ValueError khi module đã nạp mà không có __spec__ (pack khác đặt module thay thế vào sys.modules). Đã nạp thì coi là có."""
    if name in sys.modules:
        return False
    importlib.invalidate_caches()
    return importlib.util.find_spec(name) is None


def _fetch_node_pack(pack: dict[str, Any], pack_dir: str) -> tuple[bool, str]:
    """Clone hoặc pull/reset pack về commit mới nhất. Nếu git lỗi hoặc thư mục không phải repo git thì tải zip từ GitHub."""
    git_bin = shutil.which("git")
    branch = pack.get("branch", "main")
    if git_bin and os.path.isdir(os.path.join(pack_dir, ".git")):
        ok, tail = _run_install_cmd([git_bin, "-C", pack_dir, "pull", "--ff-only"], 300)
        if ok:
            return True, tail
        for br in dict.fromkeys([branch, "main", "master"]):
            f_ok, _ = _run_install_cmd([git_bin, "-C", pack_dir, "fetch", "--depth", "1", "origin", br], 300)
            if f_ok:
                r_ok, r_tail = _run_install_cmd([git_bin, "-C", pack_dir, "reset", "--hard", "FETCH_HEAD"], 120)
                if r_ok:
                    return True, r_tail
    if git_bin and not os.path.exists(pack_dir):
        clone_url = f"https://github.com/{pack['repo']}.git"
        ok, tail = _run_install_cmd([git_bin, "clone", "--depth", "1", "-b", branch, clone_url, pack_dir], 900)
        if not ok:
            ok, tail = _run_install_cmd([git_bin, "clone", "--depth", "1", clone_url, pack_dir], 900)
        if ok:
            return True, tail
    last_err = ""
    for br in dict.fromkeys([branch, "main", "master"]):
        zip_url = f"https://github.com/{pack['repo']}/archive/refs/heads/{br}.zip"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(io.BytesIO(_http_get_bytes(zip_url, timeout=600))) as archive:
                    archive.extractall(tmp)
                shutil.copytree(os.path.join(tmp, os.listdir(tmp)[0]), pack_dir, dirs_exist_ok=True)
            return True, ""
        except Exception as e:
            last_err = str(e)
    return False, last_err


_SKIP_PIP_PREFIXES = (
    "torch",
    "torchvision",
    "torchaudio",
    "nvidia-vfx",
    "amd-adlx",
    "groundingdino-py",
    "llama-cpp-python",
    "pytest",
    "git+https://github.com/facebookresearch/sam2",
)


def _should_skip_pip_spec(spec: str) -> bool:
    s_low = spec.lower().strip()
    if not s_low or s_low.startswith("-"):
        return True
    if "platform_system == \"windows\"" in s_low and os.name != "nt":
        return True
    for pref in _SKIP_PIP_PREFIXES:
        if s_low == pref or s_low.startswith(pref + ">") or s_low.startswith(pref + "<") or s_low.startswith(pref + "=") or s_low.startswith(pref + "[") or s_low.startswith(pref + ";") or s_low.startswith(pref + "-"):
            if pref == "torch" and s_low.startswith("torch-") and not s_low.startswith("torch=="):
                continue
            return True
    return False


def _pip_install_requirements(pack_dir: str) -> list[str]:
    """Cài requirements.txt an toàn (bỏ qua các gói ghi đè PyTorch hoặc gói lỗi C++ build gây treo)."""
    req_path = os.path.join(pack_dir, "requirements.txt")
    if not os.path.isfile(req_path):
        return []
    pip_install = [sys.executable, "-m", "pip", "install", "--prefer-binary"]
    specs: list[str] = []
    try:
        with open(req_path, encoding="utf-8", errors="ignore") as req_file:
            for line in req_file:
                spec = line.split("#")[0].strip()
                if spec and not _should_skip_pip_spec(spec):
                    specs.append(spec)
    except Exception:
        return []
    if not specs:
        return []
    if _run_install_cmd(pip_install + specs, 600)[0]:
        return []
    failed: list[str] = []
    for spec in specs:
        ok, tail = _run_install_cmd(pip_install + [spec], 240)
        if not ok:
            failed.append(f"{spec}: {tail[:120]}")
    return failed


def _hot_load_custom_node_pack(pack_dir: str, module_parent: str = "custom_nodes") -> bool:
    """Nạp nóng (Hot-load) một thư mục hoặc file custom_node vào RAM của ComfyUI (hỗ trợ cả V1 NODE_CLASS_MAPPINGS và V3 comfy_entrypoint)."""
    if not pack_dir or not (os.path.isfile(pack_dir) or os.path.isfile(os.path.join(pack_dir, "__init__.py"))):
        return False
    try:
        import asyncio
        import inspect
        import nodes
        abs_target = os.path.abspath(pack_dir)
        sys_module_name = os.path.splitext(pack_dir)[0] if os.path.isfile(pack_dir) else pack_dir.replace(".", "_x_")
        for mod_key in list(sys.modules.keys()):
            mod_obj = sys.modules.get(mod_key)
            mod_file = getattr(mod_obj, "__file__", "") or ""
            if mod_key == sys_module_name or mod_key.startswith(sys_module_name + ".") or (mod_file and os.path.abspath(mod_file).startswith(abs_target)):
                sys.modules.pop(mod_key, None)
        importlib.invalidate_caches()
        res = nodes.load_custom_node(pack_dir, module_parent=module_parent)
        if inspect.isawaitable(res):
            loop = asyncio.new_event_loop()
            try:
                return bool(loop.run_until_complete(res))
            finally:
                loop.close()
        return bool(res)
    except Exception as e:
        print(f"[VStone] Hot-load pack {os.path.basename(pack_dir)} warning: {e}", flush=True)
        return False


def _stop_download_processes() -> None:
    """aria2c/curl sống sót qua os.execv và sẽ ghi chồng file .part với lượt tải sau khởi động lại. Cả hai đều tải tiếp được."""
    try:
        import psutil
        for child in psutil.Process().children(recursive=True):
            if child.name() in ("aria2c", "curl"):
                child.terminate()
    except Exception as e:
        print(f"[VStone] Không dừng được tiến trình tải: {e}", flush=True)


def _restart_comfyui_after(delay_sec: float) -> None:
    """Khởi động lại ComfyUI với đúng tham số đang chạy (giữ nguyên port), cùng cách ComfyUI-Manager dùng."""
    def _restart() -> None:
        time.sleep(delay_sec)
        print("[VStone] Khởi động lại ComfyUI để nạp pack vừa cài...", flush=True)
        _stop_download_processes()
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
    """Cài pack rồi khởi động lại ngay để hết báo thiếu node, lần Queue sau mới tải model. Dòng đầu của báo cáo là kết luận."""
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

    missing_ids = _missing_node_ids([node_id for pack in LTX25_CUSTOM_NODE_PACKS for node_id in pack["node_ids"]])
    if missing_ids:
        if not fetch_failed and os.environ.get(LTX25_RESTARTED_ENV) != "1":
            os.environ[LTX25_RESTARTED_ENV] = "1"
            _restart_comfyui_after(5.0)
            summary = "🔄 Đã cài pack, ComfyUI đang khởi động lại. Chờ 1-2 phút, F5, rồi bấm Queue lần nữa để tải model."
            return "\n".join([summary] + details)
        details.append("Thiếu node: " + ", ".join(missing_ids) + ". Xem log khởi động, dòng IMPORT FAILED.")
        return "\n".join(["❌ Chưa cài xong pack LTX-2.5."] + details)

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
        return "\n".join(["❌ Pack đã đủ, còn thiếu model."] + details)
    return "\n".join(["✅ LTX-2.5 đã đủ pack và model. Mở workflow ltx_2.5/LTX25_Outpaint_Official."] + details)


# ------------------------------------------------------------------------------
# Cài pack AusBoss cho workflow outpaint Qwen của tác giả LoRA, model đi theo sync_manifest.json
# ------------------------------------------------------------------------------
AUSBOSS_PACK: dict[str, Any] = {
    "dir": "ComfyUI-AusBoss",
    "repo": "ausboss/ComfyUI-AusBoss",
    "branch": "main",
    "node_ids": [
        "AUSBOSS_NODES_Compare",
        "AUSBOSS_NODES_CropForInpaint",
        "AUSBOSS_NODES_ImageCropRotatePad",
        "AUSBOSS_NODES_LoraLoader",
        "AUSBOSS_NODES_RunTimer",
        "AUSBOSS_NODES_SaveImage",
        "AUSBOSS_NODES_Seed",
        "AUSBOSS_NODES_ShowText",
        "AUSBOSS_NODES_StitchInpaint",
        "AUSBOSS_NODES_Text",
        "AUSBOSS_NODES_WorkflowNote",
        # Workflow Tiled Upscale của tác giả cần pack 2.15.0 trở lên, pack cũ hơn thì thiếu hai node này và sẽ được pull.
        "AUSBOSS_NODES_TiledUpscale",
        "AUSBOSS_NODES_TiledUpscaleStitch",
    ],
}
AUSBOSS_RESTARTED_ENV = "VSTONE_AUSBOSS_PACK_RESTARTED"
AUSBOSS_WORKFLOW = "qwen_image_2.1/qwen_image_2_1_outpaint_ausboss"


def install_ausboss_outpaint(unique_id: Any = None) -> str:
    """Cài pack rồi khởi động lại. Lần Queue sau xác nhận node đã nạp. Dòng đầu của báo cáo là kết luận."""
    root = _get_comfy_root()
    if not root:
        return "❌ Không tìm thấy thư mục ComfyUI."
    missing = _missing_node_ids(AUSBOSS_PACK["node_ids"])
    if not missing:
        return f"✅ Pack AusBoss đã đủ node. Mở workflow {AUSBOSS_WORKFLOW}."
    # os.execv giữ biến môi trường, nên nhớ các node đã thử cài; node mới thêm vào danh sách vẫn được cài.
    tried = set(filter(None, os.environ.get(AUSBOSS_RESTARTED_ENV, "").split(",")))
    if set(missing) <= tried:
        return "\n".join([
            "❌ Đã khởi động lại nhưng vẫn thiếu node AusBoss. Xem log khởi động, dòng IMPORT FAILED.",
            "Thiếu node: " + ", ".join(missing),
        ])

    custom_nodes = os.path.join(root, "custom_nodes")
    os.makedirs(custom_nodes, exist_ok=True)
    installed_dirs = {name.lower(): name for name in os.listdir(custom_nodes)}
    pack_dir = os.path.join(custom_nodes, installed_dirs.get(AUSBOSS_PACK["dir"].lower(), AUSBOSS_PACK["dir"]))

    _vstone_ui_status(unique_id, f"Đang cài pack {AUSBOSS_PACK['dir']}...")
    ok, tail = _fetch_node_pack(AUSBOSS_PACK, pack_dir)
    if not ok or not os.path.isfile(os.path.join(pack_dir, "__init__.py")):
        return f"❌ Không tải được pack AusBoss, chưa khởi động lại. {tail}"

    os.environ[AUSBOSS_RESTARTED_ENV] = ",".join(sorted(tried | set(missing)))
    _restart_comfyui_after(5.0)
    return "🔄 Đã cài pack AusBoss, ComfyUI đang khởi động lại. Chờ 1-2 phút, F5, rồi bấm Queue lần nữa để kiểm tra."


# ------------------------------------------------------------------------------
# Cài pack AnyAngle Studio T8 để xoay vật thể, model đi theo sync_manifest.json
# ------------------------------------------------------------------------------
ANYANGLE_PACK: dict[str, Any] = {
    "dir": "Comfyui-Qwen-Image-2.1-MultiAngle-T8",
    "repo": "T8mars/Comfyui-Qwen-Image-2.1-MultiAngle-T8",
    "branch": "main",
    "node_ids": ["AnyAngleStudioT8", "AnyAngleOptionalLoRAT8", "AnyAngleReconstructionOutputT8"],
}
# requirements.txt của pack. Chỉ cài gói còn thiếu, để không nâng cấp thư viện các pack khác đang dùng.
ANYANGLE_PIP: dict[str, str] = {
    "onnxruntime": "onnxruntime>=1.18,<2",
    "cv2": "opencv-python-headless>=4.8,<5",
    "huggingface_hub": "huggingface_hub>=0.26",
}
ANYANGLE_MODELS: list[tuple[str, str]] = [
    ("diffusion_models", "triposplat_fp16.safetensors"),
    ("vae", "triposplat_vae_decoder_fp16.safetensors"),
    ("vae", "flux2-vae.safetensors"),
    ("clip_vision", "dino_v3_vit_h.safetensors"),
    ("background_removal", "birefnet.safetensors"),
    ("loras", "QI2.1_AnyAngle.safetensors"),
]
ANYANGLE_RESTARTED_ENV = "VSTONE_ANYANGLE_PACK_RESTARTED"
ANYANGLE_WORKFLOW = "anyangle_studio/AnyAngle-Studio-Qwen21"


def install_anyangle_studio(unique_id: Any = None) -> str:
    """Cài pack rồi khởi động lại. Lần Queue sau xác nhận node và model. Dòng đầu của báo cáo là kết luận."""
    root = _get_comfy_root()
    if not root:
        return "❌ Không tìm thấy thư mục ComfyUI."
    missing = _missing_node_ids(ANYANGLE_PACK["node_ids"])
    if not missing:
        waiting = [name for folder, name in ANYANGLE_MODELS if not os.path.isfile(os.path.join(root, "models", folder, name))]
        if waiting:
            return "\n".join([
                "⏳ Pack AnyAngle Studio đã đủ node, model còn đang tải ngầm. Bấm Queue lần nữa để kiểm tra lại.",
                "Chưa có: " + ", ".join(waiting),
            ])
        return f"✅ AnyAngle Studio đã đủ node và model. Mở workflow {ANYANGLE_WORKFLOW}."
    if os.environ.get(ANYANGLE_RESTARTED_ENV) == "1":
        return "\n".join([
            "❌ Đã khởi động lại nhưng vẫn thiếu node AnyAngle Studio. Xem log khởi động, dòng IMPORT FAILED.",
            "Thiếu node: " + ", ".join(missing),
        ])

    custom_nodes = os.path.join(root, "custom_nodes")
    os.makedirs(custom_nodes, exist_ok=True)
    installed_dirs = {name.lower(): name for name in os.listdir(custom_nodes)}
    pack_dir = os.path.join(custom_nodes, installed_dirs.get(ANYANGLE_PACK["dir"].lower(), ANYANGLE_PACK["dir"]))

    _vstone_ui_status(unique_id, f"Đang cài pack {ANYANGLE_PACK['dir']}, khoảng 100 MB...")
    ok, tail = _fetch_node_pack(ANYANGLE_PACK, pack_dir)
    if not ok or not os.path.isfile(os.path.join(pack_dir, "__init__.py")):
        return f"❌ Không tải được pack AnyAngle Studio, chưa khởi động lại. {tail}"

    details: list[str] = []
    for module, spec in ANYANGLE_PIP.items():
        if _module_missing(module):
            _vstone_ui_status(unique_id, f"Đang cài thư viện {spec}...")
            pip_ok, pip_tail = _run_install_cmd([sys.executable, "-m", "pip", "install", spec], 900)
            details.append(f"Thư viện {spec}: " + ("đã cài." if pip_ok else f"LỖI {pip_tail}"))

    os.environ[ANYANGLE_RESTARTED_ENV] = "1"
    _restart_comfyui_after(5.0)
    return "\n".join(["🔄 Đã cài pack AnyAngle Studio, ComfyUI đang khởi động lại. Chờ 1-2 phút, F5, rồi bấm Queue lần nữa để kiểm tra."] + details)


# ------------------------------------------------------------------------------
# Cài pack cho 3 workflow BFS đổi mặt / đổi nhân vật của tác giả, model đi theo sync_manifest.json
# ------------------------------------------------------------------------------
# "pip" là requirements.txt của từng pack (module -> gói pip). Chỉ cài gói còn thiếu,
# để không nâng cấp thư viện các pack khác đang dùng.
BFS_SWAP_PACKS: list[dict[str, Any]] = [
    {
        "dir": "ComfyUI-KJNodes",
        "repo": "kijai/ComfyUI-KJNodes",
        "branch": "main",
        "node_ids": ["ImageResizeKJv2", "ImageConcatMulti", "AddLabel"],
        "pip": {"color_matcher": "color-matcher", "matplotlib": "matplotlib", "mss": "mss", "cv2": "opencv-python-headless"},
    },
    {
        "dir": "rgthree-comfy",
        "repo": "rgthree/rgthree-comfy",
        "branch": "main",
        "node_ids": ["Image Comparer (rgthree)"],
        "pip": {},
    },
    # Workflow đổi mặt và thay nhân vật dùng sampler deis_2m, pack này thêm nó vào KSampler.
    {
        "dir": "RES4LYF",
        "repo": "ClownsharkBatwing/RES4LYF",
        "branch": "main",
        "node_ids": ["ClownsharKSampler_Beta"],
        "pip": {"cv2": "opencv-python-headless", "matplotlib": "matplotlib", "pywt": "PyWavelets"},
    },
    {
        "dir": "ComfyUI-BFSNodes",
        "repo": "alisson-anjos/ComfyUI-BFSNodes",
        "branch": "main",
        "node_ids": ["BFSSegmentPicker", "BFSVideoCompare"],
        "pip": {
            "librosa": "librosa",
            "cv2": "opencv-python-headless",
            "scenedetect": "scenedetect",
            "transformers": "transformers>=4.55.0",
            "scipy": "scipy",
        },
    },
]
BFS_SWAP_MODELS: list[tuple[str, str]] = [
    ("loras", "bfs_head_v1.1_qwen_2.1.safetensors"),
    ("loras/Downloads", "bfs_body_swap_v1.0_qwen_2.1.safetensors"),
    ("loras", "bfs_body_swap_selective_magenta_v1.0_qwen_2.1.safetensors"),
    ("loras", "p_qwen_image_2.1_8step_v0.1.safetensors"),
    ("loras/Downloads", "p_qwen_image_2.1_8step_v0.1.safetensors"),
    ("checkpoints", "sam3.1_multiplex_fp16.safetensors"),
]
BFS_SWAP_RESTARTED_ENV = "VSTONE_BFS_SWAP_PACKS_RESTARTED"
BFS_SWAP_WORKFLOW_DIR = "bfs_swap"


def install_bfs_swap_packs(unique_id: Any = None) -> str:
    """Cài các pack còn thiếu rồi khởi động lại một lần. Lần Queue sau xác nhận node và model. Dòng đầu của báo cáo là kết luận."""
    root = _get_comfy_root()
    if not root:
        return "❌ Không tìm thấy thư mục ComfyUI."
    missing = _missing_node_ids([node_id for pack in BFS_SWAP_PACKS for node_id in pack["node_ids"]])
    if not missing:
        waiting = [f"{folder}/{name}" for folder, name in BFS_SWAP_MODELS if not os.path.isfile(os.path.join(root, "models", folder, name))]
        if waiting:
            return "\n".join([
                "⏳ Đã đủ node cho BFS, model còn đang tải ngầm. Bấm Queue lần nữa để kiểm tra lại.",
                "Chưa có: " + ", ".join(waiting),
            ])
        return f"✅ BFS đã đủ node và model. Mở các workflow trong thư mục {BFS_SWAP_WORKFLOW_DIR}."
    # Biến môi trường còn nguyên sau os.execv, nên nhớ đã thử cài cho node nào. Pack mới thêm vào danh sách vẫn được cài.
    tried = set(filter(None, os.environ.get(BFS_SWAP_RESTARTED_ENV, "").split(",")))
    if set(missing) <= tried:
        return "\n".join([
            "❌ Đã khởi động lại nhưng vẫn thiếu node cho BFS. Xem log khởi động, dòng IMPORT FAILED.",
            "Thiếu node: " + ", ".join(missing),
        ])

    custom_nodes = os.path.join(root, "custom_nodes")
    os.makedirs(custom_nodes, exist_ok=True)
    installed_dirs = {name.lower(): name for name in os.listdir(custom_nodes)}
    details: list[str] = []
    for pack in BFS_SWAP_PACKS:
        if not _missing_node_ids(pack["node_ids"]):
            details.append(f"Pack {pack['dir']}: đã đủ node.")
            continue
        pack_dir = os.path.join(custom_nodes, installed_dirs.get(pack["dir"].lower(), pack["dir"]))
        _vstone_ui_status(unique_id, f"Đang cài pack {pack['dir']}...")
        ok, tail = _fetch_node_pack(pack, pack_dir)
        if not ok or not os.path.isfile(os.path.join(pack_dir, "__init__.py")):
            return "\n".join([f"❌ Không tải được pack {pack['dir']}, chưa khởi động lại. {tail}"] + details)
        details.append(f"Pack {pack['dir']}: đã tải về.")
        for module, spec in pack["pip"].items():
            if _module_missing(module):
                _vstone_ui_status(unique_id, f"Đang cài thư viện {spec}...")
                pip_ok, pip_tail = _run_install_cmd([sys.executable, "-m", "pip", "install", spec], 900)
                details.append(f"Thư viện {spec}: " + ("đã cài." if pip_ok else f"LỖI {pip_tail}"))

    os.environ[BFS_SWAP_RESTARTED_ENV] = ",".join(sorted(tried | set(missing)))
    _restart_comfyui_after(5.0)
    return "\n".join(["🔄 Đã cài pack cho BFS, ComfyUI đang khởi động lại. Chờ 1-2 phút, F5, rồi bấm Queue lần nữa để kiểm tra."] + details)


# ------------------------------------------------------------------------------
# Cài trọn bộ 28 Custom Node Packs cho Portable Workflow Pack (47 Workflows: MiniMax-H3 VFX Edit, HeadSwap/CharSwap/StyleTransfer/LMS/Live, Netflix VOID, LTX 2.5, Qwen 2.1, VOSR2/SeedVR2)
# ------------------------------------------------------------------------------
PORTABLE_PACK_CUSTOM_NODES: list[dict[str, Any]] = [
    {
        "dir": "ComfyUI-VideoHelperSuite",
        "repo": "Kosinkadink/ComfyUI-VideoHelperSuite",
        "branch": "main",
        "node_ids": ["VHS_VideoCombine", "VHS_LoadVideo", "VHS_LoadVideoFFmpeg", "VHS_VideoInfo", "VHS_VideoInfoLoaded"],
        "pip": {"cv2": "opencv-python-headless", "imageio_ffmpeg": "imageio-ffmpeg"},
    },
    {
        "dir": "ComfyUI-KJNodes",
        "repo": "kijai/ComfyUI-KJNodes",
        "branch": "main",
        "node_ids": [
            "ImageResizeKJv2",
            "ImageConcatMulti",
            "AddLabel",
            "PointsEditor",
            "MiniMaxChunkFeedForward",
            "MiniMaxLowVRAMAttention",
            "ModelPreviewOverrideKJ",
            "MiniMaxH3MemoryEfficientSageAttentionPatch",
            "LTX2SamplingPreviewOverride",
            "LTX2_NAG",
            "GrowMaskWithBlur",
            "SomethingToString",
            "ReplaceImagesInBatch",
            "InsertImagesToBatchIndexed",
            "GetImagesFromBatchIndexed",
            "GetImageRangeFromBatch",
            "SimpleCalculatorKJ",
            "VAELoaderKJ",
            "VRAM_Debug",
            "PathchSageAttentionKJ",
            "INTConstant",
        ],
        "pip": {"color_matcher": "color-matcher", "matplotlib": "matplotlib", "mss": "mss", "cv2": "opencv-python-headless", "scipy": "scipy"},
    },
    {
        "dir": "ComfyUI-AIToolkit-MiniMaxH3",
        "repo": "ostris/ComfyUI-AIToolkit-MiniMaxH3",
        "branch": "main",
        "node_ids": ["AIToolkitMiniMaxH3RefVideo"],
        "pip": {},
    },
    {
        "dir": "ComfyUI-BFSNodes",
        "repo": "alisson-anjos/ComfyUI-BFSNodes",
        "branch": "main",
        "node_ids": [
            "BFSShotH3ConditioningSimple",
            "BFSShotPlanner",
            "BFSShotUnpack",
            "BFSShotRepack",
            "BFSShotJoin",
            "BFSVideoCompare",
            "LTXMultipleControls",
            "LTXVEditAnythingLoopingSampler",
            "LTXVEditAnythingModuleLoader",
            "LTXVResizeReferenceByMask",
        ],
        "pip": {"librosa": "librosa", "cv2": "opencv-python-headless", "scenedetect": "scenedetect", "scipy": "scipy"},
    },
    {
        "dir": "Comfyui_Minimax_h3_latent_Upscaler",
        "repo": "LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler",
        "branch": "main",
        "node_ids": ["MinimaxH3LatentUpscaler3D"],
        "pip": {},
    },
    {
        "dir": "ComfyUI-SolAttn_triton",
        "repo": "kijai/ComfyUI-SolAttn_triton",
        "branch": "main",
        "node_ids": ["SolAttnPatch"],
        "pip": {},
    },
    {
        "dir": "ComfyUI-sol-attn",
        "repo": "Saganaki22/ComfyUI-sol-attn",
        "branch": "main",
        "node_ids": ["MiniMaxH3ChunkFeedForward", "MiniMaxH3MemoryEfficientSolAttentionPatch"],
        "pip": {},
    },
    {
        "dir": "comfyui-rmbg",
        "repo": "1038lab/comfyui-rmbg",
        "branch": "main",
        "node_ids": ["BiRefNetRMBG", "AILab_LamaRemover"],
        "pip": {
            "timm": "timm",
            "kornia": "kornia",
            "transparent_background": "transparent-background",
            "onnxruntime": "onnxruntime-gpu",
            "cv2": "opencv-python-headless",
            "segment_anything": "segment-anything",
            "hydra": "hydra-core",
            "omegaconf": "omegaconf",
            "iopath": "iopath",
            "decord": "decord",
            "ftfy": "ftfy",
        },
    },
    {
        "dir": "ComfyUI-Easy-Use",
        "repo": "yolain/ComfyUI-Easy-Use",
        "branch": "main",
        "node_ids": ["easy int", "easy seed", "easy cleanGpuUsed", "easy simpleMath"],
        "pip": {
            "diffusers": "diffusers",
            "accelerate": "accelerate",
            "lark": "lark",
            "onnxruntime": "onnxruntime",
            "cv2": "opencv-python-headless",
            "sentencepiece": "sentencepiece",
            "spandrel": "spandrel",
            "matplotlib": "matplotlib",
            "peft": "peft",
        },
    },
    {
        "dir": "rgthree-comfy",
        "repo": "rgthree/rgthree-comfy",
        "branch": "main",
        "node_ids": ["Image Comparer (rgthree)", "Power Lora Loader (rgthree)", "Any Switch (rgthree)"],
        "pip": {},
    },
    {
        "dir": "ComfyUI_tinyterraNodes",
        "repo": "TinyTerra/ComfyUI_tinyterraNodes",
        "branch": "main",
        "node_ids": ["ttN concat"],
        "pip": {},
    },
    {
        "dir": "ComfyUI-MiniMaxH3-Context-Loop",
        "repo": "ethanfel/ComfyUI-MiniMaxH3-Context-Loop",
        "branch": "main",
        "node_ids": [
            "MiniMaxH3ContexTrimSourceAV",
            "MiniMaxH3ContexMaskedTarget",
            "MiniMaxH3ContexMaskGridPreview",
            "MiniMaxH3ChainAssemble",
            "MiniMaxH3ChainLoopStart",
            "MiniMaxH3ChainLoopEnd",
        ],
        "pip": {},
    },
    {
        "dir": "ComfyUI-H3-Multishot",
        "repo": "jlucasmcrell/ComfyUI-H3-Multishot",
        "branch": "master",
        "node_ids": [
            "H3ClipLoaderAny",
            "H3ModelLoaderAny",
            "H3LoraStack",
            "H3MultishotMemorySampler",
            "H3MultishotSampler",
            "H3StudioControls",
            "H3Keyframes",
        ],
        "pip": {"einops": "einops", "scipy": "scipy"},
    },
    {
        "dir": "ComfyUI_MiniMax_H3_Extender",
        "repo": "tritant/ComfyUI_MiniMax_H3_Extender",
        "branch": "main",
        "node_ids": ["MiniMaxH3Extender"],
        "pip": {"imageio_ffmpeg": "imageio-ffmpeg"},
    },
    {
        "dir": "ComfyUI-MAINodes",
        "repo": "matlowai/ComfyUI-MAINodes",
        "branch": "main",
        "node_ids": ["H3ExactRecover", "H3InjectSchedule", "H3JerkOracle", "H3TimeSmear", "H3V2VInit"],
        "pip": {},
    },
    {
        "dir": "ComfyUI-Spectrum-MiniMax-H3",
        "repo": "xmarre/ComfyUI-Spectrum-MiniMax-H3",
        "branch": "main",
        "node_ids": ["SpectrumApplyMiniMaxH3"],
        "pip": {},
    },
    {
        "dir": "ComfyUI-ReservedVRAM",
        "repo": "Windecay/ComfyUI-ReservedVRAM",
        "branch": "main",
        "node_ids": ["ReservedVRAMSetter"],
        "pip": {"pynvml": "nvidia-ml-py"},
    },
    {
        "dir": "CRT-Nodes",
        "repo": "PGCRT/CRT-Nodes",
        "branch": "main",
        "node_ids": ["CRTFirstLastFrameSelector"],
        "pip": {
            "cv2": "opencv-python-headless",
            "scipy": "scipy",
            "color_matcher": "color-matcher",
            "spandrel": "spandrel",
            "librosa": "librosa",
            "imageio_ffmpeg": "imageio-ffmpeg",
            "soundfile": "soundfile",
            "einops": "einops",
            "rotary_embedding_torch": "rotary-embedding-torch",
            "timm": "timm",
            "ultralytics": "ultralytics",
        },
    },
    {
        "dir": "ComfyUI-DaSiWa-Nodes",
        "repo": "darksidewalker/ComfyUI-DaSiWa-Nodes",
        "branch": "main",
        "node_ids": ["DaSiWa_NodeStatusSwitch", "DaSiWa_RTX_UpscalerRefiner"],
        "pip": {"accelerate": "accelerate", "psutil": "psutil", "av": "av", "pynvml": "nvidia-ml-py"},
    },
    {
        "dir": "ComfyUI-VOSR2",
        "repo": "ylchen333/ComfyUI-VOSR2",
        "branch": "main",
        "node_ids": ["VOSR2ModelLoader", "VOSR2Upscale"],
        "pip": {"huggingface_hub": "huggingface_hub"},
    },
    {
        "dir": "ComfyUI-Frame-Interpolation",
        "repo": "Fannovel16/ComfyUI-Frame-Interpolation",
        "branch": "main",
        "node_ids": ["FILM VFI"],
        "pip": {"einops": "einops", "cv2": "opencv-python-headless", "scipy": "scipy", "skimage": "scikit-image"},
    },
    {
        "dir": "comfyui_controlnet_aux",
        "repo": "Fannovel16/comfyui_controlnet_aux",
        "branch": "main",
        "node_ids": ["CannyEdgePreprocessor", "DWPreprocessor", "MiDaS-DepthMapPreprocessor"],
        "pip": {
            "scipy": "scipy",
            "cv2": "opencv-python-headless",
            "filelock": "filelock",
            "einops": "einops",
            "yaml": "pyyaml",
            "skimage": "scikit-image",
            "omegaconf": "omegaconf",
        },
    },
    {
        "dir": "RES4LYF",
        "repo": "ClownsharkBatwing/RES4LYF",
        "branch": "main",
        "node_ids": ["ClownsharKSampler_Beta"],
        "pip": {"cv2": "opencv-python-headless", "matplotlib": "matplotlib", "pywt": "PyWavelets"},
    },
    {
        "dir": "ComfyUI-GGUF",
        "repo": "city96/ComfyUI-GGUF",
        "branch": "main",
        "node_ids": ["UnetLoaderGGUF"],
        "pip": {"gguf": "gguf>=0.13.0", "sentencepiece": "sentencepiece", "google.protobuf": "protobuf"},
    },
    {
        "dir": "ComfyUI_JoyAI_Echo_GGUF_Nodes",
        "repo": "RealRebelAI/ComfyUI_JoyAI_Echo_GGUF_Nodes",
        "branch": "main",
        "node_ids": ["JoyEcho_LLMEnhance"],
        "pip": {"einops": "einops", "scipy": "scipy"},
    },
    {
        "dir": "WhatDreamsCost-ComfyUI",
        "repo": "WhatDreamscost/WhatDreamsCost-ComfyUI",
        "branch": "main",
        "node_ids": ["LoadAudioUI"],
        "pip": {},
    },
    {
        "dir": "ComfyUI-Custom-Scripts",
        "repo": "pythongosssss/ComfyUI-Custom-Scripts",
        "branch": "main",
        "node_ids": ["ShowText|pysssss"],
        "pip": {},
    },
    {
        "dir": "ComfyUI-Impact-Pack",
        "repo": "ltdrdata/ComfyUI-Impact-Pack",
        "branch": "Main",
        "node_ids": ["ImpactInversedSwitch"],
        "pip": {
            "segment_anything": "segment-anything",
            "skimage": "scikit-image",
            "piexif": "piexif",
            "cv2": "opencv-python-headless",
            "scipy": "scipy",
            "dill": "dill",
            "matplotlib": "matplotlib",
        },
    },
]
PORTABLE_PACK_RESTARTED_ENV = "VSTONE_PORTABLE_PACK_RESTARTED"
_PORTABLE_PACK_LOCK = sys.__dict__.setdefault("_vstone_portable_pack_lock", threading.Lock())


def install_portable_workflow_pack(unique_id: Any = None, allow_restart: bool = True) -> str:
    """Cài/cập nhật trọn bộ 28 custom node packs + pip + bản vá Netflix VOID của Portable Workflow Pack, nạp nóng vào RAM ComfyUI."""
    root = _get_comfy_root()
    if not root:
        return "❌ Không tìm thấy thư mục ComfyUI."
    if not _PORTABLE_PACK_LOCK.acquire(blocking=allow_restart):
        return "⏳ Luồng cài đặt Portable Workflow Pack đang chạy ngầm, vui lòng chờ 1-2 phút."
    try:
        custom_nodes = os.path.join(root, "custom_nodes")
        os.makedirs(custom_nodes, exist_ok=True)
        installed_dirs = {name.lower(): name for name in os.listdir(custom_nodes)}

        details: list[str] = []
        newly_installed = 0
        total = len(PORTABLE_PACK_CUSTOM_NODES)
        for idx, pack in enumerate(PORTABLE_PACK_CUSTOM_NODES, start=1):
            existing_name = installed_dirs.get(pack["dir"].lower())
            pack_dir = os.path.join(custom_nodes, existing_name or pack["dir"])
            has_init = os.path.isfile(os.path.join(pack_dir, "__init__.py"))
            node_ids = pack.get("node_ids", [])
            missing_before = _missing_node_ids(node_ids) if node_ids else ([] if has_init else [pack["dir"]])
            pip_map = pack.get("pip", {})
            missing_pip = [ (mod, spec) for mod, spec in pip_map.items() if _module_missing(mod) ]

            if has_init and not missing_before and not missing_pip:
                details.append(f"[{idx}/{total}] ✅ {pack['dir']}: đã nạp đủ node ({len(node_ids)} node).")
                continue

            if unique_id is not None:
                _vstone_ui_status(unique_id, f"Đang cài/cập nhật [{idx}/{total}] {pack['dir']} ({pack['repo']})...")
            ok, tail = _fetch_node_pack(pack, pack_dir)
            if ok and os.path.isdir(pack_dir):
                newly_installed += 1
                for mod, spec in pip_map.items():
                    if _module_missing(mod):
                        if unique_id is not None:
                            _vstone_ui_status(unique_id, f"[{idx}/{total}] Đang cài thư viện {spec} cho {pack['dir']}...")
                        _run_install_cmd([sys.executable, "-m", "pip", "install", "--prefer-binary", spec], 300)
                for failure in _pip_install_requirements(pack_dir):
                    details.append(f"  ⚠️ pip ({pack['dir']}): {failure}")
                hot_ok = _hot_load_custom_node_pack(pack_dir)
                still_missing = _missing_node_ids(node_ids) if node_ids else []
                if not still_missing:
                    details.append(f"[{idx}/{total}] ✅ {pack['dir']}: đã cập nhật & nạp nóng thành công.")
                else:
                    details.append(f"[{idx}/{total}] 🔄 {pack['dir']}: đã tải về ({'hot-load một phần' if hot_ok else 'cần restart'}, còn thiếu: {', '.join(still_missing[:5])}).")
            else:
                details.append(f"[{idx}/{total}] ❌ {pack['dir']}: lỗi tải ({tail})")

        # Áp dụng bản vá Netflix VOID (patches/nodes_void.py -> ComfyUI/comfy_extras/nodes_void.py) & nạp nóng
        try:
            commit_ref = _get_latest_commit_sha()
            raw_ref_base = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{commit_ref}"
            void_bytes = _http_get_bytes(f"{raw_ref_base}/patches/nodes_void.py?_t={int(time.time())}", timeout=15)
            if void_bytes and len(void_bytes) > 500:
                void_dst = os.path.join(root, "comfy_extras", "nodes_void.py")
                wrote = _atomic_write_bytes(void_dst, void_bytes, min_bytes=500)
                if wrote or _missing_node_ids(["VOIDInpaintConditioning", "VOIDSampler"]):
                    _hot_load_custom_node_pack(void_dst, module_parent="comfy_extras")
                details.append("Patch Netflix VOID (comfy_extras/nodes_void.py): đã cập nhật & nạp nóng.")
        except Exception as e:
            details.append(f"Patch Netflix VOID: {e}")

        all_expected = [nid for p in PORTABLE_PACK_CUSTOM_NODES for nid in p.get("node_ids", [])]
        total_still_missing = _missing_node_ids(all_expected)
        if allow_restart and newly_installed > 0 and total_still_missing:
            tried = set(filter(None, os.environ.get(PORTABLE_PACK_RESTARTED_ENV, "").split(",")))
            if not (set(total_still_missing) <= tried):
                os.environ[PORTABLE_PACK_RESTARTED_ENV] = ",".join(sorted(tried | set(total_still_missing)))
                _restart_comfyui_after(5.0)
                return "\n".join([
                    f"🔄 Đã cài/cập nhật {newly_installed}/{total} custom node packs. ComfyUI đang khởi động lại để nạp phần còn lại — chờ 1-2 phút rồi nhấn F5!",
                    *details,
                ])
        return "\n".join([
            f"✅ Đã cài đặt & nạp nóng {total} custom node packs + bản vá Netflix VOID cho toàn bộ Workflows (01_MiniMax_H3_VFX_Edit .. 05_MiniMax_H3_Loop_And_Native)."
            + (f" (Đã cập nhật mới {newly_installed} pack — nhấn phím R hoặc F5 trên trình duyệt để làm mới danh sách node!)" if newly_installed > 0 else ""),
            *details,
        ])
    finally:
        try:
            _PORTABLE_PACK_LOCK.release()
        except Exception:
            pass


# ------------------------------------------------------------------------------
# LLM Qwen3.8-27B (Huihui Abliterated FP8) chạy bằng vLLM trong môi trường riêng (vLLM kéo torch riêng, không được đụng torch của ComfyUI).
# vLLM nghe ở cổng nội bộ, trang HTML gọi qua /vstone/llm trên chính cổng của ComfyUI, nên không cần mở thêm cổng.
# ------------------------------------------------------------------------------
LLM_ROUTE = "/vstone/llm"
LLM_MODEL_REPO = "munekazu/Huihui-Qwen3.8-27B-abliterated-FP8"
LLM_MODEL_DIR = "Huihui-Qwen3.8-27B-abliterated-FP8"
LLM_SERVED_NAME = "qwen3.8-27b-abliterated"
LLM_LEGACY_NAME = "qwen3.6-35b-a3b"
LLM_PORT = 8010
LLM_GPU_SHARE = 0.4
LLM_MAX_LEN = 65536
LLM_RESTARTED_ENV = "VSTONE_LLM_ROUTE_RESTARTED"


def _llm_paths(root: str) -> dict[str, str]:
    base = os.path.join(root, "vstone_llm")
    venv = os.path.join(base, "venv")
    python = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")
    return {
        "base": base,
        "venv": venv,
        "python": python,
        "vllm_ok": os.path.join(venv, ".vllm_installed"),
        "model": os.path.join(base, LLM_MODEL_DIR),
        "model_ok": os.path.join(base, LLM_MODEL_DIR, ".download_complete"),
        "setup_log": os.path.join(base, "setup.log"),
        "server_log": os.path.join(base, "vllm.log"),
        "pid": os.path.join(base, "vllm.pid"),
    }


def _llm_state() -> dict[str, Any]:
    state = getattr(sys, "_vstone_llm_state", None)
    if state is None:
        state = {"stage": "", "error": "", "gpu": None, "busy": False}
        setattr(sys, "_vstone_llm_state", state)
    return state


def _llm_alive() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{LLM_PORT}/v1/models", timeout=2) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status == 200 and (LLM_SERVED_NAME in body or LLM_MODEL_DIR in body)
    except Exception:
        return False


def _llm_pick_gpu() -> tuple[int, int, int]:
    """Ưu tiên card phụ (GPU > 0) nếu đủ bộ nhớ để tách khỏi ComfyUI ở GPU 0; nếu không thì chọn card trống nhất."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        rows = [[int(v) for v in line.split(",")] for line in out.strip().splitlines() if line.strip()]
        best = max(rows, key=lambda r: (r[0] > 0 and r[1] >= int(r[2] * LLM_GPU_SHARE) + 2048, r[1], r[0]))
        return best[0], best[1], best[2]
    except Exception:
        return 0, 0, 0


# Model FP8 khoảng 29 GB, cộng KV cache và phần đệm. Dưới mức này thì không bật, báo card đang bận.
LLM_MIN_GIB = 36
LLM_START_GRACE_SEC = 600


def _llm_pid_running(paths: dict[str, str]) -> tuple[int, float] | None:
    """vLLM do helper bật trước đó (kể cả trước khi ComfyUI khởi động lại) mà còn sống: (pid, lúc bật)."""
    try:
        with open(paths["pid"], encoding="utf-8") as f:
            pid_text, started_text = f.read().split()
        pid, started = int(pid_text), float(started_text)
        os.kill(pid, 0)
        return pid, started
    except Exception:
        return None


def _llm_stop(paths: dict[str, str]) -> None:
    """Tắt mọi tiến trình vLLM và EngineCore mồ côi (trên Linux venv/bin/python là symlink nên phải dò cả cmdline và tên tiến trình)."""
    killed = []
    try:
        import psutil
        venv = paths["venv"]
        venv_real = os.path.realpath(venv)
        base = paths["base"]
        me = os.getpid()
        for p in psutil.process_iter(["pid", "name", "cmdline", "exe"]):
            if p.pid == me:
                continue
            cmdline = " ".join(p.info.get("cmdline") or [])
            name = p.info.get("name") or ""
            exe = p.info.get("exe") or ""
            if (
                base in cmdline
                or venv in cmdline
                or venv_real in cmdline
                or venv_real in exe
                or "VLLM::EngineCore" in cmdline
                or "VLLM::EngineCore" in name
                or f"--port {LLM_PORT}" in cmdline
            ):
                try:
                    for child in p.children(recursive=True):
                        child.terminate()
                        killed.append(child)
                except Exception:
                    pass
                p.terminate()
                killed.append(p)
        psutil.wait_procs(killed, timeout=20)
        for p in killed:
            if p.is_running():
                p.kill()
    except Exception as e:
        print(f"[VStone] LLM: không tắt được vLLM cũ: {e}", flush=True)
    if killed:
        time.sleep(5)
    try:
        os.remove(paths["pid"])
    except OSError:
        pass


def _log_tail(path: str, lines: int = 6) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            rows = [line.strip() for line in f.read().replace("\r", "\n").splitlines() if line.strip()]
            return "\n".join(rows[-lines:])
    except OSError:
        return ""


LLM_RUN_MARK = "=== VStone: bật vLLM"
LLM_ERROR_WORDS = ("Segfault", "Assertion", "Error", "error:", "Exception", "CUDA", "memory", "OOM", "not supported", "No such file", "Traceback")


def _llm_root_cause(path: str, limit: int = 15) -> str:
    """Lỗi gốc của lần bật gần nhất: dòng cuối của log chỉ là "Engine core initialization failed", lỗi thật nằm phía trên."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return ""
    start = max((i for i, line in enumerate(lines) if line.startswith(LLM_RUN_MARK)), default=0)
    run = lines[start:]
    picked: list[str] = []
    for line in run:
        line = line.strip()
        if line and any(word in line for word in LLM_ERROR_WORDS) and line not in picked:
            picked.append(line[:300])
    head = run[0] if run and run[0].startswith(LLM_RUN_MARK) else ""
    return "\n".join(([head] if head else []) + picked[-limit:] + [f"(Log đầy đủ: {path})"])


LLM_STEPS = (
    ("Application startup complete", "Đã mở cổng, sẵn sàng nhận câu hỏi"),
    ("Capturing CUDA graph", "Chụp CUDA graph, bước cuối"),
    ("Initial profiling/warmup", "Đã chạy thử (warmup), chuẩn bị mở cổng"),
    ("torch.compile", "Biên dịch kernel (torch.compile)"),
    ("KV cache", "Tính chỗ cho KV cache"),
    ("Memory profiling", "Đo bộ nhớ"),
    ("Loading weights took", "Đã nạp xong trọng số"),
    ("Loading safetensors", "Đang nạp trọng số model"),
    ("Starting to load model", "Bắt đầu nạp model"),
    ("Initializing a V1 LLM engine", "Khởi tạo engine"),
    ("non-default args", "Đọc cấu hình"),
)


def _llm_progress(path: str) -> str:
    """vLLM đang làm gì trong lần bật gần nhất: đã chạy bao lâu, bước gần nhất, vài dòng log cuối không kèm tiền tố."""
    import re
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return ""
    start = max((i for i, line in enumerate(lines) if line.startswith(LLM_RUN_MARK)), default=0)
    run = lines[start:]
    elapsed = ""
    stamp = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", run[0]) if run else None
    if stamp:
        seconds = time.time() - time.mktime(time.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S"))
        elapsed = f"Đã chạy {int(seconds // 60)} phút {int(seconds % 60)} giây. "
    step = next((vi for key, vi in LLM_STEPS if any(key in line for line in run)), "Đang chuẩn bị")
    clean = [re.sub(r"^\(\w+ pid=\d+\)\s*(INFO|WARNING|ERROR)?\s*[\d-]+ [\d:]+ \[[^\]]+\]\s*", "", line).strip()[:200]
             for line in run[1:] if line.strip()]
    return "\n".join([f"{elapsed}Bước: {step}."] + clean[-3:])


def _llm_last_run_eager(path: str) -> bool:
    """Kiểm tra lần chạy gần nhất trong log đã là model hiện tại (LLM_MODEL_DIR) và có cờ tăng tốc (--gdn-prefill-backend) chưa."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        start = max((i for i, line in enumerate(lines) if line.startswith(LLM_RUN_MARK)), default=-1)
        return start >= 0 and "--gdn-prefill-backend" in lines[start] and LLM_MODEL_DIR in lines[start]
    except OSError:
        return False


def _llm_spec_failed(path: str) -> bool:
    """Nếu lần bật ngay trước đó có --speculative-config mà dừng sớm, lần bật lại tự bỏ cờ này để đảm bảo chạy được."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        start = max((i for i, line in enumerate(lines) if line.startswith(LLM_RUN_MARK)), default=-1)
        return start >= 0 and "--speculative-config" in lines[start] and LLM_MODEL_DIR in lines[start]
    except OSError:
        return False


def _llm_start_server(root: str) -> None:
    paths, state = _llm_paths(root), _llm_state()
    # Chỉ một vLLM: bản đang chạy hoặc đang nạp (với cấu hình tăng tốc mới) thì để yên, bản cũ hoặc treo quá lâu thì tắt.
    running = _llm_pid_running(paths)
    if running and _llm_last_run_eager(paths["server_log"]) and (_llm_alive() or time.time() - running[1] < LLM_START_GRACE_SEC):
        state["stage"] = "Khởi động vLLM"
        return
    use_mtp = not (_llm_spec_failed(paths["server_log"]) and not _llm_alive())
    _llm_stop(paths)
    gpu, free_mib, total_mib = _llm_pick_gpu()
    share, max_len = LLM_GPU_SHARE, LLM_MAX_LEN
    if total_mib:
        share = min(LLM_GPU_SHARE, round((free_mib - 1536) / total_mib - 0.005, 2))
        if share * total_mib / 1024 < LLM_MIN_GIB:
            state.update(stage="", error=f"Card trống nhất (GPU {gpu}) chỉ còn {free_mib / 1024:.0f} GB, LLM cần ít nhất {LLM_MIN_GIB} GB. "
                                         "Đợi các job ComfyUI xong hoặc giải phóng bộ nhớ rồi thử lại.")
            return
        if share < LLM_GPU_SHARE:
            max_len = 32768
    # DeepGEMM dịch kernel lúc chạy và đòi nvcc >= 12.9, CUDA trên H200 cũ hơn (log 2026-11-06).
    # --enforce-eager + MTP speculative decoding (đoán 2 token/bước từ tầng BF16 mtp sẵn có) + Triton GDN prefill + Prefix caching.
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
               VLLM_USE_DEEP_GEMM="0", VLLM_MOE_USE_DEEP_GEMM="0")
    cmd = [
        paths["python"], "-m", "vllm.entrypoints.openai.api_server",
        "--model", paths["model"], "--served-model-name", LLM_SERVED_NAME, LLM_LEGACY_NAME,
        "--host", "127.0.0.1", "--port", str(LLM_PORT),
        "--gpu-memory-utilization", str(share), "--max-model-len", str(max_len),
        "--enforce-eager",
        "--enable-prefix-caching",
        "--gdn-prefill-backend", "triton",
        "--safetensors-load-strategy", "prefetch",
        "--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
    ]
    if use_mtp:
        cmd.extend(["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 2}'])
    log = open(paths["server_log"], "a", encoding="utf-8")
    log.write(f"\n{LLM_RUN_MARK} {time.strftime('%Y-%m-%d %H:%M:%S')} trên GPU {gpu}: {' '.join(cmd[3:])}\n")
    log.flush()
    # Phiên riêng: vLLM sống sót khi ComfyUI tự khởi động lại bằng os.execv.
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    with open(paths["pid"], "w", encoding="utf-8") as f:
        f.write(f"{proc.pid} {time.time()}")
    state.update(gpu=gpu, proc=proc, stage="Khởi động vLLM")


def _llm_setup(root: str) -> None:
    state, paths = _llm_state(), _llm_paths(root)
    os.makedirs(paths["base"], exist_ok=True)
    # Tắt vLLM của model cũ (nếu đang chạy) để giải phóng VRAM trước khi tải/bật model mới.
    _llm_stop(paths)
    steps = []
    if not os.path.isfile(paths["vllm_ok"]):
        if not os.path.isfile(paths["python"]):
            steps.append(("Tạo môi trường riêng", [sys.executable, "-m", "venv", paths["venv"]], None))
        steps.append(("Cập nhật pip", [paths["python"], "-m", "pip", "install", "-U", "pip"], None))
        steps.append(("Cài vLLM, khoảng 10 phút", [paths["python"], "-m", "pip", "install", "-U", "vllm"], paths["vllm_ok"]))
    if not os.path.isfile(paths["model_ok"]):
        download = f"from huggingface_hub import snapshot_download; snapshot_download({LLM_MODEL_REPO!r}, local_dir={paths['model']!r})"
        steps.append((f"Tải model {LLM_MODEL_DIR} (khoảng 29 GB)", [paths["python"], "-c", download], paths["model_ok"]))
    try:
        with open(paths["setup_log"], "a", encoding="utf-8") as log:
            for label, cmd, marker in steps:
                state["stage"] = label
                log.write(f"\n=== {label}: {' '.join(cmd)}\n")
                log.flush()
                res = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
                if res.returncode != 0:
                    state["error"] = f"{label} lỗi (mã {res.returncode})."
                    return
                if marker:
                    os.makedirs(os.path.dirname(marker), exist_ok=True)
                    with open(marker, "w", encoding="utf-8") as f:
                        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        state["stage"] = "Khởi động vLLM"
        if not _llm_alive():
            _llm_start_server(root)
    except Exception as e:
        state["error"] = str(e)
    finally:
        state["busy"] = False


def _llm_ready(root: str) -> bool:
    paths = _llm_paths(root)
    return os.path.isfile(paths["vllm_ok"]) and os.path.isfile(paths["model_ok"])


def install_llm(unique_id: Any = None) -> str:
    """Cài và chạy LLM ngầm, mỗi lần Queue báo tiến độ. Dòng đầu của báo cáo là kết luận."""
    root = _get_comfy_root()
    if not root:
        return "❌ Không tìm thấy thư mục ComfyUI."
    state, paths = _llm_state(), _llm_paths(root)
    if _llm_alive() and _llm_last_run_eager(paths["server_log"]):
        if not getattr(sys, "_vstone_llm_route", False):
            if os.environ.get(LLM_RESTARTED_ENV) == "1":
                return "❌ vLLM đang chạy nhưng ComfyUI chưa mở được đường dẫn /vstone/llm. Xem log khởi động."
            os.environ[LLM_RESTARTED_ENV] = "1"
            _restart_comfyui_after(5.0)
            return "🔄 vLLM đã chạy. ComfyUI khởi động lại một lần để mở đường dẫn /vstone/llm. Chờ 1-2 phút, F5, rồi bấm Queue lần nữa."
        gpu = state.get("gpu")
        return (f"✅ LLM {LLM_MODEL_DIR} (tăng tốc MTP + Triton GDN) đang chạy{f' trên GPU {gpu}' if gpu is not None else ''}. "
                f"Gọi qua {LLM_ROUTE}/v1/chat/completions trên cổng ComfyUI.")
    if state["busy"]:
        detail = _llm_progress(paths["server_log"]) if state["stage"] == "Khởi động vLLM" else _log_tail(paths["setup_log"], 3)
        return "\n".join([f"⏳ Đang làm: {state['stage']}. Bấm Queue lại để xem tiến độ.", detail])
    if state["error"]:
        report = "\n".join([f"❌ {state['error']} Bấm Queue lần nữa để thử lại.", _log_tail(paths["setup_log"])])
        state["error"] = ""
        return report
    if _llm_ready(root) and _llm_last_run_eager(paths["server_log"]) and (state["stage"] == "Khởi động vLLM" or _llm_pid_running(paths) is not None):
        proc = state.get("proc")
        # Sau khi ComfyUI khởi động lại thì không còn proc trong RAM, xem file pid thay thế.
        dead = proc.poll() is not None if proc is not None else _llm_pid_running(paths) is None
        if not dead:
            state["stage"] = "Khởi động vLLM"
            return "\n".join(["⏳ vLLM đang khởi động, thường 1-2 phút. Bấm Queue lại để xem tiến độ.", _llm_progress(paths["server_log"])])
        state.update(stage="", proc=None)
        if _llm_last_run_eager(paths["server_log"]):
            return "\n".join(["❌ vLLM dừng khi khởi động. Gửi các dòng dưới đây để sửa, rồi bấm Queue để thử lại.",
                              _llm_root_cause(paths["server_log"])])
    if _llm_ready(root):
        _llm_start_server(root)
        if state["error"]:
            err = state["error"]
            state["error"] = ""
            return f"❌ {err}"
        gpu = state.get("gpu")
        return "\n".join([
            f"⏳ Đã bật vLLM ({LLM_MODEL_DIR}) trên GPU {gpu if gpu is not None else 0}, thường mất 1-2 phút nạp 6 phần trọng số. Bấm Queue lại sau 30 giây để xem % tiến độ.",
            _llm_progress(paths["server_log"]),
        ])
    state.update(busy=True, error="", stage="Chuẩn bị")
    threading.Thread(target=_llm_setup, args=(root,), name="VStoneLLMSetup", daemon=True).start()
    return (f"⏳ Bắt đầu tải và bật LLM mới ({LLM_MODEL_DIR}, khoảng 29 GB). "
            "Môi trường vLLM đã có sẵn nên chỉ cần tải model. ComfyUI vẫn dùng bình thường. Bấm Queue lại để xem tiến độ.")


def _register_llm_route() -> None:
    """Chuyển /vstone/llm/<đường dẫn> sang vLLM nội bộ, kể cả trả lời dạng stream. Chỉ đăng ký được lúc ComfyUI khởi động."""
    if globals().get("_DISABLE_DAEMON_ON_EXEC") or getattr(sys, "_vstone_llm_route", False):
        return
    try:
        from aiohttp import ClientSession, ClientTimeout, web
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception:
        return

    @routes.route("*", LLM_ROUTE + "/{tail:.*}")
    async def _vstone_llm_proxy(request):
        url = f"http://127.0.0.1:{LLM_PORT}/{request.match_info['tail']}"
        if request.query_string:
            url += "?" + request.query_string
        body = await request.read()
        try:
            async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=5)) as session:
                async with session.request(request.method, url, data=body,
                                           headers={"Content-Type": request.headers.get("Content-Type", "application/json")}) as upstream:
                    resp = web.StreamResponse(status=upstream.status,
                                              headers={"Content-Type": upstream.headers.get("Content-Type", "application/json")})
                    await resp.prepare(request)
                    async for chunk in upstream.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    return resp
        except Exception as e:
            return web.json_response({"error": f"LLM chưa chạy ({e}). Chạy mục 7 của node stoneV_CloudSyncHub."}, status=503)

    setattr(sys, "_vstone_llm_route", True)


def _llm_on_load() -> None:
    """Đã cài xong thì tự bật lại vLLM khi ComfyUI khởi động, nếu nó chưa chạy."""
    if globals().get("_DISABLE_DAEMON_ON_EXEC") or getattr(sys, "_vstone_llm_autostart", False):
        return
    root = _get_comfy_root()
    if not root or not _llm_ready(root):
        return
    setattr(sys, "_vstone_llm_autostart", True)

    def _later():
        time.sleep(20)
        if not _llm_alive():
            _llm_state()["stage"] = "Khởi động vLLM"
            _llm_start_server(root)

    threading.Thread(target=_later, name="VStoneLLMAutostart", daemon=True).start()


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
try:
    _register_llm_route()
    _llm_on_load()
except Exception as _llm_err:
    print(f"[VStone] LLM: {_llm_err}", flush=True)


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
                    "4. Cài pack AusBoss cho outpaint Qwen (tự khởi động lại)",
                    "5. Cài AnyAngle Studio để xoay vật thể (tự khởi động lại)",
                    "6. Cài pack cho BFS đổi mặt / đổi nhân vật (tự khởi động lại)",
                    "7. Cài LLM Qwen3.6 (vLLM)",
                    "8. Cài Portable Workflow Pack (22 Custom Nodes + Patch VOID, tự khởi động lại)",
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
                    "diffusion_models/vace",
                    "diffusion_models/wan",
                    "checkpoints",
                    "vae",
                    "vae_approx",
                    "text_encoders",
                    "model_patches",
                    "controlnet",
                    "latent_upscale_models",
                    "upscale_models",
                    "optical_flow",
                    "frame_interpolation",
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
        # ComfyUI giữ đối tượng node giữa các lần Queue (execution.py, caches.objects), nên sau khi nạp nóng
        # node đang nằm trên màn hình vẫn chạy code cũ. Chuyển sang lớp mới nhất đã nạp.
        import nodes
        newest = nodes.NODE_CLASS_MAPPINGS.get("stoneV_CloudSyncHub")
        if newest is not None and getattr(newest, "run_sync", None) is not stoneV_CloudSyncHub.run_sync:
            return newest.run_sync(self, action, preset_model, custom_url, custom_target_folder, unique_id)
        if action.startswith("0."):
            report = install_ltx25_one_click(unique_id)
            _vstone_ui_status(unique_id, report)
            return {"ui": {"text": [report]}, "result": (report,)}

        if action.startswith("4."):
            report = install_ausboss_outpaint(unique_id)
            _vstone_ui_status(unique_id, report)
            return {"ui": {"text": [report]}, "result": (report,)}

        if action.startswith("5."):
            report = install_anyangle_studio(unique_id)
            _vstone_ui_status(unique_id, report)
            return {"ui": {"text": [report]}, "result": (report,)}

        if action.startswith("7."):
            report = install_llm(unique_id)
            _vstone_ui_status(unique_id, report)
            return {"ui": {"text": [report]}, "result": (report,)}

        if action.startswith("6."):
            report = install_bfs_swap_packs(unique_id)
            _vstone_ui_status(unique_id, report)
            return {"ui": {"text": [report]}, "result": (report,)}

        if action.startswith("8."):
            report = install_portable_workflow_pack(unique_id)
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
                elif folder == "__ALL_WAN21_VACE__":
                    for item in WAN21_VACE_OUTPAINT_MODELS:
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


SCREENX_PAINT_MULTIPLE = 64  # LTX-2.5 cắt hình về bội số 64 ngay sau bước đệm, khung phải khớp sẵn để không mất hình

HUY_SCREENX_EDITOR_JS = r"""
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "Huy_screenx_outpain_mask";
const GRID = 64;
const MAX_PAD = 8192;
const EDGE_GRAB_PX = 9;
const VIEW_MARGIN_PX = 14;
const FOOTER_PX = 42;
const EDITOR_MIN_HEIGHT = 280;
const PLACEHOLDER_SIZE = { width: 1792, height: 1024 };
const PAD_NAMES = ["left", "top", "right", "bottom"];

const snap = (value) => Math.min(MAX_PAD, Math.max(0, Math.round(value / GRID) * GRID));

// Giống _round_screenx_pad trong node Python.
function roundPad(size, before, after) {
    const total = size + before + after;
    const extra = Math.ceil(total / GRID) * GRID - total;
    return before > 0 && after === 0 ? [before + extra, after] : [before, after + extra];
}

function createEditor(node) {
    const field = (name) => node.widgets.find((widget) => widget.name === name);
    const readPad = () => Object.fromEntries(PAD_NAMES.map((name) => [name, Number(field(name).value) || 0]));

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%;height:100%;display:block;touch-action:none;";
    const domWidget = node.addDOMWidget("screenx_editor", "custom", canvas, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => EDITOR_MIN_HEIGHT,
    });
    domWidget.serialize = false;

    let frame = null;
    let drag = null;

    const frameSize = () => ({
        width: node.properties?.frameWidth ?? PLACEHOLDER_SIZE.width,
        height: node.properties?.frameHeight ?? PLACEHOLDER_SIZE.height,
    });

    function geometry(pad) {
        const { width, height } = frameSize();
        const [left, right] = roundPad(width, pad.left, pad.right);
        const [top, bottom] = roundPad(height, pad.top, pad.bottom);
        const canvasW = left + width + right;
        const canvasH = top + height + bottom;
        let scale, originX, originY;
        if (drag) {
            // Lúc kéo giữ nguyên tỉ lệ để hình không nhảy: kéo ảnh thì khung đứng yên, kéo mép thì ảnh đứng yên.
            scale = drag.scale;
            originX = drag.mode === "move" ? drag.originX : drag.imageX - left * scale;
            originY = drag.mode === "move" ? drag.originY : drag.imageY - top * scale;
        } else {
            const viewW = canvas.clientWidth;
            const viewH = canvas.clientHeight - FOOTER_PX;
            scale = Math.max(0.001, Math.min((viewW - 2 * VIEW_MARGIN_PX) / canvasW, (viewH - 2 * VIEW_MARGIN_PX) / canvasH));
            originX = (viewW - canvasW * scale) / 2;
            originY = (viewH - canvasH * scale) / 2;
        }
        return {
            left, right, top, bottom, width, height, canvasW, canvasH, scale, originX, originY,
            imageX: originX + left * scale,
            imageY: originY + top * scale,
            imageW: width * scale,
            imageH: height * scale,
        };
    }

    function render() {
        const viewW = canvas.clientWidth;
        const viewH = canvas.clientHeight;
        if (!viewW || !viewH) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(viewW * dpr);
        canvas.height = Math.round(viewH * dpr);
        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);
        ctx.fillStyle = "#1e1e1e";
        ctx.fillRect(0, 0, viewW, viewH);

        const g = geometry(readPad());
        const boxW = g.canvasW * g.scale;
        const boxH = g.canvasH * g.scale;

        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, viewW, viewH - FOOTER_PX);
        ctx.clip();
        ctx.fillStyle = "#808080";
        ctx.fillRect(g.originX, g.originY, boxW, boxH);
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        if (frame) {
            ctx.drawImage(frame, g.imageX, g.imageY, g.imageW, g.imageH);
        } else {
            ctx.fillStyle = "#3a3a3a";
            ctx.fillRect(g.imageX, g.imageY, g.imageW, g.imageH);
            ctx.fillStyle = "#ffffff";
            ctx.font = "12px sans-serif";
            ctx.fillText("Chạy node 1 lần để hiện ảnh thật", g.imageX + g.imageW / 2, g.imageY + g.imageH / 2);
        }
        ctx.strokeStyle = "#39a0ff";
        ctx.lineWidth = 2;
        ctx.strokeRect(g.imageX, g.imageY, g.imageW, g.imageH);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.strokeRect(g.originX, g.originY, boxW, boxH);

        ctx.fillStyle = "#ffffff";
        for (const [x, y] of [
            [g.originX, g.originY + boxH / 2],
            [g.originX + boxW, g.originY + boxH / 2],
            [g.originX + boxW / 2, g.originY],
            [g.originX + boxW / 2, g.originY + boxH],
        ]) {
            ctx.fillRect(x - 4, y - 4, 8, 8);
        }

        ctx.font = "12px sans-serif";
        ctx.fillStyle = "#000000";
        const label = (value, base, regionPx, x, y) => {
            if (value > 0 && regionPx >= 34) ctx.fillText(`${value} px (${Math.round((100 * value) / base)}%)`, x, y);
        };
        const midY = g.imageY + g.imageH / 2;
        const midX = g.imageX + g.imageW / 2;
        label(g.left, g.width, g.left * g.scale, g.originX + (g.left * g.scale) / 2, midY);
        label(g.right, g.width, g.right * g.scale, g.imageX + g.imageW + (g.right * g.scale) / 2, midY);
        label(g.top, g.height, g.top * g.scale, midX, g.originY + (g.top * g.scale) / 2);
        label(g.bottom, g.height, g.bottom * g.scale, midX, g.imageY + g.imageH + (g.bottom * g.scale) / 2);
        ctx.restore();

        ctx.font = "12px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#ffffff";
        ctx.fillText(`Trái ${g.left}   Phải ${g.right}   Trên ${g.top}   Dưới ${g.bottom} px`, 8, viewH - 29);
        if (g.left + g.right + g.top + g.bottom === 0) {
            ctx.fillStyle = "#ff6b6b";
            ctx.fillText("Không có vùng xám: LTX sẽ gen lại y hệt", 8, viewH - 11);
        } else {
            ctx.fillText(`Khung ${g.canvasW}×${g.canvasH}   Ảnh ${g.width}×${g.height}`, 8, viewH - 11);
        }
    }

    function applyPad(next) {
        for (const name of PAD_NAMES) {
            const widget = field(name);
            if (widget.value !== next[name]) {
                widget.value = next[name];
                widget.callback?.(next[name], app.canvas, node);
            }
        }
        node.setDirtyCanvas?.(true, true);
        render();
    }

    const pointerPosition = (event) => {
        // Graph đang zoom thì canvas bị co giãn bằng CSS, phải quy về đơn vị của canvas.
        const rect = canvas.getBoundingClientRect();
        return {
            x: ((event.clientX - rect.left) * canvas.clientWidth) / rect.width,
            y: ((event.clientY - rect.top) * canvas.clientHeight) / rect.height,
        };
    };

    function hitTest({ x, y }) {
        const g = geometry(readPad());
        const x0 = g.originX;
        const y0 = g.originY;
        const x1 = x0 + g.canvasW * g.scale;
        const y1 = y0 + g.canvasH * g.scale;
        const withinY = y >= y0 - EDGE_GRAB_PX && y <= y1 + EDGE_GRAB_PX;
        const withinX = x >= x0 - EDGE_GRAB_PX && x <= x1 + EDGE_GRAB_PX;
        const edges = {
            left: withinY && Math.abs(x - x0) <= EDGE_GRAB_PX,
            right: withinY && Math.abs(x - x1) <= EDGE_GRAB_PX,
            top: withinX && Math.abs(y - y0) <= EDGE_GRAB_PX,
            bottom: withinX && Math.abs(y - y1) <= EDGE_GRAB_PX,
        };
        if (edges.left || edges.right || edges.top || edges.bottom) return { mode: "edge", edges };
        const insideImage = x >= g.imageX && x <= g.imageX + g.imageW && y >= g.imageY && y <= g.imageY + g.imageH;
        return insideImage ? { mode: "move", edges } : null;
    }

    function cursorFor(hit) {
        if (!hit) return "default";
        if (hit.mode === "move") return drag ? "grabbing" : "grab";
        const { left, right, top, bottom } = hit.edges;
        if ((left && top) || (right && bottom)) return "nwse-resize";
        if ((right && top) || (left && bottom)) return "nesw-resize";
        return left || right ? "ew-resize" : "ns-resize";
    }

    canvas.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        const position = pointerPosition(event);
        const hit = hitTest(position);
        if (!hit) return;
        event.stopPropagation();
        event.preventDefault();
        canvas.setPointerCapture(event.pointerId);
        const g = geometry(readPad());
        drag = {
            ...hit,
            startX: position.x,
            startY: position.y,
            startPad: { left: g.left, right: g.right, top: g.top, bottom: g.bottom },
            scale: g.scale,
            originX: g.originX,
            originY: g.originY,
            imageX: g.imageX,
            imageY: g.imageY,
        };
        canvas.style.cursor = cursorFor(hit);
    });

    canvas.addEventListener("pointermove", (event) => {
        const position = pointerPosition(event);
        if (!drag) {
            canvas.style.cursor = cursorFor(hitTest(position));
            return;
        }
        event.stopPropagation();
        event.preventDefault();
        const dx = (position.x - drag.startX) / drag.scale;
        const dy = (position.y - drag.startY) / drag.scale;
        const start = drag.startPad;
        const next = { ...start };
        if (drag.mode === "move") {
            const totalX = start.left + start.right;
            const totalY = start.top + start.bottom;
            next.left = Math.min(totalX, snap(start.left + dx));
            next.right = totalX - next.left;
            next.top = Math.min(totalY, snap(start.top + dy));
            next.bottom = totalY - next.top;
        } else {
            if (drag.edges.left) next.left = snap(start.left - dx);
            if (drag.edges.right) next.right = snap(start.right + dx);
            if (drag.edges.top) next.top = snap(start.top - dy);
            if (drag.edges.bottom) next.bottom = snap(start.bottom + dy);
        }
        applyPad(next);
    });

    const endDrag = (event) => {
        if (!drag) return;
        event.stopPropagation();
        drag = null;
        canvas.style.cursor = "default";
        render();
        node.graph?.change?.();
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    for (const name of PAD_NAMES) {
        const widget = field(name);
        const callback = widget.callback;
        widget.callback = function (...args) {
            const result = callback?.apply(this, args);
            render();
            return result;
        };
    }

    const swapButton = node.addWidget("button", "Đổi trái ↔ phải", null, () => {
        const pad = readPad();
        applyPad({ ...pad, left: pad.right, right: pad.left });
        node.graph?.change?.();
    }, { serialize: false });
    swapButton.serialize = false;

    const observer = new ResizeObserver(render);
    observer.observe(canvas);

    return {
        render,
        disconnect: () => observer.disconnect(),
        setFrame(image, width, height) {
            frame = image;
            node.properties = node.properties ?? {};
            node.properties.frameWidth = width;
            node.properties.frameHeight = height;
            render();
        },
    };
}

app.registerExtension({
    name: "vstone.HuyScreenxOutpainMask",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.screenxEditor = createEditor(this);
            const [width, height] = this.computeSize();
            this.setSize([Math.max(width, 480), Math.max(height, 480)]);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            this.screenxEditor?.render();
            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.screenxEditor?.disconnect();
            return onRemoved?.apply(this, arguments);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const info = message?.huy_preview?.[0];
            if (!info) return;
            const image = new Image();
            image.onload = () => this.screenxEditor?.setFrame(image, info.width, info.height);
            const query = new URLSearchParams({ filename: info.filename, subfolder: info.subfolder, type: info.type, rand: String(Math.random()) });
            image.src = api.apiURL(`/view?${query}`);
        };
    },
});
"""


def _install_screenx_editor_js() -> None:
    """Hot-reload không thêm được route mới, nhưng ComfyUI luôn phục vụ thư mục extensions của frontend, nên chỉ cần F5."""
    try:
        import server
        web_root = server.PromptServer.instance.web_root
    except Exception:
        return  # không chạy trong ComfyUI
    path = os.path.join(web_root, "extensions", "vstone", "huy_screenx_outpain_mask.js")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as js_file:
            js_file.write(HUY_SCREENX_EDITOR_JS)
    except OSError as e:
        print(f"[VStone] Không ghi được khung đồ họa Huy_screenx vào {path}: {e}", flush=True)


_install_screenx_editor_js()


# >>> CONCEPT_IMAGE_EDIT_HTML: sinh từ vstone/concept_image_edit_src.html bằng vstone/embed_concept_html.py, đừng sửa tay
CONCEPT_IMAGE_EDIT_HTML = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>img{max-width:100%}</style>
</head>
<body>
<title>Concept Image Edit</title>
<style>
/* Bàn vẽ tối trung tính: ảnh chiếm gần hết màn hình, màu duy nhất là màu cọ. Bấm cọ mở hộp tuỳ chỉnh ngay cạnh cọ. */
:root{
  color-scheme:dark;
  --bench:#212226;
  --well:#17181B;
  --surface:#2B2C31;
  --raised:#3A3B42;
  --line:#4B4C54;
  --ink:#EFEDE6;
  --ink-2:#B7B5AD;
  --paper:#EFEDE6;
  --paper-ink:#1B1C1F;
  --warn:#F2B14A;
  --font-ui:"Be Vietnam Pro",system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-data:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bench);color:var(--ink);font:400 13px/1.45 var(--font-ui)}
[hidden]{display:none!important}
button,input,select,textarea{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--paper);outline-offset:2px}
h1,h2,h3,p{margin:0}
svg{display:block}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}

.app{height:100%;display:flex;flex-direction:column;gap:8px;
  padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:10px}
.main{position:relative;display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:8px;min-height:0;flex:1}

.strip{width:64px;display:flex;flex-direction:column;align-items:center;gap:4px;min-height:0;overflow-y:auto;scrollbar-width:none}
.tiles{display:grid;gap:10px;padding:5px}
.tiles:empty{display:none}
.tile{position:relative;width:48px;height:48px;padding:4px;border:0;border-radius:6px;background:var(--c);color:var(--on);cursor:pointer;display:grid;place-items:center}
.tile img{position:absolute;inset:4px;width:calc(100% - 8px);height:calc(100% - 8px);max-width:none;object-fit:cover;display:block;border-radius:3px}
.tile[data-painted="false"]{opacity:.55}
.tile[data-active="true"]{opacity:1;box-shadow:0 0 0 2px var(--bench),0 0 0 4px var(--paper)}
.bang{position:absolute;right:-5px;top:-5px;width:17px;height:17px;border-radius:50%;background:var(--warn);color:#111;font:600 11px/17px var(--font-ui);text-align:center}
.more{position:absolute;right:-5px;bottom:-5px;width:18px;height:18px;border-radius:50%;background:var(--paper);color:var(--paper-ink);display:none;place-items:center}
.more svg{width:11px;height:11px}
.tile[data-active="true"] .more{display:grid}
.tilewrap{position:relative}
.tilex{position:absolute;left:-6px;top:-6px;z-index:1;width:18px;height:18px;padding:0;border:1px solid var(--line);border-radius:50%;background:var(--well);color:var(--ink);cursor:pointer;display:none;place-items:center}
.tilex svg{width:10px;height:10px}
.tilex:hover{background:var(--paper);color:var(--paper-ink);border-color:var(--paper)}
.tilewrap:hover .tilex,.tilewrap:focus-within .tilex,.tile[data-active="true"] + .tilex{display:grid}
.ltile{position:relative;width:48px;height:48px;padding:3px;border:1px solid var(--line);border-radius:6px;background:repeating-conic-gradient(var(--raised) 0 25%,var(--surface) 0 50%) 0 0/10px 10px;cursor:pointer;display:grid;place-items:center}
.ltile canvas{display:block;max-width:100%;max-height:100%}
.ltile[data-active="true"]{box-shadow:0 0 0 2px var(--bench),0 0 0 4px var(--paper)}
.ltile[data-active="true"] + .tilex{display:grid}
.floatbar{position:absolute;left:50%;top:10px;transform:translateX(-50%);z-index:3;display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:6px;width:max-content;max-width:calc(100% - 20px);padding:6px 8px;background:var(--surface);border:1px solid var(--line);border-radius:6px;box-shadow:0 8px 24px #00000080}
.floatbar .viewchip{padding:3px 10px;font-size:12px}
.floatbar .viewchip:disabled{opacity:.45;cursor:default}
.selwrap{position:absolute;left:50%;top:10px;transform:translateX(-50%);z-index:3;display:grid;justify-items:center;gap:6px;width:max-content;max-width:calc(100% - 20px)}
.selwrap > .capture-hint,.selwrap > .floatbar{position:static;transform:none;max-width:100%}
.selwrap > .capture-hint{flex-wrap:nowrap;justify-content:center;gap:6px;padding:5px 8px;white-space:nowrap}
.selwrap .selname{flex:1 1 140px;min-width:0}
.selwrap,.selwrap > *{pointer-events:none}
.selwrap input,.selwrap button,.selwrap label{pointer-events:auto}
.selname{width:200px;padding:3px 8px;font-size:12px;font-weight:400;color:var(--paper-ink);background:#fff;border:1px solid var(--paper-ink);border-radius:4px}
.selsoft{display:flex;align-items:center;gap:6px;padding:0 4px;font-size:12px;color:var(--ink-2)}
.selsoft input{width:110px;accent-color:var(--paper)}
.floatbar > span{padding:0 4px;font-size:12px;color:var(--ink-2)}
.qstate{display:flex;align-items:center;gap:6px;min-width:0;font-size:12px;color:var(--ink-2);white-space:nowrap;font-variant-numeric:tabular-nums}
.qdot{width:8px;height:8px;flex:none;border:1.5px solid var(--ink-2);border-radius:50%}
.qstate[data-busy="true"]{color:var(--ink)}
.qstate[data-busy="true"] .qdot{background:var(--warn);border-color:var(--warn)}
.add{width:48px;height:48px;flex:none;margin-block:5px;padding:0;background:transparent;border:1.5px dashed var(--ink-2);border-radius:6px;color:var(--ink);cursor:pointer;display:grid;place-items:center}
.add:hover{border-style:solid;background:var(--surface)}
.add.nudge{border-color:var(--paper);background:var(--raised)}
.env{position:relative;width:48px;height:48px;flex:none;padding:0;background:transparent;border:1px solid var(--line);border-radius:6px;color:var(--ink);cursor:pointer;display:grid;place-items:center}
.env:hover{background:var(--surface)}
.env[aria-expanded="true"]{box-shadow:0 0 0 2px var(--bench),0 0 0 4px var(--paper)}
.env img{position:absolute;inset:4px;width:calc(100% - 8px);height:calc(100% - 8px);max-width:none;object-fit:cover;display:block;border-radius:3px}
.envcount{position:absolute;right:-5px;bottom:-5px;min-width:18px;height:18px;border-radius:9px;background:var(--paper);color:var(--paper-ink);font:600 11px/18px var(--font-ui);text-align:center}
.strip hr{width:36px;border:0;border-top:1px solid var(--line);margin:4px 0;flex:none}
.ibtn{width:48px;height:36px;flex:none;padding:0;background:transparent;border:0;border-radius:4px;color:var(--ink);cursor:pointer;display:grid;place-items:center}
.ibtn:hover:not(:disabled){background:var(--surface)}
.ibtn[aria-pressed="true"]{background:var(--paper);color:var(--paper-ink)}
.ibtn:disabled{opacity:.35;cursor:default}
.ibtn:focus-within{outline:2px solid var(--paper);outline-offset:2px}

.stage{position:relative;overflow:hidden;background:var(--well);border-radius:4px;min-height:260px;min-width:0}
.frame{position:absolute;left:0;top:0}
#view{display:block;touch-action:none}
.imgedge{position:absolute;pointer-events:none;outline:1px dashed #ffffffb0}
.padtag{position:absolute;pointer-events:none;transform:translate(-50%,-50%);font:400 11px var(--font-data);color:#fff;background:#000000a8;padding:1px 5px;border-radius:2px;white-space:nowrap}
.edge{position:absolute;display:grid;place-items:center;touch-action:none}
.edge::after{content:"";background:var(--paper);border-radius:3px;box-shadow:0 0 0 1px #000}
.edge:hover::after{background:#fff}
.edge[data-side="left"],.edge[data-side="right"]{top:0;bottom:0;width:22px;cursor:ew-resize}
.edge[data-side="left"]{left:-22px}
.edge[data-side="right"]{right:-22px}
.edge[data-side="left"]::after,.edge[data-side="right"]::after{width:6px;height:56px}
.edge[data-side="top"],.edge[data-side="bottom"]{left:0;right:0;height:22px;cursor:ns-resize}
.edge[data-side="top"]{top:-22px}
.edge[data-side="bottom"]{bottom:-22px}
.edge[data-side="top"]::after,.edge[data-side="bottom"]::after{width:56px;height:6px}
.brushcur{position:absolute;left:0;top:0;pointer-events:none;border-radius:50%;border:1.5px solid #fff;box-shadow:0 0 0 1px #000}
.tag{position:absolute;left:8px;top:8px;display:flex;align-items:center;gap:5px;font:400 11px var(--font-data);background:#000000a8;color:#fff;padding:3px 6px;border:0;border-radius:3px;cursor:pointer}
.tag svg{width:11px;height:11px}
.drop{position:absolute;inset:0;display:grid;place-content:center;justify-items:center;gap:8px;cursor:pointer;color:var(--ink-2)}
.drop:hover{color:var(--ink)}
.drop svg{width:48px;height:48px}
.stage.over{outline:2px dashed var(--paper);outline-offset:-6px}

.tabs{display:flex;gap:4px;flex:none}
.tab{background:transparent;border:0;border-bottom:2px solid transparent;padding:2px 10px 4px;color:var(--ink-2);font-weight:500;cursor:pointer}
.tab:hover{color:var(--ink)}
.tab[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--paper)}
.sheetpanel{width:280px;display:grid;gap:10px;align-content:start;min-height:0;overflow-y:auto;padding:2px}
.sheetslots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.slotlabel{position:absolute;left:5px;bottom:5px;font-size:11px;background:#000000b8;color:#fff;padding:1px 5px;border-radius:2px;pointer-events:none}
.sheetbox{position:relative;display:grid;place-items:center;height:150px;border:1.5px dashed var(--ink-2);border-radius:4px;color:var(--ink-2);cursor:pointer;overflow:hidden}
.sheetbox:hover{border-style:solid}
.sheetbox:focus-within{outline:2px solid var(--paper);outline-offset:2px}
.sheetbox.over{border-color:var(--paper)}
.sheetbox img{position:absolute;inset:0;width:100%;height:100%;max-width:none;object-fit:contain;background:var(--well)}
.sheetempty-box{display:grid;justify-items:center;gap:4px}
.sheetempty-box svg{width:32px;height:32px}
.sheetx{position:absolute;right:6px;top:6px;width:24px;height:24px;padding:0;background:var(--well);border:1px solid var(--line);border-radius:3px;cursor:pointer;display:grid;place-items:center}
.sheetx svg{width:12px;height:12px}
.sheetwrap{position:relative}
.viewchips{display:flex;flex-wrap:wrap;gap:6px}
.viewchip{padding:4px 11px;background:var(--raised);border:1px solid var(--line);border-radius:14px;cursor:pointer}
.viewchip[aria-pressed="true"]{background:var(--paper);color:var(--paper-ink);border-color:var(--paper)}
.sheetpanel textarea.field{height:64px}
.sheet-empty{position:absolute;inset:0;display:grid;place-items:center;text-align:center;padding:16px;color:var(--ink-2)}
.rotslots .sheetwrap:first-child{grid-column:1/-1}
.rotslots .sheetwrap:first-child .sheetbox{height:180px}
.rotslots .sheetbox{height:96px}
.rot{position:absolute;inset:0;z-index:1;display:grid;grid-template-rows:minmax(0,1fr) auto;gap:8px;padding:8px}
.rotview{position:relative;min-height:0;overflow:hidden;border-radius:4px;cursor:grab;touch-action:none;user-select:none}
.rotview.dragging{cursor:grabbing}
.rotview canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.rotframe{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);outline:1px dashed #ffffffb0;box-shadow:0 0 0 1px #00000080;pointer-events:none}
.rotnote{position:absolute;right:8px;top:8px;padding:3px 8px;font-size:12px;color:var(--paper-ink);background:var(--warn);border-radius:3px;pointer-events:none}
.rotmsg{position:absolute;inset:0;display:grid;place-content:center;justify-items:center;gap:10px;text-align:center;padding:16px;color:var(--ink-2);pointer-events:none}
.rotmsg .bar-track{width:220px;height:4px;background:var(--raised);border-radius:2px;overflow:hidden}
.rotmsg .bar-fill{display:block;height:100%;width:0;background:var(--paper);transition:width .1s linear}
.rotctl{display:grid;justify-items:center;gap:6px}
.rotrow{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:4px 6px}
.rotrow .viewchip{padding:3px 10px;font-size:12px}
.swap{position:absolute;inset:0;z-index:1;padding:8px}
.swapview{position:relative;width:100%;height:100%;overflow:hidden;border-radius:4px;touch-action:none;user-select:none}
.swapview canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.swapview.painting{cursor:none}
.swapring{position:absolute;left:0;top:0;border:1.5px solid #fff;border-radius:50%;box-shadow:0 0 0 1px rgba(0,0,0,.55);pointer-events:none;transform:translate(-50%,-50%)}
.swapslots .sheetbox{height:150px}
.swappick{display:grid;gap:8px}
.swappick-head{display:flex;align-items:center;gap:8px;font-size:13px}
.swappick-head span:last-child{color:var(--ink-2);font-size:12px}
.swapswatch{width:14px;height:14px;border-radius:3px;background:#FF00FF;opacity:.6;flex:none}
.swapsize{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink-2)}
.swapsize input{flex:1;min-width:0}
.upscale{position:absolute;inset:0;z-index:1;padding:8px}
.upview{position:relative;width:100%;height:100%;overflow:hidden;border-radius:4px}
.upview canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.upinfo{position:absolute;left:8px;top:8px;display:grid;gap:2px;max-width:calc(100% - 16px);padding:5px 9px;font:400 12px/1.45 var(--font-data);font-variant-numeric:tabular-nums;color:var(--ink);background:var(--well);border:1px solid var(--line);border-radius:3px;pointer-events:none}
.upinfo .upwarn{font-family:var(--font-ui);color:var(--warn)}
.upslots .sheetbox{height:180px}
.upview.comparing{cursor:ew-resize;touch-action:none;user-select:none}
.upview.comparing.panning{cursor:grabbing}
.uptools{position:absolute;right:8px;top:8px;display:flex;flex-wrap:wrap;justify-content:flex-end;gap:6px}
.uptools .viewchip{padding:3px 10px;font-size:12px;background:var(--well)}
.upview .rotnote{top:auto;bottom:8px}
.rotslider{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink-2)}
.rotslider input{width:220px;accent-color:var(--paper)}
.capbox{position:absolute;pointer-events:none;outline:2px dashed #fff;box-shadow:0 0 0 1px #000,0 0 0 9999px #00000066}
.capture-hint{position:absolute;left:50%;top:10px;transform:translateX(-50%);z-index:3;display:flex;align-items:center;gap:10px;padding:8px 12px;background:var(--paper);color:var(--paper-ink);border-radius:6px;font-weight:600;white-space:nowrap;box-shadow:0 8px 24px #00000080}
.capture-hint button{background:transparent;border:1px solid var(--paper-ink);border-radius:4px;padding:2px 8px;color:inherit;font-weight:500;cursor:pointer}
.pop{position:absolute;z-index:5;width:390px;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px;display:grid;gap:10px;box-shadow:0 12px 32px #00000090}
.types{display:grid;grid-template-columns:repeat(4,1fr);gap:4px}
.type{display:grid;justify-items:center;align-content:start;gap:4px;padding:7px 1px 5px;background:transparent;border:1px solid transparent;border-radius:4px;color:var(--ink-2);font-size:10.5px;line-height:1.2;text-align:center;cursor:pointer}
.type:hover{color:var(--ink)}
.type[aria-pressed="true"]{background:var(--raised);border-color:var(--line);color:var(--ink)}
.detail{display:grid;grid-template-columns:76px minmax(0,1fr);gap:8px}
.imgwrap{position:relative;height:76px}
.imgwrap[data-off="true"]{opacity:.28}
.imgwrap[data-off="true"] .imgpick{cursor:default;border-style:dashed}
.imgpick{position:absolute;inset:0;display:grid;place-items:center;border:1.5px dashed var(--ink-2);border-radius:4px;cursor:pointer;overflow:hidden}
.imgpick:hover{border-style:solid}
.imgpick:focus-within{outline:2px solid var(--paper);outline-offset:2px}
.imgpick img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.imgx{position:absolute;right:3px;top:3px;width:20px;height:20px;padding:0;background:var(--well);border:1px solid var(--line);border-radius:3px;cursor:pointer;display:grid;place-items:center}
.imgx svg{width:11px;height:11px}
.field{background:var(--well);border:1px solid var(--line);border-radius:4px;padding:6px 8px;width:100%;min-width:0}
textarea.field{display:block;resize:none;height:76px;font:400 12px/1.45 var(--font-data)}
.popfoot{display:flex;align-items:center;gap:8px}
.sw{width:24px;height:24px;padding:0;border:0;border-radius:50%;background:var(--c);cursor:pointer}
.sw[aria-pressed="true"]{box-shadow:0 0 0 2px var(--surface),0 0 0 4px var(--paper)}
.popfoot .ibtn{margin-left:auto;width:36px;height:30px}
.typehint,.envhint{color:var(--ink-2);font-size:11.5px;line-height:1.35}
.envlora{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:12px;color:var(--ink-2)}
.envlora .viewchip{padding:3px 12px;font-size:12px}
.envslots{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.imgpick.over{border-color:var(--paper)}

.bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px}
.chat{position:absolute;inset:0;z-index:1;display:grid;grid-template-rows:minmax(0,1fr) auto;gap:8px;padding:8px}
.chatlog{overflow-y:auto;display:flex;flex-direction:column;gap:10px;padding:4px}
.msg{max-width:min(760px,92%);padding:10px 12px;border-radius:6px;white-space:pre-wrap;word-break:break-word;line-height:1.45;position:relative}
.msg.user{align-self:flex-end;background:var(--paper);color:var(--paper-ink)}
.msg.bot{align-self:flex-start;background:var(--raised);border:1px solid var(--line);padding-right:40px}
.msg.bot .msgcopy{position:absolute;top:4px;right:4px;opacity:.6}
.msg.err{color:var(--warn)}
.msgthink{margin-bottom:8px;padding:6px 8px;background:var(--well);border:1px solid var(--line);border-radius:4px;font-size:12px;color:var(--ink-2)}
.msgthink summary{cursor:pointer;font-weight:600;user-select:none}
.msgthink div{margin-top:6px;max-height:220px;overflow-y:auto;white-space:pre-wrap}
.chatempty{margin:auto;color:var(--ink-2);text-align:center}
.chatin{display:flex;gap:8px;align-items:flex-end}
.chatin textarea{flex:1;min-height:44px;max-height:200px;resize:vertical}
.size{display:flex;align-items:center;gap:7px}
.size input{width:150px;accent-color:var(--paper)}
.size span{display:block;background:var(--ink-2);border-radius:50%}
.nums{font:400 12px var(--font-data);color:var(--ink-2);font-variant-numeric:tabular-nums}
.warn{color:var(--warn);font-size:12px}
.grow{flex:1}
.model{width:auto;height:36px;padding-block:0}
.btn{display:inline-flex;align-items:center;gap:6px;background:var(--raised);border:1px solid var(--line);border-radius:4px;padding:7px 12px;cursor:pointer}
.btn:hover{border-color:var(--ink-2)}
.btn[aria-expanded="true"]{background:var(--paper);color:var(--paper-ink);border-color:var(--paper)}
.dotwarn{width:8px;height:8px;border-radius:50%;background:var(--warn)}
#gen{min-width:150px;padding:8px 18px;background:var(--paper);color:var(--paper-ink);font-weight:600;border:0;border-radius:4px}
#gen:disabled{opacity:.4;cursor:not-allowed}

.job{position:absolute;left:50%;top:10px;transform:translateX(-50%);z-index:3;width:min(460px,calc(100% - 20px));display:grid;gap:6px;padding:10px 12px;background:var(--surface);border:1px solid var(--line);border-radius:6px;box-shadow:0 8px 24px #00000080}
.job[data-kind="error"]{border-color:var(--warn)}
.job[data-kind="error"] .job-title{color:var(--warn)}
.job-head{display:flex;align-items:baseline;gap:10px}
.job-title{font-size:14px;font-weight:600;flex:1;min-width:0}
.job-time{font:500 13px var(--font-data);font-variant-numeric:tabular-nums}
.job-close{align-self:center;width:24px;height:24px;padding:0;background:transparent;border:0;border-radius:3px;cursor:pointer;display:grid;place-items:center}
.job-close:hover{background:var(--raised)}
.job-close svg{width:13px;height:13px}
.job-sub{font-size:12px;color:var(--ink-2);overflow-wrap:anywhere}
.job-sub:empty{display:none}
.job-bar{height:6px;background:var(--well);border-radius:3px;overflow:hidden}
.job-bar span{display:block;height:100%;width:0;background:var(--paper);border-radius:3px;transition:width .3s linear}
.job-steps{display:flex;flex-wrap:wrap;gap:4px 12px;list-style:none;margin:0;padding:0;font-size:11px;color:var(--ink-2)}
.job-steps li[data-state="done"]{color:var(--ink)}
.job-steps li[data-state="done"]::before{content:"✓ "}
.job-steps li[data-state="now"]{color:var(--paper-ink);background:var(--paper);border-radius:3px;padding:0 5px;font-weight:600}
@media (prefers-reduced-motion:reduce){.job-bar span{transition:none}}
.status{position:absolute;left:8px;bottom:8px;z-index:2;max-width:calc(100% - 16px);padding:3px 8px;font-size:12px;color:#fff;background:#000000b8;border-radius:3px;pointer-events:none;overflow-wrap:anywhere}
.status:empty{display:none}
.results{display:flex;gap:6px;overflow-x:auto;flex:none;padding-bottom:2px}
.results button{flex:none;width:76px;height:76px;padding:0;background:var(--well);border:1px solid var(--line);border-radius:4px;cursor:pointer;overflow:hidden}
.results button:hover{border-color:var(--paper)}
.results img{width:100%;height:100%;object-fit:cover;display:block}
.viewer{position:absolute;inset:0;z-index:4;display:grid;grid-template-rows:minmax(0,1fr) auto;gap:8px;padding:8px;background:var(--well)}
.viewer img{width:100%;height:100%;min-height:0;object-fit:contain;display:block}
.viewer-actions{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px}
a.btn{color:inherit;text-decoration:none}

.drawer{width:392px;display:grid;grid-template-rows:auto minmax(0,1fr);background:var(--surface);border-radius:4px;min-height:0}
.drawer-head{display:flex;align-items:center;justify-content:space-between;padding:6px 6px 6px 12px}
.drawer-head h2{font-size:13px;font-weight:600}
.drawer-head .ibtn{width:32px;height:28px}
.drawer-body{overflow:auto;padding:0 12px 12px;min-height:0;display:grid;gap:10px;align-content:start}
.problems{display:grid;gap:3px;margin:0;padding:6px 9px;list-style:none;border:1px solid var(--warn);border-radius:4px;color:var(--warn);font-size:12px}
.problems:empty{display:none}
.none{color:var(--ink-2)}
.pass{display:grid;gap:8px;border-top:1px solid var(--line);padding-top:10px}
.pass-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
.pass h3{font-size:13px;font-weight:600}
.pass-actions{display:flex;gap:2px}
.pass-actions .ibtn{width:30px;height:26px}
.pass-actions .ibtn svg{width:16px;height:16px}
.slots{display:flex;gap:6px;overflow-x:auto;padding-bottom:2px}
.slots figure{margin:0;flex:none;width:72px}
.slots canvas{display:block;width:72px;height:48px;object-fit:contain;background:var(--well);border:1px solid var(--line)}
.slots figcaption{font:500 11px var(--font-data);margin-top:3px}
.lines{list-style:none;margin:0;padding:0;display:grid;gap:9px}
.en{font:400 12px/1.5 var(--font-data);overflow-wrap:anywhere}
.vi{color:var(--ink-2);font-size:12px;margin-top:2px;overflow-wrap:anywhere}
.tok{background:var(--raised);border-radius:2px;padding:0 3px}
.area{box-shadow:inset 0 -3px 0 var(--c)}
.pass textarea.field{height:136px;resize:vertical}

@media (max-width:1100px){
  .drawer{position:absolute;right:0;top:0;bottom:0;width:min(392px,100%);z-index:6;box-shadow:0 12px 32px #00000090}
}
@media (max-width:780px){
  .main{grid-template-columns:minmax(0,1fr);grid-template-rows:auto minmax(0,1fr)}
  .strip{width:auto;flex-direction:row;flex-wrap:wrap;overflow:visible;align-items:center;gap:2px}
  .tiles{display:flex;gap:8px;padding:4px}
  .tile,.add,.env{width:42px;height:42px;margin-block:0}
  .strip hr{display:none}
  .ibtn{width:38px}
  .pop{width:min(100%,320px)}
  .sheetpanel{width:auto;overflow:visible}
  .sheetbox{height:110px}
  .size input{width:110px}
  #gen{flex:1}
}
</style>

<div class="app">
  <h1 class="sr">Concept Image Edit</h1>
  <div class="tabs" role="tablist" aria-label="Chế độ">
    <button class="tab" id="tabEdit" type="button" role="tab" aria-selected="true">Sửa ảnh</button>
    <button class="tab" id="tabSheet" type="button" role="tab" aria-selected="false">Char sheet</button>
    <button class="tab" id="tabRotate" type="button" role="tab" aria-selected="false">Xoay</button>
    <button class="tab" id="tabSwap" type="button" role="tab" aria-selected="false">Đổi người</button>
    <button class="tab" id="tabUpscale" type="button" role="tab" aria-selected="false">Upscale</button>
    <button class="tab" id="tabChat" type="button" role="tab" aria-selected="false">Chat</button>
  </div>
  <div class="main" id="main">
    <nav class="strip" id="strip" aria-label="Cọ và công cụ">
      <div class="tiles" id="tiles"></div>
      <button class="add" id="addBrush" type="button" title="Thêm cọ (+)" aria-label="Thêm cọ" data-icon="plus"></button>
      <hr>
      <button class="env" id="envBtn" type="button" title="Môi trường: ảnh cho model hiểu cảnh khi kéo rộng" aria-label="Môi trường" aria-expanded="false" aria-controls="envPop"><span data-icon="mountain"></span><img id="envThumb" alt="" hidden><span class="envcount" id="envCount" hidden></span></button>
      <hr>
      <button class="ibtn" id="selectTool" type="button" title="Chọn vật bằng SAM 3.1: kéo khung hoặc bấm lên vật" aria-label="Chọn vật" aria-pressed="false" data-icon="select"></button>
      <div class="tiles" id="layerTiles"></div>
      <label class="ibtn" title="Thêm layer từ ảnh khác"><span data-icon="layerAdd"></span><input class="sr" id="layerFile" type="file" accept="image/*" aria-label="Thêm layer từ ảnh khác"></label>
      <hr>
      <button class="ibtn" id="erase" type="button" title="Tẩy nét (E)" aria-label="Tẩy nét" aria-pressed="false" data-icon="eraser"></button>
      <button class="ibtn" id="undo" type="button" title="Hoàn tác (Ctrl+Z)" aria-label="Hoàn tác" data-icon="undo"></button>
      <button class="ibtn" id="redo" type="button" title="Làm lại (Ctrl+Y)" aria-label="Làm lại" data-icon="redo"></button>
      <button class="ibtn" id="clearStrokes" type="button" title="Xoá hết nét" aria-label="Xoá hết nét" data-icon="trash"></button>
      <label class="ibtn" title="Đổi ảnh gốc"><span data-icon="image"></span><input class="sr" id="baseFile" type="file" accept="image/*" aria-label="Đổi ảnh gốc"></label>
    </nav>

    <section class="sheetpanel" id="sheetPanel" aria-label="Char sheet" hidden>
      <div class="sheetslots" id="sheetSlots"></div>
      <div class="viewchips" id="sheetViews" role="group" aria-label="Góc nhìn"></div>
      <textarea class="field" id="sheetNote" placeholder="Mô tả thêm (tiếng Anh)" aria-label="Mô tả thêm"></textarea>
      <select class="field" id="sheetSize" aria-label="Cỡ bảng nhân vật" title="Càng lớn càng chậm">
        <option value="1408x768">1408 × 768</option>
        <option value="2048x1152" selected>2048 × 1152</option>
        <option value="2720x1536">2720 × 1536</option>
      </select>
    </section>

    <section class="sheetpanel" id="rotatePanel" aria-label="Xoay vật thể" hidden>
      <div class="viewchips" id="rotMode" role="group" aria-label="Cách xoay">
        <button class="viewchip" type="button" data-mode="dummy" aria-pressed="true" title="Khối giả để biết hướng, xoay theo 8 hướng">Dummy</button>
        <button class="viewchip" type="button" data-mode="splat" aria-pressed="false" title="Dựng khối 3D từ ảnh, xoay góc tuỳ ý">Splat</button>
      </div>
      <div class="sheetslots rotslots" id="rotateSlots"></div>
      <p class="envhint" id="rotHint"></p>
    </section>

    <section class="sheetpanel" id="chatPanel" aria-label="Chat" hidden>
      <div class="viewchips" id="chatMode" role="group" aria-label="Việc cần làm"></div>
      <p class="envhint" id="chatHint"></p>
      <div class="viewchips">
        <button class="viewchip" id="chatThink" type="button" aria-pressed="false" title="Bật hoặc tắt suy nghĩ từng bước trước khi trả lời">Suy nghĩ sâu</button>
        <button class="viewchip" id="chatClear" type="button">Cuộc trò chuyện mới</button>
      </div>
      <p class="envhint">Model Qwen3.8-27B (Huihui Abliterated FP8) chạy trên H200.</p>
    </section>

    <section class="sheetpanel" id="swapPanel" aria-label="Đổi người" hidden>
      <div class="viewchips" id="swapMode" role="group" aria-label="Đổi gì">
        <button class="viewchip" type="button" data-mode="head" aria-pressed="true" title="Thay cả đầu, gồm tóc">Đổi mặt</button>
        <button class="viewchip" type="button" data-mode="body" aria-pressed="false" title="Thay cả người, giữ dáng và cảnh">Thay nhân vật</button>
      </div>
      <div class="sheetslots swapslots" id="swapSlots"></div>
      <p class="envhint" id="swapHint"></p>
      <div class="swappick" id="swapPick" hidden>
        <div class="swappick-head"><span class="swapswatch" aria-hidden="true"></span><strong>Tô chọn người</strong><span>(không bắt buộc)</span></div>
        <p class="envhint">Cảnh đông người thì tô lên người cần thay. Không tô thì thay người chính trong cảnh.</p>
        <div class="viewchips" id="swapTools" role="group" aria-label="Chọn người cần thay">
          <button class="viewchip" type="button" data-tool="paint" aria-pressed="false">Cọ</button>
          <button class="viewchip" type="button" data-tool="erase" aria-pressed="false">Tẩy</button>
          <button class="viewchip" type="button" id="swapClear">Xoá hết</button>
        </div>
        <label class="swapsize">Cỡ cọ<input id="swapSize" type="range" min="8" max="300" value="60" aria-label="Cỡ cọ tô chọn người"></label>
      </div>
    </section>

    <section class="sheetpanel" id="upscalePanel" aria-label="Upscale" hidden>
      <div class="sheetslots upslots" id="upscaleSlots"></div>
      <div class="viewchips" id="upscaleSize" role="group" aria-label="Cỡ kết quả">
        <button class="viewchip" type="button" data-mp="4" aria-pressed="true" title="Kết quả khoảng 4 triệu điểm ảnh, thường vẽ lại 2 ô">4 MP</button>
        <button class="viewchip" type="button" data-mp="8" aria-pressed="false" title="Kết quả khoảng 8 triệu điểm ảnh, thường vẽ lại 4 ô">8 MP</button>
      </div>
      <div class="viewchips" id="upscaleSpeed" role="group" aria-label="Tốc độ">
        <button class="viewchip" type="button" data-fast="1" aria-pressed="true" title="8 bước, có LoRA Viggle Turbo">Nhanh</button>
        <button class="viewchip" type="button" data-fast="0" aria-pressed="false" title="25 bước, không Turbo, lâu gấp khoảng 3 lần">Kỹ</button>
      </div>
      <p class="envhint" id="upscaleHint"></p>
    </section>

    <div class="stage" id="stage">
      <div class="sheet-empty" id="sheetEmpty" hidden>Bảng nhân vật sẽ hiện ở đây</div>
      <div class="rot" id="rot" hidden>
        <div class="rotview" id="rotView" title="Kéo để xoay">
          <canvas id="rotCanvas"></canvas>
          <div class="rotframe" id="rotFrame" hidden></div>
          <div class="rotnote" id="rotNote" hidden>Giả lập để xem giao diện, chưa phải splat thật</div>
          <div class="rotmsg" id="rotMsg" hidden>
            <span id="rotMsgText"></span>
            <span class="bar-track" id="rotMsgBar" hidden><span class="bar-fill" id="rotMsgFill"></span></span>
          </div>
        </div>
        <div class="rotctl" id="rotDummyCtl">
          <div class="rotrow" id="rotDist" role="group" aria-label="Cự ly"></div>
        </div>
        <div class="rotctl" id="rotSplatCtl" hidden>
          <div class="rotrow">
            <label class="rotslider"><span aria-hidden="true">↔</span><input id="rotAzRange" type="range" min="-180" max="180" step="1" value="0" aria-label="Xoay ngang"></label>
            <label class="rotslider"><span aria-hidden="true">↕</span><input id="rotElRange" type="range" min="-80" max="80" step="1" value="0" aria-label="Xoay dọc"></label>
            <button class="viewchip" id="rotReset" type="button">Chính diện</button>
            <button class="viewchip" id="rotRetry" type="button" hidden>Dựng lại</button>
          </div>
        </div>
      </div>
      <div class="chat" id="chat" hidden>
        <div class="chatlog" id="chatLog" aria-live="polite"><p class="chatempty" id="chatEmpty">Gõ hoặc dán nội dung bên dưới</p></div>
        <div class="chatin">
          <textarea class="field" id="chatInput" placeholder="Nhập nội dung, Enter để gửi, Shift+Enter xuống dòng" aria-label="Nội dung chat"></textarea>
          <button class="btn" id="chatSend" type="button">Gửi</button>
        </div>
      </div>
      <div class="swap" id="swap" hidden>
        <div class="swapview" id="swapView">
          <canvas id="swapCanvas"></canvas>
          <div class="rotmsg" id="swapMsg">Bỏ ảnh cảnh vào</div>
          <div class="swapring" id="swapRing" hidden></div>
        </div>
      </div>
      <div class="upscale" id="upscale" hidden>
        <div class="upview" id="upView">
          <canvas id="upCanvas"></canvas>
          <div class="rotmsg" id="upMsg">Bỏ ảnh cần upscale vào</div>
          <div class="upinfo" id="upInfo" hidden></div>
          <div class="uptools" id="upTools" hidden>
            <button class="viewchip" id="upFit" type="button" title="Cả ảnh vừa khung">Vừa khung</button>
            <button class="viewchip" id="upOne" type="button" title="Xem đúng từng điểm ảnh của kết quả">100%</button>
            <button class="viewchip" id="upOpen" type="button" title="Mở ảnh kết quả để tải về hoặc sửa tiếp">Xem ảnh</button>
          </div>
          <div class="rotnote" id="upNote" hidden>Giả lập để xem giao diện, chưa phải ảnh upscale thật</div>
        </div>
      </div>
      <div class="frame" id="frame">
        <canvas id="view"></canvas>
        <div class="capbox" id="capbox" hidden></div>
        <div class="imgedge" id="imgedge"></div>
        <span class="padtag" data-tag="left"></span>
        <span class="padtag" data-tag="right"></span>
        <span class="padtag" data-tag="top"></span>
        <span class="padtag" data-tag="bottom"></span>
        <div class="edge" data-side="left" title="Kéo để nới"></div>
        <div class="edge" data-side="right" title="Kéo để nới"></div>
        <div class="edge" data-side="top" title="Kéo để nới"></div>
        <div class="edge" data-side="bottom" title="Kéo để nới"></div>
      </div>
      <div class="brushcur" id="brushcur" hidden></div>
      <button class="tag" id="sampleTag" type="button" title="Xoá ảnh mẫu">Ảnh mẫu <span data-icon="close"></span></button>
      <label class="drop" id="drop" hidden>
        <span data-icon="plus"></span>
        <span>Ảnh gốc</span>
        <input class="sr" id="baseFile2" type="file" accept="image/*" aria-label="Chọn ảnh gốc">
      </label>
      <div class="status" id="status" role="status"></div>
      <div class="selwrap" id="selWrap" hidden>
      <div class="capture-hint" id="selHint" hidden>
        <input class="field selname" id="selName" type="text" placeholder="Tìm theo tên (tiếng Anh)" title="Tên vật bằng tiếng Anh. Nhiều loại thì cách nhau bằng dấu phẩy, thêm :số sau tên để giới hạn số lượng" aria-label="Tìm vật theo tên">
        <button id="selFind" type="button">Tìm</button>
        <button id="selHelp" type="button" title="Kéo khung quanh vật hoặc bấm lên vật. Shift + kéo để thêm vật, Shift + bấm để bỏ phần thừa. Gõ tên tiếng Anh rồi Enter để tìm mọi vật cùng loại." aria-label="Cách chọn vật">?</button>
        <button id="selExit" type="button">Xong</button>
      </div>
      <div class="floatbar" id="selBar" hidden>
        <span id="selText">Đã chọn vật</span>
        <button class="viewchip" type="button" id="selRemove" title="Biến vùng chọn thành vùng tô của một cọ Xoá">Xoá vật</button>
        <button class="viewchip" type="button" id="selLayer" title="Cắt vật ra thành layer, chỗ cũ thành xám để lượt Ghép lấp lại">Tách thành layer</button>
        <button class="viewchip" type="button" id="selCopy" title="Chép vật thành layer, vật vẫn ở chỗ cũ">Chép thành layer</button>
        <button class="viewchip" type="button" id="selClear">Bỏ chọn</button>
        <label class="selsoft" title="Nới vùng chọn rộng ra, tính bằng điểm ảnh">Nới rộng<input id="selGrow" type="range" min="0" max="40" step="1" value="8" aria-label="Nới rộng vùng chọn"></label>
        <label class="selsoft" title="Độ mềm của viền vùng chọn, tính bằng điểm ảnh">Mềm viền<input id="selFeather" type="range" min="0" max="30" step="1" value="3" aria-label="Mềm viền vùng chọn"></label>
      </div>
      </div>
      <div class="floatbar" id="layerBar" hidden>
        <div class="viewchips" id="layerMode" role="group" aria-label="Cách chỉnh layer">
          <button class="viewchip" type="button" data-mode="transform" aria-pressed="true" title="Kéo để dời, kéo góc để bẻ phối cảnh, Shift + kéo góc để đổi cỡ, kéo nút tròn để xoay">Biến dạng</button>
          <button class="viewchip" type="button" data-mode="warp" aria-pressed="false" title="Kéo 9 điểm để uốn layer">Uốn</button>
        </div>
        <button class="viewchip" type="button" id="layerFlip">Lật ngang</button>
        <button class="viewchip" type="button" id="layerCut" title="Tách nền layer bằng BiRefNet">Tách nền</button>
        <button class="viewchip" type="button" id="layerKeep" aria-pressed="false" title="Bật: vật vẫn ở chỗ cũ. Tắt: chỗ cũ thành xám, lượt Ghép lấp lại">Giữ chỗ cũ</button>
        <button class="viewchip" type="button" id="blendLora" aria-pressed="false" title="LoRA Qwen-image-2.1 光影溶图 của RunningHub, thử nghiệm">LoRA ánh sáng</button>
        <button class="viewchip" type="button" id="layerSave" title="Tải layer như đang thấy, PNG nền trong suốt">Tải PNG</button>
        <button class="viewchip" type="button" id="layerToSwap" title="Dùng layer làm ảnh mặt hoặc ảnh nhân vật ở tab Đổi người">Sang Đổi người</button>
        <button class="viewchip" type="button" id="layerToUp" title="Dùng layer làm ảnh cần phóng ở tab Upscale">Sang Upscale</button>
        <span>Bấm Gen để ghép vào ảnh</span>
      </div>
      <div class="capture-hint" id="captureHint" hidden>
        <span>Kéo khung quanh vùng muốn clone</span>
        <button id="captureCancel" type="button">Huỷ</button>
      </div>
      <div class="job" id="job" role="status" aria-live="polite" hidden>
        <div class="job-head">
          <span class="job-title" id="jobTitle"></span>
          <span class="job-time" id="jobTime"></span>
          <button class="job-close" id="jobClose" type="button" title="Đóng" aria-label="Đóng thông báo" data-icon="close" hidden></button>
        </div>
        <div class="job-sub" id="jobSub"></div>
        <div class="job-bar" id="jobBarWrap"><span id="jobBar"></span></div>
        <ol class="job-steps" id="jobSteps"></ol>
      </div>
      <div class="viewer" id="viewer" hidden>
        <img id="viewerImg" alt="Ảnh kết quả">
        <div class="viewer-actions">
          <button class="btn" id="viewerUse" type="button">Sửa tiếp từ ảnh này</button>
          <a class="btn" id="viewerSave" download="concept_edit.png">Tải về</a>
          <button class="btn" id="viewerClose" type="button">Đóng</button>
        </div>
      </div>
    </div>

    <aside class="drawer" id="drawer" aria-label="Prompt" hidden>
      <div class="drawer-head">
        <h2>Prompt</h2>
        <button class="ibtn" id="drawerClose" type="button" title="Đóng" aria-label="Đóng" data-icon="close"></button>
      </div>
      <div class="drawer-body">
        <ul class="problems" id="problems"></ul>
        <p class="none" id="noPrompt">Tô hoặc kéo rộng để tạo prompt.</p>
        <div class="lines" id="passes"></div>
      </div>
    </aside>

    <div class="pop" id="pop" role="dialog" aria-label="Tuỳ chỉnh cọ" hidden>
      <div class="types" id="popTypes"></div>
      <p class="typehint" id="popHint"></p>
      <div class="detail" id="popDetail">
        <div class="imgwrap" id="popImgWrap">
          <label class="imgpick" title="Ảnh tham chiếu">
            <span data-icon="plus" id="popImgPlus"></span>
            <img id="popImg" alt="" hidden>
            <input class="sr" id="popFile" type="file" accept="image/*" aria-label="Ảnh tham chiếu">
          </label>
          <button class="imgx" id="popImgClear" type="button" title="Bỏ ảnh" aria-label="Bỏ ảnh" data-icon="close" hidden></button>
        </div>
        <textarea class="field" id="popNote" aria-label="Mô tả"></textarea>
      </div>
      <div class="popfoot">
        <div class="popfoot" id="popSwatches"></div>
        <button class="ibtn" id="popDelete" type="button" title="Xoá cọ này" aria-label="Xoá cọ này" data-icon="trash"></button>
      </div>
    </div>

    <div class="pop" id="envPop" role="dialog" aria-label="Môi trường" hidden>
      <div class="envslots" id="envSlots"></div>
      <textarea class="field" id="scene" placeholder="Mô tả cảnh, nếu cần (tiếng Anh)" aria-label="Mô tả cảnh"></textarea>
      <p class="envhint">Model xem các ảnh này khi kéo rộng để biết xung quanh có gì.</p>
      <div class="envlora">
        <span>LoRA kéo rộng</span>
        <div class="viewchips" id="envLora" role="group" aria-label="LoRA kéo rộng">
          <button class="viewchip" type="button" data-strength="0" aria-pressed="false">Tắt</button>
          <button class="viewchip" type="button" data-strength="0.6" aria-pressed="false">0,6</button>
          <button class="viewchip" type="button" data-strength="1" aria-pressed="true">1</button>
        </div>
      </div>
    </div>
  </div>

  <div class="results" id="results" aria-label="Kết quả" hidden></div>

  <div class="bar">
    <label class="size edit-only" title="Cỡ cọ ([ ])">
      <span style="width:6px;height:6px"></span>
      <input id="size" type="range" min="8" max="400" value="72" aria-label="Cỡ cọ">
      <span style="width:14px;height:14px"></span>
    </label>
    <span class="nums edit-only" id="readout"></span>
    <span class="warn edit-only" id="sizeNote"></span>
    <span class="grow"></span>
    <select class="field model edit-only" id="resolution" aria-label="Độ nét khi gen" title="Càng lớn vùng sửa càng nét và càng chậm. Ảnh ra luôn giữ cỡ ảnh đang sửa.">
      <option value="1.5">Gen ở 1,5 MP</option>
      <option value="4">Gen ở 4 MP</option>
      <option value="0">Gen ở cỡ gốc</option>
    </select>
    <span class="qstate" id="qState" title="Hàng đợi chung của ComfyUI trên H200, mọi họa sĩ dùng chung, chạy từng việc một"><span class="qdot" aria-hidden="true"></span><span id="qText">Chưa kết nối ComfyUI</span></span>
    <select class="field model" id="model" aria-label="Model">
      <option>Qwen 2.1</option>
      <option disabled>Flux 2 (chưa có model)</option>
    </select>
    <button class="btn" id="promptBtn" type="button" aria-expanded="false" aria-controls="drawer"><span data-icon="lines"></span>Prompt<span class="dotwarn" id="promptDot" hidden></span></button>
    <button id="gen" type="button" disabled title="Chưa nối ComfyUI">Gen</button>
  </div>
</div>

<script>
'use strict';
const GRID = 32;
const PAD_GRAY = '#808080';
const MAX_PAD = 8192;

const ICONS = {
  plus: 'M10 4v12M4 10h12',
  close: 'M5 5l10 10M15 5L5 15',
  person: 'M10 3a3 3 0 110 6 3 3 0 010-6zM4 17c.8-3.6 3.2-5 6-5s5.2 1.4 6 5',
  add: 'M3 3h9v9H3zM14.5 12v6M11.5 15h6',
  personAdd: 'M8 3a3 3 0 110 6 3 3 0 010-6zM2 17c.8-3.6 3-5 6-5 1.800 0 3.300.5 4.400 1.700M15.500 6v6M12.500 9h6',
  clone: 'M7 7h10v10H7zM13 7V3H3v10h4',
  replace: 'M4 7h11l-3-3M16 13H5l3 3',
  capture: 'M3 7V3h4M13 3h4v4M17 13v4h-4M7 17H3v-4',
  remove: 'M3 3h9v9H3zM12 12l5.5 5.5M17.5 12L12 17.5',
  mountain: 'M2 16l5.5-9 3.5 5.5 2-3 5 6.5z',
  pencil: 'M4 16l1-4 8-8 3 3-8 8zM11.5 5.5l3 3',
  eraser: 'M8 16l-4-4 7.5-7.5a1.4 1.4 0 012 0l3.5 3.5a1.4 1.4 0 010 2L11 16zM6 16h10',
  undo: 'M7 5L3.5 8.5 7 12M3.5 8.5H12a4.5 4.5 0 010 9H8',
  redo: 'M13 5l3.5 3.5L13 12M16.5 8.5H8a4.5 4.5 0 000 9h4',
  trash: 'M4 6h12M8 6V4h4v2M6 6l.8 10.5h6.4L14 6',
  image: 'M3 4h14v12H3zM3 13l4-4 3 3 3-2 4 4M13 6.3a1.2 1.2 0 110 2.4 1.2 1.2 0 010-2.4z',
  lines: 'M4 5h12M4 10h12M4 15h8',
  copy: 'M7 7h9v10H7zM4 13V3h9',
  check: 'M4 10l4 4 8-9',
  select: 'M3 6V3h3M9 3h2M14 3h3v3M3 9v2M3 14v3h3M10 10l7 3-3 1.2-1.2 3z',
  layerAdd: 'M3 8l6.5-3.5L16 8l-6.5 3.5zM3 12l6.5 3.5 2.5-1.300M15.5 12.500v5M13 15h5',
};
const icon = name => `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${ICONS[name]}"/></svg>`;
document.querySelectorAll('[data-icon]').forEach(el => { el.innerHTML = icon(el.dataset.icon); });

const PALETTE = [
  { id: 'red', vi: 'Đỏ', area: 'đỏ', en: 'red', hex: '#FF1E1E', on: '#111' },
  { id: 'blue', vi: 'Xanh', area: 'xanh', en: 'blue', hex: '#1464FF', on: '#fff' },
  { id: 'purple', vi: 'Tím', area: 'tím', en: 'purple', hex: '#A21EFF', on: '#fff' },
  { id: 'yellow', vi: 'Vàng', area: 'vàng', en: 'yellow', hex: '#FFE500', on: '#111' },
  { id: 'green', vi: 'Xanh lá', area: 'xanh lá', en: 'green', hex: '#00B84A', on: '#111' },
  { id: 'orange', vi: 'Cam', area: 'cam', en: 'orange', hex: '#FF8A00', on: '#111' },
];
const PAINT = Object.fromEntries(PALETTE.map(c => [c.id, c]));

// Bốn việc, không phân biệt người hay vật: ảnh trong hộp cọ và mô tả cho model biết đó là gì.
const TYPES = {
  add: { label: 'Thêm', icon: 'add', image: true, ph: 'Thêm gì, nếu không có ảnh (tiếng Anh)', hint: 'Đặt thứ trong ảnh vào vùng tô.' },
  replace: { label: 'Thay', icon: 'replace', image: true, ph: 'Đổi thành gì, nếu không có ảnh (tiếng Anh)', hint: 'Giữ dáng và vị trí, đổi thứ đang có trong vùng tô thành thứ trong ảnh.' },
  remove: { label: 'Xoá', icon: 'remove', image: false, ph: 'Xoá gì, nếu cần (tiếng Anh)', hint: 'Xoá thứ trong vùng tô, lấp bằng nền xung quanh.' },
  clone: { label: 'Clone', icon: 'clone', image: true, capture: true, ph: 'Mô tả, nếu cần (tiếng Anh)', hint: 'Chụp một vùng trên ảnh, rồi tô nơi muốn chép tới.' },
};

const joinTags = (tags, word = 'and') => (tags.length > 1 ? `${tags.slice(0, -1).join(', ')} ${word} ${tags[tags.length - 1]}` : tags[0]);
const PROMPTS = {
  outpaint: {
    en: 'Outpaint the image: replace the solid gray areas with a seamless continuation of the scene, keeping the existing picture unchanged.',
    vi: 'Nới rộng ảnh: thay các vùng xám đặc bằng phần cảnh nối liền, giữ nguyên phần ảnh đang có.',
  },
  scene: { en: t => `Scene: ${t}.`, vi: t => `Mô tả cảnh: ${t}.` },
  env: {
    en: tags => (tags.length > 1
      ? `${joinTags(tags)} show the environment this scene belongs to. Use them to understand what exists around the picture and continue the scene with that content, without copying their framing. Keep the lighting and perspective of <image1>.`
      : `${tags[0]} shows the environment this scene belongs to. Use it to understand what exists around the picture and continue the scene with that content, without copying its framing. Keep the lighting and perspective of <image1>.`),
    vi: tags => `${joinTags(tags, 'và')} cho biết môi trường của cảnh này. Dùng để hiểu xung quanh có gì và vẽ tiếp theo đó, không chép khung hình. Giữ ánh sáng và phối cảnh của <image1>.`,
  },
  add: {
    en: (c, r) => `Place what <image${r}> shows inside the ${c.en} outline in <image1>.`,
    vi: (c, r) => `Đặt thứ trong <image${r}> vào viền ${c.area} trên <image1>.`,
  },
  addText: {
    en: (c, n) => `Add ${n} inside the ${c.en} outline in <image1>.`,
    vi: (c, n) => `Thêm ${n} vào viền ${c.area} trên <image1>.`,
  },
  replace: {
    en: (c, r) => `Keep the position, pose, size and camera angle of what is inside the ${c.en} outline in <image1>, and change it into what <image${r}> shows.`,
    vi: (c, r) => `Giữ vị trí, dáng, cỡ và góc máy của thứ trong viền ${c.area} trên <image1>, đổi nó thành thứ trong <image${r}>.`,
  },
  clone: {
    en: (c, r) => `Copy all of what <image${r}> shows into the ${c.en} outline in <image1>, changing only its lighting, shadows and perspective to fit <image1>.`,
    vi: (c, r) => `Chép toàn bộ thứ trong <image${r}> vào viền ${c.area} trên <image1>, chỉ đổi ánh sáng, bóng đổ và phối cảnh cho hợp <image1>.`,
  },
  remove: {
    en: (c, n) => `Remove ${n || 'what is'} inside the ${c.en} outline in <image1> and fill it with the surrounding background.`,
    vi: (c, n) => `Xoá ${n || 'thứ'} trong viền ${c.area} trên <image1> và lấp bằng nền xung quanh.`,
  },
  note: {
    en: (c, n) => `Inside the ${c.en} outline: ${n}.`,
    vi: (c, n) => `Trong viền ${c.area}: ${n}.`,
  },
  looks: {
    en: tags => (tags.length > 1
      ? `Use ${joinTags(tags)} only for how things look: do not copy their framing, layout or lighting, and if one shows several views of the same subject, draw it once.`
      : `Use ${tags[0]} only for how things look: do not copy its framing, layout or lighting, and if it shows several views of the same subject, draw it once.`),
    vi: tags => `Chỉ dùng ${joinTags(tags, 'và')} để biết trông thế nào: không chép khung hình, bố cục hay ánh sáng, nếu có nhiều góc nhìn của cùng một thứ thì chỉ vẽ một lần.`,
  },
  finish: [
    { en: 'Stay inside the outline and match the lighting, shadows and perspective of <image1>. Remove the outline.', vi: 'Không vẽ ra ngoài viền, khớp ánh sáng, bóng đổ và phối cảnh của <image1>. Xoá viền đi.' },
    { en: 'Stay inside each outline and match the lighting, shadows and perspective of <image1>. Remove the outlines.', vi: 'Không vẽ ra ngoài các viền, khớp ánh sáng, bóng đổ và phối cảnh của <image1>. Xoá các viền đi.' },
  ],
  keep: { en: 'Keep everything else exactly the same.', vi: 'Giữ nguyên mọi thứ còn lại.' },
};
const line = (p, c, v) => ({ en: p.en(c, v), vi: p.vi(c, v) });

// Mọi cọ gom vào một lượt trên cả shot. Màu chỉ để nhận diện, ý nghĩa nằm trong hộp của cọ.
function buildEditPass(marks, problems) {
  const lines = [], slots = [], looks = [], brushes = [];
  for (const m of marks) {
    const c = PAINT[m.color];
    const note = m.note.trim().replace(/[.\s]+$/, '');
    if (m.type === 'remove') {
      lines.push(line(PROMPTS.remove, c, note));
    } else if (m.hasRef) {
      const n = slots.length + 2;
      lines.push(line(PROMPTS[m.type], c, n));
      if (note) lines.push(line(PROMPTS.note, c, note));
      slots.push({ brush: m.brush, n });
      if (m.type !== 'clone') looks.push(`<image${n}>`);
    } else if (m.type === 'clone') {
      problems.push({ brush: m.brush, text: `${c.vi}: chưa chụp vùng` });
      continue;
    } else if (note) {
      lines.push(line(m.type === 'add' ? PROMPTS.addText : PROMPTS.note, c, note));
    } else {
      problems.push({ brush: m.brush, text: `${c.vi}: thiếu ảnh hoặc mô tả` });
      continue;
    }
    brushes.push(m.brush);
  }
  if (!brushes.length) return null;
  if (looks.length) lines.push({ en: PROMPTS.looks.en(looks), vi: PROMPTS.looks.vi(looks) });
  lines.push(PROMPTS.finish[brushes.length > 1 ? 1 : 0], PROMPTS.keep);
  return { kind: 'edit', name: 'Sửa', lines, slots, brushes };
}

function buildPasses({ marks, expanded, scene, envCount, expandLora, layers, blendLora }) {
  const problems = [];
  const passes = [];
  if (expanded) {
    const out = [{ ...PROMPTS.outpaint }];
    const s = scene.trim().replace(/[.\s]+$/, '');
    if (s) out.push({ en: PROMPTS.scene.en(s), vi: PROMPTS.scene.vi(s) });
    if (envCount) {
      const tags = Array.from({ length: envCount }, (_, i) => `<image${i + 2}>`);
      out.push({ en: PROMPTS.env.en(tags), vi: PROMPTS.env.vi(tags) });
    }
    const loraNote = expandLora === 1 ? '' : expandLora ? ` (LoRA ${String(expandLora).replace('.', ',')})` : ' (không LoRA)';
    passes.push({ kind: 'expand', name: `Kéo rộng${loraNote}`, lines: out, slots: [] });
  }
  if (layers && layers.length) passes.push(buildMergePass(layers, blendLora));
  const edit = buildEditPass(marks.filter(m => m.cover), problems);
  if (edit) passes.push(edit);
  return { passes, problems };
}

function snapPad(size, raw, other) {
  const value = Math.max(0, Math.min(MAX_PAD, raw));
  if (value < GRID / 2 && other === 0) return 0;
  const total = Math.round((size + other + value) / GRID) * GRID;
  return Math.max(0, total - size - other);
}

const $ = id => document.getElementById(id);
const stage = $('stage'), frame = $('frame'), view = $('view'), imgedge = $('imgedge'), brushcur = $('brushcur'), pop = $('pop');
const layer = document.createElement('canvas');

const S = {
  base: null, w: 0, h: 0,
  pads: { left: 0, right: 0, top: 0, bottom: 0 },
  strokes: [], past: [], future: [],
  brushes: [], nextId: 1, active: null, editing: null, erase: false, size: 72,
  padsPainted: false, capture: null, baseFromResult: false, isSample: false, tab: 'edit',
  scene: '',
  expandLora: 1,
  tool: null,
  blendLora: true,
  env: [0, 1, 2, 3].map(() => ({ ref: null, refUrl: '' })),
  overrides: { expand: null, edit: null },
};
const V = { scale: 1, x: 0, y: 0 };
// Vùng chọn của SAM 3.1 và các layer, toạ độ theo ảnh gốc (không tính vùng nới).
const Sel = { box: null, points: [], hits: [], mask: null, tint: null, ring: null, raw: null, clean: null, smooth: 1, grow: 8, feather: 3, busy: false, ticket: 0, upload: { for: null, name: '', k: 1 }, ready: false };
const Layers = { list: [], active: null, mode: 'transform', nextId: 1, busy: false };
const W = () => S.w + S.pads.left + S.pads.right;
const H = () => S.h + S.pads.top + S.pads.bottom;
const isExpanded = () => S.pads.left + S.pads.right + S.pads.top + S.pads.bottom > 0;
const needsOutpaint = () => isExpanded() && !S.padsPainted;
const brushOf = id => S.brushes.find(b => b.id === id);
const canPaint = () => !!S.base && !S.tool && (S.erase || !!brushOf(S.active));
const viewCursor = () => (S.capture !== null || S.tool === 'select' ? 'crosshair' : S.tool === 'layer' ? 'move' : canPaint() ? 'none' : 'default');

function drawStroke(g, s) {
  const b = brushOf(s.brush);
  if (!s.erase && !b) return;
  if (s.mask) {
    drawMaskStroke(g, s, b);
    return;
  }
  g.globalCompositeOperation = s.erase ? 'destination-out' : 'source-over';
  g.strokeStyle = s.erase ? '#000' : PAINT[b.color].hex;
  g.lineWidth = s.size;
  g.lineCap = 'round';
  g.lineJoin = 'round';
  g.beginPath();
  g.moveTo(s.points[0].x, s.points[0].y);
  if (s.points.length === 1) g.lineTo(s.points[0].x + 0.01, s.points[0].y);
  for (const p of s.points.slice(1)) g.lineTo(p.x, p.y);
  g.stroke();
  g.globalCompositeOperation = 'source-over';
}

function rebuildLayer() {
  layer.width = Math.max(1, W());
  layer.height = Math.max(1, H());
  const g = layer.getContext('2d');
  g.setTransform(1, 0, 0, 1, S.pads.left, S.pads.top);
  for (const s of S.strokes) drawStroke(g, s);
}

// Nét tô phủ hết vùng nới thì không còn điểm xám nào cho lượt kéo rộng.
function padsFullyPainted() {
  if (!isExpanded()) return false;
  const p = S.pads;
  const g = layer.getContext('2d');
  const strips = [
    [0, 0, p.left, H()],
    [W() - p.right, 0, p.right, H()],
    [p.left, 0, S.w, p.top],
    [p.left, H() - p.bottom, S.w, p.bottom],
  ];
  for (const [x, y, w, h] of strips) {
    if (!w || !h) continue;
    const alpha = g.getImageData(x, y, w, h).data;
    for (let i = 3; i < alpha.length; i += 4) {
      if (alpha[i] < 250) return false;
    }
  }
  return true;
}

function measureCover() {
  for (const b of S.brushes) b.cover = 0;
  if (!S.base || !S.brushes.length) return;
  const k = Math.min(1, 512 / Math.max(W(), H()));
  const w = Math.max(1, Math.ceil(W() * k)), h = Math.max(1, Math.ceil(H() * k));
  const small = document.createElement('canvas');
  small.width = w; small.height = h;
  const g = small.getContext('2d', { willReadFrequently: true });
  g.drawImage(layer, 0, 0, w, h);
  const data = g.getImageData(0, 0, w, h).data;
  const rgb = S.brushes.map(b => {
    const hex = PAINT[b.color].hex;
    return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
  });
  const counts = S.brushes.map(() => 0);
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 64) continue;
    let best = 0, bestD = Infinity;
    rgb.forEach((v, n) => {
      const d = (data[i] - v[0]) ** 2 + (data[i + 1] - v[1]) ** 2 + (data[i + 2] - v[2]) ** 2;
      if (d < bestD) { bestD = d; best = n; }
    });
    counts[best]++;
  }
  S.brushes.forEach((b, n) => { b.cover = counts[n] / (w * h); });
}

let raf = 0;
function schedule() {
  if (raf) return;
  raf = requestAnimationFrame(() => { raf = 0; renderView(); });
}
function renderView() {
  if (!S.base) return;
  const dpr = window.devicePixelRatio || 1;
  const cw = W() * V.scale, ch = H() * V.scale;
  view.width = Math.max(1, Math.round(cw * dpr));
  view.height = Math.max(1, Math.round(ch * dpr));
  view.style.width = cw + 'px';
  view.style.height = ch + 'px';
  view.style.cursor = viewCursor();
  const g = view.getContext('2d');
  g.setTransform(view.width / W(), 0, 0, view.height / H(), 0, 0);
  g.imageSmoothingQuality = 'high';
  g.fillStyle = PAD_GRAY;
  g.fillRect(0, 0, W(), H());
  g.drawImage(S.base, S.pads.left, S.pads.top, S.w, S.h);
  // Nét tô chỉ để artist thấy vùng chọn, nên vẫn phải nhìn được dáng bên dưới.
  g.globalAlpha = 0.5;
  g.drawImage(layer, 0, 0);
  g.globalAlpha = 1;
  drawOverlays(g);
}

function place() {
  if (!S.base) return;
  const s = V.scale, p = S.pads;
  frame.style.transform = `translate(${V.x}px,${V.y}px)`;
  frame.style.width = W() * s + 'px';
  frame.style.height = H() * s + 'px';
  imgedge.hidden = !isExpanded();
  imgedge.style.left = p.left * s + 'px';
  imgedge.style.top = p.top * s + 'px';
  imgedge.style.width = S.w * s + 'px';
  imgedge.style.height = S.h * s + 'px';
  const midX = (p.left + S.w / 2) * s, midY = (p.top + S.h / 2) * s;
  const spots = {
    left: [p.left * s / 2, midY, p.left],
    right: [(W() - p.right / 2) * s, midY, p.right],
    top: [midX, p.top * s / 2, p.top],
    bottom: [midX, (H() - p.bottom / 2) * s, p.bottom],
  };
  for (const side in spots) {
    const tag = frame.querySelector(`[data-tag="${side}"]`);
    const [x, y, v] = spots[side];
    const roomy = (side === 'left' || side === 'right') ? v * s >= 44 : v * s >= 20;
    tag.hidden = !v || !roomy;
    tag.textContent = '+' + v;
    tag.style.left = x + 'px';
    tag.style.top = y + 'px';
  }
  renderView();
}

function fit() {
  if (!S.base) return;
  const r = stage.getBoundingClientRect();
  const margin = 34;
  V.scale = Math.max(0.02, Math.min((r.width - 2 * margin) / W(), (r.height - 2 * margin) / H(), 1.5));
  V.x = (r.width - W() * V.scale) / 2;
  V.y = (r.height - H() * V.scale) / 2;
  place();
}

function snapshot() { return { strokes: S.strokes.slice(), pads: { ...S.pads }, brushes: S.brushes.slice() }; }
function pushHistory() {
  S.past.push(snapshot());
  if (S.past.length > 60) S.past.shift();
  S.future = [];
}
function restore(from, to) {
  if (!from.length) return;
  to.push(snapshot());
  const snap = from.pop();
  S.strokes = snap.strokes;
  S.pads = snap.pads;
  for (const b of snap.brushes) {
    if (brushOf(b.id) || S.brushes.length >= PALETTE.length) continue;
    if (S.brushes.some(o => o.color === b.color)) b.color = PALETTE.find(c => !S.brushes.some(o => o.color === c.id)).id;
    S.brushes.push(b);
  }
  S.brushes.sort((a, b) => a.id - b.id);
  renderTiles();
  afterChange();
  fit();
}

function renderTiles() {
  const wrap = $('tiles');
  wrap.replaceChildren();
  S.brushes.forEach((b, i) => {
    const c = PAINT[b.color], T = TYPES[b.type];
    const t = document.createElement('button');
    t.type = 'button';
    t.className = 'tile';
    t.dataset.id = b.id;
    t.style.setProperty('--c', c.hex);
    t.style.setProperty('--on', c.on);
    t.title = `${T.label} (${i + 1})`;
    t.setAttribute('aria-label', `${T.label}, màu ${c.vi}`);
    t.innerHTML = (T.image && b.ref ? `<img src="${b.refUrl}" alt="">` : icon(T.icon)) +
      `<span class="bang" hidden>!</span><span class="more">${icon('pencil')}</span>`;
    t.addEventListener('click', () => {
      if (S.editing === b.id) { closePop(); return; }
      selectBrush(b.id);
      openPop(b.id);
    });
    const x = document.createElement('button');
    x.type = 'button';
    x.className = 'tilex';
    x.title = 'Xoá cọ này (Ctrl+Z để lấy lại)';
    x.setAttribute('aria-label', `Xoá cọ ${T.label}, màu ${c.vi}`);
    x.innerHTML = icon('close');
    x.addEventListener('click', e => { e.stopPropagation(); deleteBrush(b.id); });
    const cell = document.createElement('div');
    cell.className = 'tilewrap';
    cell.append(t, x);
    wrap.append(cell);
  });
  $('addBrush').hidden = S.brushes.length >= PALETTE.length;
}

function selectBrush(id) {
  S.active = id;
  S.erase = false;
  if (S.tool) {
    S.tool = null;
    clearSelection();
    syncToolUi();
  }
  syncTools();
}
function syncTools() {
  $('erase').setAttribute('aria-pressed', String(S.erase));
  $('size').value = S.size;
  $('undo').disabled = !S.past.length;
  $('redo').disabled = !S.future.length;
  $('clearStrokes').disabled = !S.strokes.length;
  document.querySelectorAll('.tile').forEach(t => {
    const b = brushOf(Number(t.dataset.id));
    t.dataset.active = String(!S.erase && S.active === b.id);
    t.dataset.painted = String(b.cover > 0);
  });
  view.style.cursor = viewCursor();
}

function addBrush() {
  const free = PALETTE.find(c => !S.brushes.some(b => b.color === c.id));
  if (!free) return;
  const last = S.brushes[S.brushes.length - 1];
  const b = { id: S.nextId++, color: free.id, type: last && last.type !== 'clone' ? last.type : 'add', ref: null, refUrl: '', note: '', cover: 0 };
  S.brushes.push(b);
  renderTiles();
  selectBrush(b.id);
  renderPanel();
  openPop(b.id);
}
function deleteBrush(id) {
  pushHistory();
  S.brushes = S.brushes.filter(b => b.id !== id);
  S.strokes = S.strokes.filter(s => s.brush !== id);
  if (S.active === id) S.active = S.brushes.length ? S.brushes[S.brushes.length - 1].id : null;
  closePop();
  renderTiles();
  afterChange();
}

function buildPop() {
  $('popTypes').innerHTML = Object.entries(TYPES).map(([k, t]) =>
    `<button class="type" type="button" data-type="${k}" aria-pressed="false" title="${t.hint}">${icon(t.icon)}<span>${t.label}</span></button>`).join('');
  $('popTypes').addEventListener('click', e => {
    const btn = e.target.closest('.type');
    const b = brushOf(S.editing);
    if (!btn || !b) return;
    b.type = btn.dataset.type;
    syncPop();
    renderTiles();
    renderPanel();
    if (TYPES[b.type].capture && !b.ref) startCapture(b.id);
  });
  pop.querySelector('.imgpick').addEventListener('click', e => {
    const b = brushOf(S.editing);
    if (b && TYPES[b.type].capture) {
      e.preventDefault();
      startCapture(b.id);
    }
  });
  $('popSwatches').innerHTML = PALETTE.map(c =>
    `<button class="sw" type="button" data-color="${c.id}" style="--c:${c.hex}" title="${c.vi}" aria-label="Màu ${c.vi}"></button>`).join('');
  $('popSwatches').addEventListener('click', e => {
    const sw = e.target.closest('.sw');
    const b = brushOf(S.editing);
    if (!sw || !b) return;
    const other = S.brushes.find(o => o.color === sw.dataset.color);
    if (other && other !== b) other.color = b.color;
    b.color = sw.dataset.color;
    syncPop();
    renderTiles();
    afterChange();
  });
  $('popNote').addEventListener('input', e => {
    const b = brushOf(S.editing);
    if (b) { b.note = e.target.value; renderPanel(); }
  });
  $('popFile').addEventListener('change', e => {
    const file = e.target.files[0];
    e.target.value = '';
    const b = brushOf(S.editing);
    if (file && b) loadFile(file, (img, url) => { b.ref = img; b.refUrl = url; syncPop(); renderTiles(); renderPanel(); });
  });
  $('popImgClear').addEventListener('click', () => {
    const b = brushOf(S.editing);
    if (!b) return;
    b.ref = null;
    b.refUrl = '';
    syncPop();
    renderTiles();
    renderPanel();
  });
  $('popDelete').addEventListener('click', () => { if (S.editing !== null) deleteBrush(S.editing); });
}
function syncPop() {
  const b = brushOf(S.editing);
  if (!b) return;
  const T = TYPES[b.type];
  pop.querySelectorAll('.type').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.type === b.type)));
  pop.querySelectorAll('.sw').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.color === b.color)));
  $('popHint').textContent = T.hint;
  $('popImgWrap').dataset.off = String(!T.image);
  $('popFile').disabled = !T.image || !!T.capture;
  $('popImgPlus').innerHTML = icon(T.capture ? 'capture' : 'plus');
  pop.querySelector('.imgpick').title = T.capture ? 'Chụp vùng trên ảnh' : 'Ảnh tham chiếu';
  $('popNote').value = b.note;
  $('popNote').placeholder = T.ph;
  const hasRef = !!b.ref;
  $('popImg').hidden = !hasRef;
  if (hasRef) $('popImg').src = b.refUrl;
  $('popImgPlus').hidden = hasRef;
  $('popImgClear').hidden = !hasRef || !T.image;
}
function placePop(el, anchor) {
  const box = $('main').getBoundingClientRect(), t = anchor.getBoundingClientRect();
  if (window.matchMedia('(max-width:780px)').matches) {
    el.style.left = '0px';
    el.style.top = t.bottom - box.top + 8 + 'px';
  } else {
    el.style.left = t.right - box.left + 14 + 'px';
    el.style.top = Math.max(0, Math.min(t.top - box.top, box.height - el.offsetHeight)) + 'px';
  }
}
function openPop(id) {
  S.editing = id;
  pop.hidden = false;
  syncPop();
  placePop(pop, document.querySelector(`.tile[data-id="${id}"]`));
}
function closePop() {
  S.editing = null;
  pop.hidden = true;
}
document.addEventListener('pointerdown', e => {
  if (!pop.hidden && !pop.contains(e.target) && !e.target.closest('.tile') && !e.target.closest('.add')) closePop();
  if (!$('envPop').hidden && !$('envPop').contains(e.target) && !e.target.closest('#envBtn')) toggleEnv(false);
});

const esc = s => String(s).replace(/[&<>"]/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[m]));
function markup(text) {
  return esc(text)
    .replace(/&lt;image(\d+)&gt;/g, '<span class="tok">&lt;image$1&gt;</span>')
    .replace(/\b(red|blue|purple|yellow|green|orange) outline\b/g, (m, n) => `<span class="area" style="--c:${PAINT[n].hex}">${m}</span>`)
    .replace(/[vV]iền (đỏ|xanh lá|xanh|tím|vàng|cam)/g, (m, n) => `<span class="area" style="--c:${PALETTE.find(c => c.area === n).hex}">${m}</span>`);
}

function paintThumb(cv) {
  const k = 144 / W();
  cv.width = 144;
  cv.height = Math.max(1, Math.round(H() * k));
  const g = cv.getContext('2d');
  g.setTransform(k, 0, 0, k, 0, 0);
  g.imageSmoothingQuality = 'high';
  g.fillStyle = PAD_GRAY;
  g.fillRect(0, 0, W(), H());
  g.drawImage(S.base, S.pads.left, S.pads.top, S.w, S.h);
}
function paintShotThumb(cv, ids) {
  const plan = planShot(W(), H(), 0.25);
  const clean = newCanvas(plan.outW, plan.outH);
  const c = clean.getContext('2d');
  c.setTransform(plan.outW / W(), 0, 0, plan.outH / H(), 0, 0);
  c.imageSmoothingQuality = 'high';
  c.fillStyle = PAD_GRAY;
  c.fillRect(0, 0, W(), H());
  c.drawImage(S.base, S.pads.left, S.pads.top, S.w, S.h);
  const marks = ids.map(id => {
    const b = brushOf(id);
    return { mask: brushMask(b), hex: PAINT[b.color].hex };
  });
  cv.width = 144;
  cv.height = Math.max(1, Math.round(H() * 144 / W()));
  cv.getContext('2d').drawImage(guideImage(clean, marks, plan), 0, 0, cv.width, cv.height);
}
function paintRefThumb(cv, ref) {
  const rw = ref.naturalWidth || ref.width, rh = ref.naturalHeight || ref.height;
  cv.width = 144;
  cv.height = Math.max(1, Math.round(rh * 144 / rw));
  cv.getContext('2d').drawImage(ref, 0, 0, cv.width, cv.height);
}
function slotFigure(label, title, paint) {
  const fig = document.createElement('figure');
  fig.title = title;
  const cv = document.createElement('canvas');
  paint(cv);
  const cap = document.createElement('figcaption');
  cap.textContent = label;
  fig.append(cv, cap);
  return fig;
}
function iconButton(name, title, onclick) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'ibtn';
  b.title = title;
  b.setAttribute('aria-label', title);
  b.innerHTML = icon(name);
  b.onclick = onclick;
  return b;
}

function renderPanel() {
  const marks = S.brushes.map(b => ({ brush: b.id, color: b.color, type: b.type, hasRef: !!b.ref, note: b.note, cover: b.cover }));
  const { passes, problems } = S.tab === 'sheet' ? buildSheetPasses() : S.tab === 'rotate' ? buildRotatePasses() : S.tab === 'swap' ? buildSwapPasses() : S.tab === 'upscale' ? buildUpscalePasses() : S.base ? buildPasses({ marks, expanded: needsOutpaint(), scene: S.scene, envCount: S.env.filter(s => s.ref).length, expandLora: S.expandLora, layers: Layers.list, blendLora: S.blendLora }) : { passes: [], problems: [] };

  document.querySelectorAll('.tile').forEach(t => {
    t.querySelector('.bang').hidden = !problems.some(p => p.brush === Number(t.dataset.id));
  });
  $('promptDot').hidden = !problems.length;
  $('problems').replaceChildren(...problems.map(p => { const li = document.createElement('li'); li.textContent = p.text; return li; }));
  $('noPrompt').hidden = passes.length > 0 || problems.length > 0;
  $('noPrompt').textContent = { sheet: 'Thêm ảnh nhân vật để tạo prompt.', rotate: 'Thêm ảnh chính để tạo prompt.', swap: 'Thêm ảnh cảnh để tạo prompt.', upscale: 'Thêm ảnh cần upscale để tạo prompt.' }[S.tab] || 'Tô hoặc kéo rộng để tạo prompt.';
  S.passes = passes;
  S.problems = problems;
  syncGen();

  const wrap = $('passes');
  wrap.replaceChildren();
  passes.forEach((pass, i) => {
    const sec = document.createElement('section');
    sec.className = 'pass';
    const auto = pass.lines.map(l => l.en).join(' ');
    const override = S.overrides[pass.kind] ?? null;

    const head = document.createElement('div');
    head.className = 'pass-head';
    const h = document.createElement('h3');
    h.textContent = passes.length > 1 ? `${i + 1}. ${pass.name}` : pass.name;
    const actions = document.createElement('div');
    actions.className = 'pass-actions';
    if (override === null) {
      actions.append(iconButton('pencil', 'Sửa tay', () => { S.overrides[pass.kind] = auto; renderPanel(); }));
    } else {
      actions.append(iconButton('undo', 'Dùng lại prompt tự động', () => { S.overrides[pass.kind] = null; renderPanel(); }));
    }
    const copy = iconButton('copy', 'Chép prompt', () => {
      copyText(S.overrides[pass.kind] ?? auto).then(ok => {
        if (ok) { copy.innerHTML = icon('check'); setTimeout(() => { copy.innerHTML = icon('copy'); }, 1200); }
        else copy.title = 'Không chép được, hãy bôi đen';
      });
    });
    actions.append(copy);
    head.append(h, actions);
    sec.append(head);

    const slots = document.createElement('div');
    slots.className = 'slots';
    if (pass.kind === 'sheet') {
      Sheet.slots.filter(s => s.ref).forEach((slot, n) => {
        slots.append(slotFigure(`image_${n + 1}`, slot.label, cv => paintRefThumb(cv, slot.ref)));
      });
    } else if (pass.kind === 'rotate') {
      slots.append(slotFigure('image_1', 'Ảnh chính', cv => paintRefThumb(cv, Rotate.slots[0].ref)));
      let n = 2;
      if (Rotate.mode === 'splat') slots.append(slotFigure(`image_${n++}`, 'Ảnh render từ splat ở góc đang chọn', cv => RotateView.paintThumb(cv)));
      Rotate.slots.slice(1).filter(s => s.ref).forEach(slot => {
        slots.append(slotFigure(`image_${n++}`, slot.label, cv => paintRefThumb(cv, slot.ref)));
      });
    } else if (pass.kind.startsWith('swap')) {
      const [shot, ref] = Swap.slots;
      slots.append(slotFigure('image_1', 'Ảnh cảnh', cv => paintRefThumb(cv, shot.ref)));
      let n = 2;
      if (pass.kind === 'swap_selective') slots.append(slotFigure(`image_${n++}`, 'Ảnh cảnh, người cần thay phủ hồng', cv => paintRefThumb(cv, swapMarked())));
      if (ref.ref) slots.append(slotFigure(`image_${n}`, SWAP_REF_LABEL[Swap.mode], cv => paintRefThumb(cv, ref.ref)));
    } else if (pass.kind === 'merge') {
      slots.append(slotFigure('image_1', 'Cả shot với layer đã đặt: vùng lấp viền đỏ, vùng ghép viền xanh', cv => paintMergeThumb(cv)));
    } else if (pass.kind === 'upscale') {
      slots.append(slotFigure('image_1', 'Từng ô khoảng 2 MP của ảnh đã phóng to, vẽ lại lần lượt', cv => paintRefThumb(cv, Up.slots[0].ref)));
    } else if (pass.kind === 'expand') {
      slots.append(slotFigure('image_1', 'Ảnh + vùng xám', cv => paintThumb(cv)));
      S.env.filter(s => s.ref).forEach((slot, n) => {
        slots.append(slotFigure(`image_${n + 2}`, 'Môi trường', cv => paintRefThumb(cv, slot.ref)));
      });
    } else {
      slots.append(slotFigure('image_1', 'Cả shot, model thấy viền mỏng thay cho nét tô', cv => paintShotThumb(cv, pass.brushes)));
      for (const s of pass.slots) {
        const b = brushOf(s.brush);
        slots.append(slotFigure(`image_${s.n}`, TYPES[b.type].label, cv => paintRefThumb(cv, b.ref)));
      }
    }
    sec.append(slots);

    if (override === null) {
      const ol = document.createElement('ol');
      ol.className = 'lines';
      for (const l of pass.lines) {
        const li = document.createElement('li');
        li.innerHTML = `<p class="en">${markup(l.en)}</p><p class="vi">${markup(l.vi)}</p>`;
        ol.append(li);
      }
      sec.append(ol);
    } else {
      const ta = document.createElement('textarea');
      ta.className = 'field';
      ta.id = 'override-' + pass.kind;
      ta.value = override;
      ta.setAttribute('aria-label', 'Prompt sửa tay');
      ta.addEventListener('input', () => { S.overrides[pass.kind] = ta.value; });
      sec.append(ta);
    }
    wrap.append(sec);
  });
  syncTools();
}

function syncBar() {
  const has = !!S.base;
  const mp = W() * H() / 1e6;
  $('readout').textContent = has ? `${W()} × ${H()} · ${mp.toLocaleString('vi-VN', { maximumFractionDigits: 2 })} MP` : '';
  let note = '';
  if (has && W() * H() > 2048 * 2048) note = 'Vượt 2048 × 2048';
  else if (has && needsOutpaint() && mp > 2) note = 'Quá 2 MP cho LoRA v2';
  $('sizeNote').textContent = note;
}

function afterChange() {
  rebuildLayer();
  measureCover();
  S.padsPainted = padsFullyPainted();
  place();
  syncBar();
  renderPanel();
}

function setPad(side, raw) {
  const opposite = { left: 'right', right: 'left', top: 'bottom', bottom: 'top' }[side];
  const size = side === 'left' || side === 'right' ? S.w : S.h;
  S.pads[side] = snapPad(size, raw, S.pads[opposite]);
}

let drag = null;
frame.querySelectorAll('.edge').forEach(edge => {
  edge.addEventListener('pointerdown', e => {
    e.preventDefault();
    edge.setPointerCapture(e.pointerId);
    drag = {
      side: edge.dataset.side, startPad: S.pads[edge.dataset.side], sx: e.clientX, sy: e.clientY,
      ix: V.x + S.pads.left * V.scale, iy: V.y + S.pads.top * V.scale, pushed: false,
    };
  });
  edge.addEventListener('pointermove', e => {
    if (!drag) return;
    const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
    const delta = { left: -dx, right: dx, top: -dy, bottom: dy }[drag.side] / V.scale;
    const before = S.pads[drag.side];
    const undoState = snapshot();
    setPad(drag.side, drag.startPad + delta);
    if (S.pads[drag.side] === before) return;
    if (!drag.pushed) {
      S.past.push(undoState);
      S.future = [];
      drag.pushed = true;
    }
    V.x = drag.ix - S.pads.left * V.scale;
    V.y = drag.iy - S.pads.top * V.scale;
    rebuildLayer();
    place();
    syncBar();
  });
  const end = () => {
    if (!drag) return;
    drag = null;
    afterChange();
    fit();
  };
  edge.addEventListener('pointerup', end);
  edge.addEventListener('pointercancel', end);
});

let live = null;
function toImage(e) {
  const r = view.getBoundingClientRect();
  return { x: (e.clientX - r.left) / V.scale - S.pads.left, y: (e.clientY - r.top) / V.scale - S.pads.top };
}
let capDrag = null;
function startCapture(id) {
  if (!S.base) {
    setStatus('Thêm ảnh gốc trước khi chụp vùng.');
    return;
  }
  S.capture = id;
  selectBrush(id);
  closePop();
  brushcur.hidden = true;
  $('captureHint').hidden = false;
  view.style.cursor = viewCursor();
}
function stopCapture() {
  S.capture = null;
  capDrag = null;
  $('capbox').hidden = true;
  $('captureHint').hidden = true;
  view.style.cursor = viewCursor();
}
function drawCapBox(p) {
  const x = Math.min(capDrag.x, p.x), y = Math.min(capDrag.y, p.y);
  const box = $('capbox');
  box.hidden = false;
  box.style.left = (x + S.pads.left) * V.scale + 'px';
  box.style.top = (y + S.pads.top) * V.scale + 'px';
  box.style.width = Math.abs(p.x - capDrag.x) * V.scale + 'px';
  box.style.height = Math.abs(p.y - capDrag.y) * V.scale + 'px';
}
// Chụp từ ảnh hiện tại, không lấy nét tô và không lấy vùng xám.
function finishCapture(p) {
  const b = brushOf(S.capture);
  const x0 = Math.max(0, Math.round(Math.min(capDrag.x, p.x))), y0 = Math.max(0, Math.round(Math.min(capDrag.y, p.y)));
  const x1 = Math.min(S.w, Math.round(Math.max(capDrag.x, p.x))), y1 = Math.min(S.h, Math.round(Math.max(capDrag.y, p.y)));
  const w = x1 - x0, h = y1 - y0;
  capDrag = null;
  $('capbox').hidden = true;
  if (w < 8 || h < 8 || !b) return;
  const crop = newCanvas(w, h);
  crop.getContext('2d').drawImage(S.base, x0, y0, w, h, 0, 0, w, h);
  b.ref = crop;
  b.refUrl = crop.toDataURL('image/png');
  stopCapture();
  renderTiles();
  renderPanel();
  setStatus('Đã chụp. Giờ tô vào vùng cần clone.');
}
function moveBrushCursor(e) {
  if (!canPaint() || S.capture !== null) { brushcur.hidden = true; return; }
  const r = stage.getBoundingClientRect();
  const d = S.size * V.scale;
  brushcur.hidden = false;
  brushcur.style.width = d + 'px';
  brushcur.style.height = d + 'px';
  brushcur.style.transform = `translate(${e.clientX - r.left - d / 2}px,${e.clientY - r.top - d / 2}px)`;
  brushcur.style.borderColor = S.erase ? '#fff' : PAINT[brushOf(S.active).color].hex;
}
view.addEventListener('pointerdown', e => {
  if (e.button !== 0 || !S.base) return;
  if (S.tool) {
    toolDown(e);
    return;
  }
  if (S.capture !== null) {
    view.setPointerCapture(e.pointerId);
    capDrag = toImage(e);
    return;
  }
  if (!canPaint()) {
    const add = $('addBrush');
    add.classList.add('nudge');
    setTimeout(() => add.classList.remove('nudge'), 500);
    return;
  }
  view.setPointerCapture(e.pointerId);
  pushHistory();
  live = { brush: S.erase ? null : S.active, erase: S.erase, size: S.size, points: [toImage(e)] };
  S.strokes = [...S.strokes, live];
  const g = layer.getContext('2d');
  g.setTransform(1, 0, 0, 1, S.pads.left, S.pads.top);
  drawStroke(g, live);
  schedule();
});
view.addEventListener('pointermove', e => {
  if (S.tool) {
    toolMove(e);
    return;
  }
  moveBrushCursor(e);
  if (capDrag) drawCapBox(toImage(e));
  if (!live) return;
  const p = toImage(e);
  const last = live.points[live.points.length - 1];
  if (Math.hypot(p.x - last.x, p.y - last.y) < 1) return;
  live.points.push(p);
  const g = layer.getContext('2d');
  g.setTransform(1, 0, 0, 1, S.pads.left, S.pads.top);
  drawStroke(g, { ...live, points: [last, p] });
  schedule();
});
const endStroke = () => {
  if (!live) return;
  live = null;
  afterChange();
};
view.addEventListener('pointerup', e => {
  if (S.tool || toolDrag) {
    toolUp(e);
    return;
  }
  if (capDrag) finishCapture(toImage(e));
  endStroke();
});
view.addEventListener('pointercancel', () => {
  toolDrag = null;
  capDrag = null;
  $('capbox').hidden = true;
  endStroke();
});
view.addEventListener('pointerleave', () => { if (!live) brushcur.hidden = true; });


// ---------- Chọn vật bằng SAM 3.1 và layer ----------
// SAM chạy bằng node SAM3_Detect có sẵn trong ComfyUI 0.43, model như template utility_image_segment_sam3.
const SAM_CKPT = 'sam3.1_multiplex_fp16.safetensors';
const SAM_UPLOAD_MP = 2;
const SEL_COLOR = '#34D3FF';
const SEL_HELP = 'Kéo khung quanh vật hoặc bấm lên vật. Shift + kéo để thêm vật, Shift + bấm để bỏ phần thừa. Gõ tên tiếng Anh rồi Enter để tìm mọi vật cùng loại.';
// LoRA Qwen-image-2.1 光影溶图 của RunningHub (tác giả 浩的AI日常), thử nghiệm: trang của LoRA không ghi cách đưa ảnh vào.
const BLEND_LORA = 'Qwenimag21_c2-st2000.safetensors';
// Câu đầu của LoRA là prompt mẫu của tác giả, giữ nguyên. Các câu sau bảo Qwen đổi ánh sáng của vật theo cảnh,
// vì nếu chỉ bảo giữ dáng và chi tiết thì Qwen giữ luôn ánh sáng cũ của vật.
const MERGE_PROMPTS = {
  hole: { en: 'Fill the gray area inside the red outline in <image1> with the surrounding background, as if nothing was ever there.', vi: 'Lấp vùng xám trong viền đỏ trên <image1> bằng nền xung quanh, như chưa từng có gì ở đó.' },
  lora: { en: 'pengyu, Apply consistent lighting to the product or object, enhance specular highlights, front-back spatial relationship, add natural contact shadow, and blend it seamlessly into the background.', vi: 'pengyu (từ khoá của LoRA), prompt mẫu của tác giả: ánh sáng đồng bộ cho vật, tăng ánh phản chiếu, đúng quan hệ trước sau, thêm bóng tiếp xúc, hoà liền vào nền.' },
  relight: { en: 'The object inside the blue outline in <image1> was pasted in and still has the lighting of another photo. Relight it to belong to this scene: match the light direction, rim light, color temperature, brightness, contrast, colors and color grading of its surroundings in <image1>.', vi: 'Vật trong viền xanh trên <image1> được dán vào nên còn ánh sáng của ảnh khác. Chiếu sáng lại cho hợp cảnh: khớp hướng sáng, rim light, nhiệt độ màu, độ sáng, độ tương phản, màu sắc và tông màu của xung quanh trên <image1>.' },
  shadow: { en: 'Shade it from the same light sources as the scene, add the shadow it casts on the ground and nearby surfaces in that light direction, and the reflected light and color it picks up from the scene.', vi: 'Đổ khối theo đúng nguồn sáng của cảnh, thêm bóng nó đổ xuống nền và các mặt gần đó theo hướng sáng ấy, và ánh phản xạ, màu hắt từ cảnh lên nó.' },
  keepShape: { en: 'Keep its shape, pose, proportions and identity; change only its lighting, shading and color.', vi: 'Giữ dáng, tư thế, tỉ lệ và nhận diện của vật; chỉ đổi ánh sáng, đổ khối và màu.' },
  edges: { en: 'Clean up its cut-out edges: remove jagged or aliased edges and any leftover outline, fringe or halo of its old background, and blend the edges smoothly into the new background.', vi: 'Làm sạch mép cắt: khử răng cưa, xoá viền outline, viền hay quầng của nền cũ còn sót, hoà mép mượt vào nền mới.' },
  finish: { en: 'Remove the outlines.', vi: 'Xoá các viền đi.' },
};
function buildMergePass(layers, blendLora) {
  const lines = [];
  if (layers.some(L => L.hole && !L.keep)) lines.push(MERGE_PROMPTS.hole);
  if (blendLora) lines.push(MERGE_PROMPTS.lora);
  lines.push(MERGE_PROMPTS.relight, MERGE_PROMPTS.shadow, MERGE_PROMPTS.keepShape, MERGE_PROMPTS.edges, MERGE_PROMPTS.finish, PROMPTS.keep);
  return { kind: 'merge', name: blendLora ? 'Ghép layer (LoRA ánh sáng)' : 'Ghép layer', lines, slots: [] };
}

// Ánh xạ ô vuông đơn vị lên tứ giác 4 góc (phối cảnh), hoặc qua lưới 3x3 khi uốn.
function quadMap(q) {
  const [p0, p1, p2, p3] = q;
  const dx1 = p1.x - p2.x, dx2 = p3.x - p2.x, dx3 = p0.x - p1.x + p2.x - p3.x;
  const dy1 = p1.y - p2.y, dy2 = p3.y - p2.y, dy3 = p0.y - p1.y + p2.y - p3.y;
  let g = 0, h = 0;
  if (Math.abs(dx3) > 1e-9 || Math.abs(dy3) > 1e-9) {
    const den = dx1 * dy2 - dx2 * dy1;
    g = (dx3 * dy2 - dx2 * dy3) / den;
    h = (dx1 * dy3 - dx3 * dy1) / den;
  }
  const a = p1.x - p0.x + g * p1.x, b = p3.x - p0.x + h * p3.x, d = p1.y - p0.y + g * p1.y, e = p3.y - p0.y + h * p3.y;
  return (u, v) => {
    const w = g * u + h * v + 1;
    return { x: (a * u + b * v + p0.x) / w, y: (d * u + e * v + p0.y) / w };
  };
}
const quadWeights = t => [2 * (t - 0.5) * (t - 1), -4 * t * (t - 1), 2 * t * (t - 0.5)];
function gridMap(G) {
  return (u, v) => {
    const a = quadWeights(u), b = quadWeights(v);
    let x = 0, y = 0;
    for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
      const w = a[i] * b[j];
      x += w * G[j * 3 + i].x;
      y += w * G[j * 3 + i].y;
    }
    return { x, y };
  };
}
const layerMap = L => (L.grid ? gridMap(L.grid) : quadMap(L.corners));
const activeLayer = () => Layers.list.find(L => L.id === Layers.active) || null;

// Vẽ layer bằng lưới tam giác: mỗi tam giác chép ảnh layer qua một phép biến đổi affine. pad nới tam giác để không hở đường nối.
function drawTriangle(g, img, s0, s1, s2, d0, d1, d2, pad) {
  const cx = (d0.x + d1.x + d2.x) / 3, cy = (d0.y + d1.y + d2.y) / 3;
  const grow = p => {
    const l = Math.hypot(p.x - cx, p.y - cy) || 1;
    return { x: p.x + (p.x - cx) / l * pad, y: p.y + (p.y - cy) / l * pad };
  };
  const [e0, e1, e2] = [grow(d0), grow(d1), grow(d2)];
  const ux = s1.x - s0.x, uy = s1.y - s0.y, vx = s2.x - s0.x, vy = s2.y - s0.y;
  const det = ux * vy - vx * uy;
  if (Math.abs(det) < 1e-9) return;
  const px = d1.x - d0.x, py = d1.y - d0.y, qx = d2.x - d0.x, qy = d2.y - d0.y;
  const a = (px * vy - qx * uy) / det, c = (qx * ux - px * vx) / det;
  const b = (py * vy - qy * uy) / det, d = (qy * ux - py * vx) / det;
  g.save();
  g.beginPath();
  g.moveTo(e0.x, e0.y);
  g.lineTo(e1.x, e1.y);
  g.lineTo(e2.x, e2.y);
  g.closePath();
  g.clip();
  g.transform(a, b, c, d, d0.x - a * s0.x - c * s0.y, d0.y - b * s0.x - d * s0.y);
  // Chỉ chép phần ảnh quanh tam giác, đỡ tốn khi layer lớn.
  const bx = Math.max(0, Math.floor(Math.min(s0.x, s1.x, s2.x)) - 2), by = Math.max(0, Math.floor(Math.min(s0.y, s1.y, s2.y)) - 2);
  const bw = Math.min(img.width, Math.ceil(Math.max(s0.x, s1.x, s2.x)) + 2) - bx, bh = Math.min(img.height, Math.ceil(Math.max(s0.y, s1.y, s2.y)) + 2) - by;
  if (bw > 0 && bh > 0) g.drawImage(img, bx, by, bw, bh, bx, by, bw, bh);
  g.restore();
}
// Vẽ layer bằng WebGL: các tam giác dùng chung đỉnh nên không có đường nối thành lưới như khi cắt từng tam giác trên canvas 2D.
// Máy không có WebGL thì quay về cách vẽ từng tam giác.
const LayerGL = { canvas: null, gl: null, prog: null, buf: null, textures: new Map(), failed: false };
function layerGL() {
  if (LayerGL.gl || LayerGL.failed) return LayerGL.gl;
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl', { premultipliedAlpha: true, preserveDrawingBuffer: true, antialias: true });
  const shader = (type, source) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null;
  };
  const vs = gl && shader(gl.VERTEX_SHADER, 'attribute vec2 p;attribute vec2 t;uniform vec2 size;varying vec2 vt;void main(){vt=t;vec2 c=p/size*2.0-1.0;gl_Position=vec4(c.x,-c.y,0.0,1.0);}');
  const fs = gl && shader(gl.FRAGMENT_SHADER, 'precision mediump float;varying vec2 vt;uniform sampler2D img;void main(){gl_FragColor=texture2D(img,vt);}');
  const prog = vs && fs && gl.createProgram();
  if (prog) {
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
  }
  if (!prog || !gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    LayerGL.failed = true;
    return null;
  }
  Object.assign(LayerGL, { canvas, gl, prog, buf: gl.createBuffer() });
  return gl;
}
function layerTexture(gl, L) {
  let entry = LayerGL.textures.get(L.id);
  if (entry && entry.src === L.src && entry.rev === L.rev) return entry.tex;
  if (!entry) {
    entry = { tex: gl.createTexture() };
    LayerGL.textures.set(L.id, entry);
  }
  gl.bindTexture(gl.TEXTURE_2D, entry.tex);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, L.src);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  Object.assign(entry, { src: L.src, rev: L.rev });
  return entry.tex;
}
// Vẽ layer vào khung bao của nó ở tỉ lệ scale, trả về canvas WebGL và vị trí theo toạ độ ảnh.
function renderLayerGL(L, scale, n) {
  const gl = layerGL();
  if (!gl) return null;
  const map = layerMap(L), P = [];
  for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) P.push(map(i / n, j / n));
  const x0 = Math.floor(Math.min(...P.map(p => p.x))) - 1, y0 = Math.floor(Math.min(...P.map(p => p.y))) - 1;
  const x1 = Math.ceil(Math.max(...P.map(p => p.x))) + 1, y1 = Math.ceil(Math.max(...P.map(p => p.y))) + 1;
  const max = gl.getParameter(gl.MAX_VIEWPORT_DIMS);
  const k = Math.min(scale, (max[0] - 2) / (x1 - x0), (max[1] - 2) / (y1 - y0));
  const w = Math.max(1, Math.ceil((x1 - x0) * k)), h = Math.max(1, Math.ceil((y1 - y0) * k));
  LayerGL.canvas.width = w;
  LayerGL.canvas.height = h;
  gl.viewport(0, 0, w, h);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(LayerGL.prog);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, layerTexture(gl, L));
  const v = new Float32Array(n * n * 6 * 4);
  let o = 0;
  const put = (i, j) => {
    const p = P[j * (n + 1) + i];
    v[o++] = (p.x - x0) * k;
    v[o++] = (p.y - y0) * k;
    v[o++] = i / n;
    v[o++] = j / n;
  };
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    put(i, j); put(i + 1, j); put(i + 1, j + 1);
    put(i, j); put(i + 1, j + 1); put(i, j + 1);
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, LayerGL.buf);
  gl.bufferData(gl.ARRAY_BUFFER, v, gl.DYNAMIC_DRAW);
  const pLoc = gl.getAttribLocation(LayerGL.prog, 'p'), tLoc = gl.getAttribLocation(LayerGL.prog, 't');
  gl.enableVertexAttribArray(pLoc);
  gl.vertexAttribPointer(pLoc, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(tLoc);
  gl.vertexAttribPointer(tLoc, 2, gl.FLOAT, false, 16, 8);
  gl.uniform2f(gl.getUniformLocation(LayerGL.prog, 'size'), w, h);
  gl.uniform1i(gl.getUniformLocation(LayerGL.prog, 'img'), 0);
  gl.drawArrays(gl.TRIANGLES, 0, n * n * 6);
  return { canvas: LayerGL.canvas, x: x0, y: y0, w: w / k, h: h / k };
}
function drawLayer(g, L, n, pad) {
  const t = g.getTransform();
  const out = renderLayerGL(L, Math.min(4, Math.max(0.02, Math.hypot(t.a, t.b))), 32);
  if (out) {
    g.drawImage(out.canvas, out.x, out.y, out.w, out.h);
    return;
  }
  drawLayerTriangles(g, L, n, pad);
}
function drawLayerTriangles(g, L, n, pad) {
  const map = layerMap(L), sw = L.src.width, sh = L.src.height;
  const P = [];
  for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) P.push(map(i / n, j / n));
  const S_ = (i, j) => ({ x: i / n * sw, y: j / n * sh });
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const d00 = P[j * (n + 1) + i], d10 = P[j * (n + 1) + i + 1], d01 = P[(j + 1) * (n + 1) + i], d11 = P[(j + 1) * (n + 1) + i + 1];
    drawTriangle(g, L.src, S_(i, j), S_(i + 1, j), S_(i + 1, j + 1), d00, d10, d11, pad);
    drawTriangle(g, L.src, S_(i, j), S_(i + 1, j + 1), S_(i, j + 1), d00, d11, d01, pad);
  }
}
function layerOutline(L) {
  const map = layerMap(L), pts = [];
  for (let k = 0; k < 16; k++) pts.push(map(k / 16, 0));
  for (let k = 0; k < 16; k++) pts.push(map(1, k / 16));
  for (let k = 16; k > 0; k--) pts.push(map(k / 16, 1));
  for (let k = 16; k > 0; k--) pts.push(map(0, k / 16));
  return pts;
}
function insidePolygon(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i], b = poly[j];
    if ((a.y > p.y) !== (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}
function layerCenter(L) {
  const m = layerMap(L);
  return m(0.5, 0.5);
}
function layerHandles(L) {
  const m = layerMap(L), px = 1 / V.scale;
  if (Layers.mode === 'warp') {
    const pts = L.grid || [0, 0.5, 1].flatMap(v => [0, 0.5, 1].map(u => m(u, v)));
    return pts.map((p, index) => ({ kind: 'grid', index, p }));
  }
  const corners = [m(0, 0), m(1, 0), m(1, 1), m(0, 1)].map((p, index) => ({ kind: 'corner', index, p }));
  const top = m(0.5, 0), c = layerCenter(L), l = Math.hypot(top.x - c.x, top.y - c.y) || 1;
  const rot = { kind: 'rotate', index: 0, p: { x: top.x + (top.x - c.x) / l * 28 * px, y: top.y + (top.y - c.y) / l * 28 * px }, from: top };
  return [rot, ...corners];
}
function hitHandle(L, p) {
  const r = 10 / V.scale;
  return layerHandles(L).find(h => Math.hypot(h.p.x - p.x, h.p.y - p.y) <= r) || null;
}
function layerAt(p) {
  for (let i = Layers.list.length - 1; i >= 0; i--) if (insidePolygon(p, layerOutline(Layers.list[i]))) return Layers.list[i];
  return null;
}
function holeGray(L) {
  if (!L._gray) {
    const c = newCanvas(L.hole.width, L.hole.height), g = c.getContext('2d');
    stampMask(g, L.hole, 3);
    g.globalCompositeOperation = 'source-in';
    g.fillStyle = PAD_GRAY;
    g.fillRect(0, 0, c.width, c.height);
    L._gray = c;
  }
  return L._gray;
}
function tintMask(mask, hex) {
  const c = newCanvas(mask.width, mask.height), g = c.getContext('2d');
  g.drawImage(mask, 0, 0);
  g.globalCompositeOperation = 'source-in';
  g.fillStyle = hex;
  g.fillRect(0, 0, c.width, c.height);
  return c;
}
const strokeTints = new WeakMap();
function drawMaskStroke(g, s, b) {
  if (s.erase) {
    g.globalCompositeOperation = 'destination-out';
    g.drawImage(s.mask, 0, 0);
    g.globalCompositeOperation = 'source-over';
    return;
  }
  const hex = PAINT[b.color].hex;
  let t = strokeTints.get(s.mask);
  if (!t || t.hex !== hex) {
    t = { hex, canvas: tintMask(s.mask, hex) };
    strokeTints.set(s.mask, t);
  }
  g.drawImage(t.canvas, 0, 0);
}

// Vẽ lên khung xem: vùng xám sẽ lấp, các layer, vùng chọn, khung chỉnh của layer đang chọn.
let toolDrag = null;
function drawOverlays(g) {
  const selecting = S.tool === 'select';
  if (!Layers.list.length && !selecting) return;
  const px = 1 / V.scale;
  g.save();
  g.translate(S.pads.left, S.pads.top);
  for (const L of Layers.list) if (L.hole && !L.keep) g.drawImage(holeGray(L), 0, 0);
  for (const L of Layers.list) drawLayer(g, L, 12, px);
  if (selecting) {
    if (Sel.tint) {
      g.globalAlpha = 0.45;
      g.drawImage(Sel.tint, 0, 0);
      g.globalAlpha = 1;
    }
    if (Sel.ring) g.drawImage(Sel.ring, 0, 0);
    const box = toolDrag && toolDrag.kind === 'select' && toolDrag.moved ? normBox(toolDrag.start, toolDrag.at) : Sel.busy ? Sel.box : null;
    if (box) {
      g.lineWidth = 1.5 * px;
      g.setLineDash([6 * px, 4 * px]);
      g.strokeStyle = '#000';
      g.strokeRect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
      g.lineDashOffset = 5 * px;
      g.strokeStyle = '#fff';
      g.strokeRect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
      g.setLineDash([]);
    }
    for (const p of Sel.points) {
      g.beginPath();
      g.arc(p.x, p.y, 6 * px, 0, Math.PI * 2);
      g.fillStyle = p.neg ? '#FF4D4D' : '#3DDC84';
      g.fill();
      g.lineWidth = 1.5 * px;
      g.strokeStyle = '#fff';
      g.stroke();
    }
  }
  const L = S.tool === 'layer' && activeLayer();
  if (L) {
    const outline = layerOutline(L);
    const path = () => { g.beginPath(); outline.forEach((p, i) => (i ? g.lineTo(p.x, p.y) : g.moveTo(p.x, p.y))); g.closePath(); };
    path();
    g.lineWidth = 3 * px;
    g.strokeStyle = 'rgba(0,0,0,.6)';
    g.stroke();
    g.lineWidth = 1.2 * px;
    g.strokeStyle = '#fff';
    g.stroke();
    const m = layerMap(L);
    if (Layers.mode === 'warp') {
      g.lineWidth = 1 * px;
      g.strokeStyle = 'rgba(255,255,255,.7)';
      for (const t of [0.5]) {
        g.beginPath();
        for (let k = 0; k <= 16; k++) { const p = m(t, k / 16); if (k) g.lineTo(p.x, p.y); else g.moveTo(p.x, p.y); }
        g.stroke();
        g.beginPath();
        for (let k = 0; k <= 16; k++) { const p = m(k / 16, t); if (k) g.lineTo(p.x, p.y); else g.moveTo(p.x, p.y); }
        g.stroke();
      }
    }
    for (const h of layerHandles(L)) {
      if (h.kind === 'rotate') {
        g.beginPath();
        g.moveTo(h.from.x, h.from.y);
        g.lineTo(h.p.x, h.p.y);
        g.lineWidth = 1.2 * px;
        g.strokeStyle = '#fff';
        g.stroke();
        g.beginPath();
        g.arc(h.p.x, h.p.y, 6 * px, 0, Math.PI * 2);
      } else if (h.kind === 'grid') {
        g.beginPath();
        g.arc(h.p.x, h.p.y, 5.5 * px, 0, Math.PI * 2);
      } else {
        g.beginPath();
        g.rect(h.p.x - 5 * px, h.p.y - 5 * px, 10 * px, 10 * px);
      }
      g.fillStyle = '#fff';
      g.fill();
      g.lineWidth = 1.5 * px;
      g.strokeStyle = '#111';
      g.stroke();
    }
  }
  g.restore();
}

function normBox(a, b) {
  return { x0: Math.max(0, Math.min(a.x, b.x)), y0: Math.max(0, Math.min(a.y, b.y)), x1: Math.min(S.w, Math.max(a.x, b.x)), y1: Math.min(S.h, Math.max(a.y, b.y)) };
}
function clonePts(L) {
  return { corners: L.corners.map(q => ({ ...q })), grid: L.grid && L.grid.map(q => ({ ...q })) };
}
function applyLayerDrag(d, p, shift) {
  const L = d.layer, src = d.orig, dx = p.x - d.start.x, dy = p.y - d.start.y;
  const pts = src.grid || src.corners;
  const set = arr => { if (src.grid) L.grid = arr; else L.corners = arr; };
  if (d.kind === 'move') {
    set(pts.map(q => ({ x: q.x + dx, y: q.y + dy })));
  } else if (d.kind === 'rotate' || (d.kind === 'corner' && shift)) {
    const c = d.center;
    if (d.kind === 'rotate') {
      const a = Math.atan2(p.y - c.y, p.x - c.x) - Math.atan2(d.start.y - c.y, d.start.x - c.x);
      const cos = Math.cos(a), sin = Math.sin(a);
      set(pts.map(q => ({ x: c.x + (q.x - c.x) * cos - (q.y - c.y) * sin, y: c.y + (q.x - c.x) * sin + (q.y - c.y) * cos })));
    } else {
      const f = Math.max(0.05, Math.hypot(p.x - c.x, p.y - c.y) / Math.max(1e-6, Math.hypot(d.start.x - c.x, d.start.y - c.y)));
      set(pts.map(q => ({ x: c.x + (q.x - c.x) * f, y: c.y + (q.y - c.y) * f })));
    }
  } else {
    const index = d.kind === 'corner' && src.grid ? [0, 2, 8, 6][d.index] : d.index;
    set(pts.map((q, i) => (i === index ? { x: q.x + dx, y: q.y + dy } : { ...q })));
  }
}
function toolDown(e) {
  const p = toImage(e);
  view.setPointerCapture(e.pointerId);
  if (S.tool === 'select') {
    if (!Sel.busy) toolDrag = { kind: 'select', start: p, at: p, moved: false, shift: e.shiftKey };
    return;
  }
  const L = activeLayer();
  const hit = L && hitHandle(L, p);
  const under = hit ? L : layerAt(p);
  if (!under) {
    Layers.active = null;
    S.tool = null;
    syncToolUi();
    return;
  }
  if (under.id !== Layers.active) {
    Layers.active = under.id;
    renderLayerTiles();
  }
  toolDrag = { kind: hit ? hit.kind : 'move', index: hit ? hit.index : 0, start: p, layer: under, orig: clonePts(under), center: layerCenter(under) };
  schedule();
}
function toolMove(e) {
  if (!toolDrag) return;
  const p = toImage(e);
  if (toolDrag.kind === 'select') {
    toolDrag.at = p;
    toolDrag.moved = toolDrag.moved || Math.hypot(p.x - toolDrag.start.x, p.y - toolDrag.start.y) * V.scale > 4;
  } else {
    applyLayerDrag(toolDrag, p, e.shiftKey);
  }
  schedule();
}
function toolUp(e) {
  const d = toolDrag;
  toolDrag = null;
  if (!d) return;
  if (d.kind === 'select') {
    const p = toImage(e);
    if (d.moved) {
      // Kéo khung: chọn lại từ đầu. Shift + kéo: thêm một vật vào vùng chọn.
      const hit = { kind: 'box', box: normBox(d.start, p) };
      if (d.shift) {
        Sel.hits = [...Sel.hits, hit];
      } else {
        Sel.hits = [hit];
        Sel.points = [];
      }
      requestSelection(hit);
    } else {
      // Bấm: lấy thêm phần còn thiếu. Shift + bấm: bỏ phần thừa.
      let hit = Sel.hits.find(h => h.kind === 'points');
      if (!hit && !d.shift) {
        hit = { kind: 'points', points: [] };
        Sel.hits = [...Sel.hits, hit];
      }
      if (hit) {
        hit.points = [...hit.points, { x: p.x, y: p.y, neg: d.shift }];
        Sel.points = hit.points;
        requestSelection(hit);
      }
    }
  } else {
    renderPanel();
  }
  schedule();
}

// SAM 3.1: kéo khung thì tìm vật trong khung, bấm thì tìm theo điểm (Shift + bấm là điểm bỏ).
function buildSamGraph({ image, box, points, text }) {
  const graph = {
    1: { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: SAM_CKPT } },
    4: { class_type: 'LoadImage', inputs: { image } },
    5: { class_type: 'SAM3_Detect', inputs: { model: ['1', 0], image: ['4', 0], threshold: 0.5, refine_iterations: 2, individual_masks: false } },
    8: { class_type: 'MaskToImage', inputs: { mask: ['5', 0] } },
    6: { class_type: 'PreviewImage', inputs: { images: ['8', 0] } },
  };
  const detect = graph[5].inputs;
  if (box) {
    graph[2] = { class_type: 'PrimitiveBoundingBox', inputs: box };
    detect.bboxes = ['2', 0];
  }
  const pos = points.filter(p => !p.neg).map(p => ({ x: p.x, y: p.y }));
  const neg = points.filter(p => p.neg).map(p => ({ x: p.x, y: p.y }));
  if (pos.length) {
    graph[3] = { class_type: 'PrimitiveString', inputs: { value: JSON.stringify(pos) } };
    detect.positive_coords = ['3', 0];
  }
  if (neg.length) {
    graph[7] = { class_type: 'PrimitiveString', inputs: { value: JSON.stringify(neg) } };
    detect.negative_coords = ['7', 0];
  }
  if (text) {
    graph[9] = { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text } };
    detect.conditioning = ['9', 0];
  }
  return graph;
}
// Việc nhỏ chạy riêng, không dùng nút Gen: chờ kết quả của node 6.
async function waitHelper(promptId) {
  for (let i = 0; i < 600; i++) {
    const res = await fetch(`/api/history/${promptId}`);
    const entry = res.ok ? (await res.json())[promptId] : null;
    if (entry) {
      const status = entry.status || {};
      if (status.status_str === 'error') {
        const failed = (status.messages || []).find(m => m[0] === 'execution_error');
        throw new Error(failed ? `${failed[1].node_type}: ${failed[1].exception_message}` : 'ComfyUI chạy lỗi.');
      }
      const images = ((entry.outputs || {})['6'] || {}).images;
      if (images && images[0]) return images[0];
      if (status.completed) throw new Error('ComfyUI chạy xong nhưng không trả về ảnh.');
    }
    await new Promise(resolve => setTimeout(resolve, 600));
  }
  throw new Error('Chờ ComfyUI quá lâu, hàng đợi đang dài.');
}
async function requireSam() {
  if (Sel.ready) return;
  const read = name => fetch(`/api/object_info/${name}`).then(r => (r.ok ? r.json() : {}), () => ({}));
  const [detect, ckpt] = await Promise.all([read('SAM3_Detect'), read('CheckpointLoaderSimple')]);
  if (!detect.SAM3_Detect) throw new Error('ComfyUI này chưa có node SAM3_Detect (cần ComfyUI 0.43).');
  const names = ckpt.CheckpointLoaderSimple ? ckpt.CheckpointLoaderSimple.input.required.ckpt_name[0] : [];
  if (!names.includes(SAM_CKPT)) throw new Error(`ComfyUI chưa có model ${SAM_CKPT}. Cloud Sync tải model trong nền, chờ tải xong rồi chọn lại.`);
  Sel.ready = true;
}
// Mask của SAM là trắng đen ở cỡ ảnh đã thu nhỏ. Phóng về cỡ ảnh gốc có nội suy và làm mờ nhẹ cho viền hết răng cưa,
// dựng lại đường viền trơn quanh mức 0,5, rồi làm mềm viền theo thanh Mềm viền.
// Dọn mask của SAM ở cỡ ảnh đã gửi lên: bỏ các mảnh rời nhỏ và lấp các lỗ nhỏ bên trong.
// Mảnh hoặc lỗ dưới 4% mảnh lớn nhất (tối thiểu 48 điểm ảnh) bị coi là hạt lấm tấm.
function cleanSamMask(raw, rel = 0.04) {
  const w = raw.naturalWidth || raw.width, h = raw.naturalHeight || raw.height;
  const c = newCanvas(w, h), g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(raw, 0, 0);
  const px = g.getImageData(0, 0, w, h), d = px.data, n = w * h;
  const on = new Uint8Array(n);
  for (let i = 0; i < n; i++) on[i] = d[i * 4] * d[i * 4 + 3] > 127 * 255 ? 1 : 0;
  const label = new Int32Array(n).fill(-1), stack = new Int32Array(n), areas = [], edges = [];
  // Gán nhãn các vùng liền nhau (8 hướng cho vật, 4 hướng cho nền) bằng ngăn xếp, không đệ quy.
  for (let s = 0; s < n; s++) {
    if (label[s] !== -1) continue;
    const value = on[s], id = areas.length;
    let top = 0, area = 0, edge = false;
    stack[top++] = s;
    label[s] = id;
    while (top) {
      const p = stack[--top], x = p % w, y = (p - x) / w;
      area++;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) edge = true;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if ((!dx && !dy) || (!value && dx && dy)) continue;
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const q = ny * w + nx;
        if (label[q] === -1 && on[q] === value) {
          label[q] = id;
          stack[top++] = q;
        }
      }
    }
    areas.push(value ? area : -area);
    edges.push(edge);
  }
  let largest = 0;
  for (const a of areas) if (a > largest) largest = a;
  const small = Math.max(48, largest * rel);
  for (let i = 0; i < n; i++) {
    const a = areas[label[i]];
    const keep = a > 0 ? a >= small : !edges[label[i]] && -a < small;
    d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = 255;
    d[i * 4 + 3] = keep ? 255 : 0;
  }
  g.putImageData(px, 0, 0);
  return c;
}
// Mọi bước làm trên cỡ của mask SAM (tối đa khoảng 2 MP) cho nhanh, rồi phóng về cỡ ảnh gốc có nội suy nên viền vẫn trơn.
function samSoftMask(raw, w, h, grow, feather) {
  const rw = raw.naturalWidth || raw.width, rh = raw.naturalHeight || raw.height, k = rw / w;
  const a = newCanvas(rw, rh), g = a.getContext('2d', { willReadFrequently: true });
  g.filter = 'blur(0.75px)';
  g.drawImage(raw, 0, 0);
  g.filter = 'none';
  const px = g.getImageData(0, 0, rw, rh), d = px.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = (d[i] / 255) * (d[i + 3] / 255);
    const t = Math.min(1, Math.max(0, (v - 0.3) / 0.4));
    d[i] = d[i + 1] = d[i + 2] = 255;
    d[i + 3] = Math.round(t * t * (3 - 2 * t) * 255);
  }
  g.putImageData(px, 0, 0);
  let core = a;
  if (grow > 0) {
    const grown = newCanvas(rw, rh);
    dilate(grown.getContext('2d'), a, grow * k);
    core = newCanvas(rw, rh);
    const cg = core.getContext('2d');
    cg.filter = 'blur(0.6px)';
    cg.drawImage(grown, 0, 0);
  }
  // Mềm ra phía ngoài: quầng mờ nằm ngoài vùng chọn, bên trong vật và phần đã nới vẫn đặc.
  let soft = core;
  if (feather > 0) {
    const halo = newCanvas(rw, rh);
    dilate(halo.getContext('2d'), core, feather * k);
    soft = newCanvas(rw, rh);
    const sg = soft.getContext('2d');
    sg.filter = `blur(${feather * k / 2}px)`;
    sg.drawImage(halo, 0, 0);
    sg.filter = 'none';
    sg.drawImage(core, 0, 0);
  }
  const up = c => {
    const o = newCanvas(w, h), og = o.getContext('2d');
    og.imageSmoothingQuality = 'high';
    og.drawImage(c, 0, 0, w, h);
    return o;
  };
  const out = up(soft);
  out.core = soft === core ? out : up(core);
  return out;
}
// Mask của BiRefNet đã có viền mềm, giữ nguyên độ mềm đó làm kênh trong suốt.
function alphaFromImage(img, w, h) {
  const c = newCanvas(w, h), g = c.getContext('2d', { willReadFrequently: true });
  g.imageSmoothingQuality = 'high';
  g.drawImage(img, 0, 0, w, h);
  const px = g.getImageData(0, 0, w, h), d = px.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i];
    d[i] = d[i + 1] = d[i + 2] = 255;
    d[i + 3] = v;
  }
  g.putImageData(px, 0, 0);
  return c;
}
function applySelSoftness() {
  const mask = samSoftMask(Sel.clean, S.w, S.h, Sel.grow, Sel.feather);
  const bounds = maskBounds(mask);
  Sel.mask = bounds ? mask : null;
  Sel.tint = bounds ? tintMask(mask, SEL_COLOR) : null;
  Sel.ring = bounds ? selectionOutline(mask.core || mask) : null;
  return bounds;
}
function maskBounds(mask) {
  const w = mask.width, h = mask.height;
  const data = mask.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (data[(y * w + x) * 4 + 3] > 8) {
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  return x1 < 0 ? null : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}
// Vùng chọn là hợp của nhiều lần chọn: mỗi khung là một vật, các điểm bấm là một vật, mỗi lần tìm theo tên có thể ra nhiều vật.
// Bản xem trước không có ComfyUI: khung ra hình elip, điểm ra hình tròn, tìm theo tên ra hai vật giả.
function fakeHitMask(hit) {
  const c = newCanvas(S.w, S.h), g = c.getContext('2d');
  g.fillStyle = '#fff';
  if (hit.kind === 'box') {
    const b = hit.box;
    g.beginPath();
    g.ellipse((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, Math.max(1, (b.x1 - b.x0) / 2 * 0.9), Math.max(1, (b.y1 - b.y0) / 2 * 0.9), 0, 0, Math.PI * 2);
    g.fill();
  } else if (hit.kind === 'points') {
    const r = Math.min(S.w, S.h) * 0.08;
    for (const p of hit.points) {
      g.globalCompositeOperation = p.neg ? 'destination-out' : 'source-over';
      g.beginPath();
      g.arc(p.x, p.y, r, 0, Math.PI * 2);
      g.fill();
    }
  } else {
    for (const [fx, fy] of [[0.3, 0.55], [0.7, 0.5]]) {
      g.beginPath();
      g.ellipse(S.w * fx, S.h * fy, S.w * 0.07, S.h * 0.18, 0, 0, Math.PI * 2);
      g.fill();
    }
  }
  return c;
}
async function samHitMask(hit) {
  await requireSam();
  if (Sel.upload.for !== S.base) {
    const small = fitReference(S.base, SAM_UPLOAD_MP);
    Sel.upload = { for: S.base, name: await uploadCanvas(small, `${Date.now().toString(36)}_sam.png`), k: small.width / S.w };
  }
  const k = Sel.upload.k;
  const b = hit.kind === 'box' && hit.box;
  const box = b && { x: Math.round(b.x0 * k), y: Math.round(b.y0 * k), width: Math.max(1, Math.round((b.x1 - b.x0) * k)), height: Math.max(1, Math.round((b.y1 - b.y0) * k)) };
  const points = hit.kind === 'points' ? hit.points.map(p => ({ x: Math.round(p.x * k), y: Math.round(p.y * k), neg: p.neg })) : [];
  const id = await queueGraph(buildSamGraph({ image: Sel.upload.name, box, points, text: hit.kind === 'text' ? hit.text : '' }));
  const { img } = await loadOutput(await waitHelper(id));
  return { raw: img, smooth: Math.max(1, 1 / k) };
}
async function requestSelection(hit) {
  if (!S.base || !hit) return;
  const ticket = ++Sel.ticket;
  Sel.busy = true;
  Sel.box = hit.kind === 'box' ? hit.box : null;
  syncSelUi();
  schedule();
  try {
    const got = ON_COMFY ? await samHitMask(hit) : { raw: fakeHitMask(hit), smooth: 1 };
    if (ticket !== Sel.ticket) return;
    // Tìm theo tên thường ra nhiều vật to nhỏ khác nhau, nên chỉ bỏ những mảnh thật nhỏ.
    hit.clean = cleanSamMask(got.raw, hit.kind === 'text' ? 0.005 : 0.04);
    hit.smooth = got.smooth;
    Sel.raw = got.raw;
    rebuildSelection();
  } catch (e) {
    if (ticket === Sel.ticket) setStatus(e.message || String(e));
  } finally {
    if (ticket === Sel.ticket) {
      Sel.busy = false;
      Sel.box = null;
      syncSelUi();
      schedule();
    }
  }
}
function rebuildSelection() {
  const parts = Sel.hits.filter(h => h.clean);
  if (!parts.length) return null;
  const c = newCanvas(parts[0].clean.width, parts[0].clean.height), g = c.getContext('2d');
  for (const h of parts) g.drawImage(h.clean, 0, 0);
  Sel.clean = c;
  Sel.smooth = parts[0].smooth;
  const bounds = applySelSoftness();
  setStatus(bounds ? (ON_COMFY ? '' : 'Bản xem trước: vùng chọn là giả lập, trên ComfyUI là SAM 3.1 thật.') : 'Không tìm thấy vật, thử kéo khung khác, bấm lên vật hoặc tìm theo tên khác.');
  return bounds;
}
function searchByName() {
  const name = $('selName').value.trim();
  if (!name || Sel.busy || !S.base) return;
  const hit = { kind: 'text', text: name };
  Sel.hits = [...Sel.hits, hit];
  requestSelection(hit);
}
// Nới vùng bằng các bản dịch chuyển quanh một vòng tròn. Số bản tăng theo bán kính, để ảnh lớn kéo thanh vẫn nhanh.
function dilate(g, mask, radius) {
  g.drawImage(mask, 0, 0);
  const steps = Math.min(24, Math.max(8, Math.round(radius * 1.2)));
  for (const part of radius > 6 ? [1, 0.5] : [1]) {
    for (let i = 0; i < steps; i++) {
      const t = i / steps * Math.PI * 2;
      g.drawImage(mask, Math.cos(t) * radius * part, Math.sin(t) * radius * part);
    }
  }
}
// Đường viền trắng quanh phần đặc của vùng chọn (sau khi nới), để thấy vùng chọn tới đâu. Quầng mềm nằm ngoài đường này.
function selectionOutline(core) {
  const w = core.width, h = core.height;
  const ring = newCanvas(w, h), rg = ring.getContext('2d');
  dilate(rg, core, Math.max(1.5, 1.5 / V.scale));
  rg.globalCompositeOperation = 'destination-out';
  rg.drawImage(core, 0, 0);
  rg.globalCompositeOperation = 'source-in';
  rg.fillStyle = '#fff';
  rg.fillRect(0, 0, w, h);
  return ring;
}
function clearSelection() {
  Sel.ticket++;
  Object.assign(Sel, { box: null, points: [], hits: [], mask: null, tint: null, ring: null, raw: null, clean: null, busy: false });
}
// Xoá vật: vùng chọn thành vùng tô của một cọ Xoá mới, Gen chạy như cọ Xoá tô tay.
function selectionToRemoveBrush() {
  if (!Sel.mask) return;
  const free = PALETTE.find(c => !S.brushes.some(b => b.color === c.id));
  if (!free) {
    setStatus('Đã đủ 6 cọ, xoá bớt một cọ rồi chọn lại.');
    return;
  }
  pushHistory();
  const b = { id: S.nextId++, color: free.id, type: 'remove', ref: null, refUrl: '', note: '', cover: 0 };
  S.brushes.push(b);
  S.strokes = [...S.strokes, { brush: b.id, erase: false, size: 0, points: [], mask: Sel.mask }];
  clearSelection();
  S.tool = null;
  S.active = b.id;
  renderTiles();
  syncToolUi();
  afterChange();
  setStatus('Đã tô vùng xoá bằng cọ mới. Bấm Gen để xoá.');
}
// Tách thành layer: cắt điểm ảnh trong vùng chọn, chỗ cũ thành vùng xám để lượt Ghép lấp lại.
function selectionToLayer(keep) {
  if (!Sel.mask) return;
  const r = maskBounds(Sel.mask);
  if (!r) return;
  const src = newCanvas(r.w, r.h), g = src.getContext('2d');
  g.drawImage(S.base, -r.x, -r.y);
  g.globalCompositeOperation = 'destination-in';
  g.drawImage(Sel.mask, -r.x, -r.y);
  const corners = [{ x: r.x, y: r.y }, { x: r.x + r.w, y: r.y }, { x: r.x + r.w, y: r.y + r.h }, { x: r.x, y: r.y + r.h }];
  addLayer({ src, hole: Sel.mask, corners, keep });
  clearSelection();
  setStatus((keep ? 'Đã chép, vật vẫn ở chỗ cũ. ' : '') + 'Kéo layer tới chỗ mới. Kéo góc để bẻ phối cảnh, Shift + kéo góc để đổi cỡ, kéo nút tròn để xoay.');
}
function addLayer({ src, hole, corners, keep }) {
  const L = { id: Layers.nextId++, src, hole, keep: !!keep, corners, grid: null, _gray: null };
  Layers.list.push(L);
  Layers.active = L.id;
  S.tool = 'layer';
  closePop();
  renderLayerTiles();
  syncToolUi();
  renderPanel();
}
function addLayerFromFile(file) {
  if (!S.base) {
    setStatus('Thêm ảnh gốc trước khi thêm layer.');
    return;
  }
  loadFile(file, img => {
    const w = img.naturalWidth, h = img.naturalHeight;
    const keep = Math.min(1, 2048 / Math.max(w, h));
    const src = newCanvas(Math.round(w * keep), Math.round(h * keep));
    src.getContext('2d').drawImage(img, 0, 0, src.width, src.height);
    const s = Math.min(1, 0.5 * Math.min(S.w / w, S.h / h));
    const cx = S.w / 2, cy = S.h / 2, hw = w * s / 2, hh = h * s / 2;
    addLayer({ src, hole: null, corners: [{ x: cx - hw, y: cy - hh }, { x: cx + hw, y: cy - hh }, { x: cx + hw, y: cy + hh }, { x: cx - hw, y: cy + hh }] });
    setStatus('Layer từ ảnh khác: bấm Tách nền nếu ảnh còn nền.');
  });
}
function deleteLayer(id) {
  Layers.list = Layers.list.filter(L => L.id !== id);
  if (Layers.active === id) Layers.active = null;
  if (S.tool === 'layer' && !activeLayer()) S.tool = null;
  renderLayerTiles();
  syncToolUi();
  renderPanel();
}
function resetLayers() {
  Layers.list = [];
  Layers.active = null;
  clearSelection();
  Sel.upload = { for: null, name: '', k: 1 };
  if (S.tool) S.tool = null;
  renderLayerTiles();
  syncToolUi();
}
function setLayerMode(mode) {
  Layers.mode = mode;
  const L = activeLayer();
  if (mode === 'warp' && L && !L.grid) {
    const m = quadMap(L.corners);
    L.grid = [0, 0.5, 1].flatMap(v => [0, 0.5, 1].map(u => m(u, v)));
  }
  syncToolUi();
}
function flipLayer() {
  const L = activeLayer();
  if (!L) return;
  const c = newCanvas(L.src.width, L.src.height), g = c.getContext('2d');
  g.translate(c.width, 0);
  g.scale(-1, 1);
  g.drawImage(L.src, 0, 0);
  L.src = c;
  renderLayerTiles();
  schedule();
  renderPanel();
}
// Tách nền layer bằng BiRefNet, như template utility_birefnet_remove_background.
async function cutoutLayer() {
  const L = activeLayer();
  if (!L || Layers.busy) return;
  if (!ON_COMFY) {
    setStatus('Bản xem trước: Tách nền chạy BiRefNet trên ComfyUI.');
    return;
  }
  Layers.busy = true;
  syncToolUi();
  try {
    const name = await uploadCanvas(L.src, `${Date.now().toString(36)}_layer.png`);
    const id = await queueGraph({
      1: { class_type: 'LoadBackgroundRemovalModel', inputs: { bg_removal_name: 'birefnet.safetensors' } },
      4: { class_type: 'LoadImage', inputs: { image: name } },
      3: { class_type: 'RemoveBackground', inputs: { bg_removal_model: ['1', 0], image: ['4', 0] } },
      5: { class_type: 'MaskToImage', inputs: { mask: ['3', 0] } },
      6: { class_type: 'PreviewImage', inputs: { images: ['5', 0] } },
    });
    const { img } = await loadOutput(await waitHelper(id));
    const mask = alphaFromImage(img, L.src.width, L.src.height);
    const g = L.src.getContext('2d');
    g.globalCompositeOperation = 'destination-in';
    g.drawImage(mask, 0, 0);
    g.globalCompositeOperation = 'source-over';
    L.rev = (L.rev || 0) + 1;
    renderLayerTiles();
    renderPanel();
    setStatus('Đã tách nền layer.');
  } catch (e) {
    setStatus(e.message || String(e));
  } finally {
    Layers.busy = false;
    syncToolUi();
    schedule();
  }
}
function renderLayerTiles() {
  const wrap = $('layerTiles');
  wrap.replaceChildren(...Layers.list.map((L, i) => {
    const cell = document.createElement('div');
    cell.className = 'tilewrap';
    const t = document.createElement('button');
    t.type = 'button';
    t.className = 'ltile';
    t.title = `Layer ${i + 1}`;
    t.setAttribute('aria-label', `Layer ${i + 1}`);
    t.dataset.active = String(S.tool === 'layer' && Layers.active === L.id);
    const cv = document.createElement('canvas');
    const k = 38 / Math.max(L.src.width, L.src.height);
    cv.width = Math.max(1, Math.round(L.src.width * k));
    cv.height = Math.max(1, Math.round(L.src.height * k));
    cv.getContext('2d').drawImage(L.src, 0, 0, cv.width, cv.height);
    t.append(cv);
    t.addEventListener('click', () => {
      Layers.active = L.id;
      S.tool = 'layer';
      clearSelection();
      closePop();
      syncToolUi();
    });
    const x = document.createElement('button');
    x.type = 'button';
    x.className = 'tilex';
    x.title = 'Xoá layer này';
    x.setAttribute('aria-label', `Xoá layer ${i + 1}`);
    x.innerHTML = icon('close');
    x.addEventListener('click', e => { e.stopPropagation(); deleteLayer(L.id); });
    cell.append(t, x);
    return cell;
  }));
}
function syncSelUi() {
  const on = S.tool === 'select';
  $('selWrap').hidden = !on;
  $('selHint').hidden = !on;
  $('selBar').hidden = !on || (!Sel.mask && !Sel.busy);
  $('selFind').disabled = Sel.busy;
  $('selText').textContent = Sel.busy ? 'Đang tìm vật bằng SAM 3.1...' : 'Đã chọn vật';
  ['selRemove', 'selLayer', 'selCopy'].forEach(id => { $(id).disabled = Sel.busy || !Sel.mask; });
  $('selFeather').disabled = $('selGrow').disabled = Sel.busy || !Sel.raw;
}
function syncToolUi() {
  $('selectTool').setAttribute('aria-pressed', String(S.tool === 'select'));
  const L = S.tool === 'layer' && activeLayer();
  $('layerBar').hidden = !L;
  $('layerKeep').hidden = !(L && L.hole);
  $('layerKeep').setAttribute('aria-pressed', String(!!(L && L.keep)));
  $('layerMode').querySelectorAll('.viewchip').forEach(c => c.setAttribute('aria-pressed', String(c.dataset.mode === Layers.mode)));
  $('blendLora').setAttribute('aria-pressed', String(S.blendLora));
  $('layerCut').disabled = Layers.busy;
  $('layerCut').textContent = Layers.busy ? 'Đang tách nền...' : 'Tách nền';
  document.querySelectorAll('.ltile').forEach((t, i) => { t.dataset.active = String(S.tool === 'layer' && Layers.list[i] && Layers.list[i].id === Layers.active); });
  syncSelUi();
  view.style.cursor = viewCursor();
  if (S.tool) brushcur.hidden = true;
  schedule();
}
// Ảnh của layer đúng như đang thấy (đã biến dạng, uốn, lật), nền trong suốt.
function renderLayerImage(L) {
  const pts = layerOutline(L);
  const x0 = Math.floor(Math.min(...pts.map(p => p.x))), y0 = Math.floor(Math.min(...pts.map(p => p.y)));
  const x1 = Math.ceil(Math.max(...pts.map(p => p.x))), y1 = Math.ceil(Math.max(...pts.map(p => p.y)));
  const c = newCanvas(Math.max(1, x1 - x0), Math.max(1, y1 - y0)), g = c.getContext('2d');
  g.translate(-x0, -y0);
  drawLayer(g, L, 24, 0.6);
  return c;
}
// Đổi người và Upscale đọc ảnh không có kênh trong suốt, nên phần trong suốt thành nền trắng trơn.
function flattenWhite(canvas) {
  const c = newCanvas(canvas.width, canvas.height), g = c.getContext('2d');
  g.fillStyle = '#fff';
  g.fillRect(0, 0, c.width, c.height);
  g.drawImage(canvas, 0, 0);
  return c;
}
function saveLayer() {
  const L = activeLayer();
  if (!L) return;
  if (!ON_COMFY) {
    setStatus('Bản xem trước không tải file được. Trên trang ComfyUI nút này tải layer về thành PNG nền trong suốt.');
    return;
  }
  renderLayerImage(L).toBlob(blob => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `layer_${L.id}_${Date.now().toString(36)}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, 'image/png');
}
function sendLayer(tab) {
  const L = activeLayer();
  if (!L) return;
  const img = flattenWhite(renderLayerImage(L));
  const url = img.toDataURL('image/png');
  if (tab === 'swap') {
    Swap.slots[1].ref = img;
    Swap.slots[1].refUrl = url;
  } else {
    Up.slots[0].ref = img;
    Up.slots[0].refUrl = url;
    Up.fromBase = null;
  }
  switchTab(tab);
  if (tab === 'swap') syncSwap();
  else syncUpscale();
  renderPanel();
  setStatus(tab === 'swap' ? `Đã đưa layer vào ô ${SWAP_REF_LABEL[Swap.mode].toLowerCase()}, nền trắng trơn.` : 'Đã đưa layer vào ô ảnh cần upscale, nền trắng trơn.');
}
function bindLayerTools() {
  $('selectTool').addEventListener('click', () => {
    if (!S.base) {
      setStatus('Thêm ảnh gốc trước khi chọn vật.');
      return;
    }
    S.tool = S.tool === 'select' ? null : 'select';
    if (S.tool !== 'select') clearSelection();
    else setStatus(SEL_HELP);
    Layers.active = null;
    closePop();
    renderLayerTiles();
    syncToolUi();
  });
  $('selExit').addEventListener('click', () => { S.tool = null; clearSelection(); syncToolUi(); });
  $('selFind').addEventListener('click', searchByName);
  $('selHelp').addEventListener('click', () => setStatus(SEL_HELP));
  $('selName').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); searchByName(); } });
  $('selRemove').addEventListener('click', selectionToRemoveBrush);
  $('selLayer').addEventListener('click', () => selectionToLayer(false));
  $('selCopy').addEventListener('click', () => selectionToLayer(true));
  $('layerKeep').addEventListener('click', () => {
    const L = activeLayer();
    if (!L || !L.hole) return;
    L.keep = !L.keep;
    syncToolUi();
    renderPanel();
  });
  $('layerSave').addEventListener('click', saveLayer);
  $('layerToSwap').addEventListener('click', () => sendLayer('swap'));
  $('layerToUp').addEventListener('click', () => sendLayer('upscale'));
  $('selClear').addEventListener('click', () => { clearSelection(); syncSelUi(); schedule(); });
  let softFrame = 0;
  const reshape = () => {
    if (!Sel.clean || softFrame) return;
    softFrame = requestAnimationFrame(() => {
      softFrame = 0;
      applySelSoftness();
      schedule();
    });
  };
  $('selFeather').addEventListener('input', e => { Sel.feather = Number(e.target.value); reshape(); });
  $('selGrow').addEventListener('input', e => { Sel.grow = Number(e.target.value); reshape(); });
  $('layerFile').addEventListener('change', e => {
    const file = e.target.files[0];
    e.target.value = '';
    if (file) addLayerFromFile(file);
  });
  $('layerMode').addEventListener('click', e => {
    const chip = e.target.closest('.viewchip');
    if (chip) setLayerMode(chip.dataset.mode);
  });
  $('layerFlip').addEventListener('click', flipLayer);
  $('layerCut').addEventListener('click', cutoutLayer);
  $('blendLora').addEventListener('click', () => { S.blendLora = !S.blendLora; syncToolUi(); renderPanel(); });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && S.tool && !e.target.matches('input,textarea,select')) {
      S.tool = null;
      clearSelection();
      syncToolUi();
    }
  });
}

// Lượt Ghép: ảnh có layer và vùng xám, mask là vùng lấp cộng vùng quanh layer, dán lại đúng vùng đó nên phần còn lại giữ nguyên từng điểm ảnh.
function mergeSnapshot(pass) {
  return {
    text: promptFor(pass),
    lora: S.blendLora,
    layers: Layers.list.map(L => ({ src: L.src, hole: L.hole, keep: L.keep, _gray: L._gray, corners: L.corners.map(q => ({ ...q })), grid: L.grid && L.grid.map(q => ({ ...q })) })),
  };
}
function composeMerge(current, job, pads) {
  const out = newCanvas(current.width, current.height), g = out.getContext('2d');
  g.drawImage(current, 0, 0);
  g.translate(pads.left, pads.top);
  for (const L of job.layers) if (L.hole && !L.keep) g.drawImage(holeGray(L), 0, 0);
  for (const L of job.layers) drawLayer(g, L, 24, 0.6);
  return out;
}
function mergeMasks(job, w, h, pads) {
  const hole = newCanvas(w, h), foot = newCanvas(w, h), area = newCanvas(w, h);
  const hg = hole.getContext('2d'), fg = foot.getContext('2d');
  hg.translate(pads.left, pads.top);
  fg.translate(pads.left, pads.top);
  for (const L of job.layers) {
    if (L.hole && !L.keep) stampMask(hg, L.hole, 4);
    drawLayer(fg, L, 16, 0.6);
  }
  const ag = area.getContext('2d');
  ag.drawImage(hole, 0, 0);
  stampMask(ag, foot, Math.round(Math.max(16, 0.02 * Math.max(w, h))));
  const marks = job.layers.some(L => L.hole && !L.keep) ? [{ mask: hole, hex: PAINT.red.hex }] : [];
  marks.push({ mask: foot, hex: PAINT.blue.hex });
  return { area, marks };
}
function paintMergeThumb(cv) {
  cv.width = 144;
  cv.height = Math.max(1, Math.round(H() * 144 / W()));
  const g = cv.getContext('2d');
  g.scale(144 / W(), 144 / W());
  g.fillStyle = PAD_GRAY;
  g.fillRect(0, 0, W(), H());
  g.drawImage(S.base, S.pads.left, S.pads.top, S.w, S.h);
  g.translate(S.pads.left, S.pads.top);
  for (const L of Layers.list) if (L.hole && !L.keep) g.drawImage(holeGray(L), 0, 0);
  for (const L of Layers.list) drawLayer(g, L, 8, 1);
}
async function requireBlendLora() {
  const info = await fetch('/api/object_info/LoraLoaderModelOnly').then(r => (r.ok ? r.json() : {}), () => ({}));
  if (!info.LoraLoaderModelOnly) return;
  if (!info.LoraLoaderModelOnly.input.required.lora_name[0].includes(BLEND_LORA)) {
    throw new Error(`ComfyUI chưa có LoRA ánh sáng ${BLEND_LORA}. Cloud Sync tải trong nền, chờ tải xong rồi Gen lại, hoặc tắt LoRA ánh sáng.`);
  }
}
function loadFile(file, done) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => done(img, url);
  img.src = url;
}
function setBase(img, sample, fromResult) {
  stopCapture();
  S.base = img;
  S.isSample = !!sample;
  S.baseFromResult = !!fromResult;
  S.w = img.naturalWidth || img.width;
  S.h = img.naturalHeight || img.height;
  S.pads = { left: 0, right: 0, top: 0, bottom: 0 };
  S.strokes = [];
  S.past = [];
  S.future = [];
  S.overrides = { ...S.overrides, expand: null, edit: null };
  resetLayers();
  syncStage();
  afterChange();
  fit();
}
function syncStage() {
  const edit = S.tab === 'edit';
  frame.hidden = !edit || !S.base;
  $('drop').hidden = !edit || !!S.base;
  $('sampleTag').hidden = !edit || !S.base || !S.isSample;
  $('sheetEmpty').hidden = S.tab !== 'sheet';
  $('rot').hidden = S.tab !== 'rotate';
  $('swap').hidden = S.tab !== 'swap';
  $('upscale').hidden = S.tab !== 'upscale';
  $('chat').hidden = S.tab !== 'chat';
}
function takeBase(file) {
  if (file && file.type.startsWith('image/')) loadFile(file, img => setBase(img, false));
}
['baseFile', 'baseFile2'].forEach(id => $(id).addEventListener('change', e => {
  takeBase(e.target.files[0]);
  e.target.value = '';
}));
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());
stage.addEventListener('dragover', () => stage.classList.add('over'));
stage.addEventListener('dragleave', () => stage.classList.remove('over'));
stage.addEventListener('drop', e => {
  stage.classList.remove('over');
  if (S.tab === 'sheet') takeSheetFiles(e.dataTransfer.files, Math.max(0, Sheet.slots.findIndex(s => !s.ref)));
  else if (S.tab === 'rotate') takeRotateFiles(e.dataTransfer.files, 0);
  else if (S.tab === 'swap') takeSwapFiles(e.dataTransfer.files, Swap.slots[0].ref ? 1 : 0);
  else if (S.tab === 'upscale') takeUpscaleFiles(e.dataTransfer.files, 0);
  else takeBase(e.dataTransfer.files[0]);
});

function toggleDrawer(open) {
  $('drawer').hidden = !open;
  $('promptBtn').setAttribute('aria-expanded', String(open));
}
$('promptBtn').addEventListener('click', () => toggleDrawer($('drawer').hidden));
$('drawerClose').addEventListener('click', () => toggleDrawer(false));
$('addBrush').addEventListener('click', addBrush);
$('scene').addEventListener('input', e => { S.scene = e.target.value; renderPanel(); });
$('size').addEventListener('input', e => { S.size = Number(e.target.value); });
$('erase').addEventListener('click', () => { S.erase = !S.erase; syncTools(); });
$('undo').addEventListener('click', () => restore(S.past, S.future));
$('redo').addEventListener('click', () => restore(S.future, S.past));
$('clearStrokes').addEventListener('click', () => { pushHistory(); S.strokes = []; afterChange(); });
function clearBase() {
  stopCapture();
  S.baseFromResult = false;
  S.base = null; S.w = 0; S.h = 0;
  S.pads = { left: 0, right: 0, top: 0, bottom: 0 };
  S.strokes = []; S.past = []; S.future = []; S.scene = '';
  S.overrides = { ...S.overrides, expand: null, edit: null };
  S.isSample = false;
  resetLayers();
  $('scene').value = '';
  S.env.forEach(slot => { slot.ref = null; slot.refUrl = ''; });
  syncEnv();
  closePop();
  syncStage();
  rebuildLayer();
  measureCover();
  syncBar();
  renderPanel();
}
$('sampleTag').addEventListener('click', clearBase);
window.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closePop(); $('viewer').hidden = true; stopCapture(); return; }
  if (S.tab === 'rotate' && !e.target.matches('input,textarea,select') && nudgeRotate(e.key)) { e.preventDefault(); return; }
  if (e.target.matches('input,textarea,select') || S.tab !== 'edit') return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
    e.preventDefault();
    if (e.shiftKey) restore(S.future, S.past); else restore(S.past, S.future);
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); restore(S.future, S.past); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const n = Number(e.key);
  if (n >= 1 && n <= S.brushes.length) { selectBrush(S.brushes[n - 1].id); closePop(); }
  else if (e.key === '+') addBrush();
  else if (e.key.toLowerCase() === 'e') { S.erase = !S.erase; syncTools(); }
  else if (e.key === '[') { S.size = Math.max(8, S.size - 8); syncTools(); }
  else if (e.key === ']') { S.size = Math.min(400, S.size + 8); syncTools(); }
});

// Chỉ khi trang được phục vụ từ chính ComfyUI (/extensions/vstone/...) nút Gen mới chạy. Không gọi địa chỉ nào khác.
const ON_COMFY = location.pathname.includes('/extensions/vstone/');
const OUTPAINT_LORA = 'qwen-image-2.1-outpaint-v2.safetensors';
const STITCH_FEATHER = 32;
// Do giao diện ComfyUI xuất từ workflow image_qwen_image_2_1_image_edit_light, bản đã chạy trên H200.
const EDIT_GRAPH = {
  1: { class_type: 'UNETLoader', inputs: { unet_name: 'qwen_image_2.1_int8_convrot.safetensors', weight_dtype: 'default' } },
  2: { class_type: 'CLIPLoader', inputs: { clip_name: 'qwen3vl_8b_int8_convrot.safetensors', type: 'qwen_image', device: 'default' } },
  3: { class_type: 'VAELoader', inputs: { vae_name: 'qwen_image_2.1_vae_bf16.safetensors' } },
  4: { class_type: 'LoadImage', inputs: { image: '' } },
  6: { class_type: 'SaveImage', inputs: { filename_prefix: 'Concept_edit', images: ['9', 0] } },
  7: { class_type: 'TextEncodeQwenImage21', inputs: { clip: ['2', 0], 'images.image_1': ['4', 0], vae: ['3', 0], prompt: '', negative_prompt: '', resolution: 1024 } },
  8: { class_type: 'KSampler', inputs: { model: ['1', 0], positive: ['7', 0], negative: ['7', 1], latent_image: ['7', 2], seed: 0, steps: 25, cfg: 1, sampler_name: 'euler', scheduler: 'simple', denoise: 1 } },
  9: { class_type: 'VAEDecode', inputs: { samples: ['8', 0], vae: ['3', 0] } },
};

const SHEET_VIEWS = [
  { id: 'front', label: 'Trước', en: 'front view', vi: 'nhìn trước', on: true },
  { id: 'three', label: '3/4', en: 'three-quarter view', vi: 'góc 3/4', on: true },
  { id: 'side', label: 'Nghiêng', en: 'side view', vi: 'nhìn nghiêng', on: true },
  { id: 'back', label: 'Sau', en: 'back view', vi: 'nhìn sau', on: true },
  { id: 'face', label: 'Cận mặt', on: false },
];
const Sheet = {
  slots: [
    { role: 'body', label: 'Toàn thân' },
    { role: 'face', label: 'Chân dung' },
    { role: 'extra', label: 'Ảnh thêm' },
    { role: 'extra', label: 'Ảnh thêm' },
  ].map(slot => ({ ...slot, ref: null, refUrl: '' })),
  views: new Set(SHEET_VIEWS.filter(v => v.on).map(v => v.id)),
  note: '',
};
const imageTags = numbers => numbers.map(n => `<image${n}>`);
const joinList = (items, word) => (items.length > 1 ? `${items.slice(0, -1).join(', ')} ${word} ${items[items.length - 1]}` : items[0]);

// Ô đã có ảnh được đánh số image theo thứ tự, ô trống không chiếm số.
function buildSheetPasses() {
  const filled = Sheet.slots.filter(s => s.ref).map((slot, i) => ({ role: slot.role, n: i + 1 }));
  if (!filled.length) return { passes: [], problems: [] };
  const picked = SHEET_VIEWS.filter(v => v.en && Sheet.views.has(v.id));
  if (!picked.length) return { passes: [], problems: [{ brush: null, text: 'Chọn ít nhất một góc nhìn' }] };
  const tags = imageTags(filled.map(f => f.n));
  const allEn = joinList(tags, 'and'), allVi = joinList(tags, 'và');
  const lines = [{
    en: `Using the character in ${allEn}, create a character reference sheet on a plain light gray background: ${picked.map(v => v.en).join(', ')}, full body, same scale, evenly spaced in one row.`,
    vi: `Dựa trên nhân vật trong ${allVi}, tạo bảng nhân vật trên nền xám nhạt trơn: ${picked.map(v => v.vi).join(', ')}, toàn thân, cùng tỉ lệ, cách đều trên một hàng.`,
  }];
  if (filled.length > 1) {
    lines.push({ en: `${allEn} show the same character.`, vi: `${allVi} là cùng một nhân vật.` });
    const body = filled.find(f => f.role === 'body'), face = filled.find(f => f.role === 'face');
    if (body) lines.push({ en: `Take the body and outfit from <image${body.n}>.`, vi: `Lấy vóc dáng và trang phục từ <image${body.n}>.` });
    if (face) lines.push({ en: `Take the face and hair from <image${face.n}>.`, vi: `Lấy mặt và tóc từ <image${face.n}>.` });
    const extra = imageTags(filled.filter(f => f.role === 'extra').map(f => f.n));
    if (extra.length) lines.push({ en: `Use ${joinList(extra, 'and')} for more details of the same character.`, vi: `Dùng ${joinList(extra, 'và')} để lấy thêm chi tiết của nhân vật.` });
  }
  lines.push(
    { en: `Keep the face, hair, body shape, outfit and colors identical to ${allEn} in every view.`, vi: `Giữ mặt, tóc, vóc dáng, trang phục và màu sắc giống hệt ${allVi} ở mọi góc.` },
    { en: 'Neutral standing pose, even studio lighting, no text, no labels.', vi: 'Dáng đứng trung tính, ánh sáng studio đều, không chữ, không nhãn.' },
  );
  if (Sheet.views.has('face')) lines.push({ en: 'Add a close-up of the face.', vi: 'Thêm một ô cận mặt.' });
  const note = Sheet.note.trim().replace(/[.\s]+$/, '');
  if (note) lines.push({ en: `${note}.`, vi: `${note}.` });
  return { passes: [{ kind: 'sheet', name: 'Char sheet', lines, slots: [] }], problems: [] };
}
function syncSlotBoxes(containerId, slots) {
  document.querySelectorAll(`#${containerId} .sheetwrap`).forEach((wrap, i) => {
    const slot = slots[i];
    const has = !!slot.ref;
    const img = wrap.querySelector('img');
    img.hidden = !has;
    if (has) img.src = slot.refUrl;
    wrap.querySelector('.sheetempty-box').hidden = has;
    wrap.querySelector('.slotlabel').hidden = !has;
    wrap.querySelector('.sheetx').hidden = !has;
  });
}
function syncSheetSlots() {
  syncSlotBoxes('sheetSlots', Sheet.slots);
}
// Ảnh đầu vào ô được chọn, các ảnh sau vào lần lượt những ô còn trống.
function takeFilesInto(slots, fileList, start, sync) {
  const files = [...fileList].filter(file => file.type.startsWith('image/'));
  const targets = [start, ...slots.map((slot, i) => i).filter(i => i !== start && !slots[i].ref)];
  if (files.length > targets.length) setStatus(`Mỗi lần chỉ nhận thêm được ${targets.length} ảnh, các ảnh còn lại đã bỏ qua.`);
  files.slice(0, targets.length).forEach((file, i) => {
    loadFile(file, (img, url) => {
      slots[targets[i]].ref = img;
      slots[targets[i]].refUrl = url;
      sync();
      renderPanel();
    });
  });
}
function takeSheetFiles(fileList, start) {
  takeFilesInto(Sheet.slots, fileList, start, syncSheetSlots);
}
function toggleEnv(open) {
  $('envPop').hidden = !open;
  $('envBtn').setAttribute('aria-expanded', String(open));
  if (open) placePop($('envPop'), $('envBtn'));
}
function syncEnv() {
  document.querySelectorAll('#envSlots .imgwrap').forEach((wrap, i) => {
    const slot = S.env[i], has = !!slot.ref, img = wrap.querySelector('img');
    img.hidden = !has;
    if (has) img.src = slot.refUrl;
    wrap.querySelector('.envplus').hidden = has;
    wrap.querySelector('.imgx').hidden = !has;
  });
  const filled = S.env.filter(s => s.ref);
  $('envThumb').hidden = !filled.length;
  if (filled.length) $('envThumb').src = filled[0].refUrl;
  $('envCount').hidden = !filled.length;
  $('envCount').textContent = filled.length;
}
function buildEnvBox() {
  $('envSlots').innerHTML = S.env.map((slot, i) => `
    <div class="imgwrap">
      <label class="imgpick" title="Ảnh môi trường">
        <span class="envplus">${icon('plus')}</span>
        <img alt="" hidden>
        <input class="sr" type="file" accept="image/*" multiple aria-label="Ảnh môi trường ${i + 1}">
      </label>
      <button class="imgx" type="button" title="Bỏ ảnh" aria-label="Bỏ ảnh ${i + 1}" hidden>${icon('close')}</button>
    </div>`).join('');
  document.querySelectorAll('#envSlots .imgwrap').forEach((wrap, i) => {
    const box = wrap.querySelector('.imgpick');
    wrap.querySelector('input').addEventListener('change', e => {
      takeFilesInto(S.env, e.target.files, i, syncEnv);
      e.target.value = '';
    });
    wrap.querySelector('.imgx').addEventListener('click', () => {
      S.env[i].ref = null;
      S.env[i].refUrl = '';
      syncEnv();
      renderPanel();
    });
    box.addEventListener('dragover', e => { e.preventDefault(); box.classList.add('over'); });
    box.addEventListener('dragleave', () => box.classList.remove('over'));
    box.addEventListener('drop', e => {
      e.preventDefault();
      box.classList.remove('over');
      takeFilesInto(S.env, e.dataTransfer.files, i, syncEnv);
    });
  });
  $('envBtn').addEventListener('click', () => toggleEnv($('envPop').hidden));
  // Để so sánh: tắt LoRA, LoRA 0,6 hoặc LoRA 1 cho lượt kéo rộng. Lượt sửa không đổi.
  $('envLora').addEventListener('click', e => {
    const chip = e.target.closest('.viewchip');
    if (!chip) return;
    S.expandLora = Number(chip.dataset.strength);
    $('envLora').querySelectorAll('.viewchip').forEach(c => c.setAttribute('aria-pressed', String(c === chip)));
    renderPanel();
  });
}
function buildSlotBoxes(containerId, slots, take, sync) {
  $(containerId).innerHTML = slots.map(slot => `
    <div class="sheetwrap">
      <label class="sheetbox" title="${slot.label}">
        <span class="sheetempty-box">${icon('plus')}<span>${slot.label}</span></span>
        <img alt="${slot.label}" hidden>
        <span class="slotlabel" hidden>${slot.label}</span>
        <input class="sr" type="file" accept="image/*" multiple aria-label="${slot.label}">
      </label>
      <button class="sheetx" type="button" title="Bỏ ảnh" aria-label="Bỏ ảnh ${slot.label}" hidden>${icon('close')}</button>
    </div>`).join('');
  document.querySelectorAll(`#${containerId} .sheetwrap`).forEach((wrap, i) => {
    const box = wrap.querySelector('.sheetbox');
    wrap.querySelector('input').addEventListener('change', e => {
      take(e.target.files, i);
      e.target.value = '';
    });
    wrap.querySelector('.sheetx').addEventListener('click', () => {
      slots[i].ref = null;
      slots[i].refUrl = '';
      sync();
      renderPanel();
    });
    box.addEventListener('dragover', e => { e.preventDefault(); e.stopPropagation(); box.classList.add('over'); });
    box.addEventListener('dragleave', () => box.classList.remove('over'));
    box.addEventListener('drop', e => {
      e.preventDefault();
      e.stopPropagation();
      box.classList.remove('over');
      take(e.dataTransfer.files, i);
    });
  });
}
function buildSheetSlots() {
  buildSlotBoxes('sheetSlots', Sheet.slots, takeSheetFiles, syncSheetSlots);
}

// Tab Xoay. Dummy: con mèo giả để biết hướng, 8 hướng theo LoRA QMA. Splat: khối dựng từ ảnh, xoay góc tuỳ ý theo LoRA AnyAngle.
const ROTATE_AZ = [
  { en: 'front view', vi: 'Trước' },
  { en: 'front-right quarter view', vi: 'Trước phải' },
  { en: 'right side view', vi: 'Phải' },
  { en: 'back-right quarter view', vi: 'Sau phải' },
  { en: 'back view', vi: 'Sau' },
  { en: 'back-left quarter view', vi: 'Sau trái' },
  { en: 'left side view', vi: 'Trái' },
  { en: 'front-left quarter view', vi: 'Trước trái' },
];
const ROTATE_EL = [
  { en: 'low-angle shot', vi: 'Từ dưới', deg: -30 },
  { en: 'eye-level shot', vi: 'Ngang', deg: 0 },
  { en: 'elevated shot', vi: 'Hơi cao', deg: 30 },
  { en: 'high-angle shot', vi: 'Từ trên', deg: 60 },
];
const ROTATE_DIST = [
  { en: 'wide shot', vi: 'Xa', z: 9 },
  { en: 'medium shot', vi: 'Vừa', z: 6.6 },
  { en: 'close-up', vi: 'Gần', z: 4.6 },
];
const Rotate = {
  slots: [
    { label: 'Ảnh chính' },
    { label: 'Bên phải', en: 'the right side', vi: 'bên phải' },
    { label: 'Bên trái', en: 'the left side', vi: 'bên trái' },
    { label: 'Phía sau', en: 'the back', vi: 'phía sau' },
    { label: 'Phía trên', en: 'the top', vi: 'phía trên' },
  ].map(slot => ({ ...slot, ref: null, refUrl: '' })),
  mode: 'dummy',
  az: 0,
  el: 0,
  dist: 1,
  splatZoom: 1,
  splat: null,
  builtFrom: null,
  buildTimer: 0,
  mainCanvas: null,
};
const rotateAzIndex = () => ((Math.round(Rotate.az / 45) % 8) + 8) % 8;
const rotateElIndex = () => ROTATE_EL.reduce((best, e, i) => (Math.abs(e.deg - Rotate.el) < Math.abs(ROTATE_EL[best].deg - Rotate.el) ? i : best), 0);
const wrapDegrees = d => (((d % 360) + 540) % 360) - 180;

function buildRotatePasses() {
  if (!Rotate.slots[0].ref) return { passes: [], problems: [] };
  if (Rotate.mode === 'splat' && Rotate.splat !== 'ready') return { passes: [], problems: [] };
  const extras = Rotate.slots.slice(1).filter(s => s.ref);
  const first = Rotate.mode === 'splat' ? 3 : 2;
  const lines = [];
  if (Rotate.mode === 'splat') {
    lines.push({ en: 'Change the camera angle from <image2> to <image1>.', vi: 'Đổi góc máy theo <image2>, giữ nội dung và phong cách của <image1>.' });
  } else {
    const az = ROTATE_AZ[rotateAzIndex()], el = ROTATE_EL[rotateElIndex()], dist = ROTATE_DIST[Rotate.dist];
    lines.push({ en: `<qma> ${az.en} ${el.en} ${dist.en}${extras.length ? '.' : ''}`, vi: `Xoay tới: ${az.vi}, ${el.vi}, ${dist.vi}.` });
  }
  if (extras.length) {
    lines.push({
      en: `${joinList(extras.map((s, i) => `<image${i + first}> shows ${s.en} of the same subject`), 'and')}. Use ${extras.length > 1 ? 'them' : 'it'} only for details that <image1> does not show, and keep the design, colors and materials of <image1>.`,
      vi: `${joinList(extras.map((s, i) => `<image${i + first}> là ${s.vi} của cùng vật đó`), 'và')}. Chỉ dùng chúng cho chi tiết mà <image1> không có, giữ thiết kế, màu và chất liệu của <image1>.`,
    });
  }
  return { passes: [{ kind: 'rotate', name: Rotate.mode === 'splat' ? 'Xoay bằng splat' : 'Xoay', lines, slots: [] }], problems: [] };
}

// Con mèo ngồi kiểu tượng gốm: mặt và bụng kem ở phía trước, lưng vằn và đuôi dựng ở phía sau, nhìn là biết đang xem hướng nào.
function buildCatMesh() {
  const out = [];
  const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
  const ramp = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = v => { const l = Math.hypot(v[0], v[1], v[2]); return l > 1e-9 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0]; };
  const spline = (pts, t) => {
    const f = Math.max(0, Math.min(0.9999, t)) * (pts.length - 1), i = Math.floor(f), s = f - i;
    const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
    return p1.map((_, k) => 0.5 * (2 * p1[k] + (p2[k] - p0[k]) * s + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * s * s + (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * s * s * s));
  };
  // Lưới trơn: pháp tuyến tính từ các điểm lân cận và hướng ra ngoài theo tâm tham chiếu ref(u, v).
  const surface = (fn, ref, nu, nv, color) => {
    const grid = [];
    for (let i = 0; i <= nu; i++) {
      const row = [];
      for (let j = 0; j < nv; j++) row.push(fn(i / nu, j / nv));
      grid.push(row);
    }
    const at = (i, j) => grid[Math.max(0, Math.min(nu, i))][((j % nv) + nv) % nv];
    const vertex = (i, j) => {
      const p = at(i, j), outward = sub(p, ref(i / nu, j / nv));
      let n = unit(cross(sub(at(i + 1, j), at(i - 1, j)), sub(at(i, j + 1), at(i, j - 1))));
      if (dot(n, n) < 0.5) n = unit(outward);
      if (dot(n, outward) < 0) n = [-n[0], -n[1], -n[2]];
      const c = typeof color === 'function' ? color(p, n) : color;
      out.push(p[0], p[1], p[2], n[0], n[1], n[2], c[0], c[1], c[2]);
    };
    for (let i = 0; i < nu; i++) {
      for (let j = 0; j < nv; j++) {
        for (const [a, b] of [[i, j], [i + 1, j], [i + 1, j + 1], [i, j], [i + 1, j + 1], [i, j + 1]]) vertex(a, b);
      }
    }
  };
  const ball = (c, r, color, nu = 24, nv = 36, shape) => surface((u, v) => {
    const t = Math.PI * u, f = 2 * Math.PI * v;
    const s = [Math.sin(t) * Math.cos(f), Math.cos(t), Math.sin(t) * Math.sin(f)];
    const k = shape ? shape(s) : [1, 1, 1];
    return [c[0] + r[0] * s[0] * k[0], c[1] + r[1] * s[1] * k[1], c[2] + r[2] * s[2] * k[2]];
  }, () => c, nu, nv, color);
  const tube = (pts, radius, color, nu = 40, nv = 14) => {
    const center = t => spline(pts, t);
    const frames = [];
    let side = null;
    for (let i = 0; i <= nu; i++) {
      const u = i / nu;
      const d = unit(sub(center(Math.min(1, u + 0.01)), center(Math.max(0, u - 0.01))));
      if (side) {
        const k = dot(side, d);
        side = unit([side[0] - k * d[0], side[1] - k * d[1], side[2] - k * d[2]]);
      } else {
        side = unit(cross(d, Math.abs(d[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0]));
      }
      frames.push([side, cross(d, side)]);
    }
    surface((u, v) => {
      const c = center(u), [a, b] = frames[Math.round(u * nu)];
      const f = 2 * Math.PI * v, r = radius(u);
      return [0, 1, 2].map(k => c[k] + r * (Math.cos(f) * a[k] + Math.sin(f) * b[k]));
    }, u => center(u), nu, nv, color);
  };
  const ear = (base, tip, width, thick, color, push = 0) => surface((u, v) => {
    const f = 2 * Math.PI * v, k = (1 - u) ** 0.72;
    return [
      base[0] + (tip[0] - base[0]) * u + width * k * Math.cos(f),
      base[1] + (tip[1] - base[1]) * u,
      base[2] + (tip[2] - base[2]) * u + thick * k * Math.sin(f) + push,
    ];
  }, u => [base[0] + (tip[0] - base[0]) * u, base[1] + (tip[1] - base[1]) * u, base[2] + (tip[2] - base[2]) * u + push], 24, 28, color);

  const coat = [0.93, 0.61, 0.32], stripe = [0.78, 0.44, 0.2], cream = [0.99, 0.93, 0.82], pink = [0.95, 0.63, 0.62], ink = [0.13, 0.11, 0.15], white = [1, 1, 1];
  // Thân: tròn ở dưới, thon dần lên cổ, phía trước màu kem, sau lưng có vằn.
  const bodyProfile = [[0.0, -1.1], [0.42, -1.1], [0.66, -0.92], [0.7, -0.6], [0.6, -0.22], [0.45, 0.14], [0.34, 0.42], [0.0, 0.56]];
  const bodyColor = (p, n) => {
    const base = lerp(coat, cream, ramp(0.25, 0.8, n[2]));
    const band = ramp(0.35, 0.75, Math.sin(p[1] * 13 + 0.6)) * ramp(-0.05, -0.45, n[2]) * ramp(0.75, 0.4, Math.abs(n[0]));
    return lerp(base, stripe, band);
  };
  surface((u, v) => {
    const [r, y] = spline(bodyProfile, u), f = 2 * Math.PI * v;
    return [r * Math.cos(f), y, r * 0.92 * Math.sin(f) - 0.1 * (1 - ramp(-1.1, 0.4, y))];
  }, u => [0, spline(bodyProfile, u)[1], -0.1 * (1 - ramp(-1.1, 0.4, spline(bodyProfile, u)[1]))], 64, 72, bodyColor);
  // Đùi sau và bàn chân.
  for (const s of [-1, 1]) {
    ball([s * 0.5, -0.76, -0.12], [0.29, 0.34, 0.44], (p, n) => lerp(coat, cream, ramp(0.55, 0.95, n[2]) * 0.6));
    ball([s * 0.46, -1.03, 0.3], [0.17, 0.085, 0.24], cream);
    ball([s * 0.2, -1.03, 0.56], [0.15, 0.095, 0.19], cream);
  }
  // Đầu: má hơi bè, nửa dưới mặt màu kem, trán có ba vằn.
  const headC = [0, 0.8, 0.06];
  const headColor = (p, n) => {
    const y = p[1] - headC[1];
    const face = ramp(0.15, 0.6, n[2]) * (1 - ramp(-0.08, 0.08, y - 0.02 + 0.32 * Math.abs(p[0])));
    let c = lerp(coat, cream, face);
    for (const x of [-0.15, 0, 0.15]) {
      const mark = (1 - ramp(0.02, 0.05, Math.abs(p[0] - x))) * ramp(0.2, 0.3, y) * ramp(0.05, 0.3, n[2] + 0.3);
      c = lerp(c, stripe, mark);
    }
    return c;
  };
  ball(headC, [0.6, 0.47, 0.5], headColor, 72, 96, s => {
    const cheek = ramp(0.25, -0.25, s[1]) * (1 - ramp(-0.25, -0.85, s[1]));
    return [1 + 0.1 * cheek, 1, 1];
  });
  // Tai.
  for (const s of [-1, 1]) {
    ear([s * 0.27, 0.9, 0.04], [s * 0.5, 1.6, 0.0], 0.3, 0.13, coat);
    ear([s * 0.3, 1.02, 0.11], [s * 0.47, 1.5, 0.05], 0.17, 0.035, pink);
  }
  // Mắt, mũi, miệng, ria.
  for (const s of [-1, 1]) {
    ball([s * 0.23, 0.84, 0.5], [0.068, 0.092, 0.03], ink, 14, 18);
    ball([s * 0.21, 0.875, 0.528], [0.021, 0.024, 0.008], white, 8, 10);
    tube([[0, 0.665, 0.55], [s * 0.035, 0.635, 0.552], [s * 0.085, 0.65, 0.535]], () => 0.009, ink, 14, 6);
    for (const [y, lift] of [[0.72, 0.03], [0.68, -0.03]]) {
      tube([[s * 0.3, y, 0.42], [s * 0.5, y + lift * 0.6, 0.39], [s * 0.72, y + lift, 0.31]], u => 0.007 * (1 - u * 0.7), [0.97, 0.95, 0.9], 12, 5);
    }
  }
  ball([0, 0.695, 0.56], [0.038, 0.026, 0.02], pink, 10, 14);
  // Vòng cổ và chuông ở phía trước.
  surface((u, v) => {
    const f = 2 * Math.PI * u, t = 2 * Math.PI * v, r = 0.05;
    return [(0.37 + r * Math.cos(t)) * Math.sin(f), 0.38 + r * Math.sin(t), (0.34 + r * Math.cos(t)) * Math.cos(f)];
  }, u => [0.37 * Math.sin(2 * Math.PI * u), 0.38, 0.34 * Math.cos(2 * Math.PI * u)], 72, 12, [0.83, 0.2, 0.25]);
  ball([0, 0.28, 0.4], [0.07, 0.07, 0.07], [0.98, 0.78, 0.3], 16, 22);
  // Đuôi dựng phía sau, chóp đậm.
  const tailPts = [[0, -0.92, -0.62], [0.12, -0.98, -0.98], [0.42, -0.72, -1.1], [0.55, -0.22, -1.02], [0.4, 0.2, -0.96], [0.5, 0.56, -0.88]];
  tube(tailPts, u => 0.125 * Math.sqrt(Math.max(0, 1 - ramp(0.86, 1, u) ** 2)) * (1 - 0.18 * u), (p, n) => lerp(coat, stripe, ramp(0.05, 0.3, p[1])), 72, 18);

  // Bóng mềm dưới chân và vòng mảnh, chấm sáng chỉ phía trước. Pháp tuyến 0 là không đổ sáng.
  const floor = -1.14, bg = [0.09, 0.094, 0.106], dark = [0.035, 0.037, 0.043];
  const seg = 72, rings = 14;
  for (let i = 0; i < rings; i++) {
    const r0 = 1.3 * i / rings, r1 = 1.3 * (i + 1) / rings;
    const c0 = lerp(dark, bg, ramp(0.3, 1, i / rings)), c1 = lerp(dark, bg, ramp(0.3, 1, (i + 1) / rings));
    for (let j = 0; j < seg; j++) {
      const a = 2 * Math.PI * j / seg, b = 2 * Math.PI * (j + 1) / seg;
      const put = (r, f, c) => out.push(r * Math.sin(f), floor, r * Math.cos(f), 0, 0, 0, c[0], c[1], c[2]);
      put(r0, a, c0); put(r1, a, c1); put(r1, b, c1);
      put(r0, a, c0); put(r1, b, c1); put(r0, b, c0);
    }
  }
  surface((u, v) => {
    const f = 2 * Math.PI * u, t = 2 * Math.PI * v, R = 1.42, r = 0.007;
    return [(R + r * Math.cos(t)) * Math.sin(f), floor + r * Math.sin(t), (R + r * Math.cos(t)) * Math.cos(f)];
  }, u => [1.42 * Math.sin(2 * Math.PI * u), floor, 1.42 * Math.cos(2 * Math.PI * u)], 120, 6, [0.36, 0.37, 0.42]);
  ball([0, floor, 1.42], [0.05, 0.05, 0.05], [0.94, 0.93, 0.9], 12, 16);
  return new Float32Array(out);
}

// Splat đã đổi về hệ trục Y hướng lên: vị trí, ma trận hiệp phương sai và màu của từng điểm.
function emptyGaussians(n) {
  return { n, pos: new Float32Array(n * 3), cov: new Float32Array(n * 6), col: new Uint8Array(n * 4) };
}
// Đọc file .spz (Niantic v2) do node SplatToFile3D ghi. Công thức giải mã theo _parse_spz_gaussian của ComfyUI.
const SPZ_COLOR_SCALE = 0.28209479177387814 / 0.15;
async function parseSpz(buffer) {
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  const raw = new Uint8Array(await new Response(stream).arrayBuffer());
  const head = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  if (head.getUint32(0, true) !== 0x5053474E || head.getUint32(4, true) !== 2) throw new Error('File splat không đúng định dạng.');
  const n = head.getUint32(8, true);
  const unit = 1 / (1 << raw[13]);
  const offA = 16 + n * 9, offC = offA + n, offS = offC + n * 3, offQ = offS + n * 3;
  const g = emptyGaussians(n);
  let m = 0;
  for (let i = 0; i < n; i++) {
    if (raw[offA + i] < 4) continue;
    const axis = k => {
      const p = 16 + i * 9 + k * 3;
      const v = raw[p] | (raw[p + 1] << 8) | (raw[p + 2] << 16);
      return (v & 0x800000 ? v - 0x1000000 : v) * unit;
    };
    // Splat gốc là Y xuống, Z tới; đổi về Y lên bằng phép quay 180 độ quanh trục X, giống RenderSplat của ComfyUI.
    g.pos[m * 3] = axis(0);
    g.pos[m * 3 + 1] = -axis(1);
    g.pos[m * 3 + 2] = -axis(2);
    for (let k = 0; k < 3; k++) {
      const v = ((raw[offC + i * 3 + k] / 255 - 0.5) * SPZ_COLOR_SCALE + 0.5) * 255;
      g.col[m * 4 + k] = Math.max(0, Math.min(255, v));
    }
    g.col[m * 4 + 3] = raw[offA + i];
    const s0 = Math.exp(2 * (raw[offS + i * 3] / 16 - 10)), s1 = Math.exp(2 * (raw[offS + i * 3 + 1] / 16 - 10)), s2 = Math.exp(2 * (raw[offS + i * 3 + 2] / 16 - 10));
    const x = raw[offQ + i * 3] / 127.5 - 1, y = raw[offQ + i * 3 + 1] / 127.5 - 1, z = raw[offQ + i * 3 + 2] / 127.5 - 1;
    const w = Math.sqrt(Math.max(0, 1 - x * x - y * y - z * z));
    const len = Math.hypot(x, y, z, w) || 1;
    const qx = x / len, qy = y / len, qz = z / len, qw = w / len;
    const r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - qw * qz), r02 = 2 * (qx * qz + qw * qy);
    const r10 = 2 * (qx * qy + qw * qz), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - qw * qx);
    const r20 = 2 * (qx * qz - qw * qy), r21 = 2 * (qy * qz + qw * qx), r22 = 1 - 2 * (qx * qx + qy * qy);
    g.cov[m * 6] = r00 * r00 * s0 + r01 * r01 * s1 + r02 * r02 * s2;
    g.cov[m * 6 + 1] = -(r00 * r10 * s0 + r01 * r11 * s1 + r02 * r12 * s2);
    g.cov[m * 6 + 2] = -(r00 * r20 * s0 + r01 * r21 * s1 + r02 * r22 * s2);
    g.cov[m * 6 + 3] = r10 * r10 * s0 + r11 * r11 * s1 + r12 * r12 * s2;
    g.cov[m * 6 + 4] = r10 * r20 * s0 + r11 * r21 * s1 + r12 * r22 * s2;
    g.cov[m * 6 + 5] = r20 * r20 * s0 + r21 * r21 * s1 + r22 * r22 * s2;
    m++;
  }
  g.n = m;
  return g;
}
// Khối giả lập cho bản xem ngoài ComfyUI: phồng ảnh chính thành các điểm. Trên ComfyUI dùng splat thật từ TripoSplat.
function buildDemoGaussians(img) {
  const iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
  const k = Math.min(1, 380 / Math.max(iw, ih));
  const w = Math.max(2, Math.round(iw * k)), h = Math.max(2, Math.round(ih * k));
  const c = newCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  const px = ctx.getImageData(0, 0, w, h).data;
  const step = 2.4 / Math.max(w, h);
  const g = emptyGaussians(w * h * 2);
  let m = 0;
  const put = (X, Y, Z, r, gr, b) => {
    g.pos.set([X, Y, Z], m * 3);
    g.cov.set([step * step, 0, 0, step * step, 0, step * step], m * 6);
    g.col.set([r, gr, b, 255], m * 4);
    m++;
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const u = 2 * x / (w - 1) - 1, v = 2 * y / (h - 1) - 1;
      const t = 0.55 * Math.sqrt(Math.max(0, (1 - u ** 4) * (1 - v ** 4)));
      const i = (y * w + x) * 4;
      const X = (x - (w - 1) / 2) * step, Y = ((h - 1) / 2 - y) * step;
      put(X, Y, t, px[i], px[i + 1], px[i + 2]);
      put(X, Y, -t, px[i] * 0.62, px[i + 1] * 0.62, px[i + 2] * 0.62);
    }
  }
  return g;
}
// Góc nhìn ban đầu. Có thông số máy ảnh của TripoSplat thì khối mở đúng góc ảnh gốc, cách tính theo gói AnyAngle Studio.
function splatCamera(g, W, H, source) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < g.n; i++) {
    if (g.col[i * 4 + 3] < 25) continue;
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], g.pos[i * 3 + k]);
      hi[k] = Math.max(hi[k], g.pos[i * 3 + k]);
    }
  }
  const extent = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  if (source && Array.isArray(source.camera_token) && Array.isArray(source.crop)) {
    const t = source.camera_token;
    const len = Math.hypot(t[0], t[1], t[2]) || 1;
    const radius = t[3] < 0.001 ? 1000 : 1 / t[3];
    return {
      target: [0, 0, 0], radius, tan: t[4] / radius, crop: source.crop, W, H, extent, matched: true,
      az0: Math.atan2(t[0] / len, -t[1] / len) * 180 / Math.PI, el0: Math.asin(Math.max(-1, Math.min(1, t[2] / len))) * 180 / Math.PI,
    };
  }
  const tan = Math.tan(17.5 * Math.PI / 180), side = Math.max(W, H);
  const half = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2 * 1.1;
  return {
    target: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2], radius: half / tan, tan,
    crop: [(W - side) / 2, (H - side) / 2, (W + side) / 2, (H + side) / 2], W, H, extent, matched: false, az0: 0, el0: 0,
  };
}

const RotateView = (() => {
  const canvas = $('rotCanvas');
  const gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: true, alpha: true, premultipliedAlpha: true });
  const cur = { az: 0, el: 0, z: 6.6 };
  let raf = 0, dragging = false, meshCount = 0;
  let meshProg, meshVao, meshU, splatProg, splatVao, splatU, orderBuf;
  let splat = null, cam = null, order = null, counts = null, keys = null, lastSort = '';
  const textures = [];
  const mul = (a, b) => {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) o[c * 4 + r] = a[r] * b[c * 4] + a[r + 4] * b[c * 4 + 1] + a[r + 8] * b[c * 4 + 2] + a[r + 12] * b[c * 4 + 3];
    return o;
  };
  const rotX = d => { const s = Math.sin(d * Math.PI / 180), c = Math.cos(d * Math.PI / 180); return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); };
  const rotY = d => { const s = Math.sin(d * Math.PI / 180), c = Math.cos(d * Math.PI / 180); return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); };
  const move = (x, y, z) => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
  // l, r, b, t tính ở độ sâu 1 đơn vị, nên khung lệch tâm vẫn dùng chung một công thức.
  const frustum = (l, r, b, t, near, far) => new Float32Array([
    2 / (r - l), 0, 0, 0,
    0, 2 / (t - b), 0, 0,
    (r + l) / (r - l), (t + b) / (t - b), (far + near) / (near - far), -1,
    0, 0, 2 * far * near / (near - far), 0,
  ]);
  const program = (vs, fs) => {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(p));
    return p;
  };
  function init() {
    if (!gl) return;
    meshProg = program(`#version 300 es
      layout(location=0) in vec3 p; layout(location=1) in vec3 n; layout(location=2) in vec3 c;
      uniform mat4 M, VP; out vec3 vn; out vec3 vc;
      void main(){ vn = mat3(M) * n; vc = c; gl_Position = VP * M * vec4(p, 1.0); }`, `#version 300 es
      precision highp float; in vec3 vn; in vec3 vc; out vec4 o;
      void main(){
        if (dot(vn, vn) < 0.0001) { o = vec4(vc, 1.0); return; }
        vec3 n = normalize(vn), L = normalize(vec3(0.42, 0.72, 0.62)), H = normalize(L + vec3(0.0, 0.0, 1.0));
        float wrap = clamp((dot(n, L) + 0.55) / 1.55, 0.0, 1.0);
        float sky = 0.5 + 0.5 * n.y;
        float rim = pow(1.0 - clamp(n.z, 0.0, 1.0), 3.0);
        float gloss = pow(max(dot(n, H), 0.0), 48.0);
        vec3 ambient = mix(vec3(0.6, 0.5, 0.47), vec3(0.62, 0.6, 0.58), sky);
        vec3 c = vc * (ambient + 0.46 * wrap) + vec3(1.0, 0.96, 0.9) * (0.08 * gloss + 0.07 * rim);
        o = vec4(c, 1.0);
      }`);
    meshU = { M: gl.getUniformLocation(meshProg, 'M'), VP: gl.getUniformLocation(meshProg, 'VP') };
    splatProg = program(`#version 300 es
      precision highp float; precision highp int;
      layout(location=0) in uint aIndex;
      uniform highp sampler2D tPos, tCovA, tCovB; uniform lowp sampler2D tCol;
      uniform mat4 P, MV; uniform vec2 view; uniform int texW;
      out vec4 vCol; out vec2 vPos;
      void main(){
        ivec2 uv = ivec2(int(aIndex) % texW, int(aIndex) / texW);
        vec3 c = texelFetch(tPos, uv, 0).xyz;
        vec4 a = texelFetch(tCovA, uv, 0), b = texelFetch(tCovB, uv, 0);
        vec4 cam = MV * vec4(c, 1.0);
        vCol = texelFetch(tCol, uv, 0); vPos = vec2(0.0);
        if (cam.z > -0.001) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
        mat3 R = mat3(MV);
        mat3 S = mat3(a.x, a.y, a.z, a.y, a.w, b.x, a.z, b.x, b.y);
        mat3 C = R * S * transpose(R);
        float w = -cam.z;
        vec2 f = vec2(P[0][0], P[1][1]) * view * 0.5;
        vec3 j0 = vec3(f.x / w, 0.0, f.x * cam.x / (w * w));
        vec3 j1 = vec3(0.0, f.y / w, f.y * cam.y / (w * w));
        float aa = dot(j0, C * j0), bb = dot(j0, C * j1), dd = dot(j1, C * j1);
        float mid = 0.5 * (aa + dd), rad = length(vec2(0.5 * (aa - dd), bb));
        float l1 = mid + rad, l2 = max(mid - rad, 0.05);
        vec2 v = abs(bb) > 1e-6 ? normalize(vec2(bb, l1 - aa)) : (aa >= dd ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
        vec2 major = min(sqrt(2.0 * max(l1, 0.05)), 512.0) * v;
        vec2 minor = min(sqrt(2.0 * l2), 512.0) * vec2(v.y, -v.x);
        vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, float((gl_VertexID >> 1) & 1) * 2.0 - 1.0) * 2.0;
        vPos = corner;
        vec4 clip = P * cam;
        gl_Position = vec4(clip.xy / clip.w + (corner.x * major + corner.y * minor) * 2.0 / view, 0.0, 1.0);
      }`, `#version 300 es
      precision highp float; in vec4 vCol; in vec2 vPos; out vec4 o;
      void main(){ float A = dot(vPos, vPos); if (A > 4.0) discard; float B = exp(-A) * vCol.a; o = vec4(vCol.rgb * B, B); }`);
    splatU = {};
    for (const name of ['tPos', 'tCovA', 'tCovB', 'tCol', 'P', 'MV', 'view', 'texW']) splatU[name] = gl.getUniformLocation(splatProg, name);
    const mesh = buildCatMesh();
    meshCount = mesh.length / 9;
    meshVao = gl.createVertexArray();
    gl.bindVertexArray(meshVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, mesh, gl.STATIC_DRAW);
    [[0, 0], [1, 12], [2, 24]].forEach(([loc, off]) => { gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 36, off); });
    splatVao = gl.createVertexArray();
    gl.bindVertexArray(splatVao);
    orderBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, orderBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribIPointer(0, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(0, 1);
    gl.bindVertexArray(null);
  }
  function target() {
    return { az: Rotate.az, el: Rotate.el, z: Rotate.mode === 'splat' && cam ? cam.radius * Rotate.splatZoom : ROTATE_DIST[Rotate.dist].z };
  }
  function modelView(view) {
    const az = view.az + cam.az0;
    const el = Math.max(-89, Math.min(89, view.el + cam.el0));
    const t = cam.target;
    return mul(move(0, 0, -view.z), mul(rotX(el), mul(rotY(-az), move(-t[0], -t[1], -t[2]))));
  }
  // Toạ độ khung ảnh gốc ở độ sâu 1 đơn vị: ô vuông TripoSplat dùng để dựng chính là toàn bộ cảm biến.
  function frameExtents() {
    const [x0, y0, x1, y1] = cam.crop;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, hw = (x1 - x0) / 2, hh = (y1 - y0) / 2;
    return { l: -cx / hw * cam.tan, r: (cam.W - cx) / hw * cam.tan, t: cy / hh * cam.tan, b: (cy - cam.H) / hh * cam.tan };
  }
  function drawSplat(proj, mv, w, h) {
    const key = [mv[2], mv[6], mv[10], mv[14]].map(v => v.toFixed(4)).join();
    if (key !== lastSort) {
      lastSort = key;
      let min = Infinity, max = -Infinity;
      const pos = splat.pos, n = splat.n, a = mv[2], b = mv[6], c = mv[10];
      for (let i = 0; i < n; i++) {
        const d = a * pos[i * 3] + b * pos[i * 3 + 1] + c * pos[i * 3 + 2];
        keys[i] = d;
        if (d < min) min = d;
        if (d > max) max = d;
      }
      // Sắp từ gần tới xa: lớp gần vẽ trước, lớp xa cộng vào phần còn trong suốt.
      const scale = 65535 / Math.max(1e-6, max - min);
      counts.fill(0);
      for (let i = 0; i < n; i++) counts[((max - keys[i]) * scale) | 0]++;
      let sum = 0;
      for (let i = 0; i < 65536; i++) { const v = counts[i]; counts[i] = sum; sum += v; }
      for (let i = 0; i < n; i++) order[counts[((max - keys[i]) * scale) | 0]++] = i;
      gl.bindBuffer(gl.ARRAY_BUFFER, orderBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, order);
    }
    gl.useProgram(splatProg);
    textures.forEach((tex, i) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, tex); });
    ['tPos', 'tCovA', 'tCovB', 'tCol'].forEach((name, i) => gl.uniform1i(splatU[name], i));
    gl.uniformMatrix4fv(splatU.P, false, proj);
    gl.uniformMatrix4fv(splatU.MV, false, mv);
    gl.uniform2f(splatU.view, w, h);
    gl.uniform1i(splatU.texW, splat.texW);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.bindVertexArray(splatVao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, splat.n);
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }
  function draw() {
    if (!gl) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    if (Rotate.mode === 'dummy') {
      gl.clearColor(0.09, 0.094, 0.106, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const f = Math.tan(17.5 * Math.PI / 180), aspect = w / h;
      const VP = mul(frustum(-f * aspect, f * aspect, -f, f, 0.1, 50), move(0, -0.22, -cur.z));
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(meshProg);
      gl.uniformMatrix4fv(meshU.M, false, mul(rotX(cur.el), rotY(-cur.az)));
      gl.uniformMatrix4fv(meshU.VP, false, VP);
      gl.bindVertexArray(meshVao);
      gl.drawArrays(gl.TRIANGLES, 0, meshCount);
      gl.bindVertexArray(null);
      return;
    }
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!splat || !cam) return;
    // Khung nét đứt chiếm 86% khung xem, nên nhìn thấy gì trong khung thì ảnh render đúng như vậy.
    const e = frameExtents();
    const cx = (e.l + e.r) / 2, cy = (e.b + e.t) / 2, aspect = w / h;
    const vy = Math.max((e.t - e.b) / 2 / 0.86, (e.r - e.l) / 2 / (0.86 * aspect)), vx = vy * aspect;
    const near = Math.max(0.01, cur.z - cam.extent * 2), far = cur.z + cam.extent * 2;
    drawSplat(frustum(cx - vx, cx + vx, cy - vy, cy + vy, near, far), modelView(cur), w, h);
  }
  function frame() {
    raf = 0;
    const t = target();
    const ease = dragging ? 1 : 0.22;
    let moving = false;
    for (const key of ['az', 'el', 'z']) {
      const gap = t[key] - cur[key];
      if (Math.abs(gap) > 0.01) { cur[key] += gap * ease; moving = true; } else cur[key] = t[key];
    }
    draw();
    if (moving) raf = requestAnimationFrame(frame);
  }
  const request = () => { if (!raf) raf = requestAnimationFrame(frame); };
  // Vẽ khối ra ảnh đúng khung ảnh gốc, nền xám như ảnh render trong hướng dẫn của LoRA AnyAngle.
  function capture(w, h) {
    const view = target();
    const fb = gl.createFramebuffer(), tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const e = frameExtents();
    drawSplat(frustum(e.l, e.r, e.b, e.t, Math.max(0.01, view.z - cam.extent * 2), view.z + cam.extent * 2), modelView(view), w, h);
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fb);
    gl.deleteTexture(tex);
    const out = newCanvas(w, h);
    const ctx = out.getContext('2d');
    const image = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      const src = (h - 1 - y) * w * 4, dst = y * w * 4;
      for (let x = 0; x < w * 4; x += 4) {
        const rest = 64 * (1 - px[src + x + 3] / 255);
        image.data[dst + x] = px[src + x] + rest;
        image.data[dst + x + 1] = px[src + x + 1] + rest;
        image.data[dst + x + 2] = px[src + x + 2] + rest;
        image.data[dst + x + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
    request();
    return out;
  }
  return {
    available: !!gl,
    init,
    request,
    capture,
    setDragging(on) { dragging = on; },
    setSplat(g, camera) {
      if (!gl) return;
      for (const tex of textures) gl.deleteTexture(tex);
      textures.length = 0;
      const texW = 2048, texH = Math.max(1, Math.ceil(g.n / texW));
      const a = new Float32Array(texW * texH * 4), b = new Float32Array(texW * texH * 4), c = new Float32Array(texW * texH * 4), d = new Uint8Array(texW * texH * 4);
      for (let i = 0; i < g.n; i++) {
        a.set(g.pos.subarray(i * 3, i * 3 + 3), i * 4);
        b.set(g.cov.subarray(i * 6, i * 6 + 4), i * 4);
        c.set(g.cov.subarray(i * 6 + 4, i * 6 + 6), i * 4);
        d.set(g.col.subarray(i * 4, i * 4 + 4), i * 4);
      }
      [[gl.RGBA32F, gl.FLOAT, a], [gl.RGBA32F, gl.FLOAT, b], [gl.RGBA32F, gl.FLOAT, c], [gl.RGBA8, gl.UNSIGNED_BYTE, d]].forEach(([format, type, data]) => {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texStorage2D(gl.TEXTURE_2D, 1, format, texW, texH);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, texW, texH, gl.RGBA, type, data);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        textures.push(tex);
      });
      splat = { n: g.n, pos: g.pos, texW };
      cam = camera;
      order = new Uint32Array(g.n);
      keys = new Float32Array(g.n);
      counts = new Uint32Array(65536);
      lastSort = '';
      gl.bindBuffer(gl.ARRAY_BUFFER, orderBuf);
      gl.bufferData(gl.ARRAY_BUFFER, order, gl.DYNAMIC_DRAW);
      cur.z = camera.radius * Rotate.splatZoom;
      request();
    },
    clearSplat() { splat = null; cam = null; request(); },
    paintThumb(cv) {
      if (!splat || !cam) return;
      cv.width = 144;
      cv.height = Math.max(1, Math.round(144 * cam.H / cam.W));
      cv.getContext('2d').drawImage(capture(cv.width, cv.height), 0, 0);
    },
  };
})();

function syncRotateChips() {
  $('rotDist').querySelectorAll('.viewchip').forEach((chip, i) => chip.setAttribute('aria-pressed', String(i === Rotate.dist)));
  $('rotAzRange').value = Math.round(wrapDegrees(Rotate.az));
  $('rotElRange').value = Math.round(Rotate.el);
}
function applyRotate() {
  syncRotateChips();
  RotateView.request();
}
function setRotate(change) {
  Object.assign(Rotate, change);
  applyRotate();
  renderPanel();
}
function layoutRotateFrame() {
  const main = Rotate.slots[0].ref, box = $('rotView').getBoundingClientRect();
  const show = Rotate.mode === 'splat' && Rotate.splat === 'ready' && !!main;
  $('rotFrame').hidden = !show;
  if (!show) return;
  const aspect = (main.naturalWidth || main.width) / (main.naturalHeight || main.height);
  const fit = Math.min(box.width * 0.86 / aspect, box.height * 0.86);
  $('rotFrame').style.width = `${Math.round(fit * aspect)}px`;
  $('rotFrame').style.height = `${Math.round(fit)}px`;
}
function fitMainForRotate(img) {
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  const s = Math.sqrt(1e6 / (w * h));
  const c = newCanvas(Math.max(32, Math.round(w * s)), Math.max(32, Math.round(h * s)));
  const g = c.getContext('2d');
  g.imageSmoothingQuality = 'high';
  g.drawImage(img, 0, 0, c.width, c.height);
  return c;
}
function splatReady(g, main, source) {
  Rotate.az = 0;
  Rotate.el = 0;
  Rotate.splatZoom = 1;
  RotateView.setSplat(g, splatCamera(g, main.width, main.height, source));
  Rotate.splat = 'ready';
  syncRotateMode();
  renderPanel();
}
function buildSplatPreview(main) {
  clearInterval(Rotate.buildTimer);
  $('rotMsgFill').style.width = '0';
  let progress = 0;
  Rotate.buildTimer = setInterval(() => {
    progress = Math.min(100, progress + 6);
    $('rotMsgFill').style.width = `${progress}%`;
    if (progress < 100) return;
    clearInterval(Rotate.buildTimer);
    if (Rotate.mainCanvas === main) splatReady(buildDemoGaussians(main), main, null);
  }, 90);
}
// Đường đi theo template TripoSplat của ComfyUI. Node của gói AnyAngle Studio chỉ thêm khi có, để lấy thông số máy ảnh.
function buildSplatGraph(image, withStudio) {
  const seed = 46;
  const graph = {
    1: { class_type: 'LoadImage', inputs: { image } },
    2: { class_type: 'LoadBackgroundRemovalModel', inputs: { bg_removal_name: 'birefnet.safetensors' } },
    3: { class_type: 'RemoveBackground', inputs: { bg_removal_model: ['2', 0], image: ['1', 0] } },
    15: { class_type: 'UNETLoader', inputs: { unet_name: 'triposplat_fp16.safetensors', weight_dtype: 'default' } },
    4: { class_type: 'TripoSplatPreprocessImage', inputs: { image: ['1', 0], mask: ['3', 0], erode_radius: 1, size: 1024 } },
    20: { class_type: 'CLIPVisionLoader', inputs: { clip_name: 'dino_v3_vit_h.safetensors' } },
    21: { class_type: 'VAELoader', inputs: { vae_name: 'flux2-vae.safetensors' } },
    7: { class_type: 'TripoSplatConditioning', inputs: { clip_vision: ['20', 0], vae: ['21', 0], image: ['4', 0] } },
    8: { class_type: 'KSampler', inputs: { model: ['15', 0], positive: ['7', 0], negative: ['7', 1], latent_image: ['7', 2], seed, steps: 20, cfg: 3, sampler_name: 'dpmpp_2m', scheduler: 'simple', denoise: 1 } },
    22: { class_type: 'VAELoader', inputs: { vae_name: 'triposplat_vae_decoder_fp16.safetensors' } },
    9: { class_type: 'VAEDecodeTripoSplat', inputs: { samples: ['8', 0], vae: ['22', 0], num_gaussians: 262144, seed } },
    23: { class_type: 'SplatToFile3D', inputs: { splat: ['9', 0], format: 'spz' } },
    6: { class_type: 'SaveGLB', inputs: { mesh: ['23', 0], filename_prefix: '3d/Concept_splat' } },
  };
  if (withStudio) {
    graph[24] = { class_type: 'AnyAngleReconstructionOutputT8', inputs: { splat: ['9', 0], samples: ['8', 0], reference: ['1', 0], prepared: ['4', 0], mask: ['3', 0], keep_background: false } };
  }
  return graph;
}
async function buildSplatReal(main) {
  const stamp = Date.now().toString(36);
  beginRun();
  try {
    startPass('Dựng splat');
    const name = await uploadCanvas(main, `${stamp}_splat.png`);
    const check = await fetch('/api/object_info/AnyAngleReconstructionOutputT8');
    const withStudio = check.ok && !!(await check.json()).AnyAngleReconstructionOutputT8;
    run.promptId = await queueGraph(buildSplatGraph(name, withStudio));
    advanceJob('queue');
    renderJob();
    const found = await waitForHistory(run.promptId, outputs => {
      const all = Object.values(outputs);
      const file = all.flatMap(o => o['3d'] || [])[0];
      if (!file) return null;
      const record = all.flatMap(o => o.anyangle_reconstruction || [])[0];
      return { file, source: record && record.source };
    }, 'ComfyUI chạy xong nhưng không có file splat.');
    const res = await fetch(viewUrl(found.file));
    if (!res.ok) throw new Error(`Không tải được file splat (lỗi ${res.status}).`);
    const g = await parseSpz(await res.arrayBuffer());
    if (Rotate.mainCanvas === main) splatReady(g, main, found.source);
    hideJob();
    setStatus(found.source ? 'Đã dựng xong splat.' : 'Đã dựng xong splat. Chưa có gói AnyAngle Studio nên khối có thể không mở đúng mặt ảnh, hãy xoay về đúng hướng.');
  } catch (e) {
    Rotate.splat = 'failed';
    failRun(e);
  } finally {
    endRun();
  }
}
function startSplatBuild() {
  const main = fitMainForRotate(Rotate.slots[0].ref);
  Rotate.builtFrom = Rotate.slots[0].ref;
  Rotate.mainCanvas = main;
  Rotate.splat = 'building';
  RotateView.clearSplat();
  if (ON_COMFY) buildSplatReal(main);
  else buildSplatPreview(main);
}
function syncRotateMode() {
  const splat = Rotate.mode === 'splat';
  $('rotMode').querySelectorAll('.viewchip').forEach(chip => chip.setAttribute('aria-pressed', String(chip.dataset.mode === Rotate.mode)));
  $('rotDummyCtl').hidden = splat;
  $('rotSplatCtl').hidden = !splat;
  $('rotHint').textContent = splat
    ? 'Bỏ ảnh chính vào là tự dựng khối 3D. Hợp nhất với một vật rõ ràng, nền sạch, chụp góc 3/4.'
    : 'Con mèo chỉ để biết hướng. Ảnh chính nên nhìn chính diện, ảnh các góc khác có thể bỏ trống.';
  const main = Rotate.slots[0].ref;
  if (splat && main && Rotate.builtFrom !== main && !run.active) startSplatBuild();
  if (!main && Rotate.builtFrom) {
    clearInterval(Rotate.buildTimer);
    Rotate.splat = null;
    Rotate.builtFrom = null;
    Rotate.mainCanvas = null;
    RotateView.clearSplat();
  }
  let message = '';
  if (!RotateView.available) message = 'Máy này không chạy được WebGL nên không hiện được khối 3D';
  else if (splat && !main) message = 'Bỏ ảnh chính vào để dựng splat';
  else if (splat && Rotate.splat === 'building') message = 'Đang dựng splat từ ảnh chính...';
  else if (splat && Rotate.splat === 'failed') message = 'Không dựng được splat. Bấm Dựng lại để thử.';
  $('rotMsg').hidden = !message;
  $('rotMsgText').textContent = message;
  $('rotMsgBar').hidden = !(splat && Rotate.splat === 'building' && !ON_COMFY);
  $('rotRetry').hidden = Rotate.splat !== 'failed';
  $('rotNote').hidden = !splat || Rotate.splat !== 'ready' || ON_COMFY;
  layoutRotateFrame();
  applyRotate();
}
function syncRotate() {
  syncSlotBoxes('rotateSlots', Rotate.slots);
  syncRotateMode();
}
function takeRotateFiles(fileList, start) {
  takeFilesInto(Rotate.slots, fileList, start, syncRotate);
}
function nudgeRotate(key) {
  const splat = Rotate.mode === 'splat';
  if (key === 'ArrowLeft') setRotate({ az: splat ? Rotate.az - 5 : Math.round(Rotate.az / 45) * 45 - 45 });
  else if (key === 'ArrowRight') setRotate({ az: splat ? Rotate.az + 5 : Math.round(Rotate.az / 45) * 45 + 45 });
  else if (key === 'ArrowUp') setRotate({ el: splat ? Math.min(80, Rotate.el + 5) : ROTATE_EL[Math.min(ROTATE_EL.length - 1, rotateElIndex() + 1)].deg });
  else if (key === 'ArrowDown') setRotate({ el: splat ? Math.max(-80, Rotate.el - 5) : ROTATE_EL[Math.max(0, rotateElIndex() - 1)].deg });
  else return false;
  return true;
}
function buildRotatePanel() {
  buildSlotBoxes('rotateSlots', Rotate.slots, takeRotateFiles, syncRotate);
  RotateView.init();
  const chips = (id, items, pick) => {
    $(id).innerHTML = items.map(item => `<button class="viewchip" type="button" aria-pressed="false">${item.vi}</button>`).join('');
    $(id).querySelectorAll('.viewchip').forEach((chip, i) => chip.addEventListener('click', () => pick(i)));
  };
  chips('rotDist', ROTATE_DIST, i => setRotate({ dist: i }));
  $('rotMode').addEventListener('click', e => {
    const chip = e.target.closest('.viewchip');
    if (!chip || chip.dataset.mode === Rotate.mode) return;
    Rotate.mode = chip.dataset.mode;
    S.overrides.rotate = null;
    if (Rotate.mode === 'dummy') Object.assign(Rotate, { az: Math.round(Rotate.az / 45) * 45, el: ROTATE_EL[rotateElIndex()].deg });
    syncRotateMode();
    renderPanel();
  });
  $('rotAzRange').addEventListener('input', e => { Rotate.az = Number(e.target.value); applyRotate(); });
  $('rotElRange').addEventListener('input', e => { Rotate.el = Number(e.target.value); applyRotate(); });
  $('rotAzRange').addEventListener('change', renderPanel);
  $('rotElRange').addEventListener('change', renderPanel);
  $('rotReset').addEventListener('click', () => setRotate({ az: Rotate.az + wrapDegrees(-Rotate.az), el: 0, splatZoom: 1 }));
  $('rotRetry').addEventListener('click', () => {
    Rotate.builtFrom = null;
    Rotate.splat = null;
    syncRotateMode();
    renderPanel();
  });
  const view = $('rotView');
  let start = null;
  view.addEventListener('pointerdown', e => {
    view.setPointerCapture(e.pointerId);
    start = { x: e.clientX, y: e.clientY, az: Rotate.az, el: Rotate.el };
    view.classList.add('dragging');
    RotateView.setDragging(true);
  });
  view.addEventListener('pointermove', e => {
    if (!start) return;
    const limit = Rotate.mode === 'splat' ? [-80, 80] : [-30, 60];
    Rotate.az = start.az - (e.clientX - start.x) * 0.4;
    Rotate.el = Math.max(limit[0], Math.min(limit[1], start.el + (e.clientY - start.y) * 0.3));
    applyRotate();
  });
  // Dummy: thả tay thì về nấc gần nhất, vì LoRA chỉ hiểu 8 hướng và 4 độ cao. Splat: giữ nguyên góc đang chọn.
  const finish = () => {
    if (!start) return;
    start = null;
    view.classList.remove('dragging');
    RotateView.setDragging(false);
    if (Rotate.mode === 'dummy') setRotate({ az: Math.round(Rotate.az / 45) * 45, el: ROTATE_EL[rotateElIndex()].deg });
    else setRotate({});
  };
  view.addEventListener('pointerup', finish);
  view.addEventListener('pointercancel', finish);
  view.addEventListener('wheel', e => {
    if (Rotate.mode !== 'splat') return;
    e.preventDefault();
    Rotate.splatZoom = Math.max(0.35, Math.min(3, Rotate.splatZoom * (e.deltaY > 0 ? 1.08 : 0.93)));
    RotateView.request();
  }, { passive: false });
  syncRotate();
}
function buildSheetPanel() {
  buildSheetSlots();
  $('sheetViews').innerHTML = SHEET_VIEWS.map(v => `<button class="viewchip" type="button" data-view="${v.id}" aria-pressed="${Sheet.views.has(v.id)}">${v.label}</button>`).join('');
  $('sheetViews').addEventListener('click', e => {
    const chip = e.target.closest('.viewchip');
    if (!chip) return;
    const id = chip.dataset.view;
    if (Sheet.views.has(id)) Sheet.views.delete(id); else Sheet.views.add(id);
    chip.setAttribute('aria-pressed', String(Sheet.views.has(id)));
    renderPanel();
  });
  $('sheetNote').addEventListener('input', e => { Sheet.note = e.target.value; renderPanel(); });
}
// Tab Đổi người theo ba workflow BFS cho Qwen 2.1: đổi cả đầu, thay cả người, thay đúng người được tô.
// Prompt tiếng Anh là prompt gốc trong workflow của tác giả, giữ nguyên.
const SWAP_PROMPTS = {
  head: {
    en: 'head_swap: start with <image1> as the base image, keeping its lighting, environment, and background. remove the head from <image1> completely and replace it with the head from <image2>, strictly preserving the hair, eye color, nose structure from <image2>. copy the direction of the eye, head rotation, micro expressions from <image1>, high quality, sharp details, 4k',
    vi: 'Lấy <image1> làm nền, giữ ánh sáng, môi trường và nền. Bỏ hẳn đầu trong <image1>, thay bằng đầu trong <image2>, giữ đúng tóc, màu mắt, dáng mũi của <image2>. Chép hướng nhìn, góc đầu và biểu cảm của <image1>.',
  },
  body: {
    en: 'body_swap: start with <image1> as the base image, keeping its lighting, environment, and background. replace the body from <image1> with the body from <image2>, strictly preserving the clothing, body shape and proportions from <image2>. strictly replicate the exact pose, arm positions, leg positions, hand gestures, head rotation, eye direction and micro expressions from <image1>',
    vi: 'Lấy <image1> làm nền, giữ ánh sáng, môi trường và nền. Thay người trong <image1> bằng người trong <image2>, giữ đúng quần áo, dáng người và tỉ lệ của <image2>. Chép đúng tư thế, tay, chân, cử chỉ, góc đầu, hướng nhìn và biểu cảm của <image1>.',
  },
  selective: {
    en: "body_swap: replace only the person highlighted in magenta in the second image with the character from the third image - face, hair, body and outfit. Keep the first image's scene, lighting, colours, camera and every other person unchanged, and keep the exact pose, head angle and facial expression of the replaced person. The replaced person's original hair must not remain; use only the hairstyle of the character from the third image. No magenta in the result.",
    vi: 'Chỉ thay người được phủ hồng trong ảnh thứ hai bằng nhân vật trong ảnh thứ ba, gồm mặt, tóc, thân và trang phục. Giữ nguyên cảnh, ánh sáng, màu, góc máy và mọi người khác của ảnh đầu, giữ đúng tư thế, góc đầu và biểu cảm. Không giữ tóc cũ. Kết quả không còn màu hồng.',
  },
};
const SWAP_REF_LABEL = { head: 'Ảnh mặt', body: 'Ảnh nhân vật' };
const SWAP_HINTS = {
  head: 'Thay cả đầu (gồm tóc) của người trong cảnh bằng người ở ảnh mặt. Giữ dáng, ánh sáng, nền. Hợp nhất khi cảnh có 1 người và mặt không quá nhỏ. Dùng LoRA BFS Head v1.1.',
  body: 'Thay cả người (mặt, tóc, dáng, quần áo) bằng nhân vật ở ảnh mẫu. Giữ dáng, khung hình, ánh sáng và nền của cảnh. Ảnh nhân vật nên toàn thân, chính diện, nền trơn. Dùng LoRA BFS Body Swap.',
};
// Màu và độ đậm như đầu ra marked của node BFS Segment Picker.
const SWAP_MARK = { color: '#FF00FF', alpha: 0.3 };
const Swap = {
  mode: 'head',
  slots: [{ label: 'Ảnh cảnh' }, { label: SWAP_REF_LABEL.head }].map(slot => ({ ...slot, ref: null, refUrl: '' })),
  mark: null,
  markFor: null,
  hasMark: false,
  tool: null,
  size: 60,
};
const swapFit = { x: 0, y: 0, s: 1 };
const swapVariant = () => (Swap.mode === 'head' ? 'head' : Swap.hasMark ? 'selective' : 'body');
// Đồ thị theo đúng ba workflow BFS trong workflows/bfs_swap, chỉ bỏ các node xem thử và so sánh.
// Đổi mặt và thay nhân vật: ảnh vừa khung vuông 2 MP và 1,5 MP bằng ImageResizeKJv2, sampler deis_2m (của RES4LYF), 8 bước, CFG 1.
// Thay người đã tô: cảnh và nhân vật 1 MP, ảnh phủ hồng là ảnh thứ hai thay cho node Segment Picker, euler 8 bước, CFG 1.
const BFS_RUN = {
  head: { lora: 'bfs_head_v1.1_qwen_2.1.safetensors', turbo: 'p_qwen_image_2.1_8step_v0.1.safetensors', cache: 'auto', mp: 2, latent: 'empty' },
  body: { lora: 'Downloads/bfs_body_swap_v1.0_qwen_2.1.safetensors', turbo: 'Downloads/p_qwen_image_2.1_8step_v0.1.safetensors', cache: 'gpu', mp: 1.5, latent: 'encoder' },
  selective: { lora: 'bfs_body_swap_selective_magenta_v1.0_qwen_2.1.safetensors', turbo: 'p_qwen_image_2.1_8step_v0.1.safetensors', cache: 'auto', mp: 1 },
};
const SWAP_UPLOAD_MP = 4;
function buildSwapGraph(variant, { scene, person, marked, text, seed }) {
  const run = BFS_RUN[variant];
  const graph = {
    1: { class_type: 'UNETLoader', inputs: { unet_name: 'qwen_image_2.1_int8_convrot.safetensors', weight_dtype: 'default' } },
    2: { class_type: 'CLIPLoader', inputs: { clip_name: 'qwen3vl_8b_int8_convrot.safetensors', type: 'qwen_image', device: 'default' } },
    3: { class_type: 'VAELoader', inputs: { vae_name: 'qwen_image_2.1_vae_bf16.safetensors' } },
    15: { class_type: 'LoraLoaderModelOnly', inputs: { model: ['1', 0], lora_name: run.turbo, strength_model: 1 } },
    16: { class_type: 'LoraLoaderModelOnly', inputs: { model: ['15', 0], lora_name: run.lora, strength_model: 1 } },
    17: { class_type: 'QwenImage21Cache', inputs: { model: ['16', 0], device: run.cache, dtype: 'default' } },
    4: { class_type: 'LoadImage', inputs: { image: scene } },
    20: { class_type: 'LoadImage', inputs: { image: person } },
    7: { class_type: 'TextEncodeQwenImage21', inputs: { clip: ['2', 0], vae: ['3', 0], prompt: text, negative_prompt: '', resolution: 0 } },
    9: { class_type: 'VAEDecode', inputs: { samples: ['8', 0], vae: ['3', 0] } },
    6: { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'Concept_swap' } },
  };
  const encoder = graph[7].inputs;
  if (variant === 'selective') {
    const resize = id => ({ class_type: 'ResizeImageMaskNode', inputs: { input: [id, 0], resize_type: 'scale total pixels', 'resize_type.megapixels': run.mp, scale_method: 'lanczos' } });
    graph[21] = { class_type: 'LoadImage', inputs: { image: marked } };
    graph[10] = resize('4');
    graph[11] = resize('21');
    graph[12] = resize('20');
    encoder['images.image_1'] = ['10', 0];
    encoder['images.image_2'] = ['11', 0];
    encoder['images.image_3'] = ['12', 0];
    graph[30] = { class_type: 'RandomNoise', inputs: { noise_seed: seed } };
    graph[31] = { class_type: 'CFGGuider', inputs: { model: ['17', 0], positive: ['7', 0], negative: ['7', 1], cfg: 1 } };
    graph[32] = { class_type: 'KSamplerSelect', inputs: { sampler_name: 'euler' } };
    graph[33] = { class_type: 'BasicScheduler', inputs: { model: ['17', 0], scheduler: 'simple', steps: 8, denoise: 1 } };
    graph[8] = { class_type: 'SamplerCustomAdvanced', inputs: { noise: ['30', 0], guider: ['31', 0], sampler: ['32', 0], sigmas: ['33', 0], latent_image: ['7', 2] } };
    graph[18] = { class_type: 'SplitImageWithAlpha', inputs: { image: ['9', 0] } };
    graph[6].inputs.images = ['18', 0];
    return graph;
  }
  const fit = id => ({ class_type: 'ImageResizeKJv2', inputs: { image: [id, 0], width: ['12', 0], height: ['12', 1], upscale_method: 'lanczos', keep_proportion: 'resize', pad_color: '0, 0, 0', crop_position: 'center', divisible_by: 32, device: 'cpu' } });
  graph[12] = { class_type: 'ResolutionSelector', inputs: { aspect_ratio: '1:1 (Square)', megapixels: run.mp, multiple: 32 } };
  graph[10] = fit('4');
  graph[11] = fit('20');
  encoder['images.image_1'] = ['10', 0];
  encoder['images.image_2'] = ['11', 0];
  let latent = ['7', 2];
  if (run.latent === 'empty') {
    graph[13] = { class_type: 'EmptyLatentImage', inputs: { width: ['10', 1], height: ['10', 2], batch_size: 1 } };
    latent = ['13', 0];
  }
  graph[8] = { class_type: 'KSampler', inputs: { model: ['17', 0], positive: ['7', 0], negative: ['7', 1], latent_image: latent, seed, steps: 8, cfg: 1, sampler_name: 'deis_2m', scheduler: 'simple', denoise: 1 } };
  return graph;
}
// deis_2m do pack RES4LYF thêm vào KSampler, ImageResizeKJv2 thuộc KJNodes. Mục 6 của stoneV_CloudSyncHub cài cả hai.
async function requireSwapPacks() {
  const read = name => fetch(`/api/object_info/${name}`).then(r => (r.ok ? r.json() : {}), () => ({}));
  const [sampler, resize] = await Promise.all([read('KSampler'), read('ImageResizeKJv2')]);
  const names = sampler.KSampler ? sampler.KSampler.input.required.sampler_name[0] : [];
  const missing = [!names.includes('deis_2m') && 'RES4LYF', !resize.ImageResizeKJv2 && 'KJNodes'].filter(Boolean);
  if (missing.length) {
    throw new Error(`ComfyUI chưa có pack ${missing.join(' và ')} mà workflow BFS cần. Mở ComfyUI, chạy mục 6 của node stoneV_CloudSyncHub, rồi Gen lại.`);
  }
}
async function generateSwap() {
  const pass = S.passes.find(p => p.kind.startsWith('swap_'));
  const variant = pass.kind.slice(5);
  const text = promptFor(pass);
  const scene = fitReference(Swap.slots[0].ref, SWAP_UPLOAD_MP);
  const person = fitReference(Swap.slots[1].ref, SWAP_UPLOAD_MP);
  const marked = variant === 'selective' ? fitReference(swapMarked(), SWAP_UPLOAD_MP) : null;
  const stamp = Date.now().toString(36);
  beginRun();
  try {
    startPass(pass.name);
    if (variant !== 'selective') await requireSwapPacks();
    const names = await uploadAll(marked ? [scene, person, marked] : [scene, person], `${stamp}_swap`);
    run.promptId = await queueGraph(buildSwapGraph(variant, { scene: names[0], person: names[1], marked: names[2], text, seed: randomSeed() }));
    advanceJob('queue');
    renderJob();
    const { img, url } = await loadOutput(await waitForImage(run.promptId));
    addResult(img, url, pass.name);
    if (S.tab === 'swap') openViewer(img, url);
    hideJob();
    setStatus(`Xong trong ${clock(Date.now() - job.startedAt)}.`);
  } catch (e) {
    failRun(e);
  } finally {
    endRun();
  }
}
function buildSwapPasses() {
  const [shot, ref] = Swap.slots;
  if (!shot.ref) return { passes: [], problems: [] };
  const variant = swapVariant();
  const problems = ref.ref ? [] : [{ brush: null, text: `Thêm ${SWAP_REF_LABEL[Swap.mode].toLowerCase()}` }];
  const name = { head: 'Đổi mặt', body: 'Thay nhân vật', selective: 'Thay người đã tô' }[variant];
  return { passes: [{ kind: `swap_${variant}`, name, lines: [SWAP_PROMPTS[variant]], slots: [] }], problems };
}
// Lớp tô cùng cỡ ảnh cảnh. Đổi ảnh cảnh thì bỏ vùng tô cũ.
function swapMark() {
  const shot = Swap.slots[0].ref;
  if (!shot) {
    Object.assign(Swap, { mark: null, markFor: null, hasMark: false });
    return null;
  }
  if (Swap.markFor !== shot) {
    Object.assign(Swap, { mark: newCanvas(shot.naturalWidth || shot.width, shot.naturalHeight || shot.height), markFor: shot, hasMark: false });
  }
  return Swap.mark;
}
// Ảnh cảnh có người cần thay phủ hồng, gửi làm ảnh thứ hai như workflow chọn người.
function swapMarked() {
  const shot = Swap.slots[0].ref, mark = swapMark();
  const c = imageToCanvas(shot), g = c.getContext('2d');
  g.globalAlpha = SWAP_MARK.alpha;
  g.drawImage(mark, 0, 0);
  return c;
}
function measureSwapMark() {
  const mark = Swap.mark;
  if (!mark) return false;
  const small = newCanvas(Math.max(1, Math.round(mark.width / 8)), Math.max(1, Math.round(mark.height / 8)));
  const g = small.getContext('2d');
  g.drawImage(mark, 0, 0, small.width, small.height);
  const data = g.getImageData(0, 0, small.width, small.height).data;
  for (let i = 3; i < data.length; i += 4) if (data[i] > 8) return true;
  return false;
}
function drawSwap() {
  const view = $('swapView'), canvas = $('swapCanvas');
  const box = view.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.round(box.width * dpr));
  canvas.height = Math.max(1, Math.round(box.height * dpr));
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, box.width, box.height);
  const shot = Swap.slots[0].ref;
  $('swapMsg').hidden = !!shot;
  if (!shot) return;
  const w = shot.naturalWidth || shot.width, h = shot.naturalHeight || shot.height;
  const s = Math.min(box.width / w, box.height / h);
  Object.assign(swapFit, { s, x: (box.width - w * s) / 2, y: (box.height - h * s) / 2 });
  g.imageSmoothingQuality = 'high';
  g.drawImage(shot, swapFit.x, swapFit.y, w * s, h * s);
  if (Swap.mode === 'body' && swapMark()) {
    g.globalAlpha = SWAP_MARK.alpha;
    g.drawImage(Swap.mark, swapFit.x, swapFit.y, w * s, h * s);
    g.globalAlpha = 1;
  }
}
function syncSwapMode() {
  const body = Swap.mode === 'body', shot = Swap.slots[0].ref;
  $('swapMode').querySelectorAll('.viewchip').forEach(chip => chip.setAttribute('aria-pressed', String(chip.dataset.mode === Swap.mode)));
  $('swapHint').textContent = SWAP_HINTS[Swap.mode];
  $('swapPick').hidden = !body || !shot;
  Swap.tool = body && shot ? Swap.tool || 'paint' : null;
  $('swapTools').querySelectorAll('[data-tool]').forEach(chip => chip.setAttribute('aria-pressed', String(chip.dataset.tool === Swap.tool)));
  $('swapView').classList.toggle('painting', !!Swap.tool);
  if (!Swap.tool) $('swapRing').hidden = true;
  drawSwap();
}
function syncSwap() {
  syncSlotBoxes('swapSlots', Swap.slots);
  swapMark();
  syncSwapMode();
}
function takeSwapFiles(fileList, start) {
  takeFilesInto(Swap.slots, fileList, start, syncSwap);
}
function buildSwapSlots() {
  Swap.slots[1].label = SWAP_REF_LABEL[Swap.mode];
  buildSlotBoxes('swapSlots', Swap.slots, takeSwapFiles, syncSwap);
  syncSlotBoxes('swapSlots', Swap.slots);
}
function buildSwapPanel() {
  buildSwapSlots();
  $('swapMode').addEventListener('click', e => {
    const chip = e.target.closest('.viewchip');
    if (!chip || chip.dataset.mode === Swap.mode) return;
    Swap.mode = chip.dataset.mode;
    buildSwapSlots();
    syncSwapMode();
    renderPanel();
  });
  $('swapTools').addEventListener('click', e => {
    const chip = e.target.closest('[data-tool]');
    if (!chip) return;
    Swap.tool = chip.dataset.tool;
    syncSwapMode();
  });
  $('swapClear').addEventListener('click', () => {
    if (!Swap.mark) return;
    Swap.mark.getContext('2d').clearRect(0, 0, Swap.mark.width, Swap.mark.height);
    Swap.hasMark = false;
    drawSwap();
    renderPanel();
  });
  $('swapSize').addEventListener('input', e => {
    Swap.size = Number(e.target.value);
    $('swapRing').style.width = $('swapRing').style.height = `${Swap.size}px`;
  });
  const view = $('swapView'), ring = $('swapRing');
  const toImage = e => {
    const box = view.getBoundingClientRect();
    return { x: (e.clientX - box.left - swapFit.x) / swapFit.s, y: (e.clientY - box.top - swapFit.y) / swapFit.s };
  };
  const paint = (a, b) => {
    const g = swapMark().getContext('2d');
    g.globalCompositeOperation = Swap.tool === 'erase' ? 'destination-out' : 'source-over';
    g.fillStyle = g.strokeStyle = SWAP_MARK.color;
    g.lineWidth = Swap.size / swapFit.s;
    g.lineCap = g.lineJoin = 'round';
    g.beginPath();
    g.arc(b.x, b.y, g.lineWidth / 2, 0, Math.PI * 2);
    g.fill();
    if (a) {
      g.beginPath();
      g.moveTo(a.x, a.y);
      g.lineTo(b.x, b.y);
      g.stroke();
    }
    g.globalCompositeOperation = 'source-over';
  };
  let last = null;
  view.addEventListener('pointerdown', e => {
    if (!Swap.tool || !Swap.slots[0].ref) return;
    view.setPointerCapture(e.pointerId);
    last = toImage(e);
    paint(null, last);
    drawSwap();
  });
  view.addEventListener('pointermove', e => {
    if (Swap.tool) {
      const box = view.getBoundingClientRect();
      ring.hidden = false;
      ring.style.width = ring.style.height = `${Swap.size}px`;
      ring.style.transform = `translate(${e.clientX - box.left}px, ${e.clientY - box.top}px) translate(-50%, -50%)`;
    }
    if (!last) return;
    const point = toImage(e);
    paint(last, point);
    last = point;
    drawSwap();
  });
  const finish = () => {
    if (!last) return;
    last = null;
    Swap.hasMark = measureSwapMark();
    renderPanel();
  };
  view.addEventListener('pointerup', finish);
  view.addEventListener('pointercancel', finish);
  view.addEventListener('pointerleave', () => { ring.hidden = true; });
  syncSwap();
}
// Bóng người đứng giữa cảnh mẫu, chỉ để xem giao diện khi chưa nối ComfyUI.
function sampleSwapScene() {
  const c = sampleScene(), g = c.getContext('2d');
  g.fillStyle = '#1B2126';
  g.beginPath();
  g.moveTo(628, 704);
  g.lineTo(640, 536);
  g.quadraticCurveTo(672, 508, 704, 536);
  g.lineTo(716, 704);
  g.closePath();
  g.fill();
  g.fillStyle = '#C49478';
  g.beginPath();
  g.arc(672, 486, 26, 0, Math.PI * 2);
  g.fill();
  g.fillStyle = '#2B1C15';
  g.beginPath();
  g.arc(672, 480, 27, Math.PI, 0);
  g.fill();
  return c;
}
// Tab Chat: gọi LLM qua /vstone/llm trên cổng ComfyUI (mục 7 của stoneV_CloudSyncHub), trả lời dạng stream.
const CHAT_MODES = [
  { id: 'ko2vi', label: 'Hàn → Việt', hint: 'Dán tiếng Hàn, nhận bản dịch tiếng Việt.', think: false,
    system: 'Translate the user\'s Korean text into natural, accurate Vietnamese. Keep names, numbers, formatting and line breaks. Output only the translation.' },
  { id: 'vi2ko', label: 'Việt → Hàn', hint: 'Dán tiếng Việt, nhận bản dịch tiếng Hàn.', think: false,
    system: 'Translate the user\'s Vietnamese text into natural, accurate Korean suited to a professional film studio. Keep names, numbers, formatting and line breaks. Output only the translation.' },
  { id: 'prompt', label: 'Viết prompt', hint: 'Tả bằng tiếng Việt, nhận prompt tiếng Anh cho Qwen Image.', think: false,
    system: 'Turn the user\'s description into one clear English prompt for an image generation or editing model. Describe subject, setting, lighting, camera and style concretely. Output only the prompt.' },
  { id: 'code', label: 'Code', hint: 'Hỏi về code, script, tiện ích.', think: false,
    system: 'You are a concise coding assistant. Answer in Vietnamese, with complete, working code when code is needed.' },
  { id: 'free', label: 'Tự do', hint: 'Hỏi gì cũng được.', think: false, system: '' },
];
const LLM_MODEL = 'qwen3.6-35b-a3b';
const Chat = { mode: 'ko2vi', think: false, history: [], busy: false, abort: null };
function chatBubble(role, content) {
  $('chatEmpty').hidden = true;
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  if (role === 'bot') {
    const body = document.createElement('div');
    body.textContent = content;
    const copy = iconButton('copy', 'Chép', () => copyText(body.textContent).then(ok => {
      if (ok) { copy.innerHTML = icon('check'); setTimeout(() => { copy.innerHTML = icon('copy'); }, 1200); }
    }));
    copy.classList.add('msgcopy');
    div.append(body, copy);
  } else {
    div.textContent = content;
  }
  $('chatLog').append(div);
  $('chatLog').scrollTop = $('chatLog').scrollHeight;
  return div;
}
function syncChat() {
  $('chatMode').querySelectorAll('.viewchip').forEach(chip => chip.setAttribute('aria-pressed', String(chip.dataset.mode === Chat.mode)));
  $('chatThink').setAttribute('aria-pressed', String(Chat.think));
  $('chatHint').textContent = CHAT_MODES.find(m => m.id === Chat.mode).hint;
  $('chatSend').textContent = Chat.busy ? 'Dừng' : 'Gửi';
}
async function sendChat() {
  if (Chat.busy) { if (Chat.abort) Chat.abort.abort(); return; }
  const content = $('chatInput').value.trim();
  if (!content) return;
  const mode = CHAT_MODES.find(m => m.id === Chat.mode);
  const think = Chat.think;
  $('chatInput').value = '';
  chatBubble('user', content);
  Chat.history.push({ role: 'user', content });
  const bubble = chatBubble('bot', '…');
  const textNode = bubble.firstChild;
  if (!ON_COMFY) {
    textNode.textContent = 'Bản xem trước: chưa nối LLM. Trên H200, câu trả lời sẽ hiện ở đây, chạy dần từng chữ.';
    return;
  }
  Chat.busy = true;
  Chat.abort = new AbortController();
  syncChat();
  let answer = '', thought = '', thinkBox = null, thinkSum = null, thinkBody = null;
  try {
    const sys = mode.system + (think ? ' Think briefly (under 150 words) before giving the final answer.' : '');
    const messages = (sys ? [{ role: 'system', content: sys.trim() }] : []).concat(Chat.history.slice(-20));
    const res = await fetch('/vstone/llm/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: Chat.abort.signal,
      body: JSON.stringify({
        model: LLM_MODEL,
        messages,
        stream: true,
        max_tokens: think ? 4096 : 2048,
        temperature: think ? 0.6 : 0.3,
        presence_penalty: 0.3,
        chat_template_kwargs: { enable_thinking: think },
      }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.error || (detail.message) || `LLM lỗi ${res.status}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    textNode.textContent = think ? 'Đang suy nghĩ…' : '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
        const delta = (JSON.parse(line.slice(6)).choices[0] || {}).delta || {};
        const r = delta.reasoning_content || delta.reasoning || '';
        if (r) {
          if (!thinkBox) {
            thinkBox = document.createElement('details');
            thinkBox.className = 'msgthink';
            thinkBox.open = true;
            thinkSum = document.createElement('summary');
            thinkSum.textContent = 'Đang suy nghĩ…';
            thinkBody = document.createElement('div');
            thinkBox.append(thinkSum, thinkBody);
            bubble.insertBefore(thinkBox, textNode);
            textNode.textContent = '';
          }
          thought += r;
          thinkBody.textContent = thought;
          thinkBody.scrollTop = thinkBody.scrollHeight;
          $('chatLog').scrollTop = $('chatLog').scrollHeight;
        }
        if (delta.content) {
          if (thinkSum && thinkSum.textContent !== 'Quá trình suy nghĩ') {
            thinkSum.textContent = 'Quá trình suy nghĩ';
          }
          answer += delta.content;
          textNode.textContent = answer;
          $('chatLog').scrollTop = $('chatLog').scrollHeight;
        }
      }
    }
    if (!answer) {
      textNode.textContent = thought ? '(Đã suy nghĩ xong ở khung trên. Nếu muốn trả lời thẳng nhanh hơn, hãy tắt nút "Suy nghĩ sâu".)' : '(Không nhận được câu trả lời.)';
    }
    Chat.history.push({ role: 'assistant', content: answer || thought });
  } catch (e) {
    if (e.name === 'AbortError') {
      textNode.textContent = answer || 'Đã dừng.';
      if (answer) Chat.history.push({ role: 'assistant', content: answer });
    } else {
      bubble.classList.add('err');
      textNode.textContent = `${e.message || e}. Nếu LLM chưa chạy, mở ComfyUI và chạy mục 7 của node stoneV_CloudSyncHub.`;
      Chat.history.pop();
    }
  } finally {
    Chat.busy = false;
    Chat.abort = null;
    syncChat();
  }
}
function buildChatPanel() {
  $('chatMode').innerHTML = CHAT_MODES.map(m => `<button class="viewchip" type="button" data-mode="${m.id}" aria-pressed="false">${m.label}</button>`).join('');
  $('chatMode').addEventListener('click', e => {
    const chip = e.target.closest('.viewchip');
    if (!chip) return;
    Chat.mode = chip.dataset.mode;
    Chat.think = CHAT_MODES.find(m => m.id === Chat.mode).think;
    syncChat();
    $('chatInput').focus();
  });
  $('chatThink').addEventListener('click', () => {
    Chat.think = !Chat.think;
    syncChat();
    $('chatInput').focus();
  });
  $('chatClear').addEventListener('click', () => {
    if (Chat.abort) Chat.abort.abort();
    Chat.history = [];
    $('chatLog').querySelectorAll('.msg').forEach(m => m.remove());
    $('chatEmpty').hidden = false;
  });
  $('chatSend').addEventListener('click', sendChat);
  $('chatInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendChat(); }
  });
  syncChat();
}
// Tab Upscale theo workflow Tiled Upscale của AusBoss (workflows/qwen_image_2.1/qwen_image_2_1_tiled_upscale_ausboss.json).
// Qwen 2.1 vẽ lại ảnh theo từng ô khoảng 2 MP rồi ghép, Restore LoRA giữ sát ảnh gốc, mỗi ô khớp màu với ô gốc.
// Prompt tiếng Anh là prompt gốc trong workflow của tác giả, giữ nguyên.
const UPSCALE_PROMPT = {
  en: 'Restore the photo: remove blur, noise and compression artifacts and recover sharp, natural detail, keeping the same composition, people and colors.',
  vi: 'Phục hồi ảnh: bỏ mờ, nhiễu và vết nén, lấy lại chi tiết sắc nét, tự nhiên, giữ nguyên bố cục, người và màu.',
};
const UPSCALE_HINT = 'Qwen 2.1 vẽ lại ảnh nét hơn ở cỡ lớn: chia thành các ô khoảng 2 MP, vẽ từng ô rồi ghép, mỗi ô giữ màu ảnh gốc. Chi tiết nhỏ là vẽ thêm cho hợp, mặt nhỏ có thể đổi, chữ nhỏ có thể sai. Ảnh đã to và nét thì được thêm ít. Dùng workflow Tiled Upscale và Restore LoRA của AusBoss.';
const UPSCALE_LORAS = { restore: 'qwen-image-2.1-restore.safetensors', turbo: 'Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors' };
// Các giá trị của node Tiled Upscale trong workflow tác giả: ô 2 MP, chồng 128 px, cạnh chia hết 32, vẽ lại tối đa gấp 8 lần cỡ gốc.
const UPSCALE_TILE = { tile_megapixels: 2, overlap: 128, multiple: 32, max_growth: 8 };
const UPSCALE_UPLOAD_MP = 16;
const Up = {
  slots: [{ label: 'Ảnh cần upscale', ref: null, refUrl: '' }],
  mp: 4,
  fast: true,
  fromBase: null,
};
function upscaleTarget(w, h, mp) {
  const s = Math.sqrt(mp * 1e6 / (w * h)), m = UPSCALE_TILE.multiple;
  return { w: Math.max(m, Math.round(w * s / m) * m), h: Math.max(m, Math.round(h * s / m) * m) };
}
function setUpInfo(lines) {
  $('upInfo').replaceChildren(...lines.map((line, i) => {
    const span = document.createElement('span');
    span.textContent = line;
    if (i) span.className = 'upwarn';
    return span;
  }));
}
// So sánh trước và sau như node Compare trong workflow của tác giả: rê chuột để kéo đường chia, lăn chuột để phóng, kéo để di chuyển.
const UpView = { zoom: 1, cx: 0, cy: 0, split: 0.5, drag: null };
const upResult = () => (Up.result && Up.resultFor === Up.slots[0].ref ? Up.result : null);
function upLayout(box, res) {
  const rw = res.naturalWidth || res.width, rh = res.naturalHeight || res.height;
  const fitScale = Math.min(box.width / rw, box.height / rh);
  const scale = fitScale * UpView.zoom;
  return { rw, rh, fitScale, scale, x: box.width / 2 - UpView.cx * scale, y: box.height / 2 - UpView.cy * scale };
}
function clampUpView(box, res) {
  const L = upLayout(box, res);
  const halfW = box.width / 2 / L.scale, halfH = box.height / 2 / L.scale;
  UpView.cx = L.rw <= 2 * halfW ? L.rw / 2 : Math.min(L.rw - halfW, Math.max(halfW, UpView.cx));
  UpView.cy = L.rh <= 2 * halfH ? L.rh / 2 : Math.min(L.rh - halfH, Math.max(halfH, UpView.cy));
}
// Phóng tối đa 200% điểm ảnh của kết quả.
const upMaxZoom = (box, res) => Math.max(1, 2 / upLayout(box, res).fitScale);
const upOneToOne = (box, res) => Math.max(1, 1 / upLayout(box, res).fitScale);
function zoomUpAt(box, px, py, zoom) {
  const res = upResult();
  if (!res) return;
  const before = upLayout(box, res);
  const ix = (px - before.x) / before.scale, iy = (py - before.y) / before.scale;
  UpView.zoom = Math.min(upMaxZoom(box, res), Math.max(1, zoom));
  const after = upLayout(box, res);
  UpView.cx = ix - (px - box.width / 2) / after.scale;
  UpView.cy = iy - (py - box.height / 2) / after.scale;
  drawUpscale();
}
function showUpscaleResult(ref, img, url) {
  Object.assign(Up, { result: img, resultUrl: url, resultFor: ref });
  Object.assign(UpView, { zoom: 1, split: 0.5, cx: (img.naturalWidth || img.width) / 2, cy: (img.naturalHeight || img.height) / 2, drag: null });
  drawUpscale();
}
function drawUpLabel(g, text, x, y, align) {
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  g.font = `500 12px ${css('--font-ui')}`;
  const w = g.measureText(text).width + 14;
  const left = align === 'right' ? x - w : x;
  g.fillStyle = css('--well');
  g.fillRect(left, y, w, 22);
  g.fillStyle = css('--ink');
  g.textBaseline = 'middle';
  g.fillText(text, left + 7, y + 11);
}
function drawUpscale() {
  const view = $('upView'), canvas = $('upCanvas');
  const box = view.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.round(box.width * dpr));
  canvas.height = Math.max(1, Math.round(box.height * dpr));
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, box.width, box.height);
  const ref = Up.slots[0].ref, res = upResult();
  $('upMsg').hidden = !!ref;
  $('upInfo').hidden = !ref;
  $('upTools').hidden = !res;
  view.classList.toggle('comparing', !!res);
  // Khung đang ẩn thì chưa vẽ, tránh chia cho 0 khi tính tỉ lệ.
  if (!ref || box.width < 2 || box.height < 2) return;
  const w = ref.naturalWidth || ref.width, h = ref.naturalHeight || ref.height;
  g.imageSmoothingQuality = 'high';
  if (res) {
    clampUpView(box, res);
    const L = upLayout(box, res);
    const splitX = Math.round(box.width * UpView.split);
    for (const [img, x0, x1] of [[ref, 0, splitX], [res, splitX, box.width]]) {
      g.save();
      g.beginPath();
      g.rect(x0, 0, x1 - x0, box.height);
      g.clip();
      g.drawImage(img, L.x, L.y, L.rw * L.scale, L.rh * L.scale);
      g.restore();
    }
    g.fillStyle = 'rgba(0,0,0,.55)';
    g.fillRect(splitX - 1.5, 0, 3, box.height);
    g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--paper').trim();
    g.fillRect(splitX - 0.5, 0, 1, box.height);
    if (splitX > 70) drawUpLabel(g, 'Trước', splitX - 8, box.height - 30, 'right');
    if (box.width - splitX > 60) drawUpLabel(g, 'Sau', splitX + 8, box.height - 30, 'left');
    setUpInfo([`Trước ${w} × ${h}  ·  Sau ${L.rw} × ${L.rh}  ·  ${Math.round(L.scale * 100)}%`]);
    return;
  }
  const s = Math.min(box.width / w, box.height / h);
  g.drawImage(ref, (box.width - w * s) / 2, (box.height - h * s) / 2, w * s, h * s);
  const target = upscaleTarget(w, h, Up.mp), sourceMp = w * h / 1e6;
  const lines = [`${w} × ${h}  →  khoảng ${target.w} × ${target.h}`];
  if (sourceMp > Up.mp) lines.push(`Ảnh đã lớn hơn ${Up.mp} MP, kết quả sẽ nhỏ lại.`);
  else if (Up.mp > UPSCALE_TILE.max_growth * sourceMp) lines.push('Ảnh nhỏ: Qwen vẽ lại tối đa gấp 8 lần cỡ gốc, phần còn lại phóng to thường.');
  setUpInfo(lines);
}
// Chuột: rê để kéo đường chia, lăn để phóng, kéo để di chuyển khi đã phóng, bấm đúp để chuyển giữa vừa khung và 100%.
// Màn cảm ứng: ở cỡ vừa khung thì kéo để chia, đã phóng thì kéo để di chuyển.
function bindUpscaleCompare() {
  const view = $('upView');
  const at = e => { const r = view.getBoundingClientRect(); return { r, x: e.clientX - r.left, y: e.clientY - r.top }; };
  view.addEventListener('pointerdown', e => {
    if (!upResult() || e.button !== 0) return;
    const p = at(e);
    UpView.drag = { x: p.x, y: p.y, cx: UpView.cx, cy: UpView.cy, slide: UpView.zoom <= 1.001 };
    if (UpView.drag.slide) UpView.split = Math.min(1, Math.max(0, p.x / p.r.width));
    view.setPointerCapture(e.pointerId);
    view.classList.toggle('panning', !UpView.drag.slide);
    drawUpscale();
  });
  view.addEventListener('pointermove', e => {
    const res = upResult();
    if (!res) return;
    const p = at(e);
    if (UpView.drag && !UpView.drag.slide) {
      const scale = upLayout(p.r, res).scale;
      UpView.cx = UpView.drag.cx - (p.x - UpView.drag.x) / scale;
      UpView.cy = UpView.drag.cy - (p.y - UpView.drag.y) / scale;
    } else if (UpView.drag || e.pointerType === 'mouse') {
      UpView.split = Math.min(1, Math.max(0, p.x / p.r.width));
    } else {
      return;
    }
    drawUpscale();
  });
  const end = () => { UpView.drag = null; view.classList.remove('panning'); };
  view.addEventListener('pointerup', end);
  view.addEventListener('pointercancel', end);
  view.addEventListener('wheel', e => {
    if (!upResult()) return;
    e.preventDefault();
    const p = at(e);
    zoomUpAt(p.r, p.x, p.y, UpView.zoom * Math.exp(-e.deltaY * 0.0015));
  }, { passive: false });
  view.addEventListener('dblclick', e => {
    const res = upResult();
    if (!res) return;
    const p = at(e);
    zoomUpAt(p.r, p.x, p.y, UpView.zoom > 1.001 ? 1 : upOneToOne(p.r, res));
  });
  $('upFit').addEventListener('click', () => { UpView.zoom = 1; drawUpscale(); });
  $('upOne').addEventListener('click', () => {
    const res = upResult();
    if (!res) return;
    const r = view.getBoundingClientRect();
    zoomUpAt(r, r.width / 2, r.height / 2, upOneToOne(r, res));
  });
  $('upOpen').addEventListener('click', () => { if (upResult()) openViewer(Up.result, Up.resultUrl); });
}
function syncUpscale() {
  syncSlotBoxes('upscaleSlots', Up.slots);
  $('upscaleSize').querySelectorAll('.viewchip').forEach(chip => chip.setAttribute('aria-pressed', String(Number(chip.dataset.mp) === Up.mp)));
  $('upscaleSpeed').querySelectorAll('.viewchip').forEach(chip => chip.setAttribute('aria-pressed', String((chip.dataset.fast === '1') === Up.fast)));
  drawUpscale();
}
function takeUpscaleFiles(fileList, start) {
  Up.fromBase = null;
  takeFilesInto(Up.slots, fileList, start, syncUpscale);
}
// Ô ảnh tự lấy ảnh đang sửa ở tab Sửa ảnh, trừ khi họa sĩ đã bỏ ảnh khác vào.
function syncUpscaleFromBase() {
  const slot = Up.slots[0];
  const own = slot.ref && Up.fromBase !== slot.ref;
  if (S.base && !S.isSample && !own && slot.ref !== S.base) {
    slot.ref = S.base;
    slot.refUrl = S.base.src || S.base.toDataURL('image/jpeg', 0.85);
    Up.fromBase = S.base;
  }
  syncUpscale();
  renderPanel();
}
function buildUpscalePasses() {
  if (!Up.slots[0].ref) return { passes: [], problems: [] };
  return { passes: [{ kind: 'upscale', name: `Upscale ${Up.mp} MP`, lines: [UPSCALE_PROMPT], slots: [] }], problems: [] };
}
function buildUpscalePanel() {
  buildSlotBoxes('upscaleSlots', Up.slots, takeUpscaleFiles, syncUpscale);
  $('upscaleHint').textContent = UPSCALE_HINT;
  bindUpscaleCompare();
  $('upscaleSize').addEventListener('click', e => {
    const chip = e.target.closest('.viewchip');
    if (!chip) return;
    Up.mp = Number(chip.dataset.mp);
    syncUpscale();
    renderPanel();
  });
  $('upscaleSpeed').addEventListener('click', e => {
    const chip = e.target.closest('.viewchip');
    if (!chip) return;
    Up.fast = chip.dataset.fast === '1';
    syncUpscale();
    renderPanel();
  });
  syncUpscale();
}
// Đồ thị theo đúng workflow Tiled Upscale của tác giả, chỉ bỏ node cắt ảnh (để mặc định là giữ nguyên ảnh), node so sánh và node đo giờ.
// Nhanh: bật LoRA Viggle Turbo, 8 bước. Kỹ: tắt Viggle, 25 bước. Giống công tắc Fast của tác giả.
function buildUpscaleGraph({ image, text, seed, mp, fast }) {
  const loras = [
    { name: UPSCALE_LORAS.restore, strength: 1.0, enabled: true, triggers: '' },
    { name: UPSCALE_LORAS.turbo, strength: 1.0, enabled: fast, triggers: '' },
  ];
  return {
    1: { class_type: 'UNETLoader', inputs: { unet_name: 'qwen_image_2.1_int8_convrot.safetensors', weight_dtype: 'default' } },
    2: { class_type: 'CLIPLoader', inputs: { clip_name: 'qwen3vl_8b_int8_convrot.safetensors', type: 'qwen_image', device: 'default' } },
    3: { class_type: 'VAELoader', inputs: { vae_name: 'qwen_image_2.1_vae_bf16.safetensors' } },
    4: { class_type: 'LoadImage', inputs: { image } },
    11: { class_type: 'AUSBOSS_NODES_LoraLoader', inputs: { model: ['1', 0], loras: JSON.stringify(loras), trigger_separator: ', ', on_missing: 'error' } },
    30: { class_type: 'AUSBOSS_NODES_TiledUpscale', inputs: { image: ['4', 0], megapixels: mp, ...UPSCALE_TILE } },
    17: { class_type: 'QwenImage21Cache', inputs: { model: ['11', 0], device: 'auto', dtype: 'default' } },
    7: { class_type: 'TextEncodeQwenImage21', inputs: { clip: ['2', 0], vae: ['3', 0], prompt: text, negative_prompt: '', resolution: 0, 'images.image_1': ['30', 0] } },
    8: { class_type: 'KSampler', inputs: { model: ['17', 0], positive: ['7', 0], negative: ['7', 1], latent_image: ['7', 2], seed, steps: fast ? 8 : 25, cfg: 1, sampler_name: 'euler', scheduler: 'simple', denoise: 1 } },
    9: { class_type: 'VAEDecode', inputs: { samples: ['8', 0], vae: ['3', 0] } },
    31: { class_type: 'AUSBOSS_NODES_TiledUpscaleStitch', inputs: { tiles: ['9', 0], stitcher: ['30', 1], keep_colors: 1 } },
    6: { class_type: 'SaveImage', inputs: { images: ['31', 0], filename_prefix: 'Concept_upscale' } },
  };
}
// Node Tiled Upscale có từ pack AusBoss 2.15, mục 4 của stoneV_CloudSyncHub cập nhật pack. Hai LoRA đi theo sync_manifest.json.
async function requireUpscaleParts(fast) {
  const read = name => fetch(`/api/object_info/${name}`).then(r => (r.ok ? r.json() : {}), () => ({}));
  const [tile, lora] = await Promise.all([read('AUSBOSS_NODES_TiledUpscale'), read('LoraLoaderModelOnly')]);
  if (!tile.AUSBOSS_NODES_TiledUpscale) {
    throw new Error('ComfyUI chưa có node Tiled Upscale của pack AusBoss (cần bản 2.15 trở lên). Mở ComfyUI, chạy mục 4 của node stoneV_CloudSyncHub, rồi Gen lại.');
  }
  if (!lora.LoraLoaderModelOnly) return;
  const names = lora.LoraLoaderModelOnly.input.required.lora_name[0];
  const missing = [UPSCALE_LORAS.restore, fast && UPSCALE_LORAS.turbo].filter(name => name && !names.includes(name));
  if (missing.length) {
    throw new Error(`ComfyUI chưa có ${missing.join(' và ')}. Cloud Sync tải LoRA trong nền, chờ tải xong rồi Gen lại.`);
  }
}
async function generateUpscale() {
  const pass = S.passes.find(p => p.kind === 'upscale');
  const text = promptFor(pass);
  const ref = Up.slots[0].ref;
  const source = fitReference(ref, UPSCALE_UPLOAD_MP);
  const { mp, fast } = Up;
  const stamp = Date.now().toString(36);
  beginRun();
  try {
    startPass(pass.name);
    await requireUpscaleParts(fast);
    const names = await uploadAll([source], `${stamp}_upscale`);
    run.promptId = await queueGraph(buildUpscaleGraph({ image: names[0], text, seed: randomSeed(), mp, fast }));
    advanceJob('queue');
    renderJob();
    const { img, url } = await loadOutput(await waitForImage(run.promptId));
    addResult(img, url, pass.name);
    showUpscaleResult(ref, img, url);
    hideJob();
    setStatus(`Xong trong ${clock(Date.now() - job.startedAt)}.`);
  } catch (e) {
    failRun(e);
  } finally {
    endRun();
  }
}
function switchTab(tab) {
  S.tab = tab;
  const edit = tab === 'edit';
  $('tabEdit').setAttribute('aria-selected', String(edit));
  $('tabSheet').setAttribute('aria-selected', String(tab === 'sheet'));
  $('tabRotate').setAttribute('aria-selected', String(tab === 'rotate'));
  $('tabSwap').setAttribute('aria-selected', String(tab === 'swap'));
  $('strip').hidden = !edit;
  $('sheetPanel').hidden = tab !== 'sheet';
  $('rotatePanel').hidden = tab !== 'rotate';
  $('swapPanel').hidden = tab !== 'swap';
  document.querySelectorAll('.edit-only').forEach(el => { el.hidden = !edit; });
  closePop();
  toggleEnv(false);
  stopCapture();
  if (!edit && S.tool) {
    S.tool = null;
    clearSelection();
  }
  syncToolUi();
  $('viewer').hidden = true;
  brushcur.hidden = true;
  syncStage();
  renderPanel();
  if (edit) fit();
  if (tab === 'rotate') syncRotateMode();
  if (tab === 'swap') syncSwapMode();
  $('tabUpscale').setAttribute('aria-selected', String(tab === 'upscale'));
  $('upscalePanel').hidden = tab !== 'upscale';
  if (tab === 'upscale') syncUpscaleFromBase();
  $('tabChat').setAttribute('aria-selected', String(tab === 'chat'));
  $('chatPanel').hidden = tab !== 'chat';
  document.querySelectorAll('.bar > :not(.grow)').forEach(el => { if (!el.classList.contains('edit-only')) el.hidden = tab === 'chat'; });
  if (tab === 'chat') $('chatInput').focus();
}

function buildApiPrompt({ image1, refs, text, resolution, seed, lora, latent }) {
  const graph = JSON.parse(JSON.stringify(EDIT_GRAPH));
  const encoder = graph[7].inputs;
  graph[4].inputs.image = image1;
  encoder.prompt = text;
  encoder.resolution = resolution;
  graph[8].inputs.seed = seed;
  refs.forEach((name, i) => {
    const id = String(20 + i);
    graph[id] = { class_type: 'LoadImage', inputs: { image: name } };
    encoder[`images.image_${i + 2}`] = [id, 0];
  });
  if (lora) {
    graph[15] = { class_type: 'LoraLoaderModelOnly', inputs: { model: ['1', 0], lora_name: lora, strength_model: 1 } };
    graph[8].inputs.model = ['15', 0];
  }
  if (latent) {
    // Như nhánh custom size của workflow Qwen chính thức: cỡ ảnh ra theo khung trống thay vì theo image_1.
    graph[30] = { class_type: 'EmptyLatentImage', inputs: { width: latent.w, height: latent.h, batch_size: 1 } };
    graph[8].inputs.latent_image = ['30', 0];
  }
  return graph;
}

const CLIENT_ID = 'concept-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
const run = { active: false, cancelled: false, promptId: null };
let shown = null;

const BASE_TITLE = document.title;
const JOB_STEPS = [['upload', 'Tải ảnh'], ['queue', 'Hàng đợi'], ['load', 'Nạp model'], ['read', 'Đọc ảnh, prompt'], ['sample', 'Vẽ'], ['save', 'Lưu']];
const JOB_ORDER = JOB_STEPS.map(s => s[0]);
// Số node trong EDIT_GRAPH. Ảnh tham chiếu dùng id từ 20 trở đi.
const NODE_STAGE = { 1: 'load', 2: 'load', 3: 'load', 15: 'load', 4: 'read', 7: 'read', 8: 'sample', 9: 'save', 6: 'save' };
const job = { stage: 'upload', ahead: 0, value: 0, max: 0, firstValue: 0, firstAt: 0, startedAt: 0, started: false, pass: '', cancelling: false, timer: 0 };
let socketOpen = false;

function clock(ms) {
  const total = Math.max(0, Math.round(ms / 1000));
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}
function jobHeadline() {
  if (job.cancelling) return 'Đang dừng...';
  if (job.stage === 'upload') return 'Đang tải ảnh lên ComfyUI';
  if (job.stage === 'queue') return job.ahead ? `Đang xếp hàng: ${job.ahead} việc phía trước` : 'Đang vào hàng đợi';
  if (!socketOpen) return 'ComfyUI đang chạy việc này';
  if (job.stage === 'load') return 'Đang nạp model';
  if (job.stage === 'read') return 'Đang đọc ảnh và prompt';
  if (job.stage === 'save') return 'Đang giải mã và lưu ảnh';
  return job.max ? `Đang vẽ bước ${job.value}/${job.max}` : 'Đang nạp model vào GPU';
}
function jobRemaining() {
  if (job.stage !== 'sample' || job.value - job.firstValue < 2) return '';
  const perStep = (Date.now() - job.firstAt) / (job.value - job.firstValue);
  return `còn khoảng ${clock(perStep * (job.max - job.value))}`;
}
function jobPercent() {
  if (job.stage === 'sample') return job.max ? 25 + 65 * job.value / job.max : 25;
  return { upload: 3, queue: 6, load: 12, read: 20, save: 95 }[job.stage];
}
function renderJob() {
  if (!run.active) return;
  const at = JOB_ORDER.indexOf(job.stage);
  const box = $('job');
  box.hidden = false;
  box.dataset.kind = '';
  $('jobClose').hidden = true;
  $('jobTitle').textContent = jobHeadline();
  $('jobTime').textContent = clock(Date.now() - job.startedAt);
  $('jobSub').textContent = [job.pass, jobRemaining()].filter(Boolean).join(' · ');
  $('jobBarWrap').hidden = false;
  $('jobBar').style.width = jobPercent() + '%';
  $('jobSteps').hidden = false;
  $('jobSteps').innerHTML = JOB_STEPS.map(([, label], i) => `<li data-state="${i < at ? 'done' : i === at ? 'now' : 'todo'}">${label}</li>`).join('');
  const short = job.stage === 'sample' && job.max ? `${job.value}/${job.max}` : job.stage === 'queue' ? 'Xếp hàng' : 'Đang chạy';
  document.title = `${short} · ${BASE_TITLE}`;
}
function advanceJob(stage) {
  if (JOB_ORDER.indexOf(stage) > JOB_ORDER.indexOf(job.stage)) job.stage = stage;
}
function startPass(name) {
  Object.assign(job, { stage: 'upload', ahead: 0, value: 0, max: 0, firstValue: 0, firstAt: 0, started: false, pass: name });
  renderJob();
}
function hideJob() {
  clearInterval(job.timer);
  $('job').hidden = true;
  document.title = BASE_TITLE;
}
function showJobError(message) {
  clearInterval(job.timer);
  document.title = BASE_TITLE;
  const box = $('job');
  box.hidden = false;
  box.dataset.kind = 'error';
  $('jobTitle').textContent = 'Không chạy được';
  $('jobTime').textContent = '';
  $('jobSub').textContent = message;
  $('jobBarWrap').hidden = true;
  $('jobSteps').hidden = true;
  $('jobClose').hidden = false;
}
async function checkQueue(promptId) {
  try {
    const queue = await (await fetch('/api/queue')).json();
    const running = queue.queue_running || [];
    const pending = queue.queue_pending || [];
    if (running.some(item => item[1] === promptId)) {
      job.started = true;
      advanceJob('load');
    } else {
      const mine = pending.find(item => item[1] === promptId);
      if (mine) job.ahead = running.length + pending.filter(item => item[0] < mine[0]).length;
    }
    renderJob();
  } catch (e) {
    // Không đọc được hàng đợi thì vẫn chờ kết quả, chỉ thiếu số việc phía trước.
  }
}

function setStatus(text) {
  $('status').textContent = text || '';
}
function syncGen() {
  const gen = $('gen');
  if (!ON_COMFY) { gen.disabled = true; gen.title = 'Chưa nối ComfyUI'; return; }
  gen.title = '';
  if (run.active) { gen.disabled = false; return; }
  gen.textContent = 'Gen';
  gen.disabled = !S.passes || !S.passes.length || S.problems.length > 0;
}
function promptFor(pass) {
  return S.overrides[pass.kind] ?? pass.lines.map(l => l.en).join(' ');
}
function newCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}
function imageToCanvas(img) {
  const c = newCanvas(img.naturalWidth || img.width, img.naturalHeight || img.height);
  c.getContext('2d').drawImage(img, 0, 0);
  return c;
}
function takeRunSnapshot() {
  const padded = newCanvas(W(), H());
  const g = padded.getContext('2d');
  g.fillStyle = PAD_GRAY;
  g.fillRect(0, 0, W(), H());
  g.drawImage(S.base, S.pads.left, S.pads.top, S.w, S.h);
  return { w: W(), h: H(), pads: { ...S.pads }, base: S.base, baseW: S.w, baseH: S.h, padded };
}
// Bước 4 trong hướng dẫn của LoRA outpaint: dán lại điểm ảnh gốc, làm mờ 32 px ở các cạnh được nới.
function stitchOriginal(result, snap) {
  const out = newCanvas(snap.w, snap.h);
  const g = out.getContext('2d');
  g.drawImage(result, 0, 0, snap.w, snap.h);
  const original = newCanvas(snap.baseW, snap.baseH);
  const o = original.getContext('2d');
  o.drawImage(snap.base, 0, 0, snap.baseW, snap.baseH);
  o.globalCompositeOperation = 'destination-out';
  const fade = (x0, y0, x1, y1, rx, ry, rw, rh) => {
    const grad = o.createLinearGradient(x0, y0, x1, y1);
    grad.addColorStop(0, 'rgba(0,0,0,1)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    o.fillStyle = grad;
    o.fillRect(rx, ry, rw, rh);
  };
  const f = STITCH_FEATHER, bw = snap.baseW, bh = snap.baseH;
  if (snap.pads.left) fade(0, 0, f, 0, 0, 0, f, bh);
  if (snap.pads.right) fade(bw, 0, bw - f, 0, bw - f, 0, f, bh);
  if (snap.pads.top) fade(0, 0, 0, f, 0, 0, bw, f);
  if (snap.pads.bottom) fade(0, bh, 0, bh - f, 0, bh - f, bw, f);
  g.drawImage(original, snap.pads.left, snap.pads.top);
  return out;
}

// Gửi cả shot để model thấy môi trường, vùng được vẽ đi bằng mask nới 8 px, rồi dán lại với viền mềm hẹp (như ví dụ Inpaint của AusBoss).
// Shot đã nhỏ hơn mức MP đã chọn thì giữ cỡ gốc: phóng lên chỉ tốn thêm, dán lại vẫn về cỡ cũ.
const MASK_GROW = 8;
function planShot(w, h, megapixels) {
  const scale = megapixels && w * h > megapixels * 1e6 ? Math.sqrt(megapixels * 1e6 / (w * h)) : 1;
  const to32 = v => Math.max(64, Math.round(v * scale / 32) * 32);
  return { x: 0, y: 0, w, h, outW: to32(w), outH: to32(h) };
}
function brushMask(b) {
  const mask = newCanvas(W(), H());
  const g = mask.getContext('2d');
  g.setTransform(1, 0, 0, 1, S.pads.left, S.pads.top);
  for (const s of S.strokes) drawStroke(g, s.brush === b.id ? s : { ...s, erase: true });
  return mask;
}
function stampMask(g, mask, radius) {
  g.drawImage(mask, 0, 0);
  for (const part of [1, 0.66, 0.33]) {
    for (let a = 0; a < 24; a++) {
      const t = a / 24 * Math.PI * 2;
      g.drawImage(mask, Math.cos(t) * radius * part, Math.sin(t) * radius * part);
    }
  }
}
function cropCanvas(source, crop) {
  const out = newCanvas(crop.outW, crop.outH);
  const g = out.getContext('2d');
  g.imageSmoothingQuality = 'high';
  g.drawImage(source, crop.x, crop.y, crop.w, crop.h, 0, 0, crop.outW, crop.outH);
  return out;
}
function modelMask(mask, crop) {
  const scaled = cropCanvas(mask, crop);
  const out = newCanvas(crop.outW, crop.outH);
  const g = out.getContext('2d');
  stampMask(g, scaled, MASK_GROW);
  g.globalCompositeOperation = 'source-in';
  g.fillStyle = '#fff';
  g.fillRect(0, 0, crop.outW, crop.outH);
  g.globalCompositeOperation = 'destination-over';
  g.fillStyle = '#000';
  g.fillRect(0, 0, crop.outW, crop.outH);
  return out;
}
// Viền mỏng nằm ngoài mask, chỉ có trên ảnh model nhìn, ảnh bắt đầu vẽ vẫn sạch.
function locatorImage(clean, mask, crop, hex) {
  const scaled = cropCanvas(mask, crop);
  const ring = newCanvas(crop.outW, crop.outH);
  const r = ring.getContext('2d');
  stampMask(r, scaled, MASK_GROW + 14);
  r.globalCompositeOperation = 'destination-out';
  stampMask(r, scaled, MASK_GROW + 8);
  r.globalCompositeOperation = 'source-in';
  r.fillStyle = hex;
  r.fillRect(0, 0, crop.outW, crop.outH);
  const out = newCanvas(crop.outW, crop.outH);
  const g = out.getContext('2d');
  g.drawImage(clean, 0, 0);
  g.drawImage(ring, 0, 0);
  return out;
}
function guideImage(clean, marks, crop) {
  return marks.reduce((image, m) => locatorImage(image, m.mask, crop, m.hex), clean);
}
function pasteCrop(current, result, crop, mask) {
  const scale = crop.outW / crop.w;
  const w = current.width, h = current.height;
  const grown = newCanvas(w, h);
  stampMask(grown.getContext('2d'), mask, Math.max(2, Math.round(MASK_GROW / scale)));
  const soft = newCanvas(w, h);
  const s = soft.getContext('2d');
  s.filter = `blur(${Math.max(1, Math.round(3 / scale))}px)`;
  s.drawImage(grown, 0, 0);
  s.filter = 'none';
  const edited = newCanvas(w, h);
  const e = edited.getContext('2d');
  e.imageSmoothingQuality = 'high';
  e.drawImage(result, crop.x, crop.y, crop.w, crop.h);
  e.globalCompositeOperation = 'destination-in';
  e.drawImage(soft, 0, 0);
  const out = newCanvas(w, h);
  const g = out.getContext('2d');
  g.drawImage(current, 0, 0);
  g.drawImage(edited, 0, 0);
  return out;
}
// Ảnh tham chiếu lớn làm mọi bước chậm đi, nên đưa về khoảng 1 MP trước khi gửi.
function fitReference(img, megapixels = 1) {
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  const s = Math.min(1, Math.sqrt(megapixels * 1e6 / (w * h)));
  const c = newCanvas(Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s)));
  const g = c.getContext('2d');
  g.imageSmoothingQuality = 'high';
  g.drawImage(img, 0, 0, c.width, c.height);
  return c;
}
function buildInpaintPrompt({ shot, outlined, mask, refs, text, seed }) {
  const graph = buildApiPrompt({ image1: shot, refs, text, resolution: 0, seed, lora: null });
  graph[44] = { class_type: 'LoadImage', inputs: { image: outlined } };
  graph[7].inputs['images.image_1'] = ['44', 0];
  graph[41] = { class_type: 'VAEEncode', inputs: { pixels: ['4', 0], vae: ['3', 0] } };
  graph[42] = { class_type: 'LoadImageMask', inputs: { image: mask, channel: 'red' } };
  graph[43] = { class_type: 'SetLatentNoiseMask', inputs: { samples: ['41', 0], mask: ['42', 0] } };
  graph[8].inputs.latent_image = ['43', 0];
  return graph;
}

async function uploadCanvas(canvas, name) {
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
  const form = new FormData();
  form.append('image', blob, name);
  form.append('type', 'input');
  form.append('subfolder', 'concept_edit');
  form.append('overwrite', 'true');
  const res = await fetch('/api/upload/image', { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Không tải được ảnh lên ComfyUI (lỗi ${res.status}).`);
  const data = await res.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}
function promptErrorText(data) {
  const parts = [];
  for (const node of Object.values(data.node_errors || {})) {
    for (const err of node.errors || []) parts.push(`${node.class_type}: ${err.message}${err.details ? ` (${err.details})` : ''}`);
  }
  return parts.join(' · ') || (data.error && data.error.message) || 'ComfyUI từ chối lệnh.';
}
async function queueGraph(graph) {
  const res = await fetch('/api/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: graph, client_id: CLIENT_ID }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(promptErrorText(data));
  return data.prompt_id;
}
async function waitForHistory(promptId, pick, missing) {
  for (;;) {
    if (run.cancelled) throw new Error('Đã dừng.');
    if (!job.started) await checkQueue(promptId);
    const res = await fetch(`/api/history/${promptId}`);
    const entry = res.ok ? (await res.json())[promptId] : null;
    if (entry) {
      const status = entry.status || {};
      if (status.status_str === 'error') {
        const failed = (status.messages || []).find(m => m[0] === 'execution_error');
        const interrupted = (status.messages || []).some(m => m[0] === 'execution_interrupted');
        if (interrupted) throw new Error('Đã dừng.');
        throw new Error(failed ? `${failed[1].node_type}: ${failed[1].exception_message}` : 'ComfyUI chạy lỗi.');
      }
      const found = pick(entry.outputs || {});
      if (found) return found;
      if (status.completed) throw new Error(missing);
    }
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
}
function waitForImage(promptId) {
  return waitForHistory(promptId, outputs => Object.values(outputs).flatMap(o => o.images || []).filter(i => i.type === 'output')[0], 'ComfyUI chạy xong nhưng không có ảnh ra.');
}
function viewUrl(info) {
  return `/api/view?filename=${encodeURIComponent(info.filename)}&subfolder=${encodeURIComponent(info.subfolder || '')}&type=${encodeURIComponent(info.type)}`;
}
function loadOutput(info) {
  const url = viewUrl(info);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ img, url });
    img.onerror = () => reject(new Error('Không tải được ảnh kết quả.'));
    img.src = url;
  });
}
function addResult(img, url, title) {
  const list = $('results');
  const thumb = document.createElement('button');
  thumb.type = 'button';
  thumb.title = title;
  const small = document.createElement('img');
  small.src = url;
  small.alt = title;
  thumb.append(small);
  thumb.onclick = () => openViewer(img, url);
  list.prepend(thumb);
  list.hidden = false;
}
// Gen xong thì ảnh kết quả thành ảnh đang sửa. Đang tô dở trong lúc chạy thì giữ nguyên nét, chỉ mở ảnh ra xem.
async function continueFrom(result, snap, strokesAtStart) {
  const untouched = S.base === snap.base && S.strokes === strokesAtStart && S.pads.left === snap.pads.left && S.pads.right === snap.pads.right && S.pads.top === snap.pads.top && S.pads.bottom === snap.pads.bottom;
  if (S.base === snap.base && !S.baseFromResult) {
    const before = await canvasToResult(imageToCanvas(S.base));
    addResult(before.img, before.url, 'Ảnh gốc');
    S.baseFromResult = true;
  }
  addResult(result.img, result.url, 'Ảnh kết quả');
  if (untouched) setBase(result.img, false, true);
  else openViewer(result.img, result.url);
  return untouched;
}
function openViewer(img, url) {
  shown = img;
  $('viewerImg').src = url;
  $('viewerSave').href = url;
  $('viewer').hidden = false;
}
function canvasToResult(canvas) {
  return new Promise(resolve => canvas.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => resolve({ img, url });
    img.src = url;
  }, 'image/png'));
}
function randomSeed() {
  return Math.floor(Math.random() * 2 ** 48);
}
function handleSocketMessage({ type, data }) {
  if (type === 'status' && data && data.status && data.status.exec_info) {
    Q.remaining = data.status.exec_info.queue_remaining;
    renderQueueState();
  }
  if (!run.active || !data) return;
  if (data.prompt_id && run.promptId && data.prompt_id !== run.promptId) return;
  if (type === 'execution_start') {
    job.started = true;
    advanceJob('load');
  } else if (type === 'executing' && data.node !== null && data.node !== undefined) {
    job.started = true;
    advanceJob(NODE_STAGE[data.node] || 'read');
  } else if (type === 'progress' && String(data.node) === '8') {
    job.started = true;
    advanceJob('sample');
    job.value = data.value;
    job.max = data.max;
    if (!job.firstAt) {
      job.firstAt = Date.now();
      job.firstValue = data.value;
    }
  } else {
    return;
  }
  renderJob();
}
function connectProgress() {
  let socket;
  try {
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws?clientId=${CLIENT_ID}`);
  } catch (e) {
    return;
  }
  socket.onopen = () => { socketOpen = true; renderJob(); };
  socket.onmessage = e => {
    if (typeof e.data === 'string') handleSocketMessage(JSON.parse(e.data));
  };
  socket.onclose = () => {
    socketOpen = false;
    renderJob();
    setTimeout(connectProgress, 4000);
  };
}
// Ảnh tham chiếu của lượt sửa chia nhau 2 MP, ảnh môi trường mỗi ảnh 0,5 MP, để lượt chạy không quá nặng.
const EDIT_REF_MP = 2;
const ENV_REF_MP = 0.5;
function editStep(pass) {
  const marks = pass.brushes.map(id => {
    const b = brushOf(id);
    return { mask: brushMask(b), hex: PAINT[b.color].hex };
  });
  const area = newCanvas(W(), H());
  for (const m of marks) area.getContext('2d').drawImage(m.mask, 0, 0);
  const refMp = Math.min(1, EDIT_REF_MP / pass.slots.length);
  return { name: pass.name, text: promptFor(pass), marks, area, refs: pass.slots.map(s => fitReference(brushOf(s.brush).ref, refMp)) };
}
async function uploadAll(canvases, prefix) {
  const names = [];
  for (let i = 0; i < canvases.length; i++) names.push(await uploadCanvas(canvases[i], `${prefix}${i + 1}.png`));
  return names;
}
// Hàng đợi chung của ComfyUI: nhiều họa sĩ dùng chung H200, ComfyUI chạy từng việc một. Số việc lấy từ socket, mất socket thì hỏi /api/queue.
const Q = { remaining: null };
function renderQueueState() {
  const n = Q.remaining;
  $('qState').dataset.busy = String(!!n);
  $('qText').textContent = n === null ? 'Chưa kết nối ComfyUI' : n ? `ComfyUI đang bận: ${n} việc` : 'ComfyUI đang rảnh';
}
async function pollQueue() {
  try {
    const queue = await (await fetch('/api/queue')).json();
    Q.remaining = (queue.queue_running || []).length + (queue.queue_pending || []).length;
  } catch (e) {
    Q.remaining = null;
  }
  renderQueueState();
}
// Báo khi xong lúc họa sĩ đang ở tab hoặc cửa sổ khác: đổi tiêu đề tab và kêu một tiếng ngắn.
// Trang chạy http trong mạng nội bộ nên thông báo hệ thống của trình duyệt không dùng được.
let audioCtx = null;
function unlockSound() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
  } catch (e) {
    audioCtx = null;
  }
}
function chime(ok) {
  if (!audioCtx) return;
  const t = audioCtx.currentTime;
  (ok ? [660, 880] : [440, 330]).forEach((freq, i) => {
    const osc = audioCtx.createOscillator(), gain = audioCtx.createGain();
    const at = t + i * 0.16;
    osc.type = 'sine';
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.0001, at);
    gain.gain.exponentialRampToValueAtTime(0.15, at + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.3);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start(at);
    osc.stop(at + 0.32);
  });
}
function notifyFinished(ok) {
  if (!document.hidden) return;
  document.title = `${ok ? '✓ Xong' : '✕ Lỗi'} · ${BASE_TITLE}`;
  chime(ok);
}
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && !run.active) document.title = BASE_TITLE;
});
async function generate() {
  unlockSound();
  if (run.active) {
    run.cancelled = true;
    job.cancelling = true;
    renderJob();
    if (run.promptId) {
      fetch('/api/queue', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ delete: [run.promptId] }) });
      // Có prompt_id thì ComfyUI chỉ dừng khi việc đang chạy đúng là việc này, không dừng nhầm việc của họa sĩ khác.
      fetch('/api/interrupt', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt_id: run.promptId }) });
    }
    return;
  }
  if (S.tab === 'sheet') {
    await generateSheet();
    return;
  }
  if (S.tab === 'rotate') {
    await generateRotate();
    return;
  }
  if (S.tab === 'swap') {
    await generateSwap();
    return;
  }
  if (S.tab === 'upscale') {
    await generateUpscale();
    return;
  }
  const expand = S.passes.find(p => p.kind === 'expand');
  const edit = S.passes.find(p => p.kind === 'edit');
  const snap = takeRunSnapshot();
  const expandText = expand && promptFor(expand);
  const envRefs = expand ? S.env.filter(s => s.ref).map(s => fitReference(s.ref, ENV_REF_MP)) : [];
  const step = edit && editStep(edit);
  const merge = S.passes.find(p => p.kind === 'merge');
  const mergeJob = merge && mergeSnapshot(merge);
  const megapixels = Number($('resolution').value);
  const total = (expand ? 1 : 0) + (merge ? 1 : 0) + (edit ? 1 : 0);
  const passLabel = (index, name) => (total > 1 ? `Lượt ${index}/${total}: ${name}` : name);
  const stamp = Date.now().toString(36);
  const strokesAtStart = S.strokes;
  beginRun();
  try {
    let current = snap.padded;
    if (expand) {
      startPass(passLabel(1, 'Kéo rộng'));
      const name = await uploadCanvas(current, `${stamp}_outpaint.png`);
      const refs = await uploadAll(envRefs, `${stamp}_env`);
      const graph = buildApiPrompt({ image1: name, refs, text: expandText, resolution: 0, seed: randomSeed(), lora: S.expandLora ? OUTPAINT_LORA : null });
      if (S.expandLora) graph[15].inputs.strength_model = S.expandLora;
      run.promptId = await queueGraph(graph);
      advanceJob('queue');
      renderJob();
      const { img } = await loadOutput(await waitForImage(run.promptId));
      current = stitchOriginal(img, snap);
    }
    if (mergeJob) {
      if (run.cancelled) throw new Error('Đã dừng.');
      run.promptId = null;
      startPass(passLabel(expand ? 2 : 1, merge.name));
      if (mergeJob.lora) await requireBlendLora();
      const plan = planShot(snap.w, snap.h, megapixels);
      const comp = composeMerge(current, mergeJob, snap.pads);
      const masks = mergeMasks(mergeJob, snap.w, snap.h, snap.pads);
      const clean = cropCanvas(comp, plan);
      const shot = await uploadCanvas(clean, `${stamp}_merge.png`);
      const mask = await uploadCanvas(modelMask(masks.area, plan), `${stamp}_merge_mask.png`);
      const outlined = await uploadCanvas(guideImage(clean, masks.marks, plan), `${stamp}_merge_outlined.png`);
      const graph = buildInpaintPrompt({ shot, outlined, mask, refs: [], text: mergeJob.text, seed: randomSeed() });
      if (mergeJob.lora) {
        graph[15] = { class_type: 'LoraLoaderModelOnly', inputs: { model: ['1', 0], lora_name: BLEND_LORA, strength_model: 1 } };
        graph[8].inputs.model = ['15', 0];
      }
      run.promptId = await queueGraph(graph);
      advanceJob('queue');
      renderJob();
      const { img } = await loadOutput(await waitForImage(run.promptId));
      current = pasteCrop(comp, img, plan, masks.area);
    }
    if (step) {
      if (run.cancelled) throw new Error('Đã dừng.');
      run.promptId = null;
      startPass(passLabel(total, step.name));
      const plan = planShot(snap.w, snap.h, megapixels);
      const clean = cropCanvas(current, plan);
      const shot = await uploadCanvas(clean, `${stamp}_shot.png`);
      const mask = await uploadCanvas(modelMask(step.area, plan), `${stamp}_mask.png`);
      const outlined = await uploadCanvas(guideImage(clean, step.marks, plan), `${stamp}_outlined.png`);
      const refs = await uploadAll(step.refs, `${stamp}_ref`);
      run.promptId = await queueGraph(buildInpaintPrompt({ shot, outlined, mask, refs, text: step.text, seed: randomSeed() }));
      advanceJob('queue');
      renderJob();
      const { img } = await loadOutput(await waitForImage(run.promptId));
      current = pasteCrop(current, img, plan, step.area);
    }
    const finished = await canvasToResult(current);
    const continued = await continueFrom(finished, snap, strokesAtStart);
    hideJob();
    setStatus(`Xong trong ${clock(Date.now() - job.startedAt)}.${continued ? ' Đang sửa tiếp trên ảnh này.' : ''}`);
  } catch (e) {
    failRun(e);
  } finally {
    endRun();
  }
}
function beginRun() {
  run.active = true;
  run.cancelled = false;
  run.outcome = 'done';
  run.promptId = null;
  job.startedAt = Date.now();
  job.cancelling = false;
  setStatus('');
  $('gen').textContent = 'Dừng';
  clearInterval(job.timer);
  job.timer = setInterval(renderJob, 1000);
  syncGen();
}
function failRun(e) {
  run.outcome = run.cancelled ? 'cancel' : 'error';
  if (run.cancelled) {
    hideJob();
    setStatus('Đã dừng.');
  } else {
    showJobError(e.message || String(e));
  }
}
function endRun() {
  run.active = false;
  run.promptId = null;
  if (run.outcome !== 'cancel') notifyFinished(run.outcome === 'done');
  syncGen();
  if (S.tab === 'rotate') syncRotateMode();
}
async function generateSheet() {
  const pass = S.passes.find(p => p.kind === 'sheet');
  const text = promptFor(pass);
  const references = Sheet.slots.filter(s => s.ref).map(s => imageToCanvas(s.ref));
  const [w, h] = $('sheetSize').value.split('x').map(Number);
  const stamp = Date.now().toString(36);
  beginRun();
  try {
    startPass('Char sheet');
    const names = [];
    for (let i = 0; i < references.length; i++) names.push(await uploadCanvas(references[i], `${stamp}_sheet${i + 1}.png`));
    run.promptId = await queueGraph(buildApiPrompt({ image1: names[0], refs: names.slice(1), text, resolution: 1024, seed: randomSeed(), lora: null, latent: { w, h } }));
    advanceJob('queue');
    renderJob();
    const { img, url } = await loadOutput(await waitForImage(run.promptId));
    addResult(img, url, 'Char sheet');
    if (S.tab === 'sheet') openViewer(img, url);
    hideJob();
    setStatus(`Xong trong ${clock(Date.now() - job.startedAt)}.`);
  } catch (e) {
    failRun(e);
  } finally {
    endRun();
  }
}
// Dummy theo trang LoRA QMA, Splat theo trang và sơ đồ của LoRA AnyAngle: image_1 là ảnh gốc, image_2 là ảnh render.
const ROTATE_RUN = {
  dummy: { lora: 'qwen-image-2.1-qma-camera-angles.safetensors', steps: 30, cfg: 3 },
  splat: { lora: 'QI2.1_AnyAngle.safetensors', steps: 20, cfg: 3 },
};
const ROTATE_REF_MP = 0.5;
async function generateRotate() {
  const pass = S.passes.find(p => p.kind === 'rotate');
  const text = promptFor(pass);
  const settings = ROTATE_RUN[Rotate.mode];
  const splat = Rotate.mode === 'splat';
  const main = splat ? Rotate.mainCanvas : fitMainForRotate(Rotate.slots[0].ref);
  const render = splat ? RotateView.capture(main.width, main.height) : null;
  const extras = Rotate.slots.slice(1).filter(s => s.ref).map(s => fitReference(s.ref, ROTATE_REF_MP));
  const stamp = Date.now().toString(36);
  beginRun();
  try {
    startPass(pass.name);
    const name = await uploadCanvas(main, `${stamp}_rotate.png`);
    const refs = await uploadAll(render ? [render, ...extras] : extras, `${stamp}_angle`);
    const graph = buildApiPrompt({ image1: name, refs, text, resolution: 0, seed: randomSeed(), lora: settings.lora });
    graph[8].inputs.steps = settings.steps;
    graph[8].inputs.cfg = settings.cfg;
    run.promptId = await queueGraph(graph);
    advanceJob('queue');
    renderJob();
    const { img, url } = await loadOutput(await waitForImage(run.promptId));
    addResult(img, url, 'Xoay');
    if (S.tab === 'rotate') openViewer(img, url);
    hideJob();
    setStatus(`Xong trong ${clock(Date.now() - job.startedAt)}.`);
  } catch (e) {
    failRun(e);
  } finally {
    endRun();
  }
}
function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text).then(() => true, () => false);
  const area = document.createElement('textarea');
  area.value = text;
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.append(area);
  area.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
  area.remove();
  return Promise.resolve(ok);
}
$('gen').addEventListener('click', generate);
$('jobClose').addEventListener('click', () => { $('job').hidden = true; });
$('viewerClose').addEventListener('click', () => { $('viewer').hidden = true; });
$('viewerUse').addEventListener('click', () => {
  if (!shown) return;
  $('viewer').hidden = true;
  if (S.tab !== 'edit') switchTab('edit');
  setBase(shown, false, true);
});
$('captureCancel').addEventListener('click', stopCapture);
$('tabEdit').addEventListener('click', () => switchTab('edit'));
$('tabSheet').addEventListener('click', () => switchTab('sheet'));
$('tabRotate').addEventListener('click', () => switchTab('rotate'));
$('tabSwap').addEventListener('click', () => switchTab('swap'));
$('tabUpscale').addEventListener('click', () => switchTab('upscale'));
$('tabChat').addEventListener('click', () => switchTab('chat'));

function sampleScene() {
  const c = document.createElement('canvas');
  c.width = 1344; c.height = 768;
  const g = c.getContext('2d');
  const sky = g.createLinearGradient(0, 0, 0, 560);
  sky.addColorStop(0, '#2C4A5E'); sky.addColorStop(0.55, '#7F9190'); sky.addColorStop(1, '#E8BC8E');
  g.fillStyle = sky; g.fillRect(0, 0, 1344, 768);
  const sun = g.createRadialGradient(880, 420, 10, 880, 420, 300);
  sun.addColorStop(0, 'rgba(255,226,180,.95)'); sun.addColorStop(1, 'rgba(255,226,180,0)');
  g.fillStyle = sun; g.fillRect(0, 0, 1344, 768);
  const ridges = [[440, 42, '#71848B', 0.7], [490, 58, '#526873', 1.3], [550, 66, '#37525C', 2.1], [620, 44, '#233A41', 3.4]];
  for (const [base, amp, color, seed] of ridges) {
    g.fillStyle = color;
    g.beginPath();
    g.moveTo(0, 768);
    for (let x = 0; x <= 1344; x += 8) {
      const y = base + amp * (Math.sin(x * 0.006 + seed) + 0.5 * Math.sin(x * 0.017 + seed * 2) + 0.22 * Math.sin(x * 0.051 + seed * 3));
      g.lineTo(x, y);
    }
    g.lineTo(1344, 768);
    g.closePath();
    g.fill();
  }
  g.fillStyle = '#26383E';
  g.fillRect(286, 412, 26, 120);
  g.beginPath(); g.moveTo(278, 412); g.lineTo(299, 372); g.lineTo(320, 412); g.closePath(); g.fill();
  const ground = g.createLinearGradient(0, 640, 0, 768);
  ground.addColorStop(0, '#1C2C30'); ground.addColorStop(1, '#10191B');
  g.fillStyle = ground;
  g.beginPath();
  g.moveTo(0, 768);
  for (let x = 0; x <= 1344; x += 16) g.lineTo(x, 672 + 16 * Math.sin(x * 0.008));
  g.lineTo(1344, 768);
  g.closePath();
  g.fill();
  return c;
}

buildPop();
bindLayerTools();
buildSheetPanel();
buildEnvBox();
buildRotatePanel();
buildSwapPanel();
buildUpscalePanel();
buildChatPanel();
renderTiles();
if (ON_COMFY) {
  clearBase();
  connectProgress();
  pollQueue();
  setInterval(() => { if (!socketOpen) pollQueue(); }, 10000);
} else {
  setBase(sampleScene(), true);
  const sample = sampleScene();
  Rotate.slots[0].ref = sample;
  Rotate.slots[0].refUrl = sample.toDataURL('image/jpeg', 0.8);
  syncRotate();
  const person = sampleSwapScene();
  Swap.slots[0].ref = person;
  Swap.slots[0].refUrl = person.toDataURL('image/jpeg', 0.8);
  syncSwap();
  const shot = sampleScene();
  Up.slots[0].ref = shot;
  Up.slots[0].refUrl = shot.toDataURL('image/jpeg', 0.8);
  syncUpscale();
  // Bản xem trước: kết quả giả để thấy phần so sánh, và hàng đợi giả.
  const big = upscaleTarget(shot.width, shot.height, Up.mp);
  const fake = newCanvas(big.w, big.h), fg = fake.getContext('2d');
  fg.imageSmoothingQuality = 'high';
  fg.drawImage(shot, 0, 0, big.w, big.h);
  const px = fg.getImageData(0, 0, big.w, big.h);
  for (let i = 0; i < px.data.length; i += 4) {
    const n = ((i * 2654435761) >>> 24) / 255 - 0.5;
    for (let k = 0; k < 3; k++) px.data[i + k] = Math.max(0, Math.min(255, px.data[i + k] + n * 22));
  }
  fg.putImageData(px, 0, 0);
  showUpscaleResult(shot, fake, fake.toDataURL('image/jpeg', 0.9));
  $('upNote').hidden = false;
  Q.remaining = 2;
  renderQueueState();
  $('qText').textContent += ' (giả lập)';
}
new ResizeObserver(() => {
  if (S.tab === 'rotate') { layoutRotateFrame(); RotateView.request(); }
  else if (S.tab === 'swap') drawSwap();
  else if (S.tab === 'upscale') drawUpscale();
  else if (!drag && !live) fit();
}).observe(stage);
</script>
</body>
</html>
"""
# <<< CONCEPT_IMAGE_EDIT_HTML


def _install_concept_image_edit_html() -> None:
    """Trang sửa ảnh cho nhóm concept, mở tại /extensions/vstone/concept_image_edit.html, cùng cơ chế với JS của node Huy."""
    try:
        import server
        web_root = server.PromptServer.instance.web_root
    except Exception:
        return  # không chạy trong ComfyUI
    path = os.path.join(web_root, "extensions", "vstone", "concept_image_edit.html")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as html_file:
            html_file.write(CONCEPT_IMAGE_EDIT_HTML)
    except OSError as e:
        print(f"[VStone] Không ghi được trang concept vào {path}: {e}", flush=True)


_install_concept_image_edit_html()


def _round_screenx_pad(size: int, before: int, after: int) -> tuple[int, int]:
    """Làm tròn tổng kích thước lên bội số 64, phần dư cộng vào cạnh đang nới."""
    total = size + before + after
    extra = math.ceil(total / SCREENX_PAINT_MULTIPLE) * SCREENX_PAINT_MULTIPLE - total
    return (before + extra, after) if before > 0 and after == 0 else (before, after + extra)


def _save_screenx_preview(frame: torch.Tensor) -> dict[str, Any]:
    """Lưu frame đầu vào thư mục temp để khung đồ họa trên node hiển thị."""
    import folder_paths
    from PIL import Image

    pixels = (frame[:, :, :3].cpu().numpy().clip(0.0, 1.0) * 255.0).astype("uint8")
    preview = Image.fromarray(pixels)
    width, height = preview.size
    preview.thumbnail((1024, 1024))
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"huy_screenx_preview_{time.time_ns()}.png"
    preview.save(os.path.join(temp_dir, filename), compress_level=1)
    return {"filename": filename, "subfolder": "", "type": "temp", "width": width, "height": height}


class Huy_screenx_outpain_mask:
    """
    Chuẩn bị vùng paint, không gen gì: nới khung quanh ảnh/video và trả mask vùng cần vẽ.
    Ảnh gốc giữ nguyên, vùng mới tô xám 0.5. Mask = 1 ở vùng cần vẽ, = 0 ở ảnh gốc.
    Khung đồ họa trên node (HUY_SCREENX_EDITOR_JS) cho kéo ảnh và kéo mép vùng xám, rồi ghi số vào left, top, right, bottom.
    Mask giống nhau ở mọi frame nên chỉ tính 1 lần rồi nhân đủ số frame, vì blend của LTX cắt video theo số frame của mask.
    Tổng chiều rộng và chiều cao tự làm tròn lên bội số 64.
    """

    @classmethod
    def INPUT_TYPES(cls):
        pad_input = {"min": 0, "max": 8192, "step": SCREENX_PAINT_MULTIPLE}
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 1024, **pad_input}),
                "top": ("INT", {"default": 0, **pad_input}),
                "right": ("INT", {"default": 0, **pad_input}),
                "bottom": ("INT", {"default": 0, **pad_input}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "prepare"
    OUTPUT_NODE = True  # để chạy riêng node này, hiện ảnh thật trên khung đồ họa mà không phải gen cả workflow
    CATEGORY = "stoneV"

    def prepare(
        self,
        image: torch.Tensor,
        left: int = 1024,
        top: int = 0,
        right: int = 0,
        bottom: int = 0,
    ):
        frames, height, width, channels = image.shape
        left, right = _round_screenx_pad(width, left, right)
        top, bottom = _round_screenx_pad(height, top, bottom)
        canvas_h = top + height + bottom
        canvas_w = left + width + right

        canvas = image.new_full((frames, canvas_h, canvas_w, channels), 0.5)
        canvas[:, top : top + height, left : left + width, :] = image
        mask = image.new_ones((canvas_h, canvas_w))
        mask[top : top + height, left : left + width] = 0.0
        return {
            "ui": {"huy_preview": [_save_screenx_preview(image[0])]},
            "result": (canvas, mask.repeat(frames, 1, 1)),
        }


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
    "Huy_screenx_outpain_mask": Huy_screenx_outpain_mask,
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
    # Không đăng ký node trùng tên với ComfyUI-LTXVideo / ComfyUI-KJNodes: hot-reload sẽ đè lên node của pack gốc.
}

VSTONE_NODE_DISPLAY_NAME_MAPPINGS = {
    "stoneV_LTX25_Outpaint_Loader": "stoneV LTX-2.5 Outpaint All-in-One Model Loader (Auto-Download)",
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
    "Huy_screenx_outpain_mask": "Huy_screenx outpain mask",
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
