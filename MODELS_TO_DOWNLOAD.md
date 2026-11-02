# Danh Sách Cài Đặt Tinh Gọn Cho IT (ComfyUI: Qwen-Image-2.1 & MiniMax-H3)

## 1. Yêu Cầu Môi Trường (Phần Mềm & Phần Cứng)
* **GPU:** NVIDIA RTX 4090 / 5090 / A100 / H100 (Tối thiểu **24GB VRAM**, khuyến nghị **32GB+ VRAM**; RAM hệ thống **64GB**; Ổ cứng trống **~120GB SSD**).
* **Driver & CUDA:** NVIDIA Driver mới nhất hỗ trợ **CUDA 13.0 (`cu130`)**.
* **Bản cài ComfyUI (v0.43.0+):**
  * **Windows Portable (tích hợp sẵn Python + PyTorch cu130):**
    `https://github.com/Comfy-Org/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z` *(1.90 GB)*
  * **Linux (nếu cài trên Server):**
    ```bash
    git clone https://github.com/Comfy-Org/ComfyUI.git
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
    pip install -r ComfyUI/requirements.txt
    ```

---

## 2. Danh Sách Model Tinh Gọn Nhất Cần Tải (Tổng cộng: 12 Files ~ 86.26 GB)

### A. Bộ Qwen-Image-2.1 (Image-to-Image / Multi-Image Edit) — ~25.77 GB
*Kho gốc: `https://huggingface.co/Comfy-Org/Qwen-Image-2.1`*

| # | Thư mục đích (`ComfyUI/models/...`) | Tên file | Dung lượng | Link tải trực tiếp (Direct URL) |
| :-: | :--- | :--- | :---: | :--- |
| 1 | `models/diffusion_models/` | `qwen_image_2.1_int8_convrot.safetensors` | 6.76 GB | https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors |
| 2 | `models/text_encoders/` | `qwen3vl_8b_int8_convrot.safetensors` | 8.71 GB | https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors |
| 3 | `models/text_encoders/` | `qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors` | 8.82 GB | https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors |
| 4 | `models/vae/` | `qwen_image_2.1_vae_bf16.safetensors` | 0.63 GB | https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors |
| 5 | `models/loras/` | `qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors` | 0.85 GB | https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors |

---

### B. Bộ MiniMax-H3 (Image-to-Video & Reference-to-Video + Audio) — ~60.49 GB
*Kho gốc: `https://huggingface.co/Comfy-Org/MiniMax-H3`*

| # | Thư mục đích (`ComfyUI/models/...`) | Tên file | Dung lượng | Link tải trực tiếp (Direct URL) |
| :-: | :--- | :--- | :---: | :--- |
| 6 | `models/diffusion_models/` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.53 GB | https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors |
| 7 | `models/diffusion_models/` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.53 GB | https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors |
| 8 | `models/text_encoders/` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.61 GB | https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors |
| 9 | `models/vae/` | `minimax_h3_video_vae_int8_convrot.safetensors` | 2.62 GB | https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors |
| 10 | `models/vae/` | `minimax_h3_audio_vae_fp32.safetensors` | 0.56 GB | https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors |
| 11 | `models/loras/` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 1.82 GB | https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors |
| 12 | `models/loras/` | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1.82 GB | https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors |

---

## 3. Workflow JSON Chuẩn (Tải đặt vào `ComfyUI/user/default/workflows/`)
1. **Qwen-Image-2.1 (Image Edit / I2I):**
   `https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_qwen_image_2_1_image_edit.json`
2. **MiniMax-H3 (Image-to-Video):**
   `https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_i2v.json`
3. **MiniMax-H3 (Reference-to-Video):**
   `https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_r2v.json`
