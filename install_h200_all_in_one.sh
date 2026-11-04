#!/usr/bin/env bash
# ==============================================================================
# ALL-IN-ONE INSTALLER FROM A TO Z FOR NVIDIA H200 (LINUX)
# ComfyUI (v0.43.0+ cu130) + Qwen-Image-2.1 (I2I/Edit) + MiniMax-H3 (I2V/R2V)
# ==============================================================================
# Cach dung tren server Linux H200:
#   chmod +x install_h200_all_in_one.sh
#   ./install_h200_all_in_one.sh            # Cai dat A-Z + tai 12 model chuan (86.26 GB)
#   ./install_h200_all_in_one.sh --bf16     # Tai them ban Full BF16 cho H200 141GB VRAM
#   ./install_h200_all_in_one.sh --run      # Cai dat xong tu dong bat ComfyUI tren port 8188
# ==============================================================================

set -euo pipefail

DOWNLOAD_BF16=false
AUTO_RUN=false
PORT=8188

for arg in "$@"; do
    case "$arg" in
        --bf16) DOWNLOAD_BF16=true ;;
        --run)  AUTO_RUN=true ;;
        --port=*) PORT="${arg#*=}" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_NAME="$(basename "$(dirname "$SCRIPT_DIR")")"

# Xac dinh vi tri ComfyUI (neu script nam trong custom_nodes hay nam doc lap)
if [ "$PARENT_NAME" = "custom_nodes" ]; then
    COMFY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    COMFY_DIR="$(dirname "$SCRIPT_DIR")/ComfyUI"
fi

echo "=============================================================================="
echo "  [AIO-H200] BAT DAU CAI DAT TU A-Z CHO NVIDIA H200 (LINUX)"
echo "  - Thu muc goi AIO : $SCRIPT_DIR"
echo "  - Thu muc ComfyUI : $COMFY_DIR"
echo "  - Tai them BF16   : $DOWNLOAD_BF16"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# BUOC 1: Cai dat cac goi he thong Linux (aria2, git, ffmpeg, python3-venv)
# ------------------------------------------------------------------------------
echo -e "\n[1/6] Kiem tra va cai dat cong cu he thong (aria2, git, ffmpeg)..."
if command -v apt-get &> /dev/null; then
    SUDO=""
    if [ "$(id -u)" -ne 0 ] && command -v sudo &> /dev/null; then
        SUDO="sudo"
    fi
    $SUDO apt-get update -y
    $SUDO apt-get install -y aria2 git ffmpeg curl wget build-essential python3 python3-dev python3-venv python3-pip libgl1 libglib2.0-0
elif command -v dnf &> /dev/null; then
     dnf install -y aria2 git ffmpeg curl wget python3 python3-pip
fi

# ------------------------------------------------------------------------------
# BUOC 2: Clone ComfyUI (v0.43.0+) & tao Python Virtual Environment + PyTorch cu130
# ------------------------------------------------------------------------------
echo -e "\n[2/6] Cai dat ComfyUI & PyTorch CUDA 13.0 (cu130) toi uu cho H200..."
if [ ! -d "$COMFY_DIR/.git" ]; then
    git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFY_DIR"
else
    echo "ComfyUI da ton tai, cap nhat phien ban moi nhat..."
    git -C "$COMFY_DIR" pull --ff-only || true
fi

VENV_DIR="$COMFY_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
# Cai PyTorch CUDA 13.0 (cu130) ho tro day du int8_convrot & Hopper H200 (sm_90a)
pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
pip install -r "$COMFY_DIR/requirements.txt"
# Cai them SageAttention & Triton tang toc suy luan tren H200
pip install triton sageattention huggingface_hub[hf_transfer] || true

# ------------------------------------------------------------------------------
# BUOC 3: Cai dat ComfyUI-Manager & Gắn Node All-in-One vao custom_nodes
# ------------------------------------------------------------------------------
echo -e "\n[3/6] Cai dat ComfyUI-Manager & Node ComfyUI_Qwen21_MiniMaxH3_AIO..."
mkdir -p "$COMFY_DIR/custom_nodes"

if [ ! -d "$COMFY_DIR/custom_nodes/ComfyUI-Manager" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git "$COMFY_DIR/custom_nodes/ComfyUI-Manager"
    pip install -r "$COMFY_DIR/custom_nodes/ComfyUI-Manager/requirements.txt" || true
fi

TARGET_NODE_DIR="$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO"
if [ "$SCRIPT_DIR" != "$TARGET_NODE_DIR" ]; then
    rm -rf "$TARGET_NODE_DIR"
    cp -r "$SCRIPT_DIR" "$TARGET_NODE_DIR"
fi

# Cai dat thu muc custom node doc lap custom_nodes/vstone
if [ -d "$SCRIPT_DIR/vstone" ]; then
    rm -rf "$COMFY_DIR/custom_nodes/vstone"
    cp -r "$SCRIPT_DIR/vstone" "$COMFY_DIR/custom_nodes/vstone"
fi

# ------------------------------------------------------------------------------
# BUOC 4: Chep san 12 Workflow chuan & 10 Embeddings vao ComfyUI
# ------------------------------------------------------------------------------
echo -e "\n[4/6] Tich hop 12 Workflow chuan & 10 MiniMax-H3 Embeddings..."
mkdir -p "$COMFY_DIR/user/default/workflows"
mkdir -p "$COMFY_DIR/models/embeddings"
mkdir -p "$COMFY_DIR/models/diffusion_models"
mkdir -p "$COMFY_DIR/models/text_encoders"
mkdir -p "$COMFY_DIR/models/vae"
mkdir -p "$COMFY_DIR/models/loras"
mkdir -p "$COMFY_DIR/models/model_patches"

if [ -d "$SCRIPT_DIR/workflows" ]; then
    cp -r "$SCRIPT_DIR/workflows/"* "$COMFY_DIR/user/default/workflows/"
fi
if [ -d "$SCRIPT_DIR/embeddings" ]; then
    cp -r "$SCRIPT_DIR/embeddings/"*.safetensors "$COMFY_DIR/models/embeddings/" || true
fi

# ------------------------------------------------------------------------------
# BUOC 5: Tai tu dong 12 Model cot loi (86.26 GB) bang aria2c 16 luong
# ------------------------------------------------------------------------------
echo -e "\n[5/6] Tai bo Checkpoint Qwen-Image-2.1 & MiniMax-H3 (aria2c 16 connections)..."

dl_model() {
    local url="$1"
    local subdir="$2"
    local filename="$3"
    local outdir="$COMFY_DIR/models/$subdir"
    mkdir -p "$outdir"
    echo "  -> Dang tai: models/$subdir/$filename"
    aria2c --check-certificate=false --disable-ipv6=true --file-allocation=none \
           -c -x 16 -s 16 -k 1M --console-log-level=warn --summary-interval=10 \
           -d "$outdir" -o "$filename" "$url"
}

HF_QWEN="https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"
HF_MINIMAX="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"

# --- A. Qwen-Image-2.1 Core (5 files - 25.77 GB) ---
dl_model "$HF_QWEN/diffusion_models/qwen_image_2.1_int8_convrot.safetensors"       "diffusion_models" "qwen_image_2.1_int8_convrot.safetensors"
dl_model "$HF_QWEN/text_encoders/qwen3vl_8b_int8_convrot.safetensors"              "text_encoders"    "qwen3vl_8b_int8_convrot.safetensors"
dl_model "$HF_QWEN/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors" "text_encoders" "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"
dl_model "$HF_QWEN/vae/qwen_image_2.1_vae_bf16.safetensors"                        "vae"              "qwen_image_2.1_vae_bf16.safetensors"
dl_model "$HF_QWEN/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"  "loras"            "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"
dl_model "$HF_QWEN/model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors" "model_patches" "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors"

# --- B. MiniMax-H3 Core (7 files - 60.49 GB) ---
dl_model "$HF_MINIMAX/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" "diffusion_models" "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
dl_model "$HF_MINIMAX/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" "diffusion_models" "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
dl_model "$HF_MINIMAX/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"        "text_encoders"    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
dl_model "$HF_MINIMAX/vae/minimax_h3_video_vae_int8_convrot.safetensors"                 "vae"              "minimax_h3_video_vae_int8_convrot.safetensors"
dl_model "$HF_MINIMAX/vae/minimax_h3_audio_vae_fp32.safetensors"                         "vae"              "minimax_h3_audio_vae_fp32.safetensors"
dl_model "$HF_MINIMAX/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"   "loras"            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
dl_model "$HF_MINIMAX/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"  "loras"            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

# --- C. Tuy chon H200 Full BF16 (--bf16) ---
if [ "$DOWNLOAD_BF16" = true ]; then
    echo -e "\n[5b/6] Tai them bo Full Precision BF16 danh rieng cho H200 (141GB VRAM)..."
    dl_model "$HF_QWEN/diffusion_models/qwen_image_2.1_bf16.safetensors"                  "diffusion_models" "qwen_image_2.1_bf16.safetensors"
    dl_model "$HF_QWEN/text_encoders/qwen3vl_8b_bf16.safetensors"                         "text_encoders"    "qwen3vl_8b_bf16.safetensors"
    dl_model "$HF_MINIMAX/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"      "diffusion_models" "minimax_h3_fl2va_pruned_bf16.safetensors"
    dl_model "$HF_MINIMAX/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors"     "diffusion_models" "minimax_h3_ref2va_pruned_bf16.safetensors"
    dl_model "$HF_MINIMAX/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"  "text_encoders"    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
fi

# ------------------------------------------------------------------------------
# BUOC 6: Tao script khoi chay start_h200.sh (bat --highvram + auto-sync an toan)
# ------------------------------------------------------------------------------
echo -e "\n[6/6] Cai dat script khoi chay $COMFY_DIR/start_h200.sh..."
if [ -f "$SCRIPT_DIR/start_h200.sh" ] && [ "$SCRIPT_DIR/start_h200.sh" != "$COMFY_DIR/start_h200.sh" ]; then
    cp -f "$SCRIPT_DIR/start_h200.sh" "$COMFY_DIR/start_h200.sh"
fi
chmod +x "$COMFY_DIR/start_h200.sh"

echo "=============================================================================="
echo "  [HOAN TAT A-Z] Moi truong ComfyUI + Qwen-Image-2.1 + MiniMax-H3 da san sang!"
echo "  De chay server tren H200, go lenh:"
echo "      $COMFY_DIR/start_h200.sh $PORT"
echo "=============================================================================="

if [ "$AUTO_RUN" = true ]; then
    exec "$COMFY_DIR/start_h200.sh" "$PORT"
fi
