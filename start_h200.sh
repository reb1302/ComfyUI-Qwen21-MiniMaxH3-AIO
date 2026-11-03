#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/main.py" ]; then
    COMFY_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../ComfyUI/main.py" ]; then
    COMFY_DIR="$(cd "$SCRIPT_DIR/../ComfyUI" && pwd)"
elif [ -f "$SCRIPT_DIR/../../main.py" ]; then
    COMFY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "[ERROR] Khong tim thay thu muc ComfyUI! Hay chay: bash install_h200_all_in_one.sh truoc."
    read -rp "Nhan Enter de dong..."
    exit 1
fi

PORT="${1:-8188}"
source "$COMFY_DIR/venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 1. Tu dong git pull neu dang dung trong repo ComfyUI-Qwen21-MiniMaxH3-AIO
REPO_DIR=""
if [ -d "$SCRIPT_DIR/.git" ]; then
    REPO_DIR="$SCRIPT_DIR"
elif [ -d "$COMFY_DIR/../ComfyUI-Qwen21-MiniMaxH3-AIO/.git" ]; then
    REPO_DIR="$(cd "$COMFY_DIR/../ComfyUI-Qwen21-MiniMaxH3-AIO" && pwd)"
fi

if [ -n "$REPO_DIR" ]; then
    echo "[SYNC] Dang git pull code moi nhat tai $REPO_DIR ..."
    git -C "$REPO_DIR" pull --ff-only || true
fi

# 2. Copy truc tiep node_helper_vstone.py vao CA 3 vi tri trong ComfyUI/custom_nodes/
echo "[SYNC] Dang cap nhat VStone Nodes & Workflows vao $COMFY_DIR/custom_nodes/ ..."
mkdir -p "$COMFY_DIR/custom_nodes/vstone"
mkdir -p "$COMFY_DIR/user/default/workflows"
mkdir -p "$COMFY_DIR/models/loras"

if [ -n "$REPO_DIR" ] && [ -f "$REPO_DIR/vstone/node_helper_vstone.py" ]; then
    cp -f "$REPO_DIR/vstone/node_helper_vstone.py" "$COMFY_DIR/custom_nodes/vstone/node_helper_vstone.py"
    cp -f "$REPO_DIR/vstone/__init__.py" "$COMFY_DIR/custom_nodes/vstone/__init__.py"
    cp -f "$REPO_DIR/vstone/node_helper_vstone.py" "$COMFY_DIR/custom_nodes/node_helper_vstone.py"
    if [ -d "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO" ]; then
        mkdir -p "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone"
        cp -f "$REPO_DIR/vstone/node_helper_vstone.py" "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone/node_helper_vstone.py"
        cp -f "$REPO_DIR/vstone/__init__.py" "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone/__init__.py"
    fi
    if [ -d "$REPO_DIR/workflows" ]; then
        cp -rf "$REPO_DIR/workflows/"* "$COMFY_DIR/user/default/workflows/" 2>/dev/null || true
    fi
    if [ "$SCRIPT_DIR/start_h200.sh" != "$COMFY_DIR/start_h200.sh" ]; then
        cp -f "$REPO_DIR/start_h200.sh" "$COMFY_DIR/start_h200.sh" 2>/dev/null || true
        chmod +x "$COMFY_DIR/start_h200.sh" 2>/dev/null || true
    fi
else
    SHA=$(git ls-remote https://github.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO.git refs/heads/main 2>/dev/null | awk '{print $1}')
    REF="${SHA:-main}"
    curl -4 -sL "https://raw.githubusercontent.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/$REF/vstone/node_helper_vstone.py" -o "$COMFY_DIR/custom_nodes/vstone/node_helper_vstone.py" || true
    curl -4 -sL "https://raw.githubusercontent.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/$REF/vstone/__init__.py" -o "$COMFY_DIR/custom_nodes/vstone/__init__.py" || true
    cp -f "$COMFY_DIR/custom_nodes/vstone/node_helper_vstone.py" "$COMFY_DIR/custom_nodes/node_helper_vstone.py" 2>/dev/null || true
    if [ -d "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone" ]; then
        cp -f "$COMFY_DIR/custom_nodes/vstone/node_helper_vstone.py" "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone/node_helper_vstone.py" 2>/dev/null || true
    fi
fi

# 3. Tu dong tai cac LoRA tu Laptop (GitHub Release v1.0-loras) ve ComfyUI/models/loras/ neu chua co
sync_lora() {
    local fname="$1"
    local url="$2"
    local dst="$COMFY_DIR/models/loras/$fname"
    if [ ! -f "$dst" ]; then
        echo "[LORA SYNC] Dang tai LoRA tu Laptop: $fname ..."
        curl -4 -L "$url" -o "$dst" || true
    fi
}

sync_lora "High quality.safetensors" "https://github.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/releases/download/v1.0-loras/High_quality.safetensors"
sync_lora "qwen-image-2.1-outpaint-v2.safetensors" "https://github.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/releases/download/v1.0-loras/qwen-image-2.1-outpaint-v2.safetensors"
sync_lora "qwen_image_2_1_360_panorama_outpaint_lora_v0_1.safetensors" "https://github.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/releases/download/v1.0-loras/qwen_image_2_1_360_panorama_outpaint_lora_v0_1.safetensors"

# 4. Tu dong kiem tra neu sageattention da cai thanh cong thi bat --use-sage-attention
ATTN_FLAG=""
if python -c "import sageattention" &> /dev/null; then
    echo "[INFO] Phat hien SageAttention da san sang -> Bat --use-sage-attention"
    ATTN_FLAG="--use-sage-attention"
else
    echo "[INFO] Su dung PyTorch Native FlashAttention/SDPA"
fi

echo "=============================================================================="
echo "  Khoi dong ComfyUI tren NVIDIA H200 (Port: $PORT, Mode: --highvram $ATTN_FLAG)"
echo "  Da dong bo day du VStone ScreenX Nodes + Laptop LoRAs!"
echo "=============================================================================="
python "$COMFY_DIR/main.py" --listen 0.0.0.0 --port "$PORT" --highvram $ATTN_FLAG
read -rp "Nhan Enter de dong cua so..."
