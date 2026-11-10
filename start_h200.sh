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

# 1. Tu dong git fetch + reset --hard neu dang dung trong repo ComfyUI-Qwen21-MiniMaxH3-AIO
#    (Tranh loi git pull bi ket khi file local bi 0KB hoac dirty do auto-sync)
REPO_DIR=""
if [ -d "$SCRIPT_DIR/.git" ]; then
    REPO_DIR="$SCRIPT_DIR"
elif [ -d "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/.git" ]; then
    REPO_DIR="$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO"
elif [ -d "$COMFY_DIR/../ComfyUI-Qwen21-MiniMaxH3-AIO/.git" ]; then
    REPO_DIR="$(cd "$COMFY_DIR/../ComfyUI-Qwen21-MiniMaxH3-AIO" && pwd)"
fi

if [ -n "$REPO_DIR" ]; then
    echo "[SYNC] Dang cap nhat code moi nhat tai $REPO_DIR ..."
    git -C "$REPO_DIR" fetch origin main && git -C "$REPO_DIR" reset --hard origin/main || true
fi

# 2. Cap nhat an toan (Atomic + chong 0KB) vao ComfyUI/custom_nodes/
echo "[SYNC] Dang cap nhat VStone Nodes & Workflows vao $COMFY_DIR/custom_nodes/ ..."
mkdir -p "$COMFY_DIR/custom_nodes/vstone"
mkdir -p "$COMFY_DIR/user/default/workflows"
mkdir -p "$COMFY_DIR/models/loras"

# Xoa cac file .tmp rac hoac ban copy trung lap o goc custom_nodes/ gay xung dot
rm -f "$COMFY_DIR/custom_nodes/node_helper_vstone.py" "$COMFY_DIR/custom_nodes/"*.tmp "$COMFY_DIR/custom_nodes/vstone/"*.tmp 2>/dev/null || true

safe_copy_if_valid() {
    local src="$1"
    local dst="$2"
    local min_bytes="${3:-50}"
    if [ -f "$src" ]; then
        local sz
        sz=$(wc -c < "$src" 2>/dev/null || echo 0)
        if [ "$sz" -ge "$min_bytes" ]; then
            local real_src real_dst
            real_src=$(readlink -f "$src" 2>/dev/null || echo "$src")
            real_dst=$(readlink -f "$dst" 2>/dev/null || echo "$dst")
            if [ "$real_src" != "$real_dst" ]; then
                cp -f "$src" "$dst.tmp.$$" && mv -f "$dst.tmp.$$" "$dst"
            fi
            return 0
        fi
    fi
    return 1
}

safe_curl_to() {
    local url="$1"
    local dst="$2"
    local min_bytes="${3:-50}"
    local tmp="$dst.curl.$$.tmp"
    if curl -4 -fsSL "$url" -o "$tmp" 2>/dev/null; then
        local sz
        sz=$(wc -c < "$tmp" 2>/dev/null || echo 0)
        if [ "$sz" -ge "$min_bytes" ]; then
            mv -f "$tmp" "$dst"
            return 0
        fi
    fi
    rm -f "$tmp" 2>/dev/null || true
    return 1
}

if [ -n "$REPO_DIR" ] && safe_copy_if_valid "$REPO_DIR/vstone/node_helper_vstone.py" "$COMFY_DIR/custom_nodes/vstone/node_helper_vstone.py" 10000; then
    safe_copy_if_valid "$REPO_DIR/vstone/__init__.py" "$COMFY_DIR/custom_nodes/vstone/__init__.py" 50 || true
    if [ -d "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO" ]; then
        mkdir -p "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone"
        safe_copy_if_valid "$REPO_DIR/vstone/node_helper_vstone.py" "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone/node_helper_vstone.py" 10000 || true
        safe_copy_if_valid "$REPO_DIR/vstone/__init__.py" "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone/__init__.py" 50 || true
        safe_copy_if_valid "$REPO_DIR/__init__.py" "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/__init__.py" 100 || true
    fi
    if [ -d "$REPO_DIR/workflows" ]; then
        cp -rf "$REPO_DIR/workflows/"* "$COMFY_DIR/user/default/workflows/" 2>/dev/null || true
    fi
    if [ -d "$REPO_DIR/ComfyUI-Mockin-Node" ]; then
        mkdir -p "$COMFY_DIR/custom_nodes/ComfyUI-Mockin-Node/web"
        cp -rf "$REPO_DIR/ComfyUI-Mockin-Node/"* "$COMFY_DIR/custom_nodes/ComfyUI-Mockin-Node/" 2>/dev/null || true
    fi
    if [ -d "$REPO_DIR/ComfyUI-Nukomfy-Suite" ]; then
        mkdir -p "$COMFY_DIR/custom_nodes/ComfyUI-Nukomfy-Suite"
        cp -rf "$REPO_DIR/ComfyUI-Nukomfy-Suite/"* "$COMFY_DIR/custom_nodes/ComfyUI-Nukomfy-Suite/" 2>/dev/null || true
    fi
    if [ "$SCRIPT_DIR/start_h200.sh" != "$COMFY_DIR/start_h200.sh" ]; then
        safe_copy_if_valid "$REPO_DIR/start_h200.sh" "$COMFY_DIR/start_h200.sh" 500 || true
        chmod +x "$COMFY_DIR/start_h200.sh" 2>/dev/null || true
    fi
else
    SHA=$(git ls-remote https://github.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO.git refs/heads/main 2>/dev/null | awk '{print $1}')
    REF="${SHA:-main}"
    safe_curl_to "https://raw.githubusercontent.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/$REF/vstone/node_helper_vstone.py" "$COMFY_DIR/custom_nodes/vstone/node_helper_vstone.py" 10000 || true
    safe_curl_to "https://raw.githubusercontent.com/reb1302/ComfyUI-Qwen21-MiniMaxH3-AIO/$REF/vstone/__init__.py" "$COMFY_DIR/custom_nodes/vstone/__init__.py" 50 || true
    if [ -d "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO" ]; then
        mkdir -p "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone"
        safe_curl_to "https://raw.githubusercontent.com/reb1302/ComfyUI-Qwen21-MiniMaxH3_AIO/$REF/__init__.py" "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/__init__.py" 100 || true
        safe_copy_if_valid "$COMFY_DIR/custom_nodes/vstone/node_helper_vstone.py" "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone/node_helper_vstone.py" 10000 || true
        safe_copy_if_valid "$COMFY_DIR/custom_nodes/vstone/__init__.py" "$COMFY_DIR/custom_nodes/ComfyUI_Qwen21_MiniMaxH3_AIO/vstone/__init__.py" 50 || true
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
