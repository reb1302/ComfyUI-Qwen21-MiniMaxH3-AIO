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

# Tu dong kiem tra neu sageattention da cai thanh cong thi bat --use-sage-attention, neu chua thi chay PyTorch SDPA de khong bi vang
ATTN_FLAG=""
if python -c "import sageattention" &> /dev/null; then
    echo "[INFO] Phat hien SageAttention da san sang -> Bat --use-sage-attention"
    ATTN_FLAG="--use-sage-attention"
else
    echo "[INFO] SageAttention chua duoc bien dich -> Su dung PyTorch Native FlashAttention/SDPA (khong bi vang)"
fi

echo "=============================================================================="
echo "  Khoi dong ComfyUI tren NVIDIA H200 (Port: $PORT, Mode: --highvram $ATTN_FLAG)"
echo "  Duong dan: $COMFY_DIR"
echo "=============================================================================="
python "$COMFY_DIR/main.py" --listen 0.0.0.0 --port "$PORT" --highvram $ATTN_FLAG
read -rp "Nhan Enter de dong cua so..."
