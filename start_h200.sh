#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/main.py" ]; then
    COMFY_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../ComfyUI/main.py" ]; then
    COMFY_DIR="$(cd "$SCRIPT_DIR/../ComfyUI" && pwd)"
elif [ -f "$SCRIPT_DIR/../../main.py" ]; then
    COMFY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "[ERROR] Khong tim thay thu muc ComfyUI! Hay chay: bash install_h200_all_in_one.sh truoc."
    exit 1
fi

PORT="${1:-8188}"
source "$COMFY_DIR/venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=============================================================================="
echo "  Khoi dong ComfyUI tren NVIDIA H200 (Port: $PORT, Mode: --highvram)"
echo "  Duong dan: $COMFY_DIR"
echo "=============================================================================="
exec python "$COMFY_DIR/main.py" --listen 0.0.0.0 --port "$PORT" --highvram
