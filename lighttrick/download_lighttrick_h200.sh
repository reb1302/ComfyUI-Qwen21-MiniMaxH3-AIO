#!/usr/bin/env bash
# ==============================================================================
# Lightricks LTX-2.5 + Creative Lab (15 IC-LoRAs & LoRAs) All-in-One Installer
# Dành cho Server H200 Linux & ComfyUI VStone Library
# ==============================================================================
# Cách chạy trên H200:
#   bash download_lighttrick_h200.sh [ĐƯỜNG_DẪN_COMFYUI] [HF_TOKEN_TÙY_CHỌN]
# Ví dụ:
#   bash download_lighttrick_h200.sh /workspace/ComfyUI
#   bash download_lighttrick_h200.sh /workspace/ComfyUI hf_xxxxxxxxxxxxxxxxx
# ==============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY_ROOT="${1:-}"
INPUT_HF_TOKEN="${2:-${HF_TOKEN:-}}"

if [ -z "$COMFY_ROOT" ]; then
    for cand in "/workspace/ComfyUI" "/root/ComfyUI" "/opt/ComfyUI" "$(cd "$SCRIPT_DIR/.." && pwd)"; do
        if [ -d "$cand/models" ] && [ -d "$cand/custom_nodes" ]; then
            COMFY_ROOT="$cand"
            break
        fi
    done
fi

if [ -z "$COMFY_ROOT" ] || [ ! -d "$COMFY_ROOT" ]; then
    COMFY_ROOT="/workspace/ComfyUI"
    mkdir -p "$COMFY_ROOT/models" "$COMFY_ROOT/custom_nodes"
fi

echo "=============================================================================="
echo "🚀 [Lightricks H200 Installer] Thư mục ComfyUI: $COMFY_ROOT"
echo "=============================================================================="

# 1. Lưu HF_TOKEN nếu có
if [ -n "$INPUT_HF_TOKEN" ]; then
    export HF_TOKEN="$INPUT_HF_TOKEN"
    mkdir -p "$HOME/.cache/huggingface"
    echo -n "$INPUT_HF_TOKEN" > "$HOME/.cache/huggingface/token"
    echo "🔑 Đã cấu hình HF_TOKEN để tải trực tiếp từ các repo Gated của Lightricks."
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" | tr -d '[:space:]')"
    echo "🔑 Đã tìm thấy HF_TOKEN trong ~/.cache/huggingface/token."
else
    export HF_TOKEN=""
    echo "ℹ️ Không có HF_TOKEN -> Sử dụng 100% Mirror Ungated (HTTP 200) tự động!"
fi

# 2. Cài đặt aria2c nếu chưa có
if ! command -v aria2c >/dev/null 2>&1; then
    echo "📦 Đang cài đặt aria2c..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -y && apt-get install -y aria2 git curl
    fi
fi

# 3. Cài đặt tất cả Custom Node phụ thuộc cho Lightricks LTX-2.5
CUSTOM_NODES_DIR="$COMFY_ROOT/custom_nodes"
mkdir -p "$CUSTOM_NODES_DIR"

install_or_update_node() {
    local dir_name="$1"
    local repo_url="$2"
    local branch="$3"
    local check_file="${4:-__init__.py}"
    local target_dir="$CUSTOM_NODES_DIR/$dir_name"

    echo "🔧 [Custom Node] Cài đặt bản gốc 100% $dir_name ($repo_url)..."
    if [ ! -f "$target_dir/$check_file" ]; then
        rm -rf "$target_dir"
    fi
    if [ -d "$target_dir/.git" ]; then
        GIT_LFS_SKIP_SMUDGE=1 git -C "$target_dir" -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false reset --hard HEAD || true
        GIT_LFS_SKIP_SMUDGE=1 git -C "$target_dir" -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false pull --ff-only || true
    else
        rm -rf "$target_dir"
        GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false clone --depth 1 -b "$branch" "$repo_url" "$target_dir" || \
        GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false clone --depth 1 "$repo_url" "$target_dir"
    fi

    if [ -f "$target_dir/requirements.txt" ]; then
        echo "📦 [Pip] Cài đặt phụ thuộc Python cho $dir_name..."
        grep -vE '^(torch|torchvision|torchaudio|nvidia-vfx|amd-adlx|pytest)' "$target_dir/requirements.txt" > "/tmp/${dir_name}_req.txt" || true
        python3 -m pip install --prefer-binary -r "/tmp/${dir_name}_req.txt" || true
    fi
}

python3 -m pip install --prefer-binary "colour-science>=0.4.4" "OpenImageIO>=3.1.0" "fileseq>=2.0.0" einops kornia diffusers color-matcher || true

install_or_update_node "ComfyUI-LTXVideo" "https://github.com/Lightricks/ComfyUI-LTXVideo.git" "master" "nodes_registry.py"
install_or_update_node "ComfyUI-KJNodes" "https://github.com/kijai/ComfyUI-KJNodes.git" "main" "__init__.py"
install_or_update_node "ComfyUI-Video-Depth-Anything" "https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git" "main" "__init__.py"

# 4. Đồng bộ toàn bộ Workflows gốc Lightricks 2.5 vào thư mục lighttrick của ComfyUI
WF_DEST="$COMFY_ROOT/user/default/workflows/lighttrick"
mkdir -p "$WF_DEST"
if [ -d "$SCRIPT_DIR/../workflows/lighttrick" ]; then
    rm -rf "$WF_DEST"/*
    cp -rf "$SCRIPT_DIR/../workflows/lighttrick/"* "$WF_DEST/" 2>/dev/null || true
    echo "✅ Đã chép toàn bộ Workflows gốc Lightricks 2.5 vào: $WF_DEST"
fi

# 5. Hàm tải Model tốc độ cao (aria2c 16 luồng + Fallback Mirror Ungated)
download_model() {
    local folder="$1"
    local filename="$2"
    local primary_url="$3"
    local fallback_url="${4:-}"

    local target_dir="$COMFY_ROOT/models/$folder"
    local target_path="$target_dir/$filename"
    mkdir -p "$target_dir"

    if [ -f "$target_path" ]; then
        local fsize
        fsize=$(stat -c%s "$target_path" 2>/dev/null || stat -f%z "$target_path" 2>/dev/null || echo 0)
        if [ "$fsize" -gt 1048576 ]; then
            echo "✅ [Đã có] $folder/$filename ($((fsize / 1048576)) MB)"
            return 0
        fi
        rm -f "$target_path"
    fi

    local urls=()
    if [[ "$primary_url" == *"huggingface.co/Lightricks/LTX-2.5-22b-"* ]] && [ -z "$HF_TOKEN" ] && [ -n "$fallback_url" ]; then
        urls=("$fallback_url" "$primary_url")
    elif [ -n "$fallback_url" ]; then
        urls=("$primary_url" "$fallback_url")
    else
        urls=("$primary_url")
    fi

    for u in "${urls[@]}"; do
        echo "⬇️ Đang tải $folder/$filename từ: $u"
        local auth_args=()
        if [ -n "$HF_TOKEN" ] && [[ "$u" == *"huggingface.co"* ]] && [[ "$u" != *"/buckets/"* ]]; then
            auth_args=("--header=Authorization: Bearer $HF_TOKEN")
        fi

        if command -v aria2c >/dev/null 2>&1; then
            aria2c --check-certificate=false --disable-ipv6=true --file-allocation=none \
                -c -x 16 -s 16 -k 1M --allow-overwrite=true \
                "${auth_args[@]}" -d "$target_dir" -o "${filename}.part" "$u" || true
        else
            if [ -n "$HF_TOKEN" ] && [[ "$u" == *"huggingface.co"* ]] && [[ "$u" != *"/buckets/"* ]]; then
                curl -4 -L -C - -H "Authorization: Bearer $HF_TOKEN" "$u" -o "$target_dir/${filename}.part" || true
            else
                curl -4 -L -C - "$u" -o "$target_dir/${filename}.part" || true
            fi
        fi

        if [ -f "$target_dir/${filename}.part" ]; then
            local psize
            psize=$(stat -c%s "$target_dir/${filename}.part" 2>/dev/null || stat -f%z "$target_dir/${filename}.part" 2>/dev/null || echo 0)
            if [ "$psize" -gt 1048576 ]; then
                mv -f "$target_dir/${filename}.part" "$target_path"
                echo "✅ [Hoàn tất] $folder/$filename ($((psize / 1048576)) MB)"
                return 0
            else
                rm -f "$target_dir/${filename}.part"
            fi
        fi
    done

    echo "⚠️ [Cảnh báo] Chưa tải được $folder/$filename (Nếu là repo Gated chưa có mirror, hãy truyền HF_TOKEN)."
    return 1
}

echo "=============================================================================="
echo "📥 PHẦN 1: TẢI TRỌN BỘ CORE MODEL LIGHTRICKS LTX-2.5 22B (UNGATED HTTP 200)"
echo "=============================================================================="
download_model "diffusion_models" "ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-22b-distilled-transformer-bf16.safetensors"

download_model "diffusion_models" "ltx-2.5-22b-dev-transformer-bf16.safetensors" \
    "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-22b-dev-transformer-bf16.safetensors"

download_model "loras" "ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
    "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"

download_model "text_encoders" "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"

download_model "text_encoders" "gemma4_e2b_it_bf16.safetensors" \
    "https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e2b_it_bf16.safetensors"

download_model "vae" "ltx-2.5-video-vae-bf16.safetensors" \
    "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-video-vae-bf16.safetensors"

download_model "vae" "ltx-2.5-audio-vae-bf16.safetensors" \
    "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-audio-vae-bf16.safetensors"

download_model "latent_upscale_models" "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

download_model "latent_upscale_models" "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors" \
    "https://huggingface.co/yuvraj108c/LTX-2.5/resolve/main/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"

echo "=============================================================================="
echo "📥 PHẦN 2: TẢI CÁC IC-LORA UNGATED CHUẨN TRONG WORKFLOW LIGHTRICKS 2.5"
echo "=============================================================================="
download_model "loras" "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-outpaint.safetensors" \
    "https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/resolve/main/ltx-2.3-22b-ic-lora-outpaint.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Day-To-Night/ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-deblur-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-decompression-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Decompression/ltx-2.3-22b-ic-lora-decompression-0.9.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-colorization-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Colorization/ltx-2.3-22b-ic-lora-colorization-0.9.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Water-Simulation/ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"

download_model "embeddings" "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Instant-Shave/ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"

download_model "loras" "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

download_model "loras" "ltx-2.3-22b-lora-cinemagraph-0.9.safetensors" \
    "https://huggingface.co/thinklikeajew2/ltx23-cinemagraph-mirror/resolve/main/ltx-2.3-22b-lora-cinemagraph-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-bbox-control-3000-rank-32.safetensors" \
    "https://huggingface.co/yuvraj108c/LTX-2.5-22b-IC-LoRA-BBox-Control/resolve/main/ltx-2.5-22b-ic-lora-bbox-control-3000-rank-32.safetensors"

echo "=============================================================================="
echo "📥 PHẦN 3: TẢI 15 BỘ MODEL LIGHTRICKS LTX-2.5 CREATIVE LAB (HF_TOKEN + FALLBACK)"
echo "=============================================================================="
download_model "loras" "ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Day-To-Night/resolve/main/ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Day-To-Night/ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate/resolve/main/ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-water-simulation-0.9.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Water-Simulation/resolve/main/ltx-2.5-22b-ic-lora-water-simulation-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Water-Simulation/ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler/resolve/main/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-deblur-0.9.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Deblur/resolve/main/ltx-2.5-22b-ic-lora-deblur-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-decompression-0.9.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Decompression/resolve/main/ltx-2.5-22b-ic-lora-decompression-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Decompression/ltx-2.3-22b-ic-lora-decompression-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-colorization-0.9.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Colorization/resolve/main/ltx-2.5-22b-ic-lora-colorization-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Colorization/ltx-2.3-22b-ic-lora-colorization-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-lora-cinemagraph-0.9.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-LoRA-Cinemagraph/resolve/main/ltx-2.5-22b-lora-cinemagraph-0.9.safetensors" \
    "https://huggingface.co/thinklikeajew2/ltx23-cinemagraph-mirror/resolve/main/ltx-2.3-22b-lora-cinemagraph-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-lora-slow-motion-control-1.0.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-LoRA-Slow-Motion-Control/resolve/main/ltx-2.5-22b-lora-slow-motion-control-1.0.safetensors" \
    "https://huggingface.co/thinklikeajew2/ltx23-cinemagraph-mirror/resolve/main/ltx-2.3-22b-lora-cinemagraph-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients/resolve/main/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR/resolve/main/ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"

download_model "embeddings" "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR/resolve/main/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render/resolve/main/ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details/resolve/main/ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-restore-1.0.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Restore/resolve/main/ltx-2.5-22b-ic-lora-restore-1.0.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-Deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors"

download_model "loras" "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors" \
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen/resolve/main/ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors" \
    "https://huggingface.co/buckets/raubatz/ltx/resolve/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"

echo "=============================================================================="
echo "🎉 HOÀN TẤT CÀI ĐẶT LIGHTRICKS LTX-2.5 + CREATIVE LAB TRÊN H200!"
echo "👉 Mở ComfyUI -> Workflows -> lighttrick/ để chạy toàn bộ 25 Workflows."
echo "=============================================================================="
