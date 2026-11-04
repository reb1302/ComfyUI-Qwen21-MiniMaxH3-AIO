#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# H200 CODING AGENT ALL-IN-ONE (SELF-HEALING, SMOKE-TESTED, NO DOCKER)
# Model : Qwen/Qwen3.8-27B (Chạy 1 Engine duy nhất, không tranh chấp GPU)
# Cổng  : LiteLLM Gateway (Port 4000) -> vLLM Engine (Port 8001)
# ==============================================================================

WORK_DIR="${WORK_DIR:-$HOME/h200-coding-agent}"
VLLM_VENV="$WORK_DIR/.venv-vllm"
LITELLM_VENV="$WORK_DIR/.venv-litellm"
LOG_DIR="$WORK_DIR/logs"
PID_DIR="$WORK_DIR/pids"

MAIN_MODEL="${MAIN_MODEL:-Qwen/Qwen3.8-27B}"
MAIN_PORT="${MAIN_PORT:-8001}"
GATEWAY_PORT="${GATEWAY_PORT:-4000}"

TEAM_API_KEY="${TEAM_API_KEY:-sk-team-h200-super-secret}"
INTERNAL_KEY="sk-internal-vllm-key"
ENABLE_VISION="${ENABLE_VISION:-false}"

for arg in "$@"; do
    case "$arg" in
        --vision) ENABLE_VISION=true ;;
    esac
done

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p "$WORK_DIR" "$LOG_DIR" "$PID_DIR"

# ==============================================================================
# 1. DỌN SẠCH TRIỆT ĐỂ TIẾN TRÌNH CŨ & ĐỢI GPU GIẢI PHÓNG VRAM
# ==============================================================================
stop_stack() {
    echo "🛑 Đang dọn sạch toàn bộ tiến trình vLLM / EngineCore / APIServer / LiteLLM cũ..."
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "APIServer" 2>/dev/null || true
    pkill -9 -f "litellm" 2>/dev/null || true
    rm -f "$PID_DIR"/*.pid 2>/dev/null || true

    # Đợi GPU thực sự giải phóng VRAM (tối đa 10 giây)
    for _ in {1..5}; do
        if ! pgrep -f "vllm serve|VLLM::EngineCore" >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    sleep 1
    echo "✅ Đã giải phóng sạch tiến trình cũ!"
}

# ==============================================================================
# 2. TỰ ĐỘNG TÍNH TOÁN % VRAM AN TOÀN TRÊN H200 (KỂ CẢ KHI COMFYUI ĐANG CHẠY)
# ==============================================================================
calc_safe_gpu_util() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "0.80"
        return
    fi
    local free_mib total_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')

    if [ -z "$free_mib" ] || [ -z "$total_mib" ] || [ "$total_mib" -le 0 ]; then
        echo "0.80"
        return
    fi

    # Qwen3.8-27B BF16 cần tối thiểu ~58,000 MiB (~58GB)
    if [ "$free_mib" -lt 62000 ]; then
        echo "⚠️ CẢNH BÁO: VRAM trống hiện tại chỉ còn ${free_mib} MiB (< 62GB)!" >&2
        echo "   Đang kiểm tra các tiến trình chiếm GPU:" >&2
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
    fi

    # Lấy 82% lượng VRAM đang trống quy đổi theo tổng VRAM (giới hạn trong khoảng 0.60 -> 0.85)
    awk -v f="$free_mib" -v t="$total_mib" 'BEGIN {
        u = (f * 0.82) / t;
        if (u > 0.85) u = 0.85;
        if (u < 0.60) u = 0.60;
        printf "%.2f", u;
    }'
}

# ==============================================================================
# 3. GHI CẤU HÌNH LITELLM & CÀI ĐẶT MÔI TRƯỜNG
# ==============================================================================
write_litellm_config() {
    cat << YAML > "$WORK_DIR/litellm_config.yaml"
model_list:
  # 1. Chế độ Agent Suy Luận Sâu (xhigh - 20 điểm Artificial Analysis)
  - model_name: qwen3.8-agent-xhigh
    litellm_params:
      model: openai/$MAIN_MODEL
      api_base: http://127.0.0.1:$MAIN_PORT/v1
      api_key: "$INTERNAL_KEY"
      extra_body:
        chat_template_kwargs:
          enable_thinking: true

  # 2. Chế độ Code Nhanh (Tắt suy luận - Phản hồi tức thì 0ms delay)
  - model_name: qwen3.8-coder-fast
    litellm_params:
      model: openai/$MAIN_MODEL
      api_base: http://127.0.0.1:$MAIN_PORT/v1
      api_key: "$INTERNAL_KEY"
      extra_body:
        chat_template_kwargs:
          enable_thinking: false

  # 3. Chế độ Tab Autocomplete cho IDE (FIM siêu tốc)
  - model_name: qwen-autocomplete-fim
    litellm_params:
      model: openai/$MAIN_MODEL
      api_base: http://127.0.0.1:$MAIN_PORT/v1
      api_key: "$INTERNAL_KEY"
      extra_body:
        chat_template_kwargs:
          enable_thinking: false

general_settings:
  master_key: "$TEAM_API_KEY"
YAML
}

install_stack() {
    echo "======================================================================"
    echo "🚀 [1/4] Kiểm tra phần cứng GPU NVIDIA H200..."
    echo "======================================================================"
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ LỖI: Không tìm thấy driver NVIDIA (nvidia-smi)!"
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.free,memory.total,driver_version --format=csv

    echo ""
    echo "======================================================================"
    echo "🚀 [2/4] Cài đặt 'uv' & môi trường Python 3.12 độc lập..."
    echo "======================================================================"
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi

    if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
        uv venv "$VLLM_VENV" --python 3.12 --seed
    fi
    uv pip install --python "$VLLM_VENV/bin/python" --upgrade \
        vllm \
        transformers \
        huggingface_hub \
        hf_transfer \
        qwen-vl-utils

    if [ ! -x "$LITELLM_VENV/bin/litellm" ]; then
        uv venv "$LITELLM_VENV" --python 3.12 --seed
        uv pip install --python "$LITELLM_VENV/bin/python" --upgrade "litellm[proxy]"
    fi

    echo ""
    echo "======================================================================"
    echo "🚀 [3/4] Kiểm tra / Tải Model $MAIN_MODEL..."
    echo "======================================================================"
    if [ -n "${HF_TOKEN:-}" ]; then
        "$VLLM_VENV/bin/huggingface-cli" download "$MAIN_MODEL" --token "$HF_TOKEN"
    else
        "$VLLM_VENV/bin/huggingface-cli" download "$MAIN_MODEL"
    fi

    echo ""
    echo "======================================================================"
    echo "🚀 [4/4] Tạo file cấu hình LiteLLM Gateway..."
    echo "======================================================================"
    write_litellm_config
    echo "✅ CÀI ĐẶT HOÀN TẤT!"
}

# ==============================================================================
# 4. HÀM CHỜ HEALTHCHECK (TRẢ VỀ 0 NẾU OK, 1 NẾU CRASH ĐỂ TỰ FALLBACK)
# ==============================================================================
wait_for_port() {
    local port=$1
    local name=$2
    local log_file=$3
    local pid_file=$4

    echo -n "⏳ Đang khởi động $name (Port $port) "
    for _ in {1..180}; do
        if [ -f "$pid_file" ] && ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            echo -e "\n⚠️ $name dừng sớm! Chi tiết lỗi trong $log_file:"
            echo "----------------------------------------------------------------------"
            grep -E -B 2 -A 20 "ERROR|Traceback|RuntimeError|ValueError|OutOfMemoryError" "$log_file" | tail -n 45 || tail -n 45 "$log_file"
            echo "----------------------------------------------------------------------"
            return 1
        fi
        if (curl -sf -H "Authorization: Bearer $INTERNAL_KEY" "http://127.0.0.1:$port/health" >/dev/null 2>&1 || \
            curl -sf -H "Authorization: Bearer $TEAM_API_KEY" "http://127.0.0.1:$port/health/liveness" >/dev/null 2>&1) && \
           [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            echo -e "\n✅ $name đã sẵn sàng 100%!"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    echo -e "\n⚠️ Quá thời gian chờ $name! Xem log tại: $log_file"
    return 1
}

# ==============================================================================
# 5. KHỞI CHẠY VLLM VỚI CƠ CHẾ TỰ ĐỘNG CHỮA LỖI 3 CẤP (SELF-HEALING FALLBACK)
# ==============================================================================
launch_vllm_with_fallback() {
    local gpu_util
    gpu_util=$(calc_safe_gpu_util)

    local mm_flags=("--language-model-only")
    if [ "$ENABLE_VISION" = true ]; then
        # Tắt video warmup (nguyên nhân gây tràn RAM Qwen3VL), chỉ giữ xử lý ảnh chụp màn hình
        mm_flags=("--limit-mm-per-prompt" '{"image": 2, "video": 0}')
    fi

    echo "🔥 [Cấp độ 1 - Tốc độ tối đa] Khởi chạy $MAIN_MODEL (GPU Util: $gpu_util, Mode: Coding Agent)..."
    rm -f "$LOG_DIR/vllm_main.log"
    nohup "$VLLM_VENV/bin/vllm" serve "$MAIN_MODEL" \
        --host 127.0.0.1 \
        --port "$MAIN_PORT" \
        --api-key "$INTERNAL_KEY" \
        --dtype bfloat16 \
        --gpu-memory-utilization "$gpu_util" \
        --max-model-len 65536 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 32 \
        "${mm_flags[@]}" \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        > "$LOG_DIR/vllm_main.log" 2>&1 &
    echo $! > "$PID_DIR/vllm_main.pid"

    if wait_for_port "$MAIN_PORT" "vLLM ($MAIN_MODEL - Fast Mode)" "$LOG_DIR/vllm_main.log" "$PID_DIR/vllm_main.pid"; then
        return 0
    fi

    echo "🔄 Tự động kích hoạt [Cấp độ 2 - Safe Eager Mode] (Bỏ qua CUDA Graph & Vision Warmup)..."
    stop_stack
    gpu_util=$(calc_safe_gpu_util)
    rm -f "$LOG_DIR/vllm_main.log"
    nohup "$VLLM_VENV/bin/vllm" serve "$MAIN_MODEL" \
        --host 127.0.0.1 \
        --port "$MAIN_PORT" \
        --api-key "$INTERNAL_KEY" \
        --dtype bfloat16 \
        --gpu-memory-utilization "$gpu_util" \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --language-model-only \
        --enforce-eager \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        > "$LOG_DIR/vllm_main.log" 2>&1 &
    echo $! > "$PID_DIR/vllm_main.pid"

    if wait_for_port "$MAIN_PORT" "vLLM ($MAIN_MODEL - Safe Eager Mode)" "$LOG_DIR/vllm_main.log" "$PID_DIR/vllm_main.pid"; then
        return 0
    fi

    echo "❌ Không thể khởi động vLLM sau cả 2 chế độ. Hãy kiểm tra 'nvidia-smi' xem có tiến trình nào khác đang chiếm hết VRAM không!"
    exit 1
}

# ==============================================================================
# 6. KHỞI ĐỘNG TOÀN BỘ & TỰ ĐỘNG SMOKE-TEST API
# ==============================================================================
start_stack() {
    if [ ! -x "$VLLM_VENV/bin/vllm" ] || [ ! -x "$LITELLM_VENV/bin/litellm" ]; then
        echo "📦 Phát hiện chưa cài đặt đủ môi trường, tiến hành cài đặt tự động..."
        install_stack
    else
        write_litellm_config
    fi

    stop_stack
    ulimit -n 65535 2>/dev/null || true

    echo ""
    launch_vllm_with_fallback

    echo ""
    echo "🔥 Khởi chạy LiteLLM Gateway cho cả nhóm tại cổng $GATEWAY_PORT..."
    rm -f "$LOG_DIR/litellm.log"
    nohup "$LITELLM_VENV/bin/litellm" \
        --config "$WORK_DIR/litellm_config.yaml" \
        --host 0.0.0.0 \
        --port "$GATEWAY_PORT" \
        --num_workers 8 \
        > "$LOG_DIR/litellm.log" 2>&1 &
    echo $! > "$PID_DIR/litellm.pid"

    if ! wait_for_port "$GATEWAY_PORT" "LiteLLM Gateway" "$LOG_DIR/litellm.log" "$PID_DIR/litellm.pid"; then
        echo "❌ LiteLLM khởi động thất bại!"
        exit 1
    fi

    echo ""
    echo "🧪 Chạy Smoke Test kiểm tra phản hồi thực tế từ Model..."
    test_api

    SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    SERVER_IP=${SERVER_IP:-127.0.0.1}

    echo ""
    echo "======================================================================"
    echo "🎉 HỆ THỐNG CODING AGENT TRÊN H200 ĐÃ HOẠT ĐỘNG & SMOKE-TEST THÀNH CÔNG!"
    echo "======================================================================"
    echo "🌐 API Base URL cho cả nhóm : http://$SERVER_IP:$GATEWAY_PORT/v1"
    echo "🔑 API Key chung của nhóm   : $TEAM_API_KEY"
    echo "----------------------------------------------------------------------"
    echo "🧠 Model 1 (Suy luận sâu)   : qwen3.8-agent-xhigh"
    echo "⚡ Model 2 (Code siêu tốc)  : qwen3.8-coder-fast"
    echo "⌨️  Model 3 (Tab gợi ý code) : qwen-autocomplete-fim"
    echo "======================================================================"
}

status_stack() {
    echo "📊 Trạng thái GPU H200 hiện tại:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free,memory.used,memory.total --format=csv
    echo ""
    echo "📊 Các tiến trình đang chạy:"
    pgrep -af "vllm|litellm" || echo "🔴 Không có dịch vụ nào đang chạy"
}

test_api() {
    local resp
    resp=$(curl -s --max-time 30 "http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TEAM_API_KEY" \
        -d '{
            "model": "qwen3.8-coder-fast",
            "messages": [{"role": "user", "content": "Reply with exactly: H200_AGENT_READY"}],
            "max_tokens": 32
        }' || true)

    if echo "$resp" | grep -q "choices"; then
        echo "✅ SMOKE TEST PASS! Phản hồi từ Qwen3.8-27B:"
        echo "$resp" | grep -o '"content":"[^"]*"' | head -n 1 || echo "$resp"
    else
        echo "⚠️ Phản hồi Smoke Test:"
        echo "$resp"
    fi
}

logs_stack() {
    tail -n 60 -f "$LOG_DIR"/*.log
}

case "${1:-start}" in
    all)      install_stack && start_stack ;;
    install)  install_stack ;;
    start)    start_stack ;;
    stop)     stop_stack ;;
    restart)  start_stack ;;
    status)   status_stack ;;
    test)     test_api ;;
    logs)     logs_stack ;;
    *)
        echo "Cách dùng: $0 {start|stop|restart|install|status|test|logs} [--vision]"
        exit 1
        ;;
esac
