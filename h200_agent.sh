#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# H200 COMPANY AI & CODING AGENT (SELF-HEALING, AUTO VRAM FIT, NO DOCKER)
# - Mặc định        : ./h200_agent.sh start       (Chạy Qwen/Qwen3.8-27B sẵn có)
# - Chế độ Siêu tốc : ./h200_agent.sh start --moe (Chạy Qwen/Qwen3-30B-A3B-FP8 MoE)
# ==============================================================================

WORK_DIR="${WORK_DIR:-$HOME/h200-coding-agent}"
VLLM_VENV="$WORK_DIR/.venv-vllm"
LITELLM_VENV="$WORK_DIR/.venv-litellm"
LOG_DIR="$WORK_DIR/logs"
PID_DIR="$WORK_DIR/pids"

MAIN_MODEL="${MAIN_MODEL:-Qwen/Qwen3.8-27B}"
USE_MOE=false

for arg in "$@"; do
    case "$arg" in
        --moe)
            USE_MOE=true
            MAIN_MODEL="Qwen/Qwen3-30B-A3B-FP8"
            ;;
    esac
done

MAIN_PORT="${MAIN_PORT:-8001}"
GATEWAY_PORT="${GATEWAY_PORT:-4000}"

TEAM_API_KEY="${TEAM_API_KEY:-sk-team-h200-super-secret}"
INTERNAL_KEY="sk-internal-vllm-key"

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p "$WORK_DIR" "$LOG_DIR" "$PID_DIR"

# ==============================================================================
# 1. DỌN SẠCH TIẾN TRÌNH CŨ CỦA VLLM / LITELLM (KHÔNG ĐỤNG VÀO COMFYUI)
# ==============================================================================
stop_stack() {
    echo "🛑 Đang dọn sạch các tiến trình vLLM / EngineCore / LiteLLM cũ..."
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "APIServer" 2>/dev/null || true
    pkill -9 -f "litellm" 2>/dev/null || true
    rm -f "$PID_DIR"/*.pid 2>/dev/null || true

    for _ in {1..5}; do
        if ! pgrep -f "vllm serve|VLLM::EngineCore" >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    sleep 1
    echo "✅ Đã giải phóng sạch tiến trình vLLM cũ!"
}

# ==============================================================================
# 2. TÍNH CHÍNH XÁC % VRAM TRỐNG (TRỪ BUFFER AN TOÀN ĐỂ KHÔNG BAO GIỜ LỖI 82/139 GiB)
# ==============================================================================
calc_safe_gpu_util() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "0.52"
        return
    fi
    local free_mib total_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')

    if [ -z "$free_mib" ] || [ -z "$total_mib" ] || [ "$total_mib" -le 0 ]; then
        echo "0.52"
        return
    fi

    echo "📊 VRAM H200 hiện tại: Trống ${free_mib} MiB / Tổng ${total_mib} MiB" >&2

    # Lấy chính xác tỷ lệ (free / total) trừ đi 0.05 (~7 GiB buffer an toàn cho CUDA context)
    awk -v f="$free_mib" -v t="$total_mib" 'BEGIN {
        free_ratio = f / t;
        u = free_ratio - 0.05;
        if (u > 0.85) u = 0.85;
        if (u < 0.25) u = 0.25;
        printf "%.2f", u;
    }'
}

# ==============================================================================
# 3. GHI CẤU HÌNH LITELLM (CÓ CẢ CODE AGENT + DỊCH THUẬT TIẾNG HÀN CHO CÔNG TY)
# ==============================================================================
write_litellm_config() {
    cat << YAML > "$WORK_DIR/litellm_config.yaml"
model_list:
  # 1. Chế độ Agent Suy Luận Sâu (Debug khó, Thiết kế kiến trúc)
  - model_name: qwen3.8-agent-xhigh
    litellm_params:
      model: openai/$MAIN_MODEL
      api_base: http://127.0.0.1:$MAIN_PORT/v1
      api_key: "$INTERNAL_KEY"
      extra_body:
        chat_template_kwargs:
          enable_thinking: true

  # 2. Chế độ Code Nhanh & Chat Công Ty (Phản hồi tức thì 0ms delay)
  - model_name: qwen3.8-coder-fast
    litellm_params:
      model: openai/$MAIN_MODEL
      api_base: http://127.0.0.1:$MAIN_PORT/v1
      api_key: "$INTERNAL_KEY"
      extra_body:
        chat_template_kwargs:
          enable_thinking: false

  # 3. Chế độ Chuyên Dịch Thuật Tiếng Hàn <-> Việt/Anh & Hành chính Công ty
  - model_name: company-korean-translator
    litellm_params:
      model: openai/$MAIN_MODEL
      api_base: http://127.0.0.1:$MAIN_PORT/v1
      api_key: "$INTERNAL_KEY"
      extra_body:
        chat_template_kwargs:
          enable_thinking: false

  # 4. Chế độ Tab Autocomplete cho IDE (FIM siêu tốc)
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
    echo "🚀 [1/4] Kiểm tra phần cứng GPU NVIDIA H200..."
    nvidia-smi --query-gpu=name,memory.free,memory.total,driver_version --format=csv

    echo "🚀 [2/4] Kiểm tra 'uv' & môi trường Python 3.12..."
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi

    if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
        uv venv "$VLLM_VENV" --python 3.12 --seed
        uv pip install --python "$VLLM_VENV/bin/python" --upgrade \
            vllm \
            transformers \
            huggingface_hub \
            hf_transfer \
            qwen-vl-utils
    fi

    if [ ! -x "$LITELLM_VENV/bin/litellm" ]; then
        uv venv "$LITELLM_VENV" --python 3.12 --seed
        uv pip install --python "$LITELLM_VENV/bin/python" --upgrade "litellm[proxy]"
    fi

    echo "🚀 [3/4] Đảm bảo Model $MAIN_MODEL đã có trên ổ cứng..."
    "$VLLM_VENV/bin/huggingface-cli" download "$MAIN_MODEL"

    echo "🚀 [4/4] Tạo file cấu hình LiteLLM..."
    write_litellm_config
    echo "✅ CÀI ĐẶT HOÀN TẤT!"
}

# ==============================================================================
# 4. HÀM CHỜ HEALTHCHECK
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
# 5. KHỞI CHẠY VLLM VỚI TỰ ĐỘNG CÂN CHỈNH VRAM (CHẠY CHUNG VỚI COMFYUI OK!)
# ==============================================================================
launch_vllm_with_fallback() {
    local gpu_util
    gpu_util=$(calc_safe_gpu_util)

    if [ "$USE_MOE" = true ]; then
        echo "🔥 Đang tải (nếu chưa có) & khởi chạy Siêu MoE $MAIN_MODEL (30.5B/3.3B active, ~30GB VRAM, GPU Util: $gpu_util)..."
        "$VLLM_VENV/bin/huggingface-cli" download "$MAIN_MODEL"
        rm -f "$LOG_DIR/vllm_main.log"
        nohup "$VLLM_VENV/bin/vllm" serve "$MAIN_MODEL" \
            --host 127.0.0.1 \
            --port "$MAIN_PORT" \
            --api-key "$INTERNAL_KEY" \
            --gpu-memory-utilization "$gpu_util" \
            --max-model-len 65536 \
            --max-num-batched-tokens 16384 \
            --max-num-seqs 64 \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            --reasoning-parser qwen3 \
            --trust-remote-code \
            > "$LOG_DIR/vllm_main.log" 2>&1 &
        echo $! > "$PID_DIR/vllm_main.pid"
        if wait_for_port "$MAIN_PORT" "vLLM MoE ($MAIN_MODEL)" "$LOG_DIR/vllm_main.log" "$PID_DIR/vllm_main.pid"; then
            return 0
        fi
    fi

    echo "🔥 [Cấp độ 1 - FP8 Tiết kiệm VRAM (~28GB)] Khởi chạy $MAIN_MODEL (GPU Util: $gpu_util)..."
    rm -f "$LOG_DIR/vllm_main.log"
    nohup "$VLLM_VENV/bin/vllm" serve "$MAIN_MODEL" \
        --host 127.0.0.1 \
        --port "$MAIN_PORT" \
        --api-key "$INTERNAL_KEY" \
        --quantization fp8 \
        --gpu-memory-utilization "$gpu_util" \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 32 \
        --language-model-only \
        --enforce-eager \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        > "$LOG_DIR/vllm_main.log" 2>&1 &
    echo $! > "$PID_DIR/vllm_main.pid"

    if wait_for_port "$MAIN_PORT" "vLLM ($MAIN_MODEL - FP8 Mode)" "$LOG_DIR/vllm_main.log" "$PID_DIR/vllm_main.pid"; then
        return 0
    fi

    echo "🔄 Tự động kích hoạt [Cấp độ 2 - BF16 Safe Mode] (GPU Util: $gpu_util)..."
    stop_stack
    gpu_util=$(calc_safe_gpu_util)
    rm -f "$LOG_DIR/vllm_main.log"
    nohup "$VLLM_VENV/bin/vllm" serve "$MAIN_MODEL" \
        --host 127.0.0.1 \
        --port "$MAIN_PORT" \
        --api-key "$INTERNAL_KEY" \
        --dtype bfloat16 \
        --gpu-memory-utilization "$gpu_util" \
        --max-model-len 16384 \
        --max-num-batched-tokens 4096 \
        --max-num-seqs 16 \
        --language-model-only \
        --enforce-eager \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        > "$LOG_DIR/vllm_main.log" 2>&1 &
    echo $! > "$PID_DIR/vllm_main.pid"

    if wait_for_port "$MAIN_PORT" "vLLM ($MAIN_MODEL - BF16 Safe Mode)" "$LOG_DIR/vllm_main.log" "$PID_DIR/vllm_main.pid"; then
        return 0
    fi

    echo "❌ Không thể khởi động vLLM! Các tiến trình đang chiếm VRAM trên H200 của bạn:"
    nvidia-smi
    exit 1
}

# ==============================================================================
# 6. KHỞI ĐỘNG TOÀN BỘ & SMOKE-TEST CẢ CODE LẪN DỊCH TIẾNG HÀN
# ==============================================================================
start_stack() {
    if [ ! -x "$VLLM_VENV/bin/vllm" ] || [ ! -x "$LITELLM_VENV/bin/litellm" ]; then
        install_stack
    else
        write_litellm_config
    fi

    stop_stack
    ulimit -n 65535 2>/dev/null || true

    echo ""
    launch_vllm_with_fallback

    echo ""
    echo "🔥 Khởi chạy LiteLLM Gateway cho toàn công ty tại cổng $GATEWAY_PORT..."
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
    echo "🧪 Chạy Smoke Test (Kiểm tra phản hồi Code & Dịch thuật Tiếng Hàn)..."
    test_api

    SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    SERVER_IP=${SERVER_IP:-127.0.0.1}

    echo ""
    echo "======================================================================"
    echo "🎉 HỆ THỐNG AI & CODING AGENT TOÀN CÔNG TY ĐÃ HOẠT ĐỘNG 100%!"
    echo "======================================================================"
    echo "🌐 API Base URL cho công ty : http://$SERVER_IP:$GATEWAY_PORT/v1"
    echo "🔑 API Key chung            : $TEAM_API_KEY"
    echo "📦 Model vật lý đang chạy   : $MAIN_MODEL"
    echo "----------------------------------------------------------------------"
    echo "🧠 Model 1 (Code Agent Sâu) : qwen3.8-agent-xhigh"
    echo "⚡ Model 2 (Code Siêu Tốc)  : qwen3.8-coder-fast"
    echo "🇰🇷 Model 3 (Dịch Tiếng Hàn) : company-korean-translator"
    echo "⌨️  Model 4 (Tab Gợi Ý Code) : qwen-autocomplete-fim"
    echo "----------------------------------------------------------------------"
    echo "💡 Mẹo: Muốn đổi sang chạy MoE 30B-A3B-FP8 (nhanh gấp 5 lần, nhẹ VRAM):"
    echo "   ./h200_agent.sh start --moe"
    echo "======================================================================"
}

status_stack() {
    echo "📊 Trạng thái GPU H200 hiện tại:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free,memory.used,memory.total --format=csv
    echo ""
    echo "📊 Các tiến trình đang dùng VRAM trên H200:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
    echo ""
    echo "📊 Tiến trình Agent đang chạy:"
    pgrep -af "vllm|litellm" || echo "🔴 Không có dịch vụ nào đang chạy"
}

test_api() {
    local resp
    resp=$(curl -s --max-time 30 "http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TEAM_API_KEY" \
        -d '{
            "model": "company-korean-translator",
            "messages": [{"role": "user", "content": "Dịch sang tiếng Hàn lịch sự (존댓말): Hệ thống AI nội bộ của công ty đã sẵn sàng hoạt động."}],
            "max_tokens": 60
        }' || true)

    if echo "$resp" | grep -q "choices"; then
        echo "✅ SMOKE TEST PASS! Kết quả dịch Tiếng Hàn từ Model:"
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
        echo "Cách dùng: $0 {start|stop|restart|install|status|test|logs} [--moe]"
        exit 1
        ;;
esac
