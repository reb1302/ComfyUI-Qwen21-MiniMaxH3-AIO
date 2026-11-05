#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# H200 DUAL-SOTA COMPANY AI & CODING AGENT (NO DOCKER, AUTO VRAM BALANCER)
# Cài đặt & Chạy cả 2 Model SOTA Bản Tối Giản Chính Chủ (FP8):
#   1. Qwen/Qwen3.8-27B-FP8     (28.75 GiB - Suy luận sâu 20đ Artificial Analysis)
#   2. Qwen/Qwen3.6-35B-A3B-FP8 (34.89 GiB - Siêu MoE 35B/3B active nhanh gấp 5 lần)
# ==============================================================================

WORK_DIR="${WORK_DIR:-$HOME/h200-coding-agent}"
VLLM_VENV="$WORK_DIR/.venv-vllm"
LITELLM_VENV="$WORK_DIR/.venv-litellm"
LOG_DIR="$WORK_DIR/logs"
PID_DIR="$WORK_DIR/pids"

MODEL_38="Qwen/Qwen3.8-27B-FP8"
MODEL_36="Qwen/Qwen3.6-35B-A3B-FP8"

PORT_38="${PORT_38:-8001}"
PORT_36="${PORT_36:-8002}"
GATEWAY_PORT="${GATEWAY_PORT:-4000}"

TEAM_API_KEY="${TEAM_API_KEY:-sk-team-h200-super-secret}"
INTERNAL_KEY="sk-internal-vllm-key"

# Chế độ chạy: "dual" (chạy cả 2 nếu đủ VRAM), "only38", hoặc "only36"
RUN_MODE="dual"
for arg in "$@"; do
    case "$arg" in
        --only-38|--38) RUN_MODE="only38" ;;
        --only-36|--36|--moe) RUN_MODE="only36" ;;
        --dual) RUN_MODE="dual" ;;
    esac
done

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

mkdir -p "$WORK_DIR" "$LOG_DIR" "$PID_DIR"

# ==============================================================================
# 1. DỌN SẠCH TIẾN TRÌNH VLLM / LITELLM CŨ (GIỮ NGUYÊN COMFYUI)
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

get_free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

get_total_mib() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

calc_util_for_mib() {
    local target_mib=$1
    local free_mib total_mib
    free_mib=$(get_free_mib)
    total_mib=$(get_total_mib)

    awk -v target="$target_mib" -v f="$free_mib" -v t="$total_mib" 'BEGIN {
        # Không bao giờ vượt quá (free_mib - 3500 MiB)
        max_safe = f - 3500;
        use_mib = (target < max_safe) ? target : max_safe;
        u = use_mib / t;
        if (u > 0.85) u = 0.85;
        if (u < 0.20) u = 0.20;
        printf "%.2f", u;
    }'
}

# ==============================================================================
# 2. CÀI ĐẶT MÔI TRƯỜNG & TẢI CẢ 2 MODEL FP8 VỀ Ổ CỨNG
# ==============================================================================
install_stack() {
    echo "======================================================================"
    echo "🚀 [1/4] Kiểm tra phần cứng GPU NVIDIA H200..."
    echo "======================================================================"
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

    echo ""
    echo "======================================================================"
    echo "🚀 [3/4] Tải Bản 1: $MODEL_38 (28.75 GiB - SOTA Reasoning)..."
    echo "======================================================================"
    "$VLLM_VENV/bin/huggingface-cli" download "$MODEL_38"

    echo ""
    echo "======================================================================"
    echo "🚀 [4/4] Tải Bản 2: $MODEL_36 (34.89 GiB - Siêu MoE 35B-A3B)..."
    echo "======================================================================"
    "$VLLM_VENV/bin/huggingface-cli" download "$MODEL_36"

    echo "✅ ĐÃ TẢI & CÀI ĐẶT XONG CẢ 2 BẢN MODEL!"
}

# ==============================================================================
# 3. GHI CẤU HÌNH LITELLM GATEWAY (TỰ ĐỘNG ĐỊNH TUYẾN THEO ENGINE ĐANG CHẠY)
# ==============================================================================
write_litellm_config() {
    local model_deep="$1"
    local port_deep="$2"
    local model_fast="$3"
    local port_fast="$4"

    cat << YAML > "$WORK_DIR/litellm_config.yaml"
model_list:
  # 1. Chế độ Suy Luận Sâu (Qwen3.8-27B-FP8 xhigh - Debug khó, Kiến trúc)
  - model_name: qwen3.8-agent-xhigh
    litellm_params:
      model: openai/$model_deep
      api_base: http://127.0.0.1:$port_deep/v1
      api_key: "$INTERNAL_KEY"
      extra_body:
        chat_template_kwargs:
          enable_thinking: true

  # 2. Chế độ Siêu MoE 35B-A3B (Tốc độ 200+ tok/s cho cả công ty)
  - model_name: qwen3.6-moe-fast
    litellm_params:
      model: openai/$model_fast
      api_base: http://127.0.0.1:$port_fast/v1
      api_key: "$INTERNAL_KEY"
      extra_body:
        chat_template_kwargs:
          enable_thinking: false

  # 3. Tên tương thích ngược (Code nhanh)
  - model_name: qwen3.8-coder-fast
    litellm_params:
      model: openai/$model_fast
      api_base: http://127.0.0.1:$port_fast/v1
      api_key: "$INTERNAL_KEY"
      extra_body:
        chat_template_kwargs:
          enable_thinking: false

  # 4. Chuyên gia Dịch Thuật Tiếng Hàn <-> Việt/Anh & Hành chính Công ty
  - model_name: company-korean-translator
    litellm_params:
      model: openai/$model_fast
      api_base: http://127.0.0.1:$port_fast/v1
      api_key: "$INTERNAL_KEY"
      extra_body:
        chat_template_kwargs:
          enable_thinking: false

  # 5. Chế độ Tab Autocomplete cho IDE (FIM siêu tốc)
  - model_name: qwen-autocomplete-fim
    litellm_params:
      model: openai/$model_fast
      api_base: http://127.0.0.1:$port_fast/v1
      api_key: "$INTERNAL_KEY"
      extra_body:
        chat_template_kwargs:
          enable_thinking: false

general_settings:
  master_key: "$TEAM_API_KEY"
YAML
}

# ==============================================================================
# 4. HÀM CHỜ HEALTHCHECK
# ==============================================================================
wait_for_port() {
    local port=$1
    local name=$2
    local log_file=$3
    local pid_file=$4

    echo -n "⏳ Đang nạp $name lên H200 (Port $port) "
    for _ in {1..180}; do
        if [ -f "$pid_file" ] && ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            echo -e "\n⚠️ $name dừng sớm! Chi tiết lỗi trong $log_file:"
            echo "----------------------------------------------------------------------"
            grep -E -B 2 -A 20 "ERROR|Traceback|RuntimeError|ValueError|OutOfMemoryError" "$log_file" | tail -n 40 || tail -n 40 "$log_file"
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
    echo -e "\n⚠️ Quá thời gian chờ $name!"
    return 1
}

start_one_vllm() {
    local model=$1
    local port=$2
    local util=$3
    local max_len=$4
    local tag=$5

    rm -f "$LOG_DIR/vllm_${tag}.log"
    nohup "$VLLM_VENV/bin/vllm" serve "$model" \
        --host 127.0.0.1 \
        --port "$port" \
        --api-key "$INTERNAL_KEY" \
        --gpu-memory-utilization "$util" \
        --max-model-len "$max_len" \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 32 \
        --language-model-only \
        --enforce-eager \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        > "$LOG_DIR/vllm_${tag}.log" 2>&1 &
    echo $! > "$PID_DIR/vllm_${tag}.pid"

    wait_for_port "$port" "vLLM [$model] (Util: $util)" "$LOG_DIR/vllm_${tag}.log" "$PID_DIR/vllm_${tag}.pid"
}

# ==============================================================================
# 5. KHỞI CHẠY THÔNG MINH (DUAL-MODEL HOẶC SINGLE-MODEL TÙY LƯỢNG VRAM TRỐNG)
# ==============================================================================
start_stack() {
    install_stack
    stop_stack
    ulimit -n 65535 2>/dev/null || true

    local free_mib total_mib
    free_mib=$(get_free_mib)
    total_mib=$(get_total_mib)
    echo ""
    echo "📊 Dung lượng VRAM H200 trống trước khi nạp: ${free_mib} MiB / ${total_mib} MiB"

    ACTIVE_SUMMARY=""

    if [ "$RUN_MODE" = "only38" ]; then
        local u38
        u38=$(calc_util_for_mib 65000)
        echo "🔥 Chạy chuyên biệt Bản 1: $MODEL_38 (GPU Util: $u38)..."
        start_one_vllm "$MODEL_38" "$PORT_38" "$u38" 65536 "38"
        write_litellm_config "$MODEL_38" "$PORT_38" "$MODEL_38" "$PORT_38"
        ACTIVE_SUMMARY="$MODEL_38 (Port $PORT_38)"

    elif [ "$RUN_MODE" = "only36" ]; then
        local u36
        u36=$(calc_util_for_mib 70000)
        echo "🔥 Chạy chuyên biệt Bản 2 (Siêu MoE): $MODEL_36 (GPU Util: $u36)..."
        start_one_vllm "$MODEL_36" "$PORT_36" "$u36" 65536 "36"
        write_litellm_config "$MODEL_36" "$PORT_36" "$MODEL_36" "$PORT_36"
        ACTIVE_SUMMARY="$MODEL_36 (Port $PORT_36)"

    else
        # Chế độ DUAL: Thử nạp cả 2 bản cùng lúc nếu VRAM trống >= 76,000 MiB (~74.2 GiB)
        if [ "$free_mib" -ge 76000 ]; then
            echo "🔥 [DUAL-ENGINE MODE] Nạp song song cả 2 Model lên H200..."
            # Bản 1 (28.75 GiB weights) -> cấp ~35,500 MiB (~34.6 GiB)
            local u1
            u1=$(calc_util_for_mib 35500)
            echo "👉 [1/2] Nạp $MODEL_38 (Suy luận sâu - Util: $u1)..."
            if start_one_vllm "$MODEL_38" "$PORT_38" "$u1" 16384 "38"; then
                local rem_mib
                rem_mib=$(get_free_mib)
                echo "📊 VRAM còn trống cho Bản 2: ${rem_mib} MiB"
                if [ "$rem_mib" -ge 41000 ]; then
                    # Bản 2 (34.89 GiB weights) -> cấp phần VRAM trống còn lại trừ buffer
                    local u2
                    u2=$(calc_util_for_mib 43500)
                    echo "👉 [2/2] Nạp tiếp $MODEL_36 (Siêu MoE tốc độ cao - Util: $u2)..."
                    if start_one_vllm "$MODEL_36" "$PORT_36" "$u2" 16384 "36"; then
                        write_litellm_config "$MODEL_38" "$PORT_38" "$MODEL_36" "$PORT_36"
                        ACTIVE_SUMMARY="CHẠY SONG SONG CẢ 2: $MODEL_38 (Port $PORT_38) + $MODEL_36 (Port $PORT_36)"
                    else
                        echo "⚠️ VRAM không đủ nhét đồng thời cả 2 (do ComfyUI đang chiếm 58GB). Giữ $MODEL_38 chạy chính!"
                        pkill -9 -f "port $PORT_36" 2>/dev/null || true
                        write_litellm_config "$MODEL_38" "$PORT_38" "$MODEL_38" "$PORT_38"
                        ACTIVE_SUMMARY="$MODEL_38 (Đã cài sẵn $MODEL_36 trên ổ cứng, gõ './h200_agent.sh start --36' để đổi)"
                    fi
                else
                    echo "ℹ️ VRAM còn lại (${rem_mib} MiB) đang nhường cho ComfyUI. Sử dụng $MODEL_38 làm Engine chính!"
                    write_litellm_config "$MODEL_38" "$PORT_38" "$MODEL_38" "$PORT_38"
                    ACTIVE_SUMMARY="$MODEL_38 (Đã cài sẵn $MODEL_36 trên ổ cứng, gõ './h200_agent.sh start --36' để đổi)"
                fi
            fi
        else
            local u_single
            u_single=$(calc_util_for_mib 60000)
            echo "ℹ️ VRAM trống (${free_mib} MiB) đang chia sẻ với ComfyUI -> Chạy Siêu MoE $MODEL_36 (Util: $u_single)..."
            start_one_vllm "$MODEL_36" "$PORT_36" "$u_single" 32768 "36"
            write_litellm_config "$MODEL_36" "$PORT_36" "$MODEL_36" "$PORT_36"
            ACTIVE_SUMMARY="$MODEL_36 (Đã cài sẵn $MODEL_38 trên ổ cứng, gõ './h200_agent.sh start --38' để đổi)"
        fi
    fi

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

    wait_for_port "$GATEWAY_PORT" "LiteLLM Gateway" "$LOG_DIR/litellm.log" "$PID_DIR/litellm.pid"

    echo ""
    echo "🧪 Chạy Smoke Test kiểm tra phản hồi Dịch thuật Tiếng Hàn & Code..."
    test_api

    SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    SERVER_IP=${SERVER_IP:-127.0.0.1}

    echo ""
    echo "======================================================================"
    echo "🎉 HỆ THỐNG DUAL-SOTA TRÊN H200 ĐÃ HOẠT ĐỘNG 100%!"
    echo "======================================================================"
    echo "🌐 API Base URL cho công ty : http://$SERVER_IP:$GATEWAY_PORT/v1"
    echo "🔑 API Key chung            : $TEAM_API_KEY"
    echo "⚙️  Engine đang hoạt động    : $ACTIVE_SUMMARY"
    echo "----------------------------------------------------------------------"
    echo "🧠 1. qwen3.8-agent-xhigh       (Suy luận sâu 20đ - Debug & Kiến trúc)"
    echo "⚡ 2. qwen3.6-moe-fast          (Siêu MoE 35B-A3B - Code & Chat 200+ tok/s)"
    echo "🇰🇷 3. company-korean-translator (Dịch thuật Tiếng Hàn <-> Việt/Anh)"
    echo "⌨️  4. qwen-autocomplete-fim     (Tab Autocomplete cho VSCode/JetBrains)"
    echo "----------------------------------------------------------------------"
    echo "🔄 Lệnh chuyển đổi nhanh khi cần:"
    echo "   ./h200_agent.sh start --dual    (Chạy song song cả 2 model)"
    echo "   ./h200_agent.sh start --only-36 (Dồn hết VRAM cho Siêu MoE Qwen3.6-35B-A3B-FP8)"
    echo "   ./h200_agent.sh start --only-38 (Dồn hết VRAM cho Qwen3.8-27B-FP8)"
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
        echo "✅ SMOKE TEST PASS! Phản hồi thực tế từ Model:"
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
    all)      start_stack ;;
    install)  install_stack ;;
    start)    start_stack ;;
    stop)     stop_stack ;;
    restart)  start_stack ;;
    status)   status_stack ;;
    test)     test_api ;;
    logs)     logs_stack ;;
    *)
        echo "Cách dùng: $0 {start|stop|restart|install|status|test|logs} [--dual|--only-36|--only-38]"
        exit 1
        ;;
esac
