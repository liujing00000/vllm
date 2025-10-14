#!/bin/bash

PORT=8000
MODEL=$1
TOKENS=$2

docker run -e "HF_TOKEN=$HF_TOKEN" --gpus all --shm-size 1g -p $PORT:80 \
           -v "$PWD/data:/data" \
           ghcr.io/huggingface/text-generation-inference:2.2.0 \
           --model-id "$MODEL" \
           --sharded false  \
           --max-input-length 1024 \
           --max-total-tokens 2048 \
           --max-best-of 5 \
           --max-concurrent-requests 5000 \
           --max-batch-total-tokens "$TOKENS"


#!/bin/bash
set -euo pipefail

# 固定测试温度值
TEMPERATURE=0.65

# 显示帮助信息
show_help() {
    echo "Usage: $0 --model MODEL_NAME --port PORT"
    echo ""
    echo "Benchmark Testing Script for AI Models"
    echo ""
    echo "Required Arguments:"
    echo "  --model       Model name (e.g. Qwen3-32B)"
    echo "  --port        Port number for API server"
    echo ""
    echo "Example:"
    echo "  $0 --model Qwen3-32B --port 4567"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown parameter: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证必要参数
if [[ -z "${MODEL_NAME:-}" ]]; then
    echo "Error: --model argument is required"
    show_help
    exit 1
fi

if [[ -z "${PORT:-}" ]]; then
    echo "Error: --port argument is required"
    show_help
    exit 1
fi

# 参数组合
combinations=(
    # "4  1024 1024"
    # "4  2048 2048"
    # "4  4096 1024"
    # "8  1024 1024"
    # "8  2048 2048"
    # "8  4096 1024"

    # "16  1024 1024"
    # "16  2048 2048"
    # "16  4096 1024"
    # "32  1024 1024"
    # "32  2048 2048"
    # "32  4096 1024"

    "64  1024 1024"
    # "64  2048 2048"
    # "64  4096 1024"
    # "128 1024 1024"
    # "128 2048 2048"
    # "128 4096 1024"
    # "256 1024 1024"
    # "256 2048 2048"
    # "256 4096 1024"
)

# 创建唯一结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROOT_RESULTS_DIR="./results/${MODEL_NAME}_port${PORT}_${TIMESTAMP}"
mkdir -p "$ROOT_RESULTS_DIR"

# 创建运行配置记录
{
    echo "==== Benchmark Configuration ===="
    echo "Model:       $MODEL_NAME"
    echo "API Port:    $PORT"
    echo "Start Time:  $(date)"
    echo "Result Dir:  $ROOT_RESULTS_DIR"
    echo ""
} | tee "${ROOT_RESULTS_DIR}/config.log" > "${ROOT_RESULTS_DIR}/summary.log"

# 运行所有组合
for combo in "${combinations[@]}"; do
    read num_prompts input_len output_len <<< $combo
    
    # 创建当前组合的结果目录
    COMBO_DIR="${ROOT_RESULTS_DIR}/prompts${num_prompts}_in${input_len}_out${output_len}"
    mkdir -p "$COMBO_DIR"
    
    # 日志文件
    EXEC_LOG="${COMBO_DIR}/execution.log"
    
    # 开始时间
    START_TIME=$(date +%s)
    
    # 记录参数
    echo "Testing: prompts=${num_prompts}, input=${input_len}, output=${output_len}, temp=${TEMPERATURE}" > "$EXEC_LOG"
    
    # 执行测试命令
    set +e
    vllm bench serve \
        --backend openai-chat \
        --base-url "http://127.0.0.1:${PORT}" \
        --endpoint '/v1/chat/completions' \
        --model "$MODEL_NAME" \
        --tokenizer "/workspace/models/Qwen_Qwen3-32B" \
        --trust-remote-code \
        --temperature "$TEMPERATURE" \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --ignore-eos \
        --num-prompts "$num_prompts" \
        --save-result \
        --profile \
        --result-dir "$COMBO_DIR" >> "$EXEC_LOG" 2>&1
    
    EXIT_STATUS=$?
    set -e
    
    # 计算持续时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_STR=$(printf "%02d:%02d:%02d" $((DURATION/3600)) $(( (DURATION%3600)/60 )) $((DURATION%60)))
    
    # 记录结果状态
    if [ $EXIT_STATUS -eq 0 ]; then
        STATUS="SUCCESS"
        STATUS_COLOR="\033[32m"
    else
        STATUS="FAILED"
        STATUS_COLOR="\033[31m"
    fi
    
    # 添加到摘要日志
    printf "%-10s | %-5s | %-5s | %-5s | ${STATUS_COLOR}%-7s\033[0m | %-9s | %s\n" \
        "$(date +%T)" "$num_prompts" "$input_len" "$output_len" "$STATUS" "$DURATION_STR" "$COMBO_DIR" \
        >> "${ROOT_RESULTS_DIR}/summary.log"
done

# 生成最终报告
{
    echo ""
    echo "==== Benchmark Results Summary ===="
    echo "Start Time: $(date -d @$START_TIME +'%Y-%m-%d %H:%M:%S')"
    echo "End Time:   $(date)"
    echo "Total Duration: $(( (END_TIME - START_TIME)/60 )) minutes"
    echo ""
    echo "Model: $MODEL_NAME | Port: $PORT | Temperature: $TEMPERATURE"
    echo ""
    echo "Column: Time | Prompts | Input | Output | Status | Duration | Result Directory"
    echo "-------------------------------------------------------------------------------"
    cat "${ROOT_RESULTS_DIR}/summary.log"
} | tee -a "${ROOT_RESULTS_DIR}/config.log"

echo ""
echo "Benchmark completed. All results saved to:"
echo "  $ROOT_RESULTS_DIR"
