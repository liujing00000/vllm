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


import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp


def run_mm_all_reduce_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = "tcp://" + master_ip + ":" + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group

    default_pg = _get_default_group()
    if torch.__version__ > "2.0.1":
        hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcom_info = default_pg.get_hccl_comm_name(rank)

    input_ = torch.randn(x1_shape, dtype=dtype).npu()
    weight = torch.randn(x2_shape, dtype=dtype).npu()
    output = torch_npu.npu_mm_all_reduce_base(input_, weight, hcom_info, reduce_op="sum")
    print("output: ", output)


if __name__ == "__main__":
    worksize = 8
    master_ip = "127.0.0.1"
    master_port = "50001"
    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.float16

    mp.spawn(
        run_mm_all_reduce_base,
        args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype),
        nprocs=worksize,
    )


experimental_config = torch_npu.profiler._ExperimentalConfig(
	export_type=torch_npu.profiler.ExportType.Text,
	profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
	msprof_tx=False,
	aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
	l2_cache=False,
	op_attr=False,
	data_simplification=False,
	record_op_args=False,
	gc_detect_threshold=None
)

prof = torch_npu.profiler.profile(
	activities=[
		torch_npu.profiler.ProfilerActivity.CPU,
		torch_npu.profiler.ProfilerActivity.NPU
		],
	schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
	on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
	record_shapes=False,
	profile_memory=False,
	with_stack=False,
	with_modules=False,
	with_flops=False,
	experimental_config=experimental_config)
prof.start()
for step in range(steps):
	train_one_step()
	prof.step()
prof.stop()



# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_execute_sqrt)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./")

set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")

set(LIB_PATH $ENV{NPU_HOST_LIB})

# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub/")
    set(LIB_PATH1 "/usr/local/Ascend/ascend-toolkit/latest/atc/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
include_directories(
    ${INC_PATH}/runtime/include
    ${INC_PATH}/atc/include
    ${CUST_PKG_PATH}/include
)

# add host lib path
link_directories(
    ${LIB_PATH}
    ${LIB_PATH1}
    ${CUST_PKG_PATH}/lib
)

add_executable(execute_sqrt_op
    main.cpp
)

target_link_libraries(execute_sqrt_op
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    stdc++
)

install(TARGETS execute_sqrt_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})


#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/lib64


python3 gen_data.py

if [ $? -ne 0 ]; then
    echo "ERROR: generate input data failed!"
    return 1
fi
echo "INFO: generate input data success!"
set -e
rm -rf build
mkdir -p build
cmake -B build
cmake --build build -j
(
    cd build
    ./execute_sqrt_op
)

