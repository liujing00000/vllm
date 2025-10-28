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


#include <iostream>
#include <vector>
#include <thread>
#include "aclnnop/aclnn_matmul_all_reduce_v2.h"

int ndev = 8;

#define CHECK_RET(cond, return_expr) \
do {                               \
    if (!(cond)) {                   \
    return_expr;                   \
    }                                \
} while (0)

#define LOG_PRINT(message, ...)     \
do {                              \
    printf(message, ##__VA_ARGS__); \
} while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i: shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int launchOneThreadMatmulAllReduce(Args &args) {
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcom_name[128];
    ret = HcclGetCommName(args.hcclComm, hcom_name);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
            args.context);

    std::vector<int64_t> x1Shape = {32, 64};
    std::vector<int64_t> x2Shape = {64, 128};
    std::vector<int64_t> biasShape = {128};
    std::vector<int64_t> x3Shape = {32, 128};
    std::vector<int64_t> outShape = {32, 128};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *x3DeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *x3 = nullptr;
    aclTensor *out = nullptr;

    int64_t commTurn = 0;
    int64_t streamMode = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long x3ShapeSize = GetShapeSize(x3Shape);
    long long outShapeSize = GetShapeSize(outShape);
    std::vector<int16_t> x1HostData(x1ShapeSize, 1);
    std::vector<int16_t> x2HostData(x2ShapeSize, 1);
    std::vector<int16_t> biasHostData(biasShapeSize, 1);
    std::vector<int16_t> x3HostData(x3ShapeSize, 1);
    std::vector<int16_t> outHostData(outShapeSize, 0);
    // 创建 tensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 调用第一段接口, x3位置传入out
    ret = aclnnMatmulAllReduceV2GetWorkspaceSize(x1, x2, bias, x3, hcom_name, "sum", commTurn,
                                                 streamMode, out, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMatmulAllReduceV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用第二段接口
    ret = aclnnMatmulAllReduceV2(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulAllReduceV2 failed. ERROR: %d\n", ret); return ret);
    //（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("device%d aclnnMatmulAllReduceV2 execute success \n", args.rankId);
    // 释放device资源，需要根据具体API的接口定义修改
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (x3 != nullptr) {
        aclDestroyTensor(x3);
    }
    if (out != nullptr) {
        aclDestroyTensor(out);
    }
    if (x1DeviceAddr != nullptr) {
        aclrtFree(x1DeviceAddr);
    }
    if (x2DeviceAddr != nullptr) {
        aclrtFree(x2DeviceAddr);
    }
    if (biasDeviceAddr != nullptr) {
        aclrtFree(biasDeviceAddr);
    }
    if (x3DeviceAddr != nullptr) {
        aclrtFree(x3DeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(args.stream);
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return 0;
}

int main(int argc, char *argv[]) {
    int ret;
    int32_t devices[ndev];
    for (int i = 0; i < ndev; i++) {
        devices[i] = i;
    }
    HcclComm comms[128];
    ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    // 初始化集合通信域
    for (int i = 0; i < ndev; i++) {
        ret = aclrtSetDevice(devices[i]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    Args args[ndev];
    aclrtStream stream[ndev];
    aclrtContext context[ndev];
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    }
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadMatmulAllReduce,
                              std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}


# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.14)

# 设置工程名
project(ACLNN_EXAMPLE)

# Compile options
add_compile_options(-std=c++11)

# 设置编译选项
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")    
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 设置可执行文件名（如opapi_test），并指定待运行算子文件*.cpp所在目录
add_executable(opapi_test
               test_add.cpp) 

# 设置ASCEND_PATH（CANN软件包目录，请根据实际路径修改）和INCLUDE_BASE_DIR（头文件目录）
if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")      
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()
set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
include_directories(
    ${INCLUDE_BASE_DIR}
    ${INCLUDE_BASE_DIR}/aclnn
)

# 设置链接的库文件路径
target_link_libraries(opapi_test PRIVATE
                      ${ASCEND_PATH}/lib64/libascendcl.so
                      ${ASCEND_PATH}/lib64/libnnopbase.so
                      ${ASCEND_PATH}/lib64/libopapi.so)

# 可执行文件在CMakeLists文件所在目录的bin目录下
install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})


# 设置链接的库文件路径
find_package(Threads REQUIRED)
target_link_libraries(opapi_test PRIVATE
                      ${ASCEND_PATH}/lib64/libascendcl.so
                      ${ASCEND_PATH}/lib64/libnnopbase.so
                      ${ASCEND_PATH}/lib64/libopapi.so
                      ${ASCEND_PATH}/lib64/libhccl.so      # 集合通信库文件
                      ${CMAKE_THREAD_LIBS_INIT})           # 多线程依赖的库文件


mkdir -p build 
cd build
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make
cd bin
./opapi_test



#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {16, 32};
  std::vector<int64_t> mat2Shape = {32, 16};
  std::vector<int64_t> outShape = {16, 16};
  void* selfDeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData(512, 1);
  std::vector<float> mat2HostData(512, 1);
  std::vector<float> outHostData(256, 0);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mat2 aclTensor
  ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  int8_t cubeMathType = 1;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // 调用aclnnMatmul第一段接口
  ret = aclnnMatmulGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMatmul第二段接口
  ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(mat2);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(mat2DeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
