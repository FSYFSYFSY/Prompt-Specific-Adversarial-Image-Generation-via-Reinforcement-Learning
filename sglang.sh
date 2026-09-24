export HF_HUB_DISABLE_XET=1
export HF_HOME=/autodl-fs/data/DiffusionNFT/model
export HF_ENDPOINT=https://hf-mirror.com

ENV_PYTHON=/autodl-fs/data/sglang-cu130/bin/python

# 关键修复1：系统 /usr/local/cuda 是 12.8，而 Blackwell(sm_120) 需要 CUDA>=12.9，
# flashinfer 因此拒绝 sm_120 导致 sglang 崩溃。显式指向环境自带的 cu13 工具链。
export CUDA_HOME=/autodl-fs/data/sglang-cu130/lib/python3.12/site-packages/nvidia/cu13
# 关键修复2：flashinfer JIT 编译需要 ninja/nvcc，把环境 bin 加入 PATH
export PATH=/autodl-fs/data/sglang-cu130/bin:$CUDA_HOME/bin:$PATH
# 首次 JIT 编译大量内核，用多核并行加速（机器 25 核）
export MAX_JOBS=12
# 关键修复3：容器自带 OMP_NUM_THREADS=0，libgomp 会报 "Invalid value"
# 并退化为单线程，影响 CPU 侧算子与 tokenizer 吞吐，这里显式设为合法值。
export OMP_NUM_THREADS=8

trap 'echo "清理进程..."; kill $(jobs -p) 2>/dev/null' EXIT

# ===== Qwen3-VL-8B-Instruct =====
# 串行流程下真实并发极低，max-running-requests不需要留大空间
# cuda-graph-max-bs=4：只覆盖 bs=1,2,4，贴合"单条请求串行"的真实负载
CUDA_VISIBLE_DEVICES=0 $ENV_PYTHON -m sglang.launch_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --port 17141 \
    --mem-fraction-static 0.40 \
    --context-length 2048 \
    --max-running-requests 4 \
    --cuda-graph-max-bs-decode 4 &

until curl -s http://localhost:17141/health >/dev/null; do sleep 2; done
sleep 10

# ===== JailJudge-guard =====
# 同样低并发，且是Qwen答复之后才触发的下游打分，
# 保留小范围graph capture换取单条打分的响应速度
CUDA_VISIBLE_DEVICES=0 $ENV_PYTHON -m sglang.launch_server \
    --model-path usail-hkust/JailJudge-guard \
    --port 17142 \
    --mem-fraction-static 0.3 \
    --max-running-requests 4 \
    --cuda-graph-max-bs-decode 4 &

wait