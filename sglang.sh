export HF_HUB_DISABLE_XET=1
export HF_HOME=/root/autodl-tmp/DiffusionNFT/model
export HF_ENDPOINT=https://hf-mirror.com

ENV_PYTHON=/root/autodl-tmp/sglang/bin/python

trap 'echo "清理进程..."; kill $(jobs -p) 2>/dev/null' EXIT

# ===== Qwen3-VL-8B-Instruct =====
# 串行流程下真实并发极低，max-running-requests不需要留大空间
# cuda-graph-max-bs=4：只覆盖 bs=1,2,4，贴合"单条请求串行"的真实负载
CUDA_VISIBLE_DEVICES=1 $ENV_PYTHON -m sglang.launch_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --port 17141 \
    --mem-fraction-static 0.5 \
    --context-length 2048 \
    --max-running-requests 4 \
    --cuda-graph-max-bs-decode 4 &

until curl -s http://localhost:17141/health >/dev/null; do sleep 2; done
sleep 10

# ===== JailJudge-guard =====
# 同样低并发，且是Qwen答复之后才触发的下游打分，
# 保留小范围graph capture换取单条打分的响应速度
CUDA_VISIBLE_DEVICES=1 $ENV_PYTHON -m sglang.launch_server \
    --model-path usail-hkust/JailJudge-guard \
    --port 17142 \
    --mem-fraction-static 0.8 \
    --max-running-requests 4 \
    --cuda-graph-max-bs-decode 4 &

wait