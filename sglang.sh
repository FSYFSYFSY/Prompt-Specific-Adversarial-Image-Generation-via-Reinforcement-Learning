export HF_HUB_DISABLE_XET=1
export HF_HOME=/root/autodl-tmp/DiffusionNFT/model
export HF_ENDPOINT=https://hf-mirror.com

ENV_PYTHON=/root/autodl-tmp/sglang/bin/python

# Qwen3-VL-8B-Instruct 
CUDA_VISIBLE_DEVICES=1 $ENV_PYTHON -m sglang.launch_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --port 17141 \
    --mem-fraction-static 0.57 \
    --context-length 2048 &

# JailJudge-guard 
CUDA_VISIBLE_DEVICES=1 $ENV_PYTHON -m sglang.launch_server \
    --model-path usail-hkust/JailJudge-guard \
    --port 17142 \
    --mem-fraction-static 0.30 &

wait