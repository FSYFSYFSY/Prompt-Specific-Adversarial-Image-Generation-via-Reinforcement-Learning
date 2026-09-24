#!/bin/bash
set -euo pipefail

export HF_HUB_DISABLE_XET=1
export HF_HOME=/autodl-fs/data/DiffusionNFT/model
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/autodl-fs/data/DiffusionNFT:${PYTHONPATH:-}

SD35_PY=/autodl-fs/data/sd35-cu130/bin/python
cd /autodl-fs/data/DiffusionNFT

# Use the existing configured services as-is.
# This script assumes sglang.sh has already been started in another terminal.
bash /autodl-fs/data/sglang.sh &
SG_LANG_PID=$!

# Give the judge services a moment to begin serving.
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:17141/health >/dev/null 2>&1 && curl -fsS http://127.0.0.1:17142/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

# Final health check before starting validation.
if ! curl -fsS http://127.0.0.1:17141/health >/dev/null 2>&1; then
  echo "17141 not ready" >&2
  exit 1
fi
if ! curl -fsS http://127.0.0.1:17142/health >/dev/null 2>&1; then
  echo "17142 not ready" >&2
  exit 1
fi

$SD35_PY /autodl-fs/data/DiffusionNFT/scripts/evaluation.py \
  --checkpoint_path /autodl-fs/data/DiffusionNFT/logs/nft/sd3/jailguard/checkpoints/checkpoint-42-80 \
  --model_type sd3 \
  --dataset safebench \
  --output_dir /autodl-fs/data/DiffusionNFT/logs/nft/sd3/jailguard/eval/validation-4x \
  --num_images_per_prompt 4 \
  --num_inference_steps 40 \
  --guidance_scale 1.0 \
  --resolution 512 \
  --save_images \
  --mixed_precision no

kill $SG_LANG_PID 2>/dev/null || true
wait $SG_LANG_PID 2>/dev/null || true
