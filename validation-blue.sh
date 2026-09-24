#!/bin/bash
# ============================================================
#  100-sample validation（蓝队模型可自由切换）
#
#  前置：先在另一个终端跑
#      bash /autodl-fs/data/sglang-models.sh <蓝队别名> jail
#  本脚本只负责跑 evaluation.py，不负责起服务。
#
#  ------------------------------------------------------------
#  用法:
#      bash validation-blue.sh                 # 默认蓝队 internvl3, seed 450, 1x, LoRA
#      BLUE_ALIAS=mini  bash validation-blue.sh
#      BLUE_ALIAS=llava bash validation-blue.sh
#      BLUE_ALIAS=qwen3vl bash validation-blue.sh      # 复现原来的基线
#      CKPT= bash validation-blue.sh                   # 不带 LoRA（base 模型）
#      SEED=42  bash validation-blue.sh
#      NPX=4    bash validation-blue.sh                # 每个 prompt 出 4 张
#      CKPT=logs/.../checkpoint-42-40 bash validation-blue.sh
#
#  base / lora：
#      CKPT 非空 -> 走 LoRA，读 $CKPT/lora
#      CKPT 为空 -> base 模型，不加 --checkpoint_path
# ============================================================
set -euo pipefail

FS=/autodl-fs/data
SD35_PY=$FS/sd35-cu130/bin/python
REPO=$FS/DiffusionNFT

# ---- 蓝队模型（必须和 sglang-models.sh 启动时用的别名一致）----
BLUE_ALIAS="${BLUE_ALIAS:-internvl3}"
BLUE_PORT="${BLUE_PORT:-17141}"
case "$BLUE_ALIAS" in
    qwen3vl)   BLUE_MODEL="Qwen/Qwen3-VL-8B-Instruct" ;;
    internvl3) BLUE_MODEL="InternVL3-8B" ;;
    mini)      BLUE_MODEL="MiniCPM-V-2_6" ;;
    llava)     BLUE_MODEL="llava-onevision-qwen2-7b-ov" ;;
    *)         BLUE_MODEL="$BLUE_ALIAS" ;;
esac

JUDGE_PORT=17142

# ---- 数据 / 输出 ----
PROMPT_FILE="${PROMPT_FILE:-dataset/safebench/train.txt}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
SEED="${SEED:-450}"
NPX="${NPX:-1}"
CKPT="${CKPT-logs/nft/sd3/jailguard/checkpoints/checkpoint-42-80}"
# 有 LoRA 还是 base：由 CKPT 是否为空决定
if [ -n "$CKPT" ]; then KIND="lora"; else KIND="base"; fi
# GEN_SEED 非空 -> 固定扩散采样噪声（common random numbers），并给输出目录加后缀
GEN_SEED="${GEN_SEED:-}"
GEN_ARG=(); GEN_TAG=""
if [ -n "$GEN_SEED" ]; then
    GEN_ARG=(--gen_seed "$GEN_SEED")
    GEN_TAG="-gen${GEN_SEED}"
fi
OUT="${OUT:-logs/nft/sd3/jailguard/eval/train-random100-seed${SEED}-twostage-judgeonly-${KIND}-blue${BLUE_ALIAS}-${NPX}x${GEN_TAG}}"

export HF_HUB_DISABLE_XET=1
export HF_HOME=$FS/DiffusionNFT/model
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$REPO:${PYTHONPATH:-}

# rewards.py 就是读这两个变量来选蓝队
export BLUE_VLM_BASE_URL="http://127.0.0.1:${BLUE_PORT}/v1"
export BLUE_VLM_MODEL="$BLUE_MODEL"

cd "$REPO"

# ---- 前置检查 ----
if ! curl -sf "http://127.0.0.1:${BLUE_PORT}/health" >/dev/null 2>&1; then
    echo "❌ 蓝队服务 :${BLUE_PORT} 未就绪"
    echo "   请先在另一个终端运行: bash $FS/sglang-models.sh ${BLUE_ALIAS} jail"
    exit 1
fi
if ! curl -sf "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; then
    echo "❌ 裁判服务 :${JUDGE_PORT} 未就绪"
    echo "   请先在另一个终端运行: bash $FS/sglang-models.sh ${BLUE_ALIAS} jail"
    exit 1
fi

echo "============================================================"
echo " 蓝队 : $BLUE_MODEL   ($BLUE_VLM_BASE_URL)"
echo " 裁判 : :${JUDGE_PORT}  (usail-hkust/JailJudge-guard)"
echo " 采样 : $PROMPT_FILE  取 $SAMPLE_SIZE 条  seed=$SEED"
echo " 出图 : 每个 prompt $NPX 张"
if [ "$KIND" = "lora" ]; then
    echo " 权重 : LoRA  $CKPT/lora"
else
    echo " 权重 : base（不带 LoRA）"
fi
if [ -n "$GEN_SEED" ]; then
    echo " 噪声 : 已固定 --gen_seed=$GEN_SEED（base/lora 用同一份初始噪声）"
fi
echo " 输出 : $OUT"
echo "============================================================"
echo

CKPT_ARG=()
[ -n "$CKPT" ] && CKPT_ARG=(--checkpoint_path "$CKPT")

$SD35_PY scripts/evaluation.py \
    "${CKPT_ARG[@]}" \
    "${GEN_ARG[@]}" \
    --model_type sd3 \
    --dataset safebench \
    --prompt_file "$PROMPT_FILE" \
    --prompt_sample_size "$SAMPLE_SIZE" \
    --prompt_seed "$SEED" \
    --output_dir "$OUT" \
    --num_images_per_prompt "$NPX" \
    --num_inference_steps 40 \
    --guidance_scale 1.0 \
    --resolution 512 \
    --save_images \
    --mixed_precision no

echo
echo "=== 平均分 ==="
cat "$OUT/average_scores.json"
echo
