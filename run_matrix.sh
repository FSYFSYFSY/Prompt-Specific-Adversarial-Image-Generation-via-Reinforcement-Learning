#!/bin/bash
# ============================================================
#  蓝队模型 × (base / LoRA) 矩阵评测
#
#  对每个蓝队模型：启一次服务 -> 先跑 base（无 LoRA）-> 再跑 LoRA
#  裁判只启动一次，一直复用。
#
#  用法:
#      bash run_matrix.sh                          # 默认 4 个模型全跑（seed 450）
#      SEED=451 bash run_matrix.sh                 # 换样本（100 条）
#      bash run_matrix.sh mini llava               # 只跑指定模型
#      REUSE_BLUE=1 bash run_matrix.sh qwen3vl ... # 第一个模型复用 17141 上已在跑的服务
#      LORA_CKPT=logs/.../checkpoint-42-80 bash run_matrix.sh
#      MATRIX_LOG=/tmp/x.log MATRIX_SUMM=/tmp/x.txt SEED=451 bash run_matrix.sh
# ============================================================
set -u

FS=/autodl-fs/data
EVALDIR=$FS/DiffusionNFT/logs/nft/sd3/jailguard/eval
BLUE_PORT=17141
JUDGE_PORT=17142
LOG="${MATRIX_LOG:-/tmp/sglogs/matrix.log}"
SUMM="${MATRIX_SUMM:-/tmp/sglogs/matrix_summary.txt}"

MODELS=("${@:-}")
[ -z "${MODELS[0]:-}" ] && MODELS=(qwen3vl internvl3 mini llava)

SEED="${SEED:-450}"
NPX="${NPX:-1}"
LORA_CKPT="${LORA_CKPT:-logs/nft/sd3/jailguard/checkpoints/checkpoint-42-80}"
# GEN_SEED 非空 -> 固定扩散采样噪声，base/lora 用同一份初始噪声（配对方差大幅下降）
GEN_SEED="${GEN_SEED:-}"
GEN_TAG=""
if [ -n "$GEN_SEED" ]; then
    export GEN_SEED
    GEN_TAG="-gen${GEN_SEED}"
fi

mkdir -p /tmp/sglogs
: > "$SUMM"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

stop_port() {
    local port="$1"
    pkill -9 -f "sglang.launch_server.*--port ${port}" 2>/dev/null
    # 必须确认「进程没了 + 端口不响应」再返回，并额外留几秒，
    # 让上一个 sglang-models.sh 包装脚本的 EXIT cleanup 跑完。
    # 否则包装脚本的清理动作可能晚于新服务的启动，把新服务误杀。
    for _ in $(seq 1 40); do
        if ! pgrep -f "sglang.launch_server.*--port ${port}" >/dev/null 2>&1 \
           && ! curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            sleep 5
            return 0
        fi
        sleep 2
    done
    say "⚠️ 端口 ${port} 未能清干净，仍继续"
    sleep 5
}

wait_port() {   # wait_port <port> <超时秒>
    local port="$1" limit="$2" t=0
    while [ "$t" -lt "$limit" ]; do
        curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && return 0
        sleep 10; t=$((t+10))
    done
    return 1
}

# ---------- 1) 裁判（只起一次）----------
if curl -sf "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; then
    say "裁判 ${JUDGE_PORT} 已在运行，复用"
else
    say "启动裁判 ..."
    setsid bash "$FS/sglang-models.sh" none jail >> "$LOG" 2>&1 &
    wait_port "$JUDGE_PORT" 1500 || { say "❌ 裁判启动失败"; exit 1; }
    say "裁判就绪"
fi

# ---------- 2) 逐个模型 ----------
FIRST=1
for ALIAS in "${MODELS[@]}"; do
    say "=================================================="
    say "蓝队模型 = $ALIAS"

    if [ "$FIRST" = "1" ] && [ "${REUSE_BLUE:-0}" = "1" ] \
       && pgrep -f "sglang.launch_server.*--port ${BLUE_PORT}" >/dev/null 2>&1; then
        say "复用 17141 上已在启动的服务（REUSE_BLUE=1）"
    else
        stop_port "$BLUE_PORT"
        say "启动蓝队 $ALIAS ..."
        setsid bash "$FS/sglang-models.sh" "$ALIAS" none >> "$LOG" 2>&1 &
    fi
    FIRST=0

    if ! wait_port "$BLUE_PORT" 2400; then
        say "❌ 蓝队 $ALIAS 启动失败/超时，跳过整个模型"
        printf "%-10s %-6s %-9s %s\n" "$ALIAS" "base"  "-"  "START_FAILED" >> "$SUMM"
        printf "%-10s %-6s %-9s %s\n" "$ALIAS" "lora"  "-"  "START_FAILED" >> "$SUMM"
        continue
    fi
    say "蓝队 $ALIAS 就绪"

    for KIND in base lora; do
        if [ "$KIND" = "lora" ]; then
            CK_ARG="$LORA_CKPT"
        else
            CK_ARG=""          # 空 => base 模型，不带 LoRA
        fi

        say "--- $ALIAS / $KIND 开始评测"
        BLUE_ALIAS="$ALIAS" SEED="$SEED" NPX="$NPX" CKPT="$CK_ARG" \
            bash "$FS/validation-blue.sh" >> "$LOG" 2>&1

        OUT="$EVALDIR/train-random100-seed${SEED}-twostage-judgeonly-${KIND}-blue${ALIAS}-${NPX}x${GEN_TAG}"
        N=$(wc -l < "$OUT/evaluation_results.jsonl" 2>/dev/null || echo 0)
        NZ=$(python3 -c "
import json
print(sum(1 for l in open('$OUT/evaluation_results.jsonl') if l.strip() and json.loads(l)['scores']['jailguard']>0))
" 2>/dev/null || echo NA)
        SCORE=$(python3 -c "
import json;print(round(json.load(open('$OUT/average_scores.json'))['jailguard'],4))
" 2>/dev/null || echo NA)

        say "--- $ALIAS / $KIND : jailguard=$SCORE  非零 $NZ/$N"
        if [ "$NZ" = "0" ] && [ "$N" != "0" ]; then
            say "⚠️  $ALIAS/$KIND 全 0 分，怀疑服务挂了，请查 $LOG"
        fi
        printf "%-10s %-6s %-9s %s/%s\n" "$ALIAS" "$KIND" "$SCORE" "$NZ" "$N" >> "$SUMM"
    done
done

# ---------- 3) 汇总 ----------
say "全部完成"
{
    echo "================= 汇总 ================="
    echo "seed=$SEED  npx=$NPX  gen_seed=${GEN_SEED:-未固定}"
    printf "%-10s %-6s %-9s %s\n" "蓝队" "权重" "jailguard" "非零/总数"
    cat "$SUMM"
} | tee -a "$LOG"
