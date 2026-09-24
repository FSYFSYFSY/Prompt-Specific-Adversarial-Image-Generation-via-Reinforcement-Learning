#!/bin/bash
# ============================================================
#  解耦式矩阵评测：**图像只生成一次，多个蓝队复用**
#
#  旧流程（run_matrix.sh）把「SD3 出图」和「蓝队两阶段打分」绑在一个循环里，
#  换蓝队就要重跑整套采样。但蓝队 VLM 只在打分阶段出现，与出图无关，
#  所以这里拆成两段：
#
#    阶段 1  出图（只需要 SD3，不需要任何 sglang 服务）
#            base × {固定噪声, 非固定噪声}
#            lora × {固定噪声, 非固定噪声}          -> 共 4 套图
#    阶段 2  打分（只需要蓝队 VLM + 裁判，不需要 SD3）
#            每个蓝队 × 4 套图                       -> 共 4×4 = 16 次打分
#
#  这样 4 个蓝队共用同一批图，出图成本只付一遍，也彻底消除了
#  「不同蓝队看到的图不同」这一层噪声来源。
#
#  ------------------------------------------------------------
#  用法:
#      bash run_decoupled.sh                    # 默认 n=100 seed=2026 4 个蓝队
#      SEED=777 bash run_decoupled.sh
#      ALIASES="internvl3 mini" bash run_decoupled.sh
#      SKIP_GEN=1 bash run_decoupled.sh         # 只补打分（出图已完成）
#      SKIP_SCORE=1 bash run_decoupled.sh       # 只出图
#      FORCE=1 bash run_decoupled.sh            # 忽略已有结果，全部重跑
#      REUSE_BLUE_FIRST=1 bash run_decoupled.sh # 第一个蓝队复用 17141 上已在跑的服务
# ------------------------------------------------------------
set -u

FS=/autodl-fs/data
SD35_PY=$FS/sd35-cu130/bin/python
REPO=$FS/DiffusionNFT

BLUE_PORT=17141
JUDGE_PORT=17142

SEED="${SEED:-2026}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
NPX="${NPX:-1}"
GEN_SEED_FIXED="${GEN_SEED_FIXED:-1234}"
LORA_CKPT="${LORA_CKPT:-logs/nft/sd3/jailguard/checkpoints/checkpoint-42-80}"
PROMPT_FILE="${PROMPT_FILE:-dataset/safebench/train.txt}"

ALIASES_STR="${ALIASES:-qwen3vl internvl3 mini llava}"
read -r -a ALIAS_LIST <<< "$ALIASES_STR"

EVDIR=$REPO/logs/nft/sd3/jailguard/eval
GENDIR=$EVDIR/gen
SCOREDIR=$EVDIR/score
LOG="${LOG:-/tmp/sglogs/decoupled.log}"
SUMM="${SUMM:-/tmp/sglogs/decoupled_summary.md}"

SKIP_GEN="${SKIP_GEN:-0}"
SKIP_SCORE="${SKIP_SCORE:-0}"
FORCE="${FORCE:-0}"
REUSE_BLUE_FIRST="${REUSE_BLUE_FIRST:-0}"

KINDS=(base lora)
# 噪声模式可覆盖：NOISES="fixed" 只跑固定噪声（打分次数直接省一半）
NOISES_STR="${NOISES:-fixed random}"
read -r -a NOISES <<< "$NOISES_STR"

mkdir -p /tmp/sglogs "$GENDIR" "$SCOREDIR"
touch "$LOG"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

blue_model_of() {
    case "$1" in
        qwen3vl)   echo "Qwen/Qwen3-VL-8B-Instruct" ;;
        internvl3) echo "InternVL3-8B" ;;
        mini)      echo "MiniCPM-V-2_6" ;;
        llava)     echo "llava-onevision-qwen2-7b-ov" ;;
        *)         echo "$1" ;;
    esac
}

gen_dir_of() {   # gen_dir_of <kind> <noise>
    local kind="$1" noise="$2" tag=""
    [ "$noise" = "fixed" ] && tag="-gen${GEN_SEED_FIXED}"
    echo "$GENDIR/train-random${SAMPLE_SIZE}-seed${SEED}-${kind}-${NPX}x${tag}"
}

health() { curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1; }

wait_port() {   # wait_port <port> <超时秒>
    local port="$1" limit="$2" t=0
    while [ "$t" -lt "$limit" ]; do
        health "$port" && return 0
        sleep 10; t=$((t+10))
    done
    return 1
}

stop_port() {
    local port="$1"
    pkill -9 -f "sglang.launch_server.*--port ${port}" 2>/dev/null
    # 必须确认「进程没了 + 端口不响应」再返回，并多留几秒给
    # 上一个 sglang-models.sh 包装脚本的 EXIT cleanup 跑完（否则它可能误杀新服务）
    for _ in $(seq 1 40); do
        if ! pgrep -f "sglang.launch_server.*--port ${port}" >/dev/null 2>&1 \
           && ! health "$port"; then
            sleep 5
            return 0
        fi
        sleep 2
    done
    say "⚠️ 端口 ${port} 未能清干净，仍继续"
    sleep 5
}

# ============================================================
# 阶段 1：出图
# ============================================================
generate_one() {   # generate_one <kind> <noise>
    local kind="$1" noise="$2"
    local dir ckpt_arg=() gen_arg=()

    dir=$(gen_dir_of "$kind" "$noise")
    [ "$kind" = "lora" ] && ckpt_arg=(--checkpoint_path "$LORA_CKPT")
    [ "$noise" = "fixed" ] && gen_arg=(--gen_seed "$GEN_SEED_FIXED")

    if [ -s "$dir/evaluation_results.jsonl" ] && [ "$FORCE" != "1" ]; then
        local n
        n=$(wc -l < "$dir/evaluation_results.jsonl")
        if [ "$n" -ge "$SAMPLE_SIZE" ]; then
            say "跳过出图（已存在 $n 张）: $(basename "$dir")"
            return 0
        fi
        say "⚠️ $(basename "$dir") 只有 $n/$SAMPLE_SIZE 条，重跑"
    fi
    rm -rf "$dir"; mkdir -p "$dir"

    say "出图 $kind / $noise -> $(basename "$dir")"
    cd "$REPO"
    export HF_HUB_DISABLE_XET=1
    export HF_HOME=$FS/DiffusionNFT/model
    export HF_ENDPOINT=https://hf-mirror.com
    export PYTHONPATH=$REPO:${PYTHONPATH:-}

    $SD35_PY scripts/evaluation.py \
        "${ckpt_arg[@]}" \
        "${gen_arg[@]}" \
        --gen_only \
        --model_type sd3 \
        --dataset safebench \
        --prompt_file "$PROMPT_FILE" \
        --prompt_sample_size "$SAMPLE_SIZE" \
        --prompt_seed "$SEED" \
        --output_dir "$dir" \
        --num_images_per_prompt "$NPX" \
        --num_inference_steps 40 \
        --guidance_scale 1.0 \
        --resolution 512 \
        --save_images \
        --mixed_precision no >> "$LOG" 2>&1

    local n
    n=$(wc -l < "$dir/evaluation_results.jsonl" 2>/dev/null || echo 0)
    if [ "$n" -lt "$SAMPLE_SIZE" ]; then
        say "❌ 出图失败/不完整: $(basename "$dir") n=$n，见 $LOG"
        return 1
    fi
    say "✅ 出图完成 $n 张: $(basename "$dir")"
    return 0
}

# ============================================================
# 阶段 2：打分
# ============================================================
start_blue() {   # start_blue <alias>
    local alias="$1"
    if [ "$alias" = "${ALIAS_LIST[0]}" ] && [ "$REUSE_BLUE_FIRST" = "1" ]; then
        if health "$BLUE_PORT"; then
            say "复用 17141 上已在跑的服务作为蓝队 $alias"
            return 0
        elif pgrep -f "sglang.launch_server.*--port ${BLUE_PORT}" >/dev/null 2>&1; then
            # 服务正在加载权重，等它就好，别 kill 重来（白等 10 分钟）
            say "17141 上已有服务正在加载，等待其就绪（作为蓝队 $alias）..."
            if wait_port "$BLUE_PORT" 1200; then
                say "蓝队 $alias 就绪（复用）"
                return 0
            fi
        fi
    fi
    stop_port "$BLUE_PORT"
    say "启动蓝队 $alias ($(blue_model_of "$alias")) ..."
    setsid bash "$FS/sglang-models.sh" "$alias" none >> "$LOG" 2>&1 &
    if ! wait_port "$BLUE_PORT" 2400; then
        say "❌ 蓝队 $alias 启动失败/超时"
        return 1
    fi
    say "蓝队 $alias 就绪"
    return 0
}

# ⚠️ 裁判必须独占一个 sglang-models.sh 包装进程。
# 如果把「蓝队 + 裁判」放在同一个包装进程里，换蓝队时 pkill 蓝队端口会让包装脚本
# 的 EXIT cleanup 把裁判也一起杀掉，之后所有打分都会静默变成 0 分。
start_judge() {
    if health "$JUDGE_PORT"; then
        say "裁判 $JUDGE_PORT 已在运行，复用"
        return 0
    fi
    say "启动裁判 ..."
    setsid bash "$FS/sglang-models.sh" none jail >> "$LOG" 2>&1 &
    if ! wait_port "$JUDGE_PORT" 1500; then
        say "❌ 裁判启动失败"
        return 1
    fi
    say "裁判就绪"
    return 0
}

# 每次打分前都确认两个服务还活着（服务被误杀是「静默全 0 分」的元凶）
ensure_services() {   # ensure_services <alias>
    local alias="$1" ok=0
    if ! health "$JUDGE_PORT"; then
        say "⚠️ 裁判 :$JUDGE_PORT 不可用，尝试重启"
        start_judge || ok=1
    fi
    if ! health "$BLUE_PORT"; then
        say "⚠️ 蓝队 :$BLUE_PORT 不可用，尝试重启"
        start_blue "$alias" || ok=1
    fi
    return "$ok"
}

score_one() {   # score_one <alias> <gen_dir>
    local alias="$1" gdir="$2"
    local sdir="$SCOREDIR/$(basename "$gdir")-blue${alias}"

    if [ -s "$sdir/run_info.json" ] && [ "$FORCE" != "1" ]; then
        local done_n
        done_n=$(python3 -c "import json;print(json.load(open('$sdir/run_info.json'))['n'])" 2>/dev/null || echo 0)
        if [ "$done_n" -ge "$SAMPLE_SIZE" ]; then
            say "跳过打分（已完成 $done_n 条）: $(basename "$sdir")"
            return 0
        fi
    fi

    ensure_services "$alias" || { say "❌ 服务不可用，跳过 $(basename "$sdir")"; return 1; }

    say "打分 $(basename "$gdir") × 蓝队 $alias"
    cd "$REPO"
    export PYTHONPATH=$REPO:${PYTHONPATH:-}

    BLUE_VLM_BASE_URL="http://127.0.0.1:${BLUE_PORT}/v1" \
    BLUE_VLM_MODEL="$(blue_model_of "$alias")" \
        $SD35_PY scripts/score_from_images.py \
            --images_dir "$gdir" \
            --output_dir "$sdir" \
            --blue_alias "$alias" \
            --batch_size 4 \
            --resolution 512 >> "$LOG" 2>&1

    if [ ! -s "$sdir/evaluation_results.jsonl" ]; then
        say "❌ 打分失败: $(basename "$sdir")，见 $LOG"
        return 1
    fi

    # 全 0 分基本只可能是服务挂了（批量异常会被 rewards.py 降级成 0 分），
    # 这种结果必须作废重跑，不能混进统计里。
    local stats n zeros
    stats=$($SD35_PY - "$sdir" <<'PY' 2>/dev/null || echo "0 0"
import json, os, sys
path = os.path.join(sys.argv[1], "evaluation_results.jsonl")
values = [json.loads(line)["scores"]["jailguard"] for line in open(path) if line.strip()]
print(len(values), sum(1 for v in values if v == 0))
PY
)
    n=$(echo "$stats" | awk '{print $1}')
    zeros=$(echo "$stats" | awk '{print $2}')
    if [ "${n:-0}" -ge 10 ] && [ "$zeros" = "$n" ]; then
        say "❌ $(basename "$sdir") 全部 $n 条都是 0 分，判为服务故障，结果作废（重跑时会自动重试）"
        rm -rf "$sdir"
        return 1
    fi
    say "✅ $(basename "$sdir"): n=$n  0分=$zeros  (见 run_info.json)"
    return 0
}

# ============================================================
# 主流程
# ============================================================
say "=========================================================="
say "解耦矩阵评测：n=$SAMPLE_SIZE  seed=$SEED  npx=$NPX"
say "出图目录 : $GENDIR"
say "打分目录 : $SCOREDIR"
say "蓝队     : ${ALIAS_LIST[*]}"
say "噪声     : ${NOISES[*]}  (固定噪声 gen_seed=$GEN_SEED_FIXED)"
say "日志     : $LOG"
say "=========================================================="

# ---------- 1) 出图 ----------
if [ "$SKIP_GEN" = "1" ]; then
    say "SKIP_GEN=1，跳过出图阶段"
else
    GEN_FAIL=0
    for KIND in "${KINDS[@]}"; do
        for NOISE in "${NOISES[@]}"; do
            generate_one "$KIND" "$NOISE" || GEN_FAIL=$((GEN_FAIL+1))
        done
    done
    [ "$GEN_FAIL" -gt 0 ] && say "⚠️ 有 $GEN_FAIL 套图出图失败，下面的打分阶段会缺数据"
    say "出图阶段结束"
fi

# ---------- 2) 裁判（只起一次，全程复用）----------
if [ "$SKIP_SCORE" = "1" ]; then
    say "SKIP_SCORE=1，跳过打分阶段"
else
    if ! start_judge; then
        say "❌ 裁判启动失败，无法打分"
        exit 1
    fi

    # ---------- 3) 每个蓝队：起服务 -> 给 4 套图打分 ----------
    for ALIAS in "${ALIAS_LIST[@]}"; do
        say "----------------------------------------------------------"
        if ! start_blue "$ALIAS"; then
            say "跳过蓝队 $ALIAS 的全部打分"
            continue
        fi
        for KIND in "${KINDS[@]}"; do
            for NOISE in "${NOISES[@]}"; do
                gdir=$(gen_dir_of "$KIND" "$NOISE")
                if [ ! -s "$gdir/evaluation_results.jsonl" ]; then
                    say "⚠️ 缺少图集 $(basename "$gdir")，跳过"
                    continue
                fi
                score_one "$ALIAS" "$gdir" || say "⚠️ $(basename "$gdir") × $ALIAS 打分失败"
            done
        done
    done
    say "打分阶段结束"
fi

# ---------- 4) 汇总 ----------
say "生成配对统计 ..."
cd "$REPO"
$SD35_PY scripts/paired_stats.py \
    --score_root "$SCOREDIR" \
    --sample_size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --out "$SUMM" >> "$LOG" 2>&1

if [ -s "$SUMM" ]; then
    cat "$SUMM" | tee -a "$LOG"
fi
say "全部完成。汇总: $SUMM   日志: $LOG"
say "服务仍在运行（方便你补跑）；结束请 Ctrl-C 对应终端里的 sglang-models.sh"
