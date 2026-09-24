#!/bin/bash
# ============================================================
#  蓝队 VLM + 裁判模型 一体启动脚本（可自由切换）
#
#  沿用 sglang.sh 的三处关键修复，端口与 rewards.py 保持一致：
#    蓝队 VLM : 17141   <- rewards.py 里 BLUE_VLM_BASE_URL 的默认值
#    裁判模型 : 17142   <- rewards.py 里硬编码
#
#  ------------------------------------------------------------
#  用法:
#     bash sglang-models.sh [蓝队] [裁判]
#
#     bash sglang-models.sh                # qwen3vl + jail  (等价于原 sglang.sh)
#     bash sglang-models.sh internvl3      # 蓝队换成 InternVL3-8B
#     bash sglang-models.sh mini           # 蓝队换成 MiniCPM-V-2_6
#     bash sglang-models.sh llava          # 蓝队换成 llava-onevision
#     bash sglang-models.sh llava jail     # 蓝队 llava + 裁判 jail
#     bash sglang-models.sh internvl3 none # 只起蓝队，不起裁判
#     bash sglang-models.sh qwen3vl internvl3   # 裁判也换成 VLM
#
#  ------------------------------------------------------------
#  可用别名:
#     qwen3vl   Qwen/Qwen3-VL-8B-Instruct       （HF 缓存，原默认）
#     internvl3 /root/.../models/InternVL3-8B
#     mini      /root/.../models/MiniCPM-V-2_6
#     llava     /root/.../models/llava-onevision-qwen2-7b-ov
#     jail      usail-hkust/JailJudge-guard     （裁判默认）
#     none      不起来
#     <其它>    任意本地路径或 HF repo id（会带上 --trust-remote-code）
#
#  ------------------------------------------------------------
#  可覆盖的环境变量:
#     BLUE_PORT=17141  JUDGE_PORT=17142
#     BLUE_MEM=0.30    JUDGE_MEM=0.22      # mem-fraction-static
#     MAX_REQ=4        GRAPH_BS=4          # 串行低并发调优，同原脚本
#     CTX_LEN=4096                         # 统一覆盖 context-length
#
#  ⚠️ 显存预算（本机 RTX PRO 6000，97.9 GiB 可用）
#     mem-fraction-static 是「权重 + KV 池」占整卡的比例，sglang 会一次性占满。
#     两个服务要和 DiffusionNFT 的 SD3 评测（约 35 GiB）共存，所以不能按原来的
#     0.40 + 0.30 = 0.70 配，否则 66 + 35 = 101 GiB > 95 GiB，会 CUDA OOM
#     （蓝队进程会先被 OOM 打死，日志里是 SIGQUIT / "one child failed"）。
#         蓝队 0.30 -> ~29 GiB（权重 15 + KV 14）
#         裁判 0.22 -> ~21 GiB（权重 16 + KV 5）
#         合计 ~50 GiB，给 SD3 留 ~45 GiB
#     如果只起服务、不跑 SD3，可以自己调回 BLUE_MEM=0.40 JUDGE_MEM=0.30。
# ============================================================
set -u

# ---- 路径（/root/autodl-fs 是 /autodl-fs/data 的软链）----
FS=/autodl-fs/data
ENV_PYTHON=$FS/sglang-cu130/bin/python
MODELS=$FS/models
HF_CACHE=$FS/DiffusionNFT/model/hub

# ---- 参数 ----
BLUE_ALIAS="${1:-qwen3vl}"
JUDGE_ALIAS="${2:-jail}"

BLUE_PORT="${BLUE_PORT:-17141}"
JUDGE_PORT="${JUDGE_PORT:-17142}"
BLUE_MEM="${BLUE_MEM:-0.30}"
JUDGE_MEM="${JUDGE_MEM:-0.22}"
MAX_REQ="${MAX_REQ:-4}"
GRAPH_BS="${GRAPH_BS:-4}"
# 额外参数（可选）。裁判实测会周期性挂在 CUDA illegal memory access
# (batch_result_processor.py:811 copy_done.synchronize())，09-16 与 09-18 各一次，
# 关掉 CUDA graph / overlap schedule 可绕开这类图捕获+异步拷贝的 bug：
#   JUDGE_EXTRA_ARGS="--disable-overlap-schedule --disable-cuda-graph"
BLUE_EXTRA_ARGS="${BLUE_EXTRA_ARGS:-}"
JUDGE_EXTRA_ARGS="${JUDGE_EXTRA_ARGS:-}"

# ------------------------------------------------------------
# 环境变量（sglang.sh 里的三处关键修复，别删）
# ------------------------------------------------------------
export HF_HUB_DISABLE_XET=1
export HF_HOME=$FS/DiffusionNFT/model
export HF_ENDPOINT=https://hf-mirror.com

# 关键修复1：系统 /usr/local/cuda 是 12.8，而 Blackwell(sm_120) 需要 CUDA>=12.9，
# flashinfer 因此拒绝 sm_120 导致 sglang 崩溃。显式指向环境自带的 cu13 工具链。
export CUDA_HOME=$FS/sglang-cu130/lib/python3.12/site-packages/nvidia/cu13
# 关键修复2：flashinfer JIT 编译需要 ninja/nvcc，把环境 bin 加入 PATH
export PATH=$FS/sglang-cu130/bin:$CUDA_HOME/bin:$PATH
# 首次 JIT 编译大量内核，用多核并行加速（机器 22~25 核）
export MAX_JOBS=12
# 关键修复3：容器自带 OMP_NUM_THREADS=0，libgomp 会报 "Invalid value"
# 并退化为单线程，影响 CPU 侧算子与 tokenizer 吞吐，这里显式设为合法值。
export OMP_NUM_THREADS=8

# ------------------------------------------------------------
# 别名 -> 模型
# ------------------------------------------------------------
M_PATH=""; M_ARGS=""; M_CTX=2048; M_NAME=""

resolve_model() {
    case "$1" in
        qwen3vl)
            M_PATH="Qwen/Qwen3-VL-8B-Instruct"
            M_NAME="Qwen/Qwen3-VL-8B-Instruct"
            M_CTX=2048; M_ARGS="" ;;
        internvl3)
            M_PATH="$MODELS/InternVL3-8B"
            M_NAME="InternVL3-8B"
            M_CTX=4096; M_ARGS="--trust-remote-code" ;;
        mini)
            M_PATH="$MODELS/MiniCPM-V-2_6"
            M_NAME="MiniCPM-V-2_6"
            M_CTX=4096; M_ARGS="--trust-remote-code" ;;
        llava)
            M_PATH="$MODELS/llava-onevision-qwen2-7b-ov"
            M_NAME="llava-onevision-qwen2-7b-ov"
            # anyres_max_9 一张图可能展开成上千个视觉 token，给宽一点
            M_CTX=8192; M_ARGS="--trust-remote-code" ;;
        jail)
            M_PATH="usail-hkust/JailJudge-guard"
            M_NAME="usail-hkust/JailJudge-guard"
            M_CTX=2048; M_ARGS="" ;;          # 该模型 max_position_embeddings 就是 2048
        none)
            M_PATH="" ;;
        *)
            M_PATH="$1"
            M_NAME="$(basename "$1")"
            M_CTX=4096; M_ARGS="--trust-remote-code" ;;
    esac
    [ -n "${CTX_LEN:-}" ] && M_CTX="$CTX_LEN"
}

# ------------------------------------------------------------
# 预检查
# ------------------------------------------------------------
check_llava_siglip() {
    local f="$HF_CACHE/models--google--siglip-so400m-patch14-384/snapshots"
    if ! ls "$f"/*/model.safetensors >/dev/null 2>&1; then
        echo "⚠️  警告: llava-onevision 需要 SigLIP 视觉塔 (3.5GB)，本地缓存里没有。"
        echo "    它会在 load_weights() 里调 SiglipVisionModel.from_pretrained()，"
        echo "    走 hf-mirror 很容易卡死。建议先补（modelscope 有同名镜像）："
        echo "      https://www.modelscope.cn/models/AI-ModelScope/siglip-so400m-patch14-384"
    fi
}

# 检查模型是否可用：本地目录 / HF 缓存
check_model() {
    local path="$1" tag="$2"
    if [ -e "$path" ]; then
        [ -f "$path/config.json" ] \
            && echo "  ✓ $tag 本地模型: $path" \
            || echo "  ⚠️ $tag $path 里没有 config.json，可能不完整"
        return
    fi
    local cached="$HF_CACHE/models--${path//\//--}"
    if ls "$cached"/snapshots/*/config.json >/dev/null 2>&1; then
        echo "  ✓ $tag HF 缓存命中: $path"
    else
        echo "  ⚠️ $tag $path 不在 HF 缓存里，启动时会联网下载"
        echo "     注意：hf-mirror 这条线路实测只有 97KB/s~1.5MB/s 且会卡死；"
        echo "     如需下载建议先 source /etc/network_turbo 或改用 modelscope。"
    fi
}

# ------------------------------------------------------------
# 启动 / 等待 / 清理
# ------------------------------------------------------------
BLUE_PID=""; JUDGE_PID=""

# 按「进程组」清理服务。setsid 之后服务是独立的会话/进程组组长（PGID == PID），
# 所以 kill -PGID 能一次带走 scheduler / tokenizer / detokenizer 全部子进程。
kill_tree() {
    local pid="$1" sig="$2" pgid
    [ -z "$pid" ] && return 0
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ]; then
        kill -"$sig" -"$pgid" 2>/dev/null
    else
        kill -"$sig" "$pid" 2>/dev/null
    fi
}

cleanup() {
    echo
    echo ">>> 清理进程..."
    kill_tree "$BLUE_PID"  TERM
    kill_tree "$JUDGE_PID" TERM
    sleep 4
    kill_tree "$BLUE_PID"  KILL
    kill_tree "$JUDGE_PID" KILL
    # 注意：这里【绝对不能】用 pkill -f "…--port 17141" 兜底！
    # 本脚本退出时可能已经有人在同一端口起了新服务（比如 run_matrix.sh 换蓝队），
    # 按端口扫会把别人的新服务一起杀掉（表现为新服务无报错直接被 Killed）。
    # 进程组 kill 已经足够，端口残留交给调用方处理。
    echo ">>> 已清理。"
}
trap cleanup EXIT INT TERM

SERVER_PID=""

start_server() {
    # start_server <标签> <端口> <mem> <模型路径> <name> <ctx> <args>
    # 注意：这里不能用 $(start_server ...) 命令替换来取 PID！
    # 后台 server 会继承命令替换那个管道的写端，管道永不关闭，$(...) 永不返回。
    # 所以改为把 PID 写进全局变量 SERVER_PID。
    local tag="$1" port="$2" mem="$3" path="$4" name="$5" ctx="$6" args="$7"
    echo ">>> 启动 $tag : $name"
    echo "    端口=$port  mem-fraction-static=$mem  context-length=$ctx  $args"
    # setsid: 让每个服务独占进程组，方便整组 kill
    setsid $ENV_PYTHON -m sglang.launch_server \
        --model-path "$path" \
        --served-model-name "$name" \
        --host 127.0.0.1 \
        --port "$port" \
        --mem-fraction-static "$mem" \
        --context-length "$ctx" \
        --max-running-requests "$MAX_REQ" \
        --cuda-graph-max-bs-decode "$GRAPH_BS" \
        $args &
    SERVER_PID=$!
}

wait_ready() {
    # wait_ready <标签> <端口> <pid>
    local tag="$1" port="$2" pid="$3" n=0
    printf "    等待 %s :%s 就绪" "$tag" "$port"
    while true; do
        # 注意要用 -f：/health 未就绪时返回 503，光 curl -s 是判不出来的
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo " ✅ 就绪"; return 0; }
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo " ❌ 进程已退出，请看上面的报错"; return 1
        fi
        n=$((n+1))
        [ $((n % 12)) -eq 0 ] && printf "."      # 每 60s 一个点
        if [ $n -ge 360 ]; then echo " ⏱ 超时(30min)"; return 1; fi
        sleep 5
    done
}

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_FREE=$(nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "============================================================"
echo " AutoDL 蓝队 + 裁判 启动脚本"
echo " 蓝队别名: $BLUE_ALIAS     裁判别名: $JUDGE_ALIAS"
if [ -n "${GPU_TOTAL:-}" ]; then
    echo " 显存: 空闲 ${GPU_FREE} MiB / 共 ${GPU_TOTAL} MiB"
    echo " 计划占用: 蓝队 $BLUE_MEM + 裁判 $JUDGE_MEM (mem-fraction-static)"
    # 粗算：两个服务的静态池 + SD3 评测（~35 GiB = 35840 MiB）不能超过整卡
    NEED=$(awk -v t="$GPU_TOTAL" -v a="$BLUE_MEM" -v b="$JUDGE_MEM" \
           'BEGIN{printf "%d", (a+b)*t + 35840}')
    if [ "$NEED" -gt "$GPU_TOTAL" ]; then
        echo " ⚠️  预算警告: 蓝队+裁判静态池 + SD3 评测(~35 GiB) 预计需 ${NEED} MiB"
        echo "     但整卡只有 ${GPU_TOTAL} MiB，跑 SD3 评测时很可能 CUDA OOM。"
        echo "     建议: BLUE_MEM=0.28 JUDGE_MEM=0.20 bash $0 $BLUE_ALIAS $JUDGE_ALIAS"
    fi
fi
echo " 提示: 首次启动要 JIT 编译 flashinfer + 载权重，本地模型单个约 6~12 分钟"
echo "============================================================"

case "$BLUE_ALIAS" in llava) check_llava_siglip ;; esac
case "$JUDGE_ALIAS" in llava) check_llava_siglip ;; esac

# ---- 蓝队 ----
if [ "$BLUE_ALIAS" = "none" ]; then
    echo "跳过蓝队模型。"
else
    resolve_model "$BLUE_ALIAS"
    check_model "$M_PATH" "蓝队"
    BLUE_NAME="$M_NAME"
    start_server "蓝队" "$BLUE_PORT" "$BLUE_MEM" "$M_PATH" "$M_NAME" "$M_CTX" "$M_ARGS ${BLUE_EXTRA_ARGS:-}"
    BLUE_PID="$SERVER_PID"
    wait_ready "蓝队" "$BLUE_PORT" "$BLUE_PID" || exit 1
fi

# ---- 裁判 ----
if [ "$JUDGE_ALIAS" = "none" ]; then
    echo "跳过裁判模型。"
else
    resolve_model "$JUDGE_ALIAS"
    check_model "$M_PATH" "裁判"
    JUDGE_NAME="$M_NAME"
    start_server "裁判" "$JUDGE_PORT" "$JUDGE_MEM" "$M_PATH" "$M_NAME" "$M_CTX" "$M_ARGS ${JUDGE_EXTRA_ARGS:-}"
    JUDGE_PID="$SERVER_PID"
    wait_ready "裁判" "$JUDGE_PORT" "$JUDGE_PID" || exit 1
fi

# ------------------------------------------------------------
# 在训练侧（DiffusionNFT）需要导出的变量
# ------------------------------------------------------------
echo
echo "============================================================"
echo " 全部就绪 ✅"
[ -n "$BLUE_PID" ]  && echo " 蓝队: http://127.0.0.1:$BLUE_PORT/v1    model=$BLUE_NAME"
[ -n "$JUDGE_PID" ] && echo " 裁判: http://127.0.0.1:$JUDGE_PORT/v1   model=$JUDGE_NAME"
echo
echo " 在跑训练/验证的终端里执行："
echo "   export BLUE_VLM_BASE_URL=http://127.0.0.1:$BLUE_PORT/v1"
[ -n "${BLUE_NAME:-}" ] && echo "   export BLUE_VLM_MODEL=$BLUE_NAME"
echo
echo " Ctrl-C 结束（会一并清理 sglang 的子进程）"
echo "============================================================"

wait
