#!/bin/bash
# ============================================================
#  watchdog.sh — DiffusionNFT 训练看门狗
#
#  为什么需要它（2026-09-16 事故）：
#    17:36 裁判服务 :17142 因 "CUDA error: an illegal memory access" 崩溃，
#    但 sglang-models.sh 末尾是裸 `wait`，wrapper 不退出、蓝队照常响应；
#    训练侧 rewards.py 的 except 兜底返回 0.0 -> 整批 reward=0 -> advantage=0
#    -> 梯度=0。结果白跑 2.5h 无任何告警。
#    本脚本把「服务偷偷死了」变成「立刻告警 + 立即止损」。
#
#  用法：
#    nohup bash /root/autodl-fs/watchdog.sh > /root/autodl-fs/watchdog.log 2>&1 &
#
#  可覆盖的环境变量：
#    INTERVAL=60        巡检间隔秒
#    FAIL_TOLERANCE=3   健康检查连续失败几次才判定死亡（默认 3 次 ≈ 3min，防抖动误杀）
#    BLUE_PORT=17141    JUDGE_PORT=17142
#    TRAIN_LOG=/root/autodl-fs/train-20260917.log
#    STALE_MIN=40       训练日志多少分钟不更新算卡死（0 = 关闭该检查）
#    REWARD_WATCH=1     是否检测 "Advantages mean" 连续为 0（静默失败的指纹）
#    ZERO_TOLERANCE=2   连续几个 epoch 的 Advantages mean 为 0 才止损
#                       （judge_only 口径下 reward 是离散的 {0,2/9,4/9,6/9,8/9}，
#                        组内同分时 advantage 天然为 0，可适当调大或 REWARD_WATCH=0）
#    KILL_ON_ALERT=1    检测到异常时是否杀掉训练（0 = 只告警不动手）
#
#  退出码：0 = 训练进程自然结束 / 1 = 看门狗检测到异常并止损
# ============================================================
set -u

FS=${FS:-/root/autodl-fs}
INTERVAL=${INTERVAL:-60}
FAIL_TOLERANCE=${FAIL_TOLERANCE:-3}
BLUE_PORT=${BLUE_PORT:-17141}
JUDGE_PORT=${JUDGE_PORT:-17142}
TRAIN_LOG=${TRAIN_LOG:-$FS/train-$(date +%Y%m%d).log}
STALE_MIN=${STALE_MIN:-40}
REWARD_WATCH=${REWARD_WATCH:-1}
ZERO_TOLERANCE=${ZERO_TOLERANCE:-2}
KILL_ON_ALERT=${KILL_ON_ALERT:-1}
SENTINEL=$FS/TRAINING_STOPPED_ALERT.txt

# 训练进程匹配模式：bash nft.sh -> torchrun -> 真正的 python 子进程
TRAIN_PATTERN="bash nft\.sh|torch\.distributed\.run|train_nft_sd3\.py"

BLUE_FAIL=0
JUDGE_FAIL=0
ZERO_STREAK=0
REWARD_SEEN=0

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

health() { curl -sf -m 5 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }
train_pids() { pgrep -f "$TRAIN_PATTERN" 2>/dev/null; }

alert_stop() {
    # alert_stop <原因>
    local reason="$1" pids
    say "❌ 异常：$reason"

    if [ "$KILL_ON_ALERT" = 1 ]; then
        say "🛑 止损：停止训练进程（sglang 服务保持运行，不动）"
        # 一次拿全 PID 再发信号：先杀子（python）再杀父（torchrun / nft.sh）
        pids=$(train_pids)
        [ -n "$pids" ] && kill -TERM $pids 2>/dev/null
        sleep 20
        pids=$(train_pids)
        if [ -n "$pids" ]; then
            say "   20s 后仍存活，SIGKILL: $pids"
            kill -KILL $pids 2>/dev/null
        fi
        sleep 2
        if [ -n "$(train_pids)" ]; then
            say "   ⚠️ 仍有残留进程: $(train_pids | tr '\n' ' ')"
        else
            say "   ✅ 训练已停止"
        fi
    else
        say "   KILL_ON_ALERT=0，仅告警不杀训练"
    fi

    # 无论是否杀训练都写告警文件：终端可能早就关了，这是唯一的持久记录
    {
        echo "训练被看门狗停止 —— $(ts)"
        echo "原因: $reason"
        echo
        echo "服务状态:"
        echo "  蓝队  :$BLUE_PORT  $(health "$BLUE_PORT" && echo UP || echo DOWN)"
        echo "  裁判  :$JUDGE_PORT  $(health "$JUDGE_PORT" && echo UP || echo DOWN)"
        echo
        echo "恢复正常训练步骤:"
        echo "  1) 确认/重启服务: cd $FS && BLUE_MEM=0.25 JUDGE_MEM=0.20 bash sglang-models.sh internvl3 jail"
        echo "  2) 等两个 /health 都返回 200（蓝队 ~5min, 裁判 ~10min）"
        echo "  3) nft.sh 里启用 RESUME_FROM=...checkpoint-<seed>-<step>，再 bash $FS/nft.sh"
        echo "  4) 核对首 epoch: grep -o 'Advantages mean: [-0-9.]*' $TRAIN_LOG | tail -3  # 必须非 0"
        echo
        echo "详细日志: $FS/watchdog.log"
    } > "$SENTINEL"
    say "   📄 已写入告警文件: $SENTINEL"
}

trap 'say "看门狗退出（exit=$?）"' EXIT

say "============================================================"
say " 训练看门狗启动"
say "  巡检间隔 : ${INTERVAL}s   失败阈值 : ${FAIL_TOLERANCE} 次"
say "  蓝队     : 127.0.0.1:$BLUE_PORT    裁判 : 127.0.0.1:$JUDGE_PORT"
say "  训练日志 : $TRAIN_LOG（${STALE_MIN}min 不更新算卡死）"
say "  训练进程 : $(train_pids | tr '\n' ' ')"
say "============================================================"

while true; do
    # ---- 0) 训练进程是否还在 ----
    if [ -z "$(train_pids)" ]; then
        say "✅ 训练进程已退出（正常结束或被手动停止），看门狗随之退出"
        exit 0
    fi

    # ---- 1) 服务健康 ----
    health "$BLUE_PORT"  && BLUE_FAIL=0  || BLUE_FAIL=$((BLUE_FAIL + 1))
    health "$JUDGE_PORT" && JUDGE_FAIL=0 || JUDGE_FAIL=$((JUDGE_FAIL + 1))

    if [ "$BLUE_FAIL" -gt 0 ] || [ "$JUDGE_FAIL" -gt 0 ]; then
        say "⚠️  健康检查失败计数 blue=$BLUE_FAIL judge=$JUDGE_FAIL（阈值 $FAIL_TOLERANCE）"
    fi

    if [ "$BLUE_FAIL" -ge "$FAIL_TOLERANCE" ]; then
        alert_stop "蓝队 VLM :$BLUE_PORT 连续 $BLUE_FAIL 次无响应（训练拿不到回复）"
        exit 1
    fi
    if [ "$JUDGE_FAIL" -ge "$FAIL_TOLERANCE" ]; then
        alert_stop "裁判 :$JUDGE_PORT 连续 $JUDGE_FAIL 次无响应（reward 会全 0，静默失败）"
        exit 1
    fi

    # ---- 2) reward 静默归零检测（只在新 epoch 结果出现时评估）----
    if [ "$REWARD_WATCH" = 1 ] && [ -f "$TRAIN_LOG" ]; then
        n=$(grep -c "Advantages mean:" "$TRAIN_LOG" 2>/dev/null || true)
        n=${n:-0}
        if [ "$n" -gt "$REWARD_SEEN" ]; then
            REWARD_SEEN=$n
            last=$(grep -o "Advantages mean: [-0-9.eE]*" "$TRAIN_LOG" | tail -1 | awk '{print $3}')
            # 恰好 0（或极小）视为该 batch advantage 全 0
            is_zero=$(awk -v v="${last:-x}" 'BEGIN{ if (v ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) { a=v<0?-v:v; print (a<1e-12)?1:0 } else print 0 }')
            if [ "$is_zero" = 1 ]; then
                ZERO_STREAK=$((ZERO_STREAK + 1))
                say "⚠️  Advantages mean=0（第 $ZERO_STREAK 次连续）—— reward 全 0 的指纹"
            else
                ZERO_STREAK=0
            fi
            if [ "$ZERO_STREAK" -ge "$ZERO_TOLERANCE" ]; then
                alert_stop "连续 $ZERO_STREAK 个 epoch 的 Advantages mean=0（reward 全 0，梯度为 0）"
                exit 1
            fi
        fi
    fi

    # ---- 3) 训练日志是否还在动（卡死检测）----
    if [ "$STALE_MIN" -gt 0 ] && [ -f "$TRAIN_LOG" ]; then
        age=$(( ( $(date +%s) - $(stat -c %Y "$TRAIN_LOG") ) / 60 ))
        if [ "$age" -ge "$STALE_MIN" ]; then
            alert_stop "训练日志 $age 分钟无更新（> ${STALE_MIN}min），疑似卡死"
            exit 1
        fi
    fi

    sleep "$INTERVAL"
done
