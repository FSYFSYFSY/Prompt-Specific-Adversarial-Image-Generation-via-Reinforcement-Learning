#!/bin/bash
# restart-on-checkpoint.sh
# 用途：等下一个 checkpoint 落盘后，自动重启训练，以应用 train_nft_sd3.py 的最新改动
#       （新增 wandb.define_metric + 去掉显式 step=global_step，让曲线能正常记录）
#
# 安全设计：先检查服务健康，健康才停训练；服务不健康则直接放弃，保持现有训练继续跑。
# 只重启训练，不碰蓝队/裁判服务（服务重启会因显存不足失败，详见 training-resume 笔记）。

CKPT_DIR=/autodl-fs/data/DiffusionNFT/logs/nft/sd3/jailguard-llava/checkpoints
PREV=checkpoint-42-106
NFT=/root/autodl-fs/nft.sh
LOG=/root/autodl-fs/restart-on-ckpt3.log
TRAIN_LOG=/root/autodl-fs/train-20260924-llava-resume4.log

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "=== 开始监控新 checkpoint（当前最新: $PREV）==="

# ---------- 1. 等新 checkpoint 出现且写完 ----------
NEW=""
for i in $(seq 1 1440); do          # 最多等约 8 小时
  cand=$(ls -t "$CKPT_DIR" 2>/dev/null | grep -E '^checkpoint-42-[0-9]+$' | head -1)
  if [[ -n "$cand" && "$cand" != "$PREV" ]]; then
    s1=$(du -sb "$CKPT_DIR/$cand" 2>/dev/null | cut -f1)
    sleep 8
    s2=$(du -sb "$CKPT_DIR/$cand" 2>/dev/null | cut -f1)
    if [[ -n "$s1" && "$s1" == "$s2" && "$s1" -gt 1000000 ]]; then
      NEW="$cand"
      say "发现新 checkpoint 且已写完: $NEW (size=${s1} bytes)"
      break
    fi
  fi
  sleep 20
done

if [[ -z "$NEW" ]]; then
  say "超时(8h)：未发现新 checkpoint，退出，训练不受影响"
  exit 1
fi

# ---------- 2. 服务健康检查（不健康就不动训练）----------
ok=1
curl -sf -m 5 http://127.0.0.1:17141/health >/dev/null || ok=0
curl -sf -m 5 http://127.0.0.1:17142/health >/dev/null || ok=0
if [[ $ok -eq 0 ]]; then
  say "❌ 蓝队/裁判服务不健康，放弃重启（保持现有训练继续跑）；请先修复服务"
  exit 2
fi
say "服务健康检查通过（蓝队 17141 / 裁判 17142）"

# ---------- 3. 停训练 ----------
say "停止训练进程..."
pkill -f "torch.distributed.run --nproc_per_node=1" 2>/dev/null
pkill -f "train_nft_sd3.py" 2>/dev/null
for i in $(seq 1 60); do
  pgrep -f "train_nft_sd3" >/dev/null || break
  sleep 3
done

if pgrep -f "train_nft_sd3" >/dev/null; then
  say "⚠️ 训练进程未能正常退出，强制 kill -9"
  pkill -9 -f "train_nft_sd3" 2>/dev/null
  sleep 10
fi

# 等显存释放（服务自身约 40~45G，训练约 95G）
used=""
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ -n "$used" && "$used" -lt 60000 ]] && break
  sleep 3
done
say "训练已停止，显存降至 ${used} MiB"

# ---------- 4. 更新 nft.sh 的 RESUME_FROM ----------
NEW_PATH="$CKPT_DIR/$NEW"
sed -i "s#^RESUME_FROM=.*#RESUME_FROM=$NEW_PATH#" "$NFT"
say "nft.sh RESUME_FROM -> $NEW_PATH"
grep -n "^RESUME_FROM=" "$NFT" >> "$LOG"

# ---------- 5. 重启训练 ----------
cd /root/autodl-fs
nohup setsid bash "$NFT" >> "$TRAIN_LOG" 2>&1 < /dev/null &
sleep 8
if pgrep -f "train_nft_sd3" >/dev/null; then
  say "✅ 训练已重启（新日志: $TRAIN_LOG）"
else
  say "❌ 训练重启失败，请查看 $TRAIN_LOG"
fi
