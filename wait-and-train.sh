#!/bin/bash
# wait-and-train.sh
# 等蓝队(17141)+裁判(17142) 都 /health 200 之后，再用 nft.sh 启动续训。
# 原因：mem-fraction-static 按启动时空闲显存算，训练必须先于训练启动服务，
#       否则训练占满显存后裁判无法启动 -> reward 静默为 0。
LOG=/root/autodl-fs/wait-and-train.log
TRAIN_LOG=/root/autodl-fs/train-20260924-llava-resume.log

echo "[$(date '+%F %T')] 开始等待服务就绪..." >> "$LOG"
for i in $(seq 1 120); do
  if curl -sf -m 3 http://127.0.0.1:17141/health >/dev/null && \
     curl -sf -m 3 http://127.0.0.1:17142/health >/dev/null; then
    echo "[$(date '+%F %T')] 蓝队+裁判均就绪，启动训练" >> "$LOG"
    exec bash /root/autodl-fs/nft.sh >> "$TRAIN_LOG" 2>&1
  fi
  sleep 15
done
echo "[$(date '+%F %T')] 超时(30min)：服务未就绪，放弃启动训练" >> "$LOG"
exit 1
