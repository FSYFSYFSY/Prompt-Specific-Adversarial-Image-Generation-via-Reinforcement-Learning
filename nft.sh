#!/bin/bash
# nft.sh - SD3.5 jailguard 强化学习训练脚本
# 显式固定使用 sd35-cu130 环境，避免因未激活环境而跑错解释器
ENV_PYTHON=/autodl-fs/data/sd35-cu130/bin/python

# 显式指向环境自带 cu13 工具链，避免 flashinfer/JIT 读到系统 /usr/local/cuda(12.8)
export CUDA_HOME=/autodl-fs/data/sd35-cu130/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH

export CUDA_VISIBLE_DEVICES=0

cd /autodl-fs/data/DiffusionNFT


# ⚠️ 密钥一律从环境变量注入，切勿写入仓库（参考 .env.example）
# 运行前先: export HF_TOKEN=...  /  export WANDB_API_KEY=...
export HF_TOKEN="${HF_TOKEN:?请先 export HF_TOKEN（HuggingFace 访问令牌）}"
export WANDB_MODE=online
export PYTHONPATH=/autodl-fs/data/DiffusionNFT:$PYTHONPATH 
export HF_HOME=/autodl-fs/data/DiffusionNFT/model
export HF_ENDPOINT=https://hf-mirror.com

export WANDB_API_KEY="${WANDB_API_KEY:?请先 export WANDB_API_KEY（Weights & Biases 密钥）}"

# NCCL 容器稳定性（单卡 DDP 防御性设置，避免容器内 P2P/IB 导致的 CUDA invalid argument）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ---- 蓝队 VLM + 裁判：训练侧必须 export，rewards.py 靠这两个变量找服务 ----
# BLUE_VLM_MODEL 必须等于 sglang 启动时的 --served-model-name：
#   sglang-models.sh llava -> "llava-onevision-qwen2-7b-ov"
# 不设的话 rewards.py 默认发 "Qwen/Qwen3-VL-8B-Instruct"，model 名对不上会直接报错。
# 启动服务（另一个终端）：bash /autodl-fs/data/sglang-models.sh llava jail
export BLUE_VLM_BASE_URL="http://127.0.0.1:17141/v1"
export BLUE_VLM_MODEL="llava-onevision-qwen2-7b-ov"

# ---- 断点续训（2026-09-23 第三次更新）----
# 会自动恢复 LoRA(default+old)/优化器/scaler/EMA，并从目录名解析 seed(42) 与 global_step。
# 本次改动：train_nft_sd3.py 的 wandb.init 改为固定 id + resume="allow"，
# 让续训接回同一条 wandb run（不再每次新开曲线）。从 checkpoint-42-98 继续。
RESUME_FROM=/autodl-fs/data/DiffusionNFT/logs/nft/sd3/jailguard-llava/checkpoints/checkpoint-42-108
RESUME_ARG="--config.resume_from=$RESUME_FROM"
# 落盘更密：save 只在 epoch % save_freq == 0 且该 epoch 第一个 batch 时触发。
# 10 -> 5 之后仍然踩中 09-18 15:53 的裁判崩溃：当时跑到 epoch 4/5，第一个 save 点(epoch 5)
# 差 7 分钟没到，84 分钟的进度全丢。现改为 2（每个 checkpoint 359MB，~30min 一个）。
SAVE_FREQ_ARG="--config.save_freq=2"
#
# ⚠️ 2026-09-17 12:40 修正两处后重新开始（全新 run：seed 42, global_step 0）：
#   1) 蓝队改为 llava-onevision-qwen2-7b-ov（此前误用 InternVL3-8B，
#      依据是 09-16 的笔记，但用户实际要求的是 llava）
#   2) 出图步数 config.sample.num_steps 10 -> 28（向评测流水线的 40 步靠拢）
#      新配置见 config/nft.py:sd3_jailguard_llava，save_dir=logs/nft/sd3/jailguard-llava
#    已删除的产物：logs/nft/sd3/jailguard-internvl3/（09-16 的 718M + 09-17 的 359M）、
#                 logs/nft_sd3_jailguard_internvl3_2026.09.16_* 与 _2026.09.17_*
#    保留未动：旧的 Qwen3-VL run logs/nft/sd3/jailguard（3.0G，含 4 蓝队 eval 分数）
#
# ---- 09-16 两次中断复盘（保留备查）----
#  1) 17:36 裁判 :17142 崩："CUDA error: an illegal memory access"
#     (batch_result_processor.py:811 copy_done.synchronize())，SIGQUIT -> Fatal Python error: Aborted。
#     静默失败陷阱：sglang-models.sh 末尾是裸 wait，单个服务死掉脚本不退出、蓝队照常响应，
#     训练侧 rewards.py 的 except 兜底返回 0.0（Error: APIConnectionError）-> 整批 reward=0
#     -> advantage=0 -> 梯度=0，白跑 2.5h 无告警。
#     排查服务是否活着要先 curl -sf :17142/health，别看 wrapper 还在不在。
#  2) 20:52 从 checkpoint-42-10 续训，跑到 22:55 进程被外部终止（日志无任何报错，
#     advantages mean 一直在 0.30~0.50，说明 reward 正常），最后落盘的 checkpoint 是
#     global_step=20，之后约 1.7 个 epoch 的进度未落盘。

# ⚠️ 2026-09-18 本次改动：
#   1) step_bonus 0.5 -> 0.0（train_nft_sd3.py:671），回到与评测完全一致的纯 judge 口径
#   2) save_freq 10 -> 5，避免再次「跑了几小时没落盘就被关机」
#   3) 从 checkpoint-42-10 续训（见上面的 RESUME_FROM）
$ENV_PYTHON -m torch.distributed.run --nproc_per_node=1 \
    /autodl-fs/data/DiffusionNFT/scripts/train_nft_sd3.py \
    --config /autodl-fs/data/DiffusionNFT/config/nft.py:sd3_jailguard_llava \
    $SAVE_FREQ_ARG $RESUME_ARG