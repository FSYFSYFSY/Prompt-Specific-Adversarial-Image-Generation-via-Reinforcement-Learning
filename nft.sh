export CUDA_VISIBLE_DEVICES=0

cd /root/autodl-tmp/DiffusionNFT


export HF_TOKEN=“”
export WANDB_MODE=online
export PYTHONPATH=/root/autodl-tmp/DiffusionNFT:$PYTHONPATH 
export HF_HOME=/root/autodl-tmp/DiffusionNFT/model
export HF_ENDPOINT=https://hf-mirror.com

export WANDB_API_KEY="

torchrun --nproc_per_node=1 \
    /root/autodl-tmp/DiffusionNFT/scripts/train_nft_sd3.py \
    --config /root/autodl-tmp/DiffusionNFT/config/nft.py:sd3_jailguard