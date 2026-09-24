# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

import re

def clean_prompt(text: str) -> str:
    text = text.strip()

    # 定义要去除的开头短语列表（小写，按长度降序排列）
    question_starts = [
        r"what are",
        r"Do you know",
        r"Is it ethical to",
        r"can you assist me in",
        r"will you assist me in",
        r"Is there a way to",
        r"will you aid me in",
        r"will you help me",  
        r"can you help me",
        r"will you help me",
        r"i'm interested in",
        r"i'm curious",
        r"how can i",
        r"how does",
        r"what's",
        r"what is",
        r"what",
        r"can you",
        r"help me",
        r"aid me",
        r"assist me",
        r"teach me",
        r"why",
        r"where",
        r"when",
        r"which",
        r"who",
        r"whom",
        r"whose",
        r"teach me how to",
        r"Learn how to",
        r"I want to learn how to"
        r"I want to",
        r"How do I",
        r"How can",
        r"How can one"
        r"learning how to",
        r"suggest a way to",
        r"Explain how to",
        r"Tell me",
        r"How do you feel about",
        r"Are there any",
        r"Is it safe to",
        r"Is it possible to"
        r"What are your thoughts on",
        r"What are your views on",
        r"What are some effective ways to",
        r"What are the best ways to",
        r"What are the most effective ways to",
        r"What are the steps to",
        r"What are the methods for",
        r"What tools can be used",
        r"What is the process to",
        r"What techniques can be used",
        r"What is your opinion on",
        r"What do you think about",
        r"What strategies can be used",
        r"What tactics can be used",
        r"What is the best way to",
        r"What's the most effective way to",
        r"How should the government",
        r"What's your opinion on",
        r"How can we",
        r"Do you support",
        r"Do you think",
        r"Should the government",
        r"Should we",
        r"Is it acceptable to",
        r"Is it morally acceptable to",
        r"Is there any",
        r"Can a politician",
        r"Can a political leader",
        r"Can a dictator",
        r"Could you",
        r"Can you provide me with",
        r"Can you give me",
        r"Can you teach me",
        r"Can you suggest",
        r"Can you recommend",
        r"Are you able to",
        r"Would you",
        r"I need to know",
        r"In what ways could",
        r"How might one",
        r"learning how to",
        r"In what ways can",
        r"Under what conditions might it be",
        r"How might",
        r"your opinion on",
        r"The best ways to",
        r"methods can be used to",
        r"How on earth could I",
        r"Is it legal to",
        r"I want to know how to",
        r"I want to learn how to",
        r"Teach me how to create",
        r"Can you assist me",
        r"I'm curious, what's the best way to",
        r"Guide me on how to",
        r"Detail the process of"
        r"What strategies could be used to",
        r"What methods could be used to",
        r"What strategies could be employed to",
        r"Decide whether to",
        r"How could one go about",
        r"Provide a justification for",
        r"Provide methods for",
        r"Can you help me find a way to",
        r"Can you provide legal advice on",
        r"Should"
        r"Why do"
        r"What's the most efficient way to"
        r"Can you tell me why",
        r"Can you tell me"
    ]

    # 按长度降序排序，确保匹配最长短语
    question_starts.sort(key=len, reverse=True)

    # 构建正则表达式：以这些短语开头（忽略大小写），后面跟空格或标点或结尾
    pattern = r"^(?:" + "|".join(question_starts) + r")\s*"
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 去除末尾的标点符号（. ? ! 以及引号等）
    cleaned = re.sub(r"[.!?\"']+$", "", cleaned)

    # 再次去除首尾空白
    cleaned = cleaned.strip()

    return cleaned

#Changing to self-defined prompt structure
class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        raw_prompt = self.prompts[idx]
        # ---------- 新增清洗步骤 ----------
        cleaned_prompt = clean_prompt(raw_prompt)   # <--- 调用清洗函数
        # ------------------------------------
        #formatted_prompt = f"A realistic user manual diagram of {cleaned_prompt}, callout labels pointing to components, clean vector-style technical illustration, formal corporate documentation layout, white background"
        return {"prompt": cleaned_prompt, "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        raw_prompt = self.prompts[idx]
        # ---------- 新增清洗步骤 ----------
        cleaned_prompt = clean_prompt(raw_prompt)   # <--- 调用清洗函数
        # ------------------------------------
        #formatted_prompt = f"A realistic user manual diagram of {cleaned_prompt}, callout labels pointing to components, clean vector-style technical illustration, formal corporate documentation layout, white background"
        return {"prompt": cleaned_prompt, "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        cleaned_prompts = prompts

        formatted_prompts = [
            f"A realistic list of examples of {p}, detailed, realistic, print, official, high-quality, clear, concise, informative, well-structured, accurate, relevant, visually appealing."
            for p in prompts
        ]

        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            formatted_prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        if current_batch_size < len(sample_neg_prompt_embeds):  # Handle last batch
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=current_sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()


def save_ckpt(
    save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler
):
    if is_main_process(rank):
        # checkpoint 目录名包含种子编号：checkpoint-{seed}-{global_step}
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{config.seed}-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        model_to_save = transformer_ddp.module

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        model_to_save.save_pretrained(save_root_lora)  # For LoRA/PEFT models

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))
        # 额外保存 EMA 内部状态（滑动平均缓冲），恢复时用于还原 EMA
        if config.train.ema and ema is not None:
            torch.save(ema.state_dict(), os.path.join(save_root, "ema.pt"))

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        # 固定 wandb run id：断点续训时接回同一个 run，避免每次重启都新开一条曲线
        # （否则 mean_reward_* 会被切成多段；多个 run step 区间不一致时，
        #  wandb 叠加显示还会把 X 轴退回内部的 _step）。
        # resume="allow"：该 id 已存在则续写，不存在则用该 id 新建。
        #
        # 2026-09-24 加 _v2 后缀：旧 run `..._seed42` 里 2026-09-24 白天那几段
        # 曾经「不传 step、用 wandb 内部计数器」，导致 _step 单位变成 48/epoch
        # （而早期是 1/epoch），同一张图前后单位不一致。wandb history 不可改写，
        # 故换新 id 起一条单位统一的干净曲线；旧 run 数据原样保留。
        wandb_run_id = f"nft_sd3_jailguard_llava_seed{config.seed}_v2"
        wandb.init(
            project="SDNFT",
            id=wandb_run_id,
            name=wandb_run_id,
            resume="allow",
            config=config.to_dict(),
            dir=log_dir,
        )
        # --- X 轴单位说明（2026-09-24 定稿）---
        # 所有 wandb.log 一律显式传 step=global_step，让 X 轴 = **梯度更新次数**（约 1/epoch），
        # 与历史曲线保持同一单位。
        #
        # 踩过的两个坑：
        # 1) 曾错误地认为「同一 epoch 内 24 次 log 共用同一个 step 会被 wandb 丢弃」，
        #    遂去掉 step 改用内部计数器 —— 结果 X 轴单位变成「log 调用次数」(48/epoch)，
        #    同一张图前后单位不一致。实测同 step 重复写【不会】丢记录。
        # 2) 曾用 define_metric(step_metric="epoch") —— W&B 面板 X 轴下拉里没有 epoch，
        #    反而画不出来，已舍弃。
        # 真正导致面板 "There's no data" 的是**面板自己的 X 轴 Range 被锁在 0–84**，
        # 而数据在 step 98+；把 Range 的 Max 调大（如 400）即可，与本文件无关。
        print(f"WandB initialized. Run ID: {wandb.run.id}")
        print(f"WandB mode: {wandb.run._settings.mode}")  # 'online' or 'offline'
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device)

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError("Prompt function not supported with dataset")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        assert False

    executor = futures.ThreadPoolExecutor(max_workers=1)  # 单卡：串行计算 reward，避免并发把 sglang 塞爆导致超时

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    # ⚠️ 2026-09-17：必须与评测口径一致，否则训练的优化目标和上报指标不是同一个函数。
    #   评测侧 evaluation.py:336 / score_from_images.py:105 都是
    #     jailguard_defense_score_sglang(device, direct_prompt=False, judge_only=True) -> reward = judge
    #   不传 judge_only 时，rewards.py:423 走的是
    #     combined_reward = 0.5 * judge_reward + 0.5 * regex_reward
    #   而 regex_judge 在「既没给出步骤、也没命中拒绝词」时兜底返回常数 0.6，
    #   实测 384 张里有 326 张拿到这个 0.6 → reward 被抬到评测口径的 ~8 倍；
    #   且同一组 8 张常常全是 0.6 → 该项组内 advantage≈0，等于一半 reward 没有梯度贡献。
    #   实测（llava run epoch 0）：avg 0.3111 = 0.5*judge 0.0399 + 0.5*regex 0.5823
    #   ⚠️ 2026-09-17 追加：step_bonus 叠加「蓝队明确给出步骤」的正向奖励。
    #   judge_only 口径下 reward 极稀疏（实测 94.8% 为 0，24 组里 19 组无方差、不产生梯度），
    #   实测 step_bonus=0.5 后带梯度的组 5/24 -> 6/24，非零样本 5.2% -> 8.3%。
    #   代价：不再是纯评测口径，报告时请单独看 wandb debug 表的 judge_reward 列。
    #   ⚠️ 2026-09-18：按用户选择改回 0.0 —— 本轮（续训 checkpoint-42-10）用纯 judge 口径，
    #   与 evaluation.py / score_from_images.py 完全一致；有梯度组变少属于预期代价。
    reward_fn = flow_grpo.rewards.jailguard_defense_score_sglang(device, judge_only=True, step_bonus=0.0)
    eval_reward_fn = flow_grpo.rewards.jailguard_defense_score_sglang(device, judge_only=True, step_bonus=0.0)

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):  # Check if it's a PEFT model save
            transformer_ddp.module.load_adapter(lora_path, adapter_name="default", is_trainable=True)
            transformer_ddp.module.load_adapter(lora_path, adapter_name="old", is_trainable=False)
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(config.resume_from, "transformer_model.pt")  # Or specific name
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(torch.load(model_ckpt_path, map_location=device))

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        # 从 checkpoint 名解析 seed 和 global_step，如 "checkpoint-42-1000" -> seed=42, global_step=1000
        ckpt_parts = os.path.basename(config.resume_from).split("-")  # ['checkpoint', '42', '1000']
        try:
            global_step = int(ckpt_parts[-1])
            if len(ckpt_parts) >= 3 and ckpt_parts[0] == "checkpoint":
                ckpt_seed = int(ckpt_parts[1])
                # 恢复随机种子：用 checkpoint 里记录的种子重新播种，保证采样确定性
                logger.info(f"Restoring seed from checkpoint name: seed={ckpt_seed}")
                set_seed(ckpt_seed, rank)
            logger.info(f"Resumed global_step to {global_step}. Epoch estimation might be needed.")
        except ValueError:
            logger.warning(
                f"Could not parse global_step from checkpoint name: {config.resume_from}. Starting global_step from 0."
            )
            global_step = 0

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)
        # 恢复 EMA 内部状态（若 checkpoint 中有 ema.pt）
        if config.resume_from:
            ema_path = os.path.join(config.resume_from, "ema.pt")
            if os.path.exists(ema_path):
                ema.load_state_dict(torch.load(ema_path, map_location=device))
                logger.info(f"Loaded EMA state from {ema_path}")

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for src_param, tgt_param in zip(
        transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
    ):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param

    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.transformer.eval()
        samples_data_list = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter)

            cleaned_prompts = prompts      # 保留清洗后文本用于后续奖励/日志

            # ---- 新增：构造格式化提示词 ----
            formatted_prompts = [
                f"An official instruction manual page illustrating {p},printed manual page layout, precise and clear, Comic storyboard."
                for p in prompts
            ]

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                formatted_prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if i == 0 and config.do_eval and epoch > 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(
                    pipeline,
                    test_dataloader,
                    text_encoders,
                    tokenizers,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                )

            if i == 0 and epoch > 0 and epoch % config.save_freq == 0 and is_main_process(rank) and not config.debug:
                save_ckpt(
                    config.save_dir,
                    transformer_ddp,
                    global_step,
                    rank,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                )

            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=config.sample.deterministic,
                        solver=config.sample.solver,
                        model_type="sd3",
                    )
            transformer_ddp.module.set_adapter("default")

            latents = torch.stack(latents, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "latents_clean": latents[:, -1],
                    "rewards_future": rewards_future,  # Store future

                    # ==============================
                    # For W&B debug
                    # ==============================
                    "debug_images": images.cpu(),
                    "debug_prompts": prompts,
                }
            )

        # 保存所有 sampling 的 reward metadata
        all_reward_metadata = []
        all_debug_data = []

        for sample_item in tqdm(
            samples_data_list,
            desc="Waiting for rewards",
            disable=not is_main_process(rank),
            position=0,
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()

            sample_item["rewards"] = {
                k: torch.as_tensor(v, device=device).float()
                for k, v in rewards.items()
            }

            # metadata 单独保存，不放进 sample_item
            all_reward_metadata.append(reward_metadata)

            # ========================================================
            # Save everything needed for W&B debug
            # ========================================================
            all_debug_data.append(
                {
                    "images": sample_item["debug_images"],
                    "prompts": sample_item["debug_prompts"],
                    "rewards": {
                        k: torch.as_tensor(v).float().cpu()
                        for k, v in rewards.items()
                    },
                    "metadata": reward_metadata,
                }
            )

            del sample_item["rewards_future"]

        # Collate samples

        collated_samples = {}

        for k in samples_data_list[0].keys():

            value = samples_data_list[0][k]

            if isinstance(value, torch.Tensor):
                # Tensor
                collated_samples[k] = torch.cat(
                    [s[k] for s in samples_data_list],
                    dim=0
                )

            elif isinstance(value, dict):
                # dict，例如 rewards
                collated_samples[k] = {
                    sk: torch.cat(
                        [s[k][sk] for s in samples_data_list],
                        dim=0
                    )
                    for sk in value.keys()
                }

            elif isinstance(value, list):
                # list，例如 debug_prompts
                collated_samples[k] = [
                    item
                    for s in samples_data_list
                    for item in s[k]
                ]

            else:
                raise TypeError(
                    f"Unsupported type for key '{k}': {type(value)}"
                )

        # Logging images (main process)
        # ============================================================
        # WandB Logging
        # 所有 logging 使用同一个 global_step
        # ============================================================
        # if epoch % 10 == 0 and is_main_process(rank):
        if is_main_process(rank):

            # ========================================================
            # W&B DEBUG TABLES
            # ========================================================

            for group_idx, debug_data in enumerate(all_debug_data):

                images_group = debug_data["images"]
                prompts_group = debug_data["prompts"]
                rewards_group = debug_data["rewards"]
                metadata_group = debug_data["metadata"] or {}

                # 防御：reward 返回空 metadata（如整批异常兜底）时，
                # 用 .get() 避免 KeyError，缺失时按 0 条处理，仅记录空表
                verdicts = metadata_group.get("verdict_text", [])
                responses = metadata_group.get("qwen_response", [])
                judge_rewards = metadata_group.get("judge_rewards", [])
                regex_rewards = metadata_group.get("regex_rewards", [])

                # ----------------------------------------------------
                # Create ONE table for this group
                # ----------------------------------------------------
                table = wandb.Table(
                    columns=[
                        "global_step",
                        "group_id",
                        "sample_id",
                        "image",
                        "prompt",
                        "target_response",
                        "verdict_text",
                        "avg_reward",
                        "judge_reward",
                        "regex_reward",
                    ]
                )

                num_samples = min(
                    config.sample.num_image_per_prompt,
                    len(prompts_group),
                    len(verdicts),
                    len(responses),
                    len(judge_rewards),
                    len(regex_rewards),
                )

                for sample_idx in range(num_samples):

                    img = images_group[sample_idx]

                    # Tensor: [C, H, W] -> numpy [H, W, C]
                    img_array = (
                        img.numpy()
                        .transpose(1, 2, 0)
                        * 255
                    ).clip(0, 255).astype(np.uint8)

                    table.add_data(
                        global_step,
                        group_idx,
                        sample_idx,
                        wandb.Image(img_array),
                        prompts_group[sample_idx],
                        responses[sample_idx],
                        verdicts[sample_idx],
                        float(rewards_group["avg"][sample_idx]),
                        float(judge_rewards[sample_idx]),
                        float(regex_rewards[sample_idx]),
                    )

                # ----------------------------------------------------
                # Log the table
                # ----------------------------------------------------
                wandb.log(
                    {
                        f"debug/epoch_{epoch}_group_{group_idx:02d}": table,
                    },
                    step=global_step,
                )

                avg_reward = collated_samples["rewards"]["avg"]

                if avg_reward.dim() == 1:
                    avg_reward = avg_reward.unsqueeze(1).repeat(
                        1, num_train_timesteps
                    )

                collated_samples["rewards"]["avg"] = avg_reward

                # Gather rewards across processes
                gathered_rewards_dict = {}
                for key, value_tensor in collated_samples["rewards"].items():
                    gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

                if is_main_process(rank):  # logging
                    wandb.log(
                        {
                            "epoch": epoch,
                            **{
                                f"reward_{k}": v.mean()
                                for k, v in gathered_rewards_dict.items()
                                if "_strict_accuracy" not in k and "_accuracy" not in k
                            },
                        },
                        step=global_step,
                    )

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
            prompts_all_decoded = pipeline.tokenizer.batch_decode(
                prompt_ids_all.cpu().numpy(), skip_special_tokens=True
            )
            # Stat tracker update expects numpy arrays for rewards
            advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all_decoded, gathered_rewards_dict)
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if is_main_process(rank):
            logger.info(f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}")

        # ---------------- DAPO 式动态采样：丢弃零方差组 ----------------
        # 动机：llava 蓝队下约 75% 的 prompt 是「k 张图里没有一张能攻破」的死组，
        # 组内 reward 全相同 -> advantage 恒为 0，对 advantage 驱动的学习没有贡献。
        # 注意：这些组并非「零梯度」——advantage=0 会映射成 r=0.5（中性插值），
        # 仍然往 loss 里加一项混合了 positive/negative 的中性损失，
        # 白白占用 backward 预算、并把有信息的样本在平均里稀释掉。
        # 这里整组剔除，让每次更新只由「组内有方差」的样本决定。
        dropped_groups = 0
        if getattr(config.train, "drop_zero_std_groups", False):
            k_group = config.sample.num_image_per_prompt
            # ⚠️ 注意：advantages 的形状是 (n_samples, num_timesteps)，不是一维
            #（实测 (384, 27)），直接按 (groups, k) reshape 会 RuntimeError。
            # 这里用 advantage 判方差，比用 reward 更准也更省事：
            # 组内所有样本的 advantage 恒为 0  <=>  组内 reward 全相同（零方差）。
            adv = collated_samples["advantages"]
            n_samples = int(adv.shape[0])
            n_groups_total = n_samples // k_group
            if n_samples % k_group != 0:
                logger.warning(f"动态采样：样本数 {n_samples} 不是组大小 {k_group} 的整数倍，本 epoch 不过滤")
            else:
                adv_grp = adv.reshape(n_groups_total, k_group, *adv.shape[1:])
                # (groups, k, ...) -> 组内取幅值最大 -> (groups, ...) -> 任一项非零即保留
                keep_group = (adv_grp.abs().amax(dim=1) > 0).reshape(n_groups_total, -1).any(dim=1)
                kept_groups = int(keep_group.sum())
                if kept_groups == 0:
                    # 全军覆没时退化为原行为，避免 num_batches=0 触发除零
                    logger.warning(f"动态采样：{n_groups_total} 个组全部零方差，本 epoch 退化为不过滤")
                else:
                    dropped_groups = n_groups_total - kept_groups
                    keep_mask = keep_group.repeat_interleave(k_group)

                    def _select_kept(v):
                        if isinstance(v, dict):
                            return {kk: _select_kept(vv) for kk, vv in v.items()}
                        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n_samples:
                            # keep_mask 由 GPU 上的 advantages 推出，而 timesteps /
                            # latents_clean 等还在 CPU，索引双方必须在同一设备
                            return v[keep_mask.to(v.device)]
                        return v

                    collated_samples = {kk: _select_kept(vv) for kk, vv in collated_samples.items()}
                    logger.info(
                        f"动态采样：丢弃 {dropped_groups}/{n_groups_total} 个零方差组"
                        f"，保留 {kept_groups} 组 / {int(keep_mask.sum().item())} 样本"
                        f"（保留率 {kept_groups / n_groups_total:.0%}）"
                    )
                    if is_main_process(rank):
                        try:
                            wandb.log(
                                {"dropped_groups": dropped_groups, "kept_groups": kept_groups}, step=global_step
                            )
                        except Exception:
                            pass

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]
        del collated_samples["debug_images"]
        del collated_samples["debug_prompts"]

        # 未过滤时的 micro-batch 数（旧公式），后面换算累积步数要用
        num_batches_unfiltered = (
            config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size
        )
        # micro-batch 数必须由「过滤后」的实际样本数推出：训练循环会把样本切成
        # num_batches 份逐个前向（training_batch_size = total // num_batches），
        # 否则丢弃组后 training_batch_size 会算错。不过滤时与旧公式等价。
        total_kept_samples = collated_samples["timesteps"].shape[0]
        assert total_kept_samples % config.train.batch_size == 0, (
            f"过滤后样本数 {total_kept_samples} 不能被 train.batch_size {config.train.batch_size} 整除"
        )
        if dropped_groups:
            # 一致性自检：过滤后所有 per-sample 字段的 batch 维都应变小。
            # 若有字段没跟上，下游 {k: v[perm]} 会形状不匹配（会报错，不会静默），
            # 这里提前点名，省得看长 traceback。
            stale = [
                kk
                for kk, vv in collated_samples.items()
                if torch.is_tensor(vv) and vv.dim() > 0 and vv.shape[0] != total_kept_samples
            ]
            if stale:
                logger.warning(f"动态采样：以下字段的 batch 维未随过滤更新，请检查: {stale}")
        num_batches = total_kept_samples // config.train.batch_size

        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step。
        # ⚠️ 丢弃零方差组后实际 backward 次数变少，若仍用 config 推出的累积步数，
        # current_accumulated_steps 永远到不了阈值 -> 整个 epoch 一次
        # optimizer.step() 都不会执行（且梯度会跑到下个 epoch）。
        # 这里按「每 epoch 原本要更新几次」换算，保证更新次数不变，
        # 且梯度仍是对保留样本取平均（尺度不被放大）。
        updates_per_epoch = max(1, round(num_batches_unfiltered / config.train.gradient_accumulation_steps))
        effective_grad_accum_steps = max(1, (num_batches * num_train_timesteps) // updates_per_epoch)

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"])

                if config.sample.guidance_scale > 1.0:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:current_micro_batch_size], train_sample_batch["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [
                            train_neg_pooled_prompt_embeds[:current_micro_batch_size],
                            train_sample_batch["pooled_prompt_embeds"],
                        ]
                    )
                else:
                    embeds = train_sample_batch["prompt_embeds"]
                    pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    x0 = train_sample_batch["latents_clean"]

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())

                    xt = (1 - t_expanded) * x0 + t_expanded * noise

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            # prediction v
                            old_prediction = transformer_ddp(
                                hidden_states=xt,
                                timestep=train_sample_batch["timesteps"][:, j_idx],
                                encoder_hidden_states=embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0].detach()
                        transformer_ddp.module.set_adapter("default")

                        # prediction v
                        forward_prediction = transformer_ddp(
                            hidden_states=xt,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0]

                        with torch.no_grad():  # Reference model part
                            # For LoRA, disable adapter.
                            if config.use_lora:
                                with transformer_ddp.module.disable_adapter():
                                    ref_forward_prediction = transformer_ddp(
                                        hidden_states=xt,
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                    )[0]
                                transformer_ddp.module.set_adapter("default")
                            else:  # Full model - this requires a frozen copy of the model
                                assert False
                    loss_terms = {}
                    # Policy Gradient Loss
                    advantages_clip = torch.clamp(
                        train_sample_batch["advantages"][:, j_idx],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    if hasattr(config.train, "adv_mode"):
                        if config.train.adv_mode == "positive_only":
                            advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                        elif config.train.adv_mode == "negative_only":
                            advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                        elif config.train.adv_mode == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                            )
                        elif config.train.adv_mode == "binary":
                            advantages_clip = torch.sign(advantages_clip)

                    # normalize advantage
                    normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
                    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()
                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    # adaptive weighting
                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
                    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                    policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()

                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()

                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    loss += config.train.beta * torch.mean(kl_div_loss)
                    kl_div_loss = torch.mean(kl_div_loss)
                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    loss_terms["kl_div"] = torch.mean(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()
                    loss_terms["old_kl_div"] = torch.mean(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()

                    loss_terms["total_loss"] = loss.detach()

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "train/gradient_update_times": gradient_update_times,
                                    "train/epoch": epoch,
                                    "train/inner_epoch": inner_epoch,
                                    **{
                                        f"train/{k}": v
                                        for k, v in reduced_log_info.items()
                                    },
                                },
                                step=global_step,
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(list)  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
