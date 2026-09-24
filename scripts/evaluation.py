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

import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model

from flow_grpo.rewards import multi_score, jailguard_defense_score_sglang

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
from peft import PeftModel

import logging
import re

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def setup_distributed(rank, world_size):
    """Initializes the distributed process group."""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


def is_main_process(rank):
    """Checks if the current process is the main one (rank 0)."""
    return rank == 0


def clean_prompt(text):
    text = text.strip()
    question_starts = [
        r"what are", r"Do you know", r"Is it ethical to", r"can you assist me in",
        r"will you assist me in", r"Is there a way to", r"will you aid me in",
        r"will you help me", r"can you help me", r"i'm interested in", r"i'm curious",
        r"how can i", r"how does", r"what's", r"what is", r"what", r"can you",
        r"help me", r"aid me", r"assist me", r"teach me", r"why", r"where", r"when",
        r"which", r"who", r"whom", r"whose", r"teach me how to", r"Learn how to",
        r"I want to learn how to", r"I want to", r"How do I", r"How can", r"How can one",
        r"learning how to", r"suggest a way to", r"Explain how to", r"Tell me",
        r"How do you feel about", r"Are there any", r"Is it safe to", r"Is it possible to",
        r"What are your thoughts on", r"What are your views on", r"What are some effective ways to",
        r"What are the best ways to", r"What are the most effective ways to",
        r"What are the steps to", r"What are the methods for", r"What tools can be used",
        r"What is the process to", r"What techniques can be used", r"What is your opinion on",
        r"What do you think about", r"What strategies can be used", r"What tactics can be used",
        r"What is the best way to", r"What's the most effective way to",
        r"How should the government", r"What's your opinion on", r"How can we",
        r"Do you support", r"Do you think", r"Should the government", r"Should we",
        r"Is it acceptable to", r"Is it morally acceptable to", r"Is there any",
        r"Can a politician", r"Can a political leader", r"Can a dictator", r"Could you",
        r"Can you provide me with", r"Can you give me", r"Can you teach me",
        r"Can you suggest", r"Can you recommend", r"Are you able to", r"Would you",
        r"I need to know", r"In what ways could", r"How might one", r"How might",
        r"your opinion on", r"The best ways to", r"methods can be used to",
        r"How on earth could I", r"Is it legal to", r"I want to know how to",
        r"Teach me how to create", r"Can you assist me", r"I'm curious, what's the best way to",
        r"Guide me on how to", r"Detail the process of", r"What strategies could be used to",
        r"What methods could be used to", r"What strategies could be employed to",
        r"Decide whether to", r"How could one go about", r"Provide a justification for",
        r"Provide methods for", r"Can you help me find a way to",
        r"Can you provide legal advice on", r"Should", r"Why do",
        r"What's the most efficient way to", r"Can you tell me why", r"Can you tell me",
    ]
    question_starts.sort(key=len, reverse=True)
    cleaned = re.sub(r"^(?:" + "|".join(question_starts) + r")\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"[.!?\"']+$", "", cleaned).strip()


class TextPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}.txt")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        raw_prompt = self.prompts[idx]
        return {"prompt": clean_prompt(raw_prompt), "metadata": {"raw_prompt": raw_prompt}, "original_index": idx}

class PromptFileDataset(Dataset):
    def __init__(self, prompt_file, sample_size=None, seed=42):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        conversation_markers = ("<|im_start|>", "<|im_end|>", "[inst]", "[/inst]", "user:", "assistant:")
        prompts = [prompt for prompt in prompts if not any(marker in prompt.lower() for marker in conversation_markers)]
        if sample_size is not None:
            if sample_size > len(prompts):
                raise ValueError(f"Requested {sample_size} prompts, but only {len(prompts)} eligible prompts were found")
            rng = np.random.default_rng(seed)
            prompts = rng.choice(prompts, size=sample_size, replace=False).tolist()

        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        raw_prompt = self.prompts[idx]
        return {"prompt": clean_prompt(raw_prompt), "metadata": {"raw_prompt": raw_prompt}, "original_index": idx}


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}_metadata.jsonl")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx], "original_index": idx}


def collate_fn(examples):
    prompts = [example["prompt"] for example in examples]
    metadatas = [example["metadata"] for example in examples]
    indices = [example["original_index"] for example in examples]
    return prompts, metadatas, indices


def resolve_sd3_model_path():
    default_cache_dir = "/autodl-fs/data/DiffusionNFT/model/hub/models--stabilityai--stable-diffusion-3.5-medium"
    snapshots_dir = os.path.join(default_cache_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        snapshot_names = sorted(os.listdir(snapshots_dir))
        if snapshot_names:
            return os.path.join(snapshots_dir, snapshot_names[-1])
    return "stabilityai/stable-diffusion-3.5-medium"


def main(args):
    # --- Distributed Setup ---
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if args.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None

    if is_main_process(rank):
        print(f"Running distributed evaluation with {world_size} GPUs.")
        if enable_amp:
            print(f"Using mixed precision: {args.mixed_precision}")
        os.makedirs(args.output_dir, exist_ok=True)
        if args.save_images:
            os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    results_filepath = os.path.join(args.output_dir, "evaluation_results.jsonl")

    # --- Load Model and Pipeline ---
    if is_main_process(rank):
        print("Loading model and pipeline...")

    if args.model_type == "sd3":
        model_path = resolve_sd3_model_path()
        if is_main_process(rank):
            print(f"Loading SD3.5 base model from: {model_path}")
        pipeline = StableDiffusion3Pipeline.from_pretrained(model_path, local_files_only=os.path.isdir(model_path))
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
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.lora_hf_path:
        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, args.lora_hf_path)
        pipeline.transformer = pipeline.transformer.merge_and_unload()
    elif args.checkpoint_path:
        lora_path = os.path.join(args.checkpoint_path, "lora")
        if is_main_process(rank):
            print(f"Loading LoRA weights from: {lora_path}")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                f"LoRA directory not found at {lora_path}. Ensure your checkpoint has a 'lora' subdirectory."
            )

        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
        pipeline.transformer.load_adapter(lora_path, adapter_name="default", is_trainable=False)

    pipeline.transformer.eval()
    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.transformer.to(device, dtype=text_encoder_dtype)
    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # --- Load Dataset with Distributed Sampler ---
    dataset_path = f"dataset/{args.dataset}"
    if is_main_process(rank):
        print(f"Loading dataset from: {dataset_path}")

    if args.prompt_file:
        dataset = PromptFileDataset(args.prompt_file, args.prompt_sample_size, args.prompt_seed)
        all_reward_scorers = {"jailguard": 1.0}
        eval_batch_size = max(1, 4 // args.num_images_per_prompt)
        if is_main_process(rank):
            sampled_prompt_path = os.path.join(args.output_dir, "sampled_prompts.txt")
            with open(sampled_prompt_path, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(dataset.prompts) + "\n")
            print(f"Loaded {len(dataset)} raw prompts from {args.prompt_file}")
            print(f"Saved sampled prompts to {sampled_prompt_path}")
    elif args.dataset == "geneval":
        dataset = GenevalPromptDataset(dataset_path, split="test")
        all_reward_scorers = {"geneval": 1.0}
        eval_batch_size = 14
    elif args.dataset == "ocr":
        dataset = TextPromptDataset(dataset_path, split="test")
        all_reward_scorers = {"ocr": 1.0}
        eval_batch_size = 16
    elif args.dataset == "pickscore":
        dataset = TextPromptDataset(dataset_path, split="test")
        all_reward_scorers = {
            "imagereward": 1.0,
            "pickscore": 1.0,
            "aesthetic": 1.0,
            "unifiedreward": 1.0,
            "clipscore": 1.0,
            "hpsv2": 1.0,
        }
        eval_batch_size = 16
    elif args.dataset == "drawbench":
        dataset = TextPromptDataset(dataset_path, split="test")
        all_reward_scorers = {
            "imagereward": 1.0,
            "pickscore": 1.0,
            "aesthetic": 1.0,
            "unifiedreward": 1.0,
            "clipscore": 1.0,
            "hpsv2": 1.0,
        }
        eval_batch_size = 5
    elif args.dataset == "safebench":
        dataset = TextPromptDataset(dataset_path, split="test")
        all_reward_scorers = {
            "jailguard": 1.0,
        }
        # judge 服务低并发（max-running-requests=4，cuda-graph-max-bs-decode=4），
        # 每 batch 最多 4 张图，避免图捕获规格外的 batch size。
        eval_batch_size = max(1, 4 // args.num_images_per_prompt)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # --- Instantiate Reward Models ---
    # --gen_only 时只出图，不初始化任何打分模型（因此也不需要蓝队/裁判服务在线）。
    # 图像随后可由 scripts/score_from_images.py 用多个蓝队 VLM 反复打分，避免重复采样。
    scoring_fn = None
    if args.gen_only:
        if is_main_process(rank):
            print("[gen_only] 跳过奖励模型初始化，只生成并保存图像")
    else:
        if is_main_process(rank):
            print("Initializing reward models...")
        if args.dataset == "safebench":
            # jailguard 评分器返回 {"jailguard": ..., "avg": ...} 字典（训练脚本同款格式），
            # 不走 multi_score 的通用 list 封装，由下面的循环直接解析。
            scoring_fn = jailguard_defense_score_sglang(device, direct_prompt=False, judge_only=True)
        else:
            scoring_fn = multi_score(device, all_reward_scorers)

    # --- Evaluation Loop ---
    results_this_rank = []

    for batch in tqdm(dataloader, desc=f"Evaluating (Rank {rank})", disable=not is_main_process(rank)):
        prompts, metadata, indices = batch
        original_batch_size = len(prompts)
        prompts = [prompt for prompt in prompts for _ in range(args.num_images_per_prompt)]
        metadata = [item for item in metadata for _ in range(args.num_images_per_prompt)]
        sample_ids = [
            original_index * args.num_images_per_prompt + repeat_index
            for original_index in indices
            for repeat_index in range(args.num_images_per_prompt)
        ]
        current_batch_size = len(prompts)

        # 固定扩散采样噪声（common random numbers）：
        # 用 sample_id 派生种子，base 与 lora 两次运行会用【完全相同】的初始噪声，
        # 差异只来自权重而不来自随机种子，配对比较的方差会大幅下降。
        # 以 batch 首个 sample_id 派生种子，保证结果与批大小无关且可复现。
        gen = torch.Generator(device=device).manual_seed(
            int(args.gen_seed) + int(sample_ids[0]) * 1000003
        )

        with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
            with torch.no_grad():
                images = pipeline(
                    prompts,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    output_type="pt",
                    height=args.resolution,
                    width=args.resolution,
                    generator=gen,
                )[0]

        if args.gen_only:
            all_scores = {}
        else:
            all_scores, _ = scoring_fn(images, prompts, metadata, only_strict=False)

        for i in range(current_batch_size):
            sample_idx = sample_ids[i]
            result_item = {
                "sample_id": sample_idx,
                "prompt": prompts[i],
                "metadata": metadata[i] if metadata else {},
                "scores": {},
            }

            if args.save_images:
                image_path = os.path.join(args.output_dir, "images", f"{sample_idx:05d}.jpg")
                pil_image = Image.fromarray((images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil_image.save(image_path)
                result_item["image_path"] = image_path

            for score_name, score_values in all_scores.items():
                if isinstance(score_values, torch.Tensor):
                    result_item["scores"][score_name] = score_values[i].detach().cpu().item()
                else:
                    result_item["scores"][score_name] = float(score_values[i])

            results_this_rank.append(result_item)

        del images
        torch.cuda.empty_cache()

    # --- Gather and Save Results ---
    dist.barrier()

    all_gathered_results = [None] * world_size
    dist.all_gather_object(all_gathered_results, results_this_rank)

    if is_main_process(rank):
        flat_results = [item for sublist in all_gathered_results for item in sublist]

        flat_results.sort(key=lambda x: x["sample_id"])

        with open(results_filepath, "w") as f_out:
            for result_item in flat_results:
                f_out.write(json.dumps(result_item) + "\n")

        print(f"\nEvaluation finished. All {len(flat_results)} results saved to {results_filepath}")

        if args.gen_only:
            print("[gen_only] 图像与 manifest 已落盘，跳过打分与均分统计")
            cleanup_distributed()
            return

        all_scores_agg = defaultdict(list)

        for result in flat_results:
            for score_name, score_value in result["scores"].items():
                if isinstance(score_value, (int, float)):
                    all_scores_agg[score_name].append(score_value)

        average_scores = {
            name: np.mean(list(filter(lambda score: score != -10.0, scores))) for name, scores in all_scores_agg.items()
        }

        print("\n--- Average Scores ---")
        if not average_scores:
            print("No scores were found to average.")
        else:
            for name, avg_score in sorted(average_scores.items()):
                print(f"{name:<20}: {avg_score:.4f}")
        print("----------------------")

        avg_scores_filepath = os.path.join(args.output_dir, "average_scores.json")
        with open(avg_scores_filepath, "w") as f_avg:
            json.dump(average_scores, f_avg, indent=4)
        print(f"Average scores also saved to {avg_scores_filepath}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained diffusion model in a distributed manner.")
    parser.add_argument(
        "--lora_hf_path",
        type=str,
        default="",
        help="Huggingface path for LoRA.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Local path to the LoRA checkpoint directory (e.g., './save/run_name/checkpoints/checkpoint-5000').",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["sd3"],
        help="Type of the base model ('sd3').",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["geneval", "ocr", "pickscore", "drawbench", "safebench"], help="Dataset type."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_output",
        help="Directory to save evaluation results and generated images.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=28, help="Number of inference steps for the diffusion pipeline."
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of the generated images.")
    parser.add_argument(
        "--save_images", action="store_true", help="Include this flag to save generated images to the output directory."
    )
    parser.add_argument(
        "--gen_only",
        action="store_true",
        help="Only generate + save images and the manifest; skip reward-model init and scoring. "
             "Pair with scripts/score_from_images.py to score the same images with several "
             "blue-team VLMs (generate once, score many times). Implies --save_images.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of independently sampled images to generate for each prompt.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between 'no', 'fp16', or 'bf16'.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="",
        help="Optional raw prompt file. When set, it overrides the built-in dataset split.",
    )
    parser.add_argument(
        "--prompt_sample_size",
        type=int,
        default=None,
        help="Optional number of prompts to sample without replacement from --prompt_file.",
    )
    parser.add_argument("--prompt_seed", type=int, default=42, help="Random seed for raw prompt sampling.")
    parser.add_argument(
        "--gen_seed",
        type=int,
        default=1234,
        help="Seed for the diffusion sampling generator (common random numbers). "
             "With the same seed, a base run and a LoRA run see identical initial noise, "
             "so the paired difference only reflects the weights and its variance drops "
             "a lot. Change it to get an independent replicate.",
    )

    args = parser.parse_args()
    if args.num_images_per_prompt < 1:
        parser.error("--num_images_per_prompt must be at least 1")
    if args.gen_only and not args.save_images:
        # gen_only 的唯一产物就是图像，必须落盘
        args.save_images = True
    main(args)
