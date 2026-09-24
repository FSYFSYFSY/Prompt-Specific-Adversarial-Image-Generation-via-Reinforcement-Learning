# SPDX-License-Identifier: Apache-2.0
"""
用「蓝队 VLM + 裁判」给**已经生成好的图片**离线打分。

背景：scripts/evaluation.py 原本把「SD3 出图」和「蓝队两阶段越狱打分」耦合在一个循环里，
换一个蓝队模型就要把整套扩散采样重跑一遍。但蓝队 VLM 只出现在打分阶段
（stage-1 看图输出步骤，stage-2 由 JailJudge 打分），和出图完全无关，
所以图像只需生成一次（evaluation.py --gen_only），之后被所有蓝队复用。

用法:
    BLUE_VLM_BASE_URL=http://127.0.0.1:17141/v1 BLUE_VLM_MODEL=InternVL3-8B \
    python scripts/score_from_images.py \
        --images_dir logs/nft/sd3/jailguard/eval/gen/train-random100-seed2026-base-1x-gen1234 \
        --output_dir logs/nft/sd3/jailguard/eval/score/train-random100-seed2026-base-1x-gen1234-blueinternvl3 \
        --blue_alias internvl3

输入目录需要是 evaluation.py --gen_only 的输出（含 evaluation_results.jsonl + images/）。
输出目录会写入 evaluation_results.jsonl / average_scores.json / run_info.json。

注意：蓝队与裁判的采样都是 temperature=0，本脚本对同一批图是确定性的，
同样的 (图, 蓝队) 组合重复跑应当得到相同结果；若不一致，说明服务侧有非确定性
（例如 batch 调度导致的数值差异），可用同一目录跑两次来量化。
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flow_grpo.rewards import jailguard_defense_score_sglang  # noqa: E402


def load_manifest(images_dir):
    manifest = os.path.join(images_dir, "evaluation_results.jsonl")
    if not os.path.exists(manifest):
        raise FileNotFoundError(
            f"{manifest} 不存在。--images_dir 必须指向 evaluation.py --gen_only 的输出目录。"
        )
    with open(manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f"{manifest} 是空的")
    records.sort(key=lambda r: r["sample_id"])
    return records


def resolve_image_path(images_dir, record):
    """定位图片：优先用约定的 images/<sample_id:05d>.jpg，再回退到 manifest 里记录的路径。"""
    candidates = [
        os.path.join(images_dir, "images", f"{record['sample_id']:05d}.jpg"),
    ]
    recorded = record.get("image_path")
    if recorded:
        candidates.append(recorded)
        candidates.append(os.path.join(images_dir, os.path.basename(recorded)))
        candidates.append(os.path.join(REPO_ROOT, recorded))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"sample_id={record['sample_id']} 找不到图片，试过: {candidates}")


def load_images(images_dir, records, resolution):
    batch = []
    for record in records:
        path = resolve_image_path(images_dir, record)
        image = Image.open(path).convert("RGB")
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution))
        batch.append(np.asarray(image, dtype=np.uint8))
    return np.stack(batch, axis=0)


def main(args):
    records = load_manifest(args.images_dir)
    if args.limit:
        records = records[: args.limit]

    blue_url = os.getenv("BLUE_VLM_BASE_URL", "http://127.0.0.1:17141/v1")
    blue_model = os.getenv("BLUE_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

    os.makedirs(args.output_dir, exist_ok=True)
    results_filepath = os.path.join(args.output_dir, "evaluation_results.jsonl")

    print("=" * 60)
    print(f" 图片目录 : {args.images_dir}")
    print(f" 输出目录 : {args.output_dir}")
    print(f" 蓝队     : {blue_model}  ({blue_url})")
    print(f" 裁判     : http://127.0.0.1:17142/v1  (usail-hkust/JailJudge-guard)")
    print(f" 样本数   : {len(records)}   batch_size={args.batch_size}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 注意：环境变量在工厂函数调用时被读取，所以必须在调用之前设置好 BLUE_VLM_*
    scoring_fn = jailguard_defense_score_sglang(device, direct_prompt=False, judge_only=True)

    results = []
    for start in tqdm(range(0, len(records), args.batch_size), desc="Scoring"):
        chunk = records[start : start + args.batch_size]
        images = load_images(args.images_dir, chunk, args.resolution)
        prompts = [record["prompt"] for record in chunk]
        metadata = [record.get("metadata", {}) for record in chunk]

        scores, extra_metadata = scoring_fn(images, prompts, metadata, only_strict=False)
        jailguard_scores = scores["jailguard"]

        for i, record in enumerate(chunk):
            value = jailguard_scores[i]
            value = value.item() if isinstance(value, torch.Tensor) else float(value)
            item = {
                "sample_id": record["sample_id"],
                "prompt": record["prompt"],
                "metadata": record.get("metadata", {}),
                "scores": {"jailguard": value},
                "blue_alias": args.blue_alias,
                "blue_model": blue_model,
            }
            for key, values in (extra_metadata or {}).items():
                if isinstance(values, (list, tuple)) and len(values) > i:
                    item[key] = values[i]
            results.append(item)

        # 增量落盘，长任务中途挂掉也不至于全丢
        with open(results_filepath, "w", encoding="utf-8") as f_out:
            for item in results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    results.sort(key=lambda x: x["sample_id"])
    with open(results_filepath, "w", encoding="utf-8") as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    values = np.array([item["scores"]["jailguard"] for item in results], dtype=float)
    average_scores = {"jailguard": float(values.mean())}
    with open(os.path.join(args.output_dir, "average_scores.json"), "w", encoding="utf-8") as f:
        json.dump(average_scores, f, indent=4)

    zeros = int((values == 0).sum())
    run_info = {
        "images_dir": os.path.abspath(args.images_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "blue_alias": args.blue_alias,
        "blue_model": blue_model,
        "blue_vlm_base_url": blue_url,
        "n": len(results),
        "mean_jailguard": float(values.mean()),
        "zero_count": zeros,
        "zero_ratio": zeros / max(1, len(results)),
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=4, ensure_ascii=False)

    print()
    print("--- Summary ------------------------------------------")
    print(f"n            : {len(results)}")
    print(f"jailguard mean: {values.mean():.4f}")
    print(f"0 分(未越狱) : {zeros}/{len(results)} = {zeros / max(1, len(results)):.1%}")
    print(f"结果          : {results_filepath}")
    print("------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score pre-generated images with a blue-team VLM + JailJudge.")
    parser.add_argument("--images_dir", type=str, required=True, help="evaluation.py --gen_only 的输出目录")
    parser.add_argument("--output_dir", type=str, required=True, help="打分结果输出目录")
    parser.add_argument("--blue_alias", type=str, default="unknown", help="蓝队别名，仅用于记录到结果里")
    parser.add_argument("--batch_size", type=int, default=4, help="每批打分的图片数，与 sglang 并发配置匹配")
    parser.add_argument("--resolution", type=int, default=512, help="送进蓝队 VLM 前的图像边长")
    parser.add_argument("--limit", type=int, default=None, help="只打分前 N 条（冒烟测试用）")
    main(parser.parse_args())
