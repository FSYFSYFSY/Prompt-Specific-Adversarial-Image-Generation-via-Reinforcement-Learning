# SPDX-License-Identifier: Apache-2.0
"""
汇总 run_decoupled.sh 的打分结果，输出 base vs LoRA 的**配对**统计。

配对成立的前提：base 与 lora 用同一批 prompt、同一个 sample_id 顺序，
「固定噪声」那一轮还额外共用同一份初始噪声（CRN），因此可以逐样本求差。

用法:
    python scripts/paired_stats.py --score_root <.../eval/score> --out summary.md
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

from stats_utils import t_crit, t_p_value

# train-random100-seed2026-base-1x-gen1234-blueinternvl3
DIR_RE = re.compile(
    r"^train-random(?P<n>\d+)-seed(?P<seed>\d+)-(?P<kind>base|lora)-(?P<npx>\d+)x"
    r"(?P<tag>-gen\d+)?-blue(?P<alias>.+)$"
)


def blue_model_name(alias):
    return {
        "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
        "internvl3": "InternVL3-8B",
        "mini": "MiniCPM-V-2_6",
        "llava": "llava-onevision-qwen2-7b-ov",
    }.get(alias, alias)


def load_score_dir(path):
    """读取一个打分目录 -> {sample_id: score}"""
    result_file = os.path.join(path, "evaluation_results.jsonl")
    scores = {}
    if not os.path.exists(result_file):
        return scores
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            scores[int(record["sample_id"])] = float(record["scores"]["jailguard"])
    return scores


def paired_stats(base_scores, lora_scores):
    """逐 sample_id 配对：d = lora - base"""
    common = sorted(set(base_scores) & set(lora_scores))
    if len(common) < 2:
        return None
    b = np.array([base_scores[i] for i in common], dtype=float)
    l = np.array([lora_scores[i] for i in common], dtype=float)
    d = l - b
    n = len(d)
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1))
    se = sd_d / math.sqrt(n) if n > 0 else float("nan")

    tc = t_crit(n - 1)
    t_stat = float(mean_d / se) if se > 0 else float("nan")
    p_value = t_p_value(t_stat, n - 1) if se > 0 else float("nan")

    return {
        "n": n,
        "base_mean": float(b.mean()),
        "lora_mean": float(l.mean()),
        "delta": mean_d,
        "sigma_d": sd_d,
        "se": se,
        "ci_low": mean_d - tc * se,
        "ci_high": mean_d + tc * se,
        "t": t_stat,
        "p": p_value,
        "base_zero_ratio": float((b == 0).mean()),
        "lora_zero_ratio": float((l == 0).mean()),
        "base_nonzero": int((b != 0).sum()),
        "lora_nonzero": int((l != 0).sum()),
    }


def required_n(delta, sigma, alpha=0.05, power=0.8):
    """粗略样本量估计（配对 t 检验，两正态近似）"""
    if not delta or not sigma or delta == 0:
        return None
    z_a = 1.959964
    z_b = 0.841621
    return int(math.ceil((z_a + z_b) ** 2 * sigma**2 / delta**2))


def fmt(value, spec=".4f"):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return format(value, spec)


def main(args):
    groups = {}       # key -> {"base": scores, "lora": scores}
    meta = {}

    for name in sorted(os.listdir(args.score_root)):
        path = os.path.join(args.score_root, name)
        if not os.path.isdir(path):
            continue
        match = DIR_RE.match(name)
        if not match:
            continue
        info = match.groupdict()
        if args.seed is not None and int(info["seed"]) != args.seed:
            continue
        if args.sample_size is not None and int(info["n"]) != args.sample_size:
            continue
        scores = load_score_dir(path)
        if not scores:
            continue
        key = (int(info["npx"]), int(info["seed"]), info["tag"] or "random", info["alias"])
        groups.setdefault(key, {})[info["kind"]] = scores
        meta[key] = {"n": int(info["n"])}

    if not groups:
        print(f"[paired_stats] {args.score_root} 下没有可用的打分结果"
              f"（seed={args.seed}, sample_size={args.sample_size}）")
        return 1

    # ---------- 计算 ----------
    results = {}
    for key, kinds in sorted(groups.items()):
        if "base" in kinds and "lora" in kinds:
            stat = paired_stats(kinds["base"], kinds["lora"])
            if stat:
                results[key] = stat

    n_tests = max(1, len(results))
    bonferroni_alpha = 0.05 / n_tests

    # ---------- 生成 markdown ----------
    lines = []
    lines.append("# 解耦式矩阵评测汇总（图像只生成一次，多蓝队复用）")
    lines.append("")
    seeds = sorted({key[1] for key in groups})
    sizes = sorted({meta[key]["n"] for key in groups})
    lines.append(f"- prompt 采样 seed: {seeds}，样本量: {sizes}（`dataset/safebench/train.txt`）")
    lines.append(f"- 打分口径: `jailguard = (judge_harm - 1) / 9`，**分越高 = 蓝队越容易被越狱（防御越弱）**")
    lines.append(f"- base = 无 LoRA；lora = `checkpoint-42-80/lora`")
    lines.append(f"- 「固定噪声」= 出图时 `--gen_seed 1234`（base/lora 共用同一份初始噪声）")
    lines.append("- 每个 (蓝队, 噪声模式) 都是一次**逐 sample_id 配对**比较，Δ = lora − base")
    lines.append("")

    for tag, title in (("-gen1234", "固定噪声（gen_seed=1234）"), (None, "非固定噪声（每轮随机）")):
        tag_key = tag or "random"
        rows = [(key, stat) for key, stat in results.items() if key[2] == tag_key]
        if not rows:
            continue
        lines.append(f"## Δ 表：{title}")
        lines.append("")
        lines.append("| 蓝队 | n | base | lora | Δ (lora−base) | σ_d | Δ 的 95% CI | 配对 t | p | 显著(α=0.05) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for key, stat in sorted(rows):
            alias = key[3]
            sig = "❌"
            if not math.isnan(stat["p"]):
                if stat["p"] < bonferroni_alpha:
                    sig = "✅✅"
                elif stat["p"] < 0.05:
                    sig = "✅"
            elif stat["ci_low"] > 0 or stat["ci_high"] < 0:
                sig = "✅"
            lines.append(
                f"| {alias} | {stat['n']} | {fmt(stat['base_mean'])} | {fmt(stat['lora_mean'])} "
                f"| **{stat['delta']:+.4f}** | {fmt(stat['sigma_d'])} "
                f"| [{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}] | {stat['t']:+.2f} "
                f"| {fmt(stat['p'], '.3f')} | {sig} |"
            )
        lines.append("")
        lines.append(f"✅✅ = 过 Bonferroni 校正（α = 0.05/{n_tests} = {bonferroni_alpha:.4f}）；✅ = 仅过 α=0.05；❌ = 不显著")
        lines.append("")

    # ---------- 拒绝率表 ----------
    lines.append("## 蓝队区分度（0 分 = 蓝队成功拒绝，样本原封不动）")
    lines.append("")
    lines.append("| 蓝队 | 噪声 | base 非零 | lora 非零 | base 均值 | lora 均值 |")
    lines.append("|---|---|---|---|---|---|")
    for key, stat in sorted(results.items()):
        noise = "固定" if key[2] == "-gen1234" else "非固定"
        lines.append(
            f"| {key[3]} | {noise} | {stat['base_nonzero']}/{stat['n']} "
            f"| {stat['lora_nonzero']}/{stat['n']} | {fmt(stat['base_mean'])} | {fmt(stat['lora_mean'])} |"
        )
    lines.append("")

    # ---------- 噪声模式对比 ----------
    noise_rows = []
    for alias in sorted({key[3] for key in groups}):
        for kind in ("base", "lora"):
            fixed = groups.get((1, seeds[0], "-gen1234", alias), {}).get(kind)
            rnd = groups.get((1, seeds[0], "random", alias), {}).get(kind)
            if not fixed or not rnd:
                continue
            common = sorted(set(fixed) & set(rnd))
            f = np.array([fixed[i] for i in common])
            r = np.array([rnd[i] for i in common])
            corr = float(np.corrcoef(f, r)[0, 1]) if len(common) > 2 and f.std() > 0 and r.std() > 0 else float("nan")
            noise_rows.append((alias, kind, len(common), float(f.mean()), float(r.mean()), corr))
    if noise_rows:
        lines.append("## 噪声模式对比（固定 vs 非固定，同一蓝队同一权重）")
        lines.append("")
        lines.append("| 蓝队 | 权重 | n | 固定噪声均值 | 非固定噪声均值 | 差值 | 逐样本相关系数 |")
        lines.append("|---|---|---|---|---|---|---|")
        for alias, kind, n, fm, rm, corr in noise_rows:
            lines.append(f"| {alias} | {kind} | {n} | {fm:.4f} | {rm:.4f} | {fm - rm:+.4f} | {fmt(corr, '.3f')} |")
        lines.append("")
        lines.append("固定噪声只是**换了一批噪声**（不是消除噪声）：两轮的点估计差异反映的是「换图」带来的不确定性。")
        lines.append("")

    # ---------- 样本量建议 ----------
    lines.append("## 样本量建议")
    lines.append("")
    lines.append("| 蓝队 | 噪声 | σ_d | 检出 Δ=0.05 需 n | 检出 Δ=0.10 需 n | 检出 Δ=0.15 需 n |")
    lines.append("|---|---|---|---|---|---|")
    for key, stat in sorted(results.items()):
        noise = "固定" if key[2] == "-gen1234" else "非固定"
        n5 = required_n(0.05, stat["sigma_d"])
        n10 = required_n(0.10, stat["sigma_d"])
        n15 = required_n(0.15, stat["sigma_d"])
        lines.append(f"| {key[3]} | {noise} | {stat['sigma_d']:.3f} | {n5 or '—'} | {n10 or '—'} | {n15 or '—'} |")
    lines.append("")
    lines.append("> 出图只做一遍、多个蓝队复用之后，加大 n 的边际成本主要落在蓝队 VLM 的推理上，")
    lines.append("> 因此建议先把 n 加到上表给出的量级，再下结论。")
    lines.append("")

    markdown = "\n".join(lines)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(markdown + "\n")
        with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_tests": n_tests,
                    "bonferroni_alpha": bonferroni_alpha,
                    "results": {
                        "|".join(map(str, key)): stat for key, stat in results.items()
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(markdown)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise base-vs-LoRA stats for run_decoupled.sh outputs.")
    parser.add_argument("--score_root", type=str, required=True, help="打分结果根目录（.../eval/score）")
    parser.add_argument("--out", type=str, default="", help="markdown 汇总输出路径")
    parser.add_argument("--sample_size", type=int, default=None, help="只统计该样本量")
    parser.add_argument("--seed", type=int, default=None, help="只统计该 prompt seed")
    sys.exit(main(parser.parse_args()))
