# SPDX-License-Identifier: Apache-2.0
"""
对比图：

  图 1（跨 prompt_seed）：同一个 LoRA，在不同随机样本上
      (a) 整体 Δ          (b) 条件 Δ（只在 base 拒绝的样本上）   (c) base 均值（蓝队区分度）
  图 2（噪声模式）：同一权重
      (a) 整体 Δ 固定 vs 非固定   (b) 条件 Δ 固定 vs 非固定   (c) 两轮逐样本相关

用法:
    /root/miniconda3/bin/python scripts/plot_compare.py
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from stats_utils import jailguard_map, load_roots, t_test  # noqa: E402

FIXED = "-gen1234"
NOISE_LABEL = {FIXED: "fixed noise", "random": "random noise"}
SEED_COLORS = {450: "#8d99ae", 2026: "#2f6fd0", 2027: "#e07b39", 2028: "#1a9850"}


def stars(p):
    if not math.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def analyze(entry):
    base = jailguard_map(entry["base"])
    lora = jailguard_map(entry["lora"])
    ids = sorted(set(base) & set(lora))
    overall = t_test([lora[i] - base[i] for i in ids])
    ref = [i for i in ids if base[i] == 0]
    cond = t_test([lora[i] - base[i] for i in ref]) if len(ref) >= 2 else None
    return {
        "overall": overall, "cond": cond, "n_ref": len(ref),
        "base_mean": float(np.mean([base[i] for i in ids])),
        "lora_mean": float(np.mean([lora[i] for i in ids])),
        "base": base, "lora": lora, "ids": ids,
    }


def grouped_delta_panel(ax, aliases, series, title, ylabel, annotate_p=True):
    """series: [(label, color, {alias: (mean, ci95, p)})]"""
    x = np.arange(len(aliases))
    width = 0.8 / max(1, len(series))
    for k, (label, color, values) in enumerate(series):
        means = [values.get(a, (float("nan"), 0, float("nan")))[0] for a in aliases]
        errs = [values.get(a, (float("nan"), 0, float("nan")))[1] for a in aliases]
        ps = [values.get(a, (float("nan"), 0, float("nan")))[2] for a in aliases]
        offset = (k - (len(series) - 1) / 2) * width
        pos = x + offset
        ax.bar(pos, means, width, yerr=errs, capsize=3, label=label, color=color,
               edgecolor="white")
        if annotate_p:
            for xi, (m, e, p) in zip(pos, zip(means, errs, ps)):
                if not math.isfinite(m):
                    continue
                top = m + (e if m >= 0 else -e)
                ax.annotate(stars(p), (xi, m + (e + 0.012 if m >= 0 else -e - 0.05)),
                            ha="center", fontsize=11, fontweight="bold", color="#222222")
    ax.axhline(0, color="#444444", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(aliases, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def figure_seeds(data, seeds, aliases, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    stats = {}
    for seed in seeds:
        for alias in aliases:
            entry = data.get((seed, FIXED, alias))
            if entry and "base" in entry and "lora" in entry:
                stats[(seed, alias)] = analyze(entry)

    used_seeds = [s for s in seeds if any((s, a) in stats for a in aliases)]
    series = []
    for seed in used_seeds:
        overall = {a: (stats[(seed, a)]["overall"]["mean"], stats[(seed, a)]["overall"]["ci95"],
                       stats[(seed, a)]["overall"]["p"]) for a in aliases if (seed, a) in stats}
        series.append((f"seed {seed}", SEED_COLORS.get(seed, "#666666"), overall))
    grouped_delta_panel(axes[0], aliases, series,
                        "(a) overall Δ (all samples)\nold metric — mostly not significant",
                        "Δ = lora − base  (95% CI)")

    series_cond = []
    for seed in used_seeds:
        cond = {}
        for a in aliases:
            if (seed, a) not in stats or stats[(seed, a)]["cond"] is None:
                continue
            st = stats[(seed, a)]["cond"]
            cond[a] = (st["mean"], st["ci95"], st["p"])
        series_cond.append((f"seed {seed}", SEED_COLORS.get(seed, "#666666"), cond))
    grouped_delta_panel(axes[1], aliases, series_cond,
                        "(b) conditional Δ (only samples base refused)\nthis is where LoRA actually acts",
                        "Δ among refused samples  (95% CI)")

    # (c) base 均值（蓝队区分度，跨 seed 稳定性）
    x = np.arange(len(aliases))
    width = 0.8 / len(used_seeds)
    for k, seed in enumerate(used_seeds):
        means = [stats[(seed, a)]["base_mean"] for a in aliases if (seed, a) in stats]
        pos = [xi for xi, a in zip(x, aliases) if (seed, a) in stats]
        axes[2].bar([p + (k - (len(used_seeds) - 1) / 2) * width for p in pos], means,
                    width, label=f"seed {seed}", color=SEED_COLORS.get(seed, "#666666"),
                    edgecolor="white")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(aliases, fontsize=9)
    axes[2].set_ylabel("mean jailguard of the base model")
    axes[2].set_title("(c) blue-model ranking is stable across seeds\n"
                      "(and the spread ≫ the LoRA effect)", fontsize=11)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=False, fontsize=9)

    fig.suptitle("Cross-seed comparison · JailGuard, n=100 per seed, LoRA = checkpoint-42-80, fixed noise\n"
                 "stars: * p<0.05, ** p<0.01, *** p<0.001 (paired t-test)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=200)
    print(f"图已保存: {out}")
    return stats


def figure_noise(data, seed, aliases, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    stats = {}
    for noise in (FIXED, "random"):
        for alias in aliases:
            entry = data.get((seed, noise, alias))
            if entry and "base" in entry and "lora" in entry:
                stats[(noise, alias)] = analyze(entry)

    noises = [n for n in (FIXED, "random") if any((n, a) in stats for a in aliases)]
    if len(noises) < 2:
        print(f"seed {seed} 只有 {len(noises)} 种噪声，跳过图 2")
        return stats

    series = []
    for noise in noises:
        vals = {a: (stats[(noise, a)]["overall"]["mean"], stats[(noise, a)]["overall"]["ci95"],
                    stats[(noise, a)]["overall"]["p"]) for a in aliases if (noise, a) in stats}
        series.append((NOISE_LABEL[noise], "#2f6fd0" if noise == FIXED else "#e07b39", vals))
    grouped_delta_panel(axes[0], aliases, series,
                        "(a) overall Δ by noise mode\nnoise mode barely matters",
                        "Δ = lora − base  (95% CI)")

    series_cond = []
    for noise in noises:
        vals = {}
        for a in aliases:
            if (noise, a) in stats and stats[(noise, a)]["cond"] is not None:
                st = stats[(noise, a)]["cond"]
                vals[a] = (st["mean"], st["ci95"], st["p"])
        series_cond.append((NOISE_LABEL[noise], "#2f6fd0" if noise == FIXED else "#e07b39", vals))
    grouped_delta_panel(axes[1], aliases, series_cond,
                        "(b) conditional Δ by noise mode\nsignificance is stable",
                        "Δ among refused samples  (95% CI)")

    # (c) 两轮之间的逐样本相关（同一 sample_id、不同初始噪声）
    x = np.arange(len(aliases))
    width = 0.35
    for k, noise in enumerate(noises):
        means = [stats[(noise, a)]["base_mean"] if (noise, a) in stats else np.nan for a in aliases]
        axes[2].bar(x + (k - 0.5) * width, means, width, color="#2f6fd0" if noise == FIXED else "#e07b39",
                    edgecolor="white", label=NOISE_LABEL[noise])
    corr_lines = []
    for alias in aliases:
        fixed = data.get((seed, FIXED, alias))
        rnd = data.get((seed, "random", alias))
        if not fixed or not rnd:
            continue
        bf, br = jailguard_map(fixed["base"]), jailguard_map(rnd["base"])
        ids = sorted(set(bf) & set(br))
        f = np.array([bf[i] for i in ids])
        r = np.array([br[i] for i in ids])
        corr = float(np.corrcoef(f, r)[0, 1]) if f.std() > 0 and r.std() > 0 else float("nan")
        corr_lines.append(f"{alias}: r={corr:.2f}")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(aliases, fontsize=9)
    axes[2].set_ylabel("base mean jailguard")
    axes[2].set_title("(c) same seed, different initial noise\nper-sample correlation of base scores",
                      fontsize=11)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].annotate("\n".join(corr_lines), xy=(0.02, 0.96), xycoords="axes fraction",
                     va="top", fontsize=8.5, color="#333333",
                     bbox=dict(boxstyle="round,pad=0.35", fc="#fbfbfb", ec="#bbbbbb", lw=0.8))

    fig.suptitle(f"Fixed vs random initial noise · seed {seed}, n=100, LoRA = checkpoint-42-80\n"
                 "stars: * p<0.05, ** p<0.01, *** p<0.001 (paired t-test)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=200)
    print(f"图已保存: {out}")
    return stats


def main(args):
    data = load_roots([args.score_root, args.score_root_legacy], args.sample_size)
    if not data:
        print("没有找到打分结果")
        return 1

    seeds = args.seeds or sorted({k[0] for k in data})
    order = []
    for alias in sorted({k[2] for k in data}):
        entry = data.get((seeds[-1], FIXED, alias)) or next(
            (v for k, v in data.items() if k[2] == alias), None)
        if entry and "base" in entry:
            order.append((alias, np.mean(list(jailguard_map(entry["base"]).values()))))
    aliases = [a for a, _ in sorted(order, key=lambda t: -t[1])]
    print(f"seed: {seeds}  蓝队(按 base 均值降序): {aliases}")

    figure_seeds(data, seeds, aliases, args.out_seeds)
    figure_noise(data, args.noise_seed, aliases, args.out_noise)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparison plots across seeds and noise modes.")
    parser.add_argument("--score_root", type=str, default="logs/nft/sd3/jailguard/eval/score")
    parser.add_argument("--score_root_legacy", type=str, default="logs/nft/sd3/jailguard/eval",
                        help="旧（耦合）流水线结果目录，用于纳入老 seed")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--noise_seed", type=int, default=2026, help="做噪声对比图的 seed（需两种噪声都有）")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--out_seeds", type=str, default="/root/autodl-fs/eval_compare_seeds.png")
    parser.add_argument("--out_noise", type=str, default="/root/autodl-fs/eval_compare_noise.png")
    raise SystemExit(main(parser.parse_args()))
