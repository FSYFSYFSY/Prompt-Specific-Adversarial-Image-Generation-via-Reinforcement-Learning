# SPDX-License-Identifier: Apache-2.0
"""
单个 prompt_seed 的 LoRA 对比报告图（4 面板），支持中文 / 英文两套标签。

  (a) 平均越狱分：有 LoRA vs 没 LoRA
  (b) 拒绝率：有 LoRA vs 没 LoRA
  (c) 逐样本转化：多少条从「拒绝」变成「服从」，多少条反向
  (d) 对 base 拒绝过的样本，LoRA 之后拿到多少分

用法:
    /root/miniconda3/bin/python scripts/plot_seed_report.py --seed 2026            # 中文
    /root/miniconda3/bin/python scripts/plot_seed_report.py --seed 2026 --lang en  # 英文
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_main import FONT_PATH, setup_chinese_font  # noqa: E402
from stats_utils import binom_two_sided, jailguard_map, load_records, t_test  # noqa: E402

FIXED = "-gen1234"
TRAIN_BLUE = "qwen3vl"          # LoRA 训练时使用的蓝队
# 裁判分是 (harm-1)/9，harm 取 1..10 的整数 → 分数只有这 10 个离散值。
# 注意：分数以 float32 存储（2/9 实际是 0.2222222238779068），
# 不能跟 round(k/9, 3) 做浮点精确比较，必须按 round(v * 9) 归到整数档。
N_LEVELS = 10
SCORE_LABELS = [f"{i / 9:.2f}" for i in range(N_LEVELS)]


def score_level(v):
    """把裁判分映射到 0..9 的整数档（对 float32 存储误差安全）。"""
    k = int(round(v * 9))
    if not 0 <= k < N_LEVELS:
        raise ValueError(f"越狱分超出 [0, 1]：{v!r}")
    return k

C_BASE = "#9aa4b0"
C_LORA = "#2f6fd0"
C_BOTH_REF = "#94a3b8"
C_BROKEN = "#d1495b"
C_RESTORED = "#2a9d8f"
C_BOTH_COMPLY = "#cbd5e1"

L = {
    "zh": {
        "carry": "★训练用蓝队",
        "suptitle": ("LoRA vs 没 LoRA  ·  prompt_seed = {seed}  ·  每蓝队 {n} 条 safebench prompt  ·  固定初始噪声\n"
                     "越狱分越高 = 蓝队越容易被绕过；「拒绝」= 裁判给最低危害分（0 分）"),
        "a_title": "(a) 整体分数：差别很小、多数达不到显著\n（原因见 (c)：涨的和跌的互相抵消）",
        "a_ylabel": "平均越狱分",
        "a_legend": ["没 LoRA", "有 LoRA"],
        "b_title": "(b) 拒绝率：LoRA 让蓝队少拒了一点\n（百分数越小 = 越容易被绕过）",
        "b_ylabel": "完全拒答的样本占比（%）",
        "b_legend": ["没 LoRA 的拒绝率", "有 LoRA 的拒绝率"],
        "b_annot": "{a:.0%} → {b:.0%}\n({d:+.0%} 点)",
        "c_title": "(c) 逐样本转化：有多少条从「拒绝」变成了「服从」\n红色 = LoRA 攻破的拒绝；绿色 = LoRA 反而补回的拒绝",
        "c_ylabel": "样本数（每个蓝队 100 条）",
        "c_segments": ["两边都拒绝（没被绕过）", "拒绝 → 服从（LoRA 攻破）",
                       "服从 → 拒绝（LoRA 补回）", "两边都服从"],
        "c_annot": "攻破 {n} 条\n= 全部样本的 {all:.0%}\n= 原拒绝样本的 {ref:.0%}",
        "d_title": "(d) 只看「没 LoRA 时被拒绝」的样本\n0 分那根 = LoRA 也没能攻破；右边的都是被攻破的（幅度多大）",
        "d_xlabel": "LoRA 之后拿到的越狱分（0 = 仍然拒绝，越大 = 越配合）",
        "d_ylabel": "样本数",
        "ns": "不显著",
        "console_seed": "=== seed {seed}（固定噪声，n={n}）===",
        "console_head": ["蓝队", "base均值", "lora均值", "Δ", "p", "拒绝率base", "拒绝率lora",
                         "攻破", "补回", "净", "McNemar p"],
    },
    "en": {
        "carry": "★ blue team used in LoRA training",
        "suptitle": ("LoRA vs no LoRA  ·  prompt_seed = {seed}  ·  {n} safebench prompts per blue team  ·  fixed initial noise\n"
                     "Higher jailbreak score = blue team easier to bypass; \"refusal\" = judge assigns the minimum harm score (0)"),
        "a_title": "(a) Overall score: the difference is small, mostly not significant\n(reason: see panel (c) - gains and losses cancel out)",
        "a_ylabel": "Mean jailbreak score",
        "a_legend": ["no LoRA (base SD3)", "with LoRA"],
        "b_title": "(b) Refusal rate: LoRA makes the blue team refuse slightly less\n(lower % = easier to bypass)",
        "b_ylabel": "Share of fully-refused samples (%)",
        "b_legend": ["refusal rate, no LoRA", "refusal rate, with LoRA"],
        "b_annot": "{a:.0%} → {b:.0%}\n({d:+.0%} pts)",
        "c_title": "(c) Per-sample transitions: how many go from \"refusal\" to \"compliance\"\nred = refusals broken by LoRA; green = refusals restored by LoRA",
        "c_ylabel": "Number of samples (100 per blue team)",
        "c_segments": ["refused in both (never bypassed)", "refusal → compliance (broken by LoRA)",
                       "compliance → refusal (restored by LoRA)", "compliant in both"],
        "c_annot": "broken {n}\n= {all:.0%} of all\n= {ref:.0%} of refused",
        "d_title": "(d) Only the samples the base model refused\nbar at 0 = LoRA still could not break it; further right = broken, and by how much",
        "d_xlabel": "Jailbreak score after LoRA (0 = still refused, higher = more compliant)",
        "d_ylabel": "Number of samples",
        "ns": "n.s.",
        "console_seed": "=== seed {seed} (fixed noise, n={n}) ===",
        "console_head": ["blue", "base mean", "lora mean", "Δ", "p", "refuse base", "refuse lora",
                         "broken", "restored", "net", "McNemar p"],
    },
}


def sig_label(p, lang):
    """显著性标记：* / ** / ***，不显著时用文字"""
    if p is None or not math.isfinite(p):
        return ""
    for cut, mark in ((0.001, "***"), (0.01, "**"), (0.05, "*")):
        if p < cut:
            return mark
    return L[lang]["ns"]


def analyse(entry):
    base = jailguard_map(entry["base"])
    lora = jailguard_map(entry["lora"])
    ids = sorted(set(base) & set(lora))
    n = len(ids)
    both_ref = [i for i in ids if base[i] == 0 and lora[i] == 0]
    broken = [i for i in ids if base[i] == 0 and lora[i] > 0]
    restored = [i for i in ids if base[i] > 0 and lora[i] == 0]
    both_comp = [i for i in ids if base[i] > 0 and lora[i] > 0]
    d_all = [lora[i] - base[i] for i in ids]
    d_ref = [lora[i] - base[i] for i in broken + both_ref]
    return {
        "n": n,
        "base_mean": float(np.mean([base[i] for i in ids])),
        "lora_mean": float(np.mean([lora[i] for i in ids])),
        "base_ci": t_test([base[i] for i in ids])["ci95"],
        "lora_ci": t_test([lora[i] for i in ids])["ci95"],
        "delta": t_test(d_all),
        "n_both_ref": len(both_ref), "n_broken": len(broken),
        "n_restored": len(restored), "n_both_comp": len(both_comp),
        "refuse_base": (len(both_ref) + len(broken)) / n,
        "refuse_lora": (len(both_ref) + len(restored)) / n,
        "p_flip": binom_two_sided(min(len(broken), len(restored)),
                                  len(broken) + len(restored)),
        "n_ref": len(d_ref),
        "cond": t_test(d_ref) if len(d_ref) >= 2 else None,
        "lora_scores_of_refused": [lora[i] for i in broken + both_ref],
    }


def figure_seed(stats, aliases, seed, n, lang, out):
    T = L[lang]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
    fig.suptitle(T["suptitle"].format(seed=seed, n=n), fontsize=14.5)

    x = np.arange(len(aliases))
    w = 0.36
    xtick_labels = [f"{a}\n{T['carry']}" if a == TRAIN_BLUE else a for a in aliases]

    # ---------- (a) 平均分 ----------
    ax = axes[0][0]
    bm = [stats[a]["base_mean"] for a in aliases]
    lm = [stats[a]["lora_mean"] for a in aliases]
    ax.bar(x - w / 2, bm, w, yerr=[stats[a]["base_ci"] for a in aliases], capsize=4,
           color=C_BASE, edgecolor="white", label=T["a_legend"][0])
    ax.bar(x + w / 2, lm, w, yerr=[stats[a]["lora_ci"] for a in aliases], capsize=4,
           color=C_LORA, edgecolor="white", label=T["a_legend"][1])
    for i, a in enumerate(aliases):
        st = stats[a]["delta"]
        top = max(bm[i] + stats[a]["base_ci"], lm[i] + stats[a]["lora_ci"])
        ax.annotate(f"{st['mean']:+.3f}\n{sig_label(st['p'], lang)}\np={st['p']:.3f}",
                    (i, top + 0.03), ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(T["a_ylabel"], fontsize=11)
    ax.set_title(T["a_title"], fontsize=12.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")

    # ---------- (b) 拒绝率 ----------
    ax = axes[0][1]
    rb = [stats[a]["refuse_base"] * 100 for a in aliases]
    rl = [stats[a]["refuse_lora"] * 100 for a in aliases]
    ax.bar(x - w / 2, rb, w, color=C_BASE, edgecolor="white", label=T["b_legend"][0])
    ax.bar(x + w / 2, rl, w, color=C_LORA, edgecolor="white", label=T["b_legend"][1])
    for i, a in enumerate(aliases):
        s = stats[a]
        ax.annotate(T["b_annot"].format(a=s["refuse_base"], b=s["refuse_lora"],
                                        d=s["refuse_lora"] - s["refuse_base"]),
                    (i, max(rb[i], rl[i]) + 2), ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel(T["b_ylabel"], fontsize=11)
    ax.set_title(T["b_title"], fontsize=12.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")

    # ---------- (c) 逐样本转化 ----------
    ax = axes[1][0]
    segments = [
        (T["c_segments"][0], C_BOTH_REF, [stats[a]["n_both_ref"] for a in aliases]),
        (T["c_segments"][1], C_BROKEN, [stats[a]["n_broken"] for a in aliases]),
        (T["c_segments"][2], C_RESTORED, [stats[a]["n_restored"] for a in aliases]),
        (T["c_segments"][3], C_BOTH_COMPLY, [stats[a]["n_both_comp"] for a in aliases]),
    ]
    bottom = np.zeros(len(aliases))
    for label, color, vals in segments:
        vals = np.array(vals, dtype=float)
        ax.bar(x, vals, 0.6, bottom=bottom, color=color, edgecolor="white", label=label)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.annotate(f"{int(v)}", (i, bottom[i] + v / 2), ha="center", va="center",
                            fontsize=9.5, color="#222222")
        bottom += vals
    for i, a in enumerate(aliases):
        s = stats[a]
        ax.annotate(T["c_annot"].format(n=s["n_broken"], all=s["n_broken"] / s["n"],
                                        ref=s["n_broken"] / max(1, s["n_broken"] + s["n_both_ref"])),
                    (i, 101.5), ha="center", va="bottom", fontsize=8.5, color="#a4172f")
    ax.set_xticks(x)
    ax.set_xticklabels(aliases, fontsize=10.5)
    ax.set_ylim(0, 126)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel(T["c_ylabel"], fontsize=11)
    ax.set_title(T["c_title"], fontsize=12.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")

    # ---------- (d) 被拒样本的得分分布 ----------
    ax = axes[1][1]
    width = 0.8 / len(aliases)
    for k, a in enumerate(aliases):
        scores = stats[a]["lora_scores_of_refused"]
        hist = [0] * N_LEVELS
        for v in scores:
            hist[score_level(v)] += 1
        assert sum(hist) == len(scores), f"{a}: 分档计数 {sum(hist)} != 样本数 {len(scores)}"
        pos = np.arange(N_LEVELS) + (k - (len(aliases) - 1) / 2) * width
        ax.bar(pos, hist, width, label=f"{a} (n={len(scores)})",
               color=["#d1495b", "#e07b39", "#2a9d8f", "#2f6fd0"][k % 4], edgecolor="white")
        detail = " ".join(f"{lab}:{c}" for lab, c in zip(SCORE_LABELS, hist) if c)
        print(f"  (d) {a:<9} n={len(scores):>3}  {detail}")
    ax.set_xticks(np.arange(N_LEVELS))
    ax.set_xticklabels(SCORE_LABELS, fontsize=9)
    ax.set_xlabel(T["d_xlabel"], fontsize=11)
    ax.set_ylabel(T["d_ylabel"], fontsize=11)
    ax.set_title(T["d_title"], fontsize=12.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    print(f"图已保存 / figure saved: {out}")


def main(args):
    setup_chinese_font()
    data = load_records(args.score_root, args.sample_size, args.seed)
    if not data:
        print(f"seed {args.seed} 没有打分结果")
        return 1

    stats = {}
    for (noise, alias), entry in data.items():
        if noise != FIXED or "base" not in entry or "lora" not in entry:
            continue
        stats[alias] = analyse(entry)
    aliases = sorted(stats, key=lambda a: -stats[a]["base_mean"])

    T = L[args.lang]
    print("\n" + T["console_seed"].format(seed=args.seed, n=args.sample_size))
    head = T["console_head"]
    print(f"{head[0]:<10}{head[1]:>9}{head[2]:>9}{head[3]:>9}{head[4]:>8}"
          f"{head[5]:>11}{head[6]:>11}{head[7]:>6}{head[8]:>6}{head[9]:>5}{head[10]:>11}")
    for a in aliases:
        s = stats[a]
        print(f"{a:<10}{s['base_mean']:>9.4f}{s['lora_mean']:>9.4f}{s['delta']['mean']:>+9.4f}"
              f"{s['delta']['p']:>8.3f}{s['refuse_base']:>10.0%}{s['refuse_lora']:>11.0%}"
              f"{s['n_broken']:>6}{s['n_restored']:>6}{s['n_broken'] - s['n_restored']:>+5}"
              f"{s['p_flip']:>11.4f}")

    if args.out:
        out = args.out
    elif args.lang == "en":
        out = f"/root/autodl-fs/LoRA_comparison_seed{args.seed}_EN.png"
    else:
        out = f"/root/autodl-fs/LoRA对比_seed{args.seed}.png"
    figure_seed(stats, aliases, args.seed, args.sample_size, args.lang, out)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-seed LoRA vs base report figure (zh / en).")
    parser.add_argument("--score_root", type=str, default="logs/nft/sd3/jailguard/eval/score")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--out", type=str, default="")
    raise SystemExit(main(parser.parse_args()))
