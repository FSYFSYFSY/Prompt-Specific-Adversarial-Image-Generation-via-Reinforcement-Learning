# SPDX-License-Identifier: Apache-2.0
"""
第二张图：LoRA 的越狱效应到底体现在哪里。

和 plot_results.py（看整体平均分）互补，这张图专门画「条件效应」的证据：

  (a) 只在 base 成功拒绝的样本上，lora−base 的配对 Δ 及 95% CI
      —— 8/8 显著，这才是 LoRA 真正的贡献
  (b) 效应分解：整体 Δ = w_ref · Δ_ref + (1−w_ref) · Δ_rest
      —— 说明为什么整体平均分看起来「几乎没效果」
  (c) 拒绝翻转计数：攻破的拒绝数 vs 补回的拒绝数

用法:
    /root/miniconda3/bin/python scripts/plot_mechanism.py --out /root/autodl-fs/eval_mechanism.png
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from stats_utils import binom_two_sided, jailguard_map, load_records, t_test  # noqa: E402

FIXED = "-gen1234"
NOISE_LABEL = {FIXED: "fixed noise", "random": "random noise"}


def main(args):
    data = load_records(args.score_root, args.sample_size, args.seed)
    keys = sorted([k for k, v in data.items() if "base" in v and "lora" in v])
    aliases = sorted({k[1] for k in keys})
    # 按 base 均值排序，区分度好的在前
    order = []
    for alias in aliases:
        group = data.get((FIXED, alias)) or data.get(("random", alias))
        order.append((alias, np.mean(list(jailguard_map(group["base"]).values()))))
    aliases = [a for a, _ in sorted(order, key=lambda x: -x[1])]

    fig = plt.figure(figsize=(16, 5.2))
    fig.suptitle(
        "Where the LoRA jailbreak effect actually lives\n"
        "score > 0 = blue team failed to fully refuse; Δ = lora − base (paired, per sample)",
        fontsize=13,
    )

    # ---------- (a) 条件效应 forest ----------
    ax = fig.add_subplot(1, 3, 1)
    rows = [(noise, a) for noise in (FIXED, "random") for a in aliases if (noise, a) in keys]
    y = np.arange(len(rows))[::-1]
    cond_stats = []
    for yi, (noise, alias) in zip(y, rows):
        base = jailguard_map(data[(noise, alias)]["base"])
        lora = jailguard_map(data[(noise, alias)]["lora"])
        ids = [i for i in sorted(set(base) & set(lora)) if base[i] == 0]
        st = t_test([lora[i] - base[i] for i in ids])
        cond_stats.append(st)
        color = "#2f6fd0" if noise == FIXED else "#e07b39"
        ax.errorbar(st["mean"], yi, xerr=st["ci95"], fmt="o", color=color,
                    capsize=5, markersize=7, lw=2)
        ax.annotate(f" n={st['n']}  p={st['p']:.4f}", (st["mean"] + st["ci95"], yi),
                    va="center", fontsize=8.5)
    ax.axvline(0, color="#444444", lw=1.2, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} · {'fixed' if n == FIXED else 'random'}" for n, a in rows], fontsize=9)
    # x 轴范围按实际 CI 自适应，否则标注会被裁掉
    xmax = max(st["mean"] + st["ci95"] for st in cond_stats)
    ax.set_xlim(0, xmax * 1.55)
    ax.set_xlabel("Δ among samples the base model refused")
    sig = sum(1 for st in cond_stats if st["p"] < 0.05)
    title = f"(a) conditional effect: {sig}/{len(cond_stats)} significant"
    if sig == len(cond_stats):
        title += f"\n(all p ≤ {max(st['p'] for st in cond_stats):.3f})"
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # ---------- (b) 效应分解 ----------
    ax = fig.add_subplot(1, 3, 2)
    x = np.arange(len(aliases))
    width = 0.3
    ref_contrib, rest_contrib, nets = [], [], []
    for alias in aliases:
        base = jailguard_map(data[(FIXED, alias)]["base"])
        lora = jailguard_map(data[(FIXED, alias)]["lora"])
        ids = sorted(set(base) & set(lora))
        ref_ids = [i for i in ids if base[i] == 0]
        rest_ids = [i for i in ids if base[i] > 0]
        w_ref = len(ref_ids) / len(ids)
        d_ref = np.mean([lora[i] - base[i] for i in ref_ids]) if ref_ids else 0.0
        d_rest = np.mean([lora[i] - base[i] for i in rest_ids]) if rest_ids else 0.0
        ref_contrib.append(w_ref * d_ref)
        rest_contrib.append((1 - w_ref) * d_rest)
        nets.append(np.mean([lora[i] - base[i] for i in ids]))

    ax.bar(x - width / 2, ref_contrib, width, label="from breaking refusals  ($w_{ref}\\cdot\\Delta_{ref}$)",
           color="#2a9d8f", edgecolor="white")
    ax.bar(x + width / 2, rest_contrib, width, label="from already-compliant samples",
           color="#e76f51", edgecolor="white")
    ax.plot(x, nets, "kD", markersize=8, label="net Δ (overall average)")
    for i, net in enumerate(nets):
        ax.annotate(f"{net:+.3f}", (i, net), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="#444444", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(aliases, fontsize=9)
    ax.set_ylabel("contribution to the overall Δ")
    ax.set_title("(b) the net effect is a cancellation\n(green positive, red negative)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    # ---------- (c) 拒绝翻转计数 ----------
    ax = fig.add_subplot(1, 3, 3)
    broke, restored, pm = [], [], []
    for alias in aliases:
        base = jailguard_map(data[(FIXED, alias)]["base"])
        lora = jailguard_map(data[(FIXED, alias)]["lora"])
        ids = sorted(set(base) & set(lora))
        n10 = sum(1 for i in ids if base[i] == 0 and lora[i] > 0)
        n01 = sum(1 for i in ids if base[i] > 0 and lora[i] == 0)
        broke.append(n10)
        restored.append(n01)
        pm.append(binom_two_sided(min(n10, n01), n10 + n01))
    width = 0.35
    ax.bar(x - width / 2, broke, width, label="refusal broken by LoRA", color="#2a9d8f", edgecolor="white")
    ax.bar(x + width / 2, restored, width, label="refusal restored by LoRA", color="#c44e52", edgecolor="white")
    for i, (b, r, p) in enumerate(zip(broke, restored, pm)):
        ax.annotate(f"{b} vs {r}\np={p:.3f}", (i, max(b, r)), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(aliases, fontsize=9)
    ax.set_ylabel("number of samples (fixed noise)")
    ax.set_ylim(0, max(max(broke), max(restored)) * 1.45)
    up = sum(1 for b, r in zip(broke, restored) if b > r)
    ax.set_title(f"(c) refusal flips: {up}/{len(aliases)} blues break more\n"
                 f"refusals than they restore (and none is significant)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    if args.out_pdf:
        fig.savefig(args.out_pdf)
    print(f"图已保存: {args.out}")
    for alias, b, r, p, ref, net in zip(aliases, broke, restored, pm, ref_contrib, nets):
        print(f"  {alias:<10} 攻破 {b:>3} / 补回 {r:>3}  p={p:.4f}   拒绝桶贡献 {ref:+.4f}  净值 {net:+.4f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the mechanism behind the LoRA jailbreak effect.")
    parser.add_argument("--score_root", type=str, default="logs/nft/sd3/jailguard/eval/score")
    parser.add_argument("--out", type=str, default="/root/autodl-fs/eval_mechanism.png")
    parser.add_argument("--out_pdf", type=str, default="")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    raise SystemExit(main(parser.parse_args()))
