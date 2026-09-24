# SPDX-License-Identifier: Apache-2.0
"""
把 run_decoupled.sh 的打分结果画成图。

面板:
  (a) 固定噪声下 base / lora 的 jailguard 均值（误差棒 = 均值 95% CI）
  (b) 非固定噪声下同上
  (c) forest plot: 每个 (蓝队, 噪声模式) 的 Δ = lora − base 及 95% CI
  (d) Δ 按 base 分数分箱：LoRA 的收益集中在哪些样本上

因为没有中文字体，图内文字用英文。

用法:
    /root/miniconda3/bin/python scripts/plot_results.py \
        --score_root logs/nft/sd3/jailguard/eval/score \
        --out /root/autodl-fs/eval_decoupled.png
"""

import argparse
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

DIR_RE = re.compile(
    r"^train-random(?P<n>\d+)-seed(?P<seed>\d+)-(?P<kind>base|lora)-(?P<npx>\d+)x"
    r"(?P<tag>-gen\d+)?-blue(?P<alias>.+)$"
)

BASE_C = "#7f8c9b"
LORA_C = "#2f6fd0"
NOISE_LABEL = {"-gen1234": "Fixed noise (--gen_seed 1234)", "random": "Random noise (no --gen_seed)"}


# ---------- 纯 Python 的 t 分布（没有 scipy 时用）----------
def _betacf(a, b, x, maxit=200, eps=3e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + aa / c
            c = c if abs(c) > 1e-30 else 1e-30
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def two_sided_p(t, df):
    """P(|T_df| > |t|)"""
    if not math.isfinite(t) or df <= 0:
        return float("nan")
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def t_crit(df, alpha=0.05):
    if scipy_stats is not None:
        return float(scipy_stats.t.ppf(1 - alpha / 2, df=df))
    lo, hi = 0.0, 100.0                      # 二分求解双尾分位点
    for _ in range(200):
        mid = (lo + hi) / 2
        if two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load(score_root, sample_size, seed):
    data = {}
    for name in sorted(os.listdir(score_root)):
        path = os.path.join(score_root, name)
        match = DIR_RE.match(name)
        if not os.path.isdir(path) or not match:
            continue
        info = match.groupdict()
        if sample_size is not None and int(info["n"]) != sample_size:
            continue
        if seed is not None and int(info["seed"]) != seed:
            continue
        result_file = os.path.join(path, "evaluation_results.jsonl")
        if not os.path.exists(result_file):
            continue
        scores = {}
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    scores[int(record["sample_id"])] = float(record["scores"]["jailguard"])
        noise = info["tag"] or "random"
        data.setdefault((noise, info["alias"]), {})[info["kind"]] = scores
    return data


def stats_of(base, lora):
    common = sorted(set(base) & set(lora))
    b = np.array([base[i] for i in common], dtype=float)
    l = np.array([lora[i] for i in common], dtype=float)
    d = l - b
    n = len(d)
    tc = t_crit(n - 1)
    out = {
        "n": n,
        "base_mean": float(b.mean()),
        "lora_mean": float(l.mean()),
        "base_ci": tc * float(b.std(ddof=1)) / math.sqrt(n),
        "lora_ci": tc * float(l.std(ddof=1)) / math.sqrt(n),
        "delta": float(d.mean()),
        "delta_ci": tc * float(d.std(ddof=1)) / math.sqrt(n),
        "d": d,
        "b": b,
    }
    if d.std(ddof=1) > 0:
        t_stat = d.mean() / (d.std(ddof=1) / math.sqrt(n))
        out["t"] = float(t_stat)
        out["p"] = (float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n - 1)))
                    if scipy_stats is not None else two_sided_p(t_stat, n - 1))
    else:
        out["p"] = float("nan")
        out["t"] = float("nan")
    return out


def main(args):
    data = load(args.score_root, args.sample_size, args.seed)
    if not data:
        print("没有找到打分结果")
        return 1

    # 按 base 均值从高到低排（区分度好的放前面）
    order = []
    for alias in sorted({key[1] for key in data}):
        group = data.get(("random", alias)) or data.get(("-gen1234", alias))
        order.append((alias, np.mean(list(group.get("base", {0: 0}).values()))))
    aliases = [a for a, _ in sorted(order, key=lambda x: -x[1])]

    stats = {key: stats_of(**val) for key, val in data.items() if "base" in val and "lora" in val}
    noises = [n for n in ("-gen1234", "random") if any(k[0] == n for k in stats)]

    # 只跑了一种噪声时用 1x3（柱状 / forest / 分桶），两种噪声时用 2x2
    single_noise = len(noises) == 1
    if single_noise:
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
        bar_axes, forest_ax, bucket_ax = [axes[0]], axes[1], axes[2]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        bar_axes, forest_ax, bucket_ax = [axes[0][0], axes[0][1]], axes[1][0], axes[1][1]
    letters = "abcdef"
    fig.suptitle(
        f"LoRA vs base on JailGuard (n={args.sample_size}, prompt_seed={args.seed}, "
        f"checkpoint-42-80)" + ("" if single_noise else "\n")
        + "higher score = weaker blue-team defense (easier to jailbreak)",
        fontsize=13,
    )
    # ---------- (a)(b) 分组柱状图 ----------
    for idx, (ax, noise) in enumerate(zip(bar_axes, noises)):
        x = np.arange(len(aliases))
        width = 0.36
        base_means = [stats[(noise, a)]["base_mean"] for a in aliases]
        lora_means = [stats[(noise, a)]["lora_mean"] for a in aliases]
        base_err = [stats[(noise, a)]["base_ci"] for a in aliases]
        lora_err = [stats[(noise, a)]["lora_ci"] for a in aliases]

        ax.bar(x - width / 2, base_means, width, yerr=base_err, capsize=4,
               label="base", color=BASE_C, edgecolor="white")
        ax.bar(x + width / 2, lora_means, width, yerr=lora_err, capsize=4,
               label="lora", color=LORA_C, edgecolor="white")

        for i, alias in enumerate(aliases):
            s = stats[(noise, alias)]
            top = max(s["base_mean"] + s["base_ci"], s["lora_mean"] + s["lora_ci"])
            ax.annotate(
                f"Δ{s['delta']:+.3f}\np={s['p']:.3f}",
                (i, top + 0.045), ha="center", va="bottom", fontsize=8.5, color="#333333",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(aliases)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("mean jailguard")
        ax.set_title(f"({letters[idx]}) {NOISE_LABEL[noise]}", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right")

    # ---------- forest plot ----------
    ax = forest_ax
    rows = [(noise, a) for noise in noises for a in aliases]
    y = np.arange(len(rows))[::-1]
    for yi, (noise, alias) in zip(y, rows):
        s = stats[(noise, alias)]
        color = "#2f6fd0" if noise == "-gen1234" else "#e07b39"
        ax.errorbar(s["delta"], yi, xerr=s["delta_ci"], fmt="o", color=color,
                    capsize=5, markersize=7, lw=2)
        ax.annotate(f"  {s['delta']:+.3f}  [{s['delta'] - s['delta_ci']:+.3f}, "
                    f"{s['delta'] + s['delta_ci']:+.3f}]  p={s['p']:.3f}",
                    (s["delta"] + s["delta_ci"], yi), va="center", fontsize=8.5)
    ax.axvline(0, color="#444444", lw=1.2, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} · {'fixed' if n == '-gen1234' else 'random'}" for n, a in rows])
    ax.set_xlim(-0.15, 0.35)
    ax.set_xlabel("Δ = lora − base  (95% CI)")
    ax.set_title(f"({letters[len(noises)]}) paired effect of LoRA"
                 + (f" (all {len(rows)} comparisons positive)" if all(stats[k]["delta"] > 0 for k in rows) else ""),
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # ---------- Δ 按 base 分数分箱 ----------
    ax = bucket_ax
    bins = [(0.0, 0.0001, "0\n(refused)"), (0.0001, 0.34, "0–0.33"), (0.34, 0.67, "0.33–0.67"),
            (0.67, 1.0001, "0.67–1.0")]
    x = np.arange(len(bins))
    width = 0.2
    palette = ["#2f6fd0", "#1a9850", "#e07b39", "#b03a8c"]
    noise = noises[0]
    for k, alias in enumerate(aliases):
        means, ns = [], []
        s = stats[(noise, alias)]
        for lo, hi, _ in bins:
            mask = (s["b"] >= lo) & (s["b"] < hi)
            means.append(float(s["d"][mask].mean()) if mask.sum() else np.nan)
            ns.append(int(mask.sum()))
        ax.bar(x + (k - 1.5) * width, means, width, label=alias, color=palette[k % 4],
               edgecolor="white")
        for xi, (m, cnt) in enumerate(zip(means, ns)):
            if not math.isnan(m):
                ax.annotate(f"n={cnt}", (x[xi] + (k - 1.5) * width, m + 0.012),
                            ha="center", fontsize=7, color="#555555")
    ax.axhline(0, color="#444444", lw=1.2)
    # 高亮「base 完全拒绝」这一桶：LoRA 的主要效应都发生在这里
    ax.axvspan(-0.5, 0.5, color="#f2c94c", alpha=0.13, zorder=0)
    ref_deltas = []
    for alias in aliases:
        s = stats[(noise, alias)]
        mask = (s["b"] >= 0) & (s["b"] < bins[0][1])
        if mask.sum():
            ref_deltas.append(float(s["d"][mask].mean()))
    rng = f"{min(ref_deltas):+.2f} ~ {max(ref_deltas):+.2f}" if ref_deltas else "—"
    ax.annotate(
        f"LoRA mainly converts hard refusals into partial compliance (Δ > 0 here, {rng})\n"
        "and slightly lowers scores on already-compliant samples (Δ < 0 on the right)\n"
        "→ the small net Δ is a cancellation of these two opposite effects",
        xy=(0.02, 0.03), xycoords="axes fraction", va="bottom", fontsize=8.2, color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbfb", ec="#bbbbbb", lw=0.8),
    )
    ax.set_xticks(x)
    ax.set_xticklabels([b[2] for b in bins], fontsize=9)
    ax.set_xlabel("base score bucket (how much the base model already complied)")
    ax.set_ylabel("mean Δ (lora − base)")
    ax.set_title(f"({letters[len(noises) + 1]}) where the LoRA effect comes from · {NOISE_LABEL[noise]}", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.90 if single_noise else 0.945))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)

    if args.out_pdf:
        fig.savefig(args.out_pdf)

    print(f"图已保存: {args.out}")
    print()
    print(f"{'蓝队':<10} {'噪声':<8} {'base':>7} {'lora':>7} {'Δ':>8} {'±95%CI':>8} {'p':>7}")
    for (noise, alias) in rows:
        s = stats[(noise, alias)]
        label = "固定" if noise == "-gen1234" else "非固定"
        print(f"{alias:<10} {label:<8} {s['base_mean']:>7.4f} {s['lora_mean']:>7.4f} "
              f"{s['delta']:>+8.4f} {s['delta_ci']:>8.4f} {s['p']:>7.3f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot decoupled JailGuard comparison results.")
    parser.add_argument("--score_root", type=str,
                        default="logs/nft/sd3/jailguard/eval/score")
    parser.add_argument("--out", type=str, default="/root/autodl-fs/eval_decoupled.png")
    parser.add_argument("--out_pdf", type=str, default="", help="可选：同时导出 PDF")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    raise SystemExit(main(parser.parse_args()))
