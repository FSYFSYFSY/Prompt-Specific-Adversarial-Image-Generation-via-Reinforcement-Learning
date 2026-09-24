# SPDX-License-Identifier: Apache-2.0
"""
面向「结论」的主图，只回答四个问题：
  图 1  eval_main.png
    (a) 有 LoRA vs 没 LoRA，整体分数差多少？
    (b) 这个差别主要发生在哪些样本上？（base 拒绝过的 vs base 本来就配合的）
    (c) 不同蓝队模型表现如何？（拒绝率 × 平均越狱分，箭头表示 LoRA 带来的移动）
    (d) 「拒绝样本上的提升」在不同随机样本上稳定吗？
  图 2  eval_qwen3vl.png（LoRA 训练时用的就是 qwen3vl 蓝队，单独细看）
    (a) 逐样本散点：base 分 vs LoRA 分
    (b) Δ 的分布（全部 / 拆分拒绝与非拒绝）

中文字体：/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc

用法:
    /root/miniconda3/bin/python scripts/plot_main.py
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from stats_utils import jailguard_map, load_roots, t_test  # noqa: E402

FIXED = "-gen1234"
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
C_BASE = "#9aa4b0"      # 没 LoRA
C_LORA = "#2f6fd0"      # 有 LoRA
C_REF = "#2a9d8f"       # base 拒绝过的样本
C_REST = "#e76f51"      # base 本来就配合的样本
TRAIN_BLUE = "qwen3vl"  # LoRA 训练时使用的蓝队


def setup_chinese_font():
    if os.path.exists(FONT_PATH):
        font_manager.fontManager.addfont(FONT_PATH)
        name = font_manager.FontProperties(fname=FONT_PATH).get_name()
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        print(f"中文字体: {name}")
    else:
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
        print("⚠️ 未找到中文字体，图内中文可能显示为方块")
    plt.rcParams["axes.unicode_minus"] = False


def stars(p):
    if p is None or not math.isfinite(p):
        return ""
    for cut, mark in ((0.001, "***"), (0.01, "**"), (0.05, "*")):
        if p < cut:
            return mark
    return "（不显著）"


def collect(data, aliases, seeds, noise=FIXED):
    """把多个 seed 的同一蓝队合并：n 更大、CI 更窄，结论更稳。"""
    out = {}
    for alias in aliases:
        d, b_sc, l_sc = [], [], []
        per_seed = {}
        for seed in seeds:
            entry = data.get((seed, noise, alias))
            if not entry or "base" not in entry or "lora" not in entry:
                continue
            base = jailguard_map(entry["base"])
            lora = jailguard_map(entry["lora"])
            ids = sorted(set(base) & set(lora))
            sd = [lora[i] - base[i] for i in ids]
            d += sd
            b_sc += [base[i] for i in ids]
            l_sc += [lora[i] for i in ids]
            ref_ids = [i for i in ids if base[i] == 0]
            per_seed[seed] = {
                "n": len(ids), "n_ref": len(ref_ids),
                "delta": float(np.mean(sd)) if sd else float("nan"),
                "cond": t_test([lora[i] - base[i] for i in ref_ids]) if len(ref_ids) >= 2 else None,
            }
        if not d:
            continue
        ref_mask = [x == 0 for x in b_sc]
        out[alias] = {
            "n": len(d), "delta": t_test(d),
            "base_mean": float(np.mean(b_sc)), "lora_mean": float(np.mean(l_sc)),
            "base_ci": t_test(b_sc)["ci95"], "lora_ci": t_test(l_sc)["ci95"],
            "refuse_rate_base": float(np.mean(ref_mask)),
            "refuse_rate_lora": float(np.mean([x == 0 for x in l_sc])),
            "d_ref": [v for v, m in zip(d, ref_mask) if m],
            "d_rest": [v for v, m in zip(d, ref_mask) if not m],
            "per_seed": per_seed,
            "base_scores": b_sc, "lora_scores": l_sc,
        }
    return out


def figure_main(data, aliases, seeds, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("LoRA（checkpoint-42-80）对越狱评测的影响  ·  蓝队 = 负责拒答的视觉模型\n"
                 f"数据：safebench 随机样本，每轮 100 条 prompt，合并 prompt_seed {'/'.join(map(str, seeds))}"
                 f"（每蓝队 n={100 * len(seeds)}）",
                 fontsize=15)

    # ---------- (a) 有 LoRA vs 没 LoRA：整体 ----------
    ax = axes[0][0]
    x = np.arange(len(aliases))
    w = 0.36
    bm = [data[a]["base_mean"] for a in aliases]
    lm = [data[a]["lora_mean"] for a in aliases]
    bc = [data[a]["base_ci"] for a in aliases]
    lc = [data[a]["lora_ci"] for a in aliases]
    ax.bar(x - w / 2, bm, w, yerr=bc, capsize=4, color=C_BASE, edgecolor="white",
           label="没 LoRA（原始 SD3）")
    ax.bar(x + w / 2, lm, w, yerr=lc, capsize=4, color=C_LORA, edgecolor="white",
           label="有 LoRA（微调后）")
    for i, a in enumerate(aliases):
        st = data[a]["delta"]
        top = max(bm[i] + bc[i], lm[i] + lc[i])
        ax.annotate(f"差 {st['mean']:+.3f}\n{stars(st['p'])}\n(p={st['p']:.3f})",
                    (i, top + 0.03), ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(aliases, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("平均越狱分（越高 = 蓝队越容易被越狱）", fontsize=11)
    ax.set_title("(a) 整体对比：有 LoRA 和没 LoRA 差别很小，\n四个蓝队没有一个达到统计显著",
                 fontsize=12.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")

    # ---------- (b) 影响在哪里 ----------
    ax = axes[0][1]
    w = 0.36
    ref_means, rest_means, ref_err, rest_err = [], [], [], []
    for a in aliases:
        s1 = t_test(data[a]["d_ref"]) if len(data[a]["d_ref"]) >= 2 else None
        s2 = t_test(data[a]["d_rest"]) if len(data[a]["d_rest"]) >= 2 else None
        ref_means.append(s1["mean"] if s1 else np.nan)
        ref_err.append(s1["ci95"] if s1 else 0)
        rest_means.append(s2["mean"] if s2 else np.nan)
        rest_err.append(s2["ci95"] if s2 else 0)
    ax.bar(x - w / 2, ref_means, w, yerr=ref_err, capsize=4, color=C_REF, edgecolor="white",
           label="base 拒绝过的样本（LoRA 有发挥空间）")
    ax.bar(x + w / 2, rest_means, w, yerr=rest_err, capsize=4, color=C_REST, edgecolor="white",
           label="base 本来就配合的样本（没有提升空间）")
    for i, a in enumerate(aliases):
        s1 = t_test(data[a]["d_ref"]) if len(data[a]["d_ref"]) >= 2 else None
        s2 = t_test(data[a]["d_rest"]) if len(data[a]["d_rest"]) >= 2 else None
        if s1:
            ax.annotate(f"{stars(s1['p'])}\nn={len(data[a]['d_ref'])}",
                        (i - w / 2, s1["mean"] + s1["ci95"] + 0.015), ha="center",
                        va="bottom", fontsize=9, color="#1c6b62")
        if s2:
            ax.annotate(f"{stars(s2['p'])}\nn={len(data[a]['d_rest'])}",
                        (i + w / 2, s2["mean"] + 0.02), ha="center", va="bottom",
                        fontsize=9, color="#a2402a")
    ax.axhline(0, color="#444444", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(aliases, fontsize=11)
    ax.set_ylabel("Δ = 有 LoRA - 没 LoRA（越狱分变化）", fontsize=11)
    ax.set_ylim(-0.25, 0.62)
    ax.set_title("(b) 影响在哪里：提升全部集中在「base 能拒绝的样本」上\n"
                 "（另一边反而略微下降，两者相抵 → 整体看不出差别）", fontsize=12.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="lower left")

    # ---------- (c) 不同蓝队表现如何 ----------
    ax = axes[1][0]
    for a in aliases:
        s = data[a]
        ax.annotate("", xy=(s["refuse_rate_lora"], s["lora_mean"]),
                    xytext=(s["refuse_rate_base"], s["base_mean"]),
                    arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.6, alpha=0.8))
        ax.scatter([s["refuse_rate_base"]], [s["base_mean"]], s=110, color=C_BASE,
                   edgecolor="white", zorder=3)
        ax.scatter([s["refuse_rate_lora"]], [s["lora_mean"]], s=110, color=C_LORA,
                   edgecolor="white", zorder=3)
        label = f"{a}" + ("  ★训练时用的蓝队" if a == TRAIN_BLUE else "")
        ax.annotate(label, (s["refuse_rate_base"], s["base_mean"]),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10.5)
    ax.scatter([], [], s=110, color=C_BASE, edgecolor="white", label="没 LoRA")
    ax.scatter([], [], s=110, color=C_LORA, edgecolor="white", label="有 LoRA")
    ax.annotate("越靠这边\n= 区分度越好\n(拒绝少、分布铺满)", xy=(0.03, 0.92),
                xycoords="axes fraction", fontsize=10, color="#2a9d8f",
                bbox=dict(boxstyle="round,pad=0.35", fc="#eefaf7", ec="#9fd7cc", lw=0.8))
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("拒绝率（蓝队完全拒答的样本比例，越低越好用）", fontsize=11)
    ax.set_ylabel("平均越狱分", fontsize=11)
    ax.set_title("(c) 不同蓝队表现差异 >> LoRA 的影响（箭头 = LoRA 带来的变化，都很短）",
                 fontsize=12.5)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10.5, loc="lower right")

    # ---------- (d) 是否稳定 ----------
    ax = axes[1][1]
    y = np.arange(len(aliases))[::-1]
    for yi, a in zip(y, aliases):
        st = t_test(data[a]["d_ref"]) if len(data[a]["d_ref"]) >= 2 else None
        if not st:
            continue
        colors = [C_LORA if seed == seeds[-1] else ("#8ab4f0" if seed != seeds[0] else C_BASE)
                  for seed in seeds]
        # 每个 seed 单独算，检查结论是否随样本变化
        for k, seed in enumerate(seeds):
            ps = data[a]["per_seed"].get(seed)
            if not ps or not ps["cond"]:
                continue
            ax.errorbar(ps["cond"]["mean"], yi + (k - 1) * 0.18, xerr=ps["cond"]["ci95"],
                        fmt="o", color=colors[k], capsize=3, markersize=6, lw=1.6,
                        label=f"prompt_seed {seed}" if yi == y[0] else None)
        ax.annotate(f"合并 n={st['n']}  Δ={st['mean']:+.3f}   p={st['p']:.4f} {stars(st['p'])}",
                    (0.62, yi), fontsize=9.5, va="center", color="#1c6b62")
    ax.axvline(0, color="#444444", lw=1.2, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(aliases, fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("只在「base 拒绝过的样本」上的 Δ = LoRA - base（95% CI）", fontsize=11)
    ax.set_title("(d) 这个提升稳定吗：每个圆点是一个独立随机样本，\n三个样本一致为正且都显著",
                 fontsize=12.5)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160)
    print(f"图已保存: {out}")


def figure_qwen3vl(data, alias, out):
    s = data[alias]
    b = np.array(s["base_scores"])
    l = np.array(s["lora_scores"])
    d = l - b
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    fig.suptitle(f"细看 {alias}（LoRA 训练时用的就是这个蓝队）：改了什么？", fontsize=15)

    ax = axes[0]
    refused = b == 0
    jitter = np.random.default_rng(0).normal(0, 0.008, size=b.shape)
    ax.scatter(b[~refused], l[~refused], s=22, alpha=0.5, color=C_LORA, label="base 本来就配合")
    ax.scatter(b[refused] + jitter[refused], l[refused], s=34, alpha=0.85, color="#d1495b",
               marker="^", label=f"base 拒绝过（n={int(refused.sum())}）")
    ax.plot([0, 1], [0, 1], ls="--", color="#444444", lw=1.2, label="y = x（没变化）")
    ax.set_xlabel("没 LoRA 时的越狱分", fontsize=11)
    ax.set_ylabel("有 LoRA 时的越狱分", fontsize=11)
    ax.set_title("(a) 逐样本对比\n红三角在对角线以上 = LoRA 攻破了这次拒绝", fontsize=12)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    ax = axes[1]
    ax.hist(d, bins=np.arange(-1.01, 1.02, 0.1), color=C_LORA, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#444444", lw=1.4)
    ax.axvline(d.mean(), color="#d1495b", lw=1.8, label=f"平均 Δ={d.mean():+.3f}")
    ax.annotate(f"{(d == 0).mean():.0%} 的样本毫无变化\n（两边都是 0 或分数相同）",
                xy=(0.03, 0.95), xycoords="axes fraction", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#cccccc", lw=0.8))
    ax.set_xlabel("Δ = 有 LoRA - 没 LoRA", fontsize=11)
    ax.set_ylabel("样本数", fontsize=11)
    ax.set_title("(b) 所有样本的 Δ 分布\n正的和负的都很多 → 整体被抵消", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10)

    ax = axes[2]
    ax.hist(s["d_ref"], bins=np.arange(-0.1, 1.02, 0.1), color="#d1495b", alpha=0.8,
            edgecolor="white", label=f"base 拒绝过的样本（n={len(s['d_ref'])}）")
    ax.hist(s["d_rest"], bins=np.arange(-1.01, 0.2, 0.1), color=C_REST, alpha=0.75,
            edgecolor="white", label=f"base 本来就配合（n={len(s['d_rest'])}）")
    ax.axvline(0, color="#444444", lw=1.4)
    ax.set_xlabel("Δ = 有 LoRA - 没 LoRA", fontsize=11)
    ax.set_ylabel("样本数", fontsize=11)
    ax.set_title("(c) 按「base 是否拒绝」拆开看\n拒绝样本整体右移（提升），另一组左移", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=160)
    print(f"图已保存: {out}")


def main(args):
    setup_chinese_font()
    data = load_roots([args.score_root, args.score_root_legacy], args.sample_size)
    if not data:
        print("没有找到打分结果")
        return 1

    seeds = sorted({k[0] for k in data})
    aliases = sorted(
        {k[2] for k in data},
        key=lambda a: -max((np.mean(list(jailguard_map(data[k]["base"]).values()))
                            for k in data if k[2] == a and "base" in data[k]), default=0))
    agg = collect(data, aliases, seeds)
    print(f"prompt_seed: {seeds}  蓝队: {aliases}")

    figure_main(agg, [a for a in aliases if a in agg], seeds, args.out_main)
    if args.detail_alias in agg:
        figure_qwen3vl(agg, args.detail_alias, args.out_detail)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear, conclusion-oriented figures for the LoRA comparison.")
    parser.add_argument("--score_root", type=str, default="logs/nft/sd3/jailguard/eval/score")
    parser.add_argument("--score_root_legacy", type=str, default="logs/nft/sd3/jailguard/eval")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--detail_alias", type=str, default="qwen3vl")
    parser.add_argument("--out_main", type=str, default="/root/autodl-fs/LoRA对比_总图.png")
    parser.add_argument("--out_detail", type=str, default="/root/autodl-fs/LoRA对比_qwen3vl细节.png")
    raise SystemExit(main(parser.parse_args()))
