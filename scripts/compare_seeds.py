# SPDX-License-Identifier: Apache-2.0
"""
跨 prompt_seed 的复现检验：同一个 LoRA checkpoint，换不同随机样本是否得到一致结论。

对每个 (seed, 蓝队) 输出三组量：
  1. 整体 Δ（全样本平均）            —— 老口径，通常不显著
  2. 条件 Δ（只在 base 拒绝的样本上） —— 真正的效应所在
  3. 拒绝翻转（攻破 vs 补回）+ 判官无关口径（regex_rewards，旧数据没这个字段）

兼容两种目录命名：
  新（解耦流水线）: train-random100-seed2027-base-1x-gen1234-blueqwen3vl
  旧（耦合流水线）: train-random100-seed450-twostage-judgeonly-base-blueqwen3vl-1x-gen1234

用法:
    /root/miniconda3/bin/python scripts/compare_seeds.py --seeds 450 2026 2027
"""

import argparse
import json
import os
import re

from stats_utils import binom_two_sided, jailguard_map, t_test

NEW_RE = re.compile(
    r"^train-random(?P<n>\d+)-seed(?P<seed>\d+)-(?P<kind>base|lora)-(?P<npx>\d+)x"
    r"(?P<tag>-gen\d+)?-blue(?P<alias>.+)$"
)
LEGACY_RE = re.compile(
    r"^train-random(?P<n>\d+)-seed(?P<seed>\d+)-(?P<variant>.+?)-(?P<kind>base|lora)"
    r"-blue(?P<alias>.+?)-(?P<npx>\d+)x(?P<tag>-gen\d+)?$"
)


def load_root(score_root, sample_size=None):
    """{(seed, noise, alias): {'base': {sid: rec}, 'lora': {...}}}"""
    out = {}
    if not score_root or not os.path.isdir(score_root):
        return out
    for name in sorted(os.listdir(score_root)):
        path = os.path.join(score_root, name)
        if not os.path.isdir(path):
            continue
        match = NEW_RE.match(name) or LEGACY_RE.match(name)
        if not match:
            continue
        info = match.groupdict()
        if sample_size is not None and int(info["n"]) != sample_size:
            continue
        result_file = os.path.join(path, "evaluation_results.jsonl")
        if not os.path.exists(result_file):
            continue
        rows = {}
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    rows[int(record["sample_id"])] = record
        if rows:
            key = (int(info["seed"]), info["tag"] or "random", info["alias"])
            out.setdefault(key, {})[info["kind"]] = rows
    return out


def analyse(base_rec, lora_rec):
    base = jailguard_map(base_rec)
    lora = jailguard_map(lora_rec)
    ids = sorted(set(base) & set(lora))

    overall = t_test([lora[i] - base[i] for i in ids])
    ref_ids = [i for i in ids if base[i] == 0]
    cond = t_test([lora[i] - base[i] for i in ref_ids]) if len(ref_ids) >= 2 else None

    n10 = sum(1 for i in ids if base[i] == 0 and lora[i] > 0)   # 攻破拒绝
    n01 = sum(1 for i in ids if base[i] > 0 and lora[i] == 0)   # 补回拒绝

    steps = None
    if (base_rec and all("regex_rewards" in r for r in base_rec.values())
            and all("regex_rewards" in r for r in lora_rec.values())):
        s_base = sum(1 for i in ids if base_rec[i]["regex_rewards"] == 1.0)
        s_lora = sum(1 for i in ids if lora_rec[i]["regex_rewards"] == 1.0)
        steps = (s_base / len(ids), s_lora / len(ids))

    return {"overall": overall, "cond": cond, "n_ref": len(ref_ids),
            "n10": n10, "n01": n01, "p_mcnemar": binom_two_sided(min(n10, n01), n10 + n01),
            "steps": steps, "n": len(ids)}


def pairs(data, seeds, aliases):
    for seed in seeds:
        for alias in aliases:
            key = (seed, "-gen1234", alias)
            entry = data.get(key)
            if entry and "base" in entry and "lora" in entry:
                yield seed, alias, analyse(entry["base"], entry["lora"])


def main(args):
    data = load_root(args.score_root, args.sample_size)
    if args.score_root_legacy:
        for key, value in load_root(args.score_root_legacy, args.sample_size).items():
            data.setdefault(key, value)          # 新流水线优先，旧目录只补缺
    if not data:
        print("没有找到可用的固定噪声打分结果")
        return 1

    seeds = args.seeds or sorted({k[0] for k in data})
    seeds = [s for s in seeds if any(k[0] == s for k in data)]
    aliases = args.aliases or sorted({k[2] for k in data})

    print(f"# 跨 seed 复现检验（固定噪声 gen_seed={args.gen_seed}，n={args.sample_size}）")
    print("口径: 分 > 0 = 越狱有进展；Δ = lora − base，逐 sample_id 配对。\n")

    print("## 1. 整体 Δ（全样本平均，老口径）")
    print(f"{'seed':<8}{'蓝队':<10}{'n':>5}{'Δ':>10}{'p':>10}{'95% CI':>22}")
    print("-" * 65)
    for seed, alias, res in pairs(data, seeds, aliases):
        r = res["overall"]
        lo, hi = r["mean"] - r["ci95"], r["mean"] + r["ci95"]
        print(f"{seed:<8}{alias:<10}{r['n']:>5}{r['mean']:>+10.4f}{r['p']:>10.4f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}")

    print("\n## 2. 条件 Δ（只在 base 成功拒绝的样本上）")
    print(f"{'seed':<8}{'蓝队':<10}{'拒绝数':>7}{'Δ':>10}{'t':>8}{'p':>10}{'95% CI':>22}{'显著':>6}")
    print("-" * 81)
    hits = total = 0
    for seed, alias, res in pairs(data, seeds, aliases):
        cond = res["cond"]
        if cond is None:
            continue
        total += 1
        sig = cond["p"] < 0.05
        hits += int(sig)
        lo, hi = cond["mean"] - cond["ci95"], cond["mean"] + cond["ci95"]
        print(f"{seed:<8}{alias:<10}{res['n_ref']:>7}{cond['mean']:>+10.3f}{cond['t']:>+8.2f}"
              f"{cond['p']:>10.4f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{'✅' if sig else '❌':>6}")
    print(f"\n→ 条件效应在 {hits}/{total} 个 (seed × 蓝队) 组合上 p<0.05")

    print("\n## 3. 拒绝翻转 + 判官无关口径（蓝队是否直接给出步骤）")
    print(f"{'seed':<8}{'蓝队':<10}{'攻破':>6}{'补回':>6}{'McNemar p':>11}{'步骤率 base→lora':>20}")
    print("-" * 61)
    for seed, alias, res in pairs(data, seeds, aliases):
        steps = res["steps"]
        step_txt = f"{steps[0]:.2f} -> {steps[1]:.2f}" if steps else "n/a (旧数据)"
        print(f"{seed:<8}{alias:<10}{res['n10']:>6}{res['n01']:>6}"
              f"{res['p_mcnemar']:>11.4f}{step_txt:>20}")

    print("\n读法：只有条件 Δ 跨 seed 稳定为正且显著，才说明这是 LoRA 的真实机制，")
    print("      而不是某一批样本的偶然（整体 Δ 被已配合样本的负向变化抵消，天然不显著）。")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-seed replication check.")
    parser.add_argument("--score_root", type=str, default="logs/nft/sd3/jailguard/eval/score")
    parser.add_argument("--score_root_legacy", type=str, default="logs/nft/sd3/jailguard/eval",
                        help="旧（耦合）流水线结果目录，用来把老 seed 一并纳入")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--aliases", type=str, nargs="*", default=None)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--gen_seed", type=int, default=1234)
    raise SystemExit(main(parser.parse_args()))
