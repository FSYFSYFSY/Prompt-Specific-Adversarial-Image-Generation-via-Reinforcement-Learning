# SPDX-License-Identifier: Apache-2.0
"""
共用的小统计工具：有 scipy 就用 scipy，没有就用纯 Python 兜底。

为什么要兜底：绘图/分析常用 miniconda 环境，里面没有 scipy，
之前 p 值会静默变成 nan。
"""

import json
import math
import os
import re

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

# 打分目录命名：
#   新（解耦流水线）: train-random100-seed2026-base-1x-gen1234-blueinternvl3
#   旧（耦合流水线）: train-random100-seed450-twostage-judgeonly-base-blueqwen3vl-1x-gen1234
DIR_RE = re.compile(
    r"^train-random(?P<n>\d+)-seed(?P<seed>\d+)-(?P<kind>base|lora)-(?P<npx>\d+)x"
    r"(?P<tag>-gen\d+)?-blue(?P<alias>.+)$"
)
LEGACY_RE = re.compile(
    r"^train-random(?P<n>\d+)-seed(?P<seed>\d+)-(?P<variant>.+?)-(?P<kind>base|lora)"
    r"-blue(?P<alias>.+?)-(?P<npx>\d+)x(?P<tag>-gen\d+)?$"
)


def load_roots(roots, sample_size=None):
    """
    合并多个结果目录，兼容新旧两种命名。
    返回 {(seed, noise, alias): {"base": {sid: record}, "lora": {sid: record}}}
    noise 为 "-genXXXX"（固定噪声）或 "random"（每轮随机）。
    """
    data = {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            match = DIR_RE.match(name) or LEGACY_RE.match(name)
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
                data.setdefault(key, {})[info["kind"]] = rows
    return data


def load_records(score_root, sample_size=None, seed=None):
    """
    读取打分目录，返回 {(noise, alias): {"base": {sid: record}, "lora": {sid: record}}}
    noise 取 "-gen1234"（固定噪声）或 "random"（每轮随机）。
    """
    data = {}
    if not os.path.isdir(score_root):
        return data
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
        rows = {}
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    rows[int(record["sample_id"])] = record
        if rows:
            data.setdefault((info["tag"] or "random", info["alias"]), {})[info["kind"]] = rows
    return data


def jailguard_map(records):
    return {sid: rec["scores"]["jailguard"] for sid, rec in records.items()}


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
    """双尾临界值 t_{1-alpha/2, df}"""
    if df <= 0:
        return float("nan")
    if scipy_stats is not None:
        return float(scipy_stats.t.ppf(1 - alpha / 2, df=df))
    lo, hi = 0.0, 100.0
    for _ in range(200):          # 二分：two_sided_p 随 t 单调下降
        mid = (lo + hi) / 2
        if two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def t_p_value(t, df):
    if scipy_stats is not None:
        return float(2 * (1 - scipy_stats.t.cdf(abs(t), df=df)))
    return two_sided_p(t, df)


def t_test(values):
    """单样本 t 检验（对配对差值数组）。返回 mean/t/p/se/ci95/sd/n。"""
    n = len(values)
    if n < 2:
        return {"n": n, "mean": (values[0] if n else float("nan")),
                "sd": float("nan"), "se": float("nan"), "t": float("nan"),
                "p": float("nan"), "ci95": float("nan")}
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    se = sd / math.sqrt(n)
    if se == 0:
        return {"n": n, "mean": mean, "sd": 0.0, "se": 0.0,
                "t": float("nan"), "p": float("nan"), "ci95": 0.0}
    t = mean / se
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t,
            "p": t_p_value(t, n - 1), "ci95": t_crit(n - 1) * se}


def binom_two_sided(k, n, p=0.5):
    """精确二项检验双尾 p（k 传较小的一侧计数即可，实现按「概率不大于观测」求和）"""
    if n == 0:
        return float("nan")
    if scipy_stats is not None:
        return float(scipy_stats.binomtest(k, n, p).pvalue)
    probs = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    observed = probs[k]
    return float(min(1.0, sum(pr for pr in probs if pr <= observed + 1e-12)))


def paired_delta(base, lora, subset_ids=None):
    """
    逐 sample_id 配对求差。base/lora 是 {sample_id: score}，
    subset_ids 非空时只在这些 id 上算（用于条件效应）。
    """
    ids = sorted(set(base) & set(lora))
    if subset_ids is not None:
        keep = set(subset_ids)
        ids = [i for i in ids if i in keep]
    return [lora[i] - base[i] for i in ids], ids
