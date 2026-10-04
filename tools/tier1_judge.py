"""Tier-1 擦除集判读脚本（P1-c 根因判别）—— `[实验员·砚]` R367 预注册。

用途：读 the-world-data/p1c_erase/ 的 4 seed CSV，按已锁判据判 H1/H2/不定。
用法：
    python tools/tier1_judge.py [--data-dir the-world-data/p1c_erase] [--out docs/Tier-1判读结果-YYYYMMDD.md]

判据（天平 2026-10-03 锁定，R225 红线）：
  ① pep_frac_read_moore ≥0.80 ⇒ H1 带｜≤0.50 ⇒ H2 带（pep = 被擦除集）
  ② pep_d_med ≤2.0 ⇒ H1｜≥3.0 ⇒ H2
  ③ ΔR = pos_frac_read_moore − pep_frac_read_moore（差，非比）；≤0.05 ⇒ H1｜≥0.25 ⇒ H2
  ④ 一致性锁：≥3/4 种子同带 + 池化值离带界 ≥2×SD；逐 seed 须 P1/P2 同向，异向 ⇒ 判别不出
  ⑤ 判 H2 需 SNR = (0.862 − pep池化) / 种子间SD ≥2（T1 §3.3-3，仅用于判 H2）

🔴 本脚本不跑批（R117：起跑令只在 fish）；只读已有数据。
"""

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================================
# 判据阈值（锁定，R225）
# ============================================================================================
THRESHOLDS = {
    "pep_frac_read_moore": {"H1": 0.80, "H2": 0.50},  # ≥H1 ⇒ H1带；≤H2 ⇒ H2带
    "pep_d_med": {"H1": 2.0, "H2": 3.0},  # ≤H1 ⇒ H1；≥H2 ⇒ H2
    "delta_R": {"H1": 0.05, "H2": 0.25},  # ≤H1 ⇒ H1；≥H2 ⇒ H2
    "consensus": {"min_seeds": 3, "total_seeds": 4},  # ≥3/4 同带
    "pool_margin_sd": 2.0,  # 池化值离带界 ≥2×SD
    "h2_snr_min": 2.0,  # 判 H2 需 SNR ≥2
}


# ============================================================================================
# 数据读取
# ============================================================================================
def read_p1c_csv(fp: str) -> Dict:
    """读单个 seed 的 p1c_erase CSV，返回池化统计。"""
    rows = list(csv.DictReader(open(fp, encoding="utf-8")))
    if not rows:
        return {"error": f"空文件: {fp}"}

    # 🔴 delta_R 不在 CSV 中，须由 pos_frac_read_moore − pep_frac_read_moore 现算（T1 §3.1）
    # 列名须与探针输出一致（experiments/p1c_erasure_probe.py）
    raw_cols = ["pos_frac_read_moore", "pep_frac_read_moore", "pep_d_med"]

    # 池化（跨窗口平均）
    pooled = {}
    for col in raw_cols:
        vals = [float(r[col]) for r in rows if col in r and r[col] not in ("", "NaN")]
        if vals:
            pooled[col] = {
                "mean": st.mean(vals),
                "median": st.median(vals),
                "stdev": st.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        else:
            pooled[col] = {"mean": float("nan"), "median": float("nan"), "stdev": 0.0, "n": 0}

    # ΔR = pos − pep（差，非比；T1 §3.1）
    pos_mean = pooled["pos_frac_read_moore"]["mean"]
    pep_mean = pooled["pep_frac_read_moore"]["mean"]
    if not (math.isnan(pos_mean) or math.isnan(pep_mean)):
        delta_r = pos_mean - pep_mean
        # 种子间 SD 用逐窗口差值算（更稳健）
        deltas = [
            float(r.get("pos_frac_read_moore", "NaN")) - float(r.get("pep_frac_read_moore", "NaN"))
            for r in rows
            if r.get("pos_frac_read_moore", "NaN") not in ("", "NaN")
            and r.get("pep_frac_read_moore", "NaN") not in ("", "NaN")
        ]
        delta_stdev = st.stdev(deltas) if len(deltas) > 1 else 0.0
        pooled["delta_R"] = {"mean": delta_r, "median": delta_r, "stdev": delta_stdev, "n": len(deltas)}
    else:
        pooled["delta_R"] = {"mean": float("nan"), "median": float("nan"), "stdev": 0.0, "n": 0}

    return {"file": fp, "n_windows": len(rows), "pooled": pooled}


def load_all_seeds(data_dir: str) -> Dict[int, Dict]:
    """读 4 seed 的 CSV（207/208/209/210）。"""
    seeds = {}
    for seed in [207, 208, 209, 210]:
        fp = Path(data_dir) / f"p1c_erase_s{seed}_t3000.csv"
        if not fp.exists():
            print(f"⚠️ 缺失: {fp}（跳过）")
            continue
        seeds[seed] = read_p1c_csv(str(fp))
    return seeds


# ============================================================================================
# 判据检查
# ============================================================================================
def judge_single_metric(value: float, metric_name: str) -> Tuple[Optional[str], str]:
    """单指标判 H1/H2/不定。返回 (判定, 说明)。NaN ⇒ 判不定（fail-loud，不 crash）。"""
    if math.isnan(value):
        return None, f"NaN ⇒ 不定（数据缺失）"

    th = THRESHOLDS.get(metric_name)
    if not th:
        return None, f"未知指标: {metric_name}"

    # 🔴 delta_R 方向反转：≤H1 ⇒ H1（低差 = 擦除影响小 = H1）；≥H2 ⇒ H2
    if metric_name == "delta_R":
        if value <= th["H1"]:
            return "H1", f"{value:.3f} ≤ {th['H1']} ⇒ H1"
        elif value >= th["H2"]:
            return "H2", f"{value:.3f} ≥ {th['H2']} ⇒ H2"
        else:
            return None, f"{value:.3f} 在 ({th['H1']}, {th['H2']}) ⇒ 不定"
    else:
        # pep_frac_read_moore: ≥H1 ⇒ H1；≤H2 ⇒ H2
        # pep_d_med: ≤H1 ⇒ H1；≥H2 ⇒ H2
        if metric_name == "pep_d_med":
            if value <= th["H1"]:
                return "H1", f"{value:.3f} ≤ {th['H1']} ⇒ H1"
            elif value >= th["H2"]:
                return "H2", f"{value:.3f} ≥ {th['H2']} ⇒ H2"
            else:
                return None, f"{value:.3f} 在 ({th['H1']}, {th['H2']}) ⇒ 不定"
        else:  # pep_frac_read_moore
            if value >= th["H1"]:
                return "H1", f"{value:.3f} ≥ {th['H1']} ⇒ H1"
            elif value <= th["H2"]:
                return "H2", f"{value:.3f} ≤ {th['H2']} ⇒ H2"
            else:
                return None, f"{value:.3f} 在 ({th['H2']}, {th['H1']}) ⇒ 不定"


def check_consensus(judgments: List[Optional[str]]) -> Tuple[bool, str]:
    """检查一致性锁：≥3/4 种子同带。"""
    from collections import Counter
    cnt = Counter(judgments)
    most_common, count = cnt.most_common(1)[0] if cnt else (None, 0)

    if count >= THRESHOLDS["consensus"]["min_seeds"]:
        return True, f"{count}/{THRESHOLDS['consensus']['total_seeds']} 同判 {most_common}"
    else:
        return False, f"无一致（{dict(cnt)}）"


def check_pool_margin(pool_value: float, threshold: float, sd: float) -> Tuple[bool, str]:
    """检查池化值离带界 ≥2×SD。"""
    margin = abs(pool_value - threshold)
    required = THRESHOLDS["pool_margin_sd"] * sd
    if margin >= required:
        return True, f"|{pool_value:.3f} − {threshold}| = {margin:.3f} ≥ {required:.3f} (2×SD)"
    else:
        return False, f"|{pool_value:.3f} − {threshold}| = {margin:.3f} < {required:.3f} (2×SD)"


def compute_snr(signal: float, noise_sd: float) -> float:
    """信噪比 = |signal| / noise_sd。"""
    return abs(signal) / noise_sd if noise_sd > 0 else float("inf")


# ============================================================================================
# 主判读流程
# ============================================================================================
def judge_tier1(seeds: Dict[int, Dict]) -> Dict:
    """Tier-1 总判读。返回结构化结果。"""
    result = {
        "seeds": {},
        "consensus": {},
        "final_verdict": None,
        "notes": [],
    }

    # 1. 逐 seed 判读
    for seed, data in seeds.items():
        if "error" in data:
            result["seeds"][seed] = {"error": data["error"]}
            continue

        pooled = data["pooled"]
        seed_judgment = {"file": data["file"], "n_windows": data["n_windows"], "metrics": {}}

        # 三个主指标
        for metric in ["pep_frac_read_moore", "pep_d_med", "delta_R"]:
            val = pooled[metric]["mean"]
            verdict, explanation = judge_single_metric(val, metric)
            seed_judgment["metrics"][metric] = {
                "value": val,
                "stdev": pooled[metric]["stdev"],
                "verdict": verdict,
                "explanation": explanation,
            }

        # 🔴 单 seed 综合判（T1 §3.1/3.2：P1/P2 同向才可判；异向 ⇒ 判别不出）
        # P1 = pep_frac_read_moore 方向；P2 = delta_R 方向（pep_d_med 为辅助）
        pep_v = seed_judgment["metrics"]["pep_frac_read_moore"]["verdict"]
        dr_v = seed_judgment["metrics"]["delta_R"]["verdict"]
        if pep_v is not None and dr_v is not None and pep_v == dr_v:
            seed_verdict = pep_v  # P1/P2 同向
        elif pep_v is None and dr_v is None:
            seed_verdict = None  # 两者都不定
        else:
            seed_verdict = None  # 异向 ⇒ 判别不出
        seed_judgment["seed_verdict"] = seed_verdict
        seed_judgment["p1_p2_agree"] = (pep_v == dr_v) if (pep_v and dr_v) else False

        result["seeds"][seed] = seed_judgment

    # 2. 一致性检查
    seed_verdicts = [s.get("seed_verdict") for s in result["seeds"].values() if "seed_verdict" in s]
    consensus_ok, consensus_msg = check_consensus(seed_verdicts)
    result["consensus"] = {"ok": consensus_ok, "message": consensus_msg, "verdicts": seed_verdicts}

    # 3. 池化值离界检查（用全 seed 池化）
    pool_stats = {}
    for metric in ["pep_frac_read_moore", "pep_d_med", "delta_R"]:
        vals = [s["metrics"][metric]["value"] for s in result["seeds"].values() if "metrics" in s]
        if vals:
            pool_stats[metric] = {
                "pool_mean": st.mean(vals),
                "pool_stdev": st.stdev(vals) if len(vals) > 1 else 0.0,
            }
    result["pool_stats"] = pool_stats

    # 4. 最终判定
    if not consensus_ok:
        result["final_verdict"] = "不定（一致性不满足）"
        result["notes"].append(f"⚠️ 一致性锁失败: {consensus_msg}")
    else:
        # 取多数判
        from collections import Counter
        cnt = Counter(seed_verdicts)
        final_verdict = cnt.most_common(1)[0][0]

        # 🔴 判 H2 需 SNR ≥2（T1 §3.3-3：SNR = (0.862 − pep池化) / 种子间SD）
        if final_verdict == "H2":
            pep_pool = pool_stats["pep_frac_read_moore"]["pool_mean"]
            pep_sd = pool_stats["pep_frac_read_moore"]["pool_stdev"]
            snr = compute_snr(0.862 - pep_pool, pep_sd)
            if snr < THRESHOLDS["h2_snr_min"]:
                result["final_verdict"] = "不定（H2 但 SNR < 2）"
                result["notes"].append(f"⚠️ SNR = ({0.862} − {pep_pool:.3f}) / {pep_sd:.3f} = {snr:.2f} < {THRESHOLDS['h2_snr_min']}（T1 §3.3-3）")
            else:
                result["final_verdict"] = f"H2（SNR = {snr:.2f} ≥ {THRESHOLDS['h2_snr_min']}）"
        else:
            result["final_verdict"] = final_verdict

    return result


# ============================================================================================
# 输出
# ============================================================================================
def format_markdown(result: Dict) -> str:
    """格式化为 Markdown 判读稿。"""
    lines = [
        "# Tier-1 擦除集判读结果（P1-c 根因判别）",
        "",
        f"**判读时间**：2026-10-04  ",
        f"**判据来源**：天平 2026-10-03 锁定（R225 红线） ",
        f"**数据目录**：`the-world-data/p1c_erase/` ",
        "",
        "---",
        "",
        "## 一、逐 seed 判读",
        "",
    ]

    for seed, data in result["seeds"].items():
        if "error" in data:
            lines.append(f"### seed {seed}：❌ {data['error']}")
            lines.append("")
            continue

        lines.append(f"### seed {seed}（{data['n_windows']} 窗）")
        lines.append("")
        lines.append("| 指标 | 池化均值 | SD | 判定 | 说明 |")
        lines.append("|---|---|---|---|---|")
        for metric in ["pep_frac_read_moore", "pep_d_med", "delta_R"]:
            m = data["metrics"][metric]
            lines.append(f"| `{metric}` | {m['value']:.3f} | {m['stdev']:.3f} | {m['verdict'] or '不定'} | {m['explanation']} |")
        lines.append("")
        lines.append(f"**单 seed 综合判**：{data['seed_verdict'] or '不定'}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 二、一致性检查",
        "",
        f"**结果**：{'✅ 通过' if result['consensus']['ok'] else '❌ 失败'}  ",
        f"**详情**：{result['consensus']['message']}  ",
        "",
    ])

    if result.get("pool_stats"):
        lines.extend([
            "## 三、池化统计",
            "",
            "| 指标 | 池化均值 | 池化 SD |",
            "|---|---|---|",
        ])
        for metric, stats in result["pool_stats"].items():
            lines.append(f"| `{metric}` | {stats['pool_mean']:.3f} | {stats['pool_stdev']:.3f} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 四、最终判定",
        "",
        f"**🔴 {result['final_verdict']}**",
        "",
    ])

    if result["notes"]:
        lines.append("**备注**：")
        for note in result["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "**判读人**：砚 🪶  ",
        "**复核**：待镜 D4 复核  ",
        "**纪律**：本判读按预注册判据执行（R225）；未做事后调整。",
    ])

    return "\n".join(lines)


# ============================================================================================
# CLI
# ============================================================================================
def main():
    parser = argparse.ArgumentParser(description="Tier-1 擦除集判读（P1-c 根因判别）")
    parser.add_argument("--data-dir", default="the-world-data/p1c_erase", help="数据目录")
    parser.add_argument("--out", default=None, help="输出 Markdown 路径（默认打印到 stdout）")
    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        print(f"❌ 数据目录不存在: {args.data_dir}")
        print("   提示：数据将在 208/209 云机 + 207/210 本机跑完后落入")
        return 1

    seeds = load_all_seeds(args.data_dir)
    if not seeds:
        print(f"❌ 无有效 seed 数据（需 207/208/209/210）")
        return 1

    result = judge_tier1(seeds)
    md = format_markdown(result)

    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        sys.stdout.buffer.write(f"OK 判读稿已写入: {args.out}\n".encode("utf-8"))
    else:
        sys.stdout.buffer.write(md.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")

    return 0


if __name__ == "__main__":
    exit(main())
