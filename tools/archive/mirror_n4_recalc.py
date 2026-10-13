#!/usr/bin/env python3
"""N1 密度批·镜独立复算（R371 令"真独立复算"）。

独立性声明：
- 输入源 = `the-world-data/p1c_density/p1c_density_d{patches}_s{seed}_t3000.summary.json`；
- 不复用 `tools/tier1_judge.py` / `tools/mirror_r369_recalc.py` / 探针 `p1c_erasure_probe` 内部逻辑；
- 从 JSON schema 独立读取（`pooled.pos_frac_read_moore` / `pooled.pos_n_dist_pairs`
  / `pooled.pep.frac_read_moore_pos` / `pooled.pep.n_pos` / `pooled.pos_d_med`
  / `extinct_at`）；不复用 PI 中间量。

判据 = N1《预注册-N1-密度梯度批-20261004》§3.1/§3.2/§四（跑前锁定 R225）：
- 主判据：`pos_frac_read_moore` 沿 D0→D1→D2→D3 首个 ≤0.50 落带档；
- 档级聚合：加权 Σ(run池化值 × run距离对数) / Σ(run距离对数)（N1 §3.1）；
- 三重锁 S1/S2/S3（N1 §3.2）：
  · S1 该档存活 run ≥3 且 ≥3/4 逐 run 池化 ≤0.50；
  · S2 (0.50 − 档值) ≥ 2× 种子间样本 SD（存活 run 池化值，ddof=1 等权）；
  · S3 pep_frac_read_moore ≤0.80（不落 H1）**且** pos_d_med ≥2.0；
- 生态门（N1 §四）：run 级 `extinct_at=null` 存活；档级 ≥3/4 存活；沿密侧→疏侧遇崩溃档 break 止步；
- 对照档异常（N1 §3.2 灰区红线末句）：D0 若落带 ⇒ fail-loud 判代码漂移；
- NaN 处置（N4 审 §三·② 采纳后补）：任一存活 run 池化 NaN ⇒ 该档判不定并 fail-loud。

用法：
    PYTHONIOENCODING=utf-8 python tools/mirror_n4_recalc.py \\
        [--data-dir the-world-data/p1c_density] \\
        [--patches 1700,850,425,213] [--seeds 207,208,209,210] \\
        [--out docs/N1密度批-镜独立复算-YYYYMMDD.md]

本脚本不跑批（R117）；只读已有数据。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# R98 纪律：GBK 控制台非 ASCII rc=1 兜底。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# N1 §3.1/§3.2 阈值（跑前锁定 R225）
POS_H2 = 0.50
PEP_H1 = 0.80
POS_D_MED_H1 = 2.0
S1_MIN_ALIVE = 3
S1_MIN_FALLING = 3
S2_MARGIN_FACTOR = 2.0

# N1 §2.1 密度档阶梯 D0→D3
BAND_LABEL = {1700: "D0", 850: "D1", 425: "D2", 213: "D3"}


def _fmt(v, spec=".4f"):
    return "n/a" if v is None else format(v, spec)


def _weighted(values, weights):
    if not values or not weights or len(values) != len(weights):
        return None
    tot = sum(weights)
    if tot == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / tot


def load_run(data_dir, patches, seed):
    path = Path(data_dir) / f"p1c_density_d{patches}_s{seed}_t3000.summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(summary):
    """从 summary.json 独立读取判据字段（不依赖探针内部逻辑）。"""
    pooled = summary.get("pooled") or {}
    pep = pooled.get("pep") or {}
    return {
        "seed": summary.get("seed"),
        "ticks": summary.get("ticks"),
        "extinct_at": summary.get("extinct_at"),
        # 主判据
        "pos_frac_read_moore": pooled.get("pos_frac_read_moore"),
        "pos_n_dist_pairs": pooled.get("pos_n_dist_pairs") or 0,
        "pos_d_med": pooled.get("pos_d_med"),
        # S3 判定量（d>0 子集口径，与 N1 §3.1 一致）
        "pep_frac_read_moore": pep.get("frac_read_moore_pos"),
        "pep_n_pos": pep.get("n_pos") or 0,
        "pep_d_med": pep.get("d_med_pos"),
        # 复现门留档
        "readback": summary.get("readback") or {},
        "argv": summary.get("argv") or [],
    }


def collect_band(data_dir, patches, seeds):
    alive, crashed, missing = [], [], []
    for s in seeds:
        sm = load_run(data_dir, patches, s)
        if sm is None:
            missing.append(s)
            continue
        m = extract_metrics(sm)
        if m["extinct_at"] is None:
            alive.append(m)
        else:
            crashed.append(m)
    return {"alive": alive, "crashed": crashed, "missing": missing}


def judge_band(patches, band, seeds):
    """返回判读结果（含 S1/S2/S3 逐项 + 通过 bool）。"""
    alive = band["alive"]
    out = {
        "patches": patches,
        "band": BAND_LABEL.get(patches, f"D?({patches})"),
        "n_alive": len(alive),
        "n_crashed": len(band["crashed"]),
        "n_missing": len(band["missing"]),
        "verdict": None,
        "reason": "",
        "passed": False,
        "pos_pool": None,
        "pep_pool": None,
        "pos_d_med_pool": None,
        "pep_d_med_pool": None,
        "pos_sd": None,
        "s1": None,
        "s2": None,
        "s3": None,
    }

    # 生态门：档级存活 <S1_MIN_ALIVE ⇒ 档判崩溃，不入主判据
    if len(alive) < S1_MIN_ALIVE:
        # 全缺档（数据未到位）≠ 生态崩溃——语义区分（起跑令/入仓状态自查用）
        if (len(alive) == 0 and len(band["crashed"]) == 0
                and len(band["missing"]) > 0):
            out["verdict"] = "缺档未跑"
            out["reason"] = (
                f"存活=0 崩溃=0 缺档={len(band['missing'])} ⇒ 数据未到位"
                "（起跑令未下或数据未入仓）")
        else:
            out["verdict"] = "档崩溃"
            out["reason"] = (
                f"存活 run {len(alive)}/{len(alive) + len(band['crashed'])} "
                f"< {S1_MIN_ALIVE} ⇒ 档判崩溃（N1 §四 档级）")
        return out

    # NaN fail-loud（N4 审 §三·② 采纳建议）
    for r in alive:
        if r["pos_frac_read_moore"] is None:
            raise RuntimeError(
                f"🔴 NaN fail-loud：档 {out['band']} (patches={patches}) "
                f"seed {r['seed']} pos_frac_read_moore 缺失 ⇒ 该档判不定并上板")

    pos_vals = [r["pos_frac_read_moore"] for r in alive]
    pos_ws = [r["pos_n_dist_pairs"] for r in alive]
    pos_pool = _weighted(pos_vals, pos_ws)
    pos_sd = statistics.stdev(pos_vals) if len(pos_vals) > 1 else 0.0

    # S1
    s1_count = sum(1 for v in pos_vals if v is not None and v <= POS_H2)
    s1_pass = s1_count >= S1_MIN_FALLING and len(alive) >= S1_MIN_ALIVE

    # S2
    s2_margin = (POS_H2 - pos_pool) if pos_pool is not None else None
    s2_required = S2_MARGIN_FACTOR * pos_sd
    s2_pass = (s2_margin is not None) and (s2_margin >= s2_required)

    # S3（pep 与 pos_d_med 同时满足）
    pep_vals = [r["pep_frac_read_moore"] for r in alive
                if r["pep_frac_read_moore"] is not None]
    pep_ws = [r["pep_n_pos"] for r in alive
              if r["pep_frac_read_moore"] is not None]
    pep_pool = _weighted(pep_vals, pep_ws) if pep_vals else None
    s3_pep_pass = (pep_pool is not None) and (pep_pool <= PEP_H1)

    dmed_vals = [r["pos_d_med"] for r in alive if r["pos_d_med"] is not None]
    dmed_ws = [r["pos_n_dist_pairs"] for r in alive if r["pos_d_med"] is not None]
    pos_d_med_pool = _weighted(dmed_vals, dmed_ws) if dmed_vals else None
    s3_dmed_pass = (pos_d_med_pool is not None) and (pos_d_med_pool >= POS_D_MED_H1)

    pepd_vals = [r["pep_d_med"] for r in alive if r["pep_d_med"] is not None]
    pepd_ws = [r["pep_n_pos"] for r in alive if r["pep_d_med"] is not None]
    pep_d_med_pool = _weighted(pepd_vals, pepd_ws) if pepd_vals else None

    passed = s1_pass and s2_pass and s3_pep_pass and s3_dmed_pass
    out.update({
        "pos_pool": pos_pool, "pep_pool": pep_pool,
        "pos_d_med_pool": pos_d_med_pool, "pep_d_med_pool": pep_d_med_pool,
        "pos_sd": pos_sd,
        "s1": {"count": s1_count, "alive": len(alive), "pass": s1_pass},
        "s2": {"margin": s2_margin, "required": s2_required, "pass": s2_pass},
        "s3": {"pep_pool": pep_pool, "pep_pass": s3_pep_pass,
               "pos_d_med": pos_d_med_pool, "dmed_pass": s3_dmed_pass,
               "pass": s3_pep_pass and s3_dmed_pass},
        "passed": passed,
    })
    if passed:
        out["verdict"] = "落带 H2"
        out["reason"] = "S1/S2/S3 三重锁全过"
    elif not s1_pass:
        out["verdict"] = "不落带"
        out["reason"] = (f"S1 未达（逐 run ≤0.50 的 seed 数 {s1_count}/{len(alive)} < "
                         f"{S1_MIN_FALLING}）")
    else:
        out["verdict"] = "不定"
        fails = []
        if not s2_pass:
            fails.append(f"S2 余量 {s2_margin:.3f} < 2×SD={s2_required:.3f}")
        if not s3_pep_pass:
            fails.append(f"S3.pep 档值 {_fmt(pep_pool)} > {PEP_H1}")
        if not s3_dmed_pass:
            fails.append(f"S3.pos_d_med 档值 {_fmt(pos_d_med_pool)} < {POS_D_MED_H1}")
        out["reason"] = " + ".join(fails) if fails else "三重锁不通过"
    return out


def run_batch(data_dir, patches_list, seeds):
    patches_list = sorted(patches_list, reverse=True)  # 密侧→疏侧
    per_band = {}
    for p in patches_list:
        band = collect_band(data_dir, p, seeds)
        per_band[p] = judge_band(p, band, seeds)

    trace = [per_band[p] for p in patches_list]

    # 对照档 D0 异常检测
    d0 = trace[0]
    if d0["passed"]:
        raise RuntimeError(
            f"🔴 对照档异常：D0 (patches={d0['patches']}) 判落带 H2 "
            f"（pos 档值={_fmt(d0['pos_pool'])} ≤{POS_H2}）⇒ 代码漂移嫌疑"
            "（N1 §3.2 灰区红线末句：预期 D0 pos ≈0.86–0.92 落灰区之上）；"
            "先查代码再判读。")

    # 沿密侧→疏侧遍历，遇崩溃/缺档 break 止步（更疏侧不再判读）
    STOP_VERDICTS = ("档崩溃", "缺档未跑")
    target = None
    effective_trace = []
    stopped_at = None
    for j in trace:
        effective_trace.append(j)
        if j["verdict"] in STOP_VERDICTS:
            stopped_at = j
            break
        if j["passed"] and target is None:
            target = j  # 首个落带档 = 目标（N1 §3.1 判定方向）
            break

    return {
        "data_dir": str(data_dir),
        "patches_list": patches_list,
        "seeds": seeds,
        "per_band": per_band,
        "effective_trace": effective_trace,
        "target": target,
        "stopped_at": stopped_at,
    }


def format_markdown(out):
    lines = [
        "# N1 密度批·镜独立复算（tools/mirror_n4_recalc.py）",
        "",
        f"**输入源**：`{out['data_dir']}/p1c_density_d{{patches}}_s{{seed}}_t3000.summary.json`  ",
        f"**判据**：N1《预注册-N1-密度梯度批-20261004》§3.1 主判据 "
        f"`pos_frac_read_moore ≤ {POS_H2}`（沿 D0→D3 首个落带档）  ",
        "**独立性**：不复用 tier1_judge / c22cd58 / 探针内部逻辑；"
        "从 JSON schema 独立读取；不复用 PI 中间量（R371 令）。",
        "",
        "---",
        "",
        "## 一、逐档判读",
        "",
        "| 档 | patches | 存活 | 崩溃 | 缺档 | pos 加权档值 | SD | "
        "pep 加权档值 | pos_d_med 加权 | S1 | S2 | S3 | 判定 | 说明 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for j in out["effective_trace"]:
        s1 = j.get("s1") or {}
        s2 = j.get("s2") or {}
        s3 = j.get("s3") or {}
        lines.append(
            f"| {j['band']} | {j['patches']} | {j['n_alive']} | {j['n_crashed']} | "
            f"{j['n_missing']} | {_fmt(j['pos_pool'])} | {_fmt(j['pos_sd'], '.4f')} | "
            f"{_fmt(j['pep_pool'])} | {_fmt(j['pos_d_med_pool'])} | "
            f"{'✅' if s1.get('pass') else '❌'} ({s1.get('count', 'n/a')}/{s1.get('alive', 'n/a')}) | "
            f"{'✅' if s2.get('pass') else '❌'} (余量 {_fmt(s2.get('margin'))} vs 2×SD {_fmt(s2.get('required'))}) | "
            f"{'✅' if s3.get('pass') else '❌'} (pep {_fmt(s3.get('pep_pool'))} ≤{PEP_H1} · d_med {_fmt(s3.get('pos_d_med'))} ≥{POS_D_MED_H1}) | "
            f"**{j['verdict']}** | {j['reason']} |")

    # 崩溃档/缺档披露
    for j in out["effective_trace"]:
        if j["n_crashed"] > 0 or j["n_missing"] > 0:
            lines.append("")
            lines.append(f"### {j['band']} (patches={j['patches']}) 崩溃/缺档披露")
            if j["n_crashed"] > 0:
                lines.append(f"- 崩溃 seed 数 = {j['n_crashed']}（不入主判据；详见 summary.json `extinct_at`）")
            if j["n_missing"] > 0:
                lines.append(f"- 缺档 seed 数 = {j['n_missing']}")

    lines.extend([
        "",
        "---",
        "",
        "## 二、首个落带档判定",
        "",
    ])
    t = out["target"]
    if t is not None:
        lines.extend([
            f"**🟢 阳性档**：{t['band']}（patches={t['patches']}）",
            f"- pos 加权档值 = {_fmt(t['pos_pool'])} ≤ {POS_H2} ⇒ 落 H2 带",
            f"- S1 = {t['s1']['count']}/{t['s1']['alive']} seeds 逐 run ≤0.50（阈值 ≥{S1_MIN_FALLING}）",
            f"- S2 余量 = {_fmt(t['s2']['margin'])} ≥ 2×SD = {_fmt(t['s2']['required'])}",
            f"- S3 同向 = pep 档值 {_fmt(t['s3']['pep_pool'])} ≤{PEP_H1} · pos_d_med 档值 {_fmt(t['s3']['pos_d_med'])} ≥{POS_D_MED_H1}",
            "",
            f"**语义边界**（N1 §3.3）：仅证明该密度档下 P1-c 私有性前提成立 ⇒ 可作 M1 装置定档候选，"
            "上呈 fish 裁定；不构成 M1 重启（R366 前置 5 道门照旧），不构成任何"
            "\"语言涌现\"表述（AGENTS.md 纪律锚点）。",
        ])
    elif out.get("stopped_at") is not None:
        stop = out["stopped_at"]
        prev = out["effective_trace"][-2] if len(out["effective_trace"]) >= 2 else None
        prev_desc = (f"判读至 {prev['band']} (patches={prev['patches']}) 为最高存活档"
                     if prev else "无存活档可判读")
        lines.extend([
            f"**🟡 生态门/缺档止步**：{stop['band']} (patches={stop['patches']}) "
            f"判 {stop['verdict']} —— {stop['reason']}",
            f"- {prev_desc}（N1 §四 止步规则：崩溃档更疏侧不再判读）；",
            f"- 若首个落带档在止步档**更疏侧** ⇒ 本批无法判定（须先解生态门，如降 pop / 缩 ticks 属新批）；",
            f"- 若首个落带档应存在于止步档**之前**（更密侧）⇒ 上板重议甲案阶梯设计。",
        ])
    else:
        lines.extend([
            "**🔴 四档全不落（含灰区，无生态门/缺档止步）** ⇒ 判\"甲案在本阶梯与 3000t 窗内"
            "未达私有信息带\"，**不外推**\"再多降 N 档会怎样\"；上板重议"
            "（衔接 N3 捕食/庇护所案 = 另一信息源路线）（N1 §3.4）。",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 三、生态门止步说明",
        "",
        f"- 沿 D0→D3（密侧→疏侧）遍历，遇崩溃档 break 止步（N1 §四 止步规则）；",
        f"- 崩溃 seed 的 `extinct_at` 已入披露但**不出主判据**（N1 §四 run 级）；",
        f"- 若 D0 判落带 ⇒ fail-loud 判代码漂移（N1 §3.2 灰区红线末句），不进入本表。",
        "",
        "## 四、辅助读数（报不判）",
        "",
        "| 档 | pep_d_med 加权 | ΔR 加权（pos−pep） |",
        "|---|---|---|",
    ])
    for j in out["effective_trace"]:
        dr = None
        if j["pos_pool"] is not None and j["pep_pool"] is not None:
            dr = j["pos_pool"] - j["pep_pool"]
        lines.append(f"| {j['band']} | {_fmt(j.get('pep_d_med_pool'))} | {_fmt(dr)} |")

    lines.extend([
        "",
        "---",
        "",
        "**判读人**：镜 🔍（独立复算，R371）  ",
        "**判据归属**：砚 🪶（N1 预注册 @b715222，跑前锁定 R225）  ",
        "**最终判读权**：天平（本脚本仅出复算读数，不代判）",
    ])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="N1 密度批·镜独立复算（R371）")
    ap.add_argument("--data-dir", default="the-world-data/p1c_density")
    ap.add_argument("--patches", default="1700,850,425,213",
                    help="逗号分隔密度档（默认 N1 §2.1 阶梯）")
    ap.add_argument("--seeds", default="207,208,209,210")
    ap.add_argument("--out", default=None,
                    help="Markdown 输出路径（默认打印到 stdout）")
    a = ap.parse_args()

    patches_list = [int(x) for x in a.patches.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    out = run_batch(a.data_dir, patches_list, seeds)
    md = format_markdown(out)

    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
        sys.stdout.buffer.write(f"OK 复算稿已写入: {a.out}\n".encode("utf-8"))
    else:
        sys.stdout.buffer.write(md.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
