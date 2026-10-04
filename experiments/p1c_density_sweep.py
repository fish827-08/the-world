#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N2 密度批载体（R373-①，卡 N2-DENSITY-CARRIER）：D0–D3 × seed 207–210 矩阵调度器。

判据/口径唯一权威 = `docs/预注册/预注册-N1-密度梯度批-20261004.md`（🔒 已冻结，
含镜 N4 唯一修订 S4 NaN 条）。本脚本**不裁定、不判读**——只出矩阵/清单、驱动跑批
（起跑权在 fish，R117/R372）、按锁定口径聚合读数（逐对池化主口径 + 三重锁机械检查），
判读稿归砚。

四档：patches ∈ {1700(D0 对照), 850, 425, 213} × seed {207,208,209,210} × 3000t。
D0 默认复用 Tier-1 既有数据（`the-world-data/p1c_erase/`）⇒ 新增 run = 12（§2.4）；
复用前提 = `--smoke-gate` 同机复现门通过（1700×207 与既有 CSV 逐行对拍，
排除 `ms_per_tick_window`，口径同 dm-guard `_dm_guard_mismatches`）。

R277 起跑四要素 ⇒ manifest 逐 run 落盘（§2.3）：
  ① 全 argv（复现命令）② 装置回显（probe summary.readback）
  ③ 代码 commit 指纹（git rev-parse HEAD + dirty）④ 装置字段逐项核对表（§2.2 锁定值）。

子命令（互斥）：
  --plan        只打印矩阵 + 逐 run 命令 + 核对表（可 --manifest-out 落 JSON），**不跑**
  --smoke-gate  跑 1 个复现门 run（1700×207，产物入 --gate-out，不进数据仓正式目录）
  --run-matrix  顺序驱动 12 run（须 fish 起跑令授权；默认 --dry-run 只打印命令）
  --aggregate   读数附表：逐档逐 seed 披露 + 三重锁 S1–S4 机械检查（判读稿素材）
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---- 预注册 §2.2 锁定值（改动=改口径，须修订令，R225）
PATCHES_LADDER = [1700, 850, 425, 213]        # D0→D3（密度降序）
SEEDS = [207, 208, 209, 210]
TICKS, SAMPLE, RING_DEPTH = 3000, 100, 64
LOCKED_DEVICE = {                              # §2.2 公共参数（s2 + 显式 patches）
    "rows": 480, "cols": 960, "pop": 10000, "rgm": 1.195,
    "bg_low_prod_frac": 0.0, "bg_low_cap_mult": 0.0,
}
LOCKED_READBACK = {                            # §2.2 readback 位（跑后必须回显核对）
    "signal_alphabet": "8", "memory_v2": False, "info_structure_enabled": False,
    "rd_enabled": False, "use_sim_core": False, "subpos_enabled": True,
}
BAND_NAMES = {1700: "D0", 850: "D1", 425: "D2", 213: "D3"}
# 探针判别装置指纹的 patches 维（`p1c_erasure_probe._S2 = (480, 960, 1700)`）：
# 仅此档免 `--smoke`；漂移由 test_smoke_flag_matches_probe_guard 贯通测试判红。
S2_PATCHES = 1700
PRE_REG = "docs/预注册/预注册-N1-密度梯度批-20261004.md"
DATA_DIR_DEFAULT = "the-world-data/p1c_density"
ERASE_DIR_DEFAULT = "the-world-data/p1c_erase"
# 三重锁阈值（§3.2 锁定）
S0_THRESH = 0.50          # 主判据 pos_frac_read_moore ≤0.50 ⇒ 私有信息成立带
S3_PEP_MAX = 0.80         # 同向确认：pep ≤0.80（不落 H1 带）
S3_POS_DMIN = 2.0         # 且 pos_d_med ≥ 2.0
S2_MARGIN_MULT = 2.0      # 离界余量 ≥ 2×SD（种子间样本 SD，ddof=1）
MIN_ALIVE = 3             # 档级门：存活 run ≥3/4；S1：存活 run 中 ≥3/4 逐 run ≤0.50


def out_name(patches: int, seed: int, ticks: int = TICKS) -> str:
    return f"p1c_density_d{patches}_s{seed}_t{ticks}"


def build_cmd(python: str, patches: int, seed: int, out_path: Path,
              home_range: bool = True, dm_guard: str | None = None,
              module: str = "experiments.p1c_erasure_probe") -> list[str]:
    """单 run 命令（🔴 显式 --device s2 + 显式 --patches：R5.6 不吃默认 60×120）。"""
    cmd = [python, "-u", "-m", module,
           "--device", "s2", "--patches", str(patches),
           "--rows", str(LOCKED_DEVICE["rows"]), "--cols", str(LOCKED_DEVICE["cols"]),
           "--pop", str(LOCKED_DEVICE["pop"]), "--rgm", str(LOCKED_DEVICE["rgm"]),
           "--bg-low-prod-frac", str(LOCKED_DEVICE["bg_low_prod_frac"]),
           "--bg-low-cap-mult", str(LOCKED_DEVICE["bg_low_cap_mult"]),
           "--seed", str(seed), "--ticks", str(TICKS), "--sample", str(SAMPLE),
           "--ring-depth", str(RING_DEPTH)]
    if home_range:
        cmd += ["--home-range"]
    if dm_guard:
        cmd += ["--dm-guard", str(dm_guard)]
    # D-DENSITY-SMOKE：探针 R278 守卫对 ≠(480,960,1700) 的档要求显式 `--smoke`，
    # 而密度批的 patches 正是被测变量（非判别档 = 常态）⇒ 非 D0 档必须自带该旗标。
    if patches != S2_PATCHES:
        cmd += ["--smoke"]
    cmd += ["--out", str(out_path)]
    return cmd


def git_fingerprint(cwd: Path) -> dict:
    def _run(args):
        try:
            r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                               text=True, encoding="utf-8", timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    rev = _run(["rev-parse", "HEAD"])
    if rev is None:
        raise SystemExit("🔴 manifest 四要素-③：git rev-parse HEAD 失败 ⇒ 拒落 manifest"
                         "（R277：代码指纹缺位不得起跑）。")
    dirty = bool(_run(["status", "--porcelain"]))
    return {"commit": rev, "dirty": dirty}


def field_check(patches: int) -> dict:
    """R277 四要素-④：装置字段逐项核对表（对照 §2.2 锁定值 + 本次请求值）。"""
    want = dict(LOCKED_DEVICE)
    want["patches"] = patches
    want["ticks"] = TICKS
    want["sample"] = SAMPLE
    want["ring_depth"] = RING_DEPTH
    want["seeds"] = SEEDS
    return {"source": f"{PRE_REG} §2.2", "locked": want,
            "readback_expected": dict(LOCKED_READBACK)}


def plan_rows() -> list[dict]:
    rows = []
    for p in PATCHES_LADDER:
        for s in SEEDS:
            d0_reuse = (p == 1700)
            rows.append({
                "band": BAND_NAMES[p], "patches": p, "seed": s, "ticks": TICKS,
                "status": "reuse-Tier-1(§2.4，复现门通过后)" if d0_reuse else "to-run",
                "new_run": not d0_reuse,
            })
    return rows


def cmd_plan(args) -> int:
    rows = plan_rows()
    to_run = [r for r in rows if r["new_run"]]
    print(f"# N2 密度批矩阵（{PRE_REG} §二/§2.4）")
    print(f"# 档位: D0=1700(复用/复现门) D1=850 D2=425 D3=213 | seeds={SEEDS} | "
          f"{TICKS}t sample={SAMPLE} ring={RING_DEPTH} --device s2 显式 | --home-range 开")
    print(f"# 新增 run = {len(to_run)}（D0 复用 ⇒ 12；复现门漂移 ⇒ D0 +4，上板报 PI）")
    for r in rows:
        cmd = build_cmd(args.python, r["patches"], r["seed"],
                        Path(args.data_dir) / (out_name(r["patches"], r["seed"]) + ".csv"))
        tag = "复用" if not r["new_run"] else "待跑"
        print(f"\n[{r['band']}] d{r['patches']} s{r['seed']} ({tag})")
        print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
    if args.manifest_out:
        manifest = {
            "card": "N2-DENSITY-CARRIER", "pre_reg": PRE_REG,
            "git": git_fingerprint(_ROOT), "field_check": {BAND_NAMES[p]: field_check(p)
                                                            for p in PATCHES_LADDER},
            "matrix": rows, "commands": {
                f"{BAND_NAMES[r['patches']]}_d{r['patches']}_s{r['seed']}": build_cmd(
                    args.python, r["patches"], r["seed"],
                    Path(args.data_dir) / (out_name(r["patches"], r["seed"]) + ".csv"))
                for r in rows if r["new_run"]},
            "rng_note": "见交付说明（§九-2：斑块中心流独立 default_rng(patch_seed)，"
                        "不消费引擎主 rng；跨档序列非前缀嵌套）",
        }
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest_out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        print(f"\n矩阵 manifest（R277 四要素-①③④）：{args.manifest_out}")
    print("\n# 本命令不跑批（R117）；起跑由 fish 授权后 PI 触发 --run-matrix。")
    return 0


def cmd_smoke_gate(args) -> int:
    """§2.4 复现门：1700×207 与既有擦除档逐行对拍（排除 ms 列 = dm-guard 口径）。"""
    stored = Path(args.erase_dir) / "p1c_erase_s207_t3000.csv"
    if not stored.exists():
        raise SystemExit(f"🔴 复现门：既有 Tier-1 档不存在 {stored} ⇒ 拒跑。")
    out = Path(args.gate_out)
    if out.exists() and not args.force:
        raise SystemExit(f"🔴 复现门产物已存在 {out} ⇒ 不覆盖（F-R10）；确要重跑用 --force。")
    cmd = build_cmd(args.python, 1700, 207, out, home_range=True, dm_guard=str(stored))
    print(f"# [smoke-gate] 复现门 run（D0 复用裁决用）：\n  {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=_ROOT)
    if rc == 0:
        print("✅ 复现门通过 ⇒ D0 复用 Tier-1 四 seed 全量，新增 run=12（§2.4）。")
        return 0
    print(f"🔴 复现门失败（rc={rc}）⇒ 代码漂移：D0 重跑 4 run（预算 +4，上板报 PI 批条），"
          "且 Tier-1 判读稿同步标记『待重测』（§2.4）。本脚本未自行裁定，报告已生成即止。",
          file=sys.stderr)
    return rc


def cmd_run_matrix(args) -> int:
    rows = [r for r in plan_rows() if r["new_run"]]
    if args.only_patches:
        rows = [r for r in rows if r["patches"] in set(args.only_patches)]
    if args.only_seeds:
        rows = [r for r in rows if r["seed"] in set(args.only_seeds)]
    data_dir = Path(args.data_dir)
    fp = git_fingerprint(_ROOT)
    print(f"# [run-matrix] {len(rows)} run；代码指纹 {fp['commit'][:8]}"
          f"{' DIRTY⚠️' if fp['dirty'] else ''}；dry_run={args.dry_run}")
    if fp["dirty"] and not args.allow_dirty:
        raise SystemExit("🔴 工作树 dirty ⇒ manifest 代码指纹不唯一 ⇒ 停（--allow-dirty 明示豁免）。")
    failures = []
    for r in rows:
        out = data_dir / (out_name(r["patches"], r["seed"]) + ".csv")
        cmd = build_cmd(args.python, r["patches"], r["seed"], out)
        if args.dry_run:
            print(f"[{r['band']}] dry: " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
            continue
        if out.exists() and not args.force:
            raise SystemExit(f"🔴 产物已存在 {out} ⇒ 拒覆盖（F-R10）；重跑用 --force。")
        print(f"[{r['band']}] 起跑 d{r['patches']} s{r['seed']} …")
        rc = subprocess.call(cmd, cwd=_ROOT)
        summary_p = out.with_name(out.stem + ".summary.json")
        if rc != 0 or not summary_p.exists():
            failures.append(f"{'/'.join([str(r['patches']), str(r['seed'])])} rc={rc}")
            print(f"🔴 run 失败 rc={rc} ⇒ 继续下一 run（fail-loud 汇总在末尾）。",
                  file=sys.stderr)
            continue
        summary = json.loads(summary_p.read_text(encoding="utf-8"))
        mism = verify_readback(summary, expect_patches=r["patches"])
        if mism:
            failures.append(f"d{r['patches']}_s{r['seed']} readback: " + "；".join(mism))
            print("🔴 readback 核对失配（R277 四要素-②）：" + "；".join(mism),
                  file=sys.stderr)
        manifest = {  # R277 四要素逐 run 落盘
            "band": r["band"], "patches": r["patches"], "seed": r["seed"],
            "argv": summary["argv"], "git": fp,
            "readback": summary["readback"], "field_check": field_check(r["patches"]),
            "extinct_at": summary["extinct_at"],
            "out_csv": str(out), "out_summary": str(summary_p),
            "pre_reg": PRE_REG, "authorized_by": args.authorized_by,
        }
        mp = data_dir / f"manifest_d{r['patches']}_s{r['seed']}.json"
        data_dir.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  manifest：{mp}")
    if args.dry_run:
        print("# dry-run：未起跑。真实起跑 = fish 授权（R372）后 PI 执行。")
        return 0
    if failures:
        raise SystemExit(f"🔴 {len(failures)} run 异常：\n  " + "\n  ".join(failures))
    print(f"✅ {len(rows)} run 完成且 readback 逐字段核对通过。")
    return 0


def verify_readback(summary: dict, expect_patches: int | None = None) -> list[str]:
    """R277 四要素-②：跑后 summary.readback 与 §2.2 锁定值逐字段核对。"""
    rb = summary.get("readback", {})
    mism = []
    for k, want in LOCKED_READBACK.items():
        got = rb.get(k)
        if got != want:
            mism.append(f"readback.{k}={got!r}≠锁定{want!r}")
    for k, want in [("rows", LOCKED_DEVICE["rows"]), ("cols", LOCKED_DEVICE["cols"]),
                    ("initial_pop", LOCKED_DEVICE["pop"]),
                    ("rgm", LOCKED_DEVICE["rgm"]), ("ring_depth", RING_DEPTH)]:
        if rb.get(k) != want:
            mism.append(f"readback.{k}={rb.get(k)!r}≠锁定{want!r}")
    p = rb.get("patches")
    if p not in PATCHES_LADDER:
        mism.append(f"readback.patches={p!r} 不在四档阶梯")
    if expect_patches is not None and p != expect_patches:
        mism.append(f"readback.patches={p!r}≠请求档 {expect_patches}")
    if summary.get("ticks") != TICKS or summary.get("sample") != SAMPLE:
        mism.append(f"ticks/sample={summary.get('ticks')}/{summary.get('sample')} "
                    f"≠{TICKS}/{SAMPLE}")
    return mism


# ---- 聚合（读数附表，判读稿素材；判读权在砚） ----

def _load_run_summaries(data_dir: Path, erase_dir: Path) -> dict[int, list[dict]]:
    """按档分组收集 run 读数；D0 优先本目录重跑（复现门漂移后），否则回退 Tier-1 复用。"""
    by_band: dict[int, list[dict]] = {p: [] for p in PATCHES_LADDER}
    for p in PATCHES_LADDER:
        for s in SEEDS:
            sp = data_dir / (out_name(p, s) + ".summary.json")
            if sp.exists():
                d = json.loads(sp.read_text(encoding="utf-8"))
                by_band[p].append(_extract_run(p, s, d, src=str(sp), reused=False))
    if not by_band[1700]:
        for s in SEEDS:
            sp = erase_dir / f"p1c_erase_s{s}_t3000.summary.json"
            if sp.exists():
                d = json.loads(sp.read_text(encoding="utf-8"))
                by_band[1700].append(_extract_run(
                    1700, s, d, src=f"{sp} (复用Tier-1，§2.4)", reused=True))
    return by_band


def _extract_run(p, s, d: dict, src: str, reused: bool) -> dict:
    pooled = d.get("pooled", {})
    pep = pooled.get("pep", {})
    per_sample = d.get("per_sample", [])
    ms = [r.get("ms_per_tick_window") for r in per_sample
          if isinstance(r.get("ms_per_tick_window"), (int, float))]
    return {
        "band": BAND_NAMES[p], "patches": p, "seed": s, "src": src, "reused": reused,
        "extinct_at": d.get("extinct_at"),
        "pos_frac": pooled.get("pos_frac_read_moore"),
        "pos_n_pairs": pooled.get("pos_n_dist_pairs"),
        "pos_d_med": pooled.get("pos_d_med"),
        "pep_frac": pep.get("frac_read_moore_pos"),
        "pep_n": pep.get("n_pos"),
        "pep_d_med": pep.get("d_med_pos"),
        "delta_r": (pooled.get("pos_frac_read_moore") - pep.get("frac_read_moore_pos"))
        if (pooled.get("pos_frac_read_moore") is not None
            and pep.get("frac_read_moore_pos") is not None) else None,
        "hr_run": (pooled.get("hr_run_median") or {}).get("hr_max_dist_median"),
        "ms_tick_med": (float(sorted(ms)[len(ms) // 2]) if ms else None),
        "pop_end": (per_sample[-1].get("n_agents") if per_sample else None),
        "window_mean_pos_frac": _window_mean(per_sample, "pos_frac_read_moore"),
    }


def _window_mean(per_sample: list[dict], col: str):
    vals = [r[col] for r in per_sample if isinstance(r.get(col), (int, float))]
    return float(sum(vals) / len(vals)) if vals else None


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def _sd(vals: list[float]) -> float | None:
    """样本 SD（ddof=1，§3.2-S2）；n<2 ⇒ None。"""
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def judge_band(p: int, runs: list[dict]) -> dict:
    """§3.2 三重锁 + §四 生态门 + S4 NaN fail-loud：机械检查（不定≠错误）。"""
    out = {"band": BAND_NAMES[p], "patches": p,
           "runs": [{k: r[k] for k in ("seed", "pos_frac", "pep_frac", "pos_d_med",
                                       "delta_r", "hr_run", "ms_tick_med", "pop_end",
                                       "extinct_at", "window_mean_pos_frac", "src")}
                    for r in runs]}
    alive = [r for r in runs if r["extinct_at"] is None]
    out["n_alive"] = len(alive)
    out["alive_seeds"] = [r["seed"] for r in alive]
    out["crashed_seeds"] = {str(r["seed"]): r["extinct_at"] for r in runs
                            if r["extinct_at"] is not None}
    # 🔴 S4（镜 N4 唯一修订条）：任一存活 run 主判据 NaN ⇒ 该档判不定 + fail-loud
    #    （无条件检查：先于门/数据量，不得静默跳过）
    bad = [r["seed"] for r in alive if not _is_num(r["pos_frac"])]
    if bad:
        out["verdict"] = f"不定（S4 NaN fail-loud：seed {bad} 池化值非数，不得静默跳过）"
        out["s4_nan"] = bad
        out["gate"] = False
        return out
    if len(runs) < len(SEEDS):
        out["verdict"] = f"数据不足（{len(runs)}/4 run 到位）"
        out["gate"] = False
        return out
    if len(alive) < MIN_ALIVE:
        out["verdict"] = "崩溃（档级≤2/4 存活，§四）"
        out["gate"] = False
        return out
    out["gate"] = True
    vals = [r["pos_frac"] for r in alive]
    den = sum(r["pos_n_pairs"] or 0 for r in alive)
    band_val = (sum(r["pos_frac"] * (r["pos_n_pairs"] or 0) for r in alive) / den
                if den else None)
    out["band_pooled_pos_frac"] = band_val          # 主口径：逐对池化跨 seed（§3.1）
    if band_val is None:
        out["verdict"] = "不定（池化分母=0）"
        return out
    sd = _sd(vals)
    n_le = sum(1 for v in vals if v <= S0_THRESH)
    out["s1_seeds_le"] = f"{n_le}/{len(alive)}（需≥{MIN_ALIVE}）"
    s1 = n_le >= MIN_ALIVE
    margin = S0_THRESH - band_val
    need = S2_MARGIN_MULT * sd if sd is not None else None
    out["s2_margin"] = {"band_val": band_val, "margin": margin, "sd": sd, "need": need,
                       "pass": (sd is not None and margin >= need)}
    s2 = out["s2_margin"]["pass"]
    pep_den = sum(r["pep_n"] or 0 for r in alive)
    pep_band = (sum(r["pep_frac"] * (r["pep_n"] or 0) for r in alive) / pep_den
                if pep_den and all(_is_num(r["pep_frac"]) for r in alive) else None)
    dmed_alive = [r["pos_d_med"] for r in alive if _is_num(r["pos_d_med"])]
    pos_d_med_band = (sum(r["pos_d_med"] * (r["pos_n_pairs"] or 0) for r in alive
                          if _is_num(r["pos_d_med"])) / den if den and dmed_alive else None)
    s3 = (pep_band is not None and pep_band <= S3_PEP_MAX
          and pos_d_med_band is not None and pos_d_med_band >= S3_POS_DMIN)
    out["s3"] = {"pep_band": pep_band, "pos_d_med_band_weighted_approx": pos_d_med_band,
                 "per_run_pos_d_med": dmed_alive, "pass": s3,
                 "note": "pos_d_med 跨 seed 严格逐对池化需原始对数组（未落盘）⇒ "
                         "此处为 n 加权近似 + 逐 run 披露，口径终审在砚"}
    out["s1_pass"], out["s2_pass"], out["s3_pass"] = s1, s2, s3
    out["band_val_le_050"] = band_val <= S0_THRESH
    out["gray_zone"] = (S0_THRESH < band_val < 0.80)
    if s1 and s2 and s3:
        out["verdict"] = "阳性（三重锁全过，§3.2）"
    else:
        out["verdict"] = "不定" + ("（灰区，禁挑边 R7.2）" if out["gray_zone"] else "")
    return out


def cmd_aggregate(args) -> int:
    data_dir = Path(args.data_dir)
    by_band = _load_run_summaries(data_dir, Path(args.erase_dir))
    bands = []
    for p in PATCHES_LADDER:
        if not by_band[p]:
            bands.append({"band": BAND_NAMES[p], "patches": p,
                          "verdict": "缺数据（未跑/未到位）", "gate": False, "runs": []})
            continue
        bands.append(judge_band(p, by_band[p]))
    # 止步规则（§四）：崩溃档及其更疏侧档不入"首个 <0.50"搜索
    search_order = []
    for b in bands:
        if b.get("gate") is False and "崩溃" in str(b.get("verdict")):
            break
        search_order.append(b)
    positive = next((b for b in search_order
                     if str(b.get("verdict", "")).startswith("阳性")), None)
    first_le = next((b for b in search_order
                     if b.get("band_val_le_050")), None)
    s4_hits = [b for b in bands if b.get("s4_nan")]
    md = _format_md(bands, positive, first_le)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"bands": bands, "candidate_band": positive and positive["band"],
             "first_le_050_band": first_le and first_le["band"],
             "pre_reg": PRE_REG}, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.out_md:
        sys.stdout.buffer.write(md.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
    print(f"# 机械结论：候选档={positive and positive['band']}"
          f" 首个≤0.50 档={first_le and first_le['band']}（判读稿归砚，本工具不裁定）")
    if s4_hits:
        for b in s4_hits:
            print(f"🔴 S4 fail-loud：{b['band']} 档 seed {b['s4_nan']} 池化值 NaN ⇒ "
                  "判不定，须上板（不得静默跳过）。", file=sys.stderr)
        return 3
    if not any(b.get("runs") for b in bands):
        print("🔴 聚合 fail-loud：四档均无到位 run（数据未回/路径错）⇒ 本次聚合无效，"
              "不得视作『全档不定』出表（rc=4）。", file=sys.stderr)
        return 4
    return 0


def _format_md(bands, positive, first_le) -> str:
    L = ["# 密度批读数附表（N2 载体聚合｜判读稿素材，非判读稿）", "",
         f"口径唯一来源：`{PRE_REG}`（🔒 冻结）；主口径=**逐对池化**（§3.1），"
         "30 窗等权仅披露列。判定与判读权在砚，本表只列机械检查结果。", "",
         "| 档 | patches | 存活 | 逐对池化 pos | 30窗等权(披露) | 逐run≤0.50 | S2 余量/2×SD "
         "| pep档值 | pos_d_med(近似) | ΔR | hr_max_dist_median | ms/t | 机械结论 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in bands:
        runs = b.get("runs", [])
        alive = [r for r in runs if r["extinct_at"] is None]
        def _m(key):
            vals = [r[key] for r in alive if _is_num(r.get(key))]
            return (sum(vals) / len(vals)) if vals else None
        wm = _m("window_mean_pos_frac")
        dr = _m("delta_r")
        hr = _m("hr_run")
        ms = _m("ms_tick_med")
        L.append("| {} | {} | {}/4 | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            b["band"], b["patches"], b.get("n_alive", len(alive)),
            _f(b.get("band_pooled_pos_frac")), _f(wm),
            b.get("s1_seeds_le", "—"),
            _f((b.get("s2_margin") or {}).get("margin")) + " vs " +
            _f((b.get("s2_margin") or {}).get("need")),
            _f((b.get("s3") or {}).get("pep_band")),
            _f((b.get("s3") or {}).get("pos_d_med_band_weighted_approx")),
            _f(dr), _f(hr, ".3f"), _f(ms, ".2f"),
            b.get("verdict", "—")))
    L += ["", f"**首个 ≤0.50 档**（搜索含止步规则，§四）：{first_le and first_le['band']}；"
              f"**三重锁阳性档**：{positive and positive['band']}。", "",
          "逐 seed 明细：", ""]
    for b in bands:
        L.append(f"- {b['band']} d{b['patches']}：存活={b.get('alive_seeds', [])} "
                 f"崩溃={b.get('crashed_seeds', {}) or '无'}；"
                 + "；".join(f"s{r['seed']} pos={_f(r['pos_frac'])} pep={_f(r['pep_frac'])} "
                             f"d_med={_f(r['pos_d_med'])} pop末={r['pop_end']}"
                             for r in b.get("runs", [])))
    L.append("")
    L.append("— 载体自动聚合（澜舟 N2），判读稿请砚引用本表。")
    return "\n".join(L)


def _f(v, spec: str = ".4f") -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "NaN"
    if isinstance(v, str):
        return v
    return format(float(v), spec)


def main() -> int:
    ap = argparse.ArgumentParser(description="N2 密度批载体（矩阵/复现门/驱动/聚合）")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="只出矩阵与 manifest，不跑")
    mode.add_argument("--smoke-gate", action="store_true",
                      help="§2.4 同机复现门：1700×207 与既有擦除档逐行对拍（1 run）")
    mode.add_argument("--run-matrix", action="store_true",
                      help="驱动 12 run（须 fish 起跑令；默认 dry-run）")
    mode.add_argument("--aggregate", action="store_true", help="读数附表（三重锁机械检查）")
    ap.add_argument("--python", default=sys.executable,
                    help="子进程解释器（默认=当前解释器；云机用 venv 内 python 起跑本脚本即可）")
    ap.add_argument("--data-dir", default=DATA_DIR_DEFAULT,
                    help="密度批数据目录（相对当前工作目录=仓库根，云机同理）")
    ap.add_argument("--erase-dir", default=ERASE_DIR_DEFAULT,
                    help="Tier-1 复用档目录（D0，§2.4）")
    ap.add_argument("--gate-out", default=str(Path("_trash_local") / "p1c_density"
                                              / "smoke_gate" / "p1c_erase_s207_t3000.csv"))
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--only-patches", nargs="*", type=int, default=None)
    ap.add_argument("--only-seeds", nargs="*", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", default=None)
    ap.add_argument("--force", action="store_true", help="覆盖既有产物（默认拒，F-R10）")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--authorized-by", default=None,
                    help="起跑授权记录（R372 fish 授权凭据/帖号，写入 manifest）")
    a = ap.parse_args()
    for k in ("data_dir", "erase_dir", "gate_out", "manifest_out", "out_md", "json_out"):
        v = getattr(a, k)
        if v:
            p = Path(v)
            setattr(a, k, str(p if p.is_absolute() else _ROOT / p))  # 相对=脚本所在树根
    if a.run_matrix and a.dry_run is None:
        if not a.authorized_by:
            raise SystemExit("🔴 真实起跑需 --authorized-by <R372 授权凭据>（起跑令在 fish"
                             "，R117）；先看命令用 --dry-run。")
        a.dry_run = False
    elif a.dry_run is None:
        a.dry_run = True
    if a.plan:
        return cmd_plan(a)
    if a.smoke_gate:
        return cmd_smoke_gate(a)
    if a.run_matrix:
        return cmd_run_matrix(a)
    return cmd_aggregate(a)


if __name__ == "__main__":
    raise SystemExit(main())
