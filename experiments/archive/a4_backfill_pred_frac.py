"""D-16 回算（backfill）：给**已跑完**的 R19 验收 summary 补上 R38③ 分层判据。

背景
----
D-16（``codebook_conv`` / ``pred_frac`` 入 CSV）只合入了 ``main``，而 D-23 长跑在
``feat/r19-acceptance`` 的 worktree（``_wt_r19``）上跑，用的是分支旧版脚本 ⇒
那批 CSV 与 summary **没有新判据列**。重跑代价太大（约 1.5 h/3 臂），故回算。

可回算与不可回算（严格区分，不伪造）
------------------------------------
* ✅ ``final_pred_frac``：**可精确回算**。summary 存了 ``deaths_by_cause``，
  捕食占比 = PREDATION / 全部死因。这是 R38③ 分层判据（饱和封顶 vs 捕食主导）。
* ⚠️ ``final_codebook_conv``：
  - ``arbitrary_codebook=False``（cb0）：码本恒等映射 ⇒ 按定义**恒等于 1.0**，
    可直接填 1.0，并标注 ``"identity_constant"``（**无判别力**，内评 N3）。
  - ``arbitrary_codebook=True``（cb1）：需要终局码本矩阵，summary 未存 ⇒
    **不可回算**，标记为 ``None`` + ``"unavailable_need_rerun"``，不得填默认值。
* ❌ CSV 逐 tick 的 ``pred_frac``：CSV 未记录逐点死因 ⇒ **不可回算**。
  分层是按 run 做的（终局值足够），不伪造逐点列。

用法
----
    python experiments/a4_backfill_pred_frac.py                     # 默认回算 _wt_r19
    python experiments/a4_backfill_pred_frac.py --dir _rerun_logs/a4_fix
    python experiments/a4_backfill_pred_frac.py --apply             # 写回 summary（默认只预览）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from observatory.statistics import predation_fraction  # noqa: E402  单一口径实现


# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行

# R38③ 分层阈值：<0.9 = 非捕食主导（多为饱和封顶区制）；>=0.9 = 捕食主导
PRED_DOMINANT = 0.9


def regime(pred_frac: float | None) -> str:
    if pred_frac is None:
        return "unknown"
    return "predation_dominant" if pred_frac >= PRED_DOMINANT else "non_predation"


def backfill_one(path: Path, apply: bool) -> dict:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    r = d.setdefault("result", {})
    sw = d.get("switches", {})

    # 1) pred_frac：从 deaths_by_cause 精确回算
    causes = r.get("deaths_by_cause") or {}
    pred = predation_fraction(causes)
    r["final_pred_frac"] = round(pred, 4)
    r["final_pred_frac_source"] = "backfilled_from_deaths_by_cause"
    r["regime"] = regime(pred)

    # 2) codebook_conv：cb0 恒等=1.0（无判别力）；cb1 不可回算
    cb = bool(sw.get("arbitrary_codebook"))
    if cb:
        r.setdefault("final_codebook_conv", None)
        r["final_codebook_conv_source"] = "unavailable_need_rerun"
    else:
        r["final_codebook_conv"] = 1.0
        r["final_codebook_conv_source"] = "identity_constant(no_discriminative_power)"

    # 3) 声明 CSV 无逐点值（防误读）
    r["csv_has_per_tick_pred_frac"] = False

    if apply:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)

    return {
        "name": path.name.replace(".summary.json", ""),
        "cb": int(cb),
        "seed": sw.get("seed"),
        "final_N": r.get("final_N"),
        "gate": r.get("eco_gate_pass"),
        "pred_frac": r["final_pred_frac"],
        "regime": r["regime"],
        "codebook_conv": r["final_codebook_conv"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="_wt_r19/_rerun_logs/a4_fix",
                    help="含 *.summary.json 的目录（默认 D-23 的 worktree 产出）")
    ap.add_argument("--apply", action="store_true", help="写回文件（默认只预览）")
    args = ap.parse_args()

    base = Path(args.dir)
    if not base.is_absolute():
        base = ROOT / base
    files = sorted(glob.glob(str(base / "*.summary.json")))
    if not files:
        print(f"未找到 summary：{base}")
        return 1

    print(f"目录：{base}    模式：{'写入' if args.apply else '预览（未写盘）'}")
    print("-" * 96)
    hdr = f"{'ARM':<22}{'cb':>3}{'seed':>6}{'final_N':>9}{'gate':>7}{'pred_frac':>11}{'区制':>22}{'conv':>7}"
    print(hdr)
    print("-" * 96)
    rows = [backfill_one(Path(f), args.apply) for f in files]
    for x in sorted(rows, key=lambda z: (z["cb"], z["seed"] or 0)):
        print(f"{x['name']:<22}{x['cb']:>3}{str(x['seed']):>6}{str(x['final_N']):>9}"
              f"{str(x['gate']):>7}{x['pred_frac']:>11.4f}{x['regime']:>22}"
              f"{str(x['codebook_conv']):>7}")
    print("-" * 96)

    hi = [x for x in rows if x["pred_frac"] >= PRED_DOMINANT]
    lo = [x for x in rows if x["pred_frac"] < PRED_DOMINANT]
    print(f"分层结果（阈值 {PRED_DOMINANT}）：捕食主导 {len(hi)} 臂 / 非捕食主导 {len(lo)} 臂")
    if hi and lo:
        print("\n⚠️ R38③：两层【不得合并均值】——区制差异与臂间差异不可分离。")
        for label, grp in (("捕食主导", hi), ("非捕食主导(多为饱和封顶)", lo)):
            ns = [x["final_N"] for x in grp if isinstance(x["final_N"], int)]
            if ns:
                print(f"  {label}: n={len(ns)}  final_N 中位数={sorted(ns)[len(ns)//2]}")
    if not args.apply:
        print("\n（预览模式，未写盘。加 --apply 写入 summary.json）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
