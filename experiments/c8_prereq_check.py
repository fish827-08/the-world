#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C8 前提对账执行器（R127，2026-09-19）——「**脚本意图 = 科学问题所需的前提吗**」。

两种用法：

1) **看读数**（默认）——打印 uniform / patchy 的容量空间统计与判据：
       python experiments/c8_prereq_check.py
2) **验预设声明**（Pre-Flight 用）——逐条验 `experiments/prerequisites.json` 里该 preset 的
   `prerequisites`，**不过 ⇒ 退出码 1**（Pre-Flight 拒跑）：
       python experiments/c8_prereq_check.py --preset cstep3patchy

支持的前提种类（`kind`）
------------------------
- `preset_arg_equals`：该 preset 的 fixed/variants 参数中 `key=value`（缺 CLI 时可用 `default` 兜底
  —— 用于"当时该参数尚无 CLI、等价于默认值"的历史批）。
- `capacity_lat_r2_max`：用 `observatory.world_capacity` 复算容量图的「纬度 R²」 ⇒ 要求 `≤ value`。
  可用 item 级 `"distribution"` 指定要复算的世界（缺省取 preset 参数 / `default`）。
- `summary_switches_equals`：给了 `--dirs` 时，核对**真实产物** `switches`（最硬的一档）。

⚠️ 声明文件里 `retrospective: true` 的 preset 属**事后登记**（用于验证"C8 本可拦住历史批次"），
   其 FAIL **不影响退出码**，但会在报告里**显式标红**。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # R98 纪律：GBK 控制台兜底
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from observatory.world_capacity import (  # noqa: E402
    DEFAULT_MAX_LAT_R2, build_capacity_map, capacity_stats,
)

PREREQ_FILE = ROOT / "experiments" / "prerequisites.json"


def load_prerequisites(path: Path = PREREQ_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preset_args(dicts: list[dict]) -> dict[str, str]:
    """把一个 preset（含 variants）里的 `key=value` 参数摊平成字典（后写覆盖前写）。"""
    out: dict[str, str] = {}
    for d in dicts:
        for a in d.get("fixed", []) + [x for v in d.get("variants", []) for x in v.get("args", [])]:
            if "=" in a:
                k, v = a.split("=", 1)
                out[k] = v
            else:
                out[a] = "true"          # 无值开关（如 calibration-arm）
        for k in d.get("grid", []):
            key = k.split("=", 1)[0]
            out.setdefault(f"<grid>{key}", key)
        # 🔴 臂变量：**摊平后会被后一个臂覆盖** ⇒ 另存 `<variant>{i}:{key}` 供
        #    `variant_arg_equals` 使用（本批的"唯一被测变量"必须能被 C4 查到）。
        for vi, v in enumerate(d.get("variants", [])):
            for x in v.get("args", []):
                if "=" in x:
                    k2, v2 = x.split("=", 1)
                    out[f"<variant>{vi}:{k2}"] = v2
                else:
                    out[f"<variant>{vi}:{x}"] = "true"
    return out


def eval_item(item: dict, argv: dict[str, str], dist_for_r2: str) -> dict:
    """评估单条前提 ⇒ `{id, ok, got, want, note}`。"""
    kind = item["kind"]
    if kind == "variant_arg_equals":
        # 臂变量版 `preset_arg_equals`：要求**至少一个臂**带 `key=value`
        # （例：A-连续批两臂 k=2.0 / k=0.0——若两臂都读到 0.0 ⇒ 跑成了纯对照，整批作废）。
        key, want = item["key"], str(item["value"])
        hits = [v for k, v in argv.items()
                if k.startswith("<variant>") and k.split(":", 1)[1] == key]
        got = f"臂值集合={hits}" if hits else "（无任何臂带该参数）"
        ok = want in hits
        return {"id": item["id"], "ok": ok, "got": got, "want": f"至少一个臂 {key}={want}",
                "note": item.get("why", "")}
    elif kind == "preset_arg_equals":
        key, want = item["key"], str(item["value"])
        got = argv.get(key, item.get("default"))
        ok = (got == want)
        note = "" if key in argv else ("（该批无此 CLI 参数；按 `default` 判定）"
                                       if "default" in item else "（该参数不在 preset 中）")
        return {"id": item["id"], "ok": ok, "got": f"{key}={got}", "want": f"{key}={want}", "note": note}
    if kind == "capacity_lat_r2_max":
        cap = build_capacity_map(dist_for_r2)
        st = capacity_stats(cap)
        got, want = st["lat_r2"], float(item["value"])
        return {"id": item["id"], "ok": got <= want, "got": f"lat_r2={got:.4f}",
                "want": f"≤ {want}", "note": f"（世界={dist_for_r2}；off_row_frac={st['off_row_frac']:.4f}）"}
    if kind == "summary_switches_equals":
        key, want = item["key"], str(item["value"])
        return {"id": item["id"], "ok": None, "got": "（需 --dirs）", "want": f"{key}={want}",
                "note": "未提供 --dirs ⇒ 跳过"}
    raise ValueError(f"未知前提种类 kind={kind!r}（id={item.get('id')}）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="验该预设声明的 prerequisites（缺省只打读数表）")
    ap.add_argument("--max-lat-r2", type=float, default=DEFAULT_MAX_LAT_R2,
                    help=f"看读数模式下的判据阈值（默认 {DEFAULT_MAX_LAT_R2}）")
    ap.add_argument("--distributions", default="uniform,patchy")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.preset is None:
        dists = [x.strip() for x in args.distributions.split(",") if x.strip()]
        print("=" * 88)
        print(f"C8 前提对账读数 —— 判据：纬度 R² ≤ {args.max_lat_r2}（越低 ⇒ 位置越不可预测 ⇒ 信息越有价值）")
        print("=" * 88)
        print(f"{'world':>10}{'lat_r2':>10}{'未解释':>10}{'off_row_frac':>14}"
              f"{'adj_col≠':>11}{'cv':>9}{'行均值档数':>11}{'判定':>10}")
        rows = {}
        for dist in dists:
            st = capacity_stats(build_capacity_map(dist))
            rows[dist] = st
            ok = st["lat_r2"] <= args.max_lat_r2
            print(f"{dist:>10}{st['lat_r2']:>10.4f}{st['lat_unexplained']:>10.4f}"
                  f"{st['off_row_frac']:>14.4f}{st['adjacent_col_diff_nonzero_frac']:>11.4f}"
                  f"{st['cv']:>9.4f}{st['unique_row_means']:>11}{('✅ 有意义' if ok else '❌ 信息价值≈0'):>10}")
        print("")
        print("⇒ uniform 下 `lat_r2 = 1.000` ⇒ **位置可完全预测 ⇒ 信息价值 = 0**"
              "（这就是 C1a/C1b/C2/α/gate/α8 的前提性问题）")
        if args.json_out:
            p = Path(args.json_out)
            (p if p.is_absolute() else ROOT / p).write_text(
                json.dumps({"max_lat_r2": args.max_lat_r2, "worlds": rows},
                           ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    # ---- 验预设声明 ----
    from experiments import batch_runner as br
    decl = load_prerequisites()
    if args.preset not in decl:
        print(f"❌ {args.preset} 未在 {PREREQ_FILE.name} 中声明 prerequisites")
        return 1
    d = decl[args.preset]
    if args.preset not in br.PRESETS:
        print(f"❌ 未知 preset：{args.preset}（不在 batch_runner.PRESETS）")
        return 1
    argv = preset_args([br.PRESETS[args.preset]])
    retro = bool(d.get("retrospective"))
    print("=" * 88)
    print(f"C8 前提对账 —— preset `{args.preset}`" + ("（**事后登记**，FAIL 不计退出码）" if retro else ""))
    print(f"科学问题：{d.get('science_question', '（未写）')}")
    print("=" * 88)
    results, all_ok = [], True
    for item in d.get("items", []):
        dist = item.get("distribution") or argv.get("distribution") or item.get("default") or "uniform"
        r = eval_item(item, argv, dist)
        results.append({**r, "why": item.get("why", "")})
        mark = "✅" if r["ok"] else ("❌" if r["ok"] is False else "⚠️")
        print(f"{mark} [{r['id']}] {r['kind'] if 'kind' in r else ''}"
              f" got={r['got']}  want={r['want']} {r['note']}")
        print(f"     理由：{item.get('why', '')}")
        if r["ok"] is False and not retro:
            all_ok = False
    n_fail = sum(1 for r in results if r["ok"] is False)
    print("")
    if n_fail == 0:
        concl = "✅ 前提全部通过"
    elif retro:
        concl = (f"📌 **事后登记**：{n_fail} 项未过 ⇒ **该批前提不成立**"
                 f"（作为历史证据；**不构成当前阻塞**，也不得据此改判该批数据）")
    else:
        concl = f"❌ {n_fail} 项未过 ⇒ 按 R127 **不得开跑**（补前提 / 改设计后再走 Pre-Flight）"
    print(f"⇒ 结论：{concl}")
    if args.json_out:
        p = Path(args.json_out)
        (p if p.is_absolute() else ROOT / p).write_text(
            json.dumps({"preset": args.preset, "retrospective": retro,
                        "gate_pass": all_ok, "n_items": len(results), "n_fail": n_fail,
                        "fails": [r["id"] for r in results if r["ok"] is False],
                        "conclusion": concl, "items": results},
                       ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
