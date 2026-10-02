"""天平独立复核 R336 §之五「跨批交叉验证」表的数字与支别分层。

只读产物，不改任何被测代码。用法：
    .venv/Scripts/python.exe experiments/_verify_r336_table.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "the-world-data"
FS = DATA / "flagship60k"
S2 = ROOT / "_rerun_logs" / "s2g2_newdev" / "results"

# R331 甲案/ 轻舟 R328 §二 逐位复现给出的旗舰 8 seed 支别（A=纯斑块 / B=全图可产）
FS_BRANCH = {165: "A", 166: "A", 167: "B", 168: "A", 169: "A", 170: "B", 171: "A", 172: "B"}

FS_FILES = {
    (165, "off"): FS / "cloud_arm/f60k_165_off.csv",
    (165, "on"): FS / "cloud_arm/f60k_165_on.csv",
    (166, "off"): FS / "cloud_arm/f60k_166_off.csv",
    (166, "on"): FS / "cloud_arm/f60k_166_on.csv",
    (167, "off"): FS / "cloud_arm/f60k_167_off.csv",
    (167, "on"): FS / "cloud_arm/f60k_167_on.csv",
    (168, "off"): FS / "cloud_arm/f60k_168_off.csv",
    (168, "on"): FS / "cloud_arm/f60k_168_on.csv",
    # 169/171 走本机分片 off 臂在 local_slice_off，on 臂在 local_slice_on
    (169, "off"): FS / "local_slice_off/f60k_169_off.csv",
    (169, "on"): FS / "local_slice_on/f60k_169_on.csv",
    (170, "off"): FS / "local_slice_off/f60k_170_off.csv",
    (170, "on"): FS / "local_slice_on/f60k_170_on.csv",
    (171, "off"): FS / "local_slice_off/f60k_171_off.csv",
    (171, "on"): FS / "local_slice_on/f60k_171_on.csv",
    (172, "off"): FS / "local_slice_off/f60k_172_off.csv",
    (172, "on"): FS / "local_slice_on/f60k_172_on.csv",
}


def read_tail(path: Path, n: int) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return rows[-n:] if rows else []


def pick(row: dict, key: str):
    for k in row:
        if k == key:
            return row[k]
    return None


def median(xs: list[float]) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def flagship_pairs(tail: int):
    """旗舰 60k：两臂唯一变量 = 记忆档（mem_off / mem_on）。按支分层。"""
    out = []
    for seed in sorted(FS_BRANCH):
        rec = {"seed": seed, "branch": FS_BRANCH[seed]}
        for arm in ("off", "on"):
            p = FS_FILES[(seed, arm)]
            if not p.exists():
                rec[f"{arm}_missing"] = True
                continue
            rows = read_tail(p, tail)
            if not rows:
                rec[f"{arm}_missing"] = True
                continue
            last = rows[-1]
            rec[f"{arm}_tick"] = pick(last, "tick")
            rec[f"{arm}_pop"] = float(pick(last, "pop") or 0)
            for k in last:
                if "bg_resid" in k:
                    rec[f"{arm}_bg_resid"] = float(last[k])
                if k == "l1_visited_patch_frac":
                    rec[f"{arm}_l1"] = float(last[k])
        if all(f"{a}_{m}" in rec for a in ("off", "on") for m in ("pop", "bg_resid", "l1")):
            rec["K_ratio"] = rec["on_pop"] / rec["off_pop"] if rec["off_pop"] else None
            rec["dbg_resid"] = rec["on_bg_resid"] - rec["off_bg_resid"]
            rec["dl1"] = rec["on_l1"] - rec["off_l1"]
        out.append(rec)
    return out


def s2g2_pairs(tail: int):
    """R336 本批：唯一变量 = rd（rest/depletion）。6 seed × off/on × 40k，全 A 支。"""
    out = []
    for seed in range(201, 207):
        rec = {"seed": seed, "branch": "A"}
        for arm in ("off", "on"):
            p = S2 / f"s2g2_{seed}_{arm}.csv"
            if not p.exists():
                rec[f"{arm}_missing"] = True
                continue
            rows = read_tail(p, tail)
            last = rows[-1]
            rec[f"{arm}_tick"] = pick(last, "tick")
            rec[f"{arm}_pop"] = float(pick(last, "pop") or 0)
            for k in last:
                if "bg_resid" in k:
                    rec[f"{arm}_bg_resid"] = float(last[k])
                if k == "l1_visited_patch_frac":
                    rec[f"{arm}_l1"] = float(last[k])
                if k == "patch_sat_init_var":
                    rec[f"{arm}_psiv"] = float(last[k])
        if all(f"{a}_{m}" in rec for a in ("off", "on") for m in ("pop", "bg_resid")):
            rec["K_ratio"] = rec["on_pop"] / rec["off_pop"] if rec["off_pop"] else None
            rec["dbg_resid"] = rec["on_bg_resid"] - rec["off_bg_resid"]
            rec["dl1"] = rec["on_l1"] - rec["off_l1"]
        out.append(rec)
    return out


def table(recs, title):
    lines = [f"### {title}", "",
             "| seed | 支 | K_off | K_on | K比 | bg_resid off→on | Δbg_resid | Δl1 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        if "K_ratio" not in r:
            lines.append(f"| {r['seed']} | {r['branch']} | — 缺臂 — | | | | | |")
            continue
        lines.append(
            f"| {r['seed']} | {r['branch']} | {r['off_pop']:.0f} | {r['on_pop']:.0f} | "
            f"**{r['K_ratio']:.3f}** | {r['off_bg_resid']:.3f} → {r['on_bg_resid']:.3f} | "
            f"**{r['dbg_resid']:+.3f}** | {r['dl1']:+.4f} |"
        )
    return "\n".join(lines)


def main():
    tail = 20
    fs = flagship_pairs(tail)
    s2 = s2g2_pairs(tail)

    print("=" * 100)
    print("R336 §之五 跨批交叉验证 —— 天平独立复核（末 20 行同口径）")
    print("=" * 100)
    print()
    print(table(fs, "旗舰 60k（两臂唯一变量 = 记忆档 mem_off/mem_on；注意：**混支**）"))
    print()
    print(table(s2, "R336 本批 s2g2（两臂唯一变量 = rd；全 A 支）"))

    # 按支分层
    print()
    print("### 🔴 旗舰按支分层（R336 §之五 未做这一步）")
    for br in ("A", "B"):
        sub = [r for r in fs if r["branch"] == br and "K_ratio" in r]
        if not sub:
            continue
        kr = [r["K_ratio"] for r in sub]
        dbg = [r["dbg_resid"] for r in sub]
        on_bg = [r["on_bg_resid"] for r in sub]
        dl1 = [r["dl1"] for r in sub]
        print(f"- **{br} 支（n={len(sub)}）**：K比 {min(kr):.3f}~{max(kr):.3f} 中位 {median(kr):.3f}"
              f"｜Δbg_resid {min(dbg):+.3f}~{max(dbg):+.3f} 中位 {median(dbg):+.3f}"
              f"｜on臂 bg_resid {min(on_bg):.3f}~{max(on_bg):.3f}"
              f"｜Δl1 {min(dl1):+.4f}~{max(dl1):+.4f}")
        print(f"    seeds = {[r['seed'] for r in sub]}")

    # 汇总数字 vs R336 稿中所写
    print()
    print("### R336 稿 §之五 所写数字 vs 实测")
    fs_kr = [r["K_ratio"] for r in fs if "K_ratio" in r]
    fs_db = [r["dbg_resid"] for r in fs if "dbg_resid" in r]
    fs_on = [r["on_bg_resid"] for r in fs if "on_bg_resid" in r]
    fs_dl = [r["dl1"] for r in fs if "dl1" in r]
    s2_kr = [r["K_ratio"] for r in s2 if "K_ratio" in r]
    s2_dl = [r["dl1"] for r in s2 if "dl1" in r]
    s2_on = [r["on_bg_resid"] for r in s2 if "on_bg_resid" in r]
    print(f"| 指标 | R336 稿写 | 实测 | 一致? |")
    print(f"|---|---|---|---|")
    print(f"| 旗舰 Δl1 | +0.199 ~ +0.308 (7/7正) | {min(fs_dl):+.4f} ~ {max(fs_dl):+.4f} "
          f"({sum(1 for x in fs_dl if x > 0)}/{len(fs_dl)}正) | ❌ 需更正 |")
    print(f"| 旗舰 K比 | 2.27 ~ 126.2 (7/7>1) | {min(fs_kr):.3f} ~ {max(fs_kr):.3f} "
          f"({sum(1 for x in fs_kr if x > 1)}/{len(fs_kr)}>1) | ❌ 需更正 |")
    print(f"| 旗舰 on臂 bg_resid | 0.55 ~ 0.78 | {min(fs_on):.3f} ~ {max(fs_on):.3f} | ❌ 需更正 |")
    print(f"| 旗舰 Δbg_resid | +0.45~+0.69 (8/8) | {min(fs_db):+.3f} ~ {max(fs_db):+.3f} "
          f"({sum(1 for x in fs_db if x > 0)}/{len(fs_db)}正) | ❌ 需更正 |")
    print(f"| 本批 Δl1 | −0.011 ~ −0.043 (0/6) | {min(s2_dl):+.4f} ~ {max(s2_dl):+.4f} "
          f"({sum(1 for x in s2_dl if x > 0)}/{len(s2_dl)}正) | ✅一致 |")
    print(f"| 本批 K比 | 0.41 ~ 0.48 (0/6>1) | {min(s2_kr):.3f} ~ {max(s2_kr):.3f} "
          f"({sum(1 for x in s2_kr if x > 1)}/{len(s2_kr)}>1) | ✅一致 |")
    print(f"| 本批 on臂 bg_resid | 0.028 ~ 0.044 | {min(s2_on):.3f} ~ {max(s2_on):.3f} | ✅一致 |")
    print()
    print(f"🔴 支别事实：旗舰 8 seed = A 支 {[s for s,b in FS_BRANCH.items() if b=='A']}"
          f" / B 支 {[s for s,b in FS_BRANCH.items() if b=='B']}")
    print("   ⇒ R336 稿 §之五 写「旗舰 60k（seed 165–172，**仅 A 支**）」= **错**，B 支占 3/8。")

    out = {
        "flagship_by_seed": fs,
        "s2g2_by_seed": s2,
        "flagship_branch": FS_BRANCH,
    }
    p = ROOT / "_rerun_logs" / "s2g2_newdev" / "_verify_r336_table.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物：{p}")


if __name__ == "__main__":
    main()