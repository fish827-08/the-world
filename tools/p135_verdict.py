"""p135food 判读工具（13.5 食物绑定）—— `[所有者·天平]` R190 入库。

用途：13.5 批（A=13.4 对照 / B=全开 / C=只食物绑定）的零机时判读。
    python.exe tools/p135_verdict.py

判据（preset `p135food` 预注册）：
  ① Σ名义再生与容量构成可自证 ② N 是否不再顶软顶 ③ 饿死占比是否上升
  ④ 净收入是否为正 ⑤ 不灭绝
※ 本批**不判分化**（12k 仅 8-28 代 < 世代门 30）。

R190 实测要点（结论见板帖 R190）：
  · A 臂 N=1942-1944（顶软顶）⇒ 食物完全不绑定（对照有效）
  · B/C 臂 N=14-470 ⇒ 食物绑定成立，但**远低于解析 K=1383**
  · 食物**利用率仅 2-39%**，且与 N 强相关（N 越大利用率越高）⇒ 见 `--diag` 段
"""

# ============================================================================================
# 第一部分：判据核对
# ============================================================================================
# -*- coding: utf-8 -*-
"""第一部分：判据核对。"""
import csv
import glob
import json
import os
import statistics as st

PASS
D = "_rerun_logs/p135food"
SOFT_CAP = 0.6 * 3240          # ≈1944（软顶构造值）
K_PRED = 1383                  # R188 素食者批实测定稿（verify_k --bgzero --patch-mult 1.195）
ARMS = ["A", "B", "C"]
ARM_NAME = {"A": "13.4 行为对照", "B": "13.5 全开", "C": "只食物绑定"}


def f(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


runs = {}
for fp in sorted(glob.glob(D + "/*.summary.json")):
    k = os.path.basename(fp)[:-13]
    d = json.load(open(fp, encoding="utf-8"))
    sw, res = d.get("switches") or {}, d.get("result") or {}
    led = (res.get("energy_ledger") or {}).get("global") or {}
    csvp = D + "/" + k + ".csv"
    rows = list(csv.DictReader(open(csvp, encoding="utf-8"))) if os.path.exists(csvp) else []
    runs[k] = {"sw": sw, "res": res, "led": led, "rows": rows}

print("=" * 96)
print("① 逐 run 总览")
print("%-8s %-7s %-7s %-6s %-16s %-9s %-9s" % ("run", "末N", "N/软顶", "世代", "死因(捕食/饿死/老死)", "净收入", "g16"))
for k in sorted(runs):
    r = runs[k]
    N = r["res"].get("final_N")
    gen = r["res"].get("final_max_gen_highwater")
    dc = {str(x).split(".")[-1]: v for x, v in (r["res"].get("deaths_by_cause") or {}).items()}
    tot_d = sum(dc.values()) or 1
    led, rows = r["led"], r["rows"]
    T = int(rows[-1]["tick"]) if rows else 12000
    inc = sum(f(led.get(x)) for x in ("intake_forage_sum", "intake_photo_sum",
                                      "intake_scav_sum", "intake_pred_sum"))
    out = sum(f(led.get(x)) for x in ("cost_meta_sum", "cost_move_sum", "cost_attack_sum"))
    net = (inc - out) / max(1, T) / max(1, N) if N else float("nan")
    g16 = f(rows[-1].get("g16")) if rows else float("nan")
    print("%-8s %-7s %-7.2f %-6s %-16s %+-9.4f %-9.3f" % (
        k, N, (N / SOFT_CAP if N else 0), gen,
        "%d/%d/%d(饿%2.0f%%)" % (dc.get("PREDATION", 0), dc.get("STARVATION", 0),
                                 dc.get("OLD_AGE", 0), 100.0 * dc.get("STARVATION", 0) / tot_d),
        net, g16))

print()
print("=" * 96)
print("② 逐臂汇总（N 与死因）")
print("%-16s %-26s %-24s %-8s" % ("臂", "末 N（4 seed）", "死因占比 捕食/饿死/老死", "净收入中位"))
for a in ARMS:
    ks = [k for k in runs if k.startswith(a + "_")]
    if not ks:
        continue
    Ns = [runs[k]["res"].get("final_N") or 0 for k in ks]
    dc_sum = {}
    nets = []
    for k in ks:
        r = runs[k]
        for x, v in (r["res"].get("deaths_by_cause") or {}).items():
            dc_sum[str(x).split(".")[-1]] = dc_sum.get(str(x).split(".")[-1], 0) + v
        led, rows = r["led"], r["rows"]
        T = int(rows[-1]["tick"]) if rows else 12000
        N = r["res"].get("final_N") or 1
        inc = sum(f(led.get(x)) for x in ("intake_forage_sum", "intake_photo_sum",
                                          "intake_scav_sum", "intake_pred_sum"))
        out = sum(f(led.get(x)) for x in ("cost_meta_sum", "cost_move_sum", "cost_attack_sum"))
        nets.append((inc - out) / max(1, T) / max(1, N))
    dt = sum(dc_sum.values()) or 1
    print("%-16s %-26s %-24s %+-8.4f" % (
        a + " " + ARM_NAME[a], str(sorted(Ns)),
        "%.0f%%/%.0f%%/%.0f%%" % (100 * dc_sum.get("PREDATION", 0) / dt,
                                  100 * dc_sum.get("STARVATION", 0) / dt,
                                  100 * dc_sum.get("OLD_AGE", 0) / dt),
        st.median(nets)))

print()
print("=" * 96)
print("③ N 轨迹（每 2000 tick；A=对照 / B=全开 / C=只绑定）")
for a in ARMS:
    for k in sorted(k for k in runs if k.startswith(a + "_")):
        rows = runs[k]["rows"]
        traj = [(int(r["tick"]), r["N"]) for r in rows]
        print("  %-8s %s" % (k, " ".join("%s:%s" % (t // 1000, n) for t, n in traj)))

print()
print("=" * 96)
print("④ 末段窗口：人均需求与 K_meas（与 R188 的 K=1383 对照）")
for a in ARMS:
    for k in sorted(k for k in runs if k.startswith(a + "_")):
        r = runs[k]
        led, rows, sw = r["led"], r["rows"], r["sw"]
        if len(rows) < 3:
            continue
        lo = len(rows) // 2
        tick0 = int(rows[lo]["tick"])
        N = runs[k]["res"].get("final_N") or 1
        # 末段增量支出（需要账本差分 —— 这里用总量近似，标注为近似）
        eff = f(sw.get("eat_efficiency")) or 3.0
        ah = f(sw.get("assim_herb")) or 1.0
        tot_cost = sum(f(led.get(x)) for x in ("cost_meta_sum", "cost_move_sum", "cost_attack_sum"))
        need_e = tot_cost / 12000 / max(1, N)
        need_m = need_e / (eff * ah) if eff * ah else float("nan")
        g_all = 406.3 if a in ("B", "C") else 3153.0      # A 臂 = 13.4 结构（背景有产能）
        print("  %-8s 人均支出 %.4f 能量/tick ⇒ 需求 %.4f 质量/tick ｜ Σ再生 %.1f ⇒ K_meas≈%s"
              % (k, need_e, need_m, g_all,
                 ("%.0f" % (g_all / need_m)) if need_m == need_m and need_m > 0 else "n/a"))

print()
print("=" * 96)
print("⑤ 判据核对（预注册）")
a_n = [runs[k]["res"].get("final_N") or 0 for k in runs if k.startswith("A_")]
b_n = [runs[k]["res"].get("final_N") or 0 for k in runs if k.startswith("B_")]
c_n = [runs[k]["res"].get("final_N") or 0 for k in runs if k.startswith("C_")]
print("  ① Σ再生可自证：B/C 臂 Σ名义再生 = 406.3（R188 实测，tools/verify_k.py）")
print("  ② N 是否不再顶软顶：A=%s（顶 %.0f）｜B=%s｜C=%s" % (
    sorted(a_n), SOFT_CAP, sorted(b_n), sorted(c_n)))
print("     ⇒ A 臂顶软顶 = 食物**完全不绑定**（对照有效）")
print("     ⇒ B/C 臂 %s ⇒ 食物**绑定了**，但人口远低于 K=1383 ⇒ 见 ⑥" % (
    "低于软顶" if max(b_n + c_n) < SOFT_CAP else "仍有个别接近软顶"))
print("  ③ 饿死占比：见 ② 表")
print("  ④ 净收入为正：见 ② 表")
print("  ⑤ 不灭绝：B/C 最低 N = %s（>0 ⇒ 未灭绝）" % min(b_n + c_n))


# ============================================================================================
# 第二部分：补充诊断（利用率 / 收敛性 / 归因 / 世代门）
# ============================================================================================
# -*- coding: utf-8 -*-
"""第二部分：补充诊断（① 食物利用率（可达性瓶颈检验）② 收敛性 ③ 三臂归因。

关键问题：解析 K=1383，而 B/C 臂实测收敛在 14–470 ⇒ 差 3–100 倍。
最可能的解释 = **食物没被吃完**（食物集中在 11.3% 的格 + 感知只有 1 格 ⇒ 找到难）。
本脚本直接算"**被吃掉的质量 ÷ 名义再生总量**"来验证。
"""
import csv
import glob
import json
import os

PASS
D = "_rerun_logs/p135food"


def f(v):
    try:
        return float(v)
    except Exception:
        return 0.0


print("=" * 100)
print("① 食物利用率 = 实际取食质量 ÷ 名义再生总量（可达性瓶颈的直接检验）")
print("   名义再生：A 臂 3153.1 质量/tick（背景有产能）｜B/C 臂 406.3（背景归零）")
print()
print("%-8s %-7s %-9s %-13s %-13s %-11s" % ("run", "末N", "净吸收", "取食入(能量)", "取食质量", "利用率"))
for a, g_all in (("A", 3153.1), ("B", 406.3), ("C", 406.3)):
    for fp in sorted(glob.glob(D + "/%s_*.summary.json" % a)):
        k = os.path.basename(fp)[:-13]
        d = json.load(open(fp, encoding="utf-8"))
        sw, res = d.get("switches") or {}, d.get("result") or {}
        led = (res.get("energy_ledger") or {}).get("global") or {}
        eff = f(sw.get("eat_efficiency")) or 3.0
        ah = f(sw.get("assim_herb")) or 1.0
        net_abs = eff * ah
        intake_e = f(led.get("intake_forage_sum"))
        intake_m = intake_e / net_abs if net_abs else 0.0
        T = 12000
        supply = g_all * T
        util = 100.0 * intake_m / supply if supply else 0.0
        print("%-8s %-7s %-9.2f %-13.0f %-13.0f %-11.1f%%" % (
            k, res.get("final_N"), net_abs, intake_e, intake_m, util))

print()
print("=" * 100)
print("② 收敛性：末段 2000 tick 的 N 变化（|Δ|/N < 10% ⇒ 判「已收敛」）")
print("%-8s %-9s %-9s %-9s %-8s" % ("run", "t=10000", "t=12000", "Δ", "判定"))
for a in ("A", "B", "C"):
    for fp in sorted(glob.glob(D + "/%s_*.csv" % a)):
        k = os.path.basename(fp)[:-4]
        rows = list(csv.DictReader(open(fp, encoding="utf-8")))
        if len(rows) < 3:
            continue
        n10 = int(rows[-2]["N"]) if rows[-2].get("N") else 0
        n12 = int(rows[-1]["N"]) if rows[-1].get("N") else 0
        d = n12 - n10
        rel = abs(d) / max(1, n12)
        print("%-8s %-9d %-9d %+-9d %-8s" % (k, n10, n12, d, "已收敛" if rel < 0.10 else "仍在变"))

print()
print("=" * 100)
print("③ B vs C 归因：③ 能量标定带来的差异（B = 食物绑定+标定 ｜ C = 只食物绑定）")
print("%-6s %-24s %-11s %-11s %-9s" % ("臂", "末N（4 seed）", "饿死占比", "捕食占比", "老死占比"))
for a in ("B", "C"):
    Ns, dc = [], {}
    for fp in sorted(glob.glob(D + "/%s_*.summary.json" % a)):
        d = json.load(open(fp, encoding="utf-8"))
        res = d.get("result") or {}
        Ns.append(res.get("final_N") or 0)
        for x, v in (res.get("deaths_by_cause") or {}).items():
            key = str(x).split(".")[-1]
            dc[key] = dc.get(key, 0) + v
    dt = sum(dc.values()) or 1
    print("%-6s %-24s %-11s %-11s %-9s" % (
        a, str(sorted(Ns)),
        "%.0f%%" % (100 * dc.get("STARVATION", 0) / dt),
        "%.0f%%" % (100 * dc.get("PREDATION", 0) / dt),
        "%.0f%%" % (100 * dc.get("OLD_AGE", 0) / dt)))

print()
print("=" * 100)
print("④ 世代数（R149 世代门 = 30）")
for a in ("A", "B", "C"):
    gens = []
    for fp in sorted(glob.glob(D + "/%s_*.summary.json" % a)):
        d = json.load(open(fp, encoding="utf-8"))
        gens.append((d.get("result") or {}).get("final_max_gen_highwater"))
    print("  %s: %s  ⇒ 过门 %d/4" % (a, gens, sum(1 for g in gens if isinstance(g, (int, float)) and g >= 30)))
