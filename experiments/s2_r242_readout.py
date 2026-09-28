"""R242 判读：几何放宽（斑块密度 10%）+ 背景低产能带，主看门B。

判据（R242 执行令 §三，预注册）：
  主判据 = 门B：B3 逐点胜率 > 0（on 覆盖率 ≥1 个采样点超过 off）+ B1 覆盖 on>off，
           多数 seed（≥2/3）；
  L0/L2 保持观察（rd 轮休已确证，不再作门）；L3 人均摄入作成本读数；
  附加核验：背景低产能格实际占比（确认 40% 生效）+ 生物在背景格时间占比（验证续命带被使用）。
"""
import csv
import statistics as st
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def num(r, k):
    v = r.get(k)
    if v is None or v == "" or v == "nan":
        return None
    try:
        return float(v)
    except Exception:
        return None


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(path):
    rows = load(path)
    by = {}
    for r in rows:
        by.setdefault((int(float(r["seed"])), r["arm"]), []).append(r)
    seeds = sorted({k[0] for k in by})

    print("=" * 80)
    print(f"R242 判读 —— {path}")
    print(f"配置：{rows[0].get('n_patches')} 斑块 / pop={rows[0]['pop']}@{rows[0]['tick']}")
    print("=" * 80)

    # ---------- 附加核验 ----------
    print("\n【附加核验】背景低产能带是否生效 + 续命带是否被使用")
    print(f"{'seed':>5} {'arm':>4} {'低产能占比':>10} {'低产能格数':>10} {'背景停留占比':>12}")
    for sd in seeds:
        for arm in ("off", "on"):
            v = by.get((sd, arm))
            if not v:
                continue
            last = v[-1]
            print(f"{sd:>5} {arm:>4} {num(last,'bg_low_frac_actual') or 0:>10.4f} "
                  f"{num(last,'bg_low_n') or 0:>10.0f} "
                  f"{(num(last,'bg_resid_frac') or 0):>12.4f}")

    # ---------- 主判据：门B ----------
    print("\n【主判据 · 门B（L1 探索度）】三口径")
    print(f"{'seed':>5} {'off覆盖':>9} {'on覆盖':>9} {'Δ覆盖':>9} "
          f"{'B1':>4} {'B3逐点胜率':>11} {'off首访':>8} {'on首访':>8} {'B2':>4}")
    b1 = b3 = b2 = 0
    n = 0
    for sd in seeds:
        on = sorted(by.get((sd, "on"), []), key=lambda r: float(r["tick"]))
        off = sorted(by.get((sd, "off"), []), key=lambda r: float(r["tick"]))
        if not on or not off:
            continue
        n += 1
        co_on, co_off = num(on[-1], "l1_visited_patch_frac"), num(off[-1], "l1_visited_patch_frac")
        mo_on, mo_off = num(on[-1], "l1_first_visit_median"), num(off[-1], "l1_first_visit_median")
        w1 = co_on is not None and co_off is not None and co_on > co_off
        w2 = mo_on is not None and mo_off is not None and mo_on < mo_off
        d_on = {round(float(r["tick"])): num(r, "l1_visited_patch_frac") for r in on}
        d_off = {round(float(r["tick"])): num(r, "l1_visited_patch_frac") for r in off}
        common = sorted(set(d_on) & set(d_off))
        gt = sum(1 for t in common
                 if d_on[t] is not None and d_off[t] is not None and d_on[t] > d_off[t])
        frac = gt / len(common) if common else 0.0
        b1 += int(bool(w1)); b2 += int(bool(w2)); b3 += int(frac > 0)
        print(f"{sd:>5} {co_off:>9.4f} {co_on:>9.4f} {co_on-co_off:>+9.4f} "
              f"{'是' if w1 else '否':>4} {frac:>11.3f} {mo_off:>8.0f} {mo_on:>8.0f} "
              f"{'是' if w2 else '否':>4}")

    print("-" * 80)
    print(f"B1 覆盖率 on>off      : {b1}/{n}")
    print(f"B2 首访更早 on<off    : {b2}/{n}")
    print(f"B3 逐点胜率 >0 的 seed: {b3}/{n}")
    gate_b = (b1 * 2 > n) and (b3 * 2 > n)
    print(f"⇒ 门B 判定（B1 ∧ B3 各 >2/3）: {'通过 ✅' if gate_b else '不通过 ❌'}")

    # ---------- L0 / L2 / L3 观察 ----------
    print("\n【观察层】L0 轮休 / L2 分化 / L3 摄入")
    l0 = l2g = l2v = 0
    l3r = []
    for sd in seeds:
        on, off = by.get((sd, "on"), []), by.get((sd, "off"), [])
        if not on or not off:
            continue
        if (num(on[-1], "l0_any_dead") or 0) > 0 or (num(on[-1], "l0_any_rest") or 0) > 0:
            l0 += 1
        og, ng = num(off[-1], "l2_visited_gini"), num(on[-1], "l2_visited_gini")
        ov, nv = num(off[-1], "l2_visited_var"), num(on[-1], "l2_visited_var")
        l2g += int(og is not None and ng is not None and ng > og)
        l2v += int(ov is not None and nv is not None and nv > ov)
        oi, ni = num(off[-1], "l3_intake_per_capita"), num(on[-1], "l3_intake_per_capita")
        if oi and ni:
            l3r.append(ni / oi)
    print(f"L0 轮休确证        : {l0}/{n}")
    print(f"L2 gini on>off     : {l2g}/{n}｜L2 var on>off: {l2v}/{n}")
    print(f"L3 人均摄入 on/off : {' '.join(f'{x:.3f}' for x in l3r)}"
          + (f"｜均值 {st.mean(l3r):.4f}" if l3r else ""))

    # ---------- K ----------
    print("\n【K 与饱和度】")
    for sd in seeds:
        o, nn = by.get((sd, "off")), by.get((sd, "on"))
        if o and nn:
            print(f"seed{sd}: K_off={num(o[-1],'pop'):>7.0f} K_on={num(nn[-1],'pop'):>6.0f} "
                  f"sat_off={num(o[-1],'global_sat'):.4f} sat_on={num(nn[-1],'global_sat'):.4f} "
                  f"cap_lost={num(nn[-1],'cap_lost_frac'):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/s2_r242_b2_p10k_m05.csv")
