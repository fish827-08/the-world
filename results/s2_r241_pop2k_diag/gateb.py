"""R241 门B（L1 探索度）专项判读。

用途：用户指令「B门也测一下」。标准 readout 的 L1 verdict 用 max(cov_pos, ear_pos) 的
宽松 OR，容易在只赢一个子指标时通关。本脚本做更严格的三口径联合判读：

  B1 覆盖率   cover_on > cover_off（末点累计访问斑块比例）
  B2 首访更早 first_visit_median_on < first_visit_median_off
  B3 覆盖增速 轨迹上 on 的覆盖率是否持续高于 off（逐采样点配对比较）

并给出「逐采样点 on>off 的比例」，避免只看末点的偶然性。
"""
import json
import sys

sys.path.insert(0, "/tmp/tw")


def load(csv_path):
    import csv as _csv
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rows.append(r)
    return rows


def num(r, k):
    v = r.get(k)
    if v is None or v == "" or v == "nan":
        return None
    try:
        return float(v)
    except Exception:
        return None


def main(csv_path):
    rows = load(csv_path)
    by = {}
    for r in rows:
        by.setdefault((int(float(r["seed"])), r["arm"]), []).append(r)
    seeds = sorted({k[0] for k in by})

    print("=" * 78)
    print(f"门B（L1 探索度）专项判读 —— {csv_path}")
    print("=" * 78)

    b1 = b2 = 0
    n = 0
    all_pairs = []
    for sd in seeds:
        on = sorted(by.get((sd, "on"), []), key=lambda r: float(r["tick"]))
        off = sorted(by.get((sd, "off"), []), key=lambda r: float(r["tick"]))
        if not on or not off:
            continue
        n += 1
        co_on = num(on[-1], "l1_visited_patch_frac")
        co_off = num(off[-1], "l1_visited_patch_frac")
        mo_on = num(on[-1], "l1_first_visit_median")
        mo_off = num(off[-1], "l1_first_visit_median")
        nov_on = num(on[-1], "l1_visited_patch_n")
        nov_off = num(off[-1], "l1_visited_patch_n")

        win = (co_on is not None and co_off is not None and co_on > co_off)
        ear = (mo_on is not None and mo_off is not None and mo_on < mo_off)
        b1 += int(bool(win))
        b2 += int(bool(ear))

        # B3：逐采样点配对，on 覆盖率 > off 的比例
        d_on = {round(float(r["tick"])): num(r, "l1_visited_patch_frac") for r in on}
        d_off = {round(float(r["tick"])): num(r, "l1_visited_patch_frac") for r in off}
        common = sorted(set(d_on) & set(d_off))
        gt = sum(1 for t in common
                 if d_on[t] is not None and d_off[t] is not None and d_on[t] > d_off[t])
        frac_gt = gt / len(common) if common else None
        all_pairs.append(frac_gt)

        print(f"\nseed={sd}")
        print(f"  覆盖率   off={co_off:.4f} ({nov_off:.0f}块)  on={co_on:.4f} ({nov_on:.0f}块)"
              f"   on>off? {'是' if win else '否'}"
              f"   Δ={((co_on or 0)-(co_off or 0)):+.4f}")
        print(f"  首访中位 off={mo_off}  on={mo_on}   on更早? {'是' if ear else '否'}")
        print(f"  逐点 on>off 占比 = {frac_gt if frac_gt is None else round(frac_gt,3)}"
              f"  (n_pts={len(common)})")

    print("\n" + "-" * 78)
    print(f"B1 覆盖率 on>off    : {b1}/{n}")
    print(f"B2 首访更早 on<off  : {b2}/{n}")
    # B3：跨 seed 平均逐点胜率
    b3_mean = sum(all_pairs) / len(all_pairs) if all_pairs else None
    print(f"B3 逐点胜率均值      : {b3_mean if b3_mean is None else round(b3_mean,3)}")
    print("-" * 78)
    maj = max(b1, b2) * 2 > n if n else None
    strict = (b1 * 2 > n) and (b2 * 2 > n)
    print(f"宽松口径 max(B1,B2)>2/3 : {maj}")
    print(f"严格口径 B1∧B2 各>2/3   : {strict}")
    print(f"B3 逐点胜率 >0.5        : {None if b3_mean is None else b3_mean > 0.5}")
    print("=" * 78)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/s2_r241_pop2k.csv")
