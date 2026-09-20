#!/usr/bin/env python.exe
# -*- coding: utf-8 -*-
"""calib_solve —— 能量校准预实验的**解算器**（R141/R142；零机时工具）

用途：把"校准批"的读数换算成**参数标定值**，代替网格试错。

读取（任选）：
  · summary 的 `energy_ledger`（首选；见派工单 §1.3 列名）
  · CSV 的分组净收入时间序列列 `g_{lo,mid,hi}_net_{mean,p50,var}`

做三件事：
  ① 报三腿（食草 lo / 杂食 mid / 捕食 hi）的**净收入 mean / median / 方差**（窗口默认 4k–8k）
  ② 核预注册判据：主 |hi−lo|/lo ≤ 20%｜副① mid < min(lo,hi)｜副② Var(hi) > Var(lo)
  ③ 解标定值（平衡条件 net_hi == net_lo）：
       transfer* = (I_forage_lo − C_lo + C_hi) / (k · E_prey)
       或在 transfer 目标固定时，反解所需击杀率倍数 k*/k（用于分摊到 gate / prob_coef / 成功率）

用法：
  python.exe tools/calib_solve.py --preset calib1            # 自动找 _rerun_logs/<preset>/*
  python.exe tools/calib_solve.py --dir _rerun_logs/calib1 --window 4000,8000
  python.exe tools/calib_solve.py --selftest                 # 合成数据自检（不需要真数据）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

REQUIRED_CSV = [
    'g_lo_n', 'g_lo_net_mean', 'g_lo_net_p50', 'g_lo_net_var',
    'g_mid_n', 'g_mid_net_mean', 'g_mid_net_p50', 'g_mid_net_var',
    'g_hi_n', 'g_hi_net_mean', 'g_hi_net_p50', 'g_hi_net_var',
    'mean_energy', 'prey_energy_mean',
]


def _f(v):
    """宽松转 float；空/None/非数字 ⇒ None（**不写 0**，R120 口径纪律）。"""
    if v is None:
        return None
    s = str(v).strip()
    if s == '' or s.lower() in ('none', 'nan', 'n/a'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def find_dir(preset: str | None, dirp: str | None) -> str:
    if dirp:
        return dirp
    if not preset:
        print('❌ 需给 --preset 或 --dir', file=sys.stderr)
        sys.exit(2)
    cands = [f'_rerun_logs/{preset}', f'the-world-data/_rerun_logs/{preset}']
    for c in cands:
        if os.path.isdir(c):
            return c
    print(f'❌ 找不到目录：{cands}', file=sys.stderr)
    sys.exit(2)


def load_csv_window(d: str, lo: int, hi: int) -> dict:
    """返回 {'runs': {name: rows}, 'missing_cols': [...]}（只读窗口内的行）。"""
    files = sorted(glob.glob(os.path.join(d, '*.csv')))
    files = [f for f in files if not os.path.basename(f).startswith('_')]
    if not files:
        print(f'❌ {d} 下没有 run CSV', file=sys.stderr)
        sys.exit(2)
    head = next(csv.reader(open(files[0], encoding='utf-8')))
    missing = [c for c in REQUIRED_CSV if c not in head]
    runs = {}
    for f in files:
        name = os.path.basename(f)[:-4]
        rows = []
        with open(f, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                t = _f(r.get('tick'))
                if t is None or t < lo or t > hi:
                    continue
                rows.append(r)
        if rows:
            runs[name] = rows
    return {'runs': runs, 'missing_cols': missing, 'header': head}


def leg_stats(rows: list[dict], leg: str) -> dict:
    """按 tick 取窗口内该腿的 mean/p50/var（对时间取中位 ⇒ 抗单 tick 噪声）。"""
    def col(suffix):
        return [x for x in (_f(r.get(f'g_{leg}_net_{suffix}')) for r in rows) if x is not None]

    def col_n():
        # ⚠️ 人数列名是 `g_{leg}_n`（**不含 `_net_`**）——2026-09-21 修（旧版写成
        #    `g_{leg}_net_n` ⇒ 恒 None，是**我自己的 bug**，非数据问题）。
        return [x for x in (_f(r.get(f'g_{leg}_n')) for r in rows) if x is not None]

    means, p50s, vars_, ns = col('mean'), col('p50'), col('var'), col_n()
    out = {}
    for k, v in (('mean', means), ('p50', p50s), ('var', vars_)):
        out[k] = st.median(v) if v else None
    out['n_mean'] = st.mean(ns) if ns else None
    out['n_ticks'] = len(means)
    return out


def solve_from_ledger(led: dict, transfer_target: float) -> dict:
    """从 `energy_ledger` 解 transfer* 与所需击杀率倍数。"""
    obs = _f(led.get('obs_count'))
    g = (led.get('groups') or {})
    prey = (led.get('prey') or {})
    lo, hi = g.get('lo') or {}, g.get('hi') or {}
    if not obs:
        return {'error': 'energy_ledger.obs_count 缺失'}

    def per(d, key):
        v = _f(d.get(key))
        return None if v is None else v / obs

    I_forage_lo = per(lo, 'intake_forage_sum')
    C_lo = None
    if per(lo, 'cost_meta_sum') is not None:
        C_lo = sum(x for x in (per(lo, 'cost_meta_sum'), per(lo, 'cost_move_sum')) if x is not None)
    C_hi = None
    if per(hi, 'cost_meta_sum') is not None:
        C_hi = sum(x for x in (per(hi, 'cost_meta_sum'), per(hi, 'cost_move_sum'),
                               per(hi, 'cost_attack_sum')) if x is not None)
    k = per(prey, 'kills')
    E_prey = None
    ks = _f(prey.get('energy_sum'))
    if ks is not None and _f(prey.get('kills')):
        E_prey = ks / _f(prey['kills'])

    res = dict(I_forage_lo=I_forage_lo, C_lo=C_lo, C_hi=C_hi, k=k, E_prey=E_prey)
    if None in (I_forage_lo, C_lo, C_hi, k, E_prey) or not k or not E_prey:
        res['error'] = '必需量缺失（I_forage_lo / C_lo / C_hi / k / E_prey 之一）'
        return res
    res['transfer_star'] = (I_forage_lo - C_lo + C_hi) / (k * E_prey)
    need_k = (I_forage_lo - C_lo + C_hi) / (transfer_target * E_prey)
    res['transfer_target'] = transfer_target
    res['k_needed_at_target'] = need_k
    res['flux_multiple_needed'] = (need_k / k) if k else None
    return res


def fmt(x, nd=4):
    return 'None' if x is None else f'{x:.{nd}f}'


def report(d: str, lo: int, hi: int, transfer_target: float) -> int:
    dat = load_csv_window(d, lo, hi)
    print(f'📂 {d}｜窗口 {lo}–{hi}｜run 数 {len(dat["runs"])}')
    if dat['missing_cols']:
        print('\n🔴 缺列（P0 尚未实施或列名不符）：')
        for c in dat['missing_cols']:
            print('   -', c)
        print('   ⇒ 本工具按派工单 §1.3 锁定列名；列未就绪前无法解算。')
        # 仍尝试用 summary 的 energy_ledger
    print('\n=== ① 三腿净收入（窗口内时间中位）===')
    print('%-6s %10s %10s %10s %10s %8s' % ('腿', 'mean', 'median', 'var', 'n/tick', 'tick数'))
    legs = {}
    for leg in ('lo', 'mid', 'hi'):
        vals = [leg_stats(rows, leg) for rows in dat['runs'].values()]
        agg = {}
        for k in ('mean', 'p50', 'var', 'n_mean'):
            xs = [v[k] for v in vals if v.get(k) is not None]
            agg[k] = st.median(xs) if xs else None
        legs[leg] = agg
        print('%-6s %10s %10s %10s %10s %8d' % (
            leg, fmt(agg['mean']), fmt(agg['p50']), fmt(agg['var']), fmt(agg['n_mean'], 0), vals[0]['n_ticks']))

    lo_, mid_, hi_ = legs['lo'], legs['mid'], legs['hi']
    print('\n=== ② 预注册判据 ===')
    ok_main = False
    if lo_.get('p50') is not None and hi_.get('p50') is not None and lo_['p50'] not in (0, None):
        rel = abs(hi_['p50'] - lo_['p50']) / abs(lo_['p50'])
        ok_main = rel <= 0.20
        print(f'  主 |hi−lo|/lo = {rel*100:.1f}%  (阈值 ≤20%)  ⇒ {"✅ 过" if ok_main else "❌ 未达"}')
    else:
        print('  主 判据无法算（缺 p50）')
    if lo_.get('p50') is not None and mid_.get('p50') is not None and hi_.get('p50') is not None:
        ok = mid_['p50'] < min(lo_['p50'], hi_['p50'])
        print(f'  副① 杂食最低：mid={fmt(mid_["p50"])} < min(lo={fmt(lo_["p50"])}, hi={fmt(hi_["p50"])})  ⇒ {"✅ 过" if ok else "❌ 未过"}')
    if lo_.get('var') is not None and hi_.get('var') is not None:
        ok = hi_['var'] > lo_['var']
        print(f'  副② 风险化：Var(hi)={fmt(hi_["var"])} > Var(lo)={fmt(lo_["var"])}  ⇒ {"✅ 过" if ok else "❌ 未过"}')

    print('\n=== ③ 标定解算（优先 summary.energy_ledger）===')
    solved = False
    for f in sorted(glob.glob(os.path.join(d, '*.summary.json'))):
        j = json.load(open(f, encoding='utf-8'))
        led = (j.get('result') or {}).get('energy_ledger') or {}
        if not led:
            continue
        r = solve_from_ledger(led, transfer_target)
        print(f'  · {os.path.basename(f)}')
        if r.get('error'):
            print('    ', r['error'])
            continue
        solved = True
        print(f'    人均/ tick：I_forage(lo)={fmt(r["I_forage_lo"])}  C_lo={fmt(r["C_lo"])}  C_hi={fmt(r["C_hi"])}'
              f'  k={fmt(r["k"], 6)}  E_prey={fmt(r["E_prey"], 2)}')
        print(f'    ⇒ transfer* = {fmt(r["transfer_star"])}（平衡点）')
        print(f'    ⇒ 若 transfer 固定 {transfer_target}：所需击杀率倍数 = {fmt(r["flux_multiple_needed"], 2)}×'
              f'（分摊到 gate / prob_coef / 成功率）')
    if not solved:
        print('  （无 energy_ledger ⇒ 待 P0-1 实施后重跑本工具）')

    print('\n📌 纪律：解算值是**标定输入**，不是结论；套用后必须跑**确认批**（新 seed 或新 run）复核，'
          '且参数改动需进 `docs/实验记录.md` + 预注册。')
    return 0


def selftest() -> int:
    """合成数据：验证解算公式与判据逻辑（不需要真数据/引擎）。"""
    led = {'obs_count': 1000.0,
           'groups': {'lo': {'intake_forage_sum': 900.0, 'cost_meta_sum': 600.0, 'cost_move_sum': 100.0},
                      'hi': {'cost_meta_sum': 700.0, 'cost_move_sum': 100.0, 'cost_attack_sum': 50.0}},
           'prey': {'kills': 40.0, 'energy_sum': 4000.0}}
    r = solve_from_ledger(led, 0.9)
    # 手上的解析解：I=0.9, C_lo=0.7, C_hi=0.85, k=0.04, E=100
    exp = (0.9 - 0.7 + 0.85) / (0.04 * 100)
    assert not r.get('error'), r
    assert abs(r['transfer_star'] - exp) < 1e-9, (r['transfer_star'], exp)
    assert abs(r['I_forage_lo'] - 0.9) < 1e-9 and abs(r['E_prey'] - 100.0) < 1e-9
    print('✅ selftest：解算公式正确（transfer* = %.4f，期望 %.4f）' % (r['transfer_star'], exp))
    print('   所需击杀率倍数（transfer=0.9）= %.2f×' % r['flux_multiple_needed'])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description='能量校准预实验解算器（R141/R142）')
    ap.add_argument('--preset', help='预设名（自动找 _rerun_logs/<preset>）')
    ap.add_argument('--dir', help='直接给目录')
    ap.add_argument('--window', default='4000,8000', help='窗口 tick（默认 4000,8000；实测 2k 前为暂态）')
    ap.add_argument('--transfer-target', type=float, default=0.9, help='若 transfer 固定的目标值（默认 0.9）')
    ap.add_argument('--selftest', action='store_true', help='合成数据自检')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    lo, hi = (int(x) for x in a.window.split(','))
    return report(find_dir(a.preset, a.dir), lo, hi, a.transfer_target)


if __name__ == '__main__':
    sys.exit(main())
