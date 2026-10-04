#!/usr/bin/env python.exe
# -*- coding: utf-8 -*-
"""pred_flux_recheck —— R367-N3 评估稿的判读脚本（零机时，只读既有产物）。

为 `docs/评估-EVAL/评估-捕食庇护所信息价值-三案-20261004.md` §1.2/§1.4 提供可复现口径：

  ① 通量表：捕食/食草能量通量比、E_prey、击杀率、击杀/生、捕食致死占比
     （源 = summary.json 的 `energy_ledger` + `deaths_by_cause`）
  ② 风险结构首查：steady-state（剔暂态）逐窗 d_pred 差分的 CV / lag-1 自相关 / 分散指数
     （源 = run CSV 的**累计**列 `d_pred` 与 `N`；⚠️ d_pred 是累计计数，必须差分）

默认数据源：`the-world-data/experiments/r198-migration-calendar-compass`
（2026-09-25，能量封顶纪元后、带账本的最新完整批；Rust 路径批无账本 —— 见评估稿 §1.1）。

用法：
  python.exe tools/pred_flux_recheck.py                 # 默认数据源，4 臂全跑
  python.exe tools/pred_flux_recheck.py --dir <批根目录>  # 换批（需同结构：<dir>/<arm>/*.csv|*.summary.json）
  python.exe tools/pred_flux_recheck.py --selftest       # 合成数据自检（不需要真数据）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

import numpy as np

DEFAULT_DIR = os.path.join('the-world-data', 'experiments', 'r198-migration-calendar-compass')
ARMS = ('mig_base', 'mig_g', 'mig_2g', 'mig_noseason')


def _run_csvs(arm_dir: str) -> list[str]:
    return [f for f in sorted(glob.glob(os.path.join(arm_dir, '*.csv')))
            if not os.path.basename(f).startswith('_')]


def _summary_jsons(arm_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(arm_dir, '*.summary.json')))


def flux_table(base: str) -> dict[str, dict]:
    """① 通量表（每臂 6 seed 中位）。"""
    out = {}
    for arm in ARMS:
        d = os.path.join(base, arm)
        if not os.path.isdir(d):
            continue
        rows = []
        for f in _summary_jsons(d):
            r = (json.load(open(f, encoding='utf-8')).get('result') or {})
            led = r.get('energy_ledger') or {}
            g = led.get('global') or {}
            obs = led.get('obs_count')
            prey = led.get('prey') or {}
            dc = r.get('deaths_by_cause') or {}
            died = r.get('died_total') or sum(dc.values())
            if not obs or not g or not g.get('intake_forage_sum'):
                continue
            rows.append(dict(
                pr=g['intake_pred_sum'] / g['intake_forage_sum'] * 100.0,
                ep=(prey.get('energy_sum', 0.0) / prey['kills']) if prey.get('kills') else None,
                k=prey.get('kills', 0) / obs,
                kl=(prey.get('kills', 0) / died) if died else None,
                iff=g['intake_forage_sum'] / obs,
                dsh=(dc.get('DeathCause.PREDATION', 0) / died * 100.0) if died else None,
            ))
        if not rows:
            continue

        def med(key):
            xs = [x[key] for x in rows if x[key] is not None]
            return st.median(xs) if xs else float('nan')

        out[arm] = dict(pr=med('pr'), ep=med('ep'), k=med('k'), kl=med('kl'),
                        iff=med('iff'), dsh=med('dsh'), n=len(rows))
    return out


def structure_table(base: str, min_tick: float = 4000.0) -> dict[str, dict]:
    """② 风险结构首查：d_pred 累计列差分（窗宽=采样间隔）⇒ CV / lag-1 / 分散指数。"""
    out = {}
    for arm in ARMS:
        d = os.path.join(base, arm)
        if not os.path.isdir(d):
            continue
        cvs, acs, disps, rates = [], [], [], []
        for f in _run_csvs(d):
            rs = list(csv.DictReader(open(f, encoding='utf-8-sig')))
            if not rs or 'd_pred' not in rs[0]:
                continue
            ticks = np.array([float(r['tick']) for r in rs])
            dp = np.array([float(r['d_pred']) for r in rs])
            n = np.array([float(r['N']) for r in rs])
            dd = np.diff(dp)               # ⚠️ d_pred 为累计 ⇒ 差分
            tmid, nn = ticks[1:], n[1:]
            m_ = tmid >= min_tick
            dd, nn = dd[m_], nn[m_]
            if len(dd) < 6 or nn.mean() == 0:
                continue
            rate = dd / nn * 1000.0        # 每千人每窗
            m = rate.mean()
            if m <= 0:
                continue
            cvs.append(rate.std(ddof=1) / m)
            dz = rate - rate.mean()
            den = float((dz * dz).sum())
            acs.append(float((dz[:-1] * dz[1:]).sum() / den) if den > 0 else 0.0)
            disps.append(dd.var(ddof=1) / dd.mean())
            rates.append(m)
        if not cvs:
            continue
        out[arm] = dict(rate=st.median(rates), cv=st.median(cvs),
                        ac=st.median(acs), disp=st.median(disps), n=len(cvs))
    return out


def report(base: str) -> int:
    if not os.path.isdir(base):
        print(f'❌ 数据目录不存在：{base}', file=sys.stderr)
        return 2
    print(f'📂 {base}')
    print('\n=== ① 通量表（臂×6seed 中位；源=energy_ledger + 死因表）===')
    print('%-14s %8s %9s %10s %9s %9s %10s' % ('arm', 'pred%', 'E_prey', 'kills/obs', 'kills/died', 'I_for/t', 'pred死因%'))
    fl = flux_table(base)
    for arm, v in fl.items():
        print('%-14s %7.2f%% %9.1f %10.2e %9.3f %9.2f %9.1f%%  (n=%d)'
              % (arm, v['pr'], v['ep'], v['k'], v['kl'], v['iff'], v['dsh'], v['n']))
    print('\n=== ② 风险结构首查（窗=采样间隔；剔 t<%d 暂态）===' % 4000)
    print('%-14s %13s %8s %9s %9s' % ('arm', '每千人/千t', 'CV', 'lag1-AC', 'disp'))
    stt = structure_table(base)
    for arm, v in stt.items():
        print('%-14s %13.3f %8.3f %9.3f %9.3f  (n=%d)'
              % (arm, v['rate'], v['cv'], v['ac'], v['disp'], v['n']))
    print('\n📌 纪律：d_pred 为**累计**列（差分后方可判结构）；本脚本只读既有产物，不改任何文件。')
    return 0


def selftest() -> int:
    """合成数据：验证差分逻辑与统计公式。"""
    import tempfile
    d = tempfile.mkdtemp(prefix='pred_recheck_')
    arm = os.path.join(d, 'mig_base')
    os.makedirs(arm)
    # 累计序列：每窗 +10 ⇒ 差分恒 10；N 恒 1000 ⇒ rate=10/窗
    with open(os.path.join(arm, 'x.csv'), 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['tick', 'N', 'd_pred'])
        for i in range(1, 21):
            w.writerow([i * 1000, 1000, i * 10])
    stt = structure_table(d)
    v = stt['mig_base']
    assert abs(v['rate'] - 10.0) < 1e-9, v
    assert v['cv'] < 1e-9, v          # 常数序列 ⇒ CV=0
    assert abs(v['disp']) < 1e-9, v   # 常数计数 ⇒ var=0
    # 通量表空源 ⇒ 空字典
    assert flux_table(d) == {}
    print('✅ selftest：差分/统计口径正确（rate=10.0, CV=0, disp=0）')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description='R367-N3 判读脚本（零机时）')
    ap.add_argument('--dir', default=DEFAULT_DIR, help='批根目录（默认 r198）')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    return selftest() if a.selftest else report(a.dir)


if __name__ == '__main__':
    sys.exit(main())
