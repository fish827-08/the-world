"""共享 helper：合成批次（C 步判读用）——供 `test_r110_criteria.py` 使用。

来历（W0-T，2026-10-18）：这两段原定义在 `tests/test_cstep_judge.py` 里
（模块级 `pytestmark = pytest.mark.legacy`）；该文件已整体归档到 `tests/archive/`，
而主线用例 `test_r110_criteria.py` 仍需要这批合成行 ⇒ 把**数据构造**（`_row` /
`synth_batch`）提到主线可 import 的 helper 模块，归档件保持自洽（自带一份冻结副本）。

⚠️ 本模块**不是测试文件**（文件名不匹配 `test_*.py`），不会被收集。
⚠️ 这里只是**测试夹具数据**，不含任何物理常数计算（W0-T 新测试规范第 3 条）。
"""
from __future__ import annotations

SEEDS = (42, 43, 44, 45, 46, 47)


def _row(name: str, *, ratio=None, N=3240, pred=0.4, eco=True, rho=0.0,
         resp_tri=0.02, is_cal=False, max_count=3240, rand=False) -> dict:
    return {"name": name, "kind": ("oracle" if name.startswith(("oracle_", "rand_oracle"))
                                  else ("rand" if rand else name.split("_")[0])),
            "m": (float(name.split("_m")[1].split("_")[0])
                  if "_m" in name else None),
            "is_rand": rand, "seed": int(name.rsplit("_s", 1)[1]),
            "ratio": ratio, "N": N, "pred_frac": pred, "eco_gate": eco,
            "eco_applicable": True, "rho": rho, "resp_a": 0.9, "resp_b": 0.02,
            "resp_tri": resp_tri, "rs": {}, "ledger": {"identity_ok": True},
            "is_cal": is_cal, "sw_m": 1.3, "signal_mode": "random" if rand else "state",
            "max_count": max_count, "dir": "synth"}


def synth_batch(*, m13_ratio=1.25, m10_ratio=0.80, eco=lambda s: True,
                m13_rho=(0.10, 0.12, 0.11, 0.13, 0.09, 0.12),
                zero_rho=(-0.01, 0.02, -0.02, 0.01, 0.00, -0.01),
                main_rho=(0.05, 0.07, 0.04, 0.06, 0.05, 0.06)) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for i, s in enumerate(SEEDS):
        rows[f"oracle_m1.3_s{s}"] = _row(f"oracle_m1.3_s{s}", ratio=m13_ratio,
                                        eco=eco(s), rho=m13_rho[i], is_cal=True)
        rows[f"oracle_m1.0_s{s}"] = _row(f"oracle_m1.0_s{s}", ratio=m10_ratio,
                                        eco=eco(s), rho=0.01, is_cal=True)
        rows[f"zero_s{s}"] = _row(f"zero_s{s}", rho=zero_rho[i])
        rows[f"main_s{s}"] = _row(f"main_s{s}", rho=main_rho[i])
    return rows
