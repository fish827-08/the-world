"""C8 前提对账的回归测试（R127）。

覆盖：
  · `latitude_r2` 的**判据语义**（只随行变 ⇒ 1.0；行内有变化 ⇒ <1；常量 ⇒ 1.0）
  · `capacity_stats` 的互补量（`off_row_frac` 与 R² 一致：uniform 恒 0）
  · `build_capacity_map` 复现引擎真实世界（双档可构造、总量守恒、patch_seed 可复现）
  · CLI 层：`preset_args` 摊平（grid/fixed/variants/无值开关）
  · 声明文件自洽（每个 preset 的 items 结构齐备 + **C8 会把 uniform 判为 FAIL**）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import c8_prereq_check as c8  # noqa: E402
from observatory.world_capacity import (  # noqa: E402
    build_capacity_map, capacity_stats, latitude_r2,
)


# ---------------------------------------------------------------- 纯函数
def test_latitude_r2_is_one_when_capacity_varies_only_by_row():
    """只随行（纬度）变 ⇒ 行内无残差 ⇒ **R² = 1.0**（uniform 世界的代数本质）。"""
    row = np.array([1.0, 2.0, 4.0])
    cap = np.repeat(row[:, None], 5, axis=1)          # 每行 5 列全同
    assert latitude_r2(cap) == pytest.approx(1.0)


def test_latitude_r2_drops_when_variation_is_within_row():
    """行内噪声越大 ⇒ 纬度解释不完 ⇒ **R² 越低**（用同一 between-row 信号 + 不同行内幅度比较）。"""
    row = np.array([1.0, 2.0, 3.0, 4.0])
    base = np.repeat(row[:, None], 6, axis=1)
    sign = np.where(np.arange(6) % 2 == 0, 1.0, -1.0)
    small = base + 0.1 * sign          # 行内小幅
    large = base + 5.0 * sign          # 行内大幅（压过 between-row）
    assert latitude_r2(small) < 1.0
    assert latitude_r2(large) < latitude_r2(small), "行内幅度越大 ⇒ R² 应越低"
    assert latitude_r2(base) == pytest.approx(1.0)     # 无行内噪声 ⇒ 1.0（对照）


def test_latitude_r2_constant_map_returns_one():
    """常量图（SS_总 = 0）⇒ 约定 1.0：无处可"靠信息获益"。"""
    assert latitude_r2(np.full((3, 4), 2.5)) == pytest.approx(1.0)


def test_latitude_r2_rejects_non_2d():
    with pytest.raises(ValueError):
        latitude_r2(np.arange(6))


def test_off_row_frac_complements_r2():
    """`off_row_frac` 是 R² 的互补观测量：只随行变 ⇒ 恒 0。"""
    cap = np.repeat(np.array([1.0, 2.0, 3.0])[:, None], 4, axis=1)
    assert capacity_stats(cap)["off_row_frac"] == 0.0
    cap2 = cap.copy(); cap2[1, 2] = 5.0
    assert capacity_stats(cap2)["off_row_frac"] > 0.0


# ---------------------------------------------------------------- 真实世界
def test_build_capacity_map_both_distributions():
    """用**引擎真实网格**（60×120）复算 —— patchy 在小网格上会因斑块占比过大而拒绝构造。"""
    uni = build_capacity_map("uniform")
    pat = build_capacity_map("patchy")
    assert uni.shape == pat.shape == (60, 120)
    assert latitude_r2(uni) == pytest.approx(1.0)          # uniform ⇒ 完全可预测
    assert latitude_r2(pat) < 1.0                          # patchy ⇒ 有不可预测部分
    # 容量守恒（patchy 的设计承诺：Σ_capacity 与 uniform 相等）
    assert pat.sum() == pytest.approx(uni.sum(), rel=1e-9)


def test_build_capacity_map_is_deterministic_by_seed():
    a = build_capacity_map("patchy", seed=7)
    b = build_capacity_map("patchy", seed=7)
    c = build_capacity_map("patchy", seed=8)
    assert np.array_equal(a, b), "同 seed 必须逐位复现"
    assert not np.array_equal(a, c), "换 seed 应改变斑块位置"


def test_build_capacity_map_small_grid_needs_smaller_patches():
    """小网格必须配更小的 `patch_count`/`patch_radius`，否则斑块占比过大 ⇒ 显式报错（不静默）。"""
    from simulation.config import ResourceConfig
    tiny = build_capacity_map("patchy", rows=12, cols=24, seed=42,
                              resource=ResourceConfig(patch_count=3, patch_radius=1))
    assert tiny.shape == (12, 24)
    with pytest.raises(ValueError):
        build_capacity_map("patchy", rows=12, cols=24, seed=42)   # 默认 30 个斑块 ⇒ 拒绝


def test_build_capacity_map_rejects_unknown_distribution():
    with pytest.raises(ValueError):
        build_capacity_map("hexagonal")


# ---------------------------------------------------------------- CLI 助手与声明
def test_preset_args_flattens_fixed_variants_and_flags():
    d = {"fixed": ["mode=on", "snapshot-every=2000"],
         "grid": ["seed=1,2"],
         "variants": [{"args": ["distribution=patchy", "calibration-arm"]}]}
    a = c8.preset_args([d])
    assert a["mode"] == "on" and a["distribution"] == "patchy"
    assert a["calibration-arm"] == "true"       # 无值开关 ⇒ "true"
    assert a["<grid>seed"] == "seed"            # grid 键单独标注，不作普通参数


def test_prerequisites_file_is_wellformed_and_has_patchy_gate():
    decl = c8.load_prerequisites()
    assert "cstep3patchy" in decl and "cstep3alpha8" in decl
    for name, d in decl.items():
        if name.startswith("_"):
            continue
        assert d.get("science_question"), f"{name} 缺 science_question"
        for it in d["items"]:
            assert it["kind"] in ("preset_arg_equals", "capacity_lat_r2_max",
                                  "summary_switches_equals",
                                  # 臂变量（两臂取值不同 ⇒ `preset_arg_equals` 摊平后会被覆盖）
                                  "variant_arg_equals"), f"{name}: 未知 kind"
            assert it.get("why"), f"{name}/{it.get('id')} 缺 why（R127 要求写明理由）"
    # patchy 侧必须声明「容量不可由纬度预测」这条**信息价值前提**
    kinds = {it["kind"] for it in decl["cstep3patchy"]["items"]}
    assert "capacity_lat_r2_max" in kinds


def test_c8_would_have_blocked_uniform_information_experiment():
    """🔴 本模块存在的**理由**：uniform 世界的 `lat_r2 = 1.000` ⇒ 信息价值前提不成立。

    若这条断言哪天失败（例如有人把阈值放宽到 1.0），C8 就失去了拦阻能力。
    """
    st = capacity_stats(build_capacity_map("uniform"))
    assert st["lat_r2"] > c8.DEFAULT_MAX_LAT_R2, (
        f"uniform 的 lat_r2={st['lat_r2']} 未超过阈值 {c8.DEFAULT_MAX_LAT_R2} ⇒ C8 失去拦阻力"
    )
    stp = capacity_stats(build_capacity_map("patchy"))
    assert stp["lat_r2"] <= c8.DEFAULT_MAX_LAT_R2, "patchy 应通过（否则信息价值类实验永远无法开跑）"


# ---------------------------------------------------------------- Pre-Flight 集成
def test_preflight_c8_report_passes_for_patchy_preset():
    """Pre-Flight 的 C8 入口：patchy 批应**通过**（前提成立）。"""
    from experiments.preflight_check import c8_report
    ok, lines = c8_report("cstep3patchy")
    assert ok is True
    assert any("capacity_not_latitude_predictable" in x for x in lines)


def test_preflight_c8_retrospective_fail_does_not_block():
    """事后登记 preset：基础项 FAIL，但**不计入** Pre-Flight 判定（历史证据）。"""
    from experiments.preflight_check import c8_report
    ok, lines = c8_report("cstep3alpha8")
    assert ok is True, "retrospective 的 FAIL 不应阻塞"
    assert any("❌" in x for x in lines), "但报告里必须**如实显示**该 FAIL"
    assert any("事后登记" in x for x in lines)


def test_preflight_c8_rejects_undeclared_preset():
    """未声明前提的 preset ⇒ **不通过**（R127 要求每批声明，机器可读）。"""
    from experiments.preflight_check import c8_report
    ok, lines = c8_report("preset_never_declared")
    assert ok is False
    assert any("未在 prerequisites.json 中声明" in x for x in lines)
