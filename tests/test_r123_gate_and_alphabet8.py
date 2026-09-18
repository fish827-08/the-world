"""R123 实施回归：① `"8"` 档（B③ 记忆位）② 门控臂（Δ_content 付款闸）。

纪律对应：
- **C6 仪器极性**：门控臂必须有端到端极性测试（写反必失败）
- **三条守卫**：门控 + 探针关 / 无信息不对称 / 未登记校准臂 ⇒ 一律**硬失败**
  （否则 Δ≡0 ⇒ 付款全消失 ⇒ 被读成"信号无用"的**假结论**）
- `"8"` 档：码域 ⊆ {1..8}、零码零容忍、记忆位语义（非冗余性）、码本/random 域随档
- RNG 契约：切档不改抽取**形状/个数**（本测试断言第 1 tick 的方法序列）
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import (
    InfoStructureConfig, OracleConfig, SimConfig, SIGNAL_ALPHABET_CODE_MAX,
)
from simulation.sphere_engine import SphereEngine, codebook_init_rows, encode_signal_states

SEEDS = (42, 43, 44)


def _engine(*, alphabet="16", gate=False, probe=True, radius=4, tau=0.15,
            calibration=True, seed=42, max_count=600, oracle=True) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = max_count
    d2 = InfoStructureConfig(enabled=True, learning_rate=0.05)
    d2.measure_signal_response = probe
    d2.perception_radius = radius
    d2.softmax_tau = tau
    cfg.info_structure = d2
    cfg.signal_alphabet = alphabet
    cfg.oracle = OracleConfig(
        enabled=oracle, donation=1.0, gain_multiplier=1.3 if calibration else 1.0,
        is_calibration_arm=calibration, gate_mode="delta_positive" if gate else "none",
        gate_delta="content",
    )
    return SphereEngine(cfg)


# ============================================================ ① `"8"` 档

def test_alphabet8_code_domain_and_zero_code_free():
    """`"8"` 档：12 例真实引擎跑动后，码域 ⊆ {1..8} 且 `bad_code_n == 0`。"""
    for seed in SEEDS:
        e = _engine(alphabet="8", gate=False, calibration=False, seed=seed, oracle=False)
        for _ in range(200):
            if e.extinct:
                break
            e.step()
        a = e.alphabet_stats()
        assert a["signal_alphabet"] == "8"
        assert a["n_states"] == 8 and a["code_max"] == SIGNAL_ALPHABET_CODE_MAX["8"] == 8
        assert a["bad_code_n"] == 0, "8 档不得出现码 0 或越界码（§3.5 守卫）"


def test_alphabet8_memory_bit_polarity_end_to_end():
    """🔴 极性（端到端）：记忆中**另有一个（≠当前格）**富食点 ⇒ 位为 1；否则为 0。

    `code = e_bin*2 + mem_bit + 1`。用纯函数直测（免插桩，设计稿 §3.6 手法）。
    🔴 **2026-09-19 语义修正**（内评 01:4x 抽核）：判据由「当前格 ∈ 记忆」改为
    「∃ 记忆格 ≠ 当前格」—— 旧判据 = `f_bit` 的时间延迟版（当前格食物接收者能直读）⇒ 冗余。
    """
    energy = np.array([0.5, 0.5])
    grid = np.zeros(4)
    cap = np.ones(4)
    occ = np.ones(4, dtype=np.int64)
    e_flat = np.array([10, 11], dtype=np.int64)
    emitters = np.array([0, 1], dtype=np.int64)
    wm = np.full((2, 4), -1, dtype=np.int64)
    wm[0, 0] = 77          # 个体 0：记忆里有一个**≠ 当前格(10)** 的富食点 ⇒ mem_bit=1
    # 个体 1：记忆全空 ⇒ mem_bit=0
    state, off = encode_signal_states(energy, grid, cap, occ, e_flat, emitters,
                                      max_energy=10.0, alphabet="8", work_memory=wm)
    assert off == 1
    # ⚠️ 极性判法：`state = e_bin*2 + mem_bit` ⇒ **同一 e_bin 下** mem_bit=1 使 state 恰 +1
    # （不能用奇偶判：e_bin=0 时 mem=1 ⇒ state=1 是奇数）
    assert state[0] == state[1] + 1, "记忆含非当前格 ⇒ mem_bit=1 ⇒ state 比未命中恰大 1"
    assert 0 <= state.min() and (state + off).max() <= 8, "码域必须 ⊆ 1..8"


def test_alphabet8_requires_work_memory():
    """缺 `work_memory` ⇒ **硬失败**（不静默降级：否则记忆位恒 0 = 静默退化为 "4" 档）。"""
    with pytest.raises(ValueError, match="work_memory"):
        encode_signal_states(
            np.array([0.5]), np.zeros(1), np.ones(1), np.ones(1, dtype=np.int64),
            np.array([0], dtype=np.int64), np.array([0], dtype=np.int64),
            max_energy=10.0, alphabet="8",
        )


def test_alphabet8_is_not_redundant_given_memory_differs():
    """非冗余性：同 (能量, 食物, 密度) 但**记忆不同** ⇒ 码必须不同（接收者无法直读记忆）。"""
    common = dict(
        energy=np.array([0.5, 0.5]), grid=np.zeros(4), capacity=np.ones(4),
        occ=np.ones(4, dtype=np.int64), e_flat=np.array([10, 11], dtype=np.int64),
        emitters=np.array([0, 1], dtype=np.int64), max_energy=10.0,
    )
    wm_a = np.full((2, 4), -1, dtype=np.int64); wm_a[0, 0] = 77   # ≠ 当前格(10) ⇒ mem_bit=1
    wm_b = np.full((2, 4), -1, dtype=np.int64)                    # 空记忆 ⇒ mem_bit=0
    s_a, _ = encode_signal_states(alphabet="8", work_memory=wm_a, **common)
    s_b, _ = encode_signal_states(alphabet="8", work_memory=wm_b, **common)
    assert s_a[0] != s_b[0], "记忆不同 ⇒ 码必须不同（该位携带接收者不可直读的信息）"


def test_alphabet8_codebook_and_random_domain():
    """码本初始化/理论上限：`"8"` 的初值 ⊆ [1,8]；`"4"`/`"16"` 口径不变。"""
    for alpha, hi in (("8", 8), ("4", 4), ("16", 15)):
        rows = codebook_init_rows(4, alpha)
        assert rows.shape == (4, 16)          # 宽度恒 16（存档兼容，R113 明文）
        assert rows.min() >= 0 and rows.max() <= hi


def test_alphabet8_fingerprint_and_cross_alphabet_resume_blocked(tmp_path):
    """`"8"` 进指纹 ⇒ 跨档续跑**硬报错**（防混口径续跑）；同档可续。"""
    e = _engine(alphabet="8", gate=False, calibration=False, seed=7, oracle=False)
    for _ in range(30):
        e.step()
    p = tmp_path / "s8.npz"
    e.save_snapshot(str(p))
    assert "8" in SimConfig(seed=7, signal_alphabet="8").fingerprint()
    e2 = _engine(alphabet="8", gate=False, calibration=False, seed=7, oracle=False)
    SphereEngine.load_snapshot(str(p), config=e2.config)          # 同档 ⇒ 通过
    bad = SimConfig(seed=7, signal_alphabet="4")
    with pytest.raises(ValueError, match="指纹"):
        SphereEngine.load_snapshot(str(p), config=bad)


# ============================================================ ② 门控臂

def test_gate_polarity_blocks_when_delta_nonpositive():
    """🔴 极性（C6）：付款闸 = `Δ_content > 0`。构造"未受内容影响"的到达 ⇒ **不付款**。

    构造手法：把该接收者的 `Δ_content` 与 `Δ_full` 直接置为 ≤0（模拟"信标引路但内容无用"），
    断言 `apply` 不发生、且 `gate_block` 计数 +1。
    """
    e = _engine(gate=True, seed=42)
    # 直接驱动钩子：造一个"有信号 + 有食物 + 归因命中 + 额度充足"的到达
    mid = int(e._id[0])
    e._emit_count[mid] = 100          # 额度充足（100 × EMISSION_COST）
    cell = int(e._flat[0])
    e._last_sender[cell] = mid
    e.signals._marks[cell] = 1
    e.signals._age[cell] = int(e.signals.duration)
    # ⚠️ Δ_i 是**接收者（移动者）**的决策反事实，不是发送者的 —— 写错对象会让测试
    #  "因为读到 0 而通过"（自我检查时发现的口径错位，已修）
    rid = int(e._id[1])
    e._delta_content_by_id[rid] = 0.0                       # 内容**无**贡献
    e._delta_tick_by_id[rid] = e._tick                      # 本 tick 的新鲜值
    before = e._energy.copy()
    e._oracle_after_move(np.array([1], dtype=np.int64), np.array([cell], dtype=np.int64),
                         np.array([True]), np.array([True]), energy=e._energy)
    assert e._diag_gate_block >= 1, "Δ_content ≤ 0 必须被闸门拦下"
    assert np.array_equal(before, e._energy), "被拦下 ⇒ 不得发生任何转账（守恒亦不变）"


def test_gate_polarity_pays_when_delta_content_positive():
    """极性（反向）：`Δ_content > 0` ⇒ 付款发生（且能量守恒、Σ 不变）。"""
    e = _engine(gate=True, seed=42)
    mid = int(e._id[0])
    e._emit_count[mid] = 100
    cell = int(e._flat[0])
    e._last_sender[cell] = mid
    e.signals._marks[cell] = 1
    e.signals._age[cell] = int(e.signals.duration)
    rid = int(e._id[1])
    e._delta_content_by_id[rid] = 0.5
    e._delta_tick_by_id[rid] = e._tick
    before_sum = float(e._energy.sum())
    before = e._energy.copy()
    e._oracle_after_move(np.array([1], dtype=np.int64), np.array([cell], dtype=np.int64),
                         np.array([True]), np.array([True]), energy=e._energy)
    assert e._diag_gate_pass >= 1, "Δ_content > 0 必须放行"
    assert not np.array_equal(before, e._energy), "放行 ⇒ 应发生转账"
    assert abs(float(e._energy.sum()) - before_sum) < 1e-9, "C-2 守恒仍成立"


def test_gate_stale_delta_never_pays():
    """陈旧值防护：上一 tick 的 `Δ>0` **不得**让本 tick 付款（靠 tick 标记，不需复位）。"""
    e = _engine(gate=True, seed=42)
    mid = int(e._id[0])
    e._emit_count[mid] = 100
    cell = int(e._flat[0])
    e._last_sender[cell] = mid
    e.signals._marks[cell] = 1
    e.signals._age[cell] = int(e.signals.duration)
    rid = int(e._id[1])
    e._delta_content_by_id[rid] = 0.9
    e._delta_tick_by_id[rid] = e._tick - 1        # 陈旧
    e._oracle_after_move(np.array([1], dtype=np.int64), np.array([cell], dtype=np.int64),
                         np.array([True]), np.array([True]), energy=e._energy)
    assert e._diag_gate_pass == 0 and e._diag_gate_block >= 1


def test_gate_requires_probe_otherwise_hard_fail():
    """守卫 1：门控 + **探针关** ⇒ ValueError（否则 Δ≡0 ⇒ 付款全消失 = 假结论源）。"""
    cfg = SimConfig(seed=1)
    cfg.simulation.use_sim_core = False
    d2 = InfoStructureConfig(enabled=True)
    d2.measure_signal_response = False
    cfg.info_structure = d2
    cfg.oracle = OracleConfig(enabled=True, donation=1.0, gain_multiplier=1.3,
                              is_calibration_arm=True, gate_mode="delta_positive")
    with pytest.raises(ValueError, match="measure_signal_response"):
        SphereEngine(cfg)


def test_gate_requires_asymmetric_path_and_softmax():
    """守卫 2：门控在**非信息不对称路径**或 `softmax_tau=0` 下 ⇒ ValueError。"""
    with pytest.raises(ValueError, match="信息不对称"):
        _engine(gate=True, radius=8)
    with pytest.raises(ValueError, match="信息不对称"):
        _engine(gate=True, tau=0.0)


def test_gate_requires_calibration_flag_hard_fail():
    """守卫 3：门控臂是仪器 ⇒ 未登记 `is_calibration_arm` ⇒ **OracleConfig 构造期**即失败。"""
    with pytest.raises(AssertionError, match="is_calibration_arm"):
        OracleConfig(enabled=True, donation=1.0, gain_multiplier=1.0,
                     is_calibration_arm=False, gate_mode="delta_positive")


def test_gate_fields_illegal_values_are_rejected():
    with pytest.raises(AssertionError, match="gate_mode"):
        OracleConfig(gate_mode="delta")
    with pytest.raises(AssertionError, match="gate_delta"):
        OracleConfig(gate_delta="both")


def test_gate_off_is_default_and_keeps_mode_none():
    """默认关：`gate_mode="none"` ⇒ stats 报 mode=none、计数 0（旧批次口径不变）。"""
    e = _engine(gate=False, calibration=False, seed=5, oracle=True)
    for _ in range(50):
        if e.extinct:
            break
        e.step()
    g = e.oracle_stats()["gate"]
    assert g["mode"] == "none" and g["arrivals"] == 0 and g["pass"] == 0


def test_gate_funnel_conservation_and_block_frac():
    """漏斗守恒：`pass + block == arrivals`（否则计数漏项）；block_frac 与之自洽。"""
    e = _engine(gate=True, seed=42, max_count=600)
    for _ in range(120):
        if e.extinct:
            break
        e.step()
    g = e.oracle_stats()["gate"]
    assert g["mode"] == "delta_positive" and g["delta_metric"] == "content"
    assert g["pass"] + g["block"] == g["arrivals"]
    if g["arrivals"]:
        assert g["block_frac"] == pytest.approx(g["block"] / g["arrivals"], abs=1e-6)
    fn = e.oracle_funnel()
    assert fn["gate_positive"] == g["pass"] and fn["gate_block"] == g["block"]


def test_gate_snapshot_roundtrip(tmp_path):
    """快照往返：Δ 数组与闸门计数随快照走（旧快照缺键 ⇒ 全零/-1 回退，不错位）。"""
    e = _engine(gate=True, seed=42, max_count=400)
    for _ in range(80):
        if e.extinct:
            break
        e.step()
    p = tmp_path / "gate.npz"
    e.save_snapshot(str(p))
    e2 = SphereEngine.load_snapshot(str(p), config=e.config)
    assert np.array_equal(e._delta_content_by_id, e2._delta_content_by_id)
    assert np.array_equal(e._delta_tick_by_id, e2._delta_tick_by_id)
    assert e2.oracle_stats()["gate"]["pass"] == e.oracle_stats()["gate"]["pass"]
    assert e2._gate_on is True


def test_probe_reports_both_delta_flavors():
    """⑥ 统计同时报 `Δ_full` 与 `Δ_content`（口径并列，便于离线比较两条读数）。"""
    e = _engine(gate=False, calibration=False, seed=42, oracle=False)
    for _ in range(80):
        if e.extinct:
            break
        e.step()
    s = e.signal_response_stats()
    assert "resp_b_delta" in s and "resp_b_content_delta" in s


# -------------------------------------------- 2026-09-19：构念修正 + 可观测性计数器

def test_alphabet8_current_cell_in_memory_now_yields_zero_bit():
    """🔴 **构念修正回归**（内评 2026-09-19 01:4x）：记忆里**只有当前格** ⇒ `mem_bit` 必须为 **0**。

    旧实现（「当前格 ∈ 记忆」）会给出 1 —— 而那只说明"这一格曾经富过"，等于 `f_bit` 的
    时间延迟版，接收者可从当前格直读 ⇒ 冗余。本测试**钉死修正**：改回旧判据即失败。
    """
    common = dict(
        energy=np.array([0.5]), grid=np.zeros(4), capacity=np.ones(4),
        occ=np.ones(4, dtype=np.int64), e_flat=np.array([10], dtype=np.int64),
        emitters=np.array([0], dtype=np.int64), max_energy=10.0,
    )
    only_current = np.full((1, 4), -1, dtype=np.int64); only_current[0, 0] = 10
    else_where = np.full((1, 4), -1, dtype=np.int64); else_where[0, 0] = 77
    empty = np.full((1, 4), -1, dtype=np.int64)

    s_cur, _ = encode_signal_states(alphabet="8", work_memory=only_current, **common)
    s_else, _ = encode_signal_states(alphabet="8", work_memory=else_where, **common)
    s_empty, _ = encode_signal_states(alphabet="8", work_memory=empty, **common)

    assert int(s_cur[0] & 1) == 0, "记忆里只有当前格 ⇒ mem_bit 必须为 0（构念修正）"
    assert int(s_else[0] & 1) == 1, "记忆里有别的格 ⇒ mem_bit=1"
    assert int(s_empty[0] & 1) == 0, "空记忆 ⇒ mem_bit=0"
    assert int(s_cur[0]) == int(s_empty[0]), "「只有当前格」与「空记忆」在修正后**等价**"


def test_alphabet8_mem_bit_frac_counter_end_to_end():
    """`mem_bit_frac` 计数器（闭合 E-022/E-023 记的可观测性缺口）：真跑后必须出数。

    ⚠️ 教训 19（检查工具必须用真实产物验收）：不能只断言"字段存在"，要断言
    `mem_bit_n > 0`（**仪器真的测到了发射**）且 `0 <= frac <= 1`。
    """
    e = _engine(alphabet="8", gate=False, calibration=False, seed=42, oracle=False)
    for _ in range(200):
        if e.extinct:
            break
        e.step()
    a = e.alphabet_stats()
    assert a["mem_bit_n"] > 0, "🔴 仪器没累计到任何状态编码发射 ⇒ 静默失效（教训 19 同型）"
    assert 0.0 <= a["mem_bit_frac"] <= 1.0
    assert a["mem_bit_on"] <= a["mem_bit_n"]
    assert a["mem_bit_frac"] == pytest.approx(a["mem_bit_on"] / a["mem_bit_n"], abs=1e-6)


def test_mem_bit_frac_is_none_not_zero_for_other_alphabets():
    """非 `"8"` 档 ⇒ `mem_bit_frac = None`（**未适用**，不是 0 —— 同 R120 的 ratio n/a 口径）。"""
    for alpha in ("4", "16"):
        e = _engine(alphabet=alpha, gate=False, calibration=False, seed=42, oracle=False)
        for _ in range(30):
            if e.extinct:
                break
            e.step()
        a = e.alphabet_stats()
        assert a["mem_bit_n"] == 0 and a["mem_bit_frac"] is None, (
            f'{alpha} 档不该有 mem_bit 读数（得 {a["mem_bit_frac"]!r}）'
        )


def test_mem_bit_counters_survive_snapshot(tmp_path):
    """计数器须随快照走（续跑后累计不失真）—— 否则长批读数会被静默截断。"""
    e = _engine(alphabet="8", gate=False, calibration=False, seed=7, oracle=False)
    for _ in range(80):
        if e.extinct:
            break
        e.step()
    before = e.alphabet_stats()
    if before["mem_bit_n"] == 0:
        pytest.skip("本 seed 前 80 tick 无状态编码发射 ⇒ 该断言不适用")
    path = tmp_path / "a8_snap.npz"
    e.save_snapshot(str(path))
    b = SphereEngine.load_snapshot(str(path))
    after = b.alphabet_stats()
    assert after["mem_bit_n"] == before["mem_bit_n"]
    assert after["mem_bit_on"] == before["mem_bit_on"]
