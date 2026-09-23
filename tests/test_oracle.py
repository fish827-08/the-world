"""D-8：V-1 oracle 正向对照（R39，利益对齐型 / Lewis 共利）。

对应规格 `_share/规格-V1-oracle引擎级-20260913.md`（v1.2）：
C-2 守恒 / C-3 不为负 / C-9 保本封顶 / O-4 / O-6 / O-7 / 快照双向兼容（C-7）。
实现层两处裁定（已在讨论板登记）：归因键=_id；窗口按"剩余寿命"语义
（`age >= duration - persistence`，规格 §2.3 原式方向反了）。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SIGNAL_COST, InfoStructureConfig, SimConfig  # noqa: E402
from simulation.oracle import EMISSION_COST, apply_oracle, attribution_ok  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


# ---------------- 纯转移核（函数级，C-2/C-3/C-9） ---------------------------

def test_apply_oracle_conserves_energy():
    energy = np.array([10.0, 5.0, 1.0, 100.0])
    before = energy.sum()
    total, cnt, _, _ = apply_oracle(
        energy=energy,
        receiver_slots=np.array([1, 2], dtype=np.int64),
        sender_slots=np.array([0, 0], dtype=np.int64),
        budget=np.array([1.0, 1.0], dtype=np.float64),
        donation=0.05,
    )
    assert cnt == 2
    assert abs(total - 0.1) < 1e-12
    assert abs(energy.sum() - before) < 1e-9          # O-4 守恒
    assert (energy >= 0).all()                         # C-3 不为负


def test_apply_oracle_respects_budget_and_energy():
    """额度耗尽 ⇒ 不转移；**付款方**（接收者）买不起 ⇒ 只转得起的部分。

    F-R18 后（方向 R→S）：偿付能力约束在**接收者**（付款方）身上；
    故"能量卡住"的 fixture 须让**接收者**穷，而不是发送者。
    """
    energy = np.array([5.0, 0.02])                     # [0]=发送者(收款) [1]=接收者(付款，仅 0.02)
    total, cnt, _, _ = apply_oracle(
        energy=energy,
        receiver_slots=np.array([1], dtype=np.int64),
        sender_slots=np.array([0], dtype=np.int64),
        budget=np.array([1.0], dtype=np.float64),
        donation=0.05,
    )
    assert cnt == 1 and abs(total - 0.02) < 1e-12      # 被**付款方**能量卡住
    assert abs(energy[1]) < 1e-12                      # 付款方付光

    energy2 = np.array([10.0, 5.0])
    total2, cnt2, _, _ = apply_oracle(
        energy=energy2,
        receiver_slots=np.array([1], dtype=np.int64),
        sender_slots=np.array([0], dtype=np.int64),
        budget=np.array([0.01], dtype=np.float64),     # 被预算卡住（C-9）
        donation=0.05,
    )
    assert cnt2 == 1 and abs(total2 - 0.01) < 1e-12

    total3, cnt3, _, _ = apply_oracle(
        energy=energy2,
        receiver_slots=np.array([1], dtype=np.int64),
        sender_slots=np.array([0], dtype=np.int64),
        budget=np.array([0.0], dtype=np.float64),
        donation=0.05,
    )
    assert cnt3 == 0 and total3 == 0.0


def test_attribution_window_remaining_lifetime_semantics():
    """SignalField._age 是剩余寿命（写入=duration，每 tick −1）。

    "最近 persistence tick 内写入" ⇔ age >= duration − persistence。
    规格 §2.3 的 `age <= persistence` 按此语义方向相反（已在讨论板登记更正）。
    """
    age = np.array([50, 45, 41, 40, 39, 20, 0], dtype=np.int32)
    ok = attribution_ok(age, signal_duration=50, persistence=10)
    assert ok.tolist() == [True, True, True, True, False, False, False]
    ok0 = attribution_ok(age, signal_duration=50, persistence=0)   # 仅本 tick
    assert ok0.tolist() == [True, False, False, False, False, False, False]


# ---------------- 引擎级（O-6 / O-7 / C-7 / C-8） ---------------------------

def _engine(oracle: bool, seed: int = 42) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False
    d2 = InfoStructureConfig(enabled=True, learning_rate=0.05)
    cfg.info_structure = d2
    cfg.oracle.enabled = oracle
    if oracle:
        # C5（2026-09-15）：启用 oracle 时 donation 必须 >= SIGNAL_COST，否则规格自相矛盾
        # （D-24 的 G-A 不过是这个原因）。测试助手也要给自洽档。
        cfg.oracle.donation = SIGNAL_COST
    return SphereEngine(cfg)


def _run_recording(e: SphereEngine, ticks: int) -> list[int]:
    draws = []
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
        draws.append(e.rng_draws)
    return draws


def test_o6_same_tick_rng_draws_until_first_transfer():
    """O-6：oracle 开/关两臂，在首次转移发生前每 tick 的 rng_draws 逐位相同
    （零新增 RNG 调用点 C-1 的可执行化）。首次转移所在 tick 起允许分岔（C-1′ 实验效应）。

    ⚠️ F-D2：全局 np.random 进程级共享 ⇒ 先跑完 a 再构造 b（见 test_d18 同款注释）。
    """
    a = _engine(oracle=True)
    da: list[int] = []
    fired_at = None
    for i in range(400):
        if a.extinct:
            break
        a.step()
        da.append(a.rng_draws)
        if fired_at is None and a.oracle_stats()["count"] > 0:
            fired_at = i + 1
    b = _engine(oracle=False)          # 重新 seed 全局 np.random ⇒ 与 a 起点一致
    db: list[int] = []
    for _ in range(len(da)):
        if b.extinct:
            break
        b.step()
        db.append(b.rng_draws)
    if fired_at is None:
        assert da == db, "oracle 未触发时随机流必须完全一致"
        return
    assert da[: fired_at - 1] == db[: fired_at - 1], (
        "首次转移前每 tick 的 RNG 消费必须逐位相同（C-1）"
    )


def test_o7_return_ratio_bounded_by_one():
    """O-7 / C-9：保本封顶 ⇒ return_ratio ≤ 1（由构造保证，实测复核）。"""
    e = _engine(oracle=True)
    for _ in range(400):
        if e.extinct:
            break
        e.step()
    s = e.oracle_stats()
    assert s["enabled"] is True
    assert s["oracle_return_ratio"] <= 1.0 + 1e-9
    assert s["transfers"] >= 0.0


def test_c7_snapshot_roundtrip_keeps_oracle_state():
    """C-7：_last_sender / 保本记账必须进快照；续跑后 oracle 统计与连续跑一致。

    ⚠️ F-D2：引擎快照不含全局 np.random 状态（已知缺陷）⇒ 测试按生产侧
    （a4_verify_capacity 的 rngstate.pkl）同款模式补存取，否则续跑必然分岔。
    """
    import pickle

    a = _engine(oracle=True)
    for _ in range(60):
        a.step()
        if a.extinct:
            break
    path = Path(__file__).parent / "_tmp_oracle_snap.npz"
    rng_state_path = path.with_suffix(".rngstate.pkl")
    a.save_snapshot(str(path))
    with open(rng_state_path, "wb") as fh:
        pickle.dump(np.random.get_state(), fh)          # F-D2 补偿
    np_state_before_load = np.random.get_state()
    b = SphereEngine.load_snapshot(str(path))
    with open(rng_state_path, "rb") as fh:
        np.random.set_state(pickle.load(fh))            # 恢复全局态
    del np_state_before_load
    assert np.array_equal(b._last_sender, a._last_sender)
    assert np.array_equal(b._emit_count, a._emit_count)
    assert np.allclose(b._oracle_gain, a._oracle_gain)
    assert b.oracle_stats()["count"] == a.oracle_stats()["count"]
    # 续跑各 40 tick：oracle 记账仍一致。
    # ⚠️ 全局 np.random 是进程级单例 ⇒ 两条续跑【不能交错 step】，须各自
    # 从分叉点状态独立重放（先 a 后 b，每次续跑前恢复分叉点全局态）。
    with open(rng_state_path, "rb") as fh:
        np.random.set_state(pickle.load(fh))
    for _ in range(40):
        if a.extinct:
            break
        a.step()
    a_count = a.oracle_stats()["count"]
    with open(rng_state_path, "rb") as fh:
        np.random.set_state(pickle.load(fh))
    for _ in range(40):
        if b.extinct:
            break
        b.step()
    assert b.oracle_stats()["count"] == a_count
    path.unlink(missing_ok=True)
    rng_state_path.unlink(missing_ok=True)


def test_c8_sim_core_plus_oracle_raises():
    """C-8：use_sim_core=True + oracle.enabled=True ⇒ 引擎初始化显式报错。"""
    try:
        import sim_core  # noqa: F401
        has_core = True
    except ImportError:
        has_core = False
    if not has_core:
        pytest.skip("sim_core 未安装（本机无 Rust 扩展）——C-8 在导入处已报错")
    cfg = SimConfig(seed=1)
    cfg.simulation.use_sim_core = True
    cfg.oracle.enabled = True
    with pytest.raises(RuntimeError, match="C-8"):
        SphereEngine(cfg)


def test_oracle_config_dict_fallback():
    """C-7：旧存档缺 oracle 键 ⇒ 回退默认关闭。"""
    d = SimConfig(seed=1).to_dict()
    d.pop("oracle", None)
    c2 = SimConfig.from_dict(d)
    assert c2.oracle.enabled is False
    assert c2.oracle.donation == 0.05          # 默认档（关闭态 ⇒ C5 不触发）
    # fingerprint 含 oracle（批次自证是否开了 oracle —— 规格 §七.4）
    assert "oracle" in SimConfig(seed=1).fingerprint()
    assert EMISSION_COST == 0.1


# ---------- F-R16 / F-R18（2026-09-15）：封顶顺序累加 + **方向极性** 的回归 ----------
# F-R16：额度必须**顺序累加**（原实现每对都用同一 budget 快照 ⇒ 同一收款者可多次满额）
# F-R18：方向 = **R→S**（接收者付款、发送者收款）。原实现写反（发送者倒贴）⇒
#        仪器实际"惩罚发射者" ⇒ 对 g15 是负选择 ⇒ 原理上无法达成"oracle 开 ⇒ g15 上升"。
#        R102 裁定 (A) 修方向；并立**新纪律**：仪器类通道必须有**端到端极性测试**
#        （构造成功事件 ⇒ 断言目标量向**预期方向**变化）。

def test_f_r18_direction_sender_receives_receiver_pays():
    """① 方向断言：成功转移 ⇒ 发送者（收款）增加、接收者（付款）减少。"""
    e = np.array([10.0, 10.0, 10.0])
    total, cnt, kept_s, kept_g = apply_oracle(
        energy=e,
        receiver_slots=np.array([1], dtype=np.int64),   # 付款方 = 接收者（移动者）
        sender_slots=np.array([0], dtype=np.int64),     # 收款方 = 发送者（信号写入者）
        budget=np.array([1.0], dtype=np.float64), donation=0.5,
    )
    assert cnt == 1 and abs(total - 0.5) < 1e-12
    assert e[0] == pytest.approx(10.5), "发送者必须**收款**（F-R18）"
    assert e[1] == pytest.approx(9.5), "接收者必须**付款**（F-R18）"
    assert e[2] == pytest.approx(10.0), "无关个体不受影响"
    assert list(kept_s) == [0], "返回的成交槽位必须是**收款者**（发送者）"


def test_f_r18_conservation_is_pure_redistribution():
    """② 守恒断言：Σenergy 恒等 ⇒ **纯再分配、零注入**（C-2 成立）。"""
    rng = np.random.default_rng(0)
    for _ in range(30):
        e = rng.uniform(0.0, 5.0, size=6)
        before = float(e.sum())
        apply_oracle(
            energy=e,
            receiver_slots=rng.integers(0, 6, 4),
            sender_slots=rng.integers(0, 6, 4),
            budget=rng.uniform(0.0, 0.5, 4),
            donation=0.05,
        )
        assert e.sum() == pytest.approx(before, abs=1e-12)
        assert (e >= 0).all()


def test_f_r16_budget_accumulates_sequentially_per_recipient():
    """③ 封顶压力（F-R16，角色对调后重推）：同一发送者（收款方）多笔 ⇒ 额度**顺序累加**。

    `[实测 原缺陷]` budget=0.06、donation=0.05、同一发送者 2 笔 ⇒ 旧实现实付 0.10 > 0.06。
    """
    e = np.array([0.0, 10.0, 10.0])        # [0]=发送者(收款) [1][2]=接收者(付款)
    total, cnt, _, _ = apply_oracle(
        energy=e,
        receiver_slots=np.array([1, 2], dtype=np.int64),
        sender_slots=np.array([0, 0], dtype=np.int64),
        budget=np.array([0.06, 0.06], dtype=np.float64),
        donation=0.05,
    )
    assert total <= 0.06 + 1e-12, f"C-9 被突破：收款 {total} > 额度 0.06"
    assert cnt == 2                           # 第二笔为**部分**收款（0.01）
    assert e[0] == pytest.approx(0.06)
    assert e.sum() == pytest.approx(20.0, abs=1e-12)


def test_f_r16_budget_exhausted_skips_further_transfers():
    """③ 续：额度被第一笔吃满 ⇒ 后续**跳过**（不再收一次满额）。"""
    e = np.array([0.0, 10.0, 10.0])
    total, cnt, _, _ = apply_oracle(
        energy=e,
        receiver_slots=np.array([1, 2], dtype=np.int64),
        sender_slots=np.array([0, 0], dtype=np.int64),
        budget=np.array([1.0, 1.0], dtype=np.float64),
        donation=1.0,
    )
    assert cnt == 1 and total == pytest.approx(1.0)


def test_f_r18_payer_insolvency_truncates_then_skips():
    """④ 付款方（接收者）偿付能力：不足 ⇒ **部分**付款；为 0 ⇒ 跳过。"""
    e = np.array([0.0, 0.02])                 # 接收者仅 0.02 < donation 0.05
    total, cnt, _, _ = apply_oracle(
        energy=e, receiver_slots=np.array([1], dtype=np.int64),
        sender_slots=np.array([0], dtype=np.int64),
        budget=np.array([1.0], dtype=np.float64), donation=0.05,
    )
    assert cnt == 1 and total == pytest.approx(0.02)
    assert e[1] == pytest.approx(0.0) and e[0] == pytest.approx(0.02)

    e2 = np.array([0.0, 0.0])                 # 接收者无能量 ⇒ 不转移
    t2, c2, _, _ = apply_oracle(
        energy=e2, receiver_slots=np.array([1], dtype=np.int64),
        sender_slots=np.array([0], dtype=np.int64),
        budget=np.array([1.0], dtype=np.float64), donation=0.05,
    )
    assert c2 == 0 and t2 == 0.0
    assert (e2 >= 0).all()


def test_f_r18_end_to_end_polarity_emitter_gains():
    """⑤ **端到端极性测试**（R102 新纪律）：构造一次"成功通信"⇒ 信号写入者净能量**上升**。

    构造（确定性，不依赖生态噪声）：1 号个体 = 信号写入者（id 11，有发射史 ⇒ 有额度）；
    2 号个体 = 接收者（id 22，落到"被 11 标记过且有食物"的格）。
    断言：写入者能量 **+g**、接收者 **−g**。
    这条测试**正是** F-R18 的"反向最小实验"：方向写反时它必失败。
    """
    cfg = SimConfig(seed=1)
    cfg.simulation.use_sim_core = False
    cfg.oracle.enabled = True
    cfg.oracle.donation = 0.5
    cfg.oracle.persistence = 0                 # 仅本 tick：写入即刻有效
    e = SphereEngine(cfg)

    cell = 500
    e._id = np.array([11, 22], dtype=np.int64)          # 槽位 0=写入者 / 1=接收者
    e._flat = np.array([cell, cell + 1], dtype=np.int64)
    e._energy = np.array([10.0, 10.0], dtype=np.float64)
    e._emit_count = np.zeros(23, dtype=np.int32)
    e._emit_count[11] = 10                              # 额度 = 0.1×10 = 1.0
    e._oracle_gain = np.zeros(23, dtype=np.float32)
    e._last_sender[cell] = 11                           # 该格由 id 11 写入
    e.signals._marks[cell] = 1
    e.signals._age[cell] = int(e.signals.duration)       # 刚写入 ⇒ 在归因窗口内

    before = e._energy.copy()
    e._oracle_after_move(
        mi=np.array([1], dtype=np.int64),                # 接收者 = 槽位 1（id 22）
        target_cells=np.array([cell], dtype=np.int64),
        had_signal=np.array([True]),
        true_sig=np.array([True]),
        energy=e._energy,
    )
    g = 0.5
    assert e._energy[0] == pytest.approx(before[0] + g), (
        "极性失败：信号**写入者**必须收款（F-R18；写反时这里会减少）"
    )
    assert e._energy[1] == pytest.approx(before[1] - g), "接收者必须付款"
    assert e._energy.sum() == pytest.approx(before.sum(), abs=1e-9), "C-2 守恒"


def test_r102_conservation_audit_three_accounts():
    """R102 条件 1（修正版）：**守恒三账审计** —— 引擎侧实测 Σenergy 变化恒 0。

    撤销原 R100 条件 1 的 `system_energy_injected`（机制复核已证"净注入"不成立：
    `apply_oracle` 是双向转移 ⇒ 纯再分配）。本审计**独立**核算（包住调用实测变化），
    不是"同一个数抄三遍" ⇒ 能**证伪** C-2。
    """
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = 300
    cfg.info_structure = InfoStructureConfig(enabled=True, learning_rate=0.05)
    cfg.oracle.enabled = True
    cfg.oracle.donation = 5.0                  # 大额放大任何注入
    e = SphereEngine(cfg)
    for _ in range(200):
        if e.extinct:
            break
        e.step()
    a = e.oracle_audit()
    assert a["calls"] > 0, "应至少跑过若干次 oracle 挂钩"
    assert a["sum_energy_delta"] == 0.0, f"C-2 被破坏：ΔΣenergy={a['sum_energy_delta']}"
    assert a["conserved"] is True
    assert e.oracle_stats()["audit"]["conserved"] is True   # 经 stats 也能读到


# ---------------- R102 条件 1/3 + 内评《复核-R102方向修复》§三/§四 ----------------
# 对账字段（payer 侧分解 + 偿付约束量化）与**接收侧效应**（方向翻转新引入的代价）。

def test_oracle_ledger_identity_and_payer_truncation():
    """对账恒等 + **付款方偿付截断**（内评 §四：payer_insolvent_truncated）。

    donation=0.05、付款方仅 0.02 ⇒ 部分支付 0.02、截断 0.03。
    """
    e = np.array([0.0, 0.02])
    led: dict = {}
    total, cnt, _, _ = apply_oracle(
        energy=e, receiver_slots=np.array([1], dtype=np.int64),
        sender_slots=np.array([0], dtype=np.int64),
        budget=np.array([1.0], dtype=np.float64), donation=0.05,
        ledger=led,
    )
    assert cnt == 1 and total == pytest.approx(0.02)
    assert led["payer_paid"] == pytest.approx(led["sender_received"]) == pytest.approx(0.02)
    assert led["payer_trunc_n"] == 1
    assert led["payer_trunc_amt"] == pytest.approx(0.03)
    assert led["budget_trunc_n"] == 0


def test_oracle_ledger_counts_payer_broke_and_budget_exhausted():
    """两类整笔跳过必须**分开计数**（否则"接收者付不起"被混进"额度用尽"）。"""
    e = np.array([0.0, 0.0, 5.0])
    led: dict = {}
    total, cnt, _, _ = apply_oracle(
        energy=e, receiver_slots=np.array([1], dtype=np.int64),
        sender_slots=np.array([0], dtype=np.int64),
        budget=np.array([1.0], dtype=np.float64), donation=0.05, ledger=led,
    )
    assert cnt == 0 and total == 0.0
    assert led["payer_broke_n"] == 1 and led["budget_exhausted_n"] == 0

    led2: dict = {}
    apply_oracle(
        energy=e, receiver_slots=np.array([2], dtype=np.int64),
        sender_slots=np.array([0], dtype=np.int64),
        budget=np.array([0.0], dtype=np.float64), donation=0.05, ledger=led2,
    )
    assert led2["budget_exhausted_n"] == 1 and led2["payer_broke_n"] == 0


def test_oracle_ledger_engine_accumulates_and_identity_holds():
    """引擎侧累计：三账恒等 + `sum_transfers == payer_paid`（对账字段可用）。"""
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = 400
    cfg.info_structure = InfoStructureConfig(enabled=True, learning_rate=0.05)
    cfg.oracle.enabled = True
    cfg.oracle.donation = 2.0          # 高剂量 ⇒ 大量偿付截断（付款方 2.0 很贵）
    e = SphereEngine(cfg)
    for _ in range(300):
        if e.extinct:
            break
        e.step()
    led = e.oracle_ledger()
    assert led["sum_transfers"] > 0, "该配置下应发生过转移"
    assert led["identity_ok"] is True
    assert led["payer_paid"] == pytest.approx(led["sender_received"])
    assert led["payer_paid"] == pytest.approx(led["sum_transfers"], abs=1e-6)
    assert led["role"].startswith("对账")


def test_receiver_side_effects_actually_produces_data():
    """🔴 教训 19：检查工具必须用**真实产物**验收——不能只断言"字段存在"。

    要求：跑真实引擎后 ① 字段齐全 ② `food_events > 0`（**仪器真的测到了**落点摄入）
    ③ 净额字段 == 摄入 − 付款（自洽）。
    """
    cfg = SimConfig(seed=7)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = 400
    cfg.info_structure = InfoStructureConfig(enabled=True, learning_rate=0.05)
    cfg.oracle.enabled = True
    cfg.oracle.donation = 1.0
    e = SphereEngine(cfg)
    for _ in range(400):
        if e.extinct:
            break
        e.step()
    rs = e.oracle_stats()["receiver_side"]
    for k in ("events", "food_events", "mean_intake_at_paid_cell",
              "mean_intake_all_eaters", "mean_payment_per_event",
              "net_eat_minus_pay", "net_vs_baseline", "caveat"):
        assert k in rs, f"缺字段 {k}"
    assert rs["events"] > 0, "应发生过转移"
    assert rs["food_events"] > 0, "🔴 仪器没测到落点摄入 ⇒ 静默失效（教训 19 同型）"
    assert rs["mean_intake_all_eaters"] > 0
    assert rs["net_eat_minus_pay"] == pytest.approx(
        rs["mean_intake_at_paid_cell"] - rs["mean_payment_per_event"], abs=1e-4
    )
    # ⚠️ 2026-09-23（T2）：容差 1e-9 → 1e-4 —— eat_amount 0.5→0.9 构造变更使
    #   摄入/付款的浮点路径出现 ~1e-6 级差（实测 0.103072 vs 0.103073）。断言是
    #   "净额 = 摄入 − 付款"的自洽性，1e-4 仍远超浮点噪声、远低于任何真实不平衡
    #   （机械守恒另由 energy_ledger 审计覆盖，量级 1e-12）。
    # 落点暂存：t 付款、t+1 进食时才消费 ⇒ **最后一步之后残留是正常**；
    # 但必须**有界**（不会跨 tick 累积泄漏）——上限 = 该 tick 的成交笔数 ≤ 种群规模。
    assert len(e._recv_pend_cells) <= max(1, len(e._id)), (
        f"落点暂存泄漏：{len(e._recv_pend_cells)} > 种群 {len(e._id)}"
    )
    e.step()                      # 再走一步 ⇒ 上一批必须被消费掉（不叠加）
    assert len(e._recv_pend_cells) <= max(1, len(e._id))
