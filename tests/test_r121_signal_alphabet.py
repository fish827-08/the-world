"""R113 / R121 信号字母表（16 → 4）与 §4 前置改动的测试。

设计稿：`docs/设计文档/设计-信号字母表简化与记忆进信号-20260917.md` §3.6（9 项测试计划）
R121 核阅（2026-09-18）重点强调：
  · §3.4 `codebook_convergence` **必须传 `n_states`**（漏传 ⇒ 收敛度系统性虚高）
  · §3.5 `state=0` 断言 —— "4" 档下**码 0 出现次数必须为 0**（隐形发射 + 清除他人标记）
  · 内评 09-17 §三.4 追加：`state=0` 要落成**断言**（不是注释）+ 回归测试

覆盖：档位配置 / 码域 / 冗余位删除 / 码本值域 / random 档 / **RNG 不变性** /
      快照跨档硬报错 / 指标防虚高 / 守卫正反向 / §4 两个零机时 counter。
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from observatory.statistics import codebook_convergence
from simulation.config import (
    FOOD_RICH_LEVEL,
    SIGNAL_ALPHABET_CODE_MAX,
    SIGNAL_ALPHABET_STATES,
    InfoStructureConfig,
    SimConfig,
)
from simulation.sphere_engine import (
    SphereEngine,
    codebook_init_rows,
    encode_signal_states,
)

C1A_REF = os.path.join("_rerun_logs", "cstep1a", "oracle_m1.3_s42.csv")


def _make_engine(alphabet: str = "16", *, mode: str = "state",
                 codebook: bool = True, pop: int = 50, seed: int = 42,
                 oracle: bool = False, **d2_kwargs) -> SphereEngine:
    """小引擎（Python 路径：D2 启用时 Rust 未下沉，AGENTS.md 硬约束）。"""
    cfg = SimConfig(seed=seed)
    cfg.signal_alphabet = alphabet
    cfg.signal_mode = mode
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_bottleneck=True, arbitrary_codebook=codebook,
        steels_alignment=True, **d2_kwargs,
    )
    cfg.simulation.use_sim_core = False
    cfg.population.initial_count = pop
    cfg.population.max_count = max(pop, 500)
    if oracle:
        cfg.oracle.enabled = True
        cfg.oracle.donation = 1.0
        cfg.oracle.gain_multiplier = 1.3
        cfg.oracle.is_calibration_arm = True
    return SphereEngine(cfg)


def _active_codes(e: SphereEngine) -> np.ndarray:
    """当前**有效**（未过期）标记的码值。"""
    marks = np.asarray(e.signals._marks)
    age = np.asarray(e.signals._age)
    return marks[age > 0]


# ─── 1. C7 逐位对拍（默认 "16" 必须与 C1a 旧代码一致）───────────────────

@pytest.mark.skipif(os.environ.get("R121_PARITY") != "1",
                    reason="慢测（≈数分钟）：置 R121_PARITY=1 启用")
@pytest.mark.skipif(not os.path.exists(C1A_REF), reason="C1a 参考批数据不在本机")
def test_c7_parity_16_default_matches_c1a_reference():
    """默认 `"16"` 档＋同 seed＋同配置 ⇒ 与 C1a 批（旧代码）**逐位一致**。

    这是"本次改动纯观测、零行为漂移"的最强证据：任何语义漂移都会让**首个采样点**（tick
    1000）就对不上。参考 = `_rerun_logs/cstep1a/oracle_m1.3_s42.csv`（配置见
    `batch_runner.py:317-337`；引擎子树 `git diff a672637 2e1f0c7 -- simulation/` 为空）。
    """
    from experiments.a4_verify_capacity import build

    e = build("on", True, 42, 60000, max_count=3240, oracle=True, measure=True,
              oracle_donation=1.0, gain_multiplier=1.3, calibration_arm=True)
    got = []
    for t in range(1, 2001):
        e.step()
        if t % 1000 == 0:
            got.append((t, len(e._id), round(float(e._genes[:, 15].mean()), 4)))
    ref = {}
    with open(C1A_REF, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ref[int(r["tick"])] = (int(r["N"]), float(r["g15"]))
    assert got, "未采到任何样本点"
    for t, n, g15 in got:
        assert t in ref, f"参考表缺 tick={t}"
        assert n == ref[t][0], f"tick={t} 末态 N 不一致：新={n} 旧={ref[t][0]}"
        assert abs(g15 - ref[t][1]) < 1e-9, f"tick={t} g15 不一致：{g15} vs {ref[t][1]}"


# ─── 2. "4" 码域 + 零码零容忍 ───────────────────────────────────────────

def test_alphabet4_code_domain_and_zero_code():
    """"4" 档（无码本）⇒ 码 ⊆ {1,2,3,4}，且**有效标记里不许出现 0**（设计稿 §3.5）。"""
    e = _make_engine("4", codebook=False)
    seen = set()
    for _ in range(30):
        e.step()
        codes = _active_codes(e)
        if codes.size:
            seen.update(codes.tolist())
            assert codes.min() >= 1, "码 0 出现 ⇒ 隐形发射/清除他人标记（§2.4）"
            assert codes.max() <= 4, "码越上限 ⇒ 接收端读到未学过的槽"
    assert seen, "整个测试期没有任何有效发射（样本不足，断言无效）"
    assert e.alphabet_stats()["bad_code_n"] == 0


# ─── 3. "4" 档确已删除冗余位（f_bit / n_bit）───────────────────────────

def test_alphabet4_drops_food_and_neighbor_bits():
    """同 `e_bin`、不同 (食物, 邻居) ⇒ "4" 下**必须同码**；"16" 下必须不同码。

    这条直接把"删了两位"变成可证伪断言（否则 16→4 只是"更干净的冗余"）。
    """
    en = np.array([5.0, 9.0])
    ef = np.array([0, 1])
    em = np.array([0, 1])
    occ = np.array([3, 3])                       # 邻居位 = 1（occ>1）
    cap = np.array([1.0, 1.0])
    rich = np.array([1.0, 1.0])                  # 食物位 = 1
    poor = np.array([0.0, 0.0])                  # 食物位 = 0

    s16_rich, _ = encode_signal_states(en, rich, cap, occ, ef, em, 10.0, "16")
    s16_poor, _ = encode_signal_states(en, poor, cap, occ, ef, em, 10.0, "16")
    assert not np.array_equal(s16_rich, s16_poor), '"16" 下食物位应当影响状态'

    s4_rich, off4 = encode_signal_states(en, rich, cap, occ, ef, em, 10.0, "4")
    s4_poor, _ = encode_signal_states(en, poor, cap, occ, ef, em, 10.0, "4")
    assert off4 == 1, '"4" 档的 code_offset 必须为 1（code=e_bin+1 避开 0）'
    assert np.array_equal(s4_rich, s4_poor), '"4" 下食物位必须已删（不应影响状态）'
    assert s4_rich.min() >= 0 and s4_rich.max() <= 3, '"4" 的 state 必须 ∈ 0..3'

    occ2 = np.array([1, 1])                      # 邻居位 = 0
    s4_few, _ = encode_signal_states(en, rich, cap, occ2, ef, em, 10.0, "4")
    assert np.array_equal(s4_rich, s4_few), '"4" 下邻居位必须已删'


# ─── 4. 码本变异 / 对齐后值域仍受控 ────────────────────────────────────

def test_alphabet4_codebook_domain_after_mutation():
    """码本被强制全位突变 + Steels 对齐后，有效槽值域仍 ⊆ [1,4]。"""
    e = _make_engine("4", codebook=True)
    e.config.info_structure.codebook_mutation_rate = 1.0     # 强制每位都突
    e.config.info_structure.alignment_step = 1.0             # 强制对齐全位复制
    for _ in range(80):
        e.step()
    cb = np.asarray(e._codebook[: len(e._id)])
    assert cb.min() >= 1, f"码本出现 0（{cb.min()}）⇒ 发射会写 0 码"
    assert cb.max() <= 4, f"码本越上限（{cb.max()}）⇒ 码域被破坏"
    assert e.alphabet_stats()["bad_code_n"] == 0


# ─── 5. random 档码域 ──────────────────────────────────────────────────

def test_alphabet4_random_mode_domain():
    """`signal_mode="random"` 在 "4" 档下必须只产 1–4（组合守卫，设计稿 §3.5）。"""
    e = _make_engine("4", mode="random")
    for _ in range(20):
        e.step()
    codes = _active_codes(e)
    assert codes.size, "random 档没有发射（样本不足）"
    assert codes.min() >= 1 and codes.max() <= 4


# ─── 6. RNG 不变性（切档不得改变随机流消费）────────────────────────────

class _DrawRecorder:
    """记录每次抽取的（方法名, 参数 repr）——用于比对"抽取形状"是否被切档改变。"""

    def __init__(self, gen):
        self._gen = gen
        self.log: list[tuple[str, str]] = []

    def __getattr__(self, name):
        attr = getattr(self._gen, name)
        if not callable(attr):
            return attr

        def wrapper(*a, **k):
            self.log.append((name, repr((a, tuple(sorted(k.items()))))[:100]))
            return attr(*a, **k)

        return wrapper


def _fresh(alphabet: str, *, mode: str = "state", record: bool = False):
    """新建引擎。

    🔴 **必须"顺序运行、不得交错"**（2026-09-18 实测发现）：`SphereEngine.__init__`
    会 `np.random.seed(config.seed)`（`sphere_engine.py:284`），而 D2 感知噪声
    （`:1038-1039`）与英雄洗牌（`:1197/:1267`）用的是**全局 `np.random`** ⇒ 同进程内
    `a.step(); b.step()` 交错会互相偷走对方的全局随机数（实测第 21 tick 即发散）。
    这也解释了 runner 为何要 pickle `np.random.get_state()`（F-D2）。
    """
    e = _make_engine(alphabet, mode=mode)
    rec = None
    if record:
        rec = _DrawRecorder(e.rng._gen)
        e.rng._gen = rec
    return e, rec


def _run_ticks(e: SphereEngine, n: int) -> list[int]:
    out = []
    for _ in range(n):
        e.step()
        out.append(e.rng_draws)
    return out


def test_rng_shape_preserved_by_alphabet_switch():
    """§3.3 的**正确口径**（🔴 本测试初稿写错过，见下）：

    设计稿 §3.3 主张的是「**抽取形状不变**」，**不是**"整批轨迹逐位相同"。切档后**总消费数
    必然在下游发散** —— 码语义变了 ⇒ 接收者决策变了 ⇒ **逐个体循环**（邻格觅食
    `rng.integers(0, len(nb))`，`:714-716`）的次数随之变。`[实测]` 第 1 tick 的形状序列
    完全一致；总消费数随后发散（那是**改动的目的**，不是缺陷）。

    初稿断言"逐 tick 全等" ⇒ 与设计意图不符（既不可能、也无意义）。
    本测改为断言：第 1 tick 的**抽取方法序列 + 参数形状逐项相同**（这才是 §3.3 要防的
    "RNG 错位"——它会让 C7 对拍失去诊断力）。
    """
    a, ra = _fresh("16", record=True)
    a.step()
    b, rb = _fresh("4", record=True)      # 构造会重设全局种子 ⇒ 先跑完 a 再建 b
    b.step()

    assert ra.log and rb.log, "第 1 tick 未记录到任何抽取（测试失效）"
    assert [m for m, _ in ra.log] == [m for m, _ in rb.log], (
        f"第 1 tick 抽取方法序列被改变：\n16→{[m for m, _ in ra.log]}\n4→{[m for m, _ in rb.log]}"
    )
    assert [s for _, s in ra.log] == [s for _, s in rb.log], "第 1 tick 抽取形状被改变"


def test_alphabet_switch_actually_changes_behaviour():
    """切档必须在**行为上**生效（防"开关传了没生效"型静默失败，F-R21/C5 同族）。

    断言两条：① 40 tick 后两侧总消费数**不同**；② 两侧消费数各自**单调不减**。
    """
    a, _ = _fresh("16")
    da = _run_ticks(a, 40)
    b, _ = _fresh("4")
    db = _run_ticks(b, 40)
    assert all(x <= y for x, y in zip(da, da[1:])), "16 档消费数出现回退"
    assert all(x <= y for x, y in zip(db, db[1:])), "4 档消费数出现回退"
    assert da[-1] != db[-1], (
        f"40 tick 后两侧消费数仍相同（{da[-1]}）⇒ 切档没有真正改变行为（改动被静默吞掉）"
    )


def test_same_alphabet_sequential_runs_are_deterministic():
    """对照基线：**同档、顺序运行**两次必须完全一致（否则上面的比较没有基准）。

    ⚠️ 必须顺序（见 `_fresh` 注释）：交错步进会因全局 RNG 被互相污染而分歧——那是**测试
    写法**问题，不是引擎缺陷。
    """
    a, _ = _fresh("16")
    da = _run_ticks(a, 40)
    b, _ = _fresh("16")
    db = _run_ticks(b, 40)
    assert da == db, "同档顺序运行的消费序列不一致 ⇒ 存在未受控的随机源"



# ─── 7. 快照：同档可续、跨档硬报错 ─────────────────────────────────────

def test_snapshot_cross_alphabet_hard_fails(tmp_path):
    e = _make_engine("16")
    for _ in range(5):
        e.step()
    p = tmp_path / "s.npz"
    e.save_snapshot(str(p))

    same = SphereEngine.load_snapshot(str(p))
    assert same.config.signal_alphabet == "16", "同档续跑应当成功"

    cfg4 = SimConfig(seed=42)
    cfg4.signal_alphabet = "4"
    with pytest.raises(ValueError, match="指纹"):
        SphereEngine.load_snapshot(str(p), config=cfg4)


# ─── 8. 指标口径：漏传 n_states 会虚高（构造反例）──────────────────────

def test_convergence_n_states_prevents_inflation():
    """构造反例证明 §3.4 的必要性：未用槽恒为初值 ⇒ 16 口径**系统性虚高**。"""
    cb = codebook_init_rows(10, "4")          # 有效槽 0–3 = {1,2,3,4}；槽 4–15 恒 4
    for i in range(10):
        cb[i, 0] = 1 + (i % 4)                # 只在前 4 槽制造分歧
    conv4 = codebook_convergence(cb, n_states=4)
    conv16 = codebook_convergence(cb, n_states=16)
    assert conv4 < conv16, "反例应当体现 16 口径虚高"
    assert 0.0 <= conv4 <= 1.0 and 0.0 <= conv16 <= 1.0
    assert codebook_convergence(cb[:0]) == 0.0, "空码本必须返回 0.0"
    assert codebook_convergence(None) == 0.0


def test_convergence_default_matches_legacy_16():
    """默认 `n_states=16` ⇒ 与旧口径逐位相同（已判结果不回改）。"""
    rng = np.random.default_rng(0)
    cb = rng.integers(0, 16, size=(20, 16)).astype(np.uint8)
    assert codebook_convergence(cb) == codebook_convergence(cb, n_states=16)


# ─── 9. 守卫：正反向各一 ───────────────────────────────────────────────

def test_alphabet_guards_positive_and_negative():
    # 反向①：未实施档位 ⇒ 构造期硬失败（不静默降级）
    cfg8 = SimConfig(seed=1)
    cfg8.signal_alphabet = "8"
    cfg8.simulation.use_sim_core = False
    with pytest.raises(NotImplementedError):
        SphereEngine(cfg8)
    with pytest.raises(NotImplementedError):
        codebook_init_rows(3, "8")
    with pytest.raises(NotImplementedError):
        encode_signal_states(np.ones(1), np.ones(1), np.ones(1), np.ones(1, dtype=np.int64),
                             np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64),
                             10.0, "8")

    # 正向①："4" 档 bad_code_n 恒 0 且守卫标记为"期望为 0"
    e4 = _make_engine("4")
    for _ in range(20):
        e4.step()
    s4 = e4.alphabet_stats()
    assert s4["signal_alphabet"] == "4" and s4["code_max"] == 4
    assert s4["guard_expected_zero"] is True and s4["bad_code_n"] == 0

    # 正向②（最小反例实验）：人为把有效槽写成 0 ⇒ 守卫**必须**捕捉到
    e_bad = _make_engine("4", codebook=True)
    e_bad._codebook[:, :4] = 0
    for _ in range(20):
        e_bad.step()
    assert e_bad.alphabet_stats()["bad_code_n"] > 0, "零码守卫失效（静默通过）"

    # 对照："16" 档下 0 合法 ⇒ 守卫不适用（不得误报）
    e16 = _make_engine("16")
    for _ in range(20):
        e16.step()
    assert e16.alphabet_stats()["guard_expected_zero"] is False


# ─── 10. §4 两个零机时 counter ─────────────────────────────────────────

def test_r121_section4_counters_reported():
    e = _make_engine("4", oracle=True)
    for _ in range(30):
        e.step()
    o = e.oracle_stats()
    assert "food_band_true_sig" in o
    band = o["food_band_true_sig"]
    assert band["band"] == f"(food_threshold, {FOOD_RICH_LEVEL}]"
    assert band["n"] >= 0 and 0.0 <= band["frac_of_true_sig"] <= 1.0

    inh = e.inheritance_stats()
    assert inh["mem_inherit_n"] >= 0
    assert inh["mem_inherit_far_n"] <= inh["mem_inherit_n"]
    assert 0.0 <= inh["mem_inherit_far_frac"] <= 1.0
    assert "grid_ring" in inh["distance_convention"]


# ─── 11. 配置层：开关 / 指纹 / 旧存档回退 / runner 拒绝未实施档 ────────

def test_config_alphabet_switch_and_fingerprint():
    c = SimConfig(seed=1)
    assert c.signal_alphabet == "16"
    assert "signal_alphabet" in c.to_dict()
    assert "signal_alphabet" in c.fingerprint(), "必须进指纹 ⇒ 跨档续跑硬报错"

    round_trip = SimConfig.from_dict(c.to_dict())
    assert round_trip.signal_alphabet == "16"

    legacy = c.to_dict()
    legacy.pop("signal_alphabet")
    assert SimConfig.from_dict(legacy).signal_alphabet == "16", "旧存档必须回退 16"

    consts = (SIGNAL_ALPHABET_STATES, SIGNAL_ALPHABET_CODE_MAX)
    assert consts[0]["16"] == 16 and consts[0]["4"] == 4 and consts[0]["8"] == 8
    assert consts[1]["4"] == 4 and consts[1]["8"] == 8


def test_runner_build_rejects_unimplemented_alphabet():
    from experiments.a4_verify_capacity import build

    with pytest.raises(SystemExit):
        build("on", True, 42, 100, signal_alphabet="8")
    e4 = build("on", True, 42, 100, signal_alphabet="4")
    assert e4.config.signal_alphabet == "4"
    e16 = build("on", True, 42, 100)
    assert e16.config.signal_alphabet == "16", "默认必须仍是 16（旧行为）"
