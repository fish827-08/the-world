"""`s0_kernel`（ENERGY-CLOSE 步A 硬零门）测试。

覆盖四件事，缺一即视为门禁不成立：
1. 合成闭环档每 tick 残差**恰好 0.0**（三量域都过）；
2. 变异 M1–M6 **必须变红**（新工具四律第 4 条：无此门等于没测）；
3. 实测侧与通道侧**物理隔离**（源码级检查，防"由构造恒真"回潮）；
4. 本包**不被引擎导入**（零行为面：逐字节等价门不受影响）。
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from s0_kernel.accounts import to_energy
from s0_kernel.books import DOMAIN_ACCOUNTS, Books
from s0_kernel.channels import DIS, ESC, INJ, TRF, Channel, ChannelRegistry, ChannelViolation
from s0_kernel.closed_loop import MUTATIONS, WorldOptions, closed_preset, tolerance_preset
from s0_kernel.world import (CHANNEL_DEFS, ConservationViolation, World, check_tick, run)

ROOT = Path(__file__).resolve().parents[1]
TICKS = 40
DOMAINS = ("ALL", "E", "ENERGY_ONLY")


# ── 1. 硬零门 ────────────────────────────────────────────────────
def test_closed_preset_is_exactly_zero_every_tick():
    world, books = closed_preset()
    rows = run(books, world, TICKS)
    assert len(rows) == TICKS
    for t, row in enumerate(rows):
        for domain, resid in row.items():
            assert resid == 0.0, f"tick {t} 量域 {domain} 残差 {resid!r} 非零"


def test_world_actually_runs_events_not_an_empty_loop():
    """空转的世界也能恒零——必须确认四类通道都被走过（注入/转移/散逸/逃逸）。"""
    world, books = closed_preset(WorldOptions(n=8, cells=4))
    tot = {k: 0.0 for k in (INJ, TRF, DIS, ESC)}
    for _ in range(TICKS):
        books.begin_tick()
        world.step(books)
        check_tick(books)
        for k in tot:
            tot[k] += books.from_channels(k)
    assert not all(world.alive), "40 tick 无个体死亡 ⇒ 投尸/逃逸路径没被走过"
    assert sum(world.corpses) > 0.0
    for k, v in tot.items():
        assert v > 0.0, f"kind={k} 全程为 0，等于没测"


def test_predicted_all_domain_equals_inj_minus_dis_minus_esc():
    """`D=全集` ⇒ TRF 自动抵消，退化式必须逐 tick 成立（量域争论的可算形式）。"""
    world, books = closed_preset()
    for _ in range(10):
        books.begin_tick()
        world.step(books)
        deg = books.from_channels(INJ) - books.from_channels(DIS) - books.from_channels(ESC)
        assert books.predicted_delta("ALL") == pytest.approx(deg, abs=0.0)
        assert books.close("ALL") == 0.0


def test_e_domain_keeps_transfers_and_still_closes():
    """`D={E}` ⇒ 进出 E 的转移不抵消，仍是实项；闭合照样成立（说明残差不依赖量域选边）。"""
    world, books = closed_preset()
    books.begin_tick()
    world.step(books)
    e_side = 0.0
    for cid, amt in books.per_channel().items():
        ch = books.registry[cid]
        if ch.kind == TRF and "E" in (ch.src, ch.dst):
            e_side += amt
    assert e_side > 0.0, "没有进出 E 的转移 ⇒ 本测试没测到东西"
    assert books.close("E") == 0.0
    assert books.predicted_delta("E") != books.predicted_delta("ALL")


# ── 2. 变异必须变红（砚⑦：注入已知量 ⇒ 差 = 注入量，"不崩"不算过）──
@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_mutations_show_the_injected_magnitude_with_the_right_sign(name):
    mut = MUTATIONS[name]
    world, books = closed_preset(WorldOptions(n=8))
    books.begin_tick()
    world.step(books)
    if mut.mode == "raise":
        with pytest.raises(ChannelViolation):
            mut.fn(world, books)
        return
    mut.fn(world, books)
    for domain, want in mut.expect.items():
        got = books.close(domain)
        assert got == want, (
            f"{name} 量域 {domain}：期望差 {want}（注入量·含符号），实得 {got}"
            if want else f"{name} 量域 {domain} 本应看不见（期望 0），实得 {got}")
    with pytest.raises(ConservationViolation):
        check_tick(books, domains=("ALL",), carriers=False)


def test_domain_choice_is_what_makes_resource_pool_leaks_visible():
    """M1 那类"资源池凭空多"在 `ALL` 量域红、在 `{E}` 量域**瞎**——砚① 裁全载体域的直接依据。"""
    world, books = closed_preset(WorldOptions(n=8))
    books.begin_tick()
    world.step(books)
    _m1 = MUTATIONS["M1_inject_unledgered"]
    _m1.fn(world, books)
    assert books.close("ALL") == 0.5          # 全载体域看得见
    assert books.close("E") == 0.0            # 只闭合活体能量 = 漏
    assert books.close("ENERGY_ONLY") == 0.0  # 跳过质量池同样漏


def test_carrier_columns_catch_two_offsetting_leaks_that_the_total_hides():
    """砚① 分载体五列的用意：合计 0.0 但分账非零 = 两笔反向漏账互抵，只看合计算它过了。"""
    world, books = closed_preset(WorldOptions(n=8))
    books.begin_tick()
    world.step(books)
    world._g[0] += 1.0        # +0.5 能量当量（mass×eff）
    world._e[0] -= 0.5        # −0.5 能量
    assert books.close("ALL") == 0.0, "构造前提：两笔互抵，合计恒零"
    rep = books.carrier_report()
    assert rep["G"]["residual"] == 0.5 and rep["E"]["residual"] == -0.5
    assert rep["TOTAL"]["residual"] == 0.0
    with pytest.raises(ConservationViolation, match="carrier_"):
        check_tick(books, domains=("ALL",))


def test_clean_tick_carrier_columns_are_all_zero():
    """反向自检：同一张五列表在干净 tick 必须全零（否则分账判据本身是摆设）。"""
    world, books = closed_preset(WorldOptions(n=8))
    books.begin_tick()
    world.step(books)
    assert {a: r["residual"] for a, r in books.carrier_report().items()} == \
        dict.fromkeys(("E", "S", "G", "C", "F", "TOTAL"), 0.0)


def test_heat_excludes_escape_by_its_own_ledger():
    """砚⑥ 边界款：热账 = Σ_DIS，逃逸分列、**不得并进热账**（吞掉逃逸 = 丢掉最敏感的分账）。"""
    world, books = closed_preset(WorldOptions(n=8))
    books.begin_tick()
    world.step(books)
    heat0, esc0 = books.heat(), books.escaped()
    assert heat0 == sum(a for cid, a in books.per_channel().items()
                        if books.registry[cid].kind == DIS)
    assert esc0 == sum(a for cid, a in books.per_channel().items()
                       if books.registry[cid].kind == ESC)
    with world.holder(E=1):
        books.channel("corpse_leak", "E", None, 0.125)
    assert books.heat() == heat0, "多出来的逃逸被算进热账了——违反砚⑥"
    assert books.escaped() == esc0 + 0.125


def test_cross_domain_bridge_must_declare_its_source():
    """砚①：账本凡涉 S↔E 必须注明系数来源 ⇒ 不给 `eff_source` 直接拒绝建账。"""
    world = World(WorldOptions(eff_source=""))
    with pytest.raises(ChannelViolation, match="来源"):
        world.books()


def test_to_energy_is_the_single_conversion_helper_and_rejects_unknown_accounts():
    assert to_energy(1.0, "G", 0.5) == 0.5        # mass 账户折能量域
    assert to_energy(1.0, "E", 0.5) == 1.0        # 能量账户不动
    with pytest.raises(KeyError):
        to_energy(1.0, "X", 0.5)


def test_meta_echoes_the_bridge_for_manifest():
    """轻舟约束 3 的同族预防：读数量级报告必须能回显生效系数与来源，跨树对账才不抓瞎。"""
    _, books = closed_preset()
    meta = books.meta()
    assert meta["eat_efficiency"] == 0.5
    assert "eat_efficiency" in meta["eff_source"]


def test_clean_tick_passes_the_same_gate_that_mutations_fail():
    """同一判据：不动它过、动它红——排除"红是因为判据另有前置条件"。"""
    world, books = closed_preset(WorldOptions(n=8))
    books.begin_tick()
    world.step(books)
    assert check_tick(books) == dict.fromkeys(DOMAINS, 0.0)


# ── 3. 两侧物理隔离 ─────────────────────────────────────────────
def test_measured_side_source_never_touches_the_ledger():
    src = inspect.getsource(Books.measured_delta_sigma) + inspect.getsource(Books._snap)
    assert "_ledger" not in src and "predicted" not in src


def test_predicted_side_source_never_reads_world_state():
    src = inspect.getsource(Books.predicted_delta)
    assert "measure" not in src and "_snap" not in src and "_apply" not in src


def test_close_is_the_only_meeting_point():
    """`close()` 只准取两侧之差——不许自己再算一遍（防第三本账混进来）。"""
    src = inspect.getsource(Books.close)
    assert "measured_delta_sigma" in src and "predicted_delta" in src
    assert "_ledger" not in src and "_snap" not in src


def test_bypass_write_fails_loud():
    world, _ = closed_preset()
    with world.holder(E=0):
        with pytest.raises(ChannelViolation, match="旁路"):
            world.apply("E", 1.0)


def test_holder_is_required_even_inside_a_channel():
    """落点未声明 ⇒ 不知道这笔量该进哪个体/格，当场炸（不许"默认给 0 号"这种静默兜底）。"""
    world, books = closed_preset()
    with pytest.raises(ChannelViolation, match="无持有者"):
        books.channel("photo", None, "G", 0.25)


def test_unregistered_channel_is_rejected():
    world, books = closed_preset()
    with world.holder(G=0):
        with pytest.raises(ChannelViolation, match="未登记通道"):
            books.channel("ghost", None, "G", 1.0)


def test_kind_declaration_invariants_are_enforced():
    with pytest.raises(ChannelViolation):
        Channel("bad_inj", INJ, "G", "G")          # INJ 不许有 src
    with pytest.raises(ChannelViolation):
        Channel("bad_trf", TRF, "G", None)         # TRF 两端都得是账户
    with pytest.raises(ChannelViolation):
        Channel("bad_esc", ESC, "E", "C")          # 逃逸不许有 dst
    with pytest.raises(ChannelViolation):
        Channel("bad_kind", "HOT", None, "G")      # 未知 kind


def test_negative_amount_is_rejected():
    world, books = closed_preset()
    with world.holder(G=0):
        with pytest.raises(ChannelViolation, match="非负"):
            books.channel("photo", None, "G", -1.0)


def test_duplicate_channel_registration_is_rejected():
    with pytest.raises(ChannelViolation, match="重复登记"):
        ChannelRegistry(CHANNEL_DEFS + CHANNEL_DEFS[:1])


def test_nonpositive_eff_is_rejected():
    world = World()
    world.opt.eff = 0.0
    with pytest.raises(ChannelViolation, match="eff"):
        world.books()


def test_every_registered_channel_is_exercised():
    """通道表里不许躺"声明了但没人调"的死通道——它会伪装成"已审计"。"""
    world, books = closed_preset(WorldOptions(n=8, cells=4))
    seen: set[str] = set()
    for _ in range(TICKS):
        books.begin_tick()
        world.step(books)
        seen |= set(books.per_channel())
    assert seen == set(books.registry.cids()), (
        f"漏跑通道 {sorted(set(books.registry.cids()) - seen)}")


# ── 4. 零行为面守卫 ─────────────────────────────────────────────
def test_engine_never_imports_s0_kernel():
    """本包不得进入引擎运行时（否则逐字节等价门要重开）。"""
    offenders = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "simulation").rglob("*.py"))
                 if "s0_kernel" in p.read_text(encoding="utf-8", errors="ignore")]
    assert offenders == [], f"引擎已导入 s0_kernel：{offenders}"


def test_kernel_consumes_no_rng():
    for name in ("accounts", "channels", "books", "world", "closed_loop"):
        src = inspect.getsource(importlib.import_module(f"s0_kernel.{name}"))
        assert "random" not in src and "np." not in src, name


def test_same_options_give_bit_identical_state():
    a, ba = closed_preset(WorldOptions(n=6, cells=4))
    b, bb = closed_preset(WorldOptions(n=6, cells=4))
    run(ba, a, TICKS)
    run(bb, b, TICKS)
    assert a.measure() == b.measure()
    assert a.energy == b.energy and a.grid == b.grid and a.corpses == b.corpses


# ── 5. 量域报告与容差档 ─────────────────────────────────────────
def test_domain_gap_report_explains_the_difference_by_channel():
    world, books = closed_preset(WorldOptions(n=8))
    books.begin_tick()
    world.step(books)
    gap = books.domain_gap_report()
    assert all(gap[f"residual_{d}"] == 0.0 for d in DOMAIN_ACCOUNTS)
    terms = [k for k in gap if k.startswith("term__")]
    assert terms, "两量域应存在实项差异（进出 E 的转移），报告里必须点到通道"


def test_tolerance_mode_accepts_bit_noise_but_hard_zero_mode_does_not():
    """两种判门模式语义分开：`exact=False` 容忍位噪级扰动，硬零档连 1e-12 都要炸。

    真实 `eat_efficiency` 档（非 2 幂系数）只能走容差模式——但容差模式**不是守恒结论**，
    它只是"这档没资格设 0.0 门"的开关（步B 开启批才谈阈值）。
    """
    world, books = tolerance_preset(WorldOptions(n=8, eff=0.3))
    books.begin_tick()
    world.step(books)
    world._g[0] += 1e-12                       # 位噪级扰动（原生质量）
    rows = check_tick(books, exact=False)
    assert all(abs(v) < 1e-9 for v in rows.values()), rows
    with pytest.raises(ConservationViolation):
        check_tick(books, exact=True)


def test_non_dyadic_eff_still_runs_clean_in_tolerance_mode():
    world, books = tolerance_preset(WorldOptions(n=8, eff=0.3))
    rows = run(books, world, 20, exact=False)
    assert all(abs(v) < 1e-9 for row in rows for v in row.values())


def test_hard_zero_preset_refuses_non_dyadic_constants():
    """`signal_cost=0.1` 这类常量的浮点分母是 2^55 ⇒ 与 8.0 相加必舍入。

    硬零档若收这种参数，残差 1e-15 会被误读成"守恒被破坏"——所以直接拒收并说明原因。
    """
    with pytest.raises(ChannelViolation, match="位宽前提"):
        closed_preset(WorldOptions(signal_energy=0.1))


def test_measure_returns_a_copy_not_a_view():
    """`measure()` 返回副本：拿它改状态 = 绕开门禁的后门，测试先把它堵掉。"""
    world, _ = closed_preset()
    snap = world.measure()
    snap["E"] += 100.0
    assert world.measure()["E"] == pytest.approx(sum(world.energy))


# ══ A 路（PI 22:0x 落裁③：事件集合求和 + 每 N tick 全量核对）═══════════
def test_eventset_agreement_is_exact_zero_on_clean_ticks():
    """干净 tick：全量实测与事件集合实测**逐位相同**（差恰好 0.0，非容差）。

    这是 PI 落裁③「全量 vs 事件集合 差=0」验收硬门的合成世界实证——引擎段照同一判据形。
    """
    world, books = closed_preset(WorldOptions(n=8))
    for _ in range(30):
        books.begin_tick()
        world.step(books)
        for d in ("ALL", "E", "ENERGY_ONLY"):
            assert books.eventset_agreement(d) == 0.0, f"{d}：{books.events if hasattr(books,'events') else world.events[-3:]}"
            assert books.close_eventset(d) == books.close(d) == 0.0


def test_unledgered_write_on_untouched_slot_is_invisible_to_per_tick_eventset_gate():
    """🔴 A 路的**结构性失明**（本文件最重要的一条实测结论）：

    旁路写落在"本 tick 没被写过的槽位"上 ⇒
      · 逐 tick 的 `close_eventset()` **照绿**（=0.0，假绿）
      · 全量 `close()` 变红（=注入量，砚⑦方向检验形）
      · 二者之差 `eventset_agreement()` = 注入量 ⇒ **每 N tick 全量核对是硬门，不是保险丝**
    反之若旁路写砸在**本 tick 写过的槽位**上，事件集合侧照样看得见（原像差取当前数组值）。
    """
    world, books = closed_preset(WorldOptions(n=8))
    books.begin_tick()
    world.step(books)
    # fail-loud 前提：C 池本 tick 确实没被写过（否则这条测的是别的东西）
    assert ("C", 0) not in world._touch, "前提破裂：C 槽位本 tick 被写过"
    world._c[0] += 0.5                       # 旁路写，不经 apply ⇒ 不进事件集合
    assert books.close_eventset("ALL") == 0.0,   "失明证据：A 路逐 tick 判据绿着"
    assert books.close("ALL") == 0.5,            "全量判据红，且红量 = 注入量（方向检验）"
    assert books.eventset_agreement("ALL") == 0.5
    assert books.eventset_agreement("E") == 0.0, "C 不在 {E} 量域——量域差异照旧成立"
    assert books.eventset_agreement("ENERGY_ONLY") == 0.5, "C 是 energy 账户"


def test_poke_on_touched_slot_is_caught_by_eventset_side():
    """互补证据：砸在**本 tick 写过**的槽位 ⇒ 事件集合侧当场看见（原像差非零）。"""
    world, books = closed_preset(WorldOptions(n=8))
    books.begin_tick()
    world.step(books)
    touched = [k for k in world._touch if k[0] == "E"]
    assert touched, "前提：本 tick 至少有一个 E 槽位被写过"
    acc, slot = touched[0]
    world._e[slot] += 0.25
    assert ("E", slot) in world._touch
    assert books.eventset_agreement(acc) == 0.0, "写过 ⇒ 事件集合与全量同红"
    assert books.close(acc) == 0.25


def test_eventset_side_requires_injection_no_silent_fallback_to_full_sum():
    """没注入 eventset 闭包 ⇒ 当场报错；**不许**静默退回全量重求和（那等于把 A 路伪装成已核）。"""
    world, _ = closed_preset(WorldOptions(n=4))
    bare = Books(ChannelRegistry(CHANNEL_DEFS), measure=world.measure, apply=world.apply,
                 grant=world._grant, revoke=world._revoke, eff=world.opt.eff,
                 eff_source=world.opt.eff_source)
    with pytest.raises(ChannelViolation, match="未注入事件集合侧"):
        bare.measured_delta_sigma_eventset("ALL")
    with pytest.raises(ChannelViolation, match="未注入事件集合侧"):
        bare.close_eventset("ALL")


def test_eventset_measurement_source_does_not_read_the_channel_ledger():
    """源码级隔离：事件集合侧的"实测"来自**世界数组原像**，不得访问 `_ledger`/预测账。

    用 `co_names`（属性访问名）判，不用原始 source——docstring 里提一嘴"不读 _ledger"
    不该把守卫弄红（那正是我第一版守卫的假阳性）。
    否则 A 路退化成"用账本验账本"（A1 构造恒等同族），PI 落裁③ 的核对意义全失。
    """
    forbidden = {"_ledger", "predicted", "predicted_delta", "per_channel", "_kind"}
    for fn in (Books.measured_delta_sigma_eventset, Books.eventset_agreement):
        hit = forbidden & set(fn.__code__.co_names)
        assert not hit, f"{fn.__name__} 读了通道账：{hit}"
    wforbidden = {"_ledger", "books", "predicted"}
    assert not wforbidden & set(World.eventset_delta.__code__.co_names)
    # 反向：全量实测侧同样不许读账本（既有守卫的延伸）
    assert not forbidden & set(Books.measured_delta_sigma.__code__.co_names)
    # `close_eventset` 是**会合点**（与 `close` 同族）：必须两侧都读，缺一即假绿
    both = {"measured_delta_sigma_eventset", "predicted_delta"}
    assert both <= set(Books.close_eventset.__code__.co_names)
