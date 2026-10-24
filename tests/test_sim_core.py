"""sim_core（Rust 加速核）· 3.2 regrow 对拍等价测试。

验证 Rust regrow 与 Python 侧 ResourceField.regrow 逐位一致：
- temp_sensitivity=1.0（默认配置）：逐位相等（assert_array_equal）；
- temp_sensitivity≠1.0（非线性因子）：高容差近似（1e-9），
  因为 powf 与 numpy 的 C pow 最后一位可能不同。
"""
import numpy as np
import pytest

# F-V1（C5，D-2）：缺扩展 ⇒ 硬失败（不再 module-skip）。
# 原 importorskip 会在 sim_core 未安装时静默跳过整文件，
# 导致 Rust 加速核的对拍测试完全不被覆盖（"全绿"照不到盲区）。
# 现改为硬失败，让扩展缺失在套件中明确可见。
try:
    import sim_core
except ImportError as exc:  # pragma: no cover - 构建/环境问题
    pytest.fail(
        "F-V1 硬失败：sim_core 扩展未安装，Rust 加速核对拍测试将完全不被覆盖。"
        " 请先在 sim_core/ 运行 `python -m maturin develop --release` 重建扩展。"
        f" 原始错误：{exc}",
        pytrace=False,
    )

from world.light_and_temperature import LightAndTemperature
from world.resource_field import ResourceField
from world.sphere_world import SphereWorld


def make_field(rows=30, cols=60, temp_sensitivity=1.0):
    world = SphereWorld(rows=rows, cols=cols)
    light = LightAndTemperature(world, rotation_period=2400)
    field = ResourceField(
        world, light, temp_sensitivity=temp_sensitivity
    )
    return world, light, field


@pytest.mark.parametrize("ticks", [1, 37, 200])
def test_regrow_bitwise_equal_default_sensitivity(ticks):
    """默认配置（sens=1.0）：跑 1/37/200 个 tick，Rust 与 Python 逐位相等。"""
    world, light, field = make_field()
    rust_grid = field._grid.copy()

    for t in range(1, ticks + 1):
        temps = light.temperature(np.arange(world.n_cells), t)
        sim_core.regrow(
            rust_grid, field._capacity, temps,
            field.regrowth_rate, field.temp_sensitivity,
        )
        field.regrow(t)
        np.testing.assert_array_equal(rust_grid, field._grid)


def test_regrow_near_equal_sensitivity_nonlinear():
    """sens=2.5：非线性因子，逐位可能差最后一位，1e-9 容差断言。"""
    world, light, field = make_field(temp_sensitivity=2.5)
    rust_grid = field._grid.copy()

    for t in range(1, 150):
        temps = light.temperature(np.arange(world.n_cells), t)
        sim_core.regrow(
            rust_grid, field._capacity, temps,
            field.regrowth_rate, field.temp_sensitivity,
        )
        field.regrow(t)
        np.testing.assert_allclose(rust_grid, field._grid, rtol=1e-9, atol=1e-9)


def test_regrow_inplace_no_new_object():
    """就地更新：传入的数组对象不变，内容被改。"""
    fld = make_field()[2]
    original = fld._grid.copy()
    g = original.copy()
    gid = id(g)
    temps = np.zeros(fld._grid.size, dtype=np.float64)
    sim_core.regrow(g, fld._capacity, temps, fld.regrowth_rate, 1.0)
    assert id(g) == gid  # 同一个数组对象（就地改）
    np.testing.assert_array_equal(
        g, np.minimum(fld._capacity, original + fld.regrowth_rate)
    )


# ---- L7e：regrow_patchy 函数级对拍 -------------------------------------
#
# Rust regrow_patchy 与 ResourceField._regrowth_amount（distribution="patchy"）
# 逐位等价：每格恢复量 = 再生率 × 温度因子 × 空间倍率（patch/bg），min 到容量。

def make_patchy_field(rows=30, cols=60, temp_sensitivity=1.0):
    world = SphereWorld(rows=rows, cols=cols)
    light = LightAndTemperature(world, rotation_period=2400)
    field = ResourceField(
        world, light, temp_sensitivity=temp_sensitivity,
        distribution="patchy", patch_count=6, patch_radius=2,
        patch_capacity_mult=2.0, patch_regrowth_mult=2.0,
        background_fill=0.1,
    )
    return world, light, field


@pytest.mark.parametrize("ticks", [1, 37, 100])
def test_regrow_patchy_bitwise_equal(ticks):
    """patchy：跑 1/37/100 个 tick，Rust regrow_patchy 与 Python 逐位相等。"""
    world, light, field = make_patchy_field()
    rust_grid = field._grid.copy()
    mask = field._patch_mask.astype(np.uint8)

    for t in range(1, ticks + 1):
        temps = light.temperature(np.arange(world.n_cells), t)
        sim_core.regrow_patchy(
            rust_grid, field._capacity, temps, mask,
            field._patch_regrowth_mult, field._bg_regrowth_mult,
            field.regrowth_rate, field.temp_sensitivity,
        )
        field.regrow(t)
        np.testing.assert_array_equal(rust_grid, field._grid)


def test_regrow_patchy_rejects_bad_length():
    """长度不匹配应抛 ValueError（不是 UB / panic）。"""
    world, light, field = make_patchy_field()
    grid = field._grid.copy()
    temps = np.zeros(world.n_cells, dtype=np.float64)
    mask = field._patch_mask.astype(np.uint8)
    with pytest.raises(ValueError):
        sim_core.regrow_patchy(
            grid, field._capacity[: world.n_cells - 1], temps, mask,
            2.0, 0.5, 0.5, 1.0,  # capacity 短一格
        )
    with pytest.raises(ValueError):
        sim_core.regrow_patchy(
            grid, field._capacity, temps, mask[: world.n_cells - 1],
            2.0, 0.5, 0.5, 1.0,  # mask 短一格
        )


# ---- 3.3 step_vectors：种群数值管线对拍 -----------------------------------
#
# Rust 的 stage1/stage2 与 sphere_engine._step_population 的第 1~3、5~8 步
# 逐位等价。参考实现就是把引擎那几段 numpy 代码原样搬过来（同一操作顺序），
# 然后同一份输入分别走 Python / Rust，逐位比较全部数组与掩码。

GENE_COUNT = 16  # 当前引擎基因链长度（多余位不参与行为）


def make_pop(rng, P, day_len):
    """随机造一份种群状态（含环境量），供两条路径喂同一份输入。"""
    genes = rng.random((P, GENE_COUNT))
    energy = rng.random(P) * 120.0 + 1.0
    stomach = rng.random(P) * 40.0
    age = rng.integers(0, int(day_len * 0.6) + 2, size=P)
    cooldown = rng.random(P) * 60.0
    eff_activity = rng.random(P) * 1.5
    illum = rng.random(P)
    moved_raw = rng.random(P) < 0.5
    move_cost_ind = rng.random(P) * 0.6
    return (
        genes, energy, stomach, age, cooldown,
        eff_activity, illum, moved_raw, move_cost_ind,
    )


def py_stage1(energy, stomach, genes, eff_activity, illum, age, o):
    """引擎第 1~3 步的 numpy 参考实现（与 Rust stage1 一一对应）。"""
    P = len(energy)
    # 1) 光合收入
    energy += illum * genes[:, 8] * o["photo_max"]
    # 2) 代谢转化
    metab = 0.5 + genes[:, 1] * 1.5
    rate = o["base_metabolism"] * metab * eff_activity
    digest = np.minimum(stomach, rate)
    energy += digest * o["eat_efficiency"]
    stomach -= digest
    # 3) 基础维持 + 恒温维持费
    life = o["day_length"] * (1.0 + genes[:, 3] * 7.0)
    af = age.astype(np.float64)
    age_mult = np.ones(P, dtype=np.float64)
    age_mult[af < o["maturity_fraction"] * life] = o["growth_mult"]
    age_mult[af >= o["senile_fraction"] * life] = o["senile_mult"]
    energy -= o["base_metabolism"] * metab * age_mult
    energy -= o["homeo_upkeep"] * genes[:, 9]


def py_stage2(energy, age, cooldown, genes, moved_raw, move_cost_ind, o):
    """引擎第 5(后半)~8 步的 numpy 参考实现（与 Rust stage2 一一对应）。"""
    af_orig = age.astype(np.float64)  # 推进前年龄（成熟判定用，与引擎一致）
    # 5) 移动：能量门槛 + 扣费
    moved = moved_raw & (energy >= move_cost_ind)
    energy[moved] -= move_cost_ind[moved]
    # 6) 年龄推进
    age[:] += 1
    # 7) 死亡判定（用推进后的年龄判老死）
    life = o["day_length"] * (1.0 + genes[:, 3] * 7.0)
    starved = energy <= 0.0
    expired = (~starved) & (age.astype(np.float64) >= life)
    dead = starved | expired
    # 8) 冷却倒数 + 繁殖候选（成熟用推进前的年龄）
    cooldown[:] = np.maximum(0.0, cooldown - 1.0)
    repro = (
        (~dead)
        & (energy >= (0.25 + genes[:, 2] * 0.65) * o["max_energy"])
        & (cooldown <= 0.0)
        & (af_orig >= o["maturity_fraction"] * life)
    )
    return moved, starved, expired, repro


def ocfg_dict():
    """与 OrganismConfig 默认一致的标量字典（Rust 逐项传入的参数）。"""
    return {
        "photo_max": 0.1,
        "base_metabolism": 0.6,
        "eat_efficiency": 3.0,
        "growth_mult": 1.6,
        "senile_mult": 1.4,
        "maturity_fraction": 0.15,
        "senile_fraction": 0.75,
        "homeo_upkeep": 0.15,
        "day_length": 2400.0,
        "max_energy": 300.0,
    }


def rust_stage1(energy, stomach, genes, eff_activity, illum, age, o):
    sim_core.step_vectors_stage1(
        energy, stomach, genes, eff_activity, illum, age,
        o["photo_max"], o["base_metabolism"], o["eat_efficiency"],
        o["growth_mult"], o["senile_mult"],
        o["maturity_fraction"], o["senile_fraction"],
        o["homeo_upkeep"], o["day_length"],
    )


def rust_stage2(energy, age, cooldown, genes, moved_raw, move_cost_ind, o):
    out_moved = np.empty(len(energy), dtype=bool)
    out_starved = np.empty(len(energy), dtype=bool)
    out_expired = np.empty(len(energy), dtype=bool)
    out_repro = np.empty(len(energy), dtype=bool)
    sim_core.step_vectors_stage2(
        energy, age, cooldown, genes, moved_raw, move_cost_ind,
        out_moved, out_starved, out_expired, out_repro,
        o["day_length"], o["maturity_fraction"], o["max_energy"],
    )
    return out_moved, out_starved, out_expired, out_repro


@pytest.mark.parametrize("seed,P", [(7, 5), (11, 1), (42, 200)])
def test_step_vectors_bitwise_equal(seed, P):
    """同一份随机种群：stage1+stage2 连续一拍，Rust 与 Python 逐位相等。"""
    rng = np.random.default_rng(seed)
    o = ocfg_dict()
    pop = make_pop(rng, P, o["day_length"])

    py_energy, py_stomach = pop[1].copy(), pop[2].copy()
    py_age, py_cool = pop[3].copy(), pop[4].copy()
    py_stage1(py_energy, py_stomach, pop[0], pop[5], pop[6], py_age, o)
    py_moved, py_starved, py_expired, py_repro = py_stage2(
        py_energy, py_age, py_cool, pop[0], pop[7], pop[8], o,
    )

    rs_energy, rs_stomach = pop[1].copy(), pop[2].copy()
    rs_age, rs_cool = pop[3].copy(), pop[4].copy()
    rust_stage1(rs_energy, rs_stomach, pop[0], pop[5], pop[6], rs_age, o)
    rs_moved, rs_starved, rs_expired, rs_repro = rust_stage2(
        rs_energy, rs_age, rs_cool, pop[0], pop[7], pop[8], o,
    )

    np.testing.assert_array_equal(rs_energy, py_energy)
    np.testing.assert_array_equal(rs_stomach, py_stomach)
    np.testing.assert_array_equal(rs_age, py_age)
    np.testing.assert_array_equal(rs_cool, py_cool)
    np.testing.assert_array_equal(rs_moved, py_moved)
    np.testing.assert_array_equal(rs_starved, py_starved)
    np.testing.assert_array_equal(rs_expired, py_expired)
    np.testing.assert_array_equal(rs_repro, py_repro)


def test_step_vectors_boundaries():
    """边界覆盖：全饿死 / 全老死 / 未成熟 / 付不起移动费 / 冷却中。"""
    day = 2400.0
    o = ocfg_dict()
    o["day_length"] = day
    P = 6
    genes = np.full((P, GENE_COUNT), 0.5)
    # 个体 0：能量 0 → 必饿死；个体 1：年龄远超寿命 → 必老死；
    # 个体 2：幼体（age=0，未成熟）且付不起移动费；个体 3：冷却很深；
    # 个体 4：富余能量，成熟，冷却=0 → 繁殖候选；
    # 个体 5：冷却 0.5 → 倒数后为 0 → 可繁殖（若能量够）。
    energy = np.array([0.0, 200.0, 50.0, 200.0, 200.0, 200.0], dtype=np.float64)
    stomach = np.zeros(P, dtype=np.float64)
    age = np.array([0, int(day * 10), 0, 100, 500, 500], dtype=np.int64)
    cooldown = np.array([0.0, 0.0, 0.0, 9.0, 0.0, 0.5], dtype=np.float64)
    eff_activity = np.full(P, 1.0)
    illum = np.zeros(P)
    moved_raw = np.ones(P, dtype=bool)
    # 移动代价 100 > 能量 50 → 个体 2 付不起
    move_cost_ind = np.full(P, 100.0)

    py_energy, py_stomach = energy.copy(), stomach.copy()
    py_age, py_cool = age.copy(), cooldown.copy()
    py_stage1(py_energy, py_stomach, genes, eff_activity, illum, py_age, o)
    py_m, py_s, py_e, py_r = py_stage2(
        py_energy, py_age, py_cool, genes, moved_raw, move_cost_ind, o,
    )

    rs_energy, rs_stomach = energy.copy(), stomach.copy()
    rs_age, rs_cool = age.copy(), cooldown.copy()
    rust_stage1(rs_energy, rs_stomach, genes, eff_activity, illum, rs_age, o)
    rs_m, rs_s, rs_e, rs_r = rust_stage2(
        rs_energy, rs_age, rs_cool, genes, moved_raw, move_cost_ind, o,
    )

    np.testing.assert_array_equal(rs_energy, py_energy)
    np.testing.assert_array_equal(rs_stomach, py_stomach)
    np.testing.assert_array_equal(rs_age, py_age)
    np.testing.assert_array_equal(rs_cool, py_cool)
    np.testing.assert_array_equal(rs_m, py_m)
    np.testing.assert_array_equal(rs_s, py_s)
    np.testing.assert_array_equal(rs_e, py_e)
    np.testing.assert_array_equal(rs_r, py_r)

    # 语义抽检：0 饿死、1 老死、2 移动被拒、3/5 冷却生效后（3→8 不可繁殖）
    assert bool(py_s[0]) and not bool(py_e[0])
    assert not bool(py_s[1]) and bool(py_e[1])
    # 个体 2 能量(50) < 移动代价(100) → 没移动
    assert not bool(py_m[2])
    # 个体 4：年龄 500（day=2400 时寿命 10800，成熟=1620 → 未成熟，不应可繁殖）
    # 注：gen=0.5 → 寿命=2400*4.5=10800，成熟=1620，500<1620 → 不可繁殖
    assert not bool(py_r[4])


def test_step_vectors_stage1_uses_prefeeding_stomach():
    """stage1 只动胃（消化），不碰进食：传入前/后胃一致语义由引擎保证。"""
    rng = np.random.default_rng(3)
    o = ocfg_dict()
    pop = make_pop(rng, 30, o["day_length"])
    energy = pop[1].copy()
    stomach = pop[2].copy()
    safe = stomach.copy()
    rust_stage1(energy, stomach, pop[0], pop[5], pop[6], pop[3].copy(), o)
    # 消化只减不增，且不超过原胃
    assert np.all(stomach <= safe)
    assert np.all(stomach <= safe + 1e-12)


def test_step_vectors_rejects_wrong_length():
    """长度不匹配时应抛 ValueError（不是 UB / 静默越界）。"""
    o = ocfg_dict()
    P = 8
    energy = np.ones(P)
    stomach = np.ones(P)
    genes = np.ones((P, GENE_COUNT))
    eff_activity = np.ones(P)
    illum = np.ones(P)
    age = np.zeros(P, dtype=np.int64)
    with pytest.raises(ValueError):
        sim_core.step_vectors_stage1(
            energy, stomach, genes, eff_activity, np.ones(P + 1), age,
            o["photo_max"], o["base_metabolism"], o["eat_efficiency"],
            o["growth_mult"], o["senile_mult"],
            o["maturity_fraction"], o["senile_fraction"],
            o["homeo_upkeep"], o["day_length"],
        )
    with pytest.raises(ValueError):
        sim_core.step_vectors_stage2(
            energy, age.copy(), np.zeros(P + 1), genes, np.ones(P, dtype=bool),
            np.ones(P), np.zeros(P, dtype=bool), np.zeros(P, dtype=bool),
            np.zeros(P, dtype=bool), np.zeros(P, dtype=bool),
            o["day_length"], o["maturity_fraction"], o["max_energy"],
        )


# ---- 3.5 consume_many：资源场批量消耗对拍 ------------------------------
#
# Rust 的 consume_many 与 ResourceField.consume_many（引擎第 4 步进食）
# 逐位等价。参考实现就是原方法本体：同一份存量/格子/想吃的量分别走
# Python / Rust，逐位比较"扣减后的存量"与"每只实吃量"。

def rand_consume_scene(rng, n_cells, n_org, want_mult_max=15.0):
    """随机造一批进食场景：存量随机、格子可重复、单只想吃量随机。"""
    grid = rng.random(n_cells) * 30.0
    flats = rng.integers(0, n_cells, size=n_org)
    amounts = rng.random(n_org) * want_mult_max
    return grid, flats, amounts


def rust_consume(grid, flats, amounts):
    out = np.empty(len(flats), dtype=np.float64)
    sim_core.consume_many(grid, flats, amounts, out)
    return out


@pytest.mark.parametrize("seed,n_cells,n_org", [(1, 50, 0), (2, 50, 200), (3, 8, 40), (4, 8, 200)])
def test_consume_many_bitwise_equal(seed, n_cells, n_org):
    """随机进食：同格多只/存量不足/想吃超量混合，Rust 与 numpy 逐位相等。"""
    rng = np.random.default_rng(seed)
    grid, flats, amounts = rand_consume_scene(rng, n_cells, n_org)

    py_field = ResourceField.__new__(ResourceField)  # 不触构造，直接用裸数组
    py_grid, py_cap = grid.copy(), np.full(n_cells, 1e9)
    py_field._grid, py_field._capacity = py_grid, py_cap
    py_field._lazy = False  # __new__ 绕过 __init__，需手动补惰性再生关（同 __init__ 默认）
    py_taken = py_field.consume_many(flats, amounts)

    rs_grid = grid.copy()
    rs_taken = rust_consume(rs_grid, flats, amounts)

    np.testing.assert_array_equal(rs_grid, py_grid, err_msg="存量逐位不相等")
    np.testing.assert_array_equal(rs_taken, py_taken, err_msg="实吃量逐位不相等")


def test_consume_many_full_capacity_no_overdraw():
    """存量充足且想吃量一致：每只拿满，总量恰好扣完，绝不吃成负数。"""
    grid, flats = np.array([50.0, 50.0]), np.array([0, 1, 0])
    amounts = np.array([10.0, 10.0, 10.0])
    rs_grid = grid.copy()
    rs_taken = rust_consume(rs_grid, flats, amounts)
    assert rs_taken.tolist() == [10.0, 10.0, 10.0]
    np.testing.assert_array_equal(rs_grid, np.array([30.0, 40.0]))


def test_consume_many_shared_cell_split():
    """同格多只均分：存量 12、三只各想吃 10 → 各拿 4，格清零。"""
    grid, flats = np.array([12.0]), np.array([0, 0, 0])
    amounts = np.array([10.0, 10.0, 10.0])
    rs_grid = grid.copy()
    rs_taken = rust_consume(rs_grid, flats, amounts)
    np.testing.assert_array_equal(rs_taken, np.array([4.0, 4.0, 4.0]))
    np.testing.assert_array_equal(rs_grid, np.array([0.0]))


def test_consume_many_zero_amount_noop():
    """想吃量为 0：不扣存量，实吃 0。"""
    grid, flats = np.array([10.0, 0.0]), np.array([0, 1])
    amounts = np.array([0.0, 5.0])
    rs_grid = grid.copy()
    rs_taken = rust_consume(rs_grid, flats, amounts)
    np.testing.assert_array_equal(rs_taken, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(rs_grid, np.array([10.0, 0.0]))


def test_consume_many_rejects_bad_input():
    """长度不匹配 / 越界 flat 应抛 ValueError（不是 UB / panic）。"""
    grid, flats = np.ones(5), np.zeros(3, dtype=np.int64)
    amounts = np.ones(3)
    out = np.empty(3)
    with pytest.raises(ValueError):
        sim_core.consume_many(grid, flats, np.ones(4), out)      # amounts 长度错
    with pytest.raises(ValueError):
        sim_core.consume_many(grid, flats, amounts, np.empty(4))  # out 长度错
    with pytest.raises(ValueError):
        sim_core.consume_many(                                    # flat 越界(负数)
            grid, np.array([-1, 0, 0]), amounts, out,
        )
    with pytest.raises(ValueError):
        sim_core.consume_many(                                    # flat 越界(超尾)
            grid, np.array([5, 0, 0]), amounts, out,
        )