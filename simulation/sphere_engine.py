"""SphereEngine：球面世界的生物引擎（模块二 · 文件 2/2）。

模块职责
--------
上文说清了"土地/天气/食物"（模块一），这份文件让**生物**在这上面
活起来：每个 tick，它们进食、转化能量、移动、衰老，可能死亡、繁殖。
所有个体用一张"大数据表"（NumPy 数组）同时推进，速度远快于逐个体循环。

规则语义（与主人确认过的版本）
-------------------------------
1. 代谢是"胃→能量的转化"：吃进胃里的食物不立即变能量；
   每 tick 按【代谢速率】把一部分胃转化为能量。
   - 转化总量 = 同样的食物最终都给同样能量（不会因为温度少给）；
   - 转化【速度】受温度和基因影响：合适温度快、冷的地方慢；
   - 吃得多先存在胃里，以后有空再慢慢转化（饱食度=胃容量上限）。
2. 移动是生物自己的决定（基因概率），温度只影响移动的【代价】：
   冷的地方移动更费能；爱扎堆的（g13）会朝同伴多的邻居走，
   独行侠（g13<0.5）专挑冷清的邻居走。
3. 基础维持消耗（体温/活动）每 tick 必扣；
   恒温个体（g9≈1）再多扣一点（恒温维持费），换取低温不减速。
4. 部分个体会光合（基因 g8）：白天按所在格光照获得少量能量（产能刻意小）。
5. 有觅食基因（g10）的个体，自己格不够吃时可以补吃一格外邻格的食物。
6. 繁殖时按基因 g7 决定把多少比例的能量/胃粮分给子代（默认 0.5 对半）；
   生完要休息（g12 冷却期），冷却没有过去之前攒再多能量也不生。
7. 有【成熟年龄】：未到寿命的一定比例（默认 15%，由 g3 寿命决定）绝不能繁衍，
   防止一出生就疯狂生。
8. 能量【需求随年龄变化】：幼体在长身体（维持费 ×1.6）、成年（×1）、
   老年器官退化（×1.4）——需要的能量不一样。

坐标：flat（平铺索引）代表"在哪个格子"，不用 x/y——
经度环绕、极点坍缩全部由 SphereWorld 内部处理，引擎只管搬格子。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from core.lifecycle import DeathCause
from simulation.config import (
    CALIBRATION_M_RANGE,
    FOOD_RICH_LEVEL,
    SIGNAL_ALPHABET_CODE_MAX,
    SIGNAL_ALPHABET_IMPLEMENTED,
    SIGNAL_ALPHABET_STATES,
    SIGNAL_COST,
    SimConfig,
)
from simulation.genes import Gene
from simulation.oracle import EMISSION_COST, apply_oracle, attribution_ok
from simulation.provenance import CountingRNG
from simulation.tick import TickStats
from world.light_and_temperature import LightAndTemperature
from world.resource_field import ResourceField
from world.signal_field import SignalField
from world.sphere_world import SphereWorld


def codebook_init_rows(n_rows: int, alphabet: str) -> NDArray[np.uint8]:
    """按 `signal_alphabet` 档位构造初始码本（形状 `(n_rows, 16)`，uint8）。

    R113/R121（设计稿 §3.2）——**数组宽度恒为 16**（存档兼容，R113 明文）：
      · `"16"`：恒等映射 `arange(16)`（0–15）⇒ **与旧版逐位一致**（既有批次口径不动）
      · `"4"` ：`clip(arange(16)+1, 1, 4)` ⇒ 有效槽 0–3 → `{1,2,3,4}`；槽 4–15 保持初值
                （**码 0 永不出现** ⇒ 设计稿 §2.4 的"隐形发射 + 清除他人标记"后果消失）
      · `"8"` ：`clip(arange(16)+1, 1, 8)`（B③ 载体；**本版未实施** ⇒ 调用即硬失败）

    未实施的档位**硬失败**（教训 2：静默 no-op 最危险），不降级、不报警后继续。
    """
    if alphabet not in SIGNAL_ALPHABET_IMPLEMENTED:
        raise NotImplementedError(
            f"signal_alphabet={alphabet!r} 尚未实施"
            f"（已实现：{SIGNAL_ALPHABET_IMPLEMENTED}）"
        )
    if alphabet == "16":
        row = np.arange(16, dtype=np.int64)
    else:
        row = np.clip(np.arange(16, dtype=np.int64) + 1, 1,
                      SIGNAL_ALPHABET_CODE_MAX[alphabet])
    return np.asarray(row, dtype=np.uint8)[np.newaxis, :].repeat(
        int(n_rows), axis=0
    ).copy()


def encode_signal_states(
    energy: NDArray[np.float64],
    grid: NDArray[np.float64],
    capacity: NDArray[np.float64],
    occ: NDArray[np.int64],
    e_flat: NDArray[np.int64],
    emitters: NDArray[np.int64],
    max_energy: float,
    alphabet: str,
    work_memory: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.int64], int]:
    """按 `signal_alphabet` 档位把**发射者状态**编码为 `(state, code_offset)`。

    抽成**纯函数**的目的（设计稿 §3.6 测试 3）：让"冗余位是否确被删"可**不插桩**直接验证 ——
    只需给同 `e_bin`、不同 (食物, 邻居) 的两组输入，看 `state` 是否相同。

    · `"16"`：能量档 2 位 + 食物 1 位 + 邻居 1 位 ⇒ `state = e_bin*4 + f_bit*2 + n_bit`（0–15）；
              `code_offset = 0` ⇒ 无码本时 `code = state`（**0 是合法值** = 无信号，旧语义）
    · `"4"` ：**仅**能量档 2 位 ⇒ `state = e_bin`（0–3）；`code_offset = 1`
              ⇒ 无码本时 `code = e_bin + 1 ∈ {1,2,3,4}`，**码 0 永不出现**
              （删 `f_bit`：阈值错位，拟 F-M2；删 `n_bit`：严格冗余，内评 §二.1）
    · `"8"` ：能量档 2 位 + **记忆位** 1 位 ⇒ `state = e_bin*2 + mem_bit`（0–7）；
              `code_offset = 1` ⇒ 无码本时 `code = state + 1 ∈ {1..8}`
              （`mem_bit` 语义见下方分支注释；**2026-09-19 已按内评抽核修正为"非当前格命中"**）
    · 其余档位：`NotImplementedError`（不静默降级，教训 2）

    ⚠️ `code_offset` 只在**无码本（恒等映射）**时使用；开码本时 `pattern = codebook[:, state]`，
    其值域由 `codebook_init_rows` 与变异域共同保证（"4" ⇒ ⊆ [1,4]）。
    """
    e_bin = np.clip(
        (energy[emitters] / max(1e-9, max_energy) * 4).astype(np.int64), 0, 3
    )
    if alphabet == "16":
        f_bit = (grid[e_flat] > FOOD_RICH_LEVEL * capacity[e_flat]).astype(np.int64)
        n_bit = (occ[e_flat] > 1).astype(np.int64)
        return (e_bin * 4 + f_bit * 2 + n_bit).astype(np.int64), 0
    if alphabet == "4":
        return e_bin.astype(np.int64), 1
    if alphabet == "8":
        # B③（R123 2026-09-18 实施 / **2026-09-19 构念修正**）：能量档 2 位 + **记忆位** 1 位
        #   `mem_bit = 1` ⟺ 发射者记忆中**存在一个 ≠ 当前格**的富食点（"我知道**别处**哪儿有食物"）。
        #   🔴 为何修正（内评 2026-09-19 01:4x 抽核）：原实现 = 「**当前格**在我记忆里」，而
        #   `_work_memory` 是**站在富食格上时**写入的 ⇒ 该判据 ≈「这一格曾经富过」= `f_bit` 的
        #   **时间延迟版**，而当前格的瞬时食物**接收者本来就能直读** ⇒ 该位**仍冗余**（预期 null；
        #   且其 null **不可**读作「私有内容无用」——那是仪器/构念问题，不是信号问题）。
        #   改为「非当前格命中」后，该位指向**接收者无法直读的个体历史（另一些格位）**⇒ 才符合
        #   B③ 的原始意图（设计稿 §5.1）。
        #   ⚠️ **残留口径**（内评 §三，列为**下一档选项**）：记忆中的格若落在感知半径 4 内，接收者
        #   仍可直读 ⇒ 严格的「非直读」版本需加**距离/方位**判据（本档不做，避免一次改两件事）。
        if work_memory is None:
            raise ValueError(
                'signal_alphabet="8" 需要 work_memory（记忆位 mem_bit 的输入）——收到 None'
            )
        _mem = work_memory[emitters]
        # `-1` = 空槽哨兵（`e_flat ≥ 0` ⇒ 不会误命中）；同时排除"当前格"本身
        mem_hit = ((_mem != -1) & (_mem != e_flat[:, None])).any(axis=1).astype(np.int64)
        return (e_bin * 2 + mem_hit).astype(np.int64), 1
    raise NotImplementedError(
        f"signal_alphabet={alphabet!r} 的编码尚未实施"
        f"（已实现：{SIGNAL_ALPHABET_IMPLEMENTED}）"
    )


@dataclass(frozen=True)
class PopulationView:
    """一组只读的种群统计快照（per-tick 观察窗口，够 observer 用）。"""

    tick: int
    alive_count: int
    max_generation: int
    total_energy: float
    total_food_in_stomach: float


# R135 第 -1 步④：互捕量化直方图的分箱数（g16 ∈ [0,1] ⇒ 每箱宽 0.2）
CANNIB_BINS = 5

# ---------------------------------------------------------------------------
# R141 P0：**分通道能量记账**（能量校准预实验的前提）。
#   通道索引（顺序即语义，勿随意重排）：
#     0 intake_forage   取食（斑块）摄入
#     1 intake_pred     捕食摄入（抢到的猎物能量）
#     2 cost_meta       基础维持 + 恒温
#     3 cost_move       移动扣费
#     4 cost_attack     攻击成本（每次出手必付）
#   按 **g16 三分箱**（食草 / 杂食 / 捕食 三腿）分别累计 ⇒ 才能验证
#   C 档目标 `R_捕食 ≈ R_食草 > R_杂食`（**没有"杂食腿"则"中间最低"无法验证**）。
#   ⚠️ 只在 **Python 路径**（`use_sim_core=False`）记账：Rust 路径的扣费发生在
#      Rust 内部，Python 侧看不到 ⇒ `energy_channel_stats()` 会标 `path="rust"` + 值 None
#      （**不报 0**——"没测到"与"测到 0"必须分开，本项目老坑）。
EC_FORAGE, EC_PRED, EC_PHOTO, EC_META, EC_MOVE, EC_ATTACK = range(6)
EC_N = 6
EC_NAMES = ("intake_forage", "intake_pred", "intake_photo",
            "cost_meta", "cost_move", "cost_attack")
# 三腿的分组名（派工单 §1.1：**每 tick 按个体当前 g16 现算**，不用出生标签 ⇒ 防杂交后失效）
EC_BOX_NAMES = ("lo", "mid", "hi")   # g16 < 1/3 / [1/3, 2/3] / > 2/3
# 收入侧通道（净收入 = 收入 − 支出；缺 `intake_photo` 会让净收入虚假为负——
# 光合是独立于取食的**直接收入**，不经过胃）。

# R141/R138：真决斗三级拆分的键（**顺序固定**；全部预置 0 ⇒ 输出不随事件有无而变）
#   nominal = len(attackers)：名义攻击者（= `cannibalism.n_attacks`，旧口径的分母）
#   skip_*  = 三类"根本没出手"的原因；real_attempts = 真出手 = nominal − 三类 skip
#   kills   = 致死（= `cannibalism.n_kills`）
# ⇒ **真实出手成功率 = kills / real_attempts**（旧口径 kills / nominal 会低估）
DUEL_KEYS = ("skip_no_energy", "skip_no_prey", "skip_already_eaten",
             "real_attempts", "kills")


def _genome_summary(genes) -> dict:
    """基因组摘要（t=0 基线；**只读、不消费 RNG**）。

    R135 第 -1 步③。核心字段 `corr_g16_g4`：搭档基因 g4（进食量倍率，强选择）
    与 g16（攻击性，关捕食后中性）的**初始抽样 LD**。
    实测（详见 `SphereEngine.__init__` 注释）表明该 LD 会通过**搭车效应**把 g16 拖走，
    量级足以伪造"g16 被选择" ⇒ **判读任何基因位移前必须先扣掉它**。
    """
    g = np.asarray(genes, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] < 2 or g.shape[1] < 17:
        return {}
    m = g.mean(axis=0)
    s = g.std(axis=0, ddof=1)
    out = {
        "n": int(g.shape[0]),
        "gene_count": int(g.shape[1]),
        "g16_mean": float(m[16]),
        "g16_std": float(s[16]),
        "g4_mean": float(m[4]),
        "g4_std": float(s[4]),
        "gene_means": [round(float(x), 6) for x in m],
    }
    # 相关系数：任一方退化（sd≈0）时给 None 而不是 NaN（NaN 会静默污染下游 JSON）
    if float(s[16]) > 1e-12 and float(s[4]) > 1e-12:
        out["corr_g16_g4"] = float(np.corrcoef(g[:, 16], g[:, 4])[0, 1])
    else:
        out["corr_g16_g4"] = None
    return out


class SphereEngine:
    """球面世界引擎：NumPy 数组整群推进。

    实例属性（__slots__ 说明）
    -------------------------
    config : SimConfig
        全部参数（网格/光照/资源/生物/基因/种群/运行）。
    rng : np.random.Generator
        随机数源（用 config.seed 初始化，保证可复现）。
    world : SphereWorld
        土地（模块一文件1）：格子、面积、邻居。
    light : LightAndTemperature
        天气（模块一文件2）：光照、温度、活性。
    resources : ResourceField
        食物（模块一文件3）：存量、容量、再生。
    _flat : NDArray[int64]
        每个存活个体的位置（平铺索引）。
    _energy : NDArray[float64]
        每个个体"可用能量"（能用来移动/维持/繁殖的能量）。
    _stomach : NDArray[float64]
        每个个体"胃里的食物"（等待被代谢转化为能量）。
    _genes : NDArray[float64], 形状 (N, gene_count)
        每个个体的基因链。
    _age : NDArray[int64]
        每个个体已活的 tick 数。
    _generation : NDArray[int64]
        每个个体是第几代。
    _parent : NDArray[int64]
        每个个体的亲代 id（-1 = 初始个体）。
    _id : NDArray[int64]
        每个个体的唯一编号。
    _next_id : int
        下一个可用编号。
    _max_generation : int
        全种群最深的世代。
    _tick : int / _extinct / _finished :
        运行状态。
    _history : list[TickStats]
        逐 tick 统计（history_limit>0 时只留尾部）。
    _run_born / _run_died / _run_deaths :
        运行期累计账本（长程实验防历史无限增长）。
    """

    __slots__ = (
        "config", "rng", "world", "light", "resources", "signals",
        "_flat", "_energy", "_stomach", "_genes", "_age", "_generation",
        "_parent", "_id", "_next_id", "_max_generation",
        "_repro_cooldown",
        "_valence", "_arousal", "_expectation", "_baseline", "_trust",
        "_work_memory", "_mem_ptr", "_interpret", "_nb_table",
        # A′ 记忆朝向梯度计数器（2026-09-19）—— 本类用 `__slots__`，**新属性必须登记否则无法赋值**
        "_mem_grad_dec", "_mem_grad_slots", "_mem_grad_trig", "_mem_grad_counts_valid",
        "_codebook", "_learning_count",  # D2 信息结构：任意性码本 + 学习瓶颈计数
        "_tick", "_extinct", "_finished", "_history",
        "_history_limit", "_run_born", "_run_died", "_run_deaths",
        "_use_sim_core", "_sim_core",
        "_fruit_grid", "_fruit_charge", "_seed_carried",  # L10a
        # ---- D-17/D-18/D-8（⑤⑥ 探针 + oracle）----
        # 按 _id 键控的终身账本（长度 = _next_id，只增不压缩；死亡个体保留行——
        # R42"必须含死亡个体"）。槽位会被 :1163 死亡压缩重排 ⇒ 禁止槽位键控。
        "_rs_children", "_rs_observed", "_rs_g15", "_rs_age", "_rs_energy",
        "_rs_cc0", "_rs_cohort", "_emit_count", "_oracle_gain",
        "_last_sender",                     # cell → 最近写入者的 _id（oracle 归因）
        "_oracle_transfers", "_oracle_count",
        "_resp_decisions", "_resp_exposed", "_resp_delta_sum", "_resp_flip",
        "_measure_resp", "_oracle_on",
        # ---- R123/B② 门控臂（付款闸用逐个体 Δ；纯观测计数）----
        "_delta_full_by_id", "_delta_content_by_id", "_delta_tick_by_id",
        "_gate_on", "_gate_delta",
        "_diag_gate_pass", "_diag_gate_block",
        "_gate_arrivals", "_gate_delta_full_sum", "_gate_delta_content_sum",
        "_resp_delta_content_sum",
        # ---- D-26a：oracle 四环节诊断漏斗（纯观测计数，D-24 G-A 不过后定位瓶颈）----
        # 内评 _eval/D24判读预析 §4.1：分开记 ①发射 ②归因成功 ③true_sig ④实际转移，
        # 否则只有聚合 return_ratio、不知道卡在哪一环。仅在 oracle 开启时累加。
        "_diag_had_sig", "_diag_true_sig", "_diag_sel",
        "_diag_attrib_found", "_diag_in_window", "_diag_not_self",
        "_diag_budget_ok", "_diag_applied",
        # ---- R121 §3/§4（2026-09-18）：信号字母表档位 + 两个零机时观测计数器 ----
        # ⚠️ __slots__ 是硬约束：新属性**必须**在此登记，否则运行期 AttributeError
        "_alpha", "_alpha_states", "_alpha_code_max",
        "_diag_food_band_true_sig", "_mem_inherit_n", "_mem_inherit_far_n",
        "_alpha_bad_code_n",
        "_mem_bit_on", "_mem_bit_n",   # 2026-09-19：mem_bit 取值分布（闭合可观测性缺口）
        # ---- R102/条件 1（修正版）：守恒三账审计（★在引擎侧**独立**核算，非"同一个数抄三遍"）----
        # 撤销 R100 条件 1 原定的 "system_energy_injected"：机制复核（R102 §三）已证
        # apply_oracle 是**双向转移** ⇒ 纯再分配、**C-2 守恒成立**、无注入。
        # 审计口径 = 包住 oracle 调用前后实测 Σenergy 的变化（应恒 0）。
        "_audit_calls", "_audit_sum_delta",
        # ---- R102 条件 1/3 + 内评《复核-R102方向修复》§四：**oracle 对账字段** ----
        # 定位 = 「对账（三账应恒等，由构造保证）+ 偿付约束量化」，**不是**守恒检验。
        # 内评 §四 命名建议：call it a ledger, not an audit（本项目已吃过"看着像检查、
        # 实际恒真"的亏：G-F 门 pass:True / test_o7 假绿）。
        "_oracle_payer_paid", "_oracle_sender_received",
        "_oracle_payer_trunc_n", "_oracle_payer_trunc_amt",
        "_oracle_budget_trunc_n", "_oracle_budget_trunc_amt",
        "_oracle_payer_broke_n", "_oracle_budget_exhausted_n",
        # ---- 内评 §三 观察项 1/2：**接收侧效应**（方向翻转新引入；按 cell 键控）----
        # 接收者在 t 付款、t+1 在其**成交落点**进食 ⇒ 记录落点，下一 tick 进食时实测摄入。
        # 目的：判断「吃到 − 回付」是否仍 ≥ 不通信者的平均摄入（若为负 ⇒ 规避标记格的激励）。
        "_recv_pend_cells", "_recv_pay_sum",
        "_recv_food_sum", "_recv_food_n", "_all_food_sum", "_all_food_n",
        # R135 第 -1 步③：t=0 基因组摘要（搭车诊断基线，只读）
        "_genome_t0",
        # R135 第 -1 步④：互捕结构量化（攻击者/猎物 g16 直方图，只读）
        "_cannib_atk_hist", "_cannib_prey_hist",
        # R141 P0：分通道能量记账（逐个体缓冲 + 逐 tick 统计序列）
        "_ec_global", "_ec_box", "_ec_n", "_ec_pt", "_ec_ts",
        "_ec_prey_e_sum", "_ec_prey_kill_n",
        # R145：能量封顶活体探针
        "_frac_over_cap_max", "_over_cap_seen",
        # R141/R138：真决斗三级拆分（名义/出手/致死 + skip 原因）
        "_duel",
        # R146/R149 L2 机动性（本段 = [本地开发] 线）：strict 2 圈 CSR + 读数计数器
        "_far_off", "_far_cells", "_far_len", "_far_excluded_n",
        "_run_mover_n", "_run_dash_n", "_mover_n_by_box", "_dash_n_by_box",
        "_mob_sum", "_mob_sq_sum", "_mob_n", "_agef_sum",
        "_inelig_pop_n", "_pop_n", "_g16_le_gate_n",
    )

    # ---- 性状解码表（基因位 → 行为） --------------------------------
    # 基因位与行为的一一映射就定义在本文件（引擎热路径），不依赖其他模块：
    # g0 移动概率   g1 代谢倍率   g2 繁殖阈值   g3 寿命
    # g4 进食量倍率 g5 饱食度上限 g6 移动能耗倍率 g7 繁殖投入比例
    # g8 光合利用   g9 恒温指数   g10 邻格觅食（g≥0.5 能吃邻格）
    # g11 温度偏好 g12 繁殖冷却（间隔=g×60 tick） g13 群居性（g≥0.5 聚群，<0.5 避群）
    # （完整说明见 MODULES.md「模块二 · 基因表」）

    # D1 neutral_genes 零模型的目标冻结位（C3 修正，2026-09-12）：
    # 只冻结【感知 g14 / 信号 g15】这两个"待检通路"基因，保留 g0–g13 可演化。
    _NEUTRAL_FREEZE_COLS = (int(Gene.PERCEPTION), int(Gene.SIGNAL_STRENGTH))

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        # D-1（F1/R14）：perception_radius 显式校验。
        # 原缺陷：dataclass __post_init__ 的 assert 只在构造时生效，
        # 构造后直接赋值 c.perception_radius = 6 不触发校验；
        # 使用处只判断 ==4，非 4 静默走 8 路径 → radius=6 静默 no-op。
        # 现改为引擎入口处硬校验，非法值一律 ValueError。
        _ifcfg = getattr(config, "info_structure", None)
        if _ifcfg is not None and _ifcfg.enabled:
            if _ifcfg.perception_radius not in (4, 8):
                raise ValueError(
                    f"F1 硬失败：info_structure.perception_radius={_ifcfg.perception_radius} 不合法，"
                    f"只支持 4（Von Neumann）或 8（Moore）。"
                    f"传入其他值会被静默当作 8 处理（no-op），已禁止。"
                )
        # R97 增益校准档（2026-09-16）：C5 v2 三态**在引擎入口再校验一次**。
        # 理由同 D-1（F1 教训）：dataclass 的 `__post_init__` 只在**构造时**生效，
        # 构造后直接 `cfg.oracle.gain_multiplier = 1.3` 不会触发校验 ⇒ 校验必须落在入口。
        _ocfg = getattr(config, "oracle", None)
        if _ocfg is not None and _ocfg.enabled and _ocfg.gain_multiplier != 1.0:
            if not getattr(_ocfg, "is_calibration_arm", False):
                raise ValueError(
                    f"C5 v2 硬失败：oracle.gain_multiplier={_ocfg.gain_multiplier} ≠ 1.0 "
                    f"但未登记 is_calibration_arm=True ⇒ 校准档不得静默混入科学判读"
                    f"（R100 条件 5 / v1.1 §五）。"
                )
            lo, hi = CALIBRATION_M_RANGE
            _mm = float(_ocfg.gain_multiplier)
            if not (_mm == 1.0 or lo <= _mm <= hi):
                raise ValueError(
                    f"C5 v2 硬失败：校准臂 gain_multiplier={_mm} 只允许 1.0（配对基线）"
                    f"或落在 [{lo}, {hi}]（处理臂）⇒ 其余一律失败。"
                )
        # D-19：用 CountingRNG 包一层——**随机流逐位不变**，只统计抽取次数（rng_draws），
        # 供 provenance 机械校验"两条路径/两次重跑是否消费了同一条随机流"（V-1 O-6）。
        self.rng = CountingRNG(np.random.default_rng(config.seed))
        # D2 可复现性：感知噪声/Steels 配对使用全局 np.random（非主 rng），
        # 必须随 config.seed 播种，否则同 seed 两次运行结果不同（中高危可复现性漏洞）。
        np.random.seed(int(config.seed) & 0xFFFFFFFF)
        # 模块三：use_sim_core=True 时把种群数值管线下沉到 Rust（sim_core）
        self._use_sim_core = config.simulation.use_sim_core
        if self._use_sim_core:
            try:
                import sim_core  # 本地扩展，运行时注入
            except ImportError as exc:  # pragma: no cover - 构建问题，非逻辑错误
                raise RuntimeError(
                    "use_sim_core=True 但 sim_core 未安装：请先在 sim_core/ 运行"
                    " `.venv\\Scripts\\python -m maturin develop`"
                ) from exc
            self._sim_core = sim_core
            # Rust 基因索引常量 ↔ simulation.genes 注册表对照：防双写漂移。
            # native_gene_indicators() 返回 [(Rust 语义名, 值), ...]，
            # 任一项与 Python Gene 枚举不一致 ⇒ 引擎初始化直接报错。
            _native = getattr(sim_core, "native_gene_indicators", None)
            if _native is not None:
                _rust_const = {n: int(v) for n, v in _native()}
                _py_const = {g.name: int(g) for g in Gene}
                _drift = [
                    f"sim_core.{n}={v} != Gene.{n}={_py_const[n]}"
                    for n, v in _rust_const.items()
                    if n in _py_const and _py_const[n] != v
                ]
                if _drift:
                    raise RuntimeError(
                        "基因双写漂移：" + "; ".join(_drift)
                        + " —— 请同步 simulation/genes.py 与 sim_core/src/genes.rs"
                    )
        else:
            self._sim_core = None
        self.world = SphereWorld(
            rows=config.world.rows, cols=config.world.cols
        )
        self.light = LightAndTemperature(
            self.world,
            rotation_period=config.light.rotation_period,
            t_equator=config.light.t_equator,
            t_pole=config.light.t_pole,
            day_boost=config.light.day_boost,
            lat_base_ref=config.light.lat_base_ref,
        )
        self.resources = ResourceField(
            self.world, self.light,
            capacity_per_area=config.resources.capacity_per_area,
            regrowth_rate=config.resources.regrowth_rate,
            temp_sensitivity=config.resources.temp_sensitivity,
            distribution=config.resources.distribution,
            patch_count=config.resources.patch_count,
            patch_radius=config.resources.patch_radius,
            patch_capacity_mult=config.resources.patch_capacity_mult,
            patch_regrowth_mult=config.resources.patch_regrowth_mult,
            background_fill=config.resources.background_fill,
            initial_fill=config.resources.initial_fill,
            # 用 config.seed 派生 patch 中心：可复现，且独立 rng 不消费引擎 self.rng
            patch_seed=config.seed,
        )
        # 田字格信号场（L2/L3）：生物可写入/读取 16 种标记模式
        self.signals = SignalField(self.world, duration=50)

        # 预计算统一邻居表（L6 Rust 下沉用）：普通格 8 邻，极点格 cols 邻，
        # 统一到 nb_stride 列，未用位置填 -1。世界不变，只需构建一次。
        n_cells = self.world.n_cells
        nb_stride = max(8, self.world.cols)
        self._nb_table = np.full((n_cells, nb_stride), -1, dtype=np.int64)
        for c in range(n_cells):
            nbs = self.world.neighbors(c)
            self._nb_table[c, :len(nbs)] = nbs

        # ── R146/R149 L2：**strict 2 圈**候选表（CSR；构造期一次）──────────────────
        # 语义：2 跳可达、去掉自身与 1 圈（球面拓扑用**跳数**，**不套 5×5 方窗**；
        #       极点格走 `neighbors()` 的整带语义）。只在 `l2_dash=True` 时被读。
        # `FAR_CAP` 语义 = 「strict 2 圈规模 > FAR_CAP 的格**不可冲刺**」（拓扑退化），
        #   🔴 措辞：排除的是**格**，不是"极区个体"；`[实测]` 不变区间 [16,119] ⇒ 非旋钮。
        self._far_off = np.zeros(n_cells + 1, dtype=np.int64)
        self._far_len = np.zeros(n_cells, dtype=np.int64)
        _rings: list[np.ndarray] = [np.zeros(0, dtype=np.int32) for _ in range(n_cells)]
        _far_cap = int(config.organisms.far_cap)
        for c in range(n_cells):
            one = {int(x) for x in self.world.neighbors(c)}
            two: set[int] = set()
            for d in one:
                two.update(int(x) for x in self.world.neighbors(d))
            two.discard(c)
            two -= one
            if 0 < len(two) <= _far_cap:
                _rings[c] = np.fromiter(sorted(two), dtype=np.int32, count=len(two))
        for c in range(n_cells):
            self._far_len[c] = len(_rings[c])
            self._far_off[c + 1] = self._far_off[c] + self._far_len[c]
        self._far_cells = (np.concatenate(_rings) if self._far_off[-1] else
                           np.zeros(0, dtype=np.int32))
        self._far_excluded_n = int(np.count_nonzero(self._far_len == 0))
        del _rings

        # ── H3（R146 硬约束）：L1/L2 开启时必须 **fail-loud**（本段 = [本地开发] 线）────
        # 为什么不并进 `_d2_asym`：那是**静默换路径**（开开关后悄悄走 Python ⇒ 与"静默
        # no-op"同族）。dash PR 的翻车形态就是"改了代码但实验路径没走到"。
        # ⚠️ L1 的两个开关（`l1_seek`/`l1_fear`）由 B1（所有者线）添加 ⇒ 此处用 getattr 兜底：
        #    字段缺失时**不会**因此跳过 guard（缺失 ⇒ 视为 False，等于 L1 未开）。
        _scfg0 = config.simulation
        _l1_any = bool(getattr(_scfg0, "l1_seek", False)) or bool(getattr(_scfg0, "l1_fear", False))
        if _scfg0.use_sim_core and (_l1_any or bool(getattr(_scfg0, "l2_dash", False))):
            raise NotImplementedError(
                "L1/L2 尚未下沉 Rust：use_sim_core=True 时开启 l1_seek/l1_fear/l2_dash 会"
                "**静默走旧路径**（= dash PR 的翻车形态）⇒ 硬报错。请设 use_sim_core=False。"
            )

        n = config.population.initial_count
        self._flat = np.zeros(n, dtype=np.int64)
        self._energy = np.full(
            n, config.organisms.initial_energy, dtype=np.float64
        )
        self._stomach = np.zeros(n, dtype=np.float64)
        self._genes = self.rng.uniform(
            config.genome.gene_min,
            config.genome.gene_max,
            size=(n, config.genome.gene_count),
        )
        self._age = np.zeros(n, dtype=np.int64)
        self._generation = np.zeros(n, dtype=np.int64)
        self._parent = np.full(n, -1, dtype=np.int64)
        self._repro_cooldown = np.zeros(n, dtype=np.float64)
        self._id = np.arange(n, dtype=np.int64)
        self._next_id = n
        self._max_generation = 0

        # 愉悦度系统（L2）：四数组 + 预测误差驱动
        # _valence: 情绪效价 -1~1（瞬态，RPE 响应）
        # _arousal: 唤醒度 0~1（意外事件→高唤醒）
        # _expectation: (N, 120) 预期表（120 情境，EWMA 学习）
        # _baseline: 基线慢漂移（习惯化）
        pcfg = self.config.pleasure
        self._valence = np.zeros(n, dtype=np.float64)
        self._arousal = np.full(n, 0.5, dtype=np.float64)
        self._baseline = np.zeros(n, dtype=np.float64)
        # 信任度（L5）：对信号的信任，初始 0.5（中性），真信号+假信号-
        self._trust = np.full(n, 0.5, dtype=np.float64)
        # 工作记忆（L5）：4 槽，存食物丰富格子位置（-1=空），round-robin 写入
        self._work_memory = np.full((n, 4), -1, dtype=np.int64)
        self._mem_ptr = 0
        # 信号解读表（L5 文化传递）：(N,16)，对 16 种信号模式的响应倾向
        # 正值=移向，负值=逃避，0=忽略；初始随机，幼体向周围成体学习
        # R113/R121 信号字母表档位（fail-loud：未实施的档位在**构造期**即报错，
        # 不留到运行期 —— 与 :756 的"静默 no-op"教训同族）
        _alpha = str(config.signal_alphabet)
        if _alpha not in SIGNAL_ALPHABET_IMPLEMENTED:
            raise NotImplementedError(
                f"signal_alphabet={_alpha!r} 尚未实施"
                f"（已实现：{SIGNAL_ALPHABET_IMPLEMENTED}）"
            )
        self._alpha = _alpha
        self._alpha_states = SIGNAL_ALPHABET_STATES[_alpha]
        self._alpha_code_max = SIGNAL_ALPHABET_CODE_MAX[_alpha]
        self._interpret = self.rng.normal(0.0, 0.3, size=(n, 16))
        # D2 任意性码本：(N,16)，每个状态(0~15)映射到一个信号模式(0~15)
        # 初始恒等映射 codebook[state]=state（与旧版硬编码行为一致）；
        # 繁殖时遗传+突变，映射可漂移可协商。D2 disabled 时不使用。
        self._codebook = codebook_init_rows(n, config.signal_alphabet)
        # D2 学习瓶颈计数：(N,)，每个个体已完成的观察学习次数；
        # 达到 learning_samples_max 后停止学习（=瓶颈）。D2 disabled 时不使用。
        self._learning_count = np.zeros(n, dtype=np.int32)
        # 乐观初始化：0.8 × max_reward，逼生物探索（预期高→现实可能超预期→愉悦）
        self._expectation = np.full(
            (n, pcfg.expectation_size),
            pcfg.optimism * pcfg.max_reward,
            dtype=np.float64,
        )

        # L10a 果实-种子传播（默认关闭，fruit.enabled=False 时不执行）
        fcfg = self.config.fruit
        self._fruit_grid = np.zeros(self.world.n_cells, dtype=np.float64)  # 每格果实能量
        self._fruit_charge = np.zeros(n, dtype=np.float64)  # 每植物的果实蓄力
        self._seed_carried = np.zeros(n, dtype=np.int32)  # 每动物携带种子数（L10b 完善）

        # ---- D-17/D-18/D-8：按 _id 键控的终身账本（R42/V-7；死亡个体保留行）----
        # 长度恒 = _next_id；出生时 append 零行；**绝不**随死亡压缩（否则丢死亡个体）。
        z = lambda dt: np.zeros(n, dtype=dt)
        self._rs_children = z(np.int32)   # 终身子代数（出生时给亲代 +1）
        self._rs_observed = z(bool)       # ⑤ 首次观测标记
        self._rs_g15 = z(np.float32)      # 观测时 g15
        self._rs_age = z(np.int32)        # 观测时年龄
        self._rs_energy = z(np.float32)   # 观测时能量（⑤ 控制变量）
        self._rs_cc0 = z(np.int32)        # 观测时已生子代数（剩余 = children − cc0）
        self._rs_cohort = z(np.uint8)     # 0=非饱和窗 / 1=饱和窗（诊断）
        self._emit_count = z(np.int32)    # 终身发射次数（oracle 保本记账 + return_ratio 分母）
        self._oracle_gain = z(np.float32)  # 终身已获 oracle 回馈（C-9 保本封顶）
        self._last_sender = np.full(self.world.n_cells, -1, dtype=np.int64)
        self._oracle_transfers = 0.0
        self._oracle_count = 0
        # D-18 ⑥ 探针累计器（纯观测；measure off 时恒零且零开销）
        self._resp_decisions = 0
        self._resp_exposed = 0
        self._resp_delta_sum = 0.0
        self._resp_flip = 0
        self._resp_delta_content_sum = 0.0   # R123：内容项单去的 Δ_content 累计（并列报告）
        self._measure_resp = bool(config.info_structure.measure_signal_response)
        self._oracle_on = bool(config.oracle.enabled)
        # R123/B② 门控臂：逐个体 Δ（**按 _id 键控**——死亡压缩会重排槽位，禁槽位键控）；
        # `_delta_tick_by_id` 记"该 Δ 属于哪一 tick" ⇒ 付款时要求 `== 本 tick`，
        # **免去每 tick 复位**且杜绝"陈旧正值误触发付款"（静默错型）。
        self._delta_full_by_id = z(np.float32)
        self._delta_content_by_id = z(np.float32)
        self._delta_tick_by_id = np.full(n, -1, dtype=np.int32)
        self._gate_on = False
        self._gate_delta = str(config.oracle.gate_delta)
        self._diag_gate_pass = 0
        self._diag_gate_block = 0
        self._gate_arrivals = 0
        self._gate_delta_full_sum = 0.0
        self._gate_delta_content_sum = 0.0
        # ---- 门控臂守卫（三处齐发之一；另两处 = config.__post_init__ / a4 build）----
        if str(config.oracle.gate_mode) == "delta_positive":
            if not self._oracle_on:
                raise ValueError("门控臂需要 oracle.enabled=True（否则根本无付款可闸）")
            if not self._measure_resp:
                raise ValueError(
                    "门控臂需要 info_structure.measure_signal_response=True —— 否则 Δ≡0 "
                    "⇒ **所有付款消失**（ratio=0），会被读成「信号无用」的**假结论**（比没数据更坏）"
                )
            _ifc = config.info_structure
            if not (_ifc.enabled and _ifc.perception_radius == 4 and _ifc.softmax_tau > 0):
                raise ValueError(
                    "门控臂的 Δ 由 ⑥ 探针产生，而探针只存在于**信息不对称路径**"
                    "（enabled ∧ perception_radius=4 ∧ softmax_tau>0）的 softmax 分支里 "
                    "⇒ 该组合下 Δ 恒为 0（同上假结论）"
                )
            if not config.oracle.is_calibration_arm:
                raise ValueError("门控臂是仪器（改付款规则）⇒ 必须登记 is_calibration_arm=True")
            self._gate_on = True
        # D-26a 四环节诊断累计器（纯观测；oracle off 时恒零且不累加）
        self._diag_had_sig = 0        # 移动者落点"有信号"的次数
        self._diag_true_sig = 0       # 其中落点同时"有食物"（oracle 选中的语义）
        self._diag_sel = 0            # 进入 oracle 选择集的接收者次数
        self._diag_attrib_found = 0   # 归因命中：落点登记过发送者 且 发送者仍存活
        self._diag_in_window = 0      # 归因命中 且 在 persistence 窗口内
        self._diag_not_self = 0       # 且 发送者 ≠ 接收者（不自反馈）
        self._diag_budget_ok = 0      # 且 C-9 保本额度 > 0
        self._diag_applied = 0        # 实际成交（转移发生）次数
        # ---- R121 §3.5/§4（2026-09-18）纯观测计数器（零机时；不改状态与随机流）----
        self._diag_food_band_true_sig = 0   # §4.1：true_sig 中落点 food_ratio ∈ (0.3, 0.5]
        self._mem_inherit_n = 0             # §4.2：出生继承记忆的槽位总数
        self._mem_inherit_far_n = 0         # §4.2：其中"距出生格 > 感知半径"的槽位数
        self._alpha_bad_code_n = 0          # §3.5：码 ==0 或 > 档位上限 的发射次数（须恒 0）
        # 2026-09-19（内评 01:4x + 所有者 03:02 §五 建议）：`mem_bit` 实际取值分布
        # E-022/E-023 两份记录均把"该位运行分布无读数"列为**可观测性缺口** ⇒ 加纯观测计数器。
        # `mem_bit_frac = _mem_bit_on / _mem_bit_n`（只在走"状态编码"的发射上累计；
        # random 模式无 state ⇒ 不计入 ⇒ 分母 = 状态编码发射数，报告须标此口径）
        self._mem_bit_on = 0
        self._mem_bit_n = 0
        # A′ 记忆朝向梯度（2026-09-19）：**先证"测到了"再判读**（E-022 的可观测性缺口）
        #   decisions = 走朝向梯度分支的**决策次数**（分母）；slots = 其中"记忆有内容"的次数；
        #   trigger  = 其中"梯度确实产生非 0 贡献"的次数（`|gain| > 1e-9`）。
        # ⚠️ 只在 `memory_gradient="orientation"` 时累计；`"none"` 下全为 0（见 `memory_gradient_stats`）。
        self._mem_grad_dec = 0
        self._mem_grad_slots = 0
        self._mem_grad_trig = 0
        self._mem_grad_counts_valid = True   # Rust 路径下置 False（该侧不产计数 ⇒ 按 n/a 报）
        # 守恒三账审计（R102 条件 1 修正版）
        self._audit_calls = 0
        self._audit_sum_delta = 0.0
        # oracle 对账（三账应恒等）+ 偿付约束量化（内评 §四）
        self._oracle_payer_paid = 0.0
        self._oracle_sender_received = 0.0
        self._oracle_payer_trunc_n = 0
        self._oracle_payer_trunc_amt = 0.0
        self._oracle_budget_trunc_n = 0
        self._oracle_budget_trunc_amt = 0.0
        self._oracle_payer_broke_n = 0
        self._oracle_budget_exhausted_n = 0
        # 接收侧效应（内评 §三 观察项 1）：成交落点 → 下一 tick 实测摄入
        self._recv_pend_cells: list[int] = []
        self._recv_pay_sum = 0.0
        self._recv_food_sum = 0.0
        self._recv_food_n = 0
        self._all_food_sum = 0.0
        self._all_food_n = 0
        if self._oracle_on and self._use_sim_core:
            # C-8：静默忽略会重演 R14"配置看似生效实则没生效" ⇒ 显式报错
            raise RuntimeError(
                "C-8(V-1)：use_sim_core=True 与 oracle.enabled=True 不兼容"
                "（oracle 仅 Python 路径）——请 use_sim_core=False"
            )

        # 出生位置：均匀随机格（不做地形障碍过滤，球面无障碍）
        self._flat = self.rng.integers(0, self.world.n_cells, size=n).astype(
            np.int64
        )

        self._tick = 0
        self._extinct = False
        self._finished = False
        self._history: list[TickStats] = []
        self._history_limit = config.simulation.history_limit
        self._run_born = 0
        self._run_died = 0
        self._run_deaths: Counter = Counter()

        # ---- t=0 基因组摘要（R135 第 -1 步 ③，2026-09-20）--------------------
        # 目的：让"某个基因发生了位移"事后**可归因**。
        #
        # 🔴 实测教训（s42/s45/s47 三 seed × 1200 tick）：**中性基因也会被搭车**
        #    （genetic hitchhiking）。g16 在关捕食后**无直接选择**（消费点全集只有
        #    捕食两处，见 `Gene.AGGRESSION` grep），但它与强选择基因 g4（进食量缩放）
        #    之间存在**由 seed 决定的初始抽样连锁不平衡 LD**：
        #      s42 corr0=+0.171 ⇒ Δg4=+0.246 ⇒ **Δg16=+0.095**
        #      s47 corr0=+0.056 ⇒ Δg4=+0.224 ⇒ Δg16=+0.029
        #      s45 corr0=-0.115 ⇒ Δg4=+0.138 ⇒ Δg16=-0.000
        #    实测 ≈ 2.2 × corr0 × Δg4 × (sd16/sd4)，三 seed 定量一致（符号/单调/比值）。
        # ⇒ **没有 t=0 基线，就无法区分"该基因被选择"与"该基因被搭车"**；
        #   ⇒ 推论：**单个 seed 的基因位移/分布不可跨批比较，必须同 seed 配对**。
        # （只读、不消费 RNG ⇒ 不改变任何既有批的逐位行为。）
        # R141：g16 初始投放（默认空 = 旧行为；非空 ⇒ 覆盖 g16 列，交错分配保证组大小相等）
        _cl = str(getattr(config.genome, "init_g16_clusters", "") or "")
        if _cl:
            _vals = [float(x) for x in _cl.split(",") if x.strip()]
            if _vals:
                # ⚠️ 不消费 RNG（直接赋值）⇒ 只改初始条件，不改随机流结构
                for _i, _v in enumerate(_vals):
                    self._genes[_i::len(_vals), Gene.AGGRESSION] = _v
        self._genome_t0 = _genome_summary(self._genes)
        # R135 第 -1 步④：互捕结构量化累计器（5-bin，g16 ∈ [0,1]）
        self._cannib_atk_hist = np.zeros(CANNIB_BINS, dtype=np.int64)
        self._cannib_prey_hist = np.zeros(CANNIB_BINS, dtype=np.int64)
        # R146/R149 L2 读数（B4；本段 = [本地开发] 线）。**l2_dash 关时全部不累加**
        #   ⇒ `l2_probe()` 返回 None（"未适用"），不是 0（R120/§五.12 口径铁律）。
        self._run_mover_n = 0                 # 本 run 的移动事件数（分母）
        self._run_dash_n = 0                  # 其中走 2 格的（分子）⇒ dash_frac
        self._mover_n_by_box = np.zeros(3, dtype=np.int64)
        self._dash_n_by_box = np.zeros(3, dtype=np.int64)
        self._mob_sum = 0.0                   # Σ mob_eff（行为口径）
        self._mob_sq_sum = 0.0
        self._mob_n = 0
        self._agef_sum = 0.0                  # Σ age_factor
        self._inelig_pop_n = 0                # Σ（处于不可冲刺格的人口）——**人口**口径
        self._pop_n = 0                       # Σ 人口（同时用于 frac(g16≤gate)）
        self._g16_le_gate_n = 0               # Σ 1[g16 ≤ attack_gene_gate]
        # R141 P0：分通道能量记账（5 通道 × 全球/三分箱）。见 `energy_channel_stats`。
        self._ec_global = np.zeros(EC_N, dtype=np.float64)
        self._ec_box = np.zeros((3, EC_N), dtype=np.float64)
        self._ec_n = np.zeros(3, dtype=np.int64)
        # R141 P0（派工单 §1.2）：**逐个体**当 tick 记账缓冲 + 逐 tick 统计时间序列
        self._ec_pt = np.zeros((EC_N, 0), dtype=np.float64)
        self._ec_ts: list[dict] = []
        self._ec_prey_e_sum = 0.0     # Σ 被击杀瞬间的猎物能量（R141 解方程关键输入）
        self._ec_prey_kill_n = 0
        # R145 制度补丁②：能量封顶的**活体探针**（钳制后仍越限者 ⇒ 钳制失效）
        self._frac_over_cap_max = 0.0
        self._over_cap_seen = 0
        # R141/R138：真决斗三级拆分（键预置 ⇒ 输出稳定，不随是否有事件而缺列）
        self._duel = Counter({k: 0 for k in DUEL_KEYS})

    def cannibalism_stats(self) -> dict:
        """互捕结构量化（R135 第 -1 步④；F「去互捕」的前置证据）。

        `delta_g16 = mean(攻击者 g16) − mean(猎物 g16)`：
        `> 0` ⇒ 鹰吃鸽有结构；`≈ 0` ⇒ 随机互捕。**n=0 时返回 n_a=None（不是 0）**——
        （把"没测到"写成"测到 0"是本项目的老坑，见 E-023 的 `ratio` 教训）。
        """
        a, p = self._cannib_atk_hist, self._cannib_prey_hist
        na, npr = int(a.sum()), int(p.sum())
        centers = (np.arange(CANNIB_BINS) + 0.5) / CANNIB_BINS
        ma = float((centers * a).sum() / na) if na else None
        mp = float((centers * p).sum() / npr) if npr else None
        # R141/R138：真决斗三级拆分（Python 路径才有；Rust 路径计数全 0 ⇒ 报 None）
        duel = None
        if not self._use_sim_core:
            nominal = int(na)          # = len(attackers) 累计（与 n_attacks 同源）
            real = int(self._duel["real_attempts"])
            kills = int(self._duel["kills"])
            duel = {
                "nominal_attackers": nominal,
                "skip_no_energy": int(self._duel["skip_no_energy"]),
                "skip_no_prey": int(self._duel["skip_no_prey"]),
                "skip_already_eaten": int(self._duel["skip_already_eaten"]),
                "real_attempts": real,
                # 🔴 两个成功率口径**必须分开报**：旧口径的低估是真实存在的
                "success_rate_nominal": (round(kills / nominal, 4) if nominal else None),
                "success_rate_real": (round(kills / real, 4) if real else None),
                "note": "nominal=含未出手；real=真出手（排除能量不足/无猎物/已被吃）"
                        "⇒ 旧口径 kills/nominal 会**低估**成功率",
            }
        return {
            "duel": duel,
            "bins": CANNIB_BINS,
            "n_attacks": na if na else None,
            "n_kills": npr if npr else None,
            "attacker_g16_hist": a.tolist(),
            "prey_g16_hist": p.tolist(),
            "attacker_g16_mean": round(ma, 4) if ma is not None else None,
            "prey_g16_mean": round(mp, 4) if mp is not None else None,
            "delta_g16": (round(ma - mp, 4) if (ma is not None and mp is not None) else None),
            "note": "g16 区间 [0,1] 均分 5 bin；n=None 表示本批没有该事件（不可读作 0）",
        }

    def _ec_ensure(self, n: int) -> None:
        """确保逐个体记账缓冲够长（P 会随生死/繁殖变化）。"""
        if self._ec_pt.shape[1] < n:
            pad = np.zeros((EC_N, n - self._ec_pt.shape[1]), dtype=np.float64)
            self._ec_pt = np.concatenate([self._ec_pt, pad], axis=1)

    def _ec_add(self, k: int, idx, amount) -> None:
        """逐个体记账（R141 P0；**写当 tick 缓冲**，tick 末由 `_ec_flush` 统计后清零）。

        `idx=None` ⇒ 对**全体**当前存活者记账；否则按 `idx` 子集记账。
        """
        a = np.asarray(amount, dtype=np.float64)
        if a.size == 0:
            return
        self._ec_ensure(a.size if idx is None else int(np.max(idx)) + 1)
        if idx is None:
            self._ec_pt[k, :a.size] += a
        else:
            # `np.add.at` 对重复索引安全（移动/捕食的 idx 理论上唯一，但不赌）
            np.add.at(self._ec_pt[k], np.asarray(idx, dtype=np.int64), a)

    def _ec_flush(self) -> None:
        """tick 末：按 g16 三分箱统计净收入（n / mean / p50 / var）⇒ 时间序列；然后清零。

        🔴 口径与列名由派工单锁定（`docs/tasks/派工-本地开发-P0能量记账与投放-20260921.md` §1.2/§1.3）：
          `net = intake_forage + intake_pred − cost_meta − cost_move − cost_attack`
          **不含光合**（光合是可选单列）；样本量 n=0 ⇒ 统计量写 **None 不写 0**。
        ⚠️ 为什么要**逐 tick**统计而不是窗口合并：时间自相关 ⇒ 合并会伪重复（内评 01:16 §1.3③）。
        """
        # ⚠️ P 必须取**各数组的最小一致长度**：繁殖后 `_genes` 会比 `_id` 先扩容，
        #    直接传 `len(self._id)` 会造成 `net[m]` 与 `box` 长度不匹配（IndexError，已踩）。
        P = int(min(self._genes.shape[0], self._energy.shape[0],
                    self._id.shape[0], self._ec_pt.shape[1]))
        if self._use_sim_core or P <= 0:
            return
        pt = self._ec_pt[:, :P]
        net = (pt[EC_FORAGE] + pt[EC_PRED]
               - pt[EC_META] - pt[EC_MOVE] - pt[EC_ATTACK])
        box = np.clip((self._genes[:P, Gene.AGGRESSION] * 3).astype(np.int64), 0, 2)
        row: dict = {}
        for b, name in enumerate(EC_BOX_NAMES):
            m = box == b
            n = int(m.sum())
            row[f"g_{name}_n"] = n
            if n == 0:
                for k in ("net_mean", "net_p50", "net_var"):
                    row[f"g_{name}_{k}"] = None
                continue
            v = net[m]
            row[f"g_{name}_net_mean"] = float(v.mean())
            row[f"g_{name}_net_p50"] = float(np.percentile(v, 50))
            row[f"g_{name}_net_var"] = float(v.var(ddof=1)) if n > 1 else None
            # 附：含光合的净收入（超出派工单规格的**附加**信息，供完整性核对用）
            row[f"g_{name}_net_incl_photo"] = float((v + pt[EC_PHOTO][m]).mean())
        row["forage_in_mean"] = float(pt[EC_FORAGE].sum() / P)
        row["pred_in_mean"] = float(pt[EC_PRED].sum() / P)
        row["photo_in_mean"] = float(pt[EC_PHOTO].sum() / P)
        row["mean_energy"] = float(self._energy[:P].mean())
        row["prey_energy_mean"] = (self._ec_prey_e_sum / self._ec_prey_kill_n
                                   if self._ec_prey_kill_n else None)
        self._ec_ts.append(row)
        # 累计进 summary 用的累加器（窗口无关的全批口径）
        self._ec_global[:] += pt.sum(axis=1)
        for b in range(3):
            m = box == b
            if m.any():
                self._ec_box[b, :] += pt[:, m].sum(axis=1)
                self._ec_n[b] += int(m.sum())
        self._ec_pt[:, :P] = 0.0

    def ec_timeseries(self) -> list[dict]:
        """逐 tick 的净收入统计序列（CSV 16 列的来源；派工单 §1.3 列名）。"""
        return self._ec_ts

    def energy_ledger(self) -> dict:
        """`energy_ledger` 段（summary；派工单 §1.3 结构，**列名勿改**——`calib_solve.py` 按此消费）。

        ⚠️ `use_sim_core=True` ⇒ `path="rust"` 且无数据（**未观测**，不是 0）。
        """
        if self._use_sim_core:
            return {"path": "rust",
                    "note": "Rust 路径不做 Python 侧记账 ⇒ 未观测（None，非 0）"}
        g = {EC_NAMES[i] + "_sum": round(float(self._ec_global[i]), 3) for i in range(EC_N)}
        groups = {}
        for b, name in enumerate(EC_BOX_NAMES):
            n = int(self._ec_n[b])
            groups[name] = {
                "n_sum": n,
                **{EC_NAMES[i] + "_sum": round(float(self._ec_box[b, i]), 3)
                   for i in range(EC_N)},
            }
            # 净收入（**按派工单口径，不含光合**）+ 含光合的附加量
            groups[name]["net_sum"] = round(float(
                self._ec_box[b, EC_FORAGE] + self._ec_box[b, EC_PRED]
                - self._ec_box[b, EC_META] - self._ec_box[b, EC_MOVE]
                - self._ec_box[b, EC_ATTACK]), 3)
            groups[name]["net_incl_photo_sum"] = round(float(
                groups[name]["net_sum"] + self._ec_box[b, EC_PHOTO]), 3)
        return {
            "path": "python",
            "obs_count": int(self._ec_n.sum()),
            "global": g,
            "groups": groups,
            "prey": {"kills": int(self._ec_prey_kill_n),
                     "energy_sum": round(float(self._ec_prey_e_sum), 3)},
            "attack": {"attempts": int(self._duel["real_attempts"])},
            "note": "net = intake_forage + intake_pred − cost_meta − cost_move − cost_attack"
                    "（**不含光合**，派工单 §1.2）；net_incl_photo_* 为附加口径",
        }

    def state_bounds_check(self) -> dict:
        """状态量边界自检（R145 制度补丁③：Pre-Flight 增"状态量边界检查"）。

        🔴 **R147 §二 发现 1 修正（2026-09-21 22:10 裁定）**：关档时**不得报 0**。
            钳制关闭 ⇒ `energy ≤ max_energy` 这条不变量**根本不存在** ⇒ 检查**未执行**
            ⇒ 必须报 **`None`（"没测"）**，不是 `0`（"测出零"）。否则有人只读
            `cap_residual_n = 0` 会得出"能量没问题"，而同一 run 的 CSV 明写
            **88.7% 个体超限** —— 这正是 R120 纪律的同族
            （`mem_bit_frac` 非 `"8"` 档返回 `None` 而非 0 的先例）。
            同时给出 `energy_checked` 布尔，让"未检查"是**显式**的而不是靠推断。

        ⚠️ **命名纪律（R147 §二 发现 2）**：本函数的 `cap_residual_n` 查的是**仪器**
            （钳制后仍越限？= 钳制失效报警），CSV 的 `over_cap_frac` 查的是**现象**
            （囤积规模）。两者**正交**，故名字必须区分（旧名 `energy_over_cap` 与
            `frac_over_cap` 会被读成同一个东西）。
        """
        P = len(self._id)
        ocfg = self.config.organisms
        cap_on = bool(ocfg.energy_cap_enabled)
        out = {"n": int(P),
               "cap_residual_n": None,        # 关档 ⇒ None（**未检查**，非 0）
               "energy_checked": cap_on,
               "stomach_over_eat_cap": 0, "stomach_over_pred_cap": 0, "age_negative": 0,
               "energy_cap_enabled": cap_on}
        if P == 0:
            out["note"] = "P=0：无可检个体"
            return out
        if cap_on:
            out["cap_residual_n"] = int(
                np.count_nonzero(self._energy[:P] > ocfg.max_energy))
        # 🔴 自检第一个战果（2026-09-21，`state_bounds_check` 首跑即抓到）：
        #    **胃容量在两个路径上口径不同且从未对齐** ——
        #      进食路径（:859）        `max_energy/eat_eff × 0.5 × cap_mult`（均值 ≈62.5）
        #      捕食路径（:1737 / Rust `predation.rs:43`）`max_energy/eat_eff`（=100）
        #    ⇒ 捕食转移后胃粮可**超过**进食路径的容量（最多约 1.6 倍）。
        #    这是**既有行为**（Python 与 Rust **互相一致** ⇒ 不是双路径漂移，而是两处语义并存）
        #    ⇒ 本函数**如实报两个口径**，不擅自改行为（改了就是新纪元）。
        cap_mult = 0.5 + self._genes[:P, Gene.STOMACH_CAP] * 1.5
        stomach_cap = ocfg.max_energy / max(1e-9, ocfg.eat_efficiency) * 0.5 * cap_mult
        stomach_cap_pred = ocfg.max_energy / max(1e-9, ocfg.eat_efficiency)
        out["stomach_over_eat_cap"] = int(
            np.count_nonzero(self._stomach[:P] > stomach_cap + 1e-9))
        out["stomach_over_pred_cap"] = int(
            np.count_nonzero(self._stomach[:P] > stomach_cap_pred + 1e-9))
        out["age_negative"] = int(np.count_nonzero(self._age[:P] < 0))
        out["note"] = (
            "**字段语义铁律（R148 §五.12）：`None` ⟺ 未检查；`0` ⟺ 检查过且合规。**"
            "cap_residual_n = **仪器**（钳制后仍越限 ⇒ 失效报警；关档为 None=未检查）；"
            "囤积规模见 `over_cap_frac()`（**现象**，两者正交）。"
            "stomach_over_eat_cap / stomach_over_pred_cap 是**两条独立容量**"
            "（进食口径 ×0.5×cap_mult ≈62.5 ｜ 捕食口径 max_energy/eat_eff =100；"
            "见 `OrganismConfig` 的「两条独立胃容量」声明）——"
            "两口径不同是**既有行为**，不代表双路径漂移；命名已按内评 §一.1 改为自解释"
            "（`*_eat_cap` / `*_pred_cap`，勿再读成「唯一上限」）")
        return out

    def energy_cap_probe(self) -> dict:
        """能量封顶的活体探针读数（R145 补丁②）。

        `cap_residual_frac` **应恒 0**（开档时）。⚠️ 它是**仪器**健康度
        （钳制后仍越限的比例），**不是**囤积规模 —— 后者见 `over_cap_frac()`
        （R147 §二 发现 2：两个"over_cap"语义正交，名字必须区分）。
        """
        return {"enabled": bool(self.config.organisms.energy_cap_enabled),
                "cap_residual_frac": round(float(self._frac_over_cap_max), 8),
                "over_cap_seen": int(self._over_cap_seen),
                "note": "仪器口径：钳制**之后**统计 ⇒ 开档正常恒 0；非 0 即钳制失效（报警）。"
                        "囤积**现象**规模（含关档时的 0.8873 那类读数）见 over_cap_frac()"}

    def over_cap_frac(self) -> float | None:
        """囤积**现象**：当前个体中 `energy > max_energy` 的占比（R147 §二 发现 2）。

        与 `energy_cap_probe().cap_residual_frac`（**仪器**）正交：
          · 本方法 = "囤积有多普遍"（关档时即 88.7% 那类读数）
          · 那个   = "钳制有没有生效"（关档时无意义）
        与 `state_bounds_check().cap_residual_n` 的关系：两者分子同源（超限个体数），
          但**分母/时点不同** —— 本方法按**当刻** P 归一，且**不**随开关变语义。
        `P=0` ⇒ **None**（未观测，非 0）。
        """
        P = len(self._id)
        if P == 0:
            return None
        n_over = int(np.count_nonzero(
            self._energy[:P] > self.config.organisms.max_energy))
        return round(n_over / P, 8)


    def l2_probe(self) -> dict | None:
        """L2 机动层读数（R150 B4；本段 = [本地开发] 线）。

        🔴 `l2_dash=False` ⇒ **None（未适用）**，**不是 0**（R120 / §五.12 口径铁律：
        "没测" ≠ "测出零"）。所有分母都写进 `note`，便于独立复算。
        """
        if not bool(self.config.simulation.l2_dash):
            return None
        n_mov = int(self._run_mover_n)
        n_dash = int(self._run_dash_n)
        mob_mean = (self._mob_sum / self._mob_n) if self._mob_n else None
        mob_var = ((self._mob_sq_sum / self._mob_n) - mob_mean ** 2) if self._mob_n else None
        boxes = ("herb", "omni", "carn")
        return {
            "dash_frac": (round(n_dash / n_mov, 6) if n_mov else None),
            "mover_n": n_mov,
            "dash_n": n_dash,
            "dash_frac_by_g16_box": {
                boxes[r]: (round(int(self._dash_n_by_box[r]) / int(self._mover_n_by_box[r]), 6)
                           if self._mover_n_by_box[r] else None)
                for r in range(3)},
            "mob_eff_mean": (round(float(mob_mean), 6) if mob_mean is not None else None),
            "mob_eff_std": (round(float(np.sqrt(max(0.0, mob_var))), 6)
                            if mob_var is not None else None),
            "age_factor_mean": (round(self._agef_sum / self._mob_n, 6) if self._mob_n else None),
            "nondash_cell_pop_frac": (round(self._inelig_pop_n / self._pop_n, 6)
                                      if self._pop_n else None),
            "frac_g16_le_gate": (round(self._g16_le_gate_n / self._pop_n, 6)
                                 if self._pop_n else None),
            "far_cap": int(self.config.organisms.far_cap),
            "dash_cost_exp": float(self.config.organisms.dash_cost_exp),
            "dash_ineligible_cells": int(self._far_excluded_n),
            "dash_eligible_cells": int(self.world.n_cells) - int(self._far_excluded_n),
            "note": "dash_frac 分母 = **移动事件数**（Σ 每 tick `mover_n`，非人口·tick）；"
                    "mob_eff = g18(DEFENSE→MOBILITY) × age_factor；"
                    "nondash_cell_pop_frac = Σ(处于不可冲刺格的人口)/Σ人口；"
                    "frac_g16_le_gate = Σ1[g16 ≤ attack_gene_gate]/Σ人口（**判读前置门**用）",
        }

    def genome_t0_stats(self) -> dict:
        """t=0 基因组摘要（`_genome_t0` 的副本；跨批次 irreducible 的地基凭证）。

        含 `corr_g16_g4`——**搭车诊断的关键量**：事后可直接解释每个 seed 的 g16 位移。
        """
        return dict(self._genome_t0)

    # ---- 只读状态（给观察者用） ------------------------------------------

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def rng_draws(self) -> int:
        """主 RNG 累计抽取次数（D-19 / V-1 O-6）。

        用于 provenance：同 seed 同配置的两条路径/两次重跑，`rng_draws` 必须相同，
        否则就是消费了不同的随机流（可复现性/对拍出问题的机械信号）。
        ⚠️ 只统计【主 rng】；D2 的感知噪声走全局 `np.random`（已知缺陷 F-D2），不计入。
        """
        return int(self.rng.draws)

    def alive_count(self) -> int:
        return len(self._id)

    def population_view(self) -> PopulationView:
        return PopulationView(
            tick=self._tick,
            alive_count=len(self._id),
            max_generation=self._max_generation,
            total_energy=float(self._energy.sum()),
            total_food_in_stomach=float(self._stomach.sum()),
        )

    @property
    def extinct(self) -> bool:
        return self._extinct

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def history(self) -> list[TickStats]:
        return self._history

    @property
    def total_born(self) -> int:
        """累计出生数（与 history_limit 无关，全局计数）。"""
        return self._run_born

    @property
    def total_died(self) -> int:
        """累计死亡数（与 history_limit 无关，全局计数）。"""
        return self._run_died

    def death_cause_totals(self) -> Counter:
        """累计死因分布（与 history_limit 无关，全局计数）。"""
        return self._run_deaths.copy()

    # ---- 主循环 ----------------------------------------------------------

    def step(self) -> TickStats:
        """推进一个 tick，返回统计快照。与旧引擎的 step() 对齐。

        注意：step() 会先推进内部时钟（_tick += 1）再执行整套流程，
        所以"手动逐 tick 调度"与 run() 的行为完全一致（环境时间都会前进）。
        """
        if self._finished:
            raise RuntimeError("模拟已结束，无法继续 step()")
        self._tick += 1
        stats = self._advance_one_tick()
        self._populate_history(stats)
        return stats

    def _advance_one_tick(self) -> TickStats:
        # 顺序契约：先资源再生，再种群行动
        # uniform：Rust regrow（3.2）；patchy：Rust regrow_patchy（L7e，含空间倍率守恒）。
        #   旧版 .pyd 没有 regrow_patchy 时特性检测回退 Python，保证双路径不炸。
        if self._use_sim_core and self.resources.distribution == "uniform":
            # 3.4：资源再生长沉到 Rust（与 ResourceField.regrow 逐位等价，
            # 默认 temp_sensitivity=1.0 时严格一致；≠1 有 ≤1-ULP 差异）
            self._sim_core.regrow(
                self.resources._grid, self.resources._capacity,
                self.light.temperature(
                    np.arange(self.world.n_cells), self._tick
                ),
                self.resources.regrowth_rate,
                self.resources.temp_sensitivity,
            )
        elif (
            self._use_sim_core
            and self.resources.distribution == "patchy"
            and hasattr(self._sim_core, "regrow_patchy")
        ):
            # L7e：patchy 再生下沉 Rust（斑块格×patch_mult / 背景格×bg_mult，守恒）
            self._sim_core.regrow_patchy(
                self.resources._grid, self.resources._capacity,
                self.light.temperature(
                    np.arange(self.world.n_cells), self._tick
                ),
                self.resources._patch_mask.astype(np.uint8),
                self.resources._patch_regrowth_mult,
                self.resources._bg_regrowth_mult,
                self.resources.regrowth_rate,
                self.resources.temp_sensitivity,
            )
        else:
            self.resources.regrow(self._tick)
        # 信号场时间推进（标记衰减、过期清零）
        self.signals.tick()
        born, died, deaths = self._step_population()
        # 能量封顶（R144；`7516ba9` 引入 → 2026-09-21 补开关/测试/冒烟/纪元声明）
        # 关（默认）⇒ **与 E-017~E-031/calib1 逐位一致**（旧纪元）；开 ⇒ 新纪元（禁跨比）。
        if self.config.organisms.energy_cap_enabled:
            _cap = self.config.organisms.max_energy
            self._energy = np.minimum(self._energy, _cap)
            # R145 制度补丁②：**活体探针** —— 钳制之后若仍有越限者，说明钳制失效。
            # 正常应恒 0；非 0 即报警（比"再写一条测试"更能覆盖真实运行路径）。
            _P = len(self._id)
            if _P:
                _over = int(np.count_nonzero(self._energy[:_P] > _cap))
                self._frac_over_cap_max = max(self._frac_over_cap_max,
                                              _over / float(_P))
                self._over_cap_seen += _over
        # D-17 ⑤：每 tick 末给"首次进入窗口"的存活个体记观测（含死亡个体靠 _id 账本留存）
        if self._measure_resp:
            self._observe_selection_cohort()
        return TickStats(
            tick=self._tick,
            population=len(self._id),
            born=born,
            died=died,
            deaths_by_cause=deaths,
            total_energy=float(self._energy.sum()),
            total_resource=self.resources.total(),
        )

    def run(self, ticks: Optional[int] = None) -> list[TickStats]:
        """一直跑到 ticks 或自然结束，返回历史统计。"""
        limit = self.config.simulation.ticks if ticks is None else ticks
        while not self._finished and self._tick < limit:
            self.step()
            self._finished = self._end_condition_met()
        return list(self._history)

    def _populate_history(self, stats: TickStats) -> None:
        self._history.append(stats)
        # 全局累计（与 history_limit 无关：观察台/存档用）
        self._run_born += stats.born
        self._run_died += stats.died
        self._run_deaths.update(stats.deaths_by_cause)
        # R141 P0：tick 末结算分通道记账（派工单 §1.2：**逐 tick** 出 n/mean/p50/var
        # ⇒ 避免"跨 tick 合并样本"造成的时间自相关伪重复，内评 01:16 §1.3③）
        self._ec_flush()
        if self._history_limit > 0:
            overflow = len(self._history) - self._history_limit
            if overflow > 0:
                del self._history[:overflow]

    def _end_condition_met(self) -> bool:
        if self._extinct and self.config.simulation.stop_on_extinction:
            return True
        return self._tick >= self.config.simulation.ticks

    # ---- 数值平衡：寿命与昼夜 ---------------------------------------------

    def _lifespan(self, g3: NDArray[np.float64]) -> NDArray[np.float64]:
        """寿命（tick）= 一昼夜 × lifespan_mult × (1 + g3×7)。

        用"昼夜长度"做基准而非绝对 tick 数：昼夜设置（rotation_period）
        调整时，寿命自动跟着缩放，保证生物始终能经历若干完整昼夜。
        g3=0 最少活满一昼夜（对昼夜有反应）；g3=1 最多八昼夜（慢热长寿）。
        lifespan_mult（战役参数）：不动昼夜，直接缩放寿命基准→世代缩短。
        """
        day = float(self.config.light.rotation_period)
        mult = float(self.config.organisms.lifespan_mult)
        return day * mult * (1.0 + g3 * 7.0)

    # ---- 单 tick 种群推进（核心热循环） -----------------------------------

    def _step_population(self) -> tuple[int, int, Counter]:
        """整群推进一步。返回 (born, died, deaths)。"""
        ocfg, gcfg = self.config.organisms, self.config.genome
        ccfg = self.config.culture          # 隐式选择压参数化（A2）：信任学习幅值
        P = len(self._id)
        if P == 0:
            self._extinct = True
            return 0, 0, Counter()

        # 温度相关量（一次算出全种群的那份，避免反复调用）
        activity = self.light.activity_factor(self._flat, self._tick)

        genes = self._genes[:P]
        energy = self._energy[:P]
        stomach = self._stomach[:P]
        # D1 neutral_genes：读取侧冻结——只冻结【目标位 g14 感知 / g15 信号】，
        # 其余 g0–g13 照常演化。写入侧（繁殖继承/突变）照常，不碰 RNG 顺序。
        # ⚠️ C3 修正（2026-09-12）：原实现 `genes[:] = 0.5` 把**全部**基因冻结为 0.5，
        # 连移动/代谢/繁殖等行为位也定型 → 种群灭绝（N=2~5），不是有效零模型。
        # 零模型的目的 = 断开"感知/信号"这条待检通路，其余行为生态保持不变。
        if self.config.neutral_genes:
            genes[:, self._NEUTRAL_FREEZE_COLS] = 0.5
        # 愉悦度：记录 tick 开始时的能量（步骤 1~8 会修改 energy）
        energy_before = energy.copy() if self.config.pleasure.enabled else None

        # 0) 个体化温度响应（恒温基因 g9 + 温度偏好基因 g11）：
        #    冷血（g9=0）完全随环境：低温时消化慢、行动贵；
        #    恒温（g9=1）内部温度恒定：低温不再压制，但每 tick 多扣维持费。
        homeo = genes[:, Gene.HOMEOTHERM]
        eff_activity = activity + homeo * (1.0 - activity)
        #    温度偏好（g11）：冷血个体只在自己偏好的温度附近才满速，
        #    偏离越远越慢（偏离 15° 活力掉到 ≈6 成）；恒温者不受影响。
        #    偏好温度 = -10 + g11×50，落在 [-10°C, 40°C] 覆盖全地图谱系。
        pref_temp = -10.0 + genes[:, Gene.TEMP_PREF] * 50.0
        grid_temp = self.light.temperature(self._flat, self._tick)
        niche_match = np.exp(-0.5 * ((grid_temp - pref_temp) / 15.0) ** 2)
        eff_activity = np.where(
            homeo >= 0.5,
            eff_activity,
            eff_activity * (0.4 + 0.6 * niche_match),
        )
        cold_penalty = 2.0 - eff_activity  # 冷 => 活动代价高（恒温者不受影响）

        # 1)~3) 光合 / 代谢 / 维持 —— 双路径（Rust stage1 逐位等价，或 Python 原实现）
        day_len = float(self.config.light.rotation_period)
        # lifespan_mult：Rust 零改动方案——把 day_len 参数传 day_len×mult，
        # Rust 内部用 day_len 推导成熟/老年年龄，等效压缩寿命，Rust 无感、ABI 不变。
        _day_len_eff = day_len * float(self.config.organisms.lifespan_mult)
        if self._use_sim_core:
            self._sim_core.step_vectors_stage1(
                energy, stomach, genes, eff_activity,
                self.light.illumination(self._flat, self._tick),
                self._age[:P],
                ocfg.photo_max, ocfg.base_metabolism, ocfg.eat_efficiency,
                ocfg.growth_mult, ocfg.senile_mult,
                ocfg.maturity_fraction, ocfg.senile_fraction,
                ocfg.homeo_upkeep, _day_len_eff,
            )
        else:
            # 1) 光合收入（g8）：少量、随光照。
            #    收入 = 光照(所在格,当前tick) × g8 × photo_max。
            #    植物化增强（g19）在 stage1 之后统一补，确保双路径一致。
            _pho = (
                self.light.illumination(self._flat, self._tick)
                * genes[:, Gene.PHOTOSYNTHESIS]
                * ocfg.photo_max
            )
            energy += _pho
            self._ec_add(EC_PHOTO, None, _pho)   # R141 P0：intake_photo 通道

            # 2) 代谢转化：胃 → 能量（总量不变，快慢受温度+基因影响）
            #    转化速率 = base_metabolism × metabolic_mult × eff_activity
            #    metabolic_mult = 0.5 + g1×1.5（基因放大代谢快慢）
            metab_mult = 0.5 + genes[:, Gene.METABOLIC] * 1.5
            digest_rate = ocfg.base_metabolism * metab_mult * eff_activity
            # 每 tick 最多转化这么多；不得超出胃里有的
            digest = np.minimum(stomach, digest_rate)
            _dg = digest * ocfg.eat_efficiency      # 同样食物 → 同样能量
            energy += _dg
            # R141 P0：`intake_forage` = **真正进入能量的量**（已乘 `eat_efficiency`）。
            # 🔴 口径坑（我第一版就踩了）：若记"进胃的原始食物量"（未乘 3.0），
            #    `net = 收入 − 支出` 会**虚假为负**（实测 −0.23/人·tick，而个体显然活着）
            #    ⇒ 必须与 `intake_pred`/`intake_photo`（都直接进能量）同口径。
            self._ec_add(EC_FORAGE, None, _dg)
            stomach -= digest

            # 3) 基础维持消耗（体温/活动，必扣，与温度无关基础价）
            #    随年龄的"需求"阶段：幼体在长身体（growth_mult 倍）→
            #    成年（1 倍）→ 老年器官退化（senile_mult 倍）。
            #    成熟/老年年龄按各自寿命（g3）的比例划分，寿命长的物种成熟和老化都更晚。
            life_span = self._lifespan(genes[:, Gene.LIFE_GENE])
            age_f = self._age[:P].astype(np.float64)
            maturity_age = ocfg.maturity_fraction * life_span
            senile_age = ocfg.senile_fraction * life_span
            age_mult = np.ones(P, dtype=np.float64)
            age_mult[age_f < maturity_age] = ocfg.growth_mult
            age_mult[age_f >= senile_age] = ocfg.senile_mult
            _cm = ocfg.base_metabolism * metab_mult * age_mult
            energy -= _cm
            # R141 P0：cost_meta 通道（基础维持；恒温费下一行另计，合并进同一通道）
            self._ec_add(EC_META, None, _cm)
            #    + 恒温维持费（g9）：恒温个体每 tick 另付 homeo_upkeep
            _ch = ocfg.homeo_upkeep * homeo
            energy -= _ch
            self._ec_add(EC_META, None, _ch)   # R141：并入 cost_meta 通道

        # 3.5) 植物化光合增强（g19）：统一在 stage1 之后补，确保 Rust/Python 双路径一致
        #     扎根个体（g19 高）额外获得 光照×g8×g19×photo_max 的光合收入
        _pho2 = (
            self.light.illumination(self._flat, self._tick)
            * genes[:, Gene.PHOTOSYNTHESIS]
            * genes[:, Gene.ROOTING]
            * ocfg.photo_max
        )
        energy += _pho2
        self._ec_add(EC_PHOTO, None, _pho2)   # R141 P0：植物化增强并入 intake_photo

        # 3.5) L10a 植物蓄力→结果（默认关闭；确定性数值管线，Rust 可下沉）
        if self.config.fruit.enabled:
            self._step_fruit_charge(P, genes)

        # 4) 进食：从格子里吃进胃（先吃后扣基础维持，保证当天能吃到）
        #    饱食度：胃容量上限（基础 = max_energy/eat_efficiency/2，g5 缩放 0.5~2 倍）
        #    进食量：每 tick 最多 eat_amount（g4 缩放 0.5~1.5 倍）
        # 内评 §三 观察项 1：取出上一 tick 的 oracle 成交落点（**只消费一次**，保证
        # 「t 付款 → t+1 在落点进食」的时序对应；无 oracle 时零开销）
        pend_cells = self._recv_pend_cells
        if self._oracle_on:
            self._recv_pend_cells = []
        eat_mult = 0.5 + genes[:, Gene.EAT_AMOUNT] * 1.0
        cap_mult = 0.5 + genes[:, Gene.STOMACH_CAP] * 1.5
        stomach_cap = (
            ocfg.max_energy / max(1e-9, ocfg.eat_efficiency) * 0.5 * cap_mult
        )
        if (stomach < stomach_cap).any():
            eaters = np.flatnonzero(stomach < stomach_cap)
            # R135 第 3 步 A-连续（2026-09-20）：**凸 trade-off** `forage_mult = (1−g16)^k`。
            #   k=0（默认）⇒ 下面的乘子恒 1 ⇒ **与旧版逐位一致**；
            #   k>1 ⇒ g16 高的个体**吃斑块的能力加速下降**（中间态杂食者最吃亏）。
            #   只动取食侧（捕猎侧在 Rust，按裁定不动）⇒ **无需改 Rust、无需重编**。
            #   详见 `PredationConfig.forage_tradeoff_k` 的对照表与文献锚。
            # ⚠️ 此处**不能**沿用 `pcfg` 简写：本函数里 `pcfg` 指向的是
            # `config.pleasure`（:447），捕食段那份在 :1374 才定义 ⇒ 必须全路径取。
            _tk = float(self.config.predation.forage_tradeoff_k)
            if _tk > 0.0:
                _g16 = np.clip(genes[eaters, Gene.AGGRESSION], 0.0, 1.0)
                _forage = np.power(1.0 - _g16, _tk)
            else:
                _forage = 1.0
            want = np.minimum(
                ocfg.eat_amount * eat_mult[eaters] * _forage,
                stomach_cap[eaters] - stomach[eaters],
            )
            # 3.5：批量进食双路径（Rust consume_many 与 numpy consume_many 逐位等价）
            if self._use_sim_core:
                taken = np.empty(len(eaters), dtype=np.float64)
                self._sim_core.consume_many(
                    self.resources._grid, self._flat[eaters], want, taken
                )
            else:
                taken = self.resources.consume_many(self._flat[eaters], want)
            stomach[eaters] += taken
            # 内评 §三 观察项 1（接收侧净能量效应）：**实测**成交落点上的摄入，并同时
            # 记「全体进食者」的平均摄入作基线（= "不通信者"的参照）。
            # ⚠️ 按 **cell** 键控（非个体）：同格多人时会把他们的摄入一并计入 ⇒ 属近似，
            #    报告口径须写明（见 receiver_side_effects 的 note）。
            if self._oracle_on:
                self._all_food_sum += float(taken.sum())
                self._all_food_n += int(taken.size)
                if pend_cells:
                    pc = np.unique(np.asarray(pend_cells, dtype=np.int64))
                    hit = np.isin(self._flat[eaters], pc)
                    if hit.any():
                        self._recv_food_sum += float(taken[hit].sum())
                        self._recv_food_n += int(hit.sum())
            # 邻格觅食（g10）：自己格不够吃的个体，随机吃一格外邻格
            short = want - taken
            hung = np.flatnonzero(short > 1e-9)
            hunter = self._genes[hung, Gene.FORAGE_NEIGHBOR] >= 0.5
            if hunter.any():
                hf = hung[hunter]
                targets = np.empty(int(hunter.sum()), dtype=np.int64)
                for j, fi in enumerate(hf):
                    nb = self.world.neighbors(int(self._flat[fi]))
                    targets[j] = nb[int(self.rng.integers(0, len(nb)))]
                if self._use_sim_core:
                    taken2 = np.empty(len(hf), dtype=np.float64)
                    self._sim_core.consume_many(
                        self.resources._grid, targets, short[hf], taken2
                    )
                else:
                    taken2 = self.resources.consume_many(targets, short[hf])
                stomach[hf] += taken2

        # 4.4) 工作记忆写入（L5）：食物丰富的格子记入 4 槽（round-robin）
        cur_flat = self._flat[:P]
        # R121 §4.1：字面量 0.5 → 显式常量 `FOOD_RICH_LEVEL`（**只命名、不改数值**）。
        # 注意比较符仍是 `>`（与改名前逐位一致）；该常量语义 = "多到值得记住"，
        # 与 `CultureConfig.food_threshold`(0.3，"可食/值得付款") **不同**，见 config.py 注释。
        food_rich = (
            self.resources._grid[cur_flat]
            > FOOD_RICH_LEVEL * self.resources._capacity[cur_flat]
        )
        if food_rich.any():
            rich_idx = np.flatnonzero(food_rich)
            ptr = int(self._mem_ptr)
            self._work_memory[rich_idx, ptr] = cur_flat[rich_idx]
            self._mem_ptr = (ptr + 1) % 4

        # 4.5) 信号发射（g15）：以基因概率在当前格写入田字格标记，耗能
        #     模式 = 状态哈希（能量2位 + 食物1位 + 邻居1位 = 4位=16种）
        #     use_sim_core=True：下沉到 Rust signal_emit（L7a C1），
        #     rand_emit 在 Python 侧预生成，保证 RNG 消费顺序与纯 Python 一致。
        #     S0 优化：本 tick 移动前的位置占用只算一次（occ），信号段（4.5）
        #     与移动段（5）共用；此段 _flat 尚未变化（移动在步骤 5），len(_flat)==P，
        #     语义与各调用点原 np.bincount(self._flat[:P]) 逐位一致。
        occ = np.bincount(self._flat[:P], minlength=self.world.n_cells)
        signal_gene = genes[:, Gene.SIGNAL_STRENGTH].copy()
        rand_emit = self.rng.random(P)  # 双路径共用：Python 直接用，Rust 传入
        # D1 signal_disabled：发射概率恒0（signal_gene置零），RNG消费照常保对拍
        if self.config.signal_disabled:
            signal_gene[:] = 0.0
        # D1 signal_mode=random：独立 rng 预生成随机模式（严禁消费主 rng）
        random_patterns = None
        if self.config.signal_mode == "random":
            rng_rand = np.random.default_rng(self.config.seed + 99991)  # 独立种子偏移
            # R113/R121：码域随档位（"16"⇒1–15；"4"⇒1–4）；形状与抽取数**不变**
            random_patterns = rng_rand.integers(
                1, self._alpha_code_max + 1, size=P, dtype=np.uint8
            )
        if self._use_sim_core and random_patterns is None and not (
            self.config.info_structure.enabled and self.config.info_structure.arbitrary_codebook
        ):
            dens = occ.astype(np.float64)
            self._sim_core.signal_emit(
                self._flat[:P], energy, signal_gene, rand_emit,
                dens, self.resources._grid, self.resources._capacity,
                self.signals._marks, self.signals._age,
                SIGNAL_COST, ocfg.max_energy, self.signals.duration,
            )
        else:
            emitters = np.flatnonzero(rand_emit < signal_gene)
            if len(emitters):
                can_afford = energy[emitters] >= SIGNAL_COST
                emitters = emitters[can_afford]
                if len(emitters):
                    energy[emitters] -= SIGNAL_COST
                    e_flat = self._flat[emitters]
                    if random_patterns is not None:
                        # D1 random 模式：用独立 rng 预生成的随机模式
                        patterns = random_patterns[emitters]
                    else:
                        # 状态编码（R113/R121）：抽为纯函数 ⇒ "冗余位是否确被删"可免插桩直测
                        state, _code_offset = encode_signal_states(
                            energy, self.resources._grid, self.resources._capacity,
                            occ, e_flat, emitters, ocfg.max_energy, self._alpha,
                            work_memory=self._work_memory,   # B③："8" 档的记忆位输入
                        )
                        # 2026-09-19：`"8"` 档 `state = e_bin*2 + mem_bit` ⇒ `mem_bit = state & 1`
                        # （纯观测；只在状态编码路径累计，random 模式不计入）
                        if self._alpha == "8":
                            self._mem_bit_on += int((state & 1).sum())
                            self._mem_bit_n += int(state.size)
                        # ⚠️ 本行曾被我 2026-09-18 的重写误删（UnboundLocalError 当场暴露）
                        ifcfg = self.config.info_structure
                        if ifcfg.enabled and ifcfg.arbitrary_codebook:
                            # D2-2 任意性：pattern = 个体码本[state]，映射可遗传可漂移
                            patterns = self._codebook[emitters, state].astype(np.uint8)
                        else:
                            # 无码本：恒等映射（"16"⇒code=state；"4"⇒code=state+1，避开 0）
                            patterns = (state + _code_offset).astype(np.uint8)
                        # R121 §3.5 守卫（纯观测、零机时）：码 0 或越上限 ⇒ 计数。
                        # "4" 下**必须恒为 0**（非 0 = 隐形发射/清除他人标记 或 接收端读到未学槽）
                        if self._alpha != "16":
                            _bad = (patterns == 0) | (patterns > self._alpha_code_max)
                            if _bad.any():
                                self._alpha_bad_code_n += int(_bad.sum())
                    self.signals.write_many(e_flat, patterns)
                    if self._oracle_on:
                        # D-8 归因记账：存 **_id** 而非槽位索引（死亡压缩会重排槽位，
                        # 而归因窗口 persistence=10 tick 内发送者可能已死）。
                        self._emit_count[self._id[emitters]] += 1
                        self._last_sender[e_flat] = self._id[emitters]

        # 4.5) L10a 动物吃果实→能量转移（默认关闭；确定性数值管线，Rust 可下沉）
        if self.config.fruit.enabled:
            self._step_eat_fruit(P, energy, genes)

        # 5) 移动：g19 植物化降低移动概率（g0 × (1-g19)）
        #    use_sim_core=True：移动决策下沉到 Rust（step_movement），移动耗能在 Rust 内扣；
        #    stage2 只做年龄/冷却（moved_raw 全 False，不重复扣移动耗能）。
        #    stay_prob（战役参数）：move_prob × (1-stay_prob)，统一调低移动概率=等效扩世界。
        #    RNG 契约：保持每 tick 消费 P 个 uniform 不变（rng.random(P) 原样），只改阈值。
        _stay = float(self.config.simulation.stay_prob)
        move_prob = genes[:, Gene.MOVE_PROB] * (1.0 - genes[:, Gene.ROOTING]) * (1.0 - _stay)
        move_cost_ind = ocfg.move_cost * cold_penalty * (0.5 + genes[:, Gene.MOVE_COST])
        # D2-3 信息不对称：感知半径4/噪声/softmax 暂未下沉 Rust，启用时走 Python 路径
        _ifcfg = self.config.info_structure
        _d2_asym = _ifcfg.enabled and (_ifcfg.perception_radius == 4 or _ifcfg.perception_noise > 0 or _ifcfg.softmax_tau > 0)
        if self._use_sim_core and not _d2_asym:
            moved_raw = self.rng.random(P) < move_prob
            mi = np.flatnonzero(moved_raw & (energy >= move_cost_ind))
            # stage2：moved_raw 全 False（移动耗能改由 step_movement 扣），年龄/冷却正常
            out_moved = np.empty(P, dtype=bool)
            out_starved = np.empty(P, dtype=bool)
            out_expired = np.empty(P, dtype=bool)
            out_repro = np.empty(P, dtype=bool)
            self._sim_core.step_vectors_stage2(
                energy, self._age[:P], self._repro_cooldown[:P], genes,
                np.zeros(P, dtype=bool), move_cost_ind,
                out_moved, out_starved, out_expired, out_repro,
                _day_len_eff, ocfg.maturity_fraction, ocfg.max_energy,
            )
            if len(mi) > 0:
                # 预计算环境量
                food_ratio = self.resources._grid / np.maximum(
                    self.resources._capacity, 1e-9
                )
                sig_present = (self.signals._marks > 0).astype(np.float64)
                # D0 修复：群居项量纲归一化（按邻居上限 8 归一化 + 权重），
                # 避免未归一化 bincount(0~8) 压过感知/信号项(0~1)（外部评估 D5/元宝 C4）。
                # 仅移动决策消费此数组；signal_emit/pleasure_update 各自独立计算不受影响。
                # S0：occupancy 复用信号段预计算的 occ（移动之前位置未变）。
                nb_max = float(self._nb_table.shape[1])
                smw = self.config.simulation.social_move_weight
                densities = occ.astype(np.float64) / nb_max * smw
                signal_marks = self.signals._marks.astype(np.uint8)
                # 预生成随机选择（得分无差异时用），按移动个体顺序
                rand_choice = self.rng.integers(
                    0, 1_000_000, size=len(mi), dtype=np.int64
                )
                # Rust 移动决策（含移动扣费）
                # A′：Rust 侧**不产**朝向梯度计数器（只做决策）⇒ 标记"计数不可用"，
                #     `memory_gradient_stats()` 会按 n/a 报，**不**冒充 0（防假读数）。
                self._mem_grad_counts_valid = False
                _ifc_mg = self.config.info_structure
                self._sim_core.step_movement(
                    self._flat[:P], energy, genes, self._trust[:P],
                    self._work_memory[:P].reshape(-1), self._interpret[:P],
                    food_ratio, sig_present, densities, signal_marks,
                    self._nb_table.reshape(-1),
                    mi.astype(np.int64), rand_choice, move_cost_ind,
                    self.world.n_cells, self._nb_table.shape[1],
                    int(self.world.cols),
                    1 if _ifc_mg.memory_gradient == "orientation" else 0,
                    float(_ifc_mg.memory_gradient_gain),
                )
                # 5.6) 信任学习：移动到有信号的格子后验证真假
                target_cells = self._flat[mi]
                had_signal = sig_present[target_cells] > 0
                has_food = food_ratio[target_cells] > ccfg.food_threshold
                true_sig = had_signal & has_food
                false_sig = had_signal & ~has_food
                self._trust[mi[true_sig]] = np.minimum(
                    1.0, self._trust[mi[true_sig]] + ccfg.trust_true
                )
                self._trust[mi[false_sig]] = np.maximum(
                    0.0, self._trust[mi[false_sig]] - ccfg.trust_false
                )
                # D2-1 学习瓶颈：幼体观察学习（信号+后果→更新解读表）
                ifcfg = self.config.info_structure
                if ifcfg.enabled and ifcfg.learning_bottleneck and len(mi) > 0:
                    young = self._age[mi] < ifcfg.learning_maturity_ticks
                    can_learn = self._learning_count[mi] < ifcfg.learning_samples_max
                    learners = mi[young & can_learn]
                    if len(learners) > 0:
                        l_targets = self._flat[learners]
                        l_marks = self.signals._marks[l_targets].astype(np.int64)
                        l_has_sig = l_marks > 0
                        l_has_food = food_ratio[l_targets] > ccfg.food_threshold
                        lr = ifcfg.learning_rate
                        for j, idx in enumerate(learners):
                            if l_has_sig[j]:
                                m = int(l_marks[j])
                                if l_has_food[j]:
                                    self._interpret[idx, m] += lr * (1.0 - self._interpret[idx, m])
                                else:
                                    self._interpret[idx, m] -= lr * (1.0 + self._interpret[idx, m])
                                self._learning_count[idx] += 1
        else:
            moved = self.rng.random(P) < move_prob
            moved &= energy >= move_cost_ind  # 付得起才走
            Nm = int(moved.sum())
            # ── R146/R149 L2：人口口径读数（B4；本段 = [本地开发] 线）─────────────
            # 只在 l2_dash 开启时累加 ⇒ **关档 `l2_probe()` 返回 None（未适用），不是 0**
            # （R120/§五.12 口径铁律）。与"谁在动"无关，故放在 `if Nm:` **之外**。
            _l2_on = bool(self.config.simulation.l2_dash)
            if _l2_on:
                _gate = float(self.config.predation.attack_gene_gate)
                self._pop_n += int(P)
                self._g16_le_gate_n += int(np.count_nonzero(
                    genes[:P, Gene.AGGRESSION] <= _gate))
                self._inelig_pop_n += int(np.count_nonzero(
                    self._far_len[self._flat[:P]] == 0))
            if Nm:
                mi = np.flatnonzero(moved)
                food_ratio = self.resources._grid / np.maximum(
                    self.resources._capacity, 1e-9
                )
                sig_present = (self.signals._marks > 0).astype(np.float64)
                # D0 修复：群居项量纲归一化（按邻居上限 8 归一化 + 权重），
                # 避免未归一化 bincount(0~8) 压过感知/信号项(0~1)（外部评估 D5/元宝 C4）。
                # 仅移动决策消费此数组；signal_emit/pleasure_update 各自独立计算不受影响。
                # S0：occupancy 复用信号段预计算的 occ（移动之前位置未变）。
                nb_max = float(self._nb_table.shape[1])
                smw = self.config.simulation.social_move_weight
                densities = occ.astype(np.float64) / nb_max * smw
                rand_choice = self.rng.integers(
                    0, 1_000_000, size=Nm, dtype=np.int64
                )
                targets = np.empty(Nm, dtype=np.int64)
                # ── R146/R149 L2：两段式冲刺（**本段 = [本地开发] 线**）─────────────
                # 第一段「走多远」：纯能量闸（dash 门槛 + 2 格成本，**顺序无关**）× mob 概率闸
                # 第二段「选哪格」：走 2 格时在 **strict 2 圈**里**均匀盲选**
                # ⚠️ 措辞纪律（R148-2a）：这是**同一抽数的两个位域**（`10^6 = 100 × 10^4`
                #    恰好整除 ⇒ 均匀源下两字段不相关，已实测 χ²=9775.6/df 9801）
                #    —— **不是"两次独立判定"**，报告与注释都禁写。
                # ⚠️ RNG 形状保持（H2）：本段**不新增、也不跳过**任何 `rng.*` 调用
                #    （`u = self.rng.random()` 仍按个体消费 ⇒ 每 tick 抽取数与关档一致）。
                dash = np.zeros(Nm, dtype=bool)
                _mult = np.ones(Nm, dtype=np.float64)
                mob_eff = np.zeros(Nm, dtype=np.float64)
                age_factor = np.ones(Nm, dtype=np.float64)
                if _l2_on:
                    _ocfg = self.config.organisms
                    _cells = self._flat[mi]
                    _ls = self._lifespan(genes[mi, Gene.LIFE_GENE])   # 个体自己的寿命
                    _agef = self._age[mi].astype(np.float64)
                    age_factor[_agef < _ocfg.maturity_fraction * _ls] = _ocfg.young_mob_mult
                    age_factor[_agef >= _ocfg.senile_fraction * _ls] = _ocfg.old_mob_mult
                    # g18 = DEFENSE（B1/注册表改名为 MOBILITY 后此处同步；语义变更见 §十三 13.2）
                    mob_eff = genes[mi, Gene.DEFENSE] * age_factor
                    _step = 1.0 + _ocfg.dash_cost_kappa * (2.0 ** _ocfg.dash_cost_exp - 1.0)
                    _cost2 = move_cost_ind[mi] * _step
                    dash = (((rand_choice % 100) < (100.0 * mob_eff).astype(np.int64))
                            & (energy[mi] >= _ocfg.dash_min_energy_frac * _ocfg.max_energy)
                            & (energy[mi] >= _cost2)
                            & (self._far_len[_cells] > 0))
                    _mult = np.where(dash, _step, 1.0)
                    # 读数（B4）：dash_frac 的分子/分母、mob_eff/age_factor 分布
                    self._run_mover_n += int(Nm)
                    self._run_dash_n += int(dash.sum())
                    _box = np.clip((genes[mi, Gene.AGGRESSION] * 3).astype(np.int64), 0, 2)
                    self._mover_n_by_box += np.bincount(_box, minlength=3)[:3]
                    if dash.any():
                        self._dash_n_by_box += np.bincount(_box[dash], minlength=3)[:3]
                    self._mob_sum += float(mob_eff.sum())
                    self._mob_sq_sum += float(np.dot(mob_eff, mob_eff))
                    self._mob_n += int(Nm)
                    self._agef_sum += float(age_factor.sum())
                ifcfg3 = self.config.info_structure
                d2_asym = ifcfg3.enabled and ifcfg3.perception_radius == 4
                d2_noise = ifcfg3.enabled and ifcfg3.perception_noise > 0
                d2_softmax = ifcfg3.enabled and ifcfg3.softmax_tau > 0
                # B1：R2 声誉权重（0=关闭 → sig_weight 恒 0.5，与旧版逐位一致）
                rep_w = ifcfg3.reputation_weight if ifcfg3.enabled else 0.0
                # A′ 记忆朝向梯度（2026-09-19）：**不**受 `enabled` 门控 —— 与 ⑥ 探针同规格，
                # 必须能**单独**开关，否则 A′ 会被学习瓶颈/码本/softmax 等一堆 D2 机制污染
                # （⇒ 不再是单变量实验）。`none` 下走**原式**，逐位等价。
                mem_grad_on = (ifcfg3.memory_gradient == "orientation")
                mem_grad_gain = float(ifcfg3.memory_gradient_gain)
                for i, idx in enumerate(mi):
                    # D2-3 信息不对称：感知半径4 = Von Neumann（上/左/右/下），各向同性。
                    # ⚠️ 禁用 nb[:4]：8 邻列序以 [上左,上,上右,左] 打头，取"前4个"
                    # 实际只保留"北+西" → 个体永不能向南/东移动，种群被单向驱赶至极区、
                    # 招募崩塌（见 docs/决策与评审/A4-崩溃溯源报告-20260912.md）。
                    if d2_asym:
                        nb = self.world.neighbors_von_neumann(int(self._flat[idx]))
                    else:
                        nb = np.asarray(self.world.neighbors(int(self._flat[idx])))
                    if len(nb) == 1:
                        targets[i] = nb[0]
                        continue
                    perc = genes[idx, Gene.PERCEPTION]
                    soc = (genes[idx, Gene.SOCIABILITY] - 0.5) * 2.0
                    # D2-3 感知噪声：食物/信号感知加高斯噪声（独立rng，不消费主rng）
                    fr = food_ratio[nb].copy()
                    sp = sig_present[nb].copy()
                    if d2_noise:
                        fr += np.random.normal(0, ifcfg3.perception_noise, size=len(nb))
                        sp += np.random.normal(0, ifcfg3.perception_noise, size=len(nb))
                        fr = np.clip(fr, 0.0, 1.0)
                        sp = np.clip(sp, 0.0, 1.0)
                    # B1｜R2 声誉权重（1 行）：信号项在原有 trust 权重之上再按 trust 放大
                    # （高 trust → 更重视信号）。sig_weight = trust×(0.5 + rep_w×trust)；
                    # rep_w=0 时退化为原式 0.5×trust（与旧版逐位一致）。
                    sig_weight = self._trust[idx] * (0.5 + rep_w * self._trust[idx])
                    score = perc * (
                        fr * 0.5 + sp * sig_weight
                    ) + soc * densities[nb]
                    valid_mem = self._work_memory[idx][self._work_memory[idx] >= 0]
                    if mem_grad_on:
                        # A′：先记"测到了没有"，再加朝向梯度（可观测性优先，见 memory_gradient_stats）
                        self._mem_grad_dec += 1
                        if len(valid_mem) > 0:
                            self._mem_grad_slots += 1
                            g = self._memory_orientation_cos(
                                int(self._flat[idx]), nb, valid_mem)
                            if float(np.abs(g).max()) > 1e-9:
                                self._mem_grad_trig += 1
                            score = score + mem_grad_gain * perc * g
                    elif len(valid_mem) > 0:
                        # 原式（**严格不动**：`0.3` 字面量，保与旧版逐位一致）
                        mem_in_nb = np.isin(nb, valid_mem)
                        score = score + 0.3 * perc * mem_in_nb.astype(np.float64)
                    nb_sigs = self.signals._marks[nb]
                    if (nb_sigs > 0).any():
                        interp = np.array(
                            [self._interpret[idx, int(s)] if s > 0 else 0.0 for s in nb_sigs],
                            dtype=np.float64,
                        )
                        score = score + 0.4 * perc * interp
                    # D2-3 softmax：温度采样替代argmax（tau=0时回退argmax）
                    if d2_softmax:
                        exp_s = np.exp((score - score.max()) / ifcfg3.softmax_tau)
                        probs = exp_s / exp_s.sum()
                        # 消费主rng一个uniform用于采样（保对拍）
                        u = self.rng.random()
                        cum = np.cumsum(probs)
                        chosen = min(int(np.searchsorted(cum, u)), len(nb) - 1)
                        targets[i] = nb[chosen]
                        # ---- D-18 ⑥ 探针（R43）：纯观测，零 RNG、零行为改变 ----
                        # ⑥b = Δ_i = P(选中格|含信号) − P(选中格|去信号)；
                        # "去信号"= 反事实去掉两路信号项（感知 sp×trust 项 + 解读 interp 项），
                        # 保留噪声化 fr/群居/记忆（只去信号，不去噪声）。
                        # 禁用 argmax 翻转作主口径（τ=0.15 是概率抽样，argmax 翻转系统性低估）。
                        if self._measure_resp:
                            self._resp_decisions += 1
                            if (nb_sigs > 0).any():
                                score_wo = perc * (fr * 0.5) + soc * densities[nb]
                                if len(valid_mem) > 0:
                                    score_wo = score_wo + 0.3 * perc * np.isin(
                                        nb, valid_mem
                                    ).astype(np.float64)
                                exp_w = np.exp(
                                    (score_wo - score_wo.max()) / ifcfg3.softmax_tau
                                )
                                probs_wo = exp_w / exp_w.sum()
                                self._resp_exposed += 1
                                d_full = float(probs[chosen] - probs_wo[chosen])
                                self._resp_delta_sum += d_full
                                # R123/B②：**内容项单去**的反事实（保留存在性项）⇒ Δ_content。
                                # `score` 此刻 = ... + 0.4*perc*interp，故减去该行即"无内容"。
                                score_woc = score - 0.4 * perc * interp
                                exp_c = np.exp(
                                    (score_woc - score_woc.max()) / ifcfg3.softmax_tau
                                )
                                probs_woc = exp_c / exp_c.sum()
                                d_content = float(probs[chosen] - probs_woc[chosen])
                                self._resp_delta_content_sum += d_content
                                # 逐个体 Δ（按 _id 键控；供付款闸读取）
                                _iid = int(self._id[idx])
                                self._delta_full_by_id[_iid] = d_full
                                self._delta_content_by_id[_iid] = d_content
                                self._delta_tick_by_id[_iid] = self._tick
                                if int(np.argmax(score_wo)) != int(np.argmax(score)):
                                    self._resp_flip += 1
                    elif score.max() - score.min() < 1e-9:
                        targets[i] = nb[int(rand_choice[i] % len(nb))]
                    else:
                        targets[i] = nb[int(np.argmax(score))]
                    # ── R146/R149 L2 第二段：冲刺者改走 2 格（**盲选**）─────────────
                    # 位置：**在 score/softmax 之后** ⇒ `u = self.rng.random()` 仍按个体消费
                    # ⇒ 每 tick 随机抽取数与关档**逐字一致**（H2/H1 的形状要求）。
                    # 代价：冲刺者白算一次 score（换取随机流形状不变 —— 值得）。
                    if _l2_on and dash[i]:
                        _c = int(self._flat[idx])
                        _fl = int(self._far_len[_c])
                        targets[i] = int(self._far_cells[
                            int(self._far_off[_c]) + int(rand_choice[i] // 100) % _fl])
                self._flat[mi] = targets
                if _l2_on:
                    # L2：按实际步数计费（d=1 ×1，d=2 ×(1+κ(2^p−1))）
                    _cost = move_cost_ind[mi] * _mult
                    energy[mi] -= _cost
                    self._ec_add(EC_MOVE, mi, _cost)
                else:
                    energy[mi] -= move_cost_ind[mi]
                    # R141 P0：cost_move 通道（只记 Python 路径；Rust 路径在 Rust 内扣费）
                    self._ec_add(EC_MOVE, mi, move_cost_ind[mi])
                # 5.6) 信任学习
                target_cells = self._flat[mi]
                had_signal = sig_present[target_cells] > 0
                has_food = food_ratio[target_cells] > ccfg.food_threshold
                true_sig = had_signal & has_food
                false_sig = had_signal & ~has_food
                # 5.6.0) V-1 oracle 正向对照（R39/D-8）：利益对齐回馈。
                # 位置=规格 §2.4（true_sig 之后、信任学习之前）；C-8 ⇒ 仅 Python 路径。
                # 零 RNG、守恒；复用 :829 已算好的 true_sig（不新增判定逻辑）。
                if self._oracle_on:
                    # D-26a：①②③ 上游环节计数（纯观测，不改状态/随机流）
                    self._diag_had_sig += int(had_signal.sum())
                    self._diag_true_sig += int(true_sig.sum())
                    # R121 §4.1 零机时 counter：true_sig 落点中 `food_ratio` 落在**阈值错位带**
                    # `(food_threshold, FOOD_RICH_LEVEL]` 的计数（已被 has_food 过滤，此处只取上界）
                    if true_sig.any():
                        _fr_band = food_ratio[target_cells[true_sig]]
                        self._diag_food_band_true_sig += int(np.count_nonzero(
                            (_fr_band > ccfg.food_threshold) & (_fr_band <= FOOD_RICH_LEVEL)
                        ))
                    # R102 条件 1（修正版）：守恒三账审计——**包住调用实测 Σenergy 变化**
                    # （独立核算，非把同一个数抄三遍；C-2 成立 ⇒ 应恒 0）
                    _e_before = float(energy.sum())
                    self._oracle_after_move(mi, target_cells, had_signal, true_sig, energy)
                    self._audit_calls += 1
                    self._audit_sum_delta += float(energy.sum()) - _e_before
                self._trust[mi[true_sig]] = np.minimum(
                    1.0, self._trust[mi[true_sig]] + ccfg.trust_true
                )
                self._trust[mi[false_sig]] = np.maximum(
                    0.0, self._trust[mi[false_sig]] - ccfg.trust_false
                )
                # D2-1 学习瓶颈：幼体观察学习（Python路径，与Rust路径逻辑一致）
                ifcfg_lb = self.config.info_structure
                if ifcfg_lb.enabled and ifcfg_lb.learning_bottleneck and len(mi) > 0:
                    young = self._age[mi] < ifcfg_lb.learning_maturity_ticks
                    can_learn = self._learning_count[mi] < ifcfg_lb.learning_samples_max
                    learners = mi[young & can_learn]
                    if len(learners) > 0:
                        l_targets = self._flat[learners]
                        l_marks = self.signals._marks[l_targets].astype(np.int64)
                        l_has_sig = l_marks > 0
                        l_has_food = food_ratio[l_targets] > ccfg.food_threshold
                        lr = ifcfg_lb.learning_rate
                        for j, idx in enumerate(learners):
                            if l_has_sig[j]:
                                m = int(l_marks[j])
                                if l_has_food[j]:
                                    self._interpret[idx, m] += lr * (1.0 - self._interpret[idx, m])
                                else:
                                    self._interpret[idx, m] -= lr * (1.0 + self._interpret[idx, m])
                                self._learning_count[idx] += 1

        # 5.5) 捕食（g16）+ 6.5) 文化学习（L5）：use_sim_core=True 时合并为一次 Rust 调用，
        #     共用一次 cell→个体 CSR 构建，消除重复开销。use_sim_core=False 时分步执行。
        predation_mask = np.zeros(P, dtype=bool)
        pcfg = self.config.predation           # 隐式选择压参数化（A2）：捕食段参数
        attack_gene = genes[:, Gene.AGGRESSION]
        if pcfg.enabled:
            # PC-1 S2（R134）：`enabled=True`（默认）⇒ 原式逐位不动（RNG 消费 P 个 uniform）
            hunger = np.clip(1.0 - energy / max(ocfg.max_energy, 1e-9), 0.0, 1.0)
            attack_prob = attack_gene * pcfg.attack_prob_coef * hunger
            attackers = np.flatnonzero(
                (attack_gene > pcfg.attack_gene_gate) & (self.rng.random(P) < attack_prob)
            )
        else:
            # S2 off（PC-1 单营养级构造）：**跳过攻击者选择**（连 RNG 抽取一起跳过 ⇒
            # 新配置的 RNG 轨迹，与 enabled=True 的 run 不逐位可比——预期，非缺陷）。
            # `predation_and_culture` 照常调用（文化学习/年龄推进共用该调用），
            # 空 attackers ⇒ 捕食贡献恒 0（predation_mask 全 False）。
            attackers = np.zeros(0, dtype=np.int64)
        # 文化学习需要的成熟年龄（年龄已在 stage2 推进，use_sim_core=True 时）
        age_f = self._age[:P].astype(np.float64)
        life_span = self._lifespan(genes[:, Gene.LIFE_GENE])
        maturity_age = ocfg.maturity_fraction * life_span

        if self._use_sim_core and not _d2_asym:
            # 合并调用：捕食 + 文化学习，共用一次 CSR
            # （参考实现 bug 修复：D2 信息不对称时移动段已回退 Python 分步并跳过
            #  stage2 的年龄/冷却推进，此处分支必须同步回退，否则 _age/_repro_cooldown
            #  永不推进 → 永不成熟/不繁殖。与移动段 998 行条件保持一致。）
            if len(attackers) > 0:
                rand_prey = self.rng.integers(0, 1_000_000, size=len(attackers), dtype=np.int64)
                rand_success = self.rng.random(len(attackers))
            else:
                rand_prey = np.zeros(0, dtype=np.int64)
                rand_success = np.zeros(0, dtype=np.float64)
            self._sim_core.predation_and_culture(
                energy, stomach, predation_mask, self._interpret,
                self._flat[:P], genes, self._age[:P], maturity_age,
                attackers.astype(np.int64), rand_prey, rand_success,
                self._nb_table.reshape(-1),
                self.world.n_cells, self._nb_table.shape[1],
                ocfg.max_energy, ocfg.eat_efficiency, 0.1,
                pcfg.attack_cost, pcfg.success_gene_gain,
                pcfg.success_floor, pcfg.success_ceil,
                pcfg.transfer_ratio, pcfg.stomach_transfer,
            )
            # ── D2-4 Steels 对齐：同格相遇对齐解读表+码本（Rust路径Python侧补做） ──
            ifcfg_sa_r = self.config.info_structure
            if ifcfg_sa_r.enabled and ifcfg_sa_r.steels_alignment:
                occ_sa = np.bincount(self._flat[:P], minlength=self.world.n_cells)
                crowded_sa = np.flatnonzero(occ_sa > 1)
                for cell in crowded_sa:
                    cell_idx = np.flatnonzero(self._flat[:P] == cell)
                    if len(cell_idx) < 2:
                        continue
                    np.random.shuffle(cell_idx)
                    for pair in range(0, len(cell_idx) - 1, 2):
                        i, j = int(cell_idx[pair]), int(cell_idx[pair + 1])
                        if self.rng.random() < ifcfg_sa_r.alignment_rate:
                            # 解读表对齐（浮点EWMA）
                            diff = self._interpret[j] - self._interpret[i]
                            noise_i = self.rng.normal(0, ifcfg_sa_r.alignment_noise, size=16)
                            noise_j = self.rng.normal(0, ifcfg_sa_r.alignment_noise, size=16)
                            self._interpret[i] += ifcfg_sa_r.alignment_step * diff + noise_i
                            self._interpret[j] -= ifcfg_sa_r.alignment_step * diff + noise_j
                            # 码本对齐（离散概率替换：每个状态以alignment_step概率让i采用j的映射）
                            if ifcfg_sa_r.arbitrary_codebook:
                                cb_mask = self.rng.random(16) < ifcfg_sa_r.alignment_step
                                if cb_mask.any():
                                    self._codebook[i, cb_mask] = self._codebook[j, cb_mask]
        else:
            # ── Python 分步：捕食 ──
            if len(attackers) > 0:
                rand_prey = self.rng.integers(0, 1_000_000, size=len(attackers), dtype=np.int64)
                rand_success = self.rng.random(len(attackers))
                # R141 P0 + R138 §二：**真决斗三级拆分**（名义攻击者 / 有效出手 / 致死）
                # 与**分通道记账**共用同一循环（零额外遍历）。
                # 背景：`cannibalism_stats` 报的 5504 是**名义攻击者数**，其中含大量
                # 根本没出手的 `continue`（能量不足 / 邻格无猎物 / 本 tick 已被吃）
                # ⇒ 老工的 3.02% **低估**了真实出手成功率。这里把分母拆开。
                _atk_i, _atk_amt = [], []
                _pred_i, _pred_amt = [], []
                for k, idx in enumerate(attackers):
                    if energy[idx] <= pcfg.attack_cost:
                        self._duel["skip_no_energy"] += 1
                        continue
                    nb = np.asarray(self.world.neighbors(int(self._flat[idx])))
                    nb_mask = np.isin(self._flat[:P], nb) & (np.arange(P) != idx)
                    prey_candidates = np.flatnonzero(nb_mask)
                    if len(prey_candidates) == 0:
                        self._duel["skip_no_prey"] += 1
                        continue
                    prey = int(prey_candidates[int(rand_prey[k] % len(prey_candidates))])
                    if predation_mask[prey]:
                        self._duel["skip_already_eaten"] += 1
                        continue
                    self._duel["real_attempts"] += 1
                    energy[idx] -= pcfg.attack_cost
                    _atk_i.append(idx)
                    _atk_amt.append(float(pcfg.attack_cost))
                    success_rate = np.clip(
                        (energy[idx] / max(energy[idx] + energy[prey], 1e-9))
                        * (0.5 + attack_gene[idx] * pcfg.success_gene_gain),
                        pcfg.success_floor, pcfg.success_ceil,
                    )
                    if rand_success[k] < success_rate:
                        predation_mask[prey] = True
                        # R141：ΣE_prey（**击杀瞬间**的猎物能量）——解 `transfer*` 的关键输入
                        self._ec_prey_e_sum += float(energy[prey])
                        self._ec_prey_kill_n += 1
                        _tr = float(energy[prey]) * pcfg.transfer_ratio
                        energy[idx] += _tr
                        _pred_i.append(idx)
                        _pred_amt.append(_tr)
                        self._duel["kills"] += 1
                        stomach[idx] = np.minimum(
                            stomach[idx] + stomach[prey] * pcfg.stomach_transfer,
                            ocfg.max_energy / max(1e-9, ocfg.eat_efficiency),
                        )
                        stomach[prey] = 0.0
                # R141 P0：捕食侧两条通道（出手成本 / 掠得能量）——循环外统一记账
                if _atk_amt:
                    self._ec_add(EC_ATTACK, _atk_i, _atk_amt)
                if _pred_amt:
                    self._ec_add(EC_PRED, _pred_i, _pred_amt)
            # ── Python 分步：年龄推进（Rust 路径已由 stage2 就地 +1）──
            self._age[:P] += 1
            age_f = self._age[:P].astype(np.float64)
            # ── Python 分步：文化学习 ──
            juvenile = age_f < maturity_age
            if juvenile.any():
                j_idx = np.flatnonzero(juvenile)
                for idx in j_idx:
                    nb = np.asarray(self.world.neighbors(int(self._flat[idx])))
                    nb_mask = np.isin(self._flat[:P], nb)
                    adult_nb = nb_mask & (age_f >= maturity_age)
                    adult_idx = np.flatnonzero(adult_nb)
                    if len(adult_idx) > 0:
                        mean_interpret = self._interpret[adult_idx].mean(axis=0)
                        self._interpret[idx] += 0.1 * (mean_interpret - self._interpret[idx])
            # ── D2-4 Steels 对齐：同格相遇概率性解读表对齐 ──
            ifcfg_sa = self.config.info_structure
            if ifcfg_sa.enabled and ifcfg_sa.steels_alignment:
                occ_py = np.bincount(self._flat[:P], minlength=self.world.n_cells)
                crowded = np.flatnonzero(occ_py > 1)
                for cell in crowded:
                    cell_idx = np.flatnonzero(self._flat[:P] == cell)
                    if len(cell_idx) < 2:
                        continue
                    # 随机配对（消费主rng保对拍）
                    np.random.shuffle(cell_idx)
                    for pair in range(0, len(cell_idx) - 1, 2):
                        i, j = int(cell_idx[pair]), int(cell_idx[pair + 1])
                        if self.rng.random() < ifcfg_sa.alignment_rate:
                            # 解读表对齐（浮点EWMA）
                            diff = self._interpret[j] - self._interpret[i]
                            noise_i = self.rng.normal(0, ifcfg_sa.alignment_noise, size=16)
                            noise_j = self.rng.normal(0, ifcfg_sa.alignment_noise, size=16)
                            self._interpret[i] += ifcfg_sa.alignment_step * diff + noise_i
                            self._interpret[j] -= ifcfg_sa.alignment_step * diff + noise_j
                            # 码本对齐（离散概率替换）
                            if ifcfg_sa.arbitrary_codebook:
                                cb_mask = self.rng.random(16) < ifcfg_sa.alignment_step
                                if cb_mask.any():
                                    self._codebook[i, cb_mask] = self._codebook[j, cb_mask]

        # 7) 死亡判定：饿死（energy<=0）→ 老死（age>=寿命）→ 被捕食
        # 注意：统一用 Python 计算（不用 Rust 的 out_starved/out_expired），
        # 因为捕食（步骤 5.5）在 Rust stage2 之后才执行，会改变 energy。
        starved = energy <= 0.0
        expired = (~starved) & (age_f >= life_span)
        dead = starved | expired | predation_mask
        deaths: Counter = Counter()
        n_starved = int(starved.sum())
        n_expired = int(expired.sum())
        # ---- R135 第 -1 步④：**互捕结构量化**（F「去互捕」延后的前置；2026-09-20）----
        # 只读 `predation_mask`（Python/Rust 两条路径的共同输出）+ `attackers`（Python 侧选出）
        # ⇒ **双路径天然覆盖，无需改 Rust、无需重编**。
        # 核心判据 = `Δ = mean(g16_attacker) − mean(g16_prey)`：
        #   Δ > 0 ⇒ 「鹰吃鸽」有结构；Δ ≈ 0 ⇒ 随机互捕（不存在 g16 分层）。
        # 同时留 5-bin 直方图（不只均值——双峰/分层信息在形状里，均值会抹平）。
        if predation_mask.any():
            _pm = np.flatnonzero(predation_mask)
            self._cannib_prey_hist += np.bincount(
                np.clip((attack_gene[_pm] * CANNIB_BINS).astype(np.int64), 0, CANNIB_BINS - 1),
                minlength=CANNIB_BINS)[:CANNIB_BINS]
        if attackers.size:
            self._cannib_atk_hist += np.bincount(
                np.clip((attack_gene[attackers] * CANNIB_BINS).astype(np.int64), 0, CANNIB_BINS - 1),
                minlength=CANNIB_BINS)[:CANNIB_BINS]
        n_predation = int(predation_mask.sum())
        if n_starved:
            deaths[DeathCause.STARVATION] = n_starved
        if n_expired:
            deaths[DeathCause.OLD_AGE] = n_expired
        if n_predation:
            deaths[DeathCause.PREDATION] = n_predation

        # 8) 繁殖：冷却期（g12）倒数；能量 ≥ 阈值（0.25+g2×0.65），且种群未满
        # 注意：统一用 Python 计算繁殖判定（不用 Rust 的 out_repro），
        # 因为捕食（5.5）和植物化（3.5）在 Rust stage2 之后才执行，会改变 energy。
        if not self._use_sim_core:
            self._repro_cooldown[:P] = np.maximum(
                0.0, self._repro_cooldown[:P] - 1.0
            )
        repro_thr = (0.25 + genes[:, Gene.REPRO_THRESHOLD] * 0.65) * ocfg.max_energy
        repro = (
            (~dead)
            & (energy >= repro_thr)
            & (self._repro_cooldown[:P] <= 0.0)
            & (age_f >= maturity_age)   # 未到成熟年龄不生（长大后才能繁衍）
        )
        # PC-1 S1 软顶（R134；冒烟后修订为**目标窗形式**，2026-09-20）：
        #   p_soft = clamp((N* − N)/N*, 0, 1)，N* = soft_cap_target × max_count。
        #   N*>0 ⇒ 出生率随 N 逼近 N* 线性归零 ⇒ N 稳态钉在 N* 附近（死亡≈出生的选择窗）。
        #   🔴 首版线性 (1−N/K) 实测失败：无捕食世界死亡≈0 ⇒ N 顶满硬顶（12k 冒烟
        #      N_eq=3240=K，稳态窗门不过）⇒ 补偿必须在 N* 处归零（修订已上板）。
        # 仅 soft_cap_target>0 时消费额外 RNG（每 tick P 个 uniform）⇒ 变化限于新配置；
        #   0（默认）⇒ 与旧版逐位一致。硬顶（下方的 K 截断）保留兜底。
        _sct = float(self.config.population.soft_cap_target)
        if _sct > 0.0:
            _n_star = _sct * float(self.config.population.max_count)
            p_soft = min(1.0, max(0.0, (_n_star - P) / max(_n_star, 1e-9)))
            repro = repro & (self.rng.random(P) < p_soft)
        K = min(int(repro.sum()), self.config.population.max_count - P)
        born = 0
        if K > 0:
            ri = np.flatnonzero(repro)[:K]
            # D2 学习瓶颈/任意性码本：Rust 侧暂未实现，启用时走 Python 路径
            _ifcfg_d2 = self.config.info_structure
            _d2_repro = _ifcfg_d2.enabled and (_ifcfg_d2.learning_bottleneck or _ifcfg_d2.arbitrary_codebook or _d2_asym)
            if self._use_sim_core and not _d2_repro:
                # ---- Rust 路径（T4 L7b）：reproduce_batch 一次性算完基因/能量/文化继承 ----
                # RNG 顺序必须与 Python 参考路径逐位一致：
                #   1) mut_mask = random < rate；2) (若任一变异) gene_noise = normal；
                #   3) exp_noise = normal(inheritance_noise)；4) interp_noise = normal(0.1)
                mut_mask = (
                    self.rng.random((K, gcfg.gene_count)) < gcfg.mutation_rate
                ).astype(np.uint8)
                gene_noise = np.zeros((K, gcfg.gene_count), dtype=np.float64)
                if mut_mask.any():
                    sigma = gcfg.mutation_sigma * (gcfg.gene_max - gcfg.gene_min)
                    gene_noise = self.rng.normal(
                        0.0, sigma, size=(K, gcfg.gene_count)
                    )
                pcfg = self.config.pleasure
                exp_noise = self.rng.normal(
                    0.0, pcfg.inheritance_noise, size=(K, 120)
                )
                interp_noise = self.rng.normal(0.0, 0.1, size=(K, 16))
                # Rust 路径：D2 关闭时码本恒等映射（按档位；"16" 下与旧版逐位一致）
                child_codebook = codebook_init_rows(K, self._alpha)
                child_genes = np.empty((K, gcfg.gene_count), dtype=np.float64)
                child_energy = np.empty(K, dtype=np.float64)
                child_stomach = np.empty(K, dtype=np.float64)
                child_exp = np.empty((K, 120), dtype=np.float64)
                child_interp = np.empty((K, 16), dtype=np.float64)
                child_trust = np.empty(K, dtype=np.float64)
                child_baseline = np.empty(K, dtype=np.float64)
                self._sim_core.reproduce_batch(
                    ri.astype(np.int64),
                    self._genes[:P], energy, stomach, self._repro_cooldown[:P],
                    self._expectation[:P], self._interpret[:P],
                    self._trust[:P], self._baseline[:P],
                    mut_mask, gene_noise, exp_noise, interp_noise,
                    child_genes, child_energy, child_stomach,
                    child_exp, child_interp, child_trust, child_baseline,
                    gcfg.gene_min, gcfg.gene_max, pcfg.max_reward, 60.0,
                )
            else:
                # ---- Python 参考路径 ----
                child_genes = self._genes[ri].copy()
                mut = self.rng.random((K, gcfg.gene_count)) < gcfg.mutation_rate
                if mut.any():
                    sigma = gcfg.mutation_sigma * (gcfg.gene_max - gcfg.gene_min)
                    noise = self.rng.normal(0.0, sigma, size=(K, gcfg.gene_count))
                    child_genes = np.clip(
                        child_genes + np.where(mut, noise, 0.0),
                        gcfg.gene_min, gcfg.gene_max,
                    )
                # 传代投入比例（g7）：亲代把多少比例的能量/胃粮分给子代（0.3~0.7）
                split = 0.3 + genes[ri, Gene.PARENTAL_INVEST] * 0.4
                child_energy = energy[ri] * split
                child_stomach = stomach[ri] * split
                energy[ri] -= child_energy
                stomach[ri] -= child_stomach
                # 生完进入冷却（g12）：间隔 = g12 × 60 tick，冷却没到攒再多也不生
                self._repro_cooldown[ri] = genes[ri, Gene.REPRO_COOLDOWN] * 60.0
                # 愉悦度：子代继承亲代 expectation + 噪声（文化传递载体）
                pcfg = self.config.pleasure
                child_exp = self._expectation[ri].copy()
                child_exp += self.rng.normal(
                    0.0, pcfg.inheritance_noise, size=child_exp.shape
                )
                child_exp = np.clip(child_exp, 0.0, pcfg.max_reward)
                # 信任度：子代半继承亲代，回归中性 0.5
                child_trust = self._trust[ri] * 0.8 + 0.5 * 0.2
                child_baseline = self._baseline[ri] * 0.5
                # 信号解读表：D2 学习瓶颈 vs 旧版遗传
                ifcfg = self.config.info_structure
                if ifcfg.enabled and ifcfg.learning_bottleneck:
                    # 学习瓶颈：解读表不遗传，幼体随机初始化（与成体初始分布 normal(0,0.3) 一致）
                    # 消费 (K,16) 个 normal，与旧版 interp_noise 个数相同 → 保 RNG 顺序
                    child_interp = self.rng.normal(0.0, 0.3, size=(K, 16))
                else:
                    # 旧版：子代继承亲代 + 噪声（文化传递的核心载体）
                    child_interp = self._interpret[ri].copy()
                    child_interp += self.rng.normal(0.0, 0.1, size=child_interp.shape)
                # D2 任意性码本：子代遗传亲代码本 + 突变
                if ifcfg.enabled and ifcfg.arbitrary_codebook:
                    child_codebook = self._codebook[ri].copy()
                    cb_mut = self.rng.random((K, 16)) < ifcfg.codebook_mutation_rate
                    if cb_mut.any():
                        # 突变位映射到随机模式(1~15，0保留为无信号)
                        # R113/R121：取值域随档位（"16"⇒1–15；"4"⇒1–4）；
                        # 形状 `(K, 16)` 与抽取数**不变** ⇒ RNG 顺序不变（设计稿 §3.3）
                        cb_new = self.rng.integers(
                            1, self._alpha_code_max + 1, size=(K, 16), dtype=np.uint8
                        )
                        child_codebook[cb_mut] = cb_new[cb_mut]
                else:
                    # D2 关闭：码本恒等映射（按档位；"16" 下与旧版逐位一致）
                    child_codebook = codebook_init_rows(K, self._alpha)

            ids = np.arange(self._next_id, self._next_id + K, dtype=np.int64)
            self._next_id += K
            # D-17：_id 键控账本同步扩行（子代零行）+ 亲代终身子代数 +1。
            # ⚠️ 必须用 _id[ri]（键控），绝不用 ri 本身（死亡压缩会重排槽位）。
            self._grow_id_arrays(K)
            self._rs_children[self._id[ri]] += 1
            self._id = np.concatenate([self._id, ids])
            self._flat = np.concatenate([self._flat, self._flat[ri]])
            self._energy = np.concatenate([self._energy, child_energy])
            self._stomach = np.concatenate([self._stomach, child_stomach])
            self._genes = np.concatenate([self._genes, child_genes])
            self._age = np.concatenate([self._age, np.zeros(K, dtype=np.int64)])
            self._repro_cooldown = np.concatenate(
                [self._repro_cooldown, np.zeros(K, dtype=np.float64)]
            )
            new_gen = self._generation[ri] + 1
            self._generation = np.concatenate([self._generation, new_gen])
            self._parent = np.concatenate([self._parent, self._id[ri]])
            self._expectation = np.concatenate([self._expectation, child_exp])
            self._valence = np.concatenate(
                [self._valence, np.zeros(K, dtype=np.float64)]
            )
            self._arousal = np.concatenate(
                [self._arousal, np.full(K, 0.5, dtype=np.float64)]
            )
            self._baseline = np.concatenate([self._baseline, child_baseline])
            self._trust = np.concatenate([self._trust, child_trust])
            # 工作记忆：子代继承亲代的食物位置记忆（文化传递的一部分）
            # R121 §4.2 零机时 counter：量化"生而知之"——继承记忆中"距出生格 > 感知半径"
            # 的槽位占比（纯观测，不改状态/随机流）。距离口径 = **格距**（行差 + 经度环绕
            # 列差，与邻居表 `_build_neighbor_cache` 同口径；**未做纬度余弦缩放** ⇒ 极区
            # 高估距离，该读数应作**上界**理解）。
            _mem_inh = self._work_memory[ri]
            _br, _bc = self.world.flat_to_rc(self._flat[ri])
            _mr, _mc = self.world.flat_to_rc(_mem_inh.ravel())
            _K4 = int(_mem_inh.shape[1])
            _valid_mem = _mem_inh.ravel() >= 0
            _dr = (_mr - np.repeat(_br, _K4)).astype(np.float64)
            _dc = (_mc - np.repeat(_bc, _K4)).astype(np.float64)
            _cols_w = float(self.world.cols)
            _dc = (_dc + _cols_w / 2.0) % _cols_w - _cols_w / 2.0
            _dist = np.sqrt(_dr * _dr + _dc * _dc)
            self._mem_inherit_n += int(_valid_mem.sum())
            self._mem_inherit_far_n += int(np.count_nonzero(
                _valid_mem
                & (_dist > float(self.config.info_structure.perception_radius))
            ))
            self._work_memory = np.concatenate(
                [self._work_memory, _mem_inh.copy()]
            )
            # 信号解读表：子代继承亲代 + 噪声（文化传递的核心载体，Rust 路径已算好）
            self._interpret = np.concatenate([self._interpret, child_interp])
            # D2 码本：子代遗传+突变（Rust 路径已算好；Python 路径在上方计算）
            self._codebook = np.concatenate([self._codebook, child_codebook])
            # D2 学习计数：子代从零开始
            self._learning_count = np.concatenate([self._learning_count, np.zeros(K, dtype=np.int32)])
            # L10a：子代果实蓄力清零，种子携带清零
            self._fruit_charge = np.concatenate([self._fruit_charge, np.zeros(K, dtype=np.float64)])
            self._seed_carried = np.concatenate([self._seed_carried, np.zeros(K, dtype=np.int32)])
            born = K
            new_max = int(self._generation.max())
            if new_max > self._max_generation:
                self._max_generation = new_max

        # 8.5) 愉悦度更新（RPE 预测误差驱动）：对前 P 个亲代计算
        if self.config.pleasure.enabled and energy_before is not None:
            self._update_pleasure(P, energy_before)

        # 9) 清理尸体（只清理本 tick 行动的旧行；子代不受影响）
        if dead.any():
            keep = ~dead
            n_tail = len(self._id) - P
            self._id = np.concatenate([self._id[:P][keep], self._id[P:]])
            self._flat = np.concatenate([self._flat[:P][keep], self._flat[P:]])
            self._energy = np.concatenate(
                [self._energy[:P][keep], self._energy[P:]]
            )
            self._stomach = np.concatenate(
                [self._stomach[:P][keep], self._stomach[P:]]
            )
            self._genes = np.concatenate([self._genes[:P][keep], self._genes[P:]])
            self._age = np.concatenate([self._age[:P][keep], self._age[P:]])
            self._generation = np.concatenate(
                [self._generation[:P][keep], self._generation[P:]]
            )
            self._parent = np.concatenate([self._parent[:P][keep], self._parent[P:]])
            self._repro_cooldown = np.concatenate(
                [self._repro_cooldown[:P][keep], self._repro_cooldown[P:]]
            )
            self._valence = np.concatenate(
                [self._valence[:P][keep], self._valence[P:]]
            )
            self._arousal = np.concatenate(
                [self._arousal[:P][keep], self._arousal[P:]]
            )
            self._baseline = np.concatenate(
                [self._baseline[:P][keep], self._baseline[P:]]
            )
            self._expectation = np.concatenate(
                [self._expectation[:P][keep], self._expectation[P:]]
            )
            self._trust = np.concatenate(
                [self._trust[:P][keep], self._trust[P:]]
            )
            self._work_memory = np.concatenate(
                [self._work_memory[:P][keep], self._work_memory[P:]]
            )
            self._interpret = np.concatenate(
                [self._interpret[:P][keep], self._interpret[P:]]
            )
            # D2：码本/学习计数同步清理
            self._codebook = np.concatenate(
                [self._codebook[:P][keep], self._codebook[P:]]
            )
            self._learning_count = np.concatenate(
                [self._learning_count[:P][keep], self._learning_count[P:]]
            )
            # L10a：果实蓄力/种子携带同步清理
            self._fruit_charge = np.concatenate(
                [self._fruit_charge[:P][keep], self._fruit_charge[P:]]
            )
            self._seed_carried = np.concatenate(
                [self._seed_carried[:P][keep], self._seed_carried[P:]]
            )

        if len(self._id) == 0:
            self._extinct = True
        return born, n_starved + n_expired + n_predation, Counter(deaths)

    # ---- D-17/D-18/D-8：⑤ 观测账本 / oracle 回馈 / 统计访问器 -----------------

    def _grow_id_arrays(self, k: int) -> None:
        """出生时给全部 _id 键控账本扩 k 行（零初始化）。只增不压缩。"""
        if k <= 0:
            return
        z = lambda dt: np.zeros(k, dtype=dt)
        self._rs_children = np.concatenate([self._rs_children, z(np.int32)])
        self._rs_observed = np.concatenate([self._rs_observed, z(bool)])
        self._rs_g15 = np.concatenate([self._rs_g15, z(np.float32)])
        self._rs_age = np.concatenate([self._rs_age, z(np.int32)])
        self._rs_energy = np.concatenate([self._rs_energy, z(np.float32)])
        self._rs_cc0 = np.concatenate([self._rs_cc0, z(np.int32)])
        self._rs_cohort = np.concatenate([self._rs_cohort, z(np.uint8)])
        self._emit_count = np.concatenate([self._emit_count, z(np.int32)])
        self._oracle_gain = np.concatenate([self._oracle_gain, z(np.float32)])
        # R123/B②：门控臂逐个体 Δ（新个体两数组补零、tick 补 -1 ⇒ "本 tick 未参与决策"）
        self._delta_full_by_id = np.concatenate([self._delta_full_by_id, z(np.float32)])
        self._delta_content_by_id = np.concatenate(
            [self._delta_content_by_id, z(np.float32)])
        _nt = np.full(k, -1, dtype=np.int32)
        self._delta_tick_by_id = np.concatenate([self._delta_tick_by_id, _nt])

    def _observe_selection_cohort(self) -> None:
        """D-17 ⑤：给"首次出现在某窗口的存活个体"记一行观测（预注册口径）。

        - 窗口：`P < 0.9 × max_count` ⇒ 非饱和窗（cohort=0）；否则饱和窗（cohort=1，诊断用）。
        - 每 _id 只记**首次**（之后不重复记）；包含本 tick 新生子代。
        - 死亡个体行**永久保留**（_id 键控数组不压缩）⇒ 自动满足 R42"必须含死亡个体"。
        - ⑤ 的被估量 = 观测时 g15 → **剩余终身繁殖数**（= children − cc0），
          控年龄与能量；run 结束时仍存活者为右删失（均匀删失，仪器层可接受，已在 docstring 声明）。
        """
        P = len(self._id)
        if P == 0:
            return
        max_count = self.config.population.max_count
        cohort = 0 if P < 0.9 * max_count else 1
        alive_ids = self._id
        fresh = ~self._rs_observed[alive_ids]
        if not fresh.any():
            return
        idx = alive_ids[fresh]
        self._rs_observed[idx] = True
        self._rs_g15[idx] = self._genes[fresh, Gene.SIGNAL_STRENGTH]
        self._rs_age[idx] = self._age[fresh]
        self._rs_energy[idx] = self._energy[fresh]
        self._rs_cc0[idx] = self._rs_children[alive_ids[fresh]]
        self._rs_cohort[idx] = cohort

    def _oracle_after_move(self, mi, target_cells, had_signal, true_sig, energy) -> None:
        """D-8 oracle：归因 + 保本封顶内的 **R→S** 能量转移（详见 simulation/oracle.py）。

        F-R18（2026-09-15）：方向 = **接收者（移动者）回付发送者（信号写入者）**。
        故 `receiver_slots` 传的是**付款方**（`mi` = 移动者）、`sender_slots` 传的是
        **收款方**（`_last_sender` = 信号写入者）——与 `budget`（收款方额度）、
        `_oracle_gain`（已获回馈）、`oracle_return_ratio` 的既有命名一致。
        """
        ocfg = self.config.oracle
        sel = true_sig if ocfg.require_food else had_signal
        rc = mi[sel]
        if len(rc) == 0:
            return
        cc = target_cells[sel]
        s_ids = self._last_sender[cc]
        win = attribution_ok(self.signals._age[cc], self.signals.duration, ocfg.persistence)
        r_id = self._id[rc]
        # id→slot 解析（死亡压缩重排槽位 ⇒ 禁用发射时槽位；已死/未知 ⇒ -1 跳过）
        uniq, inv = np.unique(self._id, return_inverse=True)
        pos = np.searchsorted(uniq, s_ids)
        pos_c = np.minimum(pos, len(uniq) - 1)
        found = (s_ids >= 0) & (uniq[pos_c] == s_ids)
        s_slot = np.where(found, inv[pos_c], -1).astype(np.int64)
        # C-9 保本封顶（每发送者终身）：剩余额度 = **m ×** 成本×累计发射 − 已获回馈。
        # `m = oracle.gain_multiplier`（默认 1.0 ⇒ 与旧行为**逐位一致**）；`m > 1` 只在
        # **已登记的校准臂**上出现（C5 v2 入口硬校验）。⚠️ C-2 守恒恒成立（纯再分配）。
        valid = s_ids >= 0
        safe = np.where(valid, s_ids, 0)
        _m = float(self.config.oracle.gain_multiplier)
        budget = np.where(
            valid,
            _m * EMISSION_COST * self._emit_count[safe].astype(np.float64)
            - self._oracle_gain[safe].astype(np.float64),
            -1.0,
        )
        ok = found & win & (s_ids != r_id) & (budget > 0)
        # D-26a 四环节漏斗计数（逐级累加，纯观测）：进入选择集 → 归因命中 →
        # 窗口内 → 非自反馈 → 保本额度可用 → 实际成交。用于定位 D-24 G-A 瓶颈。
        self._diag_sel += int(sel.sum())
        self._diag_attrib_found += int(found.sum())
        self._diag_in_window += int((found & win).sum())
        self._diag_not_self += int((found & win & (s_ids != r_id)).sum())
        self._diag_budget_ok += int((found & win & (s_ids != r_id) & (budget > 0)).sum())
        # ---- R123/B② 门控臂：只对"**选择确实被信号改变**"（Δ>0）的到达付款 ----
        # 计数含义：`gate_pass` = 通过闸（可付款）；`gate_block` = 被拦 ⇒
        #   **蹭归因占比 = block / arrivals**（"碰巧来/信标引路但内容无用"在全部到达中的份额）。
        if self._gate_on:
            _arr = found & win & (s_ids != r_id) & (budget > 0)
            if _arr.any():
                _rid = r_id[_arr]
                _fresh = self._delta_tick_by_id[_rid] == self._tick
                _df = np.where(_fresh, self._delta_full_by_id[_rid], 0.0)
                _dc = np.where(_fresh, self._delta_content_by_id[_rid], 0.0)
                self._gate_arrivals += int(_arr.sum())
                self._gate_delta_full_sum += float(_df.sum())
                self._gate_delta_content_sum += float(_dc.sum())
                _sel_d = _dc if self._gate_delta == "content" else _df
                _pass = _sel_d > 0.0
                self._diag_gate_pass += int(_pass.sum())
                self._diag_gate_block += int((~_pass).sum())
                _g = np.zeros_like(ok)
                _g[np.flatnonzero(_arr)] = _pass
                ok = ok & _g
        if not ok.any():
            return
        led: dict = {}
        total, cnt, s_kept, g_kept = apply_oracle(
            energy=energy,
            receiver_slots=rc[ok].astype(np.int64),
            sender_slots=s_slot[ok],
            budget=budget[ok],
            donation=ocfg.donation,
            ledger=led,          # 内评 §四：对账 + 偿付截断量化（纯观测，不改转移结果）
        )
        if cnt:
            self._diag_applied += int(cnt)
            # F-R16：必须用 `np.add.at` —— `arr[ids] += vals` 是"缓冲赋值"
            # （`arr[ids] = arr[ids] + vals`），**重复索引时只保留最后一个** ⇒
            # 同一发送者同 tick 多笔转移会**少记回馈** ⇒ budget 虚高 ⇒ 突破 C-9 封顶。
            # [实测] `a[[1,1]] += [1,2]` ⇒ [0,2,0]（非 [0,3,0]）。
            np.add.at(self._oracle_gain, self._id[s_kept], g_kept)
            self._oracle_transfers += total
            self._oracle_count += cnt
            # R102 条件 1/3：三账 + 偿付约束（逐笔累加）
            self._oracle_payer_paid += float(led["payer_paid"])
            self._oracle_sender_received += float(led["sender_received"])
            self._oracle_payer_trunc_n += int(led["payer_trunc_n"])
            self._oracle_payer_trunc_amt += float(led["payer_trunc_amt"])
            self._oracle_budget_trunc_n += int(led["budget_trunc_n"])
            self._oracle_budget_trunc_amt += float(led["budget_trunc_amt"])
            self._oracle_payer_broke_n += int(led["payer_broke_n"])
            self._oracle_budget_exhausted_n += int(led["budget_exhausted_n"])
            # 内评 §三 观察项 1：记下**成交落点**（下一 tick 进食时实测接收侧摄入）
            ai = led["applied_idx"]
            if ai:
                self._recv_pend_cells.extend(
                    cc[ok][np.asarray(ai, dtype=np.int64)].tolist()
                )
                self._recv_pay_sum += float(led["payer_paid"])

    def signal_response_stats(self) -> dict:
        """D-18 ⑥ 三联报（累计口径）：⑥a 暴露率 / ⑥b mean Δ_i / ⑥ = ⑥a×⑥b + 翻转率。

        t=0 时 ⑥≡0 不是 bug（信号场初始无标记，V-7 G-3）。argmax 翻转率仅辅助
        （τ=0.15 概率抽样下主口径是 Δ 概率差，禁用 argmax 翻转作主判据——R43）。
        """
        dec = int(self._resp_decisions)
        exp_ = int(self._resp_exposed)
        a = (exp_ / dec) if dec else 0.0
        b = (self._resp_delta_sum / exp_) if exp_ else 0.0
        b_content = (self._resp_delta_content_sum / exp_) if exp_ else 0.0
        return {
            "decisions": dec,
            "exposed": exp_,
            "resp_a_exposure": round(float(a), 6),
            "resp_b_delta": round(float(b), 6),
            # R123/B②：**内容项单去**的 Δ（付款闸口径；`resp_b_delta` = 两项全去）
            "resp_b_content_delta": round(float(b_content), 6),
            "resp_triple": round(float(a * b), 6),
            "argmax_flip_rate": round(
                float(self._resp_flip / exp_) if exp_ else 0.0, 6
            ),
        }

    def inheritance_stats(self) -> dict:
        """R121 §4.2：记忆继承卫生的**零机时观测**（"生而知之"量化）。

        `mem_inherit_far_frac` = 出生时继承记忆中"距出生格 > 感知半径"的槽位占比。
        距离口径 = **格距**（行差 + 经度环绕列差，与邻居表同口径），**未做纬度余弦缩放**
        ⇒ 极区会高估 ⇒ 该值应读作**上界**。纯观测：不改状态、不消费随机流。
        """
        n = int(self._mem_inherit_n)
        far = int(self._mem_inherit_far_n)
        return {
            "mem_inherit_n": n,
            "mem_inherit_far_n": far,
            "mem_inherit_far_frac": round(far / n, 6) if n else 0.0,
            "distance_convention": "grid_ring(行差+经度环绕列差)，未做纬度余弦缩放（上界）",
        }

    # ------------------------------------------------ A′ 记忆**朝向梯度**（2026-09-19）
    def _memory_orientation_cos(self, cur: int, nb: np.ndarray,
                                valid_mem: np.ndarray) -> np.ndarray:
        """A′：候选邻居格方向 vs **记忆点方向** 的 `max cos`（每格一个值）。

        `cos ∈ [-1, 1]` ⇒ **背向邻居会被减分** —— 这才是"梯度"而非单纯吸引。
        取 `max` 而非 `Σ`：只让**最对准的那个记忆点**说话（语义干净，槽位多不会放大）。

        ⚠️ **双路径契约**：表达式与 Rust（`sim_core/src/movement.rs` 的 `orientation` 分支）
        **逐字同式**，包括经度环绕（Rust `rem_euclid` ≡ Python `%`（正除数即 floor-mod））
        ⇒ 保**逐位一致**；改任一侧必须同步改另一侧（F-R23 家族的正面预防）。
        """
        cols = int(self.world.cols)
        half = cols / 2.0
        cr, cc = divmod(int(cur), cols)
        nr, nc = np.divmod(nb.astype(np.int64), cols)
        dnr = nr.astype(np.float64) - float(cr)
        dnc = ((nc.astype(np.float64) - float(cc)) + half) % cols - half
        mr, mc = np.divmod(valid_mem.astype(np.int64), cols)
        dmr = mr.astype(np.float64) - float(cr)
        dmc = ((mc.astype(np.float64) - float(cc)) + half) % cols - half
        un = np.sqrt(dnr * dnr + dnc * dnc)
        mn = np.sqrt(dmr * dmr + dmc * dmc)
        # 🔴 两处过滤，缺一即错（与 Rust 的 `mc < 0 { continue }` / `m == cur { continue }` 对应）：
        #   ① `-1` 空槽**必须**过滤 —— `divmod(-1, cols)` 会得到 (-1, 119) 这种"合法"行列
        #      ⇒ 不过滤就会把空槽当成真实记忆点（本测试正是这样抓出来的）
        #   ② 记忆点 == 当前格 ⇒ 方向为零向量 ⇒ 跳过
        ok = (valid_mem >= 0) & (mn > 0.0)
        if not ok.any():
            return np.zeros(len(nb), dtype=np.float64)
        dmr, dmc, mn = dmr[ok], dmc[ok], mn[ok]
        num = dnr[:, None] * dmr[None, :] + dnc[:, None] * dmc[None, :]
        den = un[:, None] * mn[None, :]
        return (num / den).max(axis=1)

    def memory_gradient_stats(self) -> dict | None:
        """A′ 的可观测性计数 —— **先证"测到了"，再判读结果**（E-022 缺口的直接对策）。

        非 `orientation` 模式 ⇒ **None（未适用）**，**不是 0**（同 R120 的 ratio n/a 口径）。
        """
        ifcfg = self.config.info_structure
        if str(ifcfg.memory_gradient) != "orientation":
            return None
        if not self._mem_grad_counts_valid:
            # Rust（use_sim_core=True）侧只做决策、不产计数 ⇒ **报 n/a，不冒充 0**
            return {
                "mode": "orientation",
                "gain": float(ifcfg.memory_gradient_gain),
                "counters_available": False,
                "note": "Rust 路径不产朝向梯度计数器（仅 Python 路径统计）⇒ 计数按 n/a 报",
            }
        d = int(self._mem_grad_dec)
        slots = int(self._mem_grad_slots)
        trig = int(self._mem_grad_trig)
        return {
            "mode": "orientation",
            "gain": float(ifcfg.memory_gradient_gain),
            "counters_available": True,
            "decisions": d,
            "with_slots": slots,
            "triggered": trig,
            "slots_frac": (round(slots / d, 6) if d else None),
            "trigger_frac": (round(trig / d, 6) if d else None),
        }

    def alphabet_stats(self) -> dict:
        """R113/R121 §3：字母表档位读数 + 码值域守卫（§3.5，纯观测）。

        `bad_code_n` 在 `"4"`/`"8"` 档下**必须恒为 0**：非零即意味着
        ① 码 0（"隐形发射" + **清除同格他人标记**，设计稿 §2.4）或
        ② 码越上限（接收端读到**未学过的槽**）——两者都是**静默**的行为改变。
        `"16"` 档下 0 是合法值 ⇒ `guard_expected_zero=False`（恒不累加）。
        """
        return {
            "signal_alphabet": self._alpha,
            "n_states": int(self._alpha_states),
            "code_max": int(self._alpha_code_max),
            "bad_code_n": int(self._alpha_bad_code_n),
            "guard_expected_zero": bool(self._alpha != "16"),
            # 2026-09-19：`mem_bit` 取值分布（闭合 E-022/E-023 记的可观测性缺口）
            # 非 "8" 档 ⇒ frac = None（**未适用**，不是 0 —— 同 R120 的 ratio n/a 口径）
            "mem_bit_on": int(self._mem_bit_on),
            "mem_bit_n": int(self._mem_bit_n),
            "mem_bit_frac": (round(self._mem_bit_on / self._mem_bit_n, 6)
                             if self._mem_bit_n else None),
        }

    def oracle_stats(self) -> dict:
        """D-8 oracle 累计统计（O-4/O-7 判读 + manifest 必录 `oracle_return_ratio`）。"""
        emissions = int(self._emit_count.sum())
        denom = emissions * EMISSION_COST
        ratio = (self._oracle_transfers / denom) if denom > 0 else 0.0
        return {
            "enabled": bool(self._oracle_on),
            "transfers": round(float(self._oracle_transfers), 6),
            "count": int(self._oracle_count),
            "emissions": emissions,
            "oracle_return_ratio": round(float(ratio), 6),
            "funnel": self.oracle_funnel(),   # D-26a 四环节诊断
            # R102 条件 1（修正版）：守恒三账审计（撤销"净注入"口径）
            "audit": self.oracle_audit(),
            # 内评 §四：对账字段（三账**应恒等**，由构造保证；**定位=对账+偿付量化**，
            # 不是"守恒检验"——守恒由转移结构即保证，加一个恒真字段必须标注定位）
            "ledger": self.oracle_ledger(),
            # 内评 §三 观察项 1/2：接收侧效应（方向翻转新引入；`[推断]` → 实测量化）
            "receiver_side": self.receiver_side_effects(),
            # R97 增益校准档（2026-09-16）：`m` 与校准臂登记 —— 供 C4 读回核对（C5 v2）
            # 与 R100 条件 5（机器强制拒收依据 `is_calibration_arm`）。
            "gain_multiplier": float(self.config.oracle.gain_multiplier),
            "is_calibration_arm": bool(self.config.oracle.is_calibration_arm),
            # R121 §4.1 零机时 counter：**阈值错位带** `(food_threshold, FOOD_RICH_LEVEL]`
            # 在 `true_sig` 中的计数与占比 ⇒ **裁定点**：占比 >5% 再议是否对齐阈值
            # （对齐属行为变更 ⇒ 须预注册，R121 明文）
            "food_band_true_sig": {
                "n": int(self._diag_food_band_true_sig),
                "frac_of_true_sig": (
                    round(self._diag_food_band_true_sig / self._diag_true_sig, 6)
                    if self._diag_true_sig else 0.0
                ),
                "band": f"(food_threshold, {FOOD_RICH_LEVEL}]",
            },
            # R123/B② 门控臂读数（仪器段；gate off 时 mode="none" 且计数恒 0）
            "gate": {
                "mode": ("delta_positive" if self._gate_on else "none"),
                "delta_metric": (self._gate_delta if self._gate_on else None),
                "arrivals": int(self._gate_arrivals),
                "pass": int(self._diag_gate_pass),
                "block": int(self._diag_gate_block),
                "block_frac": (
                    round(self._diag_gate_block / self._gate_arrivals, 6)
                    if self._gate_arrivals else 0.0
                ),
                "mean_delta_full_at_arrivals": (
                    round(self._gate_delta_full_sum / self._gate_arrivals, 6)
                    if self._gate_arrivals else 0.0
                ),
                "mean_delta_content_at_arrivals": (
                    round(self._gate_delta_content_sum / self._gate_arrivals, 6)
                    if self._gate_arrivals else 0.0
                ),
            },
        }

    def oracle_ledger(self) -> dict:
        """R102 条件 3 + 内评 §四：**对账字段**（`Σpayer 扣减 ≡ Σ收款入账`）+ 偿付约束量化。

        ⚠️ 定位说明（内评 §四 命名建议，已采纳）：三账恒等由 `apply_oracle` 的逐笔
        「扣减 == 入账」**结构**保证 ⇒ **本字段是"对账"而非"守恒检验"**，它抓不到
        "守恒被破坏"（那种情况由 `oracle_audit` 的独立实测 ΔΣenergy 负责证伪）。
        真正有信息量的是**偿付约束**三项：`payer_trunc_*`（付款方付不起 ⇒ 部分支付）、
        `payer_broke_n`（余额为 0 ⇒ 整笔跳过）、`budget_exhausted_n`（收款方额度用尽）。
        """
        return {
            "sum_transfers": round(float(self._oracle_transfers), 6),
            "payer_paid": round(float(self._oracle_payer_paid), 6),
            "sender_received": round(float(self._oracle_sender_received), 6),
            "identity_ok": bool(
                abs(self._oracle_payer_paid - self._oracle_sender_received) <= 1e-9
            ),
            "payer_trunc_n": int(self._oracle_payer_trunc_n),
            "payer_trunc_amt": round(float(self._oracle_payer_trunc_amt), 6),
            "payer_broke_n": int(self._oracle_payer_broke_n),
            "budget_trunc_n": int(self._oracle_budget_trunc_n),
            "budget_trunc_amt": round(float(self._oracle_budget_trunc_amt), 6),
            "budget_exhausted_n": int(self._oracle_budget_exhausted_n),
            "role": "对账（三账恒等由构造保证）+ 偿付约束量化；非守恒检验",
        }

    def receiver_side_effects(self) -> dict:
        """内评《复核-R102方向修复》§三 观察项 1：**接收侧净能量效应**。

        方向翻转（F-R18）把"接收者获益"变成"接收者付出" ⇒ 需量化：接收者
        **吃到（t+1 在成交落点的实测摄入）− 回付（t 的 payment）** 是否仍 ≥
        不通信者的平均摄入。若净值为负 ⇒ 移动到标记格变成净亏 ⇒ 行为会变
        （观察项 2 的 ⑥ 响应率下降即其表现；**不得**读成"信号无用"）。

        ⚠️ **口径（按 cell 键控，属近似）**：落点以**格**记录，同格若有多人，
        他们的摄入会一并计入接收侧；且接收者可能在进食前死亡 ⇒ 分子偏小。
        ⇒ 本字段作**量级判断**用，不作精确个体账。
        """
        ev = max(1, int(self._oracle_count))
        fn = max(1, int(self._recv_food_n))
        bn = max(1, int(self._all_food_n))
        mean_recv = self._recv_food_sum / fn
        mean_base = self._all_food_sum / bn
        mean_pay = self._recv_pay_sum / ev
        return {
            "events": int(self._oracle_count),
            "food_events": int(self._recv_food_n),
            "mean_intake_at_paid_cell": round(float(mean_recv), 6),
            "mean_intake_all_eaters": round(float(mean_base), 6),
            "mean_payment_per_event": round(float(mean_pay), 6),
            "net_eat_minus_pay": round(float(mean_recv - mean_pay), 6),
            "net_vs_baseline": round(float(mean_recv - mean_pay - mean_base), 6),
            "caveat": "落点按 cell 键控（同格多人并入；接收者可能进食前死亡）⇒ 量级判断用",
        }

    def oracle_audit(self) -> dict:
        """守恒三账审计（R102 §三 修正后口径）。

        撤销 R100 条件 1 原定的 `system_energy_injected`——机制复核已证
        `apply_oracle` 是**双向转移**（修复方向后 `energy[R] -= g; energy[S] += g`），
        **纯再分配、零注入** ⇒ **C-2 守恒成立，无需例外声明**。

        本审计在引擎侧**独立**核算：包住 oracle 调用前后实测 `Σenergy` 的变化，
        累加其绝对值（应恒 0，仅浮点误差）。三账恒等（`Σtransfers ≡ Σpayer 扣减
        ≡ Σ收款入账`）由 `apply_oracle` 的逐笔配对实现保证 ⇒ 本审计用于**证伪**它。
        """
        tol = 1e-6 + 1e-9 * max(1.0, abs(float(self._oracle_transfers)))
        return {
            "calls": int(self._audit_calls),
            "sum_energy_delta": round(float(self._audit_sum_delta), 9),
            "tolerance": tol,
            "conserved": bool(abs(self._audit_sum_delta) <= tol),
            "accounts": "Σtransfers ≡ Σpayer 扣减 ≡ Σ收款入账（构造保证）",
        }

    def oracle_funnel(self) -> dict:
        """D-26a oracle 四环节诊断漏斗（累计口径；内评 `_eval/D24判读预析` §4.1）。

        「发射→成功通信」的逐级衰减，用于定位 D-24 G-A 不过时卡在哪一环：

            ① emissions 发射事件数
              ⇒ had_signal  接收者落点**有信号**（人-次）
              ⇒ true_sig    其中落点**同时有食物**（= oracle 选中语义，require_food=True）
            ② attrib_found  落点登记过发送者 且 发送者**仍存活**（id→slot 解析成功）
              ⇒ in_window   且 在 persistence 归因窗口内
              ⇒ not_self    且 非自反馈（发送者 ≠ 接收者）
              ⇒ budget_ok   且 C-9 保本额度 > 0
            ④ applied       实际成交（能量转移发生）

        后级恒 ⊆ 前级（逐级互斥计数）。⚠️ `emissions` 是**发送者-次**，
        `had_signal`/`true_sig` 是**移动者-次**（量纲不同），跨量纲比值仅作量级参考。
        """
        ap = self._diag_applied
        return {
            "emissions": int(self._emit_count.sum()),
            "had_signal": self._diag_had_sig,
            "true_sig": self._diag_true_sig,
            "selected": self._diag_sel,
            "attrib_found": self._diag_attrib_found,
            "in_window": self._diag_in_window,
            "not_self": self._diag_not_self,
            # R123/B② 门控臂（gate off 时恒 0）：`gate_positive` = 闸门放行；`gate_block` = 被拦
            "gate_positive": self._diag_gate_pass,
            "gate_block": self._diag_gate_block,
            "budget_ok": self._diag_budget_ok,
            "applied": ap,
            "sel_to_applied": round(ap / self._diag_sel, 6) if self._diag_sel else 0.0,
            "had_sig_to_applied": (
                round(ap / self._diag_had_sig, 6) if self._diag_had_sig else 0.0
            ),
            "emissions_per_applied": (
                round(int(self._emit_count.sum()) / ap, 3) if ap else None
            ),
        }

    # ---- L10a 果实-种子传播（数值管线先行版） -----------------------------

    def _step_fruit_charge(self, P: int, genes: NDArray[np.float64]) -> None:
        """植物蓄力→结果（L10a 步骤 3.5）。

        植物判定：g19(ROOTING) >= plant_threshold。
        蓄力：每 tick += charge_rate × g8(PHOTOSYNTHESIS)。
        释放：蓄力 >= fruit_threshold 时，在当前格释放果实：
            fruit_grid[cell] += charge × fruit_ratio
            charge = 0
        纯确定性数值计算，无随机数，可下沉 Rust。
        """
        fcfg = self.config.fruit
        g19 = genes[:P, int(Gene.ROOTING)]
        g8 = genes[:P, int(Gene.PHOTOSYNTHESIS)]
        plant_mask = g19 >= fcfg.plant_threshold
        if not plant_mask.any():
            return

        if self._use_sim_core:
            # L10a C5：蓄力→结果下沉 Rust，双路径逐位一致
            self._sim_core.fruit_charge(
                self._flat[:P], genes[:P].reshape(-1), self.config.genome.gene_count,
                self._fruit_charge[:P], self._fruit_grid,
                int(Gene.ROOTING), int(Gene.PHOTOSYNTHESIS),
                fcfg.plant_threshold, fcfg.charge_rate, fcfg.fruit_threshold, fcfg.fruit_ratio,
            )
            return

        # 蓄力
        self._fruit_charge[:P][plant_mask] += fcfg.charge_rate * g8[plant_mask]

        # 释放判定
        ripe = plant_mask & (self._fruit_charge[:P] >= fcfg.fruit_threshold)
        if not ripe.any():
            return

        ripe_flat = self._flat[:P][ripe]
        ripe_charge = self._fruit_charge[:P][ripe]

        # 果实能量释放到格子（多植物同格累加）
        fruit_add = ripe_charge * fcfg.fruit_ratio
        np.add.at(self._fruit_grid, ripe_flat, fruit_add)

        # 蓄力清零
        self._fruit_charge[:P][ripe] = 0.0

    def _step_eat_fruit(self, P: int, energy: NDArray[np.float64],
                         genes: NDArray[np.float64]) -> None:
        """动物吃果实→能量转移（L10a 步骤 4.5）。

        动物判定：g19(ROOTING) < plant_threshold（非植物）。
        吃果实：当前格有果实（fruit_grid[cell] > 0）时，
            eat_amount = min(fruit_grid[cell] × eat_rate, stomach剩余空间)
            energy += eat_amount × digest_ratio
            fruit_grid[cell] -= eat_amount
        纯确定性数值计算，无随机数，可下沉 Rust。
        种子摄入（L10b）：本次先行版不实现，_seed_carried 保持 0。
        """
        fcfg = self.config.fruit
        g19 = genes[:P, int(Gene.ROOTING)]
        animal_mask = g19 < fcfg.plant_threshold
        if not animal_mask.any():
            return

        if self._use_sim_core:
            # L10a C5：吃果实→能量转移下沉 Rust，双路径逐位一致
            self._sim_core.eat_fruit(
                self._flat[:P], genes[:P].reshape(-1), self.config.genome.gene_count,
                energy[:P], self._fruit_grid,
                int(Gene.ROOTING),
                fcfg.plant_threshold, fcfg.eat_rate, fcfg.digest_ratio,
            )
            return

        animal_flat = self._flat[:P][animal_mask]
        cell_fruit = self._fruit_grid[animal_flat]
        has_fruit = cell_fruit > 0
        if not has_fruit.any():
            return

        # 吃果实量：格子果实 × eat_rate
        eat_amount = cell_fruit[has_fruit] * fcfg.eat_rate

        # 能量转移（注意：布尔索引链式 [animal_mask][has_fruit] 返回副本，
        # 必须用 flatnonzero 拿到原始索引才能就地修改）
        animal_idx = np.flatnonzero(animal_mask)
        eat_idx = animal_idx[has_fruit]
        energy[eat_idx] += eat_amount * fcfg.digest_ratio

        # 果实场减少（同格多动物累加扣减）
        eat_flat = animal_flat[has_fruit]
        np.add.at(self._fruit_grid, eat_flat, -eat_amount)
        # 防止浮点误差导致负值
        self._fruit_grid[self._fruit_grid < 0] = 0.0

    # ---- 愉悦度系统（L2） --------------------------------------------------

    def _update_pleasure(self, P: int, energy_before: NDArray[np.float64]) -> None:
        """对前 P 个亲代更新愉悦度（RPE 预测误差驱动）。

        核心：愉悦 = 实际获得 − 预期获得。
        1. 情境编码（120 种：能量5×食物4×邻居3×信号2）
        2. 事件收益（Δ能量 + 社会增益，信息增益待信号场接入）
        3. RPE = 收益 − expectation[情境]
        4. 更新 valence（瞬态）/arousal（唤醒）/expectation（EWMA 学习）/baseline（习惯化）

        只更新前 P 个个体（本 tick 开始时存在的亲代），子代不参与本 tick。
        """
        pcfg = self.config.pleasure
        max_e = self.config.organisms.max_energy
        idx = np.arange(P)
        flat = self._flat[:P]
        energy_now = self._energy[:P]

        if self._use_sim_core:
            # L7a C2：愉悦度更新下沉 Rust，双路径逐位一致
            densities = np.bincount(flat, minlength=self.world.n_cells).astype(np.float64)
            self._sim_core.pleasure_update(
                flat, energy_now, energy_before,
                densities, self.resources._grid, self.resources._capacity,
                self.signals._marks,
                self._valence[:P], self._arousal[:P],
                self._expectation[:P].reshape(-1), self._baseline[:P],
                max_e, pcfg.alpha, pcfg.valence_decay, pcfg.arousal_decay,
                pcfg.baseline_rate, pcfg.max_reward,
                pcfg.w_energy, pcfg.w_info, pcfg.w_social,
            )
            return

        # 1) 情境编码
        # 能量档：energy/max_energy → 0~4
        e_bin = np.clip((energy_now / max_e) * 5, 0, 4).astype(np.int64)
        # 食物档：所在格食物/capacity → 0~3
        food_ratio = np.clip(
            self.resources._grid[flat] / np.maximum(self.resources._capacity[flat], 1e-9),
            0.0, 1.0,
        )
        f_bin = np.clip((food_ratio * 4).astype(np.int64), 0, 3)
        # 邻居档：所在格密度 → 0(无)/1(1-2)/2(3+)
        densities = np.bincount(flat, minlength=self.world.n_cells)
        n_count = densities[flat]
        n_bin = np.where(n_count == 0, 0, np.where(n_count <= 2, 1, 2))
        # 信号档：第一版信号场未接入引擎，全 0（待 L3 信号基因接入后填）
        s_bin = np.zeros(P, dtype=np.int64)
        # 组合索引：e×24 + f×6 + n×2 + s
        context = e_bin * 24 + f_bin * 6 + n_bin * 2 + s_bin

        # 2) 事件收益
        delta_e = np.clip((energy_now - energy_before) / max_e, -1.0, 1.0)
        social = np.where(n_count > 0, pcfg.social_rpe, pcfg.alone_rpe)  # 有同伴→正，孤独→负
        # 信息增益：所在格有信号标记 → 获得信息（好奇心满足），权重提高
        info = np.where(self.signals._marks[flat] > 0, 0.5, 0.0)
        reward = pcfg.w_energy * delta_e + pcfg.w_info * info + pcfg.w_social * social

        # 3) RPE = 实际 − 预期
        expected = self._expectation[idx, context]
        rpe = reward - expected

        # 4) 更新四数组
        # valence：瞬态响应 + 衰减回中性
        self._valence[:P] += rpe * 0.3
        self._valence[:P] *= pcfg.valence_decay
        self._valence[:P] = np.clip(self._valence[:P], -1.0, 1.0)
        # arousal：意外事件（|RPE| 大）→ 高唤醒
        self._arousal[:P] += np.abs(rpe) * 0.2
        self._arousal[:P] *= pcfg.arousal_decay
        self._arousal[:P] = np.clip(self._arousal[:P], 0.0, 1.0)
        # expectation：EWMA 学习（慢半拍，那半拍就是愉悦来源）
        self._expectation[idx, context] += pcfg.alpha * rpe
        self._expectation[:P] = np.clip(self._expectation[:P], 0.0, pcfg.max_reward)
        # baseline：慢漂移（习惯化——长期愉悦/不愉悦会被适应）
        self._baseline[:P] = (
            self._baseline[:P] * (1.0 - pcfg.baseline_rate)
            + self._valence[:P] * pcfg.baseline_rate
        )

    def pleasure_summary(self) -> dict:
        """愉悦度系统的统计快照（给观察者/实验用）。"""
        if len(self._id) == 0:
            return {"valence_mean": 0.0, "arousal_mean": 0.0, "baseline_mean": 0.0}
        return {
            "valence_mean": float(self._valence.mean()),
            "valence_std": float(self._valence.std()),
            "arousal_mean": float(self._arousal.mean()),
            "baseline_mean": float(self._baseline.mean()),
            "expectation_mean": float(self._expectation.mean()),
        }

    # ============================================================
    # 快照机制：支持长实验分段续跑（v2，适配 L10a + 统计数组）
    # ============================================================

    SNAPSHOT_VERSION = 3

    def save_snapshot(self, path: str) -> None:
        """保存完整引擎状态到 npz 文件（压缩）。

        保存内容：种群数组 + 愉悦度/学习数组 + L10a 果实场 + 资源场/信号场状态
        + RNG 状态 + 元数据 + 配置指纹。
        恢复后继续运行的结果与"不保存连续运行"逐位一致（可复现）。

        v2 变更：新增 _run_born/_run_died/_run_deaths 统计 + L10a 果实场数组
        + resources 再生参数；与 v1 快照不兼容。
        """
        import json
        import pickle

        P = len(self._id)
        data = {}

        # --- 1. 种群 SoA 数组（只保存有效部分 [:P]）---
        data["id"] = self._id[:P].copy()
        data["flat"] = self._flat[:P].copy()
        data["energy"] = self._energy[:P].copy()
        data["stomach"] = self._stomach[:P].copy()
        data["genes"] = self._genes[:P].copy()
        data["age"] = self._age[:P].copy()
        data["generation"] = self._generation[:P].copy()
        data["parent"] = self._parent[:P].copy()
        data["repro_cooldown"] = self._repro_cooldown[:P].copy()

        # --- 2. 愉悦度数组 ---
        data["valence"] = self._valence[:P].copy()
        data["arousal"] = self._arousal[:P].copy()
        data["expectation"] = self._expectation[:P].copy()  # (P, 120)
        data["baseline"] = self._baseline[:P].copy()

        # --- 3. 学习数组 ---
        data["trust"] = self._trust[:P].copy()
        data["work_memory"] = self._work_memory[:P].copy()  # (P, 4)
        data["mem_ptr"] = np.array(self._mem_ptr)
        data["interpret"] = self._interpret[:P].copy()  # (P, 16)

        # --- 3.1 D2 信息结构数组 ---
        data["codebook"] = self._codebook[:P].copy()  # (P, 16) uint8
        data["learning_count"] = self._learning_count[:P].copy()  # (P,) int32

        # --- 3.5 L10a 果实-种子传播数组 ---
        data["fruit_grid"] = self._fruit_grid.copy()  # (n_cells,)
        data["fruit_charge"] = self._fruit_charge[:P].copy()
        data["seed_carried"] = self._seed_carried[:P].copy()

        # --- 4. 世界状态（资源场 + 信号场）---
        data["resource_grid"] = self.resources._grid.copy()
        data["resource_capacity"] = self.resources._capacity.copy()
        data["resource_patch_mask"] = (
            self.resources._patch_mask.copy()
            if self.resources._patch_mask is not None
            else np.zeros(0, dtype=bool)
        )
        data["resource_bg_regrowth_mult"] = np.array(self.resources._bg_regrowth_mult)
        data["resource_patch_regrowth_mult"] = np.array(self.resources._patch_regrowth_mult)
        data["signal_marks"] = self.signals._marks.copy()
        data["signal_age"] = self.signals._age.copy()

        # --- 5. 运行统计 ---
        data["run_born"] = np.array(self._run_born)
        data["run_died"] = np.array(self._run_died)
        # _run_deaths 是 Counter，用 pickle 序列化
        data["run_deaths"] = np.array(pickle.dumps(dict(self._run_deaths)), dtype=object)

        # --- 5.5 D-17/D-18/D-8 探针/oracle 状态（v3 追加键；旧快照加载时回退零值）---
        # 续跑正确性关键：⑤ 的"已观测"标记、oracle 保本记账（_emit_count/_oracle_gain/
        # _last_sender）必须随快照走，否则续跑后重复观测/归因失效（F-R7 同型教训）。
        data["last_sender"] = self._last_sender.copy()
        data["oracle_transfers"] = np.array(self._oracle_transfers)
        data["oracle_count"] = np.array(self._oracle_count)
        data["resp_decisions"] = np.array(self._resp_decisions)
        data["resp_exposed"] = np.array(self._resp_exposed)
        data["resp_delta_sum"] = np.array(self._resp_delta_sum)
        data["resp_flip"] = np.array(self._resp_flip)
        if self._measure_resp or self._oracle_on:
            data["rs_children"] = self._rs_children.copy()
            data["rs_observed"] = self._rs_observed.copy()
            data["rs_g15"] = self._rs_g15.copy()
            data["rs_age"] = self._rs_age.copy()
            data["rs_energy"] = self._rs_energy.copy()
            data["rs_cc0"] = self._rs_cc0.copy()
            data["rs_cohort"] = self._rs_cohort.copy()
            data["emit_count"] = self._emit_count.copy()
            data["oracle_gain"] = self._oracle_gain.copy()
            # R123/B②：门控臂的逐个体 Δ 与计数（旧快照无此键 ⇒ 载入侧回退）
            data["delta_full_by_id"] = self._delta_full_by_id.copy()
            data["delta_content_by_id"] = self._delta_content_by_id.copy()
            data["delta_tick_by_id"] = self._delta_tick_by_id.copy()
            data["gate_counters"] = np.array(
                [self._diag_gate_pass, self._diag_gate_block, self._gate_arrivals,
                 self._gate_delta_full_sum, self._gate_delta_content_sum],
                dtype=np.float64,
            )
        # D-26a 四环节诊断计数（oracle off 时恒零，仍随快照走以保证续跑后累计不失真）
        data["diag_funnel"] = np.array(
            [
                self._diag_had_sig, self._diag_true_sig, self._diag_sel,
                self._diag_attrib_found, self._diag_in_window, self._diag_not_self,
                self._diag_budget_ok, self._diag_applied,
            ],
            dtype=np.int64,
        )
        # R121 §3/§4 计数器（续跑后累计不失真；旧快照无此键 ⇒ 回退零值）
        data["r121_counters"] = np.array(
            [
                self._diag_food_band_true_sig, self._mem_inherit_n,
                self._mem_inherit_far_n, self._alpha_bad_code_n,
                self._mem_bit_on, self._mem_bit_n,
            ],
            dtype=np.int64,
        )

        # --- 6. RNG 状态（可复现的关键）---
        data["rng_state"] = np.array(
            pickle.dumps(self.rng.bit_generator.state), dtype=object
        )

        # --- 7. 元数据 ---
        data["snapshot_version"] = np.array(self.SNAPSHOT_VERSION)
        data["tick"] = np.array(self._tick)
        data["next_id"] = np.array(self._next_id)
        data["max_generation"] = np.array(self._max_generation)
        data["count"] = np.array(P)
        data["extinct"] = np.array(self._extinct)
        data["finished"] = np.array(self._finished)
        data["gene_count"] = np.array(self.config.genome.gene_count)
        data["use_sim_core"] = np.array(self._use_sim_core)
        data["config_fingerprint"] = np.array(self.config.fingerprint())
        data["config_dict"] = np.array(
            json.dumps(self.config.to_dict(), ensure_ascii=False), dtype=object
        )

        np.savez_compressed(path, **data)

    @classmethod
    def load_snapshot(cls, path: str, config=None):
        """从快照文件恢复引擎。

        Args:
            path: 快照文件路径
            config: 引擎配置。为 None 时从快照中恢复配置；
                    提供时校验 fingerprint 与快照一致。

        Returns:
            恢复后的 SphereEngine 实例

        Raises:
            ValueError: 快照版本不兼容 / 配置指纹不匹配 / gene_count 不匹配
        """
        import json
        import pickle
        from collections import Counter

        data = np.load(path, allow_pickle=True)

        # --- 1. 版本校验 ---
        snap_version = int(data["snapshot_version"])
        if snap_version != cls.SNAPSHOT_VERSION:
            raise ValueError(
                f"快照版本 {snap_version} 不兼容，当前版本 {cls.SNAPSHOT_VERSION}"
            )

        # --- 2. 配置处理 ---
        if config is None:
            from simulation.config import SimConfig
            config = SimConfig.from_dict(json.loads(str(data["config_dict"])))
        else:
            snap_fp = str(data["config_fingerprint"])
            cur_fp = config.fingerprint()
            if snap_fp != cur_fp:
                raise ValueError(
                    f"配置指纹不匹配：快照={snap_fp[:16]}...，当前={cur_fp[:16]}..."
                )

        # --- 3. gene_count 校验 ---
        snap_gc = int(data["gene_count"])
        if snap_gc != config.genome.gene_count:
            raise ValueError(
                f"gene_count 不匹配：快照={snap_gc}，当前={config.genome.gene_count}"
            )

        # --- 4. 创建引擎（__init__ 会创建初始种群，后续覆盖）---
        engine = cls(config)

        # --- 5. 恢复种群数组 ---
        engine._id = data["id"].copy()
        engine._flat = data["flat"].copy()
        engine._energy = data["energy"].copy()
        engine._stomach = data["stomach"].copy()
        engine._genes = data["genes"].copy()
        engine._age = data["age"].copy()
        engine._generation = data["generation"].copy()
        engine._parent = data["parent"].copy()
        engine._repro_cooldown = data["repro_cooldown"].copy()

        # --- 6. 恢复愉悦度数组 ---
        engine._valence = data["valence"].copy()
        engine._arousal = data["arousal"].copy()
        engine._expectation = data["expectation"].copy()
        engine._baseline = data["baseline"].copy()

        # --- 7. 恢复学习数组 ---
        engine._trust = data["trust"].copy()
        engine._work_memory = data["work_memory"].copy()
        engine._mem_ptr = int(data["mem_ptr"])
        engine._interpret = data["interpret"].copy()

        # --- 7.1 恢复 D2 信息结构数组（v3 新增；旧快照回退默认值）---
        if "codebook" in data:
            engine._codebook = data["codebook"].copy()
        else:
            engine._codebook = codebook_init_rows(len(engine._id), engine._alpha)
        if "learning_count" in data:
            engine._learning_count = data["learning_count"].copy()
        else:
            engine._learning_count = np.zeros(len(engine._id), dtype=np.int32)

        # --- 7.5 恢复 L10a 数组 ---
        engine._fruit_grid = data["fruit_grid"].copy()
        engine._fruit_charge = data["fruit_charge"].copy()
        engine._seed_carried = data["seed_carried"].copy()

        # --- 8. 恢复世界状态 ---
        engine.resources._grid = data["resource_grid"].copy()
        engine.resources._capacity = data["resource_capacity"].copy()
        _pm = data["resource_patch_mask"]
        engine.resources._patch_mask = _pm.copy() if _pm.size > 0 else None
        engine.resources._bg_regrowth_mult = float(data["resource_bg_regrowth_mult"])
        engine.resources._patch_regrowth_mult = float(data["resource_patch_regrowth_mult"])
        engine.signals._marks = data["signal_marks"].copy()
        engine.signals._age = data["signal_age"].copy()

        # --- 8.5 恢复运行统计 ---
        engine._run_born = int(data["run_born"])
        engine._run_died = int(data["run_died"])
        engine._run_deaths = Counter(pickle.loads(data["run_deaths"].item()))

        # --- 8.6 恢复 D-17/D-18/D-8 探针/oracle 状态（旧快照回退零值/重建）---
        engine._last_sender = (
            data["last_sender"].copy()
            if "last_sender" in data
            else np.full(engine.world.n_cells, -1, dtype=np.int64)
        )
        engine._oracle_transfers = float(data["oracle_transfers"]) if "oracle_transfers" in data else 0.0
        engine._oracle_count = int(data["oracle_count"]) if "oracle_count" in data else 0
        engine._resp_decisions = int(data["resp_decisions"]) if "resp_decisions" in data else 0
        engine._resp_exposed = int(data["resp_exposed"]) if "resp_exposed" in data else 0
        engine._resp_delta_sum = float(data["resp_delta_sum"]) if "resp_delta_sum" in data else 0.0
        engine._resp_flip = int(data["resp_flip"]) if "resp_flip" in data else 0
        if "diag_funnel" in data:   # D-26a：旧快照回退全零
            _df = data["diag_funnel"]
            (engine._diag_had_sig, engine._diag_true_sig, engine._diag_sel,
             engine._diag_attrib_found, engine._diag_in_window, engine._diag_not_self,
             engine._diag_budget_ok, engine._diag_applied) = (int(x) for x in _df)
        if "r121_counters" in data:   # R121 §3/§4：旧快照回退零值（不报错，语义=未观测）
            # 🔴 长度守卫（内评未闭合项 #7，2026-09-20 修）：
            #   原写法是 **6 元元组解包** ⇒ 长度必须**严格相等**；旧快照（4 元）续跑到
            #   新版会抛 ValueError 且**看不出是这个计数器**（只会看到 "not enough values"）。
            #   改为：短 ⇒ 右侧补 0（语义 = 新增计数器未观测）；长 ⇒ 截断并**显式告警**
            #   （多余的计数会被丢弃，这是信息丢失，不能静默）。
            _rc = np.asarray(data["r121_counters"], dtype=np.int64).reshape(-1)
            if _rc.size < 6:
                _rc = np.concatenate([_rc, np.zeros(6 - _rc.size, dtype=np.int64)])
            elif _rc.size > 6:
                print(
                    f"  [WARN] snapshot r121_counters has {_rc.size} entries but this "
                    f"version reads 6 => the last {_rc.size - 6} counter(s) are dropped."
                )
                _rc = _rc[:6]
            (engine._diag_food_band_true_sig, engine._mem_inherit_n,
             engine._mem_inherit_far_n, engine._alpha_bad_code_n,
             engine._mem_bit_on, engine._mem_bit_n) = (int(x) for x in _rc)
        if "rs_children" in data:
            engine._rs_children = data["rs_children"].copy()
            engine._rs_observed = data["rs_observed"].copy()
            engine._rs_g15 = data["rs_g15"].copy()
            engine._rs_age = data["rs_age"].copy()
            engine._rs_energy = data["rs_energy"].copy()
            engine._rs_cc0 = data["rs_cc0"].copy()
            engine._rs_cohort = data["rs_cohort"].copy()
            engine._emit_count = data["emit_count"].copy()
            engine._oracle_gain = data["oracle_gain"].copy()
            # R123/B②：门控臂状态（旧快照缺键 ⇒ 按零/未变处理，不静默错位）
            for _k, _attr, _dt in (
                ("delta_full_by_id", "_delta_full_by_id", np.float32),
                ("delta_content_by_id", "_delta_content_by_id", np.float32),
                ("delta_tick_by_id", "_delta_tick_by_id", np.int32),
            ):
                if _k in data:
                    setattr(engine, _attr, data[_k].copy())
                else:
                    _arr = np.zeros(int(engine._next_id), dtype=_dt)
                    if _dt is np.int32:
                        _arr[:] = -1
                    setattr(engine, _attr, _arr)
            if "gate_counters" in data:
                _gc = data["gate_counters"]
                (engine._diag_gate_pass, engine._diag_gate_block, engine._gate_arrivals,
                 engine._gate_delta_full_sum, engine._gate_delta_content_sum) = (
                    int(_gc[0]), int(_gc[1]), int(_gc[2]), float(_gc[3]), float(_gc[4]))
            engine._gate_on = bool(engine.config.oracle.gate_mode == "delta_positive")
            engine._gate_delta = str(engine.config.oracle.gate_delta)
        else:
            # 旧快照（无探针数组）：_id 键控账本扩到 _next_id（全零 = 未观测/未发射）
            need = int(engine._next_id) - len(engine._rs_children)
            engine._grow_id_arrays(need)

        # --- 9. 恢复 RNG 状态 ---
        rng_state = pickle.loads(data["rng_state"].item())
        engine.rng.bit_generator.state = rng_state

        # --- 10. 恢复元数据 ---
        engine._tick = int(data["tick"])
        engine._next_id = int(data["next_id"])
        engine._max_generation = int(data["max_generation"])
        engine._extinct = bool(data["extinct"])
        engine._finished = bool(data["finished"])

        return engine