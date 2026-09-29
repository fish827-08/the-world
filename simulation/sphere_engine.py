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
# 13.4 波 1：亚格连续坐标（本线 = [所有者·天平]）。纯几何工具，零状态 ⇒ 导入无副作用。
from world.subpos import (
    advance_sub as _sub_advance,
    flat_of_sub as _sub_flat,
    init_from_flat as _sub_init,
    speed_steps as _sub_steps,
)
# 13.4 波 2A：斑块休耕—死亡—轮作（T2，本线 = [云端·开发]）。收编件（0207aa5），纯逻辑、零引擎依赖。
from world.resource_dynamics import ResourceDynamics as _ResourceDynamics


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
EC_FORAGE, EC_PRED, EC_PHOTO, EC_META, EC_MOVE, EC_ATTACK, EC_SCAV = range(7)
EC_N = 7
EC_NAMES = ("intake_forage", "intake_pred", "intake_photo",
            "cost_meta", "cost_move", "cost_attack", "intake_scav")
# 🔴 `intake_scav`（R165 0-2，2026-09-23 立）：**尸体来源的收入单列**。
#    此前食腐质量在胃里与植物质量不可区分 ⇒ 消化时整笔记进 `intake_forage`，
#    账本无法分离"尸体收入 vs 植物收入"（复核发现 3）。现按胃内比例归因拆开，
#    **两者之和恒等于原 `digest × eat_efficiency`** ⇒ 既有净收入数值不变、只变得可分离。
# 三腿的分组名（派工单 §1.1：**每 tick 按个体当前 g16 现算**，不用出生标签 ⇒ 防杂交后失效）
EC_BOX_NAMES = ("lo", "mid", "hi")   # g16 < 1/3 / [1/3, 2/3] / > 2/3
# 收入侧通道（净收入 = 收入 − 支出；缺 `intake_photo` 会让净收入虚假为负——
# 光合是独立于取食的**直接收入**，不经过胃）。

# R141/R138：真决斗三级拆分的键（**顺序固定**；全部预置 0 ⇒ 输出不随事件有无而变）
#   nominal = len(attackers)：名义攻击者（= `cannibalism.n_attacks`，旧口径的分母）
#   skip_*  = 三类"根本没出手"的原因；real_attempts = 真出手 = nominal − 三类 skip
#   wounds   = 出手成功但**未致死**（血条模式：health 未归零；S2，wound_enabled 时累加）
#   kills   = 致死（= `cannibalism.n_kills`）
# ⇒ **真实出手成功率 = kills / real_attempts**（旧口径 kills / nominal 会低估）
DUEL_KEYS = ("skip_no_energy", "skip_no_prey", "skip_already_eaten",
             "real_attempts", "wounds", "kills")


def _build_patch_centroid(world, patch_mask: np.ndarray) -> NDArray[np.int64]:
    """S3 记忆 v2：斑块连通域标注 + 每斑块质心格（构造期一次性；零 RNG ⇒ C7 无关）。

    返回 shape=(n_cells,) 的 int64 数组：`out[i]` = 格 i 所属斑块的**质心格 id**
    （非斑块格 = -1；无斑块 = 全 -1）。质心格 ∈ 该斑块自身（到平均行列最近者）。

    ⚠️ 经度缠绕：斑块内 col 平均用**角度平均**（sin/cos 均值再反正切）而非直接平均，
       避免跨 0/cols 边界的斑块被"拉平"到错误经度；行平均直接取算术平均（无环绕）。
    """
    n = world.n_cells
    cols = world.cols
    out = np.full(n, -1, dtype=np.int64)
    patch_cells = np.flatnonzero(patch_mask)
    if patch_cells.size == 0:
        return out
    # 连通域标注（BFS；world.neighbors 尊重拓扑 + 经度缠绕；与探针 _label_patches 同式）
    labels = np.full(n, -1, dtype=np.int64)
    nxt = 0
    for seed in patch_cells:
        if labels[seed] != -1:
            continue
        lab = nxt
        nxt += 1
        q = [int(seed)]
        labels[seed] = lab
        while q:
            c = q.pop()
            for nb in world.neighbors(c):
                nb = int(nb)
                if patch_mask[nb] and labels[nb] == -1:
                    labels[nb] = lab
                    q.append(nb)
    for lab in range(nxt):
        cells = np.flatnonzero(labels == lab)
        rs = cells // cols
        cs = cells % cols
        ang = 2.0 * np.pi * cs.astype(np.float64) / float(cols)
        mc = int(round(
            (np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
             % (2.0 * np.pi)) / (2.0 * np.pi) * float(cols))) % cols
        mr = int(round(float(rs.mean())))
        best = int(cells[0])
        best_d = float("inf")
        for c in cells:
            cr = int(c) // cols
            cc = int(c) % cols
            dr = float(abs(cr - mr))
            dcol = float(min(abs(cc - mc), cols - abs(cc - mc)))
            d = dr * dr + dcol * dcol
            if d < best_d:
                best_d = d
                best = int(c)
        out[cells] = best
    return out


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
        # S3 记忆 v2（egocentric）5 数组 + 质心表（实施规格 §二；R215 教训：漏登 = 无法赋值/续跑错位）
        "_mem_az", "_mem_dist", "_mem_tick", "_mem_degraded", "_mem_food",
        "_patch_centroid", "_mem_gain", "_mem_coarse_gain", "_mem_dist_scale",
        "_mem_degrade_thr", "_mem_ttl", "_mem_noise", "_mem_noise_p",
        "_mem_weight_gene", "_mem_v2_rng",
        # P0.1（T1）：紧凑邻居表 —— `_nb_table` 引用 world 的 `(n_cells,8)`，极点带另存
        "_pole_nb", "_pole_top", "_pole_bottom",
        # A′ 记忆朝向梯度计数器（2026-09-19）—— 本类用 `__slots__`，**新属性必须登记否则无法赋值**
        "_mem_grad_dec", "_mem_grad_slots", "_mem_grad_trig", "_mem_grad_counts_valid",
        "_codebook", "_learning_count",  # D2 信息结构：任意性码本 + 学习瓶颈计数
        "_tick", "_extinct", "_finished", "_history",
        "_history_limit", "_run_born", "_run_died", "_run_deaths",
        "_use_sim_core", "_sim_core",
        # R217 §五 #1 稀疏化 B：本属性 = **资源侧**范围闸（配置开关 ∧ py 路径 ∧ ¬资源动态）；
        #   逐字段开关在 `resources._lazy` / `signals._sparse`（R231 T-E 起两侧解耦，
        #   `_sparse_fields=False` **不再**意味着信号侧退场）
        "_sparse_fields",
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
        "_far_off_c", "_far_cells_c", "_far_len_c", "_far_excluded_c", "_far_ready",
        "_cap_floor",
        "_run_mover_n", "_run_dash_n", "_mover_n_by_box", "_dash_n_by_box",
        "_mob_sum", "_mob_sq_sum", "_mob_n", "_agef_sum",
        "_inelig_pop_n", "_pop_n", "_g16_le_gate_n",
        # R146/R149 L1 感知追击（本段 = [所有者·天平] 线 = B1）：两项逐候选 + 反退化计数
        "_seek_term_sum", "_seek_term_n", "_seek_flat_n", "_seek_zero_n",
        "_fear_term_sum", "_fear_term_n", "_fear_flat_n", "_fear_ind_n", "_l1_dec_n",
        # S1 骨架（设计稿 §5.2 项 1/9）：尸体格数组 + 个体血条数组 + 空壳读数计数器
        # ⚠️ __slots__ 是硬约束：新属性必须在此登记，否则运行期 AttributeError
        "_corpse_energy", "_corpse_age", "_health",
        # S2 主机制（设计稿 §5.3 项 3）：腐烂处资源 +50% 的 boost 剩余 tick 格数组
        "_corpse_boost",
        "_corpse_eaten_n", "_wound_n", "_contest_n",
        # 🔴 R165 0-2：**胃内尸源质量归因**（质量单位，恒 ≤ `_stomach`）
        "_stomach_scav",
        # 🔴 R165 0-1：尸体池**收支闭账**四计数器（能量单位；使池守恒可自证）
        "_corpse_deposited_e", "_corpse_overflow_e",
        "_corpse_scav_e", "_corpse_decayed_e",
        # S3 交互（设计稿 §5.4 项 6/7）：争夺战持有者胜率计数 + 血条恐惧项反退化计数
        "_contest_holder_win_n", "_fearh_flat_n", "_fearh_dec_n",
        # ---- 13.4 波 1：亚格连续坐标（本线 = [所有者·天平]）----
        # 🔴 __slots__ 是硬约束：新属性**必须**登记，否则运行期 AttributeError。
        # 位置约定：**加在块尾**（与 [本地开发] 的波 0 段分开，减少同文件并发冲突面）。
        "_sub_r", "_sub_c",
        # 读数（B4 口径；关档全部不累加 ⇒ 探针返回 None 或 0 的口径见 subpos_probe）
        "_run_flat_move_n",   # Σ 真正**换格**的个体数（= 主判据 mean_flat_moves 的分子）
        "_run_slow_n",        # Σ 移动者中 steps==0（"白移动"）的个体数
        "_run_mover_sub_n",   # Σ 移动者数（分母）
        "_steps_hist",        # 各步长档计数（形状 subdiv+1）
        # ---- 13.4 波 2A：资源动态（T2，本线 = [云端·开发]）----
        # 🔴 __slots__ 硬约束：新属性必须登记（加在块尾，减少同文件并发冲突面）。
        "_rd",                # ResourceDynamics 实例（收编件 world/resource_dynamics.py）
        "_rd_intake_sum",     # 本 tick 每格被吃量累计（note_tick 输入，逐 tick 重建）
        "_rd_growth_sum",     # 本 tick 每格名义再生量累计（note_tick 输入，逐 tick 重建）
        # ---- 13.4 波 2B（T3，本线 = [云端·开发]）：F1 修复 / span / cap ----
        "_nb_len",            # 每格**实际邻居数**（F1 修复的分母；span=1 语义）
        "_nb_norm",           # densities 归一化除数（"auto"=逐格实际邻居数，数字=冻结常量）
        "_span_table",        # span=2 邻居表（ring1+ring2 合并；span=1 = None）
        "_span_len",          # 每格 span 候选数（span=1 = 实际邻居数）
        "_span_eff2",         # 每格是否生效 2 圈（F3 闸：>cap ⇒ False = 降级 1 圈）
        "_span_cap",          # F3 闸阈值（perception_cap）
        # cap 读数（T3）：满格剔除计数 / 留本格兜底计数
        "_cap_blocked_n",     # Σ 因满格被剔除的候选选择（cap 生效时的决策数）
        "_cap_stay_n",        # Σ 全候选满 ⇒ 留本格的个体数（兜底）
        # ---- 13.8 日历—罗盘式定向迁徙（本线 = [云端·开发] 云归）----
        # 🔴 __slots__ 是硬约束：新属性**必须**登记，否则运行期 AttributeError。
        # 位置约定：**加在块尾**（与既有各线分开，减少同文件并发冲突面）。
        "_mig_on",            # 开关快照（构造期取一次；关档整段不执行 ⇒ 逐位等价）
        "_mig_gain",          # 全局增益（config.migration.gain）
        "_mig_min_abs",       # |A| 门槛（config.migration.min_abs_anomaly；A∈[−0.5,0.5]）
        "_lat_abs",           # 逐格 |φ|（弧度；**一次性**预计算，罗盘轴的"米尺"）
        "_pp_anom",           # 逐格 (P−0.5)（日历轴；**逐 tick** 刷新，见 _refresh_pp_anom）
        "_pp_anom_tick",      # 上列数组对应的 tick（去重；−1 = 尚未刷新）
        # 读数（B4 口径；关档全部不累加 ⇒ migration_probe() 返回 None = "未适用"）
        "_mig_dec_n",         # 求值的个体数（分母；len(nb)==1 已提前 continue）
        "_mig_term_sum",      # Σ|项|（按候选；**绝对值**，用于 P6 量级非僭越判定）
        "_mig_term_n",        # 候选数（分母）
        "_mig_zero_n",        # 其中"项跨候选恒为 0"的个体数（= 自熄：A≈0 或全同纬）
        "_mig_flat_n",        # 🔴 其中"跨候选取同值"的个体数（R148-1 反退化必报）
         "_mig_skip_n",        # 其中因 |A| ≤ 门槛而**提前跳过**的个体数（分段诊断）
         "_mig_pp_lo",         # 运行期内 A 的**最小值**（累积；P3/P2 诊断：跨季是否真有结构）
         "_mig_pp_hi",         # 运行期内 A 的**最大值**（累积）
         # ---- 14.9 ARS 双模式觅食（本线 = [所有者·天平]）--------------------------
         "_ars_on",            # 开关快照（关档 ⇒ 整段不执行 ⇒ 逐位等价）
         "_ars_gain",          # 惯性权重 w_pers
         "_ars_theta",         # 快 < θ×慢 ⇒ 切赶路
         "_ars_kappa",         # 油箱对称耦合系数
         "_ars_giveup",        # 赶路模式连走多少 tick 没咬到就重选方向
         "_ars_fast_tau", "_ars_slow_tau", "_ars_lat_exempt",
         "_ars_cos",           # (8,8) 槽位方向余弦矩阵（由参考格邻居表推导）
        "_ars_unit",          # (8,2) 槽位单位方向向量（S3 记忆 v2 方向量化用）
         "_out_taken",         # 本 tick 每个体**实际咬到**的质量（本格+邻格；食腐不计）
         "_feed_fast",         # 快平均 EMA（τ=fast_tau）
         "_feed_slow",         # 慢平均 EMA（τ=slow_tau）＝"个体自己的期待"
         "_ars_extensive",     # True = 赶路模式（False = 驻留/原打分）
         "_heading",           # 上一步移动方向槽位（0–7；-1 = 未知/极区）
         "_giveup_ct",         # 赶路模式下已连走多少 tick 没咬到
         "_ars_dec_n",         # 求值个体数
         "_ars_inertia_sum",   # Σ|惯性项|（P6 量级诊断）
         "_ars_flat_n",        # 跨候选取同值的个体数（R148-1 反退化）
         "_ars_sw_ie_n",       # 驻留→赶路 切换次数
         "_ars_sw_ei_n",       # 赶路→驻留 切换次数
         "_ars_rerand_n",      # 失望重选方向次数
         # ---- P2② 移动决策整块向量化（本线 = [本地开发·性能线] 轻舟）------------
         "_batch_move_on",     # 快路径开关（默认 True；**对拍/调试可置 False** 走参考循环）
         # ---- R230 T-C：F-D2 修复（B′）--------------------------------------
         "_d2_rng",            # D2 感知噪声/Steels 配对的**每引擎独立** legacy RandomState
         # ---- R240 T8：气味场（本线 = [云端开发·云启]）----------------------------
         # 🔴 __slots__ 硬约束：新属性必须登记（加在块尾，减少同文件并发冲突面）。
         "smell",              # SmellField 实例；**全关 = None**（不进任何新代码路径 ⇒ 逐位不变）
         "_smell_on",          # 开关快照（构造期取一次；False ⇒ 整段不执行）
         "_smell_dummy",       # 消费端关档时传给 Rust 的占位（长度 1；标志 0 ⇒ Rust 不索引）
         "_smell_scratch",     # 消费端取数缓冲 = **按代缓存**：全场数组，值只对"本代已读格"有效
         "_smell_stamp",       # 每格的"代"标记（int32；与缓存同代才复用 ⇒ 场一更新就全体失效）
         "_smell_gen",         # 当前代（每次气味场 update() 后 +1）
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
        # F-D2 修复（R230 T-C，**B′ 方案**）：感知噪声/Steels 配对原读**进程级全局**
        # `np.random`（同进程多引擎共享一条流 ⇒ 互相污染，R226 以"假性分化"侧记）⇒
        # 改为**每引擎独立**的 legacy `RandomState`：种子与旧全局播种**完全相同**（同算法）
        # ⇒ 单引擎轨迹逐位不变（17 处钉死 digest 全绿），多引擎互不干扰。
        # 为什么不按字面"改绑 `self.rng`"（方案 A）：`self.rng` 是 default_rng(PCG64)
        # 的另一条流 ⇒ 已判 D2 基线（C7 的 (574887, 11266.746993) 一族）当场失效
        # （实测 3/3 抽样文件挂）⇒ 违反本任务自设判据①。A/B′ 实测见板上 T-C 报告。
        # 另：B′ 下 `rng_draws` 口径与修复前**完全一致**（噪声/洗牌本就不计入；
        # CountingRNG 的显式包装才计数）⇒ provenance 校验语义不变。
        self._d2_rng = np.random.RandomState(int(config.seed) & 0xFFFFFFFF)
        # 旧全局播种保留（写入点、非消费点）：runner 侧车/旧脚本假设"构造引擎后全局流被播种"；
        # 引擎自身已无任何全局 `np.random` 读取点。B′ 后侧车对引擎随机性变为 no-op（无害），
        # 续跑一致性改由快照内的 d2_rng_state 保证（见 save/load_snapshot）。
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
            # R249：仓库自带 `sim_core/`（Rust 源码目录，无 `__init__.py`）在**未编译
            # 扩展**的机器上会被 Python 3.3+ 当作"命名空间包"⇒ `import` 成功但模块
            # 为空 ⇒ 上面的 ImportError 防护被绕过，深处才 AttributeError
            # （2026-09-28 云端裸 Python 机冒烟实例：15 failed 全此因）。
            # 显式校验关键符号：缺失 ⇒ 与未安装同罪，fail-loud（禁静默）。
            if not hasattr(sim_core, "step_vectors_stage1"):
                raise RuntimeError(
                    "use_sim_core=True 但 import 到的 sim_core 不是编译扩展"
                    "（疑似把仓库 sim_core/ 源码目录当成了命名空间空包）："
                    "请在 venv 内执行 `python -m maturin develop --release`"
                    "（见 sim_core/BUILD.md），或改走 use_sim_core=False 的"
                    " Python 路径"
                )
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
            # 13.7 季节（默认 0/0 = 关 ⇒ 逐位等价；见 LightConfig 注释）
            tilt_rad=config.light.tilt_rad,
            season_period=config.light.season_period,
        )
        self.resources = ResourceField(
            self.world, self.light,
            capacity_per_area=config.resources.capacity_per_area,
            regrowth_rate=config.resources.regrowth_rate,
            temp_sensitivity=config.resources.temp_sensitivity,
            # 🔴 R196 光驱动再生（2026-09-24）：再生量乘光照因子 ⇒ 食物带随季节移动。
            #   ⚠️ 必须显式透传，否则 `ResourceConfig.light_sensitivity` 是**死字段**
            #   （同 13.5 ① `bg_production_zero` 家族：不传即"静默无变化"——C9）。
            light_sensitivity=float(
                getattr(config.resources, "light_sensitivity", 0.0)
            ),
            light_normalize=bool(
                getattr(config.resources, "light_normalize", False)
            ),
            distribution=config.resources.distribution,
            patch_count=config.resources.patch_count,
            patch_radius=config.resources.patch_radius,
            patch_capacity_mult=config.resources.patch_capacity_mult,
            patch_regrowth_mult=config.resources.patch_regrowth_mult,
            background_fill=config.resources.background_fill,
            initial_fill=config.resources.initial_fill,
            # 用 config.seed 派生 patch 中心：可复现，且独立 rng 不消费引擎 self.rng
            patch_seed=config.seed,
            # 🔴 13.5 ①（2026-09-24）：背景产能归零 —— 必须显式透传，
            #   否则 `ResourceConfig.bg_production_zero` 是**死字段**（老工 00:20 实测过：
            #   不传这一行 ⇒ 背景归零完全不生效，且现象是"静默无变化"——C9 家族）。
            bg_production_zero=bool(
                getattr(config.resources, "bg_production_zero", False)
            ),
            # 🔴 R242 背景低产能带：必须显式透传（同 13.5① 家族：不传 ⇒ 死字段 ⇒ 静默无变化）。
            #   默认 0 ⇒ 走原路径，逐位等价（C7 不动）。
            bg_low_prod_frac=float(
                getattr(config.resources, "bg_low_prod_frac", 0.0)
            ),
            bg_low_cap_mult=float(
                getattr(config.resources, "bg_cap_mult", 0.0)
            ),
        )
        # 田字格信号场（L2/L3）：生物可写入/读取 16 种标记模式
        # 🔴 P0.0 A1（2026-09-26）：寿命原为硬编码 `duration=50`，现读配置
        #    （`signals.duration_ticks`，默认 50 = 旧行为**逐位等价**）。
        #    它是**以 tick 计价**的时长 ⇒ 时间压缩时必须 ÷k；硬编码会让机械清点扫不到。
        self.signals = SignalField(
            self.world, duration=config.signals.duration_ticks
        )
        # ---- R240 T8：气味场（多通道标量场；设计稿《气味场与分功能感知》§二/§8.2）----
        #   🔴 **默认全关**（`channels == ()`）⇒ `smell is None` ⇒ tick 里只多一次 None 判断，
        #     不进任何新代码路径 ⇒ 旧行为**逐位不变**（C7 回滚点）。
        #   ⚠️ 模块自带稀疏/降采样路径（方案 (a)，设计稿 §8.4）⇒ 与 `sparse_fields` 正交。
        from world.smell_field import SmellField   # 局部导入：全关档零 import 开销（同 rd 先例）
        self._smell_on = bool(tuple(config.smell.channels or ()))
        self.smell = SmellField(self.world, config.smell) if self._smell_on else None
        self._smell_dummy = np.zeros(1, dtype=np.float64)   # R244 §二：Rust 传参占位（关档用）
        self._smell_scratch = None                          # R244 §二：消费端取数缓存（惰性分配）
        self._smell_stamp = None                            # R244 §二：每格"代"标记
        self._smell_gen = 1                                 # R244 §二：当前代（**从 1 起**：
        #   stamp 初始全 0 ⇒ 首 tick 全部视为"未算过"，不会误用零值缓存）

        # 预计算统一邻居表（L6 Rust 下沉用）：**P0.1（T1）紧凑化** —— 直接复用 world 的
        # 紧凑缓存（`(n_cells, 8)` int64 ≈ 29.5 MB @480×960），不再自建
        # `(n_cells, max(8, cols))` 的 fat 表（480×960 下 3 538.9 MB = 单 run 内存 94.7%）。
        # 极点格（邻居 = 相邻纬度带整行 cols 个）**不进主表**（主表极点行为 -1 占位），
        # 由 `_pole_nb` (2, cols) 承载；消费端（Rust 4 入口 + Python 文化学习）按
        # "行号 == 极行"分支取 ⇒ **邻居集合与顺序与旧 fat 表逐位相同** ⇒ 纯结构重构。
        self._nb_table = self.world._nb_table
        self._pole_nb = self.world._pole_nb
        self._pole_top = int(self.world._pole_top)
        self._pole_bottom = int(self.world._pole_bottom)
        n_cells = self.world.n_cells
        # 🔴 F1 修复（任务书 T3，C9 型缺陷）：**每格实际邻居数**（span=1 语义）。
        #   此前 `densities` 用 `_nb_table.shape[1]`（= stride 120）当分母 ⇒ 社交项被
        #   静默弱化 15×（注释写"0~8"但代码除 120）。改为逐格实际邻居数（众数 8、
        #   极区 120）⇒ 社交项恢复应有量级 = **构造级变更**（digest 变，须同步测试）。
        #   ⚠️ 只在 `social_norm="auto"` 时生效；数字串 ⇒ 冻结常量（兼容设计稿提议）。
        #   P0.1：旧值 = 旧表每行"非负项计数"= 非极 8 / 极 cols；新式按同一语义直接
        #   生成（极行整带在 `_pole_nb`，主表极行是 -1 占位 ⇒ 不能再数主表）⇒ 逐位相同。
        self._nb_len = np.where(
            self.world.is_pole(np.arange(n_cells)),
            float(self.world.cols), 8.0,
        )
        _snorm0 = str(getattr(config.simulation, "social_norm", "auto"))
        if _snorm0 != "auto":
            try:
                self._nb_norm = np.full(
                    n_cells, float(_snorm0), dtype=np.float64)
            except ValueError:
                self._nb_norm = self._nb_len.copy()
        else:
            self._nb_norm = self._nb_len.copy()
        # ── 13.4 波 2B（T3）：span 感知邻居表（perception_span=2 ⇒ ring1+ring2 合并）──
        # span=1（默认）⇒ **不建表、走原路径**（逐位等价）；span=2 ⇒ 两圈候选（球面跳数，
        # 普通格 24、极区 240 —— 后者由 `_perception_cap` 降级为 1 圈，见移动段）。
        # 🔴 只被 Python 移动分支消费；use_sim_core + span=2 ⇒ 构造期 H3 硬报错。
        _span0 = int(getattr(config.simulation, "perception_span", 1))
        self._span_table = None
        if _span0 == 2:
            _span_rows: list[np.ndarray] = []
            for c in range(n_cells):
                one = [int(x) for x in self.world.neighbors(c)]
                two: set[int] = set(one)
                for d in one:
                    two.update(int(x) for x in self.world.neighbors(d))
                two.discard(c)
                _span_rows.append(np.fromiter(sorted(two), dtype=np.int64))
            _span_stride = max(int(max(len(r) for r in _span_rows)), 8)
            self._span_table = np.full((n_cells, _span_stride), -1, dtype=np.int64)
            for c, r in enumerate(_span_rows):
                self._span_table[c, :len(r)] = r
            self._span_len = np.array([len(r) for r in _span_rows], dtype=np.int64)
            # F3 闸（设计稿 §一）：ring1+2 规模 > cap ⇒ 该格 `span_eff=1`（降级不是报错）
            self._span_cap = int(getattr(config.simulation, "perception_cap", 32))
            self._span_eff2 = self._span_len <= self._span_cap
        else:
            self._span_len = self._nb_len.astype(np.int64)
            self._span_eff2 = np.ones(n_cells, dtype=bool)

        # ── R146/R149 L2：**strict 2 圈**候选表（CSR；构造期一次）──────────────────
        # 语义：2 跳可达、去掉自身与 1 圈（球面拓扑用**跳数**，**不套 5×5 方窗**；
        #       极点格走 `neighbors()` 的整带语义）。只在 `l2_dash=True` 时被读。
        # `FAR_CAP` 语义 = 「strict 2 圈规模 > FAR_CAP 的格**不可冲刺**」（拓扑退化），
        #   🔴 措辞：排除的是**格**，不是"极区个体"；`[实测]` 不变区间 [16,119] ⇒ 非旋钮。
        # 🔴 S0（2026-09-25 性能；世界「画布化」讨论）：2 跳环表**改为惰性构造**。
        #   `[实测]` 480×960（46 万格）下原无条件构造耗时 **29.0 s**，而这张表
        #   **只被 `l2_dash=True` 的路径读取**（`l2_probe` 的 dash_* 读数 + 冲刺盲选目标）。
        #   构造是 O(格数 × 邻居²) ⇒ 世界放大后必然成为构造期不可逾越的墙。
        #   ⇒ 默认置空，首次真正读取时由 `_ensure_far_tables()` 构造
        #     ⇒ **逐位等价**（构造结果只依赖 `world` 与 `far_cap`，与 tick 状态无关）。
        #   ⚠️ 新增个体状态数组的四处登记纪律：本数组是**引擎级派生表**（非逐个体），
        #     故只需 `__slots__` + 本处初始化，不进出生扩容/死亡压缩清单。
        self._far_ready = False
        self._far_off_c = np.zeros(0, dtype=np.int64)
        self._far_len_c = np.zeros(0, dtype=np.int64)
        self._far_cells_c = np.zeros(0, dtype=np.int32)
        self._far_excluded_c = 0

        # 🔴 S1（2026-09-25 性能）：`food_ratio = _grid / max(_capacity, 1e-9)` 的**分母
        #   在构造期算一次**。`_capacity` 只在 `ResourceField.__init__` 里被赋值、运行期不变
        #   ⇒ 预计算与"每 tick 重算"**逐位相同**（同一次 `np.maximum` 的同一输入）。
        #   省下每 tick 一个全场 max + 一个全场临时数组。
        #   ⚠️ 若将来出现"运行期改容量"的机制（如资源动态改写 `_capacity`），
        #     必须同步失效本缓存（否则是静默 no-op 型的 C9 缺陷）。
        self._cap_floor = np.maximum(self.resources._capacity, 1e-9)

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
        # ── H3（13.4 波 1）：亚格坐标的两条 fail-loud（本线 = [所有者·天平]）─────────
        # (a) subpos ∧ use_sim_core：坐标改造**只在 Python 路径**实现（§14.7：新机制强制 Python）
        #     ⇒ Rust 路径仍用**旧整数格移动** ⇒ 开关开了行为却不变（静默 no-op 同族）⇒ 硬报错。
        # (b) subpos ∧ l2_dash：**语义互斥** —— 两者都是「走多远」的乘子：
        #       L2 = 1/2 两档、概率闸 + **盲选** far 环目标（`rand_choice % 100` / `// 100`）；
        #       subpos = 0/0.25/…/speed_max 多档、**沿用 score 选出的方向**、由 mob_eff 确定性给出。
        #     同开 ⇒ 同一个体可能"被 L2 判冲刺换了目标"且"被 subpos 判走 0.5 格"，
        #     语义冲突；且 `dash_frac` 与 `steps` 两个读数互相污染 ⇒ 归因不干净。
        _subcfg0 = config.subpos
        if bool(getattr(_subcfg0, "enabled", False)):
            if _scfg0.use_sim_core:
                raise NotImplementedError(
                    "subpos（亚格坐标）尚未下沉 Rust：use_sim_core=True 时开启会"
                    "**静默走旧整数格移动路径**（开关开了行为却不变 = 静默 no-op 的同族形态）"
                    "⇒ 硬报错。请设 use_sim_core=False（§14.7：新机制强制 Python 路径）。"
                )
            if bool(getattr(_scfg0, "l2_dash", False)):
                raise NotImplementedError(
                    "subpos 与 l2_dash **语义互斥**：两者都是「走多远」的乘子 —— "
                    "L2 是 1/2 两档且盲选目标，subpos 是它的一般化（0/0.25/…/speed_max，"
                    "且沿用 score 选出的方向）。同开会语义冲突，且 dash_frac 与 steps "
                    "两个读数互相污染 ⇒ 归因不干净。请二选一"
                    "（建议 subpos：speed_max=2.0 已包含 L2 的冲刺语义）。"
                )
        # ── H3（13.4 波 2A）：资源动态（休耕—死亡—轮作）开启时必须 **fail-loud** ────
        # 本机制只在 Python 路径实现（§14.7：新机制/非确定性强制 Python，不动 Rust）⇒
        # use_sim_core=True 时开启会**静默走旧再生路径**（开关开了行为却不变 = 静默
        # no-op 的同族形态）⇒ 构造期硬报错。
        _rdcfg0 = getattr(config, "resource_dynamics", None)
        if _scfg0.use_sim_core and bool(getattr(_rdcfg0, "enabled", False)):
            raise NotImplementedError(
                "resource_dynamics（斑块休耕—死亡—轮作）尚未下沉 Rust："
                "use_sim_core=True 时开启会**静默走旧再生路径**（开关开了行为却不变"
                " = 静默 no-op 的同族形态）⇒ 硬报错。请设 use_sim_core=False"
                "（§14.7：新机制强制 Python 路径）。"
            )
        # ── H3（13.8）：migration 开启时必须 **fail-loud**（两条守卫，都需）──────────
        # 🔴 顺序要求（R198 实测修正）：**M1 必须排在 season 守卫之前**。
        #    原因：M1（enabled ∧ use_sim_core）与 season 守卫的触发条件在"合法配置"
        #    下**必然同时成立**（M2 要求 migration 必须配 season ⇒ 想测 M1 就一定开着
        #    season）⇒ 若 season 守卫在前，它会**先**报 ⇒ M1 变成**永不可达的死代码**
        #    （曾如此，被 `test_s5` 抓出）。把 M1 前置后，迁徙用户拿到的是**针对迁徙的
        #    可操作报错**，而不是被误导向"季节没下沉 Rust"。
        # 设计稿 §3.5：日历—罗盘式定向迁徙（g23）只在 **Python 移动路径**实现 ——
        # Rust 的移动段**不消费** g23（sim_core/src/genes.rs 里 G_MIGRATE_BIAS 仅作
        # 「索引↔语义」对齐常量）⇒ use_sim_core=True 时开启会**静默走旧移动路径**
        # （开关开了行为却不变 = 静默 no-op 的同族形态）⇒ 构造期硬报错。
        # 【M1】enabled ∧ use_sim_core ⇒ 静默 no-op（同族形态）。
        _migcfg0 = getattr(config, "migration", None)
        _mig_on0 = bool(getattr(_migcfg0, "enabled", False))
        if _scfg0.use_sim_core and _mig_on0:
            raise NotImplementedError(
                "migration（日历—罗盘式定向迁徙，13.8）尚未下沉 Rust："
                "use_sim_core=True 时开启会**静默走旧移动路径**（开关开了行为却不变"
                " = 静默 no-op 的同族形态）⇒ 硬报错。请设 use_sim_core=False"
                "（§14.7：新机制强制 Python 路径）。"
            )
        # ── H3（13.7）：season 开启时必须 **fail-loud** ──────────────────────
        # 季节光照公式（太阳赤纬 δ(t)）只在 Python 路径实现
        # （`LightAndTemperature._ensure_cache` 的季节分支）——Rust 的
        # `light_temp.rs` 未实现赤纬 ⇒ use_sim_core=True 时会**静默走无季节光照**
        # （开关开了行为却不变 = 静默 no-op 的同族形态）⇒ 构造期硬报错。
        _lcfg0 = getattr(config, "light", None)
        _season_on0 = (abs(float(getattr(_lcfg0, "tilt_rad", 0.0))) > 1e-12
                       and int(getattr(_lcfg0, "season_period", 0)) > 1)
        if _scfg0.use_sim_core and _season_on0:
            raise NotImplementedError(
                "season（季节：tilt_rad≠0 且 season_period>1）尚未下沉 Rust："
                "use_sim_core=True 时开启会**静默走无季节光照**（开关开了行为却不变"
                " = 静默 no-op 的同族形态）⇒ 硬报错。请设 use_sim_core=False"
                "（§14.7：新机制强制 Python 路径）。"
            )
        # ── H3（R196）：light_sensitivity>0 时必须 **fail-loud** ───────────────
        # 光驱动再生（`ResourceField._regrowth_amount` 的 illumination 因子）只在
        # Python 路径实现 —— Rust 的 `resource.rs` 再生式**只耦合温度** ⇒
        # use_sim_core=True 时开启会**静默走无光照再生**（开关开了行为却不变
        # = 静默 no-op 的同族形态）⇒ 构造期硬报错。
        if _scfg0.use_sim_core and float(
            getattr(getattr(config, "resources", None), "light_sensitivity", 0.0)
        ) > 0.0:
            raise NotImplementedError(
                "light_sensitivity>0（光驱动再生：再生量乘光照因子）尚未下沉 Rust："
                "use_sim_core=True 时开启会**静默走只看温度的旧再生路径**"
                "（开关开了行为却不变 = 静默 no-op 的同族形态）⇒ 硬报错。"
                "请设 use_sim_core=False（§14.7：新机制强制 Python 路径）。"
            )
        # ── H3（13.4 波 2B，T3）：perception_span=2 开启时必须 **fail-loud** ──────
        # span=2 需要两圈邻居表（`_span_table`），只在 Python 移动路径实现 ⇒
        # use_sim_core=True 时开启会**静默走 1 圈旧路径**（开关开了视野却没扩 = 静默
        # no-op 的同族形态）⇒ 构造期硬报错。
        if _scfg0.use_sim_core and int(getattr(_scfg0, "perception_span", 1)) == 2:
            raise NotImplementedError(
                "perception_span=2（两圈感知）尚未下沉 Rust：use_sim_core=True 时开启"
                "会**静默走 1 圈旧路径**（开关开了视野却没扩 = 静默 no-op 的同族形态）"
                "⇒ 硬报错。请设 use_sim_core=False（§14.7：新机制强制 Python 路径）。"
            )
        # ── H3（设计稿 §5.1 / §2.5）：尸体—食腐 + 血条开关开启时必须 **fail-loud** ────
        # 本任务全程 Python 路径（§14.7：新机制/非确定性 ⇒ 强制 Python，不动 Rust）⇒
        # use_sim_core=True 时开启 corpse/wound/contest 会**静默走旧路径**（开关开了
        # 行为却不变 = 静默 no-op 的同族形态）⇒ 构造期硬报错，禁止静默忽略。
        _cwc = getattr(config, "corpse_wound", None)
        _cw_any = bool(getattr(_cwc, "corpse_enabled", False)) or bool(
            getattr(_cwc, "wound_enabled", False)) or bool(getattr(_cwc, "contest_enabled", False))
        if _scfg0.use_sim_core and _cw_any:
            raise NotImplementedError(
                "尸体—食腐/血条尚未下沉 Rust（本任务全程 Python 路径，设计稿 §5.1）⇒ "
                "use_sim_core=True 时开启 corpse_enabled/wound_enabled/contest_enabled 会"
                "**静默走旧路径** ⇒ 硬报错。请设 use_sim_core=False。"
            )
        # ── H3（13.8）：migration 的第二条守卫（M1 已前置在 season 守卫之前）──────
        # 【M2】enabled ∧ 无季节 ⇒ **数学上恒等于 no-op**（比 M1 更隐蔽，必须另拦）：
        #   季节关 ⇒ 赤纬 δ(t) ≡ 0 ⇒ 逐格日长 P(φ) ≡ 0.5（解析式
        #   cos H₀ = −tan φ·tan δ = 0 ⇒ H₀ = π/2 ⇒ P = 0.5，**与纬度无关**）⇒
        #   日历轴异常 A_i = P − 0.5 ≡ 0 ⇒ 迁移项逐候选 **恒 0** ⇒ argmax 逐位不变。
        #   ⚠️ 此时若放行，实验会读出"迁徙无效"的**假阴性**（错在实验设计而非机制）
        #   ⇒ 正是 B2「不许 no-op」要防的形态 ⇒ 构造期硬报错（fail-loud，不发警告）。
        _lcfg1 = getattr(config, "light", None)
        _season_on1 = (abs(float(getattr(_lcfg1, "tilt_rad", 0.0))) > 1e-12
                       and int(getattr(_lcfg1, "season_period", 0)) > 1)
        if _mig_on0 and not _season_on1:
            raise ValueError(
                "migration.enabled=True 但季节未开（light.tilt_rad=0 或 "
                "light.season_period≤1）：无季节 ⇒ 赤纬 δ(t)≡0 ⇒ 逐格日长 P≡0.5"
                "（与纬度无关）⇒ 日历轴异常 A≡0 ⇒ 迁移项**逐候选恒 0**"
                "（argmax 逐位不变）⇒ 实验只会读出「迁徙无效」的**假阴性**"
                "（错在实验设计，不在机制）。⇒ 硬报错。请同时开 season"
                "（tilt_rad≠0 且 season_period>1），或令 migration.enabled=False。"
            )

        n = config.population.initial_count
        self._flat = np.zeros(n, dtype=np.int64)
        # ---- 13.4 波 1：亚格坐标初始化（本线）----
        # 关档时这两个数组**仍然存在**且恒等于由 `_flat` 推出的**格中心**
        # ⇒ I2（`_flat` 与亚格恒一致）从第一步就成立，不需要"关档特判"（少一个特例 = 少一个坑）。
        _subdiv0 = int(getattr(config.subpos, "subdiv", 4))
        self._sub_r, self._sub_c = _sub_init(self._flat, _subdiv0, self.world)
        # 读数计数器（关档不累加 ⇒ `subpos_probe()` 返回 None = "未适用"，不是 0）
        self._run_flat_move_n = 0     # Σ 真正换格的个体数（主判据分子）
        self._run_slow_n = 0          # Σ 移动者中 steps==0（"白移动"）的个体数
        self._run_mover_sub_n = 0     # Σ 移动者数（主判据分母）
        self._steps_hist = np.zeros(_subdiv0 + 1, dtype=np.int64)
        # ---- 13.4 波 2A：资源动态（T2，本线 = [云端·开发]）----
        # 构造期从 ResourceField 读斑块掩码/容量/倍率（**只读**，不碰引擎状态）。
        # 关档时 `_rd.enabled=False` ⇒ 全部方法 no-op ⇒ 引擎无条件调用也不进新路径（C7）。
        _rdcfg0 = getattr(config, "resource_dynamics", None)
        self._rd = _ResourceDynamics.from_field(
            _rdcfg0, self.resources, self.world)
        # ---- S3 记忆 v2（egocentric）：fail-loud ×2 + 构造期质心表（实施规格 §一/§七-2）----
        # 🔴 两条 fail-loud（R236 §四-D-2 同族）：
        #   ① 运行时：v2 ∧ use_sim_core=True ⇒ 硬报错（Rust 未下沉 v2；静默走旧路径 = dash PR 翻车形态）
        #   ② 构造期：v2 ∧ rd.enabled ∧ not bg_production_zero ⇒ 硬报错（R236 §四-B-2：
        #      rd 搬移会改变斑块掩码 ⇒ 静态质心/方位静默失效）
        _mcfg2 = getattr(config, "info_structure", None)
        _mv2 = bool(getattr(_mcfg2, "memory_v2", False))
        if _mv2 and self._use_sim_core:
            raise NotImplementedError(
                "memory_v2 尚未下沉 Rust：use_sim_core=True 时开启 memory_v2 会静默走旧路径"
                " ⇒ 硬报错。请设 use_sim_core=False。"
            )
        if _mv2 and bool(getattr(self._rd, "enabled", False)) \
                and not bool(getattr(self._rd, "bg_production_zero", False)):
            raise NotImplementedError(
                "memory_v2 ∧ rd.enabled ∧ 背景格有产能 ⇒ 硬报错（R236 §四-B-2：rd 轮作搬移"
                "会改变斑块掩码 ⇒ 静态质心/方位静默失效）。bg_production_zero 世界（S 线）允许。"
            )
        # 构造期质心表（仅 v2 需要；一次性 BFS 标注，不在 tick 内；零 RNG ⇒ C7 无关）
        #   `_patch_centroid[i]` = 格 i 所属斑块的质心格 id（非斑块格 = -1；无斑块 = 全 -1）。
        #   运行时个体**只查表**得到一个"质心格 id"再转方位 ⇒ 不暴露任何世界坐标本体（§三-3）。
        self._patch_centroid = np.full(self.world.n_cells, -1, dtype=np.int64)
        if _mv2:
            _pm = getattr(self.resources, "_patch_mask", None)
            if _pm is not None and bool(_pm.any()):
                self._patch_centroid = _build_patch_centroid(self.world, _pm)
        # ---- R217 §五 #1 稀疏化 B：惰性再生 / 稀疏信号（默认关 ⇒ 全场路径逐位不变）----
        # 🔴 范围锁（引擎侧）：共用 ① 配置开关；② Python 路径（Rust `consume_many` /
        #    `regrow_patchy` / `signal_emit` **直写** `_grid`/`_marks`，打脏点覆盖不到）。
        #    资源侧锁③（R244 v1 改写）：原"rd 关" → **放行 `rd.enabled ∧ bgzero`**（S2/S3 主线）；
        #    一般档（会搬斑块的世界 = rd ∧ ¬bgzero）保持全场 —— 搬移 ⇒ `_capacity`/掩码逐 tick
        #    变 ⇒ "净格恒净"前提破裂（R226 修复①：bgzero 档禁搬移 ⇒ 掩码/容量恒定，实测恒真）。
        # 资源侧另有前提校验（`enable_lazy`：temp_sensitivity/light_sensitivity/倍率符号）
        # ⇒ 不满足时它返回 False 并**保持全场路径**（安全回退；`resources._lazy` 即真值）。
        # R231 T-E：**信号侧与资源侧解耦** —— 信号活跃集（`age>0`）自包含、与 rd 正交
        #    （rd 不写 `_marks`）⇒ 信号侧范围锁只剩 ①②；rd 开档（含 bgzero S 线）照常稀疏。
        #    两侧真值以 `resources._lazy` / `signals._sparse` 为准（`_sparse_fields` = 资源侧闸）。
        _sparse_cfg = bool(
            getattr(config.simulation, "sparse_fields", False)
            and not self._use_sim_core
        )
        _rd_on0 = bool(getattr(self._rd, "enabled", False))
        _bgzero0 = bool(getattr(self._rd, "bg_production_zero", False))
        if _sparse_cfg and _rd_on0 and not _bgzero0:
            # 🔴 R244 验收③：`rd ∧ sparse ⇒ bgzero` **构造期 fail-loud**（防未来开搬移时前提
            #    静默破裂）。rd 子集化的前提 = "掩码/容量逐 tick 恒定"；bgzero 档由 R226 修复①
            #    （禁搬移）保证；一般档（搬斑块）该前提不成立 ⇒ 宁炸不静默。
            raise ValueError(
                "sparse_fields=True ∧ resource_dynamics.enabled=True ∧ "
                "bg_production_zero=False：该组合未经验证 —— 轮作会搬斑块 ⇒ 掩码/容量逐 tick "
                "变化 ⇒ 脏格集（派生量）会漏标记新出现的欠容格 ⇒ 破逐位等价。"
                "请改用 bg_production_zero=True 档（S2/S3 主线），或令 sparse_fields=False。"
                "（R244 §二 验收③：rd ∧ sparse ⇒ bgzero，构造期 fail-loud）"
            )
        _rd_mults_ok = (
            float(getattr(self._rd, "dead_regen_mult", 0.0)) >= 0.0
            and float(getattr(self._rd, "rest_regen_mult", 0.0)) >= 0.0
        )
        # R244 v1 范围锁：rd 关 ⇒ 原行为；rd 开 ⇒ 仅 bgzero 档放行（负闸 ⇒ 静默退回全场）。
        self._sparse_fields = _sparse_cfg and (not _rd_on0 or (_bgzero0 and _rd_mults_ok))
        if self._sparse_fields:
            self.resources.enable_lazy()
        if _sparse_cfg:
            self.signals.enable_sparse()
        self._rd_intake_sum = np.zeros(self.world.n_cells, dtype=np.float64)
        self._rd_growth_sum = np.zeros(self.world.n_cells, dtype=np.float64)
        # 13.4 波 2B（T3）：cap 读数计数器（关档不累加 ⇒ probe None/0 口径）
        self._cap_blocked_n = 0
        self._cap_stay_n = 0
        # ---- 13.8 日历—罗盘式定向迁徙（本线 = [云端·开发] 云归）----
        # 🔴 关档（enabled=False）时：`_mig_on=False` ⇒ 移动段整段不执行（无 RNG、无状态
        #    改变）⇒ 与旧行为**逐位等价**（C7 digest 钉死）。数组**仍然**建好（代价 = 一次
        #    `|lat|` 的 O(格数) 计算，与"关档特判"相比少一个特例 = 少一个坑）。
        self._mig_on = bool(getattr(_migcfg0, "enabled", False))
        self._mig_gain = float(getattr(_migcfg0, "gain", 0.0))
        self._mig_min_abs = float(getattr(_migcfg0, "min_abs_anomaly", 0.0))
        # 罗盘轴"米尺"：逐格 |φ|（弧度）。**一次性**预计算 ⇒ 热路径零三角函数。
        # ⚠️ `latitude_of` 的入参是**行号**（不是扁平格号）⇒ 逐行算再按列重复
        #    （与 `LightAndTemperature.photoperiod` 同一写法，保证两轴口径一致）。
        # 🔴 用 |φ| 而非带符号 φ ⇒ **南北半球自动都正确**，不需要半球分支
        #    （南半球个体向北 = |φ| 增大 = 朝赤道，与"回北方繁殖"的语义一致）。
        _lat_rows = np.abs(
            self.world.latitude_of(np.arange(self.world.rows, dtype=np.int64))
        ).astype(np.float64)
        self._lat_abs = np.ascontiguousarray(
            np.repeat(_lat_rows, self.world.cols))
        # 日历轴：逐格 A = P − 0.5（初值 0 ⇒ 未刷新时中性，不会造出假信号）
        self._pp_anom = np.zeros(self.world.n_cells, dtype=np.float64)
        self._pp_anom_tick = -1
        # 读数（关档全部不累加 ⇒ `migration_probe()` 返回 None = "未适用"，不是 0）
        self._mig_dec_n = 0
        self._mig_term_sum = 0.0
        self._mig_term_n = 0
        self._mig_zero_n = 0
        self._mig_flat_n = 0
        self._mig_skip_n = 0
        # 🔴 A 的**运行期极值**（累积，不是末 tick 快照）：season_period=6000 时
        #    tick=3000 恰好 δ=0 ⇒ 瞬时 A≡0 ⇒ 只看快照会误判"日历轴没结构"。
        #    累积极值才是 P2（纬度结构）与 P3（无 NaN/越界）的正确诊断量。
        self._mig_pp_lo = 0.0
        self._mig_pp_hi = 0.0

        # ── 14.9 ARS 双模式觅食（本线 = [所有者·天平]）────────────────────────
        # H3-A1（fail-loud）：ARS 只在 Python 移动路径实现（§14.7）⇒
        # use_sim_core=True 时开启会静默走旧移动路径（= 静默 no-op 同族）⇒ 硬报错。
        _arscfg0 = getattr(config, "ars", None)
        self._ars_on = bool(getattr(_arscfg0, "enabled", False))
        if self._ars_on and self._use_sim_core:
            raise NotImplementedError(
                "ars（ARS 双模式觅食，14.9）尚未下沉 Rust：use_sim_core=True 时开启会"
                "**静默走旧移动路径**（开关开了行为却不变 = 静默 no-op 的同族形态）"
                "⇒ 硬报错。请设 use_sim_core=False（§14.7：新机制强制 Python 路径）。"
            )
        self._ars_gain = float(getattr(_arscfg0, "gain", 0.0))
        self._ars_theta = float(getattr(_arscfg0, "theta", 0.5))
        self._ars_kappa = float(getattr(_arscfg0, "kappa", 0.0))
        self._ars_giveup = int(getattr(_arscfg0, "giveup", 20))
        self._ars_fast_tau = float(getattr(_arscfg0, "fast_tau", 50.0))
        self._ars_slow_tau = float(getattr(_arscfg0, "slow_tau", 500.0))
        self._ars_lat_exempt = float(getattr(_arscfg0, "lat_exempt_deg", 85.0))
        # 槽位方向余弦矩阵：由**参考格**（中纬度）的 8 邻表推导槽位 → (dr,dc)，
        # 归一化后内积 ⇒ (8,8)。🔴 不硬编码列序（A4 教训：列序曾导致半平面 bug）。
        _ref = (self.world.rows // 2) * self.world.cols + self.world.cols // 2
        _nb_ref = np.asarray(self.world.neighbors(_ref))
        _rc = np.asarray([divmod(int(x), self.world.cols) for x in _nb_ref])
        _r0, _c0 = divmod(_ref, self.world.cols)
        _dr = _rc[:, 0].astype(np.float64) - _r0
        _dc = _rc[:, 1].astype(np.float64) - _c0
        _dc = np.where(_dc > self.world.cols / 2, _dc - self.world.cols,
                       np.where(_dc < -self.world.cols / 2, _dc + self.world.cols, _dc))
        _ln = np.sqrt(_dr * _dr + _dc * _dc)
        _ln[_ln < 1e-12] = 1.0
        _unit = np.stack([_dr / _ln, _dc / _ln], axis=1)     # (8,2)
        self._ars_cos = (_unit @ _unit.T).astype(np.float64)  # (8,8)
        # S3 记忆 v2：保留槽位单位向量（任意方向 → 最接近 Moore 槽位，内积 argmax）
        self._ars_unit = np.asarray(_unit, dtype=np.float64)   # (8,2)
        # 个体状态数组（关档仍建 ⇒ 少一个特例 = 少一个坑；关档不消费 ⇒ 零轨迹影响）
        self._out_taken = np.zeros(n, dtype=np.float64)
        self._feed_fast = np.zeros(n, dtype=np.float64)
        self._feed_slow = np.zeros(n, dtype=np.float64)
        self._ars_extensive = np.ones(n, dtype=bool)          # 出生即赶路（还没咬到过）
        self._heading = np.full(n, -1, dtype=np.int64)        # -1 = 未知/极区
        self._giveup_ct = np.zeros(n, dtype=np.int64)
        # 读数（关档全 0/None）
        self._ars_dec_n = 0
        self._ars_inertia_sum = 0.0
        self._ars_flat_n = 0
        self._ars_sw_ie_n = 0
        self._ars_sw_ei_n = 0
        self._ars_rerand_n = 0

        # ── P2② 移动决策整块向量化（本线 = [本地开发·性能线] 轻舟）──────────
        # 快路径开关：默认开（vanilla 配置走 `_move_decide_batch`）。
        # 🔴 **对拍/调试用**：置 False ⇒ 强制走原逐个体参考循环（逐位等价的可比基线）。
        # 非状态量 ⇒ 不进快照（load 后恒为 True）。
        self._batch_move_on = True

        self._energy = np.full(
            n, config.organisms.initial_energy, dtype=np.float64
        )
        self._stomach = np.zeros(n, dtype=np.float64)
        # R165 0-2：胃内"尸源"质量（归因用；关档恒 0 ⇒ 零轨迹影响）
        self._stomach_scav = np.zeros(n, dtype=np.float64)
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
        # ---- S3 前置：记忆 v2（egocentric）新增 5 数组（实施规格 §二；默认关 ⇒ 不读写）----
        # 🔴 全部只被 `memory_v2=True` 路径消费；关档不进任何新代码路径（C7 回滚点）。
        # `_mem_az`   = 记忆方位 q（自我参照系 Moore 槽位 0–7；-1 = 空槽）——**不存世界坐标**
        # `_mem_dist` = 走过的距离 d（写入时 0，每 tick + 本步格数；**只增不减** ⇒ 不可逆降级自然成立）
        # `_mem_tick` = 记录 tick（时效判定 + 槽替换依据）
        # `_mem_degraded` = 精→粗**永久降级标记**（置 True 后不清除，除非重写该槽）
        # `_mem_food` = 记录时刻食物比（留档，首期不参与打分）
        self._mem_az = np.full((n, 4), -1, dtype=np.int8)
        self._mem_dist = np.zeros((n, 4), dtype=np.float32)
        self._mem_tick = np.full((n, 4), -1, dtype=np.int64)
        self._mem_degraded = np.zeros((n, 4), dtype=bool)
        self._mem_food = np.zeros((n, 4), dtype=np.float32)
        # v2 运行期配置快照（关档不消费 ⇒ 零轨迹影响）+ 独立噪声 RNG
        # 🔴 噪声档用**独立 rng 流**（不同 `self.rng`）：无噪声档 v2 完全不消费主 rng
        #    ⇒ v2 开/关档主随机流形状不变 ⇒ 轨迹差异纯粹来自记忆机制（归因干净；§五 RNG 契约）。
        self._mem_gain = float(getattr(_mcfg2, "memory_gradient_gain", 0.3))
        self._mem_coarse_gain = float(getattr(_mcfg2, "memory_coarse_gain", 0.15))
        self._mem_dist_scale = float(getattr(_mcfg2, "memory_dist_scale", 15.0))
        self._mem_degrade_thr = float(getattr(_mcfg2, "memory_degrade_thr", 20.0))
        self._mem_ttl = int(getattr(_mcfg2, "memory_ttl", 1000))
        self._mem_noise = bool(getattr(_mcfg2, "memory_noise", False))
        self._mem_noise_p = float(getattr(_mcfg2, "memory_noise_p", 0.1))
        # 13.11 记忆权重基因位（g22）：关（默认）⇒ 固定权重（0.3/0.15，逐位不变 = C7）；
        # 开 ⇒ 精/粗 gain 同乘 `2×g22`（g22∈[0,1] 均匀初始化 ⇒ 乘子均值 1.0，
        # ⇒ 种群总体权重与 S3 固定权重批可比，个体差异才是唯一变量）。
        self._mem_weight_gene = bool(getattr(_mcfg2, "memory_weight_gene", False))
        self._mem_v2_rng = np.random.default_rng(20260928)
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
        # 🔴 14.9 T1b（青梧找回）：`init_patch_frac>0` ⇒ 混合投放——按比例把创始代撒进
        #   食物格（斑块内），其余仍均匀随机。默认 0.0 = 走原路径（逐位等价；RNG 序列不变）。
        #   用途：斑块出生子集测"富斑耗尽→失望离开"（卡点1）、荒漠出生子集测"找路"（卡点2），
        #   两子集读数分列。**实验投放参数，不改世界规则**（斑块布局/再生全不动）。
        _init_pf = float(getattr(config.population, "init_patch_frac", 0.0))
        if _init_pf > 0.0:
            _patch_cells = np.flatnonzero(self.resources._grid > 0.0)
            if len(_patch_cells) == 0:
                raise RuntimeError(
                    "init_patch_frac>0 但当前世界无任何食物格（patchy 布局未生效？）"
                )
            _n_patch = int(round(n * min(_init_pf, 1.0)))
            if _n_patch > n:
                _n_patch = n
            _fp = self.rng.choice(_patch_cells, size=_n_patch, replace=True).astype(
                np.int64
            )
            _fr = self.rng.integers(
                0, self.world.n_cells, size=n - _n_patch
            ).astype(np.int64)
            self._flat = np.concatenate([_fp, _fr])
        else:
            self._flat = self.rng.integers(0, self.world.n_cells, size=n).astype(
                np.int64
            )
        # 🔴 13.4 波 1：**撒点之后必须重新同步亚格坐标** ——
        # 上面 `_flat = np.zeros(n)` 时做的 `_sub_init` 是基于"全 0"的，而撒点把 `_flat`
        # 整体替换成随机格 ⇒ 不同步就会**脱钩**（I2 被破坏：亚格指向格 0、`_flat` 指向随机格）。
        # 实测教训：这正是 E4/E6 两条引擎级单测第一次跑时抓到的形态。
        self._sub_r, self._sub_c = _sub_init(self._flat, _subdiv0, self.world)
        if bool(getattr(config.subpos, "enabled", False)):
            # ⚠️ **极点行**的撒点可能落在"冗余槽位"：`flat_to_rc` 与 `rc_to_flat` 在极点行
            #    **不互逆**（该行所有 col 物理上坍缩为一格，`neighbour` 返回的是整行 col）。
            #    ⇒ 开档时把 `_flat` 也**规范化**，让 I2（`_flat` ≡ 亚格派生）**逐位**成立。
            #    只在开档做 ⇒ 关档路径一个字节都不动（I1 逐位等价不受影响）。
            self._flat = _sub_flat(self._sub_r, self._sub_c, _subdiv0, self.world)

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
        # R146/R149 L1 读数（B4；本段 = [所有者·天平] 线 = B1）。**l1_seek/l1_fear 全关时
        #   全部不累加** ⇒ `l1_probe()` 返回 None（"未适用"），不是 0（R120/§五.12 口径铁律）。
        #   ⚠️ `*_flat_n` 是**反退化计数**：某项对某个体**所有候选取同值** ⇒ 该项对该个体是
        #     逐位 no-op（R148-1 的形态）⇒ 段一必报"跨候选反退化占比"。
        self._seek_term_sum = 0.0             # Σ L1a 项（按候选）
        self._seek_term_n = 0                 # L1a 候选数（分母）
        self._seek_flat_n = 0                 # 其中"跨候选取同值"的个体数
        self._seek_zero_n = 0                 # 其中"项恒为 0"的个体数（= **自熄**：邻域无猎物代理）
        self._fear_term_sum = 0.0             # Σ L1b 项（按候选；**符号为负** = 扣分）
        self._fear_term_n = 0                 # L1b 候选数（分母）
        self._fear_flat_n = 0                 # 其中"跨候选取同值"的个体数（含 danger=nb 全占）
        self._fear_ind_n = 0                  # L1b 真正施加的个体数（danger 非空）
        self._l1_dec_n = 0                    # L1 求值的个体数（分母；len(nb)==1 已提前 continue）
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

        # ---- S1 骨架（设计稿 §5.2 项 1）：尸体格数组 + 个体血条数组 ----
        # 🔴 机制一律**不接线**（S2/S3 才接线）：本段只建数据结构 + 空壳读数，
        #    默认全关 ⇒ 与旧行为逐位一致（C7 digest 钉死 (573985, 8171.692943)）。
        # 格数组：corpse_energy = 每格尸体能量（初值 0）；corpse_age = 每格尸体存续 tick。
        self._corpse_energy = np.zeros(self.world.n_cells, dtype=np.float64)
        self._corpse_age = np.zeros(self.world.n_cells, dtype=np.int64)
        # S2 主机制（设计稿 §5.3 项 3）：腐烂处资源 +50% 的 boost 剩余 tick（初值 0）
        self._corpse_boost = np.zeros(self.world.n_cells, dtype=np.int64)
        # 个体数组：health ∈ [0,1]，初值 1.0（H1）；随生死扩容/压缩（与 _energy 同节拍）。
        self._health = np.ones(n, dtype=np.float64)
        # S1 空壳读数计数器（设计稿 §5.2 项 9；值可为 0 —— S2/S3 接线后才累加）
        self._corpse_eaten_n = 0
        # R165 0-1：池收支四流（见 `corpse_probe` 的 audit 段）
        self._corpse_deposited_e = 0.0   # 投放（意图，钳制前）
        self._corpse_overflow_e = 0.0    # 被 `corpse_cap_per_cell` 丢弃
        self._corpse_scav_e = 0.0        # 食腐取走（能量）
        self._corpse_decayed_e = 0.0     # 腐烂清除
        self._wound_n = 0
        self._contest_n = 0
        # S3 交互（设计稿 §5.4 项 6/7）：争夺战持有者胜率分子 + 血条恐惧项反退化计数
        self._contest_holder_win_n = 0
        self._fearh_flat_n = 0
        self._fearh_dec_n = 0

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
                # S2（wound_enabled）：成功命中但未致死 = 血条消耗战（一击必杀 → 分期付款）
                "wounds": int(self._duel["wounds"]),
                # 🔴 两个成功率口径**必须分开报**：旧口径的低估是真实存在的
                "success_rate_nominal": (round(kills / nominal, 4) if nominal else None),
                "success_rate_real": (round(kills / real, 4) if real else None),
                "note": "nominal=含未出手；real=真出手（排除能量不足/无猎物/已被吃）"
                        "⇒ 旧口径 kills/nominal 会**低估**成功率；"
                        "wounds=血条模式下命中未致死（S2，wound_enabled 时才有）",
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

    # ---- S2 尸体—食腐通道（设计稿 §5.3；`corpse_enabled`）----------------------

    def _deposit_corpse(self, dead: NDArray[np.bool_], energy: NDArray[np.float64]) -> None:
        """死亡投放尸体：三处死因共用（S2 项 2）。

        对 dead 个体：`corpse_energy[cell] += 剩余能量 × corpse_energy_frac`，
        并重置 `corpse_age[cell] = 0`（新尸体腐烂计时从 0 起）。
        **老死个体能量已耗尽 ⇒ 尸体自然很轻**（设计稿 §一，不做特判）。

        🔴 确定性数值（无 RNG 消费）⇒ 关档零开销、零轨迹影响（C7）。
        🔴 单格能量上限 `corpse_cap_per_cell` 钳制（防极点/聚集处无界堆叠）。
        """
        _cwc = self.config.corpse_wound
        cells = self._flat[: len(dead)][dead]
        if cells.size == 0:
            return
        deposit = energy[dead] * float(_cwc.corpse_energy_frac)
        # 🔴 饿死个体能量 ≤ 0 ⇒ deposit 可为负 ⇒ 钳到 ≥0（"老死/饿死自然很轻"= 0，
        #    不得写成负数污染格上总量）。
        deposit = np.maximum(0.0, deposit)
        # 🔴 R161 P1-3 修复：必须用 `np.add.at` **累加** —— 同格多具尸体时
        #    `arr[idx] = f(arr[idx])` 的 fancy-index 读-改-写**只保留最后一次写入**
        #    （实测：同格 2 具各 deposit 9.0 ⇒ 只存 9.0，丢失 50%；死亡越聚集丢得越多，
        #     恰好砍在尸源最富的格上）。
        self._corpse_deposited_e += float(deposit.sum())      # R165 0-1：投放（意图）
        np.add.at(self._corpse_energy, cells, deposit)
        # 单格上限钳制（整数组布尔就地钳制 ⇒ 无重复索引问题）
        _cap = float(_cwc.corpse_cap_per_cell)
        # R165 0-1：溢出量单列（`np.unique` 去重 ⇒ 同格多具不重复计）
        self._corpse_overflow_e += float(
            np.maximum(0.0, self._corpse_energy[np.unique(cells)] - _cap).sum())
        np.clip(self._corpse_energy, 0.0, _cap, out=self._corpse_energy)
        self._corpse_age[cells] = 0

    def _step_corpse_decay(self) -> None:
        """每 tick 腐烂：衰减 + 归还植物池 + patch_boost（S2 项 3；`corpse_enabled`）。

        - 有尸体的格：`corpse_age += 1`；达 `corpse_decay_ticks` ⇒ 腐烂
        - 腐烂：`corpse_to_plant_frac × corpse_energy` 归还植物池（受容量上限），
          其余为分解损失（能量黑洞 → 由归还率控制非 1 的部分）
        - 腐烂处 `corpse_patch_boost`：`corpse_boost[c] = 2000`（持续 2000 tick 的
          +50% 再生加成），本函数同时把 `corpse_boost` 递减并应用加成到资源格

        🔴 确定性数值（无 RNG 消费）⇒ 关档零开销、零轨迹影响（C7）。
        """
        _cwc = self.config.corpse_wound
        has = self._corpse_energy > 0.0
        if has.any():
            self._corpse_age[has] += 1
            ripe = has & (self._corpse_age >= int(_cwc.corpse_decay_ticks))
            if ripe.any():
                cells = np.flatnonzero(ripe)
                # R165 0-1：本 tick 从池中清除的总量（能量）= 腐烂前该格存量
                self._corpse_decayed_e += float(self._corpse_energy[cells].sum())
                # 归还植物池（受容量上限）
                ret = self._corpse_energy[cells] * float(_cwc.corpse_to_plant_frac)
                room = np.maximum(
                    0.0, self.resources._capacity[cells] - self.resources._grid[cells]
                )
                put = np.minimum(ret, room)
                self.resources._grid[cells] += put   # sparse:inc（尸体归还，只增）
                # 腐烂处 patch_boost：+50% 持续 2000 tick
                self._corpse_boost[cells] = 2000
                self._corpse_energy[cells] = 0.0
                self._corpse_age[cells] = 0
        # patch_boost 生效：对 corpse_boost>0 的格补 +50% 再生（受容量上限），并递减
        bo = self._corpse_boost > 0
        if bo.any():
            _bo = np.flatnonzero(bo)
            if self.resources._lazy:
                # R244 v1：只算 `bo` 子集（子集 `_regrowth_amount` ≡ 全场在其上逐位相同；
                # 前提由 `enable_lazy` 校验）—— corpse 开档不再每 tick 全场算再生链
                # （T-G profile：bo>0 占 84% tick、|bo| 中位 69 ⇒ 全场算是纯浪费）。
                _g = self.resources._regrowth_amount(self._tick, _bo)
                room = np.maximum(
                    0.0, self.resources._capacity[_bo] - self.resources._grid[_bo]
                )
                self.resources._grid[_bo] += np.minimum(   # sparse:inc（腐烂 boost，只增）
                    _g * float(_cwc.corpse_patch_boost), room)
            else:
                boost_amt = (
                    self.resources._regrowth_amount(self._tick)
                    * float(_cwc.corpse_patch_boost)
                )
                room = np.maximum(
                    0.0, self.resources._capacity[bo] - self.resources._grid[bo]
                )
                self.resources._grid[bo] += np.minimum(boost_amt[bo], room)   # sparse:inc（腐烂 boost，只增）
            self._corpse_boost[bo] -= 1

    def _step_scavenging(self, P: int, stomach, stomach_cap, genes) -> None:
        """食腐：按 `scav_mult(g16)`（Hill 平滑，**无硬门槛**）从所在格取尸体**入胃**（S2 项 4）。

        `scav_mult = g16^s / (g16^s + gate^s)`（设计稿 §一；`scav_gate` 是半效点不是门槛）。
        取食量 = `eat_amount × scav_mult`（受胃容量限），从格上 `corpse_energy` 扣减。

        🔴 **写胃、不直接写能量**（复用 `stomach` ⇒ 天然限速，防暴富，设计稿 §三）。
        🔴 确定性数值（无 RNG 消费）⇒ 关档零开销、零轨迹影响（C7）。
        🔴 **R165 0-1（2026-09-23）：尸体池 = 能量单位；本函数是唯一换算点**
        （能量 → 质量等价 ÷`eat_efficiency`）。此前池按能量记、却把同一数字当**质量**
        写进 `stomach`（消化时再 ×`eat_efficiency`）⇒ **1 单位尸体吐 3 倍能量**
        （R163 复核实测：每次死亡净创造 1.7E）。
        """
        _cwc = self.config.corpse_wound
        # ⚠️ 分配仍在**质量域**（对 `corpse_cap_per_cell` 与胃容量的语义零改动）：
        #    只把"格上存量"折算成质量等价，扣减时再乘回 `eff` ⇒ Σ取×eff ≡ 池减（精确）。
        eff = max(1e-9, float(self.config.organisms.eat_efficiency))
        g16 = np.clip(genes[:P, Gene.AGGRESSION], 0.0, 1.0)
        _s = float(_cwc.scav_s)
        _g = float(_cwc.scav_gate)
        scav_mult = g16 ** _s / (g16 ** _s + _g ** _s + 1e-12)
        # 胃容量余量（质量：`stomach` 与植物食物同池）
        room = np.maximum(0.0, stomach_cap[:P] - stomach[:P])
        want = np.minimum(
            self.config.organisms.eat_amount * scav_mult,
            room,
        )
        if not (want > 0.0).any():
            return
        # 🔴 R161 P1-2 修复（保留）：**按格汇总需求 → 按格上存量分配**，防"同格多体各自
        #    按全量取"的超发（原实现：格上 1.0 + 同格 12 个体 ⇒ 拿走 4.78、格上 **−3.78**
        #    ⇒ 凭空生成能量）。现为**逐格守恒**：Σ取 ≤ 格存量，且格上恒 ≥ 0。
        cells = self._flat[:P]
        n_cells = self.world.n_cells
        demand = np.bincount(cells, weights=want, minlength=n_cells)   # 质量
        supply_mass = np.maximum(0.0, self._corpse_energy) / eff       # 能量 → 质量等价
        alloc = np.minimum(demand, supply_mass)     # 每格实际可分配（质量）
        # 个体按"格内需求占比"取（demand>0 处；其余 scale=1 免得 0 除）
        scale = np.ones(n_cells, dtype=np.float64)
        nz = demand > 0.0
        scale[nz] = alloc[nz] / demand[nz]
        take = want * scale[cells]                                     # 入胃质量
        if (alloc > 0.0).any():
            # 扣减 = 每格分配总量 × eff（**能量单位**；整数组运算 ⇒ 无重复索引问题，
            # 且 `alloc ≤ supply_mass` ⇒ 池恒 ≥ 0）
            self._corpse_energy -= alloc * eff
            self._corpse_scav_e += float(alloc.sum()) * eff   # R165 0-1：取走（能量）
            self._stomach_scav[:P] += take       # R165 0-2：尸源归因（质量）
            stomach[:P] += take
            self._corpse_eaten_n += int(np.count_nonzero(take > 1e-12))

    def _step_contest(self, P: int, eaters, genes, energy) -> None:
        """S3 争夺食物战（设计稿 §2.3 H3/H4 项 6；`contest_enabled`）。

        触发：**同一格（或邻格）已有他人正在取食**（果实或尸体，`eaters` = 本 tick 取食者）
        ⇒ 高 g16 者可发起驱逐战。判定用 **RHP**：
            `RHP = health × (0.3+g16) × (energy/max_energy)`（初值 a=b=c=1）
            **取食方 ×(1+holder_adv)**（Parker 持有者优势）
        - **H4 升级规则**：`|RHP_A − RHP_B| > escalation_gap` ⇒ 弱方**立即撤退**（不进入多轮，
          败者仍扣血条——"撤退"= 放弃该格；自动省机时）
        - 差距小 ⇒ 进入战斗：RHP 高者胜；**败者 health −= wound_base** + 被迫离开该格
          （放弃取食机会 = 下 tick 移动决策自然离开）；**胜者 energy −= contest_cost_energy**

        🔴 确定性数值（**零 RNG**）⇒ 关档零开销、开档不新增 RNG 抽取（C7 保底）。
        🔴 记录 `contest_n` 总数 + `contest_holder_win_n`（持有者胜率分子，判据 ④）。
        """
        _cwc = self.config.corpse_wound
        max_e = max(float(self.config.organisms.max_energy), 1e-9)
        flat = self._flat[:P]
        g16 = np.clip(genes[:P, Gene.AGGRESSION], 0.0, 1.0)
        _gap = float(_cwc.escalation_gap)
        _adv = float(_cwc.holder_adv)
        _lose = float(_cwc.wound_base)          # 败者扣血（复用血条量级，Δ_loser）
        _cost = float(_cwc.contest_cost_energy) # 胜者代价
        # 🔴 R165 0-2：胜者代价**折进 `cost_attack`**（不新开通道）。
        #    逐 tick 收集后**一次**写入（争夺可达百万次 ⇒ 不能逐次调 `_ec_add`）。
        _paid: list = []
        for i in eaters:
            cell = int(flat[i])
            nb = np.asarray(self.world.neighbors(cell))
            nb_mask = np.isin(flat, nb) & (np.arange(P) != i)
            same = flat == cell
            same = same.copy()
            same[i] = False
            rivals = np.flatnonzero(nb_mask | same)
            if rivals.size == 0:
                continue
            # 挑战者 = 同格/邻格中 g16 最高者（高 g16 者可发起驱逐）
            j = int(rivals[int(np.argmax(g16[rivals]))])
            if g16[j] <= g16[i]:
                continue                       # 挑战者攻击性不比持有者高 ⇒ 不驱逐
            rhp_i = (
                float(self._health[i]) * (0.3 + g16[i])
                * (float(energy[i]) / max_e)
            ) * (1.0 + _adv)
            rhp_j = (
                float(self._health[j]) * (0.3 + g16[j])
                * (float(energy[j]) / max_e)
            )
            self._contest_n += 1
            # H4：RHP 差距大 ⇒ 弱方立即撤退（不进入多轮）
            if abs(rhp_j - rhp_i) > _gap:
                loser = i if rhp_j > rhp_i else j
                # 🔴 撤退也是胜负：挑战者（j）撤退 ⇒ holder 守住该格（holder 胜）
                if loser == j:
                    self._contest_holder_win_n += 1
                self._health[loser] = max(0.0, float(self._health[loser]) - _lose)
                continue
            # 差距小 ⇒ 进入战斗：RHP 高者胜（持有者优势已含在 RHP）
            if rhp_i >= rhp_j:
                winner, loser = i, j
            else:
                winner, loser = j, i
            if winner == i:
                self._contest_holder_win_n += 1
            self._health[loser] = max(0.0, float(self._health[loser]) - _lose)
            energy[winner] -= _cost
            _paid.append(int(winner))
        if _paid:                                  # R165 0-2：一次写入（不是逐次）
            self._ec_add(EC_ATTACK, np.asarray(_paid, dtype=np.int64), _cost)

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
        # 🔴 R165 0-2：net **必须加** `intake_scav` —— 食腐收入已从 `intake_forage`
        #    拆出，若 net 不加它，食腐收入会从净收入里凭空消失（口径破坏）。
        #    因为 forage + scav ≡ 原 forage，故 net **数值不变**（与派工单 §1.2 等价，
        #    只是通道更细）—— 这一条已标注为「与派工单 §1.2 字面不符、需追认」。
        net = (pt[EC_FORAGE] + pt[EC_PRED] + pt[EC_SCAV]
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
                + self._ec_box[b, EC_SCAV]
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
            "note": "net = intake_forage + intake_pred + intake_scav"
                    " − cost_meta − cost_move − cost_attack"
                    "（**不含光合**；R165 0-2 起把 `intake_scav` 计入收入侧 ——"
                    " 与派工单 §1.2 的 6 通道字面不同，但数值等价：forage+scav ≡ 原 forage）；"
                    "net_incl_photo_* 为附加口径",
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
        # 🔴 13.5：与取食段同口径（胃容量独立参数；0 ⇒ 旧公式）
        _scm_pb = float(getattr(ocfg, "stomach_cap_mass", 0.0) or 0.0)
        stomach_cap = (_scm_pb * cap_mult if _scm_pb > 0.0
                       else ocfg.max_energy / max(1e-9, ocfg.eat_efficiency) * 0.5 * cap_mult)
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


    def subpos_probe(self) -> dict | None:
        """亚格坐标读数（B4 口径；13.4 波 1 = `[所有者·天平]` 线）。

        🔴 **关档返回 `None`（未适用），不是 0** —— 与 `l2_probe()` /
        `memory_gradient_stats()` 同一口径铁律（R120 / §五.12）：
        **"没测到" 与 "测到 0" 必须分开**（本项目为此专门立过纪律）。

        键（分三组）：
        * **读回（C4）**：`enabled` / `subdiv` / `speed_gain` / `speed_max` /
          `min_energy_frac` / `lat_floor` / `stay_*` 五项。
          ⚠️ `stay_fear_k` **本波不接线**（依赖波 2 的威胁感知）⇒ 只报读回值，
          防它成为一个"悄悄死掉的参数"。
        * **主判据**：`mean_flat_moves` = **真正换格**的个体占移动者的比例。
          🔴 判据是它、**不是 steps** —— 位移 0.75 格在 `steps` 上有值、
          在 `flat` 层面**等于没动**（设计稿 §2.2）。
        * **反退化**：`slow_frac`（`steps==0` 的"白移动"占比）与 `steps_frac`
          （各档占比；**只有一档非零 ⇒ 速度映射塌成常数** ⇒ 机制名存实亡）。
        """
        cfg = self.config.subpos
        if not bool(getattr(cfg, "enabled", False)):
            return None
        mv = int(self._run_mover_sub_n)
        hist = np.asarray(self._steps_hist, dtype=np.int64)
        tot = int(hist.sum())
        return {
            "enabled": True,
            "subdiv": int(cfg.subdiv),
            "speed_gain": float(cfg.speed_gain),
            "speed_max": float(cfg.speed_max),
            "min_energy_frac": float(cfg.min_energy_frac),
            "lat_floor": float(cfg.lat_floor),
            "stay_base": float(cfg.stay_base),
            "stay_food_k": float(cfg.stay_food_k),
            "stay_signal_k": float(cfg.stay_signal_k),
            "stay_fear_k": float(cfg.stay_fear_k),
            "stay_max": float(cfg.stay_max),
            "mover_n": mv,
            "flat_move_n": int(self._run_flat_move_n),
            "slow_n": int(self._run_slow_n),
            "mean_flat_moves": (round(self._run_flat_move_n / mv, 6) if mv else None),
            "slow_frac": (round(self._run_slow_n / mv, 6) if mv else None),
            "steps_frac": ([round(float(x) / tot, 6) for x in hist] if tot else None),
        }

    # ---- 13.4 波 2A：资源动态读数（T2，本线 = [云端·开发]）-------------------

    def resource_dynamics_probe(self) -> dict | None:
        """斑块休耕—死亡—轮作读数（R176 §12.5 四条 + conservation 自检）。

        🔴 **关档返回 `None`（未适用），不是 0**（R120 口径铁律：没测到 ≠ 测出零）。
        由 `world.resource_dynamics.ResourceDynamics.probe()` 提供四条：
        `dead_cell_frac` / `resting_cell_frac` / `patch_kill_n` / `patch_reborn_n`
        + `mean_capacity_effective`；`conservation_check()` 随 summary 落盘（恒应过）。
        """
        if not bool(getattr(self._rd, "enabled", False)):
            return None
        return self._rd.probe()

    def resource_dynamics_conservation(self) -> dict | None:
        """两条构造期守恒自检（Σcapacity 与面积加权再生倍率，应恒 ≈1）。

        关档 ⇒ None（未适用）。随 summary 落盘供 `[所有者]` 事后抽验。
        """
        if not bool(getattr(self._rd, "enabled", False)):
            return None
        return self._rd.conservation_check()

    # ---- 13.4 波 2B：视野/单格上限读数（T3，本线 = [云端·开发]）----------------

    def wave2b_probe(self) -> dict:
        """T3 读数：span 降级 / cap 触发 / 候选数（关档口径 = 0 或 None 区分）。

        🔴 关档（span=1 且 cap 关）⇒ 计数恒 0（未观测），与"测到 0"分开：
        `span_downgrade_frac` 关档返回 None（未适用）；`cap_blocked_n`/`cap_stay_n`
        恒 0（未触发即 0，因 cap 开关默认关 ⇒ 路径不进）。
        """
        P = len(self._id)
        _sim = self.config.simulation
        _span2 = int(getattr(_sim, "perception_span", 1)) == 2
        _cap_on = bool(getattr(_sim, "cell_occupancy_cap_enabled", False))
        return {
            "perception_span": int(getattr(_sim, "perception_span", 1)),
            "cell_occupancy_cap": int(getattr(_sim, "cell_occupancy_cap", 3)),
            "cell_occupancy_cap_enabled": _cap_on,
            "social_norm": str(getattr(_sim, "social_norm", "auto")),
            "span_downgrade_frac": (
                round(float((~self._span_eff2).mean()), 6) if _span2 else None
            ),
            "cap_blocked_n": int(self._cap_blocked_n) if _cap_on else 0,
            "cap_stay_n": int(self._cap_stay_n) if _cap_on else 0,
            "mean_candidates_per_decision": (
                round(float(self._span_len.mean()), 3) if _span2 else None
            ),
            "note": "span=1 ⇒ span_downgrade/mean_candidates = None（未适用）；"
                    "cap 关 ⇒ blocked/stay = 0（路径不进，非测出零）",
        }

    def l2_probe(self) -> dict | None:
        """L2 机动层读数（R150 B4；本段 = [本地开发] 线）。

        🔴 `l2_dash=False` ⇒ **None（未适用）**，**不是 0**（R120 / §五.12 口径铁律：
        "没测" ≠ "测出零"）。所有分母都写进 `note`，便于独立复算。
        """
        if not bool(self.config.simulation.l2_dash):
            return None
        self._ensure_far_tables()   # S0：只在真的读 dash_* 时构造（关档 early-return 不付钱）
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

    def l1_probe(self) -> dict | None:
        """L1 感知追击层读数（R150 B4；本段 = [所有者·天平] 线 = B1）。

        🔴 `l1_seek`/`l1_fear` **全关** ⇒ **None（未适用）**，**不是 0**
        （R120 / §五.12 口径铁律："没测" ≠ "测出零"）。

        两个**反退化**占比是段一必报项（R148-1 的形态检查）：
        `*_flat_frac` = "某项对该个体**所有候选取同值**"的个体占比 —— 逐位 no-op 的比例。
        ⇒ ≈1.0 意味着该项其实一行行为都没改；≈0 表示逐候选真的在起作用。
        """
        sim0 = self.config.simulation
        l1_seek_on, l1_fear_on = bool(sim0.l1_seek), bool(sim0.l1_fear)
        if not (l1_seek_on or l1_fear_on):
            return None
        dec_n = int(self._l1_dec_n)
        seek_n, fear_n = int(self._seek_term_n), int(self._fear_term_n)
        fear_ind = int(self._fear_ind_n)
        return {
            "l1_seek": l1_seek_on,
            "l1_fear": l1_fear_on,
            "w_seek_max": float(sim0.w_seek_max),
            "w_fear": float(sim0.w_fear),
            "l1_prey_mode": str(sim0.l1_prey_mode),
            "seek_term_mean": (round(self._seek_term_sum / seek_n, 6) if seek_n else None),
            "seek_term_n": seek_n,
            "seek_flat_frac": (round(self._seek_flat_n / dec_n, 6) if dec_n else None),
            # 🔴 自熄占比：`lowagg_field` 在邻域全为 0 ⇒ 该项恒 0 ⇒ **逐位 no-op**。
            #    预注册允许结局「**猎物池枯竭**」（设计稿 §3.4）⇒ 它高**不等于**"没接线"。
            "seek_zero_frac": (round(self._seek_zero_n / dec_n, 6) if dec_n else None),
            "fear_term_mean": (round(self._fear_term_sum / fear_n, 6) if fear_n else None),
            "fear_term_n": fear_n,
            "fear_applied_ind_frac": (round(fear_ind / dec_n, 6) if dec_n else None),
            "fear_flat_frac": (round(self._fear_flat_n / fear_ind, 6) if fear_ind else None),
            "dec_n": dec_n,
            "note": "seek/fear_term_mean = 按**候选**求均值（分母 = 候选数）；fear 项**符号为负**"
                    "（扣分）。seek_flat_frac 分母 = 求值个体数 dec_n；fear_flat_frac 分母 = "
                    "danger 非空的个体数（fear_applied_ind_frac 即其占比）。flat = 该项对该个体"
                    "跨候选取同值 ⇒ **逐位 no-op**（R148-1 形态）。",
        }

    def migration_probe(self) -> dict | None:
        """13.8 日历—罗盘式定向迁徙读数（本线 = [云端·开发] 云归）。

        🔴 `migration.enabled=False` ⇒ **None（未适用）**，**不是 0**
        （R120 / §五.12 口径铁律："没测" ≠ "测出零"）。

        两个**反退化**占比是段一必报项（R148-1 的形态检查）：
        `mig_flat_frac` = "项对该个体**所有候选取同值**"的个体占比 —— 逐位 no-op 的比例。
        ⇒ ≈1.0 意味着接了却一行行为都没改；≈0 表示逐候选真的在起作用。
        ⚠️ 同纬候选（极点行冗余槽位 / 同一行内横移）会**天然**产生 flat>0 ⇒
           **必须报出比例**，但不能直接判"接线失败"。
        `mig_skip_frac` = 因 |A| ≤ `min_abs_anomaly` 而**提前跳过**的个体占比（分段诊断）。

        🔴 P6 量级非僭越：`mig_term_abs_mean` 与觅食项可比（后者上界 ≈ `perc·fr·0.5`）；
           本项过大会**压过**觅食 ⇒ 变成"只迁徙不吃"的伪机制 ⇒ S2 必扫 gain。
        """
        if not self._mig_on:
            return None
        dec_n = int(self._mig_dec_n)
        term_n = int(self._mig_term_n)
        return {
            "migration_enabled": bool(self._mig_on),
            "migration_gain": float(self._mig_gain),
            "migration_min_abs_anomaly": float(self._mig_min_abs),
            "migrate_gene_slot": int(Gene.MIGRATE_BIAS),
            # 🔴 本项按**候选**求绝对均值（分母 = 候选数）：迁移项有正负、单点可为负，
            #    用 |·| 才能与"觅食项量级"做同量纲比较（P6）。
            "mig_term_abs_mean": (round(self._mig_term_sum / term_n, 9) if term_n else None),
            "mig_term_n": term_n,
            "mig_dec_n": dec_n,
            # 🔴 两个反退化占比（段一必报）
            "mig_flat_frac": (round(self._mig_flat_n / dec_n, 6) if dec_n else None),
            "mig_zero_frac": (round(self._mig_zero_n / dec_n, 6) if dec_n else None),
            "mig_skip_frac": (round(self._mig_skip_n / dec_n, 6) if dec_n else None),
            "pp_anom_min": (round(float(self._pp_anom.min()), 6)
                            if self._pp_anom.size else None),
            "pp_anom_max": (round(float(self._pp_anom.max()), 6)
                            if self._pp_anom.size else None),
            # 🔴 主诊断量：**运行期累积**极值（末 tick 快照可能是 δ=0 的退化时刻）
            "pp_anom_lo_run": round(float(self._mig_pp_lo), 6),
            "pp_anom_hi_run": round(float(self._mig_pp_hi), 6),
            "lat_abs_max": (round(float(self._lat_abs.max()), 6)
                            if self._lat_abs.size else None),
            "note": "mig_term_abs_mean = Σ|项|/候选数（绝对值 ⇒ 与觅食项同量纲，供 P6 判定）；"
                    "mig_flat_frac/zero_frac 分母 = 求值个体数 mig_dec_n；flat = 项跨候选取"
                    "同值 ⇒ **逐位 no-op**（R148-1 形态；同纬候选会天然贡献，**报比例不判死**）。"
                    "pp_anom_lo_run/hi_run = A 的**运行期累积**极值（跨季结构诊断）；"
                    "pp_anom_min/max 是**末 tick 快照**，season_period 整除 tick 时可能正好 δ=0"
                    "（A≡0）⇒ **不可单独用作 P2 判定**。",
        }

    def ars_probe(self) -> dict | None:
        """14.9 ARS 双模式觅食读数（本线 = [所有者·天平]）。

        🔴 `ars.enabled=False` ⇒ **None（未适用）**，**不是 0**（R120 / §五.12 口径铁律）。
        """
        if not self._ars_on:
            return None
        dec = int(self._ars_dec_n)
        return {
            "ars_enabled": True,
            "ars_gain": float(self._ars_gain),
            "ars_theta": float(self._ars_theta),
            "ars_kappa": float(self._ars_kappa),
            "ars_giveup": int(self._ars_giveup),
            "ars_gene_slots": (int(Gene.PERSISTENCE), int(Gene.GIVE_UP)),
            "ars_dec_n": dec,
            "ars_inertia_sum": round(float(self._ars_inertia_sum), 6),
            "ars_flat_n": int(self._ars_flat_n),
            "ars_flat_frac": (round(self._ars_flat_n / dec, 6) if dec else None),
            "ars_sw_ie_n": int(self._ars_sw_ie_n),
            "ars_sw_ei_n": int(self._ars_sw_ei_n),
            "ars_rerand_n": int(self._ars_rerand_n),
            "ars_extensive_frac": (
                round(float(self._ars_extensive[: len(self._flat)].mean()), 6)
                if len(self._flat) else None
            ),
        }

    def _ensure_far_tables(self) -> None:
        """惰性构造 L2 冲刺用的 2 跳环表（S0，2026-09-25）。

        为什么惰性
        ----------
        `[实测]` 480×960（46 万格）下原 `__init__` 无条件构造耗时 **29.0 s**；
        128 万格按 O(格数×邻居²) 外推是**分钟级** ⇒ 世界放大后这是第一道墙。
        而这张表**只在 `l2_dash=True` 路径被读**（`l2_probe` 的 dash_* 读数、
        冲刺盲选目标、`step_movement` 的 far 参数）⇒ 默认档完全不需要。

        逐位等价性
        ----------
        构造只依赖 `world`（邻居表）与 `organisms.far_cap`，**与任何 tick 状态无关**
        ⇒ 首次构造出的一定是原来那份 ⇒ 惰性化不改变任何读数（C7 基线不受影响）。

        `FAR_CAP` 语义 = 「strict 2 圈规模 > FAR_CAP 的格**不可冲刺**」（拓扑退化）。
        """
        if self._far_ready:
            return
        n_cells = self.world.n_cells
        self._far_off_c = np.zeros(n_cells + 1, dtype=np.int64)
        self._far_len_c = np.zeros(n_cells, dtype=np.int64)
        _rings: list[np.ndarray] = [np.zeros(0, dtype=np.int32) for _ in range(n_cells)]
        _far_cap = int(self.config.organisms.far_cap)
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
            self._far_len_c[c] = len(_rings[c])
            self._far_off_c[c + 1] = self._far_off_c[c] + self._far_len_c[c]
        self._far_cells_c = (np.concatenate(_rings) if self._far_off_c[-1] else
                             np.zeros(0, dtype=np.int32))
        self._far_excluded_c = int(np.count_nonzero(self._far_len_c == 0))
        del _rings
        self._far_ready = True

    # 🔴 S0 只读属性（2026-09-25）：**读即构造**。
    #   为什么不用"内部调用点显式 ensure"：测试与外部代码会**直接读** `e._far_len`
    #   （`tests/test_l2_dash.py` 就是这么选合法格的）⇒ 若只在引擎内部 ensure，
    #   外部读到的是**空数组** ⇒ 静默错误（本项目最忌的失效形态）。属性化后
    #   无论谁读、什么时候读，拿到的都是与惰性化之前**完全相同**的表。

    @property
    def _far_off(self) -> np.ndarray:
        self._ensure_far_tables()
        return self._far_off_c

    @property
    def _far_len(self) -> np.ndarray:
        self._ensure_far_tables()
        return self._far_len_c

    @property
    def _far_cells(self) -> np.ndarray:
        self._ensure_far_tables()
        return self._far_cells_c

    @property
    def _far_excluded_n(self) -> int:
        self._ensure_far_tables()
        return self._far_excluded_c

    def _refresh_pp_anom(self) -> None:
        """刷新逐格日历轴异常 A = P(φ, t) − 0.5（13.8；**逐 tick**，带 tick 去重）。

        🔴 关档（`_mig_on=False`）⇒ **立即返回**，一次浮点运算都不做 ⇒ 逐位等价。
        🔴 幂等：`LightAndTemperature.photoperiod` 内部按 tick 缓存 ⇒ 同 tick 重复调用
           不重复算三角函数；本方法自己也按 tick 去重，热路径不会累积开销。
        """
        if not self._mig_on:
            return
        if getattr(self, "_pp_anom_tick", -1) == self._tick:
            return
        pp = self.light.photoperiod(
            np.arange(self.world.n_cells, dtype=np.int64), self._tick)
        self._pp_anom = np.ascontiguousarray(pp, dtype=np.float64) - 0.5
        self._pp_anom_tick = self._tick
        # 累积极值（P2/P3 诊断；末 tick 快照可能是 δ=0 的退化时刻，不可单独用于判定）
        self._mig_pp_lo = min(self._mig_pp_lo, float(self._pp_anom.min()))
        self._mig_pp_hi = max(self._mig_pp_hi, float(self._pp_anom.max()))

    def corpse_probe(self) -> dict:
        """尸体—食腐层读数（设计稿 §5.2 项 9 / §5.3）。

        🔴 `corpse_enabled=False` 时 corpse_total/eaten 恒 0（机制未运行）；
           S2 接线后开档有真实读数。**0 = 未启用**（与"测出零"区分）。
        """
        cwc = getattr(self.config, "corpse_wound", None)
        return {
            "corpse_enabled": bool(getattr(cwc, "corpse_enabled", False)),
            "corpse_energy_frac": float(getattr(cwc, "corpse_energy_frac", 0.9)),
            "corpse_decay_ticks": int(getattr(cwc, "corpse_decay_ticks", 600)),
            "corpse_to_plant_frac": float(getattr(cwc, "corpse_to_plant_frac", 0.5)),
            "corpse_patch_boost": float(getattr(cwc, "corpse_patch_boost", 0.5)),
            "corpse_cap_per_cell": int(getattr(cwc, "corpse_cap_per_cell", 200)),
            "scav_gate": float(getattr(cwc, "scav_gate", 0.5)),
            "scav_s": float(getattr(cwc, "scav_s", 2.0)),
            # 🔴 R165 0-1：**单位标记**（可自证"这批用的是哪套单位"）
            "corpse_pool_unit": str(getattr(cwc, "corpse_pool_unit", "energy")),
            "scav_to_energy_divisor": float(self.config.organisms.eat_efficiency),
            "corpse_total": round(float(self._corpse_energy.sum()), 6),
            "corpse_eaten": int(self._corpse_eaten_n),
            # R165 0-1：池收支四流 + **闭账审计**
            "corpse_deposited_e": round(float(self._corpse_deposited_e), 6),
            "corpse_overflow_e": round(float(self._corpse_overflow_e), 6),
            "corpse_scav_e": round(float(self._corpse_scav_e), 6),
            "corpse_decayed_e": round(float(self._corpse_decayed_e), 6),
            "corpse_balance_residual": round(float(
                self._corpse_energy.sum()
                - (self._corpse_deposited_e - self._corpse_overflow_e
                   - self._corpse_scav_e - self._corpse_decayed_e)), 9),
            "corpse_age_max": (int(self._corpse_age.max())
                               if self._corpse_age.size else 0),
            "corpse_boost_active": int(np.count_nonzero(self._corpse_boost > 0)),
            "note": "S2 接线后 corpse_total/eaten 反映实际尸体通道；关档恒 0（未启用）。"
                    "corpse_boost_active = 当前处于 +50% 再生加成的格数（S2 项 3）。"
                    "R165 0-1：`corpse_pool_unit=energy` ⇒ 池以**能量**记，食腐入胃处"
                    "按 ÷`eat_efficiency` 折算成质量 ⇒ Σ取×eff ≡ 池减。"
                    "`corpse_balance_residual` 应恒 0（池收支只有四条流：投放/溢出/"
                    "食腐/腐烂；非 0 ⇒ 漏记某条流）。`corpse_eaten` 是**个体计数**"
                    "（不是质量、也不是能量）—— 2026-09-23 更正，我曾误当质量用。",
        }

    def wound_probe(self) -> dict:
        """血条—受伤层读数（设计稿 §5.2 项 9 / §5.3）。

        🔴 `wound_enabled=False` 时 wound_n/contest_n 恒 0（机制未运行）；
           health_mean 恒 1.0（未受伤）。S2 接线后开档有真实读数。
        """
        cwc = getattr(self.config, "corpse_wound", None)
        P = len(self._id)
        h = self._health[:P]
        return {
            "wound_enabled": bool(getattr(cwc, "wound_enabled", False)),
            "contest_enabled": bool(getattr(cwc, "contest_enabled", False)),
            "wound_base": float(getattr(cwc, "wound_base", 0.35)),
            "wound_heal_rate": float(getattr(cwc, "wound_heal_rate", 0.001)),
            "wound_heal_energy_cost": float(getattr(cwc, "wound_heal_energy_cost", 0.05)),
            "holder_adv": float(getattr(cwc, "holder_adv", 1.2)),
            "escalation_gap": float(getattr(cwc, "escalation_gap", 0.25)),
            "contest_cost_energy": float(getattr(cwc, "contest_cost_energy", 0.5)),
            "w_fear_health": float(getattr(cwc, "w_fear_health", 0.5)),
            "need_aggression_k": float(getattr(cwc, "need_aggression_k", 0.5)),
            "health_mean": (round(float(h.mean()), 6) if P else None),
            "health_low_frac": (round(float(np.mean(h < 0.5)), 6) if P else None),
            "wound_n": int(self._wound_n),
            "contest_n": int(self._contest_n),
            "contest_win_by_holder_frac": (
                round(self._contest_holder_win_n / self._contest_n, 6)
                if self._contest_n else None
            ),
            "fear_health_flat_frac": (
                round(self._fearh_flat_n / self._fearh_dec_n, 6)
                if self._fearh_dec_n else None
            ),
            "note": "S2 接线后 wound_n 反映血条消耗战命中数；S3 接线后 contest_n/"
                    "contest_win_by_holder_frac（判据④）与 fear_health_flat_frac（反退化，"
                    "应≈0）可读；关档恒 0/None（未启用）",
        }

    def stomach_scav_audit(self) -> dict:
        """R165 0-2 归因自检：`_stomach_scav` 必须 ∈ [0, `_stomach`] 且无 NaN。

        🔴 这是"反退化断言"（教训库 #8：有测试 ≠ 不变量被验过）：归因数组有 **4 个
        维护位点**（食腐累加/繁殖传代/新生儿扩容/死亡压缩）—— 漏改任一处都会**静默**
        让归因与胃脱节，而账本只会安静地记错。故本方法给出一句话自检。
        """
        P = len(self._id)
        if P <= 0:
            return {"n": 0, "over": 0, "negative": 0, "max_excess": 0.0}
        ss = self._stomach_scav[:P]
        st = self._stomach[:P]
        return {
            "n": int(P),
            "over": int(np.count_nonzero(ss > st + 1e-9)),
            "negative": int(np.count_nonzero(ss < -1e-12)),
            "max_excess": round(float((ss - st).max()), 9),
            "note": "over>0 或 negative>0 ⇒ 归因数组与胃脱节（繁殖/死亡/扩容位点漏改）",
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
        ⚠️ 只统计【主 rng】的显式包装方法（random/integers/normal/uniform/standard_normal）。
        F-D2 修复（R230 T-C，B′）后 D2 感知噪声/Steels 配对走**每引擎独立**的 legacy
        `_d2_rng`（与修复前的全局口径相同：不计入 draws）⇒ 计数语义与旧版逐位一致，
        引擎不再有全局 `np.random` 消费点。
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

    # ---- R240 T8：气味场（稀疏源 / 读数）--------------------------------------

    def _smell_sources(self) -> dict:
        """气味场**稀疏源**（只返回「有源」的格）—— 由引擎算：只有它握有资源/个体/基因。

        v1 口径（设计稿 §二 表 / `SmellConfig` docstring）：
          · `food`：`grid > 0` 的格；注入量 = `grid / max(capacity)`（∈[0,1] 归一）
          · `prey`：全部存活个体（每个体 1.0）
          · `risk`：`g16(AGGRESSION) ≥ risk_g16_threshold` 的个体（捕食者留味）
          · `kin` ：全部存活个体（与 `prey` 同源；读端用途不同）

        🔴 **稀疏**是本机制的红线之一（设计稿 §8.2-③）：返回的 `cells` 只含源格
        （食物格 2.8% / 个体 0.65%），`SmellField.update()` 只对这些格做写。
        """
        ch = self.smell.channels
        out: dict[str, tuple] = {}
        if "food" in ch:
            g = self.resources._grid
            cells = np.flatnonzero(g > 0.0)
            if cells.size:
                cap_max = float(self.resources._capacity.max())
                denom = cap_max if cap_max > 0.0 else 1.0
                out["food"] = (cells.astype(np.int64),
                               (g[cells] / denom).astype(np.float64))
        P = len(self._id)
        if P and ("prey" in ch or "risk" in ch or "kin" in ch):
            flat = self._flat[:P].astype(np.int64)
            if "prey" in ch:
                out["prey"] = (flat, np.ones(P, dtype=np.float64))
            if "kin" in ch:
                out["kin"] = (flat, np.ones(P, dtype=np.float64))
            if "risk" in ch:
                m = (self._genes[:P, int(Gene.AGGRESSION)]
                     >= float(self.smell.cfg.risk_g16_threshold))
                if bool(m.any()):
                    out["risk"] = (flat[m], np.ones(int(m.sum()), dtype=np.float64))
        return out

    def smell_probe(self) -> dict | None:
        """气味场读数（B4 口径：**全关返回 `None`** = 「未适用」，不是 0）。

        键见 `world/smell_field.py::SmellField.probe()`（含性能分解 `ms_per_update`）。
        """
        if self.smell is None:
            return None
        return self.smell.probe()

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
        _rd_on = bool(getattr(self._rd, "enabled", False))
        if _rd_on:
            # 13.4 波 2A（T2）：资源动态**强制 Python 路径**（H3 已拦 use_sim_core）。
            # 再生闸：休耕/死亡格再生乘子（bgzero 档默认 0.5 = 减产不停产）；其余 = 1
            # （enabled=False ⇒ 全 1 = 逐位等价）。
            self._rd_intake_sum[:] = 0.0          # 逐 tick 重建 intake（note_tick 输入）
            if self.resources._lazy:
                # ---- R244 v1：rd 再生子集化（只结算脏格 = `_grid < _capacity`）----
                # 逐位等价三条：① 子集 `_regrowth_amount` ≡ 全场在其上逐位相同（R217 前提，
                # `enable_lazy` 已逐条校验）；② 净格跳过 = 逐位 no-op（`g = 名义×闸 ≥ 0`，
                # 闸非负由 `_rd_mults_ok` + `enable_lazy` 前提把关）；③ rd 前提 = "掩码/容量
                # 逐 tick 恒定"（bgzero ⇒ R226 禁搬移；非 bgzero 档已在构造期 fail-loud）。
                self._rd_growth_sum[:] = 0.0      # 逐 tick 清零（预分配数组 + 子集回填）
                idx = np.flatnonzero(self.resources._dirty_mask)   # ~0.28 ms @460800
                if idx.size:
                    nominal = self.resources._regrowth_amount(self._tick, idx)
                    self._rd_growth_sum[idx] = nominal    # 名义再生（未乘闸；note_tick 分母）
                    mult = self._rd.growth_multiplier()
                    new = self.resources._grid[idx] + nominal * mult[idx]
                    cap_i = self.resources._capacity[idx]
                    np.minimum(cap_i, new, out=new)
                    self.resources._grid[idx] = new   # sparse:lazy（rd 子集结算；脏标记在下方维护）
                    # 到顶格转净（未到顶留脏；`>=` 而非 `==`：min 后不可能 > cap）
                    self.resources._dirty_mask[idx[new >= cap_i]] = False
            else:
                self._rd_growth_sum = self.resources._regrowth_amount(self._tick)
                growth = self._rd_growth_sum * self._rd.growth_multiplier()
                np.minimum(self.resources._capacity, self.resources._grid + growth,   # sparse:n/a（rd 全场路径）
                           out=self.resources._grid)
        elif self._use_sim_core and self.resources.distribution == "uniform":
            # 3.4：资源再生长沉到 Rust（与 ResourceField.regrow 逐位等价，
            # 默认 temp_sensitivity=1.0 时严格一致；≠1 有 ≤1-ULP 差异）
            self._sim_core.regrow(   # sparse:n/a（Rust 直写；范围锁已排除）
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
            self._sim_core.regrow_patchy(   # sparse:n/a（Rust 直写；范围锁已排除）
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
        # S2 尸体腐烂（设计稿 §5.3 项 3；`corpse_enabled`）：衰减 + 归还植物池 + patch_boost
        # 🔴 放资源更新段内、regrow 之后 ⇒ 腐烂归还的养分从**下一 tick** 的再生开始被利用
        #    （不干扰本 tick 已完成的 regrow ⇒ 关档/开档的资源管线时序清晰）。
        _cwc_tick = getattr(self.config, "corpse_wound", None)
        if bool(getattr(_cwc_tick, "corpse_enabled", False)):
            self._step_corpse_decay()
        # 信号场时间推进（标记衰减、过期清零）
        self.signals.tick()
        # ---- R240 T8：气味场（**每 k tick 才更新**；设计稿 §8.2-② 是红线）----
        #   🔴 这是本机制**唯一**的全场路径（`SmellField.update()`）：全关 ⇒ `smell is None`；
        #     开档 ⇒ 每 `update_every` tick 一次（不是每 tick）⇒ 成本 ÷k。
        #   源由引擎算（**稀疏**：只给有源的格）⇒ 注入项与扩散解耦（§8.2-③）。
        if self.smell is not None and self._tick % self.smell.update_every == 0:
            self.smell.update(self._smell_sources(), self._tick)
            # R244 §二：场一变 ⇒ 消费端按代缓存**全体失效**（gen+1）
            self._smell_gen += 1
        born, died, deaths = self._step_population()
        # 13.4 波 2A（T2）：每 tick 末轮作（死格重入候选池 + 反荒漠化闸 + 到期休耕恢复）
        # + 按当前掩码**重算** `_capacity`（`_capacity` 是基准 ⇒ 动态折扣走 `capacity_multiplier`）。
        # 🔴 R226 修复①②：搬移已上移到 `note_tick`（死亡当 tick 原子完成）⇒ `rotate`
        #    不再消费引擎 RNG（rd 改用自有 `_rng`）；回写前做 **diff 校验**（fail-loud 兜底）。
        if _rd_on:
            self._rd.rotate(self._tick)
            _cap_new = self._rd.capacity_from_base()
            # R226 裁定②：回写前校验「生产格数」与「Σcapacity」不变量（破裂即报警）
            self._rd.validate_writeback(_cap_new, self._rd._mask)
            self.resources._capacity[:] = _cap_new
            # 🔴 轮作会搬移斑块掩码 ⇒ 必须回写 `ResourceField._patch_mask`，
            #    否则 `_regrowth_amount` 的斑块倍率 / 尸体 patch_boost 仍用旧掩码
            #    （双掩码漂移 = I2 同族：两处规则不一致 ⇒ 归因不干净）。
            if self.resources._patch_mask is not None:
                self.resources._patch_mask[:] = self._rd._mask
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

    def _culture_learn_python(
        self, j_idx: np.ndarray, adult_qual: np.ndarray) -> None:
        """未成年向“邻域成年均值”EWMA 学习（Python 参考路径）。

        参数
        ----
        j_idx : NDArray[int64]
            未成年个体索引（升序）。
        adult_qual : NDArray[bool]
            与全种群等长的布尔数组，True = 成年（``age_f >= maturity_age``）。

        语义：每个未成年取其所在格邻域内全部成年个体的解读表行，去重 + 按个体
        索引升序，逐行顺序求和取均值，再 ``row += 0.1 * (mean - row)``。
        逐位对拍测试见 ``tests/test_p2_culture_vec.py``。

        🔴 S2（2026-09-26 P2 向量化，`[实测]`）：逐位等价改写。旧实现（S1d 稀疏
        字典版）逐未成年跑 Python 循环（查邻表 / 收格内成年 / sorted(set) /
        (K,16).mean(axis=0)），T4 档行内剖析占整 tick 14.9%（P1 交付）。

        改这里的代码必须保持三条不变量（否则 C7 digest 漂移）：
        ① 求和 = 每组内“成年索引升序”逐行顺序累加。`np.mean(axis=0)` 对 C 连续
           (K,16) 正是顺序累加（已实测逐位一致）；但 `np.add.reduceat` **不是**
           顺序累加（分段和与顺序和实测有位差）⇒ 这里用无缓冲 `np.add.at`
           （按索引出现顺序累加），以首行做初值 —— 首行为初值可同时覆盖 K=1 段。
        ② 去重与升序在 **(未成年组, 成年) 对级** 完成：极点带附近邻居表有重复格
           （球面退化，见表构造注释）⇒ 必须去重；升序 = 旧 flatnonzero 输出序。
        ③ 均值只读成年行、更新只写未成年行 ⇒ 两集合不相交，可整体先算后写。
        另：同格未成年邻域恒相同 ⇒ 按“所在格”归组，一组只算一次（结果不变）。
        """
        if not j_idx.size:
            return
        _cells, _cell_inv = np.unique(self._flat[j_idx], return_inverse=True)
        _cell_inv = _cell_inv.reshape(-1)
        # 邻居格集合：普通格取 (U,8) 主表；极点格 = 相邻整带 (cols,)
        # （分支与 world.neighbors() 一致：极点行优先取上极带）
        _is_pole = np.asarray(self.world.is_pole(_cells))
        _u_np = np.flatnonzero(~_is_pole)
        _u_pole = np.flatnonzero(_is_pole)
        _nb_parts: list[np.ndarray] = []
        _own_parts: list[np.ndarray] = []
        if _u_np.size:
            _nb_parts.append(self._nb_table[_cells[_u_np]])
            _own_parts.append(np.repeat(_u_np, self._nb_table.shape[1]))
        if _u_pole.size:
            _prow = _cells[_u_pole] // self.world.cols
            _nb_parts.append(self._pole_nb[np.where(_prow == self._pole_top, 0, 1)])
            _own_parts.append(np.repeat(_u_pole, self.world.cols))
        _nb_cells = np.concatenate([p.ravel() for p in _nb_parts])
        _owner = np.concatenate(_own_parts)
        # 成年个体按格建 CSR（stable ⇒ 格内保持个体索引升序）
        _a_idx = np.flatnonzero(adult_qual)
        _a_order = np.argsort(self._flat[_a_idx], kind="stable")
        _a_cell = self._flat[_a_idx][_a_order]
        _a_pos = _a_idx[_a_order]
        _beg = np.searchsorted(_a_cell, _nb_cells, side="left")
        _cnt = np.searchsorted(_a_cell, _nb_cells, side="right") - _beg
        _tot = int(_cnt.sum())
        if _tot:
            # 拼接式整段收集（不补零 ⇒ 无 maxK×S 的内存退化）
            _ex = np.concatenate(([0], np.cumsum(_cnt)[:-1]))
            _pos = np.repeat(_beg, _cnt) + (
                np.arange(_tot, dtype=np.int64) - np.repeat(_ex, _cnt))
            _own_rep = np.repeat(_owner, _cnt)
            _a_rep = _a_pos[_pos]
            _ord = np.lexsort((_a_rep, _own_rep))
            _own_k, _a_k = _own_rep[_ord], _a_rep[_ord]
            _keep = np.empty(_own_k.size, dtype=bool)
            _keep[0] = True
            np.logical_or(_own_k[1:] != _own_k[:-1],
                          _a_k[1:] != _a_k[:-1], out=_keep[1:])
            _own_k, _a_k = _own_k[_keep], _a_k[_keep]
            _seg = np.flatnonzero(np.concatenate(([True], _own_k[1:] != _own_k[:-1])))
            _seg_cnt = np.diff(np.concatenate((_seg, [_own_k.size])))
            _acc = self._interpret[_a_k[_seg]]
            _rest = np.ones(_a_k.size, dtype=bool)
            _rest[_seg] = False
            if _rest.any():
                np.add.at(_acc, np.repeat(np.arange(_seg.size), _seg_cnt)[_rest],
                          self._interpret[_a_k[_rest]])
            _means = _acc / _seg_cnt[:, None]
            _seg_of_cell = np.full(_cells.size, -1, dtype=np.int64)
            _seg_of_cell[_own_k[_seg]] = np.arange(_seg.size)
            _j_seg = _seg_of_cell[_cell_inv]
            _hit = _j_seg >= 0
            if _hit.any():
                _jv = j_idx[_hit]
                _row = self._interpret[_jv]
                self._interpret[_jv] = _row + 0.1 * (_means[_j_seg[_hit]] - _row)

    # ---- P2②：移动决策整块向量化（vanilla 配置专用快路径） ----------------

    def _move_decide_batch(
        self,
        mi: NDArray[np.int64],
        food_ratio: NDArray[np.float64],
        sig_present: NDArray[np.float64],
        densities: NDArray[np.float64],
        rand_choice: NDArray[np.int64],
        rep_w: float,
        h_norm: NDArray[np.float64] | None = None,
        hm_beta: float = 0.0,
        smell_arr: "NDArray[np.float64] | None" = None,
    ) -> NDArray[np.int64]:
        """向量化移动决策（逐位等价；仅由调用点 `_batch_ok` 门启用）。

        门槛（调用点）保证 span2/asym/noise/softmax/mem_grad/L1/血条恐惧/迁徙/
        ARS/cap/L2 全关 ⇒ 参考循环退化为「纯打分 + argmax/平局」⇒ 可整块算。
        **R247 HM 例外**：饥饿调制**已在本函数内接线**（`h_norm` 逐行调制 `perc`，
        与参考循环/ Rust 逐字同式）⇒ HM 开启**不禁用**快路径（不进 `_batch_ok` 排除表）。
        逐位等价依据：
        - 打分各项与逐个体式**同操作数、同运算顺序**（列广播不改变单元素运算）；
        - 记忆项/解读项按**行条件**取舍：`np.where` 取原值 ⇒ 不做 `+0.0`
          （否则 `-0.0` 会被加成 `0.0` = 位差）；
        - `argmax(axis=1)` 与逐行 `np.argmax` 同为「首个最大值」；
        - 平局判据 `max-min < 1e-9` 与平局取值 `rand_choice % len(nb)` 同式。
        候选集按**行**分组（与 `world.neighbors()` 的派发同源）：普通行 = `_nb_table`
        行（8 邻，直接查表）；上/下行 = `_pole_nb` 整行（cols 邻）。
        RNG 契约：本函数**不消费任何随机数**（vanilla 路径无 `rng.*` 调用）。
        """
        genes = self._genes
        cells = self._flat[mi]
        cell_rows = cells // self.world.cols
        is_top = cell_rows == self.world._pole_top
        is_bot = cell_rows == self.world._pole_bottom
        targets = np.empty(mi.size, dtype=np.int64)
        normal = ~(is_top | is_bot)
        groups: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
        if normal.any():
            groups.append((np.flatnonzero(normal), self._nb_table[cells[normal]]))
        for _mask, _prow in ((is_top, 0), (is_bot, 1)):
            if _mask.any():
                _g = np.flatnonzero(_mask)
                groups.append((
                    _g,
                    np.broadcast_to(self._pole_nb[_prow],
                                    (_g.size, self.world.cols)),
                ))
        for g, nb in groups:
            gi = mi[g]
            perc = genes[gi, Gene.PERCEPTION]
            if h_norm is not None and h_norm.size > 0:
                # R247 HM ②：逐行调制 perc（与参考循环/Rust 逐字同式：`1.0 + beta * h_norm`）。
                perc = perc * (1.0 + hm_beta * h_norm[gi])
            soc = (genes[gi, Gene.SOCIABILITY] - 0.5) * 2.0
            sig_weight = self._trust[gi] * (0.5 + rep_w * self._trust[gi])
            score = (perc[:, None] * (food_ratio[nb] * 0.5
                                      + sig_present[nb] * sig_weight[:, None])
                     + soc[:, None] * densities[nb])
            wm = self._work_memory[gi]
            has_mem = (wm >= 0).any(axis=1)
            if has_mem.any():
                mem_in_nb = (nb[:, :, None] == wm[:, None, :]).any(axis=2)
                score = np.where(
                    has_mem[:, None],
                    score + (0.3 * perc)[:, None] * mem_in_nb.astype(np.float64),
                    score,
                )
            nb_sigs = self.signals._marks[nb]
            has_sig = (nb_sigs > 0).any(axis=1)
            if has_sig.any():
                interp = np.where(
                    nb_sigs > 0, self._interpret[gi[:, None], nb_sigs], 0.0)
                score = np.where(
                    has_sig[:, None],
                    score + (0.4 * perc)[:, None] * interp,
                    score,
                )
            # R244 §二：气味消费端（第三处同式；与参考循环/Rust 同位置、同操作数、同顺序）
            if smell_arr is not None:
                score = score + perc[:, None] * smell_arr[nb]
            k = nb.shape[1]
            pick = np.where(
                (score.max(axis=1) - score.min(axis=1)) < 1e-9,
                rand_choice[g] % k,
                np.argmax(score, axis=1),
            )
            targets[g] = nb[np.arange(g.size), pick]
        return targets

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
        # 13.8：日历轴（逐格日长异常 A=P−0.5）**逐 tick 刷新**（关档 = 立即返回，零开销）。
        self._refresh_pp_anom()
        # 13.4 波 2B（T3）：span/cap 开关在**函数级**定义（移动段 Nm=0 时也需捕食段可用）
        _span2_on = (int(getattr(self.config.simulation,
                                 "perception_span", 1)) == 2)
        _cap_on = bool(getattr(self.config.simulation,
                               "cell_occupancy_cap_enabled", False))
        _cap_val = int(getattr(self.config.simulation,
                               "cell_occupancy_cap", 3))

        # S2 血条/尸体（设计稿 §2.3 H2 / §5.3）：`wound_enabled` 开关 + 参数。
        # 构造期已由 H3 保证：开 + use_sim_core=True ⇒ 硬报错 ⇒ 此处仅在 Python 路径消费。
        _cwc_pred = getattr(self.config, "corpse_wound", None)
        _wound_on = bool(getattr(_cwc_pred, "wound_enabled", False))
        # 13.4 波 3（T4）：击杀能量进尸体的门控 = `corpse_enabled`（T4 反转后
        # 捕食收益走尸体通道；corpse 关 ⇒ 旧 transfer 兜底守恒）
        _corpse_on_tick = bool(getattr(_cwc_pred, "corpse_enabled", False))

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
            # 14.9 ARS（D3 对称耦合）：油箱大 ⇒ 维持也贵（否则=给高 pers 个体发能量补贴）
            if self._ars_on and self._ars_kappa > 0.0:
                digest_rate = digest_rate * (
                    1.0 + self._ars_kappa * np.clip(genes[:P, Gene.PERSISTENCE], 0.0, 1.0)
                )
            # 每 tick 最多转化这么多；不得超出胃里有的
            digest = np.minimum(stomach, digest_rate)
            # 🔴 13.5（S2）：**吸收率**（吃进去多少变成能量）+ **未吸收回流本格植物池**。
            #   默认 `assim_herb = assim_carn = 1.0`、`assim_return_frac = 0.0`
            #   ⇒ `_assim` 恒 1、无回流 ⇒ **与旧版逐位一致**。
            #   归因（人话）：胃是"混合饭盒"，用 `_stomach_scav / stomach` 比例区分
            #   "植物腿 / 尸体腿"（R165 0-2 已有该归因，此处复用，**不新增状态量**）。
            #   ⚠️ 口径：此后 `_dg` 是**净吸收能量**（已乘吸收率）⇒ 账本 `intake_*` 同步为
            #   净口径；"未吸收回流"目前只落回植物池、**无独立账户**（S3 前补一个计数）。
            _ah = float(getattr(ocfg, "assim_herb", 1.0) or 1.0)
            _ac = float(getattr(ocfg, "assim_carn", 1.0) or 1.0)
            _ar = float(getattr(ocfg, "assim_return_frac", 1.0) or 0.0)   # R208 §三：兜底 == config 默认
            if _ah < 1.0 or _ac < 1.0:
                _scf = np.clip(
                    np.divide(self._stomach_scav[:P], stomach,
                              out=np.zeros_like(stomach), where=stomach > 0.0),
                    0.0, 1.0)
                _assim = _ah + (_ac - _ah) * _scf
            else:
                _assim = 1.0
            _dg = digest * ocfg.eat_efficiency * _assim   # 食物 → 能量（含吸收率）
            energy += _dg
            if _ar > 0.0 and (_ah < 1.0 or _ac < 1.0):
                _back = digest * (1.0 - _assim) * _ar     # 质量单位
                if float(_back.sum()) > 0.0:
                    _rfg = self.resources
                    _dep = getattr(_rfg, "deposit", None)
                    if callable(_dep):
                        # 13.5 约定接口（`[本地开发]` 在 ResourceField 侧提供，按容量封顶）
                        _dep(self._flat[:P], _back)
                    else:
                        # 回退：就地写入 + 按容量封顶（语义等价；供接口未就绪时先跑通）
                        np.add.at(_rfg._grid, self._flat[:P], _back)
                        np.minimum(_rfg._grid, _rfg._capacity, out=_rfg._grid)
            # R141 P0：`intake_forage` = **真正进入能量的量**（已乘 `eat_efficiency`）。
            # 🔴 口径坑（我第一版就踩了）：若记"进胃的原始食物量"（未乘 3.0），
            #    `net = 收入 − 支出` 会**虚假为负**（实测 −0.23/人·tick，而个体显然活着）
            #    ⇒ 必须与 `intake_pred`/`intake_photo`（都直接进能量）同口径。
            # 🔴 R165 0-2（2026-09-23）：**把胃内容按比例拆成"植物/尸体"两条腿**。
            #    胃是**充分混合池**（两来源在 `stomach` 里不可区分）⇒ 按 `scav/stomach`
            #    比例归因；**两腿之和恒等于 `_dg`** ⇒ 净收入数值一分不变，只是可分离。
            #    关档（`corpse_enabled=False`）走原路径，零开销、零轨迹影响。
            if bool(self.config.corpse_wound.corpse_enabled):
                _sc_frac = np.clip(
                    np.divide(self._stomach_scav[:P], stomach,
                              out=np.zeros_like(stomach), where=stomach > 0.0),
                    0.0, 1.0,
                )
                _dg_scav = _dg * _sc_frac
                self._ec_add(EC_FORAGE, None, _dg - _dg_scav)
                self._ec_add(EC_SCAV, None, _dg_scav)
                self._stomach_scav[:P] -= digest * _sc_frac   # 同步扣减归因
            else:
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

        # 2.5) S2 愈合（设计稿 §2.3 H5；`wound_enabled`）：health += heal_rate（上限 1.0），
        #      并扣代谢能量 heal_energy_cost（愈合不免费）。
        #      🔴 确定性数值（无 RNG 消费）⇒ 关档零开销、零轨迹影响（C7）。
        if _wound_on:
            _heal = self._health[:P] < 1.0
            if _heal.any():
                self._health[:P] = np.minimum(
                    1.0, self._health[:P] + float(_cwc_pred.wound_heal_rate)
                )
                _heal_cost = float(_cwc_pred.wound_heal_energy_cost)
                energy[_heal] -= _heal_cost
                # R165 0-2：愈合耗能**折进 `cost_meta`**（维持类，不新开通道）
                self._ec_add(EC_META, np.flatnonzero(_heal), _heal_cost)

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
        #    14.9 ARS：本 tick 每个体**实际咬到量**先清零（ARS 关时不消费 ⇒ 零轨迹影响）
        if self._ars_on:
            self._out_taken[:P] = 0.0
        #    饱食度：胃容量上限（基础 = max_energy/eat_efficiency/2，g5 缩放 0.5~2 倍）
        #    进食量：每 tick 最多 eat_amount（g4 缩放 0.5~1.5 倍）
        # 内评 §三 观察项 1：取出上一 tick 的 oracle 成交落点（**只消费一次**，保证
        # 「t 付款 → t+1 在落点进食」的时序对应；无 oracle 时零开销）
        pend_cells = self._recv_pend_cells
        if self._oracle_on:
            self._recv_pend_cells = []
        eat_mult = 0.5 + genes[:, Gene.EAT_AMOUNT] * 1.0
        cap_mult = 0.5 + genes[:, Gene.STOMACH_CAP] * 1.5
        # 🔴 13.5（S1）：**胃容量做成独立参数** —— 否则它会跟着 `max_energy` 一起变大，
        #   把"体能↑、胃↓"两个意图互相抵消（R166 §三 坑②）。
        #   `stomach_cap_mass <= 0`（默认）= 旧公式 ⇒ 逐位一致。`cap_mult` 仍按基因缩放。
        _scm = float(getattr(ocfg, "stomach_cap_mass", 0.0) or 0.0)
        if _scm > 0.0:
            stomach_cap = _scm * cap_mult
        else:
            stomach_cap = (
                ocfg.max_energy / max(1e-9, ocfg.eat_efficiency) * 0.5 * cap_mult
            )
        # 🔴 13.5（S2）：**进食阈值** —— 胃低于容量的该比例才进食（"不饿不吃"）。
        #   默认 1.0 ⇒ 旧判定 `stomach < stomach_cap`（逐位一致）。
        # 14.9 ARS（D3 对称耦合）：油箱随 pers 变大（κ 上限 0.5 ⇒ 最多 +50%）
        if self._ars_on and self._ars_kappa > 0.0:
            stomach_cap = stomach_cap * (
                1.0 + self._ars_kappa * np.clip(genes[:, Gene.PERSISTENCE], 0.0, 1.0)
            )
        _eat_frac = float(getattr(ocfg, "eat_threshold_frac", 0.0) or 1.0)   # 兜底==config 默认；`or 1.0` 保语义
        _eat_gate = stomach_cap * _eat_frac
        # S2/S3 尸体—争夺（设计稿 §5.3/5.4）：开关只读一次，供 4.3/4.3b 复用
        _cwc_scav = getattr(self.config, "corpse_wound", None)
        # 13.4 波 2A（T2）：资源动态开关（note_tick/取食累计用；关档 = 零轨迹影响）
        _rd_on = bool(getattr(self._rd, "enabled", False))
        if (stomach < _eat_gate).any():
            eaters = np.flatnonzero(stomach < _eat_gate)
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
                self._sim_core.consume_many(   # sparse:n/a（Rust 直写；范围锁已排除）
                    self.resources._grid, self._flat[eaters], want, taken
                )
            else:
                taken = self.resources.consume_many(self._flat[eaters], want)
            stomach[eaters] += taken
            # 14.9 ARS：本格咬到计入（通道 1/3）
            if self._ars_on:
                self._out_taken[eaters] += taken
            # 13.4 波 2A（T2）：每格被吃量**质量**累计（note_tick 的 intake 输入；
            # 关档 `_rd_intake_sum` 不消费 ⇒ 零轨迹影响）。
            if _rd_on:
                np.add.at(self._rd_intake_sum, self._flat[eaters], taken)
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
                    self._sim_core.consume_many(   # sparse:n/a（Rust 直写；范围锁已排除）
                        self.resources._grid, targets, short[hf], taken2
                    )
                else:
                    taken2 = self.resources.consume_many(targets, short[hf])
                stomach[hf] += taken2
                # 14.9 ARS：邻格咬到计入（通道 2/3；g10 邻格觅食）
                if self._ars_on:
                    self._out_taken[hf] += taken2
                # 13.4 波 2A（T2）：邻格取食同样计入 per-cell intake（质量单位）。
                if _rd_on:
                    np.add.at(self._rd_intake_sum, targets, taken2)
            # 4.3b) S3 争夺食物战（设计稿 §2.3 H3/H4 项 6；`contest_enabled`）：同一格/邻格
            #       有他人正在取食（`eaters` = 本 tick 取食者）⇒ 高 g16 者可驱逐
            #       （RHP + 持有者优势 + 升级阈值 + 撤退）。🔴 确定性数值（零 RNG）。
            if bool(getattr(_cwc_scav, "contest_enabled", False)):
                self._step_contest(P, eaters, genes, energy)

        # ── 14.9 ARS：期待更新 + 模式切换（进食结算后、移动决策前）────────────────
        #   快平均 = 最近几十 tick 的摄入（"现在这一口还行吗"）
        #   慢平均 = 最近几百 tick 的摄入（"我最近过得算不错吗" = 个体自己的期待）
        #   驻留→赶路：快 < θ×慢（吃得明显不如之前）或 从未咬到（fast≈0）
        #   赶路→驻留：咬到了（out_taken > 0）
        if self._ars_on:
            _df = 1.0 - 1.0 / self._ars_fast_tau
            _ds = 1.0 - 1.0 / self._ars_slow_tau
            _ot = self._out_taken[:P]
            self._feed_fast[:P] = _df * self._feed_fast[:P] + _ot / self._ars_fast_tau
            self._feed_slow[:P] = _ds * self._feed_slow[:P] + _ot / self._ars_slow_tau
            _was = self._ars_extensive[:P]
            _to_ext = (~_was) & (
                (self._feed_fast[:P] < self._ars_theta * self._feed_slow[:P])
                | (self._feed_fast[:P] <= 1e-12)
            )
            _to_int = _was & (_ot > 0.0)
            self._ars_extensive[:P] = np.where(
                _to_ext, True, np.where(_to_int, False, _was)
            )
            self._ars_sw_ie_n += int(_to_ext.sum())
            self._ars_sw_ei_n += int(_to_int.sum())

        # 4.3) S2 食腐（设计稿 §5.3 项 4；`corpse_enabled`）：按 g16 Hill 平滑从所在格取
        #      尸体**入胃**（受胃容量限）。🔴 确定性数值（无 RNG 消费）⇒ 关档零轨迹影响。
        if bool(getattr(_cwc_scav, "corpse_enabled", False)):
            self._step_scavenging(P, stomach, stomach_cap, genes)

        # 4.3c) 13.4 波 2A（T2）：取食结算后记 `note_tick`（休耕 + 判死）。
        #       `_rd_intake_sum` = 本 tick 每格**被吃量**（质量单位；果实+邻格取食累加，
        #       食腐写胃不入 note_tick —— 尸体不是"本格活再生"的取食压力）。
        #       🔴 与 `_rd_growth_sum`（regrow 段已存的本 tick 名义再生）配对。
        if _rd_on:
            if self.resources._lazy and self._rd.kill_denom == "regrowth":
                # R244 v1 分母补算：regrow 段只填**脏格**名义再生 ⇒ 净格（`grid==cap`）本 tick
                # 被吃时分母缺值 ⇒ 被 `denom_k>0` 静默滤出（极端密度通道漏杀：不炸但结果不同）。
                # 只对被吃格补算（O(活跃)）；`_rd_growth_sum==0` 含"已结算但名义=0"格 ⇒ 重算同值，无害。
                _eaten = np.flatnonzero(self._rd_intake_sum)
                if _eaten.size:
                    _miss = _eaten[self._rd_growth_sum[_eaten] == 0.0]
                    if _miss.size:
                        self._rd_growth_sum[_miss] = self.resources._regrowth_amount(
                            self._tick, _miss)
            self._rd.note_tick(self._rd_intake_sum, self._rd_growth_sum, self._tick)

        # 4.4) 工作记忆写入（L5）：食物丰富的格子记入 4 槽（round-robin）
        cur_flat = self._flat[:P]
        # R121 §4.1：字面量 0.5 → 显式常量 `FOOD_RICH_LEVEL`（**只命名、不改数值**）。
        # 注意比较符仍是 `>`（与改名前逐位一致）；该常量语义 = "多到值得记住"，
        # 与 `CultureConfig.food_threshold`(0.3，"可食/值得付款") **不同**，见 config.py 注释。
        food_rich = (
            self.resources._grid[cur_flat]
            > FOOD_RICH_LEVEL * self.resources._capacity[cur_flat]
        )
        # S3 记忆 v2 写入开关（此段先于移动循环 ⇒ 独立读取，不复用移动段的 `_mv2_on`）
        _mv2_food_on = bool(getattr(
            self.config.info_structure, "memory_v2", False))
        if _mv2_food_on:
            # v2 下 **不再写/读** `_work_memory`（v2 取代 v1）。
            # 无 RNG 消费（除噪声档独立流）⇒ 主随机流形状不变（归因干净）。
            if food_rich.any():
                self._mem_v2_write(
                    np.flatnonzero(food_rich), cur_flat, int(self._tick))
        elif food_rich.any():
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
            self._sim_core.signal_emit(   # sparse:n/a（Rust 直写；范围锁已排除）
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
        # ── R247 饥饿调制（HM，实施规格 §一）：每 tick 一次，① ② 共用同一 `_h_norm` ──
        #   ① 走/停：`move_prob_eff = clip(move_prob × (1 + alpha·h_norm), 0, 1)`
        #     ⇒ 本处**就地调整** `move_prob`，下游两处 vanilla 走停（Rust 路径 + Python 路径）
        #       自然同式；subpos 路径另有 `stay_eff` 结构 ⇒ 单独调制（见下方）。
        #   ② 感知：`perc_eff = perc × (1 + beta·h_norm)`（三处同式：Python 参考循环 /
        #     Rust `movement.rs` / 向量化快路径；h_norm 由本处算好传入，保双路径同值）。
        #   🔴 关档 ⇒ 整块跳过（`_h_norm` 置空数组 = Rust 侧关档信号）⇒ 逐位等价（C7）。
        #   🔴 不新增 RNG 抽取：`rng.random(P)` 的**次数与形状**两档完全一致。
        _hmcfg = self.config.hunger_mod
        _hm_on = bool(_hmcfg.enabled)
        if _hm_on:
            _hunger = np.clip(
                1.0 - energy / max(1e-9, float(ocfg.max_energy)), 0.0, 1.0)
            _h_norm = (_hunger - float(_hmcfg.h_mid)) / max(
                1e-9, 1.0 - float(_hmcfg.h_mid))
            move_prob = np.clip(
                move_prob * (1.0 + float(_hmcfg.alpha) * _h_norm), 0.0, 1.0)
        else:
            _h_norm = np.empty(0, dtype=np.float64)
        _hm_beta = float(_hmcfg.beta)
        # D2-3 信息不对称：感知半径4/噪声/softmax 暂未下沉 Rust，启用时走 Python 路径
        _ifcfg = self.config.info_structure
        _d2_asym = _ifcfg.enabled and (_ifcfg.perception_radius == 4 or _ifcfg.perception_noise > 0 or _ifcfg.softmax_tau > 0)
        # ── R244 §二：气味场**消费端**（默认关；🔴 只读**候选格并集** ⇒ O(N×k)，不是 O(n_cells)）──
        #   口径：`_smell_arr[c] = Σ_ch w_ch·clip(S_ch(c)/S_max_ch, 0, 1)`（候选格并集**之外**恒 0）。
        #   **三处同式同位置**：Python 参考循环 / Rust 热核 / 向量化快路径
        #   （`score += perc × _smell_arr[nb]`）⇒ 无分岔（`_move_decide_batch` 收 `smell_arr` 参数）。
        _smell_use_on = bool(self.config.smell.use_in_move) and self.smell is not None
        _smell_arr = None
        if _smell_use_on:
            # 🔴 取数 = **按代缓存**：气味场只每 k tick 变一次 ⇒ 同一代内已读格的值可复用；
            #   每 tick 只需补算"本代还没算过的格"（个体移动导致的新候选）。
            #   代价：一次场更新内首次 ≈ O(候选格数)，其余 tick ≈ O(新增格) ⇒ 摊薄 ~0.3 ms/tick @N=2000。
            _cache = self._smell_scratch
            if _cache is None:
                _cache = self._smell_scratch = np.zeros(self.world.n_cells, dtype=np.float64)
                self._smell_stamp = np.zeros(self.world.n_cells, dtype=np.int32)
            _smell_arr = _cache
            _f = self._nb_table[self._flat[:P]].ravel()
            _f = _f[_f >= 0]                              # 主表极行是 -1 占位
            _pn = self._pole_nb.ravel()
            _f = np.concatenate([_f, _pn[_pn >= 0]])      # 极格候选（极带整行）
            if _f.size:
                _stamp = self._smell_stamp
                _gen = self._smell_gen
                _new = _f[_stamp[_f] != _gen]
                if _new.size:
                    _new = np.unique(_new)                # 去重（同格被多个体/多次引用）
                    _cache[_new] = self.smell.combined(_new)
                    _stamp[_new] = _gen
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
                food_ratio = self.resources._grid / self._cap_floor
                sig_present = (self.signals._marks > 0).astype(np.float64)
                # D0 修复：群居项量纲归一化 + 🔴 F1 修复（T3）：分母 = **每格实际邻居数**
                # （`_nb_norm`），不再是 stride(120)。注释此前写"0~8"但代码除 120 ⇒ 社交项
                # 被静默弱化 15×（C9 型缺陷）。构造级变更（digest 变，已同步测试）。
                # S0：occupancy 复用信号段预计算的 occ（移动之前位置未变）。
                smw = self.config.simulation.social_move_weight
                densities = np.divide(occ, self._nb_norm, dtype=np.float64) * smw
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
                    self._nb_table.reshape(-1), self._pole_nb.reshape(-1),
                    mi.astype(np.int64), rand_choice, move_cost_ind,
                    self.world.n_cells, self._nb_table.shape[1],
                    int(self.world.cols),
                    self._pole_top, self._pole_bottom,
                    1 if _ifc_mg.memory_gradient == "orientation" else 0,
                    float(_ifc_mg.memory_gradient_gain),
                    # R247 HM ②：逐个体 h_norm（空数组 = 关档）+ beta；
                    #   Rust 侧 `if hm_on { perc *= 1.0 + beta * h_norm[idx] }`（同式）。
                    _h_norm, _hm_beta,
                    # R244 §二：气味消费端（关档传 dummy 单元素数组 + 标志 0 ⇒ Rust 不索引）
                    1 if _smell_use_on else 0,
                    _smell_arr if _smell_use_on else self._smell_dummy,
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
            # ---- 13.4 波 1：亚格坐标 —— 移动语义反转（本线 = [所有者·天平]）------------
            # 老语义：`moved = u < move_prob` ⇒ **默认不动**、以概率移动。
            # 新语义（fish 00:30）：**默认走、以概率停**，且"停"由 **基因惰性 + 状态/信息**驱动：
            #     stay_prob_eff = clip(stay_base + 惰性项 + 本格余粮 + 收到信号, 0, stay_max)
            #   🔴 `stay_max < 1` 是**硬性反退化闸**（config 断言保证）：任何个体都有下限移动概率
            #      ⇒ 防"一群不动的生物"（R171 §十 的构造巧合 + 文献 Andersson 1981 的 pause-travel）。
            #   ⚠️ `stay_fear_k`（邻域有威胁 ⇒ 停）**本波不接**：它依赖"威胁感知"，而那是波 2 的
            #      `perception_span` 的事。参数已在 config，探针会报其读回值（防"死参数"）。
            #   ⚠️ RNG 形状（I3）：仍恰好消费 `rng.random(P)` **一次/个体** ⇒ 与关档同形。
            _subcfg = self.config.subpos
            _sub_on = bool(getattr(_subcfg, "enabled", False))
            if _sub_on:
                _own_cells = self._flat[:P]
                _fr_all = self.resources._grid / np.maximum(self.resources._capacity, 1e-9)
                _stay_gene = (1.0 - genes[:, Gene.MOVE_PROB]) * (1.0 - genes[:, Gene.ROOTING])
                _stay_eff = np.clip(
                    float(_subcfg.stay_base)
                    + _stay_gene * (1.0 - _stay)
                    + float(_subcfg.stay_food_k) * np.clip(_fr_all[_own_cells], 0.0, 1.0)
                    + float(_subcfg.stay_signal_k)
                    * (self.signals._marks[_own_cells] > 0).astype(np.float64),
                    0.0, float(_subcfg.stay_max),
                )
                if _hm_on:
                    # R247 HM ①（subpos 叠加语义）：本路径的"移动概率"= p_move = 1 − stay_eff
                    #   ⇒ `p_move_eff = clip(p_move × (1 + alpha·h_norm), 0, 1)`；
                    #   判定保持 `u >= 1 − p_move_eff`（与 `u >= stay` 同翻转方向）。
                    #   ⚠️ 关档必须走**原式**：`1 − (1 − stay) ≠ stay`（浮点尾差）⇒ 两分支不合并。
                    _p_move_eff = np.clip(
                        (1.0 - _stay_eff) * (1.0 + float(_hmcfg.alpha) * _h_norm),
                        0.0, 1.0)
                    moved = self.rng.random(P) >= (1.0 - _p_move_eff)
                else:
                    moved = self.rng.random(P) >= _stay_eff
            if not _sub_on:
                moved = self.rng.random(P) < move_prob
            moved &= energy >= move_cost_ind  # 付得起才走
            Nm = int(moved.sum())
            # ── R146/R149 L2：人口口径读数（B4；本段 = [本地开发] 线）─────────────
            # 只在 l2_dash 开启时累加 ⇒ **关档 `l2_probe()` 返回 None（未适用），不是 0**
            # （R120/§五.12 口径铁律）。与"谁在动"无关，故放在 `if Nm:` **之外**。
            _l2_on = bool(self.config.simulation.l2_dash)
            if _l2_on:
                # S0（2026-09-25）：惰性构造 2 跳环表（本段之后的 far 读数/盲选都要用）
                self._ensure_far_tables()
                _gate = float(self.config.predation.attack_gene_gate)
                self._pop_n += int(P)
                self._g16_le_gate_n += int(np.count_nonzero(
                    genes[:P, Gene.AGGRESSION] <= _gate))
                self._inelig_pop_n += int(np.count_nonzero(
                    self._far_len[self._flat[:P]] == 0))
            if Nm:
                mi = np.flatnonzero(moved)
                food_ratio = self.resources._grid / self._cap_floor
                sig_present = (self.signals._marks > 0).astype(np.float64)
                # D0 修复：群居项量纲归一化 + 🔴 F1 修复（T3）：分母 = **每格实际邻居数**
                # （`_nb_norm`），不再是 stride(120)。同上（Python 移动分支）。
                # S0：occupancy 复用信号段预计算的 occ（移动之前位置未变）。
                smw = self.config.simulation.social_move_weight
                densities = np.divide(occ, self._nb_norm, dtype=np.float64) * smw
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
                # ── R146/R149 L1（本段 = [所有者·天平] 线 = B1）：每 tick 预计算 ──────
                # 信息自洽（派工单 §3.2）：只用**1 圈内可直读**的量。`occ` = 移动前占用
                # （:1425 已算好，与信号段共用，0 额外成本）；两个场是**格域聚合** ⇒ 逐候选
                # 直接索引即得，**不需要**任何"候选格的邻域聚合"（那要距离-2 信息 ⇒ 越界）。
                # 关档：整段不执行（无 RNG、无状态改变）⇒ H1 逐位等价。
                _sim0 = self.config.simulation
                _l1_seek_on = bool(_sim0.l1_seek)
                _l1_fear_on = bool(_sim0.l1_fear)
                _l1_on = _l1_seek_on or _l1_fear_on
                # S3 血条恐惧项（设计稿 §2.4 项 7）也需**威胁场**（载体 = 血条），
                # 与 L1 恐惧项独立 ⇒ 任一生效时都要算 `_agg_field`（格域聚合，零 RNG）。
                _fearh_on = (_wound_on
                            and float(getattr(_cwc_pred, "w_fear_health", 0.5)) > 0.0)  # R208 §三：兜底==config 默认
                if _l1_on or _fearh_on:
                    _gate0 = float(self.config.predation.attack_gene_gate)  # 引用同一常量，不抄字面量
                    if _l1_seek_on:
                        if str(_sim0.l1_prey_mode) == "any":
                            # E′ 归因臂：猎物代理 = **任意占格者**（去掉低 g16 过滤）——
                            # 唯一能回答"分化是不是那个过滤造出来的"的对照（不进合取）
                            _seek_field = occ.astype(np.float64)
                        else:
                            _low = genes[:P, Gene.AGGRESSION] <= _gate0
                            _seek_field = np.bincount(
                                self._flat[:P][_low], minlength=self.world.n_cells
                            ).astype(np.float64)
                    if _l1_fear_on or _fearh_on:
                        # 威胁强度：只取**超过门槛**的部分（弱 g16 个体不构成威胁）
                        _thr = np.maximum(0.0, genes[:P, Gene.AGGRESSION] - _gate0)
                        _agg_field = np.bincount(
                            self._flat[:P], weights=_thr, minlength=self.world.n_cells
                        )
                ifcfg3 = self.config.info_structure
                d2_asym = ifcfg3.enabled and ifcfg3.perception_radius == 4
                d2_noise = ifcfg3.enabled and ifcfg3.perception_noise > 0
                d2_softmax = ifcfg3.enabled and ifcfg3.softmax_tau > 0
                # 13.4 波 2B（T3）：span=2 开关（只 Python 路径，H3 已拦 Rust）+ cap 开关
                _span2_on = (int(getattr(self.config.simulation,
                                         "perception_span", 1)) == 2)
                _cap_on = bool(getattr(self.config.simulation,
                                       "cell_occupancy_cap_enabled", False))
                _cap_val = int(getattr(self.config.simulation,
                                       "cell_occupancy_cap", 3))
                # B1：R2 声誉权重（0=关闭 → sig_weight 恒 0.5，与旧版逐位一致）
                rep_w = ifcfg3.reputation_weight if ifcfg3.enabled else 0.0
                # A′ 记忆朝向梯度（2026-09-19）：**不**受 `enabled` 门控 —— 与 ⑥ 探针同规格，
                # 必须能**单独**开关，否则 A′ 会被学习瓶颈/码本/softmax 等一堆 D2 机制污染
                # （⇒ 不再是单变量实验）。`none` 下走**原式**，逐位等价。
                mem_grad_on = (ifcfg3.memory_gradient == "orientation")
                mem_grad_gain = float(ifcfg3.memory_gradient_gain)
                # S3 记忆 v2（egocentric）：与 A′ 同规格（不受 `enabled` 门控，单变量）。
                #   构造期 assert 已保证 v2 ⇒ memory_gradient=="orientation" ⇒ mem_grad_on。
                _mv2_on = bool(getattr(ifcfg3, "memory_v2", False))
                # 🔴 P2②（[本地开发·性能线]）：vanilla 快路径门 —— 个体级可选机制**全关**
                #    时参考循环退化为「纯打分 + argmax/平局」⇒ 允许整块向量化（逐位等价，
                #    见 `_move_decide_batch`）。任一机制开启 ⇒ 一律走参考循环（不做半接线，
                #    A1 教训）。`_batch_move_on` 是**对拍/调试开关**（默认 True）。
                # ⚠️ R247 HM **不进**排除表：其 ② 已在 `_move_decide_batch` 内逐行接线
                #    （`h_norm`/`hm_beta` 透传 + 变异对拍测试覆盖）⇒ 开 HM 不失快路径。
                _batch_ok = (
                    self._batch_move_on
                    and not (_span2_on or d2_asym or d2_noise or d2_softmax
                             or mem_grad_on or _l1_on or _fearh_on or _cap_on
                             or _l2_on or self._mig_on or self._ars_on
                             or _mv2_on)
                    and self.world.rows >= 3
                    and self.world.cols >= 2
                )
                if _batch_ok:
                    targets = self._move_decide_batch(
                        mi, food_ratio, sig_present, densities, rand_choice, rep_w,
                        _h_norm, _hm_beta,
                        _smell_arr if _smell_use_on else None)
                # 快路径命中 ⇒ 参考循环迭代集为空（**不重排**下方 280 行参考实现）。
                for i, idx in enumerate(() if _batch_ok else mi):
                    # D2-3 信息不对称：感知半径4 = Von Neumann（上/左/右/下），各向同性。
                    # ⚠️ 禁用 nb[:4]：8 邻列序以 [上左,上,上右,左] 打头，取"前4个"
                    # 实际只保留"北+西" → 个体永不能向南/东移动，种群被单向驱赶至极区、
                    # 招募崩塌（见 docs/决策与评审/A4-崩溃溯源报告-20260912.md）。
                    # 13.4 波 2B（T3）：span=2 ⇒ 候选 = ring1+ring2（`_span_table`）；
                    #   F3 闸：该格 ring1+2 > cap ⇒ 降级为 1 圈（`_span_eff2` 为 False）。
                    #   span=1（默认）⇒ **走原路径**（逐位等价）。
                    _cell_i = int(self._flat[idx])
                    # ── 13.8 日历—罗盘式定向迁徙：开关在**个体循环内**取一次 ─────────
                    # （`_cell_i` 与 `nb` 都是本循环的局部量 ⇒ 项必须在此求值）
                    # 关档 ⇒ `_mig_on=False` ⇒ 整段不执行（无 RNG、无状态改变）
                    # ⇒ **逐位等价**（C7）。
                    _mig_on = bool(self._mig_on)
                    _mig_gain = self._mig_gain
                    _mig_min_abs = self._mig_min_abs
                    _ars_on = self._ars_on
                    if _span2_on and self._span_eff2[_cell_i]:
                        _row = self._span_table[_cell_i]
                        nb = _row[_row >= 0]
                    elif d2_asym:
                        nb = self.world.neighbors_von_neumann(_cell_i)
                    else:
                        nb = np.asarray(self.world.neighbors(_cell_i))
                    if len(nb) == 1:
                        targets[i] = nb[0]
                        continue
                    perc = genes[idx, Gene.PERCEPTION]
                    if _hm_on:
                        # R247 HM ②：**只调制 perc 赋值行** ⇒ 食物/信号/记忆/解读项随 perc 同乘、
                        #   `soc × 密度` 项不受影响（"饿 ⇒ 更专注找吃的/信号，社交项相对被压"）。
                        #   ⚠️ 与 Rust / 批量路径**逐字同式**：`1.0 + beta * h_norm[idx]`。
                        perc = perc * (1.0 + _hm_beta * _h_norm[idx])
                    soc = (genes[idx, Gene.SOCIABILITY] - 0.5) * 2.0
                    # D2-3 感知噪声：食物/信号感知加高斯噪声（F-D2 修复 B′：走每引擎独立的
                    # `_d2_rng`，种子/算法同旧全局播种 ⇒ 单引擎逐位不变、多引擎不再互相污染）
                    fr = food_ratio[nb].copy()
                    sp = sig_present[nb].copy()
                    if d2_noise:
                        fr += self._d2_rng.normal(0, ifcfg3.perception_noise, size=len(nb))
                        sp += self._d2_rng.normal(0, ifcfg3.perception_noise, size=len(nb))
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
                    if _mv2_on:
                        # S3 记忆 v2：自我参照系打分（取代 v1 全部记忆项；v2 下不再读
                        #   `_work_memory`）。gain（精/粗）已在函数内应用 ⇒ 这里只乘 perc。
                        _hd_now = int(self._heading[idx])
                        g = self._memory_egocentric_cos(
                            idx, int(self._flat[idx]), nb, _hd_now)
                        if float(np.abs(g).max()) > 1e-9:
                            score = score + perc * g
                    elif mem_grad_on:
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
                        # 整数成员判断：广播比较替代 np.isin（valid_mem≤4，结果 bool
                        #   逐位一致；profile 移动段热点）。`0.3` 字面量严格不动。
                        mem_in_nb = (nb[:, None] == valid_mem[None, :]).any(axis=1)
                        score = score + 0.3 * perc * mem_in_nb.astype(np.float64)
                    nb_sigs = self.signals._marks[nb]
                    if (nb_sigs > 0).any():
                        interp = np.array(
                            [self._interpret[idx, int(s)] if s > 0 else 0.0 for s in nb_sigs],
                            dtype=np.float64,
                        )
                        score = score + 0.4 * perc * interp
                    # ── R244 §二：气味场消费端（默认关 ⇒ `_smell_arr is None` ⇒ 整段不执行）──
                    # 🔴 与 Rust **同式同位置**（在信号项之后、L1 项之前）⇒ 双路径逐位一致。
                    if _smell_use_on:
                        score = score + perc * _smell_arr[nb]
                    # ── R146/R149 L1 两项（**逐候选**；本段 = [所有者·天平] 线 = B1）─────
                    # 🔴 R148-1 认账：若某项对**所有候选**取同值 ⇒ `argmax` 逐位不变、
                    #    `softmax` 数学无效应（`[实测]` 偏差 9.6e-15 / 20 万次 0 翻转）
                    #    ⇒ 等于"接了却一行行为没改"。故两项都按**候选格**求值，并用
                    #    `_seek_flat_n`/`_fear_flat_n` 计"跨候选取同值"的个体（段一必报）。
                    if _l1_on:
                        _g16i = float(genes[idx, Gene.AGGRESSION])
                        self._l1_dec_n += 1
                        if _l1_seek_on:
                            # L1a 追猎：朝**低攻击性个体（猎物代理）**密集的候选格走
                            _sk = (_sim0.w_seek_max * float(genes[idx, Gene.DIET])
                                   * perc * _g16i) * np.minimum(_seek_field[nb], 1.0)
                            score = score + _sk
                            self._seek_term_sum += float(_sk.sum())
                            self._seek_term_n += int(_sk.size)
                            if float(_sk.max() - _sk.min()) < 1e-12:
                                self._seek_flat_n += 1
                                # 细分：**恒为 0** = 自熄（邻域无猎物代理），与"取同值但非零"是
                                # 两种不同诊断（前者对应预注册的「猎物池枯竭」结局）⇒ 分开记
                                if float(np.abs(_sk).max()) < 1e-12:
                                    self._seek_zero_n += 1
                        if _l1_fear_on:
                            # L1b 恐惧：**背向**威胁（复用 `_memory_orientation_cos`，与 Rust 逐字同式）
                            # ⚠️ danger == nb（每个候选格都有威胁）⇒ `cos ≡ 1` ⇒ **逐位 no-op**
                            #    ⇒ 由 `_fear_flat_n` 计数暴露（R149-4 的反退化断言落点）
                            _danger = nb[_agg_field[nb] > 0.0]
                            if len(_danger) > 0:
                                _cos = self._memory_orientation_cos(
                                    int(self._flat[idx]), nb, _danger)
                                _fe = (_sim0.w_fear * perc * (1.0 - _g16i)) * _cos
                                score = score - _fe
                                self._fear_term_sum -= float(_fe.sum())   # 记**施加之量**（负）
                                self._fear_term_n += int(_fe.size)
                                self._fear_ind_n += 1
                                if float(_fe.max() - _fe.min()) < 1e-12:
                                    self._fear_flat_n += 1
                    # ── S3 血条恐惧项（设计稿 §2.4 项 7；`wound_enabled` + `w_fear_health>0`）──
                    # 与 L1 恐惧项**独立**（载体 = 血条而非 g16）：`−w_fear_health × perc ×
                    # (1−health) × cos(候选格 ← 威胁方向)`。低血条 ⇒ 更恐惧（能力导向）。
                    # 🔴 确定性数值（零 RNG）⇒ 关档零轨迹影响。
                    if _wound_on and float(getattr(_cwc_pred, "w_fear_health", 0.5)) > 0.0:  # R208 §三：兜底==config 默认
                        self._fearh_dec_n += 1
                        _danger_h = nb[_agg_field[nb] > 0.0]
                        if len(_danger_h) > 0:
                            _cos_h = self._memory_orientation_cos(
                                int(self._flat[idx]), nb, _danger_h)
                            # 🔴 13.4 波 3（T4，fish 01:20 批准）：恐惧**带门槛连续** ——
                            #    `1 − health < 0.3` 才触发（受轻伤不恐惧，重伤才怕）。
                            #    `_wound_fear_threshold` 默认 0.3（config 字段）。
                            _inj = 1.0 - float(self._health[idx])
                            if _inj >= float(getattr(_cwc_pred, "wound_fear_threshold", 0.3)):
                                _fh = (float(_cwc_pred.w_fear_health) * perc * _inj) * _cos_h
                                score = score - _fh
                                if float(_fh.max() - _fh.min()) < 1e-12:
                                    self._fearh_flat_n += 1
                    # ── 13.8 日历—罗盘式定向迁徙（本线 = [云端·开发] 云归）──────
                    # 机制（设计稿 §3.4，**一行项**）：候选 n 的分数加
                    #   score(n) += gain · g23_i · A_i(t) · Δ|φ|(n)
                    # 其中 A_i = 个体**所在格**的日长异常（"日历"：本地是不是夏天），
                    #     Δ|φ|(n) = |φ_n| − |φ_self|（"罗盘"：候选格比本格离极点**更远**还是更近）。
                    # 🔴 语义：**本地是夏天**（A>0）⇒ 想往**离极点更远**（|φ|↑，即过夏区）走；
                    #    **本地是冬天**（A<0）⇒ 反向 ⇒ 自动成为"春北秋南"的往返迁徙
                    #    （Gwinner 1986/1996 内源年节律；Berthold 1996/2001 光周期是跨类群
                    #    主导触发且是**超前信号**）。光周期是**日历**而非食物梯度 ⇒ 远离噪音
                    #    （R195/R196 已证食物带 SNR 致命：0.0015°/tick vs 个体 0.025°/tick）。
                    # 🔴 用 |φ| 而非带符号 φ ⇒ **南北半球自动都正确**，无需半球分支。
                    # 🔴 只用**1 圈内可直读**的量（信息自洽，§3.2）：本格 A（`_pp_anom[cell]`）
                    #    + 候选格与本格的 |φ|（`_lat_abs` 逐格预计算）⇒ 零距离-2 信息。
                    # 🔴 关档（_mig_on=False）⇒ 整段不执行（无 RNG、无状态改变）⇒ 逐位等价（C7）。
                    if _mig_on:
                        self._mig_dec_n += 1
                        _A = float(self._pp_anom[_cell_i])
                        if abs(_A) <= _mig_min_abs:
                            # |A| 门槛：本地日长≈半天（换季窗口/赤道）⇒ 日历信号无信息 ⇒
                            # 跳过是**有意设计**（不是 bug）⇒ 单独计数以便诊断（非退化计数）。
                            self._mig_skip_n += 1
                        else:
                            _dabs = self._lat_abs[nb] - self._lat_abs[_cell_i]
                            _m = ((_mig_gain * float(genes[idx, Gene.MIGRATE_BIAS]))
                                  * _A * _dabs)
                            score = score + _m
                            self._mig_term_sum += float(np.abs(_m).sum())
                            self._mig_term_n += int(_m.size)
                            if float(np.abs(_m).max()) < 1e-12:
                                # 细分：**恒为 0** = 自熄（A≠0 但 Δ|φ|≡0 ⇒ 项无信息）
                                self._mig_zero_n += 1
                            if float(_m.max() - _m.min()) < 1e-12:
                                # 🔴 R148-1 反退化：对**所有候选**取同值 ⇒ argmax 逐位不变
                                #    ⇒ "接了却一行行为没改"。同纬候选天然贡献 ⇒ 报比例不判死。
                                self._mig_flat_n += 1
                    # ── 14.9 ARS：赶路惯性（仅 extensive；极区/非 8 邻豁免）────────────
                    # cos 矩阵对反向天然 = −1 ⇒ **不回头软惩罚内建**（食物项可覆盖 ⇒
                    # 满足 fish「也可以往回走、评估什么价值大」）。
                    if _ars_on and bool(self._ars_extensive[idx]):
                        self._ars_dec_n += 1
                        self._giveup_ct[idx] += 1
                        if self._giveup_ct[idx] > self._ars_giveup:
                            # 失望：连走 giveup tick 没咬到 ⇒ **确定性右转 90°**（+2 槽位）
                            # 🔴 零 RNG（保住 L1 块"不新增随机抽取"的契约，test_l1_terms）；
                            #    且这是 fish 要的「轻微转向」——不是随机乱换方向。
                            self._heading[idx] = (self._heading[idx] + 2) % 8
                            self._giveup_ct[idx] = 0
                            self._ars_rerand_n += 1
                        _hd = int(self._heading[idx])
                        _lat_deg = float(self._lat_abs[_cell_i]) * 57.29577951308232
                        # 🔴 候选可能是 4 邻（D2 Von Neumann，默认 radius=4）或 8 邻（Moore）
                        #    ⇒ cos 行按候选子集取；极点（120 邻）/span=2（16 邻）豁免。
                        if (_lat_deg <= self._ars_lat_exempt and 0 <= _hd < 8
                                and len(nb) in (4, 8)):
                            if len(nb) == 8:
                                _cos_row = self._ars_cos[_hd, :]
                            else:
                                _cos_row = self._ars_cos[_hd, :][
                                    list(self.world._VON_NEUMANN_IDX)
                                ]
                            _pers_i = float(genes[idx, Gene.PERSISTENCE])
                            _m = (self._ars_gain * _pers_i) * _cos_row
                            score = score + _m
                            self._ars_inertia_sum += float(np.abs(_m).sum())
                            if float(_m.max() - _m.min()) < 1e-12:
                                self._ars_flat_n += 1
                    # ── 13.4 波 2B（T3）：单格个体上限（score 层剔除满格）──────
                    # 🔴 三条硬约束（任务书 T3 / 设计稿 §2.4）：
                    #   1. **只约束"进入"，不约束"留在"**：候选 = 本格（steps=0）不受限。
                    #      ⚠️ 本段 `nb` 不含自身（移动候选从邻居取）⇒ 天然满足。
                    #   2. 满格候选 score = −inf；**若全部候选 −inf ⇒ 留本格**（兜底，
                    #      计 `_cap_stay_n`，不收移动费）⇒ cap 永不因"平局回退"被破坏。
                    #   3. `occ` = 移动前占用（信号段已算）⇒ 零额外成本。
                    if _cap_on:
                        _crowded = occ[nb] >= _cap_val
                        if _crowded.any():
                            score = np.where(_crowded, -np.inf, score)
                            self._cap_blocked_n += 1
                        if not np.isfinite(score).any():
                            targets[i] = _cell_i       # 全满 ⇒ 留本格（兜底）
                            self._cap_stay_n += 1
                            continue
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
                    # ── 14.9 ARS：记录本步方向（供下一 tick 的惯性项用）──────────────
                    # 🔴 heading 统一存 **Moore 槽位（0–7）**：4 邻模式下把 nb 内下标
                    #    经 `_VON_NEUMANN_IDX` 映射回 Moore 槽位（两模式共用一个 cos 矩阵）。
                    # S3 记忆 v2：`_mv2_on` 时 **ARS 关档也强制维护** heading（v2 的自我参照
                    #    系需要持续朝向基准），并同步做记忆旋转/距离更新（`_mem_v2_step_one`）。
                    if (_ars_on or _mv2_on) and len(nb) in (4, 8):
                        _same = np.flatnonzero(nb == targets[i])
                        if len(_same) == 1:
                            _hd_new = (
                                int(_same[0])
                                if len(nb) == 8
                                else int(self.world._VON_NEUMANN_IDX[int(_same[0])])
                            )
                            if _mv2_on:
                                _hd_old = int(self._heading[idx])
                                self._heading[idx] = _hd_new
                                # 距离步长：本 tick 换格 1（subpos 亚格位移由 4.6 段
                                #    `steps` 变量给实际格数；此处取保守 1.0 即可满足
                                #   "只增不减 + 降级"语义 —— 换格即消耗，见规格 §二）。
                                self._mem_v2_step_one(idx, _hd_old, _hd_new, 1.0)
                            elif _ars_on:
                                self._heading[idx] = _hd_new
                    # ── R146/R149 L2 第二段：冲刺者改走 2 格（**盲选**）─────────────
                    # 位置：**在 score/softmax 之后** ⇒ `u = self.rng.random()` 仍按个体消费
                    # ⇒ 每 tick 随机抽取数与关档**逐字一致**（H2/H1 的形状要求）。
                    # 代价：冲刺者白算一次 score（换取随机流形状不变 —— 值得）。
                    # 🔴 13.4 波 2B（T3）去两段式盲选（R148 偏离 2 的前提消失）：
                    #   span=2 ⇒ ring2 已纳入候选、按 score 选过 ⇒ **跳过盲选覆盖**
                    #   （保留 score 选出的目标）；span=1 ⇒ 维持旧盲选（逐位等价）。
                    # 🔴 cap 开（任务书 T3 §5"L2 冲刺目标同理排除"）：盲选 far 候选
                    #   **不会剔除满格** ⇒ cap 开时同样跳过盲选覆盖（保留 score 目标，
                    #   score 层已剔除满格）⇒ 冲刺目标永不落满格（本格兜底不受限）。
                    if _l2_on and dash[i] and not _span2_on and not _cap_on:
                        _c = int(self._flat[idx])
                        _fl = int(self._far_len[_c])
                        targets[i] = int(self._far_cells[
                            int(self._far_off[_c]) + int(rand_choice[i] // 100) % _fl])
                if _sub_on:
                    # ── 亚格位移：方向沿用 score 选出的方向，步长由 g18×年龄**确定性**给出 ──
                    # 🔴 速度必须确定性（I3）：每 tick 抽取数与关档一致。`rand_choice` 已被 L2
                    #    的位域契约占用（`%100` / `//100`），再抽一次就破坏随机流形状。
                    _cr, _cc = self.world.flat_to_rc(self._flat[mi])
                    _tr, _tc = self.world.flat_to_rc(targets)
                    _drow = np.sign(_tr - _cr).astype(np.int64)
                    # 经度差取**最短路**（环绕）后再取符号：跨 0/cols 边界不能直接相减
                    _dcw = (((_tc - _cc) + self.world.cols // 2) % self.world.cols
                            - self.world.cols // 2)
                    _dcol = np.sign(_dcw).astype(np.int64)
                    _ocs = self.config.organisms
                    _ls = self._lifespan(genes[mi, Gene.LIFE_GENE])
                    _agef = self._age[mi].astype(np.float64)
                    _af = np.ones_like(_agef)
                    _af[_agef < _ocs.maturity_fraction * _ls] = _ocs.young_mob_mult
                    _af[_agef >= _ocs.senile_fraction * _ls] = _ocs.old_mob_mult
                    # 🔴 纬度调速（fish 00:30「极区移速更慢」）：
                    #    speed_cap = speed_max × (lat_floor + (1−lat_floor)·cos φ)
                    _coslat = np.cos(self.world.latitude_of(_cr))
                    _cap = float(_subcfg.speed_max) * (
                        float(_subcfg.lat_floor)
                        + (1.0 - float(_subcfg.lat_floor)) * np.maximum(_coslat, 0.0))
                    _spd = np.clip(genes[mi, Gene.DEFENSE] * _af
                                   * float(_subcfg.speed_gain), 0.0, _cap)
                    _subdiv = int(_subcfg.subdiv)
                    _st = _sub_steps(_spd, _subdiv)
                    # 纯能量门槛（沿用 L2 口径）：能量不够 ⇒ 走 0 步（原地）
                    _st = np.where(
                        energy[mi] >= float(_subcfg.min_energy_frac) * _ocs.max_energy,
                        _st, np.int64(0))
                    _sr2, _sc2 = _sub_advance(self._sub_r[mi], self._sub_c[mi],
                                              _drow, _dcol, _st, _subdiv, self.world)
                    self._sub_r[mi] = _sr2
                    self._sub_c[mi] = _sc2
                    _nf = _sub_flat(_sr2, _sc2, _subdiv, self.world)
                    # 读数（B4）：🔴 主判据 = **真正换格**（mean_flat_moves），**不是 steps**
                    # —— 位移 0.75 格在 steps 上有值、在 flat 上等于没动（设计稿 §2.2）
                    self._run_mover_sub_n += int(len(mi))
                    self._run_flat_move_n += int(np.count_nonzero(_nf != self._flat[mi]))
                    self._run_slow_n += int(np.count_nonzero(_st == 0))
                    self._steps_hist += np.bincount(
                        np.clip(_st, 0, _subdiv), minlength=_subdiv + 1)[:_subdiv + 1]
                    self._flat[mi] = _nf
                    # 计费：按**实际步数**比例（走 0 步 ⇒ 不扣费 ⇒ "停"真的省钱）
                    _cost = move_cost_ind[mi] * (_st.astype(np.float64) / float(_subdiv))
                    energy[mi] -= _cost
                    self._ec_add(EC_MOVE, mi, _cost)
                else:
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
            # S3 饥饿激进项（设计稿 §2.2 项 7；`wound_enabled` + `need_aggression_k>0`）：
            #   能量低 ⇒ 更激进（主动攻击概率↑）—— 与血条恐惧**方向相反** ⇒ 二分预测
            #   （能力导向 vs 资产保护）。k=0（D 臂）⇒ 退化为原式（逐位不变）。
            _need_k = float(getattr(_cwc_pred, "need_aggression_k", 0.5)) if _wound_on else 0.0  # R208 §三：兜底==config 默认
            attack_prob = attack_gene * pcfg.attack_prob_coef * hunger * (1.0 + _need_k * hunger)
            # 🔴 13.4 波 2B（T3）"看见才出手"：span=2 ⇒ 出手须**视野内有猎物**。
            #   猎物代理 = 低 g16（≤ attack_gene_gate）个体；视野 = span 圈（`_span_table`
            #   聚合，降级格用 1 圈）。⚠️ 只在 span=2 生效（span=1 逐位等价）。
            #   `_vis` 是格域 bincount ⇒ 零新增 RNG 抽取（H2 保持：`rng.random(P)` 仍全量消费）。
            if _span2_on:
                # 视野内猎物场（格域）：每个低 g16 个体把 +1 加到"能看见它的格"
                # （= 它所在格的 span 圈邻居；降级格用 1 圈）。零新增 RNG。
                _seen = np.zeros(self.world.n_cells, dtype=np.int64)
                _low16 = genes[:P, Gene.AGGRESSION] <= float(pcfg.attack_gene_gate)
                _low_cells = self._flat[:P][_low16]
                for c in np.unique(_low_cells):
                    if self._span_eff2[c]:
                        r = self._span_table[c]
                        nb_c = r[r >= 0]
                    else:
                        nb_c = np.asarray(self.world.neighbors(int(c)))
                    np.add.at(_seen, nb_c, int((_low_cells == c).sum()))
                _vis = _seen
            if _span2_on:
                attackers = np.flatnonzero(
                    (attack_gene > pcfg.attack_gene_gate)
                    & (self.rng.random(P) < attack_prob)
                    & (_vis[self._flat[:P]] > 0)
                )
            else:
                attackers = np.flatnonzero(
                    (attack_gene > pcfg.attack_gene_gate)
                    & (self.rng.random(P) < attack_prob)
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
                self._nb_table.reshape(-1), self._pole_nb.reshape(-1),
                self.world.n_cells, self._nb_table.shape[1],
                int(self.world.cols), self._pole_top, self._pole_bottom,
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
                    # F-D2 修复 B′：改绑每引擎独立的 `_d2_rng`（不计入 draws，同修复前口径）
                    self._d2_rng.shuffle(cell_idx)
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
                        # 🔴 13.4 波 3（T4，fish 00:20 裁定，与 S2 相反）：血条语义反转
                        #   **成功 ⇒ 一击毙命**（猎物能量进尸体，**不直接转移给攻击者**）；
                        #   **失败 ⇒ 扣猎物血条**（致伤，health≤0 才死）。
                        # 🔴 守恒（红线）：击杀能量进尸体受 `corpse_enabled` 门控 ——
                        #    corpse 开 ⇒ 由 7.5 投尸（能量冻结为尸体，可审计）；
                        #    corpse 关 ⇒ 沿用旧 `transfer_ratio` 转移（不消失，保持
                        #    能量守恒审计通过）。攻击者**总是**付出手成本（循环上已扣）。
                        predation_mask[prey] = True
                        self._ec_prey_e_sum += float(energy[prey])
                        self._ec_prey_kill_n += 1
                        self._duel["kills"] += 1
                        if not _corpse_on_tick:
                            # corpse 关 ⇒ 旧转移（守恒兜底；T4 反转的"进尸体"不可用）
                            _tr = float(energy[prey]) * pcfg.transfer_ratio
                            energy[idx] += _tr
                            _pred_i.append(idx)
                            _pred_amt.append(_tr)
                            stomach[idx] = np.minimum(
                                stomach[idx] + stomach[prey] * pcfg.stomach_transfer,
                                ocfg.max_energy / max(1e-9, ocfg.eat_efficiency),
                            )
                        stomach[prey] = 0.0
                    else:
                        # 失败 ⇒ 扣猎物血条（致伤）；health ≤ 0 ⇒ 致死（能量进尸体）
                        if _wound_on:
                            _delta = float(_cwc_pred.wound_base) * (
                                0.5 + 0.5 * float(attack_gene[idx])
                            )
                            self._health[prey] -= _delta
                            self._wound_n += 1
                            self._duel["wounds"] += 1
                            if self._health[prey] > 0.0:
                                continue   # 未死：猎物保留资源，下轮可再被咬
                            # health ≤ 0 ⇒ 致死（predation_mask 置 True，能量进尸体）
                            predation_mask[prey] = True
                            self._ec_prey_e_sum += float(energy[prey])
                            self._ec_prey_kill_n += 1
                            self._duel["kills"] += 1
                            if not _corpse_on_tick:
                                _tr = float(energy[prey]) * pcfg.transfer_ratio
                                energy[idx] += _tr
                                _pred_i.append(idx)
                                _pred_amt.append(_tr)
                                stomach[idx] = np.minimum(
                                    stomach[idx] + stomach[prey] * pcfg.stomach_transfer,
                                    ocfg.max_energy / max(1e-9, ocfg.eat_efficiency),
                                )
                            stomach[prey] = 0.0
                        else:
                            # 旧路径（wound 关）：失败 = 无事（猎物逃过）
                            pass
                # R141 P0：捕食侧两条通道（出手成本 / 掠得能量）——循环外统一记账
                if _atk_amt:
                    self._ec_add(EC_ATTACK, _atk_i, _atk_amt)
                if _pred_amt:
                    self._ec_add(EC_PRED, _pred_i, _pred_amt)
            # ── Python 分步：年龄推进（Rust 路径已由 stage2 就地 +1）──
            self._age[:P] += 1
            age_f = self._age[:P].astype(np.float64)
            # ── Python 分步：文化学习 ──
            # S2（2026-09-26）：逐个体 Python 循环 → 批式向量化，语义逐位不变。
            # 实现与三条不变量见 `_culture_learn_python`。
            juvenile = age_f < maturity_age
            if juvenile.any():
                self._culture_learn_python(
                    np.flatnonzero(juvenile), age_f >= maturity_age)
            # ── D2-4 Steels 对齐：同格相遇概率性解读表对齐 ──
            ifcfg_sa = self.config.info_structure
            if ifcfg_sa.enabled and ifcfg_sa.steels_alignment:
                occ_py = np.bincount(self._flat[:P], minlength=self.world.n_cells)
                crowded = np.flatnonzero(occ_py > 1)
                for cell in crowded:
                    cell_idx = np.flatnonzero(self._flat[:P] == cell)
                    if len(cell_idx) < 2:
                        continue
                    # 随机配对（F-D2 修复 B′：独立 `_d2_rng`，同修复前"不计入 draws"口径）
                    self._d2_rng.shuffle(cell_idx)
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

        # 7) 死亡判定：饿死 → 老死 → 被捕食
        # 注意：统一用 Python 计算（不用 Rust 的 out_starved/out_expired），
        # 因为捕食（步骤 5.5）在 Rust stage2 之后才执行，会改变 energy。
        # 🔴 13.5（S2 双阈值；默认 0.0 ⇒ 与旧版逐位一致）：
        #   * **力竭** `energy < exhaust_frac×max_energy` ⇒ 死（**无论胃里有没有食**）
        #   * **饿死** `energy < starve_frac×max_energy` **且胃空** ⇒ 死
        #   为什么要两个（fish 09-23）：只留"能量低且胃空"会出现"抱着一肚子食物饿不死"
        #   的僵尸态 ⇒ 必须有一个"无论胃"的更低阈值兜底。字段见 `OrganismConfig`。
        _exh_frac = float(getattr(ocfg, "exhaust_frac", 0.0) or 0.0)
        _stv_frac = float(getattr(ocfg, "starve_frac", 0.0) or 0.0)
        if _exh_frac > 0.0 or _stv_frac > 0.0:
            _z = np.zeros_like(energy, dtype=bool)
            _exh = (energy < _exh_frac * ocfg.max_energy) if _exh_frac > 0.0 else _z
            _stv = ((energy < _stv_frac * ocfg.max_energy)
                    & (self._stomach[:P] <= 1e-9)) if _stv_frac > 0.0 else _z
            starved = _exh | _stv
        else:
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

        # 7.5) S2 死亡投放尸体（设计稿 §5.3 项 2；`corpse_enabled`）
        # 🔴 三处死因**共用一个函数**（老死自然很轻：老死个体能量已耗尽 ⇒ 尸体能量≈0，
        #    设计稿 §一"老死也留尸体、能量低"用"剩余能量 ×0.9"自然实现，**不做特判**）。
        # 位置：死亡判定（7）之后、死亡压缩（9）之前 —— dead 掩码此时仍对应完整 P 槽位。
        # 🔴 确定性数值（无 RNG 消费）⇒ 关档零开销、零轨迹影响。
        _cwc = getattr(self.config, "corpse_wound", None)
        if bool(getattr(_cwc, "corpse_enabled", False)) and dead.any():
            self._deposit_corpse(dead, energy)

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
                # R165 0-2：Rust 繁殖路径的尸源归因**恒零** —— 该路径只在
                # `use_sim_core=True` 时走，而 R145 H3 守卫已保证"开 corpse 时
                # `use_sim_core=True` 直接硬报错" ⇒ 此路径下 `_stomach_scav` 恒 0。
                # ⚠️ 这条**曾经漏写**：只在 Python 分支定义 `child_stomach_scav`，
                #    结果 Rust 分支在 append 处 UnboundLocalError（11 例测试当场抓到）。
                child_stomach_scav = np.zeros(K, dtype=np.float64)
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
                    gcfg.gene_min, gcfg.gene_max, pcfg.max_reward,
                    # 🔴 T2 发现并修（[云端开发·云启] 2026-09-26）：此处原为**硬编码 60.0**
                    #    ⇒ A1 只接了 Python 参考路径（:3796 读配置），**Rust 路径仍用 60**
                    #    ⇒ `k≠1`（`repro_cooldown_gene_scale = 60/k`）时双路径在**第一次生育**就分岔
                    #    （实测：k=2.5 双路径对拍 `born 0 != 1`）。默认档 60.0 == 配置默认 ⇒
                    #    旧行为**逐位等价**（C7 不动）。
                    self.config.organisms.repro_cooldown_gene_scale,
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
                # R165 0-2：尸源归因随胃粮**同比**传给子代（比例不变）
                child_stomach_scav = self._stomach_scav[ri] * split
                energy[ri] -= child_energy
                stomach[ri] -= child_stomach
                self._stomach_scav[ri] -= child_stomach_scav
                # 生完进入冷却（g12）：间隔 = g12 × 换算系数 tick，冷却没到攒再多也不生
                # 🔴 P0.0 A1（2026-09-26）：换算系数原为硬编码 `60.0`，现读配置
                #    （`organisms.repro_cooldown_gene_scale`，默认 60.0 = 旧行为**逐位等价**）。
                #    它是 **基因 ⇒ tick** 的换算 ⇒ 以 tick 计价 ⇒ 时间压缩时必须 ÷k。
                self._repro_cooldown[ri] = (
                    genes[ri, Gene.REPRO_COOLDOWN]
                    * self.config.organisms.repro_cooldown_gene_scale
                )
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
            self._stomach_scav = np.concatenate(
                [self._stomach_scav, child_stomach_scav])
            # 13.4 波 1：亚格坐标随子代扩容 —— **与 `_flat` 完全同型**：
            # 子代继承亲代的格（`_flat[ri]`）⇒ 同时继承亲代的亚格位置（连写法都对齐，便于复核）。
            self._sub_r = np.concatenate([self._sub_r, self._sub_r[ri]])
            self._sub_c = np.concatenate([self._sub_c, self._sub_c[ri]])

            # S1 骨架：血条随子代扩容（初值 1.0，H1）；机制不接线，仅保数组同长
            self._health = np.concatenate([self._health, np.ones(K, dtype=np.float64)])
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
            # 14.9 ARS：子代扩容（出生即赶路模式；heading 未知；期待从零开始）
            self._out_taken = np.concatenate(
                [self._out_taken, np.zeros(K, dtype=np.float64)])
            self._feed_fast = np.concatenate(
                [self._feed_fast, np.zeros(K, dtype=np.float64)])
            self._feed_slow = np.concatenate(
                [self._feed_slow, np.zeros(K, dtype=np.float64)])
            self._ars_extensive = np.concatenate(
                [self._ars_extensive, np.ones(K, dtype=bool)])
            self._heading = np.concatenate(
                [self._heading, np.full(K, -1, dtype=np.int64)])
            self._giveup_ct = np.concatenate(
                [self._giveup_ct, np.zeros(K, dtype=np.int64)])
            # S3 记忆 v2：子代继承**空记忆**（初值；记忆不遗传——学习/经验属于个体生命周期）
            self._mem_az = np.concatenate(
                [self._mem_az, np.full((K, 4), -1, dtype=np.int8)])
            self._mem_dist = np.concatenate(
                [self._mem_dist, np.zeros((K, 4), dtype=np.float32)])
            self._mem_tick = np.concatenate(
                [self._mem_tick, np.full((K, 4), -1, dtype=np.int64)])
            self._mem_degraded = np.concatenate(
                [self._mem_degraded, np.zeros((K, 4), dtype=bool)])
            self._mem_food = np.concatenate(
                [self._mem_food, np.zeros((K, 4), dtype=np.float32)])
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
            self._stomach_scav = np.concatenate(
                [self._stomach_scav[:P][keep], self._stomach_scav[P:]]
            )
            # 13.4 波 1：亚格坐标随死亡压缩（**与 `_flat` 同一 `keep` 掩码**，
            # 用切片赋值而非拼接 ⇒ 与上面 `[:P][keep]` 的语义逐位一致）。
            self._sub_r = np.concatenate([self._sub_r[:P][keep], self._sub_r[P:]])
            self._sub_c = np.concatenate([self._sub_c[:P][keep], self._sub_c[P:]])
            # 14.9 ARS：随死亡压缩（**同一 `keep` 掩码**；漏掉 ⇒ 与个体错位 ⇒ 惯性项乱指）
            self._out_taken = np.concatenate(
                [self._out_taken[:P][keep], self._out_taken[P:]])
            self._feed_fast = np.concatenate(
                [self._feed_fast[:P][keep], self._feed_fast[P:]])
            self._feed_slow = np.concatenate(
                [self._feed_slow[:P][keep], self._feed_slow[P:]])
            self._ars_extensive = np.concatenate(
                [self._ars_extensive[:P][keep], self._ars_extensive[P:]])
            self._heading = np.concatenate(
                [self._heading[:P][keep], self._heading[P:]])
            self._giveup_ct = np.concatenate(
                [self._giveup_ct[:P][keep], self._giveup_ct[P:]])
            # S3 记忆 v2：随死亡压缩（**同一 `keep` 掩码**；漏掉 ⇒ 与个体错位 ⇒ 记忆乱指）
            self._mem_az = np.concatenate(
                [self._mem_az[:P][keep], self._mem_az[P:]])
            self._mem_dist = np.concatenate(
                [self._mem_dist[:P][keep], self._mem_dist[P:]])
            self._mem_tick = np.concatenate(
                [self._mem_tick[:P][keep], self._mem_tick[P:]])
            self._mem_degraded = np.concatenate(
                [self._mem_degraded[:P][keep], self._mem_degraded[P:]])
            self._mem_food = np.concatenate(
                [self._mem_food[:P][keep], self._mem_food[P:]])
            # S1 骨架：血条随死亡压缩（与 _energy 同节拍）；机制不接线，仅保数组同长
            self._health = np.concatenate(
                [self._health[:P][keep], self._health[P:]]
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

    # ------------------------------------------------ S3 记忆 v2（egocentric，2026-09-28）
    # 规格：`docs/设计文档/设计-S3记忆改造-实施细化稿-20260927.md` §二/§三。
    # 🔴 全部方法只在 `memory_v2=True` 路径被调用；关档不消费任何 RNG（C7 回滚点）。
    # 🔴 与 Rust（`sim_core/src/movement.rs`）**逐字同式** —— 改任一侧必须同步另一侧。

    def _cell_dir_slot(self, cur: int, target: int) -> int:
        """两格 → 世界 Moore 槽位（0–7；同格/退化 ⇒ -1）。

        经度环绕取最短列差（与 `_memory_orientation_cos` 同式：`+cols/2 % cols − cols/2`）；
        行差直取。方向向量与 `_ars_unit`（8 槽位单位向量）做内积 argmax。
        """
        if cur == target:
            return -1
        cols = int(self.world.cols)
        cr, cc = divmod(int(cur), cols)
        tr, tc = divmod(int(target), cols)
        dr = float(tr - cr)
        dcc = ((float(tc - cc) + cols / 2.0) % cols) - cols / 2.0
        ln = dr * dr + dcc * dcc
        if ln < 1e-12:
            return -1
        ln = ln ** 0.5
        u = np.array([dr / ln, dcc / ln], dtype=np.float64)
        return int(np.argmax(self._ars_unit @ u))

    def _mem_v2_write(self, idx_arr: np.ndarray, cur_flat: np.ndarray,
                      tick: int) -> None:
        """富食格写入（v2 分支；取代 v1 `_work_memory` 写入）。

        质心格 = `_patch_centroid` 查表（**只查表、不存世界坐标本体**）；
        `_mem_az = (质心世界方位 − heading) mod 8`；槽 = 最旧（含空槽 -1 ⇒ 空槽优先）；
        未知朝向（heading<0，含极区）**不写入**（无朝向基准 ⇒ 无法做路径整合）。
        """
        ccell = self._patch_centroid[cur_flat[idx_arr]]
        ok = ccell >= 0
        if not ok.any():
            return
        ridx = idx_arr[ok]
        cc = ccell[ok]
        hd = self._heading[ridx]
        slots = np.argmin(self._mem_tick[ridx], axis=1)
        az = np.full(ridx.size, -1, dtype=np.int64)
        for i in range(ridx.size):
            s = self._cell_dir_slot(int(cur_flat[ridx[i]]), int(cc[i]))
            if s < 0 or hd[i] < 0:
                continue
            az[i] = (s - int(hd[i])) % 8
        m = az >= 0
        if not m.any():
            return
        ri = ridx[m]
        si = slots[m]
        ai = az[m]
        self._mem_az[ri, si] = ai.astype(np.int8)
        self._mem_dist[ri, si] = 0.0
        self._mem_tick[ri, si] = tick
        self._mem_degraded[ri, si] = False
        cap = np.maximum(self.resources._capacity[cur_flat[ri]], 1e-12)
        self._mem_food[ri, si] = (
            (self.resources._grid[cur_flat[ri]] / cap).astype(np.float32))

    def _mem_v2_step_one(self, idx: int, hd_old: int, hd_new: int,
                         steps: float) -> None:
        """移动后单个体更新：① 记忆旋转（q ← (q + δ) mod 8，δ = heading 变化）；
        ② 噪声扰动（独立 rng）；③ 距离累加（**只增不减**）；④ 精→粗永久降级。

        未知朝向（hd_old/hd_new < 0，含极区）：跳过旋转（§三-5 极区豁免），
        但仍做距离更新/降级。
        """
        az = self._mem_az[idx]
        act = self._mem_tick[idx] >= 0
        # ① 旋转（只对非空槽）
        if hd_old >= 0 and hd_new >= 0 and hd_old != hd_new:
            m = az >= 0
            if m.any():
                delta = int((hd_new - hd_old) % 8)
                if delta != 0:
                    self._mem_az[idx, m] = (
                        (az[m].astype(np.int64) + delta) % 8).astype(np.int8)
        # ② 噪声扰动（独立流 ⇒ 主 rng 形状不变）：每 tick 以 p 概率 ±1 档
        if self._mem_noise:
            m = az >= 0
            if m.any():
                r1 = self._mem_v2_rng.random(int(m.sum()))
                hit = r1 < self._mem_noise_p
                if hit.any():
                    r2 = self._mem_v2_rng.random(int(hit.sum()))
                    pert = np.where(r2 < 0.5, -1, 1).astype(np.int64)
                    mi = np.flatnonzero(m)[hit]
                    self._mem_az[idx, mi] = (
                        (self._mem_az[idx, mi].astype(np.int64) + pert) % 8
                    ).astype(np.int8)
        # ③④ 距离累加 + 永久降级
        if act.any():
            self._mem_dist[idx, act] = np.minimum(
                self._mem_dist[idx, act] + steps, 1e18)
            over = act & ~self._mem_degraded[idx] & (
                self._mem_dist[idx] > self._mem_degrade_thr)
            self._mem_degraded[idx, over] = True

    def _memory_egocentric_cos(self, idx: int, cur: int, nb: np.ndarray,
                               heading_now: int) -> np.ndarray:
        """打分（v2）：候选邻居 c 的**相对朝向** vs 记忆方位 q 的 `max cos`。

        精记忆：`gain × perc × cos(ang(c) − q) × exp(−d / dist_scale)`（d 进公式）；
        粗记忆：`coarse_gain × perc × cos(ang(c) − q)`（d 不进公式，降级即丢距离信息）；
        多槽取 max（只让最对准的记忆槽说话）。
        `ang(c) = (dir(c) − heading) mod 8`（自我参照系；unknown heading ⇒ 用世界槽位退化）。

        13.11：`memory_weight_gene=True` 时 `gain ← gain × 2×g22`（精/粗同乘 ⇒ 2:1 比例不变；
        乘子在 `perc` 之前 ⇒ 与其余打分项的相对量纲不变）。该乘子**仅 Python 路径**
        （v2 ∧ use_sim_core 已 fail-loud ⇒ 无对拍侧；关档默认 ⇒ 基础式逐位不变）。

        ⚠️ 双路径契约：与 Rust（`sim_core/src/movement.rs`）逐字同式（含 `% 8` / mod）。
        """
        n_cand = len(nb)
        out = np.zeros(n_cand, dtype=np.float64)
        az = self._mem_az[idx]
        ticks = self._mem_tick[idx]
        valid = (az >= 0) & ((self._tick - ticks) <= self._mem_ttl)
        if not valid.any():
            return out
        # 候选 c → 世界槽位（逐格量化；nb ≤ 8 小数组，循环开销可忽略）
        dirs = np.full(n_cand, -1, dtype=np.int64)
        for j in range(n_cand):
            dirs[j] = self._cell_dir_slot(cur, int(nb[j]))
        okd = dirs >= 0
        if not okd.any():
            return out
        ang = (dirs[okd].astype(np.int64) - int(heading_now)) % 8 \
            if heading_now >= 0 else dirs[okd].astype(np.int64)
        qs = az[valid].astype(np.int64)
        cosm = self._ars_cos[ang[:, None], qs[None, :]]          # (n_cand, n_slot)
        deg = self._mem_degraded[idx, valid]
        dvals = self._mem_dist[idx, valid].astype(np.float64)
        gains = np.where(deg, self._mem_coarse_gain, self._mem_gain)
        if self._mem_weight_gene:
            # 13.11：个体记忆权重乘子（关档此分支不执行 ⇒ 逐位不变）
            gains = gains * (2.0 * float(self._genes[idx, Gene.MEMORY_WEIGHT]))
        expd = np.where(deg, 1.0,
                        np.exp(-np.clip(dvals, 0.0, 1e6) / self._mem_dist_scale))
        scored = gains[None, :] * cosm * expd[None, :]
        out[okd] = scored.max(axis=1)
        return out

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
            self._sim_core.pleasure_update(   # sparse:n/a（Rust 侧只读；且在范围锁外）
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
        # 13.4 波 1：亚格坐标（缺键回退见 `load_snapshot`；关档时恒为格中心 ⇒ 与 flat 冗余但无害，
        # 存它是为了让"开档续跑"读到正确亚格位置，而不是被重置到格中心）。
        data["sub_r"] = self._sub_r[:P].copy()
        data["sub_c"] = self._sub_c[:P].copy()
        data["energy"] = self._energy[:P].copy()
        data["stomach"] = self._stomach[:P].copy()
        data["stomach_scav"] = self._stomach_scav[:P].copy()   # R165 0-2 归因
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

        # --- 3.6 S1 骨架：尸体格数组 + 个体血条数组（v3 追加键；旧快照加载时回退零值/初值）---
        data["corpse_energy"] = self._corpse_energy.copy()  # (n_cells,)
        data["corpse_age"] = self._corpse_age.copy()        # (n_cells,)
        data["corpse_boost"] = self._corpse_boost.copy()    # (n_cells,) S2：腐烂处 +50% 计时
        data["health"] = self._health[:P].copy()            # (P,)
        data["corpse_eaten_n"] = np.array(self._corpse_eaten_n)
        data["corpse_flows"] = np.array(
            [self._corpse_deposited_e, self._corpse_overflow_e,
             self._corpse_scav_e, self._corpse_decayed_e], dtype=np.float64)
        data["wound_n"] = np.array(self._wound_n)
        data["contest_n"] = np.array(self._contest_n)
        data["contest_holder_win_n"] = np.array(self._contest_holder_win_n)
        data["fearh_flat_n"] = np.array(self._fearh_flat_n)
        data["fearh_dec_n"] = np.array(self._fearh_dec_n)

        # --- 3.7 14.9 ARS 六数组（按个体；R216 §四 裁定 A = **存 + 恢复**）---
        # 这六数组是**跨 tick 状态**（期待 EMA / 双模式 / 惯性 heading / 失望计数 / 本 tick 咬到量）；
        # 不进快照 ⇒ 续跑丢 ARS 状态、且长度停在构造期 initial_count ⇒ 恢复后首次死亡压缩即
        # IndexError（R215-b）。与 §七-1 的账本长度缺陷**同族**：快照不完整 ⇒ 恢复后静默错位。
        data["out_taken"] = self._out_taken[:P].copy()
        data["feed_fast"] = self._feed_fast[:P].copy()
        data["feed_slow"] = self._feed_slow[:P].copy()
        data["ars_extensive"] = self._ars_extensive[:P].copy()
        data["heading"] = self._heading[:P].copy()
        data["giveup_ct"] = self._giveup_ct[:P].copy()

        # --- 3.8 S3 记忆 v2 5 数组（按个体；实施规格 §二 —— 漏登 = 续跑丢记忆状态）---
        data["mem_az"] = self._mem_az[:P].copy()
        data["mem_dist"] = self._mem_dist[:P].copy()
        data["mem_tick"] = self._mem_tick[:P].copy()
        data["mem_degraded"] = self._mem_degraded[:P].copy()
        data["mem_food"] = self._mem_food[:P].copy()
        data["patch_centroid"] = self._patch_centroid.copy()

        # --- 4. 世界状态（资源场 + 信号场）---
        data["resource_grid"] = self.resources._grid.copy()
        data["resource_capacity"] = self.resources._capacity.copy()
        data["resource_patch_mask"] = (
            self.resources._patch_mask.copy()
            if self.resources._patch_mask is not None
            else np.zeros(0, dtype=bool)
        )
        # --- 4.1 13.4 波 2A：资源动态状态（T2；关档 = 全零/初值 ⇒ 与旧快照兼容）---
        data["rd_mask"] = self._rd._mask.copy()
        data["rd_dead"] = self._rd._dead.copy()
        data["rd_rest_until"] = self._rd._rest_until.copy()
        data["rd_dead_since"] = self._rd._dead_since.copy()
        data["rd_demoted"] = self._rd._demoted.copy()
        data["rd_damage"] = self._rd._damage.copy()   # 波2 修 v2：累计损伤（随快照走）
        data["rd_kill_n"] = np.array(self._rd.patch_kill_n)
        data["rd_reborn_n"] = np.array(self._rd.patch_reborn_n)
        data["rd_forced_reborn_n"] = np.array(self._rd.forced_reborn_n)
        data["rd_promote_n"] = np.array(self._rd.promote_n)
        data["rd_rest_set_n"] = np.array(self._rd.rest_set_n)
        data["resource_bg_regrowth_mult"] = np.array(self.resources._bg_regrowth_mult)
        data["resource_patch_regrowth_mult"] = np.array(self.resources._patch_regrowth_mult)
        data["signal_marks"] = self.signals._marks.copy()
        data["signal_age"] = self.signals._age.copy()
        # ---- R240 T8：气味场（**新的全场量** ⇒ 必须进档；全关 ⇒ 无键）----
        #   派生量（粗网格）与计时器都是函数/读数 ⇒ **不进档**（同 `sparse_fields` 纪律）。
        if self.smell is not None:
            _st = self.smell.state()          # 两级状态（近场全分辨率 + 远场粗网格）
            data["smell_S"] = _st["S"]
            data["smell_Sc"] = _st["Sc"]

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
        # R230 T-C（F-D2 修复 B′）：D2 感知噪声/Steels 配对的独立流也须入快照，
        # 否则"续跑 ≠ 连续跑"（旧版靠 runner 侧车 pickle 全局 np.random 状态；
        # B′ 后引擎不读全局 ⇒ 侧车失效，续跑一致性只能由本键保证）。
        data["d2_rng_state"] = np.array(
            pickle.dumps(self._d2_rng.get_state()), dtype=object
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

        # --- 7.6 恢复 S1 骨架数组（旧快照缺键 ⇒ 回退零值/初值，不报错）---
        # 格数组恒 (n_cells,)；health 恒 (P,)，P = len(_id)（恢复后 _id 已就位）。
        # R165 0-2 归因数组：旧快照缺键 ⇒ 回退零（**归因从 0 起**，不报错；
        # 该批此前的食腐收入已入 `intake_forage`，故"从旧快照续跑"的归因不可追认）。
        engine._stomach_scav = (
            data["stomach_scav"].copy()
            if "stomach_scav" in data
            else np.zeros(len(engine._id), dtype=np.float64)
        )
        # 13.4 波 1：亚格坐标。**旧快照缺键 ⇒ 由 `_flat` 回推格中心**（语义正确：13.3 及以前的
        # 个体都"站在格中心"）⇒ 不报错，保持"旧快照仍可读"（S1 阶段的兼容承诺）。
        if "sub_r" in data and "sub_c" in data:
            engine._sub_r = data["sub_r"].copy()
            engine._sub_c = data["sub_c"].copy()
        else:
            engine._sub_r, engine._sub_c = _sub_init(
                engine._flat, int(getattr(engine.config.subpos, "subdiv", 4)),
                engine.world,
            )
        engine._corpse_energy = (
            data["corpse_energy"].copy()
            if "corpse_energy" in data
            else np.zeros(engine.world.n_cells, dtype=np.float64)
        )
        engine._corpse_age = (
            data["corpse_age"].copy()
            if "corpse_age" in data
            else np.zeros(engine.world.n_cells, dtype=np.int64)
        )
        engine._corpse_boost = (
            data["corpse_boost"].copy()
            if "corpse_boost" in data
            else np.zeros(engine.world.n_cells, dtype=np.int64)
        )
        engine._health = (
            data["health"].copy()
            if "health" in data
            else np.ones(len(engine._id), dtype=np.float64)
        )
        engine._corpse_eaten_n = int(data["corpse_eaten_n"]) if "corpse_eaten_n" in data else 0
        if "corpse_flows" in data:      # R165 0-1（旧快照缺键 ⇒ 回退零，不报错）
            _fl = [float(x) for x in data["corpse_flows"]]
            (engine._corpse_deposited_e, engine._corpse_overflow_e,
             engine._corpse_scav_e, engine._corpse_decayed_e) = _fl[:4]
        engine._wound_n = int(data["wound_n"]) if "wound_n" in data else 0
        engine._contest_n = int(data["contest_n"]) if "contest_n" in data else 0
        engine._contest_holder_win_n = int(
            data["contest_holder_win_n"]) if "contest_holder_win_n" in data else 0
        engine._fearh_flat_n = int(data["fearh_flat_n"]) if "fearh_flat_n" in data else 0
        engine._fearh_dec_n = int(data["fearh_dec_n"]) if "fearh_dec_n" in data else 0

        # --- 7.7 恢复 14.9 ARS 六数组（R216 §四 裁定 A；旧快照缺键 ⇒ 回退构造期初值）---
        # 缺键语义 = "该快照保存时无 ARS 状态" ⇒ 回退初值（出生即赶路 / heading 未知 / 期待零），
        # 与 `__init__` 一致；长度按**恢复后的存活数** P 建（不是构造期 initial_count）⇒ 不静默错位。
        _P = len(engine._id)
        engine._out_taken = (
            data["out_taken"].copy() if "out_taken" in data
            else np.zeros(_P, dtype=np.float64))
        engine._feed_fast = (
            data["feed_fast"].copy() if "feed_fast" in data
            else np.zeros(_P, dtype=np.float64))
        engine._feed_slow = (
            data["feed_slow"].copy() if "feed_slow" in data
            else np.zeros(_P, dtype=np.float64))
        engine._ars_extensive = (
            data["ars_extensive"].copy() if "ars_extensive" in data
            else np.ones(_P, dtype=bool))
        engine._heading = (
            data["heading"].copy() if "heading" in data
            else np.full(_P, -1, dtype=np.int64))
        engine._giveup_ct = (
            data["giveup_ct"].copy() if "giveup_ct" in data
            else np.zeros(_P, dtype=np.int64))

        # --- 7.5 S3 记忆 v2 5 数组恢复（旧快照缺键 ⇒ 初值；长度按 _P 截断）---
        engine._mem_az = (
            data["mem_az"].copy() if "mem_az" in data
            else np.full((_P, 4), -1, dtype=np.int8))
        engine._mem_dist = (
            data["mem_dist"].copy() if "mem_dist" in data
            else np.zeros((_P, 4), dtype=np.float32))
        engine._mem_tick = (
            data["mem_tick"].copy() if "mem_tick" in data
            else np.full((_P, 4), -1, dtype=np.int64))
        engine._mem_degraded = (
            data["mem_degraded"].copy() if "mem_degraded" in data
            else np.zeros((_P, 4), dtype=bool))
        engine._mem_food = (
            data["mem_food"].copy() if "mem_food" in data
            else np.zeros((_P, 4), dtype=np.float32))
        if "patch_centroid" in data and data["patch_centroid"].size == engine.world.n_cells:
            engine._patch_centroid = data["patch_centroid"].copy()

        # --- 8. 恢复世界状态 ---
        engine.resources._grid = data["resource_grid"].copy()   # sparse:reset（下方 rebuild_lazy）
        engine.resources._capacity = data["resource_capacity"].copy()
        _pm = data["resource_patch_mask"]
        engine.resources._patch_mask = _pm.copy() if _pm.size > 0 else None
        # --- 8.0 恢复资源动态状态（T2；旧快照缺键 ⇒ 由当前掩码重建 = 语义正确）---
        if "rd_mask" in data and data["rd_mask"].size == engine.world.n_cells:
            engine._rd._mask = data["rd_mask"].copy()
            engine._rd._dead = data["rd_dead"].copy()
            engine._rd._rest_until = data["rd_rest_until"].copy()
            engine._rd._dead_since = data["rd_dead_since"].copy()
            engine._rd._demoted = data["rd_demoted"].copy()
            # 波2 修 v2：累计损伤（旧快照缺键 ⇒ 回退全零，语义 = 从未被啃食）
            engine._rd._damage = (
                data["rd_damage"].copy()
                if "rd_damage" in data
                else np.zeros(engine.world.n_cells, dtype=np.float64)
            )
            engine._rd.patch_kill_n = int(data["rd_kill_n"])
            engine._rd.patch_reborn_n = int(data["rd_reborn_n"])
            engine._rd.forced_reborn_n = int(data["rd_forced_reborn_n"])
            engine._rd.promote_n = int(data["rd_promote_n"])
            engine._rd.rest_set_n = int(data["rd_rest_set_n"])
        else:
            # 旧快照（13.4 前）：rd 状态 = 构造期默认（掩码随 resources._patch_mask 走）
            engine._rd._mask = (
                engine.resources._patch_mask.copy()
                if engine.resources._patch_mask is not None
                else np.zeros(engine.world.n_cells, dtype=bool)
            )
        engine.resources._bg_regrowth_mult = float(data["resource_bg_regrowth_mult"])
        engine.resources._patch_regrowth_mult = float(data["resource_patch_regrowth_mult"])
        engine.signals._marks = data["signal_marks"].copy()   # sparse:reset（下方 rebuild_sparse）
        engine.signals._age = data["signal_age"].copy()   # sparse:reset（下方 rebuild_sparse）
        # ---- R240 T8：气味场恢复（旧快照**缺键** ⇒ 空场起算 = 文档口径；形状不符 ⇒ fail-loud）----
        #   全关档（`smell is None`）调用快照**不可能**带 `smell_S`（配置指纹会先拦跨档），
        #   故这里在**同档**内只可能是"有键"；缺键只出现在"旧档升级到新代码"的场景。
        if engine.smell is not None and "smell_S" in data:
            engine.smell.restore({"S": data["smell_S"],
                                  "Sc": data["smell_Sc"] if "smell_Sc" in data else None})
            engine._smell_gen += 1     # R244 §二：恢复后缓存全体失效（stamp 可能带旧代）
        # R217 §五 #1 稀疏化 B：脏格集/活跃集是**派生量**（不进快照）⇒ 整体替换
        #   `_grid` / `_marks` / `_age` 之后必须按不变量重建，否则陈旧集合会漏结算（破等价）。
        #   （未启用时这两个方法是 no-op。）
        engine.resources.rebuild_lazy()
        engine.signals.rebuild_sparse()

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
        # --- 9. 恢复 RNG 状态 ---
        rng_state = pickle.loads(data["rng_state"].item())
        engine.rng.bit_generator.state = rng_state
        # R230 T-C（F-D2 修复 B′）：恢复 D2 独立流；旧快照无此键 ⇒ 保持构造期新播种状态
        # （语义 = 与修复前"缺侧车"的续跑一致；不静默错位）。
        if "d2_rng_state" in data:
            engine._d2_rng.set_state(pickle.loads(data["d2_rng_state"].item()))

        # --- 10. 恢复元数据 ---
        engine._tick = int(data["tick"])
        engine._next_id = int(data["next_id"])
        engine._max_generation = int(data["max_generation"])
        engine._extinct = bool(data["extinct"])
        engine._finished = bool(data["finished"])

        # --- 10.5 恢复 _id 键控终身账本（R215 §七-1：**必须晚于 §10 的 `_next_id` 赋值**）---
        # 原顺序（账本在 §10 之前）会让"补齐长度"按**构造期**的 `_next_id`（= initial_count，
        # `__init__` 建的初始种群）计算 ⇒ 旧快照（无探针键）`need = initial_count − initial_count
        # = 0` ⇒ 12 个账本长度**停在 initial_count**，而 `_next_id` 已是快照值；续跑首次出生
        # `self._rs_children[self._id[ri]] += 1` 即越界（IndexError），且 `_rs_observed[alive_ids]`
        # 同样越界 ⇒ **默认档快照续跑**（14.10 长跑复活流程）不可用。
        # 修法 = R215 §七-1 的"补齐改到赋值之后"（另一种等价修法是元数据整块前移）。
        # 🔴 数值语义未变：账本内容仍**只**来自快照（缺键 ⇒ 全零 = 未观测/未发射）；
        #    变的只是"补齐到哪个长度" ⇒ 无探针档的续跑轨迹逐位不变（见守卫测试）。
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
            # 旧快照（无探针数组）：_id 键控账本扩到 `_next_id`（全零 = 未观测/未发射）
            need = int(engine._next_id) - len(engine._rs_children)
            engine._grow_id_arrays(need)

        return engine