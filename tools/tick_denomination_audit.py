"""tick 面额清点（P0.0 / A2）—— **机械化**扫出所有"以 tick 计价"的常数。

为什么必须机械化
----------------
2026-09-25 我们**手写**了一份重标清单（`experiments/scaling_rescale.py`），
漏掉整整一类 ——「**每 tick 概率**」（正确改法是 `1−(1−p)^k`，×k 与不变**都不对**），
且是**方向性偏差**（k=5 时种群 +82%）。
⇒ **手工清单不是验收物，机械化清点才是。**

四类（必须四类）
----------------
| 类 | 改法 | 判据 |
|---|---|---|
| `RATE` | **× k** | 每 **tick** 累积/消费一次的量（速率·时长） |
| `DURATION` | **÷ k** | 以 **tick 计数**的时长 |
| `DECAY` | **^(k)** | 每 tick **乘性**衰减（保留比例） |
| `PROB` | **1−(1−p)^k** | 每 tick **独立伯努利**试次 |
| `INVARIANT` | 不变 | 量/比例/计数/基因/速度 |

⚠️ **本工具的判定是"建议 + 证据"，不是"真相"** —— 它靠**字段名/上下文关键词 + 白名单**
做启发式分类，**人工只做复核**（任务书 §三 A2 的"验收：人工只做复核"）。
⇒ 因此每条都打印**证据位置**（字段名 / 文件:行 / 字面量上下文），**未分类项必须 = 0**。

用法
----
    python3 tools/tick_denomination_audit.py                 # 全量清点（打印报告）
    python3 tools/tick_denomination_audit.py --json out.json # 同时落 JSON（机器可读）
    python3 tools/tick_denomination_audit.py --md out.md     # 同时落 Markdown 表（人工复核用）
    python3 tools/tick_denomination_audit.py --no-literals   # 只看 config 字段（快；覆盖自检仍恒跑）

退出码（T6/A2b 语义，可进 CI）
----
    0 = 干净；1 = 审计发现（未分类 > 0 或交叉校验冲突）；2 = **覆盖/扫描硬失败**（fail-loud）。

A2b 覆盖纪律（T6，2026-09-26）
----
* **扫描清单 = glob 派生**（`simulation/*.py` + `world/*.py`，排除 `config.py`/`__init__.py`），
  **禁手写清单** —— 旧版手写清单含不存在的 `world/resources.py`（真名 `resource_field.py`），
  且 `if p.is_file():` **静默跳过** ⇒ `world/` 6 个业务文件里只有 1 个真被扫到（变异测试证实有洞）。
* **缺失/0 文件/低于数量下限/文件不存在 ⇒ fail-loud 抛 `CoverageError`（退出码 2）**，不许静默。
* **覆盖自检与 `--no-literals` 解耦**：不管扫不扫字面量，覆盖**每次都验**
  （踩过的坑：把派生放进 `if not a.no_literals:` ⇒ `--no-literals` 时目录被移走也 rc=0）。
* 解析失败（SyntaxError）⇒ 也是覆盖失败，不许 `return []` 静默吞掉。
* **噪声过滤**（只降噪不删：被过滤项仍落 JSON）——`round(x, n)` 的 n、`+= 1` 计数、
  0/1/2 平凡值、索引/形状上下文 ⇒ 不进"线索区"，否则 1600+ 条淹没真信号（见 `NOISE_*`）。
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import SimConfig  # noqa: E402

# --- R98 纪律：Windows GBK 控制台兜底（本工具 print 含 ⇒/✅/🔴，无兜底则 rc=1 假失败）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行


class CoverageError(RuntimeError):
    """扫描覆盖硬失败（A2b fail-loud）⇒ 退出码 2，与"审计发现"（1）区分。"""


# ---------------------------------------------------------------------------
# 分类白名单：字段全名（`组.字段`）⇒ 类
#   🔴 与 `experiments/scaling_rescale.py` 的四张表**必须一致**；
#      本工具会做交叉校验（`--cross-check`），不一致直接报错（fail-loud）。
# ---------------------------------------------------------------------------

RATE = "RATE"
DURATION = "DURATION"
DECAY = "DECAY"
PROB = "PROB"
INVARIANT = "INVARIANT"

#: 额外类（不在"四类改法"内，但必须显式归类，否则会污染"未分类=0"）
PER_EVENT = "PER_EVENT"      # 每事件一次（走一格/咬一口），非每 tick
RUNTIME = "RUNTIME"          # 运行长度（跑多少 tick），由调用方决定
UNWIRED = "UNWIRED"          # 引擎/世界中从未出现 ⇒ 不承载 tick 面额

#: 已知分类（组.字段 → 类）。未列出的字段由启发式规则给出**建议**。
KNOWN: dict[str, str] = {
    # ---- 每 tick 速率（× k）----
    "resources.regrowth_rate": RATE,
    "organisms.base_metabolism": RATE,
    "organisms.move_cost": RATE,
    "organisms.photo_max": RATE,
    "organisms.homeo_upkeep": RATE,
    "organisms.eat_amount": RATE,
    "fruit.charge_rate": RATE,
    "fruit.eat_rate": RATE,
    "corpse_wound.wound_heal_rate": RATE,
    "corpse_wound.wound_heal_energy_cost": RATE,
    "pleasure.alpha": RATE,
    "info_structure.learning_rate": RATE,
    "info_structure.alignment_rate": RATE,
    # ---- tick 时长（÷ k）----
    "light.rotation_period": DURATION,
    "info_structure.learning_maturity_ticks": DURATION,
    "corpse_wound.corpse_decay_ticks": DURATION,
    "resource_dynamics.rest_ticks": DURATION,
    "resource_dynamics.dead_regen_ticks": DURATION,
    "ars.fast_tau": DURATION,
    "ars.slow_tau": DURATION,
    "ars.giveup": DURATION,
    "oracle.persistence": DURATION,
    # ---- P0.0 A1 新增：两个"硬编码搬进配置"的 tick 常数 ----
    "signals.duration_ticks": DURATION,
    "organisms.repro_cooldown_gene_scale": DURATION,
    # ---- R240 T8 气味场（新机制；本线 = [云端开发·云启]）----
    "smell.update_every": DURATION,          # 更新间隔（k tick）⇒ 时间压缩 ÷k（配 diffuse ×k ⇒ 每日扩散不变）
    "smell.diffuse": RATE,                   # 每次更新的扩散权重 ⇒ 每日扩散 = (更新数/日)×系数 ⇒ ÷k 次配 ×k
    "smell.decay": DECAY,                    # 「每 tick 衰减」⇒ 时间压缩 ^k
    "smell.downsample": INVARIANT,           # 空间 LOD（1/s）—— 与 tick 无关
    "smell.risk_g16_threshold": INVARIANT,   # 基因阈值（无量纲）
    "smell.inject_food": INVARIANT,          # 注入权重（无量纲）
    "smell.inject_prey": INVARIANT,
    "smell.inject_risk": INVARIANT,
    "smell.inject_kin": INVARIANT,
    "smell.channels": INVARIANT,             # 通道集（名字元组）
    # ---- R244 §二 气味场消费端（本线 = [云端开发·云启]）----
    "smell.use_in_move": INVARIANT,          # 开关（bool）—— 不承载 tick 面额
    "smell.w_food": INVARIANT,               # 通道权重（无量纲固定权重；扩基因位留批 B）
    "smell.w_prey": INVARIANT,
    "smell.w_risk": INVARIANT,
    "smell.w_kin": INVARIANT,
    "smell.norm_mode": INVARIANT,            # 归一化模式（字符串）
    # ---- 每 tick 概率（1−(1−p)^k）----
    "predation.attack_prob_coef": PROB,
    "pleasure.baseline_rate": PROB,
    "fruit.germination_prob": PROB,
    "fruit.excretion_prob": PROB,
    "fruit.seed_intake_prob": PROB,
    "info_structure.codebook_mutation_rate": PROB,
    # ---- 每 tick 衰减（^ k）----
    "pleasure.valence_decay": DECAY,
    "pleasure.arousal_decay": DECAY,
    # ======================================================================
    # 以下为 A2 **人工复核**后补录（逐条核过实际用法，证据见字段注释/引擎行号）
    # 复核原则：**只有"每 tick 结算"的量才是 tick 面额**；
    #   "每次事件结算"（走一格付一次、咬一口付一次）与"每 tick 独立伯努利"
    #   必须区分 —— 前者若事件频率随 k 变则需另立类，后者是 PROB。
    # ======================================================================
    # ---- 量：上限 / 初始值 / 一次性转移（**不随时间累积**）⇒ 不变 ----
    "organisms.initial_energy": INVARIANT,
    "organisms.stomach_cap_mass": INVARIANT,
    "corpse_wound.corpse_cap_per_cell": INVARIANT,
    "resources.capacity_per_area": INVARIANT,
    "resources.background_fill": INVARIANT,
    "resources.initial_fill": INVARIANT,
    # ---- 比例 / 系数 / 倍率（无量纲）⇒ 不变 ----
    "organisms.eat_efficiency": INVARIANT,
    "organisms.maturity_fraction": INVARIANT,
    "organisms.senile_fraction": INVARIANT,
    "organisms.assim_herb": INVARIANT,
    "organisms.assim_carn": INVARIANT,
    "organisms.dash_cost_exp": INVARIANT,
    "organisms.dash_cost_kappa": INVARIANT,
    "predation.attack_gene_gate": INVARIANT,
    "predation.transfer_ratio": INVARIANT,
    "predation.stomach_transfer": INVARIANT,
    "predation.success_floor": INVARIANT,
    "predation.success_ceil": INVARIANT,
    "predation.success_gene_gain": INVARIANT,
    "culture.food_threshold": INVARIANT,
    "culture.trust_true": INVARIANT,
    "culture.trust_false": INVARIANT,
    "fruit.digest_ratio": INVARIANT,
    "fruit.fruit_ratio": INVARIANT,
    "fruit.fruit_threshold": INVARIANT,
    "fruit.plant_threshold": INVARIANT,
    "info_structure.reputation_weight": INVARIANT,
    "info_structure.alignment_step": INVARIANT,
    "info_structure.alignment_noise": INVARIANT,
    "info_structure.perception_noise": INVARIANT,
    "info_structure.memory_gradient_gain": INVARIANT,
    "info_structure.memory_weight_gene": INVARIANT,  # 13.11 开关（bool）—— 乘子只改打分权重，不承载 tick 面额
    "info_structure.perception_radius": INVARIANT,
    "pleasure.optimism": INVARIANT,
    "pleasure.w_energy": INVARIANT,
    "pleasure.w_info": INVARIANT,
    "pleasure.w_social": INVARIANT,
    "pleasure.alone_rpe": INVARIANT,
    "pleasure.social_rpe": INVARIANT,
    "pleasure.inheritance_noise": INVARIANT,
    "population.soft_cap_target": INVARIANT,
    "corpse_wound.scav_s": INVARIANT,
    "corpse_wound.scav_gate": INVARIANT,
    "corpse_wound.holder_adv": INVARIANT,
    "corpse_wound.corpse_patch_boost": INVARIANT,
    "corpse_wound.escalation_gap": INVARIANT,
    "corpse_wound.w_fear_health": INVARIANT,
    "corpse_wound.wound_base": INVARIANT,
    "corpse_wound.wound_fear_threshold": INVARIANT,
    "resource_dynamics.damage_recovery": INVARIANT,
    "resource_dynamics.rest_threshold": INVARIANT,
    "resource_dynamics.death_threshold": INVARIANT,
    "genome.mutation_sigma": INVARIANT,
    "migration.gain": INVARIANT,
    "migration.min_abs_anomaly": INVARIANT,
    # ---- R247 饥饿调制（HM）：无量纲调制系数/分位点 ⇒ 不承载 tick 面额 ----
    #   （`hunger_mod.stay_gain` 为预留未接线字段 ⇒ 由"未接线字段"自动单列，不进本表）
    "hunger_mod.alpha": INVARIANT,
    "hunger_mod.beta": INVARIANT,
    "hunger_mod.h_mid": INVARIANT,
    # ---- R239 ASM 模式仲裁（`sir`）：显著度权重/阈值无量纲；`hold_ticks` 是 tick 时长 ----
    "action_selection.base_explore": INVARIANT,   # 探索保底显著度（无量纲，与 Ŝ∈[0,1] 同量纲）
    "action_selection.hyst": INVARIANT,           # 迟滞阈值（显著度差，无量纲）
    "action_selection.w_feed": INVARIANT,         # 四模式权重（固定参数，批 B 才进基因位）
    "action_selection.w_hunger": INVARIANT,
    "action_selection.w_flee": INVARIANT,
    "action_selection.w_join": INVARIANT,
    "action_selection.hold_ticks": DURATION,      # 最小锁定（tick 计数）⇒ 时间压缩 ÷k
    "action_selection.mode": INVARIANT,           # 模式开关（字符串）
    "ars.gain": INVARIANT,
    "ars.theta": INVARIANT,
    "ars.kappa": INVARIANT,
    "ars.lat_exempt_deg": INVARIANT,
    "oracle.gain_multiplier": INVARIANT,
    "oracle.donation": INVARIANT,
    "light.day_boost": INVARIANT,
    "light.lat_base_ref": INVARIANT,
    "light.t_equator": INVARIANT,
    "light.t_pole": INVARIANT,
    "light.tilt_rad": INVARIANT,
    "resources.light_sensitivity": INVARIANT,
    "resources.temp_sensitivity": INVARIANT,
    "simulation.social_move_weight": INVARIANT,
    "simulation.w_fear": INVARIANT,
    "subpos.lat_floor": INVARIANT,
    "subpos.speed_gain": INVARIANT,
    "subpos.stay_base": INVARIANT,
    "world.rows": INVARIANT,
    "world.cols": INVARIANT,
    "simulation.cell_occupancy_cap": INVARIANT,
    "simulation.perception_cap": INVARIANT,
    "simulation.perception_span": INVARIANT,
    "simulation.attack_range": INVARIANT,
    "simulation.history_limit": INVARIANT,
    "organisms.far_cap": INVARIANT,
    # ---- ⚠️ 事件计价（**不是每 tick**）：单次事件的一次性代价/收益 ----
    #   语义：走一格付一次 / 咬一口付一次 ⇒ 与"每 tick"不同量纲。
    #   🔴 时间压缩下是否要重标，取决于**事件频率是否随 k 变** ——
    #      若个体的每「世界日」事件数不变（锚点保证路程/日不变）⇒ **不必重标**。
    #   本工具单列 `PER_EVENT` 以便审计者一眼看到这一族。
    "predation.attack_cost": PER_EVENT,
    "corpse_wound.contest_cost_energy": PER_EVENT,
    # ---- ⚠️ 时长但**不属于重标范围**（运行长度，不是物理常数）----
    "simulation.ticks": RUNTIME,
    # ======================================================================
    # 🔴 `organisms.move_cost` —— **必须单列，不许归入四类中任何一类**
    # ======================================================================
    # 实测证据（`simulation/sphere_engine.py`）：
    #   · subpos 路径（line 2745）：`move_cost_ind = ocfg.move_cost * cold_penalty
    #     * (0.5 + genes[:, MOVE_COST])`，随后 line 2903 `_cost2 = move_cost_ind * _step`
    #     ⇒ 它是 **"走一整步（可能跨多个亚格）"的一次性代价** × dash 倍率，
    #       **不是每 tick 结算**。
    #   · 整数格路径：位移恒 1 格/tick ⇒ "每事件" 与 "每 tick" **数值上重合**，
    #     所以两条路径在 D=2400 下看不出差别 —— 这正是它危险的地方。
    #   ⇒ **结论**：按 `PER_EVENT` 归类；`scaling_rescale.py` 把它列进 `_RATE_FIELDS`
    #      在**语义上是可疑的**（见下方「已知不一致」）。
    "organisms.move_cost": PER_EVENT,
}

#: 与 `experiments/scaling_rescale.py` 的**已知语义不一致**（不是笔误，是待裁定项）
#: 格式：字段 → (本工具类, rescale 类, 说明)
KNOWN_DIVERGENCE: dict[str, tuple[str, str, str]] = {
    "organisms.move_cost": (
        "PER_EVENT", "RATE",
        "🔴 **待裁定**：本工具按实测语义归 PER_EVENT（每「走一步」付一次，"
        "subpos 下跨多亚格；整数格路径下与每 tick 重合）。"
        "`scaling_rescale.py` 归 RATE（×k）。"
        "**两者在锚点速度下数值可能一致，但语义不同** ⇒ 若将来事件频率随 k 变，"
        "×k 会错。**本轮不改 rescale**（它已被 A4 实测验证过），"
        "仅在清点表里标记为待裁定，交所有者。",
    ),
}

#: 🔴 显式"不变"（含踩过的陷阱：误分类会出硬错）
KNOWN_INVARIANT: dict[str, str] = {
    "pleasure.expectation_size": "**不是时长**：它是 `_expectation` 的**数组宽度**"
    "（`(n, expectation_size)`）⇒ 擅自 ÷k 直接 IndexError",
    "organisms.move_cost": "⚠️ 双重身份，见下方 NOTE",
}

#: 字段名正则 ⇒ 建议类（顺序敏感：先匹配先定）
_NAME_RULES: list[tuple[str, str, str]] = [
    (r"_ticks?$", DURATION, "字段名以 `_tick(s)` 结尾 ⇒ 以 tick 计数的时长"),
    (r"duration", DURATION, "字段名含 `duration` ⇒ 时长"),
    (r"period", DURATION, "字段名含 `period` ⇒ 周期（tick 数）"),
    (r"tau$", DURATION, "字段名以 `tau` 结尾 ⇒ 时间常数（tick）"),
    (r"giveup", DURATION, "AR S 放弃窗口 ⇒ tick 时长"),
    (r"persistence", DURATION, "oracle 持续窗口 ⇒ tick 时长"),
    (r"cooldown", DURATION, "冷却 ⇒ tick 时长"),
    (r"decay", DECAY, "字段名含 `decay` ⇒ 每 tick 乘性衰减（保留比例）"),
    (r"_prob|^prob", PROB, "字段名含 `prob` ⇒ 每 tick 概率（独立试次）"),
    (r"rate$", RATE, "字段名以 `rate` 结尾 ⇒ 每 tick 速率"),
    (r"per_tick", RATE, "字段名含 `per_tick` ⇒ 每 tick 速率"),
    (r"upkeep$", RATE, "字段名以 `upkeep` 结尾 ⇒ 每 tick 维持费"),
    (r"metabolism", RATE, "代谢 ⇒ 每 tick 速率"),
    (r"_frac$|_mult$|_k$", INVARIANT, "比例 / 系数 ⇒ 无量纲，不变"),
    (r"^(max|min)_|_max$|_min$", INVARIANT, "上下限 ⇒ 量，不变"),
    (r"^patch_(count|radius)$|_count$", INVARIANT, "计数 ⇒ 不变"),
    (r"subdiv|expectation_size|alphabet", INVARIANT, "计数 / 数组宽度 ⇒ 不变"),
]

#: 已知的"双重身份"或易错点，**强制打印**提示
NOTES: dict[str, str] = {    "organisms.move_cost": (
        "⚠️ **双重身份（实测，见 `KNOWN_DIVERGENCE`）**：subpos 路径下 "
        "`move_cost_ind = move_cost × cold_penalty × (0.5+g)`，随后 "
        "`_cost2 = move_cost_ind × _step` ⇒ 它是「**走一整步**」的一次性代价"
        "（跨多个亚格，**非每 tick**）；整数格路径下位移恒 1 格/tick "
        "⇒ 每事件与每 tick **数值重合**（这正是它危险之处）。"
        "⇒ 本工具归 **PER_EVENT**，`scaling_rescale.py` 归 RATE ⇒ **待所有者裁定**。"
    ),
    "pleasure.expectation_size": (
        "⚠️ **踩过的陷阱**：它不是时长而是数组宽度，误除会 IndexError。归 INVARIANT。"
    ),
    "resources.regrowth_rate": (
        "⚠️ 若该 field 被解释为「每 tick 朝 cap 收敛的比例」（乘性），"
        "正确改法是 `1−(1−r)^k` 而非 ×k —— 请核对 `world/resources.py` 的实际用法。"
    ),
}


# ---------------------------------------------------------------------------
# 1) 遍历 SimConfig 全部数值字段
# ---------------------------------------------------------------------------

def _iter_config_fields() -> list[tuple[str, str, object]]:
    """产出 (组名, 字段名, 默认值)。只取数值型（int/float/bool 除外的数字）。"""
    out: list[tuple[str, str, object]] = []
    for grp_field in dataclasses.fields(SimConfig):
        grp_name = grp_field.name
        if grp_name == "seed":
            continue
        val = grp_field.default if grp_field.default is not dataclasses.MISSING else None
        if val is None and grp_field.default_factory is not dataclasses.MISSING:
            try:
                val = grp_field.default_factory()
            except Exception:
                val = None
        if val is None or not dataclasses.is_dataclass(val):
            continue
        for f in dataclasses.fields(val):
            dv = getattr(val, f.name, None)
            if isinstance(dv, (int, float)) and not isinstance(dv, bool):
                out.append((grp_name, f.name, dv))
    return out


def _suggest(grp: str, name: str) -> tuple[str, str]:
    """启发式建议分类，返回 (类, 依据)。"""
    full = f"{grp}.{name}"
    if full in KNOWN:
        return KNOWN[full], "白名单（人工确认过的分类表）"
    for pat, cls, why in _NAME_RULES:
        if re.search(pat, name):
            return cls, why
    return "UNCLASSIFIED", "**无规则命中** ⇒ 必须人工判"


# ---------------------------------------------------------------------------
# 0) 字段是否真的被接线？（未接线的字段不承载 tick 面额 ⇒ 不该占"未分类"）
# ---------------------------------------------------------------------------

_WIRED_CACHE: dict[str, int] | None = None


def _wired_counts() -> dict[str, int]:
    """统计每个字段名在**引擎/世界**代码里出现的次数（排除 config.py 自身）。

    🔴 为什么需要这一步：`niche_floor` / `niche_gain` / `stomach_cap_mass` 这类
    字段**只存在于配置里、从未被引擎读取**。它们**不承载 tick 面额** ——
    如果按名字瞎猜分类，就是"给不存在的东西定规矩"，反而掩盖真信号。
    ⇒ 未接线字段单列一类 `UNWIRED`（**不算"未分类"**），并**显著报出**（禁静默）。
    """
    global _WIRED_CACHE
    if _WIRED_CACHE is not None:
        return _WIRED_CACHE
    counts: dict[str, int] = {}
    files = list((ROOT / "simulation").glob("*.py")) + list((ROOT / "world").glob("*.py"))
    files = [p for p in files if p.name != "config.py"]
    blob = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", blob):
        counts[m.group(0)] = counts.get(m.group(0), 0) + 1
    _WIRED_CACHE = counts
    return counts


# ---------------------------------------------------------------------------
# 2) 扫描 simulation/*.py、world/*.py 的字面量
# ---------------------------------------------------------------------------

#: 字面量所在行的上下文关键词 ⇒ 类（比字段名更弱，只作"建议"）
_LITERAL_CTX: list[tuple[str, str, str]] = [
    (r"duration\s*=|\.duration\b", DURATION, "出现在 `duration=` / `.duration` 处"),
    (r"cooldown", DURATION, "出现在 `cooldown` 计算处"),
    (r"_ticks?\b|ticks\s*=", DURATION, "上下文含 tick 计数"),
    (r"probability|_prob\b|\bprob\b", PROB, "上下文含 `prob`"),
    (r"decay", DECAY, "上下文含 `decay`"),
    (r"rate|per_tick|cost|upkeep|gain\b", RATE, "上下文含速率/成本词"),
]

# ---------------------------------------------------------------------------
# A2b：扫描清单 = **glob 派生**（禁手写清单）
#
# 旧版是手写清单 `_SCAN_FILES = [...4 个路径...]`，后果有三：
#   ① `world/resources.py` **根本不存在**（真名 `world/resource_field.py`）⇒ 该行静默 no-op；
#   ② `world/` 6 个业务文件里只有 1 个真被扫到（变异测试证实：往 `light_and_temperature.py`
#      注入假 tick 常数**扫不出**，往 `sphere_engine.py` 注入能扫出）；
#   ③ 新增文件不会自动纳入。
# ⇒ 现在改为 glob 派生 + 缺失 **fail-loud**（见 `_derive_scan_files`）。
# ---------------------------------------------------------------------------

#: 扫描目录（glob 派生；新增文件自动纳入）
_SCAN_DIRS: tuple[str, ...] = ("simulation", "world")

#: 排除：包标记 + 纯默认值模块（config 的全部数值字段已由 §3.1 枚举，扫其字面量只重复+噪声）
_SCAN_EXCLUDE_NAMES: frozenset[str] = frozenset({"__init__.py", "config.py"})

#: 数量下限（防"glob 写错/目录被移走 ⇒ 扫了个寂寞"；2026-09 实况：simulation 5 + world 6 = 11）
_SCAN_MIN_FILES_PER_DIR = 4
_SCAN_MIN_FILES_TOTAL = 8


def _derive_scan_files(root: Path | None = None) -> list[Path]:
    """**glob 派生**扫描清单（禁手写清单）。

    🔴 fail-loud 四则（缺一则重新引入"静默漏扫"，退出码 2）：
      1. 目录不存在 ⇒ `CoverageError`（不许 `is_file()` 式静默跳过）
      2. 某目录派生到 0 个文件 ⇒ `CoverageError`
      3. 单目录 < `_SCAN_MIN_FILES_PER_DIR` 或总数 < `_SCAN_MIN_FILES_TOTAL` ⇒ `CoverageError`
      4. 派生出的文件不存在 ⇒ `CoverageError`（glob 已保证；双保险：大小写/符号链接）
    """
    base = ROOT if root is None else Path(root)
    files: list[Path] = []
    for d in _SCAN_DIRS:
        dd = base / d
        if not dd.is_dir():
            raise CoverageError(
                f"🔴 [A2b fail-loud] 扫描目录不存在：{dd}\n"
                f"    扫描清单是 glob 派生的（{list(_SCAN_DIRS)}）"
                f" ⇒ 目录消失必须显式报错，不许静默跳过。")
        found = sorted(p for p in dd.glob("*.py") if p.name not in _SCAN_EXCLUDE_NAMES)
        if not found:
            raise CoverageError(
                f"🔴 [A2b fail-loud] 目录 {d}/ 派生到 **0** 个 .py"
                f"（排除 {sorted(_SCAN_EXCLUDE_NAMES)} 后）⇒ 必然说明目录被移走/改名。")
        if len(found) < _SCAN_MIN_FILES_PER_DIR:
            raise CoverageError(
                f"🔴 [A2b fail-loud] 目录 {d}/ 只派生到 {len(found)} 个文件"
                f"（< 下限 {_SCAN_MIN_FILES_PER_DIR}）⇒ glob/目录结构异常。")
        files.extend(found)
    if len(files) < _SCAN_MIN_FILES_TOTAL:
        raise CoverageError(
            f"🔴 [A2b fail-loud] glob 只派生到 {len(files)} 个文件"
            f"（< 下限 {_SCAN_MIN_FILES_TOTAL}）⇒ 拒绝「扫了个寂寞」。")
    for p in files:
        if not p.is_file():
            raise CoverageError(f"🔴 [A2b fail-loud] 派生出的文件不存在：{p}")
    return files


def _scan_summary(files: list[Path]) -> str:
    """人类可读的覆盖摘要（进报告，证明"扫了哪些"）。"""
    by_dir: dict[str, list[str]] = {}
    for p in files:
        by_dir.setdefault(p.parent.name, []).append(p.name)
    parts = [f"{d}/ {len(v)} 个（{', '.join(v)}）" for d, v in sorted(by_dir.items())]
    return f"共 {len(files)} 个文件 ｜ " + " ｜ ".join(parts)


# ---------------------------------------------------------------------------
# A2b：**噪声过滤**（只降噪不删 —— 被过滤项仍落 JSON，只是不进"线索区"）
#
# 实测（验收件 §3.2）：旧版 1213 条"命中"抽前 15 条逐一复核 ⇒ **几乎全是假阳性**
# （`round(x, 9)` 的 9、字面 `1.0`、`self._tick += 1` 的 1；唯一真收获是
#  `sphere_engine.py:1994` 的 `getattr(cwc, "corpse_decay_ticks", 600)`）。
# ⇒ 不过滤 = 真信号被淹没 = 工具不可用（这是"覆盖缺口"的另一面）。
# ⚠️ **保守优先**：宁可漏降噪，也不许把真 tick 常数降掉。
# ---------------------------------------------------------------------------

NOISE_NONE = ""
NOISE_TRIVIAL = "TRIVIAL_0_1_2"        # 0/1/2 与 0.0/1.0/2.0，且整行无 tick 关键词
NOISE_STRUCT = "STRUCTURAL_CTX"        # 索引/形状/协议上下文；`round(x,n)` 的 n；`+=` 计数操作数

#: 上下文出现这些词 ⇒ 大概率是索引/形状/协议（非 tick 面额）
_NOISE_CTX = re.compile(
    r"\b(shape|reshape|axis|dtype|ndim|size|len|range|enumerate|arange|zeros|ones|"
    r"empty|full|astype|int32|int64|float32|float64|uint8|seed|version|"
    r"encoding|errors|header|magic|mask|bit|shift|stride|offset|index|idx|"
    r"order|kind|mode|opts|round|format|width|precision|subdiv|passes)\b",
    re.IGNORECASE,
)

#: 整行含这些词 ⇒ 平凡值（0/1/2）**不降噪**（可能是真面额，如 `xyz_ticks = 1`）
_TICK_CTX = re.compile(
    r"tick|duration|cooldown|period|tau|decay|prob|rate|upkeep|metabolism|giveup|persistence",
    re.IGNORECASE,
)


def _noise_reason(v: object, ctx: str, *, structural: bool) -> str:
    """判定该字面量是否为**噪声**（只降噪，不删；证据仍进 JSON）。

    `structural` = AST 级的结构证据（`round(x,n)` 精度位 / `+=` 计数 / 负哨兵 /
    下标索引 / `max·min·range` 守卫参数 / 十六进制掩码）⇒ 确定不是 tick 面额。
    """
    if structural:
        return NOISE_STRUCT
    if isinstance(v, float) and v in (0.0, 1.0, 2.0):
        return NOISE_TRIVIAL                       # 验收件点名：字面 1.0/0.0 这类噪声（无条件）
    if isinstance(v, int) and v in (0, 1, 2) and not _TICK_CTX.search(ctx):
        return NOISE_TRIVIAL
    if isinstance(v, int) and _NOISE_CTX.search(ctx):
        return NOISE_STRUCT
    return NOISE_NONE


def _hex_mask_in(ctx: str, v: int) -> bool:
    """行内是否有与 `v` 等值的十六进制字面量（如 `0x0F` ↔ 15）⇒ 掩码，不是面额。"""
    for tok in re.findall(r"0[xX][0-9a-fA-F_]+", ctx):
        try:
            if int(tok.replace("_", ""), 16) == v:
                return True
        except ValueError:
            pass
    return False


def _rel(p: Path) -> str:
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _scan_literals(path: Path) -> list[dict]:
    """用 `ast` 抽数字字面量 + 所在行文本（**排除注释与 docstring**）。

    A2b：解析失败**不再静默 `return []`** —— 语法坏掉的文件是覆盖缺口，抛 `CoverageError`。
    """
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise CoverageError(
            f"🔴 [A2b fail-loud] 扫描目标解析失败（覆盖缺口，不许静默跳过）：{_rel(path)}"
            f" —— {type(exc).__name__}: {exc}") from exc
    lines = src.splitlines()

    # docstring 行集合（ast 的 Constant.str 节点行范围）
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            if node.end_lineno:
                doc_lines.update(range(node.lineno, node.end_lineno + 1))

    # 父节点映射（判 `round(x, n)` 的精度位 / `+= 1` 的计数操作数）
    parents: dict[int, ast.AST] = {}
    for par in ast.walk(tree):
        for child in ast.iter_child_nodes(par):
            parents[id(child)] = par

    seen: set[tuple[int, float]] = set()
    out: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            ln = node.lineno
            if ln in doc_lines:
                continue
            key = (ln, float(node.value))
            if key in seen:
                continue
            seen.add(key)
            ctx = lines[ln - 1] if 1 <= ln <= len(lines) else ""
            ctx = ctx.strip()[:120]
            par = parents.get(id(node))
            fname = ""
            if isinstance(par, ast.Call):
                if isinstance(par.func, ast.Name):
                    fname = par.func.id
                elif isinstance(par.func, ast.Attribute):
                    fname = par.func.attr
            structural = (
                # `round(x, n)` / `np.round(x, n)` 的精度位
                (bool(fname) and fname == "round" and len(par.args) >= 2 and par.args[1] is node)
                # `+= 1` 计数推进的操作数（int）
                or (isinstance(par, ast.AugAssign) and isinstance(node.value, int))
                # 负哨兵（`-1` 等，int）
                or (isinstance(par, ast.UnaryOp) and isinstance(par.op, ast.USub)
                    and isinstance(node.value, int))
                # 下标 / 切片索引
                or isinstance(par, (ast.Slice, ast.Subscript))
                # 内建守卫参数（`max(1, …)` / `range(…)` 等）
                or (bool(fname) and fname in ("max", "min", "range", "enumerate"))
                # 十六进制掩码（行内出现等值 0x… 字面量，如 0x0F ↔ 15）
                or (isinstance(node.value, int) and _hex_mask_in(ctx, node.value))
            )
            out.append({
                "file": _rel(path),
                "line": ln,
                "value": node.value,
                "ctx": ctx,
                "noise": _noise_reason(node.value, ctx, structural=structural),
            })
    return out


def _classify_literal(rec: dict) -> tuple[str, str]:
    ctx = rec["ctx"]
    for pat, cls, why in _LITERAL_CTX:
        if re.search(pat, ctx, re.IGNORECASE):
            return cls, why
    return "UNCLASSIFIED", "无上下文关键词命中"


# ---------------------------------------------------------------------------
# 3) 交叉校验：本工具的分类表 vs `scaling_rescale.py` 的四张表
# ---------------------------------------------------------------------------

def _cross_check() -> tuple[list[str], list[str]]:
    """返回 (不一致项, 说明)。不一致 = 两份清单对同一字段给了不同类。"""
    try:
        from experiments import scaling_rescale as sr
    except Exception as exc:                                  # pragma: no cover
        return [], [f"⚠️ 无法导入 `experiments/scaling_rescale.py`：{type(exc).__name__}"]
    other: dict[str, str] = {}
    for grp, name in sr._RATE_FIELDS:
        other[f"{grp}.{name}"] = RATE
    for grp, name in sr._DURATION_FIELDS:
        other[f"{grp}.{name}"] = DURATION
    for grp, name in sr._PROB_FIELDS:
        other[f"{grp}.{name}"] = PROB
    for grp, name in sr._DECAY_FIELDS:
        other[f"{grp}.{name}"] = DECAY
    bad: list[str] = []
    for full, cls in sorted(other.items()):
        mine = KNOWN.get(full)
        if mine is None:
            bad.append(f"{full}: rescale={cls} ｜ 本工具**未收录** ⇒ 漏标风险")
        elif mine != cls:
            if full in KNOWN_DIVERGENCE:
                continue          # 已知待裁定项，单独报（不算"冲突"）
            bad.append(f"{full}: rescale={cls} ≠ 本工具={mine} ⇒ **分类冲突**")
    # 反向：本工具收录了但 rescale 没有（正常，只要不是 tick 面额）
    return bad, []


def main() -> None:
    ap = argparse.ArgumentParser(description="tick 面额机械化清点（P0.0/A2）")
    ap.add_argument("--json", default="", help="同时落 JSON 的路径（空 = 不落）")
    ap.add_argument("--md", default="", help="同时落 Markdown 表的路径（空 = 不落）")
    ap.add_argument("--no-literals", action="store_true",
                    help="跳过字面量扫描（快；**覆盖自检仍恒跑**，覆盖失败照常 fail-loud）")
    a = ap.parse_args()

    print("=" * 92)
    print("tick 面额机械化清点（P0.0 / A2）")
    print("=" * 92)

    # ---------- 3.0 覆盖自检（**恒跑 + fail-fast**，与 `--no-literals` 解耦）----------
    # 踩过的坑：曾把 glob 派生放进 `if not a.no_literals:` ⇒ `--no-literals` 时
    # 连覆盖检查都不跑 ⇒ 目录被移走也 rc=0（静默）。覆盖是"工具还能不能用"的前提，必须每次验。
    try:
        scan_files = _derive_scan_files()
    except CoverageError as exc:
        print("\n【3.0】扫描覆盖自检（**glob 派生**，恒跑，与 `--no-literals` 无关）")
        print(f"        {exc}")
        print("        ⇒ **覆盖失败 = 硬失败（退出码 2）**：报告不可信，拒绝继续清点。")
        sys.exit(2)
    print("\n【3.0】扫描覆盖自检（**glob 派生**，恒跑，与 `--no-literals` 无关）")
    print(f"        ✅ {_scan_summary(scan_files)}")

    # ---------- 3.1 config 字段 ----------
    wired = _wired_counts()
    rows: list[dict] = []
    for grp, name, dv in _iter_config_fields():
        cls, why = _suggest(grp, name)
        # 未接线字段：不承载 tick 面额 ⇒ 单列，不算"未分类"
        if wired.get(name, 0) == 0 and f"{grp}.{name}" not in KNOWN:
            cls, why = UNWIRED, "**引擎/世界中从未出现** ⇒ 不承载 tick 面额（不参与重标）"
        rows.append({"scope": "config", "full": f"{grp}.{name}", "value": dv,
                     "cls": cls, "why": why})
    rows.sort(key=lambda r: (r["cls"], r["full"]))

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["cls"]] = counts.get(r["cls"], 0) + 1

    print(f"\n【3.1】SimConfig 数值字段：**{len(rows)}** 个")
    for cls in (RATE, DURATION, DECAY, PROB, INVARIANT, UNWIRED, "UNCLASSIFIED"):
        print(f"        {cls:<13} {counts.get(cls, 0)}")
    unclass = [r for r in rows if r["cls"] == "UNCLASSIFIED"]
    print(f"\n  ⇒ **未分类项 = {len(unclass)}**"
          + ("  ✅（任务书 A2 验收：未分类项 = 0）" if not unclass else "  🔴 必须人工判"))

    if unclass:
        print("\n  🔴 未分类明细（人工判后请补进 `KNOWN`）：")
        for r in unclass:
            print(f"     · {r['full']:<52} 默认={r['value']!r}")

    unwired = [r for r in rows if r["cls"] == UNWIRED]
    if unwired:
        print(f"\n  ⚪ 未接线字段 {len(unwired)} 个（**显著报出，禁静默**；不参与重标）：")
        for r in unwired:
            print(f"     · {r['full']:<52} 默认={r['value']!r}  — {r['why']}")

    # ---------- 3.2 字面量（**线索区**；`--no-literals` 只跳过本节）----------
    lit_rows: list[dict] = []
    clue: list[dict] = []
    if not a.no_literals:
        for p in scan_files:
            lit_rows.extend(_scan_literals(p))
        for r in lit_rows:
            cls, why = _classify_literal(r)
            r["cls"] = cls
            r["why"] = why
        noise = [r for r in lit_rows if r.get("noise")]
        clue = [r for r in lit_rows if not r.get("noise")]
        hit = [r for r in clue if r["cls"] in (DURATION, DECAY, PROB)]
        print(f"\n【3.2】字面量（线索区）")
        print(f"        扫描总数 {len(lit_rows)}"
              f"｜🔇 噪声过滤 {len(noise)}（索引/形状/协议/`round(x,n)` 精度/`+=` 计数；仍落 JSON）"
              f"｜净剩 {len(clue)}｜其中带关键词 **{len(hit)}**")
        print("        ⚠️ 字面量的上下文启发式**弱**于字段名 ⇒ 只作线索，必须人工复核。")
        big = sorted(hit, key=lambda r: (r["file"], r["line"]))[:15]
        if big:
            print(f"\n        线索区中的 DURATION/DECAY/PROB（前 15 / 共 {len(hit)}）：")
            for r in big:
                print(f"          {r['file']}:{r['line']:<6} {r['cls']:<9}"
                      f" {r['value']:<8} ｜ {r['ctx']}")
        else:
            print("\n        （线索区无 DURATION/DECAY/PROB 命中 ⇒ 真 tick 常数应已全在 config）")

    # ---------- 3.3 交叉校验 ----------
    bad, warn = _cross_check()
    print("\n【3.3】交叉校验：本工具 vs `experiments/scaling_rescale.py`")
    for w in warn:
        print(f"        {w}")
    if bad:
        print(f"        🔴 **{len(bad)} 项不一致**：")
        for b in bad:
            print(f"          · {b}")
    else:
        print("        ✅ 两份清单对**所有共有字段**分类一致")
    if KNOWN_DIVERGENCE:
        print(f"        ⚪ 已知语义分歧 {len(KNOWN_DIVERGENCE)} 项（**待所有者裁定**，不计冲突）：")
        for full, (mine, theirs, why) in KNOWN_DIVERGENCE.items():
            print(f"          · {full}: 本工具={mine} vs rescale={theirs}")
            print(f"            {why}")

    # ---------- 3.4 已知易错点 ----------
    print("\n【3.4】已知易错点（强制提示，来自任务书 §三 A2 + 我们的踩坑）")
    for full, note in NOTES.items():
        print(f"        · {full}\n          {note}")

    # ---------- 3.5 汇总 + 四类改法 ----------
    print("\n【3.5】四类改法速查（**必须四类**）")
    print("        RATE      ⇒ `x * k`")
    print("        DURATION  ⇒ `x / k`（int 取整且 ≥1）")
    print("        DECAY     ⇒ `x ** k`")
    print("        PROB      ⇒ `1 - (1 - x) ** k`   🔴 上一版手工清单**漏掉的正是这一类**")
    print("        INVARIANT ⇒ 不动")

    total_unclass = len(unclass)
    n_findings = total_unclass + len(bad)
    print(f"\n⇒ 汇总：config 未分类 {total_unclass} 项"
          + ("；交叉校验通过 ✅" if not bad else f"；交叉校验 🔴 {len(bad)} 项冲突"))
    print("   （字面量项为**线索**，人工复核后补进 `KNOWN` 即可消除）")

    if a.json:
        payload = {
            "config": rows,
            "literals": lit_rows,
            "cross_check": bad,
            "unclassified": total_unclass,
            "scan_files": [_rel(p) for p in scan_files],
            "coverage": {"dirs": list(_SCAN_DIRS),
                         "min_files_per_dir": _SCAN_MIN_FILES_PER_DIR,
                         "min_files_total": _SCAN_MIN_FILES_TOTAL,
                         "n_files": len(scan_files),
                         "noise_filtered": len([r for r in lit_rows if r.get("noise")]),
                         "error": ""},
        }
        Path(a.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已落 JSON：{Path(a.json).as_posix()}")

    if a.md:
        lines = ["# tick 面额清点（P0.0 / A2）", "",
                 f"- 扫描覆盖：{_scan_summary(scan_files)}",
                 f"- config 未分类：{total_unclass}；交叉校验不一致：{len(bad)}",
                 "", "## config 字段", "", "| 字段 | 默认值 | 类 | 依据 |", "|---|---|---|---|"]
        for r in rows:
            lines.append(f"| `{r['full']}` | `{r['value']!r}` | {r['cls']} | {r['why']} |")
        if clue:
            lines += ["", f"## 字面量线索区（净剩 {len(clue)} / 总扫 {len(lit_rows)}）", "",
                      "| 文件:行 | 值 | 建议类 | 上下文 |", "|---|---|---|---|"]
            for r in clue:
                ctx = r["ctx"].replace("|", "\\|")
                lines.append(f"| `{r['file']}:{r['line']}` | `{r['value']}` | {r['cls']} | `{ctx}` |")
        Path(a.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已落 Markdown：{Path(a.md).as_posix()}")

    # 退出码（T6/A2b 语义）：0 = 干净；1 = 审计发现（未分类/交叉校验冲突）；2 = 覆盖硬失败（见 §3.0，fail-fast）
    sys.exit(0 if n_findings == 0 else 1)


if __name__ == "__main__":
    main()
