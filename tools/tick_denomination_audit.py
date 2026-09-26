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
    python3 tools/tick_denomination_audit.py                 # 全量，落 Markdown 表
    python3 tools/tick_denomination_audit.py --json out.json # 同时落 JSON（机器可读）
    python3 tools/tick_denomination_audit.py --no-literals   # 只看 config 字段（快）
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

#: 明显不是 tick 面额的数字（维度/索引/开关）⇒ 直接跳过，不干扰"未分类=0"
_LITERAL_SKIP = re.compile(
    r"^\s*(#|$)"                       # 空行 / 注释行
)

#: 这些文件里绝大多数数字是索引/形状/协议常量，逐行判会淹没真信号
_SCAN_FILES = ["simulation/sphere_engine.py", "simulation/config.py",
               "world/resources.py", "world/signal_field.py"]


def _scan_literals(path: Path) -> list[dict]:
    """用 `ast` 抽数字字面量 + 所在行文本（**排除注释与 docstring**）。"""
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()

    # docstring 行集合（ast 的 Constant.str 节点行范围）
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            if node.end_lineno:
                doc_lines.update(range(node.lineno, node.end_lineno + 1))

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
            out.append({
                "file": str(path.relative_to(ROOT)),
                "line": ln,
                "value": node.value,
                "ctx": ctx.strip()[:120],
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
    ap.add_argument("--md", default="", help="落 Markdown 表的路径（空 = 只打印）")
    ap.add_argument("--no-literals", action="store_true", help="跳过字面量扫描（快）")
    a = ap.parse_args()

    print("=" * 92)
    print("tick 面额机械化清点（P0.0 / A2）")
    print("=" * 92)

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

    # ---------- 3.2 字面量扫描 ----------
    lit_rows: list[dict] = []
    if not a.no_literals:
        for rel in _SCAN_FILES:
            p = ROOT / rel
            if p.is_file():
                lit_rows.extend(_scan_literals(p))
        # 只保留"看起来像 tick 面额"的（有上下文关键词），其余不列（否则淹没）
        for r in lit_rows:
            cls, why = _classify_literal(r)
            r["cls"] = cls
            r["why"] = why
        keyed = [r for r in lit_rows if r["cls"] != "UNCLASSIFIED"]
        hit = [r for r in lit_rows if r["cls"] in (DURATION, DECAY, PROB)]
        print(f"\n【3.2】字面量（{', '.join(_SCAN_FILES)}）")
        print(f"        扫描总数 {len(lit_rows)}｜有上下文关键词 {len(keyed)}"
              f"｜**建议 RATE/DURATION/DECAY/PROB 的 {len(keyed)}**")
        print("        ⚠️ 字面量的上下文启发式**弱**于字段名 ⇒ 只作线索，必须人工复核。")
        big = sorted(hit, key=lambda r: -r["line"])[:15]
        if big:
            print("\n        最可疑的 DURATION/DECAY/PROB 字面量（前 15）：")
            for r in big:
                print(f"          {r['file']}:{r['line']:<6} {r['cls']:<9}"
                      f" {r['value']:<8} ｜ {r['ctx']}")

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
    print(f"\n⇒ 汇总：config 未分类 {total_unclass} 项"
          + ("；交叉校验通过 ✅" if not bad else f"；交叉校验 🔴 {len(bad)} 项冲突"))
    print("   （字面量项为**线索**，人工复核后补进 `KNOWN` 即可消除）")

    if a.json:
        Path(a.json).write_text(
            json.dumps({"config": rows, "literals": lit_rows,
                        "cross_check": bad, "unclassified": total_unclass},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n已落 JSON：{a.json}")

    # 退出码：未分类 = 0 且无冲突 ⇒ 0；否则 1（fail-loud，可进 CI）
    sys.exit(0 if (total_unclass == 0 and not bad) else 1)


if __name__ == "__main__":
    main()
