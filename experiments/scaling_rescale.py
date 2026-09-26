"""时间压缩重标定（14.10 / 评审 P0.0）—— 把「每昼夜 tick 数 D」当成纯分辨率旋钮。

为什么需要
----------
锚点把三个量锁在一起：`cols = v_max × L_max`、`L_max = 8 × D`（g3 拉满、lifespan_mult=1）。
把 D 从 2400 压到 480（k=5）能让**世代所需 tick 数 ÷5**（⇒ 墙钟 ÷5），
但**所有以 tick 计价的常数若不跟着改，它们相对「昼夜」的含义会自动放大 5 倍**
——这是评审 §2.4 点出的、会"改了科学却不自知"的隐患。

按量纲分**四类**（`tools/tick_denomination_audit.py` 逐条清点过；见下 §分类表）：

| 类别 | 改法 | 例 |
|---|---|---|
| **每 tick 速率** | **× k** | `regrowth_rate` `base_metabolism` `move_cost` `photo_max` `homeo_upkeep` `wound_heal_rate` `fruit.charge_rate/eat_rate` `pleasure.alpha` |
| **tick 时长** | **÷ k** | `rotation_period`(=D) `corpse_decay_ticks` `rest_ticks` `dead_regen_ticks` `learning_maturity_ticks` `ars.{fast_tau,slow_tau,giveup}` `oracle.persistence` `signals.duration_ticks` `organisms.repro_cooldown_gene_scale` |
| **每 tick 概率：日事件** | **1−(1−p)^k** | `pleasure.baseline_rate`（EWMA 时间常数的**精确**解） |
| **每 tick 概率：速率过程** | **min(1, k·p)** | `predation.attack_prob_coef`（保「每天出手次数」；`k·p>0.5` 报饱和告警） |
| **每 tick 衰减** | **^ k** | `pleasure.valence_decay` `pleasure.arousal_decay` |
| **量 / 比例 / 计数 / 每事件概率** | **不变** | `max_energy` `initial_energy` `capacity_per_area` `eat_efficiency` `assim_*` 全部 `*_frac` `*_mult` `patch_count` `patch_radius` `perception_radius` `attack_range`；`info_structure.codebook_mutation_rate`、`genome.mutation_rate`（**每繁殖事件**） |

⚠️ `signals.duration` 原为**引擎内硬编码**（`SignalField(..., duration=50)`）——已由 P0.0 A1
   搬进 `SimConfig.signals.duration_ticks`，本模块的 `apply_post_build` 仍保留兼容设置。

用法
----
    from experiments.scaling_rescale import rescale_config
    c = SimConfig()
    notes = rescale_config(c, k=5)          # 原地改；返回需要构造后手动处理的项
    eng = SphereEngine(c)
    for name, val in notes.items():
        setattr( 目标, name, val )           # 见 apply_post_build
"""
from __future__ import annotations

import sys
from typing import Any

# 每 tick 速率（× k）
_RATE_FIELDS: list[tuple[str, str]] = [
    ("resources", "regrowth_rate"),
    ("organisms", "base_metabolism"),
    ("organisms", "move_cost"),
    ("organisms", "photo_max"),
    ("organisms", "homeo_upkeep"),
    ("organisms", "eat_amount"),          # 每 tick 一次的进食事件量 ⇒ 与速率同量纲
    ("fruit", "charge_rate"),
    ("fruit", "eat_rate"),
    ("corpse_wound", "wound_heal_rate"),
    ("corpse_wound", "wound_heal_energy_cost"),
    ("pleasure", "alpha"),                # RPE 的每 tick 学习率
    # 🔴 `pleasure.baseline_rate` **不在这里** —— 见 `_PROB_FIELDS` 下的裁定说明：
    #   它的正确改法是 PROB 公式（`1−(1−r)^k`），该公式对 EWMA 是**精确解**而非近似。
    ("info_structure", "learning_rate"),
    ("info_structure", "alignment_rate"),
]
# ⚠️ `organisms.move_cost`：**每格成本 / PER_EVENT**（引擎 subpos 路径 :3290 按实际步数
#    比例计费 `move_cost × (st/subdiv)`；整数格路径每移动 1 格扣一次）⇒ 每天移动能耗 =
#    move_cost × 每天格数，**与 k 无关**。本轮保留在 RATE（×k）仅为 A4 判据口径连续
#    （云归同判：KNOWN_DIVERGENCE 待裁定）；若将来按物理语义修正 ⇒ 移出本表。

# tick 时长（÷ k；int 字段取整且至少 1）
_DURATION_FIELDS: list[tuple[str, str]] = [
    ("light", "rotation_period"),
    ("info_structure", "learning_maturity_ticks"),
    ("corpse_wound", "corpse_decay_ticks"),
    ("resource_dynamics", "rest_ticks"),
    ("resource_dynamics", "dead_regen_ticks"),
    ("ars", "fast_tau"),
    ("ars", "slow_tau"),
    ("ars", "giveup"),
    ("oracle", "persistence"),
    # 🔴 P0.0 A1（2026-09-26）：繁殖冷却换算系数原为引擎里硬编码的 `* 60.0`，
    #    现搬进 `SimConfig.organisms`。量纲 = **基因 × 每单位基因的 tick 数**
    #    = tick 时长 ⇒ 时间压缩时 **÷ k**。（不改 ⇒ 冷却相对"世界日"缩短 k 倍。）
    ("organisms", "repro_cooldown_gene_scale"),
    # 🔴 P0.0 A1（2026-09-26）：信号场寿命原为引擎里硬编码的 `duration=50`，
    #    现搬进 `SimConfig.signals`。量纲 = tick 时长 ⇒ **÷ k**。
    ("signals", "duration_ticks"),
]
# ⚠️ `pleasure.expectation_size` **不是时长**：它是 `_expectation` 的**数组宽度**
#    （`(n, expectation_size)`）⇒ 擅自 ÷k 会直接 IndexError。属于"计数类，不变"。

# 每 tick 概率（独立试次），**守恒目标是「每世界日至少发生一次」的概率**
#   ⇒ **p' = 1 − (1 − p)^k**（不是 ×k、也不是不变）
#   🔴 这一类是上一版清单**漏掉**的，正是它导致 k=5 时种群 +82% 的系统性漂移。
#
#   ⚠️ 但「守恒目标」本身要按过程类型选，**两类必须分开**（2026-09-26 追加，见下）：
#     · 本表（PROB）  = 每天**至多一次**的事件（伯努利日事件）⇒ 保 P(≥1/日)
#     · `_RATE_PROB_FIELDS`（CAP）= 速率过程（一天可发生很多次）⇒ **保次数/日**
#   把速率过程塞进 PROB 会**系统性少算次数**（k=5 时 −33%），这是可实测的量级误差。
_PROB_FIELDS: list[tuple[str, str]] = [
    # 🔴 裁定（[所有者·天平] 2026-09-26）：`pleasure.baseline_rate` **留在 PROB**。
    #   青梧主张移入 RATE（×k），但引擎 :4596 是 EWMA 慢漂移
    #   `baseline ← baseline×(1−r) + valence×r`，其**偏差按 (1−r) 每 tick 衰减**。
    #   要求「每世界日的时间常数不变」即 `(1−r')^(D/k) = (1−r)^D` ⇒ **r' = 1−(1−r)^k**
    #   —— 正是 PROB 公式，对 EWMA 是**精确解**；×k 只是它的一阶近似。
    #   数值差：r=0.001、k=5 ⇒ 0.005000 vs 0.004990（**0.2%**）⇒ 非实质分歧。
    ("pleasure", "baseline_rate"),
    ("fruit", "germination_prob"),
    ("fruit", "excretion_prob"),
    ("fruit", "seed_intake_prob"),
]
# 🔴 上面三个 `fruit.seed_*` **是死配置**（全仓除 `config.py` 与本表外**零引用**，
#   种子传播子机制未实现；同族 `fruit.seed_energy` / `fruit.max_seed_carried` 已被
#   清点工具报为 UNWIRED）。它们**留在表里是有意的**：移出会让清点工具的交叉校验
#   失去这一行、反而更难发现。正确修法是让工具的「接线检查」**优先于** `KNOWN`
#   （现在 `KNOWN` 命中就短路，见 A2c 派工）。

# 每 tick 概率里的**速率过程** ⇒ **p' = min(1, k·p)**（守恒「每世界日发生次数」）
#
# 🔴 为什么不能沿用 PROB：PROB 保的是「每天至少一次」，而捕食是速率过程，
#   `p=0.2`、`D=2400` ⇒ 名义 480 次/日。PROB 只给 322.7 次（**−33%**），
#   CAP 给 480.0 次（= k=1）。而真正决定捕食的是「**单格停留期间出手几次**」
#   （停留 = 1/15 日）：k=1 与 CAP 都是 32 次，PROB 只有 21.5 次
#   ⇒ 捕食偏弱 ⇒ 种群偏高。**实测**（`experiments/pred_rate_rule_test.py`）：
#   k=5 尾段相对差 PROB **0.48** → CAP **0.32**（主判据通过）。
#
# ⚠️ **离散化有效域**：CAP 要求 `k·p ≤ 1` 才有解。保留方差的实用上限是
#   **`k·p ≤ 0.5`**（否则 p'→1，出手变成「每 tick 一次」的确定性过程，方差被压掉）。
#   `attack_prob_coef=0.2`、g16/hunger 取 1 ⇒ `p_max=0.2` ⇒ **k ≤ 2.5 才安全**。
#   ⇒ 🔴 **k=5 对捕食开臂不可表示**（p'=1.0 恰好落在边界上，退化）。
#     要么把 k 压到 ≤2.5，要么把 `attack_prob_coef` 一起重标（纪元级，须另立案）。
_RATE_PROB_FIELDS: list[tuple[str, str]] = [
    ("predation", "attack_prob_coef"),
]

# 🔴 青梧 A2 修正（[所有者·天平] 采纳）：`info_structure.codebook_mutation_rate` 是
#   **每繁殖事件概率**（引擎 :3790 每次繁殖逐码位抛硬币，形状 `(K,16)`）⇒ 时间压缩
#   不改变「每事件」概率 ⇒ **不变**，已移出本表。若按 PROB 用 `1−(1−p)^k`：
#   k=5 时码本突变率 0.02 → 0.096（**≈5 倍**），文化码本漂移被静默加速 ⇒ **真漏标**。
# 🔴 同族（同样「每繁殖事件」⇒ 不变，已核引擎 :3707/:3748）：`genome.mutation_rate`。
#   ⚠️ 它现在**不在任何表里**（= INVARIANT，结果正确）；但**清点工具的 `KNOWN` 里也没有**，
#   是靠「不在表里」兜住的 ⇒ 应由 A2c 显式登记（防后人误改）。
#
# 每 tick 衰减（^ k）
_DECAY_FIELDS: list[tuple[str, str]] = [
    ("pleasure", "valence_decay"),
    ("pleasure", "arousal_decay"),
]


def rescale_config(cfg: Any, k: float) -> dict[str, Any]:
    """把 `cfg` 原地重标定为「时间压缩 k 倍」。返回需要构造后手动设置的项。

    参数
    ----
    cfg : SimConfig
        会被原地修改。
    k : float
        时间压缩倍率（`k = 2400 / 新 D`）。`k = 1` ⇒ 不改。

    返回
    ----
    dict : `{"signals_duration": <int>}` —— 信号寿命目前不在 config 上，
           引擎构造后请 `eng.signals.duration = 那个值`。
    """
    if k <= 0:
        raise ValueError(f"k 必须 > 0，收到 {k!r}")
    if abs(k - 1.0) < 1e-12:
        return {"signals_duration": int(cfg.signals.duration_ticks),
                "saturation_warnings": []}      # k=1 ⇒ 不改（= 配置默认值）

    for grp, name in _RATE_FIELDS:
        sub = getattr(cfg, grp)
        setattr(sub, name, float(getattr(sub, name)) * k)

    for grp, name in _DURATION_FIELDS:
        sub = getattr(cfg, grp)
        old = getattr(sub, name)
        new = old / k
        setattr(sub, name, int(round(new)) if isinstance(old, int) else float(new))

    for grp, name in _PROB_FIELDS:
        sub = getattr(cfg, grp)
        p_old = float(getattr(sub, name))
        setattr(sub, name, 1.0 - (1.0 - p_old) ** k)

    # 速率过程的每 tick 概率：`p' = min(1, k·p)`（守恒「每世界日发生次数」）。
    # 🔴 饱和告警：`k·p > 0.5` ⇒ p' 逼近 1，出手变成准确定性过程（方差被压掉）
    #    ⇒ 该 k 与这些参数**组合不可用**，必须降 k 或改参数（禁静默通过）。
    _sat: list[str] = []
    for grp, name in _RATE_PROB_FIELDS:
        sub = getattr(cfg, grp)
        p_old = float(getattr(sub, name))
        p_raw = k * p_old
        setattr(sub, name, min(1.0, p_raw))
        if p_raw > 0.5:
            _sat.append(f"{grp}.{name}: k·p={p_raw:.3f}（k={k:g}, p={p_old:g}）"
                        f" ⇒ p'={min(1.0, p_raw):.3f}，离散化饱和（建议 k ≤ {0.5 / p_old:.2f}）")

    for grp, name in _DECAY_FIELDS:
        sub = getattr(cfg, grp)
        setattr(sub, name, float(getattr(sub, name)) ** k)

    # 「tick 时长」字段的下限保护（AR S giveup=20/5=4 仍合理；k 很大时会逼近 1）
    _mins = {"ars": {"giveup": 1, "fast_tau": 1.0, "slow_tau": 2.0},
             "resource_dynamics": {"rest_ticks": 1, "dead_regen_ticks": 10},
             "oracle": {"persistence": 1},
             "info_structure": {"learning_maturity_ticks": 10},
             "corpse_wound": {"corpse_decay_ticks": 10},
             "light": {"rotation_period": 8},
             "signals": {"duration_ticks": 1}}
    for grp, names in _mins.items():
        sub = getattr(cfg, grp)
        for name, lo in names.items():
            if getattr(sub, name) < lo:
                setattr(sub, name, int(lo) if isinstance(getattr(sub, name), int) else float(lo))

    if _sat:
        print("🔴 离散化饱和告警（速率过程每 tick 概率，K·p > 0.5）：", file=sys.stderr)
        for line in _sat:
            print("   ·", line, file=sys.stderr)

    return {"signals_duration": int(cfg.signals.duration_ticks),
            "saturation_warnings": _sat}


def apply_post_build(eng: Any, notes: dict[str, Any]) -> None:
    """把构造后才知道的目标项应用上去（目前只有信号寿命）。"""
    if "signals_duration" in notes:
        eng.signals.duration = int(notes["signals_duration"])


def ancher_consistent_speed(cols: int, D: int, lifespan_mult: float = 1.0) -> float:
    """锚点反解最大速度：`cols = v_max × 8 × D × lifespan_mult`。"""
    return cols / (8.0 * D * lifespan_mult)


def gain_for(speed_max: float, age_factor_mean: float) -> float:
    """评审 §3.1 的 gain 标定：`gain = v_max / ā`（消除顶格）。"""
    return speed_max / max(age_factor_mean, 1e-9)


def subdiv_for(speed_max: float, r_target: float = 20.0) -> int:
    """评审 §3.2：档位数 R = v_max × subdiv，推荐 R ≥ 20（理想 40）。"""
    return max(1, int(round(r_target / max(speed_max, 1e-9))))
