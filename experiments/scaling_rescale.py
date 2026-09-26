"""时间压缩重标定（14.10 / 评审 P0.0）—— 把「每昼夜 tick 数 D」当成纯分辨率旋钮。

为什么需要
----------
锚点把三个量锁在一起：`cols = v_max × L_max`、`L_max = 8 × D`（g3 拉满、lifespan_mult=1）。
把 D 从 2400 压到 480（k=5）能让**世代所需 tick 数 ÷5**（⇒ 墙钟 ÷5），
但**所有以 tick 计价的常数若不跟着改，它们相对「昼夜」的含义会自动放大 5 倍**
——这是评审 §2.4 点出的、会"改了科学却不自知"的隐患。

按量纲分三类（`experiments/tick_scaling_audit.py` 逐条清点过）：

| 类别 | 改法 | 例 |
|---|---|---|
| **每 tick 速率** | **× k** | `regrowth_rate` `base_metabolism` `move_cost` `photo_max` `homeo_upkeep` `wound_heal_rate` `fruit.charge_rate/eat_rate` `pleasure.alpha` |
| **tick 时长** | **÷ k** | `rotation_period`(=D) `corpse_decay_ticks` `rest_ticks` `dead_regen_ticks` `learning_maturity_ticks` `ars.{fast_tau,slow_tau,giveup}` `pleasure.expectation_size` `oracle.persistence` `signals.duration` |
| **每 tick 衰减** | **^ k** | `pleasure.valence_decay` `pleasure.arousal_decay` |
| **量 / 比例 / 计数** | **不变** | `max_energy` `initial_energy` `capacity_per_area` `eat_efficiency` `assim_*` 全部 `*_frac` `*_mult` `patch_count` `patch_radius` `perception_radius` `attack_range` 等 |

⚠️ `signals.duration` **目前硬编码在引擎里**（`SignalField(self.world, duration=50)`），
   不在 `SimConfig` 上 ⇒ 本模块通过返回值告知调用方「构造后需手动设置」。

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
    ("pleasure", "baseline_rate"),        # 🔴 青梧 A2 修正：EWMA 慢漂移速率（引擎 :4602
                                          #   baseline×(1-r)+valence×r，指数滑移）⇒ ×k，
                                          #   **不是** 概率 1-(1-p)^k（天平/云归分类表归 PROB 待裁定）
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

# 每 tick 概率（独立试次）⇒ **p' = 1 − (1 − p)^k**（不是 ×k、也不是不变）
#   🔴 这一类是上一版清单**漏掉**的，正是它导致 k=5 时种群 +82% 的系统性漂移。
_PROB_FIELDS: list[tuple[str, str]] = [
    ("predation", "attack_prob_coef"),
    ("fruit", "germination_prob"),
    ("fruit", "excretion_prob"),
    ("fruit", "seed_intake_prob"),
]
# 🔴 青梧 A2 修正：`info_structure.codebook_mutation_rate` 是**每繁殖事件概率**
#    （引擎 :3795 每次繁殖逐码位抛硬币）⇒ 时间压缩不改变每事件概率 ⇒ **不变**，
#    已移出本表。若按 PROB 用 1-(1-p)^k：k=5 时码本突变率 0.02 → 0.096（≈5 倍），
#    会显著加速文化码本漂移——这是真漏标错误。
# 🔴 同理 `genome.mutation_rate`（:3709/:3750）也是每繁殖事件概率 ⇒ 不变。

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
        return {"signals_duration": 50}      # k=1 ⇒ 不改（与默认 50 一致）

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

    return {"signals_duration": int(cfg.signals.duration_ticks)}


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
