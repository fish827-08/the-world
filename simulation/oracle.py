"""V-1 oracle 正向对照（R39 / D-8）：利益对齐型（Lewis 共利）能量回馈。

一句话（`_share/规格-V1-oracle引擎级-20260913.md` §〇）
--------------------------------------------------------
**oracle = 给"接收者因信号而受益"这件事，接一条回给发送者的能量通道；
内容无关、零新增随机数、能量守恒。**
它不是用来证明语言的，是用来证明"我们的仪器能在选择肯定存在时测到选择"。

设计要点（实现层裁定，与规格的差异已在讨论板登记）
------------------------------------------------
1. **单一实现**：转移核只存在于本文件 `apply_oracle`（规格 §2.5：
   禁止 Rust/Python 两路径各写一遍——`nb[:4]` 事故同型预防）。
   当前 C-8（`use_sim_core=True` + oracle ⇒ 引擎显式报错）使 Rust 路径
   不可达，故引擎只有 Python 一个调用点；将来下沉 Rust 时复用本函数。
2. **归因键 = `_id`，不是槽位索引**：规格骨架写的 `_last_sender[e_flat] = emitters`
   存的是槽位索引——但 `:1149-1177` 死亡压缩会重排槽位，而归因窗口
   （persistence=10 tick）内发送者可能已死 ⇒ 必须存 `_id`，转移时按
   id→slot 解析（内评 V-7 同款教训：禁槽位键控）。
3. **窗口判定按"剩余寿命"语义**：`SignalField._age` 是**剩余寿命**
   （写入时=duration=50，每 tick −1，到 0 清除）⇒ 规格 §2.3 的
   `_age <= persistence` 方向反了（那选中的是**即将过期**的旧信号）。
   正确语义："最近 persistence tick 内写入" ⇔ **`age >= duration - persistence`**。
4. **C-9 保本封顶（每发送者终身）**：规格给的选项是"(a) 每发射总额封顶 /
   (b) 每发射单次封顶"。精确的"每发射"记账需要发射事件簿（cell×sender×tick），
   状态重且无必要——本实现取**更严格的终身保本**：
   `剩余额度 = EMISSION_COST × 累计发射次数 − 累计已获回馈`（按 `_id` 记账）。
   ⇒ `oracle_return_ratio = Σ回馈 / Σ(发射×成本) ≤ 1` **由构造保证**（O-7 上界），
   oracle 只可能"补偿"不可能"补贴"。比规格 (a) 更保守，方向一致。
5. **零新增 RNG、能量守恒、不自反馈、不使能量为负**（C-1/C-2/C-3）。
6. 🔴 **F-R18（2026-09-15 修）方向 = `R → S`（接收者回付发送者）**：原实现写成了
   `energy[S] -= g; energy[R] += g`（**发送者倒贴**），与本文件 docstring「回给发送者」、
   引擎 `budget` 注释「已获回馈」、`oracle_return_ratio` 命名、以及"**正向**对照"的
   验收判据（"oracle 开 ⇒ g15 上升"）**全部相反** —— 属"规格骨架误被逐字实现"，
   该装置实际是「惩罚发射者」⇒ 对 `g15` 是**负选择**，原理上不可能达成验收判据。
   现修正为 **接收者（觅到食物者，付得起）付款、发送者收款**；C-2 **仍守恒**（纯再分配，
   无注入）。详见 `_share/规格-V1-oracle引擎级-20260913.md` 勘误节与讨论板 R102/F-R18。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from simulation.config import SIGNAL_COST  # 单一真源（C5）；EMISSION_COST 为兼容别名

# 信号发射成本（与 sphere_engine.py 内 `SIGNAL_COST = 0.1` 同源；
# 此处作为 oracle 保本封顶的记账基准。若改发射成本，两处必须同步。）
EMISSION_COST = SIGNAL_COST   # 兼容旧名；单一真源见 config.SIGNAL_COST（C5）


def attribution_ok(sig_age: NDArray, signal_duration: int, persistence: int) -> NDArray:
    """归因窗口判定：最近 `persistence` tick 内写入过信号。

    `SignalField._age` 是**剩余寿命**（write 置 duration，每 tick −1），
    故"写入于最近 persistence tick" ⇔ `age >= duration - persistence`。
    （规格 §2.3 原式 `age <= persistence` 按剩余寿命语义是反的，见模块 docstring #3。）
    persistence=0 ⇒ 仅本 tick（age == duration）。
    """
    return np.asarray(sig_age) >= (int(signal_duration) - int(persistence))


def apply_oracle(
    *,
    energy: NDArray[np.float64],
    receiver_slots: NDArray[np.int64],
    sender_slots: NDArray[np.int64],
    budget: NDArray[np.float64],
    donation: float,
    ledger: dict | None = None,
) -> tuple[float, int, NDArray[np.int64], NDArray[np.float64]]:
    """纯转移核：**R → S** 能量转移（接收者回付发送者；F-R18 方向修正）。

    参数
    ----
    energy : 引擎工作能量数组（就地修改）
    receiver_slots / sender_slots : 已解析好的**当前槽位**（发送者 -1 = 已死/来路不明）
    budget : 每对转移中**收款方（发送者）**的剩余保本额度（≤0 ⇒ 不转移）
    donation : 单次转移量

    ⚠️ **方向约定（F-R18）**：`receiver_slots` = **付款方**（接收者/移动者，刚觅到食物），
    `sender_slots` = **收款方**（信号写入者）。与 `oracle_return_ratio`、
    「已获回馈」、「保本封顶」等既有命名一致。

    返回
    ----
    (总转移量, 次数, 成交**收款者**槽位, 成交金额) —— 供引擎按 `_id` 累计 `_oracle_gain`
    （= 发送者**已获回馈**）。

    不变量：Σenergy 守恒（C-2：**纯再分配、零注入**）；零 RNG；`energy[付款方]` 不为负；
    **同一收款者对多笔顺序累加额度**（确定性：接收者数组顺序，F-R16 修复保留）。

    `ledger`（可选，**仅写入、不参与任何计算**）—— 供引擎做 R102 §四 / 内评 §四 要求的
    **对账与偿付约束量化**；填 `payer_paid` / `sender_received` / `payer_trunc_*` /
    `budget_trunc_*` / `payer_broke_n` / `budget_exhausted_n` / `applied_idx`。
    ⚠️ 该参数**不改变随机流、不改变转移结果**（纯观测），引擎侧有「与无仪器的同配置跑
    逐位一致」的对照检查。
    """
    total = 0.0
    kept_s: list[int] = []
    kept_g: list[float] = []
    n = 0
    # R102 条件 1/3（修正版）+ 内评 §四：对账与偿付约束量化（纯观测计数）
    payer_trunc_n = 0
    payer_trunc_amt = 0.0
    budget_trunc_n = 0
    budget_trunc_amt = 0.0
    payer_broke_n = 0          # 付款方能量为 0 ⇒ 整笔跳过
    budget_exhausted_n = 0     # 收款方额度用尽 ⇒ 整笔跳过
    applied_idx: list[int] = []
    # F-R16：每**收款者**（发送者）的剩余额度（首次出现时以配对 budget 为初始值，之后逐笔扣减）
    remaining: dict[int, float] = {}
    for i, (s, r, b) in enumerate(zip(
        sender_slots.tolist(), receiver_slots.tolist(), budget.tolist()
    )):
        if s < 0 or r < 0:
            continue
        if s not in remaining:
            remaining[s] = float(b)
        rem = remaining[s]
        if rem <= 0:
            budget_exhausted_n += 1
            continue
        # F-R18：偿付能力约束在**付款方**（接收者 r）身上
        avail = float(energy[r])
        if avail <= 0:
            payer_broke_n += 1
            continue
        d = float(donation)
        g = min(d, rem, avail)
        if g <= 0:
            continue
        if g < d:                       # 部分支付：归因到更紧的那一侧
            short = d - g
            if avail <= rem:            # 付款方付不起 ⇒ **偿付截断**
                payer_trunc_n += 1
                payer_trunc_amt += short
            else:                       # 收款方额度上限 ⇒ 额度截断
                budget_trunc_n += 1
                budget_trunc_amt += short
        energy[r] -= g          # 接收者付款（付款方偿付能力受限）
        energy[s] += g          # 发送者收款（F-R18 方向修正）
        remaining[s] = rem - g
        total += g
        kept_s.append(int(s))
        kept_g.append(float(g))
        applied_idx.append(i)
        n += 1
    if ledger is not None:
        ledger["payer_paid"] = total
        ledger["sender_received"] = total
        ledger["payer_trunc_n"] = payer_trunc_n
        ledger["payer_trunc_amt"] = payer_trunc_amt
        ledger["budget_trunc_n"] = budget_trunc_n
        ledger["budget_trunc_amt"] = budget_trunc_amt
        ledger["payer_broke_n"] = payer_broke_n
        ledger["budget_exhausted_n"] = budget_exhausted_n
        ledger["applied_idx"] = applied_idx
    return (
        total,
        n,
        np.asarray(kept_s, dtype=np.int64),
        np.asarray(kept_g, dtype=np.float64),
    )
