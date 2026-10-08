"""`s0_kernel` —— ENERGY-CLOSE 步A 的**合成闭环世界**与双臂 S0 的共用骨架。

零行为面：本包**不被 `simulation/` 导入**（`tests/test_s0_kernel.py` 有守卫断言），
因此对现役引擎的逐字节等价门毫无影响；它只服务两件事——
1. 给 A2 真·守恒断言一个**能设硬零门**的容器（镜 §四 三段式第 1 段）；
2. 给双臂 S0 一个同 RNG 流形制的 harness（PI 若裁"修底盘"则退回 1 的夹具）。

设计红线（写进代码而非文档）：
- **实测账只读世界真值、通道账只读账本**（`Books.measured_delta_sigma` / `Books.predicted_delta`），
  唯一会合点是 `Books.close()` ⇒ 杜绝"由构造恒真"的尾差假绿（`_audit_*:521` / test_o7 同族）。
- **未登记的写能量动作当场 raise**（fail-loud）：旁路 `apply()` ⇒ `ChannelViolation`；
  未登记真实写点 ⇒ `close()` 非零（变异 M1–M3/M5 证明）。
- **不消费 RNG（除 `harness`）**：`accounts/channels/books/world/closed_loop/arms` 全不 import `random`/`numpy.random`，同参数跑两次逐位相同；**唯一**消费 RNG 的是 `harness`（它要模拟抽样），代价是必须**同 seed 逐位复现**（`tests/test_s0_harness.py` 钉字节级一致）。

模块分层（单向依赖，无环）：`accounts` → `channels` → `books` → `world` → `closed_loop`；获取函数层 `arms`（纯函数）→ 出货层 `harness`（`run_arm`）。
"""

from __future__ import annotations

from .accounts import ACCOUNTS, ACCOUNT_UNITS, ENERGY_ACCOUNTS, MASS_ACCOUNTS, to_energy
from .arms import (F_FORMS, INCONCLUSIVE_EXIT, SIGMA_C_PROVISIONAL, WINDOW_GUARD,
                   WINDOW_MEASURED, WINDOW_QUOTED_R197, Landscape, arm_specs,
                   assert_arms_comparable, attack_p, clamp_is_active, declare_inconclusive_exit,
                   default_predation, f_a1, f_gn, has_interior_valley, is_inside_valley_window,
                   nu_local, nu_proxy, success_s, w_a, w_b, w_b0, w_mn, w_mr)
from .books import DOMAIN_ACCOUNTS, Books, resolve_domain
from .channels import DIS, ESC, INJ, TRF, Channel, ChannelRegistry, ChannelViolation
from .harness import Demography, S0World, run_arm
from .world import (ConservationViolation, World, WorldOptions, check_tick, default_registry,
                    run)

__all__ = [
    "ACCOUNTS", "ACCOUNT_UNITS", "ENERGY_ACCOUNTS", "MASS_ACCOUNTS", "to_energy",
    "DOMAIN_ACCOUNTS", "Books", "resolve_domain",
    "INJ", "TRF", "DIS", "ESC", "Channel", "ChannelRegistry", "ChannelViolation",
    "ConservationViolation", "World", "WorldOptions", "check_tick", "default_registry", "run",
    "F_FORMS", "INCONCLUSIVE_EXIT", "SIGMA_C_PROVISIONAL", "WINDOW_GUARD", "WINDOW_MEASURED",
    "WINDOW_QUOTED_R197", "Landscape", "arm_specs", "assert_arms_comparable", "attack_p",
    "clamp_is_active", "declare_inconclusive_exit", "default_predation", "f_a1", "f_gn",
    "has_interior_valley", "is_inside_valley_window", "nu_local", "nu_proxy", "success_s",
    "w_a", "w_b", "w_b0", "w_mn", "w_mr",
    "Demography", "S0World", "run_arm",
]
