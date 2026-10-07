"""账户面（"系统"包含什么）——A2 恒等式的定义真源。

现役引擎里能量/生物量散在 5 个载体（步A 通道枚举清单 §〇）：

| 代号 | 载体 | 现役数组 | 单位 |
|---|---|---|---|
E | 活体自由能量 | `_energy` | energy |
S | 胃内容 | `_stomach` | mass |
G | 植物资源池 | `resources._grid` | mass |
C | 尸骸池 | `_corpse_energy` | energy |
F | 果实场 | `_fruit_grid` | energy |

🔴 单位不是装饰：`mass` 账户折算到能量域必须过**唯一**换算点 `eat_efficiency`
（现役引擎的唯一折算点在 `_step_scavenging:1852` 与消化 `:3321`）。
本包把它做成显式参数，杜绝"1 单位尸体吐 3 倍能量"那类单位串账（R163 实测过）。
"""

from __future__ import annotations

ACCOUNTS: tuple[str, ...] = ("E", "S", "G", "C", "F")
ACCOUNT_UNITS: dict[str, str] = {
    "E": "energy", "S": "mass", "G": "mass", "C": "energy", "F": "energy",
}
ENERGY_ACCOUNTS: tuple[str, ...] = tuple(a for a in ACCOUNTS if ACCOUNT_UNITS[a] == "energy")
MASS_ACCOUNTS: tuple[str, ...] = tuple(a for a in ACCOUNTS if ACCOUNT_UNITS[a] == "mass")

#: 砚 20:28 复核门与立场帖 §四·待定 1：A2 量域两候选，本包两者都能出，不替砚裁。
DOMAINS: tuple[str, ...] = ("E", "ALL")


def to_energy(amount: float, account: str, eff: float) -> float:
    """把某账户的量折算到能量域（`mass` 账户 × `eff`）。未知账户当场报错。"""
    if account not in ACCOUNT_UNITS:
        raise KeyError(f"未知账户 {account!r}（可选：{ACCOUNTS}）")
    return amount * eff if ACCOUNT_UNITS[account] == "mass" else amount
