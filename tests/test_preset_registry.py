"""F-R32 守卫回归：**同名 preset 不得共存**（dict 字面量会静默覆盖后者/前者）。

事故（2026-09-19，E-027）：两个会话先后登记同名 `cstep3memgrad20`（内容不同），
启动时生效的是文件序靠后的那个 ⇒ 实跑配置（16码）≠ 预检广告的配置（4码）。
`batch_runner` 顶部现有一道**源码扫描守卫**（重复即 RuntimeError）；
本测试钉死三件事：守卫存在 / 守卫的口径与 `PRESETS` 真实键集一致 / 当前无重复。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from experiments import batch_runner as br  # noqa: E402  （能 import = 守卫未触发 = 当前无重复）


def test_preset_guard_exists_in_source():
    src = (ROOT / "experiments" / "batch_runner.py").read_text(encoding="utf-8")
    assert "_preset_names" in src and "F-R32" in src, "同名守卫被删 ⇒ F-R32 复发风险"


def test_guard_scan_agrees_with_real_keys():
    """守卫的正则口径必须与 `PRESETS` 真实键集**一致**（防守卫漂移成摆设）。"""
    src = (ROOT / "experiments" / "batch_runner.py").read_text(encoding="utf-8")
    blk = src[src.index("PRESETS = {"):src.index("\n}", src.index("PRESETS = {"))]
    names = re.findall(r'^    "([a-z0-9_]+)": dict\(', blk, re.M)
    assert set(names) == set(br.PRESETS), (
        f"源码扫描 ({len(names)}) 与 PRESETS 键集 ({len(br.PRESETS)}) 不一致 ⇒ "
        "守卫正则已漂移，抓不到新增/改名")
    assert len(names) == len(set(names)), f"存在同名 preset：{names}"


def test_replication_preset_pins_alphabet():
    """E-027 教训：事故**之后**登记的复测类 preset 必须显式钉字母表（默认值会随纪元漂移）。

    ⚠️ 豁免 `cstep3memgrad20`：它是 E-027 的**实跑凭证**（16 码正是靠"未显式传参 ⇒ 默认"
    才发生的事故本身），必须原样保留、不得补改（R10：已发生的配置不追改）。
    """
    exempt = {"cstep3memgrad20"}
    for name, p in br.PRESETS.items():
        if "memgrad20" in name and name not in exempt:
            fixed = " ".join(p["fixed"])
            assert "signal-alphabet=" in fixed, f"{name} 缺显式 signal-alphabet ⇒ E-027 同型风险"
