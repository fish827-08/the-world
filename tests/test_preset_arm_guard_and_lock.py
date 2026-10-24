"""C 步预设的两类**硬守卫**（2026-09-16 首跑两处真实事故的回归）：

1. 🔴 **oracle 专属参数不得套给非 oracle 臂** —— 首跑 `cstep1a` 把 `donation=1.0` 放在
   `fixed` 里 ⇒ 套给了 `zero` 臂 ⇒ `a4` 守卫硬拒（`rc≠0`，无产出）⇒ **18 run 只起了 12 个**。
   该类错误必须在**启动前**（CI）被拦，而不是靠数 CSV 发现。
2. 🔴 **批跑器单实例锁** —— 本环境会把带副作用的命令执行两次，实测因此起了**两个
   batch_runner**（24 个子进程写同一批 CSV/快照）⇒ 数据不可信。锁必须能拒第二个实例。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import batch_runner as br  # noqa: E402

# a4 中「仅 --arm oracle 可用」的开关（收窄于 a4 的硬校验）
ORACLE_ONLY = {"--donation", "--oracle-donation", "--oracle-persistence",
               "--gain-multiplier", "--m", "--calibration-arm"}


def _arm_of(cmd: list[str]) -> str | None:
    return cmd[cmd.index("--arm") + 1] if "--arm" in cmd else None


# ---------------------------------------------------- 守卫 1：oracle 参数 × 臂类型

@pytest.mark.parametrize("name", sorted(br.PRESETS))
def test_oracle_only_flags_never_reach_non_oracle_arms(name):
    """🔴 回归（首跑实测）：非 oracle 臂**不得**出现任何 oracle 专属开关。

    `a4` 对此**硬失败**（防静默传参）⇒ 一犯就是"部分 run 起不来"且**不报错**（只在结果里缺）。
    """
    for r in br.preset_runs(name, Path(".")):
        arm = _arm_of(r.cmd)
        if arm == "oracle":
            continue
        bad = sorted(ORACLE_ONLY & set(r.cmd))
        assert not bad, (f"{name}/{r.name}（arm={arm}）带了 oracle 专属开关 {bad} ⇒ "
                         f"a4 会硬拒 ⇒ 该 run 无产出（首跑实测事故）")


def test_known_oracle_arms_do_carry_their_flags():
    """反向：oracle 臂**必须**带齐自己的开关（否则是另一个方向的静默错）。"""
    runs = {r.name: r for r in br.preset_runs("cstep1a", Path("."))}
    m13 = runs["oracle_m1.3_s42"]
    assert _arm_of(m13.cmd) == "oracle"
    assert {"--donation", "--gain-multiplier", "--calibration-arm"} <= set(m13.cmd)


# ---------------------------------------------------- 守卫 2：单实例锁

def test_single_instance_lock_refuses_second(monkeypatch, tmp_path):
    """锁：第一个实例拿到；第二个**被拒**并给出可读原因；释放后可再拿。"""
    monkeypatch.setattr(br, "LOCK_PATH", tmp_path / ".batch_runner.lock")
    assert br.acquire_lock("p1") is None, "首个实例应拿到锁"
    msg = br.acquire_lock("p2")
    assert msg and "已有实例在跑" in msg, "第二实例必须被拒"
    assert "p1" in msg, "占用描述须含占用者信息（便于判断能否接管）"
    br.release_lock()
    assert br.acquire_lock("p3") is None, "释放后应能重新获取"
    br.release_lock()


def test_stale_lock_is_taken_over(monkeypatch, tmp_path):
    """陈旧锁（持有者 PID 已死）⇒ 自动接管，避免"僵尸锁永久堵路"。"""
    monkeypatch.setattr(br, "LOCK_PATH", tmp_path / ".batch_runner.lock")
    br.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    br.LOCK_PATH.write_text(json.dumps({"pid": 999999, "preset": "dead",
                                        "started": "1970-01-01 00:00:00"}),
                            encoding="utf-8")
    assert br.acquire_lock("fresh") is None
    info = json.loads(br.LOCK_PATH.read_text(encoding="utf-8"))
    assert info["preset"] == "fresh"
    br.release_lock()


def test_corrupt_lock_file_is_taken_over(monkeypatch, tmp_path):
    """锁文件损坏（非法 JSON）不得让批跑永久无法启动。"""
    monkeypatch.setattr(br, "LOCK_PATH", tmp_path / ".batch_runner.lock")
    br.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    br.LOCK_PATH.write_text("not-json", encoding="utf-8")
    assert br.acquire_lock("x") is None
    br.release_lock()


def test_force_lock_overrides_live_holder(monkeypatch, tmp_path):
    """`--force-lock`：确知持有者已死时可强行接管。"""
    monkeypatch.setattr(br, "LOCK_PATH", tmp_path / ".batch_runner.lock")
    assert br.acquire_lock("a") is None
    assert br.acquire_lock("b", force=True) is None   # force ⇒ 允许多占
    info = json.loads(br.LOCK_PATH.read_text(encoding="utf-8"))
    assert info["preset"] == "b"
    br.release_lock()


def test_release_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr(br, "LOCK_PATH", tmp_path / ".batch_runner.lock")
    br.release_lock()          # 不存在也不得抛
    br.release_lock()
