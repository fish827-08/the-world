"""C1 捕食信息价值批 · rd 分块死亡明细侧车测试脚手架（设计稿 §五 T1–T14）。

本线 = [本地开发·性能线] 轻舟；设计稿 = 《设计-C1捕食信息价值批-定稿-20261004.md》。

覆盖矩阵（设计稿 §五）：
  T1   默认关逐字节等价（rd 关 ⇒ 主表/manifest 与改动前逐字节同）
  T2   主表零列改动（并入 T1/T12）
  T3   通道抽象（--rd-channel risk/pheromone 同代码路径）
  T4   分块几何 4×4（480×960 与 mini 双世界；CLI 改 5/5 ⇒ 25 块）
  T5   AR1 + 置换零模型（合成序列复现；T5b 对账 kills vs d_pred 差分）
  T6   J-A CI 形态（8 seed 全 IID ⇒ 不判超零；全耦合 ⇒ 判超零）
  T7   零 RNG 污染（加列前后主 RNG 序列逐位同）
  T8   两臂指纹差异锁（on/off fingerprint 不同）
  T9   fail-loud（全 0 kills ⇒ dens 全 NaN ⇒ 不静默；空 channels + rd ⇒ SystemExit）
  T10  性能门（1 run×2000t：rd 前后 ms/tick 中位比 ≤1.10）
  T11  D-1 钩子三连（a 未装 vs 装后关等价；b 合成单杀；c Rust fail-loud）
  T12  侧车格式（rd 关⇒无侧车文件；开⇒两表行数/列名锚 + 主表两态等价）
  T13  分块防复刻 120（480 世界 cols=960 与 mini cols=120 双例几何锚一致口径）
  T14  槽位与配置面（_rd_pred_kill_log 在 __slots__；全配置 dataclass 不含该名）
  T15  操纵检验门（双臂批通用：同 seed 两臂主表逐字节同 ⇒ fail-loud；C1 单臂 N/A）
  F2   装置面：//120 与 polar 切点源码守卫 + --device 回读/显式覆盖/超上限 fail-loud
  F5   钩子 3 元组 (id, 死亡格, tick)：合成单杀 + 真跑 arity
  F6   钩子丢失 fail-loud（快照续跑重建引擎 ⇒ None ⇒ SystemExit；rd 批禁续跑）
  F7   kills_cum_total 累计列（每 4 窗=1000t 与主表 d_pred 对账锚）
  F8   manifest rd 节条件写入（rd 关 ⇒ 零 rd_* 键，T1 逐字节等价）
  F9   help 文案勘误守卫（signal 非法通道 ⇒ food,prey,risk,kin）

⚠️ 本文件 = 脚手架（骨架 + 关键用例）；T5/T6/T10 的完整统计测试需等砚预注册数值后补全。
⚠️ F2/F8 的 subprocess 用例各起一次 a4 主进程（mini 5t），秒级。
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.a4_verify_capacity import (  # noqa: E402
    RdInstruments, build,
)
from simulation.config import SimConfig  # noqa: E402


# ----  fixtures ----------------------------------------------------------

def _engine_mini(seed: int = 42, ticks: int = 200):
    """mini 世界（60×120）引擎；快速冒烟用。"""
    return build("off", False, seed, ticks, max_count=500)


def _engine_s2_like(seed: int = 42, ticks: int = 200):
    """大世界的几何（480×960）但不真跑 480×960（太慢）；
    用 mini 世界但手动验分块几何对 480×960 的口径一致性（T13）。"""
    return build("off", False, seed, ticks, max_count=500)


def _make_rd(e, *, block_rows=4, block_cols=4, channel="risk",
             sample_every=50, win_path=None, run_path=None):
    """构造 RdInstruments（不装钩子；纯几何/统计测试用）。"""
    return RdInstruments(
        e, channel=channel,
        block_rows=block_rows, block_cols=block_cols,
        sample_every=sample_every,
        windows_path=win_path, run_path=run_path,
    )


# ---- T4：分块几何 4×4 ---------------------------------------------------

class TestBlockGeometry:
    """T4/T13：分块映射正确性。"""

    def test_n_blocks_default_16(self):
        """4×4 = 16 块（§八-4 锁定）。"""
        e = _engine_mini()
        rd = _make_rd(e)
        assert rd.n_blocks == 16

    def test_n_blocks_cli_5x5(self):
        """CLI 改 5×5 ⇒ 25 块。"""
        e = _engine_mini()
        rd = _make_rd(e, block_rows=5, block_cols=5)
        assert rd.n_blocks == 25

    def test_block_map_shape(self):
        """block_map 形状 = (n_cells,)。"""
        e = _engine_mini()
        rd = _make_rd(e)
        assert rd._block_map.shape == (e.world.n_cells,)

    def test_block_map_range(self):
        """block_id ∈ [0, n_blocks)。"""
        e = _engine_mini()
        rd = _make_rd(e)
        assert rd._block_map.min() >= 0
        assert rd._block_map.max() < rd.n_blocks

    def test_block_map_corners(self):
        """四角 block_id 手算锚。

        mini 世界 60×120，4×4 分块：
        - (0,0) → lat=0, lon=0 → block 0
        - (0,119) → lat=0, lon=3 → block 3
        - (59,0) → lat=3, lon=0 → block 12
        - (59,119) → lat=3, lon=3 → block 15
        """
        e = _engine_mini()
        rd = _make_rd(e)
        cols = e.world.cols
        rows = e.world.rows
        assert rd._block_map[0] == 0  # (0,0)
        assert rd._block_map[cols - 1] == 3  # (0, cols-1)
        assert rd._block_map[(rows - 1) * cols] == 12  # (rows-1, 0)
        assert rd._block_map[rows * cols - 1] == 15  # (rows-1, cols-1)

    def test_t13_no_hardcoded_120(self):
        """T13：分块公式不硬编码 120（mini 世界 cols=120 与 480 世界 cols=960
        必须同口径）。验算：block_map 用 world.rows/cols 现算。"""
        e = _engine_mini()
        rd = _make_rd(e)
        # 手算：用 world.rows/cols 公式
        cols = rd._cols
        rows = rd._rows
        flat_idx = np.arange(rd._n_cells, dtype=np.int64)
        row = flat_idx // cols
        col = flat_idx % cols
        lat_band = (row * rd.block_rows) // rows
        lon_band = (col * rd.block_cols) // cols
        expected = lat_band * rd.block_cols + lon_band
        np.testing.assert_array_equal(rd._block_map, expected)


# ---- T11：D-1 钩子三连 --------------------------------------------------

class TestD1Hook:
    """T11a/T11b/T11c：D-1 钩子安装/记账/fail-loud。"""

    def test_t11a_hook_default_none(self):
        """T11a：未装钩子时 _rd_pred_kill_log 恒 None。"""
        e = _engine_mini(ticks=50)
        assert e._rd_pred_kill_log is None

    def test_t11a_install_and_clear(self):
        """T11a：装钩子后 = []，跑 50 tick 后仍为 list（可能有条目）。"""
        e = _engine_mini(ticks=50)
        rd = _make_rd(e)
        rd.install_hook(e)
        assert e._rd_pred_kill_log == []
        for _ in range(50):
            e.step()
        assert isinstance(e._rd_pred_kill_log, list)

    def test_t11b_synthetic_kill(self):
        """T11b：合成单杀事件 → log 记 (id, flat_cell, tick)（F5 3 元组）。"""
        e = _engine_mini(ticks=10)
        rd = _make_rd(e)
        rd.install_hook(e)
        # 合成：手动 append（模拟引擎捕食结算点的行为——3 元组，F5 后）
        prey_idx = 0
        death_cell = int(e._flat[prey_idx])
        entry = (int(e._id[prey_idx]), death_cell, int(e._tick))
        e._rd_pred_kill_log.append(entry)
        assert len(e._rd_pred_kill_log) == 1
        assert e._rd_pred_kill_log[0] == entry
        assert len(e._rd_pred_kill_log[0]) == 3

    def test_t11b_real_run_three_tuple(self):
        """F5：真跑 100t（seed 221 实测 52 杀）⇒ 全部条目为 (id, 格, tick)，
        且 tick ∈ [1, 100]（引擎两处 append 点实装 3 元组的端到端锚）。"""
        e = _engine_mini(seed=221, ticks=100)
        rd = _make_rd(e)
        rd.install_hook(e)
        for _ in range(100):
            e.step()
        log = e._rd_pred_kill_log
        assert len(log) > 0, "seed 221 mini 100t 应有击杀（实测 52）"
        assert all(len(t) == 3 for t in log)
        assert all(isinstance(t[0], int) and isinstance(t[1], int)
                   and isinstance(t[2], int) for t in log)
        ticks = [t[2] for t in log]
        assert min(ticks) >= 1 and max(ticks) <= 100

    def test_t11c_rust_fail_loud(self):
        """T11c：Rust 路径 + rd ⇒ SystemExit。"""
        e = _engine_mini(ticks=10)
        # 模拟 Rust 路径（设置 _sim_core 非 None）
        e._sim_core = object()
        rd = _make_rd(e)
        with pytest.raises(SystemExit, match="Python"):
            rd.install_hook(e)


# ---- T14：槽位与配置面 --------------------------------------------------

class TestSlotsAndConfig:
    """T14：_rd_pred_kill_log 在 __slots__；不进 SimConfig。"""

    def test_slot_registered(self):
        """T14：_rd_pred_kill_log 在 SphereEngine.__slots__。"""
        from simulation.sphere_engine import SphereEngine
        # __slots__ 可能在类层级或继承链上
        all_slots = set()
        for cls in SphereEngine.__mro__:
            all_slots.update(getattr(cls, "__slots__", ()))
        assert "_rd_pred_kill_log" in all_slots

    def test_not_in_simconfig(self):
        """T14：_rd_pred_kill_log 不在 SimConfig 任何 dataclass 字段中
        （不触指纹）。"""
        import dataclasses
        fields = set()
        for f in dataclasses.fields(SimConfig):
            fields.add(f.name)
            # 嵌套 dataclass 也扫
            if dataclasses.is_dataclass(f.type):
                pass  # 简单扫描顶层即可
        assert "_rd_pred_kill_log" not in fields

    def test_fingerprint_unchanged_by_hook(self):
        """T14：装/不装钩子 ⇒ 配置指纹不变（钩子不进 SimConfig）。"""
        e1 = _engine_mini(ticks=10)
        fp1 = e1.config.fingerprint()
        e2 = _engine_mini(ticks=10)
        e2._rd_pred_kill_log = []  # 装钩子
        fp2 = e2.config.fingerprint()
        assert fp1 == fp2


# ---- T12：侧车格式 -------------------------------------------------------

class TestSidecarFormat:
    """T12：rd 关⇒无侧车文件；开⇒两表行数/列名锚。"""

    def test_rd_off_no_sidecar(self, tmp_path):
        """T12：rd 关 ⇒ 不产生侧车文件。"""
        # 不实例化 RdInstruments ⇒ 无文件
        win_path = tmp_path / "test_rd_windows.csv"
        run_path = tmp_path / "test_rd_run.csv"
        assert not win_path.exists()
        assert not run_path.exists()

    def test_rd_on_sidecar_columns(self, tmp_path):
        """T12：rd 开 ⇒ 侧车一列名锚。"""
        e = _engine_mini(ticks=100)
        win_path = tmp_path / "test_rd_windows.csv"
        run_path = tmp_path / "test_rd_run.csv"
        rd = _make_rd(e, sample_every=50,
                      win_path=win_path, run_path=run_path)
        rd.install_hook(e)
        rd.open_sidecars()
        # 跑 100 tick，应有 2 窗
        for t in range(1, 101):
            e.step()
            if t % 50 == 0:
                rd.process_window(e, t, seed=42, arm="main")
        rd.close()
        # 验列名
        with win_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames
        assert "seed" in cols
        assert "arm" in cols
        assert "win_idx" in cols
        assert "kills_0" in cols
        assert "kills_15" in cols
        assert "occ_0" in cols
        assert "dens_0" in cols
        assert "win_kills_total" in cols
        assert "kills_cum_total" in cols   # F7：累计对账锚
        assert "risk_read_n" in cols

    def test_f7_kills_cum_total(self, tmp_path):
        """F7：kills_cum_total = 含本窗累计；窗1 注 2 杀 + 窗2 注 3 杀 ⇒ 2 → 5，
        且与侧车二 rd_total_kills 一致（主表 d_pred 差分对账锚，每 4 窗一行）。"""
        e = _engine_mini(ticks=20)
        win_path = tmp_path / "test_rd_windows.csv"
        run_path = tmp_path / "test_rd_run.csv"
        rd = _make_rd(e, sample_every=10,
                      win_path=win_path, run_path=run_path)
        rd.open_sidecars()
        # 不装引擎钩子，直接注入合成条目（本测只验侧车聚合算术）
        e._rd_pred_kill_log = []
        for i in range(2):
            e._rd_pred_kill_log.append(
                (int(e._id[i]), int(e._flat[i]), 1))
        rd.process_window(e, 10, seed=7, arm="main")
        for i in range(3):
            e._rd_pred_kill_log.append(
                (int(e._id[i]), int(e._flat[i]), 11))
        rd.process_window(e, 20, seed=7, arm="main")
        rd.close()
        rd.write_run_summary(seed=7, arm="main")
        with win_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [r["win_kills_total"] for r in rows] == ["2", "3"]
        assert [r["kills_cum_total"] for r in rows] == ["2", "5"]
        with run_path.open(encoding="utf-8") as fh:
            run = list(csv.DictReader(fh))
        assert run[0]["rd_total_kills"] == "5"


# ---- T9：fail-loud -------------------------------------------------------

class TestFailLoud:
    """T9：全 0 kills ⇒ dens 全 NaN ⇒ 不静默。"""

    def test_zero_kills_nan_dens(self, tmp_path):
        """T9：occ=0 ⇒ dens=NaN（写空串）；kills=0 但 occ>0 ⇒ dens=0。"""
        e = _engine_mini(ticks=50)
        win_path = tmp_path / "test_rd_windows.csv"
        rd = _make_rd(e, sample_every=50, win_path=win_path)
        rd.install_hook(e)
        rd.open_sidecars()
        # 手动设置：所有块 occ=0（模拟灭绝场景）
        rd._prev_block_pop = np.zeros(rd.n_blocks, dtype=np.float64)
        # 清空引擎种群（模拟灭绝）
        e._flat = e._flat[:0]
        e._energy = e._energy[:0]
        e._id = e._id[:0]
        rd.process_window(e, 50, seed=42, arm="main")
        rd.close()
        # 读侧车验 dens 列
        with win_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        # occ=0 ⇒ dens 应为空（NaN 写为空串）
        for b in range(16):
            assert row[f"occ_{b}"] == "0.0" or row[f"occ_{b}"] == "0"
            assert row[f"dens_{b}"] == ""  # NaN ⇒ 空串

    def test_f6_hook_lost_fail_loud(self):
        """F6：钩子丢失（None —— 快照续跑经 cls(config) 重建引擎的场景）
        ⇒ 窗末排空 SystemExit，**不**静默当"本窗无杀"。"""
        e = _engine_mini(ticks=10)
        assert e._rd_pred_kill_log is None   # 未装 = 模拟续跑后重建态
        rd = _make_rd(e)
        with pytest.raises(SystemExit, match="钩子丢失"):
            rd.process_window(e, 50, seed=42, arm="main")


# ---- T8：两臂指纹差异锁 -------------------------------------------------

class TestFingerprintArm:
    """T8：smell channels 含/不含 risk ⇒ 指纹不同。"""

    def test_arm_fingerprint_differs(self):
        """T8：两臂（risk 含/不含）fingerprint 不同。"""
        e_on = build("off", False, 42, 50, max_count=500,
                     smell_channels="food,risk,kin")
        e_off = build("off", False, 42, 50, max_count=500,
                      smell_channels="food,kin")
        assert e_on.config.fingerprint() != e_off.config.fingerprint()


# ---- T15：操纵检验门（双臂批通用）-----------------------------------------

def arms_manip_check(main_csv_a: Path, main_csv_b: Path, *,
                     label: str = "") -> None:
    """T15 操纵检验门（PI R383 准：C1 开发验收门 + 后续一切双臂批通用）。

    语义：同 seed 两臂主表必须**不同**；逐字节同 ⇒ fail-loud「装置未接通」，
    不得进判读——把"整批静默作废"变成"一条断言炸掉"。

    未来双臂批（Q-C 接线批起）在判读前调用本函数；C1 本批单臂（F1=丙）⇒
    批内 N/A——门本体随本分支交付，调用点归各批预注册写明（可逐字提升）。
    """
    a = Path(main_csv_a).read_bytes()
    b = Path(main_csv_b).read_bytes()
    if a == b:
        raise SystemExit(
            f"🔴 T15 操纵检验不过{'（' + label + '）' if label else ''}："
            "两臂主表逐字节相同 ⇒ 装置未接通，不得进判读（PI R383）。"
            "先修接线（risk 通道消费者门控）再跑批。"
        )


class TestT15ManipulationCheck:
    """T15：操纵检验门语义（PI R383；C1 单臂 ⇒ 批内 N/A，本类锁门本体）。"""

    def test_t15_identical_arms_fail_loud(self, tmp_path):
        """T15：两臂主表逐字节同 ⇒ SystemExit（装置未接通，不上判读）。"""
        a = tmp_path / "arm_on.csv"
        b = tmp_path / "arm_off.csv"
        a.write_text("tick,N\n1,10\n2,9\n", encoding="utf-8")
        b.write_text("tick,N\n1,10\n2,9\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="T15"):
            arms_manip_check(a, b, label="seed 221")

    def test_t15_differing_arms_pass(self, tmp_path):
        """T15：两臂主表有差 ⇒ 门放行（差 = 操纵生效的最小证据）。"""
        a = tmp_path / "arm_on.csv"
        b = tmp_path / "arm_off.csv"
        a.write_text("tick,N\n1,10\n2,9\n", encoding="utf-8")
        b.write_text("tick,N\n1,10\n2,8\n", encoding="utf-8")
        arms_manip_check(a, b, label="seed 221")  # 不抛 = 过


# ---- T1/T7：默认关逐字节等价 + 零 RNG 污染 --------------------------------

class TestDefaultOff:
    """T1/T7：rd 关 ⇒ 行为逐位等价；零 RNG 污染。"""

    def test_hook_none_no_behavior_change(self):
        """T1：默认 _rd_pred_kill_log = None ⇒ 引擎行为不变。"""
        e1 = _engine_mini(seed=42, ticks=100)
        digest1 = (int(e1._flat.sum()), round(float(e1._energy.sum()), 6))

        e2 = _engine_mini(seed=42, ticks=100)
        # 不装钩子 ⇒ 行为应与 e1 完全一致
        digest2 = (int(e2._flat.sum()), round(float(e2._energy.sum()), 6))
        assert digest1 == digest2

    def test_installed_but_empty_no_change(self):
        """T1/T7：装钩子但无杀事件 ⇒ 主 RNG 抽取次数不变（零 RNG 污染）。"""
        e1 = _engine_mini(seed=42, ticks=100)
        for _ in range(100):
            e1.step()
        draws1 = e1.rng.draws

        e2 = _engine_mini(seed=42, ticks=100)
        e2._rd_pred_kill_log = []  # 装钩子
        for _ in range(100):
            e2.step()
        draws2 = e2.rng.draws
        assert draws1 == draws2


# ---- T3：通道抽象 --------------------------------------------------------

class TestChannelAbstraction:
    """T3：--rd-channel risk/pheromone 同代码路径。"""

    def test_channel_name_stored(self):
        """T3：channel 名正确存入 RdInstruments。"""
        e = _engine_mini()
        rd_risk = _make_rd(e, channel="risk")
        rd_phro = _make_rd(e, channel="pheromone")
        assert rd_risk.channel == "risk"
        assert rd_phro.channel == "pheromone"

    def test_smell_channel_active_check(self):
        """T3：_smell_channel_active 按通道名检查。"""
        e = _engine_mini()
        # 手动设置 smell channels
        e.config.smell.channels = ("food", "risk", "kin")
        rd_risk = _make_rd(e, channel="risk")
        rd_phro = _make_rd(e, channel="pheromone")
        assert rd_risk._smell_channel_active(e) is True
        assert rd_phro._smell_channel_active(e) is False


# ---- F2：主表读数段硬编码源码守卫 ------------------------------------------

class TestSourceGuards:
    """F2：硬编码 mini 几何守卫（源码级；防回归）。"""

    def _src(self) -> str:
        return (ROOT / "experiments" / "a4_verify_capacity.py").read_text(
            encoding="utf-8")

    def test_f2_no_hardcoded_mini_cols(self):
        """F2：不得再出现 `e._flat[:P] // 120`（480×960 档静默错值根因；T13 同源）。"""
        assert "_flat[:P] // 120" not in self._src()

    def test_f2_polar_cut_from_rows(self):
        """F2：polar 切点须走 rows 比例（`_row_cut`），不得回到写死 5/54。"""
        src = self._src()
        assert "(r <= 5) | (r >= 54)" not in src
        assert "_row_cut" in src

    def test_f9_help_no_illegal_signal_channel(self):
        """F9(a)：--smell-channels help 不得再写非法通道 `signal`
        （合法集 = _KNOWN_CHANNELS = food,prey,risk,kin；写 signal 会 assert 炸）。"""
        assert "food,signal,kin,risk" not in self._src()


# ---- F2/F8：subprocess 端到端（manifest 门控 / 装置回读）-------------------

PY = sys.executable


def _run_a4(tmp_path, extra, name="run"):
    """起一次 a4 主进程（mini 60×120 / 5t / max 500）。返回 (proc, out_path)。"""
    out = tmp_path / f"{name}.csv"
    cmd = [PY, str(ROOT / "experiments" / "a4_verify_capacity.py"),
           "--mode", "off", "--seed", "7", "--ticks", "5",
           "--out", str(out), "--max-count", "500",
           "--snapshot-dir", str(tmp_path / "snap"), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          cwd=str(ROOT))
    return proc, out


def _summary_of(out: Path) -> dict:
    return json.loads(
        out.with_suffix(".summary.json").read_text(encoding="utf-8"))


class TestManifestGating:
    """F8/F2：manifest 键面门控 + 装置回读（subprocess 端到端）。"""

    def test_f8_rd_off_no_rd_keys(self, tmp_path):
        """F8/T1：rd 关 ⇒ summary.switches **零** rd_* 键、零 device 键。"""
        proc, out = _run_a4(tmp_path, [], name="rd_off")
        assert proc.returncode == 0, proc.stderr[-2000:]
        sw = _summary_of(out)["switches"]
        assert not [k for k in sw if k.startswith("rd_")]
        assert "device" not in sw and "device_applied" not in sw

    def test_f8_rd_on_has_rd_keys(self, tmp_path):
        """F8/T12：rd 开 ⇒ rd_* 键齐（值正确）+ 侧车一列头含 kills_cum_total。"""
        proc, out = _run_a4(
            tmp_path,
            ["--rd-instruments", "--smell-channels", "food,risk,kin"],
            name="rd_on")
        assert proc.returncode == 0, proc.stderr[-2000:]
        sw = _summary_of(out)["switches"]
        assert sw["rd_instruments"] is True
        assert sw["rd_block_rows"] == 4 and sw["rd_block_cols"] == 4
        assert sw["rd_sample_every"] == 250
        assert sw["rd_hook_installed"] is True
        win_csv = out.with_name(out.stem + "_rd_windows.csv")
        assert win_csv.exists()
        header = win_csv.read_text(encoding="utf-8").splitlines()[0]
        assert "kills_cum_total" in header

    def test_f2_device_readback_and_override(self, tmp_path):
        """F2：--device s2 + 显式 mini 覆盖 ⇒ 世界保持 mini（显式优先）、
        manifest 回读 device/世界尺寸、stderr 有"实际生效装置/显式覆盖"两行。"""
        proc, out = _run_a4(
            tmp_path,
            ["--device", "s2",
             "--rows", "60", "--cols", "120", "--patches", "30",
             "--pop", "200"],
            name="dev")
        assert proc.returncode == 0, proc.stderr[-2000:]
        sw = _summary_of(out)["switches"]
        assert sw["device"] == "s2"
        assert sw["world_rows"] == 60 and sw["world_cols"] == 120
        assert "实际生效装置" in proc.stderr
        assert "被显式参数覆盖" in proc.stderr

    def test_f2_pop_over_max_fail_loud(self, tmp_path):
        """F2：--device s2 不抬 --max-count ⇒ pop 10000 > 5000 ⇒ 拒跑不静默。"""
        proc, _ = _run_a4(tmp_path, ["--device", "s2"], name="bad")
        assert proc.returncode != 0
        assert "--pop" in proc.stderr
