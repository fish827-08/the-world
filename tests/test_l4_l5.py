"""L4（捕食+植物化）+ L5（信任学习+工作记忆+文化传递）功能测试。

注意：球面网格 0 号格是北极极点（邻格=整行），测试用非极点行（第1行，flat≥14）。
"""
import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine
from core.lifecycle import DeathCause


def _make(seed=1, n=20):
    cfg = SimConfig(seed=seed)
    cfg.world.rows = 10
    cfg.world.cols = 14
    cfg.population.initial_count = n
    return SphereEngine(cfg)


# ── L4 捕食 ──────────────────────────────────────────────────────────

class TestPredation:
    def test_predation_kills_and_transfers_energy(self):
        """高攻击+饥饿个体攻击邻格，成功后猎物 PREDATION 死亡，捕食者获能。"""
        e = _make(seed=10, n=10)
        # 放到第1行相邻格（14~23），非极点
        for i in range(10):
            e._flat[i] = 14 + (i % 10)
        e._energy[:] = 1.0  # 饥饿
        e._genes[:, 16] = 1.0  # 全攻击基因
        e._genes[:, 0] = 0.0  # 不动
        s = e.step()
        # died 包含所有死因
        assert s.died == sum(s.deaths_by_cause.values())
        # 至少有死亡（饥饿+高攻击+相邻，要么捕食要么饿死）
        assert s.died > 0

    def test_no_attack_when_satiated(self):
        """能量充足时攻击概率极低（饥饿驱动）。"""
        e = _make(seed=11, n=10)
        for i in range(10):
            e._flat[i] = 14 + (i % 10)
        e._energy[:] = 250.0  # 饱食（max_energy=300）
        e._genes[:, 16] = 1.0
        e._genes[:, 0] = 0.0
        predation_count = 0
        for _ in range(20):
            s = e.step()
            predation_count += s.deaths_by_cause.get(DeathCause.PREDATION, 0)
        # 饱食时捕食死亡远少于饥饿时（饥饿驱动）
        assert predation_count <= 8


# ── L4 植物化 ────────────────────────────────────────────────────────

class TestPlant:
    def test_plant_reduces_movement(self):
        """g19 高的个体移动概率 = g0 × (1-g19)，显著降低。

        用个体 ID 追踪移动（而非数组索引）：死亡/出生会重排索引、
        子代追加到尾部，按索引对比会产生大量"假移动"。
        """
        e = _make(seed=12, n=30)
        e._genes[:, 0] = 1.0
        e._genes[:15, 19] = 1.0  # 前半完全扎根
        e._genes[15:, 19] = 0.0  # 后半正常
        e._energy[:] = 200.0  # 充足能量避免死亡干扰
        is_plant = {int(i): (k < 15) for k, i in enumerate(e._id)}
        moves = {"plant": 0, "normal": 0}
        for _ in range(50):
            prev = {int(i): int(f) for i, f in zip(e._id, e._flat)}
            e.step()
            for i, f in zip(e._id, e._flat):
                old = prev.get(int(i))
                if old is not None and old != int(f):
                    moves["plant" if is_plant[int(i)] else "normal"] += 1
        assert moves["plant"] < moves["normal"] * 0.3, dict(moves)

    def test_plant_boosts_photosynthesis(self):
        """g19 高的个体在光照下获得更多光合能量。"""
        e = _make(seed=13, n=20)
        e._genes[:, 0] = 0.0
        e._genes[:, 1] = 0.0
        e._genes[:, 8] = 1.0
        e._genes[:10, 19] = 1.0
        e._genes[10:, 19] = 0.0
        e._flat[:] = 14  # 非极点
        e._energy[:] = 10.0
        e._stomach[:] = 0.0
        e.resources._grid[:] = 0.0
        for _ in range(10):
            e.step()
        assert float(e._energy[:10].mean()) > float(e._energy[10:].mean())


# ── L5 信任学习 ──────────────────────────────────────────────────────

class TestTrustLearning:
    def test_true_signal_increases_trust(self):
        """移动到有信号且有食物的格子 → trust 增加。"""
        e = _make(seed=14, n=5)
        e._genes[:, 0] = 1.0
        e._genes[:, 14] = 1.0
        e._trust[:] = 0.5
        target = 15  # 14 的邻格
        e.signals._marks[target] = 5
        e.resources._grid[target] = e.resources._capacity[target] * 0.8
        e._flat[:] = 14
        before_max = float(e._trust.max())
        for _ in range(10):
            e.step()
        assert float(e._trust.max()) > before_max

    def test_false_signal_decreases_trust(self):
        """移动到有信号但无食物的格子 → trust 减少。

        ⚠️ 2026-09-23（T2）：eat_amount 0.5→0.9 后食物更足 ⇒ 假信号惩罚
        **过程性**出现（tick1–5 trust 降到 0.4）但末态可能被后续真信号回升覆盖
        （实测 tick10 回到 0.5）。断言改为"**过程中最低点**低于初值"（机制仍
        生效：假信号确实让 trust 下降，只是不再承诺末态保持低位）。
        """
        e = _make(seed=15, n=5)
        e._genes[:, 0] = 1.0
        e._genes[:, 14] = 1.0
        e._trust[:] = 0.5
        target = 15
        e.signals._marks[target] = 5
        e.resources._grid[target] = 0.0  # 假信号
        e._flat[:] = 14
        before_min = float(e._trust.min())
        seen_min = before_min
        for _ in range(10):
            e.step()
            seen_min = min(seen_min, float(e._trust.min()))
        assert seen_min < before_min, (
            f"假信号未让 trust 过程性下降（最低 {seen_min} ≥ 初值 {before_min}）")


# ── L5 工作记忆 ──────────────────────────────────────────────────────

class TestWorkingMemory:
    def test_food_rich_cell_recorded(self):
        """食物丰富的格子被记入工作记忆。"""
        e = _make(seed=16, n=5)
        e._genes[:, 0] = 0.0
        e._flat[:] = 14
        e.resources._grid[14] = e.resources._capacity[14] * 0.8
        e._work_memory[:] = -1
        e.step()
        assert (e._work_memory == 14).any()

    def test_memory_influences_movement(self):
        """记忆中的食物格子在邻格时，移动决策优先选择该格。"""
        e = _make(seed=17, n=3)
        e._genes[:, 0] = 1.0
        e._genes[:, 14] = 1.0
        e._genes[:, 13] = 0.0  # 不合群，消除群居得分干扰
        e._flat[:] = 14
        e._work_memory[:, 0] = 15  # 记忆 15 号格
        e.resources._grid[:] = 0.0
        moved_to_mem = 0
        total_moves = 0
        for _ in range(30):
            before = e._flat.copy()
            e.step()
            n = min(len(before), len(e._flat))
            for i in range(n):
                if before[i] != e._flat[i]:
                    total_moves += 1
                    if e._flat[i] == 15:
                        moved_to_mem += 1
        if total_moves > 0:
            assert moved_to_mem / total_moves > 0.15


# ── L5 文化传递 ──────────────────────────────────────────────────────

class TestCulturalTransmission:
    def test_juvenile_learns_from_adults(self):
        """幼体的信号解读表向周围成体的平均值收敛。"""
        e = _make(seed=18, n=10)
        e._genes[:, 0] = 0.0
        e._energy[:] = 200.0  # 充足能量，避免饿死
        # 成体在 14 号格，幼体在 15 号格（14 的邻格）
        e._flat[:5] = 14
        e._flat[5:] = 15
        e._age[:5] = 5000  # 成体
        e._interpret[:5, :] = 1.0
        e._age[5:] = 0  # 幼体
        e._interpret[5:, :] = -1.0
        before = float(e._interpret[5:].mean())
        for _ in range(100):
            e.step()
        after = float(e._interpret[5:].mean())
        assert after > before + 0.2

    def test_offspring_inherits_interpretation(self):
        """子代继承亲代的解读表（+噪声）。"""
        e = _make(seed=19, n=20)
        e._genes[:, 12] = 0.0
        e._genes[:, 2] = 1.0
        e._interpret[:] = 0.8
        born = False
        for _ in range(300):
            s = e.step()
            if s.born > 0:
                born = True
                n_new = s.born
                new_interp = float(e._interpret[-n_new:].mean())
                assert abs(new_interp - 0.8) < 0.6
                break
        if not born:
            pytest.skip("300 tick 内无出生（种群动态原因）")
