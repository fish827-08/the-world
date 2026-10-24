"""B4：动作意图与成本测试（显式成本 + 结构自检）。

覆盖 B1 action.py：
- 六类动作成本表完整、为正；
- intent_cost 的 MOVE 乘子语义；
- validate_intent 的 fail-loud 断言；
- Intent 不可变（frozen dataclass）。
"""
import numpy as np
import pytest

from biosphere.action import (
    ACTION_BASE_COST, ACTION_NAMES, ActionKind, Intent, intent_cost,
    validate_intent,
)
from biosphere.develop import develop
from biosphere.genome import random_genome


def _controller(seed: int = 0):
    return develop(random_genome(np.random.default_rng(seed), uid=0))


class TestActionKind:
    def test_six_kinds(self):
        assert len(ActionKind) == 6
        assert set(ACTION_NAMES) == {int(k) for k in ActionKind}

    def test_costs_positive_and_complete(self):
        assert set(ACTION_BASE_COST) == {int(k) for k in ActionKind}
        assert all(c > 0 for c in ACTION_BASE_COST.values())

    def test_reproduce_is_priciest(self):
        """繁殖是最贵动作（规格页：没有成本就没有协作与交流）。"""
        assert max(ACTION_BASE_COST.values()) == ACTION_BASE_COST[int(ActionKind.REPRODUCE)]


class TestIntentCost:
    def test_move_scales_with_mult(self):
        c = _controller(1)
        base = ACTION_BASE_COST[int(ActionKind.MOVE)]
        assert intent_cost(ActionKind.MOVE, c) == pytest.approx(
            base * c.move_cost_mult)

    def test_others_flat(self):
        c = _controller(2)
        for kind in (ActionKind.EAT, ActionKind.ATTACK, ActionKind.SHARE,
                     ActionKind.SIGNAL, ActionKind.REPRODUCE):
            assert intent_cost(kind, c) == ACTION_BASE_COST[int(kind)]


class TestValidateIntent:
    def test_move_requires_cell(self):
        with pytest.raises(ValueError):
            validate_intent(Intent(actor_id=1, kind=ActionKind.MOVE,
                                   energy_cost=0.3))

    def test_attack_requires_target(self):
        with pytest.raises(ValueError):
            validate_intent(Intent(actor_id=1, kind=ActionKind.ATTACK,
                                   energy_cost=0.4))

    def test_share_requires_positive_payload(self):
        with pytest.raises(ValueError):
            validate_intent(Intent(actor_id=1, kind=ActionKind.SHARE,
                                   target_id=2, payload=0.0, energy_cost=0.2))

    def test_negative_cost_rejected(self):
        with pytest.raises(ValueError):
            validate_intent(Intent(actor_id=1, kind=ActionKind.EAT,
                                   target_cell=5, energy_cost=-0.1))

    def test_valid_intents_pass(self):
        validate_intent(Intent(actor_id=1, kind=ActionKind.MOVE,
                               target_cell=5, energy_cost=0.3))
        validate_intent(Intent(actor_id=1, kind=ActionKind.SHARE,
                               target_id=2, payload=3.0, energy_cost=0.2))
        validate_intent(Intent(actor_id=1, kind=ActionKind.REPRODUCE,
                               energy_cost=0.5))

    def test_intent_is_frozen(self):
        it = Intent(actor_id=1, kind=ActionKind.EAT, target_cell=5,
                    energy_cost=0.05)
        with pytest.raises(Exception):
            it.energy_cost = 1.0  # type: ignore[misc]
