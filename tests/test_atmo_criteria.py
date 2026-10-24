# TIER = "fast"
"""AV 判据注册表自检：文献锚齐备、阈值冻结值钉住、套件引用合法、无重复 ID。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from benchmarks.atmo import criteria, suites


def test_registry_ids_unique():
    assert len(criteria.CRITERIA) == len({c.cid for c in criteria.CRITERIA.values()})


@pytest.mark.parametrize("cid", sorted(criteria.CRITERIA))
def test_criterion_wellformed(cid):
    c = criteria.CRITERIA[cid]
    assert c.cid == cid
    assert c.citation and c.citation.strip(), "每条判据必须有文献/基准出处"
    assert c.desc.strip()
    assert isinstance(c.mandatory, bool)
    if c.limit_lo is not None and c.limit_hi is not None:
        assert c.limit_lo <= c.limit_hi, f"{cid} 上下限颠倒"
    if c.limit_lo is not None or c.limit_hi is not None:
        assert c.unit, "数值判据必须有量纲"


@pytest.mark.parametrize("cid", sorted(criteria.CRITERIA))
def test_frozen_thresholds_pinned(cid):
    """判据冻结值钉住：改动阈值必须显式改本测试（裁判纪律）。"""
    c = criteria.CRITERIA[cid]
    expect = {
        "CONS-MASS": (None, 1e-6), "CONS-WATER": (None, 1e-3),
        "CONS-ENERGY": (None, 1.0),
        "CONS-CFL": (None, 1.0), "CONS-POLAR-AMP": (None, 3.0),
        "CONS-POLAR-TREND": (None, 0.01),
        "HS-JET-U": (15.0, 40.0), "HS-JET-LAT": (20.0, 60.0),
        "HS-SFC-U": (-8.0, 2.0), "HS-T-TROP-SFC": (290.0, 320.0),
        "HS-T-POLE-SFC": (225.0, 265.0), "HS-T-TROP-UP": (185.0, 230.0),
        "HS-EKE-ML": (1.0, None), "HS-EKE-RATIO": (1.5, None),
        "HS-FORCING": (None, 0.0), "AQ-FORCING": (None, 0.0),
        "AQ-ITCZ-LAT": (-5.0, 5.0), "AQ-ITCZ-P": (4.0, 20.0),
        "AQ-EP-SUBTROP": (0.0, None), "AQ-TOA": (-5.0, 5.0),
        "AQ-SST-TREND": (None, 0.01), "AQ-HEMI-DIFF": (None, 2.0),
        "SOLAR-XCHECK-COSZ": (None, 1e-6), "SOLAR-XCHECK-TOA": (None, 1e-6),
        "SOLAR-XCHECK-DEC": (None, 1e-6), "SOLAR-COSZ-RANGE": (None, 0.0),
        "SOLAR-GLOBAL-MEAN": (None, 0.02), "SOLAR-OBLIQ0-SYMM": (None, 1e-6),
        "SOLAR-POLAR-NIGHT": (1.0, None),
        "RCE-COL-ENERGY": (None, 5.0), "RCE-COL-EQUIL": (None, 0.05),
        "CONS-NAN": (None, None), "CONS-DET": (None, None),
    }
    assert cid in expect, f"新判据 {cid} 必须同时进本钉住表并给文献锚"
    assert (c.limit_lo, c.limit_hi) == expect[cid]


def test_suite_criteria_ids_valid():
    known = set(criteria.CRITERIA)
    for s in (suites.HELD_SUAREZ, suites.AQUAPLANET):
        assert set(s.criteria_ids) <= known, f"{s.name} 引用了未注册判据"


def test_suites_names_match_meta_contract():
    assert suites.HELD_SUAREZ.name == "held_suarez"
    assert suites.AQUAPLANET.name == "aquaplanet"
    assert not suites.HELD_SUAREZ.wet and suites.AQUAPLANET.wet


def test_hs94_params_match_held_suarez_1994():
    p = suites.HS94_PARAMS
    assert p["delta_Ty"] == 60.0 and p["delta_theta_z"] == 10.0 and p["sigma_b"] == 0.7
    assert abs(p["ka"] - 1 / (40 * 86400)) < 1e-15
    assert abs(p["ks"] - 1 / (4 * 86400)) < 1e-15
    assert abs(p["kf"] - 1 / 86400) < 1e-15
