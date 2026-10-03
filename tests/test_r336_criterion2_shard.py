# -*- coding: utf-8 -*-
"""R336 —— s2 探针 `criterion_2` 分片运行失效的回归测试。

缺陷背景
--------
`_criterion_2` 是**整批级**判定：按 seed 分组后要求同seed 的 off + on 都在`summary` 里
才能配对。而预注册（S2 判据②）要求按 `(seed, arm)` 切成多个**单run 作业**，
每个作业只产1 个 run 的 summary ⇒ `per_seed={}` / `verdict=null`（12/12 实测）。

本测试钉死修复后的三条契约：
1. **分片**（单 run 作业）也能算出整批口径的 `criterion_2`；
2. **整批**运行的结论与旧版**逐字段一致**（R278 P3 契约不被破坏）；
3. 修复真的被撤掉时本测试**必须失败**（变异检查，见文件末尾说明）。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.join(os.path.dirname(_HERE), "experiments", "s2_depletion_probe.py")


def _load_probe():
    spec = importlib.util.spec_from_file_location("s2dp_r336", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(seed, arm, init_tail, live_tail=0.05):
    """构造一行最小summary（只需criterion_2 用到的字段）。"""
    return {"seed": seed, "arm": arm,
            "patch_sat_init_var_tail": init_tail, "patch_sat_var_tail": live_tail}


# ---------- 契约 1：分片运行（单 run）也能算整批口径 ----------

def test_shard_single_run_can_pair_with_peer(tmp_path):
    """场景：两个单 run 作业（201_off / 201_on）先后跑在同目录。

    关键断言：**后跑的那个**（只知道自己那一臂）也必须算出配对 verdict。
    """
    mod = _load_probe()

    # 先跑 201_off（此刻目录里只有自己 ⇒ 无法配对 ⇒ verdict=None 但n_paired=0）
    off_json = tmp_path / "s2g2_201_off.summary.json"
    off_json.write_text(json.dumps({"summary": [_row(201, "off", 0.16)]}),
                        encoding="utf-8")

    # 再跑 201_on（目录里已有 off ⇒ 应补读并配对成功）
    on_json = tmp_path / "s2g2_201_on.summary.json"
    on_json.write_text(json.dumps({"summary": [_row(201, "on", 0.20)]}),
                       encoding="utf-8")

    rows = mod._load_peer_summaries(str(on_json), [_row(201, "on", 0.20)])
    c2 = mod._criterion_2(rows)

    assert c2["overall"]["n_paired_seeds"] == 1, "分片下必须配对到 1 个 seed"
    assert 201 in c2["per_seed"], "per_seed 必须含 201"
    assert c2["per_seed"][201]["on_gt_off_init"] is True
    assert c2["overall"]["verdict_on_gt_off_init"] is True


def test_shard_first_run_reports_zero_pairs_not_false_verdict(tmp_path):
    """🔴 回归要点：还没有同伴时，verdict 必须是 None（"无法判定"），
    而**不能**是 False（否则会被误读为"判定失败"）。且 n_paired 必须暴露 0。"""
    mod = _load_probe()
    solo_json = tmp_path / "s2g2_201_off.summary.json"
    solo_json.write_text(json.dumps({"summary": [_row(201, "off", 0.16)]}),
                         encoding="utf-8")

    rows = mod._load_peer_summaries(str(solo_json), [_row(201, "off", 0.16)])
    c2 = mod._criterion_2(rows)

    assert c2["overall"]["verdict_on_gt_off_init"] is None
    assert c2["overall"]["n_paired_seeds"] == 0
    # 🔴 新增字段：让"null"可被读懂
    assert "n_paired_seeds" in c2["overall"]


# ---------- 契约 2：整批运行逐字段一致（R278 P3 契约） ----------

def test_whole_batch_unchanged(tmp_path):
    """整批（自己就有全部 run）时，补读逻辑不得改变任何字段。

    这保证"正常跑完时 summary 逐字节等于旧版"（R278 P3 契约）。
    """
    mod = _load_probe()
    own = [_row(201, "off", 0.16), _row(201, "on", 0.14),
           _row(202, "off", 0.17), _row(202, "on", 0.15)]

    # 目录里放一份"旧版产物"（同seed 同 arm 的重复行 + 一份陈旧 verdict）
    stale = {"summary": [_row(201, "off", 0.99),      # 同 key⇒ 必须被自己优先
                        _row(203, "off", 0.10)]}      # 新 key ⇒ 会被并入
    (tmp_path / "s2g2_stale.summary.json").write_text(
        json.dumps(stale), encoding="utf-8")

    out_json = tmp_path / "s2g2_201_off.summary.json"
    out_json.write_text(json.dumps({"summary": []}), encoding="utf-8")

    rows = mod._load_peer_summaries(str(out_json), own)

    # 自己的 4 行必须逐字段保留（不被陈旧同 key 行覆盖）
    got = {(r["seed"], r["arm"]): r["patch_sat_init_var_tail"] for r in rows}
    assert got[(201, "off")] == 0.16
    assert got[(201, "on")] == 0.14
    assert got[(202, "off")] == 0.17
    assert got[(202, "on")] == 0.15
    # 陈旧的 0.99 不得污染
    assert all(v != 0.99 for v in got.values())

    c2 = mod._criterion_2(rows)
    # on 臂全部低于 off ⇒ verdict=False（on_gt_off_init 全 false）
    assert c2["overall"]["n_paired_seeds"] >= 2
    assert c2["overall"]["verdict_on_gt_off_init"] is False
    assert c2["overall"]["off_var_init_mean"] == pytest.approx(0.165, abs=1e-6)
    assert c2["overall"]["on_var_init_mean"] == pytest.approx(0.145, abs=1e-6)


def test_dedupe_same_seed_arm(tmp_path):
    """同 (seed, arm) 只保留一行（自己优先），不重复计数。"""
    mod = _load_probe()
    own = [_row(201, "on", 0.20)]
    peer = {"summary": [_row(201, "on", 0.20), _row(201, "off", 0.16)]}
    (tmp_path / "s2g2_peer.summary.json").write_text(json.dumps(peer), encoding="utf-8")
    out_json = tmp_path / "s2g2_201_on.summary.json"
    out_json.write_text(json.dumps({"summary": []}), encoding="utf-8")

    rows = mod._load_peer_summaries(str(out_json), own)
    keys = [(r["seed"], r["arm"]) for r in rows]
    assert len(keys) == len(set(keys)), f"出现重复 key: {keys}"
    assert (201, "on") in keys and (201, "off") in keys


# ---------- 契约 3：健壮性（坏文件不得中断跑批） ----------

@pytest.mark.parametrize("bad", ["", "{", "not json at all", "{\"summary\": null}"])
def test_broken_peer_file_does_not_raise(tmp_path, bad):
    """🔴 半个 .summary.json（进程被杀时常见）不得让跑批崩掉。"""
    mod = _load_probe()
    (tmp_path / "s2g2_broken.summary.json").write_text(bad, encoding="utf-8")
    out_json = tmp_path / "s2g2_201_on.summary.json"
    out_json.write_text(json.dumps({"summary": []}), encoding="utf-8")

    rows = mod._load_peer_summaries(str(out_json), [_row(201, "on", 0.2)])
    assert any(r["seed"] == 201 for r in rows), "自己的行必须保住"


def test_missing_dir_does_not_raise(tmp_path):
    mod = _load_probe()
    rows = mod._load_peer_summaries(str(tmp_path / "nope" / "x.summary.json"),
                                   [_row(1, "on", 0.1)])
    assert len(rows) == 1


# ---------- 契约 4：off 臂异质性为 0 时不配对（守住原口径） ----------

def test_zero_off_variance_not_paired(tmp_path):
    """原逻辑：off 臂 `patch_sat_init_var_tail <= 0` 时不配对（判据无意义）。
    修复不得改变这一条。"""
    mod = _load_probe()
    rows = [_row(201, "off", 0.0), _row(201, "on", 0.2)]
    c2 = mod._criterion_2(rows)
    assert c2["per_seed"] == {}
    assert c2["overall"]["verdict_on_gt_off_init"] is None


def test_verdict_all_true_only_when_all_positive():
    """verdict 是 `all(...)`，不是 any / not any。"""
    mod = _load_probe()
    rows = [_row(1, "off", 0.10), _row(1, "on", 0.20),
            _row(2, "off", 0.30), _row(2, "on", 0.20)]   # seed2 反向
    c2 = mod._criterion_2(rows)
    assert c2["per_seed"][1]["on_gt_off_init"] is True
    assert c2["per_seed"][2]["on_gt_off_init"] is False
    assert c2["overall"]["verdict_on_gt_off_init"] is False


# ======================================================================
# 🔴 契约 5：**打穿真实路径**（`_write_summary_atomic`）
# ======================================================================
#   ⚠️ 教训（本文件第一版的失败记录，务必保留）：
#   第一版测试只直接调用 `_load_peer_summaries` + `_criterion_2` 两个函数，
#   结果**变异检查失败**——把 `_write_summary_atomic` 里的
#   `rows_for_c2` 改回 `summary`（即真正撤掉修复）后，11 个用例**依然全绿**。
#   ⇒ 说明"测两个零件"不等于"测了装配"。
#   ⇒ 下面这组用例**必须走 `_write_summary_atomic` 真实路径**才能有鉴别力。

_HEADER = None


def _real_row(seed, arm, init_tail):
    """按探针的真实 CSV 表头造一行（避免手写 mock 漏列）。

    🔴 漏列会让 `main()` 在 `last["pop"]` 之类处 KeyError——
    那是 mock 的错，不是被测逻辑的错。表头从探针源码里取，保持同步。
    """
    global _HEADER
    if _HEADER is None:
        import re
        src = open(_PROBE, encoding="utf-8").read()
        m = re.search(r"header = \[(.*?)\]", src, re.S)
        assert m, "未能从探针源码解析 CSV 表头"
        _HEADER = re.findall(r'"([^"]+)"', m.group(1))

    # 🔴 数值列（探针内部会对这些做算术，如 `_tail_mean` 的 `sum(r[key])`）⇒ 必须给数值，
    #    不能给字符串。真实 CSV 读进来是什么样，这里就给什么样。
    numeric_cols = {c: 1.0 for c in _HEADER}
    numeric_cols.update({
        "pop": 100.0, "global_sat": 0.6, "abs_food": 100.0, "abs_cap": 100.0,
        "cap_lost_frac": 0.0, "d_starv": 0.0, "d_old": 0.0, "d_pred": 0.0,
        "ms_per_tick": 1.0, "n_patches": 1700.0,
        "patch_sat_mean": 0.7, "patch_sat_var": init_tail,
        "patch_sat_range": 0.1, "patch_sat_init_var": init_tail,
        "patch_abs_food_var": 1.0,
        "l0_peak_dead": 0.0, "l0_peak_rest": 0.0, "l0_peak_demoted": 0.0,
        "l0_any_dead": 0.0, "l0_any_rest": 0.0, "l0_any_demoted": 0.0,
        "l1_visited_patch_n": 100.0, "l1_visited_patch_frac": 0.5,
        "l1_first_visit_median": 1.0, "l1_first_visit_p25": 1.0,
        "l1_first_visit_p75": 1.0,
        "l2_visited_gini": 0.5, "l2_visited_var": 1.0, "l2_total_var": 1.0,
        "l2_between_share": 0.1, "l2_visited_frac": 0.5,
        "l2_visited_mean": 1.0, "l2_unvisited_mean": 1.0, "l2_total_gini": 0.5,
        "l3_forage_sum": 1.0, "l3_intake_per_capita": 1.0, "l3_net_sum": 1.0,
        "l3_net_per_capita": 1.0, "l3_pop_mean": 1.0,
        "bg_low_frac_actual": 0.0, "bg_low_n": 0.0, "bg_resid_frac": 0.1,
    })
    # 🔴 `tick` 列探针按int 用（`range(0, ticks+1, sample)` 之后取值），
    #    字符串会让比较运算出错 ⇒ 用整数。
    row = dict(numeric_cols)
    row.update({
        "seed": seed, "arm": arm, "tick": 250,
    })
    return row


def _run_main_for(monkeypatch, mod, tmp_path, *, seed, arm, out_name):
    """用mock 的 `run_one` 跑 `main()`，走完整落盘路径。"""
    monkeypatch.setattr(mod, "run_one",
                        lambda *a, **k: ([_real_row(seed, arm, 0.16 if arm == "off" else 0.20)],
                                         "tick 用尽"))
    monkeypatch.setattr(sys, "argv", [
        "s2_depletion_probe.py", "--out", str(tmp_path / out_name),
        "--rows", "480", "--cols", "960", "--patches", "1700", "--pop", "10000",
        "--ticks", "250", "--sample", "250", "--seeds", str(seed),
        "--arms", arm, "--device", "s2",
    ])
    mod.main()
    return json.loads((tmp_path / out_name.replace(".csv", ".summary.json"))
                      .read_text(encoding="utf-8"))


def test_real_write_path_shard_pairs(monkeypatch, tmp_path):
    """🔴 核心鉴别力用例：**走 main() 真实路径**的单 run 作业，
    在同目录已有同 seed 另一臂时，必须写出配对的 verdict。

    这是第一版缺失的那一层（第一版只测零件 ⇒ 变异检查恒绿）。
    """
    mod = _load_probe()
    # 先放"另一个作业"的产物：201_off（init_tail=0.16）
    # 🔴 用**完整真实行**而非最小 stub：`_criterion_2` 会读 off 臂的
    # `patch_sat_var_tail`，stub 缺字段会KeyError/静默失配。
    peer_row = _real_row(201, "off", 0.16)
    peer_row["patch_sat_init_var_tail"] = 0.16
    peer_row["patch_sat_var_tail"] = 0.16
    peer = {"params": {}, "summary": [peer_row],
            "criterion_2": {"per_seed": {}, "overall": {}}}
    (tmp_path / "s2g2_201_off.summary.json").write_text(
        json.dumps(peer), encoding="utf-8")

    got = _run_main_for(monkeypatch, mod, tmp_path,
                        seed=201, arm="on", out_name="s2g2_201_on.csv")

    c2 = got["criterion_2"]
    assert c2["overall"]["n_paired_seeds"] == 1, (
        f"走真实路径时分片作业应配对到 1 个 seed，实际 {c2['overall']}")
    # ⚠️ 经 JSON 落盘后`per_seed` 的 key 是**字符串**（json.dumps 把 int key 转 str），
    #    而函数内返回的是 int key ⇒ 比较时统一用 str。
    assert "201" in c2["per_seed"], f"per_seed 应含 201，实际 {c2['per_seed']}"
    assert c2["per_seed"]["201"]["on_gt_off_init"] is True
    assert c2["overall"]["verdict_on_gt_off_init"] is True


def test_real_write_path_without_peer_is_null_not_false(monkeypatch, tmp_path):
    """真实路径 + **无同伴**：verdict 必须是 None，且 n_paired=0（不是 False）。"""
    mod = _load_probe()
    got = _run_main_for(monkeypatch, mod, tmp_path,
                        seed=301, arm="off", out_name="s2g2_301_off.csv")

    c2 = got["criterion_2"]
    assert c2["overall"]["n_paired_seeds"] == 0
    assert c2["overall"]["verdict_on_gt_off_init"] is None, (
        "无同伴时必须是 None（无法判定），不得是 False（会被误读为判定失败）")


def _summary_stub(seed, arm, init_tail):
    """写入同伴 summary.json 时用的行（只需 criterion_2 读到的字段）。"""
    return {"seed": seed, "arm": arm,
            "patch_sat_init_var_tail": init_tail,
            "patch_sat_var_tail": init_tail}


# ======================================================================
# 变异检查（已执行，结论记录 —— 勿删这段）
# ======================================================================
#   变异 1：`_write_summary_atomic` 里 `rows_for_c2` → `summary`
#     ⇒ 修复前：11 passed（❌ 无鉴别力，第一版测试的教训）
#     ⇒ 补上契约 5 后重跑：test_real_write_path_shard_pairs 必须 FAIL（✅ 有鉴别力）
#   变异 2：删掉 `_criterion_2` 的 `n_paired_seeds` 行
#     ⇒ test_real_write_path_without_peer_is_null_not_false 必须 FAIL（KeyError）
#   （变异检查 = 证明测试真能抓 bug，而不是恒绿。教训：必须测**装配**，
#     只测零件会被"没走那条路径"的实现绕过。）