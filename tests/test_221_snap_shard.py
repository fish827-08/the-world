"""R221 §四：`steady_k_probe.py` 快照 / 续跑 / 分片（`[云端开发·云启]`，2026-09-27）。

覆盖（对应派工验收「续跑必须与不中断连续跑**逐位一致**（含 RNG 与死因账本）」）：
  A **续跑 ≡ 连续跑（端到端）**：40 tick（每 10 tick 存快照）⇒ 从**目录**续到 60，
     与不中断跑 60 对照 ⇒ CSV 逐值一致（排除 `ms_per_tick`）+ summary 的
     `tail3_pops` / **死因账本**（含累计三项）/ `K_measured` / `max_gen` 一致
  B **快照节拍不改轨迹**：带 `--save-every` 与不带 ⇒ 逐值一致
  C **分片**：`K/N` 三份并集 == 全集且两两不交；显式 `p:sd` / `rgm:p:sd`；**匹配 0 / 越界 ⇒ fail-loud**；
     端到端 `--shard 2/2` 只跑子集（CSV 里只出现该子集的 run）
  D **旧快照兼容**：剥掉可选键（codebook / ARS 六数组 / sub_r,sub_c / health …）仍可载入并续跑
  E **防呆**：续跑 `--sample` 不一致 ⇒ fail-loud（禁采样网格静默错位）

🔴 DEL-8 变异的**敏感断言** = 本文件（改坏任一处 ⇒ 必红）；证据见 `_rerun_logs/snap_shard_smoke/`。
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.steady_k_probe import (  # noqa: E402
    _ckpt_paths,
    load_ckpt,
    run_tag,
    select_runs,
)

PY = sys.executable

#: 小世界口径（60×120 / pop 50 / 采样 10 / 关早停）；跑一次只要 1–2 s
SMALL = ["--rows", "60", "--cols", "120", "--patches", "30", "--seeds", "42",
         "--pop", "50", "--sample", "10", "--subpos", "off", "--stop-stable", "0"]
TAG = run_tag(60, 120, 30, 42, 1.0, 1.195)


def _cli(args: list[str], *, expect_ok: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run([PY, "experiments/steady_k_probe.py", *args], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=1200)
    if expect_ok:
        assert r.returncode == 0, f"CLI 失败：{r.stdout[-400:]}\n{r.stderr[-1200:]}"
    return r


def _rows(path: Path) -> list[dict]:
    """读 CSV 并剔除**计时列**（`ms_per_tick` 天生不可复现）。"""
    with io.open(path, encoding="utf-8", newline="") as fh:
        return [{k: v for k, v in row.items() if k != "ms_per_tick"}
                for row in csv.DictReader(fh)]


def _cp(src: Path, dst: Path) -> None:
    dst.write_bytes(src.read_bytes())


# ─────────────────────── A：续跑 ≡ 连续跑 ───────────────────────

def test_resume_equals_continuous_end_to_end(tmp_path: Path):
    snap_dir = tmp_path / "snap"
    csv_a, csv_b, csv_c = (tmp_path / n for n in ("a.csv", "b.csv", "c.csv"))
    # ① 跑 40 tick，每 10 tick 存一组检查点
    _cli([*SMALL, "--ticks", "40", "--out", str(csv_a),
          "--save-every", "10", "--snapshot-dir", str(snap_dir)])
    snap, rngp, metap = _ckpt_paths(snap_dir, TAG)
    assert snap.exists() and rngp.exists() and metap.exists(), "检查点三件套没写全"
    meta = json.loads(metap.read_text(encoding="utf-8"))
    assert meta["sample"] == 10 and meta["tick_saved"] == 40
    assert meta["config_fingerprint"], "meta 应带配置指纹（跨机核对用）"

    # ② 模拟"中断"：把 ① 的 CSV/summary 当作既有产物，从**快照目录**续跑到 60
    _cp(csv_a, csv_b)
    _cp(tmp_path / "a.summary.json", tmp_path / "b.summary.json")
    r = _cli(["--resume-from", str(snap_dir), "--ticks", "60", "--sample", "10",
              "--out", str(csv_b)])
    assert "续跑" in r.stdout and "@ tick 40 → 60" in r.stdout

    # ③ 不中断连续跑 60
    _cli([*SMALL, "--ticks", "60", "--out", str(csv_c)])

    # ④ 逐值一致（排除 ms_per_tick）
    rb, rc = _rows(csv_b), _rows(csv_c)
    assert len(rb) == len(rc) == 6, f"行数：续跑 {len(rb)} vs 连续 {len(rc)}（期望 6 = tick 10..60）"
    assert rb == rc, "续跑 CSV 与连续跑不一致（除 ms_per_tick 外应逐值相等）"

    # ⑤ 死因账本与 K 判读一致（续跑不得丢历史累计）
    db = json.loads((tmp_path / "b.summary.json").read_text(encoding="utf-8"))
    dc = json.loads((tmp_path / "c.summary.json").read_text(encoding="utf-8"))
    sb, sc = db["result"], dc["result"]
    for key in ("tail3_pops", "death_cause_tail", "K_measured", "final_N", "final_tick",
                "max_gen", "platform_reached", "stop"):
        assert sb[key] == sc[key], f"summary.result.{key} 不一致：续跑={sb[key]} 连续={sc[key]}"
    # run 级全字段（剔除计时两列）——含**死因账本累计** deaths_cum_* 与 K/峰值/饱和度
    strip = lambda d: {k: v for k, v in d.items() if k not in ("wall_s", "ms_per_tick_tail")}
    assert strip(db["runs"][0]) == strip(dc["runs"][0]), "runs[0] 不一致（含死因账本累计三项）"


# ─────────────────────── B：存快照不改轨迹 ───────────────────────

def test_save_every_does_not_change_trajectory(tmp_path: Path):
    csv_n, csv_s = tmp_path / "n.csv", tmp_path / "s.csv"
    _cli([*SMALL, "--ticks", "40", "--out", str(csv_n)])
    _cli([*SMALL, "--ticks", "40", "--out", str(csv_s),
          "--save-every", "7", "--snapshot-dir", str(tmp_path / "snap")])
    assert _rows(csv_n) == _rows(csv_s), "存快照改变了轨迹（应零影响）"


# ─────────────────────── C：分片 ───────────────────────

def test_shard_partition_unit():
    grid = [(1.195, 30, 42), (1.195, 30, 7), (1.195, 60, 42), (1.195, 60, 7)]
    parts = [select_runs(grid, f"{i}/3") for i in (1, 2, 3)]
    flat = [g for p in parts for g in p]
    assert sorted(flat) == sorted(grid), "三份并集必须 == 全集（不漏 run）"
    assert len(flat) == len(set(flat)), "分片之间不得重叠"
    assert select_runs(grid, "30:7") == [(1.195, 30, 7)]
    assert select_runs(grid, "1.195:60:42") == [(1.195, 60, 42)]
    assert select_runs(grid, None) == grid
    with pytest.raises(ValueError):
        select_runs(grid, "99:99")          # 匹配 0 ⇒ fail-loud（禁静默空跑）
    with pytest.raises(ValueError):
        select_runs(grid, "7/3")            # K/N 越界


def test_shard_end_to_end(tmp_path: Path):
    out = tmp_path / "sh.csv"
    r = _cli(["--rows", "60", "--cols", "120", "--patches", "30,60", "--seeds", "42,7",
              "--pop", "30", "--ticks", "1", "--sample", "1", "--subpos", "off",
              "--stop-stable", "0", "--shard", "2/2", "--out", str(out)])
    assert "本机负责 **2/4** 个 run" in r.stdout
    with io.open(out, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    got = sorted({(int(x["patches"]), int(x["seed"])) for x in rows})
    assert got == [(30, 7), (60, 7)], f"2/2 应跑网格第 2、4 号 run，实得 {got}"


# ─────────────────────── D：旧快照（缺键回退）───────────────────────

def test_old_snapshot_missing_keys_still_resumable(tmp_path: Path):
    snap_dir = tmp_path / "snap"
    _cli([*SMALL, "--ticks", "20", "--out", str(tmp_path / "a.csv"),
          "--save-every", "20", "--snapshot-dir", str(snap_dir)])
    snap, _, _ = _ckpt_paths(snap_dir, TAG)
    data = dict(np.load(snap, allow_pickle=True))
    for k in ("codebook", "learning_count", "state", "out_taken", "feed_fast", "feed_slow",
              "ars_extensive", "heading", "giveup_ct", "sub_r", "sub_c", "health",
              "stomach_scav"):
        data.pop(k, None)                      # 模拟"旧格式快照"：可选键全缺
    old = tmp_path / "legacy.snapshot.npz"
    np.savez_compressed(old, **data)

    eng, meta, start = load_ckpt(old)          # 缺 meta ⇒ 标签从快照配置反推（不报错）
    assert start == 20
    assert (int(meta["patches"]), int(meta["seed"])) == (30, 42)
    for _ in range(5):                         # 能继续跑（缺键按文档回退）
        eng.step()


# ─────────────────────── E：防呆（禁静默错位）───────────────────────

def test_sample_mismatch_fails_loud(tmp_path: Path):
    snap_dir = tmp_path / "snap"
    _cli([*SMALL, "--ticks", "20", "--out", str(tmp_path / "a.csv"),
          "--save-every", "20", "--snapshot-dir", str(snap_dir)])
    r = _cli(["--resume-from", str(snap_dir), "--ticks", "30", "--sample", "5",
              "--out", str(tmp_path / "b.csv")], expect_ok=False)
    assert r.returncode != 0
    assert "续跑采样节拍不一致" in (r.stdout + r.stderr), "采样节拍不一致必须 fail-loud"