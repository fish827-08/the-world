# -*- coding: utf-8 -*-
"""tests/test_share_lock.py — `tools/share_lock.py` 回归测试（2026-09-17）

覆盖：
  1. acquire -> status -> release 基本链
  2. 他人活跃锁 => acquire 拒绝（退出码 2）
  3. 同 slot 重复 acquire => 拒绝（退出码 4）
  4. 过期锁: status 标 stale；他人 acquire 时自动清理并通过
  5. stale-clean 的 dry-run / --yes 行为
  6. release 不存在的锁 => 退出码 1
全部用临时锁目录（--lock-dir），不触碰真实 `_share/.locks/`。
"""
import json
import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "share_lock.py")


def run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, TOOL, *args],
        capture_output=True, text=True, timeout=60, cwd=cwd)


@pytest.fixture()
def lock_dir(tmp_path):
    return str(tmp_path / "locks")


def test_acquire_status_release(lock_dir):
    r = run("acquire", "--slot", "dev", "--task", "t1", "--lock-dir", lock_dir)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "已获取锁" in r.stdout

    r2 = run("status", "--json", "--lock-dir", lock_dir)
    j = json.loads(r2.stdout)
    assert len(j["locks"]) == 1
    assert j["locks"][0]["slot"] == "dev"
    assert j["locks"][0]["state"] == "active"
    assert j["locks"][0]["task"] == "t1"

    r3 = run("release", "--slot", "dev", "--lock-dir", lock_dir)
    assert r3.returncode == 0

    r4 = run("status", "--json", "--lock-dir", lock_dir)
    assert json.loads(r4.stdout)["locks"] == []


def test_conflict_other_slot(lock_dir):
    assert run("acquire", "--slot", "dev", "--lock-dir", lock_dir).returncode == 0
    r = run("acquire", "--slot", "eval", "--lock-dir", lock_dir)
    assert r.returncode == 2
    assert "拒绝" in r.stdout
    assert "dev" in r.stdout  # 打印了持有者


def test_duplicate_same_slot(lock_dir):
    assert run("acquire", "--slot", "dev", "--lock-dir", lock_dir).returncode == 0
    r = run("acquire", "--slot", "dev", "--lock-dir", lock_dir)
    assert r.returncode == 4


def test_refresh(lock_dir):
    assert run("acquire", "--slot", "dev", "--ttl", "60", "--lock-dir", lock_dir).returncode == 0
    r = run("acquire", "--slot", "dev", "--ttl", "120", "--refresh", "--lock-dir", lock_dir)
    assert r.returncode == 0
    j = json.loads(run("status", "--json", "--lock-dir", lock_dir).stdout)
    assert j["locks"][0]["ttl_s"] == 120


def test_stale_auto_clean_on_acquire(lock_dir):
    # TTL=1s 的锁，等 2 秒即过期
    assert run("acquire", "--slot", "dev", "--ttl", "1", "--lock-dir", lock_dir).returncode == 0
    time.sleep(2.1)
    j = json.loads(run("status", "--json", "--lock-dir", lock_dir).stdout)
    assert j["locks"][0]["state"] == "stale"
    # 他人 acquire 应自动清掉过期锁并通过
    r = run("acquire", "--slot", "eval", "--lock-dir", lock_dir)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "已移出过期锁" in r.stdout
    j2 = json.loads(run("status", "--json", "--lock-dir", lock_dir).stdout)
    assert len(j2["locks"]) == 1 and j2["locks"][0]["slot"] == "eval"


def test_stale_clean_dry_run_and_yes(lock_dir):
    assert run("acquire", "--slot", "dev", "--ttl", "1", "--lock-dir", lock_dir).returncode == 0
    time.sleep(2.1)
    # dry-run: 不挪
    r = run("stale-clean", "--lock-dir", lock_dir)
    assert r.returncode == 0 and "将移出" in r.stdout
    assert os.path.exists(os.path.join(lock_dir, "dev.lock"))
    # --yes: 挪出（F-R10 家族：本机不删除，挪移至 _retired/）
    r2 = run("stale-clean", "--yes", "--lock-dir", lock_dir)
    assert r2.returncode == 0 and "已移出" in r2.stdout
    assert not os.path.exists(os.path.join(lock_dir, "dev.lock"))


def test_release_missing(lock_dir):
    r = run("release", "--slot", "dev", "--lock-dir", lock_dir)
    assert r.returncode == 1
