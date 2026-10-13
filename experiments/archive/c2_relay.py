"""C2 接力器 —— 等 cstep1b（main×6）跑完后自动启动 cstep2rand（rand_m1.3×6）。

R114a：C1b/C2 **串行两批**（单实例锁约束）。
本脚本只在「cstep1b 6 份 summary 齐 且 锁已释放」时才启动，且**不重复启动**
（启动前再查一次锁；runner 自身也有单实例锁兜底）。
日志：_rerun_logs/_c2_relay.log
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "_rerun_logs" / "_c2_relay.log"
LOCK = ROOT / "_rerun_logs" / ".batch_runner.lock"
B1 = ROOT / "_rerun_logs" / "cstep1b"
B2 = ROOT / "_rerun_logs" / "cstep2rand"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def lock_active() -> bool:
    if not LOCK.exists():
        return False
    try:
        info = json.loads(LOCK.read_text(encoding="utf-8"))
    except Exception:
        return True                      # 内容异常 ⇒ 保守认为仍在跑
    pid = int(info.get("pid", 0))
    if not pid:
        return True
    try:                                 # Windows：OpenProcess 存活探测
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        k32.CloseHandle(h)
        return True
    except Exception:
        return True


def main() -> int:
    log("C2 接力器启动；等待 cstep1b 完成（6 份 summary + 锁释放）")
    deadline = time.time() + 8 * 3600
    while time.time() < deadline:
        n1 = len(list(B1.glob("*.summary.json"))) if B1.exists() else 0
        busy = lock_active()
        if n1 >= 6 and not busy:
            log(f"cstep1b 完成（{n1} 份）且锁已释放 ⇒ 启动 cstep2rand")
            break
        time.sleep(60)
    else:
        log("等待超时（8h）⇒ 不启动 C2，请人工核")
        return 1

    if lock_active():
        log("锁仍被占用 ⇒ 不启动（避免与其它批冲突）")
        return 1
    B2.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        rc = subprocess.call(
            [PY, str(ROOT / "experiments" / "batch_runner.py"),
             "--preset", "cstep2rand", "--skip-existing"],
            cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
        )
    log(f"cstep2rand 结束 rc={rc}；summary 数="
        f"{len(list(B2.glob('*.summary.json')))}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
