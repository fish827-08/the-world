"""T6 交付证据：变异测试转录（scratch，不入库）。"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
TOOL = ROOT / "tools" / "tick_denomination_audit.py"
TARGET = ROOT / "world" / "light_and_temperature.py"
INJECT = b"\n_A2B_MUTANT_DURATION_TICKS = 137  # duration\n"


def run_tool() -> str:
    r = subprocess.run([PY, str(TOOL), "--no-md"] if False else [PY, str(TOOL)],
                       cwd=str(ROOT), capture_output=True)
    return r.stdout.decode("utf-8", errors="replace")


def hits(out: str) -> list[str]:
    return [ln.strip() for ln in out.splitlines()
            if "light_and_temperature" in ln and "DURATION" in ln]


orig = TARGET.read_bytes()
sha_before = hashlib.sha256(orig).hexdigest()[:12]
print(f"[0] 注入前 sha256[:12] = {sha_before}")

out0 = run_tool()
print(f"[1] 注入前：命中行 = {hits(out0) or '（无）'}  ✅ 应为空")

lineno = orig.count(b"\n") + 2
TARGET.write_bytes(orig + INJECT)
print(f"[2] 注入 `_A2B_MUTANT_DURATION_TICKS = 137`（第 {lineno} 行）")

out1 = run_tool()
h1 = hits(out1)
print(f"[3] 注入后：命中行 = {h1}")
assert h1, "变异未被检出 ⇒ 工具无效"
assert "DURATION" in h1[0] and "137" in h1[0], h1

TARGET.write_bytes(orig)
sha_after = hashlib.sha256(TARGET.read_bytes()).hexdigest()[:12]
print(f"[4] 还原后 sha256[:12] = {sha_after} ｜ 一致 = {sha_after == sha_before}")

out2 = run_tool()
print(f"[5] 还原后：命中行 = {hits(out2) or '（无）'}  ✅ 应为空")
assert not hits(out2)
print("⇒ 变异测试通过：检出力在**新增覆盖**（world/light_and_temperature.py 旧版根本不在清单）上成立。")
