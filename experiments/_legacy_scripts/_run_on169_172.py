# -*- coding: utf-8 -*-
"""本机 169-172 on 4 路起跑（T1b 冻结树 ecdf93d/be79ffe；off 段完成后接力）。"""
import os
import subprocess
import sys
import time

ROOT = r"C:\Users\圣羽\Desktop\temp\tempCode\the-world"
TREE = os.path.join(ROOT, "_rerun_logs", "flagship60k_on")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
OUT = os.path.join(TREE, "results")
SNAP = os.path.join(OUT, "snap")
os.makedirs(SNAP, exist_ok=True)

SEEDS = ["169", "170", "171", "172"]
procs = []
for s in SEEDS:
    tag = f"{s}_on"
    logf = open(os.path.join(OUT, f"f60k_{tag}.log"), "w", encoding="utf-8")
    cmd = [PY, "-u", "-m", "experiments.s3_memory_probe", "--seeds", s, "--arms", "on",
           "--weight-gene",
           "--ticks", "60000", "--sample", "2000",
           "--save-every", "2000", "--snapshot-dir", SNAP,
           "--out", os.path.join(OUT, f"f60k_{tag}.csv")]
    p = subprocess.Popen(cmd, cwd=TREE, stdout=logf, stderr=subprocess.STDOUT)
    procs.append((tag, p, logf))
    print(f"[spawn] {tag} pid={p.pid}", flush=True)
    time.sleep(3)

print("[stage] on×4（T1b 版）已起跑，进入监控（120s 周期）", flush=True)
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(120)
    alive = [t for t, p, _ in procs if p.poll() is None]
    print(f"[mon] {time.strftime('%H:%M')} alive={len(alive)}", flush=True)

rc = 0
for t, p, logf in procs:
    logf.close()
    print(f"[done] {t} rc={p.returncode}", flush=True)
    rc |= (p.returncode or 0)
print(f"[all] on×4 完成 rc={rc}", flush=True)
sys.exit(1 if rc else 0)
