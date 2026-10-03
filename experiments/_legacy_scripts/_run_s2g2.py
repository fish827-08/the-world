# -*- coding: utf-8 -*-
"""S2 判据② 新档批的编排器（R329/预注册 v2）。

⚠️ 为什么需要它：`experiments/s2_depletion_probe.py` 的 main 是**串行**的
   （`for seed: for arm: run_one(...)`）⇒ 单进程会把 12 个 run 顺序跑完。
   本器把每个 **(seed, arm)** 切成**独立单 run 作业**，用 `--par` 路并行。

用法：
  python _run_s2g2.py --tree <冻结树> --out <产物目录>
      --seeds 201,202,203,204,205,206 --ticks 40000 --sample 250 --par 4
      [--device s2] [--wait-glob "<dir>/*.csv" --wait-rows 13]
"""
import argparse
import glob
import os
import subprocess
import sys
import time

# 🔴 R331b：**改为 `sys.executable`**（原为硬编码本机 Windows 路径）
#   ⇒ 跨机可移植（云机器/云启沙盒直接用各自 venv 启动本器即可），避免
#   `FileNotFoundError: C:\Users\...\.venv\Scripts\python.exe`。
PY = sys.executable


def wait_for(wait_glob, wait_rows, poll=60, max_h=8.0):
    if not wait_glob:
        return
    t0 = time.time()
    print("[wait] 等前置批完成（%s 需 %d 行）…" % (wait_glob, wait_rows), flush=True)
    while time.time() - t0 < max_h * 3600:
        fs = sorted(glob.glob(wait_glob))
        if fs and all(sum(1 for _ in open(f, encoding="utf-8")) >= wait_rows for f in fs):
            print("[wait] 前置完成（%d 个文件）⇒ 开工" % len(fs), flush=True)
            time.sleep(60)
            return
        time.sleep(poll)
    print("[wait] 超时 ⇒ 仍开工（请人工核对）", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--arms", default="both", help="both / off / on（rd 开=on）")
    ap.add_argument("--ticks", type=int, default=40000)
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--device", default="s2", help="显式装置预设名（拿回显行）")
    ap.add_argument("--extra", default="", help="附加透传参数（空格分隔）")
    ap.add_argument("--wait-glob", default="")
    ap.add_argument("--wait-rows", type=int, default=0)
    ap.add_argument("--wait-max-h", type=float, default=8.0)
    a = ap.parse_args()

    a.tree = os.path.abspath(a.tree)
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    if a.wait_rows:
        wait_for(a.wait_glob, a.wait_rows, max_h=a.wait_max_h)

    arms = ["off", "on"] if a.arms == "both" else [a.arms]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    jobs = [(sd, arm) for arm in arms for sd in seeds]

    print("=== 共 %d 个 run（单 run 作业），%d 路并行 ===" % (len(jobs), a.par), flush=True)
    t0 = time.time()
    running, fails = [], []
    while jobs or running:
        while jobs and len(running) < a.par:
            sd, arm = jobs.pop(0)
            tag = "s2g2_%d_%s" % (sd, arm)
            cmd = [PY, "-u", "-m", "experiments.s2_depletion_probe",
                   "--seeds", str(sd), "--arms", arm,
                   "--ticks", str(a.ticks), "--sample", str(a.sample),
                   "--device", a.device,
                   "--out", os.path.join(a.out, tag + ".csv")]
            if a.extra:
                cmd += a.extra.split()
            log = open(os.path.join(a.out, tag + ".log"), "w", encoding="utf-8")
            running.append((tag, subprocess.Popen(cmd, cwd=a.tree, stdout=log,
                                                  stderr=subprocess.STDOUT)))
            print("[start] %s  (%.1f min)" % (tag, (time.time() - t0) / 60), flush=True)
        time.sleep(5)
        still = []
        for tag, p in running:
            if p.poll() is None:
                still.append((tag, p))
            else:
                print("[done ] %s rc=%d  (%.1f min)" % (tag, p.returncode,
                                                        (time.time() - t0) / 60), flush=True)
                if p.returncode != 0:
                    fails.append(tag)
        running = still
    print("ALLDONE 总耗时 %.1f min，失败 %s" % ((time.time() - t0) / 60, fails), flush=True)


if __name__ == "__main__":
    main()
