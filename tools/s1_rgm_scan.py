"""S1 诊断扫描驱动器（R217 §三②，`[本地开发]` 老工，2026-09-27）。

**要回答的一句话**：K=3 000 被什么限制？

做法：**只扫一个变量** —— `patch_regrowth_mult ∈ {1.195, 0.8, 0.6, 0.4}` × 3 seed（s42–44），
其余配置与 **T4 逐字段一致**（`results/t4_steady_k_k25/*.summary.json` 的 `switches`）：

    480×960｜patches 480｜pop 2000｜ticks 40000｜sample 250｜subpos on
    speed_max 0.125 / gain 0.125 / subdiv 80（R = 15）｜k 2.5｜max_count 30000

🔴 为什么单进程不行：单 run ≈ 91 min（T4 实测 5453 s）⇒ 12 run 串行 ≈ 18 h。
   本驱动器按 **6 路并发 × 2 轮** 跑（R217 已批准的成本模型：≈ 81 min/轮 ⇒ 总 ≈ 2.7 h）。

预注册判据（R217 §三，**先写死后看数**）：
  * 死因以 **STARVATION 为主** ⇒ 食物确是瓶颈（但要再分"够不着 vs 不够吃"）
  * 死因以 **OLD_AGE / PREDATION 为主** ⇒ K 由能量收支/寿命/捕食顶上 ⇒ **S2 会让 K 下降**（代价必须量化）
  * 扫描中**饱和度随 mult 下降而明显下降** ⇒ 找到临界点 ⇒ S2 可直接用这个参数（**不必改机制**）
  * 扫描中**饱和度始终 ~0.85** ⇒ 单靠降再生不够 ⇒ **必须上局部休耕**（`resource_dynamics`）

用法：

    .venv/Scripts/python.exe -X utf8 tools/s1_rgm_scan.py --dry-run      # 只打印 12 条命令
    .venv/Scripts/python.exe -X utf8 tools/s1_rgm_scan.py                # 6 路 × 2 轮
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 🔴 解释器**必须显式指定**：本驱动器也用于**冻结树**（`git archive` 提取的副本，
#    其中没有 `.venv`，因为 `.venv` 不入库）⇒ 默认取本仓 venv，可用 `--python` 覆盖。
PY = ROOT / ".venv" / "Scripts" / "python.exe"

#: T4 同款配置（`results/t4_steady_k_k25/s42_full.summary.json` 的 switches）
T4 = dict(rows=480, cols=960, patches=480, pop=2000, ticks=40000, sample=250,
          speed_max=0.125, gain=0.125, subdiv=80, k=2.5, max_count=30000)


def build_cmd(rgm: float, seed: int, out_dir: Path, max_minutes: float,
              py: Path | None = None) -> list[str]:
    probe = ROOT / "experiments" / "steady_k_probe.py"
    return [
        str(py or PY), "-X", "utf8", str(probe),
        "--rows", str(T4["rows"]), "--cols", str(T4["cols"]),
        "--patches", str(T4["patches"]), "--pop", str(T4["pop"]),
        "--ticks", str(T4["ticks"]), "--sample", str(T4["sample"]),
        "--seeds", str(seed), "--subpos", "on",
        "--speed-max", str(T4["speed_max"]), "--gain", str(T4["gain"]),
        "--subdiv", str(T4["subdiv"]), "--k", str(T4["k"]),
        "--max-count", str(T4["max_count"]), "--max-minutes", str(max_minutes),
        "--patch-regrowth-mult", str(rgm),
        "--out", str(out_dir / f"rgm{rgm:g}_s{seed}.csv"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgm", default="1.195,0.8,0.6,0.4")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out-dir", default="results/s1_diag")
    ap.add_argument("--jobs", type=int, default=6, help="并发路数（R217 批准 6）")
    ap.add_argument("--max-minutes", type=float, default=150.0, help="单 run 墙钟上限")
    ap.add_argument("--python", default=None,
                    help="解释器路径（冻结树场景必给：本仓的 .venv/Scripts/python.exe）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    py = Path(a.python) if a.python else None
    rgms = [float(x) for x in a.rgm.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    out_dir = ROOT / a.out_dir
    jobs = [(r, s) for r in rgms for s in seeds]
    print(f"== S1 诊断扫描：{len(jobs)} run（{len(rgms)} 档 rgm × {len(seeds)} seed）"
          f"，{a.jobs} 路并发 ==")
    print(f"   配置 = T4 同款：{T4}")
    print(f"   输出 = {out_dir}/rgm<档>_s<seed>.csv（+ .log/.summary.json）")
    if a.dry_run:
        for i, (r, s) in enumerate(jobs, 1):
            print(f"\n[{i}/{len(jobs)}] "
                  + " ".join(build_cmd(r, s, out_dir, a.max_minutes, py)))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done: list[dict] = []
    for rnd in range(0, len(jobs), a.jobs):
        batch = jobs[rnd:rnd + a.jobs]
        procs = []
        for r, s in batch:
            log = open(out_dir / f"rgm{r:g}_s{s}.log", "w", encoding="utf-8")
            cmd = build_cmd(r, s, out_dir, a.max_minutes, py)
            print(f"  ▶ 启动 rgm={r:g} seed={s}", flush=True)
            procs.append((r, s, subprocess.Popen(cmd, cwd=str(ROOT), stdout=log,
                                                 stderr=subprocess.STDOUT), log))
        for r, s, p, log in procs:
            rc = p.wait()
            log.close()
            print(f"  ✔ 结束 rgm={r:g} seed={s} rc={rc}"
                  f"（累计 {time.time() - t0:.0f}s）", flush=True)
            done.append({"rgm": r, "seed": s, "rc": rc})

    print(f"\n== 全部结束：{len(done)}/{len(jobs)}，总墙钟 {(time.time() - t0) / 60:.1f} min ==")
    bad = [d for d in done if d["rc"] != 0]
    if bad:
        print(f"  ⚠️ 非 0 退出 {len(bad)} 个：{bad}（逐个看 .log 尾部）")
    # 采集各 run 的 summary 摘要（判读见 R217 §三）
    rows = []
    for r, s in jobs:
        sp = out_dir / f"rgm{r:g}_s{s}.summary.json"
        if not sp.exists():
            continue
        d = json.loads(sp.read_text(encoding="utf-8"))
        res = d["result"]
        rows.append((r, s, res["K_measured"], res["saturation_tail"],
                     res["death_cause_tail"], res["max_gen"], res["stop"]))
    if rows:
        print(f"\n  {'rgm':>6}{'seed':>5}{'K实测':>8}{'饱和度':>9}"
              f"{'饿/老/捕':>16}{'代数':>6}  终止")
        for r, s, k_, sat, dt, mg, stop in rows:
            sh = (f"{dt['share_starvation']}/{dt['share_old_age']}/{dt['share_predation']}"
                  if dt["n"] else "n/a")
            print(f"  {r:>6g}{s:>5}{k_:>8}{sat:>9}{sh:>16}{mg:>6.0f}  {stop}")
        (out_dir / "s1_scan_table.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  表已写入 {out_dir / 's1_scan_table.json'}")


if __name__ == "__main__":
    sys.exit(main())
