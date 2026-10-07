"""E-057 rep_w 档扫批编排（卡 BATCH-SCRIPT；澜舟 2026-10-06）。

干什么
------
把 PI 会签 @`2306949` 锁定的 **32 run 四臂矩阵**（`--rep-w 0 / 0.05 / 0.2 / 0.5`，
四臂**全部** `--rep-w-solo` × seed 207–214 × 2000t × sample 100，s2 装置全字段）
展开成逐 run argv，并在**云机**按固定 lane 数串行执行：逐 run 读 summary 回显键核对
（不符 ⇒ fail-loud 停该 lane，后续 run 不起）、随跑随登 manifest（R277 四要素）与
`status.txt`，支持断点续跑。

三模式
------
    --plan     只打印全 argv + 矩阵/装置核对表：**零子进程、零写盘**（dry-run 演示上板）
    --verify   读已有产物目录逐 run 核对回显键（跑后复核 / 读数器前置自检）
    --run      云机实跑。非 Linux 平台直接拒 ⇒ 把 R117/R398「本机不跑批」写进代码门

纪律映射
--------
· **装置锁定即代码**：`DEVICE`/`TICKS`/`SAMPLE`/`TIERS`/`SEEDS` = 预注册 §一 表格的字面化，
  `--plan` 阶段逐 argv 静态守（`check_argv`），偏离即 exit 2 —— 不允许"跑出来才发现错档"；
· **不消费 RNG**：本模块只拼 argv + 读 JSON，**不 import 引擎**（单测有 import 守卫）；
· **回显键为准**：臂识别读 summary `reputation_weight_effective` / `info_structure_enabled_effective`
  （砚 22:13 对齐请求 ①），文件名 `rpw0/rpw5/rpw20/rpw50` 只是人眼标签（不含小数点，
  避免 `os.path.splitext` 在中间点上歧义）；
· **跑前/跑中核对**：每个 run 完成即核（`verify_summary`），四臂全要求
  `info_structure_enabled_effective is True`；锚臂 `rep_w=0.0` 亦 solo ⇒ 生效值 0.0 合法，
  但 `rep_w>0` 若生效 0.0 就是空转（引擎 `sphere_engine.py:3999` 门控）⇒ 当场红；
· **ping 砚**：脚本只落 `status.txt`/`PING_YAN.txt` 并打印待发帖文本（板上/@ 的动作归人，
  不让云机无人值守时写 `_share`——那是 share-lock 仪式领地）。假设已按 R387 写进产出。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

try:  # R98：GBK 控制台兜底
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

# ---------------- 锁定矩阵（预注册 §一 / PI 会签 @2306949；改这里=改判据装置，禁） ----------------

PROBE = "experiments.s3_memory_probe"
DEVICE = "s2"
FIELD_LOCK = {                        # --device s2 之外再显式写一遍（E-056 同款双保险）
    "rows": "480", "cols": "960", "patches": "1700", "pop": "10000",
    "rgm": "1.195", "bg_low_prod_frac": "0.0", "bg_low_cap_mult": "0.0",
}
TICKS, SAMPLE = "2000", "100"
ARMS, RD_MODE, ALPHABET = "off", "off", "8"
M0_EMIT_WINDOW_FRAC = "0.25"
TIERS: tuple[float, ...] = (0.0, 0.05, 0.2, 0.5)      # 锚臂 0.0 + 三档（档位=鱼直令锁定）
SEEDS: tuple[int, ...] = tuple(range(207, 215))       # 与已用 42–100 错开（R255）
TIER_TAG = {0.0: "rpw0", 0.05: "rpw5", 0.2: "rpw20", 0.5: "rpw50"}
EXPECTED_RUNS = len(TIERS) * len(SEEDS)               # 32
CARRIER_BASE = "7faca82"             # PI 批载体指纹（main 台账行；批窗内代码面冻结）


class PlanError(Exception):
    """矩阵/装置偏离 ⇒ 用法错（exit 2），绝不起批。"""


class EchoError(Exception):
    """回显键与声明档位不符 / 产物缺失 ⇒ 执行期红（exit 4）。"""


def tier_str(tier: float) -> str:
    return repr(float(tier))          # 0.0 / 0.05 / 0.2 / 0.5（argparse 直接可解析）


def out_name(tier: float, seed: int) -> str:
    return f"e057_{TIER_TAG[tier]}_d{FIELD_LOCK['patches']}_s{seed}_t{TICKS}"


def build_argv(tier: float, seed: int, data_dir: Path) -> list[str]:
    """逐 run argv（= 云机执行的那条命令行，非二手描述）。"""
    if tier not in TIERS:
        raise PlanError(f"🔴 档位 {tier!r} 不在锁定矩阵 {TIERS}（改档=改装置，禁）")
    csv = data_dir / "e057_repw" / (out_name(tier, seed) + ".csv")
    argv = [f"-m", PROBE, "--device", DEVICE]
    argv += ["--rows", FIELD_LOCK["rows"], "--cols", FIELD_LOCK["cols"],
             "--patches", FIELD_LOCK["patches"], "--pop", FIELD_LOCK["pop"],
             "--rgm", FIELD_LOCK["rgm"],
             "--bg-low-prod-frac", FIELD_LOCK["bg_low_prod_frac"],
             "--bg-low-cap-mult", FIELD_LOCK["bg_low_cap_mult"],
             "--seeds", str(seed), "--ticks", TICKS, "--sample", SAMPLE,
             "--arms", ARMS, "--rd-mode", RD_MODE, "--alphabet", ALPHABET,
             "--m0-instruments", "--m0-emit-window-frac", M0_EMIT_WINDOW_FRAC,
             # 🔴 四臂全 solo（PI 裁定=用）：锚臂亦然 ⇒ 同树同指纹只差 rep_w
             "--rep-w", tier_str(tier), "--rep-w-solo",
             "--out", str(csv)]
    return argv


def flag_of(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def check_argv(tier: float, seed: int, argv: list[str]) -> None:
    """静态装置守：任何一条不符即 PlanError（不起批）。"""
    need = {
        "--device": DEVICE, "--rows": FIELD_LOCK["rows"], "--cols": FIELD_LOCK["cols"],
        "--patches": FIELD_LOCK["patches"], "--pop": FIELD_LOCK["pop"],
        "--rgm": FIELD_LOCK["rgm"], "--bg-low-prod-frac": FIELD_LOCK["bg_low_prod_frac"],
        "--bg-low-cap-mult": FIELD_LOCK["bg_low_cap_mult"],
        "--seeds": str(seed), "--ticks": TICKS, "--sample": SAMPLE,
        "--arms": ARMS, "--rd-mode": RD_MODE, "--alphabet": ALPHABET,
        "--m0-emit-window-frac": M0_EMIT_WINDOW_FRAC, "--rep-w": tier_str(tier),
    }
    for flg, want in need.items():
        got = flag_of(argv, flg)
        if got != want:
            raise PlanError(f"🔴 {TIER_TAG[tier]}/s{seed}：{flg} 应为 {want!r}，实为 {got!r}")
    if "--rep-w-solo" not in argv:
        raise PlanError(f"🔴 {TIER_TAG[tier]}/s{seed}：缺 --rep-w-solo"
                        f"（裸 --rep-w 在 enabled=False 下空转，引擎 :3999）")
    if "--m0-instruments" not in argv:
        raise PlanError(f"🔴 {TIER_TAG[tier]}/s{seed}：缺 --m0-instruments（仪表必开，骨架 §三-2）")


def plan_runs(data_dir: Path) -> list[dict]:
    """矩阵展开 ⇒ 逐 run 规格（顺序=档位优先，lane 内即此序切片）。"""
    runs: list[dict] = []
    for tier in TIERS:
        for seed in SEEDS:
            argv = build_argv(tier, seed, data_dir)
            check_argv(tier, seed, argv)
            csv = Path(argv[argv.index("--out") + 1])
            runs.append({
                "name": out_name(tier, seed), "tier": tier, "tier_declared": tier_str(tier),
                "seed": seed, "argv": argv, "csv": csv,
                "summary": csv.with_suffix(".summary.json"),
                "log": csv.parent.parent / "e057_logs" / f"{out_name(tier, seed)}.log",
            })
    if len(runs) != EXPECTED_RUNS:
        raise PlanError(f"🔴 矩阵应 {EXPECTED_RUNS} run，实展 {len(runs)}")
    if len({r["csv"] for r in runs}) != EXPECTED_RUNS:
        raise PlanError("🔴 逐 run 输出路径不唯一（会互相覆盖）")
    return runs


# ---------------- 回显键核对（预注册 §一「跑前逐 run 核对」项） ----------------

def verify_summary(run: dict) -> dict:
    """读 summary 核对装置/回显；不过 ⇒ EchoError。返回核对摘（进 manifest）。"""
    sp = run["summary"]
    if not sp.exists():
        raise EchoError(f"🔴 {run['name']}：summary 缺失 {sp}")
    if not run["csv"].exists() or run["csv"].stat().st_size == 0:
        raise EchoError(f"🔴 {run['name']}：CSV 缺失或空 {run['csv']}")
    d = json.loads(sp.read_text(encoding="utf-8"))
    runs = d.get("runs") or []
    if len(runs) != 1:
        raise EchoError(f"🔴 {run['name']}：应单 seed 单臂（runs 长 1），实为 {len(runs)}")
    r0, meta = runs[0], d.get("meta", {})
    if r0.get("seed") != run["seed"]:
        raise EchoError(f"🔴 {run['name']}：seed 回显 {r0.get('seed')!r} ≠ {run['seed']}")
    if r0.get("arm") != "mem_off":
        raise EchoError(f"🔴 {run['name']}：臂应为 mem_off（--arms off），回显 {r0.get('arm')!r}")
    en, eff = r0.get("info_structure_enabled_effective"), r0.get("reputation_weight_effective")
    if en is not True:
        raise EchoError(f"🔴 {run['name']}：info_structure_enabled_effective={en!r}，"
                        f"四臂全 solo ⇒ 必须 True（solo 形制没落地 = 整批白跑）")
    if eff is None or abs(float(eff) - float(run["tier_declared"])) > 1e-12:
        raise EchoError(f"🔴 {run['name']}：reputation_weight_effective={eff!r} ≠ "
                        f"声明档 {run['tier_declared']}（空转/漂移，引擎 :3999）")
    if meta.get("rep_w") != float(run["tier_declared"]) or meta.get("rep_w_solo") is not True:
        raise EchoError(f"🔴 {run['name']}：meta 回显 rep_w={meta.get('rep_w')!r} "
                        f"rep_w_solo={meta.get('rep_w_solo')!r} 与声明不符（R277）")
    for want_k, want in (("rd_enabled_effective", False),
                         ("signal_alphabet_effective", ALPHABET)):
        got = r0.get(want_k)
        if got != want:
            raise EchoError(f"🔴 {run['name']}：{want_k}={got!r} ≠ {want!r}（rd/alphabet 锁破）")
    n_csv = sum(1 for _ in open(run["csv"], encoding="utf-8", newline="")) - 1
    return {
        "reputation_weight_effective": float(eff),
        "info_structure_enabled_effective": True,
        "meta_rep_w": meta.get("rep_w"), "meta_rep_w_solo": meta.get("rep_w_solo"),
        "rd_enabled_effective": r0.get("rd_enabled_effective"),
        "signal_alphabet_effective": r0.get("signal_alphabet_effective"),
        "stop": r0.get("stop"), "wall_s": r0.get("wall_s"), "csv_rows": n_csv,
    }


# ---------------- 环境门（平台 / 内存 / 载体指纹） ----------------

def avail_gb() -> float:
    """可用内存 GB（Linux 读 /proc/meminfo；其他平台按可用比例粗算）。"""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            kv = {}
            for line in f:
                k, _, v = line.partition(":")
                kv[k.strip()] = v
            return int(kv["MemAvailable"].split()[0]) / 1048576.0
    except Exception:
        pass
    try:
        import ctypes

        class MSX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MSX()
        m.dwLength = ctypes.sizeof(MSX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.ullAvailPhys / 1073741824.0
    except Exception:
        pass
    return float("inf")


def carrier_head() -> tuple[str, str]:
    """批载体 HEAD + 基线核对（R342：脚本必须入库、指纹必须留档）。"""
    def _git(*args: str) -> str:
        r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            raise EchoError(f"🔴 git {' '.join(args)} 失败：{r.stderr.strip()[:200]}")
        return r.stdout.strip()

    head = _git("rev-parse", "HEAD")
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", CARRIER_BASE, "HEAD"],
                         cwd=str(ROOT), capture_output=True, text=True)
    if anc.returncode != 0:
        raise EchoError(f"🔴 载体基线破：{CARRIER_BASE}（PI 批载体指纹）不是 {head[:7]} 的祖先"
                        f" ⇒ 批窗代码面/分支落点不对，不起批")
    return head[:7], CARRIER_BASE


def check_run_platform(min_free_gb: float) -> None:
    if platform.system() != "Linux":
        raise EchoError(
            f"🔴 R117/R398：--run 只在云机（Linux）起，当前={platform.system()}。"
            f"本机不跑批 ⇒ 请用 --plan/--verify；确要在非云机跑请显式 --allow-non-cloud。")
    a = avail_gb()
    if a < min_free_gb:
        raise EchoError(f"🔴 可用内存 {a:.2f} GB < 门槛 {min_free_gb:.2f} GB（2G 机须留余量，勿硬起）")


# ---------------- 产物登记（R277 四要素 + status + ping 文本） ----------------

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": {}}


def manifest_row(run: dict, rc: int, echo: dict | None, head: str, err: str | None = None) -> dict:
    """R277 四要素（rows/cols/patches/pop）逐字段 + 全 argv + 回显 + 载体指纹。"""
    row = {
        "name": run["name"], "seed": run["seed"], "tier_declared": run["tier_declared"],
        "rows": int(FIELD_LOCK["rows"]), "cols": int(FIELD_LOCK["cols"]),
        "patches": int(FIELD_LOCK["patches"]), "pop": int(FIELD_LOCK["pop"]),
        "rgm": float(FIELD_LOCK["rgm"]), "ticks": int(TICKS), "sample": int(SAMPLE),
        "arms": ARMS, "rd_mode": RD_MODE, "alphabet": ALPHABET,
        "m0_instruments": True, "m0_emit_window_frac": float(M0_EMIT_WINDOW_FRAC),
        "rep_w_solo": True, "device": DEVICE,
        "argv": ["python"] + run["argv"],
        "csv": str(run["csv"]), "summary": str(run["summary"]), "log": str(run["log"]),
        "rc": rc, "carrier_head": head, "carrier_base": CARRIER_BASE,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if echo is not None:
        row["echo"] = echo
    if err is not None:
        row["error"] = err
    return row


def status_append(status: Path, line: str) -> None:
    status.parent.mkdir(parents=True, exist_ok=True)
    with open(status, "a", encoding="utf-8") as f:
        f.write(f"{line} {time.strftime('%H:%M:%S')}\n")


PING_TEXT = ("E-057 rep_w 档扫批跑完：32 run（rpw0 锚臂/rpw5/rpw20/rpw50 × seed207-214，"
             "2000t×sample100，s2 全字段 + solo 形制），status 全 RC=0，"
             "逐 run 回显键 reputation_weight_effective/info_structure_enabled_effective 已核。"
             "产物=数据仓 e057_repw/ + e057_logs/ + e057_manifest.json。@砚 请出读数全表。")


# ---------------- 执行器（lane 串行 + 断点续跑 + fail-loud） ----------------

def complete(run: dict, strict_resume: bool) -> bool:
    """断点续跑判据：CSV+summary 在场且回显核得过 ⇒ 跳；不过 ⇒ 重跑（自愈）。"""
    if not (run["csv"].exists() and run["summary"].exists()):
        return False
    if strict_resume:
        return True
    try:
        verify_summary(run)
    except (EchoError, json.JSONDecodeError):
        return False
    return True


def run_one_lane(lane_id: int, runs: list[dict], ctx: dict) -> None:
    """一条 lane 串行跑分到的 run；任一红 ⇒ 记 status 后停该 lane（后续不起）。"""
    lock: threading.Lock = ctx["lock"]
    for run in runs:
        if ctx["aborted"]:
            return
        if complete(run, ctx["strict_resume"]):
            with lock:
                status_append(ctx["status"], f"SKIP {run['name']} (resume)")
            continue
        run["log"].parent.mkdir(parents=True, exist_ok=True)
        with open(run["log"], "w", encoding="utf-8") as lf:
            env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="1")
            rc = subprocess.call([ctx["python"], "-u", *run["argv"]],
                                 cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        echo, err = None, None
        try:
            if rc != 0:
                raise EchoError(f"🔴 {run['name']}：探针 rc={rc}（fail-loud，详见 {run['log']}）")
            echo = verify_summary(run)
        except (EchoError, json.JSONDecodeError, OSError) as e:
            err = str(e)
        with lock:
            ctx["manifest"]["runs"][run["name"]] = manifest_row(
                run, rc, echo, ctx["head"], err)
            _write_json(ctx["manifest_path"], ctx["manifest"])
            if err:
                status_append(ctx["status"], f"ABORT lane{lane_id} {run['name']} RC={rc} {err}")
                ctx["aborted"] = True
                return
            status_append(ctx["status"], f"DONE {run['name']} RC=0")


def run_batch(a: argparse.Namespace) -> int:
    data_dir = Path(a.data_dir)
    if not data_dir.exists():
        raise EchoError(f"🔴 数据仓目录不存在：{data_dir}（产物无处落 = 不起批）")
    (data_dir / "e057_repw").mkdir(parents=True, exist_ok=True)
    (data_dir / "e057_logs").mkdir(parents=True, exist_ok=True)
    head, base = carrier_head()
    runs = plan_runs(data_dir)
    status = data_dir / "e057_logs" / "status.txt"
    manifest_path = data_dir / "e057_repw" / "e057_manifest.json"
    ctx = {
        "python": a.python, "aborted": False, "strict_resume": a.strict_resume,
        "status": status, "head": head, "lock": threading.Lock(),
        "manifest": load_manifest(manifest_path), "manifest_path": manifest_path,
    }
    lanes = max(1, min(a.lanes, len(runs)))
    status_append(status, f"START python={a.python} lanes={lanes} runs={len(runs)} "
                          f"HEAD={head[:7]} base={base}")
    procs = []
    for i in range(lanes):
        chunk = runs[i::lanes]
        procs.append(_LaneThread(i, chunk, ctx))
    for p in procs:
        p.start()
    warn_at = 0.0
    while any(p.alive() for p in procs):
        time.sleep(5)
        if time.time() - warn_at < 300:        # 低内存告警 5 min 节流（不刷屏 status）
            continue
        av = avail_gb()
        if av < a.min_free_gb * 0.35:          # 只排队告警，不杀已有 run（batch_runner 家规）
            with ctx["lock"]:
                status_append(status, f"WARN lowmem avail={av:.2f}GB")
            warn_at = time.time()
    for p in procs:
        p.join()
    bad = [r["name"] for r in runs
           if ctx["manifest"]["runs"].get(r["name"], {}).get("error")]
    missing = [r["name"] for r in runs if r["name"] not in ctx["manifest"]["runs"]]
    _write_json(manifest_path, ctx["manifest"])
    if bad or missing or ctx["aborted"]:
        status_append(status, f"FAILED bad={bad} missing={missing}")
        print(f"❌ 批未全绿：bad={bad} missing={missing}", file=sys.stderr)
        return 4
    status_append(status, "ALLDONE")
    (data_dir / "e057_logs" / "PING_YAN.txt").write_text(PING_TEXT + "\n", encoding="utf-8")
    print(PING_TEXT)
    print(f"[OK] 32/32 绿；manifest={manifest_path}")
    return 0


class _LaneThread:
    """极薄线程壳（lane 内部仍串行 = 与 E-056 laneA/laneB 同形制；不引并行语义风险）。"""

    def __init__(self, lane_id: int, runs: list[dict], ctx: dict):
        self.t = None
        self.lane_id, self.runs, self.ctx = lane_id, runs, ctx

    def start(self) -> None:
        self.t = threading.Thread(target=run_one_lane,
                                  args=(self.lane_id, self.runs, self.ctx), daemon=True)
        self.t.start()

    def alive(self) -> bool:
        return self.t is not None and self.t.is_alive()

    def join(self) -> None:
        if self.t is not None:
            self.t.join()


# ---------------- 只读模式 ----------------

def do_plan(runs: list[dict]) -> int:
    print(f"# E-057 矩阵：{len(runs)} run = 档位 {list(TIERS)} × seed {list(SEEDS)}"
          f"（四臂全 --rep-w-solo）")
    print(f"# 装置锁定：--device {DEVICE} + 显式 "
          + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in FIELD_LOCK.items())
          + f" | ticks={TICKS} sample={SAMPLE} arms={ARMS} rd-mode={RD_MODE} "
            f"alphabet={ALPHABET} m0-instruments=on m0-emit-window-frac={M0_EMIT_WINDOW_FRAC}")
    print(f"# 载体：base={CARRIER_BASE}（批窗内 main 代码面冻结；本脚本在 task/BATCH-SCRIPT）")
    for r in runs:
        print(f"[{r['tier_declared']:>4s}/s{r['seed']}] python " + " ".join(r["argv"]))
    print(f"# 核对：{len(runs)}/{EXPECTED_RUNS} run，逐 argv 装置守全过（含 solo 在场）")
    return 0


def do_verify(a: argparse.Namespace) -> int:
    runs = plan_runs(Path(a.data_dir))
    bad = []
    for r in runs:
        if not r["summary"].exists():
            bad.append((r["name"], "summary 缺失"))
            continue
        try:
            echo = verify_summary(r)
        except EchoError as e:
            bad.append((r["name"], str(e)))
            continue
        print(f"[OK] {r['name']:32s} rep_w_eff={echo['reputation_weight_effective']} "
              f"enabled={echo['info_structure_enabled_effective']} "
              f"rows={echo['csv_rows']} wall={echo['wall_s']}")
    if bad:
        for name, why in bad:
            print(f"[RED] {name}: {why}", file=sys.stderr)
        print(f"❌ 核对未全过 {len(bad)}/{len(runs)}", file=sys.stderr)
        return 4
    print(f"[OK] {len(runs)}/{len(runs)} 回显键核对全过")
    return 0


def default_python() -> str:
    """跑批解释器候选（🔴 不能拿 `--data-dir` 的父目录去猜：数据仓走 worktree 时父目录是
    `<data-repo>/.worktrees`，猜出来的路径必然不存在 —— 云机首次起跑就撞在这）。"""
    exe = "python.exe" if os.name == "nt" else "python"
    cands = [os.environ.get("E057_PY")]
    for root in (ROOT, _main_repo_root()):
        if root is None:
            continue
        cands += [str(root / ".venv" / d / exe) for d in ("bin", "Scripts")]
    home = Path.home() / "world" / "the-world" / ".venv"
    cands += [str(home / d / exe) for d in ("bin", "Scripts")]
    for c in cands:
        if c and Path(c).exists():
            return c
    raise EchoError("🔴 找不到可用解释器（候选：" + " ｜ ".join(x for x in cands if x)
                    + "）⇒ 用 --python 显式给定或设 $E057_PY")


def _main_repo_root() -> Path | None:
    """linked worktree 的**主仓**根（`.venv` 只住在主仓，见 R249）。"""
    try:
        r = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=str(ROOT),
                           capture_output=True, text=True)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    common = Path(r.stdout.strip())
    if not common.is_absolute():
        common = ROOT / common
    return common.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E-057 rep_w 档扫批编排（BATCH-SCRIPT）")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/world/the-world-data"),
                    help="数据仓根（产物落 <data-dir>/e057_repw 与 e057_logs）")
    ap.add_argument("--python", default=None,
                    help="跑批解释器；缺省按 $E057_PY → 本树/.venv → 主仓/.venv → ~/world/the-world/.venv 逐候选择优")
    ap.add_argument("--lanes", type=int, default=2, help="云机 lane 数（2 核 2 路，E-056 实证形制）")
    ap.add_argument("--min-free-gb", dest="min_free_gb", type=float, default=0.6)
    ap.add_argument("--strict-resume", dest="strict_resume", action="store_true",
                    help="产物在场即跳（默认还要回显核得过才跳 ⇒ 自愈式续跑）")
    ap.add_argument("--allow-non-cloud", dest="allow_non_cloud", action="store_true",
                    help="仅供单测/演练：跳过 Linux 平台门（仍不视为正式批）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="打印全 argv（零子进程零写盘）")
    g.add_argument("--verify", action="store_true", help="逐 run 读回显键核对")
    g.add_argument("--run", action="store_true", help="云机实跑（fail-loud）")
    a = ap.parse_args(argv)

    runs = plan_runs(Path(a.data_dir))       # 三模式共用同一展开路径（测的就是跑的）
    try:
        if a.plan:
            return do_plan(runs)
        if a.verify:
            return do_verify(a)
        if not a.allow_non_cloud:
            check_run_platform(a.min_free_gb)
        else:
            print("[WARN] --allow-non-cloud：跳过平台门（演练通道，非正式批）", file=sys.stderr)
        a.python = a.python or default_python()
        if not Path(a.python).exists():
            raise EchoError(f"🔴 解释器不存在：{a.python}（--python 显式给定）")
        return run_batch(a)
    except (PlanError, EchoError) as e:
        print(str(e), file=sys.stderr)
        return 2 if isinstance(e, PlanError) else 4


if __name__ == "__main__":
    sys.exit(main())
