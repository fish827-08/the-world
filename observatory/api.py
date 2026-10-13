"""observatory.api：HTTP API 服务（模块四 · 前端控制面 / Task #2）。

通俗理解
--------
`observatory.broker` 是"实时直播"（WebSocket 推快照给前端渲染）；
本模块是"遥控器 + 目录索引"：前端通过 HTTP 就能

1. **配实验并开跑**（POST /experiments）——装置档位走 `tools.device_presets`
   白名单（🔴 纪律 R5.6：**拒绝任意 rows/cols**，只认 `--device <名>`，
   吃默认值 = 与历史读数不可比）；实验用 `subprocess.Popen` 拉起独立进程，
   Windows 下带 `CREATE_NEW_PROCESS_GROUP`（便于用 `CTRL_BREAK_EVENT` 终止）；
2. **查状态**（GET /experiments、GET /experiments/{id}）——
   pending / running / finished / failed + returncode + 日志尾巴；
3. **要快照**（POST /snapshots）——触发该 run 引擎的
   `SphereEngine.save_snapshot`，**只回元数据**（npz 可能数百 MB，绝不流式返回）；
4. **索引数据包**（GET /datapacks）——扫 `the-world-data/viz/*` 与
   `the-world-data/live/*` 的 meta.json（契约见 `viz/CONTRACT.md`）。

另加：POST /experiments/{run_id}/cancel（Windows 发 `CTRL_BREAK_EVENT`、POSIX 发
`SIGTERM` 给整个子进程组，宽限期后硬 kill）、GET /health、GET /devices。

进程模型（为什么快照要"发请求文件"）
-----------------------------------
实验跑在**子进程**里，本服务进程摸不到它的 `SphereEngine` 对象。故：

* 默认 `live=true`：子进程不是直接跑脚本，而是跑本模块的 **supervise 模式**
  （`python -m observatory.api supervise --run-dir D --script S -- <脚本参数>`）。
  supervisor 在**同进程内**导入并执行脚本的 `main(argv)`，同时钩住
  `SphereEngine.__init__` 捕获"最后构造的引擎"，再起一个轮询线程盯
  `D/snapshot_request.json`；有请求就 `engine.save_snapshot(...)` 并回写
  `D/snapshot_ack.json`。API 侧写请求 → 等 ack（有界等待）→ 回元数据。
  ⚠️ 快照与 tick 推进**非原子**（另一线程正在写数组）⇒ 可能"跨 tick 撕裂"，
  仅供可视化取帧，**不得当作实验判据数据**。
* `live=false`：直接 `python <script> <args>`（无快照能力，POST /snapshots 回 409）。
* 另有**进程内引擎登记表**（`attach_engine`）：测试/内嵌场景可直接挂一台引擎，
  POST /snapshots 走 `mode="inproc"` 直接调 `save_snapshot`。

产物目录约定
------------
`<data_root>/live/<run_id>/`：
    run.json                 本服务写的 run 台账（状态/参数/returncode；重启可复原）
    stdout.log               子进程 stdout+stderr 合流日志
    snapshot_request.json    API → supervisor 的快照请求（原子替换写）
    snapshot_ack.json        supervisor → API 的快照应答（原子替换写）
    snapshots/*.npz          快照产物（可能很大）
    <脚本自身产物>            如 run_bare_world 的 npz/csv/manifest.json（靠 `--out` 注入）

`data_root` = 环境变量 `THE_WORLD_DATA_ROOT`，缺省 `<repo_root>/the-world-data`
（**不进主仓**，见 `.gitignore`；实验数据在独立数据仓）。

运行
----
    python -m observatory.api --host 127.0.0.1 --port 8000
    # 交互式文档：http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# ---- 常量 ------------------------------------------------------------------

API_VERSION = "v1"
SERVICE_NAME = "the-world-observatory-api"

#: 默认实验脚本（相对 repo root）。⚠️ 该脚本在 VIZ worktree 分支尚不存在
#: （它比 main 旧）；不存在时 POST /experiments 会 fail-loud 400 script_not_found。
DEFAULT_SCRIPT = "experiments/run_bare_world.py"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: CORS 放行源（Vite 开发服务器）；可用环境变量追加（逗号分隔）。
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

#: run 状态机（对外契约固定四态；细分原因见 ended_reason）
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"

_SNAPSHOT_REQUEST = "snapshot_request.json"
_SNAPSHOT_ACK = "snapshot_ack.json"
_RUN_LEDGER = "run.json"
_LOG_NAME = "stdout.log"

#: 台账里最多记多少条快照（防 run.json 无限膨胀）
MAX_SNAPSHOT_RECORDS = 200

#: 引擎快照 npz 的判定键（save_snapshot 顶层必写；实验数据 npz 无此键）
_SNAPSHOT_VERSION_KEY = "snapshot_version"

#: GET /snapshots 的默认/上限输出条数
DEFAULT_SNAPSHOT_LIMIT = 200

#: 导出通道白名单（真源 viz/exporter/export_datapack.py:CHANNEL_KEYS）
_EXPORT_CHANNELS = ("resource", "energy", "fruit")

#: POST /snapshots/export 子进程超时（秒）
SNAPSHOT_EXPORT_TIMEOUT_S = 120.0


# ---- 路径解析 ---------------------------------------------------------------


def repo_root() -> Path:
    """仓库根（= observatory/ 的父目录；worktree 里就是 worktree 根）。"""
    return Path(__file__).resolve().parents[1]


def get_data_root() -> Path:
    """数据仓根：环境变量 `THE_WORLD_DATA_ROOT` 优先，缺省 `<repo_root>/the-world-data`。

    每次调用现读环境变量（便于测试 monkeypatch，也便于运行时改指向）。
    """
    env = os.environ.get("THE_WORLD_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root() / "the-world-data").resolve()


def live_root() -> Path:
    return get_data_root() / "live"


def viz_root() -> Path:
    return get_data_root() / "viz"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """原子写 JSON（先写 .tmp 再 os.replace，避免读端读到半截）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ---- 装置白名单（纪律 R5.6） ------------------------------------------------


def allowed_devices() -> list[str]:
    """装置预设白名单（真源 `tools/device_presets.py:DEVICE_PRESETS`）。"""
    from tools.device_presets import preset_names

    return list(preset_names())


def device_fields(name: str) -> dict:
    from tools.device_presets import DEVICE_PRESETS

    return dict(DEVICE_PRESETS[name])


# ---- run 台账 ---------------------------------------------------------------


@dataclass
class RunState:
    """一次实验运行的全部可观测状态（对外 = run 对象）。"""

    run_id: str
    name: str
    device: str
    seed: int
    ticks: int
    arm: Optional[int]
    script: str                 # 相对 repo root 的脚本路径
    args: list[str]             # 传给脚本的完整参数（含注入的 --out）
    command: list[str]          # 实际 Popen 的命令行
    live: bool                  # True = 走 supervise（具备快照能力）
    out_dir: str                # 产物目录（绝对路径）
    log_path: str               # 日志文件（绝对路径）
    status: str = STATUS_PENDING
    pid: Optional[int] = None
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    returncode: Optional[int] = None
    ended_reason: Optional[str] = None   # exit_ok / exit_nonzero / cancelled / spawn_error / unknown_after_restart
    note: Optional[str] = None
    snapshots: list[dict] = field(default_factory=list)

    # -- 非序列化字段（进程句柄/日志句柄），to_dict 时剔除 --
    _proc: Any = field(default=None, repr=False)
    _log_fh: Any = field(default=None, repr=False)
    _cancelled: bool = field(default=False, repr=False)

    def to_dict(self, log_tail: Optional[list[str]] = None) -> dict:
        """导出 JSON 安全视图。

        ⚠️ 不用 `dataclasses.asdict`：它会 **deepcopy** 每个字段，而 `_proc`
        （subprocess.Popen）内含 `_thread.lock` ⇒ `cannot pickle '_thread.lock'`。
        改为白名单浅导出（跳过 `_` 开头的进程内字段）。
        """
        d: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name.startswith("_"):
                continue
            v = getattr(self, f.name)
            d[f.name] = list(v) if isinstance(v, list) else v
        d["log_tail"] = log_tail
        return d


class RunRegistry:
    """线程安全的 run 登记表：内存态 + `<out_dir>/run.json` 落盘（重启可复原）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, RunState] = {}
        self._rehydrated = False

    # -- 生命周期 --

    def rehydrate(self) -> int:
        """扫 `<data_root>/live/*/run.json` 复原历史 run（幂等）。

        与持久化对称：台账 = `to_dict()` 视图（含 `log_tail` 读时视图字段，
        落盘恒 null——尾巴每次响应从 stdout.log 现算），而构造器只收状态
        字段。故按 `dataclasses.fields(RunState)` **白名单过滤**后再构造：
        单一真源 = 构造器签名，未知键（log_tail / 未来版本新字段 / `_` 开头
        进程内字段）一律剥离 ⇒ 双向向前兼容（旧 API 读新台账、新 API 读旧
        台账都安全），重启后历史 run 必然完整出现在 GET /experiments。
        """
        with self._lock:
            if self._rehydrated:
                return len(self._runs)
            self._rehydrated = True
            root = live_root()
            if not root.is_dir():
                return 0
            ctor_keys = {f.name for f in dataclasses.fields(RunState)}
            for ledger in sorted(root.glob(f"*/{_RUN_LEDGER}")):
                try:
                    raw = json.loads(ledger.read_text(encoding="utf-8"))
                    run = RunState(**{k: v for k, v in raw.items() if k in ctor_keys})
                except Exception as exc:  # 坏台账不拖垮服务，但要出声
                    print(f"[api] ⚠️ 跳过坏台账 {ledger}: {exc}", file=sys.stderr, flush=True)
                    continue
                if run.status in (STATUS_PENDING, STATUS_RUNNING):
                    # API 重启 ⇒ 子进程句柄已丢，无法再观测/终止 ⇒ 诚实标失败
                    run.status = STATUS_FAILED
                    run.ended_reason = "unknown_after_restart"
                    run.note = "API 重启前遗留：进程句柄已丢失，状态不可知（判 failed）"
                    run.finished_at = run.finished_at or _now()
                self._runs[run.run_id] = run
            return len(self._runs)

    def reset(self) -> None:
        """清空登记表（**仅测试用**：重置 rehydrate 幂等标志）。"""
        with self._lock:
            for run in self._runs.values():
                self._close_log(run)
            self._runs.clear()
            self._rehydrated = False

    def add(self, run: RunState) -> None:
        with self._lock:
            self._runs[run.run_id] = run
        self.persist(run)

    def persist(self, run: RunState) -> None:
        try:
            _atomic_write_json(Path(run.out_dir) / _RUN_LEDGER, run.to_dict())
        except Exception as exc:
            print(f"[api] ⚠️ 台账落盘失败 {run.run_id}: {exc}", file=sys.stderr, flush=True)

    def get(self, run_id: str) -> Optional[RunState]:
        with self._lock:
            self.rehydrate()
            return self._runs.get(run_id)

    def all(self) -> list[RunState]:
        with self._lock:
            self.rehydrate()
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    # -- 状态推进 --

    def refresh(self, run: RunState) -> RunState:
        """惰性收割：读一次子进程退出码并推进状态机（幂等）。"""
        with self._lock:
            proc = run._proc
            if proc is None or run.status not in (STATUS_PENDING, STATUS_RUNNING):
                return run
            rc = proc.poll()
            if rc is None:
                if run.status == STATUS_PENDING:
                    run.status = STATUS_RUNNING
                return run
            run.returncode = int(rc)
            run.finished_at = _now()
            if run._cancelled:
                run.status = STATUS_FAILED
                run.ended_reason = "cancelled"
            elif rc == 0:
                run.status = STATUS_FINISHED
                run.ended_reason = "exit_ok"
            else:
                run.status = STATUS_FAILED
                run.ended_reason = "exit_nonzero"
            run._proc = None
            self._close_log(run)
        self.persist(run)
        return run

    @staticmethod
    def _close_log(run: RunState) -> None:
        fh, run._log_fh = run._log_fh, None
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


REGISTRY = RunRegistry()

#: 进程内引擎登记表（测试/内嵌用）：run_id -> engine（duck-typed SphereEngine）
_ENGINES: dict[str, Any] = {}
_ENGINES_LOCK = threading.Lock()


def attach_engine(run_id: str, engine: Any) -> None:
    """把一台**本进程内**的引擎挂到 run_id 上（POST /snapshots 走 mode=inproc）。"""
    with _ENGINES_LOCK:
        _ENGINES[run_id] = engine


def detach_engine(run_id: str) -> None:
    with _ENGINES_LOCK:
        _ENGINES.pop(run_id, None)


def _attached_engine(run_id: str) -> Any:
    with _ENGINES_LOCK:
        return _ENGINES.get(run_id)


# ---- 参数装配与校验 ---------------------------------------------------------


def _slug(text: str, maxlen: int = 32) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")
    return slug[:maxlen] or "run"


def make_run_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{_slug(name)}-{uuid.uuid4().hex[:6]}"


def resolve_script(script: str) -> Path:
    """把请求里的 script 解析成绝对路径（fail-loud）。

    规则：只接受 `.py`；相对路径按 repo root 解析；**默认禁止跳出 repo root**
    （`..` 逃逸 / 仓外绝对路径）；文件必须存在。
    逃生口：环境变量 `THE_WORLD_API_ALLOW_EXTERNAL_SCRIPT=1` 时允许仓外绝对路径
    （**仅供测试/本地调试**，生产不开：否则就是一个任意代码执行面）。
    """
    if not isinstance(script, str) or not script.strip():
        raise ApiError(400, "invalid_request", "script 不能为空")
    script = script.strip()
    if not script.endswith(".py"):
        raise ApiError(400, "invalid_request", f"script 必须是 .py 文件：{script!r}")
    p = Path(script)
    root = repo_root()
    allow_external = os.environ.get("THE_WORLD_API_ALLOW_EXTERNAL_SCRIPT", "") == "1"
    if p.is_absolute():
        target = p.resolve()
        inside = root == target or root in target.parents
        if not inside and not allow_external:
            raise ApiError(400, "script_not_found",
                           f"script 越出仓库根（{root}）：{target}"
                           f"（如确需仓外脚本，设 THE_WORLD_API_ALLOW_EXTERNAL_SCRIPT=1）",
                           {"repo_root": str(root)})
    else:
        if ".." in p.parts:
            raise ApiError(400, "invalid_request", f"script 不得含 '..'：{script!r}")
        target = (root / p).resolve()
        if root != target and root not in target.parents:
            raise ApiError(400, "script_not_found",
                           f"script 越出仓库根（{root}）：{target}",
                           {"repo_root": str(root)})
    if not target.is_file():
        raise ApiError(400, "script_not_found",
                       f"实验脚本不存在：{target}（相对 repo root = {script!r}）",
                       {"repo_root": str(root), "tried": str(target)})
    return target


# ---- 实验参数 schema（前端自主配置 → 旗标翻译） ------------------------------
#
# 旗标契约（kebab-case；布尔 = 无值旗标，数值 = --flag value）：
#   --pop N(int≥0，0=纯场) / --sun-budget X(float>0) / --stress-rate X(float≥0) /
#   --abiotic-budget / --area-weighted / --abiotic-stress(布尔) /
#   --patch-mode(seed_dilate|field_threshold) / --patch-coverage X(0<x<1) / --env(布尔) /
#   --diff / --diff-f / --diff-k / --diff-du-dv / --diff-soc(自发分化组)。
# 个别旗标若实验脚本暂未实现，提交时脚本 argparse 会 fail-loud 报错并在 UI 展示——
# 可接受的过渡态（schema 以契约为准，不迁就个别脚本的实现进度）。

_WARN_OVERRIDE = "覆盖默认将产生新配置，读数不与历史可比"
_WARN_DIFF_NOTE = "该组参数由分化模型定义，最终口径以脚本报告为准"

#: 分组元数据（collapsed = 前端默认折叠；diffusion 组口径待脚本报告 ⇒ 默认折叠）
_PARAM_GROUPS: list[dict[str, Any]] = [
    {"key": "basic", "label": "基础", "note": None, "collapsed": False},
    {"key": "c0_world", "label": "C0 世界规则", "note": None, "collapsed": False},
    {"key": "terrain", "label": "斑块与地形", "note": None, "collapsed": False},
    {"key": "diffusion", "label": "自发分化", "note": _WARN_DIFF_NOTE, "collapsed": True},
]


def _p(key: str, label: str, ptype: str, *, flag: Optional[str] = None,
       minv: Optional[float] = None, maxv: Optional[float] = None,
       choices: Optional[list[str]] = None, unit: Optional[str] = None,
       group: str = "basic", warning: Optional[str] = None) -> dict[str, Any]:
    """schema 单项构造器（default 恒为 null：留空 = 不传旗标 = 用脚本默认）。"""
    return {"key": key, "flag": flag or ("--" + key.replace("_", "-")),
            "label": label, "type": ptype, "default": None,
            "min": minv, "max": maxv, "choices": choices, "unit": unit,
            "group": group, "warning": warning}


#: run_c0_energy.py 的参数 schema（顺序即翻译/展示顺序）
_PARAM_SCHEMA_C0: list[dict[str, Any]] = [
    _p("pop", "生物数量", "int", minv=0, maxv=100000, unit="个体",
       group="basic", warning=_WARN_OVERRIDE + "（0 = 纯场无生物）"),
    _p("sun_budget", "光照总预算", "float", minv=0, group="c0_world",
       warning="须 > 0；" + _WARN_OVERRIDE),
    _p("stress_rate", "胁迫强度", "float", minv=0, group="c0_world",
       warning="须 ≥ 0；" + _WARN_OVERRIDE),
    _p("abiotic_budget", "启用非生物能量预算", "bool", group="c0_world",
       warning=_WARN_OVERRIDE),
    _p("area_weighted", "按面积加权统计", "bool", group="c0_world",
       warning=_WARN_OVERRIDE),
    _p("abiotic_stress", "启用非生物胁迫", "bool", group="c0_world",
       warning=_WARN_OVERRIDE),
    _p("patch_mode", "斑块生成方式", "choice", choices=["seed_dilate", "field_threshold"],
       group="terrain", warning=_WARN_OVERRIDE),
    _p("patch_coverage", "斑块覆盖率", "float", minv=0, maxv=1, group="terrain",
       warning="须 0 < x < 1（开区间）；" + _WARN_OVERRIDE),
    _p("env", "启用环境地形", "bool", group="terrain", warning=_WARN_OVERRIDE),
    _p("diff", "启用自发分化", "bool", group="diffusion", warning=_WARN_OVERRIDE),
    _p("diff_f", "分化参数 f", "float", group="diffusion", warning=_WARN_OVERRIDE),
    _p("diff_k", "分化参数 k", "float", group="diffusion", warning=_WARN_OVERRIDE),
    _p("diff_du_dv", "扩散系数 du/dv", "float", group="diffusion",
       warning="归一化 RD 参数；" + _WARN_OVERRIDE),
    _p("diff_soc", "分化社会耦合", "bool", group="diffusion", warning=_WARN_OVERRIDE),
]

#: script 文件名 → 参数 schema；未登记的 script ⇒ 空参数集（端点不报错）
_PARAM_SCHEMA_BY_SCRIPT: dict[str, list[dict[str, Any]]] = {
    "run_c0_energy.py": _PARAM_SCHEMA_C0,
}


def _param_schema_for(script: str) -> list[dict[str, Any]]:
    """按 script 路径取 schema（尾段文件名匹配；未知脚本 ⇒ 空参数集）。"""
    return _PARAM_SCHEMA_BY_SCRIPT.get(Path(script).name, [])


def _bad_param(key: str, msg: str, ptype: str) -> dict[str, str]:
    """422 detail.fields 单项（与 RequestValidationError handler 形状一致）。"""
    return {"field": f"params.{key}", "msg": msg, "type": ptype}


def translate_params(params: dict[str, Any],
                     schema: list[dict[str, Any]]) -> list[str]:
    """params → 旗标列表（按 schema 固定顺序；校验 fail-loud ⇒ 422 invalid_request）。

    翻译规则：bool=true → 追加无值旗标，false/null/缺省 → 不追加；数值 → --flag value；
    choice → --flag <值>。结果追加在既有 args 之后（由调用方保证）。校验失败聚合一次抛出
    （detail.fields 指明全部非法项）。
    """
    by_key = {p["key"]: p for p in schema}
    unknown = sorted(k for k in params if k not in by_key)
    fields: list[dict[str, str]] = [
        _bad_param(k, "未知参数键（不在 schema 白名单内）", "unknown_key") for k in unknown
    ]
    args: list[str] = []
    for p in schema:
        key = p["key"]
        if key in unknown or key not in params:
            continue
        raw = params[key]
        if raw is None:  # null 同“留空”：不传旗标
            continue
        flag, ptype = p["flag"], p["type"]
        if ptype == "bool":
            if not isinstance(raw, bool):  # 先查 bool（bool 是 int 子类）
                fields.append(_bad_param(key, f"须为布尔，得到 {type(raw).__name__}", ptype))
                continue
            if raw:
                args.append(flag)  # 无值旗标；false → 不追加
            continue
        if ptype == "choice":
            if not isinstance(raw, str) or raw not in p["choices"]:
                fields.append(_bad_param(
                    key, f"须为 {'|'.join(p['choices'])}，得到 {raw!r}", ptype))
                continue
            args += [flag, raw]
            continue
        # int / float（bool 已在上面拦截，不会走到这）
        is_num = isinstance(raw, (int, float))
        is_ok = is_num and (ptype == "float" or isinstance(raw, int))
        if not is_ok:
            got = type(raw).__name__
            fields.append(_bad_param(key, f"须为 {ptype}，得到 {got}", ptype))
            continue
        val = float(raw)
        if key == "sun_budget" and val <= 0:
            fields.append(_bad_param(key, "须 > 0", ptype))
            continue
        if key == "patch_coverage" and not 0 < val < 1:
            fields.append(_bad_param(key, "须 0 < x < 1（开区间）", ptype))
            continue
        lo, hi = p.get("min"), p.get("max")
        if lo is not None and val < float(lo):
            fields.append(_bad_param(key, f"须 ≥ {p['min']}", ptype))
            continue
        if hi is not None and val > float(hi):
            fields.append(_bad_param(key, f"须 ≤ {p['max']}", ptype))
            continue
        args += [flag, str(int(raw)) if ptype == "int" else str(val)]
    if fields:
        raise ApiError(422, "invalid_request",
                       f"params 校验失败（{len(fields)} 项）："
                       + "；".join(f"{f['field']} {f['msg']}" for f in fields),
                       {"fields": fields})
    return args


def build_script_args(req: "ExperimentCreate", run_dir: Path) -> list[str]:
    """装配传给实验脚本的参数（装置档位 + 臂 + 种子 + tick + 产物目录）。"""
    args: list[str] = ["--device", req.device, "--seed", str(req.seed),
                       "--ticks", str(req.ticks)]
    if req.arm is not None:
        args += ["--arm", str(req.arm)]
    extra = list(req.extra_args or [])
    # 产物目录：脚本必须接受 --out；调用方已显式给了 --out 就不重复注入
    if not any(a == "--out" or a.startswith("--out=") for a in extra):
        args += ["--out", str(run_dir)]
    args += extra
    # 结构化参数（params）翻译为旗标，追加在既有 args 之后
    args += translate_params(dict(req.params or {}), _param_schema_for(req.script))
    return args


def build_command(script_abs: Path, script_args: list[str], run_dir: Path,
                  live: bool) -> list[str]:
    """实际 Popen 的命令行。live=True ⇒ 经本模块的 supervise 模式包一层。

    live 模式下自动启用实时数据包导出（stride 由环境变量
    `THE_WORLD_LIVE_EXPORT_STRIDE` 控制，默认 20；设 0 禁用）。
    """
    exe = sys.executable or "python"
    if live:
        stride = int(os.environ.get("THE_WORLD_LIVE_EXPORT_STRIDE", "20"))
        cmd = [exe, "-m", "observatory.api", "supervise",
               "--run-dir", str(run_dir), "--script", str(script_abs)]
        if stride > 0:
            cmd += ["--live-export-stride", str(stride)]
        cmd += ["--"] + script_args
        return cmd
    return [exe, str(script_abs)] + script_args


# ---- 子进程拉起 / 终止 ------------------------------------------------------


def child_env() -> dict[str, str]:
    """子进程环境：repo root 进 PYTHONPATH（确保 import 解析到本 worktree 代码）。"""
    env = dict(os.environ)
    root = str(repo_root())
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root + (os.pathsep + pp if pp else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def spawn(run: RunState) -> RunState:
    """Popen 拉起实验；失败 fail-loud（状态置 failed / spawn_error）。"""
    out_dir = Path(run.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(run.log_path)
    # 日志文件句柄一直开着（子进程 stdout+stderr 合流写入）；进程结束时关闭
    fh = log_path.open("wb")
    kwargs: dict[str, Any] = {
        "cwd": str(repo_root()),
        "env": child_env(),
        "stdout": fh,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        # 新进程组 ⇒ 可用 CTRL_BREAK_EVENT 只终止这一棵子树，不波及本服务
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True  # 新会话 ⇒ 可 killpg 整棵子树

    run._log_fh = fh
    try:
        proc = subprocess.Popen(run.command, **kwargs)
    except OSError as exc:
        run.status = STATUS_FAILED
        run.ended_reason = "spawn_error"
        run.note = f"拉起子进程失败：{exc}"
        run.finished_at = _now()
        RunRegistry._close_log(run)
        REGISTRY.persist(run)
        raise ApiError(500, "spawn_failed", f"拉起实验进程失败：{exc}",
                       {"command": run.command}) from exc
    run._proc = proc
    run.pid = int(proc.pid)
    run.status = STATUS_RUNNING
    run.started_at = _now()
    REGISTRY.persist(run)
    return run


def terminate(run: RunState, grace_s: float = 10.0) -> RunState:
    """终止 run：Windows 发 CTRL_BREAK_EVENT（进程组），POSIX 发 SIGTERM（进程组）。"""
    proc = run._proc
    if proc is None:
        REGISTRY.refresh(run)
        if run.status in (STATUS_PENDING, STATUS_RUNNING):
            raise ApiError(409, "not_running",
                           f"run {run.run_id} 无可终止的进程（状态 {run.status}）")
        return run
    run._cancelled = True
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ValueError) as exc:
        # 无控制台/进程已没了 ⇒ 退回硬终止（仍要出声记录）
        print(f"[api] ⚠️ 优雅终止失败，改 terminate()：{exc}", file=sys.stderr, flush=True)
        try:
            proc.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
        proc.wait(timeout=5)
    return REGISTRY.refresh(run)


# ---- 日志尾巴 ---------------------------------------------------------------


def log_tail(path: str | Path, max_bytes: int = 8192, lines: int = 20) -> list[str]:
    """读日志末尾若干行（不整文件加载：日志可能几百 MB）。"""
    p = Path(path)
    if not p.is_file():
        return []
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # 丢掉可能被截断的半行
            raw = f.read()
    except OSError:
        return []
    text = raw.decode("utf-8", errors="replace")
    out = [ln for ln in text.splitlines() if ln.strip()]
    return out[-lines:]


# ---- 快照 -------------------------------------------------------------------


def _snapshot_meta(path: Path, engine: Any = None, ack: dict | None = None) -> dict:
    """组装 POST /snapshots 的响应体（**只有元数据**，不回 npz 内容）。"""
    st = path.stat()
    meta: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(st.st_size),
        "created_at": _now(),
    }
    if ack:
        for k in ("tick", "config_fingerprint", "gene_count", "snapshot_version",
                  "request_id", "engine_tick_at_request"):
            if k in ack:
                meta[k] = ack[k]
    if engine is not None:
        meta.setdefault("tick", int(getattr(engine, "tick", 0)))
        cfg = getattr(engine, "config", None)
        if cfg is not None:
            meta.setdefault("gene_count", int(cfg.genome.gene_count))
            meta.setdefault("config_fingerprint", str(cfg.fingerprint()))
    from simulation.sphere_engine import SphereEngine
    meta.setdefault("snapshot_version", int(SphereEngine.SNAPSHOT_VERSION))
    return meta


def snapshot_inproc(run: RunState, tick: Optional[int]) -> dict:
    """进程内引擎直采（mode=inproc）。"""
    engine = _attached_engine(run.run_id)
    out = Path(run.out_dir) / "snapshots"
    out.mkdir(parents=True, exist_ok=True)
    eff_tick = int(tick) if tick is not None else int(getattr(engine, "tick", 0))
    path = out / f"snapshot_t{eff_tick}.npz"
    try:
        engine.save_snapshot(str(path))
    except Exception as exc:
        raise ApiError(500, "snapshot_failed",
                       f"save_snapshot 失败：{exc}", {"run_id": run.run_id}) from exc
    meta = _snapshot_meta(path, engine=engine)
    meta.update({"run_id": run.run_id, "mode": "inproc", "tick": eff_tick,
                 "request_id": uuid.uuid4().hex[:12]})
    return meta


def snapshot_live(run: RunState, tick: Optional[int], wait_s: float) -> dict:
    """子进程（supervise 模式）快照：写请求文件 → 有界等待 ack。"""
    run_dir = Path(run.out_dir)
    if not run_dir.is_dir():
        raise ApiError(409, "run_dir_missing", f"产物目录不存在：{run_dir}")
    request_id = uuid.uuid4().hex[:12]
    _atomic_write_json(run_dir / _SNAPSHOT_REQUEST, {
        "request_id": request_id,
        "tick": tick,
        "requested_at": _now(),
    })
    ack_path = run_dir / _SNAPSHOT_ACK
    deadline = time.monotonic() + max(0.1, wait_s)
    while time.monotonic() < deadline:
        REGISTRY.refresh(run)
        if ack_path.is_file():
            try:
                ack = json.loads(ack_path.read_text(encoding="utf-8"))
            except Exception:
                ack = None
            if isinstance(ack, dict) and ack.get("request_id") == request_id:
                if ack.get("error"):
                    raise ApiError(500, "snapshot_failed",
                                   f"子进程快照失败：{ack['error']}",
                                   {"run_id": run.run_id, "request_id": request_id})
                path = Path(ack["path"])
                if not path.is_file():
                    raise ApiError(500, "snapshot_failed",
                                   f"ack 指向的快照文件不存在：{path}")
                meta = _snapshot_meta(path, ack=ack)
                meta.update({"run_id": run.run_id, "mode": "live",
                             "request_id": request_id})
                return meta
        if run.status in (STATUS_FINISHED, STATUS_FAILED) and not ack_path.is_file():
            # 进程已结束且没有 ack ⇒ 不会再有人应答了，早点出声
            break
        time.sleep(0.2)
    raise ApiError(409, "snapshot_timeout",
                   f"等待子进程快照应答超时（{wait_s}s）：脚本可能未走 supervise 模式、"
                   f"引擎尚未构造，或已退出",
                   {"run_id": run.run_id, "request_id": request_id,
                    "status": run.status, "log_tail": log_tail(run.log_path, lines=10)})


# ---- 数据包索引 -------------------------------------------------------------


def _read_json(path: Path) -> tuple[Optional[dict], Optional[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"{path.name} 解析失败：{exc}"


def _dir_stats(d: Path) -> dict:
    n_files = size = n_frames = n_entities = 0
    has_series = False
    for root, _dirs, files in os.walk(d):
        rel = Path(root).relative_to(d).as_posix()
        in_frames = rel == "frames" or rel.startswith("frames/")
        in_entities = rel == "entities" or rel.startswith("entities/")
        for name in files:
            fp = Path(root) / name
            try:
                size += fp.stat().st_size
            except OSError:
                pass
            n_files += 1
            if in_frames:
                n_frames += 1
            elif in_entities:
                n_entities += 1
            elif rel == "." and name == "series.csv":
                has_series = True
    return {"n_files": n_files, "size_bytes": size, "n_frame_files": n_frames,
            "n_entity_files": n_entities, "has_series_csv": has_series}


def _index_pack(d: Path, kind: str, include_meta: bool, stats: bool) -> dict:
    """索引一个数据包目录：meta.json 优先，退到 *.manifest.json / run.json。"""
    entry: dict[str, Any] = {
        "id": f"{kind}/{d.name}",
        "kind": kind,
        "name": d.name,
        "path": str(d.resolve()),
        "meta_source": None,
        "meta": None,
        "contract_version": None,
        "world": None,
        "tick": None,
        "stats": None,
        "run_status": None,
        "modified_at": None,
        "error": None,
    }
    try:
        entry["modified_at"] = datetime.fromtimestamp(
            d.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    except OSError:
        pass

    meta: Optional[dict] = None
    cand = d / "meta.json"
    if cand.is_file():
        entry["meta_source"] = "meta.json"
        meta, err = _read_json(cand)
        entry["error"] = err
    else:
        manifests = sorted(d.glob("*.manifest.json"))
        if manifests:
            entry["meta_source"] = manifests[0].name
            meta, err = _read_json(manifests[0])
            entry["error"] = entry["error"] or err
        elif kind == "live" and (d / _RUN_LEDGER).is_file():
            entry["meta_source"] = _RUN_LEDGER
            meta, err = _read_json(d / _RUN_LEDGER)
            entry["error"] = entry["error"] or err

    if isinstance(meta, dict):
        entry["contract_version"] = meta.get("contract_version")
        w = meta.get("world")
        if isinstance(w, dict) and "rows" in w and "cols" in w:
            entry["world"] = {"rows": w.get("rows"), "cols": w.get("cols"),
                              "n_cells": w.get("n_cells")}
        t = meta.get("tick")
        if isinstance(t, dict):
            entry["tick"] = {k: t.get(k) for k in ("start", "end", "stride", "n_frames")}
        entry["meta"] = meta if include_meta else None
        if meta.get("status") in (STATUS_PENDING, STATUS_RUNNING, STATUS_FINISHED, STATUS_FAILED):
            entry["run_status"] = meta.get("status")
    if stats:
        try:
            entry["stats"] = _dir_stats(d)
        except OSError as exc:
            entry["error"] = entry["error"] or f"目录统计失败：{exc}"
    return entry


def index_datapacks(kind: str = "all", include_meta: bool = True,
                    stats: bool = True) -> dict:
    """索引 `the-world-data/{viz,live}/*`（GET /datapacks 的实现）。"""
    root = get_data_root()
    packs: list[dict] = []
    wanted = ("viz", "live") if kind == "all" else (kind,)
    for k in wanted:
        base = viz_root() if k == "viz" else live_root()
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                packs.append(_index_pack(d, k, include_meta, stats))
    packs.sort(key=lambda e: (e["kind"], e["name"]))
    return {
        "data_root": str(root),
        "data_root_exists": root.is_dir(),
        "generated_at": _now(),
        "count": len(packs),
        "packs": packs,
    }


# ---- 快照发现与导出（前端自主加载快照：npz → v0 数据包 → 前端载入） ----------


def _dir_size_bytes(d: Path) -> int:
    """目录总字节数（浅容错；与 _dir_stats 口径一致但不数文件）。"""
    total = 0
    for root_dir, _dirs, files in os.walk(d):
        for name in files:
            try:
                total += (Path(root_dir) / name).stat().st_size
            except OSError:
                pass
    return total


def _probe_npz_snapshot(path: Path) -> Optional[dict]:
    """试探一个 npz 是否引擎快照（SphereEngine.save_snapshot 产物）。

    判定：npz 顶层含 `snapshot_version` 标量键（save_snapshot 必写；
    C0/bare_world 等实验数据 npz 无此键）。npz 是 zip 容器，这里只解压
    标量级成员（snapshot_version/tick），不整包加载，不开 allow_pickle。
    返回 {"tick": int|None}；非快照 / 不可读返回 None。
    """
    import numpy as np  # 延迟导入：与模块其余部分保持同风格（引擎相关即用即装）

    try:
        with np.load(path, allow_pickle=False) as z:
            if _SNAPSHOT_VERSION_KEY not in z.files:
                return None
            tick = None
            if "tick" in z.files:
                try:
                    t = z["tick"]
                    if t.shape == ():
                        tick = int(t)
                except Exception:
                    tick = None
            return {"tick": tick}
    except Exception:
        return None


def _snapshot_ledger_index() -> dict[str, dict]:
    """run 台账 → {rel_path(正斜杠): {run_id, tick}} 索引（补 GET /snapshots 元数据）。"""
    idx: dict[str, dict] = {}
    root = get_data_root()
    for run in REGISTRY.all():
        for s in run.snapshots or []:
            rel = s.get("rel_path") or s.get("path")
            if not rel:
                continue
            p = Path(rel)
            if p.is_absolute():
                try:
                    p = p.relative_to(root)
                except ValueError:
                    continue
            idx.setdefault(p.as_posix(), {"run_id": run.run_id, "tick": s.get("tick")})
    return idx


def list_snapshots(limit: int = DEFAULT_SNAPSHOT_LIMIT) -> dict:
    """扫描 data_root 下的引擎快照 npz（GET /snapshots 的实现）。

    范围：`live/<run_id>/snapshots/*.npz` + data_root 下散落 `*.npz`
    （排除 live/、viz/ 子树）。非快照（无 snapshot_version 键 / 损坏）跳过
    并计入 skipped。tick 优先取 npz 内标量，缺了退 run 台账；run_id 取自
    路径 `live/<run_id>/snapshots/` 段，散落快照再退台账。
    """
    root = get_data_root()
    ledger = _snapshot_ledger_index()
    candidates: list[Path] = []
    live = live_root()
    if live.is_dir():
        for d in sorted(live.iterdir()):
            sd = d / "snapshots"
            if d.is_dir() and sd.is_dir():
                candidates.extend(sorted(sd.glob("*.npz")))
    if root.is_dir():
        for p in sorted(root.rglob("*.npz")):
            parts = p.relative_to(root).parts
            if parts and parts[0] in ("live", "viz"):
                continue  # live/ 已定向扫描；viz/ 是数据包目录
            candidates.append(p)

    items: list[dict] = []
    skipped = 0
    for p in candidates:
        probe = _probe_npz_snapshot(p)
        if probe is None:
            skipped += 1
            continue
        try:
            rel_parts = p.relative_to(root).parts
            rel = "/".join(rel_parts)
        except ValueError:
            rel_parts, rel = (), str(p)
        run_id: Optional[str] = None
        if len(rel_parts) >= 3 and rel_parts[0] == "live" and rel_parts[2] == "snapshots":
            run_id = rel_parts[1]
        tick = probe.get("tick")
        meta = ledger.get(rel)
        if meta:
            run_id = run_id or meta.get("run_id")
            if tick is None:
                tick = meta.get("tick")
        try:
            st = p.stat()
            modified = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
            size = int(st.st_size)
        except OSError:
            modified, size = None, 0
        items.append({"snapshot": rel, "size_bytes": size, "tick": tick,
                      "run_id": run_id, "modified_at": modified})
    items.sort(key=lambda e: e["modified_at"] or "", reverse=True)
    return {"data_root": str(root), "count": min(len(items), limit),
            "skipped": skipped, "snapshots": items[:limit]}


def resolve_snapshot_path(snapshot: str) -> Path:
    """把请求里的快照路径解析成绝对路径（fail-loud，防穿越；仿 resolve_script）。

    规则：只接受 `.npz`；不得含 `..`；绝对路径必须落在 data_root 内；
    文件必须存在。违规一律 400 invalid_snapshot。
    """
    if not isinstance(snapshot, str) or not snapshot.strip():
        raise ApiError(400, "invalid_snapshot", "snapshot 不能为空")
    s = snapshot.strip().replace("\\", "/")
    if not s.lower().endswith(".npz"):
        raise ApiError(400, "invalid_snapshot", f"snapshot 必须是 .npz 文件：{snapshot!r}")
    p = Path(s)
    root = get_data_root()
    if ".." in p.parts:
        raise ApiError(400, "invalid_snapshot", f"snapshot 不得含 '..'：{snapshot!r}")
    if p.is_absolute():
        target = p.resolve()
        if not (root == target or root in target.parents):
            raise ApiError(400, "invalid_snapshot",
                           f"snapshot 越出 data_root（{root}）：{target}",
                           {"data_root": str(root)})
    else:
        target = (root / p).resolve()
        if root != target and root not in target.parents:
            raise ApiError(400, "invalid_snapshot",
                           f"snapshot 越出 data_root（{root}）：{target}",
                           {"data_root": str(root)})
    if not target.is_file():
        raise ApiError(400, "invalid_snapshot", f"快照文件不存在：{target}",
                       {"data_root": str(root), "tried": str(target)})
    return target


def run_snapshot_export(snapshot_abs: Path, out_dir: Path, ticks: int,
                        stride: int, channels: list[str]) -> dict:
    """subprocess 调 viz/exporter/export_datapack.py 导出 v0 数据包（超时保护）。

    stdout/err 落 `viz/.export_logs/<out_name>.log`（"." 开头 ⇒ 不会被
    GET /datapacks 当包索引）。成功返回产物摘要 {rows, cols, n_frames,
    size_bytes, log}；失败走 ApiError(snapshot_export_failed)。
    """
    exporter = repo_root() / "viz" / "exporter" / "export_datapack.py"
    if not exporter.is_file():
        raise ApiError(500, "snapshot_export_failed", f"导出器不存在：{exporter}",
                       {"exporter": str(exporter)})
    cmd = [sys.executable, str(exporter),
           "--snapshot", str(snapshot_abs), "--out", str(out_dir),
           "--ticks", str(ticks), "--stride", str(stride),
           "--channels", ",".join(channels), "--series", "auto"]
    log_dir = get_data_root() / "viz" / ".export_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (out_dir.name + ".log")

    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=str(repo_root()), env=child_env(),
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=SNAPSHOT_EXPORT_TIMEOUT_S)
        rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = None
        out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        err = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
    header = (f"# cmd: {subprocess.list2cmdline(cmd)}\n"
              f"# timeout_s: {SNAPSHOT_EXPORT_TIMEOUT_S}\n"
              f"# rc: {rc}" + ("（超时强杀）" if timed_out else "") + "\n")
    try:
        log_path.write_text(header + "--- stdout ---\n" + out + "\n--- stderr ---\n" + err,
                            encoding="utf-8")
    except OSError:
        pass

    if timed_out:
        raise ApiError(500, "snapshot_export_failed",
                       f"导出超时（>{SNAPSHOT_EXPORT_TIMEOUT_S:.0f}s）已中止："
                       "可减小 ticks 或增大 stride 后重试",
                       {"timeout_s": SNAPSHOT_EXPORT_TIMEOUT_S, "log": str(log_path)})
    if rc != 0:
        tail = [ln for ln in (err + "\n" + out).splitlines() if ln.strip()]
        raise ApiError(500, "snapshot_export_failed",
                       f"导出器失败（rc={rc}）：{tail[-1] if tail else '(无输出)'}",
                       {"rc": rc, "log": str(log_path)})

    meta, meta_err = _read_json(out_dir / "meta.json")
    if not isinstance(meta, dict) or not isinstance(meta.get("world"), dict) \
            or "rows" not in meta.get("world", {}) or "cols" not in meta.get("world", {}):
        raise ApiError(500, "snapshot_export_failed",
                       f"导出器 rc=0 但产物 meta.json 缺失/损坏：{meta_err or out_dir}",
                       {"log": str(log_path)})
    w = meta["world"]
    n_frames = int((meta.get("tick") or {}).get("n_frames") or 1)
    return {"rows": int(w["rows"]), "cols": int(w["cols"]), "n_frames": n_frames,
            "size_bytes": _dir_size_bytes(out_dir), "log": str(log_path)}


# ---- 错误信封（所有错误统一形状，前端只认一套） ------------------------------


class ApiError(HTTPException):
    """带机器可读 error_code 的 HTTP 异常（统一信封见下）。"""

    def __init__(self, status_code: int, error_code: str, message: str,
                 detail: Any = None) -> None:
        super().__init__(status_code=status_code,
                         detail={"error_code": error_code, "message": message,
                                 "detail": detail})
        self.error_code = error_code
        self.message = message


def _envelope(error_code: str, message: str, detail: Any = None) -> dict:
    return {"error_code": error_code, "message": message, "detail": detail}


# ---- 请求体模型 -------------------------------------------------------------


class ExperimentCreate(BaseModel):
    """POST /experiments 请求体（未知字段一律拒绝 = fail-loud）。"""

    model_config = ConfigDict(extra="forbid")

    device: str = Field(description="装置预设档名，必须在 tools.device_presets.preset_names() 白名单内")
    seed: int = Field(ge=0, description="随机种子（整数 ≥0）")
    ticks: int = Field(ge=1, description="模拟 tick 数（整数 ≥1）")
    arm: Optional[int] = Field(default=None, ge=1, description="实验臂编号（run_bare_world：1-4）；缺省不传 --arm")
    name: Optional[str] = Field(default=None, max_length=64, description="人类可读名（进 run_id）；缺省 = <device>-arm<arm>")
    script: str = Field(default=DEFAULT_SCRIPT, description="实验脚本（相对 repo root 的 .py，或仓内绝对路径）")
    extra_args: list[str] = Field(default_factory=list, description="原样追加给脚本的额外参数（如 ['--sample','20']）")
    params: dict[str, Any] = Field(default_factory=dict, description=(
        "结构化参数（键须在 GET /param-schema 白名单内，值按类型/范围校验；"
        "bool=true → 无值旗标，false → 不追加；数值 → --flag value；追加在既有 args 之后）"))
    live: bool = Field(default=True, description="True = 经 supervise 包装（具备 POST /snapshots 能力）")


class SnapshotCreate(BaseModel):
    """POST /snapshots 请求体。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, description="目标 run（POST /experiments 返回的 run_id）")
    tick: Optional[int] = Field(default=None, ge=0, description="标注用 tick；缺省 = 引擎当前 tick")
    wait_s: float = Field(default=20.0, gt=0, le=300, description="等待子进程应答的上限秒数")


class SnapshotExportCreate(BaseModel):
    """POST /snapshots/export 请求体（快照 → v0 数据包导出）。"""

    model_config = ConfigDict(extra="forbid")

    snapshot: str = Field(min_length=1, description="快照 npz（data_root 相对路径，须 .npz 结尾）")
    name: Optional[str] = Field(default=None, max_length=64,
                                description="数据包名（slug 化，≤64）；缺省 = UTC 时间戳")
    ticks: int = Field(default=0, ge=0, le=1_000_000,
                       description="续跑 tick 数（0=仅快照当前帧 ⇒ 静态单帧查看）")
    stride: int = Field(default=20, ge=1, le=10_000, description="采样间隔 tick/帧")
    channels: list[str] = Field(default_factory=lambda: ["resource", "energy"],
                                description="导出通道（resource/energy/fruit）")


# ---- App 工厂 ---------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动：复原历史 run 台账 + 起后台收割线程；关闭：停收割线程。"""
    REGISTRY.rehydrate()
    _start_reaper()
    try:
        yield
    finally:
        _stop_reaper()


def create_app() -> FastAPI:
    app = FastAPI(
        title="the-world observatory API",
        version=API_VERSION,
        description="数字生命模拟的实验控制面：配实验/查状态/触发快照/索引数据包。",
        lifespan=_lifespan,
    )

    origins = list(DEFAULT_CORS_ORIGINS)
    extra = os.environ.get("THE_WORLD_API_CORS_ORIGINS", "")
    origins += [o.strip() for o in extra.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- 统一错误信封 --

    @app.exception_handler(ApiError)
    async def _api_error(request, exc: ApiError):  # noqa: ANN001
        d = exc.detail if isinstance(exc.detail, dict) else {}
        return JSONResponse(status_code=exc.status_code,
                            content=_envelope(d.get("error_code", exc.error_code),
                                              d.get("message", str(exc.message)),
                                              d.get("detail")))

    @app.exception_handler(HTTPException)
    async def _http_error(request, exc: HTTPException):  # noqa: ANN001
        d = exc.detail
        if isinstance(d, dict) and "error_code" in d:
            return JSONResponse(status_code=exc.status_code,
                                content=_envelope(d["error_code"], d.get("message", ""),
                                                  d.get("detail")),
                                headers=getattr(exc, "headers", None))
        return JSONResponse(status_code=exc.status_code,
                            content=_envelope("http_error", str(d), None),
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc: RequestValidationError):  # noqa: ANN001
        fields = []
        for e in exc.errors():
            loc = [str(x) for x in e.get("loc", ()) if x not in ("body",)]
            fields.append({"field": ".".join(loc), "msg": e.get("msg", ""),
                           "type": e.get("type", "")})
        return JSONResponse(status_code=422, content=_envelope(
            "invalid_request", "请求体校验失败（缺字段 / 类型错 / 未知字段 / 取值越界）",
            {"fields": fields}))

    @app.exception_handler(Exception)
    async def _unhandled(request, exc: Exception):  # noqa: ANN001
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        return JSONResponse(status_code=500, content=_envelope(
            "internal_error", f"服务端异常：{type(exc).__name__}: {exc}", None))

    # -- 端点 --

    @app.get("/health", tags=["meta"], summary="服务自检")
    def health() -> dict:
        script_abs = repo_root() / DEFAULT_SCRIPT
        return {
            "service": SERVICE_NAME,
            "api_version": API_VERSION,
            "status": "ok",
            "server_time": _now(),
            "python": sys.version.split()[0],
            "repo_root": str(repo_root()),
            "data_root": str(get_data_root()),
            "data_root_exists": get_data_root().is_dir(),
            "default_script": DEFAULT_SCRIPT,
            "default_script_exists": script_abs.is_file(),
            "devices": allowed_devices(),
            "registered_runs": len(REGISTRY.all()),
            "endpoints": [
                "GET /health", "GET /devices",
                "POST /experiments", "GET /experiments", "GET /experiments/{run_id}",
                "POST /experiments/{run_id}/cancel", "POST /snapshots", "GET /datapacks",
            ],
        }

    @app.get("/devices", tags=["meta"], summary="装置预设白名单（R5.6 真源）")
    def devices() -> dict:
        names = allowed_devices()
        return {
            "source": "tools/device_presets.py",
            "count": len(names),
            "devices": [{"name": n, "fields": device_fields(n)} for n in names],
            "note": "只接受白名单内的 device 名；不接受任意 rows/cols（纪律 R5.6）",
        }

    @app.get("/param-schema", tags=["meta"],
             summary="实验脚本的结构化参数 schema（前端高级参数区渲染依据）")
    def param_schema(
        script: str = Query(default=DEFAULT_SCRIPT,
                            description="实验脚本（相对 repo root 的 .py；未知 script 返回空组不报错）"),
    ) -> dict:
        params = _param_schema_for(script)
        target = Path(script)
        script_abs = target if target.is_absolute() else repo_root() / target
        return {
            "script": script,
            "script_exists": script_abs.is_file(),
            "groups": [dict(g) for g in _PARAM_GROUPS] if params else [],
            "params": [dict(p) for p in params],
        }

    @app.post("/experiments", status_code=201, tags=["experiments"],
              summary="配置并启动一次实验")
    def create_experiment(req: ExperimentCreate = Body(...)) -> dict:
        names = allowed_devices()
        if req.device not in names:
            raise ApiError(400, "unknown_device",
                           f"未知 device {req.device!r}（防呆：绝不静默回落）",
                           {"allowed_devices": names, "source": "tools/device_presets.py"})
        script_abs = resolve_script(req.script)
        try:
            script_rel = str(script_abs.relative_to(repo_root())).replace(os.sep, "/")
        except ValueError:
            script_rel = str(script_abs)
        name = (req.name or f"{req.device}" + (f"-arm{req.arm}" if req.arm else "")).strip()
        run_id = make_run_id(name)
        run_dir = live_root() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        args = build_script_args(req, run_dir)
        cmd = build_command(script_abs, args, run_dir, req.live)
        run = RunState(
            run_id=run_id, name=name, device=req.device, seed=req.seed,
            ticks=req.ticks, arm=req.arm, script=script_rel, args=args,
            command=cmd, live=bool(req.live), out_dir=str(run_dir.resolve()),
            log_path=str((run_dir / _LOG_NAME).resolve()),
        )
        REGISTRY.add(run)          # 先落台账（pending），spawn 失败也有据可查
        spawn(run)
        return run.to_dict(log_tail=log_tail(run.log_path, lines=20))

    @app.get("/experiments", tags=["experiments"], summary="列出全部 run")
    def list_experiments(
        status: Optional[str] = Query(default=None,
                                      description="过滤：pending|running|finished|failed"),
        include_log: bool = Query(default=False, description="是否附日志尾巴"),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> dict:
        if status is not None and status not in (
                STATUS_PENDING, STATUS_RUNNING, STATUS_FINISHED, STATUS_FAILED):
            raise ApiError(400, "invalid_request",
                           f"未知 status 过滤值 {status!r}",
                           {"allowed": [STATUS_PENDING, STATUS_RUNNING,
                                        STATUS_FINISHED, STATUS_FAILED]})
        all_runs = [REGISTRY.refresh(r) for r in REGISTRY.all()]
        by_status: dict[str, int] = {}
        for r in all_runs:
            by_status[r.status] = by_status.get(r.status, 0) + 1
        runs = [r for r in all_runs if status is None or r.status == status]
        total = len(runs)
        page = runs[:limit]
        return {
            "count": len(page),
            "total": total,
            "by_status": by_status,
            "server_time": _now(),
            "runs": [r.to_dict(log_tail=log_tail(r.log_path) if include_log else None)
                     for r in page],
        }

    @app.get("/experiments/{run_id}", tags=["experiments"], summary="单个 run 详情")
    def get_experiment(run_id: str,
                       tail_lines: int = Query(default=40, ge=0, le=500)) -> dict:
        run = REGISTRY.get(run_id)
        if run is None:
            raise ApiError(404, "run_not_found", f"未知 run_id：{run_id!r}",
                           {"known_runs": [r.run_id for r in REGISTRY.all()][:50]})
        REGISTRY.refresh(run)
        return run.to_dict(log_tail=log_tail(run.log_path, lines=tail_lines)
                           if tail_lines else [])

    @app.post("/experiments/{run_id}/cancel", tags=["experiments"],
              summary="终止 run（CTRL_BREAK_EVENT / SIGTERM）")
    def cancel_experiment(run_id: str,
                          grace_s: float = Query(default=10.0, gt=0, le=120)) -> dict:
        run = REGISTRY.get(run_id)
        if run is None:
            raise ApiError(404, "run_not_found", f"未知 run_id：{run_id!r}")
        REGISTRY.refresh(run)
        if run.status not in (STATUS_PENDING, STATUS_RUNNING):
            raise ApiError(409, "not_running",
                           f"run {run_id} 已结束（状态 {run.status}），无需终止")
        terminate(run, grace_s=grace_s)
        return run.to_dict(log_tail=log_tail(run.log_path, lines=20))

    @app.post("/snapshots", tags=["snapshots"],
              summary="触发引擎快照（只回元数据，不回 npz）")
    def create_snapshot(req: SnapshotCreate = Body(...)) -> dict:
        run = REGISTRY.get(req.run_id)
        if run is None:
            raise ApiError(404, "run_not_found", f"未知 run_id：{req.run_id!r}",
                           {"known_runs": [r.run_id for r in REGISTRY.all()][:50]})
        REGISTRY.refresh(run)
        if _attached_engine(run.run_id) is not None:
            meta = snapshot_inproc(run, req.tick)
        elif run.live and run.status in (STATUS_RUNNING, STATUS_PENDING):
            meta = snapshot_live(run, req.tick, req.wait_s)
        elif run.live:
            raise ApiError(409, "snapshot_unsupported",
                           f"run {run.run_id} 已结束（{run.status}），引擎不在内存中，"
                           f"无法触发新快照；请读已有产物（GET /datapacks）",
                           {"status": run.status, "out_dir": run.out_dir})
        else:
            raise ApiError(409, "snapshot_unsupported",
                           f"run {run.run_id} 以 live=false 启动，无快照通道；"
                           f"请用 live=true 重新启动",
                           {"live": False})
        rel_root = get_data_root()
        try:
            meta["rel_path"] = str(Path(meta["path"]).relative_to(rel_root)).replace(os.sep, "/")
        except ValueError:
            meta["rel_path"] = meta["path"]
        run.snapshots.append({k: meta.get(k) for k in
                              ("path", "rel_path", "tick", "size_bytes", "mode",
                               "created_at", "config_fingerprint", "gene_count",
                               "snapshot_version")})
        run.snapshots = run.snapshots[-MAX_SNAPSHOT_RECORDS:]
        REGISTRY.persist(run)
        return meta

    @app.get("/datapacks", tags=["datapacks"],
             summary="索引 the-world-data/{viz,live}/* 的 meta.json")
    def datapacks(
        kind: str = Query(default="all", description="all|viz|live"),
        include_meta: bool = Query(default=True, description="是否内联 meta.json 全文"),
        stats: bool = Query(default=True, description="是否统计文件数/字节数"),
    ) -> dict:
        if kind not in ("all", "viz", "live"):
            raise ApiError(400, "invalid_request", f"未知 kind {kind!r}",
                           {"allowed": ["all", "viz", "live"]})
        return index_datapacks(kind=kind, include_meta=include_meta, stats=stats)

    @app.get("/snapshots", tags=["snapshots"],
             summary="索引 data_root 下可加载的引擎快照 npz")
    def snapshots_index(
        limit: int = Query(default=DEFAULT_SNAPSHOT_LIMIT, ge=1, le=1000,
                           description="输出条数上限（按修改时间倒序截断）"),
    ) -> dict:
        return list_snapshots(limit=limit)

    @app.post("/snapshots/export", tags=["snapshots"],
              summary="快照 npz → v0 数据包（subprocess 导出；前端随后可载入）")
    def export_snapshot(req: SnapshotExportCreate = Body(...)) -> dict:
        snapshot_abs = resolve_snapshot_path(req.snapshot)
        bad = [c for c in req.channels if c not in _EXPORT_CHANNELS]
        if not req.channels or bad:
            raise ApiError(400, "invalid_request",
                           f"未知导出通道 {bad}；可选 {sorted(_EXPORT_CHANNELS)}",
                           {"allowed": sorted(_EXPORT_CHANNELS)})
        slug = _slug(req.name, 64) if req.name else ""
        base = f"snap_{slug or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        out_dir = viz_root() / base
        if out_dir.exists():  # 同名不覆盖：追加短随机后缀（fail-safe，不静默覆盖产物）
            out_dir = viz_root() / f"{base}-{uuid.uuid4().hex[:6]}"
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        result = run_snapshot_export(snapshot_abs, out_dir, req.ticks, req.stride,
                                     req.channels)
        return {
            "pack_id": f"viz/{out_dir.name}",
            "name": out_dir.name,
            "path": str(out_dir.resolve()),
            "n_frames": result["n_frames"],
            "rows": result["rows"],
            "cols": result["cols"],
            "size_bytes": result["size_bytes"],
        }

    return app


# ---- 后台收割线程（状态自更新，不依赖前端轮询） ------------------------------

_REAPER: Optional[threading.Thread] = None
_REAPER_STOP = threading.Event()


def _reaper_loop() -> None:
    while not _REAPER_STOP.is_set():
        try:
            for run in REGISTRY.all():
                if run.status in (STATUS_PENDING, STATUS_RUNNING):
                    REGISTRY.refresh(run)
        except Exception as exc:  # 收割线程绝不能带崩服务
            print(f"[api] ⚠️ reaper 异常：{exc}", file=sys.stderr, flush=True)
        _REAPER_STOP.wait(1.0)


def _start_reaper() -> None:
    global _REAPER
    if _REAPER is not None and _REAPER.is_alive():
        return
    _REAPER_STOP.clear()
    _REAPER = threading.Thread(target=_reaper_loop, name="api-run-reaper", daemon=True)
    _REAPER.start()


def _stop_reaper() -> None:
    _REAPER_STOP.set()


#: 模块级 app（`uvicorn observatory.api:app` 与 TestClient 都直接用它）
app = create_app()


# ---- supervise 子命令（子进程内的快照代理） ---------------------------------


def _install_engine_capture() -> dict:
    """钩住 `SphereEngine.__init__`，捕获"最后构造的引擎"（幂等）。"""
    from simulation.sphere_engine import SphereEngine

    state: dict[str, Any] = {"engine": None}
    orig = SphereEngine.__init__
    if getattr(orig, "_tw_capture", False):
        return orig._tw_state  # type: ignore[attr-defined]

    def patched(self, *a, **kw):  # noqa: ANN001
        orig(self, *a, **kw)
        state["engine"] = self

    patched._tw_capture = True    # type: ignore[attr-defined]
    patched._tw_state = state     # type: ignore[attr-defined]
    SphereEngine.__init__ = patched  # type: ignore[method-assign]
    return state


def _snapshot_poller(run_dir: Path, state: dict, stop: threading.Event,
                     poll_s: float = 0.5) -> None:
    """盯 `snapshot_request.json`；有新请求就 save_snapshot 并回写 ack。"""
    req_path = run_dir / _SNAPSHOT_REQUEST
    ack_path = run_dir / _SNAPSHOT_ACK
    last_id: Optional[str] = None
    while not stop.is_set():
        try:
            if req_path.is_file():
                req = json.loads(req_path.read_text(encoding="utf-8"))
                rid = req.get("request_id")
                if rid and rid != last_id:
                    last_id = rid
                    _handle_snapshot_request(run_dir, state, req, rid, ack_path)
        except Exception as exc:
            print(f"[supervise] ⚠️ 快照轮询异常：{exc}", file=sys.stderr, flush=True)
        stop.wait(poll_s)


def _live_export_poller(run_dir: Path, state: dict, stop: threading.Event,
                        stride: int, interval_s: float = 2.0) -> None:
    """实时数据包导出线程：等引擎就绪后，按 stride 增量写 v0 数据包帧。

    产物直接写入 run_dir（meta.json + frames/ + entities/），
    前端可通过 GET /datapacks 发现并轮询 meta.json 增量拉取新帧。
    """
    # 等待引擎被捕获
    while not stop.is_set():
        if state.get("engine") is not None:
            break
        stop.wait(0.5)
    if stop.is_set():
        return

    engine = state["engine"]
    try:
        # live_export 与 export_datapack 同目录，非包结构，用 importlib 导入
        import importlib.util as _ilu
        _live_mod_path = repo_root() / "viz" / "exporter" / "live_export.py"
        _spec = _ilu.spec_from_file_location("live_export", _live_mod_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        LiveExporter = _mod.LiveExporter
        exporter = LiveExporter(
            engine=engine,
            out_dir=run_dir,
            stride=stride,
            channels=["resource", "energy"],
        )
        print(f"[supervise] live-export 已启动（stride={stride}）", flush=True)
    except Exception as exc:
        print(f"[supervise] ⚠️ live-export 初始化失败：{exc}", file=sys.stderr, flush=True)
        return

    while not stop.is_set():
        try:
            exporter.maybe_capture()
        except Exception as exc:
            print(f"[supervise] ⚠️ live-export 采集异常：{exc}", file=sys.stderr, flush=True)
            break
        stop.wait(interval_s)

    # 停止前写最终帧 + meta
    try:
        exporter.capture_now()
        exporter.finalize()
        print(f"[supervise] live-export 完成（共 {exporter.frame_count} 帧）", flush=True)
    except Exception as exc:
        print(f"[supervise] ⚠️ live-export finalize 异常：{exc}", file=sys.stderr, flush=True)


def _handle_snapshot_request(run_dir: Path, state: dict, req: dict, rid: str,
                             ack_path: Path) -> None:
    engine = state.get("engine")
    if engine is None:
        _atomic_write_json(ack_path, {
            "request_id": rid, "error": "引擎尚未构造（脚本还没走到 SphereEngine(...)）",
            "created_at": _now()})
        return
    out = run_dir / "snapshots"
    out.mkdir(parents=True, exist_ok=True)
    try:
        tick_req = req.get("tick")
        cur = int(getattr(engine, "tick", 0))
        tick = int(tick_req) if tick_req is not None else cur
        path = out / f"snapshot_t{tick}.npz"
        engine.save_snapshot(str(path))
        cfg = getattr(engine, "config", None)
        from simulation.sphere_engine import SphereEngine
        _atomic_write_json(ack_path, {
            "request_id": rid, "error": None,
            "path": str(path.resolve()),
            "tick": tick,
            "engine_tick_at_request": cur,
            "size_bytes": int(path.stat().st_size),
            "gene_count": int(cfg.genome.gene_count) if cfg is not None else None,
            "config_fingerprint": str(cfg.fingerprint()) if cfg is not None else None,
            "snapshot_version": int(SphereEngine.SNAPSHOT_VERSION),
            "created_at": _now(),
            "warning": "快照与 tick 推进非原子，可能跨 tick 撕裂（仅供可视化）",
        })
        print(f"[supervise] 快照已写 {path.name} (tick={tick})", flush=True)
    except Exception as exc:
        _atomic_write_json(ack_path, {
            "request_id": rid, "error": f"{type(exc).__name__}: {exc}",
            "created_at": _now()})


def supervise_main(argv: list[str]) -> int:
    """`python -m observatory.api supervise --run-dir D --script S -- <脚本参数...>`。

    在本进程内导入并执行实验脚本的 `main(argv)`，同时提供快照代理。
    注：`--` 之后的参数**全部归脚本**（本解析器不碰），故先手工切分。
    """
    cut = argv.index("--") if "--" in argv else len(argv)
    head, script_args = list(argv[:cut]), list(argv[cut + 1:])

    ap = argparse.ArgumentParser(prog="observatory.api supervise",
                                 description="实验子进程监督器（快照代理）")
    ap.add_argument("--run-dir", required=True, help="产物目录（含 snapshot_request.json）")
    ap.add_argument("--script", required=True, help="实验脚本绝对路径（.py）")
    ap.add_argument("--poll", type=float, default=0.5, help="快照请求轮询间隔（秒）")
    ap.add_argument("--live-export-stride", type=int, default=0,
                    help="实时数据包导出 stride（0=禁用；>0 时每 N tick 写一帧 v0 数据包）")
    a = ap.parse_args(head)

    run_dir = Path(a.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    script_abs = Path(a.script).resolve()
    if not script_abs.is_file():
        print(f"🔴 [supervise] 脚本不存在：{script_abs}", file=sys.stderr, flush=True)
        return 2

    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    state = _install_engine_capture()
    stop = threading.Event()
    th = threading.Thread(target=_snapshot_poller, args=(run_dir, state, stop, a.poll),
                          name="snapshot-agent", daemon=True)
    th.start()
    # 实时数据包导出线程（--live-export-stride > 0 时启动）
    th_export: Optional[threading.Thread] = None
    if a.live_export_stride > 0:
        th_export = threading.Thread(
            target=_live_export_poller,
            args=(run_dir, state, stop, a.live_export_stride),
            name="live-export-agent", daemon=True)
        th_export.start()
    print(f"[supervise] run_dir={run_dir} script={script_abs} args={script_args}"
          f" live_export_stride={a.live_export_stride}",
          flush=True)

    rc = 0
    try:
        mod_name = f"_tw_script_{script_abs.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, script_abs)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法从 {script_abs} 载入模块")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        # 脚本内 resolve_device 读 sys.argv[1:] 判"显式覆盖" ⇒ 必须伪装成命令行调用
        old_argv = sys.argv
        sys.argv = [str(script_abs)] + script_args
        try:
            spec.loader.exec_module(mod)
            if not hasattr(mod, "main"):
                raise RuntimeError(f"{script_abs.name} 未定义 main()，无法在 supervise 模式下运行")
            ret = mod.main(script_args)
            rc = int(ret) if isinstance(ret, int) else 0
        finally:
            sys.argv = old_argv
    except SystemExit as exc:  # 脚本自己 raise SystemExit(code)
        rc = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        rc = 1
    finally:
        stop.set()
        th.join(timeout=2.0)
        if th_export is not None:
            th_export.join(timeout=5.0)
    print(f"[supervise] 脚本结束 rc={rc} engine_captured={state.get('engine') is not None}",
          flush=True)
    return rc


# ---- 服务入口 ---------------------------------------------------------------


def _serve(host: str, port: int) -> int:
    import uvicorn

    print(f"[api] {SERVICE_NAME} {API_VERSION} → http://{host}:{port}"
          f"（docs: /docs）", flush=True)
    print(f"[api] repo_root={repo_root()}", flush=True)
    print(f"[api] data_root={get_data_root()}（存在={get_data_root().is_dir()}）", flush=True)
    print(f"[api] devices={allowed_devices()} default_script={DEFAULT_SCRIPT}"
          f"（存在={(repo_root() / DEFAULT_SCRIPT).is_file()}）", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # supervise 子命令：`--` 之后的参数归脚本，由 supervise_main 自己切
    if raw and raw[0] == "supervise":
        return supervise_main(raw[1:])

    p = argparse.ArgumentParser(
        prog="python -m observatory.api",
        description="the-world HTTP API 服务（实验控制面 / Task #2）")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"监听地址（默认 {DEFAULT_HOST}）")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"监听端口（默认 {DEFAULT_PORT}）")
    p.add_argument("--data-root", default=None,
                   help="覆盖数据仓根（等价于环境变量 THE_WORLD_DATA_ROOT）")
    a = p.parse_args(raw)
    if a.data_root:
        os.environ["THE_WORLD_DATA_ROOT"] = a.data_root
    return _serve(a.host, a.port)


if __name__ == "__main__":
    raise SystemExit(main())
