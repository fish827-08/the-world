"""`s0_kernel.harness` —— 双臂 S0 的**最小可跑世界**与 `run_arm` 出货面（S0-kernel-b 第二半）。

为什么存在
----------
预注册 §二 共同面写死了三条硬条件：同一 harness、同小网格档＋同 `N_ind` 预算（≤500）、
同 seed 与同 RNG 消费顺序。⇒ **双臂只能活在这个小容器里**（草案 §一-3：拿现役引擎做双臂
必须冻结全部装置开关，等于重造）。⚠️ 因此"用主引擎默认档跑 A 臂"是**错的**（装置不同源 ⇒
A/B 差克里混进装置差），我 01:3x 帖上那句已自纠。

四律落点
--------
- **不触主引擎**：本模块只 import `world.sphere_world`（只读用网格拓扑）与 numpy，
  **不 import `simulation.sphere_engine`**（守卫测试钉住）⇒ 现役引擎逐位等价门零影响。
- **同流形制**：两臂各自 `default_rng(seed)`（初始状态相同），每 tick 的抽取**顺序固定**
  （① 初始位置 ② 移动 ③ 变异正态 ④ 逐个出生点的变异抽签），⇒ 同 seed 同初世界、差分只落在被测通道上。
- **无密度依赖**（by design）：`W` 是 S0 唯一的**选择通道**，资源不耗尽、不排队；规模由
  `max_ind` 兜住并记 `cap_hits`（触顶不静默淘汰既存个体）。⇒ 读数只反映"g 轴上的适应度差"，
  不掺承载力动力学。这条是**装置声明**，不是省略。
- **fail-loud 三门**（砚 ack 条件①）：spec 缺字段／越界／灭绝早停／非干净退出 ⇒ 一律 raise，
  不许"少跑几个 tick"或"参数缺省凑跑"。
- **出货面**（砚 ack 条件②＋§三-2）：逐 tick 个体 g 行 CSV ＋ manifest 增量
  `arm_impl_commit` ＋ 统计码 hash（`arms_sha256`/`harness_sha256`）＋ 全部装置参数回显。

🔴 装置参数（`Demography`）＝**待随标定批锁**：预注册只钉了网格/规模/tick/seed 轴，
出生阈、变异 σ、死亡率、代谢、`intake_scale` 这些**人口学旋钮没有权威值** ⇒ 本模块给一组
显式单值并在 docstring 与 manifest 里逐条标 `[待锁]`；改任一值 = 换装置 = 读数禁与旧批比。
判据（谷深 D／峰位／出不出）**不在本模块**（R225/R394④）。
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from world.sphere_world import SphereWorld

from .arms import Landscape, arm_specs, nu_local, w_a, w_b

REQUIRED_SPEC_KEYS = ("run_id", "arm", "seed", "ticks", "sample", "alpha_W", "demography")


@dataclass
class Demography:
    """S0 最小世界的人口学旋钮（全部 **待锁**；改值即换装置）。"""

    n_ind0: int = 500              # [待锁] 初始个体数（§二 上限取满）
    rows: int = 120                # [待锁] 网格档：120×240 = 28,800 格（草案 §五 不可裁决论证用的同一尺度）
    cols: int = 240
    e0: float = 60.0               # [待锁] 初始能量（照引擎 organisms.initial_energy 口径取值，非借用引擎行为）
    meta: float = 0.5              # [待锁] 每 tick 维持代谢
    intake_scale: float = 1.0      # [待锁] W→能量/ tick 的换算（W 是 E-034 归一化速率）
    repro_at: float = 90.0         # [待锁] 出生阈（≥1.5×e0 ⇒ 一次出生两半各 45）
    repro_cost: float = 0.5        # [待锁] 出生时母体保留比例
    mut_rate: float = 0.05         # [待锁] 每代变异概率
    mut_sigma: float = 0.05        # [待锁] 变异幅度（g 轴加性高斯，clip 到 [0,1]）
    lifespan: int = 400            # [待锁] 最大年龄（tick）
    max_ind: int = 2000            # [待锁] 硬上限（防爆炸；触顶记 `cap_hits`，不静默淘汰）

    def __post_init__(self) -> None:
        if self.n_ind0 < 1 or self.rows < 3 or self.cols < 3:
            raise ValueError(f"规模非法：n_ind0={self.n_ind0} rows={self.rows} cols={self.cols}")
        if self.repro_at <= self.e0:
            raise ValueError(f"repro_at 必须 > e0（否则第 0 tick 就爆炸），实得 {self.repro_at}")
        if not 0.0 < self.repro_cost < 1.0:
            raise ValueError(f"repro_cost ∈ (0,1)，实得 {self.repro_cost}")
        if self.meta < 0.0 or self.intake_scale <= 0.0 or self.mut_sigma < 0.0:
            raise ValueError("meta ≥ 0、intake_scale > 0、mut_sigma ≥ 0")
        if self.lifespan < 1 or self.max_ind < self.n_ind0:
            raise ValueError("lifespan ≥ 1 且 max_ind ≥ n_ind0")


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class S0World:
    """双臂共用的最小世界：格上个体、按 W 进食、分裂繁殖、年龄/饥饿死亡。"""

    def __init__(self, seed: int, arm: str, alpha_W: float | None, dem: Demography,
                 land: Landscape | None = None) -> None:
        if arm not in ("A", "B"):
            raise ValueError(f"arm 只能是 'A' 或 'B'，实得 {arm!r}")
        if arm == "A" and alpha_W is not None:
            raise ValueError(f"A 臂不消费 alpha_W（须记 null，不许记 0.0），实得 {alpha_W!r}")
        if arm == "B" and alpha_W is None:
            raise ValueError("B 臂必须显式给 alpha_W（运行批 1.0／Z2 对照批 0.0）")
        self.arm = arm
        self.alpha_W = alpha_W
        self.dem, self.land = dem, (land or Landscape())
        self.world = SphereWorld(rows=dem.rows, cols=dem.cols)
        self.n_cells = self.world.n_cells
        self.rng = np.random.default_rng(int(seed))          # 每臂一条流；同 seed ⇒ 同初状态
        self.seed = int(seed)
        self.tick = 0
        self.cap_hits = 0
        self._next_id = 0
        self.cells: list[int] = []
        self.g: list[float] = []
        self.e: list[float] = []
        self.age: list[int] = []
        self.alive: list[bool] = []
        self._spawn(dem.n_ind0)

    # ---- 初始群体 ---------------------------------------------------------------
    def _spawn(self, n: int) -> None:
        dem = self.dem
        cells = self.rng.integers(0, self.n_cells, size=n, dtype=np.int64)   # 抽取①（位置）
        for c in cells:
            self.cells.append(int(c))
            self.g.append(0.5)                    # [待锁] 初表型：g 轴中点（无先验偏置）
            self.e.append(dem.e0)
            self.age.append(0)
            self.alive.append(True)
            self._next_id += 1

    # ---- 每 tick 的固定抽取顺序（②移动 ③变异 ④出生抽签）--------------------------
    def step(self) -> None:
        dem = self.dem
        occ: dict[int, list[int]] = {}
        for i, c in enumerate(self.cells):
            if self.alive[i]:
                occ.setdefault(c, []).append(i)
        nb = {c: np.asarray(self.world.neighbors(c), dtype=np.int64) for c in occ}

        r_move = self.rng.random(len(self.cells))                     # 抽取②
        r_mut = self.rng.normal(0.0, 1.0, size=len(self.g))           # 抽取③

        # 1) 进食：A 臂 W(g)，B 臂 W(g,ν)——ν 用**近邻同类核密度**（镜 §一定义）
        for i in range(len(self.g)):
            if not self.alive[i]:
                continue
            neigh_g: list[float] = []
            for c in nb.get(self.cells[i], ()):
                for j in occ.get(int(c), ()):
                    if j != i:
                        neigh_g.append(self.g[j])
            if self.arm == "A":
                intake = w_a(self.g[i], pred=None, land=self.land)
            else:
                nu = nu_local(self.g[i], neigh_g, sigma_c=self.land.sigma_c)
                intake = w_b(self.g[i], nu, self.alpha_W, land=self.land)
            self.e[i] += intake * dem.intake_scale - dem.meta

        # 2) 移动（每存活个体一次；空格/同格都算消费过一次抽取 ⇒ 两臂顺序一致）
        for i in range(len(self.cells)):
            if not self.alive[i]:
                continue
            options = nb.get(self.cells[i])
            if options is None or len(options) == 0:
                continue
            pick = options[int(r_move[i] * len(options)) % len(options)]
            self.cells[i] = int(pick)

        # 3) 死亡：饥饿 / 超龄
        for i in range(len(self.g)):
            if not self.alive[i]:
                continue
            self.age[i] += 1
            if self.e[i] <= 0.0 or self.age[i] > dem.lifespan:
                self.alive[i] = False

        # 4) 出生（遍历快照长度 ⇒ 本 tick 新个体不立即再分裂）
        n_now = len(self.g)
        for i in range(n_now):
            if not self.alive[i] or self.e[i] < dem.repro_at:
                continue
            child_g = self.g[i]
            if self.rng.random() < dem.mut_rate:        # 抽取④：逐个出生点，两臂顺序一致
                child_g = min(max(child_g + r_mut[i] * dem.mut_sigma, 0.0), 1.0)
            self.e[i] *= dem.repro_cost
            if len(self.g) >= dem.max_ind:
                self.cap_hits += 1                      # 触顶：记账、不静默淘汰既存个体
                continue
            self.cells.append(self.cells[i])
            self.g.append(float(child_g))
            self.e.append(self.e[i])
            self.age.append(0)
            self.alive.append(True)
            self._next_id += 1
        self.tick += 1

    # ---- 出货行（逐 tick 个体 g 行，砚 §三-2）--------------------------------------
    def rows(self) -> list[tuple[int, int, int, float, float, int]]:
        out = []
        for i in range(len(self.g)):
            if self.alive[i]:
                out.append((self.tick, self._id_of(i), self.cells[i],
                            round(float(self.g[i]), 9), round(float(self.e[i]), 6),
                            int(self.age[i])))
        return out

    def _id_of(self, i: int) -> int:
        return i                                  # 槽位即稳定 id（只追加、不压缩 ⇒ 跨窗可 join）

    @property
    def pop(self) -> int:
        return sum(1 for a in self.alive if a)

    def digest(self) -> tuple[int, int, float]:
        """(Σ槽位, 人口, Σ能量 6 位)——只作**同流复现自检**，不作任何判据。"""
        idx = sum(i for i in range(len(self.g)) if self.alive[i])
        return (idx, self.pop, round(float(np.sum([e for e, a in zip(self.e, self.alive) if a])), 6))


def run_arm(spec: dict, out_dir: Path | str) -> Path:
    """跑一个 S0 run 并落 CSV＋manifest；砚 ack 三条件里的"接线目标"。

    fail-loud：`spec` 缺任一 `REQUIRED_SPEC_KEYS` 或值越界 ⇒ `KeyError`/`ValueError`；
    灭绝早停 ⇒ `RuntimeError`（少跑 tick 的批不许进判读）；两臂可比性由 `arm_specs` 侧硬校验。
    """
    missing = [k for k in REQUIRED_SPEC_KEYS if k not in spec]
    if missing:
        raise KeyError(f"run_arm spec 缺字段 {missing}（REQUIRED={list(REQUIRED_SPEC_KEYS)}）")
    arm = str(spec["arm"])
    if arm == "A" and spec["alpha_W"] is not None:
        raise ValueError(
            f"A 臂 spec 的 alpha_W 必须是 null（H2 §一.3：`0.0` 是 Z2 零对照的专用语义，"
            f"A 臂是「该项不存在」）；实得 {spec['alpha_W']!r}"
        )
    if arm == "B" and spec["alpha_W"] is None:
        raise ValueError("B 臂 spec 必须显式给 alpha_W（运行批 1.0／Z2 对照批 0.0），实得 null")
    dem = spec["demography"]
    if isinstance(dem, dict):
        dem = Demography(**dem)
    if not isinstance(dem, Demography):
        raise TypeError(f"demography 必须是 Demography 或 dict，实得 {type(dem).__name__}")
    ticks, sample = int(spec["ticks"]), int(spec["sample"])
    if ticks < 1 or sample < 1:
        raise ValueError(f"ticks/sample 必须 ≥1，实得 {ticks}/{sample}")

    a_spec, b_spec = arm_specs(int(spec["seed"]),
                               alpha_W=(float(spec["alpha_W"]) if arm == "B" else 0.0))
    chosen = a_spec if spec["arm"] == "A" else b_spec
    world = S0World(int(spec["seed"]), str(spec["arm"]), chosen["alpha_W"], dem,
                    land=chosen["landscape"])
    if spec["arm"] == "A" and world.alpha_W is not None:
        raise RuntimeError("A 臂拿到的 alpha_W 非 null ⇒ 双臂隔离破了")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{spec['run_id']}.csv"
    data_rows: list[tuple] = []
    for t in range(1, ticks + 1):
        world.step()
        if world.pop <= 0:
            raise RuntimeError(
                f"灭绝早停于 tick {t}/{ticks}（run={spec['run_id']}）⇒ 本 run 不可进判读，"
                f"要么调装置档要么如实登记，不许拿短跑当同档读数")
        if t % sample == 0 or t == ticks:
            data_rows.extend(world.rows())

    if world.tick != ticks:
        raise RuntimeError(f"非干净退出：world.tick={world.tick} ≠ ticks={ticks}")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "ind_id", "cell", "g", "energy", "age"])
        w.writerows(data_rows)

    here = Path(__file__).resolve().parent
    manifest = {
        "run_id": str(spec["run_id"]), "arm": str(spec["arm"]), "seed": world.seed,
        "alpha_W": world.alpha_W, "f_form": chosen["f_form"], "ticks": ticks,
        "sample": sample, "rows_written": len(data_rows), "pop_end": world.pop,
        "cap_hits": world.cap_hits, "digest": list(world.digest()),
        "n_cells": world.n_cells, "demography": asdict(dem),
        "landscape": asdict(world.land),
        "arm_impl_commit": spec.get("arm_impl_commit", "unknown"),
        "arms_sha256": _sha256_of(here / "arms.py"),
        "harness_sha256": _sha256_of(here / "harness.py"),
        "csv": csv_path.name,
    }
    (out_dir / f"{spec['run_id']}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path
