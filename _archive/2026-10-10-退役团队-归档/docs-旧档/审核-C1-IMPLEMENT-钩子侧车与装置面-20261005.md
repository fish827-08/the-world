# 审核：C1-IMPLEMENT（捕食信息价值批 · D-1 钩子 + D-2 侧车 + F2 装置面）

- 审核人：镜 `[代码审核]`｜日期：2026-10-05 21:5x–22:0x
- 被审件：`gitee/task/C1-IMPLEMENT` = `f1aae58`（引擎钩子）+ `392d5ef`（a4 侧车/装置/测试）+ `fd032bf`（设计稿并稿 v2）；作者 **轻舟**
- 基线 / merge-base：`3e758f6`
- 卡面：`C1-IMPLEMENT` done @`3c826ed`（结果栏 delta 清单 = 本审的聚焦面）
- 派单原文（防漂移）：PI 20:3x 班「催办：C1-IMPLEMENT 已 done（3c826ed），按 SOP 进你拍=代码快审……聚焦卡结果栏 delta 清单（F1=丙 8run／F2 device／F5 三元组／F6／F7／F8／F9／T15／T-F2）；33 例靶向测试+T1 逐字节等价证据在 `_rerun_logs/t1_equiv/`。R394① 回避矩阵已核……审过落板帖，PI 下拍复审+R225 会签。」
- 纪律：🔴 **只审不改**（AGENT.md §二）；每条阻塞附可复现步骤 + 修法建议，代码与结论归作者/PI。
- 独立性（R394④）：等价/开档/对账/崩溃四类取证全部用我自己的命令现跑（TEMP 产物，未入库、未复用她的留痕）；变异在自建树 `.worktrees/jing_c1_mut` 内做，跑完 `git checkout --` 还原（现干净）。

---

## 一、结论

🟡 **有条件通过 —— 但起跑门前必处置 C-A**。核心工程承诺（默认关逐字节等价、不进 SimConfig、不触指纹、F5 三元组、四处 kills 对账）我独立复现成立；**三条 🔴 都是"守没守住"类**（不是算法错）：scipy 收尾崩、第二处钩子无对账守、T13 名义锁硬编码实为同式自证。

| 层 | 判定 |
|---|---|
| D-1 引擎钩子（`__slots__`/默认 None/两处点位/不进配置） | 🟢 成立（§二 E1–E4） |
| D-2 侧车 + F7 累计口径 + F6 fail-loud + F2 装置面 + F8/F9 | 🟢 主体成立（§二 E5–E10） |
| 33 例靶向测试 + 她的 T1 等价证据 | 🟢 复跑 33 passed；证据 byte-cmp 复现（§二 E11/E12） |
| **C-A** `write_run_summary` 的 scipy 分支零覆盖 ⇒ 正式档收尾必崩 | 🔴 **起跑门必处置** |
| **C-B** 第二处钩子（致伤致死分支）删除后 33 例全绿 | 🔴 一行对账断言可销 |
| **C-C** `test_t13_no_hardcoded_120` 同式自证 + 只在 cols=120 跑 ⇒ 对硬编码不敏感 | 🔴 判据正确性风险（正式档 960 列） |
| C-D F6 续跑禁令无测试；C-E bg 两参未复刻 `__post_init__` 断言 | 🟡 补守 |
| C-F `if ... and args.mode == "on": pass` 空壳守；C-G `np.where` 除零告警噪声；C-H T15 门本体在测试文件内、生产不可达；C-I 证据目录被 gitignore | ⚪/🟡 |

---

## 二、独立实证表（编号供板帖引用）

| # | 待证项 | 我的独立做法 | 结果 |
|---|---|---|---|
| E1 | **默认关逐字节等价（加长 60×）** | base `3e758f6` 与 head `fd032bf` 各跑 `--mode off --seed 7 --ticks 300 --log-interval 50`（不带 `--rd-instruments`），`cmp` 主表 | ✅ **逐字节相同**（3016 B × 2）；她只跑到 5 tick，我跑到 300 tick（died=86） |
| E2 | **manifest 层也不多出键** | 同 E1 两侧 `summary.json` 展平比键集与值 | ✅ 独有键 = 空；值差（除 `started`/`finished`/`git_commit`/`code_*` provenance 外）= **空**；两侧均**无任何 `rd_` 前缀键** ⇒ F8 关档语义成立 |
| E3 | 引擎面纪律 | 读 `sphere_engine.py:672/855` + `tests/.../TestSlotsAndConfig` | ✅ `__slots__` 登记 + 构造期 `= None` + 不进 `SimConfig` + 指纹不变（T14 有守） |
| E4 | **钩子点位完整性（相对项目自身 kill 定义）** | grep 全引擎 `_duel["kills"]` 与 `_ec_prey_kill_n` 的自增点 | ✅ 自增点恰为 `:4673-4674` 与 `:4706-4707`，两处**都**紧跟 append ⇒ 无第三漏点；但见 C-B（该等价性只在我的人眼核对层，测试没锁） |
| E5 | F5 三元组语义 | 读两处 append + `:672` 注释 | ✅ `(稳定 id, 死亡格, tick)`，并写明"槽位同 tick 后段压缩失效 ⇒ 不可作跨窗 join 键" |
| E6 | **开档端到端 + 四处 kills 对账** | head 跑 `--smell-channels risk --rd-instruments --rd-sample-every 150 --ticks 300` | ✅ rc=0；侧车一 `win_kills_total` = 52+3，`kills_cum_total` 末行 = 55；侧车二 `rd_total_kills` = 55；manifest `switches.rd_n_kill_log_total` = 55 ⇒ **四处一致** |
| E7 | F6 fail-loud（钩子丢失） | 读窗末 + 收尾两处 `if log is None: raise SystemExit(_RD_HOOK_LOST_MSG)`（`:387`/`:1976`）+ 文案 | ✅ 两处齐，文案点名"快照续跑 ⇒ `cls(config)` 重建 ⇒ 静默回 None"并给 `--fresh` 出路；明确禁 `or []`（防"后半程零记录"伪装成"本窗无杀"） |
| E8 | Rust 路径硬拒 | 核 `_sim_core` 赋值条件（`sphere_engine.py:740-762`：仅 `use_sim_core=True` 分支赋值）；`install_hook` 用 `getattr(e,"_sim_core",None) is not None` | ✅ 判据成立（Python 路径该槽未赋值 ⇒ getattr 缺省 None ⇒ 不误拒）；且 a4 `build()` 恒 `use_sim_core=False` ⇒ 结构上双保险 |
| E9 | F2 装置面（R5.6） | 读 `:981+`/`:1342+` 决议块 | ✅ 复用 `add_device_arg`/`resolve_device` 真源（非另写预设）；未传 `--device` ⇒ 完全不介入；rows/cols 赋值后**显式复刻**两条 `__post_init__` 断言（F1 同型教训）；`--pop > --max-count` parse 期 `ap.error` 早拒 |
| E10 | 前置 fail-loud 三件 | 实跑 | ✅ `--rd-instruments` 而 `--smell-channels` 不含 risk ⇒ `rc=2` 点名前置；分块几何 <1 / 窗宽 <1 ⇒ 早拒；F9 help 勘误（`signal` 非法通道 ⇒ 改 `food,prey,risk,kin`）我核 `_KNOWN_CHANNELS` 口径一致 |
| E11 | 33 例靶向测试 | `py -m pytest tests/test_c1_rd_instruments.py -q` 两侧各跑 | ✅ **33 passed**（4.0–4.5s），与卡面一致 |
| E12 | 她的 T1 证据 | 我直接 `cmp` 她磁盘上的 `old.csv`/`new.csv` + 展平比 `summary.json` | ✅ CSV **逐字节相同**；值差仅 8 条 provenance 键（`code_subtrees`×4 / `code_tree_sha256` / `git_commit` / `started` / `finished`），无一条读数/生态列 |
| E13 | 与 M1 判据切点口径的关系 | 对照 PI 裁②（`d7930b8`/`dcb6753`：正式批切点 = index 式 `k=int(0.25n)`） | ⚪ C1 侧车是**固定窗宽**（`--rd-sample-every 250`，§八 锁），不涉分位切点 ⇒ 与裁②不冲突（免得会签时多虑一笔） |

---

## 三、变异复核（基线 33 passed / 4.0s）

| 变异 | 内容 | 结果 |
|---|---|---|
| M1 | 删**第二处**（致伤致死分支）钩子 guard ⇒ `if False:` | 🔴 **33 passed（0 红）** ⇒ C-B |
| M2 | 删窗末 `log.clear()`（不排空） | ✅ 1 红（`test_f7_kills_cum_total`）——排空有守 |
| M3 | 删 F6「rd 批禁续跑」`if resumed: raise` | 🔴 **0 红** ⇒ C-D |
| M4 | 分块映射 `col = flat_idx % self._cols` ⇒ `% 120` 硬编码 | 🔴 **0 红** ⇒ C-C |
| M5 | `kills_cum_total` 去掉累计项（F7 退化） | ✅ 1 红（`test_f7_kills_cum_total`） |
| M6 | rd 收尾块无条件执行（关档也写 ⇒ F8 翻转） | ✅ 2 红（`test_f8_rd_off_no_rd_keys`、`test_f2_device_readback_and_override`） |
| M7 | 删 `--pop > --max-count` 防呆 | ✅ 1 红（`test_f2_pop_over_max_fail_loud`） |

未销项 = M1/M3/M4（三条 0 红）。临时树还原后 `git status --porcelain` 空。

---

## 四、发现（现象 → 复现 → 影响 → 建议修法）

### 🔴 C-A｜正式档口径下**跑完之后**才崩：唯一 scipy 入口在收尾，且 33 例进不到该分支

**现象**：`write_run_summary` 的块排序稳定性分支（`experiments/a4_verify_capacity.py:478` `if len(self._run_rows) >= 3:`）里 `:479 from scipy.stats import spearmanr` 是**全仓 a4 唯一的 scipy 引用**；本机 `import scipy` ⇒ `ModuleNotFoundError`；仓库无 `requirements.txt`/`pyproject.toml` ⇒ 依赖不声明。33 例最多只喂 **2 窗**（`test_f7_kills_cum_total` 注 2 窗 ⇒ `<3` ⇒ 分支不进），所以"33 passed"**不能**证明这条能跑。
**复现**（我实测，约 1 分钟）：
```bash
py experiments/a4_verify_capacity.py --mode off --seed 7 --ticks 300 --log-interval 50 \
   --smell-channels risk --rd-instruments --rd-sample-every 50 --out <tmp>/on.csv
# ⇒ rc=1，跑完 300 tick 后在 :1981 → :479 抛 ModuleNotFoundError: No module named 'scipy'
# ⇒ 现场：主表 CSV 已写、侧车一已 flush，但侧车二 / manifest / provenance **全丢**（无 on.summary.json）
```
**影响**：正式批 `sample_every=250` / `ticks≈2000` ⇒ **8 窗 ≥3 ⇒ 必走该分支**。后果不是"少一个诊断列"，而是**整批跑完在最贵时点崩**：manifest（R225 起跑令对账与 provenance 的载体）与侧车二不落 ⇒ 该 run 不可对账、不可进判读，机时全烧。云机是否装 scipy 我无从核（不猜）。
**建议修法**（三选一或并用，我倾向 ①+②）：① R225 起跑门 Pre-Flight 加一条"跑批机 `import scipy` 可读"并写进起跑令记录；② 把该依赖检查提到 `main()` 起始 **fail-fast**（开档即拒，别等 2000 tick 之后）；③ 把 stability 改纯 numpy（秩 + `np.corrcoef`），彻底去掉未声明依赖。另：依赖声明缺失属**存量治理项**（`observatory/statistics.py` 等同源风险），建议单开一张卡，不挂本卡。
**可判别性补一句**：改后请补一例 **≥3 窗**的 `write_run_summary` 测试（现覆盖缺口就是这条能藏到起跑后才爆的原因）。

### 🔴 C-B｜第二处钩子（致伤致死分支）没有任何"条数对账"守

**现象**：`test_t11b_real_run_three_tuple` 真跑 100 tick，断言的是"有条目 + 元组是 3 元 + tick ∈ [1,100]"——**不断言条数等于引擎自身的击杀计数**。于是删掉第二处 guard（M1）后 33 例全绿。
**复现**：把 `sphere_engine.py:4708` 的 `if self._rd_pred_kill_log is not None:` 改 `if False:` ⇒ `py -m pytest tests/test_c1_rd_instruments.py -q` ⇒ **33 passed**。
**影响**：致伤致死（wound 路径，corpse/wound 系开着时的真实死因之一）若在某次重构中漏装，侧车会**系统性少计**，而 J-A 分块 kills 序列要的正是这个量；读数量级看着正常（少的是子集），属"静默少计"家族 —— 与队列② GAP-A"一列静默零值过所有取值检查"同族。
**建议修法**（一行，顺带把点位齐性锁死）：在 `test_t11b_real_run_three_tuple` 加
```python
assert len(log) == int(e._ec_prey_kill_n)   # 侧车条目数 ≡ 引擎自身捕食击杀计数
```
（seed 221 实测 52 杀，两处点位都在该计数内 ⇒ M1 必红。）

### 🔴 C-C｜`test_t13_no_hardcoded_120` 是**同式自证**，且只在 cols=120 档跑 ⇒ 对硬编码不敏感

**现象**：该测试用 `rd._cols`/`rd._rows` **重算同一个公式**再和 `_block_map` 比 —— 实现里若把 `self._cols` 换成常量 120，`_engine_mini()` 的 `cols` 恰是 120 ⇒ 期望侧也是 120 ⇒ 恒等通过。docstring 写着"mini cols=120 与 480 世界 cols=960 必须同口径"，但**没有任何一例在非 120 列档跑**。
**复现**：`col = flat_idx % self._cols` ⇒ `col = flat_idx % 120`（M4）⇒ **33 passed**。
**影响**：正式档是 **480×960**（R5.6）。分块映射决定每条死亡记到 16 块中的哪一块；若真人在正式口径下写成 120（等距圆柱下 960/120=8 列一错 ⇒ 经度带整体错位），`kills_b`/`dens_b` 的块归属会整片错、而测试永远不会响 ⇒ J-A 的"分块差异"判据直接失真。这是本卡**判据正确性**上最贵的一条盲区。
**建议修法**：加一例**非 120 因子**的几何（如 `rows=7, cols=11`）+ **手算常量期望表**（明确写出每个 block 的行列边界，不是拿 `rd._cols` 再算同一式）断言 `_block_map`；并补一例 480×960 的**纯映射**断言（不必跑引擎，`_build_block_map` 只依赖 rows/cols/n_cells，成本近零）。

### 🟡 C-D｜F6「rd 批禁续跑」分支无测试

M3 删掉 `if resumed: raise SystemExit(...)` ⇒ 33 例全绿（`test_f6_hook_lost_fail_loud` 覆盖的是**钩子丢失**那条，不是**续跑拒绝**）。影响 = 该互斥承诺只靠读码；一旦有人"顺手支持续跑"，三重失配（钩子静默 None / `win_idx` 从 0 重编 / 侧车一 `"w"` 覆写）会回到"前半有数后半空"。修法：一例即可（构造 `resumed=True` + `rd_instruments=True` ⇒ `pytest.raises(SystemExit)`），或 subprocess 真跑一次 `--resume`。

### 🟡 C-E｜F2 的 bg 两参赋值**未复刻** `__post_init__` 断言（同一改稿内标准不齐）

同一个 F2 块对 `rows/cols` 明确写了"赋值绕过 `__post_init__` ⇒ 显式复刻两条断言（F1 同型教训）"，但 `bg_low_prod_frac` / `bg_low_cap_mult` 两行赋值后**没有**复刻 `config.py:152-153`（`0≤frac≤1` / `cap_mult≥0`）与 `:155-158`（`frac>0 ∧ cap=0 ⇒ 配置静默无效（C9）` 联立断言）。⇒ `--bg-low-prod-frac 1.5` 会被静默接受。正式批取 0.4/0.05 在范围内 ⇒ **不阻塞起跑**；但 C9 是项目踩过的坑，联立那条最有价值的正是"看起来配了、实际没生效"。修法：两行赋值后补那三条断言（与 rows/cols 同一段风格，零争议）。

### ⚪ C-F｜空壳守：`if args.rd_instruments and args.mode == "on": pass`

分支体只有 `pass`，注释说"此处仅做防御性提示"但没有任何提示/打印。真正的 Rust 拦截在 `install_hook` 与 main（E8 已核实有效）⇒ **不是漏洞**，但读 diff 时形似一道门。建议删除，或落成实际动作。

### ⚪ C-G｜`np.where` 除零告警噪声

`dens = np.where(occ_block > 0, kills / (occ_block * self.sample_every), np.nan)` 两分支都要求值 ⇒ occ=0 的块每窗刷 `RuntimeWarning: divide by zero / invalid value`。正式批 8 窗 × 16 块 × 多 run ⇒ stderr 会被噪声淹没（掩盖真告警）。建议 `with np.errstate(divide="ignore", invalid="ignore"):` 或掩码赋值。数值结果本身正确（NaN 已按 `""` 落表）。

### 🟡 C-H｜T15 操纵门**本体在测试文件里** ⇒ 生产调用路径不可达

`arms_manip_check()` 定义在 `tests/test_c1_rd_instruments.py:364`，docstring 自己写"调用点归各批预注册写明（可逐字提升）"。两个问题：① 判读件/起跑令要调它就得 `import tests.…`（把 pytest 收集目录当库）；② "逐字提升"= 复制 ⇒ 复制体与本体漂移无守。而 F1=丙 下本批单臂（门"批内 N/A"）、**Q-C 双臂接线批起每臂对必调** ⇒ 到那一拍这条会直接挡在判读前。修法：把本体移到 `tools/arms_manip_check.py`（或 `experiments/`），测试改 import 该件 —— 一行移动 + 一改 import。
**另注（起跑形制提示，非缺陷）**：丙案下两臂逐字节同是**设计事实**（risk 通道只写不读）⇒ 若起跑令把 C1 发成双臂，T15 会按设计炸。请 PI/砚 在起跑令里写明"C1 本批单臂 ⇒ T15 不适用"的判据行，避免批中途自毁。

### ⚪ C-I｜卡面证据路径在 gitignore 内

`_rerun_logs/t1_equiv/` 落在 `.gitignore:54 _rerun_logs/*` ⇒ 只在作者本机存在（我在本机磁盘上找到并独立 `cmp` 复现了，见 E12），但**换一台机（含云机）就不可复核**，卡结果栏把它当验收路径引用。建议 `git add -f` 落那两个 942 B CSV + 一行命令行，或按 AGENT.md §十一 进 `the-world-data` 仓。

---

## 五、回避与独立性声明

1. C1 定稿由轻舟按 R394④ 自出，我非设计稿作者；PI 20:3x 班已裁定快审无冲突。我出过该稿**预审** `3f4426c`（F1–F11 意见）⇒ 为免"用自己的预审当正确性依据"，本审判据一律取 **定稿 v2 文本 + PI 裁定 `697e171`/`dcb6753` + 我的实测**，未引用预审稿措辞。
2. 若 PI 认为 E4（钩子点位齐性）这类"人眼核对"需要第二双眼睛，可指澜舟/砚交叉复看 `sphere_engine` 两处 append；我不自代。
3. 未跑任何批（R117）：我的 300 tick 微缩跑只为等价/对账/崩溃取证，`--ticks 300` @60×120，未触 `--device s2` 正式档，未产生任何判据读数。
4. 不确定项：C-A 的处置取决于**跑批机是否有 scipy**，我本机无、云机未核 ⇒ 请板桥/砚 实测一行 `import scipy` 再定 ①/②/③；我不据"应该装了"下结论。

---

## 六、请 PI 处置（R384 三要素之"结论 + 证据路径 + 阻塞"）

| 项 | 请求动作 |
|---|---|
| C-A | 🔴 起跑门前必处置：确认跑批机 scipy（Pre-Flight 一行 + 写进起跑令）**或**改启动期 fail-fast **或** stability 改纯 numpy；并补一例 ≥3 窗的 `write_run_summary` 测试。改后我复跑上面那条 300t/50 窗命令即可销项 |
| C-B | 🔴 一行对账断言（`len(log) == _ec_prey_kill_n`）；改后我复跑 M1 应变红 |
| C-C | 🔴 非 120 列档 + 手算常量表的 block-map 例；改后我复跑 M4 应变红 |
| C-D / C-E | 🟡 随上述补丁同批落最省（各 1–3 行）；C-E 的联立断言建议优先（C9 教训） |
| C-F / C-G / C-I | ⚪ 建议随本卡销项补丁一并处理（成本极低），不单开拍 |
| C-H | 🟡 是否现在就把 `arms_manip_check` 提升到 `tools/`？还是留到 Q-C 接线批（我倾向现在提升，一行移动，省得那批再改判读件） |
| 起跑形制 | 请在起跑令写明"C1 本批单臂 ⇒ T15 批内 N/A"的判据行（避免双臂误发时批中途自毁） |
| 本件 | 交付 = 本审头（`task/C1-REVIEW`，候合，合并候 PI）；任务卡 `C1-IMPLEMENT` 审核结果登记请板桥/PI 代执 —— 我不写 `_share/任务卡.md`（R371 共享主树竞写） |
| 证据 | 我的取证产物在 `C:/Users/圣羽/AppData/Local/Temp/jing_c1_eq/`（`eq_base/eq_head` 等价对、`on.*` 崩溃现场、`ok.*` 四处对账），TEMP 未入库；变异在 `.worktrees/jing_c1_mut`（已还原干净）。若需入库留痕，我可下拍 `git add -f` 关键四件（≈20 KB） |

**群报说明**：本 waker `0252d9a72439` 无已启用 IM channel（`channel list` = No channels found）⇒ 群内 @PI 不可达，板帖即 R384/R387 的上报载体。
