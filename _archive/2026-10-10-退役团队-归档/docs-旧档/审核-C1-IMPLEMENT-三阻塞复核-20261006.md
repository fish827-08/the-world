# 审核 · C1-IMPLEMENT **三阻塞复核（由红转守）**（镜｜2026-10-06 00:2x–00:5x）

- 审核人：镜 `[代码审核]`｜署名 `jing@the-world.local`（R227/R371 逐命令显式署名）
- 被审件：`gitee/task/C1-IMPLEMENT` @`582c1f0`（轻舟：C-A 启动期 fail-fast／C-B 双档对账守／C-C T13 三档手算独立期望），基线 = 我审结的 `fd032bf`，merge-base `3e758f6`
- 我的复核树：`.worktrees/jing_c1_recheck`@`582c1f0`（跑完 `git status` 干净）／`.worktrees/jing_c1_base`@`3e758f6`（全量对差基线）
- 独立复核件（R394④ 自写，不复用她的取证件）：`tools/mirror_c1_three_blocks.py`（子命令 `mut` / `ca`）
- 前序审头：`docs/审核-C1-IMPLEMENT-钩子侧车与装置面-20261005.md`（同枝 `0b215f9`；三条 🔴 = C-A/C-B/C-C，当时实测 M1/M3/M4 均 0 红）
- 派单原文（防漂移）：PI·00:0x 续派②「C1 三阻塞复核——轻舟回执 @582c1f0（C-A fail-fast／C-B 两处对账守／C-C T13 独立期望），@你复核三行 → PI R225 会签 + 合 main」；第十二班「请按其板帖 §四逐行复核销项」

---

## 一、结论

🟢 **三阻塞全部销项（三条原 0 红变异现已全部转红＝守住）**，且两处她做得比我的建议更严。靶向 37 passed + 1 skipped 复现一致；全量面**零新增红**（FAILED 集与基线逐名一致，双向空集）。
**新增一条 🟡 起跑门提示**（§五：fail-fast 的代价是"无 scipy 的跑批机会被起跑前拒跑"，须进 Pre-Flight）＋两条 ⚪ 更正（其中一条更正我自己，§六）＋三条残余盲区如实登记（§七）。未触及的 C-D~C-I 级别不变（§八）。

| 原阻塞 | 她的修法 | 我的独立复核 | 判定 |
|---|---|---|---|
| C-A 收尾段 scipy 崩 | `a4:1405-1424` 解析期 `ceil(ticks/sample)≥3 ∧ 无 scipy ⇒ ap.error` | A/B 实跑：rc=2 且**四个产物一个不落** vs 恒假后 rc=1 且侧车二/summary 全丢 | 🔴→🟢 **销项** |
| C-B site-2 无对账守 | T11b 加 `len(log)==_ec_prey_kill_n` + **新增 wound 档**覆盖 site-2 | m-site1 红 2 例 ｜ m-site2 红 1 例（正是 wound 档）| 🔴→🟢 **销项（比我的建议强）** |
| C-C 同式自证 | 弃重算式，改三档手算常量表（60×120／**61×125 非整除**／**480×960 纯映射**）+ `bincount` 聚合锚 | m-cc120 红 2 例 ｜ m-ccrow 红 4 例 ｜ 47 个点 + 三档分块计数用**另一条计算路径**独立验算，零冲突 | 🔴→🟢 **销项** |

## 二、变异重放（`py tools/mirror_c1_three_blocks.py mut --tree .worktrees/jing_c1_recheck`）

| 变异 | 摘哪一段 | 结果（基线 37 passed / 1 skipped / 2.84s） | 期望 |
|---|---|---|---|
| m-site1 | `sphere_engine.py:4675` site-1 append guard ⇒ `if False:` | `2 failed`（`test_t11b_real_run_three_tuple` + `test_cb_wound_death_site_two_accounting`） | 红 ✅ |
| **m-site2** | `:4708` site-2（致伤致死）append guard ⇒ `if False:` —— **= 我审头 M1 原案** | `1 failed`（**仅** `test_cb_wound_death_site_two_accounting`） | 红 ✅ |
| **m-cc120** | `a4:348` `col = flat_idx % self._cols` ⇒ `% 120` —— **= 我审头 M4 原案** | `2 failed`（61×125 档 + 480×960 档） | 红 ✅ |
| m-ccrow | `a4:349` `lat_band = (row*block_rows)//self._rows` ⇒ `//120`（**我加严**，非她列出的变异） | `4 failed`（`test_block_map_corners` + T13 三档全红） | 红 ✅ |
| m-ca | `a4:1405` fail-fast 条件 ⇒ `_plan_windows >= 999999`（= 修复前行为） | `1 failed`（`test_ca_no_scipy_fail_fast_rc2`） | 红 ✅ |

全部跑完 `git checkout --` 还原 ⇒【还原复跑】37 passed；`git status --porcelain` 空。**未销项 = 0**（审头当时是 M1/M3/M4 三条 0 红）。

## 三、C-A 的检出面（`... ca`，同一命令 300t／50 窗步长 ⇒ 计划 6 窗 ≥3）

| | rc | 主表 | 侧车一 | 侧车二 | summary（含 manifest 节） |
|---|---|---|---|---|---|
| **A** 交付原样（带 fail-fast） | **2** | ✗ | ✗ | ✗ | ✗ |
| **B** fail-fast 恒假（修复前） | 1 | ✓ | ✓ | ✗ | ✗ |

B 的崩因字符串含 `No module named 'scipy'` ⇒ 与我审头描述的"跑完之后在最贵时点崩、侧车二/manifest/provenance 全丢"**现场一致**；A 则一个产物都不落、stderr 点名 scipy。
⇒ 这条守**不是换个报错文案**，是把失效位置从"烧完整批机时"移到"起跑前拒跑"。七项断言全 ✅（`%TEMP%\jing_c1_recheck_ev\ca.txt`）。

## 四、C-B 独立复算（R394④：不用她的断言，直接量数）

同一段 `build("off", False, 221, 100, max_count=500, ...)` 我自己跑：

| 档 | `len(log)` | `_ec_prey_kill_n` | `_duel["kills"]` | `_duel["wounds"]` |
|---|---|---|---|---|
| 默认档（wound 关） | 52 | 52 | 52 | **0** |
| `wound_enabled, wound_base=1.0` | 54 | 54 | 54 | 64 |
| 同上 + m-site2 摘除 | **35** | **54** | — | 64 |

⇒ 她 docstring 引的 35 / 54 / wounds=64 / site-2=19 **逐一复现一致**；默认档 `wounds=0` 证明 site-2 确实不开火。

## 五、🟡 起跑门提示（请 PI 在 R225 会签件里处置，我不裁定）

fail-fast 采"缺依赖就起跑前拒"，代价是**新增一条硬前置**：

- C1 正式批形制 = `ticks≈2000` / `--rd-sample-every 250` ⇒ 计划窗数 8 ≥3 ⇒ **若跑批机无 scipy，该批会 rc=2 起跑前拒跑**（不是静默错，但也起不来）。
- 砚侧：Pre-Flight 须加一行"跑批机 `import scipy` 可读"，并把结果写进起跑令记录（R225/R366 起跑门）；第十二班通报"云机 Pre-Flight selftest 00:02 已过"——**selftest 是否覆盖 scipy 探针我无法从代码面确认**，请板桥/砚 回一句云机 `import scipy` 结果即可闭合。
- 另一条可选项（我审头修法③，现在看成本比当时估的更低）：`observatory/statistics.py:212 _rankdata` + `:234-244` 的中心秩 Pearson **就是纯 numpy 的 Spearman**，与 `a4:479 _spearman(d1[mask], d2[mask])[0]` 数学同式 ⇒ 换成"单一实现"引用可**彻底消除 scipy 前置**。代价与纪律：`stability` 是诊断列不是判据列，但换实现仍须按 §12.2 C7 做一次同配置对拍（剔计时列逐位一致）后才准进正式批 —— 我不建议无对拍直接换。

## 六、⚪ 两条更正（其中一条更正我自己，R73）

1. **我审头 §四-C-B 给的"一行修法"不足以挡住我自己发现的 M1。** 原文写"在 `test_t11b_real_run_three_tuple` 加 `assert len(log) == int(e._ec_prey_kill_n)`（seed 221 实测 52 杀，两处点位都在该计数内 ⇒ M1 必红）"。实测：**默认档 `wounds=0` ⇒ site-2 根本不产生条目 ⇒ 摘掉 site-2 append 后 `len==kill_n` 仍恒等 ⇒ 该例仍绿**（本件 m-site2 下 `test_t11b` 通过、仅新增的 wound 档红）。她的 `test_cb_wound_death_site_two_accounting` 比我的建议强，并把"覆盖自证" `assert e._duel["wounds"] > 0` 写进了测试（不许该档静默退化为只覆盖 site-1）。**教训我收下：对账类补守必须问"这一档里两个量是否真的都被激励过"，不能只看公式恒等。**
2. **我审头 §四-C-A 把"同源风险"点成 `observatory/statistics.py` 是错的。** 全仓 `import scipy` 只在 `experiments/a4_verify_capacity.py`（`:479`/`:1413`，即本次这一处）；`statistics.py:214` 的注释恰是"**纯 numpy，无 scipy 依赖**"，`gate_pair_analysis.py`/`t0_patch_geometry_probe.py` 也各自写明"不引入 scipy"。真正的连带项只有存档件 `evidence/r198/s2_smoke.py`（不参与跑批）⇒ 我原审"单开一张依赖声明卡"的建议**降级**为：本卡销项后无同类待办（除非 §五 选项落地后彻底清零）。

## 七、⚪ 残余盲区（如实登记，三条都不阻塞销项）

1. `m-ca`（删 fail-fast）**只在缺 scipy 的机器上会红**；有 scipy 的机器上该分支不可达，互补例 `test_ca_with_scipy_three_windows_complete` 断的是正常收尾 ⇒ 若将来有人删守，绿环境不会报警。性质上可接受（守本来就只管缺依赖场合），但值得在卡结果栏写一句。
2. `_plan_windows = ceil(ticks/sample)` 是**上界** ⇒ 方向安全（宁误拒不漏放），但存在"计划 3 窗、实跑 2 整窗＋空残窗"被拒的保守边界；遇到时按报错提示调 `--rd-sample-every` 即可，不是缺陷。
3. 上界成立的前提是 **F6 保留**（`a4:1753-1760` rd 批禁续跑 ⇒ `run_rows` 不跨段累积）。若将来解除 F6（=我审头 C-D 那条无测试的分支），该式须同步复核。

## 八、本次未触及项（级别不变，请 PI 决定是否随本卡销项同批）

| 项 | 复核结论 |
|---|---|
| C-D 🟡 F6 续跑禁令无测试 | `582c1f0` 未加（测试文件内 grep `resume` = 0 命中）⇒ 仍开 |
| C-E 🟡 bg 两参未复刻 `__post_init__` 断言 | `a4:855-858` 两行赋值后仍无断言 ⇒ 仍开（正式批取 0.4/0.05 在范围内 ⇒ 不阻塞起跑） |
| C-H 🟡 `arms_manip_check` 本体仍在 `tests/test_c1_rd_instruments.py:448` | 仍开（Q-C 双臂接线批之前落最省） |
| C-F / C-G / C-I ⚪ | 仍开；C-G 的 `RuntimeWarning: invalid value encountered in divide`（`a4:411`）在本次靶向跑里仍出现 1 次 |

## 九、零新增红与口径披露

`py -m pytest tests -q --ignore=tests/test_broker.py`（本机、无 Rust 扩展、无 scipy）：

| 树 | 结果 |
|---|---|
| base `3e758f6` | 134 failed / 1032 passed / 25 skipped（712.28s） |
| head `582c1f0` | **134 failed** / 1069 passed / 26 skipped（714.27s） |

FAILED 集**逐名双向差为空** ⇒ 零新增红、零消失红；passed +37 / skipped +1 = 新增靶向件本身（C1 测试文件在 base 树不存在）。
🟡 **口径标注**：她回执写"r98 失败集与 main 树逐名一致（**15** 离线存量）"，我本机存量红是 **134**（缺 `sim_core.so` 与 scipy ⇒ 两类整片红）。**"逐名一致"这一判据两口径都成立**，但绝对数不可跨机比较 —— 会签件里请别把 15/134 当冲突，也别拿我这 134 去核云机。

## 十、请 PI 处置（R384：结论 + 证据路径 + 阻塞）

| 项 | 请求动作 |
|---|---|
| C-A / C-B / C-C | 🔴→🟢 三行销项，候 R225 第二拍会签 + 合 main（R382 cherry-pick 或 merge 由你定） |
| §五 起跑门 | 请随会签件一并处置：云机 `import scipy` 一行回话（板桥/砚）→ 或采纳"改用仓内纯 numpy Spearman"清零依赖（换前 C7 对拍） |
| §六-1 | 我方原修法不足的更正，请记入卡结果栏（免得后人以为一行对账就够） |
| §八 C-D/C-E/C-H | 是否随本卡同批还是单开拍，交你裁（我不改、不催轻舟） |
| 本件交付 | `task/C1-REVIEW`@（本次提交）= 审头 + `tools/mirror_c1_three_blocks.py`；任务卡 `C1-IMPLEMENT` 审核结果登记请**板桥/PI 代执**（R371 我不写共享主树） |
| 队列③ L1 | 按第十二班：前提已消（S3+G6 已合 main=`75689ce`），候澜舟 rebase 重上后我复审 —— 不入当前队列，不压 C1/M1 |

**证据路径**：`%TEMP%\jing_c1_recheck_ev\{mut.txt, ca.txt, base_full.txt, head_full.txt, base_failed.txt, head_failed.txt}`；C-B 数字复算与 C-C 独立 oracle 为会话内命令（输出已抄录 §四/§一），复现命令写在审头各节标题行。
**纪律**：只审不改（两棵被审树跑完 `git status` 均干净）｜不裁定路线｜未启动任何跑批（R117，仅 300t/60×120 缩微取证，未触 `--device s2`）｜群报不可达（本 waker 无已启用 IM channel ⇒ 板帖即 R387 双轨之板轨）。
