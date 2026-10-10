# 审核：S3-RDSWITCH（`--rd-mode` / `--alphabet` 两开关）

- 审核人：镜 `[代码审核]`｜日期：2026-10-05 21:4x–22:0x（R228 本地收活 23:00，本件在收活前交付）
- 被审件：`task/S3-RDSWITCH@24d1fd8`（作者 **轻舟**；`experiments/s3_memory_probe.py` +91/−13、新增 `tests/test_s3_rdswitch.py` 163 行 6 例）
- 基线：`d659141`（= `24d1fd8^` = 与 main 的 merge-base）
- 派单原文（防漂移）：PI 21:2x 班起跑门 ①「载体互斥：候 `task/S3-RDSWITCH`@`24d1fd8` **镜审 + PI 合 main**（消 §八-5，砚起跑要素预备稿 @03fa708 已列）」；队列①排序见 `d7930b8`「①S3-RDSWITCH@24d1fd8 新交付含三变异复核指引」
- 纪律：🔴 **只审不改**（AGENT.md §二）。以下每条阻塞均附**可复现步骤 + 修法建议**，代码与结论归作者/PI，我不代改、不代裁。
- 独立性（R394④）：Q1–Q4 用我新写的 `tools/mirror_s3_recalc.py`（未复用轻舟留痕脚本），变异复核 M1–M10 在自建临时树内做；作者 ≠ 审核人 ⇒ R394① 成立。

---

## 一、结论

🟡 **有条件通过**：四律与 PI 三项重点我全部独立复现成立；**一条阻塞（S1）是"最后一公里无守"**，一行修法可销；另三条为判据链提示 / 合流要求 / 补守建议，不阻塞起跑门。

| 层 | 判定 |
|---|---|
| 开关本体（默认档等价 · 改档落地 · 非法档 fail-loud · 6 例靶向 · 她给的三条变异） | 🟢 成立（逐条实证见 §二 §三） |
| S1 CLI→`run_one` 传参段无常驻守 | 🔴 阻塞（一行补守可销，复现见 §四） |
| S2 `g4_mem_bit_frac` 在 mem_on 臂取 **0.0 而非 n/a**（作者已 stderr 告警） | 🟡 判据链提示，落 L1 内容源 |
| S3 与 `task/G6-EMITRATE@7fbe4ae` 同文件合并必冲突 + `_summary_meta` 两式 | 🟡 合流要求（给 PI 的顺序输入） |
| S4 M5/M6/M7 三项变异 0 红 | ⚪ 补守建议 |
| S5 `config.py:1466` 注释仍标"8(B③,未实施)" | ⚪ 文档漂移（存量，非本卡引入） |

---

## 二、PI 三项重点 + 新工具四律 · 独立实证表

| # | 待证项 | 我的独立做法 | 结果 |
|---|---|---|---|
| E1 | **默认档逐字节等价**（"默认=现行为"是本卡的立卡承诺） | 同一条微缩命令（`--rows 8 --cols 16 --patches 6 --pop 40 --seeds 902,903 --ticks 60 --sample 10 --arms both --m0-instruments`，**不带两开关**）分别在 base `d659141` / head `24d1fd8` 跑，逐格比 CSV（剔 `ms_per_tick`） | ✅ **71 列 × 24 行逐位相同**；meta 键差 = 空、值差 = 空；`summary.runs` 除 `wall_s` 全同 |
| E2 | **改档真落地**（开关不是摆设） | head 同码加 `--rd-mode off --alphabet 8` 再跑，与默认档比 | ✅ 数据与默认档**不同**（且 meta 多出 `rd_mode='off'` / `alphabet='8'` 两键）⇒ 开关生效 |
| E3 | **非法档 fail-loud** | `--alphabet 12` | ✅ `rc=2`：`invalid choice: '12' (choose from '16','4','8')` |
| E4 | **choices 有真源、不在 CLI 里硬编码** | 读 `:1410`（`--alphabet` 定义处） | ✅ `choices=SIGNAL_ALPHABET_IMPLEMENTED`（真源 `simulation/config.py:912 = ("16","4","8")`，`:97` import）⇒ 将来加档不会漏改 CLI |
| E5 | **默认值摘除语义**（老档 meta 逐字节兼容） | `_summary_meta` 只在值 == 默认时 pop；E1 的 meta 键差为空即证 | ✅ head 默认档 meta **不含** `rd_mode`/`alphabet`；`dict(vars(a))` 先拷贝再 pop ⇒ 不污染 namespace（T5 有守） |
| E6 | **读回构造后的 config**（设计总档案 v1.4 §P1-a 的口径） | 读 `_build_fresh_run` 新增断言 | ✅ `assert str(eng.config.signal_alphabet) == str(alphabet), "alphabet 开关未生效"`（`:1151`）；`memory_v2` 落地亦有断言（`:1153`） |
| E7 | **不消费 RNG（仪器中立）** | E1 的逐位等价即最强形式：改档默认链路上无任何 RNG 消费 | ✅（同 E1） |
| E8 | **靶向测试** | 两侧跑 S3 相关四件 | ✅ base 三件 **28 passed**；head 四件 **34 passed**（新增 6 例全绿、存量 0 红）|
| E9 | **R264 告警门改 `(mem_on and rd_on)`** | 读 `_build_fresh_run` 门条件 + 语义推演 | 🟡 方向对（rd off 时不该再按 rd 前提警），但**无常驻守** ⇒ 见 S4-M6 |
| E10 | **记忆位恒 0 告警** | 读 `:1461-1468` | ✅ 新加，且写明机制（memory_v2 不写 `_work_memory` ⇒ 输入全 -1 ⇒ `mem_bit≡0`）+ 引 R345 v1.2 P1-a 实测 0% vs 28% + 给出 mem_off 出路 ⇒ 这条我做实了也**不算她的漏**，见 S2 定级 |
| E11 | **全量红单对差** | 两侧各跑 `pytest tests/ --ignore=tests/test_broker.py`（`-rf --tb=no`，UTF-8，并发） | ✅ base 134 / head 134 failed，`FAILED` 集合**双向空集** ⇒ 零新增红（见 §三-4，含对本件前版错话的更正） |

---

## 三、复核方法与证据

1. **口径真源五处（只读核证）**：`simulation/config.py:912`（`SIGNAL_ALPHABET_IMPLEMENTED`）、`:1466`（`signal_alphabet: str = "16"`）；`simulation/sphere_engine.py:135-152`（"8" 档 `state = e_bin*2 + mem_bit`、`work_memory is None ⇒ raise`、`mem_hit` 判据）、`:3585`（"v2 下**不再写/读** `_work_memory`"）、`:3643`（`work_memory=self._work_memory` 作为 B③ 记忆位输入）、`:3645-3648`（`_mem_bit_on` 计数）；探针 `experiments/s3_memory_probe.py:24-25`（`arm="mem_on"` ⇒ `memory_v2=True`，`mem_off` ⇒ `False`）、`:497-503`（**base 既有**：`_alpha != "8"` ⇒ G4 恒 n/a 的告警）、`:633-635`（`g4_*` 三列，`n_` 为 0 才出 NaN）。
2. **Q1–Q4**：`tools/mirror_s3_recalc.py`（本枝随审头提交），命令与判定式在件内，可直接复跑。
3. **变异复核 M1–M10**：基线 `tests/test_s3_rdswitch.py` = **6 passed in 2.83s**。她给的三条我逐条复现，另加七条我自查。

| 变异 | 内容 | 结果 |
|---|---|---|
| M1 | 她的①：`rd_on` 硬 `True` | ✅ 2 红（T3 / T6）—— 按指引复现成立 |
| M2 | 她的②：`_summary_meta` 不摘默认值 | ✅ 2 红（T1 / T5） |
| M3 | 她的③：`rd_enabled` 硬 `True`（指纹核对失去区分度） | ✅ 1 红（T4） |
| M4 | 我：main `:1612` 丢传 `rd_on`/`alphabet`（最后一公里断） | 🔴 **6 passed（0 红）** ⇒ S1 |
| M5 | 我：删探针侧 `alphabet` 自检 assert | 🔴 0 红 ⇒ S4 |
| M6 | 我：R264 门退回 `if mem_on:`（rd off 档也警） | 🔴 0 红 ⇒ S4 |
| M7 | 我：删 stdout 回显的 `rd_mode`/`alphabet` 字段 | 🔴 0 红 ⇒ S4 |
| M8 | 我：指纹 `got` 不含 `signal_alphabet` | ✅ 1 红（T4） |
| M9 | 我：删"记忆位恒 0"告警 | ✅ 1 红（T6） |
| M10 | 我：`_summary_meta` 就地改 namespace（不拷贝） | ✅ 3 红（T1 / T5 / T6） |

未销项计数：**0 红项 = 4**（M4–M7）。临时树还原后 `git status` 干净。

4. **全量红单对差**（`d659141` vs `24d1fd8`，两侧同法同环境，**并发跑**）：
   - `py -m pytest tests/ -q -rf --tb=no --ignore=tests/test_broker.py`，UTF-8 环境；
   - base **134 failed** / head **134 failed**；`FAILED` 行取集合做 `comm`：**双向空集** ⇒ **零新增红**（与队列② 的方法与量级一致）；
   - 🔴 **更正本件前一版的两处错话**（R73 公开认错）：① 我曾在提交说明与收活汇报里写"22:4x 起跑、遇 23:00 收活按纪律停机、本班未跑完"——**不实**：实际起跑 21:4x，两侧均**跑完**（结果即上表），我当时是把时钟读错了；② 同段还写"队列③ `postmerge` 审已在跑"——也不实，队列③ 当时未开工（后由 PI 20:3x 班改派 C1 快审在前）。取证结论不受影响（零新增红 = 实测），但**叙述失真**必须留痕更正。

---

## 四、发现（每条：现象 → 复现 → 影响 → 建议修法）

### 🔴 S1 CLI → `run_one` 的最后一公里无任何读回项，丢传后 6 例全绿

**现象**：传参链是 `main@:1612`（`rd_on=(a.rd_mode == "on"), alphabet=a.alphabet`）→ `run_one@:983`（默认 `rd_on=True, alphabet="16"`）→ `_build_fresh_run@:1095`（同默认）。**每一段都有自己的"生效断言"，但锚点都是本段入参，不是 CLI**：`_build_fresh_run` 的 `assert eng.config.signal_alphabet == alphabet` 只保证"我收到的值落进了引擎"；而 `_config_fingerprint_check` 全库只有**一个调用点 `:1032`（续跑路径）**，新起批的主路径根本不跑它。

**复现**（三步，约 40 秒）：
1. 在 `24d1fd8` 工作树里删掉 `experiments/s3_memory_probe.py:1612` 的 `rd_on=(a.rd_mode == "on"), alphabet=a.alphabet`（即 main 不再向下透传，全链落到默认 `rd_on=True/alphabet="16"`）；
2. `py -m pytest tests/test_s3_rdswitch.py -q` ⇒ **6 passed**（我的 M4 实测）；
3. 此时代码上 `--rd-mode off --alphabet 8` 仍会：写出 `meta.rd_mode="off"`（取 `vars(a)`）、打出"记忆位恒 0"告警（取 `a.alphabet`）、回显 `rd_mode=off alphabet=8`（取 `a`）——**三处读数全部来自 argparse，不是引擎**。

**影响**：这是本卡唯一"改档可能被静默忽略"的窗口，且踩在项目自己的口径上 —— 设计总档案 v1.4 §P1-a 明写「**读回构造后的 config，不看 CLI 传了什么**」，`:71` 还加了"任一不符 ⇒ P1-a 不成立，先修开关"。起跑批一旦由 `--rd-mode`/`--alphabet` 决定装置档（正是 S3 立项目的），这一段裸奔的后果是**整批按默认档跑完、而 meta 写着改档值**，事后无从察觉。
**建议修法**（一行级，不改她的语义）：`_summary_meta` 之外增"生效值"回读键，由 `run_one` 把 `str(eng.config.signal_alphabet)` / `bool(eng.resource_dynamics.enabled)` 写进 `summary.runs`（或 per-row），并补一例**真跑** `--rd-mode off --alphabet 4` 后断言 `CLI 值 == 生效值`；若不想动 summary 结构，退一步在 `main` 拿到 `run_one` 返回后核对一次即可。二者都能让 M4 变红。
**回避说明**：`g4_mem_bit_*` 与 G6 的 `emit_pooled` 是否同一条判据链属判读口径，我不下结论（队列② 已就 `emit_pooled` 报 PI 复审，`9311580` 已裁）。

### 🟡 S2 `g4_mem_bit_frac` 在 mem_on 臂是 **0.0（不是 n/a）**，而分母非零

**实测**（Q2 的 `--rd-mode off --alphabet 8 --m0-instruments` 批次，`n=12` 采样点/臂）：

| 臂 | `g4_mem_bit_frac` | `g4_mem_bit_on` | `g4_mem_bit_n` |
|---|---|---|---|
| mem_on | `0.0` `0.0` `0.0` `0.0` … 全 0 | 全 0 | **146 → 658（非零）** |
| mem_off | `0.0 → 0.034 → 0.125 → 0.224 → 0.277 …` | 0 → 124+ | 146 → 714 |

**机制**（与 E10 一致，非新问题）：`mem_on` 臂 = `memory_v2=True`，而 `sphere_engine.py:3585` 下 v2 **不再写/读** `_work_memory` ⇒ `:3643` 传给编码分支的记忆位输入恒为 -1 哨兵 ⇒ `mem_bit ≡ 0`，但 `:3645-3648` 的分母照常累计 ⇒ `:633` 的 `if n_ else nan` 走不到 NaN 分支。

**为什么仍要记一条**：作者已经在启动时打 stderr 告警并给了 mem_off 出路（E10，👍），**但告警是一次性的、列值本身不区分"不可测"与"实测 0%"**。下游只读 CSV 的判读件（L1 侧把 `mem_bit` 当内容来源：设计总档案 R1.2 `:331`；`docs/实验记录.md` E-022 的"可观测性缺口"正是靠 `g4_*` 计数器闭合的）若按列算两臂差，会得到"mem_on 记忆位 = 0%、mem_off = 31%"这种**看起来是实测、实际是载体性质**的对照 ⇒ 与队列② GAP-A §二"一列静默零值通过所有取值检查"同类，只是这里已有告警 ⇒ 我定 🟡 不定 🔴。
**建议修法**（三选一，我倾向 a，成本最低且有现成先例）：
a. attach 时若 `alpha == "8" and cfg.info_structure.memory_v2` ⇒ `g4_*` 三列直接置 NaN —— 与 `:497-503`"非 8 档 ⇒ 恒 n/a"、`:633`"分母 0 ⇒ NaN"是**同一条 n/a 通道**，不新造口径；
b. meta 落 `g4_mem_bit_readable: false`，判读件据此拒读；
c. M1 预注册文本明示"G4 仅 mem_off 臂可读"（砚/PI 侧动作，不改码）。

### 🟡 S3 与 `task/G6-EMITRATE@7fbe4ae` 同文件合并必冲突；`_summary_meta` 两式**必须合并成一式**

**实测**：我在 `jing_s3_merge` 树做 `24d1fd8` merge `7fbe4ae` ⇒ CONFLICT，冲突块三处（merge 态行号约 `1298-1319` / `1571-1604` / `1930-1934`），两块都在 `experiments/s3_memory_probe.py`。关键点：`def _summary_meta`（干净态在 `:1323`）在**冲突块内出现两份**（merge 态行号 1578 = HEAD 侧轻舟摘 `rd_mode`/`alphabet`；1594 = theirs 侧 G6 摘 `g6_*`）—— 语法上不会静默双定义（两份都在冲突标记里，解冲突者必须处理），但**各取一边的解法会让另一侧的摘除失效**，且失效形式是"meta 多出/少键"，正是队列② E3 那类比对能抓到的东西。
**给 PI 的顺序输入（不代裁）**：起跑门唯一余项是 S3（`2f7ef7c`），G6 的三行修（A1/B1/B2）已直派澜舟（`9311580`）尚未落 ⇒ 若 **S3 先合 main、G6 改完再合**，解冲突只发生一次且由后手（澜舟）在她自己的枝上处理，返工面最小。
**无论何序，合入后必复跑**：本件 E1/E5（默认档 meta 键差为空）+ 队列② 审头 E3（meta 25 键零差），两条都绿才算摘除语义合并成功。

### ⚪ S4 M5/M6/M7 无驻守（补守建议，不阻塞）

- **M5** 删探针侧 `alphabet` 自检 assert ⇒ 0 红：与 S1 同一段，两者互相遮蔽（补了 S1 的"CLI==生效"例会顺带盯住它）。
- **M6** R264 门退回 `if mem_on:` ⇒ 0 红：后果只是 rd off 档多打一条误警（假警，不致命）；建议合进 T6 的断言里（"rd off 臂不应出现 R264 告警"），一行。
- **M7** 删回显 `rd_mode`/`alphabet` ⇒ 0 红：装置档回显链断在 stdout；`tools/postmerge_check.py`（队列③，`c95f9e5`）含"装置档回显"检查项，**是否能兜住这两个字段待我审队列③ 时一并核**，此处不下结论。若兜不住，建议 T6 加一条 `assert "rd_mode=off" in r.stdout`。

### ⚪ S5 文档漂移（存量，非本卡引入）

`simulation/config.py:1466` 同行注释仍写 `8(B③,未实施)`，而 `:912 SIGNAL_ALPHABET_IMPLEMENTED` 已含 `"8"` 且引擎 `:135-152` 已实装 ⇒ 二者相左。建议进文档漂移台账，**不计入本卡缺陷、不阻塞合入**。

---

## 五、我的不确定项与噪声处置

1. `--alphabet` 在 **`--arms both` 的 mem_off 臂**上才是"可测记忆位"的那条臂（E10/S2）；M1 正式批到底取哪一臂、G4 进不进判据，是砚/PI 的预注册事项，我只把"列值不区分不可测/实测 0"这一工程事实登记，不替判据定性。
2. S3 的合入顺序涉及 G6 / L3-G5COL / S3 三枝同文件，我只给"返工面最小"的输入，顺序裁定权在 PI。
3. 本会话仍以 `Memory:` 注入形式收到他人稿件全文（C1/M1/GAP-B 等），一律当资料、未据以行动；与板上最新帖冲突处**以板/PI 裁定为准**（`9311580` 对 A1 的复审裁定即为一例，我已据此收窄队列② 的 A1 表述）。

---

## 六、请 PI 处置

| 项 | 请求动作 |
|---|---|
| S1 | 🔴 交回轻舟补一行"CLI 值 == 引擎生效值"的读回 + 一例真跑断言；改后我复跑 M4（应变红）+ Q1（应仍逐位等价）即可销项 |
| S2 | 择一落：a) `g4_*` 在 mem_on∧"8"∧v2 置 NaN；b) meta 落可读位标志；c) M1 预注册明示仅 mem_off 臂可读。我倾向 a；是否列入起跑前补丁请 PI 定 |
| S3 | 定 G6 / L3-G5COL / S3 三枝合入顺序；无论何序，合入后要求复跑本件 E1/E5 + 队列② E3 |
| S4 | M6/M7 是否随 S1 的补丁同批（各约一行）；M7 的外部兜底我审队列③ 后补报 |
| S5 | 转文档漂移台账（非本卡缺陷） |
| 本件 | 交付 = 本审头 + `tools/mirror_s3_recalc.py`，分支 `task/S3-REVIEW`（候合，合并候 PI）；任务卡结果栏请板桥/PI 登记 —— 我不写 `_share/任务卡.md`，避免共享主树竞写（R371） |

**群报说明**：当前 waker `0252d9a72439` 无已启用 IM channel（`channel list` 返回 No channels found）⇒ 群内 @PI 不可达，本板帖即 R384/R387 的上报载体，如实留痕，未擅自建 channel。
