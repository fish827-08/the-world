# FILE-REORG 执行报告（批A/B/C 于 task/FILE-REORG 分支完成）

> **卡号**：FILE-REORG｜**执行**：板桥（协作）｜**日期**：2026-10-05
> **授权**：fish 已确认 **D1① / D2② / D3① / D4候镜后②**（板 36b2349）；PI 派单"先 worktree setup 建自己的工作树再执行 D1/D2/D3（**禁碰主树**），D4 等镜。零删除、pathspec、署名板桥"。
> **上游提案**：`docs/方案-FILE-REORG-文件体系重整提案-v0-20261004.md`（§三目录树 / §五零断链机制 / §六执行序）

---

## 一、执行位置与形态（对上 D2②）

全部工作在 **`.worktrees/FILE-REORG`（分支 task/FILE-REORG）** 完成，**主树零改动**——这是对 D2②"C1 起跑后再执行"的字面落实：文件树重整不依赖主树路径切换，**合入时点由 PI/fish 在 C1 起跑后定**，随时可合、随时可 revert（一批一 commit）。

D4（`docs/协作-云端双角色流程-20260922.md`）**未动**，候镜表态。

## 二、批次与 commit 台账（全部 `git mv` 零删除、pathspec 提交、署名 板桥）

| 批 | commit | 内容 | ref_check 门槛 |
|---|---|---|---|
| 工具 | `4866e80` + 修正 `2df731a` | `tools/ref_check.py`（snapshot/check/diff，8 测绿）——提案 §五 前置工具 | — |
| A-1 | `45020a2` | 判读/审核/核查/读数/交付/验收 **20 件** → `docs/判读归档/`（D3①） | 新增断链 **0** ✅ |
| A-2 | `d4266d2` | 技术文档 **3 件** → `docs/技术文档/` | 0 ✅ |
| A-3 | `105fee1` | 规格与机制 **4 件** → `docs/规格与机制/` | 0 ✅ |
| A-4 | `e58add1` | 预注册稿 **2 件** → `docs/预注册/`（违规遗留清账） | 0 ✅ |
| B | `6f4adca` | 根下 `_advice/`+`_eval_reports/`+`_roadmap/` **整目录** → `docs/外来参考线/`（**D1①** 可检索不降权）；31 个活引用文件同步替换 + 1 处旧错名引用修真名（`评估报告-元宝` → 实名） | 0 ✅ |
| C | 本 commit | `docs/README.md` 索引刷新（新目录+实测文件数+散落件现状）+ 本报告 | 终检 0 ✅ |

**合计**：docs 平铺 28 件下沉 + 3 个根目录并入；docs/ 顶层只剩 活文档四件（README/记忆索引/实验台账/实验记录）+ 边界件 6（留守理由见 README §根下散落件）+ `design_attachments/`。

## 三、零断链核验数据（卡验收硬项）

- **基线**（动手前快照 `_rerun_logs/reorg/before.json`，不入册）：token **1062** 个，**存量**断链 538（历史遗留，非本批新账）。
- **逐批门槛**：`ref_check diff` 对基线，**活引用面（追踪 md/py/sh，排除历史面）每批新增断链 = 0**（gate 日志 `_rerun_logs/reorg/gate_*.txt`）。
- **终检（全树口径，含排除面）**：迁移导致新触碰的旧路径引用残留 **22 条**，全部落在**历史引用面**——`_share/讨论板.md`、`_share/archive/`（5 份板归档）、`_share/花名册.md`、`docs/设计总档案/AI-设计总档案-v1.4-定稿.md`、`_archive/归档记录.md`、评审过程归档 2 份。
  ⇒ 按提案 §四"板上/归档里的**历史引用不回头改**（留痕原则）"处理，**只登记不修改**；这些面反查新路径可用 `docs/README.md` §根下散落件 与本报告 §二映射。
- 工具残留清单全文：`_rerun_logs/reorg/residual.txt`（gitignored 工作区，需要时上板贴文）。

## 四、结构与纪律执行

- 每批 = `git mv` → 活引用面批量替换（`_rerun_logs/reorg/apply.py` 自动+人工复核）→ `ref_check diff` 门槛 → **单批单 commit**（pathspec + 逐命令 `-c user.name/​user.email` 署名 R371/R227）。
- **零删除**：全程无 `rm`；分叉返工用 `git reset --hard`（仅限本会话自建未 push commit）+ `git clean`（绕行三律①）。
- **结构性文件不代改**（协作机制 §六.2）：`_share/路线共识.md`、`_share/待办与交接.md` 中命中旧路径的段落**已回退不改**——合入后请 PI 顺手续改（各 1-2 处，见 §三排除面逻辑）。
- 署名闸门 sign_hook 在本树全程生效（作者位=卡负责人 板桥，放行）。

## 五、后续（合入时 PI 需要做的）

1. `git merge task/FILE-REORG`（一批一 commit，可整支合或分批 cherry-pick；冲突点预期仅 `_share/任务卡.md`——主树该文件在分支建后有他线更新）；
2. 合入后 `tools/ref_check.py snapshot --out x.json` 跑一次全树确认活面 0；
3. 择机把 `ref_check` 并入 `board_check.py` 体检项"游离引用"（提案 §五.3，建议下批 collab 小单）；
4. D4 候镜后按"②归档"走 `_archive/` + 登记。

## 六、验收要点对照（对上卡片）

- [x] 目录树 before/after（提案 §三 + README 新表）
- [x] 引用影响面实测（§三：1062 token 基线 / 活面 0 新增 / 历史面 22 残留登记）
- [x] 零断链核验机制（ref_check 工具 + 逐批 gate + 8 单测）
- [x] 执行全程在自建 worktree，主树未碰（§一）
- [x] D4 未动
- [ ] **合入 main = PI 动作**（D2② 时机）⇒ 卡片置 done 待合入后随板简报确认

---

**版本**：v1（2026-10-05）｜执行署名 [协作·板桥]｜工具：`tools/ref_check.py`、`collab-toolkit/tools/worktree.py`
