# `2026-10-11-根目录清理/` — 根目录点文件清理入档（批次说明）

> **触发**：owner（fish）2026-10-11 直令——"清理一下这些 . 开头的文件"（附根目录截图）。
> **原则**：沿用 F-R10（只移动、不删除）——**非本仓产物 / 历史现场先打包入档，再从工作区移除**；
> 可再生成的纯运行时件（缓存 / 孤儿环境 / 失效物化副本）直接删除、不占档。执行与登记：轻舟。

## 一、打包入档（tar.gz；按原始目录名打包，解包即原貌）

| 包 | 原始位置 | 内容 | 体积（压缩后） |
|---|---|---|---|
| `workbuddy-memory-20260912-20261007.tar.gz` | `.workbuddy/` | 外部助手 WorkBuddy 的项目记忆 `memory/`（2026-09-12 → 10-07，31 件） | 592 KB |
| `qoder-collab-runtime-20261011.tar.gz` | `.qoder-collab-runtime/` | Qoder 协同运行态：历次会话哈希目录 + 消息缓冲（`_msgs*.json`）+ 板帖草稿（`board_*.md` 等；末次写入 2026-10-09，174 件） | 312 KB |
| `git-incident-20260913-residue.tar.gz` | `.git.corrupt-20260913/` + `.git-rewrite/` | 2026-09-13 git 损坏修复现场残骸：损坏骨架（不可读）+ filter-branch 遗留工作目录（280 件） | 212 KB |

**恢复**：在仓库根执行 `tar -xzf <包名>` 即还原到原路径（原始目录名含前导 `.`，已随包保留）。
⚠️ 若仍在使用 WorkBuddy / 旧协作工具，它们可能按需重新生成对应目录。

## 二、移动入档（`git mv`，git 历史完整）

- `.trae/skills/alife-board-eval/SKILL.md` → 本目录 `trae-skills/alife-board-eval/SKILL.md`
  （团队期"跟板 + 评估"工作流技能件；团队模式退役后无活跃引用，全仓检索仅自引用。）
  **恢复**：`git mv _archive/2026-10-11-根目录清理/trae-skills/alife-board-eval/SKILL.md .trae/skills/alife-board-eval/SKILL.md`

## 三、直接删除（可再生成，未入档）

| 件 | 体积 | 说明 |
|---|---|---|
| `.venv-dev/` | ≈58 MB | 孤儿虚拟环境。`pyvenv.cfg` 显示由已不存在的 `_gitee_review/.venv-dev` 创建（Python 3.14，基座为本机 AppData 的 Python314）；全仓零引用；本仓活跃环境为 `.venv/` |
| `.pytest_cache/` | 156 KB | pytest 运行缓存；跑测试自动重建 |
| `.qoder-user-skill-runtime/` | 178 KB | Qoder 技能物化副本（3 个已失效会话哈希，均非当前会话）；按需自动重建 |
| `.idea/` | 22 KB | JetBrains 工程配置（停更 2026-09-06）；IDE 打开时自动重建 |
| `.trae/`（空壳） | 8 KB | 技能件 `git mv` 后的空目录骨架 |
| `.worktrees/VIZ/collab-toolkit/`（残壳） | 60 KB | VIZ 工作树内团队期工具残壳（仅 2 个 `__pycache__/*.pyc`） |
| `.worktrees/VIZ/.qoder-sign.env` | 458 B | 团队期署名环境文件（SIGN-HOOK 已退役卸下）；提交署名照旧用 `git -c user.name=轻舟 …` 内联 |

## 四、保留（未动，说明口径）

`.git/`（版本库）、`.gitignore`、`.venv/`（活跃解释器）、`.worktrees/`（VIZ 工作树）、
`.qoder/`（Qoder 应用会话运行态，App 自管）、`.qoder-credits/`（owner 侧报告产物）。

## 五、登记

- `_archive/归档记录.md`：本批登记（同日）
- `_archive/INDEX.md`：§一 批次总表 + §二 主题路由（同日）
