# collab-toolkit — the-world 团队协作工具包

> **协作线（`[协作]`）的工作区与交付仓**。立项：`docs/tasks/TASKS-COLLAB.md`（2026-09-17，fish 指示）。
> 性质：分支工作线——**不碰引擎代码与科学判据**，目标 = 提高团队协作效率。
> 目录设计为**自包含**：日后可整体拆出独立上传 GitHub（见"独立化"一节）。

## 内容

| 路径 | 说明 |
|------|------|
| `docs/团队协作机制-v1.md` | **核心交付**：11 项机制清单 + 4 张流程图 + 角色分工 + 成本度量（回应"治理膨胀"） |
| `docs/独立立项申请.md` | 独立立项申请书（待裁定） |
| `tools/board_post.py` | 讨论板发言 7 步一键化（锁 v1.1）：pull→取锁→重读板尾→追加→提交→push→对账→放锁；**幂等补推**；**保留置顶块** |
| `tools/board_pin.py` | 讨论板置顶区管理（T-6）：`show/set/add/remove/check`，走锁、块外字节不变 |
| `tools/board_check.py` | 板面体检（大小/署名/时钟/置顶/锁/断链/**F-R24**）+ 协作成本度量（`metrics`） |
| `tools/onboard.py` | 新会话开场提示词生成器（6 角色，角色表同步花名册） |
| `templates/新人入门包模板.md` | 派工包/开场提示词模板 |
| `tests/` | 39 例 pytest（纯标准库，无需安装依赖） |

## 快速上手

```bash
# 发言（零号规则一键执行；先 dry-run 看看）
python collab-toolkit/tools/board_post.py --slot collab --task "事项" \
    --message-file post.md --role "[协作]" --dry-run
python collab-toolkit/tools/board_post.py --slot collab --task "事项" \
    --message-file post.md --role "[协作]"

# 板面体检（--net 附 ls-remote 对账，F-R24 下推荐）
python collab-toolkit/tools/board_check.py check --net
python collab-toolkit/tools/board_check.py metrics

# 置顶区（当前要事；仅 [所有者] 维护）
python collab-toolkit/tools/board_pin.py show
python collab-toolkit/tools/board_pin.py add --text "跑批中：C1b/C2"
python collab-toolkit/tools/board_pin.py check

# 新会话开场提示词
python collab-toolkit/tools/onboard.py --role collab --net

# 测试
python -m pytest collab-toolkit/tests/ -q
```

## 设计原则

1. **只读工具不写，写的工具必带锁**：`board_check` 全程只读；`board_post` 走 `share_lock.py`。
2. **自包含**：仅依赖 Python 标准库 + git；锁功能复用主仓 `tools/share_lock.py`
   （经 `--lock-dir` 解耦，不 import 主仓代码）。
3. **GBK 控制台安全**：输出仅中文 + ASCII（F-R15 教训）。
4. **F-R24 兼容**：`refs/remotes/` 在本机不可信；落后判断一律 `git ls-remote`。

## 独立化（拆出上传 GitHub 的步骤）

本目录按独立仓标准组织（自带 docs/tools/tests/templates/README）：

```bash
# 在 the-world 根执行（保留 git 历史）
git subtree split --prefix=collab-toolkit -b collab-toolkit-export
git remote add collab <GitHub 空仓地址>
git push collab collab-toolkit-export:main
```

拆出后注意：① `board_post.py --lock-tool` 需指向独立仓内的 `share_lock.py`
（把主仓 `tools/share_lock.py` 一并拷入 `tools/`）；② 测试中的 `REAL_LOCK_TOOL` 路径同步改。

## 纪律

- 机制变更：设计稿 → 所有者核 → 公告（不静默改流程）。
- 测试先行：工具改动必须 `pytest collab-toolkit/tests/` 全绿。
- 本线产出发言走自己的工具（dogfooding）。
