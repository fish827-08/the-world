# L1/PM 审头文档防丢快照（2026-10-08 20:4x·PI 代执）

**性质**：防单点丢失的**原样快照**，**不是并档**。正式并档仍按 `_share/讨论板.md` PI 裁决②（@`fdbbf64`）随被审线收口同批、由交叉作者（澜舟）执行；判读引用请指向并档后的 `docs/审核-*.md`，勿引用本目录路径。

| 文件 | 出处 | 原 tip |
|---|---|---|
| `审核-L1-JOINTCOL-联合采样列交付-20261005.md` | `gitee-backup/task-L1-REVIEW-20261008:docs/…`（远端 `task/L1-REVIEW` @`3157fed`） | 5,319 B |
| `审核-R396-POSTMERGE-postmerge三查骨架-20261005.md` | `gitee-backup/task-PM-REVIEW-20261008:docs/…`（远端 `task/PM-REVIEW` @`1f57b40`） | 13,772 B |

**为什么要做**：这两份审头文档实测**不在 main**（`git cat-file -e main:<path>` ⇒ NOT in main），其本地分支引用已随件②并档批删除，`gitee` 远端引用是当时唯一存身之处；板桥另抓的 `gitee-backup/*-20261008` 只是**本地** remote-tracking 引用，不随远端走，仍属单点。

**作者归属**：两份正文作者＝镜（`jing@the-world.local`），本快照仅原样导出，未改一字。正式并档后本目录标注作废留痕即可。
