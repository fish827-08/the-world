# `.locks/` — `_share/` 写入锁目录

> 建立：2026-09-17（R111，fish 指示"加锁防并行写入"）
> 工具：`tools/share_lock.py` ｜ 规则：`AGENT.md` §3.1.2

---

## 这是什么

每次有人要写 `_share/`（讨论板等）时，**先在此目录放一个 `<槽位>.lock` 文件**；
写完、push 后删除。

- **目录里有活跃锁 = 有人在写**；看文件名即知是谁（`dev.lock` = 老工在写）。
- 锁文件内容 = 单行 JSON：角色、任务、时间戳（`ts`）、TTL、host —— `status` 命令直接可读。

## 槽位对照（`.lock` 文件名 → 角色）

| slot | 角色 |
|------|------|
| `owner` | `[所有者]` / 天平 |
| `dev` | `[本地开发]` / 老工 |
| `eval` | `[内评]` |
| `web` | `[联网]` / 网评 |
| `cloud` | `[云端]` |
| `collab` | `[协作]` |

（自定义槽位也可用；`tools/share_lock.py` 里 `SLOTS` 表为准。）

## 命令（三步：查 → 锁 → 写）

```bash
python tools/share_lock.py status          # ① 写前先看：谁在写？（空闲才继续）
python tools/share_lock.py acquire --slot dev --task "追加 R111 帖"   # ② 取锁
# …拉取、重读板尾、追加、commit、push…
python tools/share_lock.py release --slot dev                          # ③ 放锁
python tools/share_lock.py stale-clean --yes    # 维护：清过期锁（TTL 默认 15min）
```

## 纪律

1. **锁不是摆设**：写 `_share/` 前必查（`status`），拿到锁才写；未拿锁的写入按"流程未完成"处理。
2. **过期自动失效**（TTL 默认 900s）：崩死残留不会永久堵门——`stale-clean` 或下次 `acquire` 会自动清。
3. **禁止手删他人活跃锁**：确认对方崩死/走失时，才用 `stale-clean --force-slot <slot>` 并板上公告。
4. **同机多会话 = 硬保护**（本机制的主要目的）：同一台机器上多个会话共享文件系统，锁实时生效——本机历史上发生过并发编辑竞态与双执行事故。
5. **跨机（云端）不依赖本锁**：锁文件**不入库**（`.gitignore`）——"锁随 git 同步"对发帖场景的防护窗口 ≈ 0（锁 push 出来时写入早已完成）。跨机冲突仍靠既有防线：**git push fast-forward 拒绝 → pull 合并 → 重读板尾**。跨机协作信号的更优方案留给 `[协作]` 线（T-1）。
6. **锁文件不进 git**：`.locks/*.lock` 已被 `.gitignore`；`README.md` 入库。
