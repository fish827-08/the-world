# 设计：S0′-a 审计账导出工具 schema（逐 tick CSV 科目表）

> 作者：板桥｜10-09 22:4x｜性质＝**工具 schema 设计**（承 021d7b1 分工单板桥件③；零代码零跑批，实现随澜舟降级码面同窗，R394 分工：砚出算式·澜舟出码·本件出账面目）
> 对表对象：砚《判据起草-E-CONSERVE审计账科目表-20261009》@`task/S0P-PREREG:1181a6b5`（勘正笔＝行号 :184-185）；行号基＝主干 @`53d941e`。
> 红线叠守：不改正式读数 CSV 列面（审计走 **sidecar**，ENERGY-CLOSE 步A 同族纪律）；0/None 严格分开（R120）；无捕食不可达科目记 **0 并带披露位**，未埋点科目记 **None**（fail-loud 候选，绝不静默补 0）。

## 一、文件与协议

- 输出物：每 run 一份 `audit_econserve_<seed>_<runid>.csv`，落 run 目录 sidecar 位（与 `_audit_*`/EP 探针同区，**不入正式读数列面**）；
- 写序：**每 tick 行末 flush**，禁批尾一次性写（跑批中断＝残账可取证）；
- 表头版本键 `audit_schema_v=1` 进 manifest；schema 变更必须升版本号＋manifest 回显新旧对照，禁改列序复用旧文件；
- 数值口径：float64 原值落盘不格式化截尾；tick 列 int。

## 二、列面（与砚科目表逐栏对表）

| 组 | 列名 | 语义 | 源（砚稿锚） | 未埋点行为 |
|---|---|---|---|---|
| 键 | `tick` | 当前 tick | — | — |
| 存量 A | `stk_energy` / `stk_stomach` / `stk_food` / `stk_corpse` | Σ个体energy／Σ胃／Σ格食物／Σ尸体 | :463/:465/:987区/:1739 | 全在账；corpse 无捕食恒 **0**＋披露位 |
| 收入 B | `in_forage` / `in_photo` / `in_scav` / `in_digest` / `in_regrow` | 取食（转移正侧）／光合（外生）／食腐／消化（转移正侧）／再生（外生） | EC :187-198/:190/规则1/:828·:843 | `in_regrow` **[待埋⑤]⇒None**；scav **0** |
| 支出 C | `dis_meta` / `dis_move` / `dis_attack` / `dis_signal` | D1+D2／D5／D6+D7／D4 现场直记 | EP :207-210 | attack **0**（不可达）；余在账 |
| 逃逸 D | `esc_death_e` / `esc_clamp_e` / `esc_food_overflow_e` / `chk_repro_zero` | 乙形态蒸发腿／钳制删出数额／容量溢出数额／繁殖零和差 | :211实装／:2209区／:987-993区／:5107 | clamp/overflow **[待埋②③]⇒None**；repro_zero 实测钉非默认 [待埋④] |
| 残差 | `resid_R_t` | `Δ存量−(收入−支出−蒸发−逃逸−弃置)` 现算现记 | 砚稿§一 | 缺口列=None ⇒ R_t=None（**不拿残缺账冒充平账**） |
| 诊断位 | `unmech_flags` | 本 tick 含 None（未埋点）科目清单，分号分隔列名 | — | 出数披露用 |

对表注：①内部转移双栏（forage/digest）零-sum 对由消费侧存量差隐含核对，schema 不单设负列（防双计）；②消化差额「显式腿 vs 隐式蒸发」[砚待核] ⇒ 若判需 `dis_digest_e`，按 schema_v2 加列，不在 v1 预留空列；③逐格面（砚§三-3 折中案）不进本文件——另出 `audit_cell_<seed>.csv`（tick,row,col,存量,局部收支）**只作诊断**，恒等式判据仍以全球列面为准，格级是否入判据候 owner 明晨定夺。

## 三、导出工具接口（实现归澜舟，本件只定契约）

- `AuditExporter(world, engine, path)`：引擎主循环 tick 末调用 `write_row(t)`；
- 科目读取**只走现有 EC/EP 探针与三本账快照**，禁为导出改行为码（零行为面，R117/AST 同族取证照步A 先例）；
- Rust/`--device s2` 路径 ⇒ 构造期 raise（探针 Python 路径限定，砚稿§三-1；`path="rust"` 即拒启导出，**不降级出 None 满屏**）；
- 冒烟档（20t×≥5 seed）即本工具首个消费者：残差序列出数＝ε 先数后锁的「数」源。

（候镜审＝设计符合位；埋点五缺口归属与逐格裁点随明晨呈件，本件不代裁。）
