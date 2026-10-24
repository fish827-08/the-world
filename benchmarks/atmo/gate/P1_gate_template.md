# 《乙方案 P1 能力门》报告 —— <日期>

> 判定人：AV（独立验证线）｜判据冻结 commit：`<判据注册表所在 commit hash>`｜被测 commit：`<A1/A2 交付 hash>`
> 判定规则（先冻结，见 criteria.py；判据不因结果回调）：
> - **PASS**：Held–Suarez 全部 mandatory PASS，且 CONS 全部 mandatory PASS；若 A2 湿件已交付，AQUA 全部 mandatory 亦须 PASS。
> - **FAIL**：任一 mandatory FAIL（代码已交付且非环境原因）。不得以"调参后可过"改判；实现线修码重跑走新提交，判据不动。
> - **CONDITIONAL**：A2 湿件未交付（进度原因而非质量）且 HS/CONS 全 PASS ⇒ 湿路未验证，转甲与否由 owner 裁。
> - 对照裁判（ExoPlaSim/观测）只作参考统计，不作 PASS/FAIL 输入。

## 一、被测物与复现

| 项 | 值 |
|---|---|
| A1 动力核心 commit / 分支 | |
| A2 物理包 commit / 分支 | |
| AV 判据 commit | |
| 运行机 | |
| 复现命令 | `python -m benchmarks.atmo.run --suite held_suarez --run-dir … --run-dir-b …` |

## 二、实测读数（只写实测值；证据 = `_rerun_logs/av-*.json`）

### Held–Suarez（<运行天数> 天）

| 判据 | 实测 | 阈值 | 状态 |
|---|---|---|---|
| HS-FORCING | | | |
| HS-JET-U / HS-JET-LAT | | | |
| HS-SFC-U | | | |
| HS-T-TROP-SFC / POLE-SFC / TROP-UP | | | |
| HS-EKE-ML / RATIO | | | |
| CONS-MASS / CFL / POLAR | | | |
| CONS-DET | | | |

### 水行星湿基准（若已交付）

（同上表格式；AQ 行 + CONS-WATER/ENERGY）

### 对照裁判（只读参考统计）

| 场 | bias | RMSE | 模式相关 | 参考 |
|---|---|---|---|---|
| u 纬向平均 | | | | |
| T 纬向平均 | | | | |
| 降水 | | | | |

## 三、写死审计门

`python -m benchmarks.atmo.audit atmo_core atmo` 输出：HIGH __ 条 / MEDIUM __ 条（逐条列出；HIGH 未清零 ⇒ FAIL）。

## 四、能力门判定

- [ ] PASS / [ ] CONDITIONAL / [ ] FAIL
- 判定理由（对判据逐条，不给印象分）：

## 五、失败原因（FAIL 时）

## 六、保留资产清单（转甲时）

- A1 代码：留作参考（不并甲线路径）
- A3 耦合层/重映射：复用
- AV 套件与判据：继续有效（甲方案同判据重跑）
- 其他：

## 七、风险与未做
