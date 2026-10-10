# 证据包 · SMELL-V1

- 生成时间：2026-09-27 16:19:37
- 分支 / HEAD：`dev/yunqi-smell-v1` / `28eadc1`
- `gitee/main`：`5523aa9`
- 未提交文件数：**0**
- 当前纪元（AGENT.md §十三 末条）：**14.8 🔴 云端自审与自优化（2026-09-24 立；fish 裁定「本地无法审核」）**
- 文件数：**18**

## 说明

R240 T8 气味场 v1：默认关逐位不变 + 性能实测（配对口径，4 通道 Δ=0.39-0.57 ms/tick；稀疏地板上 +0.46）+ AST 写入点守卫（两红线）+ 快照往返 + DEL-8 四变异红/绿

## 文件清单（完整 sha256 见同目录 `SHA256.txt`）

| 文件                                                          | 字节     | 修改时间        | sha256(前16)      |
|-------------------------------------------------------------|--------|-------------|------------------|
| `_rerun_logs/smell_v1_smoke/commit_msg_smell_v1.txt`        | 2,820  | 09-27 16:13 | 400a37880badad28 |
| `_rerun_logs/smell_v1_smoke/del8_M1_write_tag_green.txt`    | 99     | 09-27 15:56 | 580688cb1ffbcd40 |
| `_rerun_logs/smell_v1_smoke/del8_M1_write_tag_red.txt`      | 1,685  | 09-27 15:56 | 07706ac82d7e5c35 |
| `_rerun_logs/smell_v1_smoke/del8_M2_cadence_gate_green.txt` | 99     | 09-27 15:56 | ada926907995aa82 |
| `_rerun_logs/smell_v1_smoke/del8_M2_cadence_gate_red.txt`   | 1,769  | 09-27 15:56 | abfe02287d3e0bee |
| `_rerun_logs/smell_v1_smoke/del8_M3_from_dict_green.txt`    | 99     | 09-27 15:56 | 102b5f09729e7de7 |
| `_rerun_logs/smell_v1_smoke/del8_M3_from_dict_red.txt`      | 1,773  | 09-27 15:56 | 1957e6cdc3a8260f |
| `_rerun_logs/smell_v1_smoke/del8_M4_s_eff_gcd_green.txt`    | 99     | 09-27 15:56 | 32039b8e0d1451d4 |
| `_rerun_logs/smell_v1_smoke/del8_M4_s_eff_gcd_red.txt`      | 1,631  | 09-27 15:56 | ed4571d4d33c9ab7 |
| `_rerun_logs/smell_v1_smoke/pytest_full_final.txt`          | 3,221  | 09-27 16:13 | 5e656d375889efcd |
| `_rerun_logs/smell_v1_smoke/self_audit_digest.txt`          | 1,695  | 09-27 16:07 | aeabba0d482af186 |
| `_rerun_logs/smell_v1_smoke/smell_perf_full.txt`            | 2,830  | 09-27 15:55 | 166a3290310ce294 |
| `experiments/smell_perf_probe.py`                           | 10,168 | 09-27 16:17 | e8d74b4a4f0c0628 |
| `results/smell_v1/provenance.json`                          | 2,605  | 09-27 16:18 | dd97719d1f28ca80 |
| `results/smell_v1/smell_perf.csv`                           | 1,007  | 09-27 15:55 | 72975b7ab840b2f5 |
| `tests/test_smell_v1.py`                                    | 13,342 | 09-27 16:17 | 1bdb4bc4b110f510 |
| `tests/test_smell_write_guard.py`                           | 8,912  | 09-27 16:17 | ecc8778564783425 |
| `world/smell_field.py`                                      | 17,965 | 09-27 16:17 | 69565d10260cbd33 |
