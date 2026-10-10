# `_audit/` —— 证据包目录（R194，2026-09-24 立）

> 用途：存放**云端自审**的证据包清单（协议全文见 `docs/tasks/协议-云端自审与自优化-20260924.md`）。
> 🔴 **回板时必须附本目录下对应阶段的 `MANIFEST.md` + `SHA256.txt`——没有它，回板视为未完成。**

## 生成方式

```bash
# 在仓库根目录
python tools/self_audit.py pack --phase S1 \
    --include _rerun_logs/<你的冒烟目录> \
    --include "experiments/*_out.txt" \
    --note "S1 冒烟：3 preset × 1 run × 2k"
```

## 目录约定

```
_audit/
  S1/MANIFEST.md      # 分支/HEAD/gitee 同步/未提交数/当前纪元/文件表/说明
  S1/SHA256.txt       # sha256sum 格式，可 `sha256sum -c SHA256.txt` 复核
  S2/ ...
```

## 🔴 纪律

1. **只记哈希与元数据，不复制文件** —— 避免仓库膨胀（文件本体留在 `_rerun_logs/` 等处）。
2. **证据包要自解释**：事后抽验的人**先看它、后看代码** ⇒ 必须能回答
   「跑的是哪个档（`switches`）」「改了什么（commit）」「结论的数从哪来（原始日志）」。
3. 本目录**入库**（`.md` / `.txt` 不受 `.gitignore` 的 `*.json` / `*.csv` 规则影响）；
   故 **不要往这里放 `.json` / `.csv`**（会被 gitignore 静默忽略，导致清单与实际不一致）。
