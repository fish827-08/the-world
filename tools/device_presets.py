"""装置预设档（R278 §三 P3 防呆交付；云归）。

动机
----
R275/批 A 出过一次事故：探针忘了传 `--rows/--cols/--patches/...` ⇒ 静默回落到
探针自带默认，跑出**另一个装置**的数据却没有任何提示。R278 §三 裁定"从根上消灭
这一类坑"，做法 = **命名预设档**：传 `--device <名>` 即可一次性把装置字段全部
摁到该档口径，且**回显展开后的全字段**供人眼复核（第三人眼校验点）。

设计原则（R278 五条要求）
------------------------
1. **默认行为不变**：不传 `--device` ⇒ 本模块**完全不介入**，探针逐位等于旧版。
2. **只加不改**：预设档只提供"值"，写入 CSV 的列/判据逻辑一概不动。
3. **fail-loud**：`--device` 名未知 ⇒ 立即报错退出（列出可用档），绝不静默回落。
4. **显式优先于预设**：用户同时传了 `--device` 与显式 `--rows 300` 时，
   **显式值获胜**，但**必须打印一行"覆盖提示"**（否则又是一次静默分歧）。
5. **单一真源**：S2/S3 两个探针共用本模块，避免"两份拷贝各自漂移"。

预设档口径
----------
`s2` = 批 A / S2 装置（R278 §三 指定）：
    rows=480 cols=960 patches=1700 pop=10000 rgm=1.195
    bg_low_prod_frac=0.4 bg_low_cap_mult=0.05
`s3` = S3 记忆探针默认装置（与本模块 `s2` 的装置字段相同，但 ticks/sample 不同族）；
    这里**只列装置字段**，`ticks/sample` 属"实验设计"而非"装置"，不纳入预设，
    以免把两个概念混在一起（改了 ticks 不等于换了装置）。
"""
from __future__ import annotations

# 只含"装置"字段（空间尺度 + 能量 + 背景带），不含 ticks/sample/seed 等实验设计字段。
# 值 = 与探针内 DEVICE/S1_BASE 逐位一致（改这里必须同步核实探针默认，见单测）。
DEVICE_PRESETS: dict[str, dict] = {
    "s2": dict(
        rows=480, cols=960, patches=1700, pop=10000, rgm=1.195,
        bg_low_prod_frac=0.4, bg_low_cap_mult=0.05,
    ),
}

# CLI 名 -> 探针 argparse 目标属性名。显式覆盖提示要用到。
_FIELD_TO_ATTR = {
    "rows": "rows", "cols": "cols", "patches": "patches", "pop": "pop",
    "rgm": "rgm",
    "bg_low_prod_frac": "bg_low_prod_frac",
    "bg_low_cap_mult": "bg_low_cap_mult",
}


def preset_names() -> list[str]:
    return sorted(DEVICE_PRESETS)


def add_device_arg(ap, default: str | None = None) -> None:
    """给 argparse 加 `--device`。`default=None` ⇒ 不传时不介入（默认行为不变的基石）。"""
    ap.add_argument(
        "--device", default=default, choices=preset_names() or None,
        help=("装置预设档（防呆，R278 §三）："
              + "/".join(f"{k}={_fmt(v)}" for k, v in DEVICE_PRESETS.items())
              + "。不传=逐位等于旧版默认；传了则把装置字段整体摁到该档口径，"
                "并与显式 --rows 等冲突时**显式值优先**并打印覆盖提示。"),
    )


def _fmt(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items())


def resolve_device(args, argv: list[str] | None = None,
                   logger=print) -> dict:
    """把 `--device` 决议结果回写进 `args`，返回"实际生效的装置字段"字典。

    调用点（探针 main 内，紧接 `a = ap.parse_args()` 之后）：

        from tools.device_presets import resolve_device
        device_applied = resolve_device(a, sys.argv[1:])

    行为：
      * `a.device is None` ⇒ 返回 {}，**不改任何字段**（默认行为不变）。
      * 否则：对每个装置字段，若用户在命令行**显式**给了该参数 ⇒ 保留用户值并发
        覆盖提示；否则写入预设值。
      * 最后打印"实际生效装置"整行（第三人眼校验）。
    """
    name = getattr(args, "device", None)
    if name is None:
        # 🔴 关键（R278 §三 要求③"默认行为不变"）：加了 --device 参数后 argparse 会
        #   给 Namespace 塞一个 device=None 键，而探针用 `vars(a)` 写进 summary 的
        #   `params` ⇒ 会在 JSON 里多出一个 `"device": null` 字段，**破坏"逐位等价"**。
        #   故此处把该键删掉，确保不传 --device 时 summary 与旧版逐字节一致。
        try:
            delattr(args, "device")
        except AttributeError:
            pass
        return {}
    if name not in DEVICE_PRESETS:
        raise SystemExit(
            f"🔴 未知 --device '{name}'；可用：{preset_names()}。中止（防呆：绝不静默回落）")
    preset = DEVICE_PRESETS[name]
    if argv is None:
        import sys as _sys
        argv = _sys.argv[1:]

    def _explicit(attr: str) -> bool:
        # 同时认 --bg-low-prod-frac 与 --bg_low_prod_frac 两种写法
        flag_dash = "--" + attr.replace("_", "-")
        flag_under = "--" + attr
        return any(tok == flag_dash or tok == flag_under
                   or tok.startswith(flag_dash + "=")
                   or tok.startswith(flag_under + "=")
                   for tok in argv)

    applied: dict = {}
    overridden: list[str] = []
    for field, val in preset.items():
        attr = _FIELD_TO_ATTR[field]
        if _explicit(attr):
            overridden.append(f"{field}={getattr(args, attr)}(显式, 覆盖预设 {val})")
            applied[field] = getattr(args, attr)
        else:
            setattr(args, attr, val)
            applied[field] = val

    logger(f"# [--device {name}] 实际生效装置：" + " ".join(
        f"{k}={v}" for k, v in applied.items()))
    if overridden:
        logger("# [--device %s] ⚠️ 以下字段被显式参数覆盖（显式优先）：%s"
               % (name, "；".join(overridden)))
    return applied
