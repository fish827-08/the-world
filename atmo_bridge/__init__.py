"""`atmo_bridge` —— 外部 GCM 的离线气候表桥接（进程外、只读表）。

边界（总纲 §一/§三）
------------------
本包**只做三件事**：① 把 GCM 输出（CSV/NetCDF）整理成 `npz-climatology-v0` 表；
② 把表从 GCM 网格**保守重映射**到世界 480×960 网格；③ 提供命令行入口（冒烟/对照）。
**不跑 GCM、不链接 GCM、不改世界物理**。GCM 在独立进程/机器上跑（EUPL/GPL 合规见
`内部技术文档/技术-A3-SpeedyWeather许可证核实-20261018.md`）。

模块
----
* `remap` —— 规则经纬网格之间的**保守（面积权）重映射**，纯 numpy。
* `table` —— 表格式读写与校验（`npz-climatology-v0`）。

为什么不用双线性
---------------
双线性**不守恒**：降水场会被凭空放大/缩小。气候表要进世界的水循环，先守恒再谈形状
（方案与判据见 `内部技术文档/技术-T31到480x960重映射方案-20261018.md`）。
"""
from __future__ import annotations

from .remap import build_weights, remap_field, remap_fields

__all__ = ["build_weights", "remap_field", "remap_fields"]
