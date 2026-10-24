"""biosphere.gpu：Taichi 内核占位（W3 未做；见 README.md）。"""
from __future__ import annotations

#: W3 规划标志（False = GPU 路径尚未实现；任何"开 GPU"请求都应失败）
GPU_ENABLED: bool = False
