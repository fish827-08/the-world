"""biosphere：B 线（生物圈与演化）包——个体 / 基因组 / 发育 / 生态 / 度量。

通俗理解
--------
这是"世界孵化器"五层里的 Biosphere 层（见
``docs/设计文档/设计总纲-世界孵化器-v0.1-20261018.md`` §三）：

- 世界字段只经 ``world_adapter`` 读入（禁直接碰 ``world/``）；
- 基因组 → 发育（``genome`` / ``develop``）→ 控制器参数；
- 六类动作意图（``action``）经两段式提交裁定（``resolve``）；
- 轨迹 → 行为标签 / 谱系 / 交互网络（``metrics``）；
- ``simulator`` 是 CPU 参考实现，``demo`` 是冒烟入口；
- ``gpu/`` 为 Taichi 内核占位（W3 再填，当前是空壳）。

W0-B 状态：MVP 骨架（点状 agent + 合成小世界），默认不接旧引擎、
不改 ``world/`` 与 ``simulation/``。
"""

__version__ = "0.1.0"
