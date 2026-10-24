# biosphere/gpu —— Taichi 内核占位（W3 再填）

本目录为 **B 线 GPU 化（W3）** 的占位空壳：CPU 参考实现（`biosphere/simulator.py`）
跑通之前不放任何 kernel 代码。

规划（总纲 §五 W3，当前**未做**）：

- SoA 布局的个体状态 + 决策 kernel（CPU/GPU 双实现）；
- 两段式提交保 L1 同卡可复现（禁原子操作；动态增删用 compaction）；
- 与 CPU 参考实现对拍：**统计等价门**（5 seed × 1000 tick，均值差 ≤1%、
  标准差差 ≤5%、KS p>0.05、定性区制一致）；机制类结论只许 L0/L1 出。
