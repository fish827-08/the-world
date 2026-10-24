"""observatory：进化观察台（模块四）。

- traits.py      统计口径：trait 表（14 基因位 + 派生指标）解码
- statistics.py  纯函数聚合：从引擎数组算一个观测点的全部统计量
- observer.py    采样器：按世代推进 + 兜底节拍采集 GenerationSample
- experiment.py  确定性实验调度 / Runner / 预置实验矩阵
- broker.py      快照桥：每 N tick 推一条 JSON 快照给 WebSocket 客户端
"""