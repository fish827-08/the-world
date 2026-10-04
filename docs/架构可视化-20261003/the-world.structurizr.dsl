workspace "the-world" "数字生命模拟系统 — 当前架构（system-modeler 产出，2026-10-03，证据锚定代码行号）" {

  model {

    // ===== 外部参与者与外部系统 =====
    owner = person "所有者/实验者（fish）" "通过 CLI 与批跑脚本发起实验，依据 CSV/summary 与协作板裁定（裁定制）"
    viewer = person "直播观看者" "浏览器订阅 observatory WebSocket 快照流"

    gitee = softwareSystem "Gitee/GitHub 远端仓库" "外部：主仓远端（双仓镜像）；零号规则=发言前 pull 发言后 push"
    datarepo = softwareSystem "the-world-data 独立数据仓" "外部：实验产物（CSV/JSON/snapshot npz）唯一正式落点；不入主仓"

    // ===== 系统边界 =====
    tw = softwareSystem "the-world 数字生命模拟系统" "60x120 球面网格上的数字生命演化模拟；死规则内涌现语言与智能（不使用 LLM）" {

      // ---- Python 仿真引擎 ----
      engine = container "Python 仿真引擎" "simulation/ — 唯一状态持有者；SoA 数组化整群推进" "Python 3.12 + NumPy" {
        se_step = component "SphereEngine tick 循环" "sphere_engine.py — _advance_one_tick(:2759) 外层9段 + _step_population(:3204) 内层约17阶段"
        cfg = component "SimConfig 配置总集" "config.py:1418 — 29 个 dataclass；每个 __post_init__ 硬断言；fingerprint() 哈希拦跨档续跑"
        gene_reg = component "基因注册表" "genes.py — Gene(IntEnum) 24 位 + GENE_META/SEMANTICS/WIRED + validate_gene_wiring()(:216)"
        oracle = component "oracle 正向对照仪器" "oracle.py — 接收者→发送者能量回馈通道；内容无关、零新增 RNG、守恒且保本"
        prov = component "provenance 出处校验" "provenance.py — CountingRNG（随机流逐位不变+计数）、git_commit、逐子树 SHA256"
      }

      // ---- Rust 加速核 ----
      simcore = container "sim_core Rust 加速核" "sim_core/src — 17 个 pyfunction + LightTempRust pyclass；与引擎同进程内联调用（非服务）" "Rust + PyO3 + Rayon" {
        sc_lib = component "pyo3 边界层" "lib.rs — require_len 长度校验、native_gene_indicators、validate_gene_wiring（rust_all 24 项硬编码比对）"
        sc_step = component "step_vectors" "stage1 光合/代谢/维持 + stage2 移动扣费/年龄/死亡/冷却/繁殖候选；两段式把 RNG 缺口留在 Python"
        sc_core_algo = component "movement / predation / culture / l4_l5" "移动 8 邻格得分决策；捕食与文化学习；l4_l5=捕食+文化合并单次 CSR"
        sc_batch = component "reproduction / signal / consume / regrow" "繁殖只吃预生成 RNG 数组；信号发射扣费在编码前；进食 bincount 同格竞争；资源再生"
        sc_env = component "pleasure / dispersal / light_temp / neighbors / genes" "愉悦度 Rayon 并行；L10a 果实确定性段；光照温度并行核；紧凑邻居寻址；基因索引常量镜像（单一事实源在 Python）"
      }

      // ---- 环境场层 ----
      worldc = container "环境场层" "world/ — 球面拓扑 + 五个场 + 亚格层" "Python + NumPy" {
        sphere = component "SphereWorld 球面拓扑" "sphere_world.py — 等距圆柱投影 60x120；极点坍缩；面积权重∝cos(lat)；邻居表缓存 (n_cells,8)+极点带。⚠️ 极区邻接有向非对称，局部算法触及极行必须回退全量"
        light = component "LightTemp 光照温度" "light_and_temperature.py — 固定太阳+自转⇒昼夜经度扫掠；photoperiod 日长轴；唯一由 world 直连 Rust 的模块（_season_on 时强制回 Python）"
        resfield = component "ResourceField 资源场" "resource_field.py — 容量=面积x倍率（极地养不活）；regrow/consume_many/deposit；R217 惰性结算只算脏格"
        rdyn = component "ResourceDynamics 资源负反馈" "resource_dynamics.py — 取食压力↑⇒斑块死亡↑⇒产能↓（note_tick/rotate/validate_writeback）；强制 Python 路径（H3 拦 use_sim_core）"
        sigfield = component "SignalField 田字格信号" "signal_field.py — 每格 uint8 低4位=4子格⇒16 pattern；_age 为剩余寿命；稀疏 tick 只推进 age>0 格"
        smellfield = component "SmellField 气味扩散场" "smell_field.py — 近场全分辨率+远场粗网格(s=8)两级；双线性插值在读路径（避免写回地板税）"
        subposm = component "subpos 亚格连续坐标" "subpos.py — subdiv x subdiv（默认4）；flat 派生自整数格⇒取食/信号/光照仍按格零开销；仅 Python 路径"
      }

      // ---- 观察台 ----
      obs = container "观察台" "observatory/ — 指标单一口径产出 + 采样 + 直播 + 最小统计推断" "Python" {
        stats = component "statistics 指标层" "generation_statistics / d2_metrics / codebook_convergence / predation_fraction(D-16唯一口径) / selection_gradient / fisher_z"
        observer = component "EvolutionObserver" "observer.py — 世代前进 + tick_interval 双触发采样，产出 GenerationSample"
        broker = component "SnapshotBroker 直播" "broker.py — pump 推引擎、有界环形缓冲 + asyncio 队列、WebSocket 推流（py -m observatory.broker）"
        inference = component "inference 推断" "permutation_test / bootstrap_ci / compare_groups（仅 numpy，固定 random_state）"
        expspec = component "ExperimentRunner" "experiment.py — derive_seed 确定性种子派生 + 5 组实验矩阵 + 三重停止条件；不引入适应度概念"
        cap8 = component "world_capacity C8 对账" "latitude_r2 — 位置对容量 R²=1 ⇒ 信息价值为 0 的先决对账仪器"
      }

      // ---- 编排层 ----
      orch = container "实验编排层" "experiments/ + tools/ — batch_runner 以 subprocess 拉起实验脚本（进程边界，数据契约=CSV/summary.json/snapshot）；run_batch.py 再包一层进度条" "Python"

      // ---- 持久化 ----
      persist = container "持久化" "两套机制：① 引擎级快照 save_snapshot(sphere_engine.py:6122, SNAPSHOT_VERSION=3)→npz 续跑用；② persistence/io.py→manifest.json+generations.csv 实验结果落盘" "npz / CSV / JSON"

      collab = container "协作与文档面" "_share/（讨论板/路线图/锁）+ docs/（台账/设计/预注册）— git 追踪的交流共识层，不承载引擎代码" "Markdown + git"
    }

    // ===== L1/L2 关系 =====
    owner -> orch "发起：py -m observatory / run_batch.py / run_d2_experiment.py"
    owner -> collab "读写讨论板、台账、裁定"
    viewer -> broker "订阅 WebSocket 快照流（实时观测，不参与演化）"
    tw -> gitee "pull/push（零号规则；GitHub 镜像同步）"
    persist -> datarepo "实验产物同步入独立数据仓（不入主仓=未交付）"

    orch -> engine "子进程 import 引擎后 for-tick 调 step()（无服务调用，纯本地进程）"
    engine -> simcore "use_sim_core=True 时调用 12 处下沉函数（同进程 FFI；Rust 只算数值不做决策）"
    engine -> worldc "tick 内按顺序契约组装/推进：light→resources→signals→smell→(rd rotate)"
    worldc -> simcore "LightTempRust 并行光照温度（world 侧唯一直连 Rust）"
    engine -> persist "快照 npz（约100键）+ TickStats→CSV/manifest"
    obs -> engine "只读鸭子类型消费引擎数组（不变更状态）"
    obs -> persist "GenerationSample → generations.csv / manifest.json"
    engine -> collab "config_fingerprint/provenance 写入 manifest 供审计"

    // ===== L3 组件内关系（engine 视图用） =====
    se_step -> cfg "读取全部机制开关（默认关=与旧行为逐位一致）"
    se_step -> gene_reg "按 Gene.MOVE_PROB 等语义索引消费 g0~g23"
    se_step -> worldc "外层：regrow/corpse/signals.tick/smell.update/rd.rotate；内层：感知/移动读场"
    se_step -> simcore "stage1/2、consume_many、signal_emit、step_movement、predation_and_culture、reproduce_batch、pleasure_update 等"
    se_step -> oracle "移动后归因回馈（仅 Python 路径；C-8 拦 use_sim_core）"
    se_step -> prov "CountingRNG 包裹全局 rng；跑前 collect 指纹"
    gene_reg -> sc_lib "validate_gene_wiring/native_gene_indicators 双向校验（引擎初始化查3项，测试查24项）"
    sc_batch -> sc_env "共用 genes.rs 常量与 neighbors.rs 寻址"
    resfield -> sphere "容量按 cell_area（极格≈赤道1/38）"
    rdyn -> resfield "直接写 _grid/_capacity 回写（validate_writeback 不变量守卫）"
    smellfield -> sphere "扩散 laplacian 用邻居表；极区增量回退全量"

    // ===== 备注 =====
    note on simcore "双路径逐位对拍契约：Rust 下沉模块必须与 Python 参考实现逐位一致（函数级+引擎级）；RNG 消费顺序不变（预生成随机数传入 Rust）。D2 启用时强制走 Python 路径（use_sim_core=False）。"
    note on engine "H3 fail-loud 纪律：机制未下沉而 use_sim_core=True ⇒ 构造期抛错（约13处拦截），禁止'开关开了行为不变'的静默降级。"
  }

  views {

    systemContext tw "L1-SystemContext" "谁在用这个系统、它对外部长什么样" {
      include *
      autoLayout
    }

    container tw "L2-Containers" "六个容器 + 外部边界；注意 orchestrator 与引擎是进程边界，engine 与 sim_core 是同进程 FFI" {
      include *
      autoLayout
    }

    component engine "L3-Engine" "引擎内部：tick 循环与配置/基因/oracle/provenance 的接线" {
      include *
      autoLayout
    }

    component simcore "L3-SimCore" "Rust 核内部分层：pyo3 边界层与纯数值核" {
      include *
      autoLayout
    }

    component worldc "L3-WorldFields" "环境场层：拓扑被所有场依赖；光照是唯一直连 Rust 的场" {
      include *
      autoLayout
    }

    component obs "L3-Observatory" "观察台：离线统计流与实时直播流两条出口" {
      include *
      autoLayout
    }

    styles {
      include "https://static.structurizr.com/max/structurizr-styles-v1.json"
      element "外部" { shape Hexagon; color #999999; }
      element "person" { shape Person; color #0842A0; }
      element "Python 仿真引擎" { background #2b6f44; color #ffffff; }
      element "sim_core Rust 加速核" { background #b7472a; color #ffffff; }
      element "环境场层" { background #4a6fa5; color #ffffff; }
      element "观察台" { background #7a5ba9; color #ffffff; }
      element "note" { shape Note; background #fff2cc; color #333333; fontSize 14; }
    }
  }
}
