"""C4a-1「热机本体」粗网格大气 —— Step 1 = 网格 / 节拍 / 守恒上下采样 / 快照载体。

口径来源（唯一权威 = 裁定件，不凭记忆）
--------------------------------------
* `内部汇报/话事人传达-20261016-C4a-1热机本体开工裁定.md`（v26）：§二 物理链与方程、
  「网格 / 频率 / 预算」、Step 1–3 分解；§三 验收判据；§四 21 条自查清单；§五 不做。
* `内部技术文档/技术-动态天气路线评估与选型-20261014.md`：§2.1-②（EBM 参数与 +14 ℃ 锚）、
  §2.2-②（数值方法五选一 ⇒ **通量形式有限体积**）、§2.2-④（CFL 与子步解耦 = split-explicit）、
  §6（C4a-1 行：120×240；每 10t × 3 子步；均摊 +2.0%、含扩散 +2.8%）。

本批物理链（v26 §二，因果方向不许颠倒）
--------------------------------------
    S(φ) 年平短波 → 地面吸收 S(1−α) → T_surf（EBM：C·∂T/∂t = S(1−α) − (A+BT) + D∇²T，
    隐式扩散）→ 单层大气（地面以长波/感热加热大气）→ **质量 proxy + 静力平衡** → P
    → 压力梯度 −∇P/ρ + Rayleigh 阻尼（**无科氏力**）→ 风 u → 水平散度 ∇·u = 垂直运动代理
    🔴 本批风**不**反过来显式平流热量（热量输运仍走 D∇²T 的统计代表；显式 −∇·(vT) 属 C4a-2）。
    🔴 **C4a-2 Step A** 挂在链尾、方向**只朝下**：`T,P,u` → 蒸发势/迎风平流 → `qv`/`bucket`/
    账本，水汽**不回写**热机链任何一项（潜热 = T3 走 A 支 = 本批不接）⇒ 开/关两档的
    `T/P/u` 逐位相同（判据 m14）。

Step 1 交付范围（v26 §二 Step 1 明文：**先不接物理**）
----------------------------------------------------
1. **粗网格**：主 480×960 ÷ `coarse_factor`(=4) ⇒ **120×240**。因子必须**同时整除**
   rows 与 cols（精确分块是"互逆守恒采样对"的前提）⇒ 不整除 = 构造期 **fail-loud**，
   不静默回落（与 `smell_field._largest_divisor_le` 的"回落 + 告警"**故意不同**：
   气味场的粗网格只是**近似远场**，大气的粗网格是**计算域本身**，非精确分块下
   "守恒"与"分辨率"不能同时谈）。2× 粗 / 全分辨率 **已否决，不重测**（v26 §二）。
2. **节拍**：每 `every_ticks`(=10) 个主 tick 推进一次，每次 `substeps`(=3) 个 Δt=`dt_s`
   子步（split-explicit，子步与主 tick 解耦）。🔴 相位确定：引擎 `step()` **先**
   `_tick += 1` 再跑整套流程 ⇒ 触发条件 `tick % every_ticks == 0` 使**首推落在 tick 10**，
   且"推进次数 = ⌊tick/10⌋"唯一可校核（v26 §四-18：子步相位必须确定/可重建）。
3. **上下采样**（v26 §四-16：必须**保守恒**，简单线性插值会丢能量/质量）：
   * `coarsen` = **面积权分块均值**，且粗格面积 `area_c` **定义为块内细格面积之和**
     ⇒ `total_fine(f) == total_coarse(coarsen(f))` **到浮点舍入**（相对残差实测进 probe/汇报）；
   * `refine` = **分片常数**（有限体积"零阶重构"）⇒ 逐块积分严格守恒、**不产生新极值**；
   * 两者互逆：`coarsen(refine(F)) ≈ F`（求和次序可差 1–2 ULP ⇒ 判据用 1e-12 门限，不吹"逐位"）。
4. **载体**：`t_surf`（粗网格地表温度，float64 (rows_c, cols_c) 展平存储）。
   🔴 Step 1 **初值与演化都属 Step 2**（EBM 初值 = 现 insolation 年平剖面）⇒ 本步
   `advance()` 只推进节拍、**不动任何数值** ⇒ "骨架不漂移"是可证伪判据：
   任意手工置入的场，跑 N tick 后**逐位不变**（`tests/test_c4a1_atmosphere.py` 钉死）。
5. **快照**（v26 §四-18）：`state()` / `restore()` 走 `smell_field` 先例 ——
   **关档无键**（旧快照形状逐位不变）、开档键追加；`steps` / `last_tick` **一并进档**
   （否则续跑相位漂移 = "续跑 ≠ 连续跑"）。另：本模块**零 RNG** ⇒ 无状态可存也无须重建。

Step 2 交付（2026-10-16 本笔）= EBM 地表温度场
---------------------------------------------
`C·∂T/∂t = S(1−α) − (A+B·T) + D∇²T`（v26 §二，climlab 口径只借方程不搬库）。
* **真源**：S(φ) = `env_field.annual_mean_insolation`（年平、**不**吃瞬时 illumination，
  §四-5）；α/c = `SURFACE_THERMOPHYSICS`（C0b 备料）按 `surface_type` 查表后**面积权粗化**
  ⇒ 与采样对同一把尺；初值 = 现 insolation 年平剖面（`annual_mean_base_temperature`）。
* **算子**：经向**通量式有限体积**（报告 §2.2-② 五选一的选择），关于 `area` **自伴** ⇒
  `Σarea·∇²T = 0` 到机器精度、`Σarea·T·∇²T ≤ 0`（§四-1/2/3 三条一次性解决）；
  极帽质心 = (μ₁−1)/2、μ=±1 处边界通量恒零 ⇒ 极圈不被重复计数（§四-13，C0b 同款坑）。
* **显式而非隐式**：实测扩散数 `μ = dt·Στ/(C·area) ≈ 1.6e-2`（门限 0.5 的 3.3%）⇒ 隐式
  在本参数组**无稳定性收益**，而 s2 实测隐式三对角解单项 ≈ +3.3%/tick > §二 预算 +2.8%；
  故改为**构造期硬校核** `μ ≤ 0.5`（谁把 dt/节拍改进不稳区就炸 = §四-6 的真实意图）。
* **单位**（§四-4 最隐蔽的一条）：`t_surf` 一律 **℃** —— A=210/B=2.0 这组线性化常数本身
  以 ℃ 为零点（v26 自己的锚 (340×0.7−210)/2 = +14 **℃** 只有在 ℃ 下成立；代 K ⇒ OLR≈784
  ⇒ T̄→−273 荒谬）。**状态方程/理想气体只许 `t_kelvin()`**（Step 3 硬约束）。
  另设下界 fail-loud：T ≥ −A/B = −105 ℃（越界 = OLR 变负 = 模型在外推区）。
* **非负/有限**（§四-15）：每子步校核 NaN/Inf、T(K)>0、OLR 下界 ⇒ 抛错，不静默截断。
* **反照率闭合**（`cfg.albedo_closure`，🔴 v26 内部冲突①）：字面"α 按 surface_type"给
  全球均值 **+49.4 ℃**（本世界面积权 α̅=0.092），与 §二/§三 的 **+14 ℃ 锚**（隐含 α_p=0.30）
  互斥 ⇒ 默认 `planetary_offset`（α_eff = α + (α_p − α̅)，保对比度、只闭合收支，实测
  **+14.10 ℃**）；`surface_raw` 档保留 ⇒ 两种口径都可一键复现，冲突交话事人裁定。
* **零模型装置**（§三，做成配置而非改代码）：① `insolation_mode="uniform"`；
  ② `diffusion_on=False`。稳态剖面锚用**线性解装置**（EBM 热惯性 C/B≈242 天，
  而每主 tick 只推进 45 物理秒 ⇒ 收敛要 ~10⁶ tick ⇒ 生态时窗内 T 是**准静态**的）。

Step 3a 交付（2026-10-16 本笔）= 质量 proxy + 静力平衡 → 地面气压
--------------------------------------------------------------
`_diagnose_pressure()`：状态方程 `ρ=P/(R_d·T)`（T **只许** `t_kelvin()` ⇒ §四-4 唯一
换算点在这里落地）+ 静力平衡 `dP/dz=−ρg` 的等温积分（压高公式）+ **层顶气压水平均匀**
的质量 proxy ⇒ `P_s = P_顶·exp(z_b·g/(R_d·T))`：**暖柱地面气压低**（§四-9 要求的符号）。
`P_顶` 由 `AWM(P_s) ≡ 101325 Pa` 反解 ⇒ **全球柱总质量守恒**（温度只能重新分配质量）。
🔴 不写 `P=P₀·T/T₀`（§四-9 点名的错写，方向相反）；`p_surf`/`mass_col` 是 `t_surf` 的
纯函数 ⇒ **不进快照**（§四-18 "确定性重建"分支，`restore()` 里重算）。
Step 3b/3c 依次在此之上接"压力梯度 + Rayleigh 阻尼 → 风"与"∇·u → 垂直运动代理 + 零模型"。

Step 3b 交付（2026-10-16 本笔）= 压力梯度 + Rayleigh 阻尼 → 风（**无科氏力**）
--------------------------------------------------------------------------
`_integrate_wind()`：`∂u/∂t = −(1/ρ)∇_h P − u/τ`（v26 §二 Step 3 原文，🔴 **无科氏力** ⇒
不是地转风，是"摩擦平衡风"：`u_eq = τ·F`，力平衡而非转平衡）。

* **几何尺**（球面、μ=sin φ、a=`_EARTH_RADIUS_M`）：经向 `F_n = −(c_i/(ρ_i·a))·∂_μP`
  （`c=cos φ=√(1−μ²)` ⇒ **极地自然归零**，"北极点没有北"不处理也不炸）；纬向
  `F_e = −(1/(ρ_i·a·c_i))·∂_λP` ⇒ 这里的 `1/c` **就是** v26 §四-14 点名的极圈奇异，
  按该条明示的"先极地滤波、不降全局步"处理：`c_eff = max(c, _POLAR_C_FLOOR)`（只裁
  **纬向力**的分母，经向一律不裁 ⇒ 与 3c 的散度算子不同尺、不影响 `Σarea·div=0`）。
* **梯度格式**：节点梯度 = 相邻**面**梯度的平均（= 中心差，二者在均匀网格逐位同式）
  ⇒ 与 `_laplacian`/3c 通量式散度共用同一套面索引 ⇒ 极行只有一面、接缝经度周期。
* **极行**（§四-13）：极点是"一圈一格"⇒ 该格的两个速度分量取**环带均值**（`u_n` = 穿出
  环带的平均经向流、`u_e` = 绕极平均自转；后者对正压、无科氏的场恒为 0）。逐子步把
  极行的**力**折成均值 ⇒ 速度行恒单值 ⇒ 同一个极点格不会被赋予 240 个互相矛盾的风。
* **τ 选型（v30 ⑤.4 的必答项，不许默认取 1）**：`rayleigh_tau_s = 86400 s`（1 天）=
  **行星边界层摩擦时标**的量级（文献常见 1–3 天；边界层风对气压力的响应即由此控制），
  与本模块唯一的几何尺 `layer_depth_m=1500 m`（同取边界层）同一层 ⇒ 两数**互相咬合**，
  不是两个独立旋钮。由 τ 与 Δt 的显式积分稳定门限 `dt/τ ≤ 0.5` 构造期硬校核
  （正式档实测 `150/86400 = 1.736e-3` = 门限的 **0.35%** ⇒ 远离不稳区）。
  🔴 **"风能否在 10-tick 节拍内跟上压力梯度"的定量回答**：一次推进 = 3 子步 × 150 s
  = **450 s**，只占 τ 的 **0.52%** ⇒ **跟不上**；2000 tick 批 = 600 子步 = 90 000 s
  = **1.04 τ** ⇒ 只达到平衡风 `u_eq=τ·F` 的 **≈65%**（解析：1−e^(−1.04)=0.646；
  s2 正式档实测随 Step 3 汇报给出——3b 定稿的阻尼格式改过一处**先乘后加**的写法，
  旧读数 0.647 是改前采的，不复用）。
  ⇒ 本批的风是**慢响应积分器**，不是平衡风；判据因此用**解析平衡装置**
  （`u/u_eq → 1−e^{−t/τ}`，单测 W 系列）而不是"跑足够长看它稳不稳"。
  ⚠️ 改 `every_ticks`/`dt_s`/`substeps` 任一项都要重述这三段占比（v30 ⑤.4 的"显式声明"）。
* **零模型③**：`damping_on=False` ⇒ `u += dt·F` 无衰减 ⇒ 风随子步**线性发散**
  （反证 Rayleigh 摩擦是本模型**必需项**而非装饰；v26 §三"全部要跑"）。
* **通量式散度（§四-11 随 3b 落地）**：`div_u()` = `(1/a)[∂_μ(c·u_n) + (1/c_eff)·∂_λ u_e]`
  的控制体写法 ⇒ `Σ area·∇·u ≡ 0`（经向项望远镜相消、纬向项逐行周期相消；极外面 `c=0`
  不进出流量）。它是 3c 垂直运动代理 `w = −z_b·∇·u` 的**算子**，本身零状态、不进快照。
* **无 CFL**：本批风**不平流任何场**（热量输运仍走 D∇²T 的统计代表，显式 −∇·(vT) 属
  C4a-2）⇒ `u` 不是被输运量、不参与稳定性约束 ⇒ §四-14 的 CFL 在本批只体现为上面的
  `1/c` 极地滤波与 `dt/τ` 门限。**如实声明，不当"已验 CFL"糊过去。**
  ⚠️ **C4a-2 Step A** 起，风在 `moisture_on=True` 时开始平流**柱水汽 `qv × mass_col`**
  （被输运量 = 柱水，不是 u，也🔴 不是比湿本身，见下节）⇒ 那条链有**自己的**迎风 CFL 硬校核
  （`_moisture_substep` 第 1 步，红线 5）；
  热量的显式平流仍属 **Step B** ⇒ 本条在 C4a-1 口径（`moisture_on=False`）逐字继续成立。
* **快照**：`u_east`/`u_north` 是**积分状态量**（非诊断量）⇒ 进档（引擎键 `atm_u_east` /
  `atm_u_north`）；旧档（Step 3a 及更早）缺键 ⇒ 风从**静止**起算（`smell_Sc` 先例，
  文档口径 = 与"新档从静止起算"同一件事）。

C4a-2 Step A 交付（2026-10-17 本笔）= 水汽平流 + 蒸发约束 + 水量账本（🔴 **纯被动**）
------------------------------------------------------------------------------------
口径来源 = `内部汇报/话事人传达-20261017-C4a1收口裁定与C4a2放行-v35.md` §三 +
`内部汇报/汇报-20261017-C4a2动手前设计三则.md`（设计三则）+ owner 派工 C4a-2 范围。

* **状态量三个**（都进快照；单位口径 = **owner 2026-10-17 修订令**，见下）：
  `qv`（层均**混合比** kg kg⁻¹，🔴 不是柱水汽 ⇒ 柱水 = `qv × mass_col` kg m⁻²）、
  `bucket`（陆面水桶，**每单位陆地面积** kg m⁻² of land ⇒ 整格贡献 = `bucket × (1−wf)`，
  容量 = `W_max` 本身，不乘占比）、`wb`（水量账本 `[E,P,R,Ocean,imbalance,P_pot]`，**mm**）。
  ⚠️ 换算尺：`1 kg m⁻² ≡ 1 mm`（水当量，写法 `÷ρ_w ×1000` ⇒ `_M_TO_MM`）⇒ 收支累积读作 mm，
  通量类年率读数 = 累积 ÷ (`steps × dt_s`) × `_SECONDS_PER_YEAR`，单位 mm/yr。
  静态面 `water_frac`/`p_rate`/`land_dist` 由 C0b 掩码 + `env_field.pr` + 网格定 ⇒
  构造期重建、不进档（同 α/S/C 口径）。
* **被输运量 = 柱水，不是比湿**：平流核 `moisture_advection()` 输运 `qv × mass_col`（kg m⁻²），
  算完再除回 `mass_col` ⇒ 三条好处都是**守恒口径**给的：① `Σarea·∇·(柱水·v) ≡ 0` 仍是
  望远复核线恒等（`mass_col` 是诊断量、不被平流 ⇒ 若改输运比湿，恒等式立刻不成立，账本就要引入
  一个"随柱质量走"的假源项）；② 保正凸组合直接给**水汽质量**非负 ⇒ 比湿随也非负
  （`mass_col > 0` 由 `_diagnose_pressure` 硬门）；③ 蒸发/降水都作用在柱水上，除回比湿只在
  写回状态时做一次（降水的取走量 ⇒ `dq = precip / mass_col`，precip 已含 `dt`）。
* **因果方向 = 单向**：`T,P,u, mass_col` → 蒸发势/平流 → `qv/bucket/wb`，**没有任何一项
  反过来写回热机链**。⇒ 可证伪判据（单测 m18）：`moisture_on` 开/关两档在任意 tick 的
  `t_surf`/`p_surf`/`u_east`/`u_north` **逐位相同**。潜热反馈（凝结放热回 EBM）= **T3 走
  A 支 = 本批明确不接**；云/辐射属 C4a-3（owner v35 §三"云不接线"同族）。
* **子格 tiling**（粗网格一格 = 一块真实地表）：`wf = coarsen(非陆地)` ∈[0,1] 是**水面占比**，
  蒸发按占比分两本账（水面不限水 / 陆面受水桶约束），降水按占比分两头（水面直入海洋库 /
  陆面入桶）⇒ 不需要第二张网格，也没有隐藏的全局混合参数。五个量一律按**整格面积**计。
* **平流 = 迎风一阶通量式有限体积**（`moisture_advection()`，与 `div_u()` 共用同一套面索引
  与分母 ⇒ 一把尺）：保正性来自"更新是邻格的**凸组合**"，其条件 `ν=Δt·Σ出流系数 ≤ 1`
  每子步硬校核（按高纬最小格距；分母含 `c_eff` ⇒ 极带的 `1/c` 放大算进 ν，不是算完再补）。
  越界 🔴 **不许全局限小步**（派工原文）⇒ 已有的极地滤波就是该做的处置，越界即炸、不 clip。
  ⚠️ **T1 走 A 支** = 一阶迎风的数值扩散 `K_num ≈ |v|·Δx/2 ≈ 1.5e6 m²/s` **明写不藏**；
  Step B 的显式热平流会与 `D∇²T` **重复计算**同一份次网格混和 ⇒ 挂账，届时一并呈报。
* **蒸发约束（Manabe 水桶）**：`E_pot = C_E·ρ_a·|v|·max(0, q_sat − qv)`（空气动力学公式，
  系数 `C_E=1.32e-3` 取大气边界层常用百慕大值；`q_sat` 与状态量 `qv` **同为混合比**口径 ⇒
  直接相减，不再除 `mass_col`）
  ⇒ `E = Δt·E_pot·wf + (1−wf)·min(Δt·E_pot, bucket)`：第一项是**水面**（无限库），第二项是
  **陆面**——需求先按"每单位陆地面积"与桶存量取 `min`，再乘陆面占比回到整格账。
  汇侧 `P = min(柱水, p_rate·Δt)`：取走量以**当前柱水汽**（= `qv·mass_col`）为上限 ⇒ 干格不
  凭空下雨，账本记的是**实际** P（`precip_realized_frac` 就是这条截断的读数）；取走量除回
  `mass_col` 才是比湿的减少。降水的**空间型**来自 `env_field.pr`（只存储层），**总量**钉在
  **T4 走 A 支 = P̄=990 mm/yr** 锚上（`AWM(p_rate) ≡ P̄` 由构造成立）。
* **径流 T2 走 A 支**：`R = (1−wf)·max(0, bucket − W_max)`（🔴 容量 `W_max` 是**每单位陆地
  面积**的界 ⇒ 式子里没有占比），溢出**瞬达海洋库**（不建汇流延迟/河网时间尺度）；
  海洋 = 边界条件而非状态量（`wb[3]` 只记净入海水量）。
* **闭合恒等式**（判据②）拆两段硬校核，容差都是 `_WB_TOL_REL=1e-12`（相对）：
  输运段 `Σarea·∇·(柱水·v) ≡ 0`（望远镜相消）+ 源汇段
  `ΔAWM(柱水)+ΔAWM(bucket·(1−wf))+AWM(ΔOcean) ≡ 0`（靠 `wf+land ≡ 1` 逐项对消）。超容差即炸。
  ⚠️ 桶那一项**必须带 `(1−wf)` 权**：状态量是"每单位陆地面积"的深，全球储量是它的面积×占比
  加和 ⇒ 直接 `AWM(bucket)` 会把 3% 陆格的湿度当成全球量（读数键 `bucket_cell_awm_mm` 与
  `bucket_awm_land_mm` 因此两条并存、各管各的问法）。
* **零模型④⑤**（配置而非改代码，与①②③同族）：`moisture_advect_on=False` = 关平流
  （只剩局地球水 ⇒ 判据①的"海→陆通量"必须塌到 0）；`water_limit_on=False` = 蒸发不受陆面
  水量约束（实现 = **允许桶透支为负**，账本仍逐位闭合 ⇒ `bucket_min_mm<0` 就是该零模型的
  反证读数，故"桶非负"的 fail-loud 只在开约束时生效）。
* 🔴 **零新增 RNG**：整条链只有乘除与 `min/max`，一次随机数都不消费（红线 2）。
* **快照**（**T5 走 A 支 = 不升 `SNAPSHOT_VERSION`**）：三个状态量走 `atm_qv`/`atm_bucket`/
  `atm_wb` 新键；🔴 关档**不写键** ⇒ 关档/旧档快照形状逐位不变；开档读到缺键（Step A 之前的
  档）⇒ 干起算/空桶/零账本，形状不符 fail-loud。⚠️ 这条回退只为"不炸"，**不等于**
  "旧档开水汽续跑 ≡ 从头开档跑"（前者前若干 tick 是干球）⇒ 判据批次一律新建，不靠续跑。

🔴 默认关与逐位等价（v26 §四-17，C7 回滚点）
--------------------------------------------
`AtmosphereConfig.enabled=False`（默认）⇒ 引擎**不构造**本模块，tick 路径只多一次
`is not None` 判断 ⇒ 主 tick 与现 main **逐位等价**（对拍 = `tools/env_field_digest.py`
同平台前后 digest；本批**未动该工具图层口径** ⇒ 无需"末尾追加"自证）。

极点口径（C0b 同款坑，v26 §四-13：极点整圈邻域不能重复计数或漏计）
----------------------------------------------------------------
`SphereWorld` 里极点是**整行坍缩成一格**：`rc_to_flat(0, 任意 col) → 0` ⇒ 引擎/生物只写
规范槽 col 0，**其余槽是未定义历史值**（`env_field` 的 priority-flood 已踩过：非规范极槽
在邻居图上孤立）。本模块因此三条：
① `coarsen` 先把主网格极点行读成"规范槽值广播"再聚合 ⇒ 不吞垃圾；
② 粗网格极点行 = **一圈一格** ⇒ 块 = 该纬度带**全列**的面积权均值（不是逐列分块），
   输出对经度恒均匀（否则同一个极点格会被赋予 240 个不同值 = 重复计数）；
③ `refine` 的极带取粗极点行**规范槽**值广播 ⇒ 主网格极带单值（与 `rc_to_flat` 同义）。
面积记账在两套写法下恒等：`cols_粗 × area_粗极槽 = cols_细 × Σ_块 w` ⇒ ②/③ 不破坏守恒。
"""
from __future__ import annotations

import warnings
from collections import deque

import numpy as np
from numpy.typing import NDArray

from world.sphere_world import SphereWorld

# ---------------------------------------------------------------------------
# EBM 常数（v26 §二 钉死值 —— 🔴 改数须有话事人裁定，不放 config 以防"调参硬凑"）
# ---------------------------------------------------------------------------
#: 出射长波线性化常数项 A（W/m²）：`OLR = A + B·T`，North/climlab 档（v26 §二 明文 A=210）。
_EBM_A_WM2 = 210.0
#: 出射长波线性化斜率 B（W/(m²·K)）（v26 §二 明文 B=2.0）。
_EBM_B_WM2K = 2.0
#: 经向热扩散系数 D（W/(m²·K)）—— 参数化"风 + 洋流送热"（v26 §二 明文 D=0.555）。
#: 🔴 本批**只**做经向扩散（v26 §二 措辞即"经向扩散"；纬向显式扩散在极带的不稳定数
#: 随 1/cos²φ 爆炸 ⇒ 报告 §2.2-④ 的 CFL 警告；纬向各向同性版本要等隐式解 + 极区滤波）。
_EBM_D_WM2K = 0.555
#: 热容量 C = ρ_w·c_w·h（10 m 水当量 ⇒ 4.182×10⁷ J/(m²·K)；v26 记作 ≈4.2×10⁷，差 0.4%
#: 只改时间尺度、不改稳态锚）。海陆按 C0b 的比热 c 线性缩放：C_i = C_ref·(c_i/4182)。
_RHO_WATER = 1000.0
#: B4.1 地球自转角速度（rad/s）——**物理常数**，不是旋钮：Ω = 2π/86400。
#: 与 `rotation_period` 解耦的依据：`SimConfig.__post_init__` 的汇率断言保证
#: `rotation_period(tick) × tick_seconds(s/tick) = 86400 s`，故一天恒为 86400 秒。
_OMEGA_DAY = 2.0 * np.pi / 86400.0
_C_WATER = 4182.0
_MIX_LAYER_M = 10.0
_C_REF = _RHO_WATER * _C_WATER * _MIX_LAYER_M
#: 行星反照率闭合值 α_p = 0.30 —— **不是**新调的参数，而是 v26 §二 自己那条解析锚
#: `T̄=(S̄(1−α)−A)/B=(340×0.7−210)/2` 里隐含的 (1−α)=0.7。⚠️ 与 `env_field._PLANETARY_ALBEDO`
#: =0.29（现代地球观测值，用在 Stefan-Boltzmann 有效辐射剖面）**不是同一个数**：
#: 用 0.29 闭合 ⇒ 全球均值 +15.8 ℃，用 0.30 ⇒ +14.10 ℃ = 锚。差 3.4 W/m² ⇒ ΔT̄ 1.7 ℃。
_ALBEDO_ANCHOR = 0.30
#: ℃ ↔ K 边界（v26 §四-4：状态方程/理想气体只许开尔文；本模块 `t_surf` 一律 ℃）
_KELVIN0 = 273.15
#: 长波线性化的物理有效下界：OLR=A+B·T≥0 ⇒ T ≥ −A/B = −105 ℃（越界 = 模型已在外推区，
#: 属"违背自然规律"级异常 ⇒ fail-loud，不静默截断；v26 §四-15 同族）。
_OLR_T_FLOOR_C = -_EBM_A_WM2 / _EBM_B_WM2K
#: 显式扩散稳定门限（Forward-Euler 通量式：dt·Στ/(w·C) ≤ 此值）；构造期实测校核。
_DIFF_STABILITY_MAX = 0.5

# ---------------------------------------------------------------------------
# Step 3 物理常数（v26 §二「单层大气 → 气压 → 风」；🔴 标准物理常数，不入 config）
# ---------------------------------------------------------------------------
#: 重力加速度（m/s²，标准值）。
_G_ACCEL = 9.80665
#: 干空气气体常数 R_d（J/(kg·K)）—— v26 §二 因果链里大气对短波近透明、由地面加热，
#: 本批**无水汽** ⇒ 一律按干空气处理（虚拟温度修正在 C4a-2 才有对象）。
_R_DRY_AIR = 287.05
#: 全球面积权平均地面气压 P₀（Pa）= **大气总质量锚**：`AWM(P_s) ≡ P₀/g` 逐位固定。
#: 物理内容 = "全球大气总质量不变" ⇒ 温度剖面只能在柱间**重新分配**质量，不能造质量
#: （v26 §四-9/11 的质量侧对应物；也是 §四-15「质量>0」的全局形式）。
_P_REF_PAS = 101325.0
#: 压高公式理想化有效域：柱压比 `exp(Δ(z_b·g/R_d/T)) − 1` 的上限。越界 = 等温薄层
#: 假设已在外推区（真实大气的地面气压对比被质量补偿压到 ≪ 此量级）⇒ **fail-loud**，
#: 不静默给出一个"量级荒谬"的气压场（v26 §四-15 同族：宁炸不猜）。
_HYPSO_RATIO_MAX = 0.5
#: 地球半径（m）—— Step 3 引入的**唯一长度尺度**（压力梯度 `∇P = (1/a)∂_μP` 与散度算子
#: 的几何尺）。🔴 Step 2 的 `D` 用**角域**拉普拉斯（已按 l=2 模校验 1.7e-4），不含 a ⇒
#: 两者不是同一把尺，别拿 a 去"修正"Step 2 的 D（那会把已验的离散一致性推翻）。
_EARTH_RADIUS_M = 6.371e6
#: 纬向力 `1/cos φ` 的**极地滤波**下限（v26 §四-14 明示"先极地滤波、不降全局步"）。
#: s2 档（120×240）**实测**：`c<0.1` 恰为南北各 4 行（|φ|≥84.75°，合计面积占比 **0.55%**），
#: 这八行的未滤波放大倍率 `1/c` = **10.8 / 15.0 / 24.2 / 54.0** ⇒ 极帽格的纬向边长只有
#: 经向的 1/54，同一条 `∂_λP` 在那里被放大 54 倍（这就是 §四-14 点名的极圈奇异）。
#: 🔴 只裁**纬向力**的分母；经向力带 `c` 因子本就归零，裁它 = 改物理。
_POLAR_C_FLOOR = 0.1
#: Rayleigh 阻尼显式积分（Forward Euler `u ← u(1−dt/τ) + dt·F`）的稳定门限：
#: 纯阻尼项衰减因子 `|1−dt/τ| ≤ 1` ⇒ `dt/τ ≤ 1` 才不反号震荡；取 **0.5** 留一倍余量
#: （越接近 1 越出现"风每子步换向"的非物理锯齿）。构造期实测校核（同 μ 扩散的门限口径）。
_DAMP_STABILITY_MAX = 0.5
#: **读数窗口**（🔴 不参与任何物理项 ⇒ 不是旋钮，只是"赤道带/副热带带"的固定问法）：
#: v26 §三 的单胞环流判据要分开回答"赤道是否辐合、副热带是否辐散"，窗口宽度变了读数
#: 会变，故把区间钉死在模块里、随汇报给出，不许事后挑对自己有利的带界。
#: ⚠️ `[实测]` 代价必须一并给出（s2，两条时间轴都量过）：辐散支起始 |φ| =
#: **EBM 稳态装置 ≈39.8°**、**批内瞬态 ≈53.2°** ⇒ 25–45° 带均值在**两种装置下都是辐合上升**
#: （稳态 `w_sub≈+7.9e-4`、批内 `+4.8e-3` m/s）——**"副热带辐散"的字面在本批结构下取不到**。
#: 机理：无科氏力 ⇒ 没有角动量约束把下沉区压回 ~30°，暖赤道低压的**入流**一直铺到辐合带边缘，
#: 其余（到极地）整体辐散下沉；真实"副热带干旱带下沉支"属 C4a-4（自转/科氏）。
#: 带界**不迁就读数**、算法不为字面改写 ⇒ 冲突按 A/B/C 交话事人（汇报 §六 冲突①）。
#: ✅ **owner 裁定 A（v35 §一-1）**：判据 8 后半改写为**单胞结构判据**（低纬辐合上升 +
#: 辐合带外整体下沉 + 南北对称 + 形状不漂）；副热带下沉带留 **C4a-4** 加科氏后验；
#: B（挪窗口）否决 —— "带界是问法不是旋钮"。带界因此**保持原值不动**。
_BAND_EQ_MAX_DEG = 15.0
_BAND_SUB_MIN_DEG = 25.0
_BAND_SUB_MAX_DEG = 45.0
#: **赤道无风带**读数窗口（同族纪律：🔴 只是固定问法，不参与任何物理项）。
#: owner v35 §二-2 把「赤道无风带」写进判据字面 ⇒ 带界必须**先钉死后读数**，不许事后
#: 挑对自己有利的宽度。取气候学 Doldrums/信风带的常规量级：无风带 = |φ|≤5°、
#: 对照带（信风侧）= 5–20°。判据 = 两者的**面积权速率比** < 1（"比信风侧静"）。
#: ⚠️ 本批无科氏力 ⇒ 这条带只是**经向风为零**的对称结果 + 经度起伏的弱风侧；
#: 无风带"长胖"（宽度随科氏参数变）属 **P-coriolis / C4a-4** 的读法，此处只留口径。
_BAND_CALM_MAX_DEG = 5.0
_BAND_TRADE_MAX_DEG = 20.0

# ---------------------------------------------------------------------------
# C4a-2 Step A 水汽 / 水量常数（🔴 全为文献值或锚值，**不入 config** ⇒ 防"调参硬凑"，
# 与 EBM 四常数、`R_d/g/P₀` 同族纪律；owner v35 §四 T1–T5 未另裁 ⇒ 一律走 A 支并在此标注）
# ---------------------------------------------------------------------------
#: Magnus 三常数（Tetens；T 用 **℃**，输出 hPa）。🔴 必须与 `env_field._MAGNUS_*` **同值**
#: —— 本模块的饱和水汽压与 ENV 的气候态降水 proxy 是**同一条曲线**，双写漂移 = 蒸发与降水
#: 不在同一物理上 ⇒ 单测 m01 逐值钉死（加基因五步曲的"Python↔Rust 双写校验"同族做法）。
_MAGNUS_A = 17.27
_MAGNUS_B = 237.3
_MAGNUS_C_HPA = 6.11
_HPA_TO_PA = 100.0
#: ε = R_d/R_v = 0.622 —— 混合比口径 `q = ε·e/(p−e)`。🔴 状态量 `qv` **就是这个数**（kg/kg，
#: 层均混合比 ⇒ owner 2026-10-17 修订）⇒ 与 `q_sat` **直接相减**才是空气动力学公式里的
#: "饱和 deficit"；一个按比湿、一个按混合比 ⇒ deficit 系统性偏大 ~0.4% 且随 T 变，
#: 不是可以"以后补"的常数偏差（单测 m04 钉 `qv` 的混合比口径）。
_EPS_WS = 0.622
#: 空气动力学（bulk）交换系数 C_E —— Large & Pond (1981) 中性层结**水汽**档 1.32×10⁻³。
#: 🔴 本批唯一新增的"率"常数；它是通量公式的量纲闭合系数（不是拟合旋钮：改它 = 改全球 E，
#: 而 E 由账本钉在 P̄ 附近 ⇒ 实测滞留时间随它线性变，读数随汇报给出）。
_C_E = 1.32e-3
#: 秒/年（990 mm yr⁻¹ → kg m⁻² s⁻¹ 的换算尺，与"mm/yr"读数的换算共用一个常数）。
_SECONDS_PER_YEAR = 3.15576e7
#: **水量锚 P̄ = 990 mm/yr**（全球平均年降水，地球观测值；T4 走 A 支 = 用锚而不用自由参数）。
#: 换算：990 kg m⁻² yr⁻¹ ÷ 3.15576×10⁷ s = 3.137×10⁻⁵ kg m⁻² s⁻¹。
#: 口径：`p_rate = P̄·coarsen(pr)/AWM(pr)` ⇒ **空间型**取自 ENV 的气候态降水 proxy（Magnus
#: × 纬度带 × 地形抬升），**全球总量**由本锚钉死 ⇒ 既不是"平地降水"，也不是可调旋钮。
_P_ANCHOR_KG_M2_S = 990.0 / _SECONDS_PER_YEAR
#: 陆面水桶容量 W_max = 150 kg m⁻²（= 150 mm 可蒸发水；Manabe & Wetherald 1967 的 S 层厚度
#: 常规档，本世界取该族文献最常见值）。🔴 owner 2026-10-17 修订：状态量 `bucket` 是**每单位
#: 陆地面积**的水量（kg m⁻² of land）⇒ 容量就是 `W_max` 本身（`bucket ≤ W_max`），**不再**乘
#: 陆面占比；整格记账时才乘 `land = 1−water_frac`。它是**陆面唯一**的水量上界，溢出即径流入海
#: （T2 走 A 支 = 瞬达，不建汇流延迟；河流/地下库属 C4a-3+）。
_W_MAX_KG_M2 = 150.0
#: 水量深度换算：`1 kg m⁻² ≡ 1 mm`（水当量）。写法 = `flux×dt` [kg/m²] `÷ ρ_w` [→m] `× 1000`
#: [→mm] ⇒ **两个因子互为倒数**（owner 2026-10-17 修订令里的 `flux × dt / SPY × 1000` 按此解：
#: 分母取水密度而非秒/年，否则账本尺被缩 3.17e-5 倍、与 P̄=990 mm/yr 锚对不上 ⇒ 字面歧义
#: 已列入首次汇报的待裁段，代码走"与锚自洽"的这一支）。数值上是恒等变换，但**明写在式子里**
#: ⇒ `wb` 六个槽的单位就是 mm（收支类读数再 ÷ 物理秒 × `_SECONDS_PER_YEAR` 换成 mm/yr）。
_M_TO_MM = 1000.0
#: 上式的**净因子**（`1000 mm/m ÷ 1000 kg/m³` ⇒ **恰为 1.0**，浮点精确 ⇒ 乘它不改任何数值）。
#: 留在式子里只为把"kg/m² 就是 mm 水当量"这条换算**显形**（藏在注释里 = 下一位改代码的人
#: 会当成"这里漏了个单位换算"而顺手补一次 ⇒ 账本尺翻 1000 倍，正是本轮修订要防的那类错）。
_MM_PER_KG_M2 = _M_TO_MM / _RHO_WATER
#: 迎风一阶平流的 **CFL** 门限（🔴 红线 5：按**高纬最小格距**校核，分母用 `c_eff` 与 `Δμ̃`）。
#: 保正凸组合条件 = `ν_i = Δt·Σ_面 出流系数 ≤ 1` ⇒ 越界 **fail-loud**（不 clip、不降步长）；
#: `(0.5, 1]` 只**一次性预警**（留一手给话事人，不静默）。正式档实测随汇报给出。
#: ⚠️ 单位修订后 ν 的**定义不变**（只由风与网格定、与被输运量的量纲无关）⇒ 门限沿用同两条。
_CFL_STABILITY_MAX = 1.0
_CFL_WARN_MAX = 0.5
#: 水量账本相对残差容差（红线 4"账本残差超容差 ⇒ 炸"）。它**同时**管两段闭合，两段的
#: 噪声底不同 ⇒ 分开校核、共用一条判据带（混在一起会让容差被大的那段决定）：
#: ① 输运段 `Σarea·∇·(qv·mass_col·v) ≡ 0`（通量式散度的望远镜相消，`_moisture_substep` 第 1 步；
#: 🔴 被输运的是**柱水** = 比湿 × 柱质量，不是比湿本身）；
#: ② 源汇段 `bal`（基准取在**平流之后** ⇒ E/P/R/Ocean 靠 `wf+land≡1` 逐项对消，代数是
#: **恒等**的，唯一数值来源 = 求和次序噪声，见 `_moisture_substep` 注）。
#: 两段都应在 1e-16 相对量级 ⇒ 放到 1e-12 = "真漏了东西"与"舍入"之间的判据带。
_WB_TOL_REL = 1e-12
#: 内陆递减的距离带（**粗网格格数**，BFS 离水距；1 = 海岸带）。同样只是**问法**，不参与物理；
#: probe 的 `qcol_dist{d}_mm` 键按本元组生成 ⇒ 改带界即改键名，不留"代码与文档两套带"。
_DIST_BANDS = (1, 2, 3)
_DIST_BAND_FAR = 4


class Atmosphere:
    """C4a-1 粗网格大气（一次构造、随 tick 推进的动态场；**由引擎按 `AtmosphereConfig` 构造**）。

    实例属性（`__slots__` 全部字段）
    ------------------------------
    cfg : `AtmosphereConfig`（参数契约）。
    world_fine : 主网格 `SphereWorld`（采样源，只读引用）。
    world : 粗网格 `SphereWorld`（rows_c×cols_c；提供 `_lat` 与极点行号等几何真源）。
    factor, rows_f, cols_f, rows_c, cols_c, n_c : 网格尺寸（`rows_c = rows_f // factor`）。
    every_ticks, substeps, dt_s : 节拍（v26 §二「每 10 tick × 3 子步」）。
    steps : int —— 已完成的**子步**总数（= 推进次数 × substeps；进快照）。
    last_tick : int —— 最近一次推进所在主 tick（-1 = 从未推进；进快照）。
    t_surf : NDArray[float64] (n_c,) —— 粗网格地表温度（°C）。Step 2 起 = EBM 状态量。
    area : NDArray[float64] (rows_c,) —— **每粗格槽**相对面积 = factor × Σ_块 w_细
        （定义即守恒的来源；所有全球积分/面积权均值/拉普拉斯的权重都用它 ⇒ **一把尺**）。
    alpha_surface, alpha_eff : NDArray[float64] (n_c,) —— 地表反照率（C0b 查表 + 面积权粗化）
        与闭合后的有效反照率（`albedo_closure`；`surface_raw` 档两者逐位相同）。
    s_in : NDArray[float64] (n_c,) —— 年平入射 S(φ)（面积权粗化；`insolation_mode="uniform"`
        ⇒ 全球均值常数 = 零模型①）。
    absorbed : NDArray[float64] (n_c,) —— S·(1−α_eff)（W/m²，Step 2 的唯一加热项）。
    heat_cap : NDArray[float64] (n_c,) —— 粗格热容量 C（J/(m²·K)，陆地 ≈1.48×10⁷ / 水 4.182×10⁷）。
    mu_edge, mu_ctr : NDArray[float64] (rows_c+1,) / (rows_c,) —— μ=sin φ 的格边界与**面积质心**
        （极帽质心 = (μ_1−1)/2，非行心 ⇒ 极点格不被重复计数，v26 §四-13）。
    tau : NDArray[float64] (rows_c−1,) —— 面传导率 D·(1−μ²)_面/(μ̄_{i+1}−μ̄_i)·(factor/Δφ)
        （对称 ⇒ 关于 `area` 自伴；因子 factor/Δφ 将 μ 尺换算成本模块的 cos 尺）。
    p_surf : NDArray[float64] (n_c,) —— 地面气压（Pa，Step 3a 诊断量，**不进快照**）。
    mass_col : NDArray[float64] (n_c,) —— 活动层柱质量 (P_s−P_顶)/g（kg/m²，同上不进档）。
    _p_z : float —— 层顶（非辐散面）气压 = 使 `AWM(P_s)≡P₀` 的均匀常数（Pa）。
    _hypso_k : float —— 压高公式指数分子 `z_b·g/R_d`（K）。
    u_east, u_north : NDArray[float64] (n_c,) —— 粗网格风的两分量（m/s，**积分状态量 ⇒ 进
        快照**；极行恒单值）。Step 3b。
    _c_ctr : NDArray[float64] (rows_c,) —— 面积质心处 `cos φ = √(1−μ_ctr²)`（经向力乘它、
        纬向力除它 ⇒ 两者的极地行为相反，见模块头 Step 3b）。
    _dmu_ctr : NDArray[float64] (rows_c−1,) —— 相邻质心 μ 距（经向面梯度的分母）。
    _dlam_c : float —— 粗网格经度格距（rad）= 2π/cols_c（纬向面梯度的分母）。
    _damp_k : float —— `dt_s/rayleigh_tau_s`（每个子步的阻尼衰减率，构造期校核 ≤0.5）。
    _mu_damp : float —— 同 `_damp_k`，存为**读数**（与 `_mu_diff` 同一呈现口径，进 probe）。
    _scale_mu2area : float —— `factor/Δφ_细`，把"一把尺"的 `area` 换成 μ 尺的常数
        （散度的格厚 `Δμ̃ = area/scale` ⇒ `Σ area·∇·u` 的系数对每行恒等 ⇒ 望远镜相消）。

    🔴 以下均为 **C4a-2 Step A 开档专属**（`moisture_on=False` ⇒ 不分配、不推进 ⇒ 逐位等价）
    qv : NDArray[float64] (n_c,) —— 粗网格**层均混合比**（kg 水汽 / kg 干空气，无量纲；
        🔴 owner 2026-10-17 修订：不是柱水汽。柱水 = `qv × mass_col` kg m⁻² ≡ mm，
        由 `column_water()` 给）；积分状态量 ⇒ 进快照。
    bucket : NDArray[float64] (n_c,) —— 陆面**水桶**（kg m⁻²，按**每单位陆地面积**计；
        上界 `W_max`（不乘占比），整格储水 = `bucket × (1−water_frac)`，溢出即径流入海）；
        积分状态量 ⇒ 进快照。
    wb : NDArray[float64] (6,) —— **水量账本**累积 `[E, P, R, ocean, imbalance, P_pot]`
        （**mm**，一律整格面积权；末位 `P_pot` = 潜在降水，只做 `precip_realized_frac` 的分母、
        **不进闭合恒等式**）；积分状态量 ⇒ 进快照（判据②的证据必须能续跑后复核）。
    water_frac : NDArray[float64] (n_c,) —— 粗格**水面占比** `wf = coarsen(非陆地)` ∈[0,1]
        （子格 tiling：`wf` 部分蒸发不限水、`1−wf` 部分受水桶约束）。
    p_rate : NDArray[float64] (n_c,) —— 降水率（kg m⁻² s⁻¹）= `P̄·coarsen(pr)/AWM(pr)`
        （空间型 = ENV 气候态降水 proxy、总量 = 990 mm/yr 锚 ⇒ 静态、不进快照）。
    land_dist : NDArray[int16] (n_c,) —— **离水距离**（BFS 粗网格格数；水面格 = 0，内陆 ≥1）；
        只服务"内陆递减"读数（🔴 不参与任何物理项），静态。
    _cfl_max : float —— 迄今**最大** CFL 数（逐子步校核的读数，进 probe）。
    _cfl_warned : bool —— `(0.5,1]` 预警是否已发过（只发一次，不刷日志）。
    _adv_res_rel : float —— 迄今**最大**平流面积权和相对残差（输运段硬校核的读数，进 probe）。
    """

    __slots__ = (
        "cfg", "world_fine", "world",
        "factor", "rows_f", "cols_f", "rows_c", "cols_c", "n_c",
        "every_ticks", "substeps", "dt_s", "steps", "last_tick",
        "t_surf", "area", "_w_fine", "_w_block", "_den", "_lat_c",
        "alpha_surface", "alpha_eff", "s_in", "absorbed", "heat_cap",
        "mu_edge", "mu_ctr", "tau", "albedo_offset", "_mu_diff",
        "p_surf", "mass_col", "_p_z", "_hypso_k",
        "u_east", "u_north", "_c_ctr", "_dmu_ctr", "_dlam_c", "_damp_k", "_mu_damp",
        "_scale_mu2area",
        # ---- C4a-2 Step A（默认关 ⇒ 不分配 ⇒ 主 tick 与 C4a-1 逐位等价）----
        "moisture_on", "qv", "bucket", "wb", "water_frac", "p_rate", "land_dist",
        "_cfl_max", "_cfl_warned", "_adv_res_rel",
        # ---- B4.1 科氏（默认关 ⇒ 不分配 f_cor）----
        "coriolis_on", "_f_cor",
    )

    def __init__(self, world_fine: SphereWorld, cfg, surface_type=None,
                 pr_fine=None) -> None:
        """构造粗网格大气（网格/节拍/采样权重 + Step 2 的 EBM 驱动场；**零 RNG**）。

        parameters
        ----------
        world_fine : SphereWorld
            主网格（正式档 s2 = 480×960）。
        cfg : AtmosphereConfig
            参数契约。⚠️ `enabled=False` 时**不应构造本模块**（引擎侧已拦）；这里再断言，
            防"构造了却全关"的静默态（`SmellField.__init__` 同族自查）。
        surface_type : NDArray[uint8] (rows_f*cols_f,) | None
            C0b 的地表类型层（`EnvField.surface_type`）。🔴 Step 2 起**必需**：α 与热容量 C
            都按 `SURFACE_THERMOPHYSICS` 查表（v26 §二 Step 2 明文"α/c 按 surface_type 接
            C0b 备料"）⇒ None = 上层没开 `env_field.c0b_enabled` ⇒ **构造期 fail-loud**，
            不静默退化成"全球一种地表"。
        pr_fine : NDArray[float64] (rows_f*cols_f,) | None
            C4a-2 Step A 的**气候态降水 proxy**（`EnvField.pr`，只存储层）。🔴 仅在
            `moisture_on=True` 时**必需**：降水的**空间型**由它给出（全球总量钉在 990 mm/yr
            锚上）⇒ 水汽开档而拿不到 `pr` = 没有降水型 ⇒ 构造期炸，不静默改成"均匀降水"
            （那会把 C1/地形抬升的经向结构抹平，判据①的"内陆递减"就失去意义）。
        """
        assert bool(cfg.enabled), (
            "enabled=False 时不应构造 Atmosphere —— 引擎侧必须拦在构造前"
            "（否则 = 「构造了却全关」的静默态，B3 家族）")
        s = int(cfg.coarse_factor)
        rows_f, cols_f = int(world_fine.rows), int(world_fine.cols)
        if s < 1 or rows_f % s or cols_f % s:
            raise ValueError(
                f"coarse_factor={s} 不能同时整除主网格 {rows_f}×{cols_f}"
                "（精确分块是 coarsen/refine 互逆守恒对的前提）——"
                f"最近可行因子：rows {rows_f} 的因子、cols {cols_f} 的因子之积的公因子"
                "（v26 正式档 = 4 ⇒ 120×240；不静默回落，宁炸）。"
            )
        self.cfg = cfg
        self.world_fine = world_fine
        self.factor = s
        self.rows_f, self.cols_f = rows_f, cols_f
        self.rows_c, self.cols_c = rows_f // s, cols_f // s
        self.n_c = self.rows_c * self.cols_c
        self.world = SphereWorld(self.rows_c, self.cols_c)
        self.every_ticks = int(cfg.every_ticks)
        self.substeps = int(cfg.substeps)
        self.dt_s = float(cfg.dt_s)
        self.steps = 0
        self.last_tick = -1

        # ---- 面积权（🔴 用 cos φ 原值，不用 `SphereWorld._area` 的 /max 归一：
        #      两网格各自归一 ⇒ 跨网格"全球积分"会差一个常数因子，守恒校核就不干净）----
        w_f = np.cos(np.asarray(world_fine._lat, dtype=np.float64))     # (rows_f,)
        self._w_fine = w_f
        self._w_block = w_f.reshape(self.rows_c, s)                     # (rows_c, s)
        self._den = self._w_block.sum(axis=1)                           # 块内行面积权和
        self.area = float(s) * self._den                                # 每粗格槽面积
        self._lat_c = np.asarray(self.world._lat, dtype=np.float64)

        # 状态载体（Step 1：零值；物理初值属 Step 2 的 EBM 初条件）
        self.t_surf = np.zeros(self.n_c, dtype=np.float64)              # atm:init
        self._prepare_ebm(surface_type)                                  # atm:ebm-prepare
        # ---- Step 3a：单层大气的质量 proxy / 地面气压（**诊断量**，零状态、零 RNG）----
        self.p_surf = np.zeros(self.n_c, dtype=np.float64)               # atm3a:p-init
        self.mass_col = np.zeros(self.n_c, dtype=np.float64)             # atm3a:m-init
        self._p_z = 0.0
        self._hypso_k = float(cfg.layer_depth_m) * _G_ACCEL / _R_DRY_AIR  # = z_b·g/R_d [K]
        self._diagnose_pressure()                                         # atm3a:p-diag
        # ---- Step 3b：风的几何尺、状态量与阻尼稳定校核（**积分状态**，初值静止）----
        self._c_ctr = np.sqrt(np.maximum(0.0, 1.0 - self.mu_ctr ** 2))     # atm3b:geom
        self._dmu_ctr = np.diff(self.mu_ctr)
        self._dlam_c = 2.0 * np.pi / float(self.cols_c)
        self.u_east = np.zeros(self.n_c, dtype=np.float64)                 # atm3b:u-init
        self.u_north = np.zeros(self.n_c, dtype=np.float64)                # atm3b:u-init
        # ---- B4.1 科氏参数（f = 2Ω·sinφ = 2Ω·μ；逐行常数，形状 (rc,1) 广播）----
        self.coriolis_on = bool(getattr(cfg, "coriolis_on", False))
        _cs = float(getattr(cfg, "coriolis_scale", 1.0))
        self._f_cor = ((2.0 * _OMEGA_DAY * _cs * self.mu_ctr)[:, None]
                       if self.coriolis_on else None)
        self._damp_k = (self.dt_s / float(cfg.rayleigh_tau_s)
                        if bool(cfg.damping_on) else 0.0)
        self._mu_damp = self.dt_s / float(cfg.rayleigh_tau_s)              # 读数用（与开关无关）
        if bool(cfg.damping_on) and self._mu_damp > _DAMP_STABILITY_MAX:
            raise ValueError(
                f"显式阻尼稳定数 dt/τ = {self._mu_damp:.3e} > {_DAMP_STABILITY_MAX}"
                f"（正式档 dt_s=150 / τ=86400 ⇒ 1.7e-3）—— 开 `damping_on` 时把 τ 改进小于"
                "两个子步 ⇒ 衰减因子变负 = 风每子步换向的非物理锯齿。宁炸（v26 §四-6 同族）。")
        # ---- C4a-2 Step A：水汽柱 / 陆面水桶 / 水量账本（🔴 默认关 ⇒ 整块跳过）----
        #   `moisture_on=False` ⇒ **不分配**任何新数组（关档内存与 tick 路径都不变），
        #   主 tick 与 C4a-1 开档**逐位等价** = 本 Step 的可证伪判据（单测 m18）。
        self.moisture_on = bool(cfg.moisture_on)
        self._cfl_max = 0.0
        self._cfl_warned = False
        self._adv_res_rel = 0.0
        if self.moisture_on:
            self._prepare_moisture(pr_fine, surface_type)     # atm4:prepare

    # ---- Step 2：EBM 驱动场、经向扩散算子、时间推进 ------------------------

    def _prepare_ebm(self, surface_type) -> None:
        """装配 EBM 的**静态驱动场**（α/C/S/τ）与初值 `t_surf`（零 RNG、确定性）。

        四条口径：
        ① **一把尺**：面积权 `area` 同时用于全球积分、面积权均值与拉普拉斯的 `1/w`
          ⇒ `Σ_i area_i·(LT_i) = 0` 到**机器精度**（自伴 ⇒ 望远镜相消，不留 1e-5 残差）；
        ② **μ 尺换算**：North(1975) 的 `D ∂_μ[(1−μ²) ∂_μ T]` 里格厚 = Δμ，而本模块的
          `area ≈ (factor/Δφ)·Δμ`（实测比值散布 < 2e-6，含极帽）⇒ 面传导率乘同一常数
          `factor/Δφ` 把 τ 换到 area 尺，**系数不换、只换尺**；
        ③ **极点不重复计数**（v26 §四-13）：极帽格质心 = (μ_1−1)/2（不是行心），
          南北两端各只有一个面通量（μ=±1 处 1−μ²=0 ⇒ 边界通量恒零）；极行的 α/S 由
          `coarsen` 的"一圈一格"口径给出，随后 `collapse_poles` 闭合；
        ④ **反照率闭合**（`cfg.albedo_closure`）：`planetary_offset` ⇒ α_eff = α + (α_p − α̅)
          （α̅ = 面积权均值，实测本世界 0.0923 ⇒ 偏移 0.2077）；`surface_raw` ⇒ 不闭合。
          🔴 两档的**全球均值差 35 ℃**（+14.10 vs +49.43）⇒ 这是 v26 的内部冲突，
          默认档选满足 §三 锚的那一档，并在汇报里作为待裁定项 A/B 呈报（不静默取舍）。
        """
        if surface_type is None:
            raise ValueError(
                "Atmosphere 需要 C0b 的 `surface_type` 层（Step 2 的 α/热容量按地表类型查表）"
                "—— 请开 `env_field.enabled` + `env_field.c0b_enabled`。"
                "不静默退化成「全球一种地表」（那会把 v26 §二「α 按 surface_type」整条口径抹掉）。")
        from world.env_field import (SURFACE_THERMOPHYSICS, annual_mean_base_temperature,
                                     annual_mean_insolation)

        st = np.asarray(surface_type)
        if st.size != self.rows_f * self.cols_f:
            raise ValueError(
                f"surface_type 尺寸 {st.shape}（{st.size}）≠ 主网格 "
                f"{self.rows_f * self.cols_f} —— 网格不一致宁炸（静默 reshape = 错位场）")
        known = sorted(SURFACE_THERMOPHYSICS)
        if st.min() < known[0] or st.max() > known[-1]:
            raise ValueError(
                f"surface_type 含未知编码 [{int(st.min())}, {int(st.max())}]"
                f"（C0b 表只覆盖 {known}）—— 新增地表类型必须先补热物性表，不许查表回落")
        tab = np.array([SURFACE_THERMOPHYSICS[k] for k in known], dtype=np.float64)
        idx = np.searchsorted(np.asarray(known, dtype=np.int64), st.astype(np.int64))
        alpha_fine = tab[idx, 0]                                   # α（细网格逐格）
        heat_fine = _C_REF * (tab[idx, 1] / _C_WATER)              # C = C_ref·c/c_w

        lat_f = np.asarray(self.world_fine._lat, dtype=np.float64)
        s_in = np.repeat(annual_mean_insolation(lat_f), self.cols_f)
        if str(self.cfg.insolation_mode) == "uniform":             # 零模型①
            w_all = np.repeat(self._w_fine, self.cols_f)
            s_in = np.full(self.rows_f * self.cols_f,
                           float((w_all * s_in).sum() / w_all.sum()))
        base = np.repeat(annual_mean_base_temperature(lat_f), self.cols_f)

        self.s_in = self.coarsen(s_in).ravel()
        self.alpha_surface = self.coarsen(alpha_fine).ravel()
        self.heat_cap = self.coarsen(heat_fine).ravel()
        if str(self.cfg.albedo_closure) == "planetary_offset":
            self.albedo_offset = _ALBEDO_ANCHOR - self.awm(self.alpha_surface)
            self.alpha_eff = self.alpha_surface + self.albedo_offset
        else:
            self.albedo_offset = 0.0
            self.alpha_eff = self.alpha_surface.copy()
        if self.alpha_eff.min() <= 0.0 or self.alpha_eff.max() >= 1.0:
            raise ValueError(
                f"闭合后反照率越界 [{self.alpha_eff.min():.4f}, {self.alpha_eff.max():.4f}]"
                "（α∉(0,1) = 物理非法：要么 α_p/世界海陆比异常，要么面积权出错，宁炸不 clip）")
        self.absorbed = self.s_in * (1.0 - self.alpha_eff)         # S(1−α_eff)

        # ---- 经向面传导率 τ（μ 尺 → area 尺；极帽质心、μ=±1 处边界通量自动为零）----
        if self.rows_c < 3:
            raise ValueError(
                f"粗网格 rows_c={self.rows_c} < 3 ⇒ 经向拉普拉斯无从定义"
                "（降采样因子相对主网格过大，不是物理口径）——宁炸")
        dphi = float(lat_f[1] - lat_f[0])                           # 细网格行距（rad）
        edge = np.empty(self.rows_c + 1, dtype=np.float64)
        edge[0], edge[-1] = -1.0, 1.0
        seam = np.arange(1, self.rows_c) * self.factor             # 块间缝 = 细行索引
        edge[1:-1] = np.sin(0.5 * (lat_f[seam - 1] + lat_f[seam]))
        dmu = np.diff(edge)
        ctr = (edge[1:] ** 2 - edge[:-1] ** 2) / (2.0 * dmu)       # 面积质心（极帽精确）
        tau_mu = _EBM_D_WM2K * (1.0 - edge[1:-1] ** 2) / np.diff(ctr)
        scale = float(self.factor) / dphi                          # area ≈ scale·Δμ
        self.mu_edge, self.mu_ctr = edge, ctr
        self._scale_mu2area = scale                                # 一把尺的换算常数（3c 散度用）
        self.tau = tau_mu * scale
        if not bool(np.all(np.isfinite(self.tau))) or float(self.tau.min()) <= 0.0:
            raise ValueError("面传导率 τ 必须有限且正（≤0/NaN = 网格几何退化，扩散变造热）")

        # ---- 显式稳定性校核（v26 §四-6/14：不稳定先查步长，别加非物理平滑）----
        sum_tau = np.empty(self.rows_c, dtype=np.float64)
        sum_tau[1:-1] = self.tau[:-1] + self.tau[1:]
        sum_tau[0], sum_tau[-1] = self.tau[0], self.tau[-1]
        mu_max = float((self.dt_s * sum_tau / (self.area * self.heat_cap.min())).max())
        if mu_max > _DIFF_STABILITY_MAX:
            raise ValueError(
                f"显式扩散稳定数 μ_max = dt·Στ/(C·area) = {mu_max:.3e} > "
                f"{_DIFF_STABILITY_MAX}（v26 §四-6：先查步长/dt_s，不许加平滑或改 D）。"
                "正式档 dt_s=150 ⇒ μ≈1.6e-2；越界说明节拍参数被改到了不稳区。")
        self._mu_diff = mu_max

        # ---- 初值：现 insolation 年平剖面（v26 §二 Step 2 明文）----
        t0 = self.coarsen(base)
        self.collapse_poles(t0)
        self.t_surf[:] = t0.ravel()

    # ---- Step 3a：质量 proxy + 静力平衡 → 地面气压（诊断量，零状态、零 RNG）------

    def _diagnose_pressure(self) -> None:
        """单层大气地面气压 `p_surf`（Pa）与柱质量 `mass_col`（kg/m²）。

        🔴 v26 §四-9 点名禁止的写法是 `P = P₀·T/T₀`（气压处处与温度**同向** ⇒ 暖区
        地面高压 ⇒ 风从赤道吹出去 = 赤道辐散 = 单胞反号）。本函数四步推导，每步可核对：

        ① **状态方程** `ρ = P/(R_d·T)`，T **只许** `t_kelvin()`（§四-4 唯一换算点 ——
           本行是全模块里 K 的唯一消费者，Step 2 全程 ℃）；
        ② **静力平衡** `dP/dz = −ρg` 在等温层积分 ⇒ 压高公式 `P(z) = P_s·e^{−z/H}`，
           标高 `H = R_d·T/g`（**暖柱更厚**）；
        ③ **质量 proxy**：层顶 `z = z_b`（`cfg.layer_depth_m`，边界层量级）气压水平
           均匀 = `P_顶` ⇒ 层内柱质量 `m = (P_s − P_顶)/g`。由 ② 反解
           `P_s = P_顶·exp(z_b/H) = P_顶·exp(z_b·g/(R_d·T))` ⇒ **暖柱地面气压低**
           （同样层顶气压下厚层装不下那么多质量 = "暖区上升、质量在高空外泄"的
           单层代理）。指数 `z_b·g/(R_d·T)` 随 T **递减** ⇒ 与 ① 合起来给
           `dP_s/dT < 0`，而 `P₀·T/T₀` 给 `dP/dT > 0` —— **反号即合规性判据**（单测 P3）。
        ④ **全球质量锚**：`P_顶` 取使 `AWM(P_s) ≡ P₀` 的常数 ⇒ 面积权平均地面气压
           （= 全球柱总质量 / 面积）恒等于标准值 ⇒ 温度场只能在柱间**重新分配**质量、
           不能凭空造/丢质量（§四-11/15 的质量侧对应物；单测 P4 钉逐位恒定）。

        有效域与异常（§四-15）：柱压比 `max/min − 1 > _HYPSO_RATIO_MAX` ⇒ 等温薄层
        理想化已在外推区（真实大气的地面气压对比被质量补偿压得更小）⇒ **fail-loud**；
        `P>0`、`m>0`、全有限三条逐次校核，一律抛错、不 clip、不静默。
        """
        tk = self.t_kelvin()
        if not bool(np.all(np.isfinite(tk))) or float(tk.min()) <= 0.0:
            raise FloatingPointError(
                f"状态方程入参非法：T(K) min={float(tk.min()):.3f}（需全有限且 > 0 K；"
                "§四-4/15 —— K 出现 0/负/NaN 时压高公式的指数已无意义，宁炸）。")
        raw = np.exp(self._hypso_k / tk)
        self._p_z = _P_REF_PAS / self.awm(raw)          # 层顶（非辐散面）气压：均匀常数
        self.p_surf = self._p_z * raw
        self.mass_col = (self.p_surf - self._p_z) / _G_ACCEL
        ratio = float(raw.max() / raw.min()) - 1.0
        if ratio > _HYPSO_RATIO_MAX:
            raise FloatingPointError(
                f"柱压比 max/min−1 = {ratio:.4f} > {_HYPSO_RATIO_MAX}（压高公式的等温薄层"
                "理想化已出有效域 ⇒ 要么 layer_depth_m 被改大、要么温度剖面异常）。"
                "不许 clip 或改常数硬凑（v26 §三/§四-15）。")
        p_min, m_min = float(self.p_surf.min()), float(self.mass_col.min())
        if not bool(np.all(np.isfinite(self.p_surf))) or p_min <= 0.0:
            raise FloatingPointError(
                f"地面气压出现非正/非有限（min={p_min:.3e} Pa，v26 §四-15 非负硬门）。")
        if m_min <= 0.0:
            raise FloatingPointError(
                f"活动层柱质量非正（min={m_min:.3e} kg/m²）—— 层顶气压高于地面 = "
                "静力平衡被破坏（v26 §四-15）。")

    # ---- Step 3b：压力梯度 + Rayleigh 阻尼 → 风（积分状态量；无科氏力）------------

    def _pressure_gradient(self):
        """返回节点处的 `(∂_μP, ∂_λP)`（两个 (rows_c, cols_c) 视图，**不改状态**）。

        面→节点的写法（与 `_laplacian` 共用同一套面索引 ⇒ 极行只有一面、经度周期）：
        * 经向 `face[i] = (P_{i+1}−P_i)/(μ̄_{i+1}−μ̄_i)`，节点 = 相邻两面之平均；
          首/末行只有内面一面 ⇒ 直接取该面（μ=±1 处无面 = 球面无"外面"）。
        * 纬向同式，但 `Δλ` 均匀且首尾**周期**（`np.roll`）⇒ 接缝自动闭合（§四-12）。
        两式都等于均匀网格上的中心差（`P_{i+1}−P_{i−1}`)/2Δ ⇒ 与"面梯度平均"逐位同值，
        写成面形式是为了让 3c 的通量式散度拿到**同一个**几何尺。
        """
        rc, cc = self.rows_c, self.cols_c
        p2 = self.p_surf.reshape(rc, cc)
        if self.rows_c < 3:                       # 理论不发生（`_prepare_ebm` 已硬闸 rows_c≥3）
            raise ValueError(f"粗网格 rows_c={self.rows_c} < 3 ⇒ 经向梯度无从定义")
        if not bool(np.all(np.isfinite(self._dmu_ctr))) or float(self._dmu_ctr.min()) <= 0.0:
            raise FloatingPointError(
                f"质心 μ 距非正/非有限（min={float(self._dmu_ctr.min()):.3e}）"
                "—— 网格几何退化 ⇒ 气压梯度会被除爆（宁炸）。")
        g_n = np.empty((rc, cc), dtype=np.float64)
        face_n = (p2[1:] - p2[:-1]) / self._dmu_ctr[:, None]
        g_n[0] = face_n[0]
        g_n[-1] = face_n[-1]
        g_n[1:-1] = 0.5 * (face_n[1:] + face_n[:-1])
        face_e = (np.roll(p2, -1, axis=1) - p2) / self._dlam_c
        g_e = 0.5 * (face_e + np.roll(face_e, 1, axis=1))
        return g_n, g_e

    def wind_forces(self):
        """气压梯度力 `(F_east, F_north)`（m/s²）—— 3b 的**唯一受力项**（无科氏力）。

        `F_n = −(c_i/(ρ_i·a))·∂_μP`、`F_e = −(1/(ρ_i·a·c_eff_i))·∂_λP`，`1/ρ = R_d·T/P`
        （状态方程的**第二次**消费 K，仍只走 `t_kelvin()` ⇒ §四-4）。两个方向的极地行为
        **相反**，这不是笔误：经向力乘 `c` ⇒ 极点归零（"北极点没有北"）；纬向力除 `c`
        ⇒ 极圈奇异 ⇒ 按 v26 §四-14 明示做极地滤波（`c_eff = max(c, 0.1)`，🔴 只裁纬向）。
        返回前把**极行的力折成环带均值**（§四-13 一圈一格 ⇒ 极点格不得有 240 个风；
        `u_e` 的环带均值对正压无科氏场恒为 0，即"绕极自转"本批无生成通道）。
        """
        g_n, g_e = self._pressure_gradient()
        rc, cc = self.rows_c, self.cols_c
        inv_rho = (_R_DRY_AIR * self.t_kelvin() / self.p_surf).reshape(rc, cc)
        c = self._c_ctr
        f_n = -(c[:, None] * inv_rho / _EARTH_RADIUS_M) * g_n
        f_e = -(inv_rho / (_EARTH_RADIUS_M * np.maximum(c, _POLAR_C_FLOOR)[:, None])) * g_e
        f_n[0] = float(f_n[0].mean())
        f_n[-1] = float(f_n[-1].mean())
        f_e[0] = float(f_e[0].mean())
        f_e[-1] = float(f_e[-1].mean())
        if not (bool(np.all(np.isfinite(f_n))) and bool(np.all(np.isfinite(f_e)))):
            raise FloatingPointError(
                "气压梯度力出现非有限值（∂P 或 1/ρ 已越界，v26 §四-15 非负/有限硬门）。")
        return f_e, f_n

    def _integrate_wind(self) -> None:
        """一个 Δt 子步的风方程 `u ← u + Δt·F − (Δt/τ)·u`（Forward Euler，显式）。

        🔴 三条口径：
        * **不反作用**：风只**读** `p_surf`/`t_surf`，不写它们、也不写任何主网格场
          ⇒ v26 §二 的因果方向（T→P→风）不颠倒；显式热平流 `−∇·(vT)` 属 C4a-2。
        * **Rayleigh 是摩擦平衡、不是地转**：`u_eq = τ·F`（力平衡）。本批无科氏力 ⇒
          没有转平衡可言，风必然**穿过**等压线流向低压（这正是 §四-7 单暖斑辐合的机制）。
          ⚠️ 离散格式对**旧 u** 衰减（先乘后加）⇒ 不动点恰为 `τ·F`；若把衰减写在加力之后，
          不动点变成 `τ·F·(1−Δt/τ)` = 同阶但系统性偏小的风，且与"平衡风"读数不自洽。
        * **阻尼稳定**：`Δt/τ ≤ 0.5` 已在构造期硬校核（`damping_on=False` 的零模型③
          里没有衰减项 ⇒ 不受该门限约束，但 `u` 随子步线性增长 = 要的就是这个反证）。
        """
        f_e, f_n = self.wind_forces()
        rc, cc = self.rows_c, self.cols_c
        ue = self.u_east.reshape(rc, cc)
        un = self.u_north.reshape(rc, cc)
        if not ue.flags["C_CONTIGUOUS"] or not un.flags["C_CONTIGUOUS"]:   # 防御性（同 EBM）
            raise ValueError("u_east/u_north 视图非 C 连续 ⇒ 原地写回不安全，宁炸")
        if bool(self.cfg.damping_on):            # 🔴 衰减必须作用在**旧 u** 上（先乘后加）
            k = 1.0 - self._damp_k
            ue *= k
            un *= k
        if self.coriolis_on:
            # 🔴 B4.1 科氏：du_e/dt = +f·u_n，du_n/dt = −f·u_e。
            #   用**同一时刻的旧值**同时更新（= 离散旋转矩阵，|u| 在无强迫时守恒）；
            #   若先更新 ue 再用新 ue 算 un，会引入 O(dt²) 的虚假增益。
            ue0 = ue.copy()
            un0 = un.copy()
            ue += self.dt_s * (self._f_cor * un0)
            un -= self.dt_s * (self._f_cor * ue0)
        ue += self.dt_s * f_e
        un += self.dt_s * f_n
        if not (bool(np.all(np.isfinite(ue))) and bool(np.all(np.isfinite(un)))):
            raise FloatingPointError(
                "风场出现非有限值（u_east/u_north 含 NaN/Inf）—— 禁止静默替换或截断"
                "（v26 §四-15）。")

    def div_u(self) -> NDArray[np.float64]:
        """风的**通量式球面散度** `∇·u`（1/m；纯诊断量，零状态、零 RNG，**不进快照**）。

        连续式 `∇·u = (1/a)[∂_μ(c·u_n) + (1/c)·∂_λ u_e]`（`c=cos φ`）写成控制体差分：

            div_ij = [c_½(ū_n)_½ − c_{−½}(ū_n)_{−½}]/(a·Δμ̃_i)
                   + [(ū_e)_{j+½} − (ū_e)_{j−½}]/(a·c_eff_i·Δλ)

        三条口径（都是 §四-11 能不能算"做到"的关键）：
        * 🔴 **`Σ area·div ≡ 0` 到机器精度**：第一行的格厚取**面积反推**的
          `Δμ̃_i = area_i/scale` ⇒ `area_i/Δμ̃_i ≡ scale` 是**定义上的同一个常数** ⇒ 经向项
          望远镜相消（实测微型档残差 **0.0**）；纬向项每行都是周期差 ⇒ 逐行恒零。
          ⚠️ 等距纬度网格下 `np.diff(mu_edge)` 其实也能相消（实测 4.0e-17：
          `Σ cos φ·Δφ = [Δφ/(2 sin(Δφ/2))]·Δμ` 对**每一行都是同一个因子** ⇒ 两把尺只差一个
          常数 1.14e-4），但那是**网格恒等式的巧合**；写成 `area/scale` 把相消变成"由构造
          保证"，换网格口径（非等距、极帽改写）时不会悄悄退化（同 §四-16 一把尺）。
        * **极外面零通量**：`μ=±1` 处 `c=0` ⇒ 极点那一面不进出流量 = 球面没有"外面"
          （§四-13 的另一半；极行的 u 已是环带均值 ⇒ 一圈一格在这里第二次自洽）。
        * **极地滤波同 3b**：纬向项分母用 `c_eff = max(c, 0.1)`（🔴 只裁纬向）。
        面速度 = 相邻节点算术平均（与 `_pressure_gradient` 同一套面索引 ⇒ 3b 的力和 3c 的
        散度用的是**同一个**几何尺，不会一边中心差、一边面差）。
        """
        rc, cc = self.rows_c, self.cols_c
        un = self.u_north.reshape(rc, cc)
        ue = self.u_east.reshape(rc, cc)
        c_face = np.sqrt(np.maximum(0.0, 1.0 - self.mu_edge[1:-1] ** 2))   # 内部面 c
        flux_n = c_face[:, None] * 0.5 * (un[1:] + un[:-1])                # (rc−1, cc)
        dmu_area = self.area / self._scale_mu2area                          # 一把尺的 Δμ
        d_m = np.empty((rc, cc), dtype=np.float64)
        d_m[0] = flux_n[0] / dmu_area[0]                     # 南极外面通量 = 0 ⇒ 只剩内面
        d_m[-1] = -flux_n[-1] / dmu_area[-1]
        d_m[1:-1] = (flux_n[1:] - flux_n[:-1]) / dmu_area[1:-1, None]
        flux_e = 0.5 * (ue + np.roll(ue, -1, axis=1))                      # 经度周期 ⇒ 接缝普通
        d_e = (flux_e - np.roll(flux_e, 1, axis=1)) / (
            _EARTH_RADIUS_M * np.maximum(self._c_ctr, _POLAR_C_FLOOR)[:, None] * self._dlam_c)
        out = d_m / _EARTH_RADIUS_M + d_e
        if not bool(np.all(np.isfinite(out))):
            raise FloatingPointError(
                "风散度出现非有限值（u 或网格几何已越界，v26 §四-15 非负/有限硬门）。")
        return out

    def vertical_velocity(self) -> NDArray[np.float64]:
        """Step 3c 的**垂直运动代理** `w(z_b) = −z_b·∇·u`（m/s，层顶；地表取 0）。

        四条口径（v26 §二「∇·u = 垂直运动代理」+ §三「暖区 w>0」+ §四-8）：
        * **不引入第二个长度尺度**：单层大气只有 `layer_depth_m` ⇒ 层顶垂直速度就是全部
          信息，地表为 0、线性到层顶 = 边界层质量连续的最低阶写法（`w_top` 不是新旋钮）。
        * **符号**：辐合（`∇·u<0`）⇒ `w>0`（上升）。暖斑地面低压 ⇒ 风辐合 ⇒ 上升 ⇒
          `w` 与 `T` 异常**正相关**（§四-8 的可观测内容；算子写反时该读数立刻变负）。
        * 🔴 **纯诊断**：零状态、零 RNG、不进快照（同 `p_surf`/`div_u`）⇒ `restore()` 之后
          由 `t_surf` + `u_*` 直接重算，不存在"重建不出来"的历史依赖。
        * ⚠️ 本批**不接**云与降水：云量 ∝ 辐合属 **C4a-3 诊断云层**（owner v30 已裁"云不在
          本批"），这里只把 `w` 算出来并给读数，不留任何下游接线。
        """
        return -float(self.cfg.layer_depth_m) * self.div_u()

    # ---- C4a-2 Step A：水汽平流 + 蒸发约束 + 水量账本（T1–T5 走 owner v35 的 A 支）------

    def _prepare_moisture(self, pr_fine, surface_type) -> None:
        """装配 Step A 的**静态**水分面（水面占比 / 降水型 / 离水距）+ 三个状态量。

        三条口径（都在构造期，跑批里不会再判一次）：
        * **子格 tiling**（红线：粗网格一格 = 一块真实地表）：`wf = coarsen(非陆地)` 是粗格
          里的**水面占比** ∈[0,1]。蒸发按占比分账：`wf` 部分取水面（不限水，海洋是无限库），
          `1−wf` 部分取陆面（受水桶约束）⇒ 一格内"有海有陆"不需要第二个网格，也不需要
          任何隐藏的全局混合参数。🔴 单位口径按 owner 修订令：`bucket` 与陆面通量都在
          **"每单位陆地面积"**尺上算（容量 `W_max` 不乘占比），只在**进账本/守恒校核**时乘
          `land = 1−wf` 回到整格尺 ⇒ 两套尺的换算是**一处乘、一处除**，不允许多处混用。
          🔴 用**同一个** `coarsen`（面积权分块均值）⇒ 与热机链一把尺；极点行走 `_fine2d` 的
          "规范槽广播"口径，不吃非规范极槽垃圾。
        * **降水型 × 降水锚**（T4 走 A 支）：`p_rate = P̄·coarsen(pr)/AWM(coarsen(pr))`。
          分母取**粗网格**的面积权均值 ⇒ `AWM(p_rate) ≡ P̄` 由构造成立（全球水量钉死在
          990 mm/yr，ENV 的 `pr` 只提供**经向/经度结构**）。⚠️ 不静默退化：`pr=None`
          （没开 `env_field`）⇒ 炸；全球无水面（没有蒸发源）⇒ 炸；`pr` 面积权 ≤0 ⇒ 炸。
        * **离水距离**（判据①"内陆递减"的问法）：4 邻域 BFS（经度**周期**、纬度不绕），
          水面格 = 0、海岸陆格 = 1、…🔴 只作读数窗口用（不参与任何通量/源汇项），
          与 `_BAND_*` 同族纪律 = 先钉死问法再读数。
        """
        from world.env_field import SURFACE_LAND

        if pr_fine is None:
            raise ValueError(
                "moisture_on=True 但拿不到 `env_field.pr`（气候态降水 proxy）：降水的**空间型**"
                "必须由它给出（全球总量钉在 P̄=990 mm/yr 锚上）。不静默退化成均匀降水 —— "
                "那会把纬度带/地形抬升的经向结构抹平，判据①「内陆递减」失去被测对象。"
                "请开 `env_field.enabled`（`pr` 是只存储层，与 c0b/c1 是否开无关）。")
        st = np.asarray(surface_type)
        if st.size != self.rows_f * self.cols_f:
            raise ValueError(
                f"surface_type 尺寸 {st.shape} ≠ 主网格 {self.rows_f * self.cols_f}"
                "（水汽与热机链必须同一张网格，宁炸不 reshape）")
        water_fine = (st.astype(np.int64) != SURFACE_LAND).astype(np.float64)
        wf = self.coarsen(water_fine)                            # (rows_c, cols_c) ∈ [0,1]
        if float(wf.min()) < 0.0 or float(wf.max()) > 1.0:
            raise FloatingPointError(
                f"水面占比越界 [{float(wf.min()):.3e}, {float(wf.max()):.3e}]（子格占比必须在 "
                "[0,1]；越界 = coarsen 的输入不是 0/1 掩码或面积权出错，宁炸不 clip）。")
        if self.awm(wf) <= 0.0:
            raise ValueError(
                "moisture_on=True 但全球**没有水面**（`wf` 面积权 = 0）⇒ 没有蒸发源，"
                "水汽场会静默恒零（= 构造了却什么都不算，B3 家族）。请开 C0b 的海陆划分。")
        pr_c = self.coarsen(pr_fine)
        if not bool(np.all(np.isfinite(pr_c))) or float(pr_c.min()) < 0.0:
            raise FloatingPointError(
                f"降水 proxy 非法（min={float(pr_c.min()):.3e}，需有限且非负）—— "
                "`env_field.pr` 是 Magnus 的乘积，出现负/NaN = 上游温度场已越界。")
        pr_mean = self.awm(pr_c)
        if pr_mean <= 0.0:
            raise FloatingPointError(
                f"降水 proxy 的面积权均值 = {pr_mean!r}（应 >0）⇒ 归一化无定义（不静默除零，"
                "同 `env_field` C1 band 的先例）。")

        self.water_frac = wf.ravel()
        self.p_rate = (_P_ANCHOR_KG_M2_S * pr_c / pr_mean).ravel()
        self.land_dist = self._land_distance()
        # 🔴 单位口径（owner 2026-10-17 修订）：`qv` = 层均**混合比** kg/kg（干起算 = 0，
        #   与"柱水 0"同一个数）；`bucket` = 每单位陆地面积的 kg/m²（空桶 = 0，
        #   纯海格的桶恒为 0：降水只给 `land>0` 的格 ⇒ 桶里不会存"海上的水"这种垃圾读数）。
        self.qv = np.zeros(self.n_c, dtype=np.float64)              # atm4:q-init（干起算）
        self.bucket = np.zeros(self.n_c, dtype=np.float64)          # atm4:b-init（空起算）
        self.wb = np.zeros(6, dtype=np.float64)                     # atm4:wb-init（mm）

    def _land_distance(self) -> NDArray[np.int16]:
        """离水距离（BFS，粗网格格数）：水面格 0，海岸陆格 1，向内递增；🔴 只服务读数。"""
        rc, cc = self.rows_c, self.cols_c
        water = self.water_frac.reshape(rc, cc) > 0.0
        dist = np.full((rc, cc), -1, dtype=np.int16)
        dist[water] = 0
        queue = deque((int(i), int(j)) for i, j in zip(*np.nonzero(water)))
        while queue:
            i, j = queue.popleft()
            d = int(dist[i, j]) + 1
            if d > np.iinfo(np.int16).max - 1:
                raise FloatingPointError(
                    f"离水距离 BFS 超过 int16 上限（rows_c={rc} 不可能到这，网格几何异常）")
            for ii in (i - 1, i + 1):
                if 0 <= ii < rc and dist[ii, j] < 0:
                    dist[ii, j] = d
                    queue.append((ii, j))
            for jj in ((j - 1) % cc, (j + 1) % cc):
                if dist[i, jj] < 0:
                    dist[i, jj] = d
                    queue.append((i, jj))
        out = dist.ravel()
        land = self.water_frac < 1.0
        if bool(np.any(land & (out < 0))):
            raise FloatingPointError(
                f"陆格离水距离不可达（{int(np.count_nonzero(land & (out < 0)))} 格）—— "
                "粗网格 4 邻域（经度周期）必然连通，不可达 = 网格几何异常，宁炸。")
        return out

    def saturation_mixing_ratio(self) -> NDArray[np.float64]:
        """饱和混合比 `q_sat = ε·e_s/(P_s − e_s)`（kg/kg；`e_s` = Magnus，T 用 **℃**）。

        🔴 口径：本函数与**状态量 `qv` 本身**都是**混合比**（每 kg 干空气的 kg 水汽，
        owner 2026-10-17 修订令）⇒ 两者直接相减就是空气动力学公式里的"饱和 deficit"，
        式子里**不出现** `mass_col`（它只在柱水 ↔ 比湿换算时出现，见 `column_water()`）；
        一个按比湿、一个按混合比 ⇒ deficit 系统性偏大 ~0.4%（`ε` 与 `1−(1−ε)·…` 的差），
        且随 T 变 ⇒ 不是常数偏差，不能"以后补"。
        `mass_col`（= 活动层柱质量 `(P_s−P_顶)/g`）是"这格大气装得下多少 kg 空气"的同一把尺
        ⇒ 水汽柱 / 空气柱 = 混合比，与压高公式那条链自洽。
        """
        den = _MAGNUS_B + self.t_surf
        if float(den.min()) <= 0.0:
            raise FloatingPointError(
                f"Magnus 分母 B+T 非正（T min={float(self.t_surf.min()):.3f} ℃）—— "
                "T ≤ −237 ℃ 已远出 EBM 有效下界（−105 ℃），宁炸（v26 §四-15）。")
        es = _HPA_TO_PA * _MAGNUS_C_HPA * np.exp(_MAGNUS_A * self.t_surf / den)
        gap = self.p_surf - es
        if float(gap.min()) <= 0.0:
            raise FloatingPointError(
                f"P_s − e_s 非正（min={float(gap.min()):.3e} Pa）：饱和水汽压 ≥ 地面气压 ⇒ "
                "混合比无定义（真实大气不可能；这里只能是温度/气压锚被改进荒谬区）。宁炸。")
        return _EPS_WS * es / gap

    def column_water(self) -> NDArray[np.float64]:
        """柱水汽 `qv × mass_col`（kg m⁻² ≡ mm，展平 (n_c,)）—— 🔴 只读，不是状态量。

        存在理由 = 单位口径只许有**一处**换算：状态量是混合比（owner 修订令），而平流、
        收支、守恒校核全都在"柱水"这把尺上 ⇒ 与其在四五个地方各乘一次 `mass_col`
        （漏一次就是一个假源项），不如把乘除集中在这里与 `_moisture_substep` 的写回点。
        """
        return self.qv * self.mass_col

    def evaporation_potential(self, q_sat=None) -> NDArray[np.float64]:
        """空气动力学**蒸发势** `E_pot = C_E·ρ_a·|v|·max(0, q_sat − qv)`（kg m⁻² s⁻¹）。

        * 🔴 只读 `t_surf`/`p_surf`/`u_*`/`qv`，**不写**任何东西 ⇒ Step A 是被动的
          （`_moisture_substep` 的因果方向口径见其 docstring）；
        * 🔴 单位口径（owner 修订令）：`qv` **本身就是混合比** ⇒ deficit 是 `q_sat − qv`，
          式子里不出现 `mass_col`（旧写法 `q_sat − qv/mass_col` 在新口径下会把 deficit
          算成天文数字）；输出的通量仍是**每单位地表面积**的 kg m⁻² s⁻¹ ⇒ 乘占比回整格账
          由 `_moisture_substep` 做；
        * `max(0, …)` = 欠饱和才蒸：过饱和**不**在这里"负蒸发"，凝结一律走降水汇
          （`p_rate`）⇒ 避免出现"蒸发项自己造降水"的第二条汇通道（T3 A 支不接潜热，
          凝结放热根本没有出口）；
        * `q_sat − qv` 用**当前**混合比 ⇒ 已饱和的水面给 0，这就是"水汽约束"的一半，
          另一半（陆面可用水）在 `_moisture_substep` 里做。
        """
        if q_sat is None:
            q_sat = self.saturation_mixing_ratio()
        rho_a = self.p_surf / (_R_DRY_AIR * self.t_kelvin())
        spd = np.sqrt(self.u_east * self.u_east + self.u_north * self.u_north)
        out = _C_E * rho_a * spd * np.maximum(0.0, q_sat - self.qv)
        if not bool(np.all(np.isfinite(out))) or float(out.min()) < 0.0:
            raise FloatingPointError(
                f"蒸发势非法（min={float(out.min()):.3e}，需有限且非负）—— 公式三个因子"
                "（ρ_a、|v|、饱和 deficit）都非负，出负/NaN = 上游场已越界，宁炸。")
        return out

    def moisture_advection(self) -> tuple[NDArray[np.float64], float]:
        """柱水汽的通量式散度 `∇·(柱水·v)` + 迎风**CFL 数**（🔴 只读，不改状态）。

        🔴 被输运量 = **柱水** `qv × mass_col`（kg m⁻²），不是比湿本身（owner 修订令后的
        唯一自洽选择：`mass_col` 是诊断量、不参与平流 ⇒ 若把比湿当守恒量输运，
        `Σarea·∇·(q·v) ≡ 0` 仍然成立但**它不再是水量守恒**，因为每格的空气柱质量不同 ⇒
        平流会凭空改变 `AWM(qv·mass_col)` = 账本里出现一个看不见源）。⇒ 本函数在柱水尺上
        工作，调用方（`_moisture_substep`）除回 `mass_col` 写回比湿。

        与 `div_u()` **共用同一套面索引、同一套分母**（`c_face`、`Δμ̃ = area/scale`、
        `c_eff = max(c, _POLAR_C_FLOOR)`、经度周期），唯一区别 = 面上的柱水不再取算术平均
        而是**迎风一阶**：`q_face = 上游柱水(sign(v·n))`。带来的三条：

        * 🔴 **保正凸组合**：面通量形如 `Σ_邻 (系数)·柱水_上游`，系数只由风与网格定 ⇒ 只要
          `ν_i = Δt·Σ_面 出流系数 ≤ 1`，更新就是柱水的**凸组合** ⇒ 柱水不可能出负 ⇒ 比湿
          也不可能（`mass_col > 0` 由 `_diagnose_pressure` 硬门保证）。这就是
          `_CFL_STABILITY_MAX` 逐子步硬校核的内容（红线 5：按**高纬最小格数**校核，分母含
          `c_eff` ⇒ 极带的 `1/c` 放大被算进 ν，不是算完再补）。
          不稳定时🔴 **不许全局限小步**（owner 派工原文）⇒ 已有的极地滤波（`c_eff`）就是
          该做的处置；越界即炸，不许悄悄 clip `qv`。
        * **守恒同尺**：`Σ area·∇·(柱水·v) ≡ 0` 的望远镜相消只依赖"同一个面通量被两侧
          反号使用" ⇒ 迎风换的是**面上的取值**，不换通量的反对称结构 ⇒ 恒等式照旧
          成立（单测 m05 逐位核）；它现在**就是**全球柱水量的守恒（尺统一到柱水之后）。
        * **明写的代价（T1 走 A 支）**：一阶迎风的数值扩散 `K_num ≈ |v|·Δx/2` ⇒ 本档
          `|v| ≲ 30 m/s`、`Δx ≈ 1.7e5 m`（赤道格距，高纬按 `1/c_eff` 更细）⇒ **K_num ≈
          1.5e6 m²/s**。⚠️ Step B 的显式热平流会**和扩散项重复计算**同一份次网格混和，
          届时必须写明（派工范围已认这条挂账），不许在此假装"平流=纯输运无耗散"。
        """
        rc, cc = self.rows_c, self.cols_c
        un = self.u_north.reshape(rc, cc)
        ue = self.u_east.reshape(rc, cc)
        q2 = self.column_water().reshape(rc, cc)      # 🔴 柱水 (kg/m²)；新数组，不写状态
        c_face = np.sqrt(np.maximum(0.0, 1.0 - self.mu_edge[1:-1] ** 2))
        un_f = c_face[:, None] * 0.5 * (un[1:] + un[:-1])        # 面法向体积通量尺 c·u_n
        q_n = np.where(un_f > 0.0, q2[:-1], q2[1:])              # 🔴 迎风：取上游格的柱水
        flux_n = un_f * q_n
        dmu_area = self.area / self._scale_mu2area               # 一把尺的 Δμ̃（同 div_u）
        d_m = np.empty((rc, cc), dtype=np.float64)
        d_m[0] = flux_n[0] / dmu_area[0]                          # 南极外面 = 零通量
        d_m[-1] = -flux_n[-1] / dmu_area[-1]
        d_m[1:-1] = (flux_n[1:] - flux_n[:-1]) / dmu_area[1:-1, None]
        ue_f = 0.5 * (ue + np.roll(ue, -1, axis=1))               # 面 j+½ 的纬向速度
        q_e = np.where(ue_f > 0.0, q2, np.roll(q2, -1, axis=1))   # 经度周期 ⇒ 接缝普通
        flux_e = ue_f * q_e
        den_e = (_EARTH_RADIUS_M * np.maximum(self._c_ctr, _POLAR_C_FLOOR)[:, None]
                 * self._dlam_c)
        d_e = (flux_e - np.roll(flux_e, 1, axis=1)) / den_e
        out = d_m / _EARTH_RADIUS_M + d_e
        if not bool(np.all(np.isfinite(out))):
            raise FloatingPointError(
                "水汽平流出现非有限值（qv/u 或网格几何已越界，v26 §四-15 硬门）。")
        # ν_i = Δt·Σ_面 出流系数（只由风与网格定 ⇒ 与 qv 无关，凸组合条件的显式形式）
        nu = np.zeros((rc, cc), dtype=np.float64)
        nu[:-1] += np.maximum(un_f, 0.0) / (_EARTH_RADIUS_M * dmu_area[:-1, None])
        nu[1:] += np.maximum(-un_f, 0.0) / (_EARTH_RADIUS_M * dmu_area[1:, None])
        nu_e = np.maximum(ue_f, 0.0) / den_e
        nu += nu_e + np.roll(nu_e, 1, axis=1)
        return out, float(self.dt_s * nu.max())

    def _moisture_substep(self) -> None:
        """一个 Δt 子步的水循环：**平流 → 蒸发（受限）→ 降水汇 → 桶溢出成径流 → 入海**。

        🔴 因果方向（派工范围 + owner v35 §三）= **Step A 是纯被动示踪**：本函数只**读**
        `t_surf`/`p_surf`/`u_*`/`mass_col`，**一个字都不写**它们，也不写任何主网格场 ⇒
        "开档的水汽链跑完后，热机链与 `moisture_on=False` **逐位相同**"是本 Step 的
        可证伪判据（单测 m18 钉死）。潜热反馈（凝结放热回 EBM）= **T3 走 A 支 = 本批明确
        不接**；云/辐射接线属 C4a-3（owner v35 §三"云不接线"同族）。

        两套尺（owner 2026-10-17 修订令 ⇒ 🔴 每个量都在 docstring 里标尺）：
        * **柱水尺** `col = qv × mass_col`（kg m⁻² ≡ mm）：平流、蒸发、降水、账本、守恒
          全在这把尺上算（"取走/加入多少水"是质量问句，与每格空气柱多重无关）；
        * **陆面尺** `bucket`（kg m⁻² of **land**）：桶的进出都在每单位陆地面积上谈
          （需求 `E_pot·Δt`、降水 `P`、容量 `W_max` 都不乘占比），**进整格账**时才乘 `land`。
        换算各只有**一次**：`col ↔ qv` 在开头乘 / 末尾除 `mass_col`；`bucket ↔ 整格` 靠
        `land` 一次乘。🔴 纯海格（`land==0`）的桶**从不被写入**（`np.where(has_land, …)`）⇒
        不会积累"海上的陆面水"这种污染 `bucket_min/max` 读数的垃圾。

        五个量（全部在柱水尺上，单位 mm）：
        ① 平流 `col −= Δt·∇·(col·v)`（迎风一阶输运**柱水**，见 `moisture_advection`）；
        ② 蒸发 `E = Δt·E_pot·wf + land·min(Δt·E_pot, bucket)` —— 第一项是**水面**
           （无限库，只受空气欠饱和约束），第二项是**陆面**受水桶约束（`water_limit_on`，
           False = 零模型⑤：不许负蒸发，而是**允许桶为负**= "永远湿"的反证，账本仍逐位闭合）；
        ③ 汇 `P = min(col_after_E, p_rate·Δt)` —— 🔴 取走量以**当前柱水汽**为上限 ⇒
           干格不会凭空下雨（`precip_realized_frac` 就是这个截断的读数），且账本记的是
           **实际** P，不是潜在 P；比湿的减少 = `P / mass_col`（owner 令的字面式）；
        ④ 径流（**T2 走 A 支**：溢出瞬达海洋库，不建汇流延迟/河网）
           `R = land·max(0, bucket − W_max)`（容量在陆面尺上 ⇒ 式子里没有占比）；
        ⑤ 海洋 `ΔOcean = P·wf + R − E_水面` —— 唯一的"库外库"（派工范围：海洋是边界条件，
           不是状态量）。

        **闭合恒等式**（代数上恒等，判据②测的是数值实现有没有漏项）。它拆成两段分别校核，
        因为两段的**噪声底不同**，混在一起会让容差被大的那段决定：
        * **输运段** `Σ area·∇·(col·v) ≡ 0` —— 通量式散度的望远镜相消（面通量被两侧反号
          使用）。它按 `|∇·(col·v)|` 的量级带机器精度噪声 ⇒ 单独记相对残差
          `adv_res_rel`（超 `_WB_TOL_REL` 即炸，本 Step 内硬校核）。
        * **源汇段**（= `wb[4]` 累积的 `bal`）：以**平流之后**为基准，
          `ΔAWM(col) + ΔAWM(bucket·land) + AWM(ΔOcean) ≡ 0` —— 展开后 `e_w`、`e_l`、`p_amt`
          （靠 `wf + land ≡ 1`）、`r` 逐项对消 ⇒ 只剩求和次序的浮点噪声（实测
          ~1e-13 mm 量级）。这段是"源汇分配漏没漏因子"的直接探针。
        ⚠️ 桶项**必须乘 `land`** 才进恒等式：状态量是"每单位陆地面积的深"，全球储量是它乘
        陆面占比再加权 ⇒ 直接 `AWM(bucket)` 是另一条问句（probe 两个键都给）。
        两段相加 = 全子步闭合 `ΔAWM(col + bucket·land) + AWM(ΔOcean) ≡ 0`（判据②的字面口径）。
        `wb = [E, P, R, Ocean, imbalance, P_pot]`（整格面积权累积，单位 **mm**；换算因子
        `_MM_PER_KG_M2 = 1000/ρ_w` **恰为 1.0** ⇒ 乘它不改数值、只把尺写在式子里）。末位
        `P_pot` = 潜在降水累积，**不进恒等式**，只给 `precip_realized_frac` 做分母。
        """
        dt = self.dt_s
        rc, cc = self.rows_c, self.cols_c
        q2 = self.qv.reshape(rc, cc)                       # 比湿视图（末尾一次写回）
        b2 = self.bucket.reshape(rc, cc)                   # 陆面尺：kg/m² of land
        if not (q2.flags["C_CONTIGUOUS"] and b2.flags["C_CONTIGUOUS"]):   # 防御（同 EBM）
            raise ValueError("qv/bucket 视图非 C 连续 ⇒ 原地写回不安全，宁炸")
        wf = self.water_frac.reshape(rc, cc)
        land = 1.0 - wf
        has_land = land > 0.0                             # 桶只服务"格内真有一块陆"
        mc = self.mass_col.reshape(rc, cc)                 # 空气柱 (kg/m²)
        if not bool(np.all(np.isfinite(mc))) or float(mc.min()) <= 0.0:
            raise FloatingPointError(
                f"柱质量为非正/非有限（min={float(mc.min()):.3e} kg/m²）⇒ 比湿 ↔ 柱水换算"
                "无定义（`_diagnose_pressure` 已硬门；走到这里 = 有人在诊断之后改了气压）。")
        col = q2 * mc                                      # 🔴 柱水尺（kg/m² ≡ mm）

        # ---- 1) 平流（零模型④：`moisture_advect_on=False` ⇒ 整块跳过 = 局地基线）----
        if bool(self.cfg.moisture_advect_on):
            div_q, cfl = self.moisture_advection()
            if cfl > _CFL_STABILITY_MAX:
                raise FloatingPointError(
                    f"迎风平流 CFL 数 ν_max = {cfl:.3f} > {_CFL_STABILITY_MAX} ⇒ 凸组合条件"
                    "被破坏（柱水会变负 = 造负水汽）。红线 5：🔴 **不许全局限小步**、不许 clip "
                    "`qv` ⇒ 该做的处置是高纬滤波（`c_eff` 地板，已内置）/改粗网格因子，"
                    "两者都超出本 Step 授权 ⇒ 停手报话事人（A/B/C）。")
            if cfl > _CFL_WARN_MAX and not self._cfl_warned:
                self._cfl_warned = True                         # 只发一次，不刷日志
                warnings.warn(
                    f"C4a-2 水汽平流 CFL 数 ν_max = {cfl:.3f} ∈ ({_CFL_WARN_MAX}, "
                    f"{_CFL_STABILITY_MAX}]：仍在保正域内但已过半 ⇒ 预警（读数进 probe "
                    "`cfl_max`/`cfl_warned`）。", RuntimeWarning, stacklevel=2)
            if cfl > self._cfl_max:
                self._cfl_max = cfl
            # 输运段守恒硬校核：Σarea·∇·(col·v) ≡ 0（面通量两侧反号 ⇒ 只剩求和次序噪声）
            adv = self.awm(np.abs(div_q))
            res = abs(self.awm(div_q)) / adv if adv > 0.0 else 0.0
            if res > _WB_TOL_REL:
                raise FloatingPointError(
                    f"水汽平流的面积权和残差 |Σarea·∇·(col·v)|/Σarea·|∇·(col·v)| = {res:.3e} > "
                    f"{_WB_TOL_REL} —— 通量式散度按构造应当望远镜相消（同一面通量被两侧反号"
                    "使用）。越界 = 面索引/分母与 `div_u` 那套尺脱钩 ⇒ 平流在造/吞水（红线 4）。")
            if res > self._adv_res_rel:
                self._adv_res_rel = res
            col -= dt * div_q
            if float(col.min()) < 0.0:
                raise FloatingPointError(
                    f"平流后柱水汽为负（min={float(col.min()):.3e} kg/m²）—— 迎风一阶在 "
                    "CFL≤1 下是保正凸组合，出负 = 格式或校核被绕过（红线 4）。不 clip。")
        q_before = self.awm(col)
        b_before = self.awm(b2 * land)    # 平流不碰桶 ⇒ 同点取基准；🔴 整格尺（乘 land）

        # ---- 2) 蒸发（水面不限水；陆面受水桶约束 = `water_limit_on`）----
        # 🔴 `evaporation_potential()` 走的是 `t_surf`/`p_surf`/`qv` 的**展平**口径（与快照/
        #    probe 同形），本函数体内一律 2-D ⇒ 在这里换形一次（不是两处都能用的一种形状）。
        e_dem = dt * self.evaporation_potential().reshape(rc, cc)    # 每单位地表面积的需水
        e_w = e_dem * wf                                            # 水面：无限库
        e_l_land = np.where(has_land, e_dem if not bool(self.cfg.water_limit_on)
                            else np.minimum(e_dem, b2), 0.0)        # 陆面尺（零模型⑤=不封顶）
        e_l = e_l_land * land                                       # → 整格账
        col += e_w + e_l
        b2 -= e_l_land

        # ---- 3) 汇（降水：空间型 × 锚，取走量 ≤ 当前柱水；陆面那份按陆面尺入桶）----
        p_pot = self.p_rate.reshape(rc, cc) * dt
        p_amt = np.minimum(col, p_pot)
        col -= p_amt
        b2 += np.where(has_land, p_amt, 0.0)      # 同一水深落到陆地上 ⇒ 整格 = p_amt·land

        # ---- 4) 径流（桶满即溢，瞬达海洋库；T2 = A 支。容量在陆面尺上 ⇒ 不乘占比）----
        r_land = np.maximum(b2 - _W_MAX_KG_M2, 0.0)
        r_cell = r_land * land
        b2 -= r_land

        # ---- 5) 海洋与账本（🔴 记的是**实际**通量，不是潜在通量；单位 mm）----
        ocn = p_amt * wf + r_cell - e_w
        e_tot, p_tot = self.awm(e_w + e_l), self.awm(p_amt)
        r_tot, ocn_tot = self.awm(r_cell), self.awm(ocn)
        self.wb[0] += _MM_PER_KG_M2 * e_tot
        self.wb[1] += _MM_PER_KG_M2 * p_tot
        self.wb[2] += _MM_PER_KG_M2 * r_tot
        self.wb[3] += _MM_PER_KG_M2 * ocn_tot
        self.wb[5] += _MM_PER_KG_M2 * self.awm(p_pot)   # 潜在降水 ⇒ 只服务 realized_frac
        bal = ((self.awm(col) - q_before) + (self.awm(b2 * land) - b_before) + ocn_tot)
        self.wb[4] += _MM_PER_KG_M2 * bal
        scale = max(abs(self.awm(col)) + abs(self.awm(b2 * land)) + e_tot + p_tot + r_tot
                    + abs(ocn_tot), 1e-300)
        if abs(bal) > _WB_TOL_REL * scale:
            raise FloatingPointError(
                f"水量账本源汇段单子步残差 |ΔAWM(col)+ΔAWM(bucket·land)+AWM(ΔOcean)| = "
                f"{abs(bal):.3e} mm > {_WB_TOL_REL}×{scale:.3e}（红线 4：账本不闭合即炸）。"
                "基准取在**平流之后**⇒ 展开后 E/P/R/Ocean 各项应逐项对消（靠 `wf + land ≡ 1`），"
                "唯一数值来源 = 求和次序噪声 ⇒ 越界说明源汇分配漏了因子（水面/陆面占比、"
                "取走量与入桶量不等、或桶项忘了乘 `land`）。")

        # ---- 6) 写回状态量 + 非负 / 有限 / 上界 fail-loud（红线 4；一律抛错，不截断）----
        q2[:] = col / mc                     # 🔴 柱水 → 比湿：本函数里唯一一次除
        q_min, b_min, b_max = float(col.min()), float(b2.min()), float(b2.max())
        if not (np.isfinite(q_min) and np.isfinite(b_min) and np.isfinite(b_max)):
            raise FloatingPointError(
                f"水汽/水桶出现非有限值（柱水 min={q_min}, bucket min={b_min}, "
                f"bucket max={b_max}）—— 禁止静默替换（v26 §四-15）。")
        if q_min < 0.0:
            raise FloatingPointError(
                f"柱水汽为负（min={q_min:.3e} kg/m²）—— 平流保正 + `min(柱水,·)` 限汇 ⇒ 出负 "
                "= 接线被改坏，宁炸不 clip（红线 4）。")
        if bool(self.cfg.water_limit_on) and b_min < 0.0:
            raise FloatingPointError(
                f"开水量约束时陆面水桶为负（min={b_min:.3e} kg/m² of land）—— "
                "`min(潜蒸发, 桶)` 保证取水量 ≤ 桶存量 ⇒ 出负 = 约束被绕过（零模型⑤关约束时"
                "**允许**为负，那正是它要展示的东西）。")
        if b_max > _W_MAX_KG_M2 * (1.0 + _WB_TOL_REL):
            raise FloatingPointError(
                f"水桶超过容量（max={b_max:.3e} > W_max={_W_MAX_KG_M2:.3e} kg/m² of land）—— "
                "溢出项 `max(0, bucket−W_max)` 应把它压回容量 ⇒ 超容 = 径流项没接上"
                "（静默蓄水 = 判据②假象）。")

    # ---- 上下采样（v26 §四-16：面积守恒）------------------------------------

    def _fine2d(self, fine) -> NDArray[np.float64]:
        """把主网格平铺场读成 (rows_f, cols_f)，**极行按规范槽广播**（模块头「极点口径」①）。

        非规范极槽 = 未定义历史值（`rc_to_flat` 把极点整行坍缩成 col 0）⇒ 一律不吃。
        采样与积分**共用本读法** ⇒ `Σ_细 w·f == Σ_粗 area·F` 对任意输入恒成立
        （不要求极行预先均匀，守恒判据不留后门）。
        """
        f = np.asarray(fine, dtype=np.float64)
        if f.size != self.rows_f * self.cols_f:
            raise ValueError(
                f"主网格场尺寸 {f.shape}（{f.size} 元素）≠ {self.rows_f * self.cols_f}"
                f"（{self.rows_f}×{self.cols_f}）—— 网格/因子与入参不一致，宁炸")
        f2 = f.reshape(self.rows_f, self.cols_f).copy()
        f2[0, :] = f2[0, 0]
        f2[-1, :] = f2[-1, 0]
        return f2

    def coarsen(self, fine) -> NDArray[np.float64]:
        """主网格 → 粗网格：**面积权分块均值**（极行按「一圈一格」取全带均值）。

        参数：`fine` = 主网格平铺 (rows_f*cols_f,)（引擎各场一律展平存储）。
        返回：(rows_c, cols_c) 粗网格 2-D 场。

        守恒性：输出满足 `Σ_粗 area·F == total_fine(fine)`（到浮点舍入，相对残差 ~1e-16），
        因为 `area_c` 就**定义**为块内细格面积之和、而 `F_c = Σ_块 w·f / Σ_块 w`。
        """
        f2 = self._fine2d(fine)
        s = self.factor
        col_mean = f2.reshape(self.rows_f, self.cols_c, s).mean(axis=2)     # (rows_f, cols_c)
        # 纬度方向：把相邻 `s` 个细行折成一块（行主序 ⇒ (rows_c, s, cols_c) 的分组
        # 恰好就是"连续 s 行"），再按 cos φ 面积权加权（🔴 漏乘 cos φ = v26 §四-3 头号坑）
        block = col_mean.reshape(self.rows_c, s, self.cols_c)
        out = (np.einsum("rs,rsc->rc", self._w_block, block)
               / self._den[:, None])                                        # atm:sample
        # 🔴 极点行 = 一圈一格 ⇒ 块 = 该纬度带**全列**（列分块会把带内真实经度起伏
        #   塞进同一个极点格的 240 个槽 = 重复计数；见模块头「极点口径」②）
        g_all = f2.mean(axis=1)                                  # 每细行的经度均值
        pole_val = float((self._w_block[0] * g_all[:s]).sum() / self._den[0])
        out[0, :] = pole_val
        pole_val_s = float((self._w_block[-1] * g_all[-s:]).sum() / self._den[-1])
        out[-1, :] = pole_val_s
        return out

    def refine(self, coarse) -> NDArray[np.float64]:
        """粗网格 → 主网格：**分片常数**（有限体积零阶重构；逐块积分严格守恒）。

        参数：`coarse` = 粗网格场，2-D (rows_c, cols_c) 或展平 (n_c,) 皆可（同 `total_coarse`）。
        返回主网格平铺 (rows_f*cols_f,)。极带（首/末 `factor` 行）取粗极点行**规范槽**值
        广播 ⇒ 主网格极带单值（与 `rc_to_flat` 的极点坍缩同义；模块头「极点口径」③）。
        """
        c = np.asarray(coarse, dtype=np.float64)
        if c.size != self.n_c:
            raise ValueError(
                f"粗网格场尺寸 {c.shape}（{c.size} 元素）≠ {self.n_c}"
                f"（{self.rows_c}×{self.cols_c}）—— 网格/因子与入参不一致，宁炸")
        c = c.reshape(self.rows_c, self.cols_c)
        s = self.factor
        out = np.repeat(np.repeat(c, s, axis=0), s, axis=1)             # atm:sample
        out[:s, :] = c[0, 0]
        out[-s:, :] = c[-1, 0]
        return out.ravel()

    def refine_bilinear(self, coarse) -> NDArray[np.float64]:
        """粗网格 → 主网格：**面积权重心双线性**（🔴 诊断场专用，不用于守恒量）。

        与 `refine()` 的职责分工（v37/B2.1）：
        * `refine()`  = 分片常数（零阶重构）⇒ 块内方差恒 0、`coarsen(refine(F))` **严格还原**
          ⇒ 守恒量（总质量/总能量）只能走它；
        * `refine_bilinear()` = 四邻**凸组合**（权重非负、和 1）⇒ 块内方差 > 0（下游读取
          `t_surf`/`p_surf`/`u`/`qv`/降水率时更平滑），但 `coarsen(refine_bilinear(F))`
          **不再还原**（不是投影算子）。

        凸组合 ⇒ **输出必落在原场 `[min, max]` 内**（不产生新极值，v26 §四-15 同族要求）。
        几何：细格中心在粗坐标里的位置 `(r+0.5)/s − 0.5`；行方向越界钳到端点粗行
        （极带取极点行的经向插值），列方向按经度周期环绕。
        """
        c = np.asarray(coarse, dtype=np.float64)
        if c.size != self.n_c:
            raise ValueError(
                f"粗网格场尺寸 {c.shape}（{c.size} 元素）≠ {self.n_c}"
                f"（{self.rows_c}×{self.cols_c}）—— 网格/因子与入参不一致，宁炸")
        c = c.reshape(self.rows_c, self.cols_c)
        s = self.factor
        rc, cc = self.rows_c, self.cols_c
        fi = (np.arange(self.rows_f) + 0.5) / s - 0.5          # 细行 → 粗行坐标
        fj = (np.arange(self.cols_f) + 0.5) / s - 0.5          # 细列 → 粗列坐标
        i0 = np.floor(fi).astype(np.int64)
        ti = fi - i0
        j0 = np.floor(fj).astype(np.int64)
        tj = fj - j0
        i0c = np.clip(i0, 0, rc - 1)
        i1c = np.clip(i0 + 1, 0, rc - 1)                       # 极带外 ⇒ 钳到端点（凸）
        j0c = j0 % cc
        j1c = (j0 + 1) % cc                                    # 经度周期（接缝普通）
        w00 = (1.0 - ti)[:, None] * (1.0 - tj)[None, :]
        w01 = (1.0 - ti)[:, None] * tj[None, :]
        w10 = ti[:, None] * (1.0 - tj)[None, :]
        w11 = ti[:, None] * tj[None, :]
        out = (w00 * c[np.ix_(i0c, j0c)] + w01 * c[np.ix_(i0c, j1c)]
               + w10 * c[np.ix_(i1c, j0c)] + w11 * c[np.ix_(i1c, j1c)])
        return out.ravel()

    def total_fine(self, fine) -> float:
        """主网格平铺场的**面积权积分** `Σ_细 w·f`（守恒校核的左端）。

        🔴 极行经 `_fine2d` 取规范槽 ⇒ 与 `coarsen` 同一把尺；非规范极槽的垃圾不进积分。
        """
        return float((np.repeat(self._w_fine, self.cols_f) * self._fine2d(fine).ravel()).sum())

    def total_coarse(self, coarse) -> float:
        """粗网格场的**面积权积分** `Σ_粗 area·F`（守恒校核的右端；接受 2-D 或展平）。"""
        c = np.asarray(coarse, dtype=np.float64).ravel()
        return float((np.repeat(self.area, self.cols_c) * c).sum())

    def awm(self, arr) -> float:
        """粗网格面积权均值（全项目归一纪律：f_g/f_c/band 都用它，口径必须一致）。"""
        a = np.asarray(arr, dtype=np.float64).ravel()
        w = np.repeat(self.area, self.cols_c)
        return float((w * a).sum() / w.sum())

    def aw_corr(self, a, b) -> float:
        """**面积权** Pearson 相关系数（同 `awm` 一把尺）。

        只服务判据（§四-8/9：w 与 T 异常正相关、P 与 T 反号相关），不被计算路径调用；
        分母为 0（常数场）⇒ 返回 `nan` 由调用方显式处理，不静默给 0（= "无相关"假象）。
        """
        w = np.repeat(self.area, self.cols_c)
        x = np.asarray(a, dtype=np.float64).ravel()
        y = np.asarray(b, dtype=np.float64).ravel()
        xm = (w * x).sum() / w.sum()
        ym = (w * y).sum() / w.sum()
        dx, dy = x - xm, y - ym
        den = float(np.sqrt((w * dx * dx).sum() * (w * dy * dy).sum()))
        return float((w * dx * dy).sum() / den) if den > 0.0 else float("nan")

    def awm_band(self, arr, lat_lo: float, lat_hi: float) -> float:
        """**|纬度| ∈ [lat_lo, lat_hi] 带内**的面积权均值（🔴 与 `awm` 同一把 `area` 尺）。

        只服务 §三 的单胞环流读数（赤道带/副热带带的辐合辐散分开问）。带界按 `mu_ctr`
        的**面积质心**纬度取，不是行心 ⇒ 极帽那行不会被当成"一整块高纬"重复计数。
        权重 = `np.repeat(area, cols_c)`（**与 `awm` 同一个数组**，不是"每行一个权重"）⇒
        `awm_band(arr, 0, 90)` 与 `awm(arr)` 相等到**舍入**（求和顺序不同：1-D 展平 vs 2-D
        掩码）；分子按格求和、分母只按行求和会差整整一个 `cols_c`（= 120 倍），有符号场还
        看不出来 ⇒ 恒等式与分带可加性两条都要钉（W30）。
        🔴 纬度取**绝对值** ⇒ 南北两半球的同名带一起进：v26 §三 的判据本身是"关于赤道对称"
        的两个带，只挑北半球在对称场上给出的数与全球几乎一样，口径错也看不出来（本函数的
        第一版就栽在这里，由 W30 的全带恒等式抓出）。
        空带 ⇒ `ValueError`（静默返回 0 会把"粗网格太稀、该带没有行"伪装成"均值为 0"）。
        """
        lat = np.abs(np.degrees(np.arcsin(self.mu_ctr)))
        m = (lat >= lat_lo) & (lat <= lat_hi)
        if not bool(m.any()):
            raise ValueError(
                f"|纬度|∈[{lat_lo},{lat_hi}]° 带内没有粗网格行（rows_c={self.rows_c}"
                f"，质心纬度覆盖到 {float(lat.max()):.2f}°）—— 换带界或加分辨率，不给 0")
        w = np.repeat(np.asarray(self.area, dtype=np.float64), self.cols_c)
        w2 = w.reshape(self.rows_c, self.cols_c)[m]
        a2 = np.asarray(arr, dtype=np.float64).reshape(self.rows_c, self.cols_c)[m]
        return float((w2 * a2).sum() / w2.sum())

    def awm_band_nan(self, arr, lat_lo: float, lat_hi: float) -> float:
        """`awm_band` 的**读数专用**变体：带内没有行 ⇒ `nan`（口径本身仍是 `awm_band`）。

        ⚠️ 为什么这一处允许不炸：`awm_band` 的 `ValueError` 是对的（"粗网格太稀、该带没有行"
        必须在**判据**处暴露，不许伪装成"均值为 0"）。但 `probe()` 要在微型档也能跑
        （rows_c=3 只有 |φ|=0/48.6° 两档 ⇒ 5–20° 信风带根本不存在），返回 `nan` 与
        `aw_corr`（常数场）、`u_eq_ratio_awm`（零力）同族 ⇒ **nan = "这条问不了"**，
        不是"答案是 0"、也不是"没算"。🔴 判据一律在带内确有行的档上跑（单测钉档名与行数）。
        """
        try:
            return self.awm_band(arr, lat_lo, lat_hi)
        except ValueError:
            return float("nan")

    def zonal_mean(self, arr) -> NDArray[np.float64]:
        """粗网格纬带均值（每行按经度算术均值；极行本就单值 ⇒ 无特殊分支）。"""
        return np.asarray(arr, dtype=np.float64).reshape(self.rows_c, self.cols_c).mean(axis=1)

    # ---- P-eq 装置（owner v35 §二-2 首件）：赤道线剖面 + 镜像残差（全只读）--------

    def equator_rows(self) -> tuple[int, int]:
        """赤道所涉粗网格行号 `(i_south, i_north)`（奇偶两档都定义；`(m, m)` = 赤道是行）。

        🔴 用**行列数奇偶**判定，不用 `argmin|μ_ctr|`、也不用"μ 恰好等于 0"的浮点比较：
        * `rows_c` **偶** ⇒ 赤道落在两行之间的**面**上 ⇒ 返回 `(mid−1, mid)`，
          赤道线读数取这两行的**算术平均**（与 `div_u` 的面速度 `0.5·(u[i]+u[i+1])`
          同一套口径 ⇒ 剖面的尺 = 散度的尺）；
        * `rows_c` **奇** ⇒ 赤道是**中心行本身** ⇒ 返回 `(mid, mid)`，直接取该行。
        `argmin` 在偶档会**偏向固定一侧**（两行到 0 的距离完全相等 ⇒ 结果取决于实现）⇒
        奇偶两档的"赤道"不再是同一个几何对象，判据换分辨率就不可比（本模块的极行/带界
        纪律同族：先钉死问法，再读数）。校核只查"赤道是否被这两行夹住"（严格不等号 ⇒
        容得下 `mu_ctr` 的 ±0.0 与 1e-17 级质心噪声，不吃"恰好为零"）。
        """
        rc = self.rows_c
        mid = rc // 2
        if rc % 2:
            if not (self.mu_ctr[mid - 1] < 0.0 < self.mu_ctr[mid + 1]):
                raise FloatingPointError(
                    f"rows_c={rc} 的中心行不是赤道行（μ_ctr={float(self.mu_ctr[mid]):.3e}，"
                    f"邻行 {float(self.mu_ctr[mid - 1]):.3e}/{float(self.mu_ctr[mid + 1]):.3e}"
                    "）—— 网格几何已不对称，宁炸")
            return mid, mid
        if not (self.mu_ctr[mid - 1] < 0.0 < self.mu_ctr[mid]):
            raise FloatingPointError(
                f"rows_c={rc} 的第 {mid - 1}/{mid} 行没有夹住赤道（μ_ctr="
                f"{float(self.mu_ctr[mid - 1]):.3e}/{float(self.mu_ctr[mid]):.3e}）"
                "—— 网格几何已不对称，宁炸")
        return mid - 1, mid

    def equator_profile(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """赤道线上的 **(u_e, u_n) 经度剖面**（各 `cols_c` 个；纯读数，零状态）。

        偶档 = 赤道面两侧两行的算术平均（见 `equator_rows`）；奇档 = 中心行。
        返回的是**节点场的面平均**，不是再插值 ⇒ 与 `div_u`/`_pressure_gradient` 的
        面索引口径一致，"赤道无风"与"经向通量过零"用的是同一条线。
        """
        rc, cc = self.rows_c, self.cols_c
        i_s, i_n = self.equator_rows()
        ue = self.u_east.reshape(rc, cc)
        un = self.u_north.reshape(rc, cc)
        if i_s == i_n:
            return ue[i_s].copy(), un[i_s].copy()
        return 0.5 * (ue[i_s] + ue[i_n]), 0.5 * (un[i_s] + un[i_n])

    def mirror_rel(self, arr, odd: bool = False) -> float:
        """南北镜像的**相对残差** `max|arr ∓ flipud(arr)| / max|arr|`（P-eq 对称性判据口径）。

        行序反转 = 关于赤道的镜像（偶档把赤道面映到自身、奇档把中心行映到自身 ⇒ 两档都成立）。
        `odd=True` 用于**反对称**场（`u_n`：越过赤道换向 ⇒ 比较 `arr + flipud(arr)`）。
        分母取 `max|arr|` 而不是面积权均值 ⇒ 与 s2 [实测] 读数、W23 的断言**同一个尺**，
        历史数字可直接续用。恒零场 ⇒ `0.0`（镜像恒成立，不是"没算"；给 nan 反而让
        判据在零模型①档下变成不可判）。
        """
        a2 = np.asarray(arr, dtype=np.float64)
        if a2.shape != (self.rows_c, self.cols_c):
            raise ValueError(
                f"镜像残差要求粗网格 2-D 场 {self.rows_c}×{self.cols_c}，收到 {a2.shape}")
        scale = float(np.abs(a2).max())
        if scale == 0.0:
            return 0.0
        dif = a2 + a2[::-1] if odd else a2 - a2[::-1]
        return float(np.abs(dif).max() / scale)

    def collapse_poles(self, arr2d) -> None:
        """把 2-D 场的极点行回写成"经度面积权均值"（= 一圈一格的单值闭合）。

        极点行各槽面积**相等** ⇒ 取均值是**守恒**操作（不改变 `total_coarse`）；
        物理上防的是：驱动项一旦带经度起伏（如 α 逐格赋值），极点格会被赋予
        cols_c 个互相矛盾的值（v26 §四-13 的"重复计数"变体）。
        """
        arr2d[0, :] = float(arr2d[0].mean())
        arr2d[-1, :] = float(arr2d[-1].mean())

    # ---- 节拍与 EBM 时间推进（唯一全场路径；引擎按 `every_ticks` 调）----------

    def _laplacian(self, t2: NDArray[np.float64]) -> NDArray[np.float64]:
        """经向扩散 `D ∂_μ[(1−μ²) ∂_μ T]` 的**通量式有限体积**离散（自伴，面积尺 `area`）。

        `LT_i = (τ_{i+½}(T_{i+1}−T_i) − τ_{i−½}(T_i−T_{i−1})) / area_i`，τ 对称 ⇒
        ① `Σ_i area_i·LT_i = 0`（望远镜相消，到**机器精度** ⇒ 扩散不造热不漏热，v26 §四-1/2）；
        ② `Σ_i area_i·T_i·LT_i = −Σ_面 τ·(ΔT)² ≤ 0`（纯耗散，§四-2）；
        ③ 极点只有一面通量（μ=±1 处 1−μ²=0 ⇒ 边界通量恒零）⇒ 极帽不被重复计数（§四-13）。
        与连续算子的偏差：l=2 勒让德模实测 1.7e-4（s2 档，含极行）⇒ 写进汇报。
        """
        out = np.zeros_like(t2)
        f = self.tau[:, None] * (t2[1:] - t2[:-1])                  # 面通量（未乘面积）
        out[1:-1] = (f[1:] - f[:-1]) / self.area[1:-1, None]
        out[0] = f[0] / self.area[0]
        out[-1] = -f[-1] / self.area[-1]
        return out

    def _ebm_substep(self) -> None:
        """一个 Δt 子步：`C·ΔT = [S(1−α_eff) − (A+B·T) + D∇²T]·Δt`（Forward Euler）。

        🔴 单位口径（v26 §四-4，最隐蔽的"违背自然规律"位）：
        * 本函数的 `T` 一律 **℃**。理由不是"项目场是 ℃"，而是 A=210/B=2.0 这组线性化
          常数**本身**以 ℃ 为零点拟合（v26 §二 自己的解析锚 (340×0.7−210)/2=+14 **℃**
          只有在 ℃ 下成立；代 K 得 OLR≈784 W/m² ⇒ T̄→−273 ⇒ 荒谬）。
        * 任何走**状态方程/理想气体**的量（Step 3 的 P=ρRT）必须先 `t_kelvin()` 换 K。
        * 下界 `_OLR_T_FLOOR_C = −A/B = −105 ℃`：越界 ⇒ 线性化 OLR 变负 = 模型在外推区
          ⇒ fail-loud（不 clip、不静默）。

        🔴 为何显式而非隐式（v26 §二/§四-6 写"隐式扩散"）：实测本参数组的显式扩散数
        `μ = dt·Στ/(C·area) ≈ 1.6×10⁻²`（门限 0.5 的 3.3%）⇒ 稳定性根本不约束步长，
        而隐式三对角解在 s2 实测**单项就 ≈ +3.3%/tick**（> §二 预算 +2.8%）。
        构造期把 μ 做成硬校核（`_DIFF_STABILITY_MAX`）⇒ 谁把 dt/节拍改进不稳区就炸，
        这比"默认隐式"更能兑现 §四-6 的真实意图（"不稳定先查步长"）。冲突呈报话事人。
        """
        rc, cc = self.rows_c, self.cols_c
        t2 = self.t_surf.reshape(rc, cc)
        if not t2.flags["C_CONTIGUOUS"]:                            # 理论不发生（防御性）
            raise ValueError("t_surf 视图非 C 连续 ⇒ 原地写回不安全，宁炸")
        ab = self.absorbed.reshape(rc, cc)
        hc = self.heat_cap.reshape(rc, cc)
        lt = (self._laplacian(t2) if self.cfg.diffusion_on else 0.0)
        t2 += self.dt_s * (ab - (_EBM_A_WM2 + _EBM_B_WM2K * t2) + lt) / hc   # atm:substep
        # 🔴 非负/有限 fail-loud（v26 §四-15）：NaN/Inf/T(K)≤0/OLR 下界越界 ⇒ 直接报错
        t_min, t_max = float(t2.min()), float(t2.max())
        if not (np.isfinite(t_min) and np.isfinite(t_max)):
            raise FloatingPointError(
                f"EBM 子步后温度出现非有限值（min={t_min}, max={t_max}）——"
                "禁止静默截断/替换（v26 §四-15）。")
        if t_min + _KELVIN0 <= 0.0:
            raise FloatingPointError(
                f"T={t_min:.3f} ℃ ⇒ 绝对温度 ≤ 0 K（v26 §四-15 非负硬门）。")
        if t_min < _OLR_T_FLOOR_C:
            raise FloatingPointError(
                f"T={t_min:.3f} ℃ 低于长波线性化有效下界 {_OLR_T_FLOOR_C:.1f} ℃"
                "（OLR=A+B·T 已变负 = 模型在外推区，v26 §四-4/15）。")
        # 极行"一圈一格"闭合：只在真的出现经度起伏时才回写（均匀行保持逐位不变）
        if float(t2[0].max() - t2[0].min()) > 0.0 or float(t2[-1].max() - t2[-1].min()) > 0.0:
            self.collapse_poles(t2)

    def advance(self, tick: int) -> None:
        """推进一个大气时刻（= `substeps` 个 Δt 子步；split-explicit，子步与主 tick 解耦）。

        Step 2 = 只跑 EBM（温度场）；Step 3a 在每个子步的 EBM 之后**诊断**气压
        （P = f(T) 的纯函数 ⇒ 3 次诊断里只有末次留下，但顺序固定 = 确定性可重建）；
        Step 3b 在每个子步的气压之后**积分**风（u 是状态量 ⇒ 三个子步的受力**累积**，
        不像 P 那样被覆盖 ⇒ 快照必须带 u，见 `state()`）；
        **C4a-2 Step A** 在风之后追加一个**被动**水分子步（只读 T/P/u/mass_col，
        不写它们 ⇒ 同 Δt 内 `T→P→u` 的因果链一字不动）。
        🔴 因果顺序 = v26 §二：`T → P → u`，**不许**颠倒（风不加热、不平流）。
        ⚠️ 水分步的**位置**是判据的一部分：放在 `_integrate_wind()` 之后 ⇒ 水汽读到的是
        本子步刚积分出的风（同 Δt 内的风，不是上一 Δt 的）；而热机链完全不读水汽 ⇒
        `moisture_on` 开/关两档的 `t_surf`/`p_surf`/`u_*` 在任意 tick **逐位相同**
        （= 本 Step 的可证伪判据，单测 m18 钉死；放前面也一样成立，但口径要跟着改）。
        🔴 本批大气**不写任何主网格场** ⇒ 开档与关档的主 tick 仍逐位等价（C7 侧翼判据），
        演化只发生在自己的粗网格载体上（单测 E11 钉死这两条同时成立）。
        """
        for _ in range(self.substeps):
            self._ebm_substep()
            self._diagnose_pressure()                       # atm3a:advance-diag
            self._integrate_wind()                          # atm3b:advance-wind
            if self.moisture_on:
                self._moisture_substep()                    # atm4:advance-moist
        self.steps += int(self.substeps)
        self.last_tick = int(tick)

    def t_kelvin(self) -> NDArray[np.float64]:
        """粗网格温度的**开尔文**视图（+273.15）—— Step 3 的状态方程只许用它。"""
        return self.t_surf + _KELVIN0

    def tendency(self, t2: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """诊断用：返回当前 `dT/dt`（K/s，(rows_c, cols_c)）；不改状态、不被计算路径调用。"""
        cur = (self.t_surf.reshape(self.rows_c, self.cols_c) if t2 is None
               else np.asarray(t2, dtype=np.float64).reshape(self.rows_c, self.cols_c))
        lt = (self._laplacian(cur) if self.cfg.diffusion_on else 0.0)
        return ((self.absorbed.reshape(self.rows_c, self.cols_c)
                 - (_EBM_A_WM2 + _EBM_B_WM2K * cur) + lt)
                / self.heat_cap.reshape(self.rows_c, self.cols_c))

    def due(self, tick: int) -> bool:
        """该主 tick 是否轮到大气推进（相位 = `tick % every_ticks == 0`）。

        单独成方法 = 让"触发相位"成为**可钉死判据**的对象（单测直接扫 1..N），
        引擎侧钩子与本方法同式（改一处必改另一处 ⇒ 不许在引擎里另写判据）。
        """
        return int(tick) % self.every_ticks == 0

    # ---- 快照（v26 §四-18；`smell_field.state/restore` 先例）---------------

    def state(self) -> dict:
        """动态场 + 节拍计数（引擎侧写成 `atm_*` 键；**关档时引擎不构造 ⇒ 无键**）。

        🔴 Step 3a 的 `p_surf`/`mass_col` **不进档**：它们是 `t_surf` 的纯函数（诊断量），
        由 `restore()` 逐位重算 ⇒ 满足 v26 §四-18 的"进快照**或**确定性重建"二选一，
        且不放大快照体积（同 Step 2 的 α/S/C/τ 静态驱动场口径）。
        🔴 Step 3b 的 `u_east`/`u_north` **必须进档**：它们是时间积分的状态量（同一 P 场
        在不同历史下给出不同风）⇒ 不进档则"续跑 ≠ 连续跑"（§四-18 的另一分支）。
        🔴 **C4a-2 Step A** 的 `qv`/`bucket`/`wb` 同理必须进档（层均比湿与陆面水桶是累积状态，
        账本是"这一批总共蒸/降/径流了多少 mm"的唯一来源）；`water_frac`/`p_rate`/`land_dist`
        是**静态驱动面**（由 C0b 掩码 + `pr` + 网格定 ⇒ 构造期重建，不进档，同 α/S/C 口径）。
        ⚠️ 关档（`moisture_on=False`）**不写这三个键** ⇒ 旧档/关档档的快照形状逐位不变
        （T5 走 A 支 = **不升 `SNAPSHOT_VERSION`**；理由与"缺键回退"见 `restore()`）。
        """
        out = {
            "t_surf": self.t_surf.copy(),
            "u_east": self.u_east.copy(),
            "u_north": self.u_north.copy(),
            "steps": np.array(self.steps, dtype=np.int64),
            "last_tick": np.array(self.last_tick, dtype=np.int64),
        }
        if self.moisture_on:
            out["qv"] = self.qv.copy()                          # atm4:state
            out["bucket"] = self.bucket.copy()
            out["wb"] = self.wb.copy()
        return out

    def restore(self, payload) -> None:
        """从快照恢复（形状不符 ⇒ **fail-loud**，不静默重置；smell 同口径）。

        恢复场后立即重算诊断量 ⇒ "续跑首个推进之前"的读数与连续跑一致（P6 单测钉死）。
        `u_east`/`u_north` **缺键** = Step 3a 及更早的旧档 ⇒ 风从**静止**起算
        （`smell_Sc` 先例；与"新档本来就从静止起算"是同一口径，不是第三条物理）。
        `qv`/`bucket`/`wb` 同理：**缺键** = Step A 之前的旧档，或该档本就关着水汽 ⇒
        从**干起算**（`qv=0`）、**空桶**、**零账本**。⚠️ 这条回退的语义边界必须写实：
        给旧档开水汽续跑，前若干 tick 没有水汽柱（干球），它与"从头开档跑"**不可比**；
        回退是为兼容而做的让步，不是"等价重建"（判据批次一律从 `from_dict` 新建，
        不靠旧档续跑 ⇒ 单测 m17 只钉"缺键不炸 + 有键逐位回"两条，不钉可比性）。
        🔴 水汽三键的**值域**校验与写入侧不变式同语义（复核件 §六-1 ⇒ owner 2026-10-17 令）：
        `qv` 非负 + 有限；`bucket` 有限，且**只在 `water_limit_on=True` 时**要求非负
        （零模型⑤的反证臂按设计带负桶，单测 m11/m27）；`wb` 六槽有限，**不**要求非负
        （`Ocean` 是净收支、`imbalance` 是带符号残差）。越界一律抛错，不 clip、不静默重置。
        """
        t = np.asarray(payload["t_surf"], dtype=np.float64)
        if t.shape != self.t_surf.shape:
            raise ValueError(
                f"快照大气场形状 {t.shape} ≠ 当前粗网格 ({self.n_c},)"
                "（网格/因子与存档不一致 ⇒ 续跑会静默错位，宁炸）。")
        self.t_surf[:] = t
        for key, dst in (("u_east", self.u_east), ("u_north", self.u_north)):
            v = payload.get(key)
            if v is None:
                dst[:] = 0.0                                      # atm3b:restore-static
                continue
            v = np.asarray(v, dtype=np.float64)
            if v.shape != dst.shape:
                raise ValueError(
                    f"快照风场 `{key}` 形状 {v.shape} ≠ 当前粗网格 ({self.n_c},)"
                    "（同 `t_surf` 口径：网格/因子不符宁炸，不做静默 reshape）。")
            dst[:] = v
        if self.moisture_on:                                      # atm4:restore
            for key, dst in (("qv", self.qv), ("bucket", self.bucket), ("wb", self.wb)):
                v = payload.get(key)
                if v is None:
                    dst[:] = 0.0                                  # 干起算 / 空桶 / 零账本
                    continue
                v = np.asarray(v, dtype=np.float64)
                if v.shape != dst.shape:
                    raise ValueError(
                        f"快照水汽场 `{key}` 形状 {v.shape} ≠ 当前粗网格 ({self.n_c,}) / "
                        f"账本 ({self.wb.shape})（同 `t_surf` 口径，不做静默 reshape）。")
                dst[:] = v
            if float(self.qv.min()) < 0.0:
                raise FloatingPointError(
                    f"快照层均比湿为负（min={float(self.qv.min()):.3e} kg/kg）—— 平流保正 + "
                    "`min(柱水,·)` 限汇 ⇒ 档里带负水汽说明写入侧被改坏，宁炸不 clip（红线 4）。")
            if not bool(np.all(np.isfinite(self.qv))):
                raise FloatingPointError("快照比湿含 NaN/Inf（v26 §四-15 非有限硬门）。")
            # ---- `bucket` / `wb` 的同语义校验（复核件 §六-1 ⇒ owner 2026-10-17 转达令）----
            # 口径 = "档里的值必须落在**写入侧不变式**的范围内"，与 `qv` 一条尺；两条必须写实的
            # 例外（不是放松，是同一实现不变式的另一半）：
            # * 桶的非负只在 `water_limit_on=True` 下由构造保证（`min(潜蒸发, 桶)` ⇒ 取水 ≤ 存量，
            #   与 `_moisture_substep` 第 6 步的校核同条件）。零模型⑤**故意**让桶透支为负
            #   （单测 m11 的反证臂）⇒ 该档的快照必须能续跑，这里只校有限性、不校非负。
            # * 账本六槽 `[E, P, R, Ocean, imbalance, P_pot]` 里 `Ocean`（海洋库净收支）与
            #   `imbalance`（残差，带符号）**按定义可负** ⇒ 对它们做"非负"会误杀合法账本，
            #   故本槽只校有限性。
            if not bool(np.all(np.isfinite(self.bucket))):
                raise FloatingPointError("快照陆面水桶含 NaN/Inf（v26 §四-15 非有限硬门）。")
            if bool(self.cfg.water_limit_on) and float(self.bucket.min()) < 0.0:
                raise FloatingPointError(
                    f"快照陆面水桶为负（min={float(self.bucket.min()):.3e} kg/m² of land）—— "
                    "开水量约束时 `min(潜蒸发, 桶)` 保证取水 ≤ 存量 ⇒ 档里带负桶 = 写入侧被改坏，"
                    "或该档是零模型⑤（`water_limit_on=False`）却用开约束的配置续跑（口径错配）。"
                    "宁炸不 clip（红线 4）。")
            if not bool(np.all(np.isfinite(self.wb))):
                raise FloatingPointError(
                    f"快照水量账本含 NaN/Inf（wb={np.array2string(self.wb)}）—— 六槽都必须是"
                    "有限累积量（`imbalance` 是可正可负的**残差读数**，不是允许 NaN 的占位；"
                    "v26 §四-15 非有限硬门）。")
        self.steps = int(payload["steps"])
        self.last_tick = int(payload["last_tick"])
        self._diagnose_pressure()                             # atm3a:restore-diag

    # ---- 读数（只读；不进任何计算路径）------------------------------------

    def probe(self) -> dict:
        """骨架验收读数（网格/节拍/守恒残差/载体状态/气压/风）。

        🔴 只在**开档**被调用（关档引擎不构造本模块 ⇒ 旧 probe/manifest 逐位不变）。
        🔴 只读：内部调用 `wind_forces()`/`div_u()` 都是 `p_surf`/`u_*` 的**纯函数**
        （重算力、不写状态）⇒ 单测 P12/W13 钉"probe 前后所有场与计数逐位不变"。
        """
        w_fine = np.repeat(self._w_fine, self.cols_f)
        one = np.ones(self.rows_f * self.cols_f, dtype=np.float64)
        fe, fn = self.wind_forces()                             # 纯函数，不改状态
        dv = self.div_u()                                       # 同上（§四-11 读数的来源）
        wx = self.vertical_velocity()                           # = −z_b·dv，仍只读
        ue2 = self.u_east.reshape(self.rows_c, self.cols_c)
        un2 = self.u_north.reshape(self.rows_c, self.cols_c)
        spd = np.sqrt(ue2 * ue2 + un2 * un2)
        fmag = np.sqrt(fe * fe + fn * fn)
        tau = float(self.cfg.rayleigh_tau_s)
        f_awm, u_awm = self.awm(fmag), self.awm(spd)
        # 平衡风倍数 `u / (τ·|F|)`：开阻尼且批够长 ⇒ →1；`damping_on=False`（零模型③）⇒
        # 无界增长（该读数就是"摩擦是必需项"的证据）。🔴 力恒零（**零模型①**：均匀驱动
        # ⇒ 无梯度 ⇒ 根本没有平衡风可比）⇒ 返回 `nan` 而不是 0/0 崩或"0 除 0 = 达标"假象。
        u_eq_ratio = (u_awm / (tau * f_awm)
                      if (tau > 0.0 and f_awm > 0.0 and np.isfinite(f_awm)) else float("nan"))
        # ---- P-eq 读数（只读；owner v35 §二-2 首件）----
        i_eq_s, i_eq_n = self.equator_rows()
        eq_u, eq_v = self.equator_profile()
        eq_spd = np.sqrt(eq_u * eq_u + eq_v * eq_v)
        t2 = self.t_surf.reshape(self.rows_c, self.cols_c)
        p2 = self.p_surf.reshape(self.rows_c, self.cols_c)
        spd_calm = self.awm_band_nan(spd, 0.0, _BAND_CALM_MAX_DEG)
        spd_trade = self.awm_band_nan(spd, _BAND_CALM_MAX_DEG, _BAND_TRADE_MAX_DEG)
        # 无风带/信风侧**速率比**：信风侧带内无行（微型档）或恒零风 ⇒ nan（"问不了"），不是 1
        calm_ratio = (spd_calm / spd_trade
                      if (np.isfinite(spd_trade) and spd_trade > 0.0) else float("nan"))
        rep = {
            "rows_f": self.rows_f, "cols_f": self.cols_f, "factor": self.factor,
            "rows_c": self.rows_c, "cols_c": self.cols_c, "n_c": self.n_c,
            "every_ticks": self.every_ticks, "substeps": self.substeps,
            "dt_s": self.dt_s, "steps": self.steps, "last_tick": self.last_tick,
            "t_min": float(self.t_surf.min()), "t_max": float(self.t_surf.max()),
            "t_awm": self.awm(self.t_surf),
            "area_total_fine": float(w_fine.sum()),
            "area_total_coarse": float(np.repeat(self.area, self.cols_c).sum()),
            # 单位场的积分残差（相对）：coarsen(1)=1 且两端面积和恒等 ⇒ 采样对守恒的自检
            "area_rel_residual": abs(float((w_fine * one).sum())
                                     - self.total_coarse(np.ones((self.rows_c, self.cols_c))))
            / max(float((w_fine * one).sum()), 1e-300),
            # ---- Step 2：EBM 读数（收支 / 自伴 / 稳定性 / 剖面）----
            "ebm_A_WM2": _EBM_A_WM2, "ebm_B_WM2K": _EBM_B_WM2K,
            "ebm_D_WM2K": _EBM_D_WM2K, "ebm_C_ref": _C_REF,
            "albedo_closure": str(self.cfg.albedo_closure),
            "albedo_offset": float(self.albedo_offset),
            "alpha_surface_awm": self.awm(self.alpha_surface),
            "alpha_eff_min": float(self.alpha_eff.min()),
            "alpha_eff_max": float(self.alpha_eff.max()),
            "insolation_mode": str(self.cfg.insolation_mode),
            "diffusion_on": bool(self.cfg.diffusion_on),
            "s_awm": self.awm(self.s_in),
            "absorbed_awm": self.awm(self.absorbed),
            "olr_awm": _EBM_A_WM2 + _EBM_B_WM2K * self.awm(self.t_surf),
            # 收支闭合残差（相对）：全球均值意义下 |∫吸收 − ∫OLR| 的失衡占通量的比例
            "imbalance_rel": abs(self.awm(self.absorbed) - self.awm(
                _EBM_A_WM2 + _EBM_B_WM2K * self.t_surf))
            / max(abs(self.awm(self.absorbed)), 1e-300),
            "heat_cap_min": float(self.heat_cap.min()),
            "heat_cap_max": float(self.heat_cap.max()),
            "mu_diff_max": self._mu_diff,
            "t_kelvin_min": float(self.t_kelvin().min()),
            "t_eq": float(self.t_surf.reshape(self.rows_c, self.cols_c)[
                int(np.argmin(np.abs(self.mu_ctr))), 0]),
            "t_pole_s": float(self.t_surf[0]), "t_pole_n": float(self.t_surf[-1]),
            "zonal_spread_max": float(np.max(
                self.t_surf.reshape(self.rows_c, self.cols_c).max(axis=1)
                - self.t_surf.reshape(self.rows_c, self.cols_c).min(axis=1))),
            # ---- Step 3a：气压 / 质量 proxy 读数（§四-9 反号 + §四-11 质量锚 + §四-15 非负）----
            "layer_depth_m": float(self.cfg.layer_depth_m),
            "hypso_k": self._hypso_k,
            "p_top_pa": self._p_z,
            "p_awm_pa": self.awm(self.p_surf),
            # 全球质量锚残差：`AWM(P_s)` 应恒等于 P₀（构造性归一 ⇒ 只测浮点求和次序）
            "p_awm_rel_residual": abs(self.awm(self.p_surf) - _P_REF_PAS) / _P_REF_PAS,
            "p_min_pa": float(self.p_surf.min()), "p_max_pa": float(self.p_surf.max()),
            "p_col_ratio_minus1": float(self.p_surf.max() / self.p_surf.min() - 1.0),
            # 🔴 §四-9 的**符号**判据：地面气压与柱温必须**反号相关**（暖区地面低压）。
            # `P=P₀·T/T₀` 的错写在这里必然给正号 ⇒ 该读数是"没有硬写 P∝T"的证据。
            "p_t_corr": self.aw_corr(self.p_surf, self.t_kelvin()),
            "p_t_eq_minus_pole_pa": float(
                self.p_surf.reshape(self.rows_c, self.cols_c)[
                    int(np.argmin(np.abs(self.mu_ctr))), 0] - self.p_surf[0]),
            "mass_awm_kg_m2": self.awm(self.mass_col),
            "mass_min_kg_m2": float(self.mass_col.min()),
            "mass_max_kg_m2": float(self.mass_col.max()),
            # ---- Step 3b：风（v30 ⑤.4 的 τ 必答读数 + §四-7 风向的能量侧证）----
            "rayleigh_tau_s": tau,
            "damping_on": bool(self.cfg.damping_on),
            "polar_c_floor": _POLAR_C_FLOOR,
            "polar_rows_filtered": int((self._c_ctr < _POLAR_C_FLOOR).sum()),
            # 一次推进 = substeps×dt_s 物理秒；占 τ 的比例 = "风跟不跟得上节拍"的定量回答
            "advance_s": self.dt_s * self.substeps,
            "advance_over_tau": self.dt_s * self.substeps / tau,
            "dt_over_tau": self._mu_damp,
            "force_max_ms2": float(fmag.max()),
            "force_awm_ms2": self.awm(fmag.ravel()),
            # 平衡风尺 `u_eq = τ·|F|`（Rayleigh 力平衡，非地转）—— 实测 `u/u_eq` 就是
            # "批内达到平衡的百分之几十"的直接读数（v30 ⑤.4 预言 2000 tick ≈ 65%）
            "u_eq_max_ms": tau * float(fmag.max()),
            "u_eq_awm_ms": tau * self.awm(fmag.ravel()),
            "u_max_ms": float(spd.max()),
            # ---- B4.1 判据② 风带：面积权带内平均（正 = 向东；东风 ⇒ 负）----
            "u_zonal_eq_ms": self.awm_band(ue2, 0.0, 30.0),
            "u_zonal_mid_ms": self.awm_band(ue2, 30.0, 60.0),
            "coriolis_on": bool(self.coriolis_on),
            "u_rms_awm_ms": float(np.sqrt(self.awm(spd * spd))),
            "u_eq_ratio_awm": u_eq_ratio,
            "ke_awm_m2_s2": 0.5 * self.awm(spd * spd),
            # ---- §四-11 质量连续：`Σ area·∇·u` 必须望远镜相消到**机器精度**（非 1e-6）----
            # 分母用 `Σ area·|∇·u|`（同一把尺）⇒ 读数就是"相对通量"的口径，与 Step 2 的
            # `Σarea·∇²T` 残差可比。风全零时两者同零 ⇒ 给 0.0（不是 nan，也不掩盖"没算"）。
            "div_area_sum_abs": float(np.abs(np.sum(self.area[:, None] * dv))),
            "div_area_flux_scale": float(np.sum(self.area[:, None] * np.abs(dv))),
            "div_area_sum_rel_residual": abs(float(np.sum(self.area[:, None] * dv)))
            / max(float(np.sum(self.area[:, None] * np.abs(dv))), 1e-300),
            "div_min_per_m": float(dv.min()), "div_max_per_m": float(dv.max()),
            "div_awm_per_m": self.awm(dv.ravel()),
            # ---- Step 3c：单胞环流（§三 结构判据 + §四-8 w 与 T 异常同号）----
            # 🔴 带界是**读数窗口**（模块常数 `_BAND_*`，不参与任何物理项）；赤道带应辐合
            # （div<0 ⇒ w>0 上升），副热带带应辐散（div>0 ⇒ w<0）。符号写反在这里必露。
            # ⚠️ 事实与判据的张力（v26 §三 的"副热带辐散"在本批结构下**取不到**）：
            # 辐散支起始 |φ| = 稳态装置 ≈39.8° / 批内瞬态 ≈53.2° ⇒ 25–45° 带均值在两种装置下
            # 都是**辐合上升**（`w_sub_band_ms > 0`），不是判据字面的辐散。机理 = 本批无科氏力
            # ⇒ 无角动量约束把下沉区压回 ~30°（那是 C4a-4 的内容）。带界不迁就读数、算法不为
            # 字面改写 ⇒ 作为 A/B/C 冲突项随汇报交 owner；单胞的**结构**判据（赤道辐合 +
            # 带外高纬辐散 + 关于赤道对称 + 形状不随时间漂移 + 两支面积权相消）全部成立。
            "w_top_absmax_ms": float(np.abs(wx).max()),
            "w_top_awm_ms": self.awm(wx.ravel()),
            "w_eq_band_ms": self.awm_band(wx, 0.0, _BAND_EQ_MAX_DEG),
            "w_sub_band_ms": self.awm_band(wx, _BAND_SUB_MIN_DEG, _BAND_SUB_MAX_DEG),
            "div_eq_band_per_m": self.awm_band(dv, 0.0, _BAND_EQ_MAX_DEG),
            "div_sub_band_per_m": self.awm_band(dv, _BAND_SUB_MIN_DEG, _BAND_SUB_MAX_DEG),
            "w_t_anom_corr": self.aw_corr(wx, self.t_surf - self.awm(self.t_surf)),
            # §四-13 极点不重复计数：四条极行的两个分量都必须经度恒单值（实测须 = 0.0）
            "u_polar_spread": float(max(
                ue2[0].max() - ue2[0].min(), ue2[-1].max() - ue2[-1].min(),
                un2[0].max() - un2[0].min(), un2[-1].max() - un2[-1].min())),
            # ---- P-eq（owner v35 §二-2 开工清单首件）：赤道线剖面 + 镜像对称 + 无风带 ----
            # 🔴 全部只读、不参与任何物理项；无风带/信风侧带界（5°/20°）与 `_BAND_EQ/_BAND_SUB`
            # 同族纪律 = **先钉死问法再读数**，不许事后挑对自己有利的宽度。
            # ⚠️ 镜像残差的地板必须写实（s2 [实测]：`u_n` 3.23e-13 过 1e-12、
            # `div` 2.10e-12 **不过** 1e-12）：残差来自 `_steady_t` 的 LU 三对角解，
            # 不在风/散度链 ⇒ 呈报不放宽、不改带界（迁就读数 = 禁区）。
            # ⚠️ 本批**无科氏力** ⇒ "赤道无风带"只是经向风为零的对称结果 + 纬向弱风侧；
            # 无风带随科氏参数"长胖"属 P-coriolis / C4a-4，这里只留口径与实测数。
            "equator_i_south": i_eq_s, "equator_i_north": i_eq_n,
            "equator_is_face": bool(i_eq_s != i_eq_n),
            "equator_v_mean_ms": float(eq_v.mean()),
            "equator_v_absmax_ms": float(np.abs(eq_v).max()),
            "equator_u_mean_ms": float(eq_u.mean()),
            "equator_u_min_ms": float(eq_u.min()),
            "equator_u_max_ms": float(eq_u.max()),
            "equator_spd_mean_ms": float(eq_spd.mean()),
            "equator_spd_absmax_ms": float(eq_spd.max()),
            "mirror_rel_t_surf": self.mirror_rel(t2),
            "mirror_rel_p_surf": self.mirror_rel(p2),
            "mirror_rel_u_east": self.mirror_rel(ue2),
            "mirror_rel_u_north": self.mirror_rel(un2, odd=True),
            "mirror_rel_div": self.mirror_rel(dv),
            "mirror_rel_w": self.mirror_rel(wx),
            "spd_calm_band_ms": spd_calm,
            "spd_trade_band_ms": spd_trade,
            "calm_trade_ratio": calm_ratio,
        }
        # ---- C4a-2 Step A：水汽/水量账本读数（🔴 关档 ⇒ 整块不进 ⇒ 旧 probe 键集逐位不变）----
        if self.moisture_on:
            rep.update(self._moisture_probe())
        return rep

    def _moisture_probe(self) -> dict:
        """Step A 的**只读**判据读数（判据①海→陆通量与内陆递减、判据②账本闭合、水滞留时间）。

        🔴 全部为纯函数：调用的 `moisture_advection()`/`evaporation_potential()` 只读
        `qv`/`u_*`/`t_surf`/`p_surf`/`mass_col`，不写状态（与 `wind_forces()`/`div_u()` 同口径 ⇒
        单测 P12 的"probe 前后逐位不变"必须覆盖水汽开档）。
        单位口径（owner 2026-10-17 修订令 ⇒ 🔴 键名后缀就是单位，不看文档也不会读错）：
        * `_kg_kg` = **状态量比湿**（层均混合差，无量纲）；
        * `_mm` = **柱水/储水**（kg m⁻² 水当量 ≡ mm，换算尺 `1 kg/m² = 1 mm`）；
        * `_mm_yr` = 收支年率 = 累积(mm) ÷ (`steps × dt_s` 物理秒) × `_SECONDS_PER_YEAR`
          （与 P̄=990 mm/yr 锚同一条尺；时间基准是**物理秒**不是主 tick ⇒ 三元组重标若发生，
          "年"的字面要一并复核）。
        两套尺的问句都给（`qv_*` 与 `qcol_*`、`bucket_awm_land_mm` 与 `bucket_cell_awm_mm`），
        因为"空气有多湿"和"大气里共有多少水"是**两条不同判据**，只报一条会被追问另一条。
        """
        sec = float(self.steps) * self.dt_s
        mm_yr = (lambda cum: cum / sec * _SECONDS_PER_YEAR if sec > 0.0 else float("nan"))
        rc, cc = self.rows_c, self.cols_c
        q2 = self.qv.reshape(rc, cc)                     # 比湿 (kg/kg)
        col = self.column_water().reshape(rc, cc)        # 柱水 (mm) 🔴 判据①②的尺
        b2 = self.bucket.reshape(rc, cc)                 # 每单位陆地面积 (mm of land)
        wf = self.water_frac.reshape(rc, cc)
        land = 1.0 - wf
        dist = self.land_dist.reshape(rc, cc)
        out = {
            "moisture_on": True,
            "moisture_advect_on": bool(self.cfg.moisture_advect_on),
            "water_limit_on": bool(self.cfg.water_limit_on),
            # ---- 库存：大气（比湿 + 柱水两条问句）----
            "qv_awm_kg_kg": self.awm(q2), "qv_min_kg_kg": float(q2.min()),
            "qv_max_kg_kg": float(q2.max()),
            "qcol_awm_mm": self.awm(col), "qcol_min_mm": float(col.min()),
            "qcol_max_mm": float(col.max()),
            # ---- 库存：陆面水桶（陆面尺的湿度 + 整格尺的储量）----
            "bucket_awm_land_mm": self._masked_awm(b2, land > 0.0),
            "bucket_min_mm": float(b2.min()), "bucket_max_mm": float(b2.max()),
            "bucket_cell_awm_mm": self.awm(b2 * land),
            # 陆面均桶 / 容量（🔴 两者同在陆面尺上 ⇒ 不乘占比；零模型⑤的反证看 `bucket_min_mm`）
            "bucket_fill_of_capacity": (self._masked_awm(b2, land > 0.0) / _W_MAX_KG_M2),
            "w_max_kg_m2_land": _W_MAX_KG_M2,
            "p_anchor_mm_yr": _P_ANCHOR_KG_M2_S * _SECONDS_PER_YEAR,
            # ---- 收支年率（mm/yr；P̄ 锚 = 990，判据②的字面尺）----
            "evap_mm_yr": mm_yr(self.wb[0]),
            "precip_mm_yr": mm_yr(self.wb[1]),
            "runoff_mm_yr": mm_yr(self.wb[2]),
            "ocean_mm_yr": mm_yr(self.wb[3]),
            # 实际降水 / 潜在降水（= P̄ 锚给的量）：干格拿不到水 ⇒ <1 是**物理**（柱水上限），
            # 不是数值损失。这条同时是"P 记的是取走量而非潜在量"的自证。
            "precip_realized_frac": (self.wb[1] / self.wb[5]
                                     if self.wb[5] > 0.0 else float("nan")),
            # ---- 判据②：水量账本闭合 ----
            # `wb[4]` = 各子步源汇段残差的**符号累积**（= 净造/吞的水，mm）；逐子步硬校核
            # 已在 `_moisture_substep` 里做（超容差直接炸），这里给"整批漂了多少"的诚实读数。
            # ⚠️ 分母用**柱水 + 陆面储量(整格尺) + 海洋净通量**三项 ⇒ 与恒等式同尺。
            "wb_imbalance_mm": float(self.wb[4]),
            "wb_imbalance_rel": abs(float(self.wb[4])) / max(
                abs(self.awm(col)) + abs(self.awm(b2 * land)) + abs(self.wb[3]), 1e-300),
            # ---- 判据①：海→陆水汽输运 + 内陆递减 ----
            # 陆域水汽**辐合** = −AWM_{陆面份额}(∇·(柱水·v))：内部面通量两侧反号 ⇒ 按构造
            # 恰等于跨海岸线的净入流量（mm/yr）。关风/关平流 ⇒ 恒 0（反证臂 = 单测 m09/m10，
            # 正证主装置 = m08）。分带读数用**柱水**（"内陆还有多少水"是质量问句，不是湿度问句）。
            "sea2land_flux_mm_yr": self._sea2land_flux_mm_yr(),
            "qcol_dist0_mm": self._masked_awm(col, dist == 0),   # 水面格（源）
            "qcol_dist4p_mm": self._masked_awm(col, dist >= _DIST_BAND_FAR),
            # ---- 平流的守恒与稳定性（红线 5）----
            "cfl_max": self._cfl_max, "cfl_warned": bool(self._cfl_warned),
            "cfl_stability_max": _CFL_STABILITY_MAX, "cfl_warn_max": _CFL_WARN_MAX,
            "adv_hardcheck_rel_max": self._adv_res_rel,
            # ---- 湿度 / 面占比 ----
            "water_area_frac": self.awm(wf), "land_area_frac": 1.0 - self.awm(wf),
            # ---- 水滞留时间 = 库存 / 汇通量（大气柱水汽被降水清出的 e-folding 尺）----
            # 🔴 分子必须是**柱水**（mm）、不是比湿；分母 `wb[1]/sec` 是 mm s⁻¹ ⇒ ×86400
            # 才是 mm day⁻¹（这里曾因多除一次 86400 给出 1e10 day 的荒谬数 ⇒ 单测 m16 钉量级）。
            "residence_days_global": (self.awm(col) / (self.wb[1] / sec * 86400.0)
                                      if (sec > 0.0 and self.wb[1] > 0.0)
                                      else float("nan")),
        }
        # 内陆递减分带（键名由 `_DIST_BANDS` 生成 ⇒ 带界与键同源，改一处两处一起改）
        for _d in _DIST_BANDS:
            out[f"qcol_dist{_d}_mm"] = self._masked_awm(col, dist == _d)
        # 相对湿度（🔴 修订后 `qv` 本身就是混合比 ⇒ `RH = qv/q_sat`，不再除 `mass_col`；
        # 旧写法 `qv/mass_col/q_sat` 在新口径下会把 RH 算成 ~1e-5 量级的常数小值）
        q_sat = self.saturation_mixing_ratio()
        rh = self.qv / q_sat
        out["rh_awm"] = self.awm(rh)
        out["rh_land_awm"] = self._masked_awm(rh.reshape(rc, cc), land > 0.0)
        out["rh_ocean_awm"] = self._masked_awm(rh.reshape(rc, cc), wf > 0.0)
        # 平流残差的**当次**重算（只读；与 `_moisture_substep` 内的硬校核同式，供对账）
        if bool(self.cfg.moisture_advect_on):
            dq, _cfl = self.moisture_advection()
            num = abs(float(np.sum(self.area[:, None] * dq)))
            den = float(np.sum(self.area[:, None] * np.abs(dq)))
            out["adv_area_sum_rel_residual"] = num / max(den, 1e-300)
        else:
            out["adv_area_sum_rel_residual"] = 0.0
        return out

    def _masked_awm(self, arr, mask) -> float:
        """**掩码内**的面积权均值（判据分带读数用；🔴 与 `awm` 同一把 `area` 尺）。

        空掩码（微型档里可能整带无格）⇒ 返回 `nan`（"问不了"），不给 0.0（= "该带无水汽"
        假象）—— 与 `awm_band_nan` / `calm_trade_ratio` 的 nan 口径同族。
        """
        a = np.asarray(arr, dtype=np.float64)
        m = np.asarray(mask, dtype=bool)
        w = self.area[:, None] * m
        den = float(w.sum())
        return float((w * a).sum() / den) if den > 0.0 else float("nan")

    def _sea2land_flux_mm_yr(self) -> float:
        """**陆面份额加权**的水汽辐合年率（mm/yr）= 判据①"海→陆输运 > 0"的字面读数（只读）。

        `−Σ area·(1−wf)·∇·(柱水·v) / Σ area·(1−wf)`：陆域内部的**陆–陆面**通量两侧反号 ⇒
        分子只剩**跨海岸面**的净流入（这就是"海→陆"的可测定义，不需要真的找一条海岸线）。
        🔴 修订后分子**天然**是水量通量（散度作用在柱水 `qv·mass_col` 上 ⇒ kg m⁻² s⁻¹ ≡
        mm s⁻¹）⇒ 不用再假设"比湿的辐合 ≈ 水汽的辐合"（旧口径下两者只差一个 `mass_col`
        的空间分布，而 `mass_col` 随温度纬向变化 ⇒ 那个假设是不可核验的）。
        🔴 权重必须是**陆面份额 `1−wf`** 而不是"只要有一点点陆就算陆"的掩码：混合列的**水面
        份额是不受约束的蒸发源**，用 `wf<1` 掩码会让近海岸的混合列反过来主导全球均值。
        `[实测]`（旧口径、柱水汽尺）同一装置两口径的差（中纬陆块、热力风）：掩码口径
        **−257 mm/yr**、份额口径 **−118**、纯陆格（`wf==0`）**+50**；而**海岸对齐粗格**的
        装置三者相同（`wf` 恰为 0/1）⇒ 差异全来自混合列，不是符号写反。份额口径是两者的
        连续推广，且与"子格 tiling"这条本就一致 ⇒ 选它，并在单测 m22 同时钉"对齐档三口径一致"。
        零模型④（关平流）⇒ 不调用散度、直接给 0.0（判据①的反证臂：没有风就没有海陆输运）。
        """
        if not bool(self.cfg.moisture_advect_on):
            return 0.0
        dq, _cfl = self.moisture_advection()
        w = self.area[:, None] * (1.0 - self.water_frac.reshape(self.rows_c, self.cols_c))
        den = float(w.sum())
        if den <= 0.0:
            return float("nan")
        return -float((w * dq).sum() / den) * _SECONDS_PER_YEAR
