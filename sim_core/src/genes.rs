//! 基因位语义常量（Rust 侧，与 simulation/genes.py 注册表一一对应）。
//!
//! 单一事实源是 Python 侧 `simulation/genes.py` 的 `Gene` 枚举；
//! 本文件只是把同样索引复制为 Rust 常量，避免在 movement/predation 里
//! 写魔法数字混淆。lib.rs 的 `validate_gene_wiring` 会用 Python 传入的
//! 语义名字符串数组（与 Gene 枚举名一致）逐位对照校验，两侧一旦漂移
//! 会在引擎初始化时直接报错，而不是静默产生错误行为。
//!
//! 新增基因位时：simulation/genes.py 追加成员 → 本文件追加同名常量 →
//! 按需在 movement/predation/step_vectors 消费。

// g0  移动概率（被 g19 植物化缩放）
pub const G_MOVE_PROB: usize = 0;
// g1  代谢倍率（消化快慢）
pub const G_METABOLIC: usize = 1;
// g2  繁殖能量门槛
pub const G_REPRO_THRESHOLD: usize = 2;
// g3  寿命基因（影响 lifespan 派生）
pub const G_LIFE_GENE: usize = 3;
// g4  进食量
pub const G_EAT_AMOUNT: usize = 4;
// g5  胃容量
pub const G_STOMACH_CAP: usize = 5;
// g6  移动能耗
pub const G_MOVE_COST: usize = 6;
// g7  传代投入（分给子代的能量比例）
pub const G_PARENTAL_INVEST: usize = 7;
// g8  光合产能
pub const G_PHOTOSYNTHESIS: usize = 8;
// g9  恒温性
pub const G_HOMEOTHERM: usize = 9;
// g10 邻格觅食倾向
pub const G_FORAGE_NEIGHBOR: usize = 10;
// g11 温度偏好
pub const G_TEMP_PREF: usize = 11;
// g12 繁殖冷却长度
pub const G_REPRO_COOLDOWN: usize = 12;
// g13 群居性
pub const G_SOCIABILITY: usize = 13;
// g14 感知半径
pub const G_PERCEPTION: usize = 14;
// g15 信号发射概率
pub const G_SIGNAL_STRENGTH: usize = 15;
// g16 攻击性
pub const G_AGGRESSION: usize = 16;
// g17 食性（预留）
pub const G_DIET: usize = 17;
// g18 防御（预留）
pub const G_DEFENSE: usize = 18;
// g19 植物化扎根
pub const G_ROOTING: usize = 19;
// g20 赶路惯性（ARS 14.9）—— ⚠️ Rust 侧**未实现**该机制（ARS 强制 Python 路径），
//   本常量仅作「索引 ↔ 语义」对齐用；引擎侧 fail-loud 由 sphere_engine 守卫负责。
pub const G_PERSISTENCE: usize = 20;
// g21 失望阈值（ARS 14.9）—— 同上，仅对齐用。
pub const G_GIVE_UP: usize = 21;
// g22 信任阈值（预留）
pub const G_TRUST_GENE: usize = 22;
// g23 迁徙倾向（日历—罗盘式定向迁徙，13.8）—— ⚠️ Rust 侧**未实现**该机制，
// 本常量仅作「索引 ↔ 语义」对齐用；值与历史 G_RESERVED 相同（=23，**基因组长度不变**）。
// Rust 移动路径**不消费**本基因位（13.8 强制 Python 路径，fail-loud 由引擎侧守卫负责）。
pub const G_MIGRATE_BIAS: usize = 23;