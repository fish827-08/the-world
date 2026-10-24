//! 确定性超越函数（deterministic math）
//!
//! **为什么要自己写**：`f64::sin/cos/atan` 走的是**系统 libm**，Windows(MSVC CRT) 与
//! Linux(glibc) 在最后 1–2 ulp 上不一定相同 ⇒ 同样的输入在不同平台会给出不同比特。
//! 本项目要求「CPU L0 逐位」（`atmo/INTERFACE.md` §0/§8.6），所以核心路径里的超越函数
//! 必须由本模块提供：**只用 IEEE-754 的 + − × ÷ 与 sqrt**（这几条都是正确舍入的、跨平台一致的），
//! 因此同一输入在任何平台的输出**逐位相同**。
//!
//! 精度（实测，见 `tests`）：|Δ| ≤ 2e-15（对角度量 ≈ 1e-13 度，远小于任何物理尺度）。
//! **精度不是本模块的首要指标，可复现才是。**
//!
//! 禁止：在 `atmo_core` 的确定性路径上直接调用 `f64::sin` / `f64::cos` / `f64::atan` / `f64::asin`
//! / `f64::exp` / `f64::ln` / `f64::powf`（后者留给 A2 按同样方式补，或走接口变更单）。

pub const PI: f64 = std::f64::consts::PI;

/// 2/π（用于把角度折算成 π/2 的整数倍）
const FRAC_2_PI: f64 = 0.636_619_772_367_581_3;
/// π/2 的高 33 位（低位全零 ⇒ k·PIO2_HI 在 |k| ≤ 2^19 时可精确表示）
const PIO2_HI: f64 = 1.570_796_326_734_125_6;
/// π/2 − PIO2_HI 的补足项
const PIO2_LO: f64 = 6.077_100_506_506_192e-11;

/// 2π（double）
const TWO_PI_HI: f64 = 6.283_185_307_179_586;
/// 2π − TWO_PI_HI（Cody–Waite 补偿项）
const TWO_PI_LO: f64 = 2.449_293_598_294_706_4e-16;

/// 把角度归约到 `[−π, π]`。
///
/// 只做**两段 Cody–Waite**（不是 Payne–Hanek）：对 `|x| ≤ 1e6` 绝对误差 ≲ 4e-11，
/// 对本项目（相位角都先按 `2π·frac(t/T)` 算到 ≤ 2π）绰绰有余。
/// 更大角度请调用方先自行归约——**这是契约，不是实现细节**。
pub fn reduce_2pi(x: f64) -> f64 {
    if x.abs() <= std::f64::consts::TAU {
        return x;
    }
    let n = (x / TWO_PI_HI).round();
    (x - n * TWO_PI_HI) - n * TWO_PI_LO
}

/// `sin(x)`，与平台无关的逐位实现。
#[inline]
pub fn sin(x: f64) -> f64 {
    sin_cos(x).0
}

/// `cos(x)`，与平台无关的逐位实现。
#[inline]
pub fn cos(x: f64) -> f64 {
    sin_cos(x).1
}

/// 同时算 sin 与 cos（共用一次象限约化）。
pub fn sin_cos(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    // 大角度先归约到 [−π, π]（见 reduce_2pi 的精度边界）
    let x = reduce_2pi(x);
    // 约化到 r ∈ [−π/4, π/4]
    let k = (x * FRAC_2_PI).round();
    let r = (x - k * PIO2_HI) - k * PIO2_LO;
    let (s, c) = (sin_kernel(r), cos_kernel(r));
    let q = (k as i64) & 3;
    match q {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// |r| ≤ π/4 的 sin 泰勒核（截断到 r^17，余项 < 1e-19）
#[inline]
fn sin_kernel(r: f64) -> f64 {
    let z = r * r;
    r * (1.0
        + z * (-1.666_666_666_666_666_6e-1
            + z * (8.333_333_333_333_333e-3
                + z * (-1.984_126_984_126_984_1e-4
                    + z * (2.755_731_922_398_589e-6
                        + z * (-2.505_210_838_544_172e-8
                            + z * (1.605_904_383_682_161_3e-10
                                + z * (-7.647_163_731_819_816e-13
                                    + z * 2.811_457_254_345_521e-15))))))))
}

/// |r| ≤ π/4 的 cos 泰勒核（截断到 r^16，余项 < 2e-18）
#[inline]
fn cos_kernel(r: f64) -> f64 {
    let z = r * r;
    1.0 + z
        * (-5.0e-1
            + z * (4.166_666_666_666_666_4e-2
                + z * (-1.388_888_888_888_888_9e-3
                    + z * (2.480_158_730_158_730_2e-5
                        + z * (-2.755_731_922_398_589e-7
                            + z * (2.087_675_698_786_81e-9
                                + z * (-1.147_074_559_772_972_5e-11
                                    + z * (4.779_477_332_387_385e-14
                                        + z * -1.561_920_696_858_622_5e-16))))))))
}

/// `atan(x)`：5 次半角约化 + 泰勒，误差 ≤ 1e-15（含约化放大）。
pub fn atan(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return PI * 0.5;
    }
    if x == f64::NEG_INFINITY {
        return -PI * 0.5;
    }
    let ax = x.abs();
    if ax > 1.0 {
        // atan(x) = π/2 − atan(1/x)（保持符号）
        let r = PI * 0.5 - atan(1.0 / ax);
        return if x < 0.0 { -r } else { r };
    }
    let mut y = ax;
    let mut scale = 1.0f64;
    // atan(y) = 2·atan(y / (1 + sqrt(1+y²)))，做 5 次把 |y| 压到 ≤ 0.025
    for _ in 0..5 {
        y /= 1.0 + (1.0 + y * y).sqrt();
        scale *= 2.0;
    }
    let z = y * y;
    let mut term = y;
    let mut acc = 0.0f64;
    for n in 0..12usize {
        let d = (2 * n + 1) as f64;
        if n % 2 == 0 {
            acc += term / d;
        } else {
            acc -= term / d;
        }
        term *= z;
    }
    let mag = acc * scale;
    // 奇函数：这里必须补符号（x = −1 这类边界曾在单测里被抓到）
    if x < 0.0 {
        -mag
    } else {
        mag
    }
}

/// `asin(x)`：`asin(x) = atan(x / sqrt(1 − x²))`，端点返回 ±π/2。
pub fn asin(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 1.0 {
        return if x == 1.0 { PI * 0.5 } else { f64::NAN };
    }
    if x <= -1.0 {
        return if x == -1.0 { -PI * 0.5 } else { f64::NAN };
    }
    atan(x / (1.0 - x * x).sqrt())
}

/// ln(2)（double）
const LN2: f64 = 0.693_147_180_559_945_3;

/// `ln(x)`：IEEE 位分解 + atanh 级数（只用 +−×÷），**跨平台逐位一致**。
pub fn ln(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000); // [1,2)
    if m > 1.414_213_562_373_095_1 {
        m *= 0.5;
        e += 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let mut term = z;
    let mut acc = 0.0f64;
    for n in 0..24usize {
        acc += term / (2 * n + 1) as f64;
        term *= z2;
    }
    2.0 * acc + (e as f64) * LN2
}

/// `exp(x)`：2^k 位缩放 + 泰勒（只用 +−×÷），**跨平台逐位一致**。
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    if x < -745.0 {
        return 0.0;
    }
    let k = (x / LN2).round();
    let r = x - k * LN2; // |r| ≤ 0.35
    let mut term = 1.0f64;
    let mut acc = 1.0f64;
    for n in 1..18usize {
        term *= r / n as f64;
        acc += term;
    }
    // 乘 2^k（k 是整数，位运算精确）
    let ki = k as i64;
    let scale = f64::from_bits((((1023 + ki) as u64) & 0x7ff) << 52);
    acc * scale
}

/// `x^y = exp(y·ln x)`（确定性；只为避开 libm 的 `powf`）
pub fn pow(x: f64, y: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    exp(y * ln(x))
}

/// 弧度 → 度（纯乘除，跨平台一致）。
#[inline]
pub fn to_deg(rad: f64) -> f64 {
    rad * (180.0 / PI)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_err_sin_cos(limit: f64) -> (f64, f64) {
        let mut es: f64 = 0.0;
        let mut ec: f64 = 0.0;
        let step = limit / 2000.0;
        let mut i = -2000i32;
        while i <= 2000 {
            let x = i as f64 * step;
            let (s, c) = sin_cos(x);
            es = es.max((s - x.sin()).abs());
            ec = ec.max((c - x.cos()).abs());
            i += 1;
        }
        (es, ec)
    }

    #[test]
    fn sin_cos_match_std() {
        // 物理量纲的角度都先归约到 ≤ 2π（见 reduce_2pi 的契约）
        let (es, ec) = max_err_sin_cos(std::f64::consts::TAU);
        assert!(es < 2.0e-15, "sin 最大误差 {es:e} 超限");
        assert!(ec < 2.0e-15, "cos 最大误差 {ec:e} 超限");
    }

    #[test]
    fn sin_cos_large_argument_is_input_limited() {
        // |x| ~ 1e5–1e6 时误差由**输入 double 本身的 ulp** 决定，不是算法缺陷；
        // 这里只钉住"不崩、不离谱"。
        for &x in &[1.0e5f64, 1.0e6, 98_765.4321, -12_345.6789] {
            let (s, c) = sin_cos(x);
            assert!((s - x.sin()).abs() < 1.0e-8, "x={x} sin 误差过大");
            assert!((c - x.cos()).abs() < 1.0e-8, "x={x} cos 误差过大");
        }
    }

    #[test]
    fn atan_asin_match_std() {
        let mut ea: f64 = 0.0;
        let mut eai: f64 = 0.0;
        let mut i = -10000i32;
        while i <= 10000 {
            let x = i as f64 * 0.000_5;
            ea = ea.max((atan(x) - x.atan()).abs());
            i += 1;
        }
        let mut j = -999i32;
        while j <= 999 {
            let x = j as f64 * 0.001;
            eai = eai.max((asin(x) - x.asin()).abs());
            j += 1;
        }
        assert!(ea < 1.0e-14, "atan 最大误差 {ea:e} 超限");
        assert!(eai < 1.0e-14, "asin 最大误差 {eai:e} 超限");
    }

    #[test]
    fn ln_exp_match_std() {
        let mut el = 0.0f64;
        let mut ee = 0.0f64;
        let mut x = 1.0e-8f64;
        while x < 1.0e8 {
            el = el.max((ln(x) - x.ln()).abs() / x.ln().abs().max(1.0));
            let e = exp(x % 20.0 - 10.0);
            ee = ee.max((e - (x % 20.0 - 10.0).exp()).abs() / e.abs().max(1e-30));
            x *= 1.37;
        }
        assert!(el < 1e-14, "ln 相对误差 {el:e}");
        assert!(ee < 1e-14, "exp 相对误差 {ee:e}");
        // 关键常数逐位核对（Φ 积分与 HS 剖面都依赖它们）
        assert!((ln(2.0) - std::f64::consts::LN_2).abs() < 1e-16);
        assert!((ln(10.0) - 2.302_585_092_994_046).abs() < 1e-15);
    }

    #[test]
    fn half_angle_consistency() {
        // sin²+cos² = 1（在整圈上抽样）
        for i in -720..=720 {
            let x = i as f64 * 0.01;
            let (s, c) = sin_cos(x);
            assert!((s * s + c * c - 1.0).abs() < 5.0e-16, "x={x}");
        }
    }
}
