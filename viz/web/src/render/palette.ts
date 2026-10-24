/**
 * 调色板：纯函数、零依赖（供 WorldCanvas 与后续单元测试复用）。
 *
 * - `viridisRgb` / `viridisLut`：数据帧 u8 → 近似 viridis RGBA（六次多项式拟合，
 *   拟合源 = matplotlib viridis，通道误差约 ≲ 3/255）。
 * - `energyRgb` / `energyColorCss`：实体能量暖色带（低 = 暗红橙 → 高 = 亮黄白）。
 */

export type Rgb = readonly [number, number, number];

const clamp01 = (t: number): number => (t < 0 ? 0 : t > 1 ? 1 : t);

const toByte = (x: number): number => Math.round(clamp01(x) * 255);

/** 第 i 次项系数，r/g/b 三通道各一列（t = 0 时取第 0 行 = 深紫起点） */
const VIRIDIS_POLY: readonly (readonly [number, number, number])[] = [
  [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
  [0.1050930431085774, 1.404613529898575, 1.384590162594685],
  [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
  [-4.634230498983486, -5.799100973351585, -19.33244095627987],
  [6.228269936347081, 14.17993336680509, 56.69055260068105],
  [4.776384997670288, -13.74514537774601, -65.35303263337234],
  [-5.435455855934631, 4.645852612178535, 26.3124352495832],
];

const evalPoly = (t: number, ch: 0 | 1 | 2): number => {
  let acc = VIRIDIS_POLY[6][ch];
  for (let i = 5; i >= 0; i--) acc = acc * t + VIRIDIS_POLY[i][ch];
  return acc;
};

/** t ∈ [0,1] → viridis RGB（0-255，越界自动钳制） */
export function viridisRgb(t: number): Rgb {
  const x = clamp01(t);
  return [toByte(evalPoly(x, 0)), toByte(evalPoly(x, 1)), toByte(evalPoly(x, 2))];
}

/** 256 级 viridis RGBA 查找表（长度 1024，每级 `[r,g,b,255]`，可直接逐像素喂 ImageData） */
export function viridisLut(): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(1024);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = viridisRgb(i / 255);
    const o = i * 4;
    lut[o] = r;
    lut[o + 1] = g;
    lut[o + 2] = b;
    lut[o + 3] = 255;
  }
  return lut;
}

const ENERGY_STOPS: readonly Rgb[] = [
  [180, 55, 40],
  [255, 150, 50],
  [255, 250, 215],
];

/** t ∈ [0,1]（能量归一）→ 暖色带 RGB（低 = 暗红橙，高 = 亮黄白） */
export function energyRgb(t: number): Rgb {
  const x = clamp01(t) * (ENERGY_STOPS.length - 1);
  const i = Math.min(Math.floor(x), ENERGY_STOPS.length - 2);
  const f = x - i;
  const a = ENERGY_STOPS[i];
  const b = ENERGY_STOPS[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

const ENERGY_CSS_CACHE: string[] = [];

/** t ∈ [0,1] → css `rgb(...)`（按 1/64 量化缓存，避免逐实体建串） */
export function energyColorCss(t: number): string {
  const q = Math.round(clamp01(t) * 64);
  let s = ENERGY_CSS_CACHE[q];
  if (s === undefined) {
    const [r, g, b] = energyRgb(q / 64);
    s = `rgb(${r},${g},${b})`;
    ENERGY_CSS_CACHE[q] = s;
  }
  return s;
}
