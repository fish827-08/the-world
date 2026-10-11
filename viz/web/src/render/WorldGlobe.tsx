/**
 * WorldGlobe —— 三维球体视图（three.js 地球仪；设计稿：
 * docs/设计文档/设计-可视化v1-三维球体-20261011.md）。
 *
 * 与 WorldCanvas 同构：同一份 meta/grid/entities 进、画面出（`scale` 不消费 —— 契约 §3
 * 量化已含标定，仅为签名完整保留）。数据流：
 *   图层帧 uint8 → viridis LUT 上色 → DataTexture 贴 SphereGeometry（NearestFilter，保格块感）；
 *   实体 7 列 → 行列 / 亚格（与平面共用 ./subcell 帧级裁决）→ 经纬度 → 球面坐标 → THREE.Points。
 *
 * 视效（v1.1，owner 诉求：更美观 + 更大放大倍率）：
 *   缩放 minDistance 1.001 / maxDistance 20（R=1），相机 near 0.001 保证贴近不裁切；
 *   球面 256×128 段消除极区三角化棱块；MeshStandardMaterial + 环境光/平行光出昼夜明暗；
 *   1.02R 大气层辉光（BackSide + Additive + 菲涅尔 pow）、远壳星空、实体点像素级 LOD、
 *   EffectComposer + UnrealBloomPass + OutputPass 让过曝的实体点团簇核心发光
 *   （阈值 1.05 > 1.0 ⇒ 受光地表与单点羽化边缘不进 Bloom，v1.2 实机验收调参）。
 *
 * 行序：数据行 0 = 北极；DataTexture（flipY=false）缓冲第 0 行对应 UV v=0（南极端）
 * ⇒ 填缓冲时整行翻转一次（dest 行 = rows-1-r），v 轴只有这一个换算点。
 * 实体点与贴图共用同一"经纬 → 球面"映射（与 SphereGeometry 顶点公式同式），保证两种表示对齐。
 *
 * 极区：球面自然收敛（平面图上极区的横向拉伸失真在球体上消失）。
 * 生命周期：StrictMode 双挂载安全 —— 卸载全量 dispose（rAF / RO / controls / 几何 / 材质 / 纹理 / renderer）。
 * 空闲自动缓转（owner 拍板默认开）：交互即停，松手 IDLE_RESUME_MS 后恢复。
 */
import { useEffect, useRef } from "react";
import type { JSX } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import type { Meta } from "../contract";
import { energyRgb, viridisLut } from "./palette";
import { resolveSubdiv } from "./subcell";

const LUT = viridisLut();
const TAU = Math.PI * 2;

/** 球半径（世界单位）；相机距离 / 点大小均以此为基准 */
const R = 1;
/** 实体点浮起系数（避免与球面深度冲突） */
const LIFT = 1.004;
/** 空闲多久（ms）恢复自动缓转 */
const IDLE_RESUME_MS = 10_000;
const AUTO_ROTATE_SPEED = 0.6;

/** 大气层光晕壳半径（BackSide + AdditiveBlending，边缘菲涅尔辉光） */
const GLOW_R = 1.02;
/** 辉光峰值强度；相机贴近/穿入壳内时在 [GLOW_R, GLOW_FADE_FROM] 区间淡出
 *  （否则 BackSide 内壁会把整屏染蓝）。
 *  v1.2 实机验收调参 2.0→3.5：实体层撤增益后暖雾消失，蓝辉需 3.5 才能从"一线"变可显形
 *  （可见 rim 峰值 ≈ 0.6³×3.5 ≈ 0.76，加性叠在近黑背景上是明显但不致过曝的蓝色光带） */
const GLOW_STRENGTH = 3.5;
const GLOW_FADE_FROM = 1.45;
/** 星空撒点壳半径 / 点数（相机 far=60 之内） */
const STAR_R = 40;
const STAR_COUNT = 1600;
/** 实体点 LOD：屏幕像素尺寸钳制区间（拉远不至于看不见、贴近不至于胀大） */
const LOD_MIN_PX = 2.2;
const LOD_MAX_PX = 9;
/** Bloom：阈值 1.05 > 1.0 ⇒ 只有加性叠加后真正过曝的实体点核心/团簇进 Bloom。
 *  v1.2 实机验收调参 0.85→1.05：旧阈值低于受光地表峰值（≈0.8）与单点加性叠加值，
 *  全球像素普遍进 Bloom 形成暖雾，把 viridis 暗紫洗成红棕、吞掉昼夜界线 */
const BLOOM_THRESHOLD = 1.05;
/** 强度保持 0.55：过亮根因是低阈值而非强度，阈值修正后无需再压。
 *  若实机仍觉团簇光晕偏艳，下一档 0.4 */
const BLOOM_STRENGTH = 0.55;
const BLOOM_RADIUS = 0.7;
/** 光照：环境光保夜半球可读（LUT 色阶不被洗掉），方向光出昼夜明暗界线 */
const AMBIENT_INTENSITY = 0.45;
const SUN_INTENSITY = 1.8;

/** 实体点颜色增益：v1.1 曾 ×1.25 补加性混合的"吃色"，实机验收发现上万点叠加时
 *  人为增益加速暖雾积累（全球洗成红棕/白黄过曝）⇒ 撤回 1.0（色带 = energyRgb 原值） */
const ENT_GAIN = 1.0;
/** 能量暖色带 → 工作色彩空间 THREE.Color（1/64 量化缓存，同 energyColorCss 口径） */
const ENT_COLORS: readonly THREE.Color[] = Array.from({ length: 65 }, (_, q) => {
  const [r, g, b] = energyRgb(q / 64);
  const c = new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
  c.r = Math.min(1, c.r * ENT_GAIN);
  c.g = Math.min(1, c.g * ENT_GAIN);
  c.b = Math.min(1, c.b * ENT_GAIN);
  return c;
});

export interface WorldGlobeProps {
  meta: Meta;
  /** rows*cols 当前图层帧（uint8，行主序，行 0 = 北极端）；null/空 = 无图层 */
  grid: Uint8Array | null;
  /** 交错 7 列 [flat,sub_r,sub_c,energy,age,generation,mode]；null/空 = 无实体 */
  entities: Float32Array | null;
  /** 当前图层标定（meta.channels[].scale）；u8 帧已量化 ⇒ 绘制不消费 */
  scale: { p_lo: number; p_hi: number };
  /** 空闲自动缓转（默认开，owner 拍板）；交互即停、空闲后恢复 */
  autoRotate?: boolean;
}

interface GlobeCtx {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  texture: THREE.DataTexture;
  texData: Uint8Array;
  geometry: THREE.BufferGeometry;
  posAttr: THREE.BufferAttribute;
  colAttr: THREE.BufferAttribute;
  posArr: Float32Array;
  colArr: Float32Array;
  capacity: number;
  raf: number;
  resumeTimer: number | null;
  disposed: boolean;
}

export function WorldGlobe({
  meta,
  grid,
  entities,
  autoRotate = true,
}: WorldGlobeProps): JSX.Element {
  const rows = Math.max(0, Math.floor(meta.world.rows));
  const cols = Math.max(0, Math.floor(meta.world.cols));

  const hostRef = useRef<HTMLDivElement | null>(null);
  const ctxRef = useRef<GlobeCtx | null>(null);

  // ---- 场景生命周期（世界尺寸或缓转开关变化 ⇒ 重建；StrictMode 双挂载 ⇒ 全量清理重来） ----
  useEffect(() => {
    const host = hostRef.current;
    if (!host || rows <= 0 || cols <= 0) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      host.textContent = "WebGL 不可用——无法渲染三维球体，可切回平面视图。";
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.style.display = "block";
    renderer.domElement.setAttribute("role", "img");
    renderer.domElement.setAttribute("aria-label", `三维球体渲染 ${rows}×${cols}`);
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    // near 压到 0.001：允许 minDistance≈1.001 贴近观察而不被近裁面切进/深度冲突
    const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 60);
    camera.position.set(0, 1.0, 2.9);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = false;
    controls.minDistance = 1.001;
    controls.maxDistance = 20;
    controls.rotateSpeed = 0.75;
    controls.zoomSpeed = 0.75;
    controls.autoRotate = autoRotate;
    controls.autoRotateSpeed = AUTO_ROTATE_SPEED;

    // 贴图：固定一张 DataTexture，逐帧只更新像素缓冲 + needsUpdate（设计稿 §3.2）
    const texData = new Uint8Array(rows * cols * 4);
    const texture = new THREE.DataTexture(
      texData,
      cols,
      rows,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
    );
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
    texture.generateMipmaps = false;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;

    // 光照：环境光 + 平行光（MeshStandardMaterial 才吃光照，原 MeshBasicMaterial 不吃）
    const ambient = new THREE.AmbientLight(0xffffff, AMBIENT_INTENSITY);
    const sun = new THREE.DirectionalLight(0xffffff, SUN_INTENSITY);
    sun.position.set(5, 3, 4);
    scene.add(ambient, sun);

    // 256×128 段：消除极区大三角面片的可见棱块感（原 128×64 贴近时可辨）
    const sphereGeo = new THREE.SphereGeometry(R, 256, 128);
    const sphereMat = new THREE.MeshStandardMaterial({
      map: texture,
      roughness: 0.82,
      metalness: 0.0,
    });
    const sphere = new THREE.Mesh(sphereGeo, sphereMat);
    scene.add(sphere);

    // 大气层光晕：1.02R 壳 BackSide + AdditiveBlending，边缘 pow 菲涅尔（贴边最亮，正面透明）
    const glowGeo = new THREE.SphereGeometry(GLOW_R, 96, 48);
    const glowMat = new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: new THREE.Color(0.42, 0.62, 1.0) },
        uPower: { value: 3.0 },
        uStrength: { value: GLOW_STRENGTH },
      },
      vertexShader: /* glsl */ `
        varying vec3 vNormalW;
        varying vec3 vViewDir;
        void main() {
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          vNormalW = normalize(normalMatrix * normal);
          vViewDir = normalize(-mvPos.xyz);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor;
        uniform float uPower;
        uniform float uStrength;
        varying vec3 vNormalW;
        varying vec3 vViewDir;
        void main() {
          float rim = max(0.0, 0.6 - dot(normalize(vNormalW), normalize(vViewDir)));
          float intensity = pow(rim, uPower) * uStrength;
          gl_FragColor = vec4(uColor * intensity, intensity);
        }
      `,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthWrite: false,
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    scene.add(glow);

    // 星空背景：远壳随机撒点（尺寸衰减关 ⇒ 恒定星点），亮度微随机
    const starGeo = new THREE.BufferGeometry();
    {
      const sp = new Float32Array(STAR_COUNT * 3);
      const sc = new Float32Array(STAR_COUNT * 3);
      for (let i = 0; i < STAR_COUNT; i++) {
        // 均匀球面撒点
        const u = Math.random() * 2 - 1;
        const th = Math.random() * TAU;
        const s = Math.sqrt(1 - u * u);
        sp[i * 3] = STAR_R * s * Math.cos(th);
        sp[i * 3 + 1] = STAR_R * u;
        sp[i * 3 + 2] = STAR_R * s * Math.sin(th);
        const b = 0.35 + Math.random() * 0.65;
        sc[i * 3] = b;
        sc[i * 3 + 1] = b;
        sc[i * 3 + 2] = b;
      }
      starGeo.setAttribute("position", new THREE.BufferAttribute(sp, 3));
      starGeo.setAttribute("color", new THREE.BufferAttribute(sc, 3));
    }
    const starMat = new THREE.PointsMaterial({
      size: 1.6,
      sizeAttenuation: false,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
    });
    const stars = new THREE.Points(starGeo, starMat);
    stars.frustumCulled = false;
    scene.add(stars);

    // 实体层：Points + 顶点色（能量 → 暖色亮度）；点径每帧按相机距离 LOD（见主循环）
    const capacity = 1024;
    const posArr = new Float32Array(capacity * 3);
    const colArr = new Float32Array(capacity * 3);
    const geometry = new THREE.BufferGeometry();
    const posAttr = new THREE.BufferAttribute(posArr, 3);
    const colAttr = new THREE.BufferAttribute(colArr, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    colAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("position", posAttr);
    geometry.setAttribute("color", colAttr);
    geometry.setDrawRange(0, 0);

    const dotTex = makeDotTexture();
    const arc = TAU / cols; // 赤道格距（世界单位）
    const basePointSize = Math.min(Math.max(arc * 0.55, 0.022), 0.03);
    const pointsMat = new THREE.PointsMaterial({
      // 点径注意：three 的 pointsize 非物理标定（gl_PointSize = size·H/2/z）⇒ 实测校准：
      // 0.011 在大世界默认视距仅 ~1.8px（不可见），0.024 ≈ 3.9px（小世界实证观感好）。
      // 此为 LOD 基准值，主循环按相机距离在 [LOD_MIN_PX, LOD_MAX_PX] 像素内自适应钳制
      size: basePointSize,
      sizeAttenuation: true,
      map: dotTex,
      vertexColors: true,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const points = new THREE.Points(geometry, pointsMat);
    points.frustumCulled = false; // drawRange 动态 + 点全在球面邻域，直接关剔除
    scene.add(points);

    // 后处理：RenderPass → UnrealBloomPass（实体点发光）→ OutputPass（回 sRGB / tone mapping）
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(1, 1),
      BLOOM_STRENGTH,
      BLOOM_RADIUS,
      BLOOM_THRESHOLD,
    );
    composer.addPass(bloomPass);
    composer.addPass(new OutputPass());

    const ctx: GlobeCtx = {
      renderer,
      scene,
      camera,
      controls,
      texture,
      texData,
      geometry,
      posAttr,
      colAttr,
      posArr,
      colArr,
      capacity,
      raf: 0,
      resumeTimer: null,
      disposed: false,
    };
    ctxRef.current = ctx;

    // ---- 尺寸 ----
    const applySize = () => {
      const w = Math.max(1, host.clientWidth);
      const h = Math.max(1, host.clientHeight);
      renderer.setSize(w, h);
      composer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    applySize();
    const ro = new ResizeObserver(applySize);
    ro.observe(host);

    // ---- 空闲自动缓转：交互即停，松手 IDLE_RESUME_MS 后恢复 ----
    const onStart = () => {
      if (ctx.resumeTimer !== null) {
        window.clearTimeout(ctx.resumeTimer);
        ctx.resumeTimer = null;
      }
      controls.autoRotate = false;
    };
    const onEnd = () => {
      if (!autoRotate) return;
      if (ctx.resumeTimer !== null) window.clearTimeout(ctx.resumeTimer);
      ctx.resumeTimer = window.setTimeout(() => {
        ctx.resumeTimer = null;
        if (!ctx.disposed) controls.autoRotate = true;
      }, IDLE_RESUME_MS);
    };
    controls.addEventListener("start", onStart);
    controls.addEventListener("end", onEnd);

    // ---- 主循环 ----
    const animate = () => {
      ctx.raf = requestAnimationFrame(animate);
      controls.update();

      // 实体点 LOD：保持屏幕像素尺寸在 [LOD_MIN_PX, LOD_MAX_PX] ——
      // gl_PointSize = size·H/(2·dist) ⇒ 先算基准点径的像素尺寸，钳制后反解 world size
      const dist = Math.max(camera.position.length(), 1e-4);
      const h = Math.max(1, renderer.domElement.clientHeight);
      let targetPx = (basePointSize * h) / (2 * dist);
      if (targetPx < LOD_MIN_PX) targetPx = LOD_MIN_PX;
      if (targetPx > LOD_MAX_PX) targetPx = LOD_MAX_PX;
      pointsMat.size = (targetPx * 2 * dist) / h;

      // 大气层随贴近程度淡出（二次曲线，避免壳边界处突兀）
      const glowFade = Math.min(Math.max((dist - GLOW_R) / (GLOW_FADE_FROM - GLOW_R), 0), 1);
      glow.visible = glowFade > 0.002;
      glowMat.uniforms.uStrength.value = GLOW_STRENGTH * glowFade * glowFade;

      composer.render();
    };
    ctx.raf = requestAnimationFrame(animate);

    // ---- 全量清理 ----
    return () => {
      ctx.disposed = true;
      window.cancelAnimationFrame(ctx.raf);
      if (ctx.resumeTimer !== null) window.clearTimeout(ctx.resumeTimer);
      ro.disconnect();
      controls.removeEventListener("start", onStart);
      controls.removeEventListener("end", onEnd);
      controls.dispose();
      scene.remove(sphere);
      scene.remove(points);
      scene.remove(glow);
      scene.remove(stars);
      scene.remove(ambient, sun);
      sphereGeo.dispose();
      sphereMat.dispose();
      glowGeo.dispose();
      glowMat.dispose();
      starGeo.dispose();
      starMat.dispose();
      texture.dispose();
      geometry.dispose();
      pointsMat.dispose();
      dotTex.dispose();
      composer.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      ctxRef.current = null;
    };
  }, [rows, cols, autoRotate]);

  // ---- 逐帧数据：图层贴图 + 实体点（meta/grid/entities 或世界尺寸变化即更新） ----
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || ctx.disposed) return;

    fillGridTexture(ctx.texData, grid, rows, cols);
    ctx.texture.needsUpdate = true;

    const nEnt = entities ? Math.floor(entities.length / 7) : 0;
    if (nEnt > ctx.capacity) {
      const cap = Math.max(nEnt, ctx.capacity * 2);
      ctx.capacity = cap;
      ctx.posArr = new Float32Array(cap * 3);
      ctx.colArr = new Float32Array(cap * 3);
      const pa = new THREE.BufferAttribute(ctx.posArr, 3);
      const ca = new THREE.BufferAttribute(ctx.colArr, 3);
      pa.setUsage(THREE.DynamicDrawUsage);
      ca.setUsage(THREE.DynamicDrawUsage);
      ctx.geometry.setAttribute("position", pa);
      ctx.geometry.setAttribute("color", ca);
      ctx.posAttr = pa;
      ctx.colAttr = ca;
    }
    const drawn = fillEntities(meta, entities, rows, cols, ctx.posArr, ctx.colArr);
    ctx.geometry.setDrawRange(0, drawn);
    ctx.posAttr.needsUpdate = true;
    ctx.colAttr.needsUpdate = true;
  }, [meta, grid, entities, rows, cols]);

  return <div ref={hostRef} className="globe-host" />;
}

/** 图层帧 → 纹理缓冲：行翻转一次（dest 行 = rows-1-r，见文件头注释）；无帧 ⇒ LUT[0] 铺底 */
function fillGridTexture(
  texData: Uint8Array,
  grid: Uint8Array | null,
  rows: number,
  cols: number,
): void {
  const n = rows * cols;
  const m = grid ? Math.min(grid.length, n) : 0;
  for (let i = 0; i < n; i++) {
    const v = i < m && grid ? grid[i] : 0;
    const r = Math.floor(i / cols);
    const c = i - r * cols;
    const o = v * 4;
    const p = ((rows - 1 - r) * cols + c) * 4;
    texData[p] = LUT[o];
    texData[p + 1] = LUT[o + 1];
    texData[p + 2] = LUT[o + 2];
    texData[p + 3] = 255;
  }
}

/** 实体 → 点缓冲（位置/颜色），返回写入点数。归一与回落口径与 WorldCanvas 一致 */
function fillEntities(
  meta: Meta,
  entities: Float32Array | null,
  rows: number,
  cols: number,
  pos: Float32Array,
  colBuf: Float32Array,
): number {
  if (!entities || entities.length < 7) return 0;
  const nEnt = Math.floor(entities.length / 7);

  const norm = meta.entities?.norm?.energy;
  let lo = 0;
  let hi = 1;
  if (norm && Number.isFinite(norm.p_lo) && Number.isFinite(norm.p_hi) && norm.p_hi > norm.p_lo) {
    lo = norm.p_lo;
    hi = norm.p_hi;
  } else {
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < nEnt; i++) {
      const e = entities[i * 7 + 3];
      if (Number.isFinite(e)) {
        if (e < mn) mn = e;
        if (e > mx) mx = e;
      }
    }
    if (Number.isFinite(mn) && mx > mn) {
      lo = mn;
      hi = mx;
    }
  }
  const span = hi - lo;

  const subdiv = resolveSubdiv(entities, nEnt, cols, meta.entities.subdiv);

  let w = 0;
  const lift = R * LIFT;
  for (let i = 0; i < nEnt; i++) {
    const b = i * 7;
    const flat = entities[b];
    if (!Number.isFinite(flat)) continue;
    const row = Math.floor(flat / cols);
    const col = flat - row * cols;
    if (row < 0 || row >= rows || col < 0 || col >= cols) continue;

    let u: number;
    let v: number;
    const sr = entities[b + 1];
    const sc = entities[b + 2];
    if (subdiv > 0 && Number.isFinite(sr) && Number.isFinite(sc)) {
      u = (sc + 0.5) / (cols * subdiv);
      v = (sr + 0.5) / (rows * subdiv);
    } else {
      u = (col + 0.5) / cols;
      v = (row + 0.5) / rows;
    }
    if (!(u >= 0 && u <= 1 && v >= 0 && v <= 1)) continue;

    // 经纬 → 球面（与 SphereGeometry 顶点公式同式，保证与贴图对齐）
    const theta = v * Math.PI;
    const phi = u * TAU;
    const st = Math.sin(theta);
    const x = -lift * Math.cos(phi) * st;
    const y = lift * Math.cos(theta);
    const z = lift * Math.sin(phi) * st;

    const e = entities[b + 3];
    let t = 0.5;
    if (Number.isFinite(e) && span > 0) t = (e - lo) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const c = ENT_COLORS[Math.round(t * 64)];

    const p = w * 3;
    pos[p] = x;
    pos[p + 1] = y;
    pos[p + 2] = z;
    colBuf[p] = c.r;
    colBuf[p + 1] = c.g;
    colBuf[p + 2] = c.b;
    w++;
  }
  return w;
}

/** 圆形光点贴图（径向渐变白点；配合 AdditiveBlending 出柔晕） */
function makeDotTexture(): THREE.CanvasTexture {
  const s = 64;
  const cv = document.createElement("canvas");
  cv.width = s;
  cv.height = s;
  const g = cv.getContext("2d");
  if (g) {
    const grad = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    grad.addColorStop(0, "rgba(255,255,255,1)");
    grad.addColorStop(0.4, "rgba(255,255,255,0.85)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, s, s);
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export default WorldGlobe;
