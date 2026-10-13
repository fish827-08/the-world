/**
 * 亚格坐标消费规则（契约 v0 修订1 §4）—— 平面/球体两条渲染路径共用的唯一裁决点。
 *
 * `resolveSubdiv`：值取自 `meta.entities.subdiv`；缺失 或 无亚格数据 ⇒ 0 = 回退格中心。
 * 声明 subdiv 时做帧级一致性校验 —— 判据 `sub_r // subdiv == floor(flat/cols)`
 * 且 `sub_c // subdiv == col`（同一实体的两种表示）；不过（实测：subpos 关档续跑时
 * sub 列与 flat 漂移）⇒ 一律 0 = 全帧回退格中心，不允许部分实体按亚格绘制。
 */
export function resolveSubdiv(
  entities: Float32Array,
  nEnt: number,
  cols: number,
  metaSubdiv: number | undefined,
): number {
  if (typeof metaSubdiv !== "number" || !Number.isFinite(metaSubdiv) || metaSubdiv < 1) {
    return 0;
  }
  let total = 0;
  let bad = 0;
  for (let i = 0; i < nEnt; i++) {
    const b = i * 7;
    const sr = entities[b + 1];
    const sc = entities[b + 2];
    if (sr === 0 && sc === 0) continue;
    const flat = entities[b];
    if (!Number.isFinite(flat) || !Number.isFinite(sr) || !Number.isFinite(sc)) continue;
    const row = Math.floor(flat / cols);
    const col = flat - row * cols;
    total++;
    if (Math.floor(sr / metaSubdiv) !== row || Math.floor(sc / metaSubdiv) !== col) bad++;
  }
  if (total === 0) return 0;
  return bad === 0 ? metaSubdiv : 0;
}
