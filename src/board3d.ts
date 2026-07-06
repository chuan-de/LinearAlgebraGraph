/**
 * 极简 3D 黑板：正交投影 + 拖拽旋转，专为“三张平面交于一点”这类演示。
 * 约定 z 轴朝上；yaw 绕 z 轴、pitch 绕屏幕水平轴。
 */

export interface V3 {
  x: number;
  y: number;
  z: number;
}

export class Board3D {
  readonly canvas: HTMLCanvasElement;
  readonly ctx: CanvasRenderingContext2D;
  scale = 42;
  yaw = -0.6;
  pitch = 0.42;
  onRedraw: (() => void) | null = null;
  private w = 0;
  private h = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.resize();
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.scale = Math.min(120, Math.max(14, this.scale * Math.exp(-e.deltaY * 0.0012)));
        this.onRedraw?.();
      },
      { passive: false },
    );
    // 拖拽旋转视角
    let dragging = false;
    let px = 0;
    let py = 0;
    canvas.addEventListener("pointerdown", (e) => {
      dragging = true;
      px = e.clientX;
      py = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      this.yaw += (e.clientX - px) * 0.008;
      this.pitch = Math.min(1.45, Math.max(-1.45, this.pitch + (e.clientY - py) * 0.008));
      px = e.clientX;
      py = e.clientY;
      this.onRedraw?.();
    });
    canvas.addEventListener("pointerup", () => (dragging = false));
    canvas.addEventListener("pointercancel", () => (dragging = false));
  }

  resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.w = rect.width;
    this.h = rect.height;
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /** 返回 [屏幕x, 屏幕y, 深度]，深度越大越远 */
  project(v: V3): [number, number, number] {
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    const x1 = v.x * cy - v.y * sy;
    const y1 = v.x * sy + v.y * cy;
    const depth = y1 * cp - v.z * sp;
    const up = y1 * sp + v.z * cp;
    return [this.w / 2 + x1 * this.scale, this.h / 2 - up * this.scale, depth];
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.w, this.h);
  }

  line(a: V3, b: V3, color: string, width = 1, dash?: number[]): void {
    const [x1, y1] = this.project(a);
    const [x2, y2] = this.project(b);
    const ctx = this.ctx;
    ctx.save();
    if (dash) ctx.setLineDash(dash);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.restore();
  }

  axes(len: number, color: string, labelColor: string): void {
    const o: V3 = { x: 0, y: 0, z: 0 };
    const ends: [V3, string][] = [
      [{ x: len, y: 0, z: 0 }, "x"],
      [{ x: 0, y: len, z: 0 }, "y"],
      [{ x: 0, y: 0, z: len }, "z"],
    ];
    const ctx = this.ctx;
    for (const [end, label] of ends) {
      this.line(o, end, color, 1.4);
      this.line(o, { x: -end.x, y: -end.y, z: -end.z }, color, 0.7, [3, 5]);
      const [x, y] = this.project({ x: end.x * 1.09, y: end.y * 1.09, z: end.z * 1.09 });
      ctx.fillStyle = labelColor;
      ctx.font = "italic 15px Cambria, Georgia, serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x, y);
    }
  }

  polygon(pts: V3[], fill: string, alpha: number, stroke: string): void {
    if (pts.length < 3) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath();
    const [x0, y0] = this.project(pts[0]);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < pts.length; i++) {
      const [x, y] = this.project(pts[i]);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.globalAlpha = Math.min(1, alpha * 2.4);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.restore();
  }

  point(v: V3, color: string, r = 5, label = ""): void {
    const [x, y] = this.project(v);
    const ctx = this.ctx;
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    if (label) {
      ctx.fillStyle = color;
      ctx.font = "13px 'Segoe UI', 'Microsoft YaHei', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(label, x + r + 4, y - r - 2);
    }
  }

  /** 平面深度（用于画远近排序）：取多边形形心的投影深度 */
  depthOf(pts: V3[]): number {
    let x = 0;
    let y = 0;
    let z = 0;
    for (const p of pts) {
      x += p.x;
      y += p.y;
      z += p.z;
    }
    const n = pts.length || 1;
    return this.project({ x: x / n, y: y / n, z: z / n })[2];
  }
}

/** 把平面 n·p = d 裁剪到立方体 [-box, box]³，返回按角度排序的多边形顶点 */
export function planeInBox(n: V3, d: number, box: number): V3[] {
  const mag = Math.hypot(n.x, n.y, n.z);
  if (mag < 1e-12) return [];
  const f = (p: V3): number => n.x * p.x + n.y * p.y + n.z * p.z - d;
  const corners: V3[] = [];
  for (const sx of [-box, box])
    for (const sy of [-box, box]) for (const sz of [-box, box]) corners.push({ x: sx, y: sy, z: sz });
  // 立方体 12 条棱
  const edges: [number, number][] = [
    [0, 1], [2, 3], [4, 5], [6, 7], // z 方向
    [0, 2], [1, 3], [4, 6], [5, 7], // y 方向
    [0, 4], [1, 5], [2, 6], [3, 7], // x 方向
  ];
  const pts: V3[] = [];
  for (const [i, j] of edges) {
    const fa = f(corners[i]);
    const fb = f(corners[j]);
    if ((fa < 0 && fb < 0) || (fa > 0 && fb > 0)) continue;
    const t = Math.abs(fa - fb) < 1e-12 ? 0.5 : fa / (fa - fb);
    if (t < -1e-9 || t > 1 + 1e-9) continue;
    const a = corners[i];
    const b = corners[j];
    pts.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t });
  }
  if (pts.length < 3) return [];
  // 以平面内的正交基按角度排序
  const u0: V3 = Math.abs(n.x) < 0.9 ? { x: 1, y: 0, z: 0 } : { x: 0, y: 1, z: 0 };
  const u: V3 = normalize3(cross(n, u0));
  const v: V3 = normalize3(cross(n, u));
  const c: V3 = {
    x: pts.reduce((s, p) => s + p.x, 0) / pts.length,
    y: pts.reduce((s, p) => s + p.y, 0) / pts.length,
    z: pts.reduce((s, p) => s + p.z, 0) / pts.length,
  };
  return pts
    .map((p) => {
      const dx = { x: p.x - c.x, y: p.y - c.y, z: p.z - c.z };
      return { p, ang: Math.atan2(dot3(dx, v), dot3(dx, u)) };
    })
    .sort((a, b) => a.ang - b.ang)
    .map((e) => e.p);
}

function cross(a: V3, b: V3): V3 {
  return { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x };
}

function dot3(a: V3, b: V3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function normalize3(v: V3): V3 {
  const n = Math.hypot(v.x, v.y, v.z);
  return n < 1e-12 ? { x: 0, y: 0, z: 0 } : { x: v.x / n, y: v.y / n, z: v.z / n };
}
