import { I, type Mat, type Vec } from "./math";
import { COLORS } from "./theme";

const ORIGIN: Vec = { x: 0, y: 0 };

/**
 * 以画布中心为原点的 2D 坐标平面，y 轴向上。
 * 封装世界坐标 ↔ 屏幕坐标、网格/箭头/直线等绘制原语，支持滚轮缩放。
 */
export class Plane {
  readonly canvas: HTMLCanvasElement;
  readonly ctx: CanvasRenderingContext2D;
  /** 每单位长度的 CSS 像素数 */
  scale = 55;
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
        this.scale = Math.min(220, Math.max(12, this.scale * Math.exp(-e.deltaY * 0.0012)));
        this.onRedraw?.();
      },
      { passive: false },
    );
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

  toScreen(v: Vec): [number, number] {
    return [this.w / 2 + v.x * this.scale, this.h / 2 - v.y * this.scale];
  }

  toWorld(px: number, py: number): Vec {
    return { x: (px - this.w / 2) / this.scale, y: (this.h / 2 - py) / this.scale };
  }

  pointerWorld(e: PointerEvent): Vec {
    return this.toWorld(e.offsetX, e.offsetY);
  }

  /** 注册拖拽：按下/移动时以世界坐标回调 */
  attachDrag(onDrag: (v: Vec) => void): void {
    let dragging = false;
    this.canvas.addEventListener("pointerdown", (e) => {
      dragging = true;
      this.canvas.setPointerCapture(e.pointerId);
      onDrag(this.pointerWorld(e));
    });
    this.canvas.addEventListener("pointermove", (e) => {
      if (dragging) onDrag(this.pointerWorld(e));
    });
    this.canvas.addEventListener("pointerup", () => (dragging = false));
    this.canvas.addEventListener("pointercancel", () => (dragging = false));
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.w, this.h);
  }

  /** 画经过矩阵 m 变换后的整张网格（m 省略时为标准网格） */
  grid(m: Mat = I, color = COLORS.faint, width = 1): void {
    const N = 45;
    const col1: Vec = { x: m[0], y: m[2] };
    const col2: Vec = { x: m[1], y: m[3] };
    for (let k = -N; k <= N; k++) {
      // 竖线 x=k 的像：过 k·col1，方向 col2
      this.infLine({ x: k * col1.x, y: k * col1.y }, col2, color, width);
      // 横线 y=k 的像：过 k·col2，方向 col1
      this.infLine({ x: k * col2.x, y: k * col2.y }, col1, color, width);
    }
  }

  /** 过点 p、方向 dir 的无限直线 */
  infLine(p: Vec, dir: Vec, color: string, width = 1, dash?: number[]): void {
    const mag = Math.hypot(dir.x, dir.y);
    if (mag < 1e-9) return;
    const T = (this.w + this.h) / this.scale + Math.hypot(p.x, p.y) + 2;
    const ux = (dir.x / mag) * T;
    const uy = (dir.y / mag) * T;
    const [x1, y1] = this.toScreen({ x: p.x - ux, y: p.y - uy });
    const [x2, y2] = this.toScreen({ x: p.x + ux, y: p.y + uy });
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

  /** 坐标轴 + 整数刻度 */
  axes(): void {
    this.infLine(ORIGIN, { x: 1, y: 0 }, COLORS.axis, 1.5);
    this.infLine(ORIGIN, { x: 0, y: 1 }, COLORS.axis, 1.5);
    const step = this.scale > 35 ? 1 : this.scale > 16 ? 2 : 5;
    const ctx = this.ctx;
    ctx.fillStyle = COLORS.tick;
    ctx.font = "11px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const nx = Math.ceil(this.w / 2 / this.scale / step) * step;
    for (let k = -nx; k <= nx; k += step) {
      if (k === 0) continue;
      const [x, y] = this.toScreen({ x: k, y: 0 });
      ctx.fillText(String(k), x, y + 4);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const ny = Math.ceil(this.h / 2 / this.scale / step) * step;
    for (let k = -ny; k <= ny; k += step) {
      if (k === 0) continue;
      const [x, y] = this.toScreen({ x: 0, y: k });
      ctx.fillText(String(k), x - 5, y);
    }
  }

  arrow(from: Vec, to: Vec, color: string, width = 2.5, label = ""): void {
    const [x1, y1] = this.toScreen(from);
    const [x2, y2] = this.toScreen(to);
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    const ctx = this.ctx;
    ctx.save();
    // 主向量带一点粉笔的微光
    if (width >= 3) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
    }
    if (len > 1) {
      const head = Math.min(12, len * 0.45);
      const ux = dx / len;
      const uy = dy / len;
      const bx = x2 - ux * head;
      const by = y2 - uy * head;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(bx, by);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(bx - uy * head * 0.45, by + ux * head * 0.45);
      ctx.lineTo(bx + uy * head * 0.45, by - ux * head * 0.45);
      ctx.closePath();
      ctx.fill();
      if (label) {
        ctx.font = "italic 16px Cambria, Georgia, 'Times New Roman', serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, x2 + ux * 16 - uy * 12, y2 + uy * 16 + ux * 12);
      }
    } else if (label) {
      ctx.fillStyle = color;
      ctx.font = "italic 16px Cambria, Georgia, 'Times New Roman', serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x2 + 14, y2 - 14);
    }
    ctx.restore();
  }

  fillPoly(points: Vec[], color: string, alpha = 0.25): void {
    if (points.length < 3) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    const [x0, y0] = this.toScreen(points[0]);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < points.length; i++) {
      const [x, y] = this.toScreen(points[i]);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  point(v: Vec, color: string, r = 5, label = ""): void {
    const [x, y] = this.toScreen(v);
    const ctx = this.ctx;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    if (label) {
      ctx.font = "13px system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(label, x + r + 3, y - r - 1);
    }
  }

  ring(v: Vec, color: string, r = 12, width = 2.5): void {
    const [x, y] = this.toScreen(v);
    const ctx = this.ctx;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.stroke();
  }
}
