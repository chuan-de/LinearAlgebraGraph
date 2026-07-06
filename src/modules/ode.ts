import { apply, eigen, solve2, type Mat, type Vec } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt, matrixInput } from "../ui";

const PRESETS: { name: string; m: Mat }[] = [
  { name: "鞍点 [1 0.5; 0.5 -1]", m: [1, 0.5, 0.5, -1] },
  { name: "稳定结点 [-1 0; 0 -2]", m: [-1, 0, 0, -2] },
  { name: "中心 [0 -1; 1 0]（纯虚特征值）", m: [0, -1, 1, 0] },
  { name: "稳定螺旋 [-0.3 -1; 1 -0.3]", m: [-0.3, -1, 1, -0.3] },
  { name: "不稳定螺旋 [0.2 -1; 1 0.2]", m: [0.2, -1, 1, 0.2] },
];

const H = 0.02; // RK4 步长
const MAX_STEPS = 1500;
const DOT_SPEED = 1.6; // 动画点速度：世界单位/秒

export function mountOde(root: HTMLElement): () => void {
  let A: Mat = [...PRESETS[0].m] as Mat;
  let u0: Vec = { x: 2, y: 1.2 };
  let traj: Vec[] = [];
  let cum: number[] = []; // 轨迹弧长前缀和
  let totalLen = 0;
  let raf = 0;
  let t0 = performance.now();

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "微分方程 du/dt = Au 与相图"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 23。淡箭头是向量场：每一点的箭头指向 Au，解曲线处处与它相切。" +
        "拖动设置初始点 u(0)，金色曲线是从它出发的解轨迹（RK4 数值积分），" +
        "小点沿轨迹匀速播放。特征值决定相图类型：实 λ 沿特征方向（紫色虚线）指数增长/衰减，" +
        "复 λ = a±bi 则旋转，实部 a 决定螺旋向内还是向外。",
    ),
  );

  const mi = matrixInput(A, (m) => {
    A = m;
    recompute();
  });
  const miRow = el("div", "row");
  miRow.appendChild(el("span", "", "A ="));
  miRow.appendChild(mi.root);
  panel.appendChild(miRow);

  const sel = el("select");
  sel.appendChild(el("option", "", "选择预设矩阵…"));
  PRESETS.forEach((p, i) => {
    const o = el("option", "", p.name);
    o.value = String(i);
    sel.appendChild(o);
  });
  sel.value = "0";
  sel.onchange = () => {
    const p = PRESETS[Number(sel.value)];
    if (!p) return;
    A = [...p.m] as Mat;
    mi.set(A);
    recompute();
  };
  panel.appendChild(sel);

  const readout = el("div", "readout");
  panel.appendChild(readout);

  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 画布 ----
  const canvasCol = el("div", "canvas-col");
  const canvas = el("canvas", "plane");
  canvasCol.appendChild(canvas);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(canvasCol);
  root.appendChild(moduleEl);

  const plane = new Plane(canvas);
  plane.onRedraw = () => render();
  plane.attachDrag((v) => {
    if (Math.hypot(v.x, v.y) < 0.1) return;
    u0 = v;
    recompute();
  });

  function deriv(v: Vec): Vec {
    return apply(A, v);
  }

  function rk4Step(v: Vec): Vec {
    const k1 = deriv(v);
    const k2 = deriv({ x: v.x + (H / 2) * k1.x, y: v.y + (H / 2) * k1.y });
    const k3 = deriv({ x: v.x + (H / 2) * k2.x, y: v.y + (H / 2) * k2.y });
    const k4 = deriv({ x: v.x + H * k3.x, y: v.y + H * k3.y });
    return {
      x: v.x + (H / 6) * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
      y: v.y + (H / 6) * (k1.y + 2 * k2.y + 2 * k3.y + k4.y),
    };
  }

  /** 从 u(0) 积分轨迹并重建弧长表，然后重绘 */
  function recompute(): void {
    traj = [u0];
    cum = [0];
    totalLen = 0;
    let cur = u0;
    for (let i = 0; i < MAX_STEPS; i++) {
      const next = rk4Step(cur);
      const n = Math.hypot(next.x, next.y);
      totalLen += Math.hypot(next.x - cur.x, next.y - cur.y);
      traj.push(next);
      cum.push(totalLen);
      cur = next;
      if (n > 12 || n < 0.01) break;
    }
    t0 = performance.now();
    render();
  }

  /** 沿轨迹弧长 d 处的点（二分查找 + 线性插值） */
  function pointAt(d: number): Vec {
    let lo = 0;
    let hi = cum.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] < d) lo = mid + 1;
      else hi = mid;
    }
    if (lo === 0) return traj[0];
    const seg = cum[lo] - cum[lo - 1];
    const t = seg > 1e-12 ? (d - cum[lo - 1]) / seg : 0;
    const a = traj[lo - 1];
    const b = traj[lo];
    return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
  }

  function drawField(): void {
    const ctx = plane.ctx;
    let maxM = 0;
    for (let x = -5; x <= 5; x++) {
      for (let y = -5; y <= 5; y++) {
        const v = deriv({ x, y });
        maxM = Math.max(maxM, Math.hypot(v.x, v.y));
      }
    }
    if (maxM < 1e-9) return;
    for (let x = -5; x <= 5; x++) {
      for (let y = -5; y <= 5; y++) {
        const p: Vec = { x, y };
        const v = deriv(p);
        const m = Math.hypot(v.x, v.y);
        if (m < 1e-6) continue;
        const to: Vec = { x: p.x + (v.x / m) * 0.35, y: p.y + (v.y / m) * 0.35 };
        ctx.save();
        ctx.globalAlpha = 0.45 + 0.55 * (m / maxM);
        plane.arrow(p, to, COLORS.dim, 1.2);
        ctx.restore();
      }
    }
  }

  function render(): void {
    const o: Vec = { x: 0, y: 0 };
    const eg = eigen(A);
    const ctx = plane.ctx;

    plane.clear();
    plane.grid();
    plane.axes();
    drawField();

    // 特征方向（实特征值时）
    if (eg.real) {
      eg.vectors.forEach((v, i) => {
        plane.infLine(o, v, COLORS.purple, 1.2, [7, 6]);
        const tip = { x: v.x * 3.4, y: v.y * 3.4 };
        const [sx, sy] = plane.toScreen(tip);
        ctx.fillStyle = COLORS.purple;
        ctx.font = "13px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`λ${i === 0 ? "₁" : "₂"} = ${fmt(eg.values[i])}`, sx, sy - 8);
      });
    }

    // 解轨迹
    if (traj.length > 1) {
      ctx.save();
      ctx.strokeStyle = COLORS.gold;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      const [x0, y0] = plane.toScreen(traj[0]);
      ctx.moveTo(x0, y0);
      for (let i = 1; i < traj.length; i++) {
        const [x, y] = plane.toScreen(traj[i]);
        ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    // 初始点 + 沿轨迹匀速前进的动画点
    plane.point(u0, COLORS.chalk, 5, "u(0)");
    plane.ring(u0, COLORS.dim, 10, 1.5);
    if (totalLen > 1e-6) {
      const d = (((performance.now() - t0) / 1000) * DOT_SPEED) % totalLen;
      plane.point(pointAt(d), COLORS.red, 5.5);
    }

    // ---- 读出区 ----
    let text: string;
    if (eg.real) {
      const [l1, l2] = eg.values;
      const [v1, v2] = eg.vectors;
      text =
        `λ₁ = ${fmt(l1)}   v₁ = (${fmt(v1.x)}, ${fmt(v1.y)})\n` +
        `λ₂ = ${fmt(l2)}   v₂ = (${fmt(v2.x)}, ${fmt(v2.y)})\n\n` +
        "通解 u(t) = c₁e^{λ₁t}v₁ + c₂e^{λ₂t}v₂\n";
      const c = solve2([v1.x, v2.x, v1.y, v2.y], u0);
      if (c) {
        text += `由 u(0) 定出 c₁ = ${fmt(c.x)}，c₂ = ${fmt(c.y)}\n\n`;
      } else {
        text += "v₁ ∥ v₂（重根且特征向量不足），u(0) 无法按特征向量分解\n\n";
      }
      text += "每个特征方向独立演化：λ > 0 沿 v 指数远离原点，λ < 0 指数趋向原点。";
    } else {
      const [a, b] = eg.values;
      text =
        `特征值为复数：λ = ${fmt(a)} ± ${fmt(b)}i\n\n` +
        `解含振荡因子 e^{${fmt(a)}t}(cos ${fmt(b)}t, sin ${fmt(b)}t)：\n` +
        `虚部 b = ${fmt(b)} 提供旋转，实部 a = ${fmt(a)} 决定模长——\n`;
      if (a < -1e-6) text += "a < 0 → 边转边衰减，螺旋卷向原点。";
      else if (a > 1e-6) text += "a > 0 → 边转边放大，螺旋向外发散。";
      else text += "a = 0 → 模长不变，轨迹是闭合的椭圆轨道。";
    }
    readout.textContent = `u(0) = (${fmt(u0.x)}, ${fmt(u0.y)})\n\n` + text;

    // ---- 相图分类 ----
    status.innerHTML = `相图类型：<span class="status-ok">${classify(eg.real, eg.values)}</span>`;
  }

  function classify(real: boolean, values: [number, number]): string {
    const EPS = 1e-6;
    if (!real) {
      if (values[0] < -EPS) return "稳定螺旋（实部 < 0，卷向原点）";
      if (values[0] > EPS) return "不稳定螺旋（实部 > 0，向外发散）";
      return "中心（纯虚特征值，轨迹闭合）";
    }
    const [l1, l2] = values;
    if (l1 * l2 < -EPS) return "鞍点（λ 异号：一个方向进、一个方向出）";
    if (l1 < -EPS && l2 < -EPS) return "稳定结点（λ 都 < 0，全部流向原点）";
    if (l1 > EPS && l2 > EPS) return "不稳定结点（λ 都 > 0，全部流出）";
    return "退化情形（有 λ ≈ 0，沿该特征方向是一整条不动的直线）";
  }

  const onResize = (): void => {
    plane.resize();
    render();
  };
  window.addEventListener("resize", onResize);

  recompute();

  const frame = (): void => {
    render();
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);

  return () => {
    cancelAnimationFrame(raf);
    window.removeEventListener("resize", onResize);
  };
}
