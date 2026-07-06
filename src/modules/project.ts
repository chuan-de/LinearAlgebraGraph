import { solve2, type Mat, type Vec } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt } from "../ui";

const EPS = 1e-9;

export function mountProject(root: HTMLElement): () => void {
  // 左图状态：投影 p = a(aᵀb/aᵀa)
  let a: Vec = { x: 3, y: 1 };
  let b: Vec = { x: 1, y: 2.5 };
  let leftTarget: "a" | "b" | null = null;

  // 右图状态：5 个数据点 (tᵢ, yᵢ)
  const pts: Vec[] = [
    { x: -3, y: -1.5 },
    { x: -1.5, y: 0 },
    { x: 0, y: 0.5 },
    { x: 1.5, y: 1 },
    { x: 3, y: 2.8 },
  ];
  let rightTarget = -1;

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "投影与最小二乘"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 15–16。<b>左图</b>：把 b 投影到 a 的方向上，" +
        "p = a(aᵀb/aᵀa)，误差 e = b − p 与 a 垂直——这是让 ‖e‖ 最小的选择。" +
        "<b>右图</b>：Ax = b 无解时退而求其次，解正规方程 AᵀAx̂ = Aᵀb，" +
        "得到最佳直线 y = C + Dt。两张图说的是同一件事：" +
        "拟合就是把 b 投影到 A 的列空间上，竖直的玫瑰色线段合起来就是误差 e。",
    ),
  );

  const readout = el("div", "readout");
  panel.appendChild(readout);
  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 两块画布 ----
  const projCol = el("div", "canvas-col");
  projCol.appendChild(el("h3", "", "向量投影：p = a(aᵀb/aᵀa)，拖动 a 或 b"));
  const projCanvas = el("canvas", "plane");
  projCol.appendChild(projCanvas);

  const lsqCol = el("div", "canvas-col");
  lsqCol.appendChild(el("h3", "", "最小二乘拟合 y = C + Dt，拖动数据点"));
  const lsqCanvas = el("canvas", "plane");
  lsqCol.appendChild(lsqCanvas);

  const canvases = el("div", "canvas-row");
  canvases.appendChild(projCol);
  canvases.appendChild(lsqCol);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(canvases);
  root.appendChild(moduleEl);

  const projPlane = new Plane(projCanvas);
  const lsqPlane = new Plane(lsqCanvas);
  projPlane.scale = 48;
  lsqPlane.scale = 48;
  projPlane.onRedraw = () => render();
  lsqPlane.onRedraw = () => render();

  // 左图拖拽：按下时选较近的向量尖端，拖动期间锁定
  projPlane.attachDrag((v) => {
    if (leftTarget === null) {
      const da = Math.hypot(v.x - a.x, v.y - a.y);
      const db = Math.hypot(v.x - b.x, v.y - b.y);
      leftTarget = da <= db ? "a" : "b";
    }
    if (leftTarget === "a") a = v;
    else b = v;
    render();
  });
  const endLeft = (): void => {
    leftTarget = null;
  };
  projCanvas.addEventListener("pointerup", endLeft);
  projCanvas.addEventListener("pointercancel", endLeft);

  // 右图拖拽：按下时选最近的数据点，拖动期间锁定
  lsqPlane.attachDrag((v) => {
    if (rightTarget < 0) {
      let best = 0;
      let bestD = Infinity;
      pts.forEach((p, i) => {
        const d = Math.hypot(v.x - p.x, v.y - p.y);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      });
      rightTarget = best;
    }
    pts[rightTarget] = v;
    render();
  });
  const endRight = (): void => {
    rightTarget = -1;
  };
  lsqCanvas.addEventListener("pointerup", endRight);
  lsqCanvas.addEventListener("pointercancel", endRight);

  /** 屏幕坐标下的线段（用于虚线误差、残差、直角标记） */
  function segment(
    plane: Plane,
    from: Vec,
    to: Vec,
    color: string,
    width: number,
    dash?: number[],
  ): void {
    const [x1, y1] = plane.toScreen(from);
    const [x2, y2] = plane.toScreen(to);
    const ctx = plane.ctx;
    ctx.save();
    if (dash) ctx.setLineDash(dash);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.restore();
  }

  /** 在点 p 处画直角标记：两条短边分别沿 dirA（a 线方向）与 dirE（误差方向） */
  function rightAngleMark(p: Vec, dirA: Vec, dirE: Vec): void {
    const na = Math.hypot(dirA.x, dirA.y);
    const ne = Math.hypot(dirE.x, dirE.y);
    if (na < EPS || ne < EPS) return;
    const [px, py] = projPlane.toScreen(p);
    // 屏幕坐标 y 轴向下，取反
    const ax = dirA.x / na;
    const ay = -dirA.y / na;
    const ex = dirE.x / ne;
    const ey = -dirE.y / ne;
    const s = 9;
    const ctx = projPlane.ctx;
    ctx.save();
    ctx.strokeStyle = COLORS.dim;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px + ax * s, py + ay * s);
    ctx.lineTo(px + ax * s + ex * s, py + ay * s + ey * s);
    ctx.lineTo(px + ex * s, py + ey * s);
    ctx.stroke();
    ctx.restore();
  }

  function render(): void {
    const o: Vec = { x: 0, y: 0 };

    // ---- 左图：向量投影 ----
    projPlane.clear();
    projPlane.grid();
    projPlane.axes();

    const aa = a.x * a.x + a.y * a.y;
    const ab = a.x * b.x + a.y * b.y;
    let projText: string;
    if (aa < EPS) {
      projPlane.arrow(o, b, COLORS.chalk, 3, "b");
      projText = "a 是零向量：没有方向可投影，请把 a 拖离原点。";
    } else {
      const xhat = ab / aa;
      const p: Vec = { x: a.x * xhat, y: a.y * xhat };
      const e: Vec = { x: b.x - p.x, y: b.y - p.y };

      // a 张成的一维子空间（淡虚线）
      projPlane.infLine(o, a, COLORS.dim, 1, [6, 6]);
      // e = b − p：从 p 到 b 尖端的玫瑰色虚线
      segment(projPlane, p, b, COLORS.red, 2, [7, 5]);
      if (Math.hypot(e.x, e.y) > 0.04) {
        const [mx, my] = projPlane.toScreen({ x: p.x + e.x / 2, y: p.y + e.y / 2 });
        const ctx = projPlane.ctx;
        ctx.save();
        ctx.fillStyle = COLORS.red;
        ctx.font = "italic 15px Cambria, Georgia, 'Times New Roman', serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("e", mx + 10, my);
        ctx.restore();
        // 直角标记：沿 a 线（朝原点一侧）与 e 方向
        const dirA: Vec =
          Math.hypot(p.x, p.y) > EPS ? { x: -p.x, y: -p.y } : { x: a.x, y: a.y };
        rightAngleMark(p, dirA, e);
      }

      projPlane.arrow(o, a, COLORS.green, 3, "a");
      projPlane.arrow(o, b, COLORS.chalk, 3, "b");
      projPlane.arrow(o, p, COLORS.gold, 3, "p");

      projText =
        `投影：aᵀb = ${fmt(ab)}，aᵀa = ${fmt(aa)}，x̂ = aᵀb/aᵀa = ${fmt(xhat)}\n` +
        `p = x̂·a = (${fmt(p.x)}, ${fmt(p.y)})   e = b − p = (${fmt(e.x)}, ${fmt(e.y)})\n` +
        `aᵀe = ${fmt(a.x * e.x + a.y * e.y)}（垂直 ⇒ ‖e‖ 最小）`;
    }

    // ---- 右图：最小二乘 ----
    lsqPlane.clear();
    lsqPlane.grid();
    lsqPlane.axes();

    const n = pts.length;
    let st = 0;
    let stt = 0;
    let sy = 0;
    let sty = 0;
    for (const pt of pts) {
      st += pt.x;
      stt += pt.x * pt.x;
      sy += pt.y;
      sty += pt.x * pt.y;
    }
    const AtA: Mat = [n, st, st, stt];
    const Atb: Vec = { x: sy, y: sty };
    const sol = solve2(AtA, Atb); // sol.x = C, sol.y = D

    let lsqText: string;
    if (sol) {
      const C = sol.x;
      const D = sol.y;
      // 拟合直线与竖直残差
      lsqPlane.infLine({ x: 0, y: C }, { x: 1, y: D }, COLORS.gold, 2.5);
      let sse = 0;
      for (const pt of pts) {
        const fit: Vec = { x: pt.x, y: C + D * pt.x };
        const r = pt.y - fit.y;
        sse += r * r;
        segment(lsqPlane, pt, fit, COLORS.red, 1.5);
      }
      for (const pt of pts) lsqPlane.point(pt, COLORS.chalk, 5);

      lsqText =
        `最小二乘：AᵀA x̂ = Aᵀb（A 的列是 [1, tᵢ]）\n` +
        `AᵀA = [${fmt(n)} ${fmt(st)}; ${fmt(st)} ${fmt(stt)}]   ` +
        `Aᵀb = (${fmt(sy)}, ${fmt(sty)})\n` +
        `C = ${fmt(sol.x)}，D = ${fmt(sol.y)}  →  y = ${fmt(sol.x)} + ${fmt(sol.y)}t\n` +
        `残差平方和 ‖e‖² = ${fmt(sse)}`;
    } else {
      for (const pt of pts) lsqPlane.point(pt, COLORS.chalk, 5);
      lsqText = "AᵀA 奇异：所有数据点的 t 相同，A 的两列共线，直线不唯一。";
    }

    readout.textContent = `${projText}\n\n${lsqText}`;

    if (aa < EPS) {
      status.textContent = "左图：a 是零向量，无法定义投影方向。";
    } else if (!sol) {
      status.textContent = "右图：把数据点拖到不同的 t 上，AᵀA 才可逆。";
    } else {
      status.innerHTML =
        `左：拖动 a、b 看 p 与 e 如何变化；右：拖动数据点，金色直线始终是` +
        `<span class="status-ok">让残差平方和最小</span>的那条。`;
    }
  }

  const onResize = (): void => {
    projPlane.resize();
    lsqPlane.resize();
    render();
  };
  window.addEventListener("resize", onResize);

  render();

  return () => {
    window.removeEventListener("resize", onResize);
  };
}
