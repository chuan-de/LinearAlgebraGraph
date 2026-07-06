import { solve2, type Mat, type Vec } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt, matrixInput, sliderRow, vecInput } from "../ui";

const HIT_TOL = 0.12;

export function mountRowCol(root: HTMLElement): () => void {
  let A: Mat = [2, -1, 1, 1];
  let b: Vec = { x: 1, y: 5 };
  let trial: Vec = { x: 1, y: 1 }; // 试探解 (x, y)

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "Ax = b：行图像 vs 列图像"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 1。同一个方程组的两种看法：" +
        "<b>行图像</b>——每个方程是一条直线，解是交点；" +
        "<b>列图像</b>——寻找列向量的线性组合 x·c₁ + y·c₂ = b。" +
        "拖动左图中的白点，或用滑块调 x、y，观察右图的组合如何逼近 b。",
    ),
  );

  const mi = matrixInput(A, (m) => {
    A = m;
    render();
  });
  const vi = vecInput(b, (v) => {
    b = v;
    render();
  });
  const eqRow = el("div", "row");
  eqRow.appendChild(mi.root);
  eqRow.appendChild(el("span", "", "[x, y]ᵀ ="));
  eqRow.appendChild(vi.root);
  panel.appendChild(eqRow);

  const eqReadout = el("div", "readout");
  panel.appendChild(eqReadout);

  const sx = sliderRow("x", -5, 5, 0.05, trial.x, (v) => {
    trial = { ...trial, x: v };
    render();
  });
  const sy = sliderRow("y", -5, 5, 0.05, trial.y, (v) => {
    trial = { ...trial, y: v };
    render();
  });
  panel.appendChild(sx.root);
  panel.appendChild(sy.root);

  const solveBtn = el("button", "btn primary", "求解 x = A⁻¹b");
  solveBtn.onclick = () => {
    const sol = solve2(A, b);
    if (sol) setTrial(sol);
    else render();
  };
  panel.appendChild(solveBtn);

  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 两块画布 ----
  const rowCol = el("div", "canvas-col");
  rowCol.appendChild(
    el(
      "h3",
      "",
      `行图像：<span class="dot" style="background:#7fd0cf"></span>方程 1、` +
        `<span class="dot" style="background:#e88a9c"></span>方程 2，交点即解（拖动白点试探）`,
    ),
  );
  const rowCanvas = el("canvas", "plane");
  rowCol.appendChild(rowCanvas);

  const colCol = el("div", "canvas-col");
  colCol.appendChild(
    el(
      "h3",
      "",
      `列图像：<span class="dot" style="background:#8fd6a2"></span>x·c₁ 接 ` +
        `<span class="dot" style="background:#e88a9c"></span>y·c₂，目标是 ` +
        `<span class="dot" style="background:#e6c860"></span>b`,
    ),
  );
  const colCanvas = el("canvas", "plane");
  colCol.appendChild(colCanvas);

  const canvases = el("div", "canvas-row");
  canvases.appendChild(rowCol);
  canvases.appendChild(colCol);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(canvases);
  root.appendChild(moduleEl);

  const rowPlane = new Plane(rowCanvas);
  const colPlane = new Plane(colCanvas);
  rowPlane.scale = 42;
  colPlane.scale = 42;
  rowPlane.onRedraw = () => render();
  colPlane.onRedraw = () => render();
  rowPlane.attachDrag((v) => setTrial(v));

  function setTrial(v: Vec): void {
    trial = v;
    const bound = Math.max(5, Math.ceil(Math.max(Math.abs(v.x), Math.abs(v.y))) + 1);
    sx.setRange(-bound, bound);
    sy.setRange(-bound, bound);
    sx.set(v.x);
    sy.set(v.y);
    render();
  }

  /** 直线 ax + by = e：返回 [线上离原点最近的点, 方向]，系数全零时返回 null */
  function lineOf(a: number, bb: number, e: number): [Vec, Vec] | null {
    const n2 = a * a + bb * bb;
    if (n2 < 1e-12) return null;
    return [
      { x: (a * e) / n2, y: (bb * e) / n2 },
      { x: -bb, y: a },
    ];
  }

  function render(): void {
    const o = { x: 0, y: 0 };
    const c1: Vec = { x: A[0], y: A[2] };
    const c2: Vec = { x: A[1], y: A[3] };
    const p1: Vec = { x: trial.x * c1.x, y: trial.x * c1.y }; // x·c₁
    const end: Vec = { x: p1.x + trial.y * c2.x, y: p1.y + trial.y * c2.y }; // x·c₁ + y·c₂
    const sol = solve2(A, b);
    const hit = Math.hypot(end.x - b.x, end.y - b.y) < HIT_TOL;

    // --- 行图像 ---
    rowPlane.clear();
    rowPlane.grid();
    rowPlane.axes();
    const l1 = lineOf(A[0], A[1], b.x);
    const l2 = lineOf(A[2], A[3], b.y);
    if (l1) rowPlane.infLine(l1[0], l1[1], COLORS.cyan, 2);
    if (l2) rowPlane.infLine(l2[0], l2[1], COLORS.red, 2);
    if (sol) {
      rowPlane.point(sol, COLORS.gold, 5.5, `解 (${fmt(sol.x)}, ${fmt(sol.y)})`);
      rowPlane.ring(sol, COLORS.gold, 10, 1.5);
    }
    rowPlane.point(trial, COLORS.chalk, 6, `(${fmt(trial.x)}, ${fmt(trial.y)})`);

    // --- 列图像 ---
    colPlane.clear();
    colPlane.grid();
    colPlane.axes();
    colPlane.arrow(o, c1, COLORS.green, 2, "c₁");
    colPlane.arrow(o, c2, COLORS.red, 2, "c₂");
    colPlane.arrow(o, b, COLORS.gold, 3.5, "b");
    // 组合链：先走 x·c₁，再接 y·c₂
    colPlane.arrow(o, p1, "rgba(143,214,162,0.75)", 4);
    colPlane.arrow(p1, end, "rgba(232,138,156,0.75)", 4);
    colPlane.point(end, hit ? COLORS.gold : COLORS.chalk, 5);
    if (hit) colPlane.ring(b, COLORS.gold, 14, 2.5);

    // --- 面板文字 ---
    const eq = (a: number, c: number, e: number): string =>
      `${fmt(a)}x ${c < 0 ? "−" : "+"} ${fmt(Math.abs(c))}y = ${fmt(e)}`;
    eqReadout.textContent =
      `${eq(A[0], A[1], b.x)}\n` +
      `${eq(A[2], A[3], b.y)}\n\n` +
      `${fmt(trial.x)}·c₁ + ${fmt(trial.y)}·c₂ = (${fmt(end.x)}, ${fmt(end.y)})\n` +
      `目标 b = (${fmt(b.x)}, ${fmt(b.y)})，误差 ${fmt(Math.hypot(end.x - b.x, end.y - b.y))}`;

    if (hit) {
      status.innerHTML =
        `<span class="status-ok">✓ 命中！x·c₁ + y·c₂ = b。</span>` +
        `左图中白点也正好落在两条直线的交点上——两种图像说的是同一件事。`;
    } else if (!sol) {
      status.innerHTML =
        `det(A) = 0：两列共线（行图像中两条直线平行或重合）。` +
        `b 不在列空间中时<b>无解</b>，在列空间中时有<b>无穷多解</b>。`;
    } else {
      status.textContent = "调整 x、y 让组合的终点落到金色向量 b 的尖端上，或点「求解」。";
    }
  }

  const onResize = (): void => {
    rowPlane.resize();
    colPlane.resize();
    render();
  };
  window.addEventListener("resize", onResize);

  render();

  return () => {
    window.removeEventListener("resize", onResize);
  };
}
