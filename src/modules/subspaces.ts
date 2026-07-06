import { apply, det, normalize, type Mat, type Vec } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt, matrixInput } from "../ui";

const PRESETS: { name: string; m: Mat }[] = [
  { name: "秩 1：[1 2; 2 4]", m: [1, 2, 2, 4] },
  { name: "秩 1：[1 -1; -2 2]", m: [1, -1, -2, 2] },
  { name: "满秩（可逆）：[2 1; 1 2]", m: [2, 1, 1, 2] },
  { name: "零矩阵（秩 0）", m: [0, 0, 0, 0] },
];

interface Spaces {
  rank: 0 | 1 | 2;
  row?: Vec; // C(Aᵀ) 方向
  nul?: Vec; // N(A) 方向
  col?: Vec; // C(A) 方向
  lnul?: Vec; // N(Aᵀ) 方向
}

function analyze(A: Mat): Spaces {
  if (A.every((v) => Math.abs(v) < 1e-9)) return { rank: 0 };
  if (Math.abs(det(A)) > 1e-9) return { rank: 2 };
  const r1: Vec = { x: A[0], y: A[1] };
  const r2: Vec = { x: A[2], y: A[3] };
  const row = normalize(Math.hypot(r1.x, r1.y) > 1e-9 ? r1 : r2);
  const c1: Vec = { x: A[0], y: A[2] };
  const c2: Vec = { x: A[1], y: A[3] };
  const col = normalize(Math.hypot(c1.x, c1.y) > 1e-9 ? c1 : c2);
  return {
    rank: 1,
    row,
    nul: { x: -row.y, y: row.x },
    col,
    lnul: { x: -col.y, y: col.x },
  };
}

export function mountSubspaces(root: HTMLElement): () => void {
  let A: Mat = [1, 2, 2, 4];
  let x: Vec = { x: 2, y: 1.5 };

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "四个基本子空间"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 6–10。秩 1 矩阵最能说明问题：输入空间被劈成互相垂直的" +
        "<b>行空间 C(Aᵀ)</b> 与<b>零空间 N(A)</b>。把 x 分解为 x = x_row + x_null，" +
        "零空间分量被 A 湮灭，Ax 完全由行空间分量决定，并且永远落在<b>列空间 C(A)</b> 里。" +
        "拖动左图中的 x 试试。",
    ),
  );

  const mi = matrixInput(A, (m) => {
    A = m;
    render();
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
  sel.onchange = () => {
    const p = PRESETS[Number(sel.value)];
    if (!p) return;
    A = [...p.m];
    mi.set(A);
    render();
  };
  panel.appendChild(sel);

  const readout = el("div", "readout");
  panel.appendChild(readout);

  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 两块黑板 ----
  const leftCol = el("div", "canvas-col");
  leftCol.appendChild(
    el(
      "h3",
      "",
      `输入空间 ℝ²：<span class="dot" style="background:#8fd6a2"></span>行空间 C(Aᵀ)、` +
        `<span class="dot" style="background:#e88a9c"></span>零空间 N(A)（拖动 x）`,
    ),
  );
  const leftCanvas = el("canvas", "plane");
  leftCol.appendChild(leftCanvas);

  const rightCol = el("div", "canvas-col");
  rightCol.appendChild(
    el(
      "h3",
      "",
      `输出空间 ℝ²：<span class="dot" style="background:#82b4e8"></span>列空间 C(A)、` +
        `<span class="dot" style="background:#9aa8a0"></span>左零空间 N(Aᵀ)`,
    ),
  );
  const rightCanvas = el("canvas", "plane");
  rightCol.appendChild(rightCanvas);

  const canvases = el("div", "canvas-row");
  canvases.appendChild(leftCol);
  canvases.appendChild(rightCol);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(canvases);
  root.appendChild(moduleEl);

  const left = new Plane(leftCanvas);
  const right = new Plane(rightCanvas);
  left.scale = 46;
  right.scale = 46;
  left.onRedraw = () => render();
  right.onRedraw = () => render();
  left.attachDrag((v) => {
    x = v;
    render();
  });

  /** 在直线方向尽头写标签 */
  function lineLabel(plane: Plane, dir: Vec, text: string, color: string, len = 4.4): void {
    const [sx, sy] = plane.toScreen({ x: dir.x * len, y: dir.y * len });
    const ctx = plane.ctx;
    ctx.fillStyle = color;
    ctx.font = "12.5px 'Segoe UI', 'Microsoft YaHei', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, sx, sy - 12);
  }

  function render(): void {
    const o = { x: 0, y: 0 };
    const sp = analyze(A);
    const ax = apply(A, x);

    // 分解 x = x_row + x_null（秩 1 时）
    let xr: Vec = x;
    let xn: Vec = o;
    if (sp.rank === 1 && sp.row) {
      const d = x.x * sp.row.x + x.y * sp.row.y;
      xr = { x: sp.row.x * d, y: sp.row.y * d };
      xn = { x: x.x - xr.x, y: x.y - xr.y };
    } else if (sp.rank === 0) {
      xr = o;
      xn = x;
    }

    // --- 左：输入空间 ---
    left.clear();
    left.grid();
    left.axes();
    if (sp.rank === 1 && sp.row && sp.nul) {
      left.infLine(o, sp.row, COLORS.green, 2);
      left.infLine(o, sp.nul, COLORS.red, 2);
      lineLabel(left, sp.row, "C(Aᵀ) 行空间", COLORS.green);
      lineLabel(left, sp.nul, "N(A) 零空间", COLORS.red);
      // 分解的平行四边形辅助线
      left.infLine(x, sp.row, COLORS.dim, 1, [4, 5]);
      left.infLine(x, sp.nul, COLORS.dim, 1, [4, 5]);
      left.arrow(o, xr, COLORS.gold, 3, "x_row");
      left.arrow(o, xn, COLORS.cyan, 3, "x_null");
    }
    left.arrow(o, x, COLORS.chalk, 3, "x");

    // --- 右：输出空间 ---
    right.clear();
    right.grid();
    right.axes();
    if (sp.rank === 1 && sp.col && sp.lnul) {
      right.infLine(o, sp.col, COLORS.blue, 2);
      right.infLine(o, sp.lnul, COLORS.dim, 1.5, [7, 6]);
      lineLabel(right, sp.col, "C(A) 列空间", COLORS.blue);
      lineLabel(right, sp.lnul, "N(Aᵀ) 左零空间", COLORS.tick);
    }
    right.arrow(o, ax, COLORS.gold, 3, "Ax");

    // --- 面板 ---
    const r = sp.rank;
    readout.textContent =
      `rank(A) = ${r}\n` +
      `dim C(A)  = ${r}   dim N(A)  = ${2 - r}\n` +
      `dim C(Aᵀ) = ${r}   dim N(Aᵀ) = ${2 - r}\n\n` +
      `x      = (${fmt(x.x)}, ${fmt(x.y)})\n` +
      `x_row  = (${fmt(xr.x)}, ${fmt(xr.y)})\n` +
      `x_null = (${fmt(xn.x)}, ${fmt(xn.y)})\n\n` +
      `A·x    = (${fmt(ax.x)}, ${fmt(ax.y)})\n` +
      `A·x_row= (${fmt(apply(A, xr).x)}, ${fmt(apply(A, xr).y)}) ← 相同`;

    if (r === 1) {
      status.innerHTML =
        `<span class="status-ok">N(A) ⟂ C(Aᵀ)</span>：把 x 沿零空间方向怎么挪，Ax 都纹丝不动——` +
        `A 只“看得见” x 的行空间分量。`;
    } else if (r === 2) {
      status.textContent =
        "满秩：零空间只剩原点，行空间与列空间都是整个 ℝ²。选一个秩 1 的预设才能看到四个子空间劈开平面。";
    } else {
      status.textContent = "零矩阵：所有 x 都被打到 0，零空间是整个平面，列空间只剩原点。";
    }
  }

  const onResize = (): void => {
    left.resize();
    right.resize();
    render();
  };
  window.addEventListener("resize", onResize);

  render();

  return () => {
    window.removeEventListener("resize", onResize);
  };
}
