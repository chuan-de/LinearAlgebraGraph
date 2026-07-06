import { solve2, solve3, type M3, type Mat, type Vec } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt, matrix3Input, sliderRow } from "../ui";

const PRESETS: { name: string; m: M3 }[] = [
  {
    name: "天气模型（晴/雨/雪）",
    m: [
      [0.7, 0.3, 0.2],
      [0.2, 0.5, 0.3],
      [0.1, 0.2, 0.5],
    ],
  },
  {
    name: "一步到位（各列相同）",
    m: [
      [0.5, 0.5, 0.5],
      [0.3, 0.3, 0.3],
      [0.2, 0.2, 0.2],
    ],
  },
  {
    name: "慢收敛（λ₂ 接近 1）",
    m: [
      [0.98, 0.01, 0.01],
      [0.01, 0.98, 0.01],
      [0.01, 0.01, 0.98],
    ],
  },
  {
    name: "循环倾向（螺旋着收敛）",
    m: [
      [0.1, 0.8, 0.1],
      [0.1, 0.1, 0.8],
      [0.8, 0.1, 0.1],
    ],
  },
];

/** 概率单纯形三角形的三个顶点（画布坐标） */
const TRI: Vec[] = [
  { x: -4.3, y: -2.8 },
  { x: 4.3, y: -2.8 },
  { x: 0, y: 4.6 },
];

/** 概率向量 → 画布坐标 */
function baryToPlane(u: number[]): Vec {
  return {
    x: u[0] * TRI[0].x + u[1] * TRI[1].x + u[2] * TRI[2].x,
    y: u[0] * TRI[0].y + u[1] * TRI[1].y + u[2] * TRI[2].y,
  };
}

/** 画布坐标 → 概率向量（夹回单纯形内） */
function planeToBary(v: Vec): number[] {
  const m: Mat = [
    TRI[0].x - TRI[2].x,
    TRI[1].x - TRI[2].x,
    TRI[0].y - TRI[2].y,
    TRI[1].y - TRI[2].y,
  ];
  const s = solve2(m, { x: v.x - TRI[2].x, y: v.y - TRI[2].y });
  let u = s ? [s.x, s.y, 1 - s.x - s.y] : [1 / 3, 1 / 3, 1 / 3];
  u = u.map((x) => Math.max(0, x));
  const sum = u[0] + u[1] + u[2];
  return sum < 1e-9 ? [1 / 3, 1 / 3, 1 / 3] : u.map((x) => x / sum);
}

function applyM3(A: M3, u: number[]): number[] {
  return [0, 1, 2].map((i) => A[i][0] * u[0] + A[i][1] * u[1] + A[i][2] * u[2]);
}

/** 稳态：解 (A−I)π = 0 且 Σπ = 1 */
function steadyState(A: M3): number[] | null {
  const sys: M3 = [
    [A[0][0] - 1, A[0][1], A[0][2]],
    [A[1][0], A[1][1] - 1, A[1][2]],
    [1, 1, 1],
  ];
  const s = solve3(sys, [0, 0, 1]);
  return s ? [s[0], s[1], s[2]] : null;
}

export function mountMarkov(root: HTMLElement): () => void {
  let A: M3 = PRESETS[0].m.map((r) => [...r]);
  let u0: number[] = [0.9, 0.05, 0.05];
  let K = 12;

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "马尔可夫矩阵与稳态"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 24。马尔可夫矩阵<b>每列和为 1 且元素非负</b>——它把概率向量搬来搬去而总量不变。" +
        "三角形是所有概率分布组成的<b>单纯形</b>，拖动起点 u₀，" +
        "看 u₀ → Au₀ → A²u₀ → … 被吸向 λ=1 的特征向量：<b>稳态 π</b>（金色星）。",
    ),
  );

  const mi = matrix3Input(A, (m) => {
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
    A = p.m.map((r) => [...r]);
    mi.set(A);
    render();
  };
  panel.appendChild(sel);

  const normBtn = el("button", "btn", "把各列归一化为 1");
  normBtn.onclick = () => {
    for (let j = 0; j < 3; j++) {
      const s = A[0][j] + A[1][j] + A[2][j];
      if (Math.abs(s) > 1e-9) for (let i = 0; i < 3; i++) A[i][j] /= s;
    }
    mi.set(A);
    render();
  };
  panel.appendChild(normBtn);

  const kSlider = sliderRow("K", 1, 30, 1, K, (v) => {
    K = v;
    render();
  });
  panel.appendChild(kSlider.root);

  const readout = el("div", "readout");
  panel.appendChild(readout);

  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 黑板 ----
  const canvasCol = el("div", "canvas-col");
  canvasCol.appendChild(el("h3", "", "概率单纯形：三个顶点是“全在状态 i”的纯分布，内部每一点是一种概率分布"));
  const canvas = el("canvas", "plane");
  canvasCol.appendChild(canvas);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(canvasCol);
  root.appendChild(moduleEl);

  const plane = new Plane(canvas);
  plane.onRedraw = () => render();
  plane.attachDrag((v) => {
    u0 = planeToBary(v);
    render();
  });

  function render(): void {
    const colSums = [0, 1, 2].map((j) => A[0][j] + A[1][j] + A[2][j]);
    const stochastic = colSums.every((s) => Math.abs(s - 1) < 1e-6) && A.flat().every((v) => v >= -1e-9);

    // 轨迹
    const traj: number[][] = [u0];
    for (let k = 0; k < K; k++) traj.push(applyM3(A, traj[k]));
    const pi = steadyState(A);

    plane.clear();
    const ctx = plane.ctx;

    // 单纯形三角形
    const corners = TRI.map((t) => plane.toScreen(t));
    ctx.strokeStyle = COLORS.dim;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    ctx.lineTo(corners[1][0], corners[1][1]);
    ctx.lineTo(corners[2][0], corners[2][1]);
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = COLORS.tick;
    ctx.font = "13px 'Segoe UI', 'Microsoft YaHei', sans-serif";
    ctx.textAlign = "center";
    const labels = ["状态 1", "状态 2", "状态 3"];
    const offs: Vec[] = [
      { x: -0.35, y: -0.35 },
      { x: 0.35, y: -0.35 },
      { x: 0, y: 0.4 },
    ];
    labels.forEach((lab, i) => {
      const [x, y] = plane.toScreen({ x: TRI[i].x + offs[i].x * 1.4, y: TRI[i].y + offs[i].y * 1.4 });
      ctx.fillText(lab, x, y);
    });

    // 轨迹：连线 + 渐显的点
    for (let k = 0; k + 1 < traj.length; k++) {
      const a = baryToPlane(traj[k]);
      const b = baryToPlane(traj[k + 1]);
      const alpha = 0.2 + (0.5 * k) / traj.length;
      plane.arrow(a, b, `rgba(236,231,214,${alpha.toFixed(3)})`, 1.5);
    }
    traj.forEach((u, k) => {
      const p = baryToPlane(u);
      const alpha = 0.35 + (0.65 * k) / (traj.length - 1 || 1);
      plane.point(p, `rgba(236,231,214,${alpha.toFixed(3)})`, k === 0 ? 6 : 3.5, k === 0 ? "u₀（拖我）" : "");
    });

    // 稳态
    if (pi && stochastic) {
      const p = baryToPlane(pi);
      plane.point(p, COLORS.gold, 6, `π (${fmt(pi[0])}, ${fmt(pi[1])}, ${fmt(pi[2])})`);
      plane.ring(p, COLORS.gold, 11, 1.5);
    }

    // 收敛速度估计：最后两步到 π 的距离比
    let rateText = "";
    if (pi) {
      const dist = (u: number[]): number =>
        Math.abs(u[0] - pi[0]) + Math.abs(u[1] - pi[1]) + Math.abs(u[2] - pi[2]);
      const d1 = dist(traj[traj.length - 2] ?? traj[0]);
      const d2 = dist(traj[traj.length - 1]);
      rateText = `‖u_K − π‖₁ = ${fmt(d2)}` + (d1 > 1e-9 ? `，每步 ×${fmt(d2 / d1)}（≈|λ₂|）` : "");
    }

    const uK = traj[traj.length - 1];
    readout.textContent =
      `列和 = (${colSums.map((s) => fmt(s)).join(", ")})\n` +
      (pi ? `稳态 π = (${fmt(pi[0])}, ${fmt(pi[1])}, ${fmt(pi[2])})（Aπ = π）\n` : "稳态不唯一（λ=1 是重根）\n") +
      `u₀  = (${fmt(u0[0])}, ${fmt(u0[1])}, ${fmt(u0[2])})\n` +
      `u_${K} = (${fmt(uK[0])}, ${fmt(uK[1])}, ${fmt(uK[2])})\n` +
      rateText;

    if (!stochastic) {
      status.innerHTML =
        `<b>这不是马尔可夫矩阵</b>：列和 ≠ 1 或有负元素，轨迹会跑出单纯形。` +
        `点「把各列归一化为 1」修正。`;
    } else if (pi) {
      status.innerHTML =
        `<span class="status-ok">列和为 1 ⇒ λ₁ = 1 必然存在</span>，其余 |λ| ≤ 1：` +
        `不管从哪儿出发，A 一步步把分布拖向 π。第二特征值 |λ₂| 越小收敛越快。`;
    } else {
      status.textContent = "λ = 1 是重根：存在多个稳态方向，极限依赖初始分布。";
    }
  }

  const onResize = (): void => {
    plane.resize();
    render();
  };
  window.addEventListener("resize", onResize);

  render();

  return () => {
    window.removeEventListener("resize", onResize);
  };
}
