import { apply, det, eigen, normalize, type Mat, type Vec } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt, matrixInput } from "../ui";

const PRESETS: { name: string; m: Mat }[] = [
  { name: "对称矩阵 [2 1; 1 2]", m: [2, 1, 1, 2] },
  { name: "剪切 [1 1; 0 1]（重根，只有一个方向）", m: [1, 1, 0, 1] },
  { name: "旋转 60°（无实特征向量）", m: [0.5, -0.866, 0.866, 0.5] },
  { name: "马尔可夫矩阵 [0.8 0.3; 0.2 0.7]", m: [0.8, 0.3, 0.2, 0.7] },
  { name: "关于 y=x 反射（λ = ±1）", m: [0, 1, 1, 0] },
  { name: "对角矩阵 [3 0; 0 -1]", m: [3, 0, 0, -1] },
];

const TRAIL_MAX = 14;

export function mountEigen(root: HTMLElement): () => void {
  let A: Mat = [2, 1, 1, 2];
  let x: Vec = { x: 2.2, y: 0.6 };
  let trail: Vec[] = [];
  let autoTimer = 0;

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "特征向量探索器"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 21–22。在画布上拖动白色向量 x，观察蓝色的 Ax。" +
        "当 Ax 与 x 平行（方向相同或相反）时，x 就是特征向量，Ax = λx——两支箭头会变金。" +
        "「幂迭代」反复计算 x ← Ax 并归一化，看它收敛到最大特征值的方向。",
    ),
  );

  const mi = matrixInput(A, (m) => {
    A = m;
    trail = [];
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
    trail = [];
    render();
  };
  panel.appendChild(sel);

  const stepBtn = el("button", "btn", "幂迭代一步 x ← Ax");
  const autoBtn = el("button", "btn primary", "▶ 自动迭代");
  const btnRow = el("div", "row");
  btnRow.appendChild(stepBtn);
  btnRow.appendChild(autoBtn);
  panel.appendChild(btnRow);

  stepBtn.onclick = () => powerStep();
  autoBtn.onclick = () => {
    if (autoTimer) stopAuto();
    else {
      autoTimer = window.setInterval(powerStep, 650);
      autoBtn.textContent = "⏸ 停止迭代";
    }
  };

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
    if (Math.hypot(v.x, v.y) < 0.15) return;
    x = v;
    trail = [];
    render();
  });

  function stopAuto(): void {
    if (autoTimer) {
      clearInterval(autoTimer);
      autoTimer = 0;
      autoBtn.textContent = "▶ 自动迭代";
    }
  }

  function powerStep(): void {
    const ax = apply(A, x);
    if (Math.hypot(ax.x, ax.y) < 1e-9) {
      stopAuto();
      status.textContent = "Ax = 0：x 落在零空间里，幂迭代无法继续。换个方向试试。";
      return;
    }
    trail.push(x);
    if (trail.length > TRAIL_MAX) trail.shift();
    const u = normalize(ax);
    x = { x: u.x * 2.5, y: u.y * 2.5 };
    render();
  }

  function render(): void {
    const o = { x: 0, y: 0 };
    const ax = apply(A, x);
    const eg = eigen(A);

    // 平行判定：|x × Ax| / (|x||Ax|) 即夹角的正弦
    const nx = Math.hypot(x.x, x.y);
    const nax = Math.hypot(ax.x, ax.y);
    const sinTheta = nax > 1e-9 ? Math.abs(x.x * ax.y - x.y * ax.x) / (nx * nax) : 1;
    const aligned = sinTheta < 0.015 && nax > 1e-9;
    const lambdaEst = (x.x * ax.x + x.y * ax.y) / (nx * nx); // Rayleigh 商

    plane.clear();
    plane.grid();
    plane.axes();

    if (eg.real) {
      eg.vectors.forEach((v, i) => {
        plane.infLine(o, v, COLORS.purple, 1.2, [7, 6]);
        const tip = { x: v.x * 3.2, y: v.y * 3.2 };
        const [sx2, sy2] = plane.toScreen(tip);
        plane.ctx.fillStyle = COLORS.purple;
        plane.ctx.font = "13px system-ui, sans-serif";
        plane.ctx.textAlign = "center";
        plane.ctx.fillText(`λ${i === 0 ? "₁" : "₂"} = ${fmt(eg.values[i])}`, sx2, sy2 - 8);
      });
    }

    // 幂迭代轨迹（越旧越淡）
    trail.forEach((v, i) => {
      const alpha = (0.45 * (i + 1)) / trail.length;
      plane.arrow(o, v, `rgba(236,231,214,${alpha.toFixed(3)})`, 1.5);
    });

    const mainColor = aligned ? COLORS.gold : COLORS.chalk;
    const axColor = aligned ? COLORS.gold : COLORS.blue;
    plane.arrow(o, ax, axColor, 3, "Ax");
    plane.arrow(o, x, mainColor, 3, "x");

    let eigenText: string;
    if (eg.real) {
      eigenText =
        `λ₁ = ${fmt(eg.values[0])}   v₁ = (${fmt(eg.vectors[0].x)}, ${fmt(eg.vectors[0].y)})\n` +
        `λ₂ = ${fmt(eg.values[1])}   v₂ = (${fmt(eg.vectors[1].x)}, ${fmt(eg.vectors[1].y)})`;
    } else {
      eigenText = `特征值为复数 ${fmt(eg.values[0])} ± ${fmt(eg.values[1])}i\n纯旋转成分：没有实特征向量，怎么拖都不会平行`;
    }
    const tr = A[0] + A[3];
    readout.textContent =
      `特征多项式 λ² - ${fmt(tr)}λ + ${fmt(det(A))} = 0\n` +
      `（迹 = ${fmt(tr)} = λ₁+λ₂，det = ${fmt(det(A))} = λ₁·λ₂）\n\n` +
      eigenText +
      `\n\nx = (${fmt(x.x)}, ${fmt(x.y)})   Ax = (${fmt(ax.x)}, ${fmt(ax.y)})\n` +
      `Rayleigh 商 x·Ax / x·x = ${fmt(lambdaEst)}`;

    if (aligned) {
      status.innerHTML = `<span class="status-ok">✓ Ax ∥ x：这是特征向量方向！λ ≈ ${fmt(lambdaEst)}</span>`;
    } else if (trail.length > 0) {
      status.textContent = `幂迭代第 ${trail.length} 步：x 正被“拉向”绝对值最大的特征值方向。`;
    } else {
      status.textContent = "拖动白色向量，寻找让 Ax 与 x 共线的方向（紫色虚线处）。";
    }
  }

  const onResize = (): void => {
    plane.resize();
    render();
  };
  window.addEventListener("resize", onResize);

  render();

  return () => {
    stopAuto();
    window.removeEventListener("resize", onResize);
  };
}
