import { I, apply, det, eigen, lerpMat, type Mat } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt, matrixInput } from "../ui";

const PRESETS: { name: string; m: Mat }[] = [
  { name: "旋转 45°", m: [Math.SQRT1_2, -Math.SQRT1_2, Math.SQRT1_2, Math.SQRT1_2] },
  { name: "旋转 90°", m: [0, -1, 1, 0] },
  { name: "剪切 (shear)", m: [1, 1, 0, 1] },
  { name: "缩放", m: [2, 0, 0, 0.5] },
  { name: "关于 y=x 反射", m: [0, 1, 1, 0] },
  { name: "投影到 x 轴", m: [1, 0, 0, 0] },
  { name: "奇异矩阵 (det=0)", m: [2, 1, 4, 2] },
  { name: "对称矩阵", m: [2, 1, 1, 2] },
];

function ease(t: number): number {
  return t * t * (3 - 2 * t);
}

export function mountTransform(root: HTMLElement): () => void {
  let A: Mat = [1, 1, 0, 1];
  let t = 0;
  let playing = true;
  let showEigen = true;
  let showArea = true;

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "线性变换播放器"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 1–3。矩阵是对整个平面的线性变换：网格从单位阵连续形变到 A，" +
        "基向量 <span style='color:var(--green-ink)'>î</span>、<span style='color:var(--red-ink)'>ĵ</span> " +
        "分别落到 A 的第 1、2 列上。滚轮缩放视图。",
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
  sel.appendChild(el("option", "", "选择预设变换…"));
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
    t = 0;
    playing = true;
  };
  panel.appendChild(sel);

  const playBtn = el("button", "btn primary", "▶ 播放");
  const tSlider = el("input");
  tSlider.type = "range";
  tSlider.min = "0";
  tSlider.max = "1";
  tSlider.step = "0.001";
  tSlider.value = "0";
  const playRow = el("div", "row");
  playRow.appendChild(playBtn);
  playRow.appendChild(tSlider);
  panel.appendChild(playRow);

  playBtn.onclick = () => {
    if (playing) {
      playing = false;
    } else {
      if (t >= 1) t = 0;
      playing = true;
    }
    syncControls();
  };
  tSlider.oninput = () => {
    t = tSlider.valueAsNumber;
    playing = false;
    syncControls();
    render();
  };

  const areaCheck = el("label", "check");
  const areaInput = el("input");
  areaInput.type = "checkbox";
  areaInput.checked = showArea;
  areaInput.onchange = () => {
    showArea = areaInput.checked;
    render();
  };
  areaCheck.appendChild(areaInput);
  areaCheck.appendChild(el("span", "", "显示单位正方形的像（面积 = |det|）"));
  panel.appendChild(areaCheck);

  const eigenCheck = el("label", "check");
  const eigenInput = el("input");
  eigenInput.type = "checkbox";
  eigenInput.checked = showEigen;
  eigenInput.onchange = () => {
    showEigen = eigenInput.checked;
    render();
  };
  eigenCheck.appendChild(eigenInput);
  eigenCheck.appendChild(el("span", "", "显示 A 的特征向量方向（紫色虚线）"));
  panel.appendChild(eigenCheck);

  const readout = el("div", "readout");
  panel.appendChild(readout);

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

  function syncControls(): void {
    tSlider.value = String(t);
    playBtn.textContent = playing ? "⏸ 暂停" : t >= 1 ? "↺ 重播" : "▶ 播放";
  }

  function render(): void {
    const M = lerpMat(I, A, ease(t));
    plane.clear();
    plane.grid();
    plane.axes();
    plane.grid(M, COLORS.grid, 1.3);

    const o = { x: 0, y: 0 };
    const e1 = apply(M, { x: 1, y: 0 });
    const e2 = apply(M, { x: 0, y: 1 });
    if (showArea) {
      const corner = apply(M, { x: 1, y: 1 });
      plane.fillPoly([o, e1, corner, e2], det(M) >= 0 ? COLORS.gold : COLORS.red, 0.28);
    }

    let eigenText: string;
    const eg = eigen(A);
    if (eg.real) {
      if (showEigen) {
        eg.vectors.forEach((v, i) => {
          plane.infLine(o, v, COLORS.purple, 1.2, [7, 6]);
          const target = apply(M, v);
          plane.arrow(o, target, COLORS.purple, 2, `v${i === 0 ? "₁" : "₂"}`);
        });
      }
      eigenText =
        `特征值  λ₁ = ${fmt(eg.values[0])}, λ₂ = ${fmt(eg.values[1])}\n` +
        `特征向量 v₁=(${fmt(eg.vectors[0].x)}, ${fmt(eg.vectors[0].y)})` +
        ` v₂=(${fmt(eg.vectors[1].x)}, ${fmt(eg.vectors[1].y)})`;
    } else {
      eigenText = `特征值为复数 ${fmt(eg.values[0])} ± ${fmt(eg.values[1])}i\n（含旋转成分，平面上没有方向不变的实向量）`;
    }

    plane.arrow(o, e1, COLORS.green, 3, "î");
    plane.arrow(o, e2, COLORS.red, 3, "ĵ");

    readout.textContent =
      `M(t) = [ ${fmt(M[0])}\t${fmt(M[1])} ]\n` +
      `       [ ${fmt(M[2])}\t${fmt(M[3])} ]\n\n` +
      `det M(t) = ${fmt(det(M))}\n` +
      `det A    = ${fmt(det(A))}   ← 面积缩放因子\n\n` +
      eigenText;
  }

  // ---- 动画循环 ----
  let raf = 0;
  let last = performance.now();
  function loop(now: number): void {
    raf = requestAnimationFrame(loop);
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (playing) {
      t = Math.min(1, t + dt / 1.8);
      if (t >= 1) playing = false;
      syncControls();
      render();
    }
  }
  raf = requestAnimationFrame(loop);

  const onResize = (): void => {
    plane.resize();
    render();
  };
  window.addEventListener("resize", onResize);

  syncControls();
  render();

  return () => {
    cancelAnimationFrame(raf);
    window.removeEventListener("resize", onResize);
  };
}
