import { I, apply, det, eigen, lerpMat, mul, type Mat, type Vec } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt, matrixInput } from "../ui";

const PRESETS: { name: string; m: Mat }[] = [
  { name: "一般矩阵", m: [3, 1, 1, 2] },
  { name: "剪切 (shear)", m: [1, 1, 0, 1] },
  { name: "含反射 (det<0)", m: [1, 2, 2, 1] },
  { name: "奇异 (σ₂=0)", m: [2, 1, 4, 2] },
];

function ease(t: number): number {
  return t * t * (3 - 2 * t);
}

interface Svd {
  U: Mat;
  S: Mat;
  Vt: Mat;
  s1: number;
  s2: number;
  v1: Vec;
  v2: Vec;
  u1: Vec;
  u2: Vec;
}

/** 2×2 SVD：由 AᵀA 的特征分解取 vᵢ 与 σᵢ = √λᵢ，再 uᵢ = Avᵢ/σᵢ */
function svd2(A: Mat): Svd {
  const ab = A[0] * A[1] + A[2] * A[3];
  const ata: Mat = [A[0] * A[0] + A[2] * A[2], ab, ab, A[1] * A[1] + A[3] * A[3]];
  const eg = eigen(ata); // 对称半正定，恒为实特征值且 λ₁ ≥ λ₂
  const s1 = Math.sqrt(Math.max(0, eg.values[0]));
  const s2 = Math.sqrt(Math.max(0, eg.values[1]));
  const v1 = eg.vectors[0];
  // v₂ 取 v₁ 旋转 90°，保证 V 严格正交（det V = +1），σ 保持 ≥ 0
  const v2: Vec = { x: -v1.y, y: v1.x };
  const av1 = apply(A, v1);
  const av2 = apply(A, v2);
  const u1: Vec = s1 > 1e-9 ? { x: av1.x / s1, y: av1.y / s1 } : { x: 1, y: 0 };
  const u2: Vec =
    s2 > 1e-9 ? { x: av2.x / s2, y: av2.y / s2 } : { x: -u1.y, y: u1.x };
  return {
    U: [u1.x, u2.x, u1.y, u2.y],
    S: [s1, 0, 0, s2],
    Vt: [v1.x, v1.y, v2.x, v2.y],
    s1,
    s2,
    v1,
    v2,
    u1,
    u2,
  };
}

export function mountSvd(root: HTMLElement): () => void {
  let A: Mat = [3, 1, 1, 2];
  let dec = svd2(A);
  let s = 0; // 0..3：三段动画进度
  let playing = true;

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "奇异值分解 A = UΣVᵀ"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 29–31。任何矩阵都能拆成「旋转 · 拉伸 · 旋转」：" +
        "Vᵀ 把主方向 v₁, v₂ 转到坐标轴，Σ 沿轴拉伸 σ₁, σ₂，U 再转到输出方向 u₁, u₂。" +
        "白色曲线是单位圆的像——最终变成半轴 σ₁, σ₂ 的椭圆。滚轮缩放视图。",
    ),
  );

  const mi = matrixInput(A, (m) => {
    A = m;
    dec = svd2(A);
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
    dec = svd2(A);
    s = 0;
    playing = true;
  };
  panel.appendChild(sel);

  const playBtn = el("button", "btn primary", "▶ 播放");
  const sSlider = el("input");
  sSlider.type = "range";
  sSlider.min = "0";
  sSlider.max = "3";
  sSlider.step = "0.001";
  sSlider.value = "0";
  const playRow = el("div", "row");
  playRow.appendChild(playBtn);
  playRow.appendChild(sSlider);
  panel.appendChild(playRow);

  playBtn.onclick = () => {
    if (playing) {
      playing = false;
    } else {
      if (s >= 3) s = 0;
      playing = true;
    }
    syncControls();
  };
  sSlider.oninput = () => {
    s = sSlider.valueAsNumber;
    playing = false;
    syncControls();
    render();
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

  function syncControls(): void {
    sSlider.value = String(s);
    playBtn.textContent = playing ? "⏸ 暂停" : s >= 3 ? "↺ 重播" : "▶ 播放";
  }

  /** 当前动画矩阵：I→Vᵀ，再 (I→Σ)·Vᵀ，再 (I→U)·Σ·Vᵀ */
  function currentM(): Mat {
    const { U, S, Vt } = dec;
    if (s <= 1) return lerpMat(I, Vt, ease(s));
    if (s <= 2) return mul(lerpMat(I, S, ease(s - 1)), Vt);
    return mul(mul(lerpMat(I, U, ease(s - 2)), S), Vt);
  }

  function circlePath(M: Mat): void {
    const ctx = plane.ctx;
    ctx.beginPath();
    for (let k = 0; k <= 128; k++) {
      const th = (k / 128) * Math.PI * 2;
      const p = apply(M, { x: Math.cos(th), y: Math.sin(th) });
      const [x, y] = plane.toScreen(p);
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
  }

  function render(): void {
    const M = currentM();
    plane.clear();
    plane.grid();
    plane.axes();
    plane.grid(M, COLORS.grid, 1);

    const ctx = plane.ctx;
    // 原始单位圆（淡虚线参考）
    ctx.save();
    ctx.strokeStyle = COLORS.dim;
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 6]);
    circlePath(I);
    ctx.stroke();
    ctx.restore();
    // 单位圆的像：微微发光的粉笔白闭合曲线
    ctx.save();
    ctx.strokeStyle = COLORS.chalk;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = COLORS.chalk;
    ctx.shadowBlur = 7;
    circlePath(M);
    ctx.stroke();
    ctx.restore();

    const o = { x: 0, y: 0 };
    const mv1 = apply(M, dec.v1);
    const mv2 = apply(M, dec.v2);
    const done = s >= 2.99;
    plane.arrow(o, mv1, COLORS.green, 3, done ? "" : "v₁");
    plane.arrow(o, mv2, COLORS.red, 3, done ? "" : "v₂");
    if (done) {
      // 终点：M·vᵢ = σᵢuᵢ，金色标注
      plane.arrow(mv1, mv1, COLORS.gold, 2, "σ₁u₁");
      plane.arrow(mv2, mv2, COLORS.gold, 2, "σ₂u₂");
    }

    // ---- readout ----
    const { U, S, Vt, s1, s2 } = dec;
    const R = mul(mul(U, S), Vt);
    let err = 0;
    for (let i = 0; i < 4; i++) err = Math.max(err, Math.abs(R[i] - A[i]));
    const dA = det(A);
    const m2 = (m: Mat, name: string): string =>
      `${name} = [ ${fmt(m[0])}\t${fmt(m[1])} ]\n${" ".repeat(name.length)}   [ ${fmt(m[2])}\t${fmt(m[3])} ]`;
    readout.textContent =
      `σ₁ = ${fmt(s1)},  σ₂ = ${fmt(s2)}\n\n` +
      `${m2(U, "U")}\n${m2(S, "Σ")}\n${m2(Vt, "Vᵀ")}\n\n` +
      `max |UΣVᵀ − A| = ${err < 1e-4 ? err.toExponential(1) : fmt(err)}\n` +
      `det A = ${fmt(dA)}` +
      (dA < -1e-9
        ? "\n（det<0：U 或 V 含反射，线性插值中途会经过退化形状，属正常现象）"
        : "");

    // ---- status ----
    if (s < 1) {
      status.textContent =
        `第 1 段  Vᵀ：旋转对齐到主方向——把 v₁, v₂ 转到 x、y 轴上。`;
    } else if (s < 2) {
      status.textContent = `第 2 段  Σ：沿轴拉伸 σ₁ = ${fmt(s1)}, σ₂ = ${fmt(s2)}，单位圆变成椭圆。`;
    } else if (s < 3) {
      status.textContent = "第 3 段  U：旋转到输出方向 u₁, u₂。";
    } else {
      status.innerHTML =
        `<span class="status-ok">✓ A = UΣVᵀ 完成</span>：v₁ ↦ σ₁u₁, v₂ ↦ σ₂u₂，` +
        "单位圆的像是半轴 σ₁, σ₂ 的椭圆。";
    }
  }

  // ---- 动画循环 ----
  let raf = 0;
  let last = performance.now();
  function loop(now: number): void {
    raf = requestAnimationFrame(loop);
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (playing) {
      s = Math.min(3, s + dt / 1.5);
      if (s >= 3) playing = false;
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
