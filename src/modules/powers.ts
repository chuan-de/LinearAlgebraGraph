import { apply, eigen, solve2, type Mat, type Vec } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt, matrixInput, sliderRow } from "../ui";

const PRESETS: { name: string; m: Mat }[] = [
  { name: "斐波那契 [1 1; 1 0]（λ₁ = 黄金比 1.618…）", m: [1, 1, 1, 0] },
  { name: "马尔可夫 [0.9 0.2; 0.1 0.8]（λ₁=1：稳态方向）", m: [0.9, 0.2, 0.1, 0.8] },
  { name: "螺旋收缩 [0.76 -0.57; 0.57 0.76]（复特征值，|λ|<1）", m: [0.76, -0.57, 0.57, 0.76] },
  { name: "不稳定 [1.1 0.3; 0.1 1.2]（|λ|>1，爆炸增长）", m: [1.1, 0.3, 0.1, 1.2] },
];

const SUB = "₀₁₂₃₄₅₆₇₈₉";

function sub(n: number): string {
  return String(n)
    .split("")
    .map((c) => SUB[Number(c)])
    .join("");
}

export function mountPowers(root: HTMLElement): () => void {
  let A: Mat = [...PRESETS[1].m] as Mat;
  let u0: Vec = { x: 2.4, y: 0.8 };
  let K = 8;

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "差分方程与矩阵的幂 Aᵏ"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 22。差分方程 u_{k+1} = A·u_k 的解是 u_k = Aᵏu₀。" +
        "在画布上拖动设置初始向量 u₀，观察轨迹如何被特征方向（紫色虚线）“指挥”：" +
        "把 u₀ = c₁v₁ + c₂v₂ 按特征向量分解后，每一步只是给各分量乘上 λ——" +
        "|λ|>1 的方向增长，|λ|<1 的方向衰减，长期行为由绝对值最大的 λ 主导。",
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
  sel.value = "1";
  sel.onchange = () => {
    const p = PRESETS[Number(sel.value)];
    if (!p) return;
    A = [...p.m] as Mat;
    mi.set(A);
    render();
  };
  panel.appendChild(sel);

  const kSlider = sliderRow("步数 K", 1, 24, 1, K, (v) => {
    K = Math.round(v);
    render();
  });
  panel.appendChild(kSlider.root);

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
    u0 = v;
    render();
  });

  /** 轨迹 u_k = Aᵏu₀；模长超过 1e4 时截断 */
  function computeTraj(): { pts: Vec[]; truncated: boolean } {
    const pts: Vec[] = [u0];
    let cur = u0;
    let truncated = false;
    for (let k = 1; k <= K; k++) {
      cur = apply(A, cur);
      if (Math.hypot(cur.x, cur.y) > 1e4) {
        truncated = true;
        break;
      }
      pts.push(cur);
    }
    return { pts, truncated };
  }

  function render(): void {
    const o: Vec = { x: 0, y: 0 };
    const eg = eigen(A);
    const { pts, truncated } = computeTraj();
    const ctx = plane.ctx;

    plane.clear();
    plane.grid();
    plane.axes();

    // 特征方向（实特征值时）
    if (eg.real) {
      eg.vectors.forEach((v, i) => {
        plane.infLine(o, v, COLORS.purple, 1.2, [7, 6]);
        const tip = { x: v.x * 3.2, y: v.y * 3.2 };
        const [sx, sy] = plane.toScreen(tip);
        ctx.fillStyle = COLORS.purple;
        ctx.font = "13px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`λ${sub(i + 1)} = ${fmt(eg.values[i])}`, sx, sy - 8);
      });
    }

    // 连接轨迹的细线
    if (pts.length > 1) {
      ctx.save();
      ctx.strokeStyle = COLORS.dim;
      ctx.lineWidth = 1;
      ctx.beginPath();
      const [x0, y0] = plane.toScreen(pts[0]);
      ctx.moveTo(x0, y0);
      for (let i = 1; i < pts.length; i++) {
        const [x, y] = plane.toScreen(pts[i]);
        ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    // 各点：越早越淡；末点金色并标注
    const last = pts.length - 1;
    pts.forEach((p, k) => {
      ctx.save();
      ctx.globalAlpha = last === 0 ? 1 : 0.25 + (0.75 * k) / last;
      if (k === last && last > 0) {
        plane.point(p, COLORS.gold, 6, `u${sub(k)}`);
      } else {
        plane.point(p, COLORS.chalk, 4.5, k === 0 ? "u₀" : "");
      }
      ctx.restore();
    });
    plane.ring(u0, COLORS.dim, 10, 1.5);

    // ---- 读出区 ----
    let text: string;
    if (eg.real) {
      const [l1, l2] = eg.values;
      const [v1, v2] = eg.vectors;
      text =
        `λ₁ = ${fmt(l1)}   v₁ = (${fmt(v1.x)}, ${fmt(v1.y)})\n` +
        `λ₂ = ${fmt(l2)}   v₂ = (${fmt(v2.x)}, ${fmt(v2.y)})\n\n`;
      const c = solve2([v1.x, v2.x, v1.y, v2.y], u0);
      if (c) {
        text +=
          `u₀ = c₁v₁ + c₂v₂，c₁ = ${fmt(c.x)}，c₂ = ${fmt(c.y)}\n` +
          `u_k = c₁·λ₁ᵏ·v₁ + c₂·λ₂ᵏ·v₂\n\n`;
      } else {
        text += "v₁ ∥ v₂（重根且特征向量不足），u₀ 无法按特征向量分解\n\n";
      }
      text += `${growthDesc(l1, 1)}\n${growthDesc(l2, 2)}\n`;
      const domIdx = Math.abs(l1) >= Math.abs(l2) ? 0 : 1;
      text += `长期行为由主特征方向 v${sub(domIdx + 1)} 主导：u_k ≈ c${sub(domIdx + 1)}·λ${sub(domIdx + 1)}ᵏ·v${sub(domIdx + 1)}`;
    } else {
      const [re, im] = eg.values;
      const mod = Math.hypot(re, im);
      text =
        `特征值为复数：λ = ${fmt(re)} ± ${fmt(im)}i，|λ| = ${fmt(mod)}\n\n` +
        `每乘一次 A，向量旋转一个角度并把模长乘 |λ|，轨迹是螺旋：\n`;
      if (mod < 1 - 1e-6) text += "|λ| < 1 → 螺旋向内收缩，u_k → 0";
      else if (mod > 1 + 1e-6) text += "|λ| > 1 → 螺旋向外发散";
      else text += "|λ| = 1 → 绕原点转圈，模长不变";
    }
    readout.textContent =
      `u₀ = (${fmt(u0.x)}, ${fmt(u0.y)})   K = ${K}\n` +
      `u_K = (${fmt(pts[last].x)}, ${fmt(pts[last].y)})（k = ${last}）\n\n` +
      text;

    // ---- 状态行 ----
    if (truncated) {
      status.textContent = `增长太快：k = ${last + 1} 时 |u_k| 已超过 10⁴，轨迹被截断。`;
    } else if (eg.real && Math.abs(Math.abs(eg.values[0]) - 1) < 1e-6) {
      status.innerHTML =
        '<span class="status-ok">λ₁ = 1：沿 v₁ 的分量永不衰减，u_k 收敛到稳态方向。</span>';
    } else {
      status.textContent = "在画布上拖动设置 u₀，滚轮缩放；调节 K 观察 Aᵏu₀ 的走向。";
    }
  }

  function growthDesc(l: number, i: number): string {
    const a = Math.abs(l);
    const li = `λ${sub(i)}`;
    if (Math.abs(a - 1) < 1e-6) return `|${li}| = 1：该方向分量保持不变（稳态方向）`;
    return a > 1
      ? `|${li}| = ${fmt(a)} > 1：该方向分量按 ${li}ᵏ 增长`
      : `|${li}| = ${fmt(a)} < 1：该方向分量衰减 → 0`;
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
