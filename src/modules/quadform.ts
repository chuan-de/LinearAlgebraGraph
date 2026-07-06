import { det, eigen } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt } from "../ui";

const PRESETS: { name: string; a: number; b: number; d: number }[] = [
  { name: "正定（碗形椭圆）", a: 2, b: 1, d: 2 },
  { name: "临界半正定（槽形）", a: 1, b: 1, d: 1 },
  { name: "不定（鞍形双曲线）", a: 1, b: 2, d: 1 },
  { name: "负定（倒扣的碗）", a: -2, b: 0, d: -1 },
];

/** 等高线等级：正等级金色，负等级玫瑰色 */
const LEVELS = [1, 2, 4, 8, -1, -2, -4, -8];

const EPS = 1e-9;

export function mountQuadform(root: HTMLElement): () => void {
  let a = 2;
  let b = 1;
  let d = 2;

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "对称矩阵与正定性"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 25–28。对称矩阵 A = [a b; b d] 决定二次型 " +
        "f(x, y) = ax² + 2bxy + dy²。黑板上画的是 f 的等高线：" +
        "<span style='color:var(--gold)'>金色</span>为正等级 1,2,4,8，" +
        "<span style='color:var(--red)'>玫瑰色</span>为负等级。" +
        "紫色虚线是特征向量方向——正是椭圆/双曲线的主轴。滚轮缩放视图。",
    ),
  );

  // 2×2 对称矩阵输入：右上/左下共用同一个 b（左下为只读镜像）
  const matBox = el("div", "matrix");
  const mkInput = (value: number, onValue: (v: number) => void): HTMLInputElement => {
    const inp = el("input");
    inp.type = "number";
    inp.step = "0.1";
    inp.value = String(value);
    inp.oninput = () => {
      const v = Number.parseFloat(inp.value);
      onValue(Number.isFinite(v) ? v : 0);
      render();
    };
    return inp;
  };
  const inpA = mkInput(a, (v) => (a = v));
  const inpB = mkInput(b, (v) => {
    b = v;
    mirrorB.value = inpB.value;
  });
  const mirrorB = el("input");
  mirrorB.type = "number";
  mirrorB.value = String(b);
  mirrorB.readOnly = true;
  mirrorB.tabIndex = -1;
  mirrorB.style.opacity = "0.55";
  mirrorB.title = "对称位置：与右上角的 b 相同";
  const inpD = mkInput(d, (v) => (d = v));
  matBox.appendChild(inpA);
  matBox.appendChild(inpB);
  matBox.appendChild(mirrorB);
  matBox.appendChild(inpD);
  const matRow = el("div", "row");
  matRow.appendChild(el("span", "", "A ="));
  matRow.appendChild(matBox);
  panel.appendChild(matRow);

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
    a = p.a;
    b = p.b;
    d = p.d;
    inpA.value = fmt(a);
    inpB.value = fmt(b);
    mirrorB.value = fmt(b);
    inpD.value = fmt(d);
    render();
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

  /** marching squares：在可视范围的采样网格上提取 f = L 的等值线段并直接绘制 */
  function drawContours(): void {
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    const tl = plane.toWorld(0, 0);
    const br = plane.toWorld(cw, ch);
    const xmin = tl.x;
    const xmax = br.x;
    const ymin = br.y;
    const ymax = tl.y;
    const N = 140;
    const dx = (xmax - xmin) / N;
    const dy = (ymax - ymin) / N;
    const F = new Float64Array((N + 1) * (N + 1));
    for (let j = 0; j <= N; j++) {
      const y = ymin + j * dy;
      for (let i = 0; i <= N; i++) {
        const x = xmin + i * dx;
        F[j * (N + 1) + i] = a * x * x + 2 * b * x * y + d * y * y;
      }
    }
    const ctx = plane.ctx;
    for (const L of LEVELS) {
      ctx.save();
      ctx.strokeStyle = L > 0 ? COLORS.gold : COLORS.red;
      ctx.lineWidth = 1.25;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      for (let j = 0; j < N; j++) {
        const yA = ymin + j * dy;
        const yB = yA + dy;
        for (let i = 0; i < N; i++) {
          const f00 = F[j * (N + 1) + i] - L;
          const f10 = F[j * (N + 1) + i + 1] - L;
          const f01 = F[(j + 1) * (N + 1) + i] - L;
          const f11 = F[(j + 1) * (N + 1) + i + 1] - L;
          if (
            (f00 > 0 && f10 > 0 && f01 > 0 && f11 > 0) ||
            (f00 < 0 && f10 < 0 && f01 < 0 && f11 < 0)
          ) {
            continue;
          }
          const xA = xmin + i * dx;
          const xB = xA + dx;
          // 四条边上的交点，按 下/右/上/左 收集
          const pts: [number, number][] = [];
          if (f00 * f10 < 0) pts.push([xA + (f00 / (f00 - f10)) * dx, yA]);
          if (f10 * f11 < 0) pts.push([xB, yA + (f10 / (f10 - f11)) * dy]);
          if (f01 * f11 < 0) pts.push([xA + (f01 / (f01 - f11)) * dx, yB]);
          if (f00 * f01 < 0) pts.push([xA, yA + (f00 / (f00 - f01)) * dy]);
          for (let k = 0; k + 1 < pts.length; k += 2) {
            const [x1, y1] = plane.toScreen({ x: pts[k][0], y: pts[k][1] });
            const [x2, y2] = plane.toScreen({ x: pts[k + 1][0], y: pts[k + 1][1] });
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
          }
        }
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  function render(): void {
    plane.clear();
    plane.grid();
    plane.axes();

    const degenerate = Math.abs(a) < EPS && Math.abs(b) < EPS && Math.abs(d) < EPS;
    if (!degenerate) drawContours();

    // 特征向量方向：等高线的主轴（对称矩阵特征值恒为实数）
    const eg = eigen([a, b, b, d]);
    const [l1, l2] = eg.values;
    if (!degenerate) {
      const o = { x: 0, y: 0 };
      const R = (Math.min(canvas.clientWidth, canvas.clientHeight) / 2 / plane.scale) * 0.82;
      const ctx = plane.ctx;
      ctx.save();
      ctx.fillStyle = COLORS.purple;
      ctx.font = "italic 15px Cambria, Georgia, 'Times New Roman', serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      eg.vectors.forEach((v, i) => {
        plane.infLine(o, v, COLORS.purple, 1.2, [7, 6]);
        const [x, y] = plane.toScreen({ x: v.x * R, y: v.y * R });
        ctx.fillText(`λ${i === 0 ? "₁" : "₂"} = ${fmt(eg.values[i])}`, x + 10, y - 12);
      });
      ctx.restore();
    }

    // ---- readout：主元 / 特征值 / det & trace / 配方 ----
    const detA = det([a, b, b, d]);
    const trA = a + d;
    const pivotOk = Math.abs(a) > EPS;
    const p2 = d - (b * b) / a;
    const pivotLine = pivotOk
      ? `主元：p₁ = a = ${fmt(a)},  p₂ = d − b²/a = ${fmt(p2)}`
      : "主元：a ≈ 0，第一格不能作主元，需行交换后再消元";
    const squareLine = pivotOk
      ? `配方：f = ${fmt(a)}(x ${b / a >= 0 ? "+" : "−"} ${fmt(Math.abs(b / a))}y)² + ${fmt(p2)}y²`
      : "配方：a ≈ 0，无法直接对 x 配方（先换主元）";
    readout.textContent =
      `f(x,y) = ${fmt(a)}x² + ${fmt(2 * b)}xy + ${fmt(d)}y²\n` +
      `${pivotLine}\n` +
      `特征值：λ₁ = ${fmt(l1)},  λ₂ = ${fmt(l2)}\n` +
      `det A = ${fmt(detA)},  trace A = ${fmt(trA)}\n` +
      `${squareLine}`;

    // ---- status：正定性判定 + 至少两种判据的数值 ----
    const crits: string[] = [];
    if (pivotOk) crits.push(`主元 ${fmt(a)}, ${fmt(p2)}`);
    crits.push(`特征值 ${fmt(l1)}, ${fmt(l2)}`);
    crits.push(`顺序主子式 Δ₁ = ${fmt(a)}, Δ₂ = ${fmt(detA)}`);
    const critText = "判据：" + crits.join("；") + "。";
    if (degenerate) {
      status.textContent = "A = 0，f ≡ 0：完全退化，没有等高线可画。";
    } else if (l2 > EPS) {
      status.innerHTML =
        `<span class="status-ok">正定</span>：等高线是同心椭圆（碗形，原点是唯一最小值点），` +
        `主轴沿特征向量方向、半轴长 ∝ 1/√λ。${critText}（三组数全为正）`;
    } else if (l1 < -EPS) {
      status.innerHTML =
        `负定：−A 正定，f 是倒扣的碗，只出现玫瑰色负等级椭圆。${critText}` +
        `（特征值全负；主子式符号交替 Δ₁ &lt; 0, Δ₂ &gt; 0）`;
    } else if (l1 > EPS && l2 < -EPS) {
      status.innerHTML =
        `不定：等高线是双曲线（鞍形），原点是鞍点，沿 v₁ 上坡、沿 v₂ 下坡。${critText}` +
        `（特征值一正一负，det = λ₁λ₂ &lt; 0）`;
    } else if (l1 > EPS) {
      status.innerHTML =
        `<span class="status-ok">半正定（临界退化）</span>：det = λ₁λ₂ = 0，` +
        `等高线退化为平行直线族（槽形山谷），沿 λ = 0 的特征向量方向 f 恒为 0。${critText}`;
    } else {
      status.innerHTML =
        `半负定（临界退化）：λ₁ = 0 而 λ₂ ≤ 0，倒扣的槽形，` +
        `沿零特征向量方向 f 恒为 0。${critText}`;
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
