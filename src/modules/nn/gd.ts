import type { Vec } from "../../math";
import { Plane } from "../../plane";
import { COLORS } from "../../theme";
import { el, fmt, sliderRow } from "../../ui";

/** 一维目标函数预设 */
const PRESETS1: { name: string; f: (x: number) => number; df: (x: number) => number; start: number }[] = [
  {
    name: "碗 f(x) = x²",
    f: (x) => x * x,
    df: (x) => 2 * x,
    start: 2.5,
  },
  {
    name: "双坑 f(x) = x⁴/4 − 2x² + x/2（有局部极小）",
    f: (x) => (x * x * x * x) / 4 - 2 * x * x + x / 2,
    df: (x) => x * x * x - 4 * x + 0.5,
    start: 2.8,
  },
  {
    name: "陡崖缓坡 f(x) = e⁻ˣ + 0.3x（左陡右缓）",
    f: (x) => Math.exp(-x) + 0.3 * x - 0.66,
    df: (x) => -Math.exp(-x) + 0.3,
    start: 4,
  },
];

const LEVELS = [0.5, 1, 2, 4, 8, 16]; // 二维等高线等级
const BETA = 0.9; // 动量系数
const MAX_TRAIL_1D = 40;
const MAX_TRAIL_2D = 200;
const STEP_MS = 80;
const DIVERGE_X = 50; // 一维发散判定
const DIVERGE_R = 300; // 二维发散判定

/** lr 跨三个数量级，fmt 会把 0.001 抹成 0，单独格式化 */
function fmtLr(v: number): string {
  return v >= 0.01 ? String(Math.round(v * 1000) / 1000) : v.toExponential(1);
}

export function mountGd(root: HTMLElement): () => void {
  // ---- 状态 ----
  let preset = 0;
  let x1 = PRESETS1[0].start; // 一维当前位置
  let v1 = 0; // 一维动量速度
  let hist1: number[] = [];
  let diverged1 = false;

  let kappa = 10; // 条件数 κ
  let p2: Vec = { x: 4.5, y: 2 }; // 二维当前点
  let v2: Vec = { x: 0, y: 0 };
  let hist2: Vec[] = [];
  let diverged2 = false;

  let lrExp = -1; // lr = 10^lrExp
  let momentum = false;
  let steps = 0;
  let timer = 0; // 自动模式 interval

  const lr = (): number => Math.pow(10, lrExp);
  const f1 = (x: number): number => PRESETS1[preset].f(x);
  const df1 = (x: number): number => PRESETS1[preset].df(x);
  const grad2 = (p: Vec): Vec => ({ x: p.x, y: kappa * p.y });

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "梯度下降实验台"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应《Zero to Hero》视频①后半：梯度 ∇f 指向上坡最陡的方向，" +
        "<b>梯度下降就是反着它走一小步</b>：x ← x − lr·∇f(x)，学习率 lr 就是步长——" +
        "太小磨蹭半天，太大一步迈过头。左边在曲线上滚球，右边在椭圆碗里沿等高线下山，" +
        "拖动小球换起点，调 lr 感受「步长」的分寸。",
    ),
  );

  const sel = el("select");
  PRESETS1.forEach((p, i) => {
    const o = el("option", "", p.name);
    o.value = String(i);
    sel.appendChild(o);
  });
  sel.value = "0";
  sel.onchange = () => {
    preset = Number(sel.value) || 0;
    reset1D();
    steps = 0;
    render();
  };
  panel.appendChild(sel);

  // lr 滑块：滑块值是指数（对数感），旁边显示实际 lr
  const lrRow = el("div", "row");
  lrRow.appendChild(el("span", "slider-label", "lr"));
  const lrInput = el("input");
  lrInput.type = "range";
  lrInput.min = "-3";
  lrInput.max = "0.3";
  lrInput.step = "0.02";
  lrInput.value = String(lrExp);
  const lrVal = el("span", "slider-value", fmtLr(lr()));
  lrInput.oninput = () => {
    lrExp = lrInput.valueAsNumber;
    lrVal.textContent = fmtLr(lr());
    render();
  };
  lrRow.appendChild(lrInput);
  lrRow.appendChild(lrVal);
  panel.appendChild(lrRow);

  const kappaSlider = sliderRow("κ", 1, 40, 1, kappa, (v) => {
    kappa = v;
    render();
  });
  panel.appendChild(kappaSlider.root);

  const btnRow = el("div", "row");
  const btnStep = el("button", "btn", "单步");
  const btnAuto = el("button", "btn primary", "▶ 自动");
  const btnReset = el("button", "btn", "重置");
  btnRow.appendChild(btnStep);
  btnRow.appendChild(btnAuto);
  btnRow.appendChild(btnReset);
  panel.appendChild(btnRow);

  const momLabel = el("label", "check");
  const momInput = el("input");
  momInput.type = "checkbox";
  momLabel.appendChild(momInput);
  momLabel.appendChild(document.createTextNode("动量（β = 0.9：v ← βv − lr·∇f，x ← x + v）"));
  momInput.onchange = () => {
    momentum = momInput.checked;
    v1 = 0;
    v2 = { x: 0, y: 0 };
    render();
  };
  panel.appendChild(momLabel);

  const readout = el("div", "readout");
  panel.appendChild(readout);

  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 两块黑板 ----
  const leftCol = el("div", "canvas-col");
  leftCol.appendChild(el("h3", "", "一维：在曲线上滚球（拖动设初始 x，红色是切线）"));
  const leftCanvas = el("canvas", "plane");
  leftCol.appendChild(leftCanvas);

  const rightCol = el("div", "canvas-col");
  rightCol.appendChild(el("h3", "", "二维：碗 f(x,y) = ½(x² + κy²) 等高线（青色是 −∇f 方向）"));
  const rightCanvas = el("canvas", "plane");
  rightCol.appendChild(rightCanvas);

  const canvases = el("div", "canvas-row");
  canvases.appendChild(leftCol);
  canvases.appendChild(rightCol);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(canvases);
  root.appendChild(moduleEl);

  const plane1 = new Plane(leftCanvas);
  const plane2 = new Plane(rightCanvas);
  plane1.scale = 30;
  plane2.scale = 30;
  plane1.onRedraw = () => render();
  plane2.onRedraw = () => render();

  plane1.attachDrag((v) => {
    x1 = Math.max(-20, Math.min(20, v.x));
    v1 = 0;
    hist1 = [];
    diverged1 = false;
    steps = 0;
    render();
  });
  plane2.attachDrag((v) => {
    p2 = v;
    v2 = { x: 0, y: 0 };
    hist2 = [];
    diverged2 = false;
    steps = 0;
    render();
  });

  // ---- 迭代 ----
  function reset1D(): void {
    x1 = PRESETS1[preset].start;
    v1 = 0;
    hist1 = [];
    diverged1 = false;
  }

  function reset2D(): void {
    p2 = { x: 4.5, y: 2 };
    v2 = { x: 0, y: 0 };
    hist2 = [];
    diverged2 = false;
  }

  /** 两个画布同时各走一步 */
  function step(): void {
    const eta = lr();
    if (!diverged1) {
      hist1.push(x1);
      if (hist1.length > MAX_TRAIL_1D) hist1.shift();
      const g = df1(x1);
      if (momentum) {
        v1 = BETA * v1 - eta * g;
        x1 += v1;
      } else {
        x1 -= eta * g;
      }
      if (!Number.isFinite(x1) || Math.abs(x1) > DIVERGE_X) {
        diverged1 = true;
        stopAuto();
      }
    }
    if (!diverged2) {
      hist2.push(p2);
      if (hist2.length > MAX_TRAIL_2D) hist2.shift();
      const g = grad2(p2);
      if (momentum) {
        v2 = { x: BETA * v2.x - eta * g.x, y: BETA * v2.y - eta * g.y };
        p2 = { x: p2.x + v2.x, y: p2.y + v2.y };
      } else {
        p2 = { x: p2.x - eta * g.x, y: p2.y - eta * g.y };
      }
      if (!Number.isFinite(p2.x + p2.y) || Math.hypot(p2.x, p2.y) > DIVERGE_R) {
        diverged2 = true;
        stopAuto();
      }
    }
    steps++;
    render();
  }

  function startAuto(): void {
    if (timer) return;
    timer = window.setInterval(step, STEP_MS);
    btnAuto.textContent = "⏸ 停";
  }

  function stopAuto(): void {
    if (timer) {
      window.clearInterval(timer);
      timer = 0;
    }
    btnAuto.textContent = "▶ 自动";
  }

  btnStep.onclick = () => step();
  btnAuto.onclick = () => (timer ? stopAuto() : startAuto());
  btnReset.onclick = () => {
    stopAuto();
    reset1D();
    reset2D();
    steps = 0;
    render();
  };

  // ---- 状态判定 ----
  /** 二维锯齿：最近几步 y 坐标来回变号 */
  function zigzagging(): boolean {
    if (momentum || kappa < 5 || hist2.length < 7) return false;
    const ys = hist2.slice(-7).map((p) => p.y);
    let flips = 0;
    for (let i = 1; i < ys.length; i++) {
      if (Math.abs(ys[i]) > 0.02 && ys[i] * ys[i - 1] < 0) flips++;
    }
    return flips >= 4;
  }

  function statusText(): string {
    if (diverged1 || diverged2) {
      const which = diverged1 && diverged2 ? "两边都" : diverged1 ? "一维" : "二维";
      return (
        `<b>${which}发散了</b>：学习率太大，一步迈过了头，越荡越高。` +
        "把 lr 调小一个数量级，重置再试。"
      );
    }
    if (zigzagging()) {
      return (
        "<b>锯齿路径</b>：陡的方向在震荡、缓的方向在蠕动——这就是病态条件数（κ 大）。" +
        "动量 / Adam 就是为它发明的，勾选「动量」试试。"
      );
    }
    const g1 = df1(x1);
    if (preset === 1 && steps > 0 && x1 > 0.5 && Math.abs(g1) < 0.08) {
      return (
        "<b>掉进局部极小</b>：右边这个坑不是最深的（全局极小在 x ≈ −2.06），" +
        "但梯度已经 ≈ 0，普通梯度下降出不去了。换个初始点，或勾选动量借惯性冲过山脊。"
      );
    }
    const g2 = grad2(p2);
    if (steps > 0 && Math.abs(g1) < 1e-3 && Math.hypot(g2.x, g2.y) < 1e-3) {
      return '<span class="status-ok">已收敛</span>：两边的梯度都 ≈ 0，到达（局部）最低点。';
    }
    if (steps > 0 && lr() < 0.005) {
      return "lr 很小：方向没错，但步子太碎，收敛会很慢——这就是「学习率太小磨蹭」的样子。";
    }
    return "点「单步」看一步怎么走，或「▶ 自动」连续下山；调大 κ 和 lr 能看到锯齿。";
  }

  // ---- 绘制 ----
  function render1(): void {
    const ctx = plane1.ctx;
    plane1.clear();
    plane1.grid();
    plane1.axes();

    // 采样折线画曲线
    const xMin = plane1.toWorld(0, 0).x;
    const xMax = -xMin;
    ctx.save();
    ctx.strokeStyle = COLORS.chalk;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    const N = 240;
    for (let i = 0; i <= N; i++) {
      const x = xMin + ((xMax - xMin) * i) / N;
      const [sx, sy] = plane1.toScreen({ x, y: f1(x) });
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    ctx.restore();

    // 历史痕迹：越旧越淡
    if (hist1.length > 1) {
      ctx.save();
      ctx.strokeStyle = COLORS.dim;
      ctx.lineWidth = 1;
      ctx.beginPath();
      hist1.forEach((x, i) => {
        const [sx, sy] = plane1.toScreen({ x, y: f1(x) });
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
      ctx.restore();
    }
    hist1.forEach((x, i) => {
      ctx.save();
      ctx.globalAlpha = 0.12 + (0.55 * (i + 1)) / hist1.length;
      plane1.point({ x, y: f1(x) }, COLORS.chalk, 2.5);
      ctx.restore();
    });

    // 当前球处的切线短段
    const g = df1(x1);
    const dx = 0.7 / Math.hypot(1, g);
    const [tx1, ty1] = plane1.toScreen({ x: x1 - dx, y: f1(x1) - g * dx });
    const [tx2, ty2] = plane1.toScreen({ x: x1 + dx, y: f1(x1) + g * dx });
    ctx.save();
    ctx.strokeStyle = COLORS.red;
    ctx.lineWidth = 2.2;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(tx1, ty1);
    ctx.lineTo(tx2, ty2);
    ctx.stroke();
    ctx.restore();

    // 小球
    const ball: Vec = { x: x1, y: f1(x1) };
    plane1.point(ball, COLORS.gold, 5.5);
    plane1.ring(ball, COLORS.gold, 10, 1.8);
  }

  function render2(): void {
    const ctx = plane2.ctx;
    plane2.clear();
    plane2.grid();
    plane2.axes();

    // 等高线：½(x² + κy²) = c ⇒ x = √(2c)cosθ, y = √(2c/κ)sinθ
    ctx.save();
    ctx.strokeStyle = COLORS.dim;
    ctx.lineWidth = 1;
    for (const c of LEVELS) {
      const a = Math.sqrt(2 * c);
      const b = Math.sqrt((2 * c) / kappa);
      ctx.beginPath();
      const SEG = 72;
      for (let i = 0; i <= SEG; i++) {
        const t = (i / SEG) * Math.PI * 2;
        const [sx, sy] = plane2.toScreen({ x: a * Math.cos(t), y: b * Math.sin(t) });
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();
    }
    ctx.restore();

    // 历史路径（金色连线）
    if (hist2.length > 0) {
      ctx.save();
      ctx.strokeStyle = COLORS.gold;
      ctx.lineWidth = 1.8;
      ctx.lineJoin = "round";
      ctx.beginPath();
      const pts = [...hist2, p2];
      pts.forEach((p, i) => {
        const [sx, sy] = plane2.toScreen(p);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
      ctx.restore();
    }

    // 负梯度方向短箭头
    const g = grad2(p2);
    const gm = Math.hypot(g.x, g.y);
    if (gm > 1e-6) {
      const to: Vec = { x: p2.x - (g.x / gm) * 0.9, y: p2.y - (g.y / gm) * 0.9 };
      plane2.arrow(p2, to, COLORS.cyan, 2);
    }

    // 当前点
    plane2.point(p2, COLORS.gold, 5.5);
    plane2.ring(p2, COLORS.gold, 10, 1.8);
  }

  function render(): void {
    render1();
    render2();
    const g2 = grad2(p2);
    readout.textContent =
      `一维  x = ${fmt(x1)}   f(x) = ${fmt(f1(x1))}   f'(x) = ${fmt(df1(x1))}\n` +
      `二维  (x, y) = (${fmt(p2.x)}, ${fmt(p2.y)})   f = ${fmt(0.5 * (p2.x * p2.x + kappa * p2.y * p2.y))}` +
      `   ‖∇f‖ = ${fmt(Math.hypot(g2.x, g2.y))}\n` +
      `lr = ${fmtLr(lr())}   κ = ${fmt(kappa)}   步数 = ${steps}`;
    status.innerHTML = statusText();
  }

  const onResize = (): void => {
    plane1.resize();
    plane2.resize();
    render();
  };
  window.addEventListener("resize", onResize);

  render();

  return () => {
    stopAuto();
    window.removeEventListener("resize", onResize);
  };
}
