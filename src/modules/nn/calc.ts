import { Plane } from "../../plane";
import { COLORS } from "../../theme";
import { el, fmt, sliderRow } from "../../ui";

/** 教程页：导数与链式法则（Zero to Hero 基础概念） */

interface FnDef {
  name: string;
  f: (x: number) => number;
  df: (x: number) => number;
}

const FNS: FnDef[] = [
  { name: "f(x) = x²", f: (x) => x * x, df: (x) => 2 * x },
  { name: "f(x) = sin x", f: Math.sin, df: Math.cos },
  { name: "f(x) = tanh x", f: Math.tanh, df: (x) => 1 - Math.tanh(x) ** 2 },
  { name: "f(x) = x³ − 3x", f: (x) => x ** 3 - 3 * x, df: (x) => 3 * x * x - 3 },
];

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
const f3 = (n: number): string => (Object.is(n, -0) ? 0 : n).toFixed(3);

export function mountCalc(root: HTMLElement): () => void {
  // ---- 状态 ----
  let fnIdx = 0;
  let x0 = 0.8; // 画布 1：切点位置
  let h = 1; // 画布 1：割线步长
  let cx = 1.2; // 画布 2：链式法则的输入 x

  // ================= 文章骨架 =================
  const article = el("div", "article");

  article.appendChild(el("h2", "", "导数与链式法则"));
  article.appendChild(
    el(
      "p",
      "",
      "训练一个神经网络，说穿了只是一件事：网络里有几百万个可以调的旋钮（<b>参数</b>），" +
        "还有一个衡量“错得多离谱”的数——<b>loss</b>。训练就是不停地微调这些旋钮，让 loss 一点点变小。",
    ),
  );
  article.appendChild(
    el(
      "p",
      "",
      "问题是：旋钮这么多，每一个该往哪边拧、拧多少？回答这个问题所需要的<b>全部数学</b>，" +
        "就是导数。这一页我们把它讲透——不需要任何微积分背景。",
    ),
  );

  // ---- 第 1 节 ----
  article.appendChild(el("h3", "", "1. 导数是灵敏度"));
  article.appendChild(
    el(
      "p",
      "",
      "先忘掉教科书里“极限”的形式定义，抓住直觉：导数回答的问题是——" +
        "<b>轻轻推一下输入，输出会动多少？</b>想象 f 是一台机器，x 是进料口的旋钮。" +
        "把 x 从当前位置轻推一小步 h，看输出 f(x+h) 相比 f(x) 变了多少，再除以步长 h，" +
        "得到的就是“每单位推动引起的输出变化”：",
    ),
  );
  article.appendChild(el("div", "formula", "f′(x) ≈ (f(x+h) − f(x)) / h"));
  article.appendChild(
    el(
      "p",
      "",
      "这个比值在几何上是连接两点的<b>割线</b>的斜率。h 取得越小，割线就越贴近曲线在这一点的" +
        "<b>切线</b>——切线的斜率就是导数的精确值。导数是正的，说明往右推 x 输出会涨；" +
        "是负的，输出会跌；绝对值越大，这个点越“灵敏”。",
    ),
  );
  article.appendChild(
    el(
      "p",
      "",
      "在下面的黑板上试一试：<b>拖动曲线上的亮点</b>改变 x，再把 h 从 2 慢慢缩到 0.01，" +
        "看玫瑰色的割线如何逐渐“躺”到金色切线上，读出区里的数值斜率也随之收敛到真实导数。",
    ),
  );

  const wrap1 = el("div", "board-wrap");
  const canvas1 = el("canvas", "plane");
  canvas1.style.height = "320px";
  wrap1.appendChild(canvas1);
  article.appendChild(wrap1);

  const selRow = el("div", "row");
  selRow.appendChild(el("span", "slider-label", "f"));
  const sel = el("select");
  FNS.forEach((fn, i) => {
    const o = el("option", "", fn.name);
    o.value = String(i);
    sel.appendChild(o);
  });
  sel.value = "0";
  sel.onchange = () => {
    fnIdx = Number(sel.value);
    render1();
  };
  selRow.appendChild(sel);
  article.appendChild(selRow);

  const hSlider = sliderRow("h", 0.01, 2, 0.01, h, (v) => {
    h = v;
    render1();
  });
  article.appendChild(hSlider.root);

  const readout1 = el("div", "readout");
  article.appendChild(readout1);

  article.appendChild(
    el(
      "p",
      "",
      "顺便留意 tanh：它在 0 附近很陡（导数接近 1），到了两侧却几乎躺平（导数趋近 0）。" +
        "这正是深度网络里“梯度消失”的源头——以后你会在激活函数那一章再遇到它。",
    ),
  );

  // ---- 第 2 节 ----
  article.appendChild(el("h3", "", "2. 链式法则：灵敏度相乘"));
  article.appendChild(
    el(
      "p",
      "",
      "神经网络不是一个函数，而是<b>一长串函数首尾相接</b>：上一层的输出是下一层的输入。" +
        "所以真正要回答的问题是：复合函数的导数怎么算？设 y = f(g(x))，" +
        "把中间结果记作 u = g(x)，信号的流向就是 x → u → y。",
    ),
  );
  article.appendChild(
    el(
      "p",
      "",
      "推理只有一句话：把 x 轻推一小步，u 会跟着动 <b>du/dx</b> 倍的一小步；" +
        "u 的这一小步又让 y 动 <b>dy/du</b> 倍。两级传动连起来，总的放大倍数自然是两者<b>相乘</b>：",
    ),
  );
  article.appendChild(el("div", "formula", "dy/dx = (dy/du) · (du/dx)"));
  article.appendChild(
    el(
      "p",
      "",
      "这就是链式法则——<b>灵敏度沿着传动链相乘</b>。就像两级齿轮：第一级 1:3，第二级 1:2，" +
        "整体就是 1:6。链条再长也一样，把沿途每一段的局部导数一路乘过去。",
    ),
  );
  article.appendChild(
    el(
      "p",
      "",
      "下面的三条数轴就是一条两级传动链：x 经过平方得到 u = x²，u 再经过正弦得到 y = sin u。" +
        "拖动滑块，看三个标记点如何联动，两段“齿轮比”（局部导数）怎样相乘出总导数。" +
        "特别试试 x = 0 附近：第一级齿轮比 2x ≈ 0，无论第二级多灵敏，总导数都被乘成了 0。",
    ),
  );

  const wrap2 = el("div", "board-wrap");
  const canvas2 = el("canvas", "plane");
  canvas2.style.height = "320px";
  wrap2.appendChild(canvas2);
  article.appendChild(wrap2);

  const xSlider = sliderRow("x", -2, 2, 0.01, cx, (v) => {
    cx = v;
    render2();
  });
  article.appendChild(xSlider.root);

  const readout2 = el("div", "readout");
  article.appendChild(readout2);

  // ---- 第 3 节 ----
  article.appendChild(el("h3", "", "3. 结语：反向传播只是链式法则"));
  article.appendChild(
    el(
      "p",
      "",
      "到这里，你已经掌握了训练神经网络所需的全部微积分。<b>反向传播</b>（backpropagation）" +
        "听起来吓人，其实没有任何新数学——它只是把网络画成一张计算图，然后从输出端往回走，" +
        "把链式法则机械地、一段一段地执行一遍，为每个参数算出它对 loss 的灵敏度（<b>梯度</b>）。" +
        "知道了灵敏度，往反方向拧一点旋钮，loss 就会下降。",
    ),
  );
  article.appendChild(
    el(
      "p",
      "",
      "准备好了就去「计算图与反向传播」模块，亲眼看看链式法则在一张图上跑起来的样子。",
    ),
  );

  const moduleEl = el("div", "module");
  moduleEl.appendChild(article);
  root.appendChild(moduleEl);

  // ================= 画布 1：切线与割线 =================
  const plane1 = new Plane(canvas1);
  plane1.scale = 55;
  plane1.onRedraw = () => render1();
  plane1.attachDrag((v) => {
    x0 = clamp(v.x, -3, 3);
    render1();
  });

  /** 世界坐标线段 */
  function seg(
    plane: Plane,
    a: { x: number; y: number },
    b: { x: number; y: number },
    color: string,
    width = 2,
    dash?: number[],
  ): void {
    const ctx = plane.ctx;
    const [x1, y1] = plane.toScreen(a);
    const [x2, y2] = plane.toScreen(b);
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

  function drawCurve(plane: Plane, f: (x: number) => number): void {
    const w = plane.canvas.clientWidth;
    const xl = plane.toWorld(0, 0).x;
    const xr = plane.toWorld(w, 0).x;
    const ctx = plane.ctx;
    ctx.save();
    ctx.strokeStyle = COLORS.chalk;
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.beginPath();
    const n = 260;
    for (let i = 0; i <= n; i++) {
      const x = xl + ((xr - xl) * i) / n;
      const [sx, sy] = plane.toScreen({ x, y: f(x) });
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    ctx.restore();
  }

  function render1(): void {
    const { f, df } = FNS[fnIdx];
    const fx = f(x0);
    const fxh = f(x0 + h);
    const secant = (fxh - fx) / h;
    const exact = df(x0);

    plane1.clear();
    plane1.grid();
    plane1.axes();
    drawCurve(plane1, f);

    // 割线（玫瑰粉笔，虚线）：沿弦方向向两侧各延伸一点
    const dirx = h;
    const diry = fxh - fx;
    seg(
      plane1,
      { x: x0 - 0.5 * dirx, y: fx - 0.5 * diry },
      { x: x0 + 1.5 * dirx, y: fx + 1.5 * diry },
      COLORS.red,
      2,
      [6, 6],
    );
    // 两个端点到 x 轴的淡辅助线，标出步长 h
    seg(plane1, { x: x0, y: 0 }, { x: x0, y: fx }, COLORS.dim, 1, [3, 5]);
    seg(plane1, { x: x0 + h, y: 0 }, { x: x0 + h, y: fxh }, COLORS.dim, 1, [3, 5]);

    // 切线（金色实线段）
    const L = 1.4;
    seg(
      plane1,
      { x: x0 - L, y: fx - L * exact },
      { x: x0 + L, y: fx + L * exact },
      COLORS.gold,
      2.5,
    );

    plane1.point({ x: x0 + h, y: fxh }, COLORS.red, 4.5, "x+h");
    plane1.point({ x: x0, y: fx }, COLORS.chalk, 5.5, "x");
    plane1.ring({ x: x0, y: fx }, COLORS.dim, 11, 1.5);

    readout1.textContent = [
      `x = ${fmt(x0)}   h = ${fmt(h)}   f(x) = ${f3(fx)}`,
      `割线斜率 (f(x+h) − f(x)) / h = ${f3(secant)}`,
      `切线斜率 f′(x)（精确导数）   = ${f3(exact)}`,
      `两者之差 = ${f3(Math.abs(secant - exact))} —— h 越小越接近`,
    ].join("\n");
  }

  // ================= 画布 2：链式法则数轴 =================
  const plane2 = new Plane(canvas2);
  plane2.onRedraw = () => render2();

  const X_RANGE: [number, number] = [-2, 2];
  const U_RANGE: [number, number] = [0, 4];
  const Y_RANGE: [number, number] = [-1, 1];

  function render2(): void {
    const W = canvas2.clientWidth;
    const H = canvas2.clientHeight;
    const top = 52;
    const bottom = H - 34;
    const ax = [W * 0.18, W * 0.5, W * 0.82];
    const ranges = [X_RANGE, U_RANGE, Y_RANGE];
    const names = ["x", "u = x²", "y = sin u"];

    const u = cx * cx;
    const y = Math.sin(u);
    const dudx = 2 * cx;
    const dydu = Math.cos(u);
    const vals = [cx, u, y];
    const markColors = [COLORS.green, COLORS.gold, COLORS.red];

    const py = (v: number, r: [number, number]): number =>
      bottom - ((v - r[0]) / (r[1] - r[0])) * (bottom - top);

    plane2.clear();
    const ctx = plane2.ctx;

    // 三条竖直数轴
    for (let i = 0; i < 3; i++) {
      ctx.strokeStyle = COLORS.axis;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ax[i], top);
      ctx.lineTo(ax[i], bottom);
      ctx.stroke();
      // 端点刻度
      for (const end of [0, 1] as const) {
        const v = ranges[i][end === 0 ? 1 : 0];
        const yy = end === 0 ? top : bottom;
        ctx.strokeStyle = COLORS.axis;
        ctx.beginPath();
        ctx.moveTo(ax[i] - 5, yy);
        ctx.lineTo(ax[i] + 5, yy);
        ctx.stroke();
        ctx.fillStyle = COLORS.tick;
        ctx.font = "11px Consolas, monospace";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(fmt(v), ax[i] - 9, yy);
      }
      // 轴名
      ctx.fillStyle = COLORS.chalk;
      ctx.font = "italic 15px Cambria, Georgia, serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(names[i], ax[i], top - 14);
    }

    // 标记点位置
    const pts = vals.map((v, i) => ({ x: ax[i], y: py(v, ranges[i]) }));

    // 连线（淡色）
    ctx.strokeStyle = COLORS.dim;
    ctx.lineWidth = 1.5;
    for (let i = 0; i < 2; i++) {
      ctx.beginPath();
      ctx.moveTo(pts[i].x + 7, pts[i].y);
      ctx.lineTo(pts[i + 1].x - 7, pts[i + 1].y);
      ctx.stroke();
    }

    // 每段旁标注局部导数
    const derivLabels = [`du/dx = 2x = ${fmt(dudx)}`, `dy/du = cos u = ${fmt(dydu)}`];
    const derivColors = [COLORS.cyan, COLORS.purple];
    ctx.font = "12.5px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    for (let i = 0; i < 2; i++) {
      const mx = (pts[i].x + pts[i + 1].x) / 2;
      const my = (pts[i].y + pts[i + 1].y) / 2;
      ctx.fillStyle = derivColors[i];
      ctx.fillText(derivLabels[i], mx, my - 9);
    }

    // 标记点 + 数值
    ctx.font = "12px Consolas, monospace";
    for (let i = 0; i < 3; i++) {
      ctx.fillStyle = markColors[i];
      ctx.beginPath();
      ctx.arc(pts[i].x, pts[i].y, 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.textAlign = i === 0 ? "right" : "left";
      ctx.textBaseline = "middle";
      const label = ["x", "u", "y"][i] + " = " + fmt(vals[i]);
      ctx.fillText(label, pts[i].x + (i === 0 ? -12 : 12), pts[i].y);
    }

    // 数值导数验证（中心差分）
    const hh = 1e-4;
    const F = (t: number): number => Math.sin(t * t);
    const numeric = (F(cx + hh) - F(cx - hh)) / (2 * hh);

    readout2.textContent = [
      `x = ${fmt(cx)}   →   u = x² = ${f3(u)}   →   y = sin u = ${f3(y)}`,
      `链式法则：dy/dx = dy/du · du/dx = ${f3(dydu)} × ${f3(dudx)} = ${f3(dydu * dudx)}`,
      `数值验证：(F(x+h) − F(x−h)) / 2h = ${f3(numeric)}   (h = 0.0001)`,
    ].join("\n");
  }

  // ================= resize / cleanup =================
  const onResize = (): void => {
    plane1.resize();
    plane2.resize();
    render1();
    render2();
  };
  window.addEventListener("resize", onResize);

  render1();
  render2();

  return () => {
    window.removeEventListener("resize", onResize);
  };
}
