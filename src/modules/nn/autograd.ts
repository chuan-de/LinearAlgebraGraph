import { COLORS } from "../../theme";
import { el, fmt } from "../../ui";

/** 计算图节点（micrograd 的 Value） */
interface Node {
  label: string;
  data: number;
  grad: number | null; // null = 还没轮到它
  op: "" | "+" | "×" | "tanh";
  prev: Node[];
  depth: number;
  x: number;
  y: number;
}

interface BackStep {
  node: Node;
  /** 惰性生成：必须在前序步骤 apply 之后再调用，才能拿到真实梯度值 */
  desc: () => string;
  apply: () => void;
}

interface Graph {
  nodes: Node[];
  out: Node;
  steps: BackStep[];
}

interface Preset {
  name: string;
  leaves: { label: string; value: number }[];
  build: (v: number[], mk: Maker) => Node;
}

interface Maker {
  leaf(label: string, data: number): Node;
  add(label: string, a: Node, b: Node): Node;
  mul(label: string, a: Node, b: Node): Node;
  tanh(label: string, a: Node): Node;
}

const PRESETS: Preset[] = [
  {
    name: "入门：一次乘法",
    leaves: [
      { label: "a", value: 2 },
      { label: "b", value: -3 },
    ],
    build: (v, mk) => mk.mul("L", mk.leaf("a", v[0]), mk.leaf("b", v[1])),
  },
  {
    name: "Karpathy 经典表达式",
    leaves: [
      { label: "a", value: 2 },
      { label: "b", value: -3 },
      { label: "c", value: 10 },
      { label: "f", value: -2 },
    ],
    build: (v, mk) => {
      const e = mk.mul("e", mk.leaf("a", v[0]), mk.leaf("b", v[1]));
      const d = mk.add("d", e, mk.leaf("c", v[2]));
      return mk.mul("L", d, mk.leaf("f", v[3]));
    },
  },
  {
    name: "一个神经元：tanh(x·w + b)",
    leaves: [
      { label: "x₁", value: 2 },
      { label: "w₁", value: -3 },
      { label: "x₂", value: 0 },
      { label: "w₂", value: 1 },
      { label: "b", value: 6.881 },
    ],
    build: (v, mk) => {
      const p1 = mk.mul("x₁w₁", mk.leaf("x₁", v[0]), mk.leaf("w₁", v[1]));
      const p2 = mk.mul("x₂w₂", mk.leaf("x₂", v[2]), mk.leaf("w₂", v[3]));
      const s = mk.add("Σ", p1, p2);
      const n = mk.add("n", s, mk.leaf("b", v[4]));
      return mk.tanh("L", n);
    },
  },
];

/** 由预设 + 叶子值构建图、前向求值并预生成反向传播步骤 */
function buildGraph(preset: Preset, values: number[]): Graph {
  const nodes: Node[] = [];
  const mkNode = (label: string, data: number, op: Node["op"], prev: Node[]): Node => {
    const n: Node = { label, data, grad: null, op, prev, depth: 0, x: 0, y: 0 };
    nodes.push(n);
    return n;
  };
  const mk: Maker = {
    leaf: (label, data) => mkNode(label, data, "", []),
    add: (label, a, b) => mkNode(label, a.data + b.data, "+", [a, b]),
    mul: (label, a, b) => mkNode(label, a.data * b.data, "×", [a, b]),
    tanh: (label, a) => mkNode(label, Math.tanh(a.data), "tanh", [a]),
  };
  const out = preset.build(values, mk);

  // 反向传播步骤：先把 ∂L/∂L 置 1，再按逆拓扑序分发（nodes 数组本身就是拓扑序）
  const steps: BackStep[] = [
    {
      node: out,
      desc: () => `起点：L 对自己的导数是 1，所以 ∂L/∂L = 1`,
      apply: () => (out.grad = 1),
    },
  ];
  const g = (n: Node): number => n.grad ?? 0;
  const acc = (n: Node, amount: number): void => {
    n.grad = (n.grad ?? 0) + amount;
  };
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n = nodes[i];
    if (n.prev.length === 0) continue;
    if (n.op === "+") {
      const [a, b] = n.prev;
      steps.push({
        node: n,
        desc: () =>
          `「+」的局部导数是 1，梯度原样流过：\n` +
          `∂L/∂${a.label} += 1 × ${fmt(g(n))} = ${fmt(g(n))}\n` +
          `∂L/∂${b.label} += 1 × ${fmt(g(n))} = ${fmt(g(n))}`,
        apply: () => {
          acc(a, g(n));
          acc(b, g(n));
        },
      });
    } else if (n.op === "×") {
      const [a, b] = n.prev;
      steps.push({
        node: n,
        desc: () =>
          `「×」的局部导数是另一个因子：\n` +
          `∂L/∂${a.label} += ${b.label}的值(${fmt(b.data)}) × ${fmt(g(n))} = ${fmt(b.data * g(n))}\n` +
          `∂L/∂${b.label} += ${a.label}的值(${fmt(a.data)}) × ${fmt(g(n))} = ${fmt(a.data * g(n))}`,
        apply: () => {
          acc(a, b.data * g(n));
          acc(b, a.data * g(n));
        },
      });
    } else if (n.op === "tanh") {
      const [a] = n.prev;
      const local = 1 - n.data * n.data;
      steps.push({
        node: n,
        desc: () =>
          `tanh 的局部导数 = 1 − tanh² = 1 − (${fmt(n.data)})² = ${fmt(local)}\n` +
          `∂L/∂${a.label} += ${fmt(local)} × ${fmt(g(n))} = ${fmt(local * g(n))}`,
        apply: () => acc(a, local * g(n)),
      });
    }
  }
  layout(nodes);
  return { nodes, out, steps };
}

/** 布局：叶子在左，输出在右；同列均匀铺开并按子节点重心排序 */
function layout(nodes: Node[]): void {
  for (const n of nodes) {
    n.depth = n.prev.length ? Math.max(...n.prev.map((p) => p.depth)) + 1 : 0;
  }
  const maxDepth = Math.max(...nodes.map((n) => n.depth));
  const W = 880;
  const H = 560;
  const colW = (W - 200) / Math.max(1, maxDepth);
  for (let d = 0; d <= maxDepth; d++) {
    const col = nodes.filter((n) => n.depth === d);
    col.sort((a, b) => {
      const ya = a.prev.length ? a.prev.reduce((s, p) => s + p.y, 0) / a.prev.length : 0;
      const yb = b.prev.length ? b.prev.reduce((s, p) => s + p.y, 0) / b.prev.length : 0;
      return ya - yb;
    });
    col.forEach((n, i) => {
      n.x = 100 + d * colW;
      n.y = (H / (col.length + 1)) * (i + 1);
    });
  }
}

export function mountAutograd(root: HTMLElement): () => void {
  let presetIdx = 1;
  let values: number[] = PRESETS[1].leaves.map((l) => l.value);
  let stepCount = 0; // 已执行的反向传播步数
  let graph = buildGraph(PRESETS[presetIdx], values);

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "计算图与反向传播"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应视频①（micrograd）。任何表达式都是一张图：每个节点记得自己是谁算出来的。" +
        "<b>前向</b>从左到右算出值；<b>反向</b>从 L 出发，按链式法则把梯度逐节点往回传。" +
        "改左边的输入值，或者一步步点「下一步」，看梯度（金色）怎么流。",
    ),
  );

  const sel = el("select");
  PRESETS.forEach((p, i) => {
    const o = el("option", "", p.name);
    o.value = String(i);
    if (i === presetIdx) o.selected = true;
    sel.appendChild(o);
  });
  sel.onchange = () => {
    presetIdx = Number(sel.value);
    values = PRESETS[presetIdx].leaves.map((l) => l.value);
    stepCount = 0;
    rebuildInputs();
    refresh();
  };
  panel.appendChild(sel);

  const inputsBox = el("div", "");
  panel.appendChild(inputsBox);

  const prevBtn = el("button", "btn", "◀ 上一步");
  const nextBtn = el("button", "btn primary", "下一步 ▶");
  const endBtn = el("button", "btn", "⏭ 一步到底");
  const resetBtn = el("button", "btn", "↺ 清零");
  const btnRow = el("div", "row");
  [prevBtn, nextBtn, endBtn, resetBtn].forEach((b) => btnRow.appendChild(b));
  panel.appendChild(btnRow);

  prevBtn.onclick = () => {
    stepCount = Math.max(0, stepCount - 1);
    refresh();
  };
  nextBtn.onclick = () => {
    stepCount = Math.min(graph.steps.length, stepCount + 1);
    refresh();
  };
  endBtn.onclick = () => {
    stepCount = graph.steps.length;
    refresh();
  };
  resetBtn.onclick = () => {
    stepCount = 0;
    refresh();
  };

  const readout = el("div", "readout");
  panel.appendChild(readout);

  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 黑板 ----
  const canvasCol = el("div", "canvas-col");
  canvasCol.appendChild(el("h3", "", "计算图：值（粉笔白）从左往右算，梯度（金色）从右往左流"));
  const canvas = el("canvas", "plane");
  canvas.style.width = "800px";
  canvas.style.height = "540px";
  canvas.style.maxWidth = "100%";
  canvasCol.appendChild(canvas);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(canvasCol);
  root.appendChild(moduleEl);

  const ctx = canvas.getContext("2d")!;
  let cw = 0;
  let ch = 0;
  function resizeCanvas(): void {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    cw = rect.width;
    ch = rect.height;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resizeCanvas();

  function rebuildInputs(): void {
    inputsBox.innerHTML = "";
    PRESETS[presetIdx].leaves.forEach((lf, i) => {
      const row = el("div", "row");
      const lab = el("span", "slider-label", lf.label);
      const inp = el("input");
      inp.type = "number";
      inp.step = "0.5";
      inp.value = String(values[i]);
      inp.style.width = "80px";
      inp.className = "";
      inp.oninput = () => {
        const v = Number.parseFloat(inp.value);
        values[i] = Number.isFinite(v) ? v : 0;
        refresh();
      };
      const wrap = el("div", "matrix vec");
      wrap.appendChild(inp);
      row.appendChild(lab);
      row.appendChild(wrap);
      inputsBox.appendChild(row);
    });
  }

  /** 重建图 + 应用前 stepCount 步反向传播 + 重绘 */
  function refresh(): void {
    graph = buildGraph(PRESETS[presetIdx], values);
    for (let s = 0; s < stepCount; s++) graph.steps[s].apply();
    draw();
    writePanel();
  }

  function nodeSize(n: Node): [number, number] {
    return [n.prev.length ? 118 : 96, 64];
  }

  function draw(): void {
    ctx.clearRect(0, 0, cw, ch);
    const sx = cw / 880;
    const sy = ch / 560;
    const px = (n: Node): [number, number] => [n.x * sx, n.y * sy];
    const current = stepCount > 0 && stepCount <= graph.steps.length ? graph.steps[stepCount - 1].node : null;

    // 边
    for (const n of graph.nodes) {
      const [nx, ny] = px(n);
      const [nw2] = [nodeSize(n)[0] / 2];
      for (const p of n.prev) {
        const [pxx, pyy] = px(p);
        const pw2 = nodeSize(p)[0] / 2;
        const hot = current === n;
        ctx.strokeStyle = hot ? COLORS.gold : COLORS.dim;
        ctx.lineWidth = hot ? 2.2 : 1.2;
        ctx.beginPath();
        ctx.moveTo(pxx + pw2, pyy);
        ctx.bezierCurveTo(pxx + pw2 + 40, pyy, nx - nw2 - 40, ny, nx - nw2, ny);
        ctx.stroke();
        // 箭头
        ctx.fillStyle = hot ? COLORS.gold : COLORS.dim;
        ctx.beginPath();
        ctx.moveTo(nx - nw2, ny);
        ctx.lineTo(nx - nw2 - 9, ny - 4.5);
        ctx.lineTo(nx - nw2 - 9, ny + 4.5);
        ctx.closePath();
        ctx.fill();
      }
    }

    // 节点卡片
    for (const n of graph.nodes) {
      const [x, y] = px(n);
      const [w, h] = nodeSize(n);
      const hot = current === n;
      ctx.save();
      if (hot) {
        ctx.shadowColor = COLORS.gold;
        ctx.shadowBlur = 14;
      }
      ctx.fillStyle = "rgba(236,231,214,0.06)";
      ctx.strokeStyle = hot ? COLORS.gold : n.grad !== null ? "rgba(230,200,96,0.55)" : COLORS.dim;
      ctx.lineWidth = hot ? 2.2 : 1.3;
      ctx.beginPath();
      ctx.roundRect(x - w / 2, y - h / 2, w, h, 9);
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      ctx.textAlign = "center";
      // 标题行：e = a × b
      ctx.fillStyle = COLORS.chalk;
      ctx.font = "italic 14px Cambria, Georgia, serif";
      const title = n.prev.length
        ? n.op === "tanh"
          ? `${n.label} = tanh(${n.prev[0].label})`
          : `${n.label} = ${n.prev[0].label} ${n.op} ${n.prev[1].label}`
        : n.label;
      ctx.fillText(title, x, y - h / 2 + 16);
      // 值
      ctx.font = "12.5px Consolas, monospace";
      ctx.fillStyle = COLORS.chalk;
      ctx.fillText(`值 ${fmt(n.data)}`, x, y - h / 2 + 34);
      // 梯度
      if (n.grad !== null) {
        ctx.fillStyle = COLORS.gold;
        ctx.fillText(`∂L/∂ = ${fmt(n.grad)}`, x, y - h / 2 + 51);
      } else {
        ctx.fillStyle = "rgba(236,231,214,0.3)";
        ctx.fillText("∂L/∂ = ?", x, y - h / 2 + 51);
      }
    }
  }

  function writePanel(): void {
    const done = stepCount >= graph.steps.length;
    const lines: string[] = [`L = ${fmt(graph.out.data)}`, `反向传播 第 ${stepCount}/${graph.steps.length} 步`];
    if (stepCount > 0) {
      lines.push("", graph.steps[stepCount - 1].desc());
    }
    if (done) {
      // 数值检验：轻推每个叶子，对照解析梯度
      lines.push("", "数值检验（轻推 h=0.0001）：");
      const h = 1e-4;
      const leaves = graph.nodes.filter((n) => n.prev.length === 0);
      PRESETS[presetIdx].leaves.forEach((lf, i) => {
        const bumped = [...values];
        bumped[i] += h;
        const g2 = buildGraph(PRESETS[presetIdx], bumped);
        const numeric = (g2.out.data - graph.out.data) / h;
        const leaf = leaves.find((n) => n.label === lf.label);
        lines.push(`${lf.label}: 解析 ${fmt(leaf?.grad ?? 0)}  数值 ${fmt(numeric)}`);
      });
    }
    readout.textContent = lines.join("\n");

    if (done) {
      status.innerHTML =
        `<span class="status-ok">✓ 反向传播完成</span>：每个输入的 grad 就是它的“灵敏度”——` +
        `轻推它 0.01，L 大约变化 0.01 × grad。数值检验和解析梯度一致，这就是 micrograd 的全部秘密。`;
    } else if (stepCount === 0) {
      status.textContent = "前向传播已完成（每个节点有了值）。点「下一步」开始反向传播。";
    } else {
      status.textContent = "梯度正在往回流：每一步只用当前节点的局部导数 × 它已有的梯度。";
    }
  }

  const onResize = (): void => {
    resizeCanvas();
    draw();
  };
  window.addEventListener("resize", onResize);

  rebuildInputs();
  refresh();

  return () => {
    window.removeEventListener("resize", onResize);
  };
}
