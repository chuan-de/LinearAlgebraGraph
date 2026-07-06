import { COLORS } from "../../theme";
import { el, fmt } from "../../ui";

/* ============================================================
   自注意力机制（Karpathy《Zero to Hero》视频⑦：Let's build GPT）
   把单头 self-attention 的每一步拆开看：
   x = 嵌入 + 位置编码 → Q/K/V → scores = QKᵀ → (÷√d) →
   因果掩码 → softmax → 注意力矩阵 A → 输出 = A·V。
   ============================================================ */

const D = 8; // 向量维度
const MAXT = 12; // 序列长度上限
const SIZE = 460; // 画布 CSS 尺寸
const LABEL = 26; // 热力图四周标签留白

/** 32 位 LCG（Numerical Recipes 参数），返回 [0,1) */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function hexToRgb(hex: string): [number, number, number] {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

const GOLD_RGB = hexToRgb(COLORS.gold);
const RED_RGB = hexToRgb(COLORS.red);

/** 字符嵌入：按字符码用固定种子生成，每个字母的向量恒定不变 */
function charEmbed(code: number): number[] {
  const rnd = lcg(7777 + code * 131);
  const v: number[] = [];
  for (let k = 0; k < D; k++) v.push(2 * rnd() - 1);
  return v;
}

/** 位置编码：sin/cos，幅度 0.5 */
function posEncode(p: number): number[] {
  const v: number[] = [];
  for (let k = 0; k < D; k++) {
    const freq = 1 / Math.pow(50, Math.floor(k / 2) / (D / 2));
    v.push(0.5 * (k % 2 === 0 ? Math.sin(p * freq) : Math.cos(p * freq)));
  }
  return v;
}

/** D×D 随机权重矩阵 */
function makeWeight(seed: number): number[][] {
  const rnd = lcg(seed);
  const w: number[][] = [];
  for (let a = 0; a < D; a++) {
    w.push([]);
    for (let b = 0; b < D; b++) w[a].push((2 * rnd() - 1) * 0.6);
  }
  return w;
}

/** (T×D)·(D×D) */
function matmul(x: number[][], w: number[][]): number[][] {
  return x.map((row) => {
    const out = new Array<number>(D).fill(0);
    for (let a = 0; a < D; a++) {
      for (let b = 0; b < D; b++) out[b] += row[a] * w[a][b];
    }
    return out;
  });
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let k = 0; k < D; k++) s += a[k] * b[k];
  return s;
}

/** 单行 softmax（减最大值保证数值稳定，-Infinity 项概率为 0） */
function softmaxRow(row: number[]): number[] {
  let m = -Infinity;
  for (const v of row) m = Math.max(m, v);
  const e = row.map((v) => Math.exp(v - m));
  let s = 0;
  for (const v of e) s += v;
  return e.map((v) => v / s);
}

/** 一次前向计算的全部中间量 */
interface AttnResult {
  chars: string;
  scores: number[][]; // (÷√d 后、掩码前) QKᵀ
  masked: boolean[][]; // 哪些格子被因果掩码遮住
  A: number[][]; // softmax 后的注意力矩阵
  out: number[][]; // A·V
  variance: number; // 全部未掩码分数的方差（softmax 前）
}

function makeBoard(): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const canvas = el("canvas", "plane");
  canvas.style.width = `${SIZE}px`;
  canvas.style.height = `${SIZE}px`;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = SIZE * dpr;
  canvas.height = SIZE * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { canvas, ctx };
}

export function mountAttention(root: HTMLElement): () => void {
  // ---- 状态 ----
  let chars = "attention";
  let qkvSeed = 20260706;
  let useScale = true; // ÷√d
  let useMask = true; // 因果掩码
  let stage: 1 | 2 | 3 = 3; // 热力图显示阶段
  let selRow = chars.length - 1; // 选中的 query 行
  let hover: { i: number; j: number } | null = null;
  let res: AttnResult;

  function compute(): AttnResult {
    const T = chars.length;
    // 输入向量 x = 字符嵌入 + 位置编码
    const X: number[][] = [];
    for (let p = 0; p < T; p++) {
      const e = charEmbed(chars.charCodeAt(p));
      const pe = posEncode(p);
      X.push(e.map((v, k) => v + pe[k]));
    }
    const Q = matmul(X, makeWeight(qkvSeed + 1));
    const K = matmul(X, makeWeight(qkvSeed + 2));
    const V = matmul(X, makeWeight(qkvSeed + 3));

    const scale = useScale ? Math.sqrt(D) : 1;
    const scores: number[][] = [];
    const masked: boolean[][] = [];
    for (let i = 0; i < T; i++) {
      scores.push([]);
      masked.push([]);
      for (let j = 0; j < T; j++) {
        scores[i].push(dot(Q[i], K[j]) / scale);
        masked[i].push(useMask && j > i);
      }
    }
    // 未掩码分数的方差（教学点：不 ÷√d 时方差明显变大）
    let sum = 0;
    let sum2 = 0;
    let n = 0;
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        if (!masked[i][j]) {
          sum += scores[i][j];
          sum2 += scores[i][j] ** 2;
          n++;
        }
      }
    }
    const variance = n > 0 ? sum2 / n - (sum / n) ** 2 : 0;

    const A = scores.map((row, i) =>
      softmaxRow(row.map((v, j) => (masked[i][j] ? -Infinity : v))),
    );
    // 输出 = A·V（数值算出来，主要用于 readout）
    const out: number[][] = A.map((row) => {
      const o = new Array<number>(D).fill(0);
      for (let j = 0; j < T; j++) {
        for (let k = 0; k < D; k++) o[k] += row[j] * V[j][k];
      }
      return o;
    });
    return { chars, scores, masked, A, out, variance };
  }

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "自注意力：Q·K 对暗号，softmax 分权重"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应视频⑦《Let's build GPT》。注意力一句话：每个位置发出 <b>query</b>（我在找什么），" +
        "跟所有位置的 <b>key</b>（我有什么）对暗号，匹配度经 softmax 归一后，" +
        "对各位置的 <b>value</b>（我实际给出的信息）<b>加权平均</b>。" +
        "两个开关各藏一个知识点：关掉 <b>÷√d</b>，原始分数方差变大、softmax 变尖（梯度变差）；" +
        "<b>因果掩码</b> = 生成时不能偷看未来，去掉它就是 BERT 式双向注意力。",
    ),
  );

  const inputRow = el("div", "row");
  inputRow.appendChild(el("span", "", "输入序列"));
  const textInput = el("input");
  textInput.type = "text";
  textInput.value = chars;
  textInput.maxLength = 24;
  textInput.style.cssText =
    "flex:1;min-width:120px;background:#faf6ea;border:1px solid var(--paper-edge);" +
    "border-radius:7px;padding:6px 10px;font:14px var(--mono);color:var(--ink);";
  textInput.oninput = () => {
    const filtered = textInput.value.toLowerCase().replace(/[^a-z]/g, "").slice(0, MAXT);
    chars = filtered.length > 0 ? filtered : "attention";
    selRow = chars.length - 1;
    hover = null;
    render();
  };
  inputRow.appendChild(textInput);
  panel.appendChild(inputRow);
  panel.appendChild(
    el("p", "hint", `只保留小写字母 a–z，最长 ${MAXT} 个字符；留空则回到默认 "attention"。`),
  );

  const rerollBtn = el("button", "btn", "重掷 QKV 权重");
  rerollBtn.onclick = () => {
    qkvSeed = (Math.imul(qkvSeed, 48271) + 11) >>> 0;
    render();
  };
  panel.appendChild(rerollBtn);

  function makeCheck(text: string, checked: boolean, onChange: (v: boolean) => void): HTMLElement {
    const lab = el("label", "check");
    const box = el("input");
    box.type = "checkbox";
    box.checked = checked;
    box.onchange = () => onChange(box.checked);
    lab.appendChild(box);
    lab.appendChild(el("span", "", text));
    return lab;
  }
  panel.appendChild(
    makeCheck(`缩放 ÷√d（d = ${D}，√d ≈ ${Math.sqrt(D).toFixed(2)}）`, useScale, (v) => {
      useScale = v;
      render();
    }),
  );
  panel.appendChild(
    makeCheck("因果掩码（遮住 j > i：不许偷看未来）", useMask, (v) => {
      useMask = v;
      render();
    }),
  );

  const stageRow = el("div", "row");
  const stageBtns: HTMLButtonElement[] = [];
  const stageNames: [1 | 2 | 3, string][] = [
    [1, "① 原始分数 QKᵀ"],
    [2, "② 掩码后"],
    [3, "③ softmax 后"],
  ];
  for (const [s, name] of stageNames) {
    const b = el("button", "btn", name);
    b.onclick = () => {
      stage = s;
      syncStage();
      render();
    };
    stageBtns.push(b);
    stageRow.appendChild(b);
  }
  function syncStage(): void {
    stageNames.forEach(([s], idx) => stageBtns[idx].classList.toggle("primary", stage === s));
  }
  syncStage();
  panel.appendChild(stageRow);

  const readout = el("div", "readout");
  panel.appendChild(readout);
  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 两块黑板 ----
  const row = el("div", "canvas-row");
  const colA = el("div", "canvas-col");
  colA.appendChild(el("h3", "", "注意力矩阵（行 = query 位置，列 = key 位置；点击选行）"));
  const boardA = makeBoard();
  colA.appendChild(boardA.canvas);
  const colB = el("div", "canvas-col");
  colB.appendChild(el("h3", "", "弧线图：选中的 query 在「看」哪些位置"));
  const boardB = makeBoard();
  boardB.canvas.style.cursor = "default";
  colB.appendChild(boardB.canvas);
  row.appendChild(colA);
  row.appendChild(colB);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(row);
  root.appendChild(moduleEl);

  // ---- 画布 1：T×T 热力图 ----
  function drawHeatmap(): void {
    const ctx = boardA.ctx;
    const T = res.chars.length;
    const cell = (SIZE - LABEL) / T;
    ctx.clearRect(0, 0, SIZE, SIZE);

    // 归一化尺度：阶段①用全部分数，阶段②只看未被遮的
    let maxAbs = 0;
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        if (stage === 1 || !res.masked[i][j]) {
          maxAbs = Math.max(maxAbs, Math.abs(res.scores[i][j]));
        }
      }
    }
    if (maxAbs < 1e-12) maxAbs = 1;

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        const x = LABEL + j * cell;
        const y = LABEL + i * cell;
        const isMasked = stage >= 2 && res.masked[i][j];
        let alpha = 0;
        if (isMasked) {
          // 被遮的格子：近黑 + 深色叉
          ctx.fillStyle = "rgba(6,9,8,0.8)";
          ctx.fillRect(x, y, cell, cell);
          ctx.strokeStyle = "rgba(236,231,214,0.14)";
          ctx.lineWidth = 1.5;
          const p = cell * 0.28;
          ctx.beginPath();
          ctx.moveTo(x + p, y + p);
          ctx.lineTo(x + cell - p, y + cell - p);
          ctx.moveTo(x + cell - p, y + p);
          ctx.lineTo(x + p, y + cell - p);
          ctx.stroke();
        } else if (stage === 3) {
          // 概率：金色，强度 ∝ A[i][j]
          alpha = res.A[i][j];
          ctx.fillStyle = `rgba(${GOLD_RGB[0]},${GOLD_RGB[1]},${GOLD_RGB[2]},${alpha.toFixed(3)})`;
          ctx.fillRect(x, y, cell, cell);
        } else {
          // 分数：正 → 金，负 → 玫瑰，强度 ∝ |值|/max
          const v = res.scores[i][j];
          alpha = Math.abs(v) / maxAbs;
          const [r, g, b] = v >= 0 ? GOLD_RGB : RED_RGB;
          ctx.fillStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
          ctx.fillRect(x, y, cell, cell);
        }
        // 格内数值
        if (!isMasked) {
          const v = stage === 3 ? res.A[i][j] : res.scores[i][j];
          ctx.font = "10px Consolas, monospace";
          ctx.fillStyle = alpha > 0.55 ? "rgba(20,26,23,0.85)" : "rgba(236,231,214,0.62)";
          ctx.fillText(v.toFixed(2), x + cell / 2, y + cell / 2);
        }
      }
    }

    // 网格线
    ctx.strokeStyle = "rgba(236,231,214,0.09)";
    ctx.lineWidth = 1;
    for (let t = 0; t <= T; t++) {
      const p = LABEL + t * cell;
      ctx.beginPath();
      ctx.moveTo(p, LABEL);
      ctx.lineTo(p, SIZE);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(LABEL, p);
      ctx.lineTo(SIZE, p);
      ctx.stroke();
    }

    // 行 / 列字符标注
    ctx.font = "13px Consolas, monospace";
    for (let t = 0; t < T; t++) {
      const c = LABEL + (t + 0.5) * cell;
      ctx.fillStyle = t === selRow ? COLORS.cyan : COLORS.tick;
      ctx.fillText(res.chars[t], LABEL - 12, c); // 行标（query）
      ctx.fillStyle = COLORS.tick;
      ctx.fillText(res.chars[t], c, LABEL - 12); // 列标（key）
    }

    // 选中 query 行的高亮框
    ctx.strokeStyle = COLORS.cyan;
    ctx.lineWidth = 2;
    ctx.strokeRect(LABEL + 1, LABEL + selRow * cell + 1, T * cell - 2, cell - 2);

    // hover 高亮
    if (hover) {
      ctx.strokeStyle = COLORS.chalk;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(LABEL + hover.j * cell + 1, LABEL + hover.i * cell + 1, cell - 2, cell - 2);
    }
  }

  // ---- 画布 2：弧线图 ----
  function drawArcs(): void {
    const ctx = boardB.ctx;
    const T = res.chars.length;
    ctx.clearRect(0, 0, SIZE, SIZE);

    const margin = 34;
    const baseY = SIZE - 96;
    const step = (SIZE - 2 * margin) / T;
    const xOf = (j: number): number => margin + (j + 0.5) * step;
    const weights = res.A[selRow];

    // 弧线：线宽和不透明度 ∝ 注意力权重
    for (let j = 0; j < T; j++) {
      const w = weights[j];
      if (w < 0.004) continue;
      ctx.strokeStyle = `rgba(${GOLD_RGB[0]},${GOLD_RGB[1]},${GOLD_RGB[2]},${(0.12 + 0.88 * w).toFixed(3)})`;
      ctx.lineWidth = 0.6 + 7 * w;
      const x0 = xOf(selRow);
      const x1 = xOf(j);
      if (j === selRow) {
        // 自己看自己：头顶画一个小环
        ctx.beginPath();
        ctx.arc(x0, baseY - 15, 12, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        const h = Math.min(26 + Math.abs(x1 - x0) * 0.38, baseY - 26);
        ctx.beginPath();
        ctx.moveTo(x0, baseY);
        ctx.quadraticCurveTo((x0 + x1) / 2, baseY - h, x1, baseY);
        ctx.stroke();
        // 弧顶标权重
        if (w >= 0.08) {
          ctx.font = "11px Consolas, monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillStyle = `rgba(${GOLD_RGB[0]},${GOLD_RGB[1]},${GOLD_RGB[2]},0.9)`;
          ctx.fillText(w.toFixed(2), (x0 + x1) / 2, baseY - h / 2 - 4);
        }
      }
      // 被看位置的落点
      ctx.fillStyle = `rgba(${GOLD_RGB[0]},${GOLD_RGB[1]},${GOLD_RGB[2]},${(0.25 + 0.75 * w).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(x1, baseY, 2 + 4 * w, 0, Math.PI * 2);
      ctx.fill();
    }

    // 字符横排（Consolas 大字）
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const charY = baseY + 30;
    for (let j = 0; j < T; j++) {
      ctx.font = "28px Consolas, monospace";
      ctx.fillStyle = j === selRow ? COLORS.cyan : COLORS.chalk;
      ctx.fillText(res.chars[j], xOf(j), charY);
      ctx.font = "10px Consolas, monospace";
      ctx.fillStyle = COLORS.tick;
      ctx.fillText(String(j), xOf(j), charY + 22);
    }
    // query 字符圈出
    ctx.strokeStyle = COLORS.cyan;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(xOf(selRow), charY, 17, 0, Math.PI * 2);
    ctx.stroke();

    // 左上角说明当前 query
    ctx.textAlign = "left";
    ctx.font = "13px Consolas, monospace";
    ctx.fillStyle = COLORS.cyan;
    ctx.fillText(`query = 位置 ${selRow} 的 '${res.chars[selRow]}'`, 14, 22);

    // 底部一行小字
    ctx.textAlign = "center";
    ctx.font = '12.5px "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.fillStyle = COLORS.dim;
    ctx.fillText("输出 = 按这些权重对各位置的 value 加权平均", SIZE / 2, SIZE - 18);
  }

  // ---- 文案 ----
  function defaultStatus(): void {
    status.textContent =
      "把鼠标移到左图格子上查看数值和含义；点击某一行选中该 query，右图会跟着切换。";
  }

  function updateReadout(): void {
    const T = res.chars.length;
    const dist = res.A[selRow]
      .map((w, j) => (res.masked[selRow][j] ? null : `${res.chars[j]}${j}=${w.toFixed(2)}`))
      .filter((s): s is string => s !== null)
      .join(" ");
    readout.textContent = [
      `T = ${T}，d = ${D}，√d ≈ ${Math.sqrt(D).toFixed(3)}（缩放：${useScale ? "开" : "关"}）`,
      `因果掩码：${useMask ? "开（j>i 置 −∞，只看过去）" : "关（双向，BERT 式）"}`,
      `未掩码分数方差（softmax 前）= ${fmt(res.variance)}`,
      `选中 query：位置 ${selRow} 的 '${res.chars[selRow]}'`,
      `注意力分布：${dist}`,
      `输出 out[${selRow}] = A·V = [${res.out[selRow].map((v) => fmt(v)).join(", ")}]`,
    ].join("\n");
  }

  function render(): void {
    res = compute();
    if (selRow >= res.chars.length) selRow = res.chars.length - 1;
    drawHeatmap();
    drawArcs();
    updateReadout();
    if (!hover) defaultStatus();
  }

  // ---- 画布 1 交互 ----
  function cellAt(ev: PointerEvent): { i: number; j: number } | null {
    const rect = boardA.canvas.getBoundingClientRect();
    const T = res.chars.length;
    const cell = (SIZE - LABEL) / T;
    const j = Math.floor((ev.clientX - rect.left - LABEL) / cell);
    const i = Math.floor((ev.clientY - rect.top - LABEL) / cell);
    if (i < 0 || i >= T || j < 0 || j >= T) return null;
    return { i, j };
  }

  function hoverStatus(i: number, j: number): void {
    const ci = res.chars[i];
    const cj = res.chars[j];
    const who = `位置 ${i} 的 '${ci}' 对位置 ${j} 的 '${cj}'`;
    if (stage >= 2 && res.masked[i][j]) {
      status.textContent =
        `${who}：被因果掩码遮住（j=${j} > i=${i}，生成第 ${i} 个字符时还看不到未来）` +
        (stage === 3 ? "，softmax 后注意力 = 0。" : "，分数被置为 −∞。");
    } else if (stage === 3) {
      status.textContent = `${who} 的注意力 = ${res.A[i][j].toFixed(2)}（本行所有权重之和 = 1）。`;
    } else {
      status.textContent =
        `${who} 的${stage === 2 ? "掩码后" : "原始"}分数 q·k${useScale ? "/√d" : ""} = ` +
        `${res.scores[i][j].toFixed(3)}（越大表示 query 与 key 越「对得上暗号」）。`;
    }
  }

  const onMove = (ev: PointerEvent): void => {
    const c = cellAt(ev);
    if ((c === null) !== (hover === null) || (c && hover && (c.i !== hover.i || c.j !== hover.j))) {
      hover = c;
      drawHeatmap();
      if (hover) hoverStatus(hover.i, hover.j);
      else defaultStatus();
    }
  };
  const onLeave = (): void => {
    if (hover) {
      hover = null;
      drawHeatmap();
      defaultStatus();
    }
  };
  const onDown = (ev: PointerEvent): void => {
    const c = cellAt(ev);
    if (c && c.i !== selRow) {
      selRow = c.i;
      drawHeatmap();
      drawArcs();
      updateReadout();
      status.textContent =
        `已选中 query：位置 ${selRow} 的 '${res.chars[selRow]}'——右图的弧线就是它按注意力权重「看」各位置的样子。`;
    }
  };
  boardA.canvas.addEventListener("pointermove", onMove);
  boardA.canvas.addEventListener("pointerleave", onLeave);
  boardA.canvas.addEventListener("pointerdown", onDown);

  render();

  return () => {
    boardA.canvas.removeEventListener("pointermove", onMove);
    boardA.canvas.removeEventListener("pointerleave", onLeave);
    boardA.canvas.removeEventListener("pointerdown", onDown);
  };
}
