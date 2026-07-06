import { COLORS } from "../../theme";
import { el, fmt, sliderRow } from "../../ui";
import { NAMES } from "./names";

/* ============================================================
   网络健康监视器（Karpathy《Zero to Hero》视频④：
   makemore Part 3 —— Activations & Gradients, BatchNorm）

   实验：字符 3-gram 上下文 → 嵌入 C[27×10] 拼成 30 维
   → 5 层 tanh 全连接（宽 64）→ 输出 27。
   只做初始化后的一次前向 + 反向，看「初始化增益 g」
   如何决定各层激活 / 梯度的分布形状。
   ============================================================ */

const CHARS = ".abcdefghijklmnopqrstuvwxyz";
const V = 27;
const CTX = 3; // 上下文长度
const EMB = 10; // 嵌入维度
const NIN = CTX * EMB; // 30
const H = 64; // 隐层宽度
const NLAYERS = 5; // 隐层数
const B = 256; // batch 大小
const NBINS = 40; // 直方图 bin 数
const SAT_T = 0.97; // 饱和阈值 |h| > 0.97
const GAIN_TANH = 5 / 3; // tanh 的推荐增益

// 画布布局
const CW = 880;
const CHH = 560;
const TOP = 40; // 标题行以下
const ROW_H = (CHH - TOP - 8) / NLAYERS;
const HIST_H = ROW_H - 30;
const LX = 62; // 左列直方图 x
const RX = 470; // 右列直方图 x
const HW = 300; // 直方图宽

/* ---------------- 随机数：LCG 固定种子 ---------------- */

function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** 近似标准高斯：12 个均匀随机数之和 − 6 */
function gauss(rand: () => number): number {
  let t = 0;
  for (let i = 0; i < 12; i++) t += rand();
  return t - 6;
}

/* ---------------- 数据集：滑窗 3-gram ---------------- */

function buildDataset(): { X: Int32Array; Y: Int32Array } {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const name of NAMES) {
    let c0 = 0;
    let c1 = 0;
    let c2 = 0;
    for (const ch of name + ".") {
      const ix = CHARS.indexOf(ch);
      if (ix < 0) continue;
      xs.push(c0, c1, c2);
      ys.push(ix);
      c0 = c1;
      c1 = c2;
      c2 = ix;
    }
  }
  return { X: Int32Array.from(xs), Y: Int32Array.from(ys) };
}

/* ---------------- 矩阵小工具（行主序扁平数组） ---------------- */

/** out[b,j] = Σ_k a[b,k]·w[k,j]，a:[rows×K] w:[K×N] */
function matmul(a: Float64Array, w: Float64Array, rows: number, K: number, N: number): Float64Array {
  const out = new Float64Array(rows * N);
  for (let b = 0; b < rows; b++) {
    const ab = b * K;
    const ob = b * N;
    for (let k = 0; k < K; k++) {
      const av = a[ab + k];
      if (av === 0) continue;
      const wk = k * N;
      for (let j = 0; j < N; j++) out[ob + j] += av * w[wk + j];
    }
  }
  return out;
}

/** dprev[b,k] = Σ_j d[b,j]·w[k,j]（即 d @ Wᵀ），w:[K×N] d:[rows×N] */
function matmulBT(d: Float64Array, w: Float64Array, rows: number, N: number, K: number): Float64Array {
  const out = new Float64Array(rows * K);
  for (let b = 0; b < rows; b++) {
    const db = b * N;
    const ob = b * K;
    for (let k = 0; k < K; k++) {
      const wk = k * N;
      let s = 0;
      for (let j = 0; j < N; j++) s += d[db + j] * w[wk + j];
      out[ob + k] = s;
    }
  }
  return out;
}

function stdOf(a: Float64Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m += a[i];
  m /= a.length;
  let v = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - m;
    v += d * d;
  }
  return Math.sqrt(v / a.length);
}

/* ---------------- 实验：一次前向 + 反向 ---------------- */

interface RunResult {
  loss: number;
  acts: Float64Array[]; // 每个隐层的 tanh 输出 [B×H]
  grads: Float64Array[]; // 每个隐层激活的梯度 ∂L/∂h [B×H]
  actStd: number[];
  satPct: number[]; // |h| > 0.97 的比例（%）
  gradStd: number[];
}

function runExperiment(g: number, bn: boolean, seed: number, X: Int32Array, Y: Int32Array): RunResult {
  const rand = makeRng(seed);

  // ---- 初始化：W ~ 近似高斯 × g/√fan_in，b = 0（一次前反向中恒为 0，故省略）----
  const C = new Float64Array(V * EMB);
  for (let i = 0; i < C.length; i++) C[i] = gauss(rand);
  const Ws: Float64Array[] = [];
  for (let l = 0; l < NLAYERS; l++) {
    const fin = l === 0 ? NIN : H;
    const w = new Float64Array(fin * H);
    const sc = g / Math.sqrt(fin);
    for (let i = 0; i < w.length; i++) w[i] = gauss(rand) * sc;
    Ws.push(w);
  }
  const Wout = new Float64Array(H * V);
  const scOut = g / Math.sqrt(H);
  for (let i = 0; i < Wout.length; i++) Wout[i] = gauss(rand) * scOut;

  // ---- 随机取一个 batch ----
  const n = Y.length;
  const bi = new Int32Array(B);
  for (let b = 0; b < B; b++) bi[b] = Math.floor(rand() * n);

  // ---- 前向 ----
  const emb = new Float64Array(B * NIN);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < CTX; t++) {
      const ch = X[bi[b] * CTX + t];
      for (let e = 0; e < EMB; e++) emb[b * NIN + t * EMB + e] = C[ch * EMB + e];
    }
  }
  const acts: Float64Array[] = [];
  const sigmas: Float64Array[] = []; // BN 每个神经元的 σ（供反向 ÷σ）
  let h = emb;
  let hin = NIN;
  for (let l = 0; l < NLAYERS; l++) {
    const pre = matmul(h, Ws[l], B, hin, H);
    const sigma = new Float64Array(H).fill(1);
    if (bn) {
      // BatchNorm：用当前 batch 的 μ/σ 归一化（γ=1, β=0）
      for (let j = 0; j < H; j++) {
        let mu = 0;
        for (let b = 0; b < B; b++) mu += pre[b * H + j];
        mu /= B;
        let va = 0;
        for (let b = 0; b < B; b++) {
          const d = pre[b * H + j] - mu;
          va += d * d;
        }
        const sg = Math.sqrt(va / B + 1e-5);
        sigma[j] = sg;
        for (let b = 0; b < B; b++) pre[b * H + j] = (pre[b * H + j] - mu) / sg;
      }
    }
    const out = new Float64Array(B * H);
    for (let i = 0; i < out.length; i++) out[i] = Math.tanh(pre[i]);
    acts.push(out);
    sigmas.push(sigma);
    h = out;
    hin = H;
  }
  const logits = matmul(h, Wout, B, H, V);

  // ---- 平均 NLL loss + dlogits = (softmax − onehot)/B ----
  let loss = 0;
  const dlogits = new Float64Array(B * V);
  for (let b = 0; b < B; b++) {
    const o = b * V;
    let mx = -Infinity;
    for (let j = 0; j < V; j++) mx = Math.max(mx, logits[o + j]);
    let se = 0;
    for (let j = 0; j < V; j++) se += Math.exp(logits[o + j] - mx);
    const y = Y[bi[b]];
    loss -= logits[o + y] - mx - Math.log(se);
    for (let j = 0; j < V; j++) {
      const p = Math.exp(logits[o + j] - mx) / se;
      dlogits[o + j] = (p - (j === y ? 1 : 0)) / B;
    }
  }
  loss /= B;

  // ---- 反向：逐层回传激活梯度 ----
  const grads: Float64Array[] = new Array(NLAYERS);
  let dh = matmulBT(dlogits, Wout, B, V, H); // ∂L/∂h₅
  for (let l = NLAYERS - 1; l >= 0; l--) {
    grads[l] = dh;
    if (l === 0) break; // 再往前是嵌入层，可视化不需要
    const hl = acts[l];
    const dpre = new Float64Array(B * H);
    for (let i = 0; i < dpre.length; i++) dpre[i] = dh[i] * (1 - hl[i] * hl[i]); // tanh 局部导数 1−h²
    if (bn) {
      // BN 反向：直通近似 —— 梯度直接 ÷σ 传回（忽略 μ/σ 对输入的依赖，简化演示）
      const sg = sigmas[l];
      for (let b = 0; b < B; b++) for (let j = 0; j < H; j++) dpre[b * H + j] /= sg[j];
    }
    dh = matmulBT(dpre, Ws[l], B, H, H); // ∂L/∂h_{l−1}
  }

  // ---- 统计 ----
  const actStd: number[] = [];
  const satPct: number[] = [];
  const gradStd: number[] = [];
  for (let l = 0; l < NLAYERS; l++) {
    const a = acts[l];
    actStd.push(stdOf(a));
    let sat = 0;
    for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > SAT_T) sat++;
    satPct.push((sat / a.length) * 100);
    gradStd.push(stdOf(grads[l]));
  }
  return { loss, acts, grads, actStd, satPct, gradStd };
}

/* ---------------- 小工具 ---------------- */

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

/** 小数字的科学计数法显示，如 3.2e-3 */
function expFmt(v: number): string {
  if (v === 0) return "0";
  return v.toExponential(1).replace("e-", "e−");
}

/* ---------------- 模块入口 ---------------- */

export function mountHealth(root: HTMLElement): () => void {
  const { X, Y } = buildDataset();

  // ---- 交互状态 ----
  let gain = 1;
  let bnOn = false;
  let seed = 42;
  let result = runExperiment(gain, bnOn, seed, X, Y);

  // ---- 控制面板（讲义纸） ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "网络健康监视器（激活与梯度）"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Zero to Hero 视频④（Activations &amp; Gradients, BatchNorm）。" +
        "<b>健康的网络 = 每一层的激活和梯度分布都大致相同</b>——既不逐层坍缩到 0，也不挤进饱和区。" +
        "这里是一个 5 层 tanh MLP（3-gram 字符模型）初始化后的一次前向 + 反向：" +
        "拖动增益 <b>g</b>，看各层分布如何随初始化变化。tanh 的推荐增益是 <b>5/3 ≈ 1.67</b>。",
    ),
  );

  const gSlider = sliderRow("初始化增益 g", 0.1, 3, 0.05, gain, (v) => {
    gain = v;
    recompute();
  });
  panel.appendChild(gSlider.root);

  // BatchNorm 开关
  const bnLabel = el("label", "check");
  const bnBox = el("input");
  bnBox.type = "checkbox";
  bnBox.onchange = () => {
    bnOn = bnBox.checked;
    recompute();
  };
  bnLabel.appendChild(bnBox);
  bnLabel.appendChild(document.createTextNode("开启 BatchNorm（tanh 之前按 batch 归一化，γ=1 β=0）"));
  panel.appendChild(bnLabel);

  // 按钮行：换一批 / g 预设
  const btnRow = el("div", "row");
  const rerollBtn = el("button", "btn primary", "换一批 / 重掷");
  rerollBtn.onclick = () => {
    seed = (Math.imul(seed, 48271) + 1) >>> 0;
    recompute();
  };
  const presetBtn = el("button", "btn", "g → 5/3（tanh 推荐）");
  presetBtn.onclick = () => {
    gain = GAIN_TANH;
    gSlider.set(gain);
    recompute();
  };
  btnRow.appendChild(rerollBtn);
  btnRow.appendChild(presetBtn);
  panel.appendChild(btnRow);

  const readout = el("div", "readout");
  panel.appendChild(readout);

  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 黑板画布 ----
  const col = el("div", "canvas-col");
  col.appendChild(el("h3", "", "5 层 tanh 隐层：左 = 激活直方图，右 = 激活梯度直方图（初始化后一次前反向）"));
  const canvas = el("canvas", "plane");
  canvas.style.width = `${CW}px`;
  canvas.style.height = `${CHH}px`;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = CW * dpr;
  canvas.height = CHH * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  col.appendChild(canvas);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(col);
  root.appendChild(moduleEl);

  // ---- 绘制 ----

  /** 画一个直方图：data 落入 [lo,hi] 分 NBINS 个 bin，柱高归一到 histH */
  function drawHist(data: Float64Array, lo: number, hi: number, x: number, yBase: number, color: string): void {
    const bins = new Array<number>(NBINS).fill(0);
    const span = hi - lo;
    for (let i = 0; i < data.length; i++) {
      let k = Math.floor(((data[i] - lo) / span) * NBINS);
      if (k < 0) k = 0;
      if (k >= NBINS) k = NBINS - 1;
      bins[k]++;
    }
    let mx = 1;
    for (const c of bins) mx = Math.max(mx, c);
    const bw = HW / NBINS;
    ctx.fillStyle = color;
    for (let k = 0; k < NBINS; k++) {
      if (bins[k] === 0) continue;
      const bh = (bins[k] / mx) * HIST_H;
      ctx.fillRect(x + k * bw + 0.5, yBase - bh, bw - 1, bh);
    }
    // 基线 + 中线（0 刻度）
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, yBase + 0.5);
    ctx.lineTo(x + HW, yBase + 0.5);
    ctx.stroke();
    ctx.strokeStyle = "rgba(236,231,214,0.12)";
    ctx.beginPath();
    ctx.moveTo(x + HW / 2 + 0.5, yBase - HIST_H);
    ctx.lineTo(x + HW / 2 + 0.5, yBase);
    ctx.stroke();
  }

  function draw(): void {
    ctx.clearRect(0, 0, CW, CHH);

    // 顶部标题行
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.font = "14px Cambria, Georgia, serif";
    ctx.fillStyle = COLORS.chalk;
    ctx.fillText("激活分布（tanh 输出，[−1, 1]）", LX + HW / 2, 24);
    ctx.fillText("梯度分布（∂loss/∂h，自动范围）", RX + HW / 2, 24);

    const goldFill = hexToRgba(COLORS.gold, 0.55);
    const cyanFill = hexToRgba(COLORS.cyan, 0.55);

    for (let l = 0; l < NLAYERS; l++) {
      const y0 = TOP + l * ROW_H;
      const yBase = y0 + 10 + HIST_H;

      // 层标签
      ctx.font = "12px Consolas, monospace";
      ctx.fillStyle = COLORS.tick;
      ctx.textAlign = "center";
      ctx.fillText(`第${l + 1}层`, LX / 2 - 4, y0 + 10 + HIST_H / 2);

      // 左：激活直方图（固定 [−1,1]）
      drawHist(result.acts[l], -1, 1, LX, yBase, goldFill);
      ctx.fillStyle = COLORS.tick;
      ctx.font = "10px Consolas, monospace";
      ctx.fillText("−1", LX, yBase + 12);
      ctx.fillText("0", LX + HW / 2, yBase + 12);
      ctx.fillText("1", LX + HW, yBase + 12);

      // 左侧标注：std + 饱和度
      ctx.textAlign = "left";
      ctx.font = "12px Consolas, monospace";
      ctx.fillStyle = COLORS.chalk;
      ctx.fillText(`std ${result.actStd[l].toFixed(2)}`, LX + HW + 12, y0 + 10 + HIST_H / 2 - 8);
      const sat = result.satPct[l];
      ctx.fillStyle = sat > 15 ? COLORS.red : COLORS.tick;
      ctx.fillText(`饱和 ${sat.toFixed(1)}%`, LX + HW + 12, y0 + 10 + HIST_H / 2 + 10);

      // 右：梯度直方图（对称自动范围）
      const gArr = result.grads[l];
      let gm = 0;
      for (let i = 0; i < gArr.length; i++) gm = Math.max(gm, Math.abs(gArr[i]));
      if (gm === 0) gm = 1e-12;
      ctx.textAlign = "center";
      drawHist(gArr, -gm, gm, RX, yBase, cyanFill);
      ctx.fillStyle = COLORS.tick;
      ctx.font = "10px Consolas, monospace";
      ctx.fillText(`−${expFmt(gm)}`, RX + 18, yBase + 12);
      ctx.fillText("0", RX + HW / 2, yBase + 12);
      ctx.fillText(expFmt(gm), RX + HW - 18, yBase + 12);

      // 右侧标注：梯度 std
      ctx.textAlign = "left";
      ctx.font = "12px Consolas, monospace";
      ctx.fillStyle = COLORS.chalk;
      ctx.fillText(`std ${expFmt(result.gradStd[l])}`, RX + HW + 12, y0 + 10 + HIST_H / 2);
      ctx.textAlign = "center";
    }
  }

  function updateReadout(): void {
    const lines = [
      `网络：C[27×10] → 30 → 64 ×${NLAYERS} (tanh) → 27`,
      `样本：${Y.length} 个 3-gram 窗口 · batch ${B} · 种子 ${seed}`,
      `初始平均 NLL = ${result.loss.toFixed(4)}（均匀基线 ln27 ≈ ${Math.log(27).toFixed(4)}）`,
      `g = ${fmt(gain)}（tanh 推荐 5/3 ≈ 1.67）`,
    ];
    if (bnOn) lines.push("BN 反向：直通近似（梯度 ÷σ 直传），简化演示");
    readout.textContent = lines.join("\n");
  }

  function updateStatus(): void {
    if (bnOn) {
      status.innerHTML =
        `<span class="status-ok">BatchNorm 开启：</span>不管 g 怎么调，每层的分布都被拉回标准形——` +
        "这就是 BatchNorm 让深网络好训的原因（对初始化不再敏感）。";
    } else if (gain < 0.5) {
      status.textContent =
        "g 太小：激活逐层坍缩到 0，梯度也跟着消失——深层网络学不动。";
    } else if (gain > 2.2) {
      status.textContent =
        "g 太大：tanh 大面积饱和（|h|≈1），局部导数 1−h²≈0，梯度被掐死。";
    } else if (Math.abs(gain - GAIN_TANH) <= 0.15) {
      status.innerHTML =
        `<span class="status-ok">g ≈ 5/3：各层分布基本一致</span>——这就是 Xavier / Kaiming 初始化的由来：` +
        "按 1/√fan_in 缩放，再乘上抵消非线性收缩的增益。";
    } else {
      status.textContent =
        "拖动 g 观察 std 逐层的走势：略偏小 → 激活逐层收缩；略偏大 → 饱和比例逐层上升。试试 5/3 ≈ 1.67。";
    }
  }

  function recompute(): void {
    result = runExperiment(gain, bnOn, seed, X, Y);
    updateReadout();
    updateStatus();
    draw();
  }

  updateReadout();
  updateStatus();
  draw();

  return () => {
    /* 无全局监听 / 定时器，无需清理 */
  };
}
