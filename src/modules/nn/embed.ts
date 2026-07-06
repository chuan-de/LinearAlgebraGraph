import { COLORS } from "../../theme";
import { el } from "../../ui";

/* ============================================================
   MLP 字符嵌入（Karpathy《Zero to Hero》视频③：makemore Part 2）
   上下文 3 个字符 → 预测下一个字符。
   每个字符先查表 C（27×2）变成 2 维向量——查表本身就是可学习
   的参数。在浏览器里真实训练，实时观看 27 个字符的嵌入点移动、
   聚类：几千步后元音会自己聚成一团，这就是"表示学习"。
   ============================================================ */

import { NAMES } from "./names";

const CHARS = ".abcdefghijklmnopqrstuvwxyz";
const V = 27; // 词表大小
const CTX = 3; // 上下文长度
const D = 2; // 嵌入维度（取 2 便于直接画出来）
const IN = CTX * D; // MLP 输入维度 = 6
const H = 48; // 隐藏层宽度
const B = 48; // minibatch 大小

const N_PARAMS = V * D + IN * H + H + H * V + V; // 54+288+48+1296+27 = 1713
const LR_HI = 0.3;
const LR_LO = 0.03;
const LR_DECAY_STEP = 10000; // 一万步后学习率衰减（模仿视频里的 lr decay）
const STEPS_PER_FRAME = 10;
const LOSS_KEEP = 800; // loss 曲线保留最近的步数
const BIGRAM_NLL = 2.41; // 对照：bigram 计数模型的平均 NLL

const SIZE = 580; // 散点区（正方形）
const LOSS_H = 60; // 底部 loss 曲线条
const TOTAL_H = SIZE + LOSS_H;
const PAD = 26; // 散点区内边距

const VOWELS = "aeiou";

/* ---------------- 数据集：滑窗样本 (3 字符上下文 → 下一字符) ---------------- */

function buildDataset(): { X: Int32Array; Y: Int32Array; n: number } {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const name of NAMES) {
    // 前缀相当于 "..."（上下文初始全 '.'），末尾补 '.' 作为结束符
    let c0 = 0;
    let c1 = 0;
    let c2 = 0;
    const s = `${name}.`;
    for (const ch of s) {
      const ix = CHARS.indexOf(ch);
      if (ix < 0) continue;
      xs.push(c0, c1, c2);
      ys.push(ix);
      c0 = c1;
      c1 = c2;
      c2 = ix;
    }
  }
  return { X: Int32Array.from(xs), Y: Int32Array.from(ys), n: ys.length };
}

export function mountEmbed(root: HTMLElement): () => void {
  const { X, Y, n: nSamples } = buildDataset();

  /* ---------------- LCG 伪随机 + 高斯近似（12 个均匀数求和 − 6） ---------------- */

  let rngState = 88675123;
  const rand = (): number => {
    rngState = (Math.imul(rngState, 1664525) + 1013904223) >>> 0;
    return rngState / 4294967296;
  };
  const gauss = (): number => {
    let s = 0;
    for (let i = 0; i < 12; i++) s += rand();
    return s - 6;
  };

  /* ---------------- 参数 ---------------- */

  const C = new Float64Array(V * D); // 嵌入表 27×2
  const W1 = new Float64Array(IN * H); // 6×H
  const b1 = new Float64Array(H);
  const W2 = new Float64Array(H * V); // H×27
  const b2 = new Float64Array(V);

  let stepCount = 0;
  let lossHist: number[] = [];

  function initParams(): void {
    for (let i = 0; i < C.length; i++) C[i] = gauss() * 1.0; // 初始散点铺开些
    for (let i = 0; i < W1.length; i++) W1[i] = gauss() * 0.1;
    b1.fill(0);
    for (let i = 0; i < W2.length; i++) W2[i] = gauss() * 0.1;
    b2.fill(0);
    stepCount = 0;
    lossHist = [];
  }
  initParams();

  /* ---------------- 前向 / 反向缓冲（一次分配，反复使用） ---------------- */

  const bx = new Int32Array(B * CTX); // 本批上下文字符索引
  const by = new Int32Array(B); // 本批目标字符
  const emb = new Float64Array(B * IN); // B×6
  const hbuf = new Float64Array(B * H); // B×H
  const probs = new Float64Array(B * V); // B×27（logits 原地变 softmax）
  const dlogits = new Float64Array(B * V);
  const dpre = new Float64Array(B * H); // 先当 dh 用，再原地乘 (1−h²)
  const demb = new Float64Array(B * IN);
  const gC = new Float64Array(V * D);
  const gW1 = new Float64Array(IN * H);
  const gb1 = new Float64Array(H);
  const gW2 = new Float64Array(H * V);
  const gb2 = new Float64Array(V);

  /** 一步 minibatch SGD，返回本批平均 NLL */
  function trainStep(lr: number): number {
    // -- 随机采样一个 minibatch --
    for (let b = 0; b < B; b++) {
      const i = (Math.random() * nSamples) | 0;
      bx[b * CTX] = X[i * CTX];
      bx[b * CTX + 1] = X[i * CTX + 1];
      bx[b * CTX + 2] = X[i * CTX + 2];
      by[b] = Y[i];
    }

    // -- 前向：查表拼接 emb (B×6) --
    for (let b = 0; b < B; b++) {
      for (let k = 0; k < CTX; k++) {
        const c = bx[b * CTX + k];
        emb[b * IN + k * D] = C[c * D];
        emb[b * IN + k * D + 1] = C[c * D + 1];
      }
    }
    // h = tanh(emb·W1 + b1)  (B×H)
    for (let b = 0; b < B; b++) {
      for (let j = 0; j < H; j++) {
        let s = b1[j];
        for (let i = 0; i < IN; i++) s += emb[b * IN + i] * W1[i * H + j];
        hbuf[b * H + j] = Math.tanh(s);
      }
    }
    // logits = h·W2 + b2 → softmax → probs (B×27)，顺便算 loss
    let loss = 0;
    for (let b = 0; b < B; b++) {
      let mx = -Infinity;
      for (let j = 0; j < V; j++) {
        let s = b2[j];
        for (let i = 0; i < H; i++) s += hbuf[b * H + i] * W2[i * V + j];
        probs[b * V + j] = s;
        if (s > mx) mx = s;
      }
      let sum = 0;
      for (let j = 0; j < V; j++) {
        const e = Math.exp(probs[b * V + j] - mx);
        probs[b * V + j] = e;
        sum += e;
      }
      for (let j = 0; j < V; j++) probs[b * V + j] /= sum;
      loss -= Math.log(probs[b * V + by[b]] + 1e-12);
    }
    loss /= B;

    // -- 反向 --
    // dlogits = (p − onehot)/B  (B×27)
    for (let b = 0; b < B; b++) {
      for (let j = 0; j < V; j++) {
        dlogits[b * V + j] = (probs[b * V + j] - (j === by[b] ? 1 : 0)) / B;
      }
    }
    // dW2 = hᵀ·dlogits (H×27)，db2 = 列和
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < V; j++) {
        let s = 0;
        for (let b = 0; b < B; b++) s += hbuf[b * H + i] * dlogits[b * V + j];
        gW2[i * V + j] = s;
      }
    }
    for (let j = 0; j < V; j++) {
      let s = 0;
      for (let b = 0; b < B; b++) s += dlogits[b * V + j];
      gb2[j] = s;
    }
    // dh = dlogits·W2ᵀ (B×H)，再原地 ⊙(1−h²) 得 dpre
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < H; i++) {
        let s = 0;
        for (let j = 0; j < V; j++) s += dlogits[b * V + j] * W2[i * V + j];
        const h = hbuf[b * H + i];
        dpre[b * H + i] = s * (1 - h * h);
      }
    }
    // dW1 = embᵀ·dpre (6×H)，db1 = 列和
    for (let i = 0; i < IN; i++) {
      for (let j = 0; j < H; j++) {
        let s = 0;
        for (let b = 0; b < B; b++) s += emb[b * IN + i] * dpre[b * H + j];
        gW1[i * H + j] = s;
      }
    }
    for (let j = 0; j < H; j++) {
      let s = 0;
      for (let b = 0; b < B; b++) s += dpre[b * H + j];
      gb1[j] = s;
    }
    // demb = dpre·W1ᵀ (B×6)，按 3 段散射累加回 dC
    gC.fill(0);
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < IN; i++) {
        let s = 0;
        for (let j = 0; j < H; j++) s += dpre[b * H + j] * W1[i * H + j];
        demb[b * IN + i] = s;
      }
      for (let k = 0; k < CTX; k++) {
        const c = bx[b * CTX + k];
        gC[c * D] += demb[b * IN + k * D];
        gC[c * D + 1] += demb[b * IN + k * D + 1];
      }
    }

    // -- SGD 更新 --
    for (let i = 0; i < C.length; i++) C[i] -= lr * gC[i];
    for (let i = 0; i < W1.length; i++) W1[i] -= lr * gW1[i];
    for (let i = 0; i < H; i++) b1[i] -= lr * gb1[i];
    for (let i = 0; i < W2.length; i++) W2[i] -= lr * gW2[i];
    for (let i = 0; i < V; i++) b2[i] -= lr * gb2[i];

    return loss;
  }

  /* ---------------- 单条上下文的前向（供采样用），返回 27 维概率 ---------------- */

  const sEmb = new Float64Array(IN);
  const sH = new Float64Array(H);
  const sP = new Float64Array(V);
  function forwardOne(c0: number, c1: number, c2: number): Float64Array {
    sEmb[0] = C[c0 * D];
    sEmb[1] = C[c0 * D + 1];
    sEmb[2] = C[c1 * D];
    sEmb[3] = C[c1 * D + 1];
    sEmb[4] = C[c2 * D];
    sEmb[5] = C[c2 * D + 1];
    for (let j = 0; j < H; j++) {
      let s = b1[j];
      for (let i = 0; i < IN; i++) s += sEmb[i] * W1[i * H + j];
      sH[j] = Math.tanh(s);
    }
    let mx = -Infinity;
    for (let j = 0; j < V; j++) {
      let s = b2[j];
      for (let i = 0; i < H; i++) s += sH[i] * W2[i * V + j];
      sP[j] = s;
      if (s > mx) mx = s;
    }
    let sum = 0;
    for (let j = 0; j < V; j++) {
      sP[j] = Math.exp(sP[j] - mx);
      sum += sP[j];
    }
    for (let j = 0; j < V; j++) sP[j] /= sum;
    return sP;
  }

  function sampleName(): string {
    let c0 = 0;
    let c1 = 0;
    let c2 = 0;
    let out = "";
    for (let step = 0; step < 24; step++) {
      const p = forwardOne(c0, c1, c2);
      let r = Math.random();
      let j = V - 1;
      for (let k = 0; k < V; k++) {
        r -= p[k];
        if (r <= 0) {
          j = k;
          break;
        }
      }
      if (j === 0) break;
      out += CHARS[j];
      c0 = c1;
      c1 = c2;
      c2 = j;
    }
    return out;
  }

  /* ---------------- 面板 ---------------- */

  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "MLP 字符嵌入（makemore Part 2）"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Zero to Hero 视频③。<b>上下文 = 前 3 个字符</b>，每个字符先查表 C " +
        "变成一个 <b>2 维向量</b>，三个向量拼起来送进 MLP 预测下一个字符——" +
        "<b>查表本身就是可学习的参数</b>，会被梯度推着在平面上移动。",
    ),
  );

  const ctrlRow = el("div", "row");
  const trainBtn = el("button", "btn primary", "▶ 训练");
  const resetBtn = el("button", "btn", "重置");
  const sampleBtn = el("button", "btn", "采样 5 个名字");
  ctrlRow.appendChild(trainBtn);
  ctrlRow.appendChild(resetBtn);
  ctrlRow.appendChild(sampleBtn);
  panel.appendChild(ctrlRow);

  const samplesDiv = el("div", "readout");
  samplesDiv.textContent = "（采样结果会列在这里）";
  panel.appendChild(samplesDiv);

  const readout = el("div", "readout");
  panel.appendChild(readout);

  const status = el("div", "status");
  panel.appendChild(status);

  /* ---------------- 黑板画布 ---------------- */

  const col = el("div", "canvas-col");
  col.appendChild(el("h3", "", "27 个字符的 2D 嵌入空间（C 的每一行是一个点）＋ loss 曲线"));
  const canvas = el("canvas", "plane");
  canvas.style.width = `${SIZE}px`;
  canvas.style.height = `${TOTAL_H}px`;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = SIZE * dpr;
  canvas.height = TOTAL_H * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  col.appendChild(canvas);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(col);
  root.appendChild(moduleEl);

  /* ---------------- 绘制 ---------------- */

  function charColor(k: number): string {
    if (k === 0) return COLORS.cyan;
    if (VOWELS.includes(CHARS[k])) return COLORS.gold;
    return COLORS.chalk;
  }

  function drawScatter(): void {
    // 数据范围 + 15% 边距
    let lox = Infinity;
    let hix = -Infinity;
    let loy = Infinity;
    let hiy = -Infinity;
    for (let k = 0; k < V; k++) {
      lox = Math.min(lox, C[k * D]);
      hix = Math.max(hix, C[k * D]);
      loy = Math.min(loy, C[k * D + 1]);
      hiy = Math.max(hiy, C[k * D + 1]);
    }
    const rx = Math.max(hix - lox, 1e-6);
    const ry = Math.max(hiy - loy, 1e-6);
    lox -= rx * 0.15;
    hix += rx * 0.15;
    loy -= ry * 0.15;
    hiy += ry * 0.15;
    const mx = (x: number): number => PAD + ((x - lox) / (hix - lox)) * (SIZE - 2 * PAD);
    const my = (y: number): number => SIZE - PAD - ((y - loy) / (hiy - loy)) * (SIZE - 2 * PAD);

    // 数据空间的坐标轴（若 0 在视野内）
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    if (lox < 0 && hix > 0) {
      ctx.beginPath();
      ctx.moveTo(mx(0), PAD * 0.4);
      ctx.lineTo(mx(0), SIZE - PAD * 0.4);
      ctx.stroke();
    }
    if (loy < 0 && hiy > 0) {
      ctx.beginPath();
      ctx.moveTo(PAD * 0.4, my(0));
      ctx.lineTo(SIZE - PAD * 0.4, my(0));
      ctx.stroke();
    }

    // 27 个字符
    ctx.font = "16px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let k = 0; k < V; k++) {
      const x = mx(C[k * D]);
      const y = my(C[k * D + 1]);
      ctx.fillStyle = charColor(k);
      ctx.fillText(CHARS[k], x, y);
    }

    // 图例
    ctx.font = "11px Consolas, monospace";
    ctx.textAlign = "left";
    ctx.fillStyle = COLORS.gold;
    ctx.fillText("● 元音 aeiou", 10, 14);
    ctx.fillStyle = COLORS.cyan;
    ctx.fillText("● '.'（边界符）", 110, 14);
    ctx.fillStyle = COLORS.tick;
    ctx.fillText("● 辅音", 226, 14);
  }

  function drawLossStrip(): void {
    // 分隔线
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, SIZE + 0.5);
    ctx.lineTo(SIZE, SIZE + 0.5);
    ctx.stroke();

    ctx.font = "10px Consolas, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillStyle = COLORS.tick;
    ctx.fillText(`loss（最近 ${LOSS_KEEP} 步）`, 8, SIZE + 5);

    if (lossHist.length < 2) return;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of lossHist) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    if (hi - lo < 1e-6) hi = lo + 1e-6;
    const top = SIZE + 8;
    const bot = TOTAL_H - 6;
    const ly = (v: number): number => bot - ((v - lo) / (hi - lo)) * (bot - top);

    // 对照线：bigram 计数模型 NLL ≈ 2.41
    if (BIGRAM_NLL > lo && BIGRAM_NLL < hi) {
      ctx.strokeStyle = COLORS.gold;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, ly(BIGRAM_NLL));
      ctx.lineTo(SIZE, ly(BIGRAM_NLL));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.gold;
      ctx.textAlign = "right";
      ctx.fillText("bigram 2.41", SIZE - 6, ly(BIGRAM_NLL) - 11);
    }

    ctx.strokeStyle = COLORS.red;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const m = lossHist.length;
    for (let i = 0; i < m; i++) {
      const x = (i / (LOSS_KEEP - 1)) * SIZE;
      const y = ly(lossHist[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function draw(): void {
    ctx.clearRect(0, 0, SIZE, TOTAL_H);
    drawScatter();
    drawLossStrip();
  }

  /* ---------------- 读出 / 状态 ---------------- */

  function avgLoss(): number {
    const m = Math.min(lossHist.length, 100);
    if (m === 0) return NaN;
    let s = 0;
    for (let i = lossHist.length - m; i < lossHist.length; i++) s += lossHist[i];
    return s / m;
  }

  function updateReadout(): void {
    const lr = stepCount < LR_DECAY_STEP ? LR_HI : LR_LO;
    const al = avgLoss();
    readout.textContent = [
      `数据集：${NAMES.length} 个名字 → ${nSamples} 条样本`,
      `步数：${stepCount}（lr = ${lr}，第 ${LR_DECAY_STEP} 步起降为 ${LR_LO}）`,
      `loss（近 100 步平均）：${Number.isNaN(al) ? "—" : al.toFixed(4)}`,
      `参数量：C ${V * D} + W1 ${IN * H} + b1 ${H} + W2 ${H * V} + b2 ${V} = ${N_PARAMS}`,
      `对照：bigram 计数模型 NLL ≈ ${BIGRAM_NLL}，这个 MLP 能压到 ~2.2 以下`,
    ].join("\n");
  }

  function updateStatus(): void {
    if (stepCount === 0) {
      status.textContent =
        "训练前：27 个点是随机初始化的噪声，散点没有任何结构。点「▶ 训练」开始，注意盯着金色的元音点。";
    } else if (stepCount < 3000) {
      status.textContent =
        "训练中：嵌入点被梯度推着移动。留意金色点（元音 a e i o u）——再过几千步它们会自己聚起来……";
    } else {
      status.innerHTML =
        `<span class="status-ok">✓ 观察金色点：元音 a e i o u 多半已经聚成一团。</span>` +
        "没有人告诉网络什么是元音——它为了预测下一个字符，自发发现了" +
        "「元音在名字里的行为相似」，把它们放到了嵌入空间的同一片区域。这就是嵌入 / 表示学习。";
    }
  }

  function render(): void {
    draw();
    updateReadout();
    updateStatus();
  }

  /* ---------------- 训练循环 ---------------- */

  let training = false;
  let rafId = 0;

  function trainLoop(): void {
    if (!training) return;
    for (let s = 0; s < STEPS_PER_FRAME; s++) {
      const lr = stepCount < LR_DECAY_STEP ? LR_HI : LR_LO;
      const loss = trainStep(lr);
      stepCount++;
      lossHist.push(loss);
      if (lossHist.length > LOSS_KEEP) lossHist.shift();
    }
    render();
    rafId = requestAnimationFrame(trainLoop);
  }

  function stopTraining(): void {
    training = false;
    trainBtn.textContent = "▶ 训练";
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
  }

  trainBtn.onclick = () => {
    if (training) {
      stopTraining();
    } else {
      training = true;
      trainBtn.textContent = "⏸ 暂停";
      rafId = requestAnimationFrame(trainLoop);
    }
  };

  resetBtn.onclick = () => {
    stopTraining();
    initParams();
    samplesDiv.textContent = "（采样结果会列在这里）";
    render();
  };

  sampleBtn.onclick = () => {
    const names: string[] = [];
    for (let k = 0; k < 5; k++) {
      const s = sampleName();
      names.push(s === "" ? "（空）" : s);
    }
    samplesDiv.textContent = `从 '...' 上下文逐字抽样：\n${names.join("\n")}`;
  };

  render();

  return () => {
    stopTraining();
  };
}
