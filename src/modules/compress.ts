import { COLORS } from "../theme";
import { el, sliderRow } from "../ui";

const N = 8;
const SIZE = 300; // 每块画布的 CSS 尺寸
const CELL = SIZE / N;
const BOARD_DARK: [number, number, number] = [27, 36, 32]; // #1b2420
const CHALK: [number, number, number] = [236, 231, 214]; // #ece7d6

/** 一维 DCT-II 正交基：T[u][i] = α(u)·cos((2i+1)uπ/16) */
const T: number[][] = (() => {
  const t: number[][] = [];
  for (let u = 0; u < N; u++) {
    const alpha = u === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
    t.push([]);
    for (let i = 0; i < N; i++) {
      t[u].push(alpha * Math.cos(((2 * i + 1) * u * Math.PI) / (2 * N)));
    }
  }
  return t;
})();

/** 2D DCT-II：F = T · f · Tᵀ（f、F 都是按行展开的 64 维数组） */
function dct2(f: number[]): number[] {
  const F = new Array<number>(N * N).fill(0);
  for (let u = 0; u < N; u++) {
    for (let v = 0; v < N; v++) {
      let s = 0;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          s += T[u][i] * T[v][j] * f[i * N + j];
        }
      }
      F[u * N + v] = s;
    }
  }
  return F;
}

/** 2D 逆 DCT：f = Tᵀ · F · T */
function idct2(F: number[]): number[] {
  const f = new Array<number>(N * N).fill(0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let u = 0; u < N; u++) {
        for (let v = 0; v < N; v++) {
          s += T[u][i] * T[v][j] * F[u * N + v];
        }
      }
      f[i * N + j] = s;
    }
  }
  return f;
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

/** 简单 LCG，保证噪声预设每次一样 */
function noisePattern(): number[] {
  let seed = 20260706;
  const next = (): number => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const arr: number[] = [];
  for (let i = 0; i < N * N; i++) arr.push(next());
  return arr;
}

const SMILEY = [
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 1, 0, 0, 1, 0, 0,
  0, 0, 1, 0, 0, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 0, 0, 0, 0, 1, 0,
  0, 0, 1, 1, 1, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
];

const PRESETS: { name: string; make: () => number[]; noisy?: boolean }[] = [
  { name: "笑脸", make: () => SMILEY.slice() },
  {
    name: "十字",
    make: () => {
      const a = new Array<number>(N * N).fill(0);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (i === 3 || i === 4 || j === 3 || j === 4) a[i * N + j] = 1;
        }
      }
      return a;
    },
  },
  {
    name: "对角渐变",
    make: () => {
      const a: number[] = [];
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) a.push((i + j) / (2 * N - 2));
      }
      return a;
    },
  },
  {
    name: "竖条纹",
    make: () => {
      const a: number[] = [];
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) a.push(j % 2);
      }
      return a;
    },
  },
  { name: "随机噪声", make: noisePattern, noisy: true },
];

/** 创建一块 300×300 的方形画布（含 dpr 处理），返回 canvas 和 2D context */
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

function drawGridLines(ctx: CanvasRenderingContext2D): void {
  ctx.strokeStyle = "rgba(236,231,214,0.08)";
  ctx.lineWidth = 1;
  for (let t = 0; t <= N; t++) {
    const p = t * CELL;
    ctx.beginPath();
    ctx.moveTo(p, 0);
    ctx.lineTo(p, SIZE);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, p);
    ctx.lineTo(SIZE, p);
    ctx.stroke();
  }
}

/** 灰度图：黑板底色 → 粉笔白 插值 */
function drawImage(ctx: CanvasRenderingContext2D, img: number[]): void {
  ctx.clearRect(0, 0, SIZE, SIZE);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const v = Math.min(1, Math.max(0, img[i * N + j]));
      const r = Math.round(BOARD_DARK[0] + (CHALK[0] - BOARD_DARK[0]) * v);
      const g = Math.round(BOARD_DARK[1] + (CHALK[1] - BOARD_DARK[1]) * v);
      const b = Math.round(BOARD_DARK[2] + (CHALK[2] - BOARD_DARK[2]) * v);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(j * CELL, i * CELL, CELL, CELL);
    }
  }
  drawGridLines(ctx);
}

/** 系数图：正 → 金，负 → 玫瑰红，透明度 ∝ |c|/maxAbs */
function drawCoeffs(ctx: CanvasRenderingContext2D, F: number[], kept: boolean[]): void {
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.fillStyle = "#1b2420";
  ctx.fillRect(0, 0, SIZE, SIZE);
  let maxAbs = 0;
  for (const c of F) maxAbs = Math.max(maxAbs, Math.abs(c));
  if (maxAbs < 1e-12) maxAbs = 1;
  for (let u = 0; u < N; u++) {
    for (let v = 0; v < N; v++) {
      const c = F[u * N + v];
      const a = Math.abs(c) / maxAbs;
      if (a > 1e-4) {
        const [r, g, b] = c >= 0 ? GOLD_RGB : RED_RGB;
        ctx.fillStyle = `rgba(${r},${g},${b},${a.toFixed(3)})`;
        ctx.fillRect(v * CELL, u * CELL, CELL, CELL);
      }
      // 被保留的系数描一圈淡框，直观看到「留了哪 k 个」
      if (kept[u * N + v] && Math.abs(c) > 1e-12) {
        ctx.strokeStyle = "rgba(236,231,214,0.35)";
        ctx.lineWidth = 1;
        ctx.strokeRect(v * CELL + 1.5, u * CELL + 1.5, CELL - 3, CELL - 3);
      }
    }
  }
  drawGridLines(ctx);
}

export function mountCompress(root: HTMLElement): () => void {
  let img: number[] = SMILEY.slice();
  let k = 10;
  let erase = false; // 画笔 / 橡皮
  let noisyPreset = false;

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "基变换与图像压缩（DCT）"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 31。一张 8×8 灰度图就是一个 <b>64 维向量</b>：" +
        "在像素基下，每个分量只描述一个点；换到 <b>DCT 余弦基</b>后，" +
        "前几个分量就描述了整体明暗和大块结构，小系数扔掉几乎不损失——" +
        "这正是 JPEG 压缩的核心想法。",
    ),
  );

  const brushRow = el("div", "row");
  brushRow.appendChild(el("span", "", "工具"));
  const brushBtn = el("button", "btn", "画笔（白）");
  const eraseBtn = el("button", "btn", "橡皮（黑）");
  const syncTool = (): void => {
    brushBtn.classList.toggle("primary", !erase);
    eraseBtn.classList.toggle("primary", erase);
  };
  brushBtn.onclick = () => {
    erase = false;
    syncTool();
  };
  eraseBtn.onclick = () => {
    erase = true;
    syncTool();
  };
  syncTool();
  brushRow.appendChild(brushBtn);
  brushRow.appendChild(eraseBtn);
  panel.appendChild(brushRow);

  const sel = el("select");
  sel.appendChild(el("option", "", "选择预设图案…"));
  PRESETS.forEach((p, i) => {
    const o = el("option", "", p.name);
    o.value = String(i);
    sel.appendChild(o);
  });
  sel.onchange = () => {
    const p = PRESETS[Number(sel.value)];
    if (!p) return;
    img = p.make();
    noisyPreset = !!p.noisy;
    render();
  };
  panel.appendChild(sel);

  const slider = sliderRow("保留系数 k", 1, 64, 1, k, (v) => {
    k = Math.round(v);
    render();
  });
  panel.appendChild(slider.root);

  const readout = el("div", "readout");
  panel.appendChild(readout);

  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 三块黑板 ----
  const row = el("div", "canvas-row");

  const colA = el("div", "canvas-col");
  colA.appendChild(el("h3", "", "原图（点击/拖动作画）"));
  const boardA = makeBoard();
  colA.appendChild(boardA.canvas);

  const colB = el("div", "canvas-col");
  colB.appendChild(el("h3", "", "DCT 系数"));
  const boardB = makeBoard();
  colB.appendChild(boardB.canvas);

  const colC = el("div", "canvas-col");
  colC.appendChild(el("h3", "", `重构（保留前 k 个）`));
  const boardC = makeBoard();
  colC.appendChild(boardC.canvas);

  row.appendChild(colA);
  row.appendChild(colB);
  row.appendChild(colC);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(row);
  root.appendChild(moduleEl);

  // ---- 作画交互 ----
  let painting = false;
  const paintAt = (ev: PointerEvent): void => {
    const rect = boardA.canvas.getBoundingClientRect();
    const j = Math.floor((ev.clientX - rect.left) / CELL);
    const i = Math.floor((ev.clientY - rect.top) / CELL);
    if (i < 0 || i >= N || j < 0 || j >= N) return;
    const value = erase || ev.shiftKey ? 0 : 1;
    if (img[i * N + j] !== value) {
      img[i * N + j] = value;
      noisyPreset = false;
      render();
    }
  };
  const onDown = (ev: PointerEvent): void => {
    painting = true;
    boardA.canvas.setPointerCapture(ev.pointerId);
    paintAt(ev);
  };
  const onMove = (ev: PointerEvent): void => {
    if (painting) paintAt(ev);
  };
  const onUp = (): void => {
    painting = false;
  };
  boardA.canvas.addEventListener("pointerdown", onDown);
  boardA.canvas.addEventListener("pointermove", onMove);
  boardA.canvas.addEventListener("pointerup", onUp);
  boardA.canvas.addEventListener("pointercancel", onUp);
  boardA.canvas.style.cursor = "crosshair";

  function render(): void {
    const F = dct2(img);

    // 按 |c| 从大到小保留前 k 个
    const order = F.map((c, idx) => ({ idx, abs: Math.abs(c) })).sort((a, b) => b.abs - a.abs);
    const kept = new Array<boolean>(N * N).fill(false);
    for (let t = 0; t < k; t++) kept[order[t].idx] = true;
    const Fk = F.map((c, idx) => (kept[idx] ? c : 0));

    // 逆变换重构，截断到 [0,1]
    const rec = idct2(Fk).map((v) => Math.min(1, Math.max(0, v)));

    // 误差与能量
    let err = 0;
    for (let idx = 0; idx < N * N; idx++) err += (img[idx] - rec[idx]) ** 2;
    let eAll = 0;
    let eKept = 0;
    for (let idx = 0; idx < N * N; idx++) {
      eAll += F[idx] ** 2;
      if (kept[idx]) eKept += F[idx] ** 2;
    }
    const energy = eAll < 1e-12 ? 1 : eKept / eAll;

    drawImage(boardA.ctx, img);
    drawCoeffs(boardB.ctx, F, kept);
    drawImage(boardC.ctx, rec);

    readout.textContent = [
      `非零系数：${k} / 64`,
      `压缩比：64/${k} ≈ ${(64 / k).toPrecision(3)} : 1`,
      `重构误差 ‖原图 − 重构‖² = ${err.toPrecision(3)}`,
      `能量占比：Σc²(保留) / Σc²(全部) = ${(energy * 100).toFixed(1)}%`,
    ].join("\n");

    const base = "换一组基，信息集中到少数几个系数上——这就是压缩。";
    if (noisyPreset && energy < 0.9) {
      status.innerHTML =
        base +
        " 但<b>噪声</b>的能量摊在所有 64 个系数上，不往低频集中——DCT 也压不动它。";
    } else if (energy > 0.9 && k < 64) {
      status.innerHTML =
        base +
        ` <span class="status-ok">✓ 只留 ${k} 个系数就保住了 ${(energy * 100).toFixed(1)}% 的能量，图案仍可辨认。</span>`;
    } else {
      status.textContent = base + " 试着减小 k，看看多少个系数就足够认出图案。";
    }
  }

  render();

  return () => {
    boardA.canvas.removeEventListener("pointerdown", onDown);
    boardA.canvas.removeEventListener("pointermove", onMove);
    boardA.canvas.removeEventListener("pointerup", onUp);
    boardA.canvas.removeEventListener("pointercancel", onUp);
  };
}
