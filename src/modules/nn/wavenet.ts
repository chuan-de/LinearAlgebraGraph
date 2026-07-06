import { COLORS } from "../../theme";
import { el, sliderRow } from "../../ui";

/* ============================================================
   层级感受野（Karpathy《Zero to Hero》视频⑥：makemore Part 5 — WaveNet）
   纯结构可视化（不训练）：把「一口气拼接 n 个字符压进一层」
   换成「两两融合、逐层升高」的二叉树——上下文每层翻倍，
   但每层只做一次小小的融合。
   ============================================================ */

const W = 800; // 画布 CSS 宽
const H = 540; // 画布 CSS 高
const MARGIN = 50; // 左右留白
const Y_BOTTOM = 462; // 输入行中心 y
const Y_TOP = 96; // 输出节点中心 y

/** 示意超参数：嵌入维度 d 与平坦 MLP 的隐层宽度 H */
const DIM = 24;
const HID = 128;

/** 示例名字（前补 '.' 作为起始符），截取/填充到宽度 n */
const SAMPLE = ".deandre";
function contextChars(n: number): string[] {
  let s = SAMPLE;
  if (s.length >= n) s = s.slice(-n);
  else s = ".".repeat(n - s.length) + s;
  return s.split("");
}

interface NodeBox {
  lvl: number; // 0 = 输入层，L = 顶端输出
  idx: number;
  x: number; // 中心 x
  y: number; // 中心 y
  w: number;
  h: number;
}

function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.arcTo(x + w, y, x + w, y + rr, rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.arcTo(x + w, y + h, x + w - rr, y + h, rr);
  ctx.lineTo(x + rr, y + h);
  ctx.arcTo(x, y + h, x, y + h - rr, rr);
  ctx.lineTo(x, y + rr);
  ctx.arcTo(x, y, x + rr, y, rr);
  ctx.closePath();
}

export function mountWavenet(root: HTMLElement): () => void {
  let L = 3; // 层数（树高），输入长度 = 2^L
  let flat = false; // 对比：平坦 MLP
  let pinned: { lvl: number; idx: number } | null = null; // 点击固定的节点
  let hover: { lvl: number; idx: number } | null = null; // 悬停节点
  let nodes: NodeBox[] = []; // 每次渲染后重建，用于命中检测

  // ---- 控制面板（讲义纸） ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "层级感受野（WaveNet 树状融合）"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应视频⑥（makemore Part 5: WaveNet）。想看更长的上下文，" +
        "<b>平坦 MLP</b> 只能把所有字符一次性拼起来压进第一层——" +
        "参数随上下文长度爆炸，而且所有信息被一口气挤扁。" +
        "<b>WaveNet 式层级结构</b>每层只融合<b>相邻两块</b>，" +
        "信息逐层慢慢压缩，感受野（能看见的上下文）每升一层就<b>翻倍</b>。" +
        "把鼠标移到任意节点上，看看它的感受野覆盖哪些字符。",
    ),
  );

  const slider = sliderRow("层数 L", 2, 4, 1, L, (v) => {
    L = Math.round(v);
    pinned = null;
    hover = null;
    render();
  });
  panel.appendChild(slider.root);

  const checkLabel = el("label", "check");
  const checkbox = el("input");
  checkbox.type = "checkbox";
  checkbox.onchange = () => {
    flat = checkbox.checked;
    hover = null;
    render();
  };
  checkLabel.appendChild(checkbox);
  checkLabel.appendChild(el("span", "", "对比：平坦 MLP（一层挤压全部上下文）"));
  panel.appendChild(checkLabel);

  const readout = el("div", "readout");
  panel.appendChild(readout);

  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 黑板画布 ----
  const col = el("div", "canvas-col");
  col.appendChild(el("h3", "", "两两融合的二叉树：每层感受野翻倍"));
  const canvas = el("canvas", "plane");
  canvas.style.width = `${W}px`;
  canvas.style.height = `${H}px`;
  canvas.style.cursor = "default";
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  col.appendChild(canvas);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(col);
  root.appendChild(moduleEl);

  // ---- 几何布局 ----
  function layoutTree(): NodeBox[] {
    const n = 2 ** L;
    const slotW = (W - 2 * MARGIN) / n;
    const boxes: NodeBox[] = [];
    for (let lvl = 0; lvl <= L; lvl++) {
      const count = 2 ** (L - lvl);
      const span = slotW * 2 ** lvl; // 该层每个节点占据的水平跨度
      const y = Y_BOTTOM - (lvl * (Y_BOTTOM - Y_TOP)) / L;
      const w = lvl === 0 ? Math.min(44, slotW - 6) : Math.min(56, span - 12);
      const h = lvl === 0 ? 36 : 32;
      for (let idx = 0; idx < count; idx++) {
        const x = MARGIN + (idx + 0.5) * span;
        boxes.push({ lvl, idx, x, y, w, h });
      }
    }
    return boxes;
  }

  function findNode(lvl: number, idx: number): NodeBox {
    return nodes.find((b) => b.lvl === lvl && b.idx === idx)!;
  }

  /** 节点 (lvl, idx) 是否在选中节点的子树里 */
  function inSubtree(sel: { lvl: number; idx: number }, lvl: number, idx: number): boolean {
    return lvl <= sel.lvl && Math.floor(idx / 2 ** (sel.lvl - lvl)) === sel.idx;
  }

  function drawNodeBox(b: NodeBox, highlighted: boolean): void {
    roundRectPath(ctx, b.x - b.w / 2, b.y - b.h / 2, b.w, b.h, 7);
    if (highlighted) {
      ctx.fillStyle = "rgba(230,200,96,0.14)";
      ctx.fill();
    }
    ctx.strokeStyle = highlighted ? COLORS.gold : COLORS.chalk;
    ctx.lineWidth = highlighted ? 2 : 1.4;
    ctx.stroke();
  }

  function drawTree(sel: { lvl: number; idx: number }): void {
    const chars = contextChars(2 ** L);

    // 层与层之间的连线：淡粉笔在下，金色高亮在上
    for (const pass of [0, 1] as const) {
      for (let lvl = 1; lvl <= L; lvl++) {
        const count = 2 ** (L - lvl);
        for (let idx = 0; idx < count; idx++) {
          const parent = findNode(lvl, idx);
          const hi = inSubtree(sel, lvl, idx);
          if ((pass === 0) === hi) continue; // 第 0 遍画普通线，第 1 遍画高亮线
          for (const c of [2 * idx, 2 * idx + 1]) {
            const child = findNode(lvl - 1, c);
            ctx.beginPath();
            ctx.moveTo(child.x, child.y - child.h / 2);
            ctx.lineTo(parent.x, parent.y + parent.h / 2);
            ctx.strokeStyle = hi ? COLORS.gold : COLORS.dim;
            ctx.lineWidth = hi ? 2 : 1;
            ctx.stroke();
          }
        }
      }
    }

    // 节点
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const b of nodes) {
      const hi = inSubtree(sel, b.lvl, b.idx);
      drawNodeBox(b, hi);
      if (b.lvl === 0) {
        // 输入方块里的字符
        ctx.fillStyle = hi ? COLORS.gold : COLORS.chalk;
        ctx.font = "16px Consolas, monospace";
        ctx.fillText(chars[b.idx], b.x, b.y + 1);
        // 被选中节点覆盖的输入：额外加一圈金框
        if (hi) {
          roundRectPath(ctx, b.x - b.w / 2 - 3.5, b.y - b.h / 2 - 3.5, b.w + 7, b.h + 7, 9);
          ctx.strokeStyle = COLORS.gold;
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
      } else if (b.lvl === L) {
        ctx.fillStyle = hi ? COLORS.gold : COLORS.chalk;
        ctx.font = "17px Consolas, monospace";
        ctx.fillText("?", b.x, b.y + 1);
      } else {
        // 中间融合节点：画一个 ⊕ 表示「两块融合」
        ctx.fillStyle = hi ? COLORS.gold : COLORS.tick;
        ctx.font = "15px Consolas, monospace";
        ctx.fillText("⊕", b.x, b.y + 1);
      }
    }

    // 顶端标注与左侧每层节点数
    const top = findNode(L, 0);
    ctx.fillStyle = COLORS.chalk;
    ctx.font = "14px 'Segoe UI', sans-serif";
    ctx.fillText("预测下一个字符", top.x, top.y - top.h / 2 - 16);
    ctx.textAlign = "left";
    ctx.fillStyle = COLORS.tick;
    ctx.font = "11.5px Consolas, monospace";
    for (let lvl = 0; lvl <= L; lvl++) {
      const y = Y_BOTTOM - (lvl * (Y_BOTTOM - Y_TOP)) / L;
      const label = lvl === 0 ? `输入 ×${2 ** L}` : `×${2 ** (L - lvl)}`;
      ctx.fillText(label, 6, y + 4);
    }
  }

  function drawFlat(): void {
    const n = 2 ** L;
    const chars = contextChars(n);
    const slotW = (W - 2 * MARGIN) / n;
    const inW = Math.min(44, slotW - 6);

    // 又宽又扁的隐层节点
    const hidW = Math.min(320, W - 2 * MARGIN - 60);
    const hidH = 40;
    const hidY = (Y_BOTTOM + Y_TOP) / 2 + 20;

    // 输入 → 隐层：n 条线全部汇聚
    for (let i = 0; i < n; i++) {
      const x = MARGIN + (i + 0.5) * slotW;
      const tx = W / 2 - hidW / 2 + ((i + 0.5) / n) * hidW;
      ctx.beginPath();
      ctx.moveTo(x, Y_BOTTOM - 18);
      ctx.lineTo(tx, hidY + hidH / 2);
      ctx.strokeStyle = COLORS.dim;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    // 隐层 → 输出
    ctx.beginPath();
    ctx.moveTo(W / 2, hidY - hidH / 2);
    ctx.lineTo(W / 2, Y_TOP + 16);
    ctx.strokeStyle = COLORS.dim;
    ctx.lineWidth = 1;
    ctx.stroke();

    // 输入方块
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) {
      const x = MARGIN + (i + 0.5) * slotW;
      roundRectPath(ctx, x - inW / 2, Y_BOTTOM - 18, inW, 36, 7);
      ctx.strokeStyle = COLORS.chalk;
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.fillStyle = COLORS.chalk;
      ctx.font = "16px Consolas, monospace";
      ctx.fillText(chars[i], x, Y_BOTTOM + 1);
    }

    // 又宽又扁的隐层节点（玫瑰粉笔强调「挤压」）
    roundRectPath(ctx, W / 2 - hidW / 2, hidY - hidH / 2, hidW, hidH, 9);
    ctx.fillStyle = "rgba(232,138,156,0.10)";
    ctx.fill();
    ctx.strokeStyle = COLORS.red;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = COLORS.red;
    ctx.font = "14px 'Segoe UI', sans-serif";
    ctx.fillText("一层挤压全部上下文", W / 2, hidY + 1);

    // 输出节点
    roundRectPath(ctx, W / 2 - 28, Y_TOP - 16, 56, 32, 7);
    ctx.strokeStyle = COLORS.chalk;
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.fillStyle = COLORS.chalk;
    ctx.font = "17px Consolas, monospace";
    ctx.fillText("?", W / 2, Y_TOP + 1);
    ctx.font = "14px 'Segoe UI', sans-serif";
    ctx.fillText("预测下一个字符", W / 2, Y_TOP - 32);

    // 参数标注
    ctx.fillStyle = COLORS.tick;
    ctx.font = "12px Consolas, monospace";
    ctx.fillText(`第一层参数 ∝ (${n}×${DIM})×${HID} = ${n * DIM * HID}`, W / 2, hidY + hidH / 2 + 20);
  }

  function selection(): { lvl: number; idx: number } {
    return hover ?? pinned ?? { lvl: L, idx: 0 };
  }

  function render(): void {
    ctx.clearRect(0, 0, W, H);
    nodes = layoutTree();

    if (flat) {
      drawFlat();
    } else {
      drawTree(selection());
    }

    // ---- readout ----
    const n = 2 ** L;
    const seq: string[] = [];
    for (let lvl = 0; lvl <= L; lvl++) seq.push(String(2 ** (L - lvl)));
    const flatParams = n * DIM * HID;
    const treePerLayer = 2 * DIM * DIM;
    readout.textContent = [
      `层数 L = ${L}，上下文长度 = 2^${L} = ${n}`,
      `每层节点数：${seq.join(" → ")}`,
      `平坦 MLP 第一层 ∝ (${n}·d)·H = (${n}×${DIM})×${HID} = ${flatParams}`,
      `树状每层融合 ∝ (2·d)·d = (2×${DIM})×${DIM} = ${treePerLayer}（共 ${L} 层 ≈ ${L * treePerLayer}）`,
      `（示意值 d = ${DIM}，H = ${HID}）`,
    ].join("\n");

    // ---- status ----
    if (flat) {
      status.innerHTML =
        `平坦 MLP：这个隐层节点一口气看全部 <b>${n}</b> 个字符，` +
        `第一层参数 ${flatParams} 个——上下文每翻倍，参数也跟着翻倍。`;
    } else {
      const sel = selection();
      const k = 2 ** sel.lvl;
      const chars = contextChars(n)
        .slice(sel.idx * k, (sel.idx + 1) * k)
        .join("");
      status.innerHTML =
        `这个节点看得见 <b>${k}</b> 个字符：「${chars}」` +
        (sel.lvl === L
          ? ` —— 顶端输出的感受野覆盖全部输入。`
          : ` —— 每升一层，感受野翻倍。`);
    }
  }

  // ---- 交互：hover / 点击 ----
  function pick(ev: PointerEvent): { lvl: number; idx: number } | null {
    const rect = canvas.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    const py = ((ev.clientY - rect.top) / rect.height) * H;
    for (const b of nodes) {
      if (Math.abs(px - b.x) <= b.w / 2 + 4 && Math.abs(py - b.y) <= b.h / 2 + 4) {
        return { lvl: b.lvl, idx: b.idx };
      }
    }
    return null;
  }

  const onMove = (ev: PointerEvent): void => {
    if (flat) return;
    const hit = pick(ev);
    canvas.style.cursor = hit ? "pointer" : "default";
    const changed =
      (hit === null) !== (hover === null) ||
      (hit !== null && hover !== null && (hit.lvl !== hover.lvl || hit.idx !== hover.idx));
    if (changed) {
      hover = hit;
      render();
    }
  };
  const onLeave = (): void => {
    if (hover !== null) {
      hover = null;
      render();
    }
  };
  const onClick = (ev: PointerEvent): void => {
    if (flat) return;
    const hit = pick(ev);
    if (hit) {
      pinned = hit;
      hover = null;
      render();
    }
  };
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);
  canvas.addEventListener("pointerdown", onClick);

  render();

  return () => {
    canvas.removeEventListener("pointermove", onMove);
    canvas.removeEventListener("pointerleave", onLeave);
    canvas.removeEventListener("pointerdown", onClick);
  };
}
