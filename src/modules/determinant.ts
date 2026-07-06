import { det, I, type Mat, type Vec } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt, matrixInput } from "../ui";

export function mountDeterminant(root: HTMLElement): () => void {
  let A: Mat = [2, -1, 1, 1];
  let dragTarget: 0 | 1 | null = null; // 0 = c₁，1 = c₂

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "行列式与有向面积"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 18–20。矩阵 A 的两列 c₁、c₂ 张成一个平行四边形，" +
        "det A 就是它的<b>有向面积</b>：|det| 是面积大小，符号记录定向" +
        "（c₁ 逆时针转到 c₂ 为正）。拖动两支箭头的尖端改变 A，" +
        "或点击按钮验证行列式的三条基本性质。",
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

  const status = el("div", "status");

  /** 应用性质操作：改 A、回写输入框、解释 det 如何变化 */
  function applyOp(next: Mat, explain: (d0: number, d1: number) => string): void {
    const d0 = det(A);
    A = next;
    const d1 = det(A);
    mi.set(A);
    status.innerHTML = explain(d0, d1);
    render();
  }

  const swapBtn = el("button", "btn", "交换两列");
  swapBtn.onclick = () =>
    applyOp([A[1], A[0], A[3], A[2]], (d0, d1) =>
      `交换两列：det 从 ${fmt(d0)} 变为 ${fmt(d1)}` +
      `<span class="status-ok">——变号</span>，定向翻转，面积不变。`);

  const scaleBtn = el("button", "btn", "c₁ ×2");
  scaleBtn.onclick = () =>
    applyOp([2 * A[0], A[1], 2 * A[2], A[3]], (d0, d1) =>
      `c₁ 拉长为 2 倍：det 从 ${fmt(d0)} 变为 ${fmt(d1)}` +
      `<span class="status-ok">——加倍</span>，行列式对每一列都是线性的。`);

  const addBtn = el("button", "btn", "c₂ += c₁");
  addBtn.onclick = () =>
    applyOp([A[0], A[1] + A[0], A[2], A[3] + A[2]], (d0, d1) =>
      `把 c₁ 加到 c₂ 上：det 从 ${fmt(d0)} 变为 ${fmt(d1)}` +
      `<span class="status-ok">——不变</span>。平行四边形被剪切但底和高都没变，` +
      `所以消元不改变行列式。`);

  const resetBtn = el("button", "btn", "重置为单位阵");
  resetBtn.onclick = () =>
    applyOp([...I], () => `A = I：单位正方形，det I = 1。`);

  const btnRow1 = el("div", "row");
  btnRow1.appendChild(swapBtn);
  btnRow1.appendChild(scaleBtn);
  const btnRow2 = el("div", "row");
  btnRow2.appendChild(addBtn);
  btnRow2.appendChild(resetBtn);
  panel.appendChild(btnRow1);
  panel.appendChild(btnRow2);

  const readout = el("div", "readout");
  panel.appendChild(readout);
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
    if (dragTarget === null) {
      // 按下时选取离指针较近的那个列向量尖端，拖动期间锁定
      const d1 = Math.hypot(v.x - A[0], v.y - A[2]);
      const d2 = Math.hypot(v.x - A[1], v.y - A[3]);
      dragTarget = d1 <= d2 ? 0 : 1;
    }
    A = dragTarget === 0 ? [v.x, A[1], v.y, A[3]] : [A[0], v.x, A[2], v.y];
    mi.set(A);
    render();
  });
  const endDrag = (): void => {
    dragTarget = null;
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  function render(): void {
    const o: Vec = { x: 0, y: 0 };
    const c1: Vec = { x: A[0], y: A[2] };
    const c2: Vec = { x: A[1], y: A[3] };
    const sum: Vec = { x: c1.x + c2.x, y: c1.y + c2.y };
    const d = det(A);

    plane.clear();
    plane.grid();
    plane.axes();

    // 平行四边形：det 的符号决定填色
    const fill = d >= 0 ? COLORS.gold : COLORS.red;
    plane.fillPoly([o, c1, sum, c2], fill, 0.3);

    plane.arrow(o, c1, COLORS.green, 3, "c₁");
    plane.arrow(o, c2, COLORS.red, 3, "c₂");

    // 形心处标出有向面积
    if (Math.hypot(sum.x, sum.y) > 1e-9 || Math.abs(d) > 1e-9) {
      const [cx, cy] = plane.toScreen({ x: sum.x / 2, y: sum.y / 2 });
      const ctx = plane.ctx;
      ctx.save();
      ctx.fillStyle = COLORS.chalk;
      ctx.font = "italic 15px Cambria, Georgia, 'Times New Roman', serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(fmt(d), cx, cy);
      ctx.restore();
    }

    const [a, b, c, dd] = A;
    let signText: string;
    if (d > 1e-10) {
      signText = "det > 0：正定向（c₁ 逆时针转到 c₂），金色填充";
    } else if (d < -1e-10) {
      signText = "det < 0：定向翻转（c₁ 顺时针转到 c₂），玫瑰色填充";
    } else {
      signText = "det = 0：两列共线，平行四边形被压扁成一条线段";
    }
    readout.textContent =
      `det A = ad − bc\n` +
      `      = (${fmt(a)})(${fmt(dd)}) − (${fmt(b)})(${fmt(c)}) = ${fmt(d)}\n\n` +
      `|det| = ${fmt(Math.abs(d))} = 平行四边形面积\n` +
      signText;
  }

  status.textContent = "拖动绿色 c₁ 或玫瑰色 c₂ 的尖端，观察有向面积如何变化。";

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
