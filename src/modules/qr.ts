import { normalize, type Vec } from "../math";
import { Plane } from "../plane";
import { COLORS } from "../theme";
import { el, fmt } from "../ui";

const PRESETS: { name: string; a: Vec; b: Vec }[] = [
  { name: "一般情形", a: { x: 3, y: 1 }, b: { x: 1, y: 2.5 } },
  { name: "夹角很小（数值上危险）", a: { x: 3, y: 0.5 }, b: { x: 2.8, y: 0.9 } },
  { name: "已经正交", a: { x: 2, y: 1 }, b: { x: -1.2, y: 2.4 } },
  { name: "共线（Gram-Schmidt 失败）", a: { x: 2, y: 1 }, b: { x: 3, y: 1.5 } },
];

const STEP_DESC = [
  "第 0 步：原始的两列 a、b —— 不正交也不单位化",
  "第 1 步：归一化 q₁ = a / ‖a‖，作为第一个正交基",
  "第 2 步：从 b 里减掉沿 q₁ 的分量：B = b − (q₁ᵀb)q₁，剩下的 B 与 q₁ 垂直",
  "第 3 步：归一化 q₂ = B / ‖B‖，得到标准正交基 Q = [q₁ q₂]，A = QR",
];

export function mountQr(root: HTMLElement): () => void {
  let a: Vec = { x: 3, y: 1 };
  let b: Vec = { x: 1, y: 2.5 };
  let step = 0;

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "Gram-Schmidt 与 A = QR"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 17。把一组“歪的”列向量改造成标准正交基：" +
        "先归一化 a 得 <b>q₁</b>，再从 b 中<b>减掉它在 q₁ 上的投影</b>（这就是上一讲的投影！），" +
        "剩下的部分归一化得 q₂。改造过程记在上三角 R 里，所以 A = QR。" +
        "拖动 a、b 观察每一步。",
    ),
  );

  const stepBtnsRow = el("div", "row");
  const stepBtns: HTMLButtonElement[] = [];
  for (let i = 0; i <= 3; i++) {
    const btn = el("button", "btn", `${i}`);
    btn.onclick = () => {
      step = i;
      render();
    };
    stepBtns.push(btn);
    stepBtnsRow.appendChild(btn);
  }
  const stepLabel = el("span", "", "步骤");
  stepBtnsRow.prepend(stepLabel);
  panel.appendChild(stepBtnsRow);

  const sel = el("select");
  sel.appendChild(el("option", "", "选择预设向量组…"));
  PRESETS.forEach((p, i) => {
    const o = el("option", "", p.name);
    o.value = String(i);
    sel.appendChild(o);
  });
  sel.onchange = () => {
    const p = PRESETS[Number(sel.value)];
    if (!p) return;
    a = { ...p.a };
    b = { ...p.b };
    render();
  };
  panel.appendChild(sel);

  const readout = el("div", "readout");
  panel.appendChild(readout);

  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 黑板 ----
  const canvasCol = el("div", "canvas-col");
  canvasCol.appendChild(el("h3", "", "拖动 a、b 的尖端；淡圆是单位圆——q₁、q₂ 都落在它上面"));
  const canvas = el("canvas", "plane");
  canvasCol.appendChild(canvas);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(canvasCol);
  root.appendChild(moduleEl);

  const plane = new Plane(canvas);
  plane.scale = 80;
  plane.onRedraw = () => render();

  let dragTarget: "a" | "b" = "a";
  plane.attachDrag((v) => {
    if (Math.hypot(v.x, v.y) < 0.15) return;
    // 按下时锁定较近者：attachDrag 每次回调都会进来，用距离持续判断会“跳换”，
    // 所以只在指针远离两者时保持原目标，靠近某个尖端时切换
    const da = Math.hypot(v.x - a.x, v.y - a.y);
    const db = Math.hypot(v.x - b.x, v.y - b.y);
    if (Math.min(da, db) < 0.6) dragTarget = da <= db ? "a" : "b";
    if (dragTarget === "a") a = v;
    else b = v;
    render();
  });

  function unitCircle(): void {
    const ctx = plane.ctx;
    const [cx, cy] = plane.toScreen({ x: 0, y: 0 });
    ctx.save();
    ctx.strokeStyle = COLORS.dim;
    ctx.setLineDash([3, 6]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, plane.scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function rightAngle(q1: Vec, q2: Vec): void {
    const s = 0.22;
    const p1 = { x: q1.x * s, y: q1.y * s };
    const p2 = { x: q2.x * s, y: q2.y * s };
    const pc = { x: p1.x + p2.x, y: p1.y + p2.y };
    const ctx = plane.ctx;
    const [x1, y1] = plane.toScreen(p1);
    const [xc, yc] = plane.toScreen(pc);
    const [x2, y2] = plane.toScreen(p2);
    ctx.strokeStyle = COLORS.gold;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(xc, yc);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  function render(): void {
    const o = { x: 0, y: 0 };
    const na = Math.hypot(a.x, a.y);
    const q1 = normalize(a);
    const r12 = q1.x * b.x + q1.y * b.y; // q₁ᵀb
    const p: Vec = { x: q1.x * r12, y: q1.y * r12 };
    const B: Vec = { x: b.x - p.x, y: b.y - p.y };
    const nB = Math.hypot(B.x, B.y);
    const degenerate = nB < 1e-6;
    const q2 = degenerate ? { x: 0, y: 0 } : normalize(B);

    stepBtns.forEach((btn, i) => btn.classList.toggle("primary", i === step));

    plane.clear();
    plane.grid();
    plane.axes();
    unitCircle();

    // 原始向量：step 0 是主角，之后淡出为参考
    const rawWidth = step === 0 ? 3 : 1.5;
    const rawColorA = step === 0 ? COLORS.green : COLORS.dim;
    const rawColorB = step === 0 ? COLORS.chalk : COLORS.dim;
    plane.arrow(o, a, rawColorA, rawWidth, "a");
    plane.arrow(o, b, rawColorB, rawWidth, "b");

    if (step >= 1) {
      plane.arrow(o, q1, COLORS.green, 3.5, "q₁");
    }
    if (step >= 2 && !degenerate) {
      // 投影与减法：p 沿 q₁，B 从 p 的尖端指到 b 的尖端
      plane.infLine(o, q1, COLORS.dim, 1, [4, 5]);
      plane.arrow(o, p, COLORS.gold, 2.5, "(q₁ᵀb)q₁");
      plane.arrow(p, b, COLORS.red, 3, "B");
    }
    if (step >= 3 && !degenerate) {
      plane.arrow(o, q2, COLORS.red, 3.5, "q₂");
      rightAngle(q1, q2);
    }

    const rows = [
      `a = (${fmt(a.x)}, ${fmt(a.y)})   b = (${fmt(b.x)}, ${fmt(b.y)})`,
      "",
      `q₁ = a/‖a‖ = (${fmt(q1.x)}, ${fmt(q1.y)})`,
      `r₁₂ = q₁ᵀb = ${fmt(r12)}`,
      `B = b − r₁₂·q₁ = (${fmt(B.x)}, ${fmt(B.y)})`,
      degenerate ? "q₂ 不存在（B = 0）" : `q₂ = B/‖B‖ = (${fmt(q2.x)}, ${fmt(q2.y)})`,
      "",
      `R = [ ${fmt(na)}  ${fmt(r12)} ]`,
      `    [ 0     ${fmt(nB)} ]`,
      degenerate ? "" : `验证 q₁ᵀq₂ = ${fmt(q1.x * q2.x + q1.y * q2.y)} ≈ 0`,
    ];
    readout.textContent = rows.join("\n");

    if (degenerate) {
      status.innerHTML =
        `<b>a、b 共线</b>：减掉投影后 B = 0，没有第二个方向可用——` +
        `Gram-Schmidt 要求各列线性无关（这正是 R 可逆的条件）。`;
    } else if (step === 3) {
      status.innerHTML =
        `<span class="status-ok">✓ Q = [q₁ q₂] 标准正交，A = QR</span>` +
        `　R 记录了改造用掉的系数：对角线是长度，右上角是投影系数。`;
    } else {
      status.textContent = STEP_DESC[step];
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
