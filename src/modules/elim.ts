import { el, fmt } from "../ui";

type M3 = number[][];

interface Step {
  desc: string;
  E: M3 | null;
  A: M3;
  pivot?: [number, number];
  target?: [number, number];
}

const PRESETS: { name: string; m: M3 }[] = [
  { name: "Strang 经典例题", m: [[1, 2, 1], [3, 8, 1], [0, 4, 1]] },
  { name: "需要行交换（主元为 0）", m: [[0, 2, 1], [1, 1, 1], [2, 3, 4]] },
  { name: "奇异矩阵（秩 2，缺主元）", m: [[1, 2, 3], [2, 4, 6], [1, 1, 1]] },
  { name: "对角占优", m: [[4, 1, 0], [1, 4, 1], [0, 1, 4]] },
];

const clone = (m: M3): M3 => m.map((r) => [...r]);
const identity = (): M3 => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

interface ElimResult {
  steps: Step[];
  U: M3;
  L: M3;
  P: M3;
  swapped: boolean;
  pivots: number[];
}

function eliminate(A0: M3, record: boolean): ElimResult {
  const A = clone(A0);
  const L = identity();
  const P = identity();
  let swapped = false;
  const steps: Step[] = record ? [{ desc: "初始矩阵 A", E: null, A: clone(A) }] : [];

  for (let col = 0; col < 3; col++) {
    if (Math.abs(A[col][col]) < 1e-10) {
      let r = -1;
      for (let k = col + 1; k < 3; k++) {
        if (Math.abs(A[k][col]) > 1e-10) {
          r = k;
          break;
        }
      }
      if (r === -1) {
        if (record) {
          steps.push({
            desc: `第 ${col + 1} 列找不到非零主元 —— <span class="op">矩阵奇异</span>，这一列没有主元，继续看下一列`,
            E: null,
            A: clone(A),
            pivot: [col, col],
          });
        }
        continue;
      }
      [A[col], A[r]] = [A[r], A[col]];
      [P[col], P[r]] = [P[r], P[col]];
      // 已记录的乘数跟着行一起换，保证 PA = LU 成立
      for (let j = 0; j < col; j++) [L[col][j], L[r][j]] = [L[r][j], L[col][j]];
      swapped = true;
      if (record) {
        const Pm = identity();
        [Pm[col], Pm[r]] = [Pm[r], Pm[col]];
        steps.push({
          desc: `位置 (${col + 1},${col + 1}) 的主元是 0 —— <span class="op">交换 行${col + 1} ↔ 行${r + 1}</span>（左乘置换矩阵 P）`,
          E: Pm,
          A: clone(A),
          pivot: [col, col],
        });
      }
    }
    for (let row = col + 1; row < 3; row++) {
      const m = A[row][col] / A[col][col];
      if (Math.abs(m) < 1e-12) continue;
      for (let j = 0; j < 3; j++) A[row][j] -= m * A[col][j];
      A[row][col] = 0;
      L[row][col] = m;
      if (record) {
        const E = identity();
        E[row][col] = -m;
        steps.push({
          desc: `<span class="op">行${row + 1} ← 行${row + 1} − (${fmt(m)})·行${col + 1}</span>，消去位置 (${row + 1},${col + 1})，乘数 ℓ${row + 1}${col + 1} = ${fmt(m)}`,
          E,
          A: clone(A),
          pivot: [col, col],
          target: [row, col],
        });
      }
    }
  }
  const pivots = [0, 1, 2].map((i) => A[i][i]).filter((v) => Math.abs(v) > 1e-10);
  return { steps, U: A, L, P, swapped, pivots };
}

function mxEl(m: M3, pivot?: [number, number], target?: [number, number]): HTMLElement {
  const box = el("div", "mx");
  box.style.gridTemplateColumns = "repeat(3, auto)";
  m.forEach((row, i) =>
    row.forEach((v, j) => {
      const s = el("span", "", fmt(Math.abs(v) < 1e-10 ? 0 : v));
      if (pivot && i === pivot[0] && j === pivot[1]) s.classList.add("pivot");
      if (target && i === target[0] && j === target[1]) s.classList.add("target");
      if (!pivot && !target && j < i && Math.abs(v) < 1e-10) s.classList.add("zeroed");
      box.appendChild(s);
    }),
  );
  return box;
}

function matrix3Input(initial: M3, onChange: (m: M3) => void): { root: HTMLElement; set(m: M3): void } {
  const root = el("div", "matrix m3");
  const inputs: HTMLInputElement[] = [];
  const read = (): M3 =>
    [0, 1, 2].map((i) =>
      [0, 1, 2].map((j) => {
        const v = Number.parseFloat(inputs[i * 3 + j].value);
        return Number.isFinite(v) ? v : 0;
      }),
    );
  initial.flat().forEach((v) => {
    const inp = el("input");
    inp.type = "number";
    inp.step = "1";
    inp.value = String(v);
    inp.oninput = () => onChange(read());
    root.appendChild(inp);
    inputs.push(inp);
  });
  return {
    root,
    set(m: M3) {
      m.flat().forEach((v, i) => (inputs[i].value = fmt(v)));
    },
  };
}

export function mountElim(root: HTMLElement): () => void {
  let A: M3 = clone(PRESETS[0].m);
  let result = eliminate(A, true);
  let k = 0; // 当前步

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "消元与 A = LU"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Lecture 2–4。高斯消元 = 一串初等矩阵 E 从左边打到 A 上，把它化成上三角 U。" +
        "所有乘数 ℓᵢⱼ 装进下三角 L，就得到 <b>A = LU</b>。" +
        "遇到 0 主元需要交换行时，记成 <b>PA = LU</b>。金色 = 当前主元，玫瑰色 = 正被消去的位置。",
    ),
  );

  const mi = matrix3Input(A, (m) => {
    A = m;
    recompute();
  });
  const miRow = el("div", "row");
  miRow.appendChild(el("span", "", "A ="));
  miRow.appendChild(mi.root);
  panel.appendChild(miRow);

  const sel = el("select");
  sel.appendChild(el("option", "", "选择预设矩阵…"));
  PRESETS.forEach((p, i) => {
    const o = el("option", "", p.name);
    o.value = String(i);
    sel.appendChild(o);
  });
  sel.onchange = () => {
    const p = PRESETS[Number(sel.value)];
    if (!p) return;
    A = clone(p.m);
    mi.set(A);
    recompute();
  };
  panel.appendChild(sel);

  const prevBtn = el("button", "btn", "◀ 上一步");
  const nextBtn = el("button", "btn primary", "下一步 ▶");
  const endBtn = el("button", "btn", "⏭ 一步到底");
  const btnRow = el("div", "row");
  btnRow.appendChild(prevBtn);
  btnRow.appendChild(nextBtn);
  btnRow.appendChild(endBtn);
  panel.appendChild(btnRow);

  prevBtn.onclick = () => {
    k = Math.max(0, k - 1);
    render();
  };
  nextBtn.onclick = () => {
    k = Math.min(result.steps.length - 1, k + 1);
    render();
  };
  endBtn.onclick = () => {
    k = result.steps.length - 1;
    render();
  };

  const readout = el("div", "readout");
  panel.appendChild(readout);

  // ---- 作业纸 ----
  const sheet = el("div", "worksheet");

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(sheet);
  root.appendChild(moduleEl);

  function recompute(): void {
    result = eliminate(A, true);
    k = 0;
    render();
  }

  function render(): void {
    const step = result.steps[k];
    const last = k === result.steps.length - 1;
    sheet.innerHTML = "";
    sheet.appendChild(el("h3", "", `消元过程 —— 第 ${k} 步 / 共 ${result.steps.length - 1} 步`));
    sheet.appendChild(el("p", "step-desc", step.desc));

    const line = el("div", "mx-line");
    if (step.E) {
      line.appendChild(mxEl(step.E));
      line.appendChild(el("span", "", "·"));
    }
    line.appendChild(mxEl(step.A, step.pivot, step.target));
    sheet.appendChild(line);

    if (last) {
      const singular = result.pivots.length < 3;
      sheet.appendChild(
        el(
          "h3",
          "",
          singular
            ? `消元结束：只有 ${result.pivots.length} 个主元，rank(A) = ${result.pivots.length} < 3，矩阵不可逆`
            : result.swapped
              ? "分解结果：PA = LU"
              : "分解结果：A = LU",
        ),
      );
      const sum = el("div", "mx-line");
      if (result.swapped) {
        sum.appendChild(el("span", "", "P ="));
        sum.appendChild(mxEl(result.P));
      }
      sum.appendChild(el("span", "", "L ="));
      sum.appendChild(mxEl(result.L));
      sum.appendChild(el("span", "", "U ="));
      sum.appendChild(mxEl(result.U));
      sheet.appendChild(sum);
      const detVal = result.pivots.length < 3 ? 0 : result.pivots.reduce((a, b) => a * b, 1);
      sheet.appendChild(
        el(
          "p",
          "step-desc",
          `主元：${result.pivots.map((p) => fmt(p)).join("，") || "（无）"}` +
            `　→　det A = ${result.swapped ? "±" : ""}主元之积 = ${fmt(detVal)}`,
        ),
      );
    }

    readout.textContent =
      `进度: 第 ${k}/${result.steps.length - 1} 步\n` +
      `主元个数（秩）: ${result.pivots.length}\n` +
      (result.swapped ? "发生了行交换 → 写成 PA = LU\n" : "无需行交换 → A = LU\n") +
      `L 是所有乘数 ℓᵢⱼ 的收纳盒，\nU 的对角线就是主元`;
  }

  render();
  return () => {};
}
