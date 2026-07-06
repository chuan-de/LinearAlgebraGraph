import type { Mat, Vec } from "./math";

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls = "",
  html = "",
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}

/** 数字格式化：保留两位小数并去掉多余的 0 和 -0 */
export function fmt(n: number): string {
  const v = Math.round(n * 100) / 100;
  return (Object.is(v, -0) ? 0 : v).toString();
}

function numInput(value: number, onInput: () => void): HTMLInputElement {
  const inp = el("input");
  inp.type = "number";
  inp.step = "0.1";
  inp.value = String(value);
  inp.oninput = onInput;
  return inp;
}

export interface MatrixInput {
  root: HTMLElement;
  set(m: Mat): void;
}

/** 带方括号样式的 2×2 矩阵输入 */
export function matrixInput(initial: Mat, onChange: (m: Mat) => void): MatrixInput {
  const root = el("div", "matrix");
  const read = (): Mat =>
    inputs.map((i) => {
      const v = Number.parseFloat(i.value);
      return Number.isFinite(v) ? v : 0;
    }) as Mat;
  const inputs = initial.map((v) => {
    const inp = numInput(v, () => onChange(read()));
    root.appendChild(inp);
    return inp;
  });
  return {
    root,
    set(m: Mat) {
      m.forEach((v, i) => (inputs[i].value = fmt(v)));
    },
  };
}

export interface VecInput {
  root: HTMLElement;
  set(v: Vec): void;
}

/** 带方括号样式的 2 维列向量输入 */
export function vecInput(initial: Vec, onChange: (v: Vec) => void): VecInput {
  const root = el("div", "matrix vec");
  const read = (): Vec => {
    const x = Number.parseFloat(ix.value);
    const y = Number.parseFloat(iy.value);
    return { x: Number.isFinite(x) ? x : 0, y: Number.isFinite(y) ? y : 0 };
  };
  const ix = numInput(initial.x, () => onChange(read()));
  const iy = numInput(initial.y, () => onChange(read()));
  root.appendChild(ix);
  root.appendChild(iy);
  return {
    root,
    set(v: Vec) {
      ix.value = fmt(v.x);
      iy.value = fmt(v.y);
    },
  };
}

export interface SliderRow {
  root: HTMLElement;
  set(v: number): void;
  setRange(min: number, max: number): void;
}

export function sliderRow(
  label: string,
  min: number,
  max: number,
  step: number,
  value: number,
  onInput: (v: number) => void,
): SliderRow {
  const root = el("div", "row");
  const lab = el("span", "slider-label", label);
  const input = el("input");
  input.type = "range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  const val = el("span", "slider-value", fmt(value));
  input.oninput = () => {
    val.textContent = fmt(input.valueAsNumber);
    onInput(input.valueAsNumber);
  };
  root.appendChild(lab);
  root.appendChild(input);
  root.appendChild(val);
  return {
    root,
    set(v: number) {
      input.value = String(v);
      val.textContent = fmt(v);
    },
    setRange(mn: number, mx: number) {
      input.min = String(mn);
      input.max = String(mx);
    },
  };
}
