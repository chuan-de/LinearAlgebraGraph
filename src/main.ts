import "./style.css";
import { mountTransform } from "./modules/transform";
import { mountRowCol } from "./modules/rowcol";
import { mountEigen } from "./modules/eigen";
import { mountElim } from "./modules/elim";
import { mountSubspaces } from "./modules/subspaces";
import { mountProject } from "./modules/project";
import { mountDeterminant } from "./modules/determinant";
import { mountPowers } from "./modules/powers";
import { mountOde } from "./modules/ode";
import { mountQuadform } from "./modules/quadform";
import { mountSvd } from "./modules/svd";

interface ModuleDef {
  key: string;
  lect: string;
  title: string;
  mount: (root: HTMLElement) => () => void;
}

const UNITS: { name: string; items: ModuleDef[] }[] = [
  {
    name: "单元一 · Ax = b",
    items: [
      { key: "rowcol", lect: "L1", title: "行图像 vs 列图像", mount: mountRowCol },
      { key: "elim", lect: "L2–4", title: "消元与 A = LU", mount: mountElim },
      { key: "transform", lect: "L3·30", title: "线性变换", mount: mountTransform },
      { key: "subspaces", lect: "L6–10", title: "四个基本子空间", mount: mountSubspaces },
    ],
  },
  {
    name: "单元二 · 正交与行列式",
    items: [
      { key: "project", lect: "L15–16", title: "投影与最小二乘", mount: mountProject },
      { key: "determinant", lect: "L18–20", title: "行列式与面积", mount: mountDeterminant },
    ],
  },
  {
    name: "单元三 · 特征值",
    items: [
      { key: "eigen", lect: "L21–22", title: "特征向量", mount: mountEigen },
      { key: "powers", lect: "L22", title: "差分方程与 Aᵏ", mount: mountPowers },
      { key: "ode", lect: "L23", title: "微分方程与相图", mount: mountOde },
      { key: "quadform", lect: "L25–28", title: "对称矩阵与正定性", mount: mountQuadform },
      { key: "svd", lect: "L29", title: "奇异值分解 SVD", mount: mountSvd },
    ],
  },
];

const modules = new Map<string, ModuleDef>();
for (const unit of UNITS) for (const m of unit.items) modules.set(m.key, m);

const app = document.getElementById("app")!;
const nav = document.getElementById("nav")!;
let cleanup: (() => void) | null = null;

for (const unit of UNITS) {
  const label = document.createElement("div");
  label.className = "nav-unit";
  label.textContent = unit.name;
  nav.appendChild(label);
  for (const m of unit.items) {
    const btn = document.createElement("button");
    btn.className = "nav-item";
    btn.dataset.key = m.key;
    btn.innerHTML = `<span class="lect">${m.lect}</span><span>${m.title}</span>`;
    btn.onclick = () => {
      location.hash = m.key;
    };
    nav.appendChild(btn);
  }
}

function route(): void {
  let key = location.hash.slice(1);
  if (!modules.has(key)) key = "rowcol";
  cleanup?.();
  app.innerHTML = "";
  nav.querySelectorAll<HTMLButtonElement>(".nav-item").forEach((b) => {
    b.classList.toggle("active", b.dataset.key === key);
  });
  cleanup = modules.get(key)!.mount(app);
}

window.addEventListener("hashchange", route);
route();
