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
import { mountQr } from "./modules/qr";
import { mountMarkov } from "./modules/markov";
import { mountCompress } from "./modules/compress";
import { mountCalc } from "./modules/nn/calc";
import { mountProb } from "./modules/nn/prob";
import { mountAutograd } from "./modules/nn/autograd";
import { mountGd } from "./modules/nn/gd";
import { mountBigram } from "./modules/nn/bigram";
import { mountEmbed } from "./modules/nn/embed";
import { mountHealth } from "./modules/nn/health";
import { mountWavenet } from "./modules/nn/wavenet";
import { mountAttention } from "./modules/nn/attention";
import { mountBpe } from "./modules/nn/bpe";

interface ModuleDef {
  key: string;
  lect: string;
  title: string;
  mount: (root: HTMLElement) => () => void;
}

interface Unit {
  name: string;
  items: ModuleDef[];
}

interface Course {
  key: string;
  title: string;
  units: Unit[];
}

const LA_UNITS: Unit[] = [
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
      { key: "qr", lect: "L17", title: "Gram-Schmidt 与 QR", mount: mountQr },
      { key: "determinant", lect: "L18–20", title: "行列式与面积", mount: mountDeterminant },
    ],
  },
  {
    name: "单元三 · 特征值",
    items: [
      { key: "eigen", lect: "L21–22", title: "特征向量", mount: mountEigen },
      { key: "powers", lect: "L22", title: "差分方程与 Aᵏ", mount: mountPowers },
      { key: "ode", lect: "L23", title: "微分方程与相图", mount: mountOde },
      { key: "markov", lect: "L24", title: "马尔可夫矩阵与稳态", mount: mountMarkov },
      { key: "quadform", lect: "L25–28", title: "对称矩阵与正定性", mount: mountQuadform },
      { key: "svd", lect: "L29", title: "奇异值分解 SVD", mount: mountSvd },
      { key: "compress", lect: "L31", title: "基变换与图像压缩", mount: mountCompress },
    ],
  },
];

const NN_UNITS: Unit[] = [
  {
    name: "第〇章 · 从零的基础",
    items: [
      { key: "calc", lect: "基础", title: "导数与链式法则", mount: mountCalc },
      { key: "prob", lect: "基础", title: "概率、Softmax 与交叉熵", mount: mountProb },
    ],
  },
  {
    name: "视频① · micrograd",
    items: [
      { key: "autograd", lect: "①·⑤", title: "计算图与反向传播", mount: mountAutograd },
      { key: "gd", lect: "①", title: "梯度下降实验台", mount: mountGd },
    ],
  },
  {
    name: "视频②③④ · makemore",
    items: [
      { key: "bigram", lect: "②", title: "Bigram 语言模型", mount: mountBigram },
      { key: "embed", lect: "③", title: "MLP 与嵌入空间", mount: mountEmbed },
      { key: "health", lect: "④", title: "激活与梯度健康", mount: mountHealth },
    ],
  },
  {
    name: "视频⑥⑦⑧ · 走向 GPT",
    items: [
      { key: "wavenet", lect: "⑥", title: "WaveNet 层级感受野", mount: mountWavenet },
      { key: "attention", lect: "⑦", title: "注意力机制", mount: mountAttention },
      { key: "bpe", lect: "⑧", title: "BPE 分词器", mount: mountBpe },
    ],
  },
];

const COURSES: Course[] = [
  { key: "la", title: "MIT 18.06 线性代数", units: LA_UNITS },
  { key: "nn", title: "Zero to Hero 神经网络", units: NN_UNITS },
];

const modules = new Map<string, ModuleDef>();
const courseOf = new Map<string, Course>();
for (const course of COURSES) {
  for (const unit of course.units) {
    for (const m of unit.items) {
      modules.set(m.key, m);
      courseOf.set(m.key, course);
    }
  }
}

const app = document.getElementById("app")!;
const nav = document.getElementById("nav")!;
const courseNav = document.getElementById("courses")!;
let cleanup: (() => void) | null = null;

for (const course of COURSES) {
  const btn = document.createElement("button");
  btn.textContent = course.title;
  btn.dataset.course = course.key;
  btn.onclick = () => {
    location.hash = course.units[0].items[0].key;
  };
  courseNav.appendChild(btn);
}

function renderNav(course: Course, activeKey: string): void {
  nav.innerHTML = "";
  for (const unit of course.units) {
    const label = document.createElement("div");
    label.className = "nav-unit";
    label.textContent = unit.name;
    nav.appendChild(label);
    for (const m of unit.items) {
      const btn = document.createElement("button");
      btn.className = "nav-item" + (m.key === activeKey ? " active" : "");
      btn.innerHTML = `<span class="lect">${m.lect}</span><span>${m.title}</span>`;
      btn.onclick = () => {
        location.hash = m.key;
      };
      nav.appendChild(btn);
    }
  }
}

function route(): void {
  let key = location.hash.slice(1);
  if (!modules.has(key)) key = "rowcol";
  const course = courseOf.get(key)!;
  cleanup?.();
  app.innerHTML = "";
  courseNav.querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
    b.classList.toggle("active", b.dataset.course === course.key);
  });
  renderNav(course, key);
  cleanup = modules.get(key)!.mount(app);
}

window.addEventListener("hashchange", route);
route();
