import { Plane } from "../../plane";
import { COLORS } from "../../theme";
import { el, fmt, sliderRow } from "../../ui";

/** 教程页：概率、Softmax 与交叉熵（Zero to Hero 基础概念） */

const LETTERS = ["a", "b", "c", "d", "e"];

/** 画布 2 的固定观测数据 */
const DATA = ["a", "a", "b", "a", "c", "b", "a", "a", "a", "c"];
const OBS_LETTERS = ["a", "b", "c"];
const COUNTS = OBS_LETTERS.map((L) => DATA.filter((d) => d === L).length); // [6, 2, 2]
const N = DATA.length;
const FREQ = COUNTS.map((c) => c / N);
/** 理论下限：数据的熵 H = −Σ f·log f */
const ENTROPY = -FREQ.reduce((s, f) => s + f * Math.log(f), 0);

const f3 = (n: number): string => (Object.is(n, -0) ? 0 : n).toFixed(3);

function softmax(z: number[], T: number): number[] {
  const s = z.map((v) => v / T);
  const m = Math.max(...s);
  const e = s.map((v) => Math.exp(v - m));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

export function mountProb(root: HTMLElement): () => void {
  // ---- 状态 ----
  const logits = [2, 1, 0, -1, -2];
  let temp = 1;
  const weights = [0.34, 0.33, 0.33]; // 画布 2：用户给 a、b、c 的“分数”，内部归一化

  // ================= 文章骨架 =================
  const article = el("div", "article");

  article.appendChild(el("h2", "", "概率、Softmax 与交叉熵"));
  article.appendChild(
    el(
      "p",
      "",
      "语言模型做的事情出乎意料地朴素：看着已有的上下文，给“下一个字符”的每一种可能" +
        "<b>打一个分</b>，分高的更可能出现。这一页解决两个问题：打出来的分数怎么变成概率？" +
        "有了概率之后，又怎么衡量模型的好坏——也就是 loss 从哪里来。",
    ),
  );

  // ---- 第 1 节 ----
  article.appendChild(el("h3", "", "1. 从计数到概率"));
  article.appendChild(
    el(
      "p",
      "",
      "最原始的“打分”就是数数。假如在一大堆文本里，字母 q 后面跟 u 出现了 90 次、" +
        "跟 a 出现了 10 次，别的都没见过，那 q 后面接 u 的概率是多少？自然的做法是" +
        "把次数除以总数：90/100 = 0.9。这个“除以总和”的动作叫<b>归一化</b>，" +
        "它把一组非负的计数变成一组加起来等于 1 的数——这就是一个<b>概率分布</b>。",
    ),
  );
  article.appendChild(
    el(
      "p",
      "",
      "概率分布只有两条纪律：每一项 ≥ 0，全部加起来 = 1。计数天然满足第一条，" +
        "归一化保证第二条。麻烦在于，神经网络的输出层可不会体贴地吐出计数——" +
        "它吐出的是一堆<b>任意实数</b>，有正有负，我们称之为 <b>logits</b>。",
    ),
  );

  // ---- 第 2 节 ----
  article.appendChild(el("h3", "", "2. Softmax：把任意实数变成概率"));
  article.appendChild(
    el(
      "p",
      "",
      "怎么把一堆任意实数 z₁ … zₙ 变成合法的概率分布？两步。第一步，<b>取指数</b> e^z：" +
        "无论 z 是正是负，e^z 一定是正数，而且保序——原来谁大，取完指数还是谁大。" +
        "第二步，<b>归一化</b>：除以总和。合起来就是 softmax：",
    ),
  );
  article.appendChild(el("div", "formula", "pᵢ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T)"));
  article.appendChild(
    el(
      "p",
      "",
      "式子里还多了一个<b>温度 T</b>：先把所有 logits 除以 T 再取指数。" +
        "T 小，logits 之间的差距被放大，分布变得尖锐；T 大，差距被抹平，分布趋于均匀。",
    ),
  );
  article.appendChild(
    el(
      "p",
      "",
      "在下面的黑板上拖一拖，有三件事值得记住：① 给<b>所有</b> logit 同时加一个常数，" +
        "分布纹丝不动——分子分母同乘 e^c 被约掉了，softmax 只在乎<b>相对差距</b>（平移不变性）；" +
        "② 把 T 压到 0.2，概率几乎全部涌向最大的那个 logit（趋向 one-hot）；" +
        "③ 把 T 拉到 5，五根柱子趋于一样高（趋向均匀分布，虚线标出了 1/5）。",
    ),
  );

  const wrap1 = el("div", "board-wrap");
  const canvas1 = el("canvas", "plane");
  canvas1.style.height = "320px";
  wrap1.appendChild(canvas1);
  article.appendChild(wrap1);

  LETTERS.forEach((L, i) => {
    article.appendChild(
      sliderRow(L, -4, 4, 0.1, logits[i], (v) => {
        logits[i] = v;
        render1();
      }).root,
    );
  });
  article.appendChild(
    sliderRow("T", 0.2, 5, 0.05, temp, (v) => {
      temp = v;
      render1();
    }).root,
  );

  const readout1 = el("div", "readout");
  article.appendChild(readout1);

  // ---- 第 3 节 ----
  article.appendChild(el("h3", "", "3. 似然与交叉熵：概率如何变成 loss"));
  article.appendChild(
    el(
      "p",
      "",
      "有了会输出概率的模型，怎么评价它的好坏？标准很直白：" +
        "<b>好模型应该给实际发生的数据以高概率</b>。这个“模型赋予数据的概率”叫<b>似然</b>" +
        "（likelihood）。数据集里有 N 个观测，模型给每个观测 xᵢ 的概率是 p(xᵢ)，" +
        "整个数据集的似然就是它们的连乘：p(x₁)·p(x₂)·…·p(x_N)。",
    ),
  );
  article.appendChild(
    el(
      "p",
      "",
      "但连乘在实践中是灾难：一万个都小于 1 的数乘在一起，结果小到计算机的浮点数直接归零" +
        "（<b>下溢</b>）。补救的办法是<b>取对数</b>——对数把连乘变成连加，数值就安分了。" +
        "又因为习惯上 loss 要“越小越好”，而对数似然是越大越好，于是再取负号、除以 N 求平均，" +
        "得到<b>平均负对数似然</b>（NLL），也叫交叉熵损失：",
    ),
  );
  article.appendChild(el("div", "formula", "loss = −(1/N) Σᵢ log p(xᵢ)"));
  article.appendChild(
    el(
      "p",
      "",
      "模型给真实数据的概率越高，log p 越接近 0，loss 越小；反过来，给实际发生的事只留" +
        "0.01 的概率，就要付出 −log 0.01 ≈ 4.6 的沉重代价。交叉熵惩罚的正是“对现实感到意外”。",
    ),
  );
  article.appendChild(
    el(
      "p",
      "",
      "下面是一个最小实验。观测数据固定为十个字符：<b>a a b a c b a a a c</b>" +
        "（a 出现 6 次，b、c 各 2 次）。现在<b>由你来扮演模型</b>：用三个滑块设定给 a、b、c 的" +
        "概率（会自动归一化），黑板上并排画出你的分布（金色）与经验频率（淡色），" +
        "读出区实时计算你的平均 NLL。试着把 loss 压到最低——调不动了再点「揭示最优解」。",
    ),
  );

  const wrap2 = el("div", "board-wrap");
  const canvas2 = el("canvas", "plane");
  canvas2.style.height = "320px";
  wrap2.appendChild(canvas2);
  article.appendChild(wrap2);

  const wSliders = OBS_LETTERS.map((L, i) =>
    sliderRow(L, 0.01, 1, 0.01, weights[i], (v) => {
      weights[i] = v;
      render2();
    }),
  );
  wSliders.forEach((s) => article.appendChild(s.root));

  const btnRow = el("div", "row");
  const revealBtn = el("button", "btn", "揭示最优解");
  revealBtn.onclick = () => {
    FREQ.forEach((f, i) => {
      weights[i] = f;
      wSliders[i].set(f);
    });
    render2();
  };
  btnRow.appendChild(revealBtn);
  article.appendChild(btnRow);

  const readout2 = el("div", "readout");
  article.appendChild(readout2);
  const status = el("div", "status");
  article.appendChild(status);

  article.appendChild(
    el(
      "p",
      "",
      "你会发现最优解就是<b>经验频率本身</b>：p(a)=0.6、p(b)=0.2、p(c)=0.2。" +
        "此时的 NLL 达到理论下限——这个下限就是数据的<b>熵</b>。任何偏离真实频率的预测，" +
        "都会被交叉熵多罚一点。",
    ),
  );

  // ---- 第 4 节 ----
  article.appendChild(el("h3", "", "4. 结语"));
  article.appendChild(
    el(
      "p",
      "",
      "把整条流水线串起来：网络输出 logits → softmax 变成概率分布 → 在真实的下一个字符上" +
        "取 −log 得到 loss → 对所有位置求平均。<b>训练语言模型 = 用梯度下降把这个平均 NLL " +
        "压到最低</b>。下一站去「Bigram 语言模型」模块，看这套流程在真实字符数据上第一次跑通。",
    ),
  );

  const moduleEl = el("div", "module");
  moduleEl.appendChild(article);
  root.appendChild(moduleEl);

  // ================= 画布 1：softmax 柱状图 =================
  const plane1 = new Plane(canvas1);
  plane1.onRedraw = () => render1();

  function render1(): void {
    const W = canvas1.clientWidth;
    const H = canvas1.clientHeight;
    const left = 46;
    const right = 24;
    const top = 40;
    const bottom = H - 44;
    const p = softmax(logits, temp);

    plane1.clear();
    const ctx = plane1.ctx;

    // 标题
    ctx.fillStyle = COLORS.tick;
    ctx.font = "12.5px Consolas, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("p = softmax(z / T)", left, 14);

    // 均匀分布参考线 1/5
    const yUniform = bottom - 0.2 * (bottom - top);
    ctx.save();
    ctx.strokeStyle = COLORS.dim;
    ctx.setLineDash([4, 6]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, yUniform);
    ctx.lineTo(W - right, yUniform);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = COLORS.tick;
    ctx.font = "11px Consolas, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("1/5", left - 6, yUniform);

    // 底线
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(left, bottom);
    ctx.lineTo(W - right, bottom);
    ctx.stroke();

    // 柱子
    const slot = (W - left - right) / LETTERS.length;
    const barW = slot * 0.5;
    for (let i = 0; i < LETTERS.length; i++) {
      const cx = left + slot * i + slot / 2;
      const bh = p[i] * (bottom - top);
      ctx.save();
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = COLORS.gold;
      ctx.fillRect(cx - barW / 2, bottom - bh, barW, bh);
      ctx.restore();
      // 顶端概率值
      ctx.fillStyle = COLORS.chalk;
      ctx.font = "12px Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(p[i].toFixed(2), cx, bottom - bh - 4);
      // 下方字母与 logit
      ctx.font = "italic 15px Cambria, Georgia, serif";
      ctx.textBaseline = "top";
      ctx.fillText(LETTERS[i], cx, bottom + 6);
      ctx.fillStyle = COLORS.tick;
      ctx.font = "11px Consolas, monospace";
      ctx.fillText(`z=${fmt(logits[i])}`, cx, bottom + 24);
    }

    readout1.textContent = [
      `pᵢ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T)   T = ${fmt(temp)}`,
      `z = [${logits.map(fmt).join(", ")}]`,
      `p = [${p.map((v) => v.toFixed(3)).join(", ")}]   Σp = 1`,
    ].join("\n");
  }

  // ================= 画布 2：模型分布 vs 经验频率 =================
  const plane2 = new Plane(canvas2);
  plane2.onRedraw = () => render2();

  function render2(): void {
    const W = canvas2.clientWidth;
    const H = canvas2.clientHeight;
    const left = 46;
    const right = 24;
    const top = 46;
    const bottom = H - 44;
    const sum = weights.reduce((a, b) => a + b, 0);
    const p = weights.map((w) => w / sum);
    // 平均 NLL = −(1/N) Σᵢ log p(xᵢ) = −Σ freq·log p
    const nll = -FREQ.reduce((s, f, i) => s + f * Math.log(p[i]), 0);

    plane2.clear();
    const ctx = plane2.ctx;

    // 图例
    ctx.font = "12.5px Consolas, monospace";
    ctx.textBaseline = "middle";
    ctx.fillStyle = COLORS.gold;
    ctx.fillRect(left, 14, 12, 12);
    ctx.fillStyle = COLORS.chalk;
    ctx.textAlign = "left";
    ctx.fillText("你的模型 p", left + 18, 20);
    ctx.fillStyle = COLORS.dim;
    ctx.fillRect(left + 130, 14, 12, 12);
    ctx.fillStyle = COLORS.chalk;
    ctx.fillText("经验频率（计数/N）", left + 148, 20);

    // 底线
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(left, bottom);
    ctx.lineTo(W - right, bottom);
    ctx.stroke();

    const slot = (W - left - right) / OBS_LETTERS.length;
    const barW = slot * 0.22;
    const gap = slot * 0.06;
    for (let i = 0; i < OBS_LETTERS.length; i++) {
      const cx = left + slot * i + slot / 2;
      // 模型柱（金色，左）
      const mh = p[i] * (bottom - top);
      ctx.save();
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = COLORS.gold;
      ctx.fillRect(cx - gap / 2 - barW, bottom - mh, barW, mh);
      ctx.restore();
      // 频率柱（淡粉笔，右）
      const fh = FREQ[i] * (bottom - top);
      ctx.fillStyle = COLORS.dim;
      ctx.fillRect(cx + gap / 2, bottom - fh, barW, fh);
      // 顶端数值
      ctx.font = "11.5px Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillStyle = COLORS.gold;
      ctx.fillText(p[i].toFixed(2), cx - gap / 2 - barW / 2, bottom - mh - 4);
      ctx.fillStyle = COLORS.tick;
      ctx.fillText(FREQ[i].toFixed(2), cx + gap / 2 + barW / 2, bottom - fh - 4);
      // 下方字母 + 出现次数
      ctx.fillStyle = COLORS.chalk;
      ctx.font = "italic 15px Cambria, Georgia, serif";
      ctx.textBaseline = "top";
      ctx.fillText(OBS_LETTERS[i], cx, bottom + 6);
      ctx.fillStyle = COLORS.tick;
      ctx.font = "11px Consolas, monospace";
      ctx.fillText(`×${COUNTS[i]}`, cx, bottom + 24);
    }

    readout2.textContent = [
      `你的模型: p(a) = ${f3(p[0])}   p(b) = ${f3(p[1])}   p(c) = ${f3(p[2])}   （滑块已归一化）`,
      `经验频率: 6/10 = 0.600   2/10 = 0.200   2/10 = 0.200`,
      `平均 NLL = −(1/N) Σ log p(xᵢ) = ${f3(nll)}   （log 取自然对数）`,
      `理论下限 = 数据的熵 H = ${f3(ENTROPY)}`,
    ].join("\n");

    if (nll <= ENTROPY + 0.05) {
      status.innerHTML =
        `<span class="status-ok">✓ 你逼近了理论下限——最优预测就是真实频率</span>` +
        `　此时 NLL 只比熵高 ${f3(nll - ENTROPY)}。`;
    } else {
      status.textContent = `当前 NLL 比理论下限高 ${f3(nll - ENTROPY)}。哪个字母被你低估了，就把它的滑块推高一点。`;
    }
  }

  // ================= resize / cleanup =================
  const onResize = (): void => {
    plane1.resize();
    plane2.resize();
    render1();
    render2();
  };
  window.addEventListener("resize", onResize);

  render1();
  render2();

  return () => {
    window.removeEventListener("resize", onResize);
  };
}
