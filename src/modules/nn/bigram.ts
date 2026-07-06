import { COLORS } from "../../theme";
import { el } from "../../ui";

/* ============================================================
   Bigram 语言模型（Karpathy《Zero to Hero》视频②：makemore）
   只看前一个字符，预测下一个字符。
   两条路：① 直接数数得到条件概率表；② 用梯度下降训练一个
   27×27 的 logits 矩阵 —— 收敛后两者几乎一样。
   ============================================================ */

/** 数据集：常见英文名（全小写），首尾各补一个 '.' 作为开始/结束符 */
const NAMES: string[] = Array.from(
  new Set([
    // ---- 常见女名 ----
    "emma", "olivia", "ava", "isabella", "sophia", "charlotte", "mia", "amelia", "harper", "evelyn",
    "abigail", "emily", "elizabeth", "mila", "ella", "avery", "sofia", "camila", "aria", "scarlett",
    "victoria", "madison", "luna", "grace", "chloe", "penelope", "layla", "riley", "zoey", "nora",
    "lily", "eleanor", "hannah", "lillian", "addison", "aubrey", "ellie", "stella", "natalie", "zoe",
    "leah", "hazel", "violet", "aurora", "savannah", "audrey", "brooklyn", "bella", "claire", "skylar",
    "lucy", "paisley", "everly", "anna", "caroline", "nova", "genesis", "emilia", "kennedy", "samantha",
    "maya", "willow", "kinsley", "naomi", "aaliyah", "elena", "sarah", "ariana", "allison", "gabriella",
    "alice", "madelyn", "cora", "ruby", "eva", "serenity", "autumn", "adeline", "hailey", "gianna",
    "valentina", "isla", "eliana", "quinn", "nevaeh", "ivy", "sadie", "piper", "lydia", "alexa",
    "josephine", "emery", "julia", "delilah", "arianna", "vivian", "kaylee", "sophie", "brielle", "madeline",
    "peyton", "rylee", "clara", "hadley", "melanie", "mackenzie", "reagan", "adalynn", "liliana", "aubree",
    "jade", "katherine", "isabelle", "natalia", "raelynn", "maria", "athena", "ximena", "arya", "leilani",
    "taylor", "faith", "rose", "kylie", "alexandra", "mary", "margaret", "lyla", "ashley", "amaya",
    "eliza", "brianna", "bailey", "andrea", "khloe", "jasmine", "melody", "iris", "isabel", "norah",
    "annabelle", "valeria", "emerson", "adalyn", "ryleigh", "eden", "emersyn", "anastasia", "kayla", "alyssa",
    "juliana", "charlie", "esther", "ariel", "cecilia", "valerie", "alina", "molly", "reese", "aliyah",
    "lilly", "parker", "finley", "morgan", "sydney", "jordyn", "eloise", "trinity", "daisy", "kimberly",
    "lauren", "genevieve", "sara", "arabella", "harmony", "elise", "remi", "teagan", "alexis", "london",
    "sloane", "laila", "lucia", "diana", "juliette", "sienna", "elliana", "londyn", "ayla", "callie",
    "gracie", "josie", "amara", "jocelyn", "angela", "ana", "daniela", "paris", "camille", "adelaide",
    "dakota", "wren", "alana", "esme", "gemma", "blake", "harlow", "june", "freya", "vera",
    "joanna", "phoebe", "nina", "tessa", "heidi", "camryn", "kate", "fiona", "georgia", "catalina",
    "hope", "gwendolyn", "jane", "mckenna", "cassidy", "virginia", "evangeline", "olive", "alani", "selena",
    "kaia", "phoenix", "joy", "celeste", "ophelia", "dahlia", "holly", "miriam", "cheyenne", "michelle",
    "angelina", "carmen", "adrianna", "kali", "lola", "celine", "mabel", "ainsley", "rachel", "dorothy",
    "rebecca", "edith", "lena", "jessica", "laura", "susan", "karen", "linda", "nancy", "donna",
    "carol", "sandra", "ruth", "sharon", "cynthia", "kathleen", "amy", "shirley", "helen", "deborah",
    "pamela", "debra", "stephanie", "carolyn", "christine", "janet", "maureen", "diane", "julie", "joyce",
    "heather", "teresa", "doris", "gloria", "jean", "cheryl", "mildred", "frances", "martha", "irene",
    // ---- 常见男名 ----
    "liam", "noah", "oliver", "elijah", "james", "william", "benjamin", "lucas", "henry", "theodore",
    "jack", "levi", "alexander", "jackson", "mateo", "daniel", "michael", "mason", "sebastian", "ethan",
    "logan", "owen", "samuel", "jacob", "asher", "aiden", "john", "joseph", "wyatt", "david",
    "leo", "luke", "julian", "hudson", "grayson", "matthew", "ezra", "gabriel", "carter", "isaac",
    "jayden", "luca", "anthony", "dylan", "lincoln", "thomas", "maverick", "elias", "josiah", "charles",
    "caleb", "christopher", "ezekiel", "miles", "jaxon", "isaiah", "andrew", "joshua", "nathan", "nolan",
    "adrian", "cameron", "santiago", "eli", "aaron", "ryan", "angel", "cooper", "waylon", "easton",
    "kai", "christian", "landon", "colton", "roman", "axel", "brooks", "jonathan", "robert", "jameson",
    "ian", "everett", "greyson", "wesley", "jeremiah", "hunter", "leonardo", "jordan", "jose", "bennett",
    "silas", "nicholas", "beau", "weston", "austin", "connor", "carson", "dominic", "xavier", "jaxson",
    "jace", "emmett", "adam", "declan", "rowan", "micah", "kayden", "gael", "river", "ryder",
    "kingston", "damian", "sawyer", "luka", "evan", "vincent", "legend", "myles", "harrison", "august",
    "bryson", "amir", "giovanni", "chase", "diego", "milo", "jasper", "walker", "jason", "brayden",
    "cole", "nathaniel", "george", "lorenzo", "zion", "luis", "archer", "enzo", "jonah", "thiago",
    "theo", "ayden", "zachary", "calvin", "braxton", "ashton", "rhett", "atlas", "jude", "bentley",
    "carlos", "ryker", "adriel", "arthur", "ace", "tyler", "jayce", "max", "elliot", "graham",
    "kaiden", "maxwell", "juan", "dean", "matteo", "malachi", "ivan", "elliott", "jesus", "emiliano",
    "gavin", "maddox", "camden", "hayden", "leon", "antonio", "justin", "tucker", "brandon", "abel",
    "judah", "finn", "king", "brody", "xander", "emmanuel", "barrett", "tristan", "brady", "kaleb",
    "preston", "oscar", "alejandro", "marcus", "alan", "joel", "victor", "abraham", "kaden", "edward",
    "grant", "eric", "patrick", "peter", "richard", "kevin", "steven", "mark", "paul", "kenneth",
    "timothy", "jeffrey", "scott", "brian", "frank", "gregory", "raymond", "dennis", "jerry", "walter",
    "terry", "gerald", "keith", "roger", "lawrence", "sean", "albert", "jesse", "willie", "bruce",
    "ralph", "roy", "eugene", "russell", "bobby", "philip", "harry", "howard", "fred", "alfred",
    "stanley", "ernest", "martin", "leroy", "tony", "mike", "danny", "carl", "clarence", "earl",
    "joe", "allen", "marvin", "glenn", "dale", "arnold", "harold", "norman", "warren", "chad",
    "kyle", "jared", "wayne", "billy", "corey", "derek", "trevor", "travis", "todd", "brent",
    "craig", "curtis", "darren", "dustin", "shawn", "shane", "troy", "marc", "neil", "ross",
    "drew", "colin", "spencer", "garrett", "mitchell", "clayton", "wade", "dallas", "chance", "dawson",
    "gunnar", "hank", "otto", "hugo", "felix", "cyrus", "ronan", "callum", "killian", "cormac",
    "desmond", "quentin", "reid", "sterling", "clark", "wilson", "ford", "reed", "wells", "banks",
    "hayes", "lane", "knox", "nash", "zane", "remy", "tate", "cash", "colt", "duke",
    "jax", "rex", "cruz", "kane", "seth", "noel", "ellis", "orion", "otis", "amos",
  ]),
);

const CHARS = ".abcdefghijklmnopqrstuvwxyz"; // 27 个 token
const V = 27;

const SIZE = 600; // 画布 CSS 尺寸
const LABEL = 26; // 四周标签留白
const CELL = (SIZE - LABEL) / V;

function hexToRgb(hex: string): [number, number, number] {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

const CHALK_RGB = hexToRgb(COLORS.chalk);
const GOLD_RGB = hexToRgb(COLORS.gold);

/** 粉笔白 → 金色 插值，t∈[0,1] */
function heatColor(t: number, alpha: number): string {
  const r = Math.round(CHALK_RGB[0] + (GOLD_RGB[0] - CHALK_RGB[0]) * t);
  const g = Math.round(CHALK_RGB[1] + (GOLD_RGB[1] - CHALK_RGB[1]) * t);
  const b = Math.round(CHALK_RGB[2] + (GOLD_RGB[2] - CHALK_RGB[2]) * t);
  return `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
}

/** 计数矩阵 N[i][j]：bigram (i→j) 在数据集中出现的次数 */
function buildCounts(): number[][] {
  const n: number[][] = [];
  for (let i = 0; i < V; i++) n.push(new Array<number>(V).fill(0));
  for (const name of NAMES) {
    const s = `.${name}.`;
    for (let k = 0; k + 1 < s.length; k++) {
      const i = CHARS.indexOf(s[k]);
      const j = CHARS.indexOf(s[k + 1]);
      n[i][j]++;
    }
  }
  return n;
}

/** 单行 softmax（减去最大值保证数值稳定） */
function softmaxRow(row: number[]): number[] {
  let m = -Infinity;
  for (const v of row) m = Math.max(m, v);
  const e = row.map((v) => Math.exp(v - m));
  let s = 0;
  for (const v of e) s += v;
  return e.map((v) => v / s);
}

export function mountBigram(root: HTMLElement): () => void {
  // ---- 模型数据 ----
  const N = buildCounts();
  const cntRow = N.map((row) => row.reduce((a, b) => a + b, 0));
  const total = cntRow.reduce((a, b) => a + b, 0);
  let maxCount = 0;
  for (const row of N) for (const c of row) maxCount = Math.max(maxCount, c);

  // 计数模型：P[i][j] = (N+1) / Σ(N+1)（+1 拉普拉斯平滑）
  const Pcount: number[][] = N.map((row) => {
    const s = row.reduce((a, b) => a + b + 1, 0);
    return row.map((c) => (c + 1) / s);
  });
  let nllCount = 0;
  for (let i = 0; i < V; i++) {
    for (let j = 0; j < V; j++) {
      if (N[i][j] > 0) nllCount -= N[i][j] * Math.log(Pcount[i][j]);
    }
  }
  nllCount /= total;

  // 神经网络模型：27×27 logits 矩阵 W，P = 逐行 softmax(W)
  let W: number[][] = [];
  let Pnn: number[][] = [];
  let nllNN = 0;
  let trainFrames = 0;
  const resetW = (): void => {
    W = [];
    for (let i = 0; i < V; i++) W.push(new Array<number>(V).fill(0));
    trainFrames = 0;
    refreshNN();
  };
  function refreshNN(): void {
    Pnn = W.map(softmaxRow);
    let loss = 0;
    for (let i = 0; i < V; i++) {
      for (let j = 0; j < V; j++) {
        if (N[i][j] > 0) loss -= N[i][j] * Math.log(Pnn[i][j]);
      }
    }
    nllNN = loss / total;
  }
  resetW();

  /** 全批量梯度下降一步：grad[i][j] = (cntRow[i]·softmax(W[i])[j] − N[i][j]) / total */
  function gdStep(lr: number): void {
    for (let i = 0; i < V; i++) {
      if (cntRow[i] === 0) continue;
      const sm = softmaxRow(W[i]);
      for (let j = 0; j < V; j++) {
        const grad = (cntRow[i] * sm[j] - N[i][j]) / total;
        W[i][j] -= lr * grad;
      }
    }
  }

  // ---- 交互状态 ----
  let countMode = true; // true=计数模型 false=神经网络
  let training = false;
  let rafId = 0;
  let hoverI = -1;
  let hoverJ = -1;
  let sampleRow = -1; // 采样动画：当前行
  let samplePick = -1; // 采样动画：被抽中的列
  let sampleTimer = 0;
  let curName = "";
  let samples: string[] = [];

  const activeP = (): number[][] => (countMode ? Pcount : Pnn);

  function pickFromRow(probs: number[]): number {
    let r = Math.random();
    for (let j = 0; j < V; j++) {
      r -= probs[j];
      if (r <= 0) return j;
    }
    return V - 1;
  }

  /** 瞬间采样一个完整名字（无动画） */
  function sampleOnce(): string {
    const P = activeP();
    let i = 0;
    let out = "";
    for (let step = 0; step < 40; step++) {
      const j = pickFromRow(P[i]);
      if (j === 0) break;
      out += CHARS[j];
      i = j;
    }
    return out;
  }

  // ---- 控制面板 ----
  const panel = el("div", "panel");
  panel.appendChild(el("h2", "", "Bigram 语言模型（makemore）"));
  panel.appendChild(
    el(
      "p",
      "hint",
      "对应 Zero to Hero 视频②。<b>bigram</b> = 只看前一个字符来预测下一个字符：" +
        "热力图的<b>行 = 当前字符</b>，<b>列 = 下一个字符</b>，'.' 是名字的开始/结束符。" +
        "把数据集里所有相邻字符对数一遍，就得到一张条件概率表——这就是最简单的语言模型。",
    ),
  );

  // 模式切换
  const modeRow = el("div", "row");
  modeRow.appendChild(el("span", "", "模型"));
  const countBtn = el("button", "btn", "计数模型");
  const nnBtn = el("button", "btn", "神经网络");
  modeRow.appendChild(countBtn);
  modeRow.appendChild(nnBtn);
  panel.appendChild(modeRow);

  // 训练控制（仅神经网络模式）
  const trainRow = el("div", "row");
  const trainBtn = el("button", "btn", "▶ 训练");
  const resetBtn = el("button", "btn", "重置 W");
  trainRow.appendChild(trainBtn);
  trainRow.appendChild(resetBtn);
  panel.appendChild(trainRow);

  // 采样
  const sampleRowEl = el("div", "row");
  const sampleBtn = el("button", "btn primary", "采样一个名字");
  const sample10Btn = el("button", "btn", "采样 ×10");
  sampleRowEl.appendChild(sampleBtn);
  sampleRowEl.appendChild(sample10Btn);
  panel.appendChild(sampleRowEl);

  // 当前采样的名字（大号等宽）
  const nameDiv = el("div", "readout");
  nameDiv.style.fontSize = "22px";
  nameDiv.style.letterSpacing = "2px";
  nameDiv.style.minHeight = "46px";
  nameDiv.style.textAlign = "center";
  nameDiv.textContent = "·";
  panel.appendChild(nameDiv);

  // 采样结果列表
  const samplesDiv = el("div", "readout");
  samplesDiv.textContent = "（采样结果会列在这里）";
  panel.appendChild(samplesDiv);

  // 数据集与损失读出
  const readout = el("div", "readout");
  panel.appendChild(readout);

  const status = el("div", "status");
  panel.appendChild(status);

  // ---- 热力图黑板 ----
  const col = el("div", "canvas-col");
  col.appendChild(el("h3", "", "27×27 bigram 热力图（行 = 当前字符 → 列 = 下一个字符）"));
  const canvas = el("canvas", "plane");
  canvas.style.width = `${SIZE}px`;
  canvas.style.height = `${SIZE}px`;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = SIZE * dpr;
  canvas.height = SIZE * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  col.appendChild(canvas);

  const moduleEl = el("div", "module");
  moduleEl.appendChild(panel);
  moduleEl.appendChild(col);
  root.appendChild(moduleEl);

  // ---- 绘制 ----
  const cellX = (j: number): number => LABEL + j * CELL;
  const cellY = (i: number): number => LABEL + i * CELL;

  function draw(): void {
    ctx.clearRect(0, 0, SIZE, SIZE);

    // 标签
    ctx.font = "11px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let k = 0; k < V; k++) {
      const active = k === sampleRow || k === hoverI;
      ctx.fillStyle = active ? COLORS.gold : COLORS.tick;
      ctx.fillText(CHARS[k], LABEL / 2, cellY(k) + CELL / 2); // 行标签
      ctx.fillStyle = k === samplePick || k === hoverJ ? COLORS.gold : COLORS.tick;
      ctx.fillText(CHARS[k], cellX(k) + CELL / 2, LABEL / 2); // 列标签
    }

    // 格子
    let maxP = 0;
    if (!countMode) {
      for (const row of Pnn) for (const p of row) maxP = Math.max(maxP, p);
    }
    for (let i = 0; i < V; i++) {
      for (let j = 0; j < V; j++) {
        let t: number;
        if (countMode) {
          t = Math.log(1 + N[i][j]) / Math.log(1 + maxCount);
        } else {
          t = maxP > 0 ? Pnn[i][j] / maxP : 0;
        }
        if (t > 1e-3) {
          ctx.fillStyle = heatColor(t, 0.06 + 0.94 * t);
          ctx.fillRect(cellX(j), cellY(i), CELL, CELL);
        }
      }
    }

    // 网格线
    ctx.strokeStyle = "rgba(236,231,214,0.07)";
    ctx.lineWidth = 1;
    for (let k = 0; k <= V; k++) {
      ctx.beginPath();
      ctx.moveTo(cellX(k), LABEL);
      ctx.lineTo(cellX(k), SIZE);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(LABEL, cellY(k));
      ctx.lineTo(SIZE, cellY(k));
      ctx.stroke();
    }

    // 采样动画高亮：当前行 + 被抽中的格子
    if (sampleRow >= 0) {
      ctx.strokeStyle = COLORS.cyan;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(LABEL + 0.5, cellY(sampleRow) + 0.5, SIZE - LABEL - 1, CELL - 1);
      if (samplePick >= 0) {
        ctx.strokeStyle = COLORS.red;
        ctx.lineWidth = 2.5;
        ctx.strokeRect(cellX(samplePick) + 1, cellY(sampleRow) + 1, CELL - 2, CELL - 2);
      }
    }

    // hover 高亮框
    if (hoverI >= 0 && hoverJ >= 0) {
      ctx.strokeStyle = COLORS.chalk;
      ctx.lineWidth = 2;
      ctx.strokeRect(cellX(hoverJ) + 1, cellY(hoverI) + 1, CELL - 2, CELL - 2);
    }
  }

  function updateReadout(): void {
    readout.textContent = [
      `数据集：${NAMES.length} 个名字，${total} 个 bigram`,
      `计数模型平均 NLL = ${nllCount.toFixed(4)}`,
      `神经网络当前 NLL = ${nllNN.toFixed(4)}`,
      `均匀分布基线 ln 27 ≈ ${Math.log(27).toFixed(4)}`,
    ].join("\n");
  }

  function defaultStatus(): void {
    if (countMode) {
      status.innerHTML =
        "计数模型：格子亮度 ∝ log(1+次数)。鼠标悬停任意格子查看条件概率 P(下一个|当前)。";
    } else if (nllNN < nllCount + 0.03 && trainFrames > 0) {
      status.innerHTML =
        `<span class="status-ok">✓ 神经网络 NLL ≈ 计数模型 NLL——梯度下降学出了和数数一样的答案。</span>` +
        " 这正是视频②的中心论点：softmax 回归收敛到的解，就是频率统计。";
    } else if (training) {
      status.textContent = "训练中：每帧做几步全批量梯度下降，看热力图逐渐长出计数模型的花纹……";
    } else {
      status.textContent =
        "神经网络模式：W 初始全 0，softmax 后每行都是均匀分布（一片均匀的雾）。点「▶ 训练」开始梯度下降。";
    }
  }

  function render(): void {
    draw();
    updateReadout();
  }

  // ---- 模式切换 ----
  function syncMode(): void {
    countBtn.classList.toggle("primary", countMode);
    nnBtn.classList.toggle("primary", !countMode);
    trainBtn.disabled = countMode;
    resetBtn.disabled = countMode;
    trainRow.style.opacity = countMode ? "0.45" : "1";
    defaultStatus();
    render();
  }
  countBtn.onclick = () => {
    countMode = true;
    stopTraining();
    syncMode();
  };
  nnBtn.onclick = () => {
    countMode = false;
    syncMode();
  };

  // ---- 训练循环 ----
  function trainLoop(): void {
    if (!training) return;
    for (let s = 0; s < 8; s++) gdStep(40);
    trainFrames++;
    refreshNN();
    defaultStatus();
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
      defaultStatus();
    } else {
      training = true;
      trainBtn.textContent = "⏸ 暂停";
      rafId = requestAnimationFrame(trainLoop);
    }
  };
  resetBtn.onclick = () => {
    stopTraining();
    resetW();
    defaultStatus();
    render();
  };

  // ---- 采样 ----
  function pushSample(name: string): void {
    samples.unshift(name === "" ? "（空）" : name);
    if (samples.length > 12) samples = samples.slice(0, 12);
    samplesDiv.textContent = samples.join("\n");
  }
  function stopSampleAnim(): void {
    if (sampleTimer) window.clearInterval(sampleTimer);
    sampleTimer = 0;
    sampleRow = -1;
    samplePick = -1;
  }
  sampleBtn.onclick = () => {
    stopSampleAnim();
    curName = "";
    sampleRow = 0;
    nameDiv.textContent = "·";
    sampleTimer = window.setInterval(() => {
      const P = activeP();
      const j = pickFromRow(P[sampleRow]);
      samplePick = j;
      if (j === 0 || curName.length >= 40) {
        nameDiv.textContent = curName === "" ? "（空）" : curName;
        pushSample(curName);
        window.clearInterval(sampleTimer);
        sampleTimer = 0;
        render();
        // 让最后一格的高亮停留片刻再消失
        window.setTimeout(() => {
          if (!sampleTimer) {
            sampleRow = -1;
            samplePick = -1;
            render();
          }
        }, 600);
        return;
      }
      curName += CHARS[j];
      nameDiv.textContent = curName + "▌";
      sampleRow = j;
      render();
    }, 220);
  };
  sample10Btn.onclick = () => {
    stopSampleAnim();
    for (let k = 0; k < 10; k++) pushSample(sampleOnce());
    nameDiv.textContent = samples[0];
    render();
  };

  // ---- hover ----
  const onMove = (ev: PointerEvent): void => {
    const rect = canvas.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * SIZE;
    const y = ((ev.clientY - rect.top) / rect.height) * SIZE;
    const j = Math.floor((x - LABEL) / CELL);
    const i = Math.floor((y - LABEL) / CELL);
    if (i < 0 || i >= V || j < 0 || j >= V) {
      if (hoverI !== -1) {
        hoverI = -1;
        hoverJ = -1;
        defaultStatus();
        draw();
      }
      return;
    }
    if (i === hoverI && j === hoverJ) return;
    hoverI = i;
    hoverJ = j;
    const p = activeP()[i][j];
    status.innerHTML =
      `<b>${CHARS[i]} → ${CHARS[j]}</b>：P(${CHARS[j]}|${CHARS[i]}) = ${p.toFixed(4)}，` +
      `出现 ${N[i][j]} 次`;
    draw();
  };
  const onLeave = (): void => {
    hoverI = -1;
    hoverJ = -1;
    defaultStatus();
    draw();
  };
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);

  syncMode();

  return () => {
    stopTraining();
    stopSampleAnim();
    canvas.removeEventListener("pointermove", onMove);
    canvas.removeEventListener("pointerleave", onLeave);
  };
}
