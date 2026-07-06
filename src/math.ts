export interface Vec {
  x: number;
  y: number;
}

/** 2×2 矩阵，行主序：[a, b, c, d] 表示 [[a, b], [c, d]] */
export type Mat = [number, number, number, number];

export const I: Mat = [1, 0, 0, 1];

export function apply(m: Mat, v: Vec): Vec {
  return { x: m[0] * v.x + m[1] * v.y, y: m[2] * v.x + m[3] * v.y };
}

export function det(m: Mat): number {
  return m[0] * m[3] - m[1] * m[2];
}

/** 矩阵乘法 a·b */
export function mul(a: Mat, b: Mat): Mat {
  return [
    a[0] * b[0] + a[1] * b[2],
    a[0] * b[1] + a[1] * b[3],
    a[2] * b[0] + a[3] * b[2],
    a[2] * b[1] + a[3] * b[3],
  ];
}

export function lerpMat(a: Mat, b: Mat, t: number): Mat {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
    a[3] + (b[3] - a[3]) * t,
  ];
}

/** 解 Ax = b；A 奇异时返回 null */
export function solve2(m: Mat, b: Vec): Vec | null {
  const d = det(m);
  if (Math.abs(d) < 1e-10) return null;
  return {
    x: (b.x * m[3] - m[1] * b.y) / d,
    y: (m[0] * b.y - m[2] * b.x) / d,
  };
}

export function normalize(v: Vec): Vec {
  const n = Math.hypot(v.x, v.y);
  return n < 1e-12 ? { x: 0, y: 0 } : { x: v.x / n, y: v.y / n };
}

/** 3×3 矩阵（行的数组） */
export type M3 = number[][];

export function det3(m: M3): number {
  return (
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  );
}

/** Cramer 法则解 3×3 方程组，奇异时返回 null */
export function solve3(A: M3, b: number[]): [number, number, number] | null {
  const d = det3(A);
  if (Math.abs(d) < 1e-10) return null;
  const rep = (k: number): M3 => A.map((row, i) => row.map((v, j) => (j === k ? b[i] : v)));
  return [det3(rep(0)) / d, det3(rep(1)) / d, det3(rep(2)) / d];
}

export interface EigenResult {
  /** 特征值是否为实数；false 时 values 存 [实部, 虚部]，vectors 无意义 */
  real: boolean;
  values: [number, number];
  vectors: [Vec, Vec];
}

/** 2×2 矩阵的特征值/特征向量（特征向量已单位化） */
export function eigen(m: Mat): EigenResult {
  const tr = m[0] + m[3];
  const d = det(m);
  const disc = tr * tr - 4 * d;
  if (disc < -1e-12) {
    return {
      real: false,
      values: [tr / 2, Math.sqrt(-disc) / 2],
      vectors: [
        { x: 1, y: 0 },
        { x: 0, y: 1 },
      ],
    };
  }
  const s = Math.sqrt(Math.max(0, disc));
  const l1 = (tr + s) / 2;
  const l2 = (tr - s) / 2;
  return {
    real: true,
    values: [l1, l2],
    vectors: [eigenvector(m, l1, 0), eigenvector(m, l2, 1)],
  };
}

function eigenvector(m: Mat, lambda: number, idx: 0 | 1): Vec {
  // (A - λI)v = 0：第一行 (a-λ)x + b·y = 0 → v = (b, λ-a)
  const v1: Vec = { x: m[1], y: lambda - m[0] };
  if (Math.hypot(v1.x, v1.y) > 1e-9) return normalize(v1);
  // 第二行 c·x + (d-λ)y = 0 → v = (λ-d, c)
  const v2: Vec = { x: lambda - m[3], y: m[2] };
  if (Math.hypot(v2.x, v2.y) > 1e-9) return normalize(v2);
  // A = λI：任意方向都是特征向量，返回标准基
  return idx === 0 ? { x: 1, y: 0 } : { x: 0, y: 1 };
}
