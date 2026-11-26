/* kernels.h -- the work for ONE node, shared by all four versions.
 * The backends only decide which thread runs which nodes. */
#ifndef KERNELS_H
#define KERNELS_H
#include "tsne.h"
#include <math.h>

#define BISECT_TOL   1e-5
#define BISECT_MAXIT 200

/* ---------------------------------------------------------------- P ---
 * Row i: find beta_i so that the entropy of p_{j|i} = w_ij^beta / sum
 * equals log min(u, d_i); write p_{j|i} into cond[ptr[i]..ptr[i+1]-1].
 * Uses l_k = log(w_k / w_max) <= 0, so x^beta = exp(beta*l_k) <= 1: no
 * overflow. With S = sum exp(beta l_k):  H = log S - beta * sum(e_k l_k)/S.
 * H decreases as beta grows, so bisect. Returns beta_i (0 if uniform). */
static inline double row_beta(const graph_csr *G, int32_t i, double u) {
  const int64_t lo = G->ptr[i], hi = G->ptr[i + 1];
  const int64_t d = hi - lo;
  double *c = G->cond;
  if (d == 0) return 0.0;
  double wmax = G->w[lo], wmin = G->w[lo];
  for (int64_t k = lo + 1; k < hi; k++) {
    if (G->w[k] > wmax) wmax = G->w[k];
    if (G->w[k] < wmin) wmin = G->w[k];
  }
  if (u >= (double)d || wmax == wmin) {          /* uniform row */
    for (int64_t k = lo; k < hi; k++) c[k] = 1.0 / (double)d;
    return 0.0;
  }
  const double target = log(u);
  for (int64_t k = lo; k < hi; k++) c[k] = log(G->w[k] / wmax);  /* l_k */

  double beta = 1.0, bmin = 0.0, bmax = INFINITY;
  for (int it = 0; it < BISECT_MAXIT; it++) {
    double S = 0.0, SL = 0.0;
    for (int64_t k = lo; k < hi; k++) {
      double e = exp(beta * c[k]);
      S += e; SL += e * c[k];
    }
    double diff = log(S) - beta * SL / S - target;
    if (fabs(diff) < BISECT_TOL) break;
    if (diff > 0) {            /* entropy too high: sharpen, beta up */
      bmin = beta;
      beta = isinf(bmax) ? 2.0 * beta : 0.5 * (beta + bmax);
    } else {                   /* entropy too low: beta down */
      bmax = beta;
      beta = 0.5 * (beta + bmin);
    }
  }
  double S = 0.0;
  for (int64_t k = lo; k < hi; k++) { c[k] = exp(beta * c[k]); S += c[k]; }
  for (int64_t k = lo; k < hi; k++) c[k] /= S;
  return beta;
}

/* Symmetrize row i: p_ij = (p_{j|i} + p_{i|j}) / 2n. The entry (j,i) is
 * found by binary search in row j (the column indices are sorted). */
static inline void row_symmetrize(const graph_csr *G, int32_t i) {
  const double inv2n = 0.5 / (double)G->n;
  for (int64_t k = G->ptr[i]; k < G->ptr[i + 1]; k++) {
    int32_t j = G->col[k];
    int64_t a = G->ptr[j], b = G->ptr[j + 1] - 1;
    while (a < b) {                    /* find col == i in row j */
      int64_t m = (a + b) >> 1;
      if (G->col[m] < i) a = m + 1; else b = m;
    }
    G->P[k] = (G->cond[k] + G->cond[a]) * inv2n;
  }
}

/* ------------------------------------------------------- attraction ---
 * Row i of (D - A) Y^T, with A_ij = p_ij / (1 + d_ij^2) computed on the
 * fly:  F_i = sum_j A_ij (y_i - y_j). Cost proportional to deg(i). */
static inline void attr_row(const graph_csr *G, const double *Y, int32_t i,
                            double *F) {
  const double xi = Y[2 * i], yi = Y[2 * i + 1];
  double fx = 0.0, fy = 0.0;
  for (int64_t k = G->ptr[i]; k < G->ptr[i + 1]; k++) {
    int32_t j = G->col[k];
    double dx = xi - Y[2 * j], dy = yi - Y[2 * j + 1];
    double a = G->P[k] / (1.0 + dx * dx + dy * dy);
    fx += a * dx; fy += a * dy;
  }
  F[2 * i] = fx; F[2 * i + 1] = fy;
}

/* ---------------------------------------------------------- quadtree --- */
/* spread the low 16 bits of x to the even bits */
static inline uint32_t spread16(uint32_t x) {
  x &= 0xFFFFu;
  x = (x | (x << 8)) & 0x00FF00FFu;
  x = (x | (x << 4)) & 0x0F0F0F0Fu;
  x = (x | (x << 2)) & 0x33333333u;
  x = (x | (x << 1)) & 0x55555555u;
  return x;
}

/* leaf (Morton code) of point (x, y) */
static inline uint32_t leaf_code(const quadtree *T, double x, double y) {
  const int g = 1 << T->L;
  int ix = (int)((x - T->x0) / T->side * g);
  int iy = (int)((y - T->y0) / T->side * g);
  if (ix < 0) ix = 0; if (ix >= g) ix = g - 1;
  if (iy < 0) iy = 0; if (iy >= g) iy = g - 1;
  return spread16((uint32_t)ix) | (spread16((uint32_t)iy) << 1);
}

/* center of mass of leaf c from its sorted points */
static inline void leaf_com(quadtree *T, int64_t c) {
  cell_t *cl = &T->cells[T->off[T->L] + c];
  double sx = 0.0, sy = 0.0;
  for (int64_t s = T->lstart[c]; s < T->lstart[c + 1]; s++) {
    sx += T->ys[2 * s]; sy += T->ys[2 * s + 1];
  }
  double n = (double)(T->lstart[c + 1] - T->lstart[c]);
  cl->n = n;
  cl->x = n > 0 ? sx / n : 0.0;
  cl->y = n > 0 ? sy / n : 0.0;
}

/* cell c at level l from its 4 children at level l+1 */
static inline void parent_com(quadtree *T, int l, int64_t c) {
  const cell_t *ch = &T->cells[T->off[l + 1] + 4 * c];
  double n = 0.0, sx = 0.0, sy = 0.0;
  for (int k = 0; k < 4; k++) {
    n += ch[k].n; sx += ch[k].n * ch[k].x; sy += ch[k].n * ch[k].y;
  }
  cell_t *p = &T->cells[T->off[l] + c];
  p->n = n;
  p->x = n > 0 ? sx / n : 0.0;
  p->y = n > 0 ? sy / n : 0.0;
}

/* ------------------------------------------------------ Barnes-Hut ---
 * Walk the tree for the s-th sorted point. Accumulates the unnormalized
 * repulsion (fx, fy) and its contribution to Z. A cell is accepted as
 * one body when diag / dist < theta, i.e. diag^2 < theta^2 dist^2.
 * The point itself is skipped by its sorted index at the leaves; it is
 * never inside an accepted cell, since for a cell that contains y_i,
 * dist(y_i, com) <= diag, so diag/dist >= 1 > theta. */
static inline void walk_point(const quadtree *T, double theta2, int64_t s,
                              double *fx_out, double *fy_out, double *z_out) {
  const double xi = T->ys[2 * s], yi = T->ys[2 * s + 1];
  double fx = 0.0, fy = 0.0, z = 0.0;
  int     stl[64 * 4];       /* explicit stack: level, code */
  int64_t stc[64 * 4];
  int top = 0;
  stl[0] = 0; stc[0] = 0; top = 1;
  while (top > 0) {
    --top;
    const int l = stl[top];
    const int64_t c = stc[top];
    const cell_t *cl = &T->cells[T->off[l] + c];
    if (cl->n == 0.0) continue;
    if (l == T->L) {                                   /* leaf: one by one */
      for (int64_t t = T->lstart[c]; t < T->lstart[c + 1]; t++) {
        if (t == s) continue;
        double dx = xi - T->ys[2 * t], dy = yi - T->ys[2 * t + 1];
        double q = 1.0 / (1.0 + dx * dx + dy * dy);
        z += q; fx += q * q * dx; fy += q * q * dy;
      }
      continue;
    }
    double dx = xi - cl->x, dy = yi - cl->y;
    double d2 = dx * dx + dy * dy;
    if (T->diag2[l] < theta2 * d2) {                   /* far: one body */
      double q = 1.0 / (1.0 + d2);
      z += cl->n * q;
      fx += cl->n * q * q * dx; fy += cl->n * q * q * dy;
    } else {                                           /* open the cell */
      for (int k = 0; k < 4; k++) { stl[top] = l + 1; stc[top] = 4 * c + k; top++; }
    }
  }
  *fx_out = fx; *fy_out = fy; *z_out = z;
}

/* exact repulsion of point i: O(n) */
static inline void direct_point(const double *Y, int32_t n, int32_t i,
                                double *fx_out, double *fy_out, double *z_out) {
  const double xi = Y[2 * i], yi = Y[2 * i + 1];
  double fx = 0.0, fy = 0.0, z = 0.0;
  for (int32_t j = 0; j < n; j++) {
    double dx = xi - Y[2 * j], dy = yi - Y[2 * j + 1];
    double q = 1.0 / (1.0 + dx * dx + dy * dy);
    q = (j == i) ? 0.0 : q;
    z += q; fx += q * q * dx; fy += q * q * dy;
  }
  *fx_out = fx; *fy_out = fy; *z_out = z;
}


/* ===================================================================
 * Chunk-level kernels: a range of rows / points / leaves / cells.
 * Every version calls these; it only decides which thread gets which range.
 * =================================================================== */
#include <string.h>

static inline int64_t imin64(int64_t a, int64_t b) { return a < b ? a : b; }

/* the b-th of nb near-equal blocks of 0..n-1 */
static inline void block_range(int64_t n, int nb, int b, int64_t *lo, int64_t *hi) {
  *lo = n * b / nb; *hi = n * (b + 1) / nb;
}

static inline void P_rows(graph_csr *G, double u, int64_t lo, int64_t hi) {
  for (int64_t i = lo; i < hi; i++) G->beta[i] = row_beta(G, (int32_t)i, u);
}
static inline void sym_rows(const graph_csr *G, int64_t lo, int64_t hi) {
  for (int64_t i = lo; i < hi; i++) row_symmetrize(G, (int32_t)i);
}
static inline void attr_rows(const graph_csr *G, const double *Y, double *F,
                             int64_t lo, int64_t hi) {
  for (int64_t i = lo; i < hi; i++) attr_row(G, Y, (int32_t)i, F);
}

/* --- tree build, as a stable parallel counting sort in 5 steps:
 *  1. bbox_block:   min/max of each block of points        (parallel over point blocks)
 *  2. hist_block:   leaf code of each point, and per-block leaf counts h[b][c]
 *  3. leaf_totals:  sum over blocks of each leaf's count, per block of leaves
 *     (then a short sequential scan over the blocks of leaves)
 *  4. leaf_offsets: lstart[c], and h[b][c] := first slot of block b in leaf c
 *  5. scatter_block: each point to its slot. Block b writes after blocks < b
 *     in every leaf, so the order is the same as the sequential sort. */
static inline void bbox_block(const double *Y, int64_t lo, int64_t hi, double b[4]) {
  double xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
  for (int64_t i = lo; i < hi; i++) {
    double x = Y[2 * i], y = Y[2 * i + 1];
    if (x < xmin) xmin = x; if (x > xmax) xmax = x;
    if (y < ymin) ymin = y; if (y > ymax) ymax = y;
  }
  b[0] = xmin; b[1] = xmax; b[2] = ymin; b[3] = ymax;
}
static inline void hist_block(quadtree *T, const double *Y, int64_t lo, int64_t hi,
                              int32_t *h) {
  memset(h, 0, (size_t)T->nleaf * sizeof(int32_t));
  for (int64_t i = lo; i < hi; i++) {
    uint32_t c = leaf_code(T, Y[2 * i], Y[2 * i + 1]);
    T->code[i] = c; h[c]++;
  }
}
static inline int64_t leaf_totals(const quadtree *T, const int32_t *hist, int nb,
                                  int64_t c0, int64_t c1) {
  int64_t s = 0;
  for (int b = 0; b < nb; b++) {
    const int32_t *h = hist + (size_t)b * T->nleaf;
    for (int64_t c = c0; c < c1; c++) s += h[c];
  }
  return s;
}
static inline void leaf_offsets(quadtree *T, int32_t *hist, int nb,
                                int64_t c0, int64_t c1, int64_t run) {
  for (int64_t c = c0; c < c1; c++) {
    T->lstart[c] = run;
    for (int b = 0; b < nb; b++) {
      int32_t *h = hist + (size_t)b * T->nleaf;
      int32_t cnt = h[c]; h[c] = (int32_t)run; run += cnt;
    }
  }
}
static inline void scatter_block(quadtree *T, const double *Y, int64_t lo, int64_t hi,
                                 int32_t *h) {
  for (int64_t i = lo; i < hi; i++) {
    int64_t s = h[T->code[i]]++;
    T->perm[s] = (int32_t)i;
    T->ys[2 * s] = Y[2 * i]; T->ys[2 * s + 1] = Y[2 * i + 1];
  }
}

/* --- centers of mass --- */
static inline void leaf_com_range(quadtree *T, int64_t c0, int64_t c1) {
  for (int64_t c = c0; c < c1; c++) leaf_com(T, c);
}
static inline void parent_com_range(quadtree *T, int l, int64_t c0, int64_t c1) {
  for (int64_t c = c0; c < c1; c++) parent_com(T, l, c);
}

/* --- the walk: chunk ch = sorted points ch*WALK_CHUNK ...; writes the
 * unnormalized repulsion of each point and the chunk's share of Z. --- */
static inline void walk_chunk(const quadtree *T, double theta2, int64_t ch,
                              double *Frep, double *zchunk) {
  const int64_t lo = ch * WALK_CHUNK, hi = imin64(lo + WALK_CHUNK, T->n);
  double zc = 0.0;
  for (int64_t s = lo; s < hi; s++) {
    double fx, fy, z;
    walk_point(T, theta2, s, &fx, &fy, &z);
    int32_t i = T->perm[s];
    Frep[2 * i] = fx; Frep[2 * i + 1] = fy;
    zc += z;
  }
  zchunk[ch] = zc;
}
static inline void direct_chunk(const double *Y, int32_t n, int64_t ch,
                                double *Frep, double *zchunk) {
  const int64_t lo = ch * WALK_CHUNK, hi = imin64(lo + WALK_CHUNK, n);
  double zc = 0.0;
  for (int64_t i = lo; i < hi; i++) {
    double fx, fy, z;
    direct_point(Y, n, (int32_t)i, &fx, &fy, &z);
    Frep[2 * i] = fx; Frep[2 * i + 1] = fy;
    zc += z;
  }
  zchunk[ch] = zc;
}
static inline int64_t nchunks(int64_t n, int64_t chunk) { return (n + chunk - 1) / chunk; }

/* Z = sum of the chunk sums, always in the same order (cheap: n/256 terms) */
static inline double sum_chunks(const double *zchunk, int64_t nch) {
  double Z = 0.0;
  for (int64_t c = 0; c < nch; c++) Z += zchunk[c];
  return Z;
}
static inline void grad_range(const double *Fa, const double *Fr, double *g,
                              int64_t lo, int64_t hi) {
  for (int64_t k = lo; k < hi; k++) g[k] = 4.0 * (Fa[k] - Fr[k]);
}
static inline void scale_range(double *F, double a, int64_t lo, int64_t hi) {
  for (int64_t k = lo; k < hi; k++) F[k] *= a;
}

#endif
