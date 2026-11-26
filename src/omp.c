/* omp.c -- OpenMP version.
 *
 * P, attraction, walk: rows/points have very different costs (degree 1 vs
 *   3,299; points in dense vs sparse regions), so chunks are handed out
 *   dynamically: schedule(dynamic, chunk).
 * Tree build: one block of points per thread (static), a stable parallel
 *   counting sort (see kernels.h).
 * Centers of mass: leaves in parallel, then each level in parallel (the top
 *   levels have few cells and run sequentially: not worth a fork/join).
 */
#include "kernels.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>

const char *backend_name = "omp";
void backend_init(int p) { omp_set_num_threads(p); }

#define PAR_MIN 4096   /* below this many cells, do a level sequentially */

void compute_P(graph_csr *G, double u) {
  const int64_t n = G->n;
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic, ROW_CHUNK)
    for (int64_t i = 0; i < n; i++) G->beta[i] = row_beta(G, (int32_t)i, u);
    /* implicit barrier: all p_{j|i} are ready before symmetrizing */
    #pragma omp for schedule(dynamic, ROW_CHUNK)
    for (int64_t i = 0; i < n; i++) row_symmetrize(G, (int32_t)i);
  }
}

void attraction(const graph_csr *G, const double *Y, double *F) {
  const int64_t n = G->n;
  #pragma omp parallel for schedule(dynamic, ROW_CHUNK)
  for (int64_t i = 0; i < n; i++) attr_row(G, Y, (int32_t)i, F);
}

void tree_build(quadtree *T, const double *Y) {
  const int64_t n = T->n;
  const int nb = nthreads;
  const int64_t nlb = nchunks(T->nleaf, LEAF_BLOCK);   /* blocks of leaves */
  int32_t *hist = tree_hist(T, nb);
  double *bb = malloc(4 * (size_t)nb * sizeof(double));
  int64_t *lbsum = malloc((nlb + 1) * sizeof(int64_t));

  /* 1. bounding box: per-block min/max, then combine */
  #pragma omp parallel for schedule(static, 1)
  for (int b = 0; b < nb; b++) {
    int64_t lo, hi; block_range(n, nb, b, &lo, &hi);
    bbox_block(Y, lo, hi, bb + 4 * b);
  }
  double xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
  for (int b = 0; b < nb; b++) {
    xmin = fmin(xmin, bb[4 * b]);     xmax = fmax(xmax, bb[4 * b + 1]);
    ymin = fmin(ymin, bb[4 * b + 2]); ymax = fmax(ymax, bb[4 * b + 3]);
  }
  tree_set_box(T, xmin, xmax, ymin, ymax);

  #pragma omp parallel
  {
    /* 2. leaf codes and per-block histograms */
    #pragma omp for schedule(static, 1)
    for (int b = 0; b < nb; b++) {
      int64_t lo, hi; block_range(n, nb, b, &lo, &hi);
      hist_block(T, Y, lo, hi, hist + (size_t)b * T->nleaf);
    }
    /* 3. number of points in each block of leaves */
    #pragma omp for schedule(static)
    for (int64_t lb = 0; lb < nlb; lb++)
      lbsum[lb] = leaf_totals(T, hist, nb, lb * LEAF_BLOCK,
                              imin64((lb + 1) * LEAF_BLOCK, T->nleaf));
    /* exclusive scan over the blocks of leaves (short: nleaf/4096 terms) */
    #pragma omp single
    {
      int64_t run = 0;
      for (int64_t lb = 0; lb < nlb; lb++) { int64_t t = lbsum[lb]; lbsum[lb] = run; run += t; }
      T->lstart[T->nleaf] = run;
    }
    /* 4. leaf starts, and each block's first slot in each leaf */
    #pragma omp for schedule(static)
    for (int64_t lb = 0; lb < nlb; lb++)
      leaf_offsets(T, hist, nb, lb * LEAF_BLOCK,
                   imin64((lb + 1) * LEAF_BLOCK, T->nleaf), lbsum[lb]);
    /* 5. scatter */
    #pragma omp for schedule(static, 1)
    for (int b = 0; b < nb; b++) {
      int64_t lo, hi; block_range(n, nb, b, &lo, &hi);
      scatter_block(T, Y, lo, hi, hist + (size_t)b * T->nleaf);
    }
  }
  free(bb); free(lbsum);
}

void tree_com(quadtree *T) {
  const int64_t nlb = nchunks(T->nleaf, LEAF_BLOCK);
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic, 1)
    for (int64_t lb = 0; lb < nlb; lb++)
      leaf_com_range(T, lb * LEAF_BLOCK, imin64((lb + 1) * LEAF_BLOCK, T->nleaf));
    for (int l = T->L - 1; l >= 0; l--) {          /* every thread runs this loop */
      const int64_t ncell = (int64_t)1 << (2 * l);
      if (ncell >= PAR_MIN) {
        #pragma omp for schedule(static)
        for (int64_t c = 0; c < ncell; c++) parent_com(T, l, c);
      } else {
        #pragma omp single
        parent_com_range(T, l, 0, ncell);
      }
      /* both end in an implicit barrier: level l is done before level l-1 */
    }
  }
}

double tree_walk(const quadtree *T, double theta, double *Frep) {
  const int64_t nch = nchunks(T->n, WALK_CHUNK);
  const int64_t m = 2 * (int64_t)T->n;
  double *zc = malloc(nch * sizeof(double));
  #pragma omp parallel for schedule(dynamic, 1)
  for (int64_t ch = 0; ch < nch; ch++) walk_chunk(T, theta * theta, ch, Frep, zc);
  double Z = sum_chunks(zc, nch);
  free(zc);
  const double a = 1.0 / Z;
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < m; k++) Frep[k] *= a;
  return Z;
}

double direct_repulsion(const double *Y, int32_t n, double *Frep) {
  const int64_t nch = nchunks(n, WALK_CHUNK);
  const int64_t m = 2 * (int64_t)n;
  double *zc = malloc(nch * sizeof(double));
  #pragma omp parallel for schedule(dynamic, 1)
  for (int64_t ch = 0; ch < nch; ch++) direct_chunk(Y, n, ch, Frep, zc);
  double Z = sum_chunks(zc, nch);
  free(zc);
  const double a = 1.0 / Z;
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < m; k++) Frep[k] *= a;
  return Z;
}

void gradient(const double *Fa, const double *Fr, double *g, int64_t m) {
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < m; k++) g[k] = 4.0 * (Fa[k] - Fr[k]);
}
