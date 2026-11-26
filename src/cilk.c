/* cilk.c -- OpenCilk version.
 *
 * cilk_for splits a loop recursively in halves; idle workers steal halves
 * from busy ones (work stealing), so uneven rows/points are balanced
 * automatically. The grainsize pragma sets the smallest piece, the same
 * chunk sizes as the other versions. The tree build uses one block of
 * points per worker, as the others do.
 * The number of workers is set with CILK_NWORKERS, before the runtime starts.
 */
#include "kernels.h"
#include <cilk/cilk.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const char *backend_name = "cilk";
void backend_init(int p) {
  char s[16];
  snprintf(s, sizeof s, "%d", p);
  setenv("CILK_NWORKERS", s, 1);       /* read when the runtime starts */
}

#define PAR_MIN 4096

void compute_P(graph_csr *G, double u) {
  #pragma cilk grainsize 1
  cilk_for (int64_t c = 0; c < nchunks(G->n, ROW_CHUNK); c++)
    P_rows(G, u, c * ROW_CHUNK, imin64((c + 1) * ROW_CHUNK, G->n));
  /* a cilk_for ends with a sync: all p_{j|i} are ready */
  #pragma cilk grainsize 1
  cilk_for (int64_t c = 0; c < nchunks(G->n, ROW_CHUNK); c++)
    sym_rows(G, c * ROW_CHUNK, imin64((c + 1) * ROW_CHUNK, G->n));
}

void attraction(const graph_csr *G, const double *Y, double *F) {
  #pragma cilk grainsize 1
  cilk_for (int64_t c = 0; c < nchunks(G->n, ROW_CHUNK); c++)
    attr_rows(G, Y, F, c * ROW_CHUNK, imin64((c + 1) * ROW_CHUNK, G->n));
}

void tree_build(quadtree *T, const double *Y) {
  const int64_t n = T->n;
  const int nb = nthreads;
  const int64_t nlb = nchunks(T->nleaf, LEAF_BLOCK);
  int32_t *hist = tree_hist(T, nb);
  double *bb = malloc(4 * (size_t)nb * sizeof(double));
  int64_t *lbsum = malloc((nlb + 1) * sizeof(int64_t));

  /* 1. bounding box */
  #pragma cilk grainsize 1
  cilk_for (int b = 0; b < nb; b++) {
    int64_t lo, hi; block_range(n, nb, b, &lo, &hi);
    bbox_block(Y, lo, hi, bb + 4 * b);
  }
  double xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
  for (int b = 0; b < nb; b++) {
    xmin = fmin(xmin, bb[4 * b]);     xmax = fmax(xmax, bb[4 * b + 1]);
    ymin = fmin(ymin, bb[4 * b + 2]); ymax = fmax(ymax, bb[4 * b + 3]);
  }
  tree_set_box(T, xmin, xmax, ymin, ymax);
  /* 2. histograms */
  #pragma cilk grainsize 1
  cilk_for (int b = 0; b < nb; b++) {
    int64_t lo, hi; block_range(n, nb, b, &lo, &hi);
    hist_block(T, Y, lo, hi, hist + (size_t)b * T->nleaf);
  }
  /* 3. totals per block of leaves, then a short scan */
  #pragma cilk grainsize 1
  cilk_for (int64_t lb = 0; lb < nlb; lb++)
    lbsum[lb] = leaf_totals(T, hist, nb, lb * LEAF_BLOCK,
                            imin64((lb + 1) * LEAF_BLOCK, T->nleaf));
  int64_t run = 0;
  for (int64_t lb = 0; lb < nlb; lb++) { int64_t t = lbsum[lb]; lbsum[lb] = run; run += t; }
  T->lstart[T->nleaf] = run;
  /* 4. offsets */
  #pragma cilk grainsize 1
  cilk_for (int64_t lb = 0; lb < nlb; lb++)
    leaf_offsets(T, hist, nb, lb * LEAF_BLOCK,
                 imin64((lb + 1) * LEAF_BLOCK, T->nleaf), lbsum[lb]);
  /* 5. scatter */
  #pragma cilk grainsize 1
  cilk_for (int b = 0; b < nb; b++) {
    int64_t lo, hi; block_range(n, nb, b, &lo, &hi);
    scatter_block(T, Y, lo, hi, hist + (size_t)b * T->nleaf);
  }
  free(bb); free(lbsum);
}

void tree_com(quadtree *T) {
  #pragma cilk grainsize 1
  cilk_for (int64_t lb = 0; lb < nchunks(T->nleaf, LEAF_BLOCK); lb++)
    leaf_com_range(T, lb * LEAF_BLOCK, imin64((lb + 1) * LEAF_BLOCK, T->nleaf));
  for (int l = T->L - 1; l >= 0; l--) {
    const int64_t ncell = (int64_t)1 << (2 * l);
    if (ncell >= PAR_MIN) {
      #pragma cilk grainsize 1
      cilk_for (int64_t cb = 0; cb < nchunks(ncell, LEAF_BLOCK); cb++)
        parent_com_range(T, l, cb * LEAF_BLOCK, imin64((cb + 1) * LEAF_BLOCK, ncell));
    } else {
      parent_com_range(T, l, 0, ncell);
    }
  }
}

static double finish(double *zc, int64_t nch, double *F, int64_t m) {
  double Z = sum_chunks(zc, nch);
  const double a = 1.0 / Z;
  cilk_for (int64_t k = 0; k < m; k++) F[k] *= a;
  return Z;
}

double tree_walk(const quadtree *T, double theta, double *Frep) {
  const int64_t nch = nchunks(T->n, WALK_CHUNK);
  double *zc = malloc(nch * sizeof(double));
  const double th2 = theta * theta;
  #pragma cilk grainsize 1
  cilk_for (int64_t ch = 0; ch < nch; ch++) walk_chunk(T, th2, ch, Frep, zc);
  double Z = finish(zc, nch, Frep, 2 * (int64_t)T->n);
  free(zc);
  return Z;
}

double direct_repulsion(const double *Y, int32_t n, double *Frep) {
  const int64_t nch = nchunks(n, WALK_CHUNK);
  double *zc = malloc(nch * sizeof(double));
  #pragma cilk grainsize 1
  cilk_for (int64_t ch = 0; ch < nch; ch++) direct_chunk(Y, n, ch, Frep, zc);
  double Z = finish(zc, nch, Frep, 2 * (int64_t)n);
  free(zc);
  return Z;
}

void gradient(const double *Fa, const double *Fr, double *g, int64_t m) {
  cilk_for (int64_t k = 0; k < m; k++) g[k] = 4.0 * (Fa[k] - Fr[k]);
}
