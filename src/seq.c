/* seq.c -- the sequential version: the reference for the other three. */
#include "kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

const char *backend_name = "seq";
void backend_init(int p) { (void)p; }

void compute_P(graph_csr *G, double u) {
  P_rows(G, u, 0, G->n);
  sym_rows(G, 0, G->n);
}

void attraction(const graph_csr *G, const double *Y, double *F) {
  attr_rows(G, Y, F, 0, G->n);
}

/* Build: bounding square, leaf code of every point, counting sort by leaf. */
void tree_build(quadtree *T, const double *Y) {
  const int32_t n = T->n;
  double b[4];
  bbox_block(Y, 0, n, b);
  tree_set_box(T, b[0], b[1], b[2], b[3]);
  /* histogram of leaves (one block: the parallel versions use one per thread) */
  int32_t *h = tree_hist(T, 1);
  hist_block(T, Y, 0, n, h);
  /* prefix sum: lstart[c] = first sorted position of leaf c; h[c] = cursor */
  leaf_offsets(T, h, 1, 0, T->nleaf, 0);
  T->lstart[T->nleaf] = n;
  /* scatter (stable) */
  scatter_block(T, Y, 0, n, h);
}

/* Centers of mass: leaves from their points, then level by level up. */
void tree_com(quadtree *T) {
  leaf_com_range(T, 0, T->nleaf);
  for (int l = T->L - 1; l >= 0; l--)
    parent_com_range(T, l, 0, (int64_t)1 << (2 * l));
}

/* The walk, in sorted order (neighbors in the tree are neighbors in memory). */
double tree_walk(const quadtree *T, double theta, double *Frep) {
  const int64_t nch = nchunks(T->n, WALK_CHUNK);
  double *zc = malloc(nch * sizeof(double));
  for (int64_t ch = 0; ch < nch; ch++) walk_chunk(T, theta * theta, ch, Frep, zc);
  double Z = sum_chunks(zc, nch);
  free(zc);
  scale_range(Frep, 1.0 / Z, 0, 2 * (int64_t)T->n);
  return Z;
}

double direct_repulsion(const double *Y, int32_t n, double *Frep) {
  const int64_t nch = nchunks(n, WALK_CHUNK);
  double *zc = malloc(nch * sizeof(double));
  for (int64_t ch = 0; ch < nch; ch++) direct_chunk(Y, n, ch, Frep, zc);
  double Z = sum_chunks(zc, nch);
  free(zc);
  scale_range(Frep, 1.0 / Z, 0, 2 * (int64_t)n);
  return Z;
}

void gradient(const double *Fa, const double *Fr, double *g, int64_t m) {
  grad_range(Fa, Fr, g, 0, m);
}
