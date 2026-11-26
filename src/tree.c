/* tree.c -- quadtree memory and geometry, shared by all versions. */
#include "tsne.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void tree_alloc(quadtree *T, int32_t n, int L) {
  if (L < 1 || L > 15) { fprintf(stderr, "depth L must be 1..15\n"); exit(1); }
  T->L = L; T->n = n;
  T->nleaf = (int64_t)1 << (2 * L);
  int64_t total = 0;
  for (int l = 0; l <= L; l++) { T->off[l] = total; total += (int64_t)1 << (2 * l); }
  T->lstart = malloc((T->nleaf + 1) * sizeof(int64_t));
  T->code   = malloc((size_t)n * sizeof(uint32_t));
  T->perm   = malloc((size_t)n * sizeof(int32_t));
  T->ys     = malloc(2 * (size_t)n * sizeof(double));
  T->cells  = malloc((size_t)total * sizeof(cell_t));
  T->hist = NULL; T->hist_nb = 0;
  if (!T->lstart || !T->code || !T->perm || !T->ys || !T->cells) {
    fprintf(stderr, "out of memory for the tree (L = %d)\n", L); exit(1);
  }
}

void tree_free(quadtree *T) {
  free(T->lstart); free(T->code); free(T->perm); free(T->ys); free(T->cells);
  free(T->hist);
}

/* The bounding square (slightly enlarged so the max point falls inside)
 * and the squared cell diagonal at every level. */
void tree_set_box(quadtree *T, double xmin, double xmax, double ymin, double ymax) {
  double side = fmax(xmax - xmin, ymax - ymin) * (1.0 + 1e-9) + 1e-12;
  T->x0 = xmin; T->y0 = ymin; T->side = side;
  for (int l = 0; l <= T->L; l++) {
    double s = side / (double)(1 << l);
    T->diag2[l] = 2.0 * s * s;
  }
}

/* Work array for the parallel counting sort, allocated once and reused
 * (allocating it in every build would be timed as part of the build). */
int32_t *tree_hist(quadtree *T, int nb) {
  if (nb > T->hist_nb) {
    free(T->hist);
    T->hist = malloc((size_t)nb * (size_t)T->nleaf * sizeof(int32_t));
    if (!T->hist) { fprintf(stderr, "out of memory for the histograms\n"); exit(1); }
    T->hist_nb = nb;
  }
  return T->hist;
}
