/* tsne.h -- data structures and the interface every version implements.
 *
 * The program is split in two:
 *   main.c     reads the data, times each phase, checks against the
 *              reference, and saves the results. Same for all versions.
 *   seq.c      one "backend" per parallel model. Each implements the
 *   omp.c      functions below; only HOW the loops are split among
 *   pth.c      threads differs. The per-node work itself (one row of P,
 *   cilk.c     one row of the sparse product, one tree walk) is shared,
 *              in kernels.h, so all four versions compute the same thing.
 */
#ifndef TSNE_H
#define TSNE_H
#include <stdint.h>

/* The graph in CSR form (row i = column i, since W is symmetric).
 * Neighbors of node i: col[k], k = ptr[i] .. ptr[i+1]-1.
 * col is int32 (n < 2^31) to halve the memory traffic of the index array. */
typedef struct {
  int32_t  n;
  int64_t  nnz;
  int64_t *ptr;   /* n+1 */
  int32_t *col;   /* nnz */
  double  *w;     /* nnz, the affinities W */
  double  *P;     /* nnz, the symmetric joint probabilities p_ij */
  double  *cond;  /* nnz, work array: conditional p_{j|i} */
  double  *beta;  /* n */
} graph_csr;

/* One cell of the quadtree: number of points and their center of mass. */
typedef struct { double x, y, n; } cell_t;

/* Fixed-depth quadtree. Level l is a dense array of 4^l cells, indexed by
 * Morton (Z-order) code; the children of cell c at level l are cells
 * 4c..4c+3 at level l+1. The points are sorted by leaf (counting sort),
 * so the points of ANY cell are a contiguous range of the sorted order. */
typedef struct {
  int      L;          /* depth: leaves form a 2^L x 2^L grid */
  int32_t  n;
  double   x0, y0;     /* lower-left corner of the bounding square */
  double   side;       /* side of the bounding square */
  int64_t  nleaf;      /* 4^L */
  int64_t *lstart;     /* nleaf+1: points of leaf c are sorted[lstart[c] .. lstart[c+1]-1] */
  uint32_t *code;      /* n: leaf code of each point (original order) */
  int32_t *perm;       /* n: perm[s] = original index of the s-th sorted point */
  double  *ys;         /* 2n: coordinates in sorted order */
  cell_t  *cells;      /* all levels, level l starts at off[l] */
  int64_t  off[17];
  double   diag2[17];  /* squared diagonal of a cell at each level */
  int32_t *hist;       /* nb x nleaf: per-block leaf counts (parallel counting sort) */
  int      hist_nb;    /* number of blocks hist was allocated for */
} quadtree;

/* Fixed chunk sizes. Z is summed per chunk of points, then the chunk sums in
 * order, so every version adds in the same order and all four give
 * bit-identical results, whatever the number of threads. */
#define WALK_CHUNK   256     /* sorted points per chunk in the walk / direct sum */
#define ROW_CHUNK    256     /* rows per chunk in P and the attraction */
#define LEAF_BLOCK   4096    /* leaves per block in the tree build */

/* number of threads, set by main from -t (ignored by seq) */
extern int nthreads;
extern char **main_argv;   /* for the Cilk version, which may restart itself */

/* ---- the backend interface: implemented by seq.c, omp.c, pth.c, cilk.c ---- */
extern const char *backend_name;
void backend_init(int p);                                       /* set the thread count */
void compute_P(graph_csr *G, double u);                         /* beta, cond, P */
void attraction(const graph_csr *G, const double *Y, double *Fattr);
void tree_build(quadtree *T, const double *Y);                  /* bbox, sort by leaf */
void tree_com(quadtree *T);                                     /* counts, centers of mass */
double tree_walk(const quadtree *T, double theta, double *Frep);/* returns Z, Frep normalized */
double direct_repulsion(const double *Y, int32_t n, double *Frep); /* O(n^2), returns Z */
void gradient(const double *Fattr, const double *Frep, double *g, int64_t m); /* g = 4(Fattr - Frep) */

/* tree memory (shared by all backends, in tree.c) */
void tree_alloc(quadtree *T, int32_t n, int L);
void tree_free(quadtree *T);
void tree_set_box(quadtree *T, double xmin, double xmax, double ymin, double ymax);
int32_t *tree_hist(quadtree *T, int nb);   /* nb x nleaf work array */

#endif
