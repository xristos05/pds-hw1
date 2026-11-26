/* pth.c -- Pthreads version.
 *
 * Everything goes through two helpers:
 *   par_dynamic(nitems, body, arg): p threads take items (chunks) one at a
 *     time from a shared counter (an atomic fetch-and-add): dynamic load
 *     balancing, as OpenMP's schedule(dynamic).
 *   par_blocks(body, arg): thread b runs body(b), b = 0..p-1: one block each.
 * Threads are created at each parallel loop and joined at its end; the
 * join is the barrier between phases (e.g. between computing all p_{j|i}
 * and symmetrizing). Creating 16 threads costs some tens of microseconds,
 * small next to every phase except the top levels of the tree, which run
 * sequentially.
 */
#include "kernels.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>

const char *backend_name = "pth";
void backend_init(int p) { (void)p; }   /* nthreads is read directly */

#define PAR_MIN 4096

/* ------------------------------------------------------------ helpers --- */
typedef void (*item_fn)(int64_t item, void *arg);
typedef struct {
  item_fn  body;
  void    *arg;
  int64_t  nitems;
  int64_t  next;        /* shared counter: the next item to take */
} dyn_job;

static void *dyn_worker(void *p) {
  dyn_job *J = p;
  for (;;) {
    int64_t it = __atomic_fetch_add(&J->next, 1, __ATOMIC_RELAXED);
    if (it >= J->nitems) break;
    J->body(it, J->arg);
  }
  return NULL;
}

static void par_dynamic(int64_t nitems, item_fn body, void *arg) {
  dyn_job J = {body, arg, nitems, 0};
  int p = nthreads;
  if (p > nitems) p = (int)(nitems > 0 ? nitems : 1);
  pthread_t th[p];
  for (int t = 1; t < p; t++) pthread_create(&th[t], NULL, dyn_worker, &J);
  dyn_worker(&J);                                  /* the main thread works too */
  for (int t = 1; t < p; t++) pthread_join(th[t], NULL);
}

typedef void (*block_fn)(int b, void *arg);
typedef struct { block_fn body; void *arg; int b; } blk_job;

static void *blk_worker(void *p) { blk_job *J = p; J->body(J->b, J->arg); return NULL; }

static void par_blocks(block_fn body, void *arg) {
  int p = nthreads;
  pthread_t th[p];
  blk_job J[p];
  for (int b = 0; b < p; b++) { J[b].body = body; J[b].arg = arg; J[b].b = b; }
  for (int b = 1; b < p; b++) pthread_create(&th[b], NULL, blk_worker, &J[b]);
  blk_worker(&J[0]);
  for (int b = 1; b < p; b++) pthread_join(th[b], NULL);
}

static int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

/* ----------------------------------------------------------------- P --- */
typedef struct { graph_csr *G; double u; const double *Y; double *F; } g_arg;

static void P_item(int64_t c, void *a) {
  g_arg *A = a;
  P_rows(A->G, A->u, c * ROW_CHUNK, imin64((c + 1) * ROW_CHUNK, A->G->n));
}
static void sym_item(int64_t c, void *a) {
  g_arg *A = a;
  sym_rows(A->G, c * ROW_CHUNK, imin64((c + 1) * ROW_CHUNK, A->G->n));
}
void compute_P(graph_csr *G, double u) {
  g_arg A = {G, u, NULL, NULL};
  int64_t nch = ceil_div(G->n, ROW_CHUNK);
  par_dynamic(nch, P_item, &A);
  par_dynamic(nch, sym_item, &A);   /* after the join: all p_{j|i} are ready */
}

/* -------------------------------------------------------- attraction --- */
static void attr_item(int64_t c, void *a) {
  g_arg *A = a;
  attr_rows(A->G, A->Y, A->F, c * ROW_CHUNK, imin64((c + 1) * ROW_CHUNK, A->G->n));
}
void attraction(const graph_csr *G, const double *Y, double *F) {
  g_arg A = {(graph_csr *)G, 0.0, Y, F};
  par_dynamic(ceil_div(G->n, ROW_CHUNK), attr_item, &A);
}

/* -------------------------------------------------------- tree build --- */
typedef struct {
  quadtree *T; const double *Y; int nb; int32_t *hist;
  double *bb; int64_t *lbsum; int64_t nlb;
} b_arg;

static void bbox_blk(int b, void *a) {
  b_arg *A = a; int64_t lo, hi;
  block_range(A->T->n, A->nb, b, &lo, &hi);
  bbox_block(A->Y, lo, hi, A->bb + 4 * b);
}
static void hist_blk(int b, void *a) {
  b_arg *A = a; int64_t lo, hi;
  block_range(A->T->n, A->nb, b, &lo, &hi);
  hist_block(A->T, A->Y, lo, hi, A->hist + (size_t)b * A->T->nleaf);
}
static void totals_item(int64_t lb, void *a) {
  b_arg *A = a;
  A->lbsum[lb] = leaf_totals(A->T, A->hist, A->nb, lb * LEAF_BLOCK,
                             imin64((lb + 1) * LEAF_BLOCK, A->T->nleaf));
}
static void offsets_item(int64_t lb, void *a) {
  b_arg *A = a;
  leaf_offsets(A->T, A->hist, A->nb, lb * LEAF_BLOCK,
               imin64((lb + 1) * LEAF_BLOCK, A->T->nleaf), A->lbsum[lb]);
}
static void scatter_blk(int b, void *a) {
  b_arg *A = a; int64_t lo, hi;
  block_range(A->T->n, A->nb, b, &lo, &hi);
  scatter_block(A->T, A->Y, lo, hi, A->hist + (size_t)b * A->T->nleaf);
}

void tree_build(quadtree *T, const double *Y) {
  b_arg A;
  A.T = T; A.Y = Y; A.nb = nthreads;
  A.hist = tree_hist(T, A.nb);
  A.nlb = ceil_div(T->nleaf, LEAF_BLOCK);
  A.bb = malloc(4 * (size_t)A.nb * sizeof(double));
  A.lbsum = malloc((A.nlb + 1) * sizeof(int64_t));

  par_blocks(bbox_blk, &A);                                    /* 1 */
  double xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
  for (int b = 0; b < A.nb; b++) {
    xmin = fmin(xmin, A.bb[4 * b]);     xmax = fmax(xmax, A.bb[4 * b + 1]);
    ymin = fmin(ymin, A.bb[4 * b + 2]); ymax = fmax(ymax, A.bb[4 * b + 3]);
  }
  tree_set_box(T, xmin, xmax, ymin, ymax);
  par_blocks(hist_blk, &A);                                    /* 2 */
  par_dynamic(A.nlb, totals_item, &A);                         /* 3 */
  int64_t run = 0;
  for (int64_t lb = 0; lb < A.nlb; lb++) { int64_t t = A.lbsum[lb]; A.lbsum[lb] = run; run += t; }
  T->lstart[T->nleaf] = run;
  par_dynamic(A.nlb, offsets_item, &A);                        /* 4 */
  par_blocks(scatter_blk, &A);                                 /* 5 */
  free(A.bb); free(A.lbsum);
}

/* --------------------------------------------------- centers of mass --- */
typedef struct { quadtree *T; int l; int64_t ncell; } c_arg;

static void leafcom_item(int64_t lb, void *a) {
  c_arg *A = a;
  leaf_com_range(A->T, lb * LEAF_BLOCK, imin64((lb + 1) * LEAF_BLOCK, A->T->nleaf));
}
static void parent_item(int64_t cb, void *a) {
  c_arg *A = a;
  parent_com_range(A->T, A->l, cb * LEAF_BLOCK, imin64((cb + 1) * LEAF_BLOCK, A->ncell));
}
void tree_com(quadtree *T) {
  c_arg A = {T, T->L, T->nleaf};
  par_dynamic(ceil_div(T->nleaf, LEAF_BLOCK), leafcom_item, &A);
  for (int l = T->L - 1; l >= 0; l--) {
    A.l = l; A.ncell = (int64_t)1 << (2 * l);
    if (A.ncell >= PAR_MIN) par_dynamic(ceil_div(A.ncell, LEAF_BLOCK), parent_item, &A);
    else parent_com_range(T, l, 0, A.ncell);
  }
}

/* -------------------------------------------------------------- walk --- */
typedef struct {
  const quadtree *T; const double *Y; int32_t n; double th2; double *F; double *zc;
} w_arg;

static void walk_item(int64_t ch, void *a) {
  w_arg *A = a; walk_chunk(A->T, A->th2, ch, A->F, A->zc);
}
static void direct_item(int64_t ch, void *a) {
  w_arg *A = a; direct_chunk(A->Y, A->n, ch, A->F, A->zc);
}
typedef struct { double *F; double a; int64_t m; } s_arg;
static void scale_blk(int b, void *p) {
  s_arg *S = p; int64_t lo, hi;
  block_range(S->m, nthreads, b, &lo, &hi);
  scale_range(S->F, S->a, lo, hi);
}

static double finish(double *zc, int64_t nch, double *F, int64_t m) {
  double Z = sum_chunks(zc, nch);
  s_arg S = {F, 1.0 / Z, m};
  par_blocks(scale_blk, &S);
  return Z;
}

double tree_walk(const quadtree *T, double theta, double *Frep) {
  int64_t nch = nchunks(T->n, WALK_CHUNK);
  w_arg A = {T, NULL, T->n, theta * theta, Frep, malloc(nch * sizeof(double))};
  par_dynamic(nch, walk_item, &A);
  double Z = finish(A.zc, nch, Frep, 2 * (int64_t)T->n);
  free(A.zc);
  return Z;
}

double direct_repulsion(const double *Y, int32_t n, double *Frep) {
  int64_t nch = nchunks(n, WALK_CHUNK);
  w_arg A = {NULL, Y, n, 0.0, Frep, malloc(nch * sizeof(double))};
  par_dynamic(nch, direct_item, &A);
  double Z = finish(A.zc, nch, Frep, 2 * (int64_t)n);
  free(A.zc);
  return Z;
}

typedef struct { const double *Fa, *Fr; double *g; int64_t m; } g2_arg;
static void grad_blk(int b, void *p) {
  g2_arg *A = p; int64_t lo, hi;
  block_range(A->m, nthreads, b, &lo, &hi);
  grad_range(A->Fa, A->Fr, A->g, lo, hi);
}
void gradient(const double *Fa, const double *Fr, double *g, int64_t m) {
  g2_arg A = {Fa, Fr, g, m};
  par_blocks(grad_blk, &A);
}
