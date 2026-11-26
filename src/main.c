/* main.c -- driver, the same for all four versions.
 *
 *   ./tsne_<version> data/<graph> [options]
 *     -t P        number of threads (default: all hardware threads)
 *     -L L        quadtree depth, leaves form a 2^L x 2^L grid (default: auto)
 *     -theta X    opening angle (default: from <graph>-Y0.mat, 0.5)
 *     -r R        repetitions; each phase reports its fastest run (default 5)
 *     -direct     also compute the exact O(n^2) repulsion and time it
 *     -o DIR      save beta, Fattr, Frep, Z to DIR/<graph>-<version>.mat
 *     -csv FILE   append one line of timings to FILE
 *     -cmp FILE   compare with results saved by another version (e.g. the
 *                 sequential one): prints the largest differences
 *     -crossover FILE   time Barnes-Hut (build+com+walk) and the direct sum on
 *                 the first m points, m = 250, 500, 1000, ..., n; append to FILE
 */
#include "tsne.h"
#include "../starter/data.h"
#include <matio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int nthreads = 1;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* max |x - ref| / max |ref| */
static double rel_err(const double *x, const double *ref, int64_t m) {
  double num = 0.0, den = 0.0;
  for (int64_t k = 0; k < m; k++) {
    num = fmax(num, fabs(x[k] - ref[k]));
    den = fmax(den, fabs(ref[k]));
  }
  return den > 0 ? num / den : num;
}

static double max_abs_diff(const double *x, const double *ref, int64_t m) {
  double e = 0.0;
  for (int64_t k = 0; k < m; k++) e = fmax(e, fabs(x[k] - ref[k]));
  return e;
}

/* read a dense real variable of a .mat file (NULL if missing) */
static double *mat_read(mat_t *f, const char *name, size_t *len) {
  matvar_t *v = Mat_VarRead(f, name);
  if (!v || v->data_type != MAT_T_DOUBLE) { if (v) Mat_VarFree(v); return NULL; }
  size_t m = 1;
  for (int d = 0; d < v->rank; d++) m *= v->dims[d];
  double *x = malloc(m * sizeof(double));
  memcpy(x, v->data, m * sizeof(double));
  Mat_VarFree(v);
  *len = m;
  return x;
}

static void save_mat(const char *path, int32_t n, const double *beta,
                     const double *Fattr, const double *Frep, double Z) {
  mat_t *f = Mat_CreateVer(path, NULL, MAT_FT_MAT73);
  if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
  size_t d1[2] = {(size_t)n, 1}, d2[2] = {2, (size_t)n}, d0[2] = {1, 1};
  matvar_t *v;
  v = Mat_VarCreate("beta", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, d1, (void *)beta, MAT_F_DONT_COPY_DATA);
  Mat_VarWrite(f, v, MAT_COMPRESSION_ZLIB); Mat_VarFree(v);
  v = Mat_VarCreate("Fattr", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, d2, (void *)Fattr, MAT_F_DONT_COPY_DATA);
  Mat_VarWrite(f, v, MAT_COMPRESSION_ZLIB); Mat_VarFree(v);
  v = Mat_VarCreate("Frep", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, d2, (void *)Frep, MAT_F_DONT_COPY_DATA);
  Mat_VarWrite(f, v, MAT_COMPRESSION_ZLIB); Mat_VarFree(v);
  v = Mat_VarCreate("Z", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, d0, &Z, MAT_F_DONT_COPY_DATA);
  Mat_VarWrite(f, v, MAT_COMPRESSION_NONE); Mat_VarFree(v);
  Mat_Close(f);
}

/* default depth: about 4 points per leaf, 8 per occupied leaf (see README) */
static int auto_depth(int32_t n) {
  int L = (int)ceil(log((double)n / 4.0) / log(4.0));
  return L < 2 ? 2 : (L > 12 ? 12 : L);
}

/* Barnes-Hut vs direct on the first m points, m doubling up to n */
static void crossover(const char *file, const char *gname, const double *Y,
                      int32_t n, double theta, int reps) {
  FILE *f = fopen(file, "r");
  int need_header = (f == NULL);
  if (f) fclose(f);
  f = fopen(file, "a");
  if (!f) return;
  if (need_header) fprintf(f, "graph,version,threads,m,L,t_bh,t_direct\n");
  double *F = malloc(2 * (size_t)n * sizeof(double));
  printf("  crossover (m, L, Barnes-Hut s, direct s):\n");
  for (int64_t m = 250; ; m *= 2) {
    if (m > n) m = n;
    int L = auto_depth((int32_t)m);
    quadtree T;
    tree_alloc(&T, (int32_t)m, L);
    double tb = INFINITY, td = INFINITY;
    for (int r = 0; r < reps; r++) {
      double t0 = now();
      tree_build(&T, Y); tree_com(&T); tree_walk(&T, theta, F);
      double t1 = now();
      direct_repulsion(Y, (int32_t)m, F);
      double t2 = now();
      tb = fmin(tb, t1 - t0); td = fmin(td, t2 - t1);
    }
    tree_free(&T);
    printf("    %8lld  %2d  %.6f  %.6f\n", (long long)m, L, tb, td);
    fprintf(f, "%s,%s,%d,%lld,%d,%.7f,%.7f\n", gname, backend_name, nthreads, (long long)m, L, tb, td);
    if (m == n) break;
  }
  free(F);
  fclose(f);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s data/<graph> [-t P] [-L L] [-theta X] [-r R] "
                    "[-direct] [-o DIR] [-csv FILE]\n", argv[0]);
    return 1;
  }
  const char *base = argv[1];
  int L = 0, reps = 5, do_direct = 0;
  double theta = -1.0;
  const char *outdir = NULL, *csv = NULL, *cmpfile = NULL, *xfile = NULL;
  nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
  for (int a = 2; a < argc; a++) {
    if (!strcmp(argv[a], "-t") && a + 1 < argc) nthreads = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-L") && a + 1 < argc) L = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-theta") && a + 1 < argc) theta = atof(argv[++a]);
    else if (!strcmp(argv[a], "-r") && a + 1 < argc) reps = atoi(argv[++a]);
    else if (!strcmp(argv[a], "-direct")) do_direct = 1;
    else if (!strcmp(argv[a], "-o") && a + 1 < argc) outdir = argv[++a];
    else if (!strcmp(argv[a], "-csv") && a + 1 < argc) csv = argv[++a];
    else if (!strcmp(argv[a], "-cmp") && a + 1 < argc) cmpfile = argv[++a];
    else if (!strcmp(argv[a], "-crossover") && a + 1 < argc) xfile = argv[++a];
    else { fprintf(stderr, "unknown option %s\n", argv[a]); return 1; }
  }
  if (nthreads < 1 || !strcmp(backend_name, "seq")) nthreads = 1;
  if (reps < 1) reps = 1;
  backend_init(nthreads);

  /* ---- read (not timed) ---- */
  graph_t W; state_t S; ref_t R;
  if (read_graph(base, &W) || read_state(base, &S)) return 1;
  int have_ref = (read_ref(base, &R) == 0);
  if (theta < 0) theta = S.theta;
  const double u = S.u;

  graph_csr G;
  G.n = (int32_t)W.n; G.nnz = W.nnz;
  G.ptr = W.colptr;                       /* symmetric: column j = row j */
  G.col = malloc(W.nnz * sizeof(int32_t));
  for (int64_t k = 0; k < W.nnz; k++) G.col[k] = (int32_t)W.row[k];
  G.w = W.val;
  G.P = malloc(W.nnz * sizeof(double));
  G.cond = malloc(W.nnz * sizeof(double));
  G.beta = malloc(W.n * sizeof(double));
  const int32_t n = G.n;
  const double *Y = S.Y0;
  if (L <= 0) L = auto_depth(n);

  double *Fattr = malloc(2 * (size_t)n * sizeof(double));
  double *Frep  = malloc(2 * (size_t)n * sizeof(double));
  double *grad  = malloc(2 * (size_t)n * sizeof(double));
  quadtree T;
  tree_alloc(&T, n, L);

  /* ---- the phases, each timed; keep the fastest of reps runs ---- */
  enum { PH_P, PH_ATTR, PH_BUILD, PH_COM, PH_WALK, PH_GRAD, NPH };
  const char *phname[NPH] = {"P", "attr", "build", "com", "walk", "grad"};
  double best[NPH];
  for (int p = 0; p < NPH; p++) best[p] = INFINITY;
  double Z = 0.0;
  for (int r = 0; r < reps; r++) {
    double t[NPH + 1];
    t[0] = now();
    compute_P(&G, u);                         t[1] = now();
    attraction(&G, Y, Fattr);                 t[2] = now();
    tree_build(&T, Y);                        t[3] = now();
    tree_com(&T);                             t[4] = now();
    Z = tree_walk(&T, theta, Frep);           t[5] = now();
    gradient(Fattr, Frep, grad, 2 * (int64_t)n);
    t[6] = now();
    for (int p = 0; p < NPH; p++) best[p] = fmin(best[p], t[p + 1] - t[p]);
  }
  double total = 0.0;
  for (int p = 0; p < NPH; p++) total += best[p];

  /* ---- optional exact repulsion ---- */
  double t_direct = NAN, Zd = NAN;
  double *Fd = NULL;
  if (do_direct) {
    Fd = malloc(2 * (size_t)n * sizeof(double));
    t_direct = INFINITY;
    for (int r = 0; r < reps; r++) {
      double t0 = now();
      Zd = direct_repulsion(Y, n, Fd);
      t_direct = fmin(t_direct, now() - t0);
    }
  }

  /* ---- report ---- */
  const char *gname = strrchr(base, '/') ? strrchr(base, '/') + 1 : base;
  int64_t occupied = 0;
  for (int64_t c = 0; c < T.nleaf; c++) occupied += (T.lstart[c + 1] > T.lstart[c]);
  printf("%s [%s, %d threads]  n = %d, nnz = %lld, L = %d (%dx%d grid, %.1f points per occupied leaf), theta = %g, u = %g\n",
         gname, backend_name, nthreads, n, (long long)G.nnz, L, 1 << L, 1 << L,
         (double)n / (double)occupied, theta, u);
  printf("  time (s):");
  for (int p = 0; p < NPH; p++) printf("  %s %.4f", phname[p], best[p]);
  printf("  | total %.4f\n", total);
  if (do_direct) printf("  direct O(n^2) repulsion: %.4f s  (Barnes-Hut build+com+walk: %.4f s, %.1fx faster)\n",
                        t_direct, best[PH_BUILD] + best[PH_COM] + best[PH_WALK],
                        t_direct / (best[PH_BUILD] + best[PH_COM] + best[PH_WALK]));
  double eb = NAN, ea = NAN, er = NAN, ez = NAN;
  if (have_ref) {
    eb = max_abs_diff(G.beta, R.beta, n);
    ea = rel_err(Fattr, R.Fattr, 2 * (int64_t)n);
    er = rel_err(Frep, R.Frep, 2 * (int64_t)n);
    ez = fabs(Z - R.Z) / R.Z;
    printf("  vs reference: beta max|diff| %.2e   Fattr %.2e   Frep %.2e   Z %.2e  (Z = %.10g, ref %.10g)\n",
           eb, ea, er, ez, Z, R.Z);
    if (do_direct)
      printf("  direct vs reference:  Frep %.2e   Z %.2e\n",
             rel_err(Fd, R.Frep, 2 * (int64_t)n), fabs(Zd - R.Z) / R.Z);
  }

  if (cmpfile) {
    mat_t *f = Mat_Open(cmpfile, MAT_ACC_RDONLY);
    size_t m1, m2, m3, m4;
    double *cb = f ? mat_read(f, "beta", &m1) : NULL, *ca = f ? mat_read(f, "Fattr", &m2) : NULL;
    double *cr = f ? mat_read(f, "Frep", &m3) : NULL, *cz = f ? mat_read(f, "Z", &m4) : NULL;
    if (f) Mat_Close(f);
    if (!cb || !ca || !cr || !cz || m1 != (size_t)n) {
      fprintf(stderr, "cannot compare with %s\n", cmpfile);
    } else {
      double db = max_abs_diff(G.beta, cb, n), da = max_abs_diff(Fattr, ca, 2 * (int64_t)n);
      double dr = max_abs_diff(Frep, cr, 2 * (int64_t)n), dz = fabs(Z - cz[0]);
      printf("  vs %s: max|diff| beta %.1e  Fattr %.1e  Frep %.1e  Z %.1e  -> %s\n", cmpfile,
             db, da, dr, dz, (db == 0 && da == 0 && dr == 0 && dz == 0) ? "IDENTICAL" : "differ");
    }
    free(cb); free(ca); free(cr); free(cz);
  }

  if (csv) {
    FILE *f = fopen(csv, "r");
    int need_header = (f == NULL);
    if (f) fclose(f);
    f = fopen(csv, "a");
    if (f) {
      if (need_header)
        fprintf(f, "graph,version,threads,n,nnz,L,theta,t_P,t_attr,t_build,t_com,t_walk,t_grad,t_total,t_direct,err_beta,err_Fattr,err_Frep,err_Z\n");
      fprintf(f, "%s,%s,%d,%d,%lld,%d,%g,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3e,%.3e,%.3e,%.3e\n",
              gname, backend_name, nthreads, n, (long long)G.nnz, L, theta,
              best[0], best[1], best[2], best[3], best[4], best[5], total, t_direct,
              eb, ea, er, ez);
      fclose(f);
    }
  }
  if (outdir) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s-%s.mat", outdir, gname, backend_name);
    save_mat(path, n, G.beta, Fattr, Frep, Z);
    printf("  saved %s\n", path);
  }

  if (xfile) crossover(xfile, gname, Y, n, theta, reps);

  tree_free(&T);
  free(Fattr); free(Frep); free(grad); free(Fd);
  free(G.col); free(G.P); free(G.cond); free(G.beta);
  free_graph(&W); free(S.Y0);
  if (have_ref) { free(R.beta); free(R.Fattr); free(R.Frep); }
  return 0;
}
