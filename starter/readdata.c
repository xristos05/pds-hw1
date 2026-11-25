/* readdata.c -- load a graph, its given state, and the reference values,
 * and print a summary. Usage: ./readdata data/k49_norm_10NN */
#include "data.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: %s data/<graph>\n", argv[0]); return 1; }
  graph_t W; state_t S; ref_t R;
  if (read_graph(argv[1], &W) || read_state(argv[1], &S) || read_ref(argv[1], &R)) return 1;
  int64_t dmax = 0;
  for (int64_t j = 0; j < W.n; j++) {
    int64_t d = W.colptr[j + 1] - W.colptr[j];
    if (d > dmax) dmax = d;
  }
  printf("%s: n = %lld, nnz = %lld (max degree %lld)\n", argv[1],
         (long long)W.n, (long long)W.nnz, (long long)dmax);
  printf("  given state: %lld points, u = %g, theta = %g; y_1 = (%.6g, %.6g)\n",
         (long long)S.n, S.u, S.theta, S.Y0[0], S.Y0[1]);
  printf("  reference: Z = %.10g, beta_1 = %.6g, Fattr_1 = (%.6g, %.6g), Frep_1 = (%.6g, %.6g)\n",
         R.Z, R.beta[0], R.Fattr[0], R.Fattr[1], R.Frep[0], R.Frep[1]);
  if (S.n != W.n || R.n != W.n) { fprintf(stderr, "sizes do not match\n"); return 1; }
  free_graph(&W); free(S.Y0); free(R.beta); free(R.Fattr); free(R.Frep);
  return 0;
}
