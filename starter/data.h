/* data.h -- read the Homework 1 data with matio (MAT v7.3, HDF5).
 *
 *   graph_t  W;   read_graph("data/k49_norm_10NN", &W);   W.mat
 *   state_t  S;   read_state("data/k49_norm_10NN", &S);   W-Y0.mat
 *   ref_t    R;   read_ref  ("data/k49_norm_10NN", &R);   W-ref.mat
 *
 * Sparse matrices are in compressed sparse column (CSC) form, 0-based:
 * the nonzeros of column j are val[k], at rows row[k], for k in
 * colptr[j] .. colptr[j+1]-1. W is symmetric, so column j is also row j.
 * Dense 2 x n matrices (Y0, Fattr, Frep) are stored column by column:
 * point i is (Y[2*i], Y[2*i+1]).
 */
#ifndef HW1_DATA_H
#define HW1_DATA_H
#include <stdint.h>

typedef struct {
  int64_t n, nnz;
  int64_t *colptr;   /* n + 1 */
  int64_t *row;      /* nnz   */
  double  *val;      /* nnz   */
  int64_t *label;    /* n, for pictures only (0: none) */
} graph_t;

typedef struct { int64_t n; double *Y0; double u, theta; } state_t;

typedef struct { int64_t n; double *beta, *Fattr, *Frep; double Z; } ref_t;

int  read_graph(const char *base, graph_t *W);
int  read_state(const char *base, state_t *S);
int  read_ref(const char *base, ref_t *R);
void free_graph(graph_t *W);

#endif
