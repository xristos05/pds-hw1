/* data.c -- see data.h. */
#include "data.h"
#include <matio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mat_t *open_mat(const char *base, const char *suffix) {
  char path[4096];
  snprintf(path, sizeof path, "%s%s.mat", base, suffix);
  mat_t *f = Mat_Open(path, MAT_ACC_RDONLY);
  if (!f) fprintf(stderr, "cannot open %s\n", path);
  return f;
}

/* read a real variable into a new array of doubles; returns its length */
static double *read_doubles(mat_t *f, const char *name, size_t *len) {
  matvar_t *v = Mat_VarRead(f, name);
  if (!v || v->isComplex || v->class_type == MAT_C_SPARSE) {
    fprintf(stderr, "variable %s: missing or not a real dense array\n", name);
    if (v) Mat_VarFree(v);
    return NULL;
  }
  size_t n = 1;
  for (int d = 0; d < v->rank; d++) n *= v->dims[d];
  double *x = malloc(n * sizeof *x);
  for (size_t k = 0; k < n; k++) {
    switch (v->data_type) {
      case MAT_T_DOUBLE: x[k] = ((double *)v->data)[k]; break;
      case MAT_T_INT64:  x[k] = (double)((int64_t *)v->data)[k]; break;
      case MAT_T_UINT64: x[k] = (double)((uint64_t *)v->data)[k]; break;
      case MAT_T_INT32:  x[k] = (double)((int32_t *)v->data)[k]; break;
      default: fprintf(stderr, "variable %s: unexpected type %d\n", name, v->data_type);
               free(x); Mat_VarFree(v); return NULL;
    }
  }
  *len = n;
  Mat_VarFree(v);
  return x;
}

int read_graph(const char *base, graph_t *W) {
  mat_t *f = open_mat(base, "");
  if (!f) return -1;
  matvar_t *v = Mat_VarRead(f, "A");
  if (!v || v->class_type != MAT_C_SPARSE || v->data_type != MAT_T_DOUBLE) {
    fprintf(stderr, "%s: A is not a real sparse matrix\n", base);
    Mat_Close(f); return -1;
  }
  mat_sparse_t *s = v->data;
  W->n = (int64_t)v->dims[1];
  W->nnz = (int64_t)s->jc[W->n];
  W->colptr = malloc((W->n + 1) * sizeof(int64_t));
  W->row = malloc(W->nnz * sizeof(int64_t));
  W->val = malloc(W->nnz * sizeof(double));
  for (int64_t j = 0; j <= W->n; j++) W->colptr[j] = s->jc[j];
  for (int64_t k = 0; k < W->nnz; k++) W->row[k] = s->ir[k];
  memcpy(W->val, s->data, W->nnz * sizeof(double));
  Mat_VarFree(v);
  size_t len;
  double *lab = read_doubles(f, "label", &len);
  W->label = malloc(W->n * sizeof(int64_t));
  for (int64_t i = 0; i < W->n; i++) W->label[i] = lab ? (int64_t)lab[i] : 0;
  free(lab);
  Mat_Close(f);
  return 0;
}

int read_state(const char *base, state_t *S) {
  mat_t *f = open_mat(base, "-Y0");
  if (!f) return -1;
  size_t len, one;
  S->Y0 = read_doubles(f, "Y0", &len);
  double *u = read_doubles(f, "u", &one), *t = read_doubles(f, "theta", &one);
  Mat_Close(f);
  if (!S->Y0 || !u || !t) return -1;
  S->n = (int64_t)(len / 2); S->u = *u; S->theta = *t;
  free(u); free(t);
  return 0;
}

int read_ref(const char *base, ref_t *R) {
  mat_t *f = open_mat(base, "-ref");
  if (!f) return -1;
  size_t len, one;
  R->beta  = read_doubles(f, "beta", &len);
  R->Fattr = read_doubles(f, "Fattr", &len);
  R->Frep  = read_doubles(f, "Frep", &len);
  double *z = read_doubles(f, "Z", &one);
  Mat_Close(f);
  if (!R->beta || !R->Fattr || !R->Frep || !z) return -1;
  R->n = (int64_t)(len / 2); R->Z = *z; free(z);
  return 0;
}

void free_graph(graph_t *W) {
  free(W->colptr); free(W->row); free(W->val); free(W->label);
}
