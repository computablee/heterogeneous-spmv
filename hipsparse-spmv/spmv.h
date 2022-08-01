/*
Assumes it is in CSR format
*/

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void my_read_csr(char *fname, int *m, int *n, int *nnz, int **row_start,
                 int **col_idx, double **v);

void omp_spmv(int m, int n, int nnz, int *row_start, int *col_idx, double *val,
              double *x, double *y);
