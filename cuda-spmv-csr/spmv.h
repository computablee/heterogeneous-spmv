/*
Assumes it is in CSR format
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void my_read_csr(char *fname, int *m, int *n, int *nnz,
	      int **row_start, int **col_idx, double **v);

void omp_spmv(int m, int n, int nnz,
	      int *row_start, int *col_idx, double *val,
	      double *x, double *y);
