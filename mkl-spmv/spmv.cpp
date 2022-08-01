/*
Assumes it is in CSR format
*/

#include "spmv.h"
#include <chrono>
#include <cmath>
#include <mkl.h>
#include <mkl_spblas.h>
#include <iostream>
#include <omp.h>

#if __AVX512F__ == 1
#pragma message "*** AVX512 enabled ***"
#else
#pragma message "*** AVX512 disabled ***"
#endif

//#define DO_WARMUP

void my_read_csr(char *fname, int *m, int *n, int *nnz, int **row_start,
                 int **col_idx, float **val) {
  FILE *fp = fopen(fname, "r");
  if (fp == NULL) {
    printf("Error\n");
  }

  fscanf(fp, "%d %d  %d \n", m, n, nnz);

  // malloc memory for vectors
  *row_start = (int *)malloc((*m + 1) * sizeof(int));
  *col_idx = (int *)malloc(*nnz * sizeof(int));
  *val = (float *)malloc(*nnz * sizeof(float));

  if (*row_start == NULL || *col_idx == NULL || *val == NULL) {
    printf("Error, Malloc Matrix Memory\n");
    return;
  }

  // first the row_start
  int i;
  for (i = 0; i < *m + 1; ++i) {
    int temp;
    fscanf(fp, "%d ", &temp);
    (*row_start)[i] = temp;
  }

  // 2 col_idx
  for (i = 0; i < *nnz; ++i) {
    int temp;
    fscanf(fp, "%d ", &temp);
    (*col_idx)[i] = temp;
  }

  // 3 vals
  for (i = 0; i < *nnz; ++i) {
    float temp;
    fscanf(fp, "%f ", &temp);
    (*val)[i] = temp;
  }

  fclose(fp);

} // read_csr

void make_vector(int m, float **x, float val) {
  *x = (float *)malloc(m * sizeof(float));

  int i;

#pragma omp parallel for schedule(runtime)
  for (i = 0; i < m; ++i) {
    (*x)[i] = val;
  }
}

void test_spmv(int m, int n, int nnz, int *row_start, int *col_idx, float *val,
               float *x, float *y) {
  int row;

  for (row = 0; row < m; ++row) {
    int val_idx;
    float temp = 0;

    for (val_idx = row_start[row]; val_idx < row_start[row + 1]; ++val_idx) {
      temp += val[val_idx] * x[col_idx[val_idx]];
    }

    y[row] = temp;
  }
} // test_spmv

int main(int argc, char *argv[]) {

  if (argc < 2) {
    printf("./spmv.exe inputfie.csr num_runs\n");
    exit(0);
  }

  int m, n, nnz;
  int *row_ptr, *col_idx;
  float *val;

  printf("Before Read\n");
  my_read_csr(argv[1], &m, &n, &nnz, &row_ptr, &col_idx, &val);
  printf("After Read\n");

  float *x, *y;
  printf("---------BEFORE FIRST FILL---------\n");
  make_vector(m, &x, 1.0);

  printf("-----------------BEFORE SECOND FILL-----------\n");
  make_vector(m, &y, 0.0);

  printf("------------------BEFORE THIRD FILL-----------\n");
  float *yhat;
  make_vector(m, &yhat, 0.0);

  printf("--------------END THIRD-----------\n");

  int N = atoi(argv[2]);
  int i;

  test_spmv(m, n, nnz, row_ptr, col_idx, val, x, yhat);

  printf("-----------After test_spmv--------\n");

  float min = 9999.0;
  float max = 0.0;
  float avg = 0.0;

  sparse_matrix_t matrix;
  mkl_sparse_s_create_csr(&matrix, SPARSE_INDEX_BASE_ZERO, m, n, &row_ptr[0],
                          &row_ptr[1], col_idx, val);
  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;

// do 5 warmup runs with mkl_sparse_s_mv
#ifdef DO_WARMUP
  for (i = 0; i < 5; ++i) {
    mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrix, descr, x, 0.0,
                    y);
  }
#endif

  for (i = 0; i < N; ++i) {
    auto tic = std::chrono::steady_clock::now();
    mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrix, descr, x, 0.0,
                    y);
    std::chrono::duration<float> toc = std::chrono::steady_clock::now() - tic;
    float tock = toc.count();
    std::cout << i + 1 << "th iteration yielded " << tock << std::endl;
    avg += tock;
    if (tock < min) {
      min = tock;
    }
    if (tock > max) {
      max = tock;
    }
  }

  printf("TimeMin: %lg\n", min);
  printf("TimeMax: %lg\n", max);
  printf("TimeAvg: %lg\n", avg / N);

  printf("TEAST\n");

  // Check solution

  int num_wrong = 0;
  for (i = 0; i < m; ++i) {
    float temp = y[i] - yhat[i];
    if (temp > .01 || temp < -.01) {
      printf("wrong loc: %d \n", i);
      printf("y: %f yhat: %f \n", y[i], yhat[i]);
      num_wrong++;
    }
  }

  printf("Number Wrong: %d \n", num_wrong);

  free(x);
  free(y);
  free(yhat);
  free(row_ptr);
  free(col_idx);
  free(val);

  exit(0);

  return 0;
}
