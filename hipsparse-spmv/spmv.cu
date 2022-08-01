/*
Assumes it is in CSR format
*/

#include "spmv.h"
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include "hipsparse/hipsparse.h"

void my_read_csr(char *fname, int *m, int *n, int *nnz, int **row_start,
                 int **col_idx, float **val, int *m_gpu, int *n_gpu,
                 int *nnz_gpu, int **row_start_gpu, int **col_idx_gpu,
                 float **val_gpu) {
  FILE *fp = fopen(fname, "r");
  if (fp == NULL) {
    printf("Error\n");
  }

  fscanf(fp, "%d %d  %d \n", m, n, nnz);

  // malloc memory for vectors
  *row_start = (int *)malloc((*m + 1) * sizeof(int));
  *col_idx = (int *)malloc(*nnz * sizeof(int));
  *val = (float *)malloc(*nnz * sizeof(float));
  hipMalloc(row_start_gpu, (*m + 1) * sizeof(int));
  hipMalloc(col_idx_gpu, (*nnz + 4) * sizeof(int));
  hipMalloc(val_gpu, (*nnz + 4) * sizeof(float));

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

  hipMemcpy(*row_start_gpu, *row_start, (*m + 1) * sizeof(int),
             hipMemcpyHostToDevice);
  hipMemcpy(*col_idx_gpu, *col_idx, *nnz * sizeof(int),
             hipMemcpyHostToDevice);
  hipMemcpy(*val_gpu, *val, *nnz * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(m_gpu, m, sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(n_gpu, n, sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(nnz_gpu, nnz, sizeof(int), hipMemcpyHostToDevice);

} // read_csr

void make_vector(int m, float **x, float **x_gpu, float val) {
  *x = (float *)malloc(m * sizeof(float));

  if (x_gpu)
    hipMalloc(x_gpu, m * sizeof(float));

  int i;

#pragma omp parallel for schedule(runtime)
  for (i = 0; i < m; ++i) {
    (*x)[i] = val;
  }

  if (x_gpu)
    hipMemcpy(*x_gpu, *x, m * sizeof(float), hipMemcpyHostToDevice);
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
  int *m_gpu, *n_gpu, *nnz_gpu;
  int *row_ptr_gpu, *col_idx_gpu;
  float *val_gpu;

  hipMalloc(&m_gpu, sizeof(int));
  hipMalloc(&n_gpu, sizeof(int));
  hipMalloc(&nnz_gpu, sizeof(int));

  printf("Before Read\n");
  my_read_csr(argv[1], &m, &n, &nnz, &row_ptr, &col_idx, &val, m_gpu, n_gpu,
              nnz_gpu, &row_ptr_gpu, &col_idx_gpu, &val_gpu);
  printf("After Read\n");

  float *x, *y;
  float *x_gpu, *y_gpu;
  printf("---------BEFORE FIRST FILL---------\n");
  make_vector(m, &x, &x_gpu, 1.0);

  printf("-----------------BEFORE SECOND FILL-----------\n");
  make_vector(m, &y, &y_gpu, 0.0);

  printf("------------------BEFORE THIRD FILL-----------\n");
  float *yhat;
  make_vector(m, &yhat, NULL, 0.0);

  printf("--------------END THIRD-----------\n");

  int N = atoi(argv[2]);
  int i;

  test_spmv(m, n, nnz, row_ptr, col_idx, val, x, yhat);

  printf("-----------After test_spmv--------\n");

  float min = 9999.0;
  float max = 0.0;
  float avg = 0.0;

  hipsparseHandle_t handle;
  hipsparseSpMatDescr_t mat;
  hipsparseDnVecDescr_t vex, vey;
  size_t bufferSize;
  void *buffer;

  float alpha = 1, beta = 0;

  hipsparseCreate(&handle);
  hipsparseCreateCsr(&mat, m, n, nnz, row_ptr_gpu, col_idx_gpu, val_gpu,
                    HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                    HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
  hipsparseCreateDnVec(&vex, n, x_gpu, HIP_R_32F);
  hipsparseCreateDnVec(&vey, m, y_gpu, HIP_R_32F);
  hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat,
                          vex, &beta, vey, HIP_R_32F, HIPSPARSE_MV_ALG_DEFAULT,
                          &bufferSize);

  hipMalloc(&buffer, bufferSize);

  for (i = 0; i < 5; ++i) {
    hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vex,
                 &beta, vey, HIP_R_32F, HIPSPARSE_MV_ALG_DEFAULT, buffer);
    hipDeviceSynchronize();
  }

  for (i = 0; i < N; ++i) {
    auto tic = std::chrono::steady_clock::now();
    hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vex,
                 &beta, vey, HIP_R_32F, HIPSPARSE_MV_ALG_DEFAULT, buffer);
    hipDeviceSynchronize();
    std::chrono::duration<float> toc = std::chrono::steady_clock::now() - tic;
    auto error = hipGetLastError();
    printf("%s - %s\n", hipGetErrorName(error), hipGetErrorString(error));
    float tock = toc.count();
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

  hipMemcpy(y, y_gpu, m * sizeof(float), hipMemcpyDeviceToHost);

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
  hipFree(n_gpu);
  hipFree(m_gpu);
  hipFree(nnz_gpu);
  hipFree(row_ptr_gpu);
  hipFree(col_idx_gpu);
  hipFree(val_gpu);
  hipFree(x_gpu);
  hipFree(y_gpu);

  exit(0);

  return 0;
}
