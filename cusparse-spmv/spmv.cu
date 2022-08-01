/*
Assumes it is in CSR format
*/

#include "spmv.h"
#include <chrono>
#include <cmath>
#include <cuda.h>
#include <cusparse.h>
#include <iostream>
#include <omp.h>

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
  cudaMalloc(row_start_gpu, (*m + 1) * sizeof(int));
  cudaMalloc(col_idx_gpu, (*nnz + 4) * sizeof(int));
  cudaMalloc(val_gpu, (*nnz + 4) * sizeof(float));

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

  cudaMemcpy(*row_start_gpu, *row_start, (*m + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(*col_idx_gpu, *col_idx, *nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(*val_gpu, *val, *nnz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(m_gpu, m, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(n_gpu, n, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nnz_gpu, nnz, sizeof(int), cudaMemcpyHostToDevice);

} // read_csr

void make_vector(int m, float **x, float **x_gpu, float val) {
  *x = (float *)malloc(m * sizeof(float));

  if (x_gpu)
    cudaMalloc(x_gpu, m * sizeof(float));

  int i;

#pragma omp parallel for schedule(runtime)
  for (i = 0; i < m; ++i) {
    (*x)[i] = val;
  }

  if (x_gpu)
    cudaMemcpy(*x_gpu, *x, m * sizeof(float), cudaMemcpyHostToDevice);
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

  cudaMalloc(&m_gpu, sizeof(int));
  cudaMalloc(&n_gpu, sizeof(int));
  cudaMalloc(&nnz_gpu, sizeof(int));

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

  cusparseHandle_t handle;
  cusparseSpMatDescr_t mat;
  cusparseDnVecDescr_t vex, vey;
  size_t bufferSize;
  void *buffer;

  float alpha = 1, beta = 0;

  cusparseCreate(&handle);
  cusparseCreateCsr(&mat, m, n, nnz, row_ptr_gpu, col_idx_gpu, val_gpu,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateDnVec(&vex, n, x_gpu, CUDA_R_32F);
  cusparseCreateDnVec(&vey, m, y_gpu, CUDA_R_32F);
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat,
                          vex, &beta, vey, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT,
                          &bufferSize);

  cudaMalloc(&buffer, bufferSize);

  for (i = 0; i < N; ++i) {
    auto tic = std::chrono::steady_clock::now();
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vex,
                 &beta, vey, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, buffer);
    cudaDeviceSynchronize();
    std::chrono::duration<float> toc = std::chrono::steady_clock::now() - tic;
    auto error = cudaGetLastError();
    printf("%s - %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
    float tock = toc.count();
    std::cout << i + 1 << "th iteration took " << tock << std::endl;
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

  cudaMemcpy(y, y_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);

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
  cudaFree(n_gpu);
  cudaFree(m_gpu);
  cudaFree(nnz_gpu);
  cudaFree(row_ptr_gpu);
  cudaFree(col_idx_gpu);
  cudaFree(val_gpu);
  cudaFree(x_gpu);
  cudaFree(y_gpu);

  exit(0);

  return 0;
}
