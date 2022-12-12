/********************************************************************
 * This is an example showing how to use csr-k library to compute
 * sparse matrix vector multiplication (SpMV) and sparse triangular
 * solution (STS).
 *     SpMV:
 *     		Input Args: CSR-format for a sparse matrix (A)
 *     			    config file
 *     			    x vector
 *
 *     		Output Args: y vector ( y = A * x)
 *
 *     STS:
 *     		Input Args: CSR-format for a sparse matrix (A)
 *     			    config file
 *
 *     		Output Args: x vector ( L * x = b )
 *
 * Author: Humayun Kabir (kabir@psu.edu), Phillip Lane ( pal0009@uah.edu )
 **********************************************************************/

#include "csrk.cuh"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <string>

#define GPU_CORES 5120

using namespace std;

extern unsigned int *N_gpu;
extern unsigned int *M_gpu;
extern unsigned int *NNZ_gpu;
extern unsigned int *numCoarsestRows_gpu;
extern unsigned int *mapCoarseToFinerRows_gpu_outer;
extern unsigned int *mapCoarseToFinerRows_gpu_inner;
extern unsigned int *r_vec_gpu;
extern unsigned int *c_vec_gpu;
extern float *val_gpu;
extern float *x_test_gpu;
extern float *y_gpu;
extern unsigned int *coarseRowIndex;

void my_read_csr(char *fname, int *m, int *n, int *nnz,
                 unsigned int **row_start, unsigned int **col_idx,
                 float **val) {
  // printf("%s \n", fname);
  FILE *fp = fopen(fname, "r");
  if (fp == NULL) {
    printf("Error\n");
  }
  // int tt = fgetc(fp);
  // printf("%c \n", tt);
  fscanf(fp, "%d %d  %d \n", m, n, nnz);

  // printf("size %d %d %d \n", *m, *n, *nnz);
  // malloc memory for vectors
  *row_start = (unsigned int *)malloc((*m + 1) * sizeof(int));
  *col_idx = (unsigned int *)malloc(*nnz * sizeof(int));
  *val = (float *)malloc(*nnz * sizeof(float));

  if (*row_start == NULL || *col_idx == NULL || *val == NULL) {
    printf("Error, Malloc Matrix Memory\n");
    return;
  }

  // first the row_start
  int i;
  for (i = 0; i < *m + 1; ++i) {
    int temp;
    fscanf(fp, "%u ", &temp);
    (*row_start)[i] = temp - 1;
  }

  // 2 col_idx
  for (i = 0; i < *nnz; ++i) {
    int temp;
    fscanf(fp, "%u ", &temp);
    (*col_idx)[i] = temp - 1;
  }

  // 3 vals
  for (i = 0; i < *nnz; ++i) {
    float temp;
    fscanf(fp, "%f ", &temp);
    (*val)[i] = temp;
  }

  fclose(fp);

} // read_csr

void test_spmv(int m, int n, int nnz, unsigned int *row_start,
               unsigned int *col_idx, float *val, float *x, float *y) {
  int row;

  for (row = 0; row < m; ++row) {
    unsigned int val_idx;
    float temp = 0;

    for (val_idx = row_start[row]; val_idx < row_start[row + 1]; ++val_idx) {
      temp += val[val_idx] * x[col_idx[val_idx]];
    }

    y[row] = temp;
  }
} // test_spmv

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Syntax: %s inputfile num_runs\n", argv[0]);
    return 0;
  }
  int nRows, nCols;
  int NNZ;
  unsigned int *num_edges;
  unsigned int *adj;
  float *val;
  my_read_csr(argv[1], &nRows, &nCols, &NNZ, &num_edges, &adj, &val);

  string kernelType, orderingType, corseningType;
  int *supRowSizes = new int[2];

  kernelType = "SpMV";
  corseningType = "HAND";
  int k = 3;

  double d = (double)NNZ / (double)nRows;
  int blockDimx = 8, blockDimy = 12;
  int ssrs = (int)floor(0.5 + (9.175 - 1.32 * log(d)));
  int srs = (int)floor(0.5 + (20.5 - 3.5 * log(d)));
  bool vec = false;
  int veclevel = 4;

  if (d > 8.0 && d <= 16.0) {
    vec = true;
    srs = ssrs * 4;
  } else if (d > 16.0 && d <= 32.0) {
    vec = true;
    veclevel = 8;
    blockDimy = 8;
    ssrs = (int)floor((double)ssrs * 2.5 + 0.5);
    srs = ssrs * 3;
  } else if (d > 32.0 && d <= 64.0) {
    vec = true;
    veclevel = 16;
    blockDimy = 4;
    ssrs *= 2;
    srs = ssrs * 2;
  } else if (d > 64.0) {
    vec = true;
    veclevel = 32;
    blockDimy = 2;
    ssrs = (int)floor((double)ssrs * 2.7 + 0.5);
    srs = (int)floor((double)ssrs / 4 + 0.5);
  }

  printf("using ssrs %d, srs %d\n", ssrs, srs);
  if (vec)
    printf("Block dim is %dx%dx%d\n", veclevel, blockDimx, blockDimy);
  else
    printf("Block dim is %dx%d\n", blockDimx, blockDimy);
  supRowSizes[0] = ssrs;
  supRowSizes[1] = srs;

  cout << "Read in matrix and config file." << endl;
  // Initialize a matrix object
  CSRk_Graph A_mat(nRows, nRows, NNZ, num_edges, adj, val, kernelType,
                   orderingType, corseningType, false, k, supRowSizes);

  // Put the matrix in csrk format
  A_mat.putInCSRkFormat();
  cout << "In CSR-k format." << endl;

  float *x = new float[nRows];
#pragma omp for schedule(static)
  for (int i = 0; i < nRows; i++)
    x[i] = 1.0f;

  // Initializes vector x
  A_mat.setX(x);

  // Output vector
  float *y = new float[nRows];
  // Output vector for serial CSR test
  float *y_csrser = new float[nRows];

  A_mat.setY(y);

  int N = atoi(argv[2]);
  float min = 9999.0;
  float max = 0.0;
  float avg = 0.0;
  cout << "Vectors set, running test." << endl;
  // Computes y
  cout << "SSRS: " << ssrs << ", SRS: " << srs << endl;

  int dev_cnt = 0;
  cudaGetDeviceCount(&dev_cnt);

  cout << "device count: " << dev_cnt << endl;

  for (int i = 0; i < N; i++) {
    auto tic = std::chrono::steady_clock::now();
    if (vec)
      cuSpMV_3_vec<<<A_mat.getNumCoarsestRows(),
                     dim3(veclevel, blockDimx, blockDimy)>>>(
          numCoarsestRows_gpu, mapCoarseToFinerRows_gpu_outer,
          mapCoarseToFinerRows_gpu_inner, r_vec_gpu, c_vec_gpu, val_gpu,
          x_test_gpu, y_gpu);
    else
      cuSpMV_3<<<A_mat.getNumCoarsestRows(), dim3(blockDimx, blockDimy)>>>(
          numCoarsestRows_gpu, mapCoarseToFinerRows_gpu_outer,
          mapCoarseToFinerRows_gpu_inner, r_vec_gpu, c_vec_gpu, val_gpu,
          x_test_gpu, y_gpu);

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();
    std::chrono::duration<float> toc = std::chrono::steady_clock::now() - tic;
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

  test_spmv(nRows, nCols, NNZ, num_edges, adj, val, x, y_csrser);

  unsigned int *permutation = A_mat.getPermutation();
  y = A_mat.getY();
  cudaDeviceSynchronize();

  int num_wrong = 0;
  for (int i = 0; i < nRows; ++i) {
    // float temp = y[i] - y_csrser[permutation[i]];
    float temp = y[i] - y_csrser[i];
    if (temp > .01 || temp < -.01) {
      printf("wrong loc: %d \n", i);
      // printf("y: %f y_csrser: %f \n", y[i], y_csrser[i]);
      printf("y: %f y_csrser: %f \n", y[i], y_csrser[permutation[i]]);
      num_wrong++;
    }
  }

  printf("Number Wrong: %d \n", num_wrong);

  // Get permutation
  // unsigned int *permutation = A_mat.getPermutation();

  // for(int i =0; i < 4; i++)
  //	cout<<permutation[i]<<"-th value: "<<y[i]<<endl;

  /*
          //Put the matrix in csrk format
          A_mat.putInCSRkFormat();

          //Find the structure of lower and upper triangular matrices
          A_mat.incomplete_choloskey();

          //Computes b = L * x
          A_mat.compute_b();

          //performs lower triangular solution
          A_mat.lowerSTS();

          //Computes error
          A_mat.checkError();
  */

  delete[] y;
  delete[] y_csrser;
  delete[] x;
  delete[] supRowSizes;
  free(num_edges);
  free(val);
  free(adj);
  return 0;
}
