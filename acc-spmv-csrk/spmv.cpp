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

#include "csrk.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

using namespace std;

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
    (*row_start)[i] = temp;
  }

  // 2 col_idx
  for (i = 0; i < *nnz; ++i) {
    int temp;
    fscanf(fp, "%u ", &temp);
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

void test_spmv(int m, int n, int nnz, unsigned int *row_start,
               unsigned int *col_idx, float *val, float *x, float *y) {
  unsigned int row;

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
  if (argc < 4) {
    printf("Syntax: %s inputfile num_runs super_row_size\n", argv[0]);
    return 0;
  }
  int nRows, nCols;
  int NNZ;
  unsigned int *num_edges;
  unsigned int *adj;
  float *val;
  my_read_csr(argv[1], &nRows, &nCols, &NNZ, &num_edges, &adj, &val);

  string kernelType, orderingType, corseningType;
  int k = 0;
  int *supRowSizes = new int[1];

  // Read the config file
  // readConfigFile( "configSpMV.txt", kernelType, orderingType, corseningType,
  // k, supRowSizes);
  kernelType = "SpMV";
  corseningType = "HAND";
  k = 2;
  supRowSizes[0] = atoi(argv[3]);
  //	readConfigFile( "configSTS.txt", kernelType, orderingType,
  //corseningType, k, supRowSizes);

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

  for (int i = 0; i < N; i++) {
    auto tic = std::chrono::steady_clock::now();
    A_mat.SpMV();
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

  int num_wrong = 0;
  for (int i = 0; i < nRows; ++i) {
    // float temp = y[i] - y_csrser[permutation[i]];
    float temp = y[i] - y_csrser[i];
    if (temp > .01 || temp < -.01) {
      printf("wrong loc: %d \n", i);
      printf("y: %f y_csrser: %f \n", y[i], y_csrser[i]);
      // printf("y: %f y_csrser: %f \n", y[i], y_csrser[permutation[i]]);
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
