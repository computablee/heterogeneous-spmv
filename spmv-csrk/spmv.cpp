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
 * Author: Humayun Kabir (kabir@psu.edu)
 **********************************************************************/

#include "csrk.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#define CSRK_LEVEL 2

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

  printf("size %d %d %d \n", *m, *n, *nnz);
  // malloc memory for vectors
  *row_start = (unsigned int *)hbw_malloc((*m + 1) * sizeof(unsigned int));
  *col_idx = (unsigned int *)hbw_malloc(*nnz * sizeof(unsigned int));
  *val = (float *)hbw_malloc(*nnz * sizeof(float));

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
  if (argc < 3 + CSRK_LEVEL - 1) {
    printf("Syntax: %s inputfile num_runs ", argv[0]);
    for (int i = 0; i < CSRK_LEVEL - 1; i++) {
      // print "super_row_size" depending on how many levels we have
      for (int j = 0; j < CSRK_LEVEL - 1 - i; j++) {
        printf("super_");
      }
      printf("row_size ");
    }
    printf("\n");
    return 0;
  }
  int nRows, nCols;
  int NNZ;
  unsigned int *num_edges;
  unsigned int *adj;
  float *val;
  my_read_csr(argv[1], &nRows, &nCols, &NNZ, &num_edges, &adj, &val);

  string kernelType, orderingType, corseningType;
  int k = CSRK_LEVEL;
  int *supRowSizes = new int[CSRK_LEVEL - 1];

  // Read the config file
  // readConfigFile( "configSpMV.txt", kernelType, orderingType, corseningType,
  // k, supRowSizes);
  kernelType = "SpMV";
  corseningType = "HAND";
  // k = 2;
  for (int i = 0; i < CSRK_LEVEL - 1; i++)
    supRowSizes[i] = atoi(argv[i + 3]);
  std::cout << kernelType << std::endl
            << corseningType << std::endl
            << k << std::endl;
  for (int i = 0; i < CSRK_LEVEL - 1; i++)
    std::cout << supRowSizes[i] << std::endl;
  //	readConfigFile( "configSTS.txt", kernelType, orderingType,
  // corseningType, k, supRowSizes);

  cout << "Read in matrix and config file." << endl;
  // Initialize a matrix object
  CSRk_Graph A_mat(nRows, nRows, NNZ, num_edges, adj, val, kernelType,
                   orderingType, corseningType, false, k, supRowSizes);

  // Put the matrix in csrk format
  A_mat.putInCSRkFormat();
  cout << "In CSR-k format." << endl;

  float *x = (float *)hbw_malloc(nRows * sizeof(float));
#pragma omp for schedule(static)
  for (int i = 0; i < nRows; i++)
    x[i] = 1.0f;

  // Initializes vector x
  A_mat.setX(x);
  // return 0;
  // Output vector
  float *y; // = (float *) hbw_malloc(nRows*sizeof(float));
  posix_memalign((void **)&y, 64, nRows * sizeof(float));
  // Output vector for serial CSR test
  float *y_csrser = (float *)hbw_malloc(nRows * sizeof(float));
  //posix_memalign((void **)&y_csrser, 64, nRows * sizeof(float));

  int N = atoi(argv[2]);
  float min = 9999.0;
  float max = 0.0;
  float avg = 0.0;
  cout << "Vectors set, running test." << endl;
  // Computes 5

  // do 5 warmup runs
  for (int i = 0; i < 5; i++) {
    A_mat.SpMV(y);
  }

  // return 0;

  for (int i = 0; i < N; i++) {
    auto tic = std::chrono::steady_clock::now();
    A_mat.SpMV(y);
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

  int num_wrong = 0;
  // printf("%d\n", permutation[0]);
  for (int i = 0; i < nRows; ++i)
    printf("%d ", permutation[i]);
  printf("\n");
  for (int i = 0; i < nRows; ++i) {
    float temp = y[i] - y_csrser[permutation[i]];
    if (temp > .01 || temp < -.01) {
      // printf("wrong loc: %d \n", i);
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

  hbw_free(y);
  hbw_free(y_csrser);
  hbw_free(x);
  delete[] supRowSizes;
  hbw_free(num_edges);
  hbw_free(val);
  hbw_free(adj);
  return 0;
}
