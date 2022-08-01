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

using namespace std;

void my_write_csr(char *fname, CSRk_Graph *matrix) {
  // printf("%s \n", fname);
  FILE *fp = fopen(fname, "w");
  if (fp == NULL) {
    printf("Error\n");
  }
  // int tt = fgetc(fp);
  // printf("%c \n", tt);
  fprintf(fp, "%ld %ld %ld %ld %ld \n", matrix->numCoarsestRows,
          matrix->numCoarserRows, matrix->M, matrix->N, matrix->NNZ);

  // first the row_start
  int i;

  for (i = 0; i < matrix->numCoarsestRows + 1; ++i) {
    fprintf(fp, "%u ", matrix->mapCoarseToFinerRows[matrix->k - 1][i]);
  }

  for (i = 0; i < matrix->numCoarserRows + 1; ++i) {
    fprintf(fp, "%u ", matrix->mapCoarseToFinerRows[matrix->k - 2][i]);
  }

  for (i = 0; i < matrix->M + 1; ++i) {
    fprintf(fp, "%u ", matrix->r_vec[i]);
  }

  for (i = 0; i < matrix->NNZ; ++i) {
    fprintf(fp, "%u ", matrix->c_vec[i]);
  }

  for (i = 0; i < matrix->NNZ; ++i) {
    fprintf(fp, "%.6f ", matrix->val[i]);
  }
  fclose(fp);

} // read_csr

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
  *row_start = (unsigned int *)malloc((*m + 1) * sizeof(unsigned int));
  *col_idx = (unsigned int *)malloc(*nnz * sizeof(unsigned int));
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
  if (argc < 3) {
    printf("Syntax: %s inputfile outputfile\n", argv[0]);
    return 0;
  }
  int nRows, nCols;
  int NNZ;
  unsigned int *num_edges;
  unsigned int *adj;
  float *val;
  my_read_csr(argv[1], &nRows, &nCols, &NNZ, &num_edges, &adj, &val);

  string kernelType, orderingType, corseningType;
  int k = 3;
  int *supRowSizes = new int[2];

  // Read the config file
  // readConfigFile( "configSpMV.txt", kernelType, orderingType, corseningType,
  // k, supRowSizes);
  kernelType = "SpMV";
  corseningType = "HAND";
  // k = 2;
  double d = (double)NNZ / (double)nRows;
  int ssrs = (int)floor(8.89888 - 1.25 * log(d) + 0.5);
  int srs = (int)floor(10.14618 - 1.5 * log(d) + 0.5);

  if (d > 8.0 && d <= 16.0) {
    ssrs = (int)floor((double)ssrs * 1.5 + 0.5);
    srs = ssrs * 2;
  } else if (d > 16.0 && d <= 32.0) {
    ssrs *= 4;
    srs = ssrs >> 1;
  } else if (d > 32 && d <= 64.0) {
    ssrs *= 5;
    srs = ssrs >> 1;
  } else if (d > 64) {
    ssrs *= 5;
    srs = ssrs >> 1;
  }
  printf("using ssrs %d, srs %d\n", ssrs, srs);
  supRowSizes[0] = ssrs;
  supRowSizes[1] = srs;
  std::cout << kernelType << std::endl
            << corseningType << std::endl
            << k << std::endl
            << supRowSizes[0] << supRowSizes[1] << std::endl;
  //	readConfigFile( "configSTS.txt", kernelType, orderingType,
  // corseningType, k, supRowSizes);

  cout << "Read in matrix and config file." << endl;
  // Initialize a matrix object
  CSRk_Graph A_mat(nRows, nRows, NNZ, num_edges, adj, val, kernelType,
                   orderingType, corseningType, false, k, supRowSizes);

  // Put the matrix in csrk format

  auto tic = std::chrono::steady_clock::now();
  A_mat.putInCSRkFormat();
  std::chrono::duration<float> toc = std::chrono::steady_clock::now() - tic;
  float tock = toc.count();
  cout << argv[1] << " reordered in " << tock << " seconds." << endl;
  cout << "In CSR-k format." << endl;

  my_write_csr(argv[2], &A_mat);

  delete[] supRowSizes;
  free(num_edges);
  free(val);
  free(adj);
  return 0;
}
