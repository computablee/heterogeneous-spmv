/*
Assumes it is in CSR format
*/

#include "spmv.h"
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Cuda.hpp>
#include <chrono>
#include <cmath>
#include <cuda.h>
#include <omp.h>

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
    (*row_start)[i] = temp - 1;
  }

  // 2 col_idx
  for (i = 0; i < *nnz; ++i) {
    int temp;
    fscanf(fp, "%d ", &temp);
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

using Scalar = float;
using Cardinal = unsigned int;
using Offset = unsigned int;

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

  Kokkos::initialize();

  using device_type =
      typename Kokkos::Device<Kokkos::Cuda, typename Kokkos::CudaSpace>;
  using matrix_type =
      typename KokkosSparse::CrsMatrix<Scalar, Cardinal, device_type, void>;
  using graph_type = typename matrix_type::staticcrsgraph_type;
  using row_map_type = typename graph_type::row_map_type;
  using entries_type = typename graph_type::entries_type;
  using values_type = typename matrix_type::values_type;

  Cardinal numRows = m;
  Offset numNNZ = nnz;

  typename row_map_type::non_const_type row("row ptr", numRows + 1);
  row_map_type::non_const_type::HostMirror row_mirror =
      Kokkos::create_mirror_view(row);
  for (unsigned int i = 0; i < numRows + 1; i++)
    row_mirror(i) = row_ptr[i];
  Kokkos::deep_copy(Kokkos::Cuda(), row, row_mirror);

  typename entries_type::non_const_type col("col idx", numNNZ);
  entries_type::non_const_type::HostMirror col_mirror =
      Kokkos::create_mirror_view(col);
  for (unsigned int i = 0; i < numNNZ; i++)
    col_mirror(i) = col_idx[i];
  Kokkos::deep_copy(Kokkos::Cuda(), col, col_mirror);

  typename values_type::non_const_type values("values", numNNZ);
  values_type::non_const_type::HostMirror values_mirror =
      Kokkos::create_mirror_view(values);
  for (unsigned int i = 0; i < numNNZ; i++)
    values_mirror(i) = val[i];
  Kokkos::deep_copy(Kokkos::Cuda(), values, values_mirror);

  graph_type myGraph(col, row);
  matrix_type myMatrix("matr", numRows, values, myGraph);
  Scalar alpha = 1, beta = 0;

  typename values_type::non_const_type x_vec("lhs", numRows);
  values_type::non_const_type::HostMirror x_vec_mirror =
      Kokkos::create_mirror_view(x_vec);
  for (unsigned int i = 0; i < numRows; i++)
    x_vec_mirror(i) = x[i];
  Kokkos::deep_copy(Kokkos::Cuda(), x_vec, x_vec_mirror);

  typename values_type::non_const_type y_vec("rhs", numRows);
  values_type::non_const_type::HostMirror y_vec_mirror =
      Kokkos::create_mirror_view(y_vec);
  for (unsigned int i = 0; i < numRows; i++)
    y_vec_mirror(i) = y[i];
  Kokkos::deep_copy(Kokkos::Cuda(), y_vec, y_vec_mirror);

  Kokkos::fence();

  for (i = 0; i < N; ++i) {
    auto tic = std::chrono::steady_clock::now();
    KokkosSparse::spmv("N", alpha, myMatrix, x_vec, beta, y_vec);
    Kokkos::Cuda().fence();
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

  printf("TEAST\n");

  // Check solution
  Kokkos::deep_copy(Kokkos::Cuda(), y_vec_mirror, y_vec);
  Kokkos::fence();

  for (unsigned int i = 0; i < numRows; i++)
    y[i] = y_vec_mirror(i);

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
  Kokkos::finalize();

  exit(0);

  return 0;
}
