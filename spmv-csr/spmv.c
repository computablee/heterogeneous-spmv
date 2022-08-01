/*
Assumes it is in CSR format
*/

//#include "spmv.h"
#include "myTime.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void my_read_csr(char *fname, int *m, int *n, int *nnz, int **row_start,
                 int **col_idx, float **val) {
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

  int row, col;

  for (row = 0; row < m; ++row) {
    // printf("row: %d rowi: %d %d\n",
    //	     row, row_start[row], row_start[row+1]);
    int val_idx;
    float temp = 0;
    for (val_idx = row_start[row]; val_idx < row_start[row + 1]; ++val_idx) {
      /*
      printf("col: %d val: %lg x: %lg y: %lg \n",
             col_idx[val_idx], val[val_idx],
             x[col_idx[val_idx]], y[row]);
      */
      temp += val[val_idx] * x[col_idx[val_idx]];
    }
    y[row] = temp;
    // printf("Done with row: %d value: %lg \n", row, y[row]);
  }

} // test_spmv

void omp_spmv(int m, int n, int nnz, int *row_start, int *col_idx, float *val,
              float *x, float *y) {

  int row, col;
#pragma omp parallel for schedule(runtime)
  for (row = 0; row < m; ++row) {
    int val_idx;
    float temp = 0;
#ifdef __clang__
#pragma message "vectorizing according to clang"
#pragma clang loop vectorize(enable) interleave(enable)
#else
//#pragma message "vectorizing according to icc"
//#pragma vector always
//#pragma vector aligned vectorlength(8)
#endif
    for (val_idx = row_start[row]; val_idx < row_start[row + 1]; ++val_idx) {
      temp += val[val_idx] * x[col_idx[val_idx]];
    }
    y[row] = temp;
  }

} // end omp_spmv

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
  // return 0;
  printf("-----------------BEFORE SECOND FILL-----------\n");
  make_vector(m, &y, 0.0);

  printf("------------------BEFORE THIRD FILL-----------\n");
  float *yhat;
  make_vector(m, &yhat, 0.0);

  printf("--------------END THIRD-----------\n");

  int N = atoi(argv[2]);
  int i;

  for (i = 0; i < N; ++i) {
    test_spmv(m, n, nnz, row_ptr, col_idx, val, x, yhat);
  }

  printf("-----------After test_spmv--------\n");

  //__sync_synchronize();

  float min = 9999.0;
  float max = 0.0;
  float avg = 0.0;

  // printf("------------------DONE FILL-------------\n");

  // int N = atoi(argv[2]);
  // int i ;

  for (i = 0; i < 5; ++i) {
    omp_spmv(m, n, nnz, row_ptr, col_idx, val, x, y);
  }

  for (i = 0; i < N; ++i) {

    double tic = clock_time();
    omp_spmv(m, n, nnz, row_ptr, col_idx, val, x, y);
    double toc = clock_time();
    toc = toc - tic;
    avg += toc;
    if (toc < min) {
      min = toc;
    }
    if (toc > max) {
      max = toc;
    }
  }

  printf("TimeMin: %lg\n", min);
  printf("TimeMax: %lg\n", max);
  printf("TimeAvg: %lg\n", avg / N);

  // pthread_exit(NULL);

  printf("TEAST\n");

  // Check solution
  /*
  int num_wrong = 0;
  for(i=0; i < m; ++i)
    {
      float temp = y[i] - yhat[i];
      if(temp > .01 || temp < -.01)
        {
          printf("wrong loc: %d \n", i);
          printf("y: %f yhat: %f \n", y[i], yhat[i]);
          num_wrong++;
        }
    }
  printf("Number Wrong: %d \n", num_wrong);
  */

  free(x);
  free(y);
  free(yhat);
  free(row_ptr);
  free(col_idx);
  free(val);

  /*
  printf("Solution:\n");
  for(i=0; i < m; i++)
    {
      printf("%lg \n", y[i]);
    }
  */

  exit(0);

  return 0;
}
