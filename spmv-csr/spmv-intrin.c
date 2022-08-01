/*
Assumes it is in CSR format
*/

//#include "spmv.h"
#include "myTime.h"
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void print_vectori(__m512i vec) {
  int *part_res;
  posix_memalign((void **)&part_res, 64, sizeof(int) * 16);
  _mm512_store_epi32(&part_res[0], vec);
  for (int i = 0; i < 16; i++)
    printf("%d ", part_res[i]);
  printf("\n");
}

void print_vectors(__m512 vec) {
  float *part_res;
  posix_memalign((void **)&part_res, 64, sizeof(float) * 16);
  _mm512_store_ps(&part_res[0], vec);
  for (int i = 0; i < 16; i++)
    printf("%f ", part_res[i]);
  printf("\n");
}

void omp_spmv_lt4(int m, int n, int nnz, int *restrict row_start,
                  int *restrict col_idx, float *restrict val, float *restrict x,
                  float *restrict y);
void omp_spmv_gt4(int m, int n, int nnz, int *restrict row_start,
                  int *restrict col_idx, float *restrict val, float *restrict x,
                  float *restrict y);

void omp_spmv(int m, int n, int nnz, int *restrict row_start,
              int *restrict col_idx, float *restrict val, float *restrict x,
              float *restrict y) {
  float rdensity = (float)nnz / (float)(m);
  if (rdensity < 4)
    omp_spmv_lt4(m, n, nnz, row_start, col_idx, val, x, y);
  else
    omp_spmv_gt4(m, n, nnz, row_start, col_idx, val, x, y);
}

void omp_spmv_lt4(int m, int n, int nnz, int *restrict row_start,
                  int *restrict col_idx, float *restrict val, float *restrict x,
                  float *restrict y) {
  int row, col;
#pragma omp parallel
  {
    int single_add[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    __mmask16 all_mask = 0xFFFF;
    __m512 zero_vec = _mm512_setzero_ps();
    __m512i zero_vec_i = _mm512_setzero_epi32();
    __m512i m_vec = _mm512_set1_epi32(m);
    __m512i single_add_vec =
        _mm512_mask_expandloadu_epi32(m_vec, all_mask, single_add);
#pragma omp for schedule(runtime) nowait
    for (row = 0; row < m - 3; row += 4) {
      int val_idx;
      float temp[4] = {0, 0, 0, 0};
#pragma message "vectorizing according to me"
      for (val_idx = row_start[row]; val_idx < row_start[row + 4];
           val_idx += 16) {
        __m512i start_of_loop;
        __m512i end_of_loop;
        __m512i mask_prep;
        __m512i col_idx_vec;
        __m512 val_vec;
        __m512 x_vec;
        __m512 fma_res;
        __mmask16 mask_0, mask_1, mask_2, mask_3;
        __m512i part_vec_0, part_vec_1, part_vec_2, part_vec_3;
        __m512i indices;
        __m512 temp_vec_0, temp_vec_1, temp_vec_2, temp_vec_3;
        start_of_loop = _mm512_set1_epi32(val_idx);
        mask_prep = _mm512_add_epi32(start_of_loop, single_add_vec);
        end_of_loop = _mm512_set1_epi32(row_start[row + 4]);
        mask_0 = _mm512_cmplt_epi32_mask(mask_prep, end_of_loop);
        indices = _mm512_add_epi32(start_of_loop, single_add_vec);
        col_idx_vec = _mm512_mask_i32gather_epi32(zero_vec_i, mask_0, indices,
                                                  &col_idx[0], 4);
        val_vec =
            _mm512_mask_i32gather_ps(zero_vec, mask_0, indices, &val[0], 4);
        x_vec =
            _mm512_mask_i32gather_ps(zero_vec, mask_0, col_idx_vec, &x[0], 4);
        fma_res = _mm512_mul_ps(val_vec, x_vec);

        start_of_loop = _mm512_set1_epi32(val_idx);
        part_vec_0 = _mm512_set1_epi32(row_start[row + 1]);
        part_vec_1 = _mm512_set1_epi32(row_start[row + 2]);
        part_vec_2 = _mm512_set1_epi32(row_start[row + 3]);
        part_vec_3 = _mm512_set1_epi32(row_start[row + 4]);
        mask_prep = _mm512_add_epi32(start_of_loop, single_add_vec);
        mask_0 = _mm512_cmplt_epi32_mask(mask_prep, part_vec_0);
        mask_1 = _mm512_cmplt_epi32_mask(mask_prep, part_vec_1);
        mask_2 = _mm512_cmplt_epi32_mask(mask_prep, part_vec_2);
        mask_3 = _mm512_cmplt_epi32_mask(mask_prep, part_vec_3);
        temp_vec_0 = _mm512_maskz_mov_ps(mask_0, fma_res);
        temp_vec_1 = _mm512_maskz_mov_ps(~mask_0 & mask_1, fma_res);
        temp_vec_2 = _mm512_maskz_mov_ps(~mask_0 & ~mask_1 & mask_2, fma_res);
        temp_vec_3 =
            _mm512_maskz_mov_ps(~mask_0 & ~mask_1 & ~mask_2 & mask_3, fma_res);
        temp[0] += _mm512_reduce_add_ps(temp_vec_0);
        temp[1] += _mm512_reduce_add_ps(temp_vec_1);
        temp[2] += _mm512_reduce_add_ps(temp_vec_2);
        temp[3] += _mm512_reduce_add_ps(temp_vec_3);
      }
      memcpy(&y[row], &temp[0], sizeof(float) * 4);
    }
#pragma omp single
    for (; row < m; ++row) {
      float temp = 0;
      int val_idx;
#pragma vector always
#pragma vector aligned vectorlength(16)
      for (val_idx = row_start[row]; val_idx < row_start[row + 1]; ++val_idx) {
        temp += val[val_idx] * x[col_idx[val_idx]];
      }
      y[row] = temp;
    }
  }
} // end omp_spmv

void omp_spmv_gt4(int m, int n, int nnz, int *restrict row_start,
                  int *restrict col_idx, float *restrict val, float *restrict x,
                  float *restrict y) {
#pragma omp parallel for schedule(runtime)
  for (int row = 0; row < m; ++row) {
    float temp = 0;
    int val_idx;
#pragma vector always
#pragma vector aligned vectorlength(16)
    for (val_idx = row_start[row]; val_idx < row_start[row + 1]; ++val_idx) {
      temp += val[val_idx] * x[col_idx[val_idx]];
    }
    y[row] = temp;
  }
}

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

#if 1
  for (i = 0; i < 5; ++i) {
    omp_spmv(m, n, nnz, row_ptr, col_idx, val, x, y);
  }
#endif

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

  int num_wrong = 0;
  for (i = 0; i < m; ++i) {
    float temp = y[i] - yhat[i];
    if (temp > .01 || temp < -.01) {
      /*if (num_wrong < 100) {
        printf("wrong loc: %d \n", i);
        printf("y: %f yhat: %f \n", y[i], yhat[i]);
      }*/
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
