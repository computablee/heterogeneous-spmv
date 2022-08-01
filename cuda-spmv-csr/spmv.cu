/*
Assumes it is in CSR format
*/

#include "spmv.h"
#include <omp.h>
#include <cuda.h>
#include <cmath>
#include <chrono>

void my_read_csr(char* fname, int *m, int *n, int *nnz,
	      int **row_start, int **col_idx, float **val,
        int *m_gpu, int *n_gpu, int *nnz_gpu,
        int **row_start_gpu, int **col_idx_gpu, float **val_gpu)
{
  FILE *fp = fopen(fname, "r");
  if(fp == NULL)
  {
    printf("Error\n");
  }
  
  fscanf(fp, "%d %d  %d \n", m, n, nnz);

  //malloc memory for vectors
  *row_start = (int *) malloc((*m + 1)*sizeof(int));
  *col_idx   = (int *) malloc(*nnz*sizeof(int));
  *val       = (float *) malloc(*nnz*sizeof(float));
  cudaMalloc(row_start_gpu, (*m + 1)*sizeof(int));
  cudaMalloc(col_idx_gpu, (*nnz + 4)*sizeof(int));
  cudaMalloc(val_gpu, (*nnz + 4)*sizeof(float));

  if (*row_start == NULL || *col_idx == NULL || *val == NULL)
  {
    printf("Error, Malloc Matrix Memory\n");
    return; 
  }
 
  //first the row_start
  int i;
  for(i = 0; i < *m+1; ++i)
  {
    int temp;
    fscanf(fp, "%d ", &temp);
    (*row_start)[i] = temp;
  }


  //2 col_idx
  for(i = 0; i < *nnz; ++i)
  {
    int temp;
    fscanf(fp, "%d ", &temp);
    (*col_idx)[i] = temp;
  }

  //3 vals
  for(i = 0; i < *nnz; ++i)
  {
    float temp;
    fscanf(fp, "%f ", &temp);
    (*val)[i] = temp;
  }
    
  fclose(fp);

  cudaMemcpy(*row_start_gpu, *row_start, (*m + 1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*col_idx_gpu, *col_idx, *nnz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*val_gpu, *val, *nnz*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(m_gpu, m, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(n_gpu, n, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nnz_gpu, nnz, sizeof(int), cudaMemcpyHostToDevice);

}//read_csr


void make_vector(int m, float **x, float **x_gpu, float val)
{
  *x = (float *)malloc(m*sizeof(float));

  if (x_gpu)
    cudaMalloc(x_gpu, m*sizeof(float));

  int i;
  
  #pragma omp parallel for schedule(runtime)
  for(i = 0; i < m; ++i)
  {
    (*x)[i] = val;
  }

  if (x_gpu)
    cudaMemcpy(*x_gpu, *x, m*sizeof(float), cudaMemcpyHostToDevice);
}


void test_spmv(int m, int n , int nnz, 
	       int *row_start, int *col_idx, float *val, 
	       float *x, float *y)
{
  int row;

  for(row = 0; row < m; ++row)
  {
    int val_idx;
    float temp = 0;

    for(val_idx = row_start[row]; 
        val_idx < row_start[row+1]; ++val_idx)
    {
      temp += val[val_idx]*x[col_idx[val_idx]];
    }

    y[row] = temp;
  }
}//test_spmv

__global__ void cuda_spmv(int *__restrict__ m_gpu, int *__restrict__ n_gpu, int *__restrict__ nnz_gpu,
	      int *__restrict__ row_start, int *__restrict__ col_idx, float *__restrict__ val,
	      float *__restrict__ x, volatile float *__restrict__ y)
{
  int uid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int row;
  int m = *m_gpu;

  for(row = uid; row < m; row += stride)
  {
    int val_idx = row_start[row], end = row_start[row+1];
    int end_m3 = end - 3;
    float temp = 0;

    float val_prefetch[8];
    int col_idx_prefetch[8];

    val_prefetch[0] = val[val_idx];
    val_prefetch[1] = val[val_idx + 1];
    val_prefetch[2] = val[val_idx + 2];
    val_prefetch[3] = val[val_idx + 3];
    col_idx_prefetch[0] = col_idx[val_idx];
    col_idx_prefetch[1] = col_idx[val_idx + 1];
    col_idx_prefetch[2] = col_idx[val_idx + 2];
    col_idx_prefetch[3] = col_idx[val_idx + 3];

    for(; val_idx < end_m3; val_idx += 4)
    {
      val_prefetch[4] = val[val_idx + 4];
      val_prefetch[5] = val[val_idx + 5];
      val_prefetch[6] = val[val_idx + 6];
      val_prefetch[7] = val[val_idx + 7];
      col_idx_prefetch[4] = col_idx[val_idx + 4];
      col_idx_prefetch[5] = col_idx[val_idx + 5];
      col_idx_prefetch[6] = col_idx[val_idx + 6];
      col_idx_prefetch[7] = col_idx[val_idx + 7];
      temp = __fmaf_rn(val_prefetch[0], x[col_idx_prefetch[0]], temp);
      temp = __fmaf_rn(val_prefetch[1], x[col_idx_prefetch[1]], temp);
      temp = __fmaf_rn(val_prefetch[2], x[col_idx_prefetch[2]], temp);
      temp = __fmaf_rn(val_prefetch[3], x[col_idx_prefetch[3]], temp);
      val_prefetch[0] = val_prefetch[4];
      val_prefetch[1] = val_prefetch[5];
      val_prefetch[2] = val_prefetch[6];
      val_prefetch[3] = val_prefetch[7];
      col_idx_prefetch[0] = col_idx_prefetch[4];
      col_idx_prefetch[1] = col_idx_prefetch[5];
      col_idx_prefetch[2] = col_idx_prefetch[6];
      col_idx_prefetch[3] = col_idx_prefetch[7];
    }

    val_prefetch[0] = val[val_idx];
    col_idx_prefetch[0] = col_idx[val_idx];
    for(; val_idx < end; ++val_idx)
    {
      val_prefetch[1] = val[val_idx + 1];
      col_idx_prefetch[1] = col_idx[val_idx + 1];
      temp = __fmaf_rn(val_prefetch[0], x[col_idx_prefetch[0]], temp);
      val_prefetch[0] = val_prefetch[1];
      col_idx_prefetch[0] = col_idx_prefetch[1];
    }

    y[row] = temp;
  }
}//end omp_spmv


int main( int argc, char *argv[])
{
  
  if(argc < 2)
  {
    printf("./spmv.exe inputfie.csr num_runs\n");
    exit(0);
  }


  int m,n,nnz;
  int *row_ptr, *col_idx;
  float *val;
  int *m_gpu,*n_gpu,*nnz_gpu;
  int *row_ptr_gpu, *col_idx_gpu;
  float *val_gpu;

  cudaMalloc(&m_gpu, sizeof(int));
  cudaMalloc(&n_gpu, sizeof(int));
  cudaMalloc(&nnz_gpu, sizeof(int));
  
  printf("Before Read\n");
  my_read_csr(argv[1], &m, &n, &nnz,
	  &row_ptr, &col_idx, &val,
    m_gpu, n_gpu, nnz_gpu,
    &row_ptr_gpu, &col_idx_gpu, &val_gpu);
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
  int i ;

  test_spmv(m,n, nnz, 
    row_ptr, col_idx, val, 
    x, yhat);

  printf("-----------After test_spmv--------\n");

  float min = 9999.0;
  float max = 0.0;
  float avg = 0.0;

  for(i =0 ; i < N; ++i)
  {    
    auto tic = std::chrono::steady_clock::now();
    cuda_spmv<<<m/32,32>>>(m_gpu,n_gpu, nnz_gpu, 
      row_ptr_gpu, col_idx_gpu, val_gpu, 
      x_gpu, y_gpu);
    //auto error = cudaGetLastError();
    //printf("%s - %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
    cudaDeviceSynchronize();
    std::chrono::duration<float> toc = std::chrono::steady_clock::now() - tic;
    float tock = toc.count();
    avg += tock;
    if(tock < min)
    {
      min = tock;
    }
    if(tock > max)
    {
      max = tock;
    }
  }
 
  printf("TimeMin: %lg\n", min);
  printf("TimeMax: %lg\n", max);
  printf("TimeAvg: %lg\n", avg/N);
  
  printf("TEAST\n");

  //Check solution
  
  cudaMemcpy(y, y_gpu, m*sizeof(float), cudaMemcpyDeviceToHost);

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
