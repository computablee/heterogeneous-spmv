/*
Assumes it is in CSR format
*/

//#include "spmv.h"
#include "myTime.h"
#include <iostream>
#include <omp.h>

void my_read_csr3(char* fname, int *sup_sup_row_size, int *sup_row_size, int *m, int *n, int *nnz,
	      int **row_start, int **col_idx, double **val, int ***sup_rows)
{
  //printf("%s \n", fname);
  FILE *fp = fopen(fname, "r");
  if(fp == NULL)
    {
      printf("Error\n");
    }
  //int tt = fgetc(fp); 
  //printf("%c \n", tt);
  fscanf(fp, "%d %d %d %d %d \n", sup_sup_row_size, sup_row_size, m, n, nnz);

int k=3;
  //printf("size %d %d %d \n", *m, *n, *nnz);
  //malloc memory for vectors
  *row_start = (int *) malloc((*m+1)*sizeof(int));
  *col_idx   = (int *) malloc(*nnz*sizeof(int));
  *val       = (double *) malloc(*nnz*sizeof(double));
  *sup_rows  = (int **) malloc((k+1)*sizeof(int*));
  (*sup_rows)[k-1] = (int *) malloc((*sup_sup_row_size+1)*sizeof(int));
  (*sup_rows)[k-2] = (int *) malloc((*sup_row_size+1)*sizeof(int));

  if (*row_start == NULL || *col_idx == NULL || *val == NULL)
    {
      printf("Error, Malloc Matrix Memory\n");
      return; 
    }

  //first the row_start
  int i;
  for(i = 0; i < *sup_sup_row_size+1; ++i)
    {
      int temp;
      fscanf(fp, "%d ", &temp);
      (*sup_rows)[k-1][i] = temp;
    }
  for(i = 0; i < *sup_row_size+1; ++i)
    {
      int temp;
      fscanf(fp, "%d ", &temp);
      (*sup_rows)[k-2][i] = temp;
    }
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
      double temp;
      fscanf(fp, "%lg ", &temp);
      (*val)[i] = temp;
    }

  fclose(fp);

}//read_csr





int main( int argc, char *argv[])
{
  
  if(argc < 2)
    {
      printf("./stats.exe inputfie.csr \n");
      exit(0);
    }


  int k=3,sup_sup_row_size,sup_row_size,m,n,nnz;
  int *row_ptr, *col_idx;
  int **sup_rows;
  double *val;

  my_read_csr3(argv[1], &sup_sup_row_size, &sup_row_size, &m, &n, &nnz,
	   &row_ptr, &col_idx, &val, &sup_rows);

  
  //double *x, *y;
  //make_vector(m, &x, 1.0);
  //make_vector(m, &y, 0.0);

  int max_nnz = 0;
  int min_nnz = nnz;
  int max_band = 0;
  int min_band = nnz;
  double var_nnz = 0;
  double avg = (nnz*1.0)/(m*1.0);
  long long avg_band = 0;

  int row; 
  for(row = 0; row < m; ++row)
    {
      
      int row_nnz = row_ptr[row+1] - row_ptr[row];
      int band = col_idx[row_ptr[row+1]-1]- col_idx[row_ptr[row]];

//      printf("band: %d \n", band);

      if(row_nnz < min_nnz)
	min_nnz = row_nnz;
      if(row_nnz > max_nnz)
	max_nnz = row_nnz;
      if(band > max_band)
	max_band = band;
      if(band < min_band)
	min_band = band;

      var_nnz += (row_nnz - avg)*(row_nnz - avg);
      avg_band += band;

    }

  //avg_band = avg_band/ (1.0*m);
  printf("NNZ Avg: %f \n", (nnz*1.0)/(m*1.0));
  printf("NNZ Min: %d  Percent: %f \n", min_nnz, (min_nnz*1.0)/(m*1.0));
  printf("NNZ Max: %d  Percent: %f \n", max_nnz, (max_nnz*1.0)/(m*1.0));
  printf("NNZ Var: %f \n", var_nnz/(m*1.0));
  printf("Band Avg: %f \n", avg_band/(1.0*m));
  printf("Band Max: %d Percent: %f \n", max_band, (max_band*1.0)/(m*1.0));
  printf("Band Min: %d Percent: %f \n", min_band, (min_band*1.0)/(m*1.0));

  double avg_band_d = avg_band/(1.0*m);
  double var_band = 0;
  for(row=0; row < m; ++row)
    {
      int band = col_idx[row_ptr[row+1]-1] - col_idx[row_ptr[row]];
      var_band = (band - avg_band_d)*(band - avg_band_d);
    }
  printf("Band Var: %f \n", var_band/(m*1.0));
  printf("Total NNZ: %d\n", nnz);
  printf("Dim: %dx%d\n", m, n);  

  return 0;
}
