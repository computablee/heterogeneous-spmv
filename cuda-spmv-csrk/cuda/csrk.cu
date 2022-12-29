/* Contains the function definitions for modifying matrices and putting them in
 * proper format that would help to perform the SpMV and TriSolve in an
 * efficient manner Author: Humayun Kabir ( kabir@psu.edu ), Phillip Lane (
 * pal0009@uah.edu ) */

#include "csrk.cuh"
#include <algorithm>
#include <cuda.h>
#include <string.h>
#include <vector>

using namespace std;

unsigned int *N_gpu;
unsigned int *M_gpu;
unsigned int *NNZ_gpu;
unsigned int *numCoarsestRows_gpu;
unsigned int *mapCoarseToFinerRows_gpu_outer;
unsigned int *mapCoarseToFinerRows_gpu_inner;
unsigned int *r_vec_gpu;
unsigned int *c_vec_gpu;
unsigned int *coarseRowIndex;
float *val_gpu;
float *x_test_gpu;
float *y_gpu;

// Read the config file
void readConfigFile(char *configFileName, string &kernelType,
                    string &orderingType, string &corseningType, int &k,
                    int *&superRowSizes) {

  ifstream inFile(configFileName);

  if (inFile.is_open()) {

    inFile >> kernelType;

    int supRowSizesLen = 0;
    if (kernelType == "SpMV") {

      orderingType = "none";
      inFile >> corseningType;
      inFile >> k;

      superRowSizes = new int[k - 1];
      supRowSizesLen = k - 1;
    } else if (kernelType == "STS") {
      inFile >> orderingType;
      inFile >> corseningType;
      inFile >> k;

      superRowSizes = new int[k - 2];
      supRowSizesLen = k - 2;
    } else {
      cout << "This kernel type is not supported" << endl;
      exit(1);
    }

    for (int i = 0; i < supRowSizesLen; i++) {
      inFile >> superRowSizes[i];
    }

    inFile.close();
  } else {
    cout << "Unable to open file: " << configFileName << endl;
    exit(1);
  }
}

// Functions to compare different structs
bool compare_deg_id_pair(degree_id_pair p1, degree_id_pair p2) {
  return p1.deg < p2.deg;
}

bool compAdjDeg(adj_deg adjL, adj_deg adjR) { return adjL.adj <= adjR.adj; }

bool myComparePair(Pair p1, Pair p2) { return p1.first < p2.first; }

bool compare_rev_deg_id_pair(rev_deg_id_pair p1, rev_deg_id_pair p2) {
  return p1.deg > p2.deg;
}

bool myCompare(Pair_CSR p1, Pair_CSR p2) { return p1.first < p2.first; }

int C_GRAPH::numCopy = 0;

/////////////////////////////////////////////////////
//     Definitions of functions in CSRk_Graph      //
/////////////////////////////////////////////////////

void CSRk_Graph::setX(float *x) {
  if (k == 1) {
    for (int i = 0; i < N; i++)
      x_test[i] = x[i];
  } else {
    for (int i = 0; i < N; i++)
      x_test[i] = x[permBigG[i]];
  }
  cudaMalloc(&x_test_gpu, sizeof(float) * N);
  cudaMemcpy(x_test_gpu, x_test, sizeof(float) * N, cudaMemcpyHostToDevice);
}

void CSRk_Graph::setY(float *y_vec) {
  y = y_vec;
  cudaMalloc(&y_gpu, sizeof(float) * N);
  cudaMemcpy(y_gpu, y, sizeof(float) * N, cudaMemcpyHostToDevice);
}

float *CSRk_Graph::getY() {
  cudaMemcpy(y, y_gpu, sizeof(float) * N, cudaMemcpyDeviceToHost);
  return y;
}

// Generate Code For Different --k
// Solve for x, in L*x = b
void CSRk_Graph::lowerSTS() {
  if (k == 2) {
    for (int packIdx = 0; packIdx < numPacks; packIdx++) {
#pragma omp parallel for schedule(runtime)
      for (int j = packsPointer[packIdx]; j < packsPointer[packIdx + 1]; j++) {
        float tempVal = 0.0;
        for (unsigned int k = num_edges_L[j]; k < num_edges_L[j + 1] - 1; k++)
          tempVal += (val_L[k] * x[adj_L[k]]);
        x[j] = (b[j] - tempVal) / val_L[num_edges_L[j + 1] - 1];
      }
    }
  } else if (k == 3) {

    for (int packIdx = 0; packIdx < numPacks; packIdx++) {
#pragma omp parallel for schedule(runtime)
      for (int j = packsPointer[packIdx]; j < packsPointer[packIdx + 1]; j++) {

        const unsigned int stratRowIndex = mapCoarseToFinerRows[k - 2][j];
        const unsigned int endRowIndex = mapCoarseToFinerRows[k - 2][j + 1];

        float tempVal = 0.0;
        for (unsigned int rowIndex = stratRowIndex; rowIndex < endRowIndex;
             rowIndex++) {
          tempVal = 0.0;
          for (unsigned int k = num_edges_L[rowIndex];
               k < num_edges_L[rowIndex + 1] - 1; k++)
            tempVal += (val_L[k] * x[adj_L[k]]);
          x[rowIndex] =
              (b[rowIndex] - tempVal) / val_L[num_edges_L[rowIndex + 1] - 1];
        }
      }
    }
  } else if (k == 4) {

    for (int packIdx = 0; packIdx < numPacks; packIdx++) {
#pragma omp parallel for schedule(runtime)
      for (int j = packsPointer[packIdx]; j < packsPointer[packIdx + 1]; j++) {

        const unsigned int startSupRow = mapCoarseToFinerRows[k - 2][j];
        const unsigned int endSupRow = mapCoarseToFinerRows[k - 2][j + 1];

        for (unsigned int supRowIndex = startSupRow; supRowIndex < endSupRow;
             supRowIndex++) {

          const unsigned int stratRowIndex =
              mapCoarseToFinerRows[k - 3][supRowIndex];
          const unsigned int endRowIndex =
              mapCoarseToFinerRows[k - 3][supRowIndex + 1];

          float tempVal = 0.0;

          for (unsigned int rowIndex = stratRowIndex; rowIndex < endRowIndex;
               rowIndex++) {
            tempVal = 0.0;
            for (unsigned int k = num_edges_L[rowIndex];
                 k < num_edges_L[rowIndex + 1] - 1; k++)
              tempVal += (val_L[k] * x[adj_L[k]]);
            x[rowIndex] =
                (b[rowIndex] - tempVal) / val_L[num_edges_L[rowIndex + 1] - 1];
          }
        }
      }
    }
  } else {
    cout << "STS is not defined for this k-value" << endl;
  }
}

__global__ void
cuSpMV_3_vec(unsigned int *__restrict__ numCoarsestRows_gpu,
             unsigned int *__restrict__ mapCoarseToFinerRows_gpu_outer,
             unsigned int *__restrict__ mapCoarseToFinerRows_gpu_inner,
             unsigned int *__restrict__ r_vec_gpu,
             unsigned int *__restrict__ c_vec_gpu, float *__restrict__ val_gpu,
             float *__restrict__ x_test_gpu, float *__restrict__ y_gpu) {
  int block = blockIdx.x;
  int othread = threadIdx.z;
  int otstride = blockDim.z;
  int mthread = threadIdx.y;
  int mtstride = blockDim.y;
  int ithread = threadIdx.x;
  int itstride = blockDim.x;
  int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x +
            threadIdx.x;

  int numCoarsestRows;

  numCoarsestRows = *numCoarsestRows_gpu;

  const unsigned int ostart_index = mapCoarseToFinerRows_gpu_outer[block];
  const unsigned int oend_index = mapCoarseToFinerRows_gpu_outer[block + 1];

  for (int i_sup_row_i = ostart_index + othread; i_sup_row_i < oend_index;
       i_sup_row_i += otstride) {
    const unsigned int start_index =
        mapCoarseToFinerRows_gpu_inner[i_sup_row_i];
    const unsigned int end_index =
        mapCoarseToFinerRows_gpu_inner[i_sup_row_i + 1];

    for (unsigned int rowIndex = start_index + mthread; rowIndex < end_index;
         rowIndex += mtstride) {
      int nnz_index_start = (int)r_vec_gpu[rowIndex];
      int nnz_index_end = (int)r_vec_gpu[rowIndex + 1];
      float temp = 0;

      __shared__ volatile float values[544];
      for (int nnz_index = nnz_index_start + ithread; nnz_index < nnz_index_end;
           nnz_index += itstride) {
        temp += val_gpu[nnz_index] * x_test_gpu[c_vec_gpu[nnz_index]];
      }

      values[tid] = temp;

      if (itstride == 32)
        values[tid] += values[tid + 16];
      if (itstride >= 16)
        values[tid] += values[tid + 8];
      if (itstride >= 8)
        values[tid] += values[tid + 4];
      values[tid] += values[tid + 2];
      values[tid] += values[tid + 1];

      if (ithread == 0)
        y_gpu[rowIndex] = values[tid];
    }
  }
}

__global__ void
cuSpMV_3(unsigned int *__restrict__ numCoarsestRows_gpu,
         unsigned int *__restrict__ mapCoarseToFinerRows_gpu_outer,
         unsigned int *__restrict__ mapCoarseToFinerRows_gpu_inner,
         unsigned int *__restrict__ r_vec_gpu,
         unsigned int *__restrict__ c_vec_gpu, float *__restrict__ val_gpu,
         float *__restrict__ x_test_gpu, float *__restrict__ y_gpu) {
  int block = blockIdx.x;
  int othread = threadIdx.y;
  int otstride = blockDim.y;
  int mthread = threadIdx.x;
  int mtstride = blockDim.x;

  int numCoarsestRows;

  numCoarsestRows = *numCoarsestRows_gpu;

  const unsigned int ostart_index = mapCoarseToFinerRows_gpu_outer[block];
  const unsigned int oend_index = mapCoarseToFinerRows_gpu_outer[block + 1];

  for (int i_sup_row_i = ostart_index + othread; i_sup_row_i < oend_index;
       i_sup_row_i += otstride) {
    const unsigned int start_index =
        mapCoarseToFinerRows_gpu_inner[i_sup_row_i];
    const unsigned int end_index =
        mapCoarseToFinerRows_gpu_inner[i_sup_row_i + 1];

    for (unsigned int rowIndex = start_index + mthread; rowIndex < end_index;
         rowIndex += mtstride) {
      int nnz_index_start = (int)r_vec_gpu[rowIndex];
      int nnz_index_end = (int)r_vec_gpu[rowIndex + 1];
      float temp = 0;

      int nnz = nnz_index_end - nnz_index_start;

      while (nnz > 7) {
        temp +=
            val_gpu[nnz_index_start] * x_test_gpu[c_vec_gpu[nnz_index_start]];
        temp += val_gpu[nnz_index_start + 1] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 1]];
        temp += val_gpu[nnz_index_start + 2] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 2]];
        temp += val_gpu[nnz_index_start + 3] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 3]];
        nnz -= 4;
        nnz_index_start += 4;
      }

      if (nnz & 0b100) {
        temp +=
            val_gpu[nnz_index_start] * x_test_gpu[c_vec_gpu[nnz_index_start]];
        temp += val_gpu[nnz_index_start + 1] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 1]];
        temp += val_gpu[nnz_index_start + 2] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 2]];
        temp += val_gpu[nnz_index_start + 3] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 3]];
        nnz_index_start += 4;
      }
      if (nnz & 0b010) {
        temp +=
            val_gpu[nnz_index_start] * x_test_gpu[c_vec_gpu[nnz_index_start]];
        temp += val_gpu[nnz_index_start + 1] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 1]];
        nnz_index_start += 2;
      }
      if (nnz & 0b001) {
        temp +=
            val_gpu[nnz_index_start] * x_test_gpu[c_vec_gpu[nnz_index_start]];
      }

      y_gpu[rowIndex] = temp;
    }
  }
}

__global__ void cuSpMV_2(unsigned int *__restrict__ numCoarsestRows_gpu,
                         unsigned int *__restrict__ mapCoarseToFinerRows_gpu,
                         unsigned int *__restrict__ r_vec_gpu,
                         unsigned int *__restrict__ c_vec_gpu,
                         float *__restrict__ val_gpu,
                         float *__restrict__ x_test_gpu,
                         float *__restrict__ y_gpu) {
  int block = blockIdx.x;
  int bstride = gridDim.x;
  int othread = threadIdx.z * blockDim.y * blockDim.x +
                threadIdx.y * blockDim.x + threadIdx.x;
  int otstride = blockDim.x * blockDim.y * blockDim.z;
  int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x +
            threadIdx.x;

  int numCoarsestRows;

  numCoarsestRows = *numCoarsestRows_gpu;

  for (int i_sup_row = block; i_sup_row < numCoarsestRows;
       i_sup_row += bstride) {
    const unsigned int start_index = mapCoarseToFinerRows_gpu[i_sup_row];
    const unsigned int end_index = mapCoarseToFinerRows_gpu[i_sup_row + 1];

    for (unsigned int rowIndex = start_index + othread; rowIndex < end_index;
         rowIndex += otstride) {
      int nnz_index_start = (int)r_vec_gpu[rowIndex];
      int nnz_index_end = (int)r_vec_gpu[rowIndex + 1];
      float temp = 0;

      int nnz = nnz_index_end - nnz_index_start;

      while (nnz > 7) {
        temp +=
            val_gpu[nnz_index_start] * x_test_gpu[c_vec_gpu[nnz_index_start]];
        temp += val_gpu[nnz_index_start + 1] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 1]];
        temp += val_gpu[nnz_index_start + 2] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 2]];
        temp += val_gpu[nnz_index_start + 3] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 3]];
        nnz -= 4;
        nnz_index_start += 4;
      }

      if (nnz & 0b100) {
        temp +=
            val_gpu[nnz_index_start] * x_test_gpu[c_vec_gpu[nnz_index_start]];
        temp += val_gpu[nnz_index_start + 1] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 1]];
        temp += val_gpu[nnz_index_start + 2] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 2]];
        temp += val_gpu[nnz_index_start + 3] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 3]];
        nnz_index_start += 4;
      }
      if (nnz & 0b010) {
        temp +=
            val_gpu[nnz_index_start] * x_test_gpu[c_vec_gpu[nnz_index_start]];
        temp += val_gpu[nnz_index_start + 1] *
                x_test_gpu[c_vec_gpu[nnz_index_start + 1]];
        nnz_index_start += 2;
      }
      if (nnz & 0b001) {
        temp +=
            val_gpu[nnz_index_start] * x_test_gpu[c_vec_gpu[nnz_index_start]];
      }

      y_gpu[rowIndex] = temp;
    }
  }
}

// Generate Code For Different --k
// Computes SpMV y =  A * x
void CSRk_Graph::SpMV() {

  if (k == 1) {

#pragma omp parallel for schedule(runtime)
    for (int rowIndex = 0; rowIndex < N; rowIndex++) {
      float temp_y = 0;
      for (unsigned int nnz_index = r_vec[rowIndex];
           nnz_index < r_vec[rowIndex + 1]; nnz_index++) {
        temp_y += val[nnz_index] * x_test[c_vec[nnz_index]];
      }
      y[rowIndex] = temp_y;
    }

  } else if (k == 2) // k=2 or CSR-2
  {
#pragma omp parallel for schedule(runtime)
    for (int i_sup_row = 0; i_sup_row < numCoarsestRows; i_sup_row++) {
      const unsigned int start_index = mapCoarseToFinerRows[k - 1][i_sup_row];
      const unsigned int end_index = mapCoarseToFinerRows[k - 1][i_sup_row + 1];

      for (unsigned int rowIndex = start_index; rowIndex < end_index;
           rowIndex++) {
        float temp_y = 0;
        unsigned int nnz_index_start = r_vec[rowIndex];
        unsigned int nnz_index_end = r_vec[rowIndex + 1];

        for (unsigned int nnz_index = nnz_index_start;
             nnz_index < nnz_index_end; nnz_index++) {
          temp_y += val[nnz_index] * x_test[c_vec[nnz_index]];
        }
        y[rowIndex] = temp_y;
      }
    }
  } else if (k == 3) // k = 3 or CSR-3
  {

#pragma omp parallel for schedule(runtime)
    for (unsigned int supSupRowIndex = 0; supSupRowIndex < numCoarsestRows;
         supSupRowIndex++) {
      const unsigned int startSupRow =
          mapCoarseToFinerRows[k - 1][supSupRowIndex];
      const unsigned int endSupRow =
          mapCoarseToFinerRows[k - 1][supSupRowIndex + 1];

      for (unsigned int supRowIndex = startSupRow; supRowIndex < endSupRow;
           supRowIndex++) {

        const unsigned int stratRowIndex =
            mapCoarseToFinerRows[k - 2][supRowIndex];
        const unsigned int endRowIndex =
            mapCoarseToFinerRows[k - 2][supRowIndex + 1];

        float temp_y = 0;

        for (unsigned int rowIndex = stratRowIndex; rowIndex < endRowIndex;
             rowIndex++) {
          temp_y = 0;
          for (unsigned int nnzIndex = r_vec[rowIndex];
               nnzIndex < r_vec[rowIndex + 1]; nnzIndex++) {
            temp_y += val[nnzIndex] * x_test[c_vec[nnzIndex]];
          }
          y[rowIndex] = temp_y;
        }
      }
    }
  } else if (k == 4) {
    // 3 2 1
#pragma omp parallel for schedule(runtime)
    for (unsigned int supSupSupRowIndex = 0;
         supSupSupRowIndex < numCoarsestRows; supSupSupRowIndex++) {
      const unsigned int startSupSupRow =
          mapCoarseToFinerRows[k - 1][supSupSupRowIndex];
      const unsigned int endSupSupRow =
          mapCoarseToFinerRows[k - 1][supSupSupRowIndex + 1];

      for (unsigned int supSupRowIndex = startSupSupRow;
           supSupRowIndex < endSupSupRow; supSupRowIndex++) {
        const unsigned int startSupRow =
            mapCoarseToFinerRows[k - 2][supSupRowIndex];
        const unsigned int endSupRow =
            mapCoarseToFinerRows[k - 2][supSupRowIndex + 1];

        for (unsigned int supRowIndex = startSupRow; supRowIndex < endSupRow;
             supRowIndex++) {

          const unsigned int stratRowIndex =
              mapCoarseToFinerRows[k - 3][supRowIndex];
          const unsigned int endRowIndex =
              mapCoarseToFinerRows[k - 3][supRowIndex + 1];

          float temp_y = 0;

          for (unsigned int rowIndex = stratRowIndex; rowIndex < endRowIndex;
               rowIndex++) {
            temp_y = 0;
            for (unsigned int nnzIndex = r_vec[rowIndex];
                 nnzIndex < r_vec[rowIndex + 1]; nnzIndex++) {
              temp_y += val[nnzIndex] * x_test[c_vec[nnzIndex]];
            }
            y[rowIndex] = temp_y;
          }
        }
      }
    }

  } else {
    cout << "SpMV-Kernel is not defined for this k-value" << endl;
  }
  // k -- is given
}

// Default Constructor
CSRk_Graph::CSRk_Graph() {
#ifdef DEBUG
  cout << "CSRk_Graph::Default Constructor called. " << endl;
#endif

  N = 0;
  M = 0;
  NNZ = 0;

  r_vec = NULL;
  c_vec = NULL;
  val = NULL;

  k = 0;
  numCoarsestRows = 0;

  mapCoarseToFinerRows = NULL;
  permBigG = NULL;

  kernelType = "SpMV";
  ifTuned = false;
}

// Constructor with input parameters
CSRk_Graph::CSRk_Graph(long nRows, long nCols, long nnz, unsigned int *rVec,
                       unsigned int *cVec, float *value, string kernelCalled,
                       string inOrderType, string inCoarsenType, bool isTuned,
                       int inK, int *inSupRowSizes) {
#ifdef DEBUG
  cout << "CSRk_Graph::Parameterized Constructor called. " << endl;
#endif
  N = nRows;
  M = nCols;
  NNZ = nnz;

  r_vec = new unsigned int[N + 1];
  if (r_vec == NULL) {
    cout << "Memory can't be allocated for: r_vec. " << endl
         << "Exiting." << endl
         << endl;
  }

  c_vec = new unsigned int[NNZ];
  if (c_vec == NULL) {
    cout << "Memory can't be allocated for: c_vec. " << endl
         << "Exiting." << endl
         << endl;
  }

  val = new float[NNZ];
  if (val == NULL) {
    cout << "Memory can't be allocated for: val. " << endl
         << "Exiting." << endl
         << endl;
  }

#pragma omp parallel
  {
#pragma omp for schedule(static) nowait
    for (int i = 0; i < nRows + 1; i++)
      r_vec[i] = rVec[i];
#pragma omp barrier

#pragma omp for schedule(static) nowait
    for (int i = 0; i < NNZ; i++)
      c_vec[i] = cVec[i];
#pragma omp barrier

#pragma omp for schedule(static) nowait
    for (int i = 0; i < NNZ; i++)
      val[i] = value[i];
#pragma omp barrier
  }
  cudaMalloc(&val_gpu, sizeof(float) * NNZ);
  cudaMalloc(&r_vec_gpu, sizeof(unsigned int) * (N + 1));
  cudaMalloc(&c_vec_gpu, sizeof(unsigned int) * NNZ);
  cudaMemcpy(val_gpu, val, sizeof(float) * NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(r_vec_gpu, r_vec, sizeof(unsigned int) * (N + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(c_vec_gpu, c_vec, sizeof(unsigned int) * NNZ,
             cudaMemcpyHostToDevice);

  kernelType = kernelCalled;
  orderType = inOrderType;
  coarsenType = inCoarsenType;
  ifTuned = isTuned;

  k = inK;
  mapCoarseToFinerRows = NULL;
  supRowSizes = NULL;
  permBigG = NULL;
  numCoarsestRows = 0;

  x_test = new float[N];

  if (kernelType == "SpMV" && k > 1) {
    permBigG = new unsigned int[nRows];

    mapCoarseToFinerRows = new unsigned int *[k + 1];
    permOfFinerRows = new unsigned int *[k];

    for (int i = 0; i < k; i++) {
      mapCoarseToFinerRows[i] = NULL;
      permOfFinerRows[i] = NULL;
    }

    supRowSizes = new unsigned int[k - 1];

    for (int i = 0; i < k - 1; i++)
      supRowSizes[i] = inSupRowSizes[i];

  } else if (kernelType == "STS" && k > 2) {
    permBigG = new unsigned int[nRows];

    mapCoarseToFinerRows = new unsigned int *[k];
    permOfFinerRows = new unsigned int *[k];

    for (int i = 0; i < k; i++) {
      mapCoarseToFinerRows[i] = NULL;
      permOfFinerRows[i] = NULL;
    }

    supRowSizes = new unsigned int[k - 1];

    for (int i = 0; i < k - 1; i++)
      supRowSizes[i] = inSupRowSizes[i];
  } else if (kernelType == "STS" && k == 2) {
    permBigG = new unsigned int[nRows];
  }

#ifdef DEBUG
  cout << "CSRk_Graph::Done creating Object with Parameterized Constructor. "
       << endl;
#endif
}

// Destructor definition
CSRk_Graph::~CSRk_Graph() {
#ifdef DEBUG
  cout << "CSRk_Graph: Destructor called. " << endl;
#endif

  if (r_vec != NULL)
    delete[] r_vec;

  if (c_vec != NULL)
    delete[] c_vec;

  if (val != NULL)
    delete[] val;

  if (permBigG != NULL)
    delete[] permBigG;

  if (x_test != NULL)
    delete[] x_test;

  if (mapCoarseToFinerRows != NULL) {
    for (int i = 0; i < k; i++) {
      if (mapCoarseToFinerRows[i] != NULL)
        delete[] mapCoarseToFinerRows[i];
    }

    delete[] mapCoarseToFinerRows;

    for (int i = 0; i < k; i++) {
      if (permOfFinerRows[i] != NULL)
        delete[] permOfFinerRows[i];
    }

    delete[] permOfFinerRows;
  }

  if (supRowSizes != NULL)
    delete[] supRowSizes;

  // For STS
  // Just check the kernel type and decide
  if (this->kernelType == STSKernel) {

    if (packsPointer != NULL)
      delete[] packsPointer;

    if (num_edges_L != NULL) {
      delete[] num_edges_L;
    }

    if (adj_L != NULL)
      delete[] adj_L;

    if (val_L != NULL)
      delete[] val_L;

    if (num_edges_U != NULL)
      delete[] num_edges_U;

    if (adj_U != NULL)
      delete[] adj_U;

    if (val_U != NULL)
      delete[] val_U;

    if (b != NULL)
      delete[] b;

    if (x != NULL)
      delete[] x;
  }

#ifdef DEBUG
  cout << "CSRk_Graph: Done destructing the Object. " << endl;
#endif
}

// Renumbers the vertices and puts the values at proper places
void CSRk_Graph::reorderA() {
#ifdef DEBUG
  cout << "CSRk_Graph::reorderA invoked." << endl;
#endif

  // Change the CSR of original matrix, using permutation permBigG
  unsigned int *A_r_vec = new unsigned int[N + 1];
  unsigned int *A_c_vec = new unsigned int[NNZ];
  float *A_val_vec = new float[NNZ];

#pragma omp parallel
  {
// Copy r_vec
#pragma omp for schedule(static) nowait
    for (unsigned int i = 0; i < N + 1; i++) {
      if (r_vec[i] > NNZ) {
        cout << "WRONG NNZ " << NNZ << " r-vec val: " << r_vec[i] << " i: " << i
             << endl;
      }
      A_r_vec[i] = r_vec[i];
    }
#pragma omp barrier

// Copy c_vec
#pragma omp for schedule(static) nowait
    for (unsigned int i = 0; i < NNZ; i++) {
      A_c_vec[i] = c_vec[i];
    }
#pragma omp barrier

    // Copy val
#pragma omp for schedule(static) nowait
    for (unsigned int i = 0; i < NNZ; i++) {
      A_val_vec[i] = val[i];
    }
  }

  unsigned int numNNZ = 0;
  r_vec[0] = 0;

  for (unsigned int i = 0; i < N; i++) {
    unsigned int old_vtx = permBigG[i];

    unsigned int start_index = A_r_vec[old_vtx];
    unsigned int end_index = A_r_vec[old_vtx + 1];

    numNNZ = numNNZ + (end_index - start_index);

    r_vec[i + 1] = numNNZ;
  }

  // assert( r_vec [N] == A_r_vec[N] );

  unsigned int *forwardMap = new unsigned int[N];

#pragma omp parallel
  {

#pragma omp for schedule(static) nowait
    for (long i = 0; i < N; i++) {
      // if(permBigG[i] >= N)
      // cout<<"Wrong wrong "<<endl;
      forwardMap[permBigG[i]] = i;
    }
#pragma omp barrier

#pragma omp for schedule(static)
    for (long i = 0; i < N; i++) {
      unsigned int old_vtx = permBigG[i];

      unsigned int start_index = A_r_vec[old_vtx];
      unsigned int end_index = A_r_vec[old_vtx + 1];

      unsigned int new_start_index = r_vec[i];

      // if(new_start_index >= NNZ)
      // cout<<"Wrong wrong"<<endl;
      for (unsigned int j = start_index; j < end_index; j++) {
        c_vec[new_start_index] = forwardMap[A_c_vec[j]];
        val[new_start_index] = A_val_vec[j];
        // Make a map from index to index
        // Map new to old NNZ index:
        // Map_index[new_start_index] = j

        new_start_index++;
      }
    }
  }

  std::vector<degree_id_pair> deg_id_vec(NNZ);

#pragma omp parallel
  {
#pragma omp for schedule(static) nowait
    for (long i = 0; i < NNZ; i++) {
      deg_id_vec[i].deg = c_vec[i];
      deg_id_vec[i].id = i;

      A_val_vec[i] = val[i];
    }
#pragma omp barrier

#pragma omp for schedule(static) nowait
    for (long i = 0; i < N; i++)
      std::sort(deg_id_vec.begin() + r_vec[i],
                deg_id_vec.begin() + r_vec[i + 1], compare_deg_id_pair);
#pragma omp barrier

#pragma omp for schedule(static) nowait
    for (long i = 0; i < N; i++) {
      unsigned int startIndex = r_vec[i];
      unsigned int endIndex = r_vec[i + 1];

      for (unsigned int j = startIndex; j < endIndex; j++) {
        c_vec[j] = deg_id_vec[j].deg;
        val[j] = A_val_vec[deg_id_vec[j].id];
        // val[j] = A_val_vec[ Map_index[ deg_id_vec[j].id ] ];
      }
    }
  }

  delete[] A_val_vec;
  delete[] A_r_vec;
  delete[] A_c_vec;

  cudaMalloc(&mapCoarseToFinerRows_gpu_outer,
             sizeof(unsigned int) * (numCoarsestRows + 1));
  cudaMemcpy(mapCoarseToFinerRows_gpu_outer, mapCoarseToFinerRows[k - 1],
             sizeof(unsigned int) * (numCoarsestRows + 1),
             cudaMemcpyHostToDevice);
  cudaMalloc(&mapCoarseToFinerRows_gpu_inner,
             sizeof(unsigned int) * (numCoarsishRows + 1));
  cudaMemcpy(mapCoarseToFinerRows_gpu_inner, mapCoarseToFinerRows[k - 2],
             sizeof(unsigned int) * (numCoarsishRows + 1),
             cudaMemcpyHostToDevice);
  cudaMalloc(&numCoarsestRows_gpu, sizeof(unsigned int));
  cudaMalloc(&N_gpu, sizeof(unsigned int));
  cudaMalloc(&M_gpu, sizeof(unsigned int));
  cudaMalloc(&NNZ_gpu, sizeof(unsigned int));
  cudaMemcpy(N_gpu, &N, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(M_gpu, &M, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(NNZ_gpu, &NNZ, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(numCoarsestRows_gpu, &numCoarsestRows, sizeof(unsigned int),
             cudaMemcpyHostToDevice);

#ifdef DEBUG
  cout << "CSRk_Graph::reorderA completed." << endl;
#endif
}

// Corsens the matrix k-1 times
// Finds the coarsen to fine graph mapping
// Also computes the matrix reordering
void CSRk_Graph::putInCSRkFormat() {
#ifdef DEBUG
  cout << "CSRk_Graph::putInCSRkFormat invoked" << endl;
#endif

  if (k < 2)
    return;

  if (kernelType == SpMVKernel) {
    BAND_k bandk(k);
    bandk.preprocessingForSpMV(*this);
    reorderA();
  } else if (kernelType == STSKernel) {
    BAND_k bandk(k - 1);
    bandk.preprocessingForSTS(*this);
    reorderA();
  } else {
    // Add a new kernel here
    cout << "Undefined kernel invoked" << endl;
    exit(1);
  }

#ifdef DEBUG
  cout << "CSRk_Graph::putInCSRkFormat completed" << endl;
#endif
}

void CSRk_Graph::incomplete_choloskey() {

#ifdef DEBUG
  cout << "CSRk_Graph::incomplete_choloskey non-zero structure invoked" << endl;
#endif

  long num_non_zeros_in_L = 0;
  long num_non_zeros_in_U = 0;
  bool self_edge_found = false;

  for (unsigned int i_row = 0; i_row < N; i_row++) {
    for (unsigned int j_ind = r_vec[i_row]; j_ind < r_vec[i_row + 1]; j_ind++) {
      if (i_row == c_vec[j_ind]) {
        self_edge_found = true;
        num_non_zeros_in_L++;
        num_non_zeros_in_U++;
      } else if (c_vec[j_ind] < i_row) {
        num_non_zeros_in_L++;
      } else if (c_vec[j_ind] > i_row) {
        num_non_zeros_in_U++;
      }
    }

    if (!self_edge_found) {
      printf("Self edge not found\n");
      exit(1);
      num_non_zeros_in_L++;
      num_non_zeros_in_U++;
    } else {
      self_edge_found = false;
    }
  }

  num_edges_L = new unsigned int[N + 1];
  adj_L = new unsigned int[num_non_zeros_in_L];
  val_L = new float[num_non_zeros_in_L];

  num_edges_U = new unsigned int[N + 1];
  adj_U = new unsigned int[num_non_zeros_in_U];
  val_U = new float[num_non_zeros_in_U];

  num_non_zeros_in_L = 0;
  num_non_zeros_in_U = 0;

  for (unsigned int i_row = 0; i_row < N; i_row++) {
    num_edges_L[i_row] = num_non_zeros_in_L;
    num_edges_U[i_row] = num_non_zeros_in_U;

    for (unsigned int j_ind = r_vec[i_row]; j_ind < r_vec[i_row + 1]; j_ind++) {
      if (c_vec[j_ind] < i_row) {
        adj_L[num_non_zeros_in_L] = c_vec[j_ind];
        val_L[num_non_zeros_in_L] = val[j_ind];
        num_non_zeros_in_L++;
      } else if (c_vec[j_ind] == i_row) {
        adj_L[num_non_zeros_in_L] = c_vec[j_ind];
        val_L[num_non_zeros_in_L] = val[j_ind];
        num_non_zeros_in_L++;

        adj_U[num_non_zeros_in_U] = c_vec[j_ind];
        val_U[num_non_zeros_in_U] = val[j_ind];
        num_non_zeros_in_U++;

      } else if (c_vec[j_ind] > i_row) {

        adj_U[num_non_zeros_in_U] = c_vec[j_ind];
        val_U[num_non_zeros_in_U] = val[j_ind];
        num_non_zeros_in_U++;
      }
    }
  }

  num_edges_L[N] = num_non_zeros_in_L;
  num_edges_U[N] = num_non_zeros_in_U;

#ifdef DEBUG
  cout << "CSRk_Graph::incomplete_choloskey non-zero structure completed"
       << endl;
  cout << "NNZ in L:" << num_non_zeros_in_L
       << " NNZ in U: " << num_non_zeros_in_U << endl;
#endif
}

void CSRk_Graph::compute_b() {
  x_test = new float[N];
  x = new float[N];
  b = new float[N];

  for (int i = 0; i < N; i++) {
    x_test[i] = 1.0;
    x[i] = 0.0;
  }

  for (int i = 0; i < N; i++) {
    float temVal = 0.0;
    for (unsigned int j = num_edges_L[i]; j < num_edges_L[i + 1]; j++) {
      temVal += val_L[j] * x_test[adj_L[j]];
    }
    b[i] = temVal;
  }
}

void CSRk_Graph::checkError() {
  float totError = 0;

  if (N > 10) {
    cout << "First 5 values:" << endl;

    for (int i = 0; i < 5; i++)
      cout << x[i] << "\t";
    cout << "" << endl << endl;

    cout << "Last 5 values:" << endl;
    for (int i = N - 5; i < N; i++)
      cout << x[i] << "\t";
    cout << "" << endl << endl;
  } else {
    cout << "The x values:" << endl;
    for (int i = 0; i < N; i++)
      cout << x[i] << "\t";
    cout << "" << endl << endl;
  }

  for (int i = 0; i < N; i++)
    totError += (x[i] - x_test[i]);

  cout << "Total Error: " << totError << endl;
}

/////////////////////////////////////////////////////
//       Definitions of functions in Band-k        //
/////////////////////////////////////////////////////

// Put the matrix in CSR-k format SPMV
void BAND_k::preprocessingForSpMV(CSRk_Graph &csrkGraph) {
#ifdef DEBUG
  cout << "BAND_k::preprocessingForSpMV invoked" << endl;
  cout << "CSRk_Graph::k " << csrkGraph.k << endl;
  cout << "BAND_k::k " << k << endl;
#endif

  unsigned int **graphPermutations =
      (unsigned int **)malloc(sizeof(unsigned int *) * k);
  unsigned int **reverseGraphPermutations =
      (unsigned int **)malloc(sizeof(unsigned int *) * k);

  for (int i = 0; i < k; i++) {
    graphPermutations[i] = NULL;
    reverseGraphPermutations[i] = NULL;
  }

#ifdef DEBUG
  cout << "BAND_k::preprocessingForSpMV invoked" << endl;
#endif

  smallGraphs[0].N = csrkGraph.N;
  smallGraphs[0].NNZ = csrkGraph.NNZ;

  smallGraphs[0].r_vec = new unsigned int[smallGraphs[0].N + 1];
  smallGraphs[0].c_vec = new unsigned int[smallGraphs[0].NNZ];
  unsigned int *inDegree = new unsigned int[smallGraphs[0].NNZ];
  unsigned int *inVertexWeight = new unsigned int[smallGraphs[0].N];
  unsigned int *inEdgeWeight = new unsigned int[smallGraphs[0].NNZ];

  for (int i = 0; i < smallGraphs[0].N; i++) {
    smallGraphs[0].r_vec[i] = csrkGraph.r_vec[i];
    inVertexWeight[i] = 1;
  }
  smallGraphs[0].r_vec[smallGraphs[0].N] = csrkGraph.r_vec[smallGraphs[0].N];

  for (int i = 0; i < smallGraphs[0].NNZ; i++) {
    inDegree[i] = 1;
    inEdgeWeight[i] = 1;
    smallGraphs[0].c_vec[i] = csrkGraph.c_vec[i];
  }
  smallGraphs[0].degree = inDegree;
  smallGraphs[0].vertexWeight = inVertexWeight;
  smallGraphs[0].edgeWeight = inEdgeWeight;

  graphPermutations[0] = new unsigned int[csrkGraph.N];

  for (int i = 0; i < csrkGraph.N; i++) {
    graphPermutations[0][i] = i;
  }

  // int supNodeSize = 80*csrkGraph.NNZ / csrkGraph.N;

  // Corsen k-1 times and do RCM everytime for hand-coarsening
  for (int i = 1; i < k; i++) {
    int supRowSizeInNNZ = csrkGraph.supRowSizes[i - 1] *
                          smallGraphs[i - 1].NNZ / smallGraphs[i - 1].N;

    // Coarsen the (i-1)-th graph to get i-th graph
    coarsenTheGraph(i, (int)csrkGraph.supRowSizes[i - 1], supRowSizeInNNZ,
                    csrkGraph);

    // do RCM everytime for hand-coarsening
    if (csrkGraph.coarsenType == HAND) {
      // RCM on i-th graph and change the mapping from i-th graph to (i-1)-graph
      int numVertices = smallGraphs[i].N;

      // new to old map
      unsigned int *corsenedGraphPerm = new unsigned int[numVertices];
      unsigned int *ReverseGraphPerm = new unsigned int[numVertices];

      unsigned int *mask = new unsigned int[numVertices];

      for (int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++)
        mask[vertexIndex] = 1;

      int firstVtxinBFS, lastVtxinBFS, ccSize = 0, root = 0;

      // To do RCM on the whole graph call RCM on each connected component
      for (int i_mask = 0; i_mask < numVertices; i_mask++) {
        if (ccSize >= numVertices)
          break;

        if (mask[i_mask] != 0) {
          root = i_mask;

          rcm_reordering_g(numVertices, (int *)smallGraphs[i].r_vec,
                           (int *)smallGraphs[i].c_vec,
                           (int *)smallGraphs[i].degree, root, (int *)mask,
                           (int *)ReverseGraphPerm, (int *)corsenedGraphPerm,
                           firstVtxinBFS, lastVtxinBFS, ccSize);
        }
      }

#ifdef DEBUG
      cout << "Done with RCM on: " << i << "-th Coarsened graph " << endl;
#endif

      // Reorder the graph using the reordered vartices
      // No need to renumber the k-1-th graph --that is the last graph
      renumberGraphUsingReorderedVertices(
          numVertices, smallGraphs[i].NNZ, &smallGraphs[i].r_vec,
          &smallGraphs[i].c_vec, &smallGraphs[i].degree, ReverseGraphPerm,
          corsenedGraphPerm);

      graphPermutations[i] = corsenedGraphPerm;
      reverseGraphPermutations[i] = ReverseGraphPerm;

      delete[] mask;
    } else {
      // csrkGraph.mapCoarseToFinerRows[i]
      // csrkGraph.permOfFinerRows[i]
      if (i == k - 1) {
        int numVertices = smallGraphs[i].N;

        // new to old map
        unsigned int *corsenedGraphPerm = new unsigned int[numVertices];
        // Old to New map
        unsigned int *ReverseGraphPerm = new unsigned int[numVertices];

        unsigned int *mask = new unsigned int[numVertices];

        for (int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++)
          mask[vertexIndex] = 1;

        int firstVtxinBFS, lastVtxinBFS, ccSize = 0, root = 0;

        // To do RCM on the whole graph call RCM on each connected component
        for (int i_mask = 0; i_mask < numVertices; i_mask++) {
          if (ccSize >= numVertices)
            break;

          if (mask[i_mask] != 0) {
            root = i_mask;

            rcm_reordering_g(numVertices, (int *)smallGraphs[i].r_vec,
                             (int *)smallGraphs[i].c_vec,
                             (int *)smallGraphs[i].degree, root, (int *)mask,
                             (int *)ReverseGraphPerm, (int *)corsenedGraphPerm,
                             firstVtxinBFS, lastVtxinBFS, ccSize);
          }
        }

#ifdef DEBUG
        cout << "Done with RCM on: " << i << "-th Coarsened graph " << endl;
#endif

        /*
                                  //Reorder the graph using the reordered
           vartices
                                  //No need to renumber the k-1-th graph --that
           is the last graph renumberGraphUsingReorderedVertices(numVertices,
           smallGraphs[i].NNZ, &smallGraphs[i].r_vec, &smallGraphs[i].c_vec,
           &smallGraphs[i].degree, ReverseGraphPerm, corsenedGraphPerm);

                                  cout<<"After function call. "<<endl; */
        // graphPermutations[i] = corsenedGraphPerm;
        // reverseGraphPermutations[i] = ReverseGraphPerm;

        // New to old Id
        csrkGraph.permOfFinerRows[i] = corsenedGraphPerm;

        delete[] mask;
      }
    }
  }

  // csrkGraph.mapCoarseToFinerRows -- this contains map from super-rows to rows
  // --BEFORE reordering
  csrkGraph.numCoarsestRows = smallGraphs[k - 1].N;
  csrkGraph.numCoarsishRows = smallGraphs[k - 2].N;

  if (csrkGraph.coarsenType == HAND) {
    // Now uncoarsen the coarsened graphs and find permutation of original graph
    // Also find k-1 mapping from coarsened graphs to finer graphs
    for (int i = k - 1; i >= 1; i--) {

      // Use RCM maps to renumber the mapping when computing CSR-k
      // Change mapping from i-th graph to (i-1)-th graph using reordered
      // super-rows graphPermutations reverseGraphPermutations

      // Uncoarsen the coarsened graphs and find the reordering of the original
      // graph
      uncoarsenTheGraph(i, csrkGraph, graphPermutations[i],
                        graphPermutations[i - 1]);

      // RCM on i-th graph and change the mapping from i-th graph to (i-1)-graph
    }

    // Use the permutation of the original matrix to reorder the matrix
    // This sets the permutation of original matrix
    // Copy the permutation so you can free memory

    for (int orgVertexId = 0; orgVertexId < smallGraphs[0].N; orgVertexId++)
      csrkGraph.permBigG[orgVertexId] = graphPermutations[0][orgVertexId];

  } else {
    for (int i = k - 1; i >= 1; i--) {
      // Renumber the vertices in graph smallGraphs[i-1] using the reordering of
      // smallGraphs[i] vertices Uncoarsen the coarsened graphs and find the
      // reordering of the original graph i-th reordering is
      // csrkGraph.permOfFinerRows[i] and (i-1)-th reordering is
      // csrkGraph.permOfFinerRows[i]
      // Mapping from i-th to (i-1)-th graph is determined by
      // csrkGraph.mapCoarseToFinerRows[i] and csrkGraph.permOfFinerRows[i-1]
      matchingUncoarsenTheGraph(i, csrkGraph);
    }

    for (int orgVertexId = 0; orgVertexId < smallGraphs[0].N; orgVertexId++)
      csrkGraph.permBigG[orgVertexId] =
          csrkGraph.permOfFinerRows[0][orgVertexId];
  }

  for (int i = 0; i < k; i++) {
    if (graphPermutations[i] != NULL)
      delete[] graphPermutations[i];

    if (reverseGraphPermutations[i] != NULL)
      delete[] reverseGraphPermutations[i];
  }

  free(graphPermutations);
  free(reverseGraphPermutations);

#ifdef DEBUG
  cout << "BAND_k::preprocessingForSpMV completed" << endl;
#endif
}

// You an Do an RCM on the graph in csrkGraph.permOfVtxs[level-1]
void BAND_k::matchingUncoarsenTheGraph(int level, CSRk_Graph &csrkGraph) {
#ifdef DEBUG
  cout << "BAND_k::matchingUncoarsenTheGraph invoked " << endl;
#endif

  long numSupRows = smallGraphs[level].N;

  unsigned int *copyMapCoarseToFiner = new unsigned int[numSupRows + 1];

  for (int i = 0; i < numSupRows + 1; i++)
    copyMapCoarseToFiner[i] = csrkGraph.mapCoarseToFinerRows[level][i];

  int startVtxId = 0;

  csrkGraph.mapCoarseToFinerRows[level][0] = 0;

  for (int i = 0; i < numSupRows; i++) {
    unsigned int oldVtxId = csrkGraph.permOfFinerRows[level][i];
    startVtxId = startVtxId + (copyMapCoarseToFiner[oldVtxId + 1] -
                               copyMapCoarseToFiner[oldVtxId]);

    csrkGraph.mapCoarseToFinerRows[level][i + 1] = startVtxId;
  }

  // Now renumber the vertices in smallGraphs[level-1]
  // The rows in a super-row is already known
  /*********************
  *    REORDER each super-row  THE SUPER-ROW HERE ****************
  //You can reorder the vertices in a super-row using RCM -- now assume you
  don't have to reorder *
  ***********************************************************************/

  // new to old id -- PI2
  unsigned int *newPermutation = new unsigned int[smallGraphs[level - 1].N];

  for (int i = 0; i < numSupRows; i++) {
    unsigned int oldVtxId = csrkGraph.permOfFinerRows[level][i];

    unsigned int oldStartNNZ = copyMapCoarseToFiner[oldVtxId];
    unsigned int oldEndNNZ = copyMapCoarseToFiner[oldVtxId + 1];

    unsigned int newNNZ = csrkGraph.mapCoarseToFinerRows[level][i];

    for (unsigned int j = oldStartNNZ; j < oldEndNNZ; j++) {
      newPermutation[newNNZ] = csrkGraph.permOfFinerRows[level - 1][j];
      newNNZ++;
    }
  }

  // Copy the old permutation PI1
  unsigned int *copyOldPermutation = new unsigned int[smallGraphs[level - 1].N];

  for (int i = 0; i < smallGraphs[level - 1].N; i++)
    copyOldPermutation[i] = csrkGraph.permOfFinerRows[level - 1][i];

  // Need to change the permutation: composition of permutations--PI1.PI2
  // graphPermutation[level-1].NewPermutation (Function composition)
  // permFinerGraph -- gives new to old permutation of vertices in
  // smallGraphs[level-1]

  for (int i = 0; i < smallGraphs[level - 1].N; i++) {
    csrkGraph.permOfFinerRows[level - 1][i] =
        copyOldPermutation[newPermutation[i]];
  }

  delete[] newPermutation;
  delete[] copyOldPermutation;
  delete[] copyMapCoarseToFiner;

#ifdef DEBUG
  cout << "BAND_k::matchingUncoarsenTheGraph completed " << endl;
#endif
}

// This function changes the mapping from coarser to finer graph vertices(level
// to level -1) This also changes the permutation of finer graph (level-1)

// Uncoarsen a coarsen graph to find a mapping of the finer graph
void BAND_k::uncoarsenTheGraph(int level, CSRk_Graph &csrkGraph,
                               unsigned int *permCorserGraph,
                               unsigned int *&permFinerGraph) {
#ifdef DEBUG
  cout << "BAND_k::uncoarsenTheGraph invoked " << endl;
#endif

  // maps level to level -1
  // csrkGraph.mapCoarseToFinerRows[level]

  long numSupRows = smallGraphs[level].N;

  unsigned int *copyMapCoarseToFiner = new unsigned int[numSupRows + 1];

  for (int i = 0; i < numSupRows + 1; i++)
    copyMapCoarseToFiner[i] = csrkGraph.mapCoarseToFinerRows[level][i];

  int startVtxId = 0;

  csrkGraph.mapCoarseToFinerRows[level][0] = 0;

  for (int i = 0; i < numSupRows; i++) {
    unsigned int oldVtxId = permCorserGraph[i];
    startVtxId = startVtxId + (copyMapCoarseToFiner[oldVtxId + 1] -
                               copyMapCoarseToFiner[oldVtxId]);

    csrkGraph.mapCoarseToFinerRows[level][i + 1] = startVtxId;
  }

  // Now renumber the vertices in smallGraphs[level-1]
  // The renage of rows in a super-row is already known

  /*********************
   *    REORDER each super-row  THE SUPER-ROW HERE ****************
          //You can reorder the vertices in a super-row using RCM -- now assume
  you don't have to reorder *
  ***********************************************************************/

  // new to old id -- PI2
  unsigned int *newPermutation = new unsigned int[smallGraphs[level - 1].N];

  for (int i = 0; i < numSupRows; i++) {
    unsigned int oldVtxId = permCorserGraph[i];

    unsigned int oldStartNNZ = copyMapCoarseToFiner[oldVtxId];
    unsigned int oldEndNNZ = copyMapCoarseToFiner[oldVtxId + 1];

    unsigned int newNNZ = csrkGraph.mapCoarseToFinerRows[level][i];

    for (unsigned int j = oldStartNNZ; j < oldEndNNZ; j++) {
      newPermutation[newNNZ] = j;
      newNNZ++;
    }
  }

  // Copy the old permutation PI1
  unsigned int *copyOldPermutation = new unsigned int[smallGraphs[level - 1].N];

  for (int i = 0; i < smallGraphs[level - 1].N; i++)
    copyOldPermutation[i] = permFinerGraph[i];

  // Need to change the permutation: composition of permutations--PI1.PI2
  // graphPermutation[level-1].NewPermutation (Function composition)
  // permFinerGraph -- gives new to old permutation of vertices in
  // smallGraphs[level-1]
  for (int i = 0; i < smallGraphs[level - 1].N; i++) {
    permFinerGraph[i] = copyOldPermutation[newPermutation[i]];
  }

  delete[] newPermutation;
  delete[] copyOldPermutation;
  delete[] copyMapCoarseToFiner;

#ifdef DEBUG
  cout << "BAND_k::uncoarsenTheGraph completed " << endl;
#endif
}

void BAND_k::coarsenTheGraph(int level, int superNodeSize, int super_node_nnz,
                             CSRk_Graph &csrkGraph) {
#ifdef DEBUG
  cout << "BAND_k::coarsenTheGraph invoked" << endl;
#endif

  if (csrkGraph.coarsenType == HAND) {
    handCoarsen(level, super_node_nnz, csrkGraph);
  } else {
    coarsenUsingMatching(level, superNodeSize, csrkGraph);
  }

#ifdef DEBUG
  cout << "BAND_k::coarsenTheGraph completed" << endl;
#endif
}

void BAND_k::handCoarsen(int level, int super_node_nnz, CSRk_Graph &csrkGraph) {
#ifdef DEBUG
  cout << "BAND_k::handCoarsen invoked" << endl;
#endif

  unsigned int num_coarse_vtxs = 0;
  unsigned int temp_nnz_count = 0;

  long N = smallGraphs[level - 1].N;
  unsigned int *r_vec = smallGraphs[level - 1].r_vec;
  unsigned int *c_vec = smallGraphs[level - 1].c_vec;

  for (unsigned int i = 0; i < N; i++) {
    if (temp_nnz_count < (unsigned int)super_node_nnz) {
      temp_nnz_count += (r_vec[i + 1] - r_vec[i]);
    } else {
      num_coarse_vtxs++;
      temp_nnz_count = (r_vec[i + 1] - r_vec[i]);
    }
  }

  if (temp_nnz_count > 0)
    num_coarse_vtxs++;

  unsigned int *r_start_coarsened = new unsigned int[num_coarse_vtxs + 1];
  unsigned int *orgVtx_sup_map = new unsigned int[N];

  r_start_coarsened[0] = 0;

  temp_nnz_count = 0;
  num_coarse_vtxs = 0;

  for (int i = 0; i < N; i++) {
    if (temp_nnz_count < (unsigned int)super_node_nnz) {
      temp_nnz_count += (r_vec[i + 1] - r_vec[i]);
    } else {
      num_coarse_vtxs++;
      temp_nnz_count = (r_vec[i + 1] - r_vec[i]);
      r_start_coarsened[num_coarse_vtxs] = i;
    }
    orgVtx_sup_map[i] = num_coarse_vtxs;
  }

  if (temp_nnz_count > 0) {
    num_coarse_vtxs++;
    r_start_coarsened[num_coarse_vtxs] = N;
  }

  // Now make the coarsened graph
  unsigned int *adj_count = new unsigned int[num_coarse_vtxs + 1];
  for (unsigned int i_c_vtx = 0; i_c_vtx < num_coarse_vtxs + 1; i_c_vtx++)
    adj_count[i_c_vtx] = 0;

  for (unsigned int i_c_vtx = 0; i_c_vtx < num_coarse_vtxs; i_c_vtx++) {
    for (unsigned int i_orgId = r_start_coarsened[i_c_vtx];
         i_orgId < r_start_coarsened[i_c_vtx + 1]; i_orgId++) {
      unsigned int firstNeighbor = r_vec[i_orgId];
      unsigned int lastNeighbor = r_vec[i_orgId + 1];

      for (unsigned int j_colId = firstNeighbor; j_colId < lastNeighbor;
           j_colId++) {
        if (c_vec[j_colId] >= r_start_coarsened[i_c_vtx]) {
          if (c_vec[j_colId] >= N)
            cout << "Wrong Worng " << endl;

          unsigned int index = orgVtx_sup_map[c_vec[j_colId]];
          if (index == i_c_vtx) {
            adj_count[i_c_vtx]++;
          } else {
            adj_count[i_c_vtx]++;
            adj_count[index]++;
          }
        }
      }
    }
  }

  unsigned int *coarse_r_vec = new unsigned int[num_coarse_vtxs + 1];
  unsigned int cumulative_index = 0;

  coarse_r_vec[0] = 0;

  for (unsigned int i_c_vtx = 0; i_c_vtx < num_coarse_vtxs; i_c_vtx++) {
    cumulative_index += adj_count[i_c_vtx];
    coarse_r_vec[i_c_vtx + 1] = cumulative_index;
    adj_count[i_c_vtx] = coarse_r_vec[i_c_vtx];
  }
  adj_count[num_coarse_vtxs] = coarse_r_vec[num_coarse_vtxs];

  unsigned int *coarse_c_vec = new unsigned int[cumulative_index];

  for (unsigned int i_c_vtx = 0; i_c_vtx < num_coarse_vtxs; i_c_vtx++) {
    for (unsigned int i_orgId = r_start_coarsened[i_c_vtx];
         i_orgId < r_start_coarsened[i_c_vtx + 1]; i_orgId++) {
      unsigned int firstNeighbor = r_vec[i_orgId];
      unsigned int lastNeighbor = r_vec[i_orgId + 1];

      for (unsigned int j_colId = firstNeighbor; j_colId < lastNeighbor;
           j_colId++) {
        if (c_vec[j_colId] >= r_start_coarsened[i_c_vtx]) {
          unsigned int index = orgVtx_sup_map[c_vec[j_colId]];
          if (index == i_c_vtx) {
            coarse_c_vec[adj_count[i_c_vtx]] = i_c_vtx;
            adj_count[i_c_vtx]++;
          } else {
            coarse_c_vec[adj_count[i_c_vtx]] = index;
            adj_count[i_c_vtx]++;
            coarse_c_vec[adj_count[index]] = i_c_vtx;
            adj_count[index]++;
          }
        }
      }
    }
  }

  std::vector<unsigned int> adj_vector(coarse_c_vec,
                                       coarse_c_vec + cumulative_index);

// The neighbors are already in sorted order -- WE DON"T NEED THIS --TEST THIS
#pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < num_coarse_vtxs; i++)
    std::sort(adj_vector.begin() + coarse_r_vec[i],
              adj_vector.begin() + coarse_r_vec[i + 1]);

  // Delete the duplicate vertices
  int prev_neighbor = 0;
  int curr_neighbor = 0, adj_index = 0;
  int degree = 0;

  unsigned int *distinct_neighbor = new unsigned int[num_coarse_vtxs + 1];
  unsigned int *distinct_adj = new unsigned int[cumulative_index];
  unsigned int *adj_degree = new unsigned int[cumulative_index];

  distinct_neighbor[0] = 0;

  for (unsigned int i = 0; i < num_coarse_vtxs; i++) {
    prev_neighbor = adj_vector[coarse_r_vec[i]];
    degree = 1;
    distinct_adj[adj_index] = prev_neighbor;
    adj_index++;

    for (unsigned int j = coarse_r_vec[i] + 1; j < coarse_r_vec[i + 1]; j++) {
      curr_neighbor = adj_vector[j];
      if (prev_neighbor == curr_neighbor) {
        degree++;
      } else {
        adj_degree[adj_index - 1] = degree;
        prev_neighbor = curr_neighbor;
        distinct_adj[adj_index] = prev_neighbor;
        adj_index++;

        degree = 1;
      }
    }
    adj_degree[adj_index - 1] = degree;
    distinct_neighbor[i + 1] = adj_index;
  }

  smallGraphs[level].N = num_coarse_vtxs;
  smallGraphs[level].NNZ = distinct_neighbor[num_coarse_vtxs];

  smallGraphs[level].r_vec = new unsigned int[num_coarse_vtxs + 1];
  smallGraphs[level].c_vec = new unsigned int[smallGraphs[level].NNZ];
  smallGraphs[level].degree = new unsigned int[smallGraphs[level].NNZ];

  for (unsigned int i = 0; i < num_coarse_vtxs; i++) {
    smallGraphs[level].r_vec[i] = distinct_neighbor[i];
    for (unsigned int j = distinct_neighbor[i]; j < distinct_neighbor[i + 1];
         j++) {
      smallGraphs[level].c_vec[j] = distinct_adj[j];
      smallGraphs[level].degree[j] = adj_degree[j];
    }
  }
  smallGraphs[level].r_vec[num_coarse_vtxs] =
      distinct_neighbor[num_coarse_vtxs];

  csrkGraph.mapCoarseToFinerRows[level] = r_start_coarsened;

  // Free allocated memory
  delete[] distinct_adj;
  delete[] adj_degree;
  delete[] distinct_neighbor;

  delete[] coarse_r_vec;
  delete[] coarse_c_vec;
  delete[] adj_count;
  delete[] orgVtx_sup_map;

#ifdef DEBUG
  cout << "BAND_k::handCoarsen completed" << endl;
#endif
}

void BAND_k::coarsenUsingMatching(int level, int superRowSize,
                                  CSRk_Graph &csrkGraph) {
#ifdef DEBUG
  cout << "BAND_k::coarsenUsingMatching invoked" << endl;
#endif

  string MatchingType = csrkGraph.coarsenType;

  vector<unsigned int *> mappingVectors;
  vector<unsigned int> vectorSizes;

  C_GRAPH localG = smallGraphs[level - 1];
  C_GRAPH coarsendGraph;

  int finalNumVtxs = localG.N / superRowSize;

  // default value is g.n
  // unsigned int maxSupRowSize = supRowSize * alpha;

  unsigned int *perm;
  while (finalNumVtxs < localG.N) {

    perm = new unsigned int[localG.N];

    if (MatchingType == RAND) {
      randomMatching(localG, perm, coarsendGraph);
    } else if (MatchingType == HEM) {
      heavyEdgeMatching(localG, perm, coarsendGraph);
    } else if (MatchingType == LEM) {
      lightEdgeMatching(localG, perm, coarsendGraph);
    }

    // heavyEdgeMatching(localG, perm, coarsendGraph);
    // lightEdgeMatching(localG, perm, coarsendGraph);

    vectorSizes.push_back(localG.N);
    mappingVectors.push_back(perm);

    localG = coarsendGraph;
  }

  smallGraphs[level].N = localG.N;
  smallGraphs[level].NNZ = localG.NNZ;

  smallGraphs[level].r_vec = new unsigned int[smallGraphs[level].N + 1];
  ;
  smallGraphs[level].c_vec = new unsigned int[smallGraphs[level].NNZ];
  smallGraphs[level].vertexWeight = new unsigned int[smallGraphs[level].N];
  smallGraphs[level].edgeWeight = new unsigned int[smallGraphs[level].NNZ];
  smallGraphs[level].degree = new unsigned int[smallGraphs[level].NNZ];

  for (int rowIdx = 0; rowIdx < smallGraphs[level].N; rowIdx++) {
    smallGraphs[level].r_vec[rowIdx] = localG.r_vec[rowIdx];
    smallGraphs[level].vertexWeight[rowIdx] = localG.vertexWeight[rowIdx];
  }
  smallGraphs[level].r_vec[smallGraphs[level].N] =
      localG.r_vec[smallGraphs[level].N];

  for (int colIdx = 0; colIdx < smallGraphs[level].NNZ; colIdx++) {
    smallGraphs[level].c_vec[colIdx] = localG.c_vec[colIdx];
    smallGraphs[level].edgeWeight[colIdx] = localG.edgeWeight[colIdx];
    smallGraphs[level].degree[colIdx] = localG.edgeWeight[colIdx];
  }

  unsigned int *mapSupRowstoRows = new unsigned int[smallGraphs[level - 1].N];
  unsigned int *numEdgesSupRowsToRows;

  // Find the final mapping from the coarsest graph to the fine graph
  findFinalMapping(localG.N, vectorSizes, mappingVectors, numEdgesSupRowsToRows,
                   mapSupRowstoRows);
  csrkGraph.mapCoarseToFinerRows[level] = numEdgesSupRowsToRows;
  csrkGraph.permOfFinerRows[level - 1] = mapSupRowstoRows;

  for (unsigned int i = 0; i < mappingVectors.size(); i++)
    if (mappingVectors[i] != NULL)
      delete[] mappingVectors[i];

#ifdef DEBUG
  cout << "BAND_k::coarsenUsingMatching completed" << endl;
#endif
}

// Put the matrix in CSR-k for STS
void BAND_k::preprocessingForSTS(CSRk_Graph &csrkGraph) {
#ifdef DEBUG
  cout << "BAND_k::preprocessingForSTS invoked" << endl;
  cout << "Value of CSRk_Graph::k : " << csrkGraph.k << endl;
  cout << "Value of BAND_k::k : " << k << endl;
#endif

  // DO an RCM on the coarsened graph except for the last level graph
  // Color or LS order the last level graph

  // This is STS-2
  if (k == 1) {

    if (csrkGraph.orderType == COLOR) {
      int numColors = 0;
      int *colorPtr;
      size_t *ix = (size_t *)malloc(csrkGraph.N * sizeof(size_t));

      BGL_ordering(csrkGraph.NNZ, csrkGraph.N, (int *)csrkGraph.r_vec,
                   (int *)csrkGraph.c_vec, ix, &numColors, &colorPtr);

      // Put the colors in increasing order
      size_t *ix_old = (size_t *)malloc(csrkGraph.N * sizeof(size_t));
      int *colorPtr_old = (int *)malloc((numColors + 1) * sizeof(int));

      if ((ix_old == NULL) || (colorPtr_old == NULL)) {
        printf("Can't allocate memory for: ix_old, colorPtr_old\n");
        exit(1);
      }

      for (int i_lvl = 0; i_lvl < numColors + 1; i_lvl++)
        colorPtr_old[i_lvl] = colorPtr[i_lvl];

      for (int i_lvl_j = 0; i_lvl_j < colorPtr[numColors]; i_lvl_j++)
        ix_old[i_lvl_j] = ix[i_lvl_j];

      int neigbors_count = 0;
      colorPtr[0] = 0;

      int old_index = numColors - 1;
      for (int i_lvl = 0; i_lvl < numColors; i_lvl++) {
        int idx_a = colorPtr_old[old_index];
        int idx_b = colorPtr_old[old_index + 1];

        neigbors_count = neigbors_count + (idx_b - idx_a);

        colorPtr[i_lvl + 1] = neigbors_count;

        old_index--;
      }
      assert(colorPtr[numColors] == colorPtr_old[numColors]);

      old_index = numColors - 1;
      for (int i_lvl = 0; i_lvl < numColors; i_lvl++) {
        int idx_a = colorPtr_old[old_index];
        int idx_b = colorPtr_old[old_index + 1];

        int idx_x = colorPtr[i_lvl];

        for (int j = idx_a; j < idx_b; j++) {
          ix[idx_x] = ix_old[j];
          idx_x++;
        }
        old_index--;
      }

      free(ix_old);
      free(colorPtr_old);

      // Now the colors are in increasing order of their size

      for (int orgVertexId = 0; orgVertexId < csrkGraph.N; orgVertexId++)
        csrkGraph.permBigG[orgVertexId] = ix[orgVertexId];

      csrkGraph.numPacks = numColors;
      csrkGraph.packsPointer = new int[numColors + 1];

      for (int i = 0; i < numColors + 1; i++)
        csrkGraph.packsPointer[i] = colorPtr[i];

      if (colorPtr != NULL)
        free(colorPtr);

      if (ix != NULL)
        free(ix);

    } else if (csrkGraph.orderType == LS) {
      int numLvls = 0;
      int *levelPtr;
      int *adjLevel = new int[csrkGraph.N];

      find_levels(csrkGraph.NNZ, csrkGraph.N, (int *)csrkGraph.r_vec,
                  (int *)csrkGraph.c_vec, adjLevel, numLvls, levelPtr);

      // Put the levels in increasing order of their size
      std::vector<degree_id_pair> levelVecSize(numLvls);

      for (int i = 0; i < numLvls; i++) {
        levelVecSize[i] = degree_id_pair(levelPtr[i + 1] - levelPtr[i], i);
      }

      int *adjLevelOld = new int[csrkGraph.N];
      int *levelPtrOld = new int[numLvls + 1];

      assert(adjLevelOld != NULL);
      assert(levelPtrOld != NULL);

      for (int i = 0; i < numLvls + 1; i++)
        levelPtrOld[i] = levelPtr[i];

      for (int i = 0; i < csrkGraph.N; i++)
        adjLevelOld[i] = adjLevel[i];

      // This sorts the levels in increasing order of size
      std::sort(levelVecSize.begin(), levelVecSize.begin() + numLvls,
                compare_deg_id_pair);

      levelPtr[0] = 0;
      int cumulativeIndex = 0;
      for (int i = 0; i < numLvls; i++) {
        int oldLevel = levelVecSize[i].id;

        int startEdge = levelPtrOld[oldLevel];
        int endEdge = levelPtrOld[oldLevel + 1];

        for (int j = startEdge; j < endEdge; j++) {
          adjLevel[cumulativeIndex] = adjLevelOld[j];
          cumulativeIndex++;
        }
        levelPtr[i + 1] = cumulativeIndex;
      }

      delete[] adjLevelOld;
      delete[] levelPtrOld;

      // Now the levels are in increasing order of their size
      for (int orgVertexId = 0; orgVertexId < csrkGraph.N; orgVertexId++)
        csrkGraph.permBigG[orgVertexId] = adjLevel[orgVertexId];

      csrkGraph.numPacks = numLvls;
      csrkGraph.packsPointer = new int[numLvls + 1];

      for (int idxLvl = 0; idxLvl < numLvls + 1; idxLvl++)
        csrkGraph.packsPointer[idxLvl] = levelPtr[idxLvl];

      if (levelPtr != NULL) {
        delete[] levelPtr;
      }
      if (adjLevel != NULL) {
        delete[] adjLevel;
      }
    }

  } else { // this is for STS-k  for k >= 3

    // if k = 3, this->k is 2
    // Size of smallGraphs is k
    // This is initialized in BAND_k

    if (csrkGraph.coarsenType == HAND) {
      stsPreprocessingForHAND(csrkGraph);
    } else {
      stsPreprocessingWithMatching(csrkGraph);
    }
  }

#ifdef DEBUG
  cout << "BAND_k::preprocessingForSTS completed" << endl;
#endif
}

// Put the matrix in CSR-k for STS Using HAND coarsening
void BAND_k::stsPreprocessingForHAND(CSRk_Graph &csrkGraph) {
#ifdef DEBUG
  cout << "BAND_k::stsPreprocessingForHAND invoked" << endl;
  cout << "Value of CSRk_Graph::k : " << csrkGraph.k << endl;
  cout << "Value of BAND_k::k : " << k << endl;
#endif

  // New to old vertex mapping
  unsigned int **graphPermutations = new unsigned int *[k];

  // Old to new vertex mapping
  unsigned int **reverseGraphPermutations = new unsigned int *[k];

  for (int i = 0; i < k; i++) {
    graphPermutations[i] = NULL;
    reverseGraphPermutations[i] = NULL;
  }

  smallGraphs[0].N = csrkGraph.N;
  smallGraphs[0].NNZ = csrkGraph.NNZ;

  smallGraphs[0].r_vec = new unsigned int[smallGraphs[0].N + 1];
  smallGraphs[0].c_vec = new unsigned int[smallGraphs[0].NNZ];
  unsigned int *inDegree = new unsigned int[smallGraphs[0].NNZ];
  unsigned int *inVertexWeight = new unsigned int[smallGraphs[0].N];
  unsigned int *inEdgeWeight = new unsigned int[smallGraphs[0].NNZ];

  for (int i = 0; i < smallGraphs[0].N; i++) {
    smallGraphs[0].r_vec[i] = csrkGraph.r_vec[i];
    inVertexWeight[i] = 1;
  }
  smallGraphs[0].r_vec[smallGraphs[0].N] = csrkGraph.r_vec[smallGraphs[0].N];

  for (int i = 0; i < smallGraphs[0].NNZ; i++) {
    inDegree[i] = 1;
    smallGraphs[0].c_vec[i] = csrkGraph.c_vec[i];
    inEdgeWeight[i] = 1;
  }
  smallGraphs[0].degree = inDegree;
  smallGraphs[0].vertexWeight = inVertexWeight;
  smallGraphs[0].edgeWeight = inEdgeWeight;

  graphPermutations[0] = new unsigned int[csrkGraph.N];

  for (int i = 0; i < csrkGraph.N; i++) {
    graphPermutations[0][i] = i;
  }

  // Coarsen k -1 time this is actually CSR-(k-2)

  // Corsen k-1 times and do RCM everytime for hand-coarsening except the last
  // graph For (k-1)-th time do a color/LS ordering

  for (int i = 1; i < k; i++) {
    int supNodeSize = csrkGraph.supRowSizes[i - 1] * smallGraphs[i - 1].NNZ /
                      smallGraphs[i - 1].N;

    // HAND coarsen the graph such that each super row has supNodeSize non-zeros
    // Coarsen smallGraphs[i-1] to get smallGraphs[i]
    // And set csrkGraph.mapCoarseToFinerRows[i] -- maps super rows from
    // smallGraphs[i] to rows in smallGraphs[i-1]
    coarsenTheGraph(i, (int)csrkGraph.supRowSizes[i - 1], supNodeSize,
                    csrkGraph);

    // RCM on i-th graph and change the mapping from i-th graph to (i-1)-graph
    int numVertices = smallGraphs[i].N;

    // new to old mapping -- given new vertex number this will return old vertex
    // number
    unsigned int *corsenedGraphPerm = new unsigned int[numVertices];

    // Old to new mapping
    unsigned int *ReverseGraphPerm = new unsigned int[numVertices];

    if (i == k - 1) {

      if (csrkGraph.orderType == COLOR) {

        // Color the graph and put the colors in increasing order of size
        int numColors = 0;
        int *colorPtr;
        size_t *ix = (size_t *)malloc(smallGraphs[i].N * sizeof(size_t));

        BGL_ordering(smallGraphs[i].NNZ, smallGraphs[i].N,
                     (int *)smallGraphs[i].r_vec, (int *)smallGraphs[i].c_vec,
                     ix, &numColors, &colorPtr);

        // Put the colors in increasing order
        size_t *ix_old = (size_t *)malloc(smallGraphs[i].N * sizeof(size_t));
        int *colorPtr_old = (int *)malloc((numColors + 1) * sizeof(int));

        if ((ix_old == NULL) || (colorPtr_old == NULL)) {
          printf("Can't allocate memory for: ix_old, colorPtr_old\n");
          exit(1);
        }

        for (int i_lvl = 0; i_lvl < numColors + 1; i_lvl++)
          colorPtr_old[i_lvl] = colorPtr[i_lvl];

        for (int i_lvl_j = 0; i_lvl_j < colorPtr[numColors]; i_lvl_j++)
          ix_old[i_lvl_j] = ix[i_lvl_j];

        int neigbors_count = 0;
        colorPtr[0] = 0;

        int old_index = numColors - 1;
        for (int i_lvl = 0; i_lvl < numColors; i_lvl++) {
          int idx_a = colorPtr_old[old_index];
          int idx_b = colorPtr_old[old_index + 1];

          neigbors_count = neigbors_count + (idx_b - idx_a);

          colorPtr[i_lvl + 1] = neigbors_count;

          old_index--;
        }
        assert(colorPtr[numColors] == colorPtr_old[numColors]);

        old_index = numColors - 1;
        for (int i_lvl = 0; i_lvl < numColors; i_lvl++) {
          int idx_a = colorPtr_old[old_index];
          int idx_b = colorPtr_old[old_index + 1];

          int idx_x = colorPtr[i_lvl];

          for (int j = idx_a; j < idx_b; j++) {
            ix[idx_x] = ix_old[j];
            idx_x++;
          }
          old_index--;
        }

        free(ix_old);
        free(colorPtr_old);

        for (int vtxIdx = 0; vtxIdx < numVertices; vtxIdx++) {
          // New to old index
          corsenedGraphPerm[vtxIdx] = ix[vtxIdx];

          // Old to new index
          ReverseGraphPerm[ix[vtxIdx]] = vtxIdx;
        }

        csrkGraph.numPacks = numColors;
        csrkGraph.packsPointer = new int[numColors + 1];

        for (int packIdx = 0; packIdx < numColors + 1; packIdx++)
          csrkGraph.packsPointer[packIdx] = colorPtr[packIdx];

        if (colorPtr != NULL)
          free(colorPtr);

        if (ix != NULL)
          free(ix);
      } else if (csrkGraph.orderType == LS) {

        int numLvls = 0;
        int *levelPtr;
        int *adjLevel = new int[smallGraphs[i].N];

        // Find level from maximum degree vertex

        find_levels_from_maxDegree_vertex(
            smallGraphs[i].NNZ, smallGraphs[i].N, (int *)smallGraphs[i].r_vec,
            (int *)smallGraphs[i].c_vec, adjLevel, numLvls, levelPtr);

        // Put the levels in increasing order of their size
        std::vector<degree_id_pair> levelVecSize(numLvls);

        for (int indexLvl = 0; indexLvl < numLvls; indexLvl++) {
          levelVecSize[indexLvl] = degree_id_pair(
              levelPtr[indexLvl + 1] - levelPtr[indexLvl], indexLvl);
        }

        int *adjLevelOld = new int[smallGraphs[i].N];
        int *levelPtrOld = new int[numLvls + 1];

        assert(adjLevelOld != NULL);
        assert(levelPtrOld != NULL);

        for (int indexLvl = 0; indexLvl < numLvls + 1; indexLvl++)
          levelPtrOld[indexLvl] = levelPtr[indexLvl];

        for (int indexLvl = 0; indexLvl < smallGraphs[i].N; indexLvl++)
          adjLevelOld[indexLvl] = adjLevel[indexLvl];

        // This sorts the levels in increasing order of size
        std::sort(levelVecSize.begin(), levelVecSize.begin() + numLvls,
                  compare_deg_id_pair);

        levelPtr[0] = 0;
        int cumulativeIndex = 0;
        for (int indexLvl = 0; indexLvl < numLvls; indexLvl++) {
          int oldLevel = levelVecSize[indexLvl].id;

          int startEdge = levelPtrOld[oldLevel];
          int endEdge = levelPtrOld[oldLevel + 1];

          for (int j = startEdge; j < endEdge; j++) {
            adjLevel[cumulativeIndex] = adjLevelOld[j];
            cumulativeIndex++;
          }
          levelPtr[indexLvl + 1] = cumulativeIndex;
        }

        delete[] adjLevelOld;
        delete[] levelPtrOld;

        for (int vtxIdx = 0; vtxIdx < numVertices; vtxIdx++) {
          // New to old index
          corsenedGraphPerm[vtxIdx] = adjLevel[vtxIdx];

          // Old to new index
          ReverseGraphPerm[adjLevel[vtxIdx]] = vtxIdx;
        }
        csrkGraph.numPacks = numLvls;
        csrkGraph.packsPointer = new int[numLvls + 1];

        for (int indexLvl = 0; indexLvl < numLvls + 1; indexLvl++)
          csrkGraph.packsPointer[indexLvl] = levelPtr[indexLvl];

        if (levelPtr != NULL) {
          delete[] levelPtr;
        }
        if (adjLevel != NULL) {
          delete[] adjLevel;
        }

      } // End of if Ordering type LS

    } else {
      unsigned int *mask = new unsigned int[numVertices];

      for (int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++)
        mask[vertexIndex] = 1;

      int firstVtxinBFS, lastVtxinBFS, ccSize = 0, root = 0;

      // To do RCM on the whole graph call RCM on each connected component
      for (int i_mask = 0; i_mask < numVertices; i_mask++) {
        if (ccSize >= numVertices)
          break;

        if (mask[i_mask] != 0) {
          root = i_mask;

          rcm_reordering_g(numVertices, (int *)smallGraphs[i].r_vec,
                           (int *)smallGraphs[i].c_vec,
                           (int *)smallGraphs[i].degree, root, (int *)mask,
                           (int *)ReverseGraphPerm, (int *)corsenedGraphPerm,
                           firstVtxinBFS, lastVtxinBFS, ccSize);
        }
      }

#ifdef DEBUG
      cout << "Done with RCM on: " << i << "-th Coarsened graph " << endl;
#endif

      // Renumber the graph using the reordered vartices
      // No need to renumber the (k-1)-th graph --that is the last graph
      // This renumbers the graph smallGraphs[i] according to the new ordering
      // that is found by RCM
      renumberGraphUsingReorderedVertices(
          numVertices, smallGraphs[i].NNZ, &smallGraphs[i].r_vec,
          &smallGraphs[i].c_vec, &smallGraphs[i].degree, ReverseGraphPerm,
          corsenedGraphPerm);

      delete[] mask;
    }

    graphPermutations[i] = corsenedGraphPerm;
    reverseGraphPermutations[i] = ReverseGraphPerm;
  }

  // csrkGraph.mapCoarseToFinerRows -- this contains mapping from super-rows to
  // rows --BEFORE reordering using RCM of smallGraphs[i]
  csrkGraph.numCoarsestRows = smallGraphs[k - 1].N;
  csrkGraph.numCoarsishRows = smallGraphs[k - 2].N;

  // Now uncoarsen the coarsened graphs and find permutation of original graph
  // Also find k-1 mapping from coarsened graphs to finer graphs
  for (int i = k - 1; i >= 1; i--) {

    /*
    uncoarsenTheGraph  -- does the following
            (i) Renumber the vertices of finer graph smallGraphs[i-1] according
to the new RCM ordering (ii) It changes the csrkGraph.mapCoarseToFinerRows[i]
superRowsToRows mapping using the new numbering (iii) Also changes the
permutation graphPermutations[i-1] -- so it reflects the new to old mapping
after renumbering the vertices in smallGraphs[i-1]
* 		*/

    // Use RCM maps to renumber the mapping when computing CSR-k
    // Change mapping from i-th graph to (i-1)-th graph using reordered
    // super-rows graphPermutations reverseGraphPermutations

    // Uncoarsen the coarsened graphs and find the reordering of the original
    // graph
    uncoarsenTheGraph(i, csrkGraph, graphPermutations[i],
                      graphPermutations[i - 1]);
    // RCM on i-th graph and change the mapping from i-th graph to (i-1)-graph
  }

  // Use the permutation of the original matrix to reorder the matrix
  // This sets the permutation of original matrix
  // Copy the permutation so you can free memory

  // Use the permutation of the original matrix to reorder the matrix
  // This sets the permutation of original matrix
  // Copy the permutation so you can free memory

  for (int orgVertexId = 0; orgVertexId < smallGraphs[0].N; orgVertexId++)
    csrkGraph.permBigG[orgVertexId] = graphPermutations[0][orgVertexId];

  for (int i = 0; i < k; i++) {
    if (graphPermutations[i] != NULL)
      delete[] graphPermutations[i];

    if (reverseGraphPermutations[i] != NULL)
      delete[] reverseGraphPermutations[i];
  }

  delete[] graphPermutations;
  delete[] reverseGraphPermutations;

#ifdef DEBUG
  cout << "BAND_k::stsPreprocessingForHAND completed" << endl;
#endif
}

// Put the matrix in CSR-k for STS using HEM/LEM/RAND coarsening
void BAND_k::stsPreprocessingWithMatching(CSRk_Graph &csrkGraph) {
#ifdef DEBUG
  cout << "BAND_k::stsPreprocessingWithMatching invoked" << endl;
  cout << "Value of CSRk_GRAPH::k " << csrkGraph.k << endl;
  cout << "Value of BAND_k::k  " << k << endl;
#endif

  // Color or LS order the last level graph

  // if k = 3, this->k is 2
  // Size of smallGraphs is k
  // This is initialized in BAND_k

  smallGraphs[0].N = csrkGraph.N;
  smallGraphs[0].NNZ = csrkGraph.NNZ;

  smallGraphs[0].r_vec = new unsigned int[smallGraphs[0].N + 1];
  smallGraphs[0].c_vec = new unsigned int[smallGraphs[0].NNZ];
  unsigned int *inDegree = new unsigned int[smallGraphs[0].NNZ];
  unsigned int *inVertexWeight = new unsigned int[smallGraphs[0].N];
  unsigned int *inEdgeWeight = new unsigned int[smallGraphs[0].NNZ];

  for (int i = 0; i < smallGraphs[0].N; i++) {
    smallGraphs[0].r_vec[i] = csrkGraph.r_vec[i];
    inVertexWeight[i] = 1;
  }
  smallGraphs[0].r_vec[smallGraphs[0].N] = csrkGraph.r_vec[smallGraphs[0].N];

  for (int i = 0; i < smallGraphs[0].NNZ; i++) {
    inDegree[i] = 1;
    smallGraphs[0].c_vec[i] = csrkGraph.c_vec[i];
    inEdgeWeight[i] = 1;
  }
  smallGraphs[0].degree = inDegree;
  smallGraphs[0].vertexWeight = inVertexWeight;
  smallGraphs[0].edgeWeight = inEdgeWeight;

  /*graphPermutations[0] = new unsigned int[csrkGraph.N];

  for(int i = 0; i < csrkGraph.N; i++)
  {
          graphPermutations[0][i] = i;
  } */

  // Coarsen k -1 time this is actually CSR-(k-2)

  // Color or level-set of original graph and reorder according to color/LS
  // Corsen k-1 times and do RCM everytime for hand-coarsening except the last
  // graph For (k-1)-th time do a color/LS ordering

  for (int i = 1; i < k; i++) {
    int supNodeSize = csrkGraph.supRowSizes[i - 1] * smallGraphs[i - 1].NNZ /
                      smallGraphs[i - 1].N;

    // This is hand coarsening such that each super row has supNodeSize
    // non-zeros Coarsen the (i-1)-th graph to get i-th graph Coarsen
    // smallGraphs[i-1] to get smallGraphs[i] And set
    // csrkGraph.mapCoarseToFinerRows[i] This maps super rows from smallGraphs[i]
    // to rows in smallGraphs[i-1]
    coarsenTheGraph(i, (int)csrkGraph.supRowSizes[i - 1], supNodeSize,
                    csrkGraph);

    if (i == k - 1) {

      int numVertices = smallGraphs[i].N;

      // new to old mapping -- given new vertex number this will return old
      // vertex number
      unsigned int *corsenedGraphPerm = new unsigned int[numVertices];

      // Old to new mapping
      unsigned int *ReverseGraphPerm = new unsigned int[numVertices];

      if (csrkGraph.orderType == COLOR) {

        // Color the graph and put the colors in increasing order of size
        int numColors = 0;
        int *colorPtr;
        size_t *ix = (size_t *)malloc(smallGraphs[i].N * sizeof(size_t));

        BGL_ordering(smallGraphs[i].NNZ, smallGraphs[i].N,
                     (int *)smallGraphs[i].r_vec, (int *)smallGraphs[i].c_vec,
                     ix, &numColors, &colorPtr);

        // Put the colors in increasing order
        size_t *ix_old = (size_t *)malloc(smallGraphs[i].N * sizeof(size_t));
        int *colorPtr_old = (int *)malloc((numColors + 1) * sizeof(int));

        if ((ix_old == NULL) || (colorPtr_old == NULL)) {
          printf("Can't allocate memory for: ix_old, colorPtr_old\n");
          exit(1);
        }

        for (int i_lvl = 0; i_lvl < numColors + 1; i_lvl++)
          colorPtr_old[i_lvl] = colorPtr[i_lvl];

        for (int i_lvl_j = 0; i_lvl_j < colorPtr[numColors]; i_lvl_j++)
          ix_old[i_lvl_j] = ix[i_lvl_j];

        int neigbors_count = 0;
        colorPtr[0] = 0;

        int old_index = numColors - 1;
        for (int i_lvl = 0; i_lvl < numColors; i_lvl++) {
          int idx_a = colorPtr_old[old_index];
          int idx_b = colorPtr_old[old_index + 1];

          neigbors_count = neigbors_count + (idx_b - idx_a);

          colorPtr[i_lvl + 1] = neigbors_count;

          old_index--;
        }
        assert(colorPtr[numColors] == colorPtr_old[numColors]);

        old_index = numColors - 1;
        for (int i_lvl = 0; i_lvl < numColors; i_lvl++) {
          int idx_a = colorPtr_old[old_index];
          int idx_b = colorPtr_old[old_index + 1];

          int idx_x = colorPtr[i_lvl];

          for (int j = idx_a; j < idx_b; j++) {
            ix[idx_x] = ix_old[j];
            idx_x++;
          }
          old_index--;
        }

        free(ix_old);
        free(colorPtr_old);

        for (int vtxIdx = 0; vtxIdx < numVertices; vtxIdx++) {
          // New to old index
          corsenedGraphPerm[vtxIdx] = ix[vtxIdx];

          // Old to new index
          ReverseGraphPerm[ix[vtxIdx]] = vtxIdx;
        }

        csrkGraph.numPacks = numColors;
        csrkGraph.packsPointer = new int[numColors + 1];

        for (int packIdx = 0; packIdx < numColors + 1; packIdx++)
          csrkGraph.packsPointer[packIdx] = colorPtr[packIdx];

        if (colorPtr != NULL)
          free(colorPtr);

        if (ix != NULL)
          free(ix);
      } else if (csrkGraph.orderType == LS) {

        int numLvls = 0;
        int *levelPtr;
        int *adjLevel = new int[smallGraphs[i].N];

        // Find level from maximum degree vertex
        find_levels_from_maxDegree_vertex(
            smallGraphs[i].NNZ, smallGraphs[i].N, (int *)smallGraphs[i].r_vec,
            (int *)smallGraphs[i].c_vec, adjLevel, numLvls, levelPtr);

        // Put the levels in increasing order of their size
        std::vector<degree_id_pair> levelVecSize(numLvls);

        for (int indexLvl = 0; indexLvl < numLvls; indexLvl++) {
          levelVecSize[indexLvl] = degree_id_pair(
              levelPtr[indexLvl + 1] - levelPtr[indexLvl], indexLvl);
        }

        int *adjLevelOld = new int[smallGraphs[i].N];
        int *levelPtrOld = new int[numLvls + 1];

        assert(adjLevelOld != NULL);
        assert(levelPtrOld != NULL);

        for (int indexLvl = 0; indexLvl < numLvls + 1; indexLvl++)
          levelPtrOld[indexLvl] = levelPtr[indexLvl];

        for (int indexLvl = 0; indexLvl < smallGraphs[i].N; indexLvl++)
          adjLevelOld[indexLvl] = adjLevel[indexLvl];

        // This sorts the levels in increasing order of size
        std::sort(levelVecSize.begin(), levelVecSize.begin() + numLvls,
                  compare_deg_id_pair);

        levelPtr[0] = 0;
        int cumulativeIndex = 0;
        for (int indexLvl = 0; indexLvl < numLvls; indexLvl++) {
          int oldLevel = levelVecSize[indexLvl].id;

          int startEdge = levelPtrOld[oldLevel];
          int endEdge = levelPtrOld[oldLevel + 1];

          for (int j = startEdge; j < endEdge; j++) {
            adjLevel[cumulativeIndex] = adjLevelOld[j];
            cumulativeIndex++;
          }
          levelPtr[indexLvl + 1] = cumulativeIndex;
        }

        delete[] adjLevelOld;
        delete[] levelPtrOld;

        for (int vtxIdx = 0; vtxIdx < numVertices; vtxIdx++) {
          // New to old index
          corsenedGraphPerm[vtxIdx] = adjLevel[vtxIdx];

          // Old to new index
          ReverseGraphPerm[adjLevel[vtxIdx]] = vtxIdx;
        }
        csrkGraph.numPacks = numLvls;
        csrkGraph.packsPointer = new int[numLvls + 1];

        for (int indexLvl = 0; indexLvl < numLvls + 1; indexLvl++)
          csrkGraph.packsPointer[indexLvl] = levelPtr[indexLvl];

        if (levelPtr != NULL) {
          delete[] levelPtr;
        }
        if (adjLevel != NULL) {
          delete[] adjLevel;
        }

      } // End of if Ordering type LS

      // New to old Id
      csrkGraph.permOfFinerRows[i] = corsenedGraphPerm;
    }

    // graphPermutations[i] = corsenedGraphPerm;
    // reverseGraphPermutations[i] = ReverseGraphPerm;
  }

  // csrkGraph.mapCoarseToFinerRows -- this contains mapping from super-rows to
  // rows --BEFORE reordering using RCM of smallGraphs[i]
  csrkGraph.numCoarsestRows = smallGraphs[k - 1].N;
  csrkGraph.numCoarsishRows = smallGraphs[k - 2].N;

  // Now uncoarsen the coarsened graphs and find permutation of original graph
  // Also find k-1 mapping from coarsened graphs to finer graphs
  for (int i = k - 1; i >= 1; i--) {

    // Renumber the vertices in graph smallGraphs[i-1] using the reordering of
    // smallGraphs[i] vertices Uncoarsen the coarsened graphs and find the
    // reordering of the original graph i-th reordering is
    // csrkGraph.permOfFinerRows[i] and (i-1)-th reordering is
    // csrkGraph.permOfFinerRows[i]
    // Mapping from i-th to (i-1)-th graph is determined by
    // csrkGraph.mapCoarseToFinerRows[i] and csrkGraph.permOfFinerRows[i-1]

    matchingUncoarsenTheGraph(i, csrkGraph);
  }

  for (int orgVertexId = 0; orgVertexId < smallGraphs[0].N; orgVertexId++)
    csrkGraph.permBigG[orgVertexId] = csrkGraph.permOfFinerRows[0][orgVertexId];

#ifdef DEBUG
  cout << "BAND_k::stsPreprocessingWithMatching completed" << endl;
#endif
}

/******************************************************************************
 ***********                  RCM Related Functions            ****************
 * ****************************************************************************/

int rcm_reordering_g(int num_verts, int *num_edges, int *adj_list,
                     int *adj_degree, int root, int mask[],
                     int oldToNewOrdering[], int newToOldOrdering[],
                     int &firstVtxinBFS, int &lastVtxinBFS, int &ccSize) {
#ifdef DEBUG
  cout << "entering rcm_reordering_g" << endl;
#endif
  int num_lvls = 0;
  int *r_vec_lvlStruc = new int[num_verts];
  int *c_vec_lvlStruc = new int[num_verts];

  findPseudoPeripheralVertex(root, num_edges, adj_list, mask, num_lvls,
                             r_vec_lvlStruc, c_vec_lvlStruc);

  firstVtxinBFS = root;

  // int copy_mask[num_verts];

  // for(int i=0; i< num_verts; i++)
  // {
  //    copy_mask[i] = mask[i];
  // }

  int Q_size = std::max(num_edges[num_verts], num_verts);
  // std::cout << Q_size << std::endl;
  Queue_Arry vertex_Q(Q_size);

  vertex_Q.enQueue(root);

  int num_visited_verts = 0;
  int new_start_vertex = ccSize;

  while (!vertex_Q.isEmpty()) {
    int parent_vertex = vertex_Q.deQueue();
    if (mask[parent_vertex] == 1) {
      mask[parent_vertex] = 0;

      newToOldOrdering[new_start_vertex] = parent_vertex;
      num_visited_verts++;
      new_start_vertex++;

      int start_edge = num_edges[parent_vertex];
      int end_edge = num_edges[parent_vertex + 1];

      std::vector<rev_deg_id_pair> rev_deg_id_vec(end_edge - start_edge);
      int count = 0;

      for (int i = start_edge; i < end_edge; i++) {
        int adj_vertex_id = adj_list[i];
        if (mask[adj_vertex_id] == 1) {
          rev_deg_id_vec[count].deg = adj_degree[i];
          rev_deg_id_vec[count].id = adj_vertex_id;
          count++;
        }
      }

      std::sort(rev_deg_id_vec.begin(), rev_deg_id_vec.begin() + count,
                compare_rev_deg_id_pair);
      for (int i = 0; i < count; i++)
        vertex_Q.enQueue(rev_deg_id_vec[i].id);
    }
  }

  int mid_value = num_visited_verts / 2, last_index = new_start_vertex - 1;

  for (int i_vtx_id = 0; i_vtx_id < mid_value; i_vtx_id++) {
    int temp_old_vtx_id = newToOldOrdering[last_index];
    newToOldOrdering[last_index] = newToOldOrdering[ccSize + i_vtx_id];
    newToOldOrdering[ccSize + i_vtx_id] = temp_old_vtx_id;
    oldToNewOrdering[newToOldOrdering[last_index]] = last_index;
    oldToNewOrdering[newToOldOrdering[ccSize + i_vtx_id]] = ccSize + i_vtx_id;

    last_index--;
  }
  if ((num_visited_verts % 2) == 1)
    oldToNewOrdering[newToOldOrdering[ccSize + mid_value]] = ccSize + mid_value;

  ccSize = ccSize + num_visited_verts;

  delete[] r_vec_lvlStruc;
  delete[] c_vec_lvlStruc;
#ifdef DEBUG
  cout << "exiting rcm_reordering_g" << endl;
#endif
  return 0;
}

// On return, root contains the psedo-peripheral vertex
void findPseudoPeripheralVertex(int &root, int *r_vec, int *c_vec, int *mask,
                                int &num_lvls, int *r_vec_lvlStruc,
                                int *c_vec_lvlStruc) {

  int j, j1, k, ccSize, minDeg, numDeg, vertex, new_Levl;

  // determine the level structure rooted at root
  findRootedLevelStructures(root, r_vec, c_vec, mask, num_lvls, r_vec_lvlStruc,
                            c_vec_lvlStruc);
  ccSize = r_vec_lvlStruc[num_lvls];

  if (num_lvls == 1 || num_lvls == ccSize)
    return;

  // pick a vertex with minimum degree from the last leve
  while (true) {
    j1 = r_vec_lvlStruc[num_lvls - 1];
    minDeg = ccSize;
    root = c_vec_lvlStruc[j1];

    if (ccSize != j1) {
      for (j = j1; j < ccSize; j++) {
        vertex = c_vec_lvlStruc[j];
        numDeg = 0;

        for (k = r_vec[vertex]; k < r_vec[vertex + 1]; k++) {
          if (mask[c_vec[k]] > 0)
            numDeg = numDeg + 1;
        }
        if (numDeg < minDeg) {
          root = vertex;
          minDeg = numDeg;
        }
      }
    }

    // Generate its rooted level structure
    findRootedLevelStructures(root, r_vec, c_vec, mask, new_Levl,
                              r_vec_lvlStruc, c_vec_lvlStruc);

    if (new_Levl <= num_lvls || num_lvls >= ccSize)
      return;
    num_lvls = new_Levl;
  }
}

// For the given root, it gives the rooted level structures, actual level is
// returned level -1

void findRootedLevelStructures(int root, int *r_vec, int *c_vec, int *mask,
                               int &num_lvls, int *r_vec_lvlStruc,
                               int *c_vec_lvlStruc) {

  int i, j, levelBegin, levelEnd, ccSize, levelSize, neighbor;

  mask[root] = 0;
  num_lvls = 0;
  levelEnd = 0;
  ccSize = 1;    // current visited vertex count
  levelSize = 1; // current level wid

  c_vec_lvlStruc[0] = root;

  // levelBegin is the pointer to the beginning of the current level, and
  // levelEnd points to the end of this level

  while (levelSize > 0) {
    levelBegin = levelEnd;
    r_vec_lvlStruc[num_lvls] = levelBegin;
    levelEnd = ccSize;

    // generate the next level by finding all the masked neighbors of vertices
    // in the current level

    for (i = levelBegin; i < levelEnd; i++) {
      for (j = r_vec[c_vec_lvlStruc[i]]; j < r_vec[c_vec_lvlStruc[i] + 1];
           j++) {
        neighbor = c_vec[j];
        if (mask[neighbor] != 0) {
          c_vec_lvlStruc[ccSize] = neighbor;
          ccSize = ccSize + 1;
          mask[neighbor] = 0;
        }
      }
    }

    // compute the current level width
    //           //if it is nonzero, generate the next level
    levelSize = ccSize - levelEnd;
    num_lvls = num_lvls + 1;
  }

  // reset mask to one for the vertices in the level structure
  r_vec_lvlStruc[num_lvls] = levelEnd;

  for (i = 0; i < ccSize; i++) {
    mask[c_vec_lvlStruc[i]] = 1;
  }
}

/*int RCMonSupNodes(graph_t *g, graph_bigAndMapping *g_big)
{

         supNodeToIndexMap supNodeIndexInBigGraph[g_big->n];
#pragma omp parallel for
         for(int i=0; i < g_big->N; i++)
         {
                const unsigned int s_edge = g_big->num_edges_coarsened[i];
                const unsigned int e_edge = g_big->num_edges_coarsened[i+1];

                for(int j= s_edge; j < e_edge; j++)
                {
                  supNodeIndexInBigGraph[g_big->adj_coarsened[j]].supNodeIndex =
i;

                  supNodeIndexInBigGraph[g_big->adj_coarsened[j]].indexInSuperNode
= j-s_edge;
                }
           }

          float time_rcm = timer();
         for(int i=0; i < g_big->N; i++)
         {
                const unsigned int s_edge = g_big->num_edges_coarsened[i];
                const unsigned int e_edge = g_big->num_edges_coarsened[i+1];

                const unsigned int start_group_vtx =
g_big->adj_coarsened[s_edge] ; const unsigned int end_group_vtx =
g_big->adj_coarsened[e_edge-1] ;

                unsigned int superNodeContents[e_edge - s_edge];
                Pair superNode_map[e_edge - s_edge];

                for(int j= s_edge; j < e_edge; j++)
                {
                  superNodeContents[j-s_edge] = g_big->adj_coarsened[j];
                  superNode_map[j-s_edge].first = j-s_edge;
                  superNode_map[j-s_edge].second =  g_big->adj_coarsened[j] ;
                }

                int min_adj_index = g_big->num_edges[ start_group_vtx ];
                int max_adj_index = g_big->num_edges[ end_group_vtx + 1  ];

                int superNodeNum_edges[e_edge - s_edge + 1];
                int superNode_adj[ max_adj_index - min_adj_index ];

                int minDegree = e_edge - s_edge +1, minDegreeVertex = 0;

                int superNode_index = 0;
                int superNode_adj_index = 0;

                for(int j= s_edge; j < e_edge; j++)
                {

                        unsigned int orgVertex_n = g_big->adj_coarsened[j] ;
                        superNodeNum_edges[ superNode_index ] =
superNode_adj_index ;

                        const unsigned int a = g_big->num_edges[orgVertex_n];
                        const unsigned int b = g_big->num_edges[orgVertex_n+1];

                        int vertex_degree = 0;
                        for(int org_adj_index = a; org_adj_index <b;
org_adj_index++)
                        {
                              unsigned int orgVertex_adj =
g_big->adj[org_adj_index] ; if(
supNodeIndexInBigGraph[orgVertex_adj].supNodeIndex == i )
                                {
                                       superNode_adj[superNode_adj_index] =
supNodeIndexInBigGraph[orgVertex_adj].indexInSuperNode; superNode_adj_index ++;
                                       vertex_degree ++;
                                }


                        }
                       if( minDegree > vertex_degree )
                        {
                              minDegree = vertex_degree;
                              minDegreeVertex = superNode_index;
                        }
                        superNode_index ++;
                }
                superNodeNum_edges[ superNode_index ] = superNode_adj_index;

                 int num_vertices_subgraph = e_edge - s_edge;
                 int oldToNewOrdering[ num_vertices_subgraph ];
                 int newToOldOrdering[ num_vertices_subgraph ];
                 int mask [ num_vertices_subgraph ];

                 for(int i_mask =0; i_mask < num_vertices_subgraph; i_mask++)
                     mask[i_mask] = 1;

                 int firstVtxinBFS, lastVtxinBFS, ccSize=0, root =0;


                 for(int i_mask =0; i_mask < num_vertices_subgraph; i_mask++)
                 {
                      if( ccSize >= num_vertices_subgraph)
                       break;

                      if ( mask[i_mask] != 0 )
                      {
                         root = i_mask;
                         rcm_reordering(num_vertices_subgraph,
superNodeNum_edges, superNode_adj, root, mask, oldToNewOrdering,
newToOldOrdering,firstVtxinBFS, lastVtxinBFS, ccSize);
                      }
                 }

                int local_vtx_id = 0;
                for(int j= s_edge; j < e_edge; j++)
                {
                        int curr_loc_vtx_id = oldToNewOrdering [
superNode_map[local_vtx_id].first ] ;

                        if( (curr_loc_vtx_id < 0) || curr_loc_vtx_id >
(e_edge-s_edge-1)) printf("out of bound, new id: %d\n", curr_loc_vtx_id );
                        g_big->adj_coarsened[j] = superNode_map[ curr_loc_vtx_id
].second ;

                        local_vtx_id ++;
                }


         }
          printf("SEQ_RCM TIME: = %0.4lf \n", timer() - time_rcm);


  return 0;
} */

int rcm_reordering(int num_verts, int *num_edges, int *adj_list, int root,
                   int mask[], int oldToNewOrdering[], int newToOldOrdering[],
                   int &firstVtxinBFS, int &lastVtxinBFS, int &ccSize) {

  int num_lvls = 0;
  int r_vec_lvlStruc[num_verts];
  int c_vec_lvlStruc[num_verts];

  // Find the source vertex
  findPseudoPeripheralVertex(root, num_edges, adj_list, mask, num_lvls,
                             r_vec_lvlStruc, c_vec_lvlStruc);

  firstVtxinBFS = root;

  // Now do the BFS
  int copy_mask[num_verts];

  for (int i = 0; i < num_verts; i++) {
    copy_mask[i] = mask[i];
  }

  int Q_size = std::max(num_edges[num_verts], num_verts);

  Queue_Arry vertex_Q(Q_size);

  vertex_Q.enQueue(root);

  int num_visited_verts = 0;
  int new_start_vertex = ccSize;

  while (!vertex_Q.isEmpty()) {
    int parent_vertex = vertex_Q.deQueue();
    if (mask[parent_vertex] == 1) {
      mask[parent_vertex] = 0;
      newToOldOrdering[new_start_vertex] = parent_vertex;
      num_visited_verts++;
      new_start_vertex++;

      int start_edge = num_edges[parent_vertex];
      int end_edge = num_edges[parent_vertex + 1];

      std::vector<degree_id_pair> deg_id_vec(end_edge - start_edge);
      int count = 0;

      for (int i = start_edge; i < end_edge; i++) {
        int adj_vertex_id = adj_list[i];
        if (mask[adj_vertex_id] == 1) {
          int deg_of_vertex =
              num_edges[adj_vertex_id + 1] - num_edges[adj_vertex_id];

          int i_deg = 0;
          for (int index_r = num_edges[adj_vertex_id];
               index_r < num_edges[adj_vertex_id + 1]; index_r++) {
            if (copy_mask[adj_list[index_r]] == 1)
              i_deg++;
          }

          deg_id_vec[count].deg = deg_of_vertex;
          if (deg_of_vertex != i_deg) {
            deg_id_vec[count].deg = i_deg;
          }

          deg_id_vec[count].id = adj_vertex_id;
          count++;
        }
      }

      std::sort(deg_id_vec.begin(), deg_id_vec.begin() + count,
                compare_deg_id_pair);
      for (int i = 0; i < count; i++)
        vertex_Q.enQueue(deg_id_vec[i].id);
    }
  }

  int mid_value = num_visited_verts / 2, last_index = new_start_vertex - 1;

  for (int i_vtx_id = 0; i_vtx_id < mid_value; i_vtx_id++) {
    int temp_old_vtx_id = newToOldOrdering[last_index];
    newToOldOrdering[last_index] = newToOldOrdering[ccSize + i_vtx_id];
    newToOldOrdering[ccSize + i_vtx_id] = temp_old_vtx_id;

    oldToNewOrdering[newToOldOrdering[last_index]] = last_index;
    oldToNewOrdering[newToOldOrdering[ccSize + i_vtx_id]] = ccSize + i_vtx_id;

    last_index--;
  }
  if ((num_visited_verts % 2) == 1)
    oldToNewOrdering[newToOldOrdering[ccSize + mid_value]] = ccSize + mid_value;

  ccSize = ccSize + num_visited_verts;

  return 0;
}

// Find levels given a graph
void find_levels(int nnz, int nRows, int *num_edges, int *adj,
                 int *&adjLevelSet, int &numLevel, int *&numEdgesLevelSet) {
#ifdef DEBUG
  cout << "Function:find_levels() invoked" << endl;
#endif

  int *adj_copy = new int[nnz];
  int *levelSetPtr = new int[nRows + 1];
  bool *processedVerts = new bool[nRows];
  bool *notCandidateVerts = new bool[nRows];

  for (int i = 0; i < nRows; i++) {
    processedVerts[i] = false;
    notCandidateVerts[i] = false;
  }

  for (int j = 0; j < nnz; j++)
    adj_copy[j] = adj[j];

  int *candLevelSet = new int[nRows];

  // int numVisitedVtxs = 0;
  int indexVer = 0;
  int candIndex = 0;

  levelSetPtr[0] = 0;
  numLevel = 0;

  while (indexVer < nRows) {
    numLevel++;

    if (numLevel == 1) {
      for (int iRow = 0; iRow < nRows; iRow++) {
        if (adj_copy[num_edges[iRow]] == iRow) {
          adjLevelSet[indexVer] = iRow;
          processedVerts[iRow] = true;
          notCandidateVerts[iRow] = true;
          indexVer++;
        }
      }
    } else { // Just check the candidate level set vertices
      for (int iRow = 0; iRow < candIndex; iRow++) {
        int candLevelVertex = candLevelSet[iRow];
        bool firstPositiveNeighbor = true;
        for (int jInd = num_edges[candLevelVertex];
             jInd < num_edges[candLevelVertex + 1]; jInd++) {
          if (adj_copy[jInd] >= 0) {
            if (adj_copy[jInd] == candLevelVertex) {
              adjLevelSet[indexVer] = candLevelVertex;
              indexVer++;
              processedVerts[candLevelVertex] = true;
              firstPositiveNeighbor = false;
            } else {
              notCandidateVerts[candLevelVertex] = false;
              firstPositiveNeighbor = false;
            }
          }
          if (!firstPositiveNeighbor) {
            break;
          }
        }
      }
    }
    levelSetPtr[numLevel] = indexVer;
    // numVisitedVtxs += (indexVer - levelSetPtr [numLevel-1]);
    candIndex = 0;

    for (int indexInLevel = levelSetPtr[numLevel - 1];
         indexInLevel < levelSetPtr[numLevel]; indexInLevel++) {
      int gVertex = adjLevelSet[indexInLevel];
      for (int jInd = num_edges[gVertex]; jInd < num_edges[gVertex + 1];
           jInd++) {
        // These are the vertices connected to the vertices in level set
        int neighborLvlVtx = adj[jInd];

        // From the adjacency of this vertex, delete the dependency on vertices
        // in level set
        if (!processedVerts[neighborLvlVtx]) {
          if (!notCandidateVerts[neighborLvlVtx]) {
            candLevelSet[candIndex] = neighborLvlVtx;
            notCandidateVerts[neighborLvlVtx] = true;
            candIndex++;
          }
          for (int jNeighborIndex = num_edges[neighborLvlVtx];
               jNeighborIndex < num_edges[neighborLvlVtx + 1];
               jNeighborIndex++) {
            if (adj_copy[jNeighborIndex] == gVertex)
              adj_copy[jNeighborIndex] = VERTEX_MARKER;
          }
        }
      }
    }
    if (levelSetPtr[numLevel] - levelSetPtr[numLevel - 1] == 0) {
      cout << "Number of vertices did not change in level" << endl;
      break;
    }
  }

  numEdgesLevelSet = new int[numLevel + 1];
  for (int i = 0; i < numLevel + 1; i++) {
    numEdgesLevelSet[i] = levelSetPtr[i];
  }

  // Clean memory
  delete[] adj_copy;
  delete[] processedVerts;
  delete[] notCandidateVerts;
  delete[] candLevelSet;
  delete[] levelSetPtr;

#ifdef DEBUG
  cout << "Function:find_levels() completed" << endl;
  cout << "Vertices in Graph: " << nRows << " Visited Vtxs: " << indexVer
       << " Num-Level: " << numLevel << endl;
#endif
}

// Find levels starting from the maximum degree vertex
void find_levels_from_maxDegree_vertex(int nnz, int nRows, int *num_edges,
                                       int *adj, int *&adjLevelSet,
                                       int &numLevel, int *&numEdgesLevelSet) {

#ifdef DEBUG
  cout << "Function:find_levels_from_maxDegree_vertex() invoked" << endl;
#endif

  bool *processedVerts = new bool[nRows];
  assert(processedVerts != NULL);

  bool *cantBeInLevelSets = new bool[nRows];
  assert(cantBeInLevelSets != NULL);

  for (int irow = 0; irow < nRows; irow++) {
    processedVerts[irow] = false;
    cantBeInLevelSets[irow] = false;
  }

  int *candidateLevelVerts = new int[nRows];
  assert(candidateLevelVerts != NULL);

  int *levelSetPtr = new int[nRows + 1];
  assert(levelSetPtr != NULL);

  int indexVert = 0;
  int candIndex = 0;

  numLevel = 0;
  levelSetPtr[0] = 0;

  while (indexVert < nRows) {
    numLevel++;

    if ((numLevel == 1) || (candIndex == 0)) {
      // Find maximum degree vertex and put it in the first level
      int maxDeg = 0, maxDegVtx = 0;
      for (int iRow = 0; iRow < nRows; iRow++) {
        if (!processedVerts[iRow]) {
          int local_deg = num_edges[iRow + 1] - num_edges[iRow];
          if (local_deg > maxDeg) {
            maxDeg = local_deg;
            maxDegVtx = iRow;
          }
        }
      }

      adjLevelSet[indexVert] = maxDegVtx;
      processedVerts[maxDegVtx] = true;
      indexVert++;
    } else { // just check the candidate level set vertices
      for (int iRow = 0; iRow < candIndex; iRow++) {

        int candLvlVtx = candidateLevelVerts[iRow];

        adjLevelSet[indexVert] = candLvlVtx;
        indexVert++;

        for (int jIndex = num_edges[candLvlVtx];
             jIndex < num_edges[candLvlVtx + 1]; jIndex++) {
          if (cantBeInLevelSets[adj[jIndex]])
            cantBeInLevelSets[adj[jIndex]] = false;
        }
      }
    }

    levelSetPtr[numLevel] = indexVert;
    candIndex = 0;

    for (int indexInLevel = levelSetPtr[numLevel - 1];
         indexInLevel < levelSetPtr[numLevel]; indexInLevel++) {
      int gVtx = adjLevelSet[indexInLevel];

      for (int jIndex = num_edges[gVtx]; jIndex < num_edges[gVtx + 1];
           jIndex++) {
        // these are the vertices connected to the vertices in current level
        int neighborLevelSetVtx = adj[jIndex];

        // From the adjacency of this vertex, delete the dependency on vertices
        // in level set Also make sure; two connected vertices are not in the
        // same level
        if (!processedVerts[neighborLevelSetVtx] &&
            !cantBeInLevelSets[neighborLevelSetVtx]) {
          // need to make sure that it's neighbor is not in candidateLevelVerts
          candidateLevelVerts[candIndex] = neighborLevelSetVtx;
          candIndex++;
          processedVerts[neighborLevelSetVtx] = true;

          for (int jCantBeIndex = num_edges[neighborLevelSetVtx];
               jCantBeIndex < num_edges[neighborLevelSetVtx + 1];
               jCantBeIndex++) {
            if ((adj[jCantBeIndex] != neighborLevelSetVtx) &&
                !processedVerts[adj[jCantBeIndex]])
              cantBeInLevelSets[adj[jCantBeIndex]] = true;
          }
        }
      }
    }

    if (levelSetPtr[numLevel] - levelSetPtr[numLevel - 1] == 0) {
      cout << "Visited vertex count do not change" << endl;
      cout << "Candidate Vertex count:" << candIndex << endl;
      break;
    }
  }

  numEdgesLevelSet = new int[numLevel + 1];
  for (int i = 0; i < numLevel + 1; i++)
    numEdgesLevelSet[i] = levelSetPtr[i];

  // Cleaning memory
  delete[] processedVerts;
  delete[] cantBeInLevelSets;
  delete[] candidateLevelVerts;
  delete[] levelSetPtr;

#ifdef DEBUG
  cout << "Function:find_levels_from_maxDegree_vertex() completed" << endl;
  cout << "Vertices in Graph: " << nRows << " Visited Vtxs: " << indexVert
       << " Num-Level: " << numLevel << endl;
#endif
}

// Color the graph using Boost library
void BGL_ordering(int nnz, int nRows, int *num_edges, int *adj, size_t *p,
                  int *numColors, int **colorPtr) {
#ifdef DEBUG
  cout << "BGL_ordering invoked " << endl;

#endif

  typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS>
      Graph;

  typedef boost::graph_traits<Graph>::vertices_size_type vertices_size_type;
  typedef boost::property_map<Graph, boost::vertex_index_t>::const_type
      vertex_index_map;
  typedef std::pair<int, int> Edge;

  std::vector<Edge> edge_array(nnz - nRows);

  size_t k = 0;
  for (size_t i = 0; i < (unsigned int)nRows; ++i) {
    for (int j = num_edges[i]; j < num_edges[i + 1]; ++j) {
      if ((int)i != adj[j]) {
        edge_array[k] = Edge((int)i, adj[j]);
        ++k;
      }
    }
  }

  Graph g(&edge_array[0], &edge_array[0] + edge_array.size(), nRows);
  std::vector<vertices_size_type> color_vec(num_vertices(g));
  boost::iterator_property_map<vertices_size_type *, vertex_index_map> color(
      &color_vec.front(), boost::get(boost::vertex_index, g));

  boost::sequential_vertex_coloring(g, color);
  std::map<int, std::set<int>> colorMap;
  for (int i = 0; i < nRows; ++i) {
    colorMap[color[i]].insert(i);
  }

  if (NULL != colorPtr && NULL != numColors) {
    *numColors = colorMap.size();
    *colorPtr = (int *)malloc((*numColors + 1) * sizeof(int));
  }

  int cIdx = 0;
  k = 0;
  (*colorPtr)[0] = 0;
  for (std::map<int, std::set<int>>::const_iterator it = colorMap.begin();
       it != colorMap.end(); ++it) {
    ++cIdx;
    (*colorPtr)[cIdx] = (*colorPtr)[cIdx - 1] + it->second.size();
    for (std::set<int>::const_iterator it2 = it->second.begin();
         it2 != it->second.end(); ++it2) {
      p[k] = *it2;
      ++k;
    }
  }

#ifdef DEBUG
  cout << "BGL_ordering completed" << endl;
  cout << "Num colors: " << colorMap.size() << endl;
#endif

  return;
}

// Renumber the vertices according to a reordering
void renumberGraphUsingReorderedVertices(long numVerts, long NNZ,
                                         unsigned int **inrVec,
                                         unsigned int **incVec,
                                         unsigned int **inDegree,
                                         unsigned int *oldToNewPerm,
                                         unsigned int *newToOldPerm) {

#ifdef DEBUG
  cout << "Function:renumberGraphUsingReorderedVertices invoked" << endl;
#endif

  unsigned int *rVec = *inrVec;
  unsigned int *cVec = *incVec;
  unsigned int *degree = *inDegree;

  unsigned int *copycVec = new unsigned int[NNZ];
  unsigned int *copyDegree = new unsigned int[NNZ];
  unsigned int *copyrVec = new unsigned int[numVerts + 1];

// Copy edge vector to temporary
#pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < numVerts + 1; i++) {
    copyrVec[i] = rVec[i];
  }

// Copy adj vector to temporary
#pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < NNZ; i++) {
    copycVec[i] = cVec[i];
    copyDegree[i] = degree[i];
  }

  unsigned int nnzCount = 0;
  rVec[0] = 0;

  for (unsigned int i = 0; i < numVerts; i++) {
    unsigned int oldVertex = newToOldPerm[i];

    unsigned int firstNNZ = copyrVec[oldVertex];
    unsigned int lastNNZ = copyrVec[oldVertex + 1];

    nnzCount = nnzCount + (lastNNZ - firstNNZ);

    rVec[i + 1] = nnzCount;
  }

  assert(rVec[numVerts] == copyrVec[numVerts]);

#pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < numVerts; i++) {
    unsigned int oldVertex = newToOldPerm[i];

    unsigned int firstNNZ = copyrVec[oldVertex];
    unsigned int lastNNZ = copyrVec[oldVertex + 1];

    unsigned int new_firstNNZ = rVec[i];

    for (unsigned int j = firstNNZ; j < lastNNZ; j++) {
      cVec[new_firstNNZ] = oldToNewPerm[copycVec[j]];
      degree[new_firstNNZ] = copyDegree[j];
      new_firstNNZ++;
    }
  }

  // Sort the adjacency list of the graph
  std::vector<adj_deg> adjVector(NNZ);

#pragma omp parallel
  {

#pragma omp for schedule(static) nowait
    for (int i = 0; i < NNZ; i++) {
      adjVector[i].adj = cVec[i];
      adjVector[i].deg = degree[i];
    }
#pragma omp barrier

#pragma omp for schedule(static) nowait
    for (int i = 0; i < numVerts; i++)
      std::sort(adjVector.begin() + rVec[i], adjVector.begin() + rVec[i + 1],
                compAdjDeg);
#pragma omp barrier

#pragma omp for schedule(static) nowait
    for (int i = 0; i < numVerts; i++) {
      unsigned int firstNNZ = rVec[i];
      unsigned int lastNNZ = rVec[i + 1];

      for (unsigned int j = firstNNZ; j < lastNNZ; j++) {
        cVec[j] = adjVector[j].adj;
        degree[j] = adjVector[j].deg;
      }
    }
#pragma omp barrier
  }

  delete[] copycVec;
  delete[] copyDegree;
  delete[] copyrVec;

#ifdef DEBUG
  cout << "Function:renumberGraphUsingReorderedVertices completed" << endl;
#endif
}

/* Given intermediate mapping of vertices find the final mapping from the
 * coarsest vertices to the finest vertices Input: numSupRows - number of
 * coarsest vertices vectorSizes - vector containing number of vertices in
 * intermediate coarsened graphs mappingVectors - vector containing the mapping
 * of fine vertices to coarse vertices Output: numEdgesSupRowsToRows - points to
 * mapSupRowstoRows to indicate where each coarsest row starts mapSupRowstoRows
 * - contains the vertices of the finest graph
 */
void findFinalMapping(int numSupRows, vector<unsigned int> &vectorSizes,
                      vector<unsigned int *> &mappingVectors,
                      unsigned int *&numEdgesSupRowsToRows,
                      unsigned int *&mapSupRowstoRows) {

  vector<vector<unsigned int>> finalAdjLists;
  vector<vector<unsigned int>> tempVectAdjLists;

  for (int i = vectorSizes.size() - 1; i >= 0; i--) {

    if (i == (int)vectorSizes.size() - 1) {
      finalAdjLists.resize(numSupRows);

      for (unsigned int j = 0; j < vectorSizes[i]; j++)
        finalAdjLists[mappingVectors[i][j]].push_back(j);
    } else {
      tempVectAdjLists.resize(vectorSizes[i + 1]);
      for (unsigned int j = 0; j < vectorSizes[i + 1]; j++)
        tempVectAdjLists[j].resize(0);

      for (unsigned int j = 0; j < vectorSizes[i]; j++)
        tempVectAdjLists[mappingVectors[i][j]].push_back(j);

      for (int j = 0; j < numSupRows; j++) {
        vector<unsigned int> curAdjList;
        for (unsigned int k = 0; k < finalAdjLists[j].size(); k++) {
          int finerRowIndex = finalAdjLists[j][k];
          for (unsigned int l = 0; l < tempVectAdjLists[finerRowIndex].size();
               l++)
            curAdjList.push_back(tempVectAdjLists[finerRowIndex][l]);
        }
        finalAdjLists[j] = curAdjList;
      }
    }
  }

  numEdgesSupRowsToRows = new unsigned int[numSupRows + 1];

  numEdgesSupRowsToRows[0] = 0;
  int mapIndex = 0;
  for (int i = 0; i < numSupRows; i++) {
    numEdgesSupRowsToRows[i + 1] =
        numEdgesSupRowsToRows[i] + finalAdjLists[i].size();
    for (unsigned int j = 0; j < finalAdjLists[i].size(); j++) {
      mapSupRowstoRows[mapIndex] = finalAdjLists[i][j];
      mapIndex++;
    }
  }
}

/* randomMatching - Find a maximal matching of the vertcies. Visit a vertex u in
 * random order and match u randomly with one of its unmatched neighbors Input:
 * g - the graph with edge and vertex weight Output: perm - mapping of vertices
 * of g to the vertices of coarsendG coarsenedG - graph of the coarsened
 * vertices */

void BAND_k::randomMatching(C_GRAPH g, unsigned int *perm,
                            C_GRAPH &coarsenedG) {
#ifdef DEBUG
  cout << "BAND_k::randomMatching invoked" << endl;
#endif
  // Matched edges
  set<Edge> matchEdges;

  cout << g.N << endl;
  // Initially vertices are not matched
  bool isMatched[g.N];

  srand(unsigned(std::time(0)));
  vector<unsigned int> myvector;

  // Initialize these values
  for (unsigned int i = 0; i < g.N; ++i) {
    isMatched[i] = false;
    myvector.push_back(i);
  }

  // Randomly shuffles the vertices
  std::random_shuffle(myvector.begin(), myvector.end());

  // Visit a vertex at random and randomly match it with one of its neighbors
  for (int i = 0; i < g.N; ++i) {
    unsigned int currVtx = myvector[i];

    if (!isMatched[currVtx]) {
      // Make a vector of unmatched neighbors
      vector<unsigned int> neighbors;

      for (unsigned int j = g.r_vec[currVtx]; j < g.r_vec[currVtx + 1]; j++) {
        if ((g.c_vec[j] != currVtx) && !isMatched[g.c_vec[j]]) {
          neighbors.push_back(g.c_vec[j]);
        }
      }

      // Randomly select one of its unmatched neighbors
      int numVtxs = neighbors.size();
      if (numVtxs == 1) {
        isMatched[currVtx] = true;
        isMatched[neighbors[0]] = true;
        matchEdges.insert(Edge(currVtx, neighbors[0]));
      } else if (numVtxs > 1) {
        int randIndex = rand() % numVtxs;
        isMatched[currVtx] = true;
        isMatched[neighbors[randIndex]] = true;
        matchEdges.insert(Edge(currVtx, neighbors[randIndex]));
      }
    }
  }

  // Count the vertices in the coarsened graph
  // Store a mapping from coarsend vertex to the vertices in the finer graph
  int numConargenedVertices = 0;
  vector<vector<unsigned int>> superRowsToRows;

  for (set<Edge>::iterator it = matchEdges.begin(); it != matchEdges.end();
       ++it) {
    unsigned int u = it->either();
    unsigned int v = it->other(u);

    perm[u] = numConargenedVertices;
    perm[v] = numConargenedVertices;

    vector<unsigned int> supRowContains(2, 0);
    supRowContains[0] = u;
    supRowContains[1] = v;
    superRowsToRows.push_back(supRowContains);
    numConargenedVertices++;
  }

  for (int i = 0; i < g.N; ++i) {
    if (!isMatched[i]) {
      perm[i] = numConargenedVertices;
      vector<unsigned int> supRowContains(1, i);
      superRowsToRows.push_back(supRowContains);
      numConargenedVertices++;
    }
  }

  coarsenedG.N = numConargenedVertices;

  // Calculate the vertex weights of the coarsened graph
  coarsenedG.vertexWeight = new unsigned int[numConargenedVertices];
  for (int i = 0; i < coarsenedG.N; i++) {
    unsigned int currVtxWeight = 0;
    for (unsigned int j = 0; j < superRowsToRows[i].size(); j++) {
      currVtxWeight = currVtxWeight + g.vertexWeight[superRowsToRows[i][j]];
    }
    coarsenedG.vertexWeight[i] = currVtxWeight;
  }

  coarsenedG.r_vec = new unsigned int[coarsenedG.N + 1];

  // vectors to compute adjacency of coarsened graph and edge weights
  vector<unsigned int> adjVector;
  vector<unsigned int> weightVector;

  coarsenedG.r_vec[0] = 0;

  for (int i = 0; i < coarsenedG.N; i++) {

    // Maps neighbor to weight
    map<unsigned int, unsigned int> adjNeighborWeightMap;

    for (unsigned int j = 0; j < superRowsToRows[i].size(); j++) {
      unsigned int vtxInCurSuperRow = superRowsToRows[i][j];
      for (unsigned int j = g.r_vec[vtxInCurSuperRow];
           j < g.r_vec[vtxInCurSuperRow + 1]; j++) {
        unsigned int adjVtx = g.c_vec[j];
        unsigned int connectedSupRow = perm[adjVtx];

        map<unsigned int, unsigned int>::iterator it =
            adjNeighborWeightMap.find(connectedSupRow);

        if (it != adjNeighborWeightMap.end()) {
          int totalWeight = g.edgeWeight[j] + it->second;
          it->second = totalWeight;
        } else {
          adjNeighborWeightMap.insert(pair<unsigned int, unsigned int>(
              connectedSupRow, g.edgeWeight[j]));
        }
      }
    }

    // Sort the neighbors according to weight in increasing order
    vector<pair<unsigned int, unsigned int>> adjWeightVector(
        adjNeighborWeightMap.begin(), adjNeighborWeightMap.end());
    sort(adjWeightVector.begin(), adjWeightVector.end(),
         sortBySecond<unsigned int, unsigned int>());

    // Put the neighbors in the graph_t struct
    coarsenedG.r_vec[i + 1] = coarsenedG.r_vec[i] + adjWeightVector.size();

    for (unsigned int j = 0; j < adjWeightVector.size(); j++) {
      adjVector.push_back(adjWeightVector[j].first);
      weightVector.push_back(adjWeightVector[j].second);
    }
  }

  coarsenedG.NNZ = adjVector.size();

  assert(coarsenedG.NNZ == coarsenedG.r_vec[coarsenedG.N]);

  coarsenedG.c_vec = new unsigned int[coarsenedG.NNZ];
  coarsenedG.edgeWeight = new unsigned int[coarsenedG.NNZ];

  for (int j = 0; j < coarsenedG.NNZ; j++) {
    coarsenedG.c_vec[j] = adjVector[j];
    coarsenedG.edgeWeight[j] = weightVector[j];
  }
#ifdef DEBUG
  cout << "BAND_k::randomMatching completed" << endl;
#endif
}

/* headyEdgeMatching - Find a maximal matching of the vertcies. Visit a vertex u
 * in random order and match u with v such that w(u,v) - weight of edge (u,v) is
 * the maximum among the valid neighbors of u. Input: g - the graph with edge
 * and vertex weight Output: perm - mapping of vertices of g to the vertices of
 * coarsendG coarsenedG - graph of the coarsened vertices */

void BAND_k::heavyEdgeMatching(C_GRAPH g, unsigned int *perm,
                               C_GRAPH &coarsenedG) {
#ifdef DEBUG
  cout << "BAND_k::heavyEdgeMatching invoked" << endl;
#endif

  // Matched edges
  set<Edge> matchEdges;

  // Initially vertices are not matched
  bool *isMatched = new bool[g.N];

  srand(std::time(0));
  vector<unsigned int> myvector;

  // Initalize thse values
  for (unsigned int i = 0; i < g.N; i++) {
    isMatched[i] = false;
    myvector.push_back(i);
  }

  // Randomly shuffles the vertices
  std::random_shuffle(myvector.begin(), myvector.end());

  // Visit a vertex at random and match it with the heaviest edge
  for (int i = 0; i < g.N; i++) {
    unsigned int currVtx = myvector[i];

    if (!isMatched[currVtx]) {
      for (int j = (g.r_vec[currVtx + 1] - 1); j >= (int)g.r_vec[currVtx];
           j--) {

        if ((g.c_vec[j] != currVtx) && !isMatched[g.c_vec[j]]) {
          isMatched[currVtx] = true;
          isMatched[g.c_vec[j]] = true;
          matchEdges.insert(Edge(currVtx, g.c_vec[j]));
          break;
        }
      }
    }
  }

  // Count the vertices in the coarsened graph
  // Store a mapping from coarsend vertex to the vertices in the finer graph
  int numConargenedVertices = 0;
  vector<vector<unsigned int>> superRowsToRows;

  for (set<Edge>::iterator it = matchEdges.begin(); it != matchEdges.end();
       ++it) {
    unsigned int u = it->either();
    unsigned int v = it->other(u);

    perm[u] = numConargenedVertices;
    perm[v] = numConargenedVertices;

    vector<unsigned int> supRowContains(2, 0);
    supRowContains[0] = u;
    supRowContains[1] = v;
    superRowsToRows.push_back(supRowContains);
    numConargenedVertices++;
  }

  for (int i = 0; i < g.N; ++i) {
    if (!isMatched[i]) {
      perm[i] = numConargenedVertices;
      vector<unsigned int> supRowContains(1, i);
      superRowsToRows.push_back(supRowContains);
      numConargenedVertices++;
    }
  }

  coarsenedG.N = numConargenedVertices;

  // Calculate the vertex weights of the coarsened graph
  coarsenedG.vertexWeight = new unsigned int[numConargenedVertices];
  for (int i = 0; i < coarsenedG.N; i++) {
    unsigned int currVtxWeight = 0;
    for (unsigned int j = 0; j < superRowsToRows[i].size(); j++) {
      currVtxWeight = currVtxWeight + g.vertexWeight[superRowsToRows[i][j]];
    }
    coarsenedG.vertexWeight[i] = currVtxWeight;
  }

  coarsenedG.r_vec = new unsigned int[coarsenedG.N + 1];

  // vectors to compute adjacency and edge weights of coarsened graph
  vector<unsigned int> adjVector;
  vector<unsigned int> weightVector;

  coarsenedG.r_vec[0] = 0;

  for (int i = 0; i < coarsenedG.N; i++) {

    // Maps neighbor to weight
    map<unsigned int, unsigned int> adjNeighborWeightMap;

    for (unsigned int j = 0; j < superRowsToRows[i].size(); j++) {
      unsigned int vtxInCurSuperRow = superRowsToRows[i][j];
      for (unsigned int j = g.r_vec[vtxInCurSuperRow];
           j < g.r_vec[vtxInCurSuperRow + 1]; j++) {
        unsigned int adjVtx = g.c_vec[j];
        unsigned int connectedSupRow = perm[adjVtx];

        map<unsigned int, unsigned int>::iterator it =
            adjNeighborWeightMap.find(connectedSupRow);

        if (it != adjNeighborWeightMap.end()) {
          int totalWeight = g.edgeWeight[j] + it->second;
          it->second = totalWeight;
        } else {
          adjNeighborWeightMap.insert(pair<unsigned int, unsigned int>(
              connectedSupRow, g.edgeWeight[j]));
        }
      }
    }

    // Sort the neighbors according to weights in increasing order
    vector<pair<unsigned int, unsigned int>> adjWeightVector(
        adjNeighborWeightMap.begin(), adjNeighborWeightMap.end());
    sort(adjWeightVector.begin(), adjWeightVector.end(),
         sortBySecond<unsigned int, unsigned int>());

    // Put the neighbors in the graph_t struct
    coarsenedG.r_vec[i + 1] = coarsenedG.r_vec[i] + adjWeightVector.size();

    for (unsigned int j = 0; j < adjWeightVector.size(); j++) {
      adjVector.push_back(adjWeightVector[j].first);
      weightVector.push_back(adjWeightVector[j].second);
    }
  }

  coarsenedG.NNZ = adjVector.size();

  assert(coarsenedG.NNZ == coarsenedG.r_vec[coarsenedG.N]);

  coarsenedG.c_vec = new unsigned int[coarsenedG.NNZ];
  coarsenedG.edgeWeight = new unsigned int[coarsenedG.NNZ];

  for (int j = 0; j < coarsenedG.NNZ; j++) {
    coarsenedG.c_vec[j] = adjVector[j];
    coarsenedG.edgeWeight[j] = weightVector[j];
  }

  delete[] isMatched;

#ifdef DEBUG
  cout << "BAND_k::heavyEdgeMatching completed" << endl;
#endif
}

/* lightEdgeMatching - Finds a maximal matching of the vertcies. Visit a vertex
 * u in random order and match u with v such that w(u,v) - weight of edge (u,v)
 * is the minimum among the valid neighbors of u. Input: g - the graph with edge
 * and vertex weight Output: perm - mapping of vertices of g to the vertices of
 * coarsendG coarsenedG - graph of the coarsened vertices */

void BAND_k::lightEdgeMatching(C_GRAPH g, unsigned int *perm,
                               C_GRAPH &coarsenedG) {

#ifdef DEBUG
  cout << "BAND_k::lightEdgeMatching invoked" << endl;
#endif

  // Matched edges
  set<Edge> matchEdges;

  // Initially vertices are not matched
  bool isMatched[g.N];

  srand(unsigned(std::time(0)));
  vector<unsigned int> myvector;

  // Initialize these values
  for (unsigned int i = 0; i < g.N; i++) {
    isMatched[i] = false;
    myvector.push_back(i);
  }

  // Randomly shuffles the vertices
  std::random_shuffle(myvector.begin(), myvector.end());

  // Visit a vertex at random and match it with the lightest edge
  for (int i = 0; i < g.N; i++) {
    unsigned int currVtx = myvector[i];

    if (!isMatched[currVtx]) {
      for (unsigned int j = g.r_vec[currVtx]; j < g.r_vec[currVtx + 1]; j++) {

        if ((g.c_vec[j] != currVtx) && !isMatched[g.c_vec[j]]) {
          isMatched[currVtx] = true;
          isMatched[g.c_vec[j]] = true;
          matchEdges.insert(Edge(currVtx, g.c_vec[j]));
          break;
        }
      }
    }
  }

  // Count the vertices in the coarsened graph
  // Store a mapping from coarsend vertex to the vertices in the finer graph
  int numConargenedVertices = 0;
  vector<vector<unsigned int>> superRowsToRows;

  for (set<Edge>::iterator it = matchEdges.begin(); it != matchEdges.end();
       ++it) {
    unsigned int u = it->either();
    unsigned int v = it->other(u);

    perm[u] = numConargenedVertices;
    perm[v] = numConargenedVertices;

    vector<unsigned int> supRowContains(2, 0);
    supRowContains[0] = u;
    supRowContains[1] = v;
    superRowsToRows.push_back(supRowContains);
    numConargenedVertices++;
  }

  for (int i = 0; i < g.N; ++i) {
    if (!isMatched[i]) {
      perm[i] = numConargenedVertices;
      vector<unsigned int> supRowContains(1, i);
      superRowsToRows.push_back(supRowContains);
      numConargenedVertices++;
    }
  }

  coarsenedG.N = numConargenedVertices;

  // Calculate the vertex weights of the coarsened graph
  coarsenedG.vertexWeight =
      (unsigned int *)malloc(numConargenedVertices * sizeof(unsigned int));
  for (int i = 0; i < coarsenedG.N; i++) {
    unsigned int currVtxWeight = 0;
    for (unsigned int j = 0; j < superRowsToRows[i].size(); j++) {
      currVtxWeight = currVtxWeight + g.vertexWeight[superRowsToRows[i][j]];
    }
    coarsenedG.vertexWeight[i] = currVtxWeight;
  }

  coarsenedG.r_vec = new unsigned int[coarsenedG.N + 1];

  // vectors to compute adjacency and edge weights of coarsened graph
  vector<unsigned int> adjVector;
  vector<unsigned int> weightVector;

  coarsenedG.r_vec[0] = 0;

  for (int i = 0; i < coarsenedG.N; i++) {

    // Maps neighbor to weight
    map<unsigned int, unsigned int> adjNeighborWeightMap;

    for (unsigned int j = 0; j < superRowsToRows[i].size(); j++) {
      unsigned int vtxInCurSuperRow = superRowsToRows[i][j];
      for (unsigned int j = g.r_vec[vtxInCurSuperRow];
           j < g.r_vec[vtxInCurSuperRow + 1]; j++) {
        unsigned int adjVtx = g.c_vec[j];
        unsigned int connectedSupRow = perm[adjVtx];

        map<unsigned int, unsigned int>::iterator it =
            adjNeighborWeightMap.find(connectedSupRow);

        if (it != adjNeighborWeightMap.end()) {
          int totalWeight = g.edgeWeight[j] + it->second;
          it->second = totalWeight;
        } else {
          adjNeighborWeightMap.insert(pair<unsigned int, unsigned int>(
              connectedSupRow, g.edgeWeight[j]));
        }
      }
    }

    // Sort the neighbors according to weights in increasing order
    vector<pair<unsigned int, unsigned int>> adjWeightVector(
        adjNeighborWeightMap.begin(), adjNeighborWeightMap.end());
    sort(adjWeightVector.begin(), adjWeightVector.end(),
         sortBySecond<unsigned int, unsigned int>());

    // Put the neighbors in the graph_t struct
    coarsenedG.r_vec[i + 1] = coarsenedG.r_vec[i] + adjWeightVector.size();

    for (unsigned int j = 0; j < adjWeightVector.size(); j++) {
      adjVector.push_back(adjWeightVector[j].first);
      weightVector.push_back(adjWeightVector[j].second);
    }
  }

  coarsenedG.NNZ = adjVector.size();

  assert(coarsenedG.NNZ == coarsenedG.r_vec[coarsenedG.N]);

  coarsenedG.c_vec = new unsigned int[coarsenedG.NNZ];
  coarsenedG.edgeWeight = new unsigned int[coarsenedG.NNZ];

  for (int j = 0; j < coarsenedG.NNZ; j++) {
    coarsenedG.c_vec[j] = adjVector[j];
    coarsenedG.edgeWeight[j] = weightVector[j];
  }

#ifdef DEBUG
  cout << "BAND_k::lightEdgeMatching completed" << endl;
#endif
}
