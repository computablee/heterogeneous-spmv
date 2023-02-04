# SpMV Benchmark
This is the source code that is the focus of my research towards my Master's thesis for an M.S. Computer Science at The University of Alabama in Huntsville.
The goal of this research is to determine whether multilevel representations of sparse matrices can improve performance of the sparse matrix-vector multiplication kernel on graphics accelerators and manycore CPUs.
The general idea is that multilevel representations of sparse matrices, though more demanding on the memory subsystem, allow more elegant and effective parallelization across the many dimensions of a GPU kernel, such as across the dimensions of blocks or grids.
Though less well-defined on CPU, in both cases this can result in more effective cache accesses and less memory stalls in the pipeline.

To complete this research, I have collected a large amount of SpMV kernels.
There are kernels using CSR and CSR-k that are written for CPU (using OpenMP and MKL) and GPU (using OpenACC, CUDA, KokkosKernels, and cuSPARSE).

UPDATE: This research is now published in [Parallel Computing](https://www.sciencedirect.com/science/article/pii/S0167819123000030) and [ProQuest](https://www.proquest.com/docview/2741089208/933BA884F54D49DCPQ).

## Directory Format
### acc-spmv-csr
Traditional SpMV kernel using OpenACC parallelization.

### acc-spmv-csrk
CSR-k SpMV kernel using OpenACC parallelization.

### cuda-spmv-csr
Traditional SpMV kernel using hand-tuned CUDA.

### cuda-spmv-csrk
CSR-k SpMV kernel using hand-tuned CUDA.

### spmv-csr
Traditional SpMV kernel using OpenMP parallelization.
Also contains a file to analyze the stats about an .mtx.rcm.csr file.

### spmv-csrk
CSR-k SpMV kernel using OpenMP parallelization.

### helpers
MATLAB/GNU Octave scripts to read in a matrix in MatrixMarket format, permute using RCM, and write out as a CSR file.
Also provides scripts to show runtime parameters for an input matrix.

### cusparse-spmv
CSR SpMV kernel using Nvidia's cuSPARSE library for comparison testing.

### hipsparse-spmv
CSR SpMV kernel using hipSPARSE for comparison testing.

### kokkos-spmv
CSR SpMV kernel using Sandia's KokkosKernels library for comparison testing.

### mkl-spmv
CSR SpMV kernel using Intel's MKL library for comparison testing.

### run_scripts
A set of Python scripts for managing execution and management of the programs in this repository.

### reformat-csr-to-csr3
Reformats a .mtx file to a .mtx.rcm.csr3 file for analysis of the banding algorithm.
Also contains an application for analyzing the stats about an .mtx.rcm.csr3 file.

## Build Notes
The OpenACC versions are compiled using the PGI compiler.
Version 20.1 does not work for `acc-spmv-csrk` due to compiler bugs related to `#pragma acc enter data copyin` directives in class constructors.
Instead, I have been using version 19.5.
This is what I get when I run `pgcc --version`:
```
pgcc 19.5-0 LLVM 64-bit target on x86-64 Linux -tp sandybridge 
PGI Compilers and Tools
Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
```

The CUDA versions and comparison implementations are compiled using the NVHPC compiler.
This is what I get when I run `nvcc --version`:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Wed_Jun__2_19:15:15_PDT_2021
Cuda compilation tools, release 11.4, V11.4.48
Build cuda_11.4.r11.4/compiler.30033411_0
```

The OpenMP versions are compiled with the ICC compiler by Intel when tested on Intel systems.
This is what I get when I run `icc --version`:
```
icc (ICC) 19.1.1.217 20200306
Copyright (C) 1985-2020 Intel Corporation.  All rights reserved.
```

We compile the OpenMP versions with AOCC when tested on AMD systems.
Someone remind me someday to post the `clang++ --version` output on Bridges-2.

Building the CSR-k benchmarks require an install of Boost.
We try to use versions 1.68 or 1.72, as later versions deprecate some headers used.

## Acknowledgements
This work was made possible in part by support from the Alabama Supercomputer Authority, Sandia National Laboratories, and NSF2044633. This work used the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number ACI-1548562.
