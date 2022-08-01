#!/bin/bash
module load nvhpc/21.5
module load cuda/11.4.0

mkdir /scratch-local/lane-matrices
cp -r ~/matrices/csr-bandk /scratch-local/lane-matrices

lspci -vnn > ../info_gpu_kokkos.txt
lscpu >> ../info_gpu_kokkos.txt
python3 run_kokkos.py
