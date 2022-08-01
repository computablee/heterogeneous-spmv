#!/bin/bash
module load boost
module load nvhpc/2020_20.5
module load cuda/11.4.0

#mkdir /scratch-local/lane-matrices
#mkdir /scratch-local/lane-matrices/norm
#cp -r ~/matrices/norm /scratch-local/lane-matrices

lspci -vnn > ../info_gpu_cuda.txt
lscpu >> ../info_gpu_cuda.txt
python3 run_cuda.py
