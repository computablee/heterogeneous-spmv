#!/bin/bash
#SBATCH -N 1
##SBATCH -n 80
##SBATCH -p icx-normal
#SBATCH -t 00:10:00

module load intel/oneAPI/hpc-toolkit/2022.1.2

lscpu > ../info_cpu_omp.txt
python3 -u run_norm.py ~
