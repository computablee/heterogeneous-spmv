#!/bin/bash
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -p RM
#SBATCH -t 00:10:00

lscpu > ../info_cpu_omp.txt
python3 -u run_norm_epyc.py $PROJECT
