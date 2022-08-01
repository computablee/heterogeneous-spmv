#!/bin/bash
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -p RM
#SBATCH -t 2:00:00

cp -r $PROJECT/matrices $RAMDISK

module load intel/20.4
#module load gcc/7.1.0
#module load boost/1.64
python3 run_epyc_mkl.py $RAMDISK
