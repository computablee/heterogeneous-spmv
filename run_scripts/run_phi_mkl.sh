#!/bin/bash
#SBATCH -N 1
#SBATCH -n 68
#SBATCH -p flat-quadrant
#SBATCH -t 6:00:00

cp -r $WORK/matrices $SCRATCH

module load intel/19.1.1
#module load gcc/7.1.0
#module load boost/1.64
python3 run_phi_mkl.py $SCRATCH
