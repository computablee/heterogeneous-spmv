#!/bin/bash
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -p RM
#SBATCH -t 4:00:00

cp -r $PROJECT/matrices $RAMDISK

module load aocc
#module load gcc/7.1.0
#module load boost/1.64
python3 run_epyc.py $RAMDISK