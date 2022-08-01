#!/bin/bash
#BSUB -n 1
#BSUB -q rhel7W
#BSUB -o weaverjob.txt
#BSUB -e weaverjob.err
#BSUB -J volta_test

#module load boost/1.75.0/gcc/7.2.0
#module load cuda/10.1.243

module load boost/1.68.0 nvhpc/2020_20.5 cuda/11.4.0

#mkdir /scratch-local/lane-matrices
#cp -r ~/matrices /scratch-local/lane-matrices

python3 run_cuda_new.py /scratch-local/lane-matrices