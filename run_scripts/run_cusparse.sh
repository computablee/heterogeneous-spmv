#!/bin/bash
#BSUB -n 1
#BSUB -q rhel7W
#BSUB -o weaverjob.txt
#BSUB -e weaverjob.err
#BSUB -J cusparse_test

module load boost/1.75.0/gcc/7.2.0
module load cuda/10.1.243

#module load boost
#module load nvhpc/2020_20.5
#module load cuda/11.4.0

python3 run_cusparse.py /scratch-local/lane-matrices
