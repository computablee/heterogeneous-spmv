#!/bin/bash
#BSUB -n 1
#BSUB -q rhel7W
#BSUB -o weaverjob.txt
#BSUB -e weaverjob.err
#BSUB -J cusparse_test

#module load boost/1.75.0/gcc/7.2.0
#module load cuda/10.1.243

#mkdir /scratch-local/lane-matrices
#cp -r ~/matrices/new_rcm /scratch-local/lane-matrices

module load boost/1.68.0
module load nvhpc/21.5
module load cuda/11.4.0

cp -r ~/matrices/new_rcm /scratch-local/lane-matrices

python3 run_cusparse.py /scratch-local/lane-matrices
