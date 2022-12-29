#!/bin/bash
module load boost/1.68.0
module load nvhpc/21.5
module load cuda/11.7.0

#mkdir /scratch-local/lane-matrices
#cp -r ~/matrices/csr-bandk /scratch-local/lane-matrices

python3 run_kokkos.py
