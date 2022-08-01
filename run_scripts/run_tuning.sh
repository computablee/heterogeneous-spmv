#!/bin/bash
module load boost/1.68.0
module load nvhpc/2020_20.5
module load cuda/11.4.0

rm -rf /scratch-local/lane-matrices
mkdir /scratch-local/lane-matrices
mkdir /scratch-local/lane-matrices/norm
cp -r ~/matrices/norm/normnew/* /scratch-local/lane-matrices/norm
#mv /scratch-local/lane-matrices/norm/normnew/* /scratch-local/lane-matrices/norm/
#rm -r /scratch-local/lane-matrices/norm/normnew

lspci -vnn > ../info_gpu_cuda.txt
lscpu >> ../info_gpu_cuda.txt
python3 run_tuning.py
