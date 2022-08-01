#!/bin/bash
module load boost
module load pgi/19.5

lspci -vnn > ../info_gpu_acc.txt
lscpu >> ../info_gpu_acc.txt
python3 run_gpu.py
