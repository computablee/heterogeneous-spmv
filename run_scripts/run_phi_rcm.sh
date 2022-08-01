#!/bin/bash
#SBATCH -N 1
#SBATCH -n 68
#SBATCH -p flat-quadrant
#SBATCH -t 48:00:00

module load intel/19.1.1  python3/3.7.0
module load boost/1.72
#module load gcc/7.1.0
#module load boost/1.64
python3 run_phi.py rcm
