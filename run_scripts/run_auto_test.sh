#!/bin/bash
module load boost
module load nvhpc/2020_20.5
module load cuda/10.2.89

for filename in ~/matrices/norm/*.mtx.csr
do
	echo "$filename"
	~/gpu-spmv-csr/cuda-spmv-csrk/spmv-auto $filename 10
done

#~/gpu-spmv-csr/cuda-spmv-csrk/spmv-auto ~/matrices/norm/bmwcra_1.mtx.csr 20
#~/gpu-spmv-csr/cuda-spmv-csrk/spmv-auto ~/matrices/norm/bmwcra_1.mtx.csr 20
#~/gpu-spmv-csr/cuda-spmv-csrk/spmv-auto ~/matrices/norm/bmwcra_1.mtx.csr 20
