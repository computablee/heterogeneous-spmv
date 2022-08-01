#!/bin/bash

/home/uahpal001/cmake-3.21.2-linux-x86_64/bin/cmake .. \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DCMAKE_INSTALL_PREFIX=/home/uahpal001/kokkos-build \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_VOLTA70=ON
