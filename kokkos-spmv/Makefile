#
# Makefile for Spmv examples
# Joshua Dennis Booth and Phillip Allen Lane
# Jun 2021
#-----------------


export CC = nvcc
export CFLAGS += -Xcompiler -fopenmp,-Wall,-Wno-sign-compare -O3 -arch=compute_70 -code=sm_70
export CFLAGS += -I/home/uahpal001/kokkos/kokkos-build/include,/home/uahpal001/kokkos/kokkos-kernels-build/include
export CFLAGS += -lm -L/home/uahpal001/kokkos/kokkos-build/lib64,/home/uahpal001/kokkos/kokkos-kernels-build/lib64 -lkokkoskernels -lkokkoscore -lcublas -lcusparse

all: spmv stats

spmv.o: spmv.cu spmv.h
	$(CC)  $(CFLAGS) -c spmv.cu

spmv: spmv.o
	$(CC)  $(CFLAGS) -o spmv.exe $^

stats: stats.c spmv.h
	$(CC)  -o stats.exe stats.c

clean:
	rm *.o; rm spmv.exe; rm stats.exe

