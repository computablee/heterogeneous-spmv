#
# Makefile for Spmv examples
# Joshua Dennis Booth and Phillip Allen Lane
# Jun 2021
#-----------------


export CC = icc
export CFLAGS = -fopenmp -O3 -std=gnu++11 -mkl

all: spmv stats

spmv.o: spmv.cpp spmv.h
	$(CC)  -c spmv.cpp $(CFLAGS)

spmv: spmv.o
	$(CC)  -o spmv.exe $< $(CFLAGS)

spmv-knl: spmv.cpp spmv.h
	$(CC) -march=knl -o spmv.exe $^ $(CFLAGS)

spmv-icx: spmv.cpp spmv.h
	$(CC) -march=icelake-server -o spmv.exe $^ $(CFLAGS)

spmv-skylake: spmv.cpp spmv.h
	$(CC) -march=skylake-avx512 -o spmv.exe $^ $(CFLAGS)

stats: stats.c spmv.h
	$(CC)  -o stats.exe stats.c

clean:
	rm -f *.o; rm -f spmv.exe; rm -f stats.exe

