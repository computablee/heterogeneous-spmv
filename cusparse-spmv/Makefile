#
# Makefile for Spmv examples
# Joshua Dennis Booth and Phillip Allen Lane
# Jun 2021
#-----------------


export CC = nvcc
#export CC = gcc
export CFLAGS += -Xcompiler -fopenmp,-Wall -std=c++11 -O3
export CFLAGS += -lm -lcusparse

all: spmv stats

spmv.o: spmv.cu spmv.h
	$(CC)  $(CFLAGS) -c spmv.cu

spmv: spmv.o
	$(CC)  $(CFLAGS) -o spmv.exe $^

stats: stats.c spmv.h
	$(CC)  -o stats.exe stats.c

clean:
	rm *.o; rm spmv.exe; rm stats.exe

