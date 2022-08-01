#
# Makefile for Spmv examples
# Joshua Dennis Booth and Phillip Allen Lane
# Jun 2021
#-----------------


export CC = nvcc
export CFLAGS += -Xcompiler -fopenmp,-Wall
export CFLAGS += -O3 -arch=compute_70 -code=sm_70
export CFLAGS += -Xptxas -warn-lmem-usage,-warn-spills,-warn-double-usage,-v
export CFLAGS += -lm

all: spmv stats

spmv.o: spmv.cu spmv.h
	$(CC)  $(CFLAGS) -c spmv.cu

spmv: spmv.o
	$(CC)  $(CFLAGS) -o spmv.exe $^

stats: stats.c spmv.h
	$(CC)  -o stats.exe stats.c

clean:
	rm *.o; rm spmv.exe; rm stats.exe

