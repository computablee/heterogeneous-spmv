#
# Makefile for Spmv examples
# Joshua Dennis Booth and Phillip Allen Lane
# Jun 2021
#-----------------

export CC = hipcc
#export CC = gcc
export CFLAGS += -Wall -O3 --offload-arch=gfx908
export LFLAGS = -L. -lhipsparse

all: spmv

spmv.o: spmv.cu spmv.h
	$(CC)  $(CFLAGS) -c spmv.cu

spmv: spmv.o
	cp $(ROCM_ROOT)/hipsparse/lib/libhipsparse.so.0 ./libhipsparse.so && \
	$(CC)  $(CFLAGS) $(LFLAGS) -o spmv.exe $^

stats: stats.c spmv.h
	$(CC)  -o stats.exe stats.c

clean:
	rm -f *.o spmv.exe stats.exe *.so

