#CC = icc
#CC = g++
#CC = nvcc
CC = hipcc

export CFLAGS = -std=c++11 -O3 --offload-arch=gfx908
export CFLAGS += -Wall -I${BOOST_ROOT}/include

all: spmv spmv-auto-mi100

spmv.o: spmv.cu csrk.cuh
	$(CC) $(CFLAGS) spmv.cu -c -o spmv.o

spmv-auto-mi100.o: spmv-auto-mi100.cu csrk.cuh
	$(CC) $(CFLAGS) spmv-auto-mi100.cu -c -o spmv-auto-mi100.o

csrk.o: csrk.cu csrk.cuh
	$(CC) $(CFLAGS) csrk.cu -c -o csrk.o

spmv: spmv.o csrk.o
	$(CC) $(CFLAGS) spmv.o csrk.o -o ./spmv

spmv-auto-mi100: spmv-auto-mi100.o csrk.o
	$(CC) $(CFLAGS) spmv-auto-mi100.o csrk.o -o ./spmv-auto-mi100

clean:
	rm -f ./spmv *.o ./spmv-auto-volta ./spmv-auto-ampere


