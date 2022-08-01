#CC = icc
#CC = g++
CC = nvcc

export CFLAGS = -std=c++11 -O4
export CFLAGS += -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80
export CFLAGS += -Xcompiler -fopenmp,-Wall
export CFLAGS += -Xptxas -warn-lmem-usage,-warn-spills,-warn-double-usage,-v
export CFLAGS += -lm -I${BOOST_ROOT}/include -I${BOOST_DIR}/include

all: spmv spmv-auto-volta spmv-auto-ampere

spmv.o: spmv.cu csrk.cuh
	$(CC) $(CFLAGS) spmv.cu -c -o spmv.o

spmv-auto-volta.o: spmv-auto-volta.cu csrk.cuh
	$(CC) $(CFLAGS) spmv-auto-volta.cu -c -o spmv-auto-volta.o

spmv-auto-ampere.o: spmv-auto-ampere.cu csrk.cuh
	$(CC) $(CFLAGS) spmv-auto-ampere.cu -c -o spmv-auto-ampere.o

csrk.o: csrk.cu csrk.cuh
	$(CC) $(CFLAGS) csrk.cu -c -o csrk.o

spmv: spmv.o csrk.o
	$(CC) $(CFLAGS) spmv.o csrk.o -o ./spmv

spmv-auto-volta: spmv-auto-volta.o csrk.o
	$(CC) $(CFLAGS) spmv-auto-volta.o csrk.o -o ./spmv-auto-volta

spmv-auto-ampere: spmv-auto-ampere.o csrk.o
	$(CC) $(CFLAGS) spmv-auto-ampere.o csrk.o -o ./spmv-auto-ampere

clean:
	rm -f ./spmv *.o ./spmv-auto-volta ./spmv-auto-ampere


