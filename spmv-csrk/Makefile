CC = icc
#CC = g++
#CC = armclang++ -march=armv8.2-a -mcpu=a64fx
#CC = clang++

#export OMPSCHDIR=/home1/07803/tg871024

CFLAGS = -O3 -std=c++11 -g
PCFLAGS = -I${TACC_BOOST_INC} -L${TACC_BOOST_LIB}
PCFLAGS += -I/home1/07803/tg871024/memkind-1.13.0/include -L/home1/07803/tg871024/memkind-1.13.0/.libs -lmemkind -fopenmp
#PCFLAGS += -Wl,-rpath,"$(OMPSCHDIR)/yan_tune_25_test/mogomp_lib/lib" -fopenmp -lnuma -lm

CSRKINCPATH=.

all: spmv

spmv: spmv.cpp csrk.cpp csrk.h
	$(CC) $(CFLAGS) -fopenmp spmv.cpp csrk.cpp -o ./spmv

spmv-epyc: spmv.cpp csrk.cpp csrk.h
	clang++ $(CFLAGS) -march=znver2 -fopenmp spmv.cpp csrk.cpp -o ./spmv

spmv-knl: spmv.cpp csrk.cpp csrk.h
	$(CC) $(CFLAGS) $(PCFLAGS) -march=knl -ipo spmv.cpp csrk.cpp -o ./spmv

spmv-icx: spmv.cpp csrk.cpp csrk.h
	$(CC) $(CFLAGS) $(PCFLAGS) -march=icelake-server -ipo spmv.cpp csrk.cpp -o ./spmv

asm: csrk.cpp csrk.h
	$(CC) $(CFLAGS) $(PCFLAGS) -march=knl -S -masm=intel csrk.cpp -o ./csrk.s

clean:
	rm -f ./spmv *.o *.s


