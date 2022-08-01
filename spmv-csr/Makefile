#
# Makefile for Spmv examples
# Joshua Dennis Booth
# Jun 2017
#-----------------


export CC = icc
#export CC = armclang -march=armv8.2-a -mcpu=a64fx
#export CFLAGS += -I$(INCDIR)
CFLAGS = -O3 -fopenmp

LIBS = -lgomp -lm

spmv: spmv.c spmv.h myTime.c stats
	$(CC) $(CFLAGS) -march=native -o spmv.exe spmv.c myTime.c

spmv-icx: spmv.c spmv.h myTime.c stats
	$(CC) $(CFLAGS) -march=icelake-server -ipo -o spmv.exe spmv.c myTime.c

spmv-icx-asm: spmv.c spmv.h myTime.c stats
	$(CC) $(CFLAGS) -march=icelake-server -ipo -masm=intel -S -o spmv.s spmv.c

spmv-epyc: spmv.c spmv.h myTime.c stats
	clang $(CFLAGS) -march=znver2 -o spmv.exe spmv.c myTime.c $(OMPLIB)

spmv-skylake: spmv.c spmv.h myTime.c stats
	$(CC) $(CFLAGS) -march=skylake-avx512 -ipo spmv.c myTime.c -o spmv.exe

stats: stats.c spmv.h
	gcc -o stats.exe stats.c $(OMPLIB)

clean:
	rm -f *.o rm spmv.exe rm stats.exe

