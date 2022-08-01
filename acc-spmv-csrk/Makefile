#CC = icc
#CC = g++
CC = pgc++

CFLAGS = -O3 -std=c++11 -Minfo=accel -acc -mp -g

CSRKINCPATH=.

all: spmv

csrk.o: csrk.cpp csrk.h
	$(CC) $(CFLAGS) -c csrk.cpp -o csrk.o

spmv.o: spmv.cpp csrk.h
	$(CC) $(CFLAGS) -c spmv.cpp -o spmv.o

spmv: spmv.o csrk.o
	$(CC) $(CFLAGS) spmv.o csrk.o -o spmv


clean:
	rm -f ./spmv *.o


