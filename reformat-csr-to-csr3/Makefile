#CC = icc
CC = g++
#CC = clang++

CFLAGS = -O3 -std=c++11 -fopenmp

all: spmv stats spmv-auto

spmv: spmv.cpp csrk.cpp csrk.h
	$(CC) $(CFLAGS) -std=c++11 spmv.cpp csrk.cpp -o ./reformat

spmv-auto: spmv-auto.cpp csrk.cpp csrk.h
	$(CC) $(CFLAGS) -std=c++11 spmv-auto.cpp csrk.cpp -o ./reformat-auto

stats: stats.c myTime.h myTime.c
	$(CC) $(CFLAGS) stats.c myTime.c -o ./stats


clean:
	rm -f ./reformat ./reformat-auto ./stats *.o


