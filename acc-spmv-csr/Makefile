#
# Makefile for Spmv examples
# Joshua Dennis Booth
# Jun 2017
#-----------------


export CC = pgc++
export CFLAGS = -O3 -Minfo=accel -acc -mp -g
export LIBS = -lgomp -lm

all: spmv stats

myTime.o: myTime.c myTime.h
	$(CC)  $(CFLAGS) -c myTime.c $(OMPLIB)

spmv.o: spmv.c spmv.h
	$(CC)  $(CFLAGS) -c spmv.c $(OMPLIB)

spmv: spmv.o myTime.o  
	$(CC)  $(CFLAGS) -o spmv.exe $^ $(OMPLIB)

stats: stats.c spmv.h
	$(CC)  -o stats.exe stats.c $(OMPLIB)

clean:
	rm *.o; rm spmv.exe; rm stats.exe

