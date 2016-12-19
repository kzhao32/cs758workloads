ZSIM_INCLUDE_DIR := $(zsim)/misc/hooks
CCFLAGS = -O3 -Wall -I $(ZSIM_INCLUDE_DIR)
CC = gcc

default: all

.PHONY: all
all: histogram_omp

histogram_omp: histogram.o 
	$(CXX) -o $@ $^ -lpthread

%.o: %.c
	$(CC) $(CCFLAGS) -c -o $@ $<

clean:
	-rm -f *.o
	-rm -f histogram_omp

srcclean:
	-rm -f *~
