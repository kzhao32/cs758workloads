#ZSIM_INCLUDE_DIR := $(zsim)/misc/hooks
#CCFLAGS = -O3 -Wall -I $(ZSIM_INCLUDE_DIR)

CCFLAGS = -O3 -Wall -I./include 
CC = gcc

default: all

.PHONY: all
all: ime-sad

ime-sad: ime-sad.o
  $(CXX) -o $@ $^ --static -lpthread -lm

%.o: %.c
  $(CC) $(CCFLAGS) -c -o $@ $<

clean:
  -rm -f *.o
  -rm -f ime-sad
	-rm -rf *.out *.txt

