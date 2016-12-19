MPU_LIBRARY_DIR :=  $(MPU_ROOT)/libmpu
MPU_INCLUDE_DIR := $(MPU_LIBRARY_DIR)/include
ZSIM_INCLUDE_DIR := $(zsim)/misc/hooks
CCFLAGS = -g -O3 -Wall -fno-tree-vectorize -I $(MPU_INCLUDE_DIR) -I $(ZSIM_INCLUDE_DIR)
LDFLAGS := -Wl,-rpath,$(MPU_LIBRARY_DIR)
CC = gcc

PROGS := histogram-ref histogram-pthread histogram-pthread-zsim

.PHONY: default all clean

all: $(PROGS)

OBJ = histogram.o 

histogram-ref: $(OBJ)
	$(CC) -o $@ $^ $(CCFLAGS) $(LDFLAGS) -L $(MPU_LIBRARY_DIR) -lmpu_ref

histogram-pthread: $(OBJ)
	$(CC) -o $@ $^ $(CCFLAGS) $(LDFLAGS) -L $(MPU_LIBRARY_DIR) -lmpu_pthread

histogram-pthread-zsim: $(OBJ)
	$(CC) -o $@ $^ $(CCFLAGS) $(LDFLAGS) -L $(MPU_LIBRARY_DIR) -lmpu_pthread_zsim

%.o: %.c
	$(CC) -c -o $@ $< $(CCFLAGS)

clean:
	-rm -f *.o
	-rm -f histogram-ref
	-rm -f histogram-pthread
	-rm -f histogram-pthread-zsim

srcclean:
	-rm -f *~
