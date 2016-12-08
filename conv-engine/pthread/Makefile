PWD=$(shell pwd)
TEST=$(shell basename $(PWD))
ARGS=

CC=gcc
OPT?=-O3
FF?=-DFF

NO_MCPAT?=--no-mcpat

CFLAGS+=${FF} ${OPT} 

all: $(TEST) 

$(TEST): ${TEST}.c ${TEST}.h
	$(CC) ${CFLAGS} $< -o $@ -lpthread --static -lm

clean: 
	rm -rf $(TEST)
	rm -rf *.out *.txt .txt

