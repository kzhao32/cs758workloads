ifndef RISCV
$(error RISCV is undefined)
endif

PWD=$(shell pwd)
#TEST=$(shell basename $(PWD))
TEST=sift_dog
ARGS=

KERN_CPU=$(TEST)
KERN_SB=$(TEST)_sb
#ALG=stencil2d

CFLAGS?=-O3 -Wall -Wno-unused-label 

CPU_SRCS=$(KERN_CPU).c
CPU_FILES=$(CPU_SRCS) $(KERN_CPU).h sim.h

##SB_Related
SB_SRCS=$(KERN_SB).c 
SB_FILES=$(SB_SRCS) $(KERN_CPU).h sim.h

SB_CPP=riscv64-unknown-elf-g++
SB_CONFIG=$(RISCV)/configs/diannao_simd64.sbmodel

DFGS=$(KERN_SB).dfg
DFG_HEADERS=$(DFGS:.dfg=.h)

$(DFG_HEADERS): %.h: %.dfg
	$(RISCV)/bin/sb_sched $(SB_CONFIG) $<

$(KERN_SB): $(SB_FILES) $(DFG_HEADERS) 
		$(SB_CPP) $(CFLAGS) -I../../common -o $(KERN_SB) $(SB_SRCS) --static -lm 

run_sb: $(KERN_SB) 
	SBCONFIG=$(SB_CONFIG) spike  --ic=128:4:64 --dc=128:4:64 --l2=1024:8:64  --extension=softbrain $(RISCV)/riscv64-unknown-elf/bin/pk  ./$(KERN_SB)

$(KERN_CPU): $(CPU_FILES) 
		$(CC) $(CFLAGS) -I../../common -o $(KERN_CPU) $(CPU_SRCS) --static -lm

run: $(KERN_CPU) 
	./$(KERN_CPU)

clean:
	rm -f $(KERN_CPU) 
	rm -f $(KERN_SB) $(DFG_HEADERS) *.s *.o
	rm -rf dots/ gams/ $(KERN_SB)-0.cfg
