CC = /usr/bin/mpiCC

CCFLAGS = -lpthread
DEBUGFLAGS = -g

BINARIES = ocean_main
SOURCES = ocean_main.c sequential_ocean.c omp_ocean.c pthread_ocean.c mpi_ocean.c

all: $(BINARIES)

$(BINARIES): $(SOURCES)
	$(CC) $(CCFLAGS) $(DEBUGFLAGS) -o $(BINARIES) $(SOURCES)

clean:
	rm -fv $(BINARIES) core* *.o *~
