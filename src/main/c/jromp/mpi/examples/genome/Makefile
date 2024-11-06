MPI_PATH=../../../../../../../libs/ompi
MPI_INCLUDE=$(MPI_PATH)/include
MPI_LIB=$(MPI_PATH)/lib
MPI_BIN=$(MPI_PATH)/bin

OPTIMIZATION_LEVEL=3

CC=$(MPI_BIN)/mpicc
CFLAGS=-I$(MPI_INCLUDE) -Wall -fopenmp
LDFLAGS=-L$(MPI_LIB) -lmpi -lm

# Use env variable or default value
TASKS=$(or $(MPI_TASKS), 6)
DATA_DIR=/home/samuel/IdeaProjects/IntelliJ/jromp-mpi-hpc/data/dna

all: genome.o main_genome.o
	@$(CC) $(CFLAGS) -o genome.out genome.o main_genome.o $(LDFLAGS)

genome.o: genome.c genome.h
	@$(CC) $(CFLAGS) -c genome.c

main_genome.o: main_genome.c genome.h
	@$(CC) $(CFLAGS) -c main_genome.c

run: all
	@$(MPI_BIN)/mpirun -np $(TASKS) ./genome.out $(DATA_DIR)

clean:
	@rm -f genome.o main_genome.o genome.out

.PHONY: all run clean
