MPI_PATH=../../../../../../../libs/ompi
MPI_INCLUDE=$(MPI_PATH)/include
MPI_LIB=$(MPI_PATH)/lib
MPI_BIN=$(MPI_PATH)/bin

CC=$(MPI_BIN)/mpicc
CFLAGS=-I$(MPI_INCLUDE) -Wall
LDFLAGS=-L$(MPI_LIB) -lmpi -lm

# Use env variable or default value
TASKS=$(or $(MPI_TASKS), 4)
ARGS=/home/samuel/IdeaProjects/IntelliJ/jromp-mpi-hpc/data/dna

all: genome

genome: genome.c
	@$(CC) $(CFLAGS) -o $@.o $< $(LDFLAGS)

run: genome
	@$(MPI_BIN)/mpirun -np $(TASKS) ./genome.o $(ARGS)

clean:
	@rm -f genome

.PHONY: all clean
