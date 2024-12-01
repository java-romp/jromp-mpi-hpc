#!/bin/bash

PROJECT_BASE_PATH=../../../../../../..
MPI_PATH=$PROJECT_BASE_PATH/libs/ompi
MPI_INCLUDE=$MPI_PATH/include
MPI_LIB=$MPI_PATH/lib
MPI_BIN=$MPI_PATH/bin

CC="$MPI_BIN/mpicc"
CFLAGS="-I$MPI_INCLUDE -Wall -fopenmp"
LDFLAGS="-L$MPI_LIB"

# Compile the code
# Parameters:
#   1: Optimization level
function compile {
  $CC $CFLAGS -o big_multiplication.o big_multiplication.c -O"$1" $LDFLAGS
}

# Clean the code
# Parameters:
#   None
function clean {
  rm -f big_multiplication.o
}

# Run the code
# Parameters:
#   1: Number of ranks
#   2: Matrix size
#   3: Number of threads
#   4: Optimization level
function run {
  $MPI_BIN/mpirun --bind-to none -np "$1" ./big_multiplication.o "$2" "$3" "$4"
}

function main {
  local ranks=5
  local matrix_size=2000
  local threads=4
  local optimization_level=3

  clean
  compile $optimization_level
  run $ranks $matrix_size $threads $optimization_level
}

main
