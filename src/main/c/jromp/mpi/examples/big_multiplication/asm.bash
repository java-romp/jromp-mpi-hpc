#!/bin/bash

# This script is used to get the assembly code of the program passed as argument
# The program is compiled with the -S flag to generate the assembly code
# The assembly code is then stored in a file with the same name as the program
# but with the .s extension (and removing the .c extension)

PROJECT_BASE_PATH=../../../../../../..
MPI_PATH=$PROJECT_BASE_PATH/libs/ompi
MPI_INCLUDE=$MPI_PATH/include
MPI_LIB=$MPI_PATH/lib
MPI_BIN=$MPI_PATH/bin

CC="$MPI_BIN/mpicc"
CFLAGS="-I$MPI_INCLUDE -Wall -fopenmp"
LDFLAGS="-L$MPI_LIB"

function assembly {
  $CC $CFLAGS -S "$1" -o "$2" $LDFLAGS
}

function main {
  local program=$1.c
  local asm_file="${program%.*}.s"

  assembly "$program" "$asm_file"
}

main "$@"

