#!/bin/bash

MPI_DIR=../../../../../../../libs/ompi
MPIEXEC=$MPI_DIR/bin/mpiexec
MPICC=$MPI_DIR/bin/mpicc
DEBUG_FLAGS="-g"
OPTIMIZATION_FLAGS="-O3"
CFLAGS="-Wall -pedantic -std=c99 -fopenmp"
LFLAGS="-lm"

VALGRIND=valgrind
VALGRIND_FLAGS="--tool=callgrind --dump-instr=yes --collect-jumps=yes --callgrind-out-file=results_callgrind/callgrind.out.%p"

function clean {
  rm -f mandelbrot.o mandelbrot.bmp
  rm -f results_callgrind/*
}

function compile {
  $MPICC $CFLAGS -o mandelbrot.o mandelbrot.c $LFLAGS
}

function run {
  local processes=5
  local width=2000
  local height=2000
  local iterations=20000
  local block_size=10

  echo "Running with $processes processes, $width x $height resolution, $iterations iterations and $block_size block size"

  $MPIEXEC -np $processes ./mandelbrot.o -n $iterations -c $width -r $height -b $block_size -o mandelbrot.bmp
}

function profile {
  local processes=4
  local width=2048
  local height=2048
  local iterations=6000
  local block_size=$((height / processes))

  echo "Profiling with $processes processes, $width x $height resolution, $iterations iterations and $block_size block size"

  $MPIEXEC -np $processes $VALGRIND $VALGRIND_FLAGS ./mandelbrot.o -n $iterations -a 3.5 -c $width -r $height -b $block_size -o mandelbrot.bmp
}

function usage {
  echo "Usage: $0 [clean|compile|run|profile]"
}

if [ $# -ne 1 ]; then
  usage
  exit 1
fi

case $1 in
  clean)
    clean
    ;;
  compile)
    compile
    ;;
  run)
    run
    ;;
  profile)
    profile
    ;;
  *)
    usage
    exit 1
    ;;
esac

exit 0
