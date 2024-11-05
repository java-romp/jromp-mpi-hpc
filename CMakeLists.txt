#
# This file is only used for adding support for development with CLion
# with syntax highlighting and code completion, not meant for building
# the project. For building the project, use other build scripts (if any).
#
cmake_minimum_required(VERSION 3.24)
project(jromp_mpi_hpc C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP REQUIRED)

set(MPI_HOME ${CMAKE_SOURCE_DIR}/libs/ompi)
include_directories(${MPI_HOME}/include)

set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/src/main/c/jromp/mpi/examples)

add_executable(blocking ${EXAMPLES_DIR}/blocking.c)
add_executable(burro ${EXAMPLES_DIR}/burro.c)
add_executable(cross ${EXAMPLES_DIR}/cross.c)
add_executable(full_parallel ${EXAMPLES_DIR}/full_parallel.c)
add_executable(mandelbrot_parallel ${EXAMPLES_DIR}/mandelbrot/mandelbrot_parallel.c)
add_executable(mandelbrot_parallel_with_openmp ${EXAMPLES_DIR}/mandelbrot/mandelbrot_parallel_with_openmp.c)
add_executable(genome ${EXAMPLES_DIR}/genome/genome.c
        ${EXAMPLES_DIR}/genome/genome.h
        ${EXAMPLES_DIR}/genome/cvector.h
        ${EXAMPLES_DIR}/genome/cvector_utils.h)
