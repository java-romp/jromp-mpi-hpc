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

add_executable(all_c
        src/main/c/jromp/mpi/examples/blocking.c
        src/main/c/jromp/mpi/examples/burro.c
        src/main/c/jromp/mpi/examples/cross.c
        src/main/c/jromp/mpi/examples/full_parallel.c
        src/main/c/jromp/mpi/examples/mandelbrot/mandelbrot_parallel.c
        src/main/c/jromp/mpi/examples/mandelbrot/mandelbrot_parallel_with_openmp.c)
