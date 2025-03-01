#
# This file is only used for adding support for development with CLion
# with syntax highlighting and code completion, not meant for building
# the project. For building the project, use other build scripts (if any).
#
cmake_minimum_required(VERSION 3.24)
project(jromp_mpi_hpc C CUDA)

set(CMAKE_C_STANDARD 17)

find_package(OpenMP REQUIRED)

set(MPI_HOME ${CMAKE_SOURCE_DIR}/libs/ompi)
set(CMAKE_PREFIX_PATH ${MPI_HOME})
find_package(MPI REQUIRED)

set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/src/main/c/jromp/mpi/examples)

add_executable(blocking ${EXAMPLES_DIR}/blocking.c)
add_executable(burro ${EXAMPLES_DIR}/burro.c)
add_executable(cross ${EXAMPLES_DIR}/cross.c)
add_executable(full_parallel ${EXAMPLES_DIR}/full_parallel.c)
add_executable(genome ${EXAMPLES_DIR}/genome/genome.c
        ${EXAMPLES_DIR}/genome/genome.h
        ${EXAMPLES_DIR}/genome/cvector.h
        ${EXAMPLES_DIR}/genome/cvector_utils.h
        ${EXAMPLES_DIR}/genome/main_genome.c)
add_executable(fractals ${EXAMPLES_DIR}/fractals/mandelbrot.h
        ${EXAMPLES_DIR}/fractals/mandelbrot.c)
add_executable(gemm ${EXAMPLES_DIR}/gemm/gemm.c
        ${EXAMPLES_DIR}/gemm/gemm.h)
add_executable(gemm_cuda ${EXAMPLES_DIR}/gemm/gemm.cu
        ${EXAMPLES_DIR}/gemm/gemm.cuh)

target_link_libraries(genome PUBLIC MPI::MPI_C OpenMP::OpenMP_C)
target_link_libraries(full_parallel PUBLIC MPI::MPI_C OpenMP::OpenMP_C)
target_link_libraries(fractals PUBLIC MPI::MPI_C OpenMP::OpenMP_C m)
target_link_libraries(gemm PUBLIC MPI::MPI_C OpenMP::OpenMP_C m)
target_link_libraries(gemm_cuda PUBLIC MPI::MPI_C OpenMP::OpenMP_C m cublas)
