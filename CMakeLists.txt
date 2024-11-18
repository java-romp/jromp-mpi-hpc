#
# This file is only used for adding support for development with CLion
# with syntax highlighting and code completion, not meant for building
# the project. For building the project, use other build scripts (if any).
#
cmake_minimum_required(VERSION 3.24)
project(jromp_mpi_hpc C)

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
add_executable(mandelbrot_parallel ${EXAMPLES_DIR}/mandelbrot/mandelbrot_parallel.c)
add_executable(mandelbrot_parallel_with_openmp ${EXAMPLES_DIR}/mandelbrot/mandelbrot_parallel_with_openmp.c)
add_executable(genome ${EXAMPLES_DIR}/genome/genome.c
        ${EXAMPLES_DIR}/genome/genome.h
        ${EXAMPLES_DIR}/genome/cvector.h
        ${EXAMPLES_DIR}/genome/cvector_utils.h
        ${EXAMPLES_DIR}/genome/main_genome.c)
add_executable(jacobi ${EXAMPLES_DIR}/jacobi/jacobi_openmp.c)
add_executable(molecular_dynamics ${EXAMPLES_DIR}/molecular_dynamics/md_openmp.c)
add_executable(fft ${EXAMPLES_DIR}/fft/fft_openmp.c)
add_executable(sgefa ${EXAMPLES_DIR}/sgefa/sgefa_openmp.c)
add_executable(ziggurat ${EXAMPLES_DIR}/ziggurat/ziggurat_openmp.c)
add_executable(heated_plate ${EXAMPLES_DIR}/heated_plate/heated_plate_openmp.c
        ${EXAMPLES_DIR}/heated_plate/heated.c)
add_executable(fractals ${EXAMPLES_DIR}/fractals/mandelbrot.h
        ${EXAMPLES_DIR}/fractals/mandelbrot.c)
add_executable(big_multiplication ${EXAMPLES_DIR}/big_multiplication/big_multiplication.c
        ${EXAMPLES_DIR}/big_multiplication/big_multiplication.h)

target_link_libraries(genome PUBLIC MPI::MPI_C OpenMP::OpenMP_C)
target_link_libraries(mandelbrot_parallel PUBLIC MPI::MPI_C)
target_link_libraries(mandelbrot_parallel_with_openmp PUBLIC MPI::MPI_C OpenMP::OpenMP_C)
target_link_libraries(full_parallel PUBLIC MPI::MPI_C OpenMP::OpenMP_C)
target_link_libraries(jacobi PUBLIC MPI::MPI_C OpenMP::OpenMP_C m)
target_link_libraries(fft PUBLIC MPI::MPI_C OpenMP::OpenMP_C m fftw3)
target_link_libraries(sgefa PUBLIC MPI::MPI_C OpenMP::OpenMP_C m)
target_link_libraries(ziggurat PUBLIC MPI::MPI_C OpenMP::OpenMP_C m)
target_link_libraries(heated_plate PUBLIC MPI::MPI_C OpenMP::OpenMP_C m)
target_link_libraries(fractals PUBLIC MPI::MPI_C OpenMP::OpenMP_C m)
target_link_libraries(big_multiplication PUBLIC MPI::MPI_C OpenMP::OpenMP_C m)
