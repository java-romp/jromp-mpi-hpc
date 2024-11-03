#
# This file is only used for adding support for development with CLion
# with syntax highlighting and code completion, not meant for building
# the project. For building the project, use other build scripts (if any).
#
cmake_minimum_required(VERSION 3.24)
project(jromp_mpi_hpc_calendula C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP REQUIRED)

set(MPI_HOME ${CMAKE_SOURCE_DIR}/libs/ompi)
include_directories(${MPI_HOME}/include)

function(add_exe name path)
    add_executable(${name} src/main/c/jromp/mpi/examples/${path})
endfunction()

add_exe(blocking blocking.c)
add_exe(burro burro.c)
add_exe(cross cross.c)
add_exe(full_parallel full_parallel.c)
add_exe(mandelbrot_parallel mandelbrot/mandelbrot_parallel.c)
