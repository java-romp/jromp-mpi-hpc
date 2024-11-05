#!/bin/bash

# This script is used to prepare the libraries for the build process.
# It builds them and copies the necessary files to the correct locations.
# The script must be run from the root of the project.

CURRENT_DIR=$PWD
icelake_modules_path=/soft/calendula/icelake/rocky8/EB/software

# Build openmpi
cd 3rd-party/openmpi-5.0.5 &&
  echo "Configuring Open MPI" &&
  ./configure \
    --prefix="$CURRENT_DIR"/libs/ompi \
    --disable-mpi-fortran \
    --enable-mpi-java \
    --with-libevent=$icelake_modules_path/libevent/2.1.12-GCCcore-11.2.0 \
    --with-hwloc=$icelake_modules_path/hwloc/2.5.0-GCCcore-11.2.0 \
    --with-pmix=internal \
    --with-prrte=internal \
    --with-libfabric=$icelake_modules_path/libfabric/1.13.2-GCCcore-11.2.0 \
    --with-ucx=$icelake_modules_path/UCX/1.11.2-GCCcore-11.2.0 &&
  echo "Building Open MPI" &&
  make -j 16 all &&
  echo "Installing Open MPI" &&
  make install &&
  echo "Open MPI built and installed"
