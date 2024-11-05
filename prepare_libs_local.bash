#!/bin/bash

# This script is used to prepare the libraries for the build process.
# It builds them and copies the necessary files to the correct locations.
# The script must be run from the root of the project.

CURRENT_DIR=$PWD

# Build openmpi
cd 3rd-party/openmpi-5.0.5 &&
  echo "Configuring Open MPI" &&
  ./configure \
    --prefix="$CURRENT_DIR"/libs/ompi \
    --disable-mpi-fortran \
    --enable-mpi-java \
    --with-libevent=internal \
    --with-hwloc=internal \
    --with-pmix=internal \
    --with-prrte=internal &&
  echo "Building Open MPI" &&
  make -j 8 all &&
  echo "Installing Open MPI" &&
  make install &&
  echo "Open MPI built and installed"
