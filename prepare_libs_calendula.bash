#!/bin/bash

# This script is used to prepare the libraries for the build process.
# It builds them and copies the necessary files to the correct locations.
# The script must be run from the root of the project.

# Command to create a node allocation to run this script:
# salloc --account=ule_formacion_9 --partition=formacion --qos=formacion --time=00:25:00 --cpus-per-task=32 --mem-per-cpu=1500 --nodelist=cn6001

CURRENT_DIR=$PWD

# Build ompi
cd 3rd-party/ompi &&
  echo "Building Open MPI" &&
  ./autogen.pl &&
  echo "Configuring Open MPI" &&
  ./configure \
    --prefix="$CURRENT_DIR"/libs/ompi \
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
