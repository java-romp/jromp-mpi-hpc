#!/bin/bash

# This script is used to prepare the libraries for the build process.
# The script must be run from the root of the project.

module purge &&
  module use /soft/calendula/icelake/rocky8/EB/modules/all/ &&
  module use /soft/calendula/icelake/rocky8/modules/ &&
  module load OpenMPI/4.1.1-GCC-11.2.0 &&
  module load Java/21.0.2 &&
  module load gradle_8.10.2 &&
  module load Autoconf/2.71-GCCcore-11.2.0 &&
  module load Automake/1.16.4-GCCcore-11.2.0 &&
  module load libtool/2.4.6-GCCcore-11.2.0 &&
  module load flex/2.6.4-GCCcore-11.2.0
