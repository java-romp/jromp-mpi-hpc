#!/bin/bash

# This script is used to prepare the libraries for the build process.
# The script must be run from the root of the project.

module purge
module use /soft/calendula/icelake/rocky8/EB/modules/all/
module use /soft/calendula/icelake/rocky8/modules/
module load OpenMPI/4.1.1-GCC-11.2.0
module load Java/21.0.2
module load Maven/3.9.7
module load gradle_8.10.2
module load CUDA/12.1.1
module load Autoconf/2.71-GCCcore-11.2.0
module load Automake/1.16.4-GCCcore-11.2.0
module load libtool/2.4.6-GCCcore-11.2.0
module load flex/2.6.4-GCCcore-11.2.0

# [ule_formacion_9_4@frontend2 jromp-mpi-hpc-calendula]$ module use /soft/calendula/icelake/rocky8/EB/modules/all/
# [ule_formacion_9_4@frontend2 jromp-mpi-hpc-calendula]$ module use /soft/calendula/icelake/rocky8/modules/
# [ule_formacion_9_4@frontend2 jromp-mpi-hpc-calendula]$ module load Java/21.0.2
# [ule_formacion_9_4@frontend2 jromp-mpi-hpc-calendula]$ module load icelake/gcc_9.4.0
# [ule_formacion_9_4@frontend2 jromp-mpi-hpc-calendula]$ module list
# Currently Loaded Modulefiles:
#  1) Java/21.0.2(21)   2) icelake/gcc_9.4.0

# Open MPI built and installed
#
# real    12m21.631s
# user    9m47.786s
# sys     3m38.129s
# [ule_formacion_9_4@cn6001 jromp-mpi-hpc-calendula]$

# Nota: he cargado el icelake porque lo compilo y ejecuto en los nodos 6xxx.
