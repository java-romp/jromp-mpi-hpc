#!/bin/bash

# This script is used to prepare the libraries for the build process.
# The script must be run from the root of the project.

module purge
# los centos van del 1-8, 9-26 son rocky (25 y 26 están inactivos)
module use /soft/calendula/cascadelake/centos7/EB/modules/all/
module use /soft/calendula/cascadelake/centos7/modules/
module use /soft/calendula/icelake/rocky8/EB/modules/all/
module use /soft/calendula/icelake/rocky8/modules/

# Compiling Open MPI in Rocky 8
module use /soft/calendula/icelake/rocky8/EB/modules/all/ &&
  module use /soft/calendula/icelake/rocky8/modules/ &&
  module load Java/21.0.2 &&
  module load gradle_8.10.2 &&
  module load GCC/11.2.0 &&
  module load GCCcore/11.2.0 &&
  module load Autoconf/2.71-GCCcore-11.2.0 &&
  module load Automake/1.16.4-GCCcore-11.2.0 &&
  module load libtool/2.4.6-GCCcore-11.2.0 &&
  module load flex/2.6.4-GCCcore-11.2.0 &&
  module load libevent/2.1.12-GCCcore-11.2.0 &&
  module load hwloc/2.5.0-GCCcore-11.2.0 &&
  module load libfabric/1.13.2-GCCcore-11.2.0 &&
  module load UCX/1.11.2-GCCcore-11.2.0

module load Maven/3.9.7

# Use Java and Gradle
module use /soft/calendula/icelake/rocky8/EB/modules/all/ /soft/calendula/icelake/rocky8/modules/ && module load Java/21.0.2 gradle_8.10.2

# Con la 10.3 se podría llegar a compilar, todas las deps están para esa versión
salloc --account=ule_formacion_9 --partition=formacion --qos=formacion --time=00:25:00 --cpus-per-task=8 --nodelist=cn6010
time ./prepare_libs_calendula.bash

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
