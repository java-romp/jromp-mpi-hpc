#!/bin/bash

# ---------------------------------------------------- #
# --------------- Immutable properties --------------- #
# ---------------------------------------------------- #
#SBATCH --partition=formacion                          # Partition where the Jobs are sent
#SBATCH --account=ule_formacion_9                      # Account to charge the job
#SBATCH --qos=formacion                                # QOS to launch this partition
#SBATCH --time=00:25:00                                # Estimated maximum time for the job
#SBATCH --mail-user=scastd00@estudiantes.unileon.es    # Mail to receive info (start/cancel/end)
#SBATCH --mail-type=FAIL                               # We want to receive all the failures
#SBATCH --mem=0                                        # Memory required (0 = no limit) Node for me

# ---------------------------------------------------- #
# ------- Properties set on run_calendula.bash ------- #
# ---------------------------------------------------- #
# --ntasks           # Number of tasks (processes)
# --ntasks-per-node  # Number of task per node
# --cpus-per-task    # Number of cpus per task
# --nodes            # Number of nodes
# --nodelist=        # Node where the job is sent

# ---------------------------------------------------- #
# --------------- Properties to adjust --------------- #
# ---------------------------------------------------- #
#SBATCH --job-name=jromp-mpi                          # Name of the job to be sent
#SBATCH --output=outputs/out/jromp-mpi_%A.out # Output filename
#SBATCH --error=outputs/err/jromp-mpi_%A.err  # Error filename

# ---------------------------------------------------- #
# ------------------ Job execution ------------------- #
# ---------------------------------------------------- #
# Parameters:
# 1: Main class to execute (e.g. "jromp.examples.pi.Pi")

# Configure the variables to execute the job and keep it clean
PROJECT_PATH="$(pwd)"
LIBS_CLASSPATH="$PROJECT_PATH"/libs/java/commons-lang3-3.16.0.jar:"$PROJECT_PATH"/libs/java/jromp-2.2.0.jar:"$PROJECT_PATH"/libs/ompi/lib/mpi.jar
CLASSPATH="$PROJECT_PATH"/build/classes/java/main:"$PROJECT_PATH"/build/resources/main:"$LIBS_CLASSPATH"
MPI_RUN_EXECUTABLE="$PROJECT_PATH"/libs/ompi/bin/mpirun

# Hide warnings
export PRTE_MCA_plm_slurm_disable_warning=true
export PRTE_MCA_plm_slurm_ignore_args=true
export LD_LIBRARY_PATH="$PROJECT_PATH"/libs/ompi/lib

# Compile the Java classes
./gradlew compileJava

if [ "$SLURM_NNODES" == 1 ]; then
    echo "1 node"
    #export FI_PROVIDER=tcp
    #export FI_PSM3_CONN_TIMEOUT=60
fi

# Execute the job
$MPI_RUN_EXECUTABLE --bind-to none -np "$SLURM_NTASKS" java -Xint -cp "$CLASSPATH" "$1"

# -Xint: Disable JIT compilation. This is needed to prevent errors to occur when running the program.

# See why I execute mpirun directly: https://docs.open-mpi.org/en/main/launching-apps/slurm.html
