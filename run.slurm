#!/bin/bash

# ---------------------------------------------------- #
# --------------- Immutable properties --------------- #
# ---------------------------------------------------- #
#SBATCH --partition=formacion                          # Partition where the Jobs are sent
#SBATCH --account=ule_formacion_9                      # Account to charge the job
#SBATCH --qos=normal                                   # QOS to launch this partition (5 days max)
#SBATCH --time=5-00:00:00                              # Estimated maximum time for the job
#SBATCH --mail-user=scastd00@estudiantes.unileon.es    # Mail to receive info (start/cancel/end)
#SBATCH --mail-type=FAIL,END                           # We want to receive all the failures and end information
#SBATCH --mem=0                                        # Memory required (0 = no limit) Node for me

# ---------------------------------------------------- #
# ------- Properties set on run_calendula.bash ------- #
# ---------------------------------------------------- #
# --ntasks=1                                     # Number of tasks (processes)
# --ntasks-per-node=1                            # Number of task per node
# --cpus-per-task=32                             # Number of cpus per task
# --nodes=1                                      # Number of nodes
# --nodelist=cn6009                              # Node where the job is sent

# ---------------------------------------------------- #
# --------------- Properties to adjust --------------- #
# ---------------------------------------------------- #
#SBATCH --job-name=jromp-mpi-java-gemm                           # Name of the job to be sent
#SBATCH --output=outputs/out/jromp-mpi-java-gemm_%A.out          # Output filename
#SBATCH --error=outputs/err/jromp-mpi-java-gemm_%A.err           # Error filename

# ---------------------------------------------------- #
# ------------------ Job execution ------------------- #
# ---------------------------------------------------- #
# Parameters:
# 1: Main class to execute (e.g. "jromp.examples.pi.Pi")

function print_slurm_info() {
    echo ""
    echo ""
    echo "##################################################"
    echo "Date: $(date +'%d/%m/%Y %H:%M:%S')"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Job Nodes: $SLURM_NNODES ($SLURM_JOB_NODELIST)"
    echo "Tasks: $SLURM_NTASKS ($SLURM_NTASKS_PER_NODE task(s)/node) - $((SLURM_NTASKS - 1)) worker(s) + 1 master"
    echo "CPUs: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK)) ($SLURM_CPUS_PER_TASK CPU(s)/task)"
    echo "##################################################"
    echo ""
    echo ""
}

# Configure the variables to execute the job and keep it clean
PROJECT_PATH="$(pwd)"
LIBS_CLASSPATH="$PROJECT_PATH"/libs/java/commons-lang3-3.16.0.jar:"$PROJECT_PATH"/libs/java/jromp-3.0.0.jar:"$PROJECT_PATH"/libs/ompi/lib/mpi.jar
CLASSPATH="$PROJECT_PATH"/build/classes/java/main:"$PROJECT_PATH"/build/resources/main:"$LIBS_CLASSPATH"
MPI_RUN_EXECUTABLE="$PROJECT_PATH"/libs/ompi/bin/mpirun
JAVA_OPTS="-XX:+UseParallelGC -XX:-TieredCompilation -Xmx32G"

# Hide warnings
export PRTE_MCA_plm_slurm_disable_warning=true
export PRTE_MCA_plm_slurm_ignore_args=true

# Use mlx5_0:1 interface for Infiniband
export UCX_NET_DEVICES=mlx5_0:1

# Compile the Java classes
# ./gradlew compileJava

# Print the SLURM info
print_slurm_info

# Get the fully qualified class name
fq_class_name="$1"
other_args="${*:2}"

# Execute the job
$MPI_RUN_EXECUTABLE --bind-to none -np "$SLURM_NTASKS" java $JAVA_OPTS -cp "$CLASSPATH" "$fq_class_name" "$other_args"

# See why I execute mpirun directly: https://docs.open-mpi.org/en/main/launching-apps/slurm.html
