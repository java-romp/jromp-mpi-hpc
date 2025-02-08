#!/bin/bash

# ---------------------------------------------------- #
# --------------- Immutable properties --------------- #
# ---------------------------------------------------- #
#SBATCH --partition=formacion                            # Partition where the Jobs are sent
#SBATCH --account=ule_formacion_9                        # Account to charge the job
#SBATCH --qos=normal                                     # QOS to launch this partition (5 days max)
#SBATCH --time=5-00:00:00                                # Estimated maximum time for the job
#SBATCH --mail-user=scastd00@estudiantes.unileon.es      # Mail to receive info (start/cancel/end)
#SBATCH --mail-type=FAIL,END                             # We want to receive all the failures and end information
#SBATCH --mem=0                                          # Memory required (0 = no limit) Node for me
#SBATCH --constraint=icelake                             # Node type
#SBATCH --exclude=cn[6001-6008]                          # Nodes to exclude

# ---------------------------------------------------- #
# ------- Properties set on run_calendula.bash ------- #
# ---------------------------------------------------- #
# --ntasks=1                                             # Number of tasks (processes)
# --ntasks-per-node=1                                    # Number of task per node
# --cpus-per-task=32                                     # Number of cpus per task
# --nodes=1                                              # Number of nodes
# --nodelist=cn6009                                      # Node where the job is sent

# ---------------------------------------------------- #
# --------------- Properties to adjust --------------- #
# ---------------------------------------------------- #
#SBATCH --job-name=jromp-mpi-c-gemm                      # Name of the job to be sent
#SBATCH --output=outputs/out/jromp-mpi-c-gemm_%A.out     # Output filename
#SBATCH --error=outputs/err/jromp-mpi-c-gemm_%A.err      # Error filename

# ---------------------------------------------------- #
# ------------------ Job execution ------------------- #
# ---------------------------------------------------- #
# Parameters:
# $1: Number of ranks
# $2: Matrix size
# $3: Number of threads
# $4: Optimization level

PROJECT_BASE_PATH=../../../../../../..
MPI_PATH=$PROJECT_BASE_PATH/libs/ompi
MPI_INCLUDE=$MPI_PATH/include
MPI_LIB=$MPI_PATH/lib
MPI_BIN=$MPI_PATH/bin

CC="$MPI_BIN/mpicc"
CFLAGS="-I$MPI_INCLUDE -Wall -fopenmp"
LDFLAGS="-L$MPI_LIB"
MPIEXEC="$MPI_BIN/mpirun"
COMPILED_CODE="gemm.o"

# If argument 5 is present, use it as the filename of the compiled code
if [ -n "$5" ]; then
    COMPILED_CODE="$5"
fi

# Compile the code
$CC $CFLAGS -o $COMPILED_CODE gemm.c -O"$4" $LDFLAGS

# Hide warnings
export PRTE_MCA_plm_slurm_disable_warning=true
export PRTE_MCA_plm_slurm_ignore_args=true

# Use mlx5_0:1 interface for Infiniband
export UCX_NET_DEVICES=mlx5_0:1

# Print the SLURM info
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

$MPIEXEC --bind-to none -np "$1" ./"$COMPILED_CODE" "$2" "$3" "$4"

# See why I execute mpirun directly: https://docs.open-mpi.org/en/main/launching-apps/slurm.html
