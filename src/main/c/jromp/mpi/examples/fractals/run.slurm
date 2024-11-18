#!/bin/bash

# ---------------------------------------------------- #
# --------------- Immutable properties --------------- #
# ---------------------------------------------------- #
#SBATCH --partition=formacion                          # Partition where the Jobs are sent
#SBATCH --account=ule_formacion_9                      # Account to charge the job
#SBATCH --qos=formacion                                # QOS to launch this partition
#SBATCH --time=24:00:00                                # Estimated maximum time for the job
#SBATCH --mail-user=scastd00@estudiantes.unileon.es    # Mail to receive info (start/cancel/end)
#SBATCH --mail-type=FAIL                               # We want to receive all the failures
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
#SBATCH --job-name=jromp-mpi                           # Name of the job to be sent
#SBATCH --output=outputs/out/jromp-mpi_%A.out          # Output filename
#SBATCH --error=outputs/err/jromp-mpi_%A.err           # Error filename

# ---------------------------------------------------- #
# ------------------ Job execution ------------------- #
# ---------------------------------------------------- #
# Parameters:
# 1: Number of iterations
# 2: Maximum value for the real part of the complex number
# 3: Width of the image
# 4: Height of the image
# 5: Block size
# 6: Number of processes

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

MPI_DIR=../../../../../../../libs/ompi
MPIEXEC=$MPI_DIR/bin/mpiexec
MPICC=$MPI_DIR/bin/mpicc

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <iterations> <max_real> <width> <height> <block_size> <num_processes>"
    exit 1
fi

$MPICC -Wall -pedantic -std=c99 -fopenmp -o mandelbrot.o mandelbrot.c -lm

# Hide warnings
export PRTE_MCA_plm_slurm_disable_warning=true
export PRTE_MCA_plm_slurm_ignore_args=true

# Use mlx5_0:1 interface for Infiniband
export UCX_NET_DEVICES=mlx5_0:1

# Print the SLURM info
print_slurm_info

$MPIEXEC --bind-to none -np "$6" ./mandelbrot.o -n "$1" -a "$2" -c "$3" -r "$4" -b "$5"
