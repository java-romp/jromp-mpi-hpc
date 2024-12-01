#!/bin/bash

# ---------------------------------------------------- #
# --------------- Immutable properties --------------- #
# ---------------------------------------------------- #
#SBATCH --partition=formacion                          # Partition where the Jobs are sent
#SBATCH --account=ule_formacion_9                      # Account to charge the job
#SBATCH --qos=normal                                   # QOS to launch this partition (5 days max)
#SBATCH --time=5-00:00:00                              # Estimated maximum time for the job
#SBATCH --mail-user=scastd00@estudiantes.unileon.es    # Mail to receive info (start/cancel/end)
#SBATCH --mail-type=FAIL                               # We want to receive all the failures
#SBATCH --mem=0                                        # Memory required (0 = no limit) Node for me
#SBATCH --gres=gpu:1                                   # Number of GPUs required
#SBATCH --constraint="gpu_a100"                        # GPU model required

# ---------------------------------------------------- #
# ------- Properties set on run_calendula.bash ------- #
# ---------------------------------------------------- #
#SBATCH --ntasks=1                                     # Number of tasks (processes)
#SBATCH --ntasks-per-node=1                            # Number of task per node
#SBATCH --cpus-per-task=32                             # Number of cpus per task
#SBATCH --nodes=1                                      # Number of nodes
#SBATCH --nodelist=cn6024                              # Node where the job is sent

# ---------------------------------------------------- #
# --------------- Properties to adjust --------------- #
# ---------------------------------------------------- #
#SBATCH --job-name=jromp-mpi-cuda-multiplication                           # Name of the job to be sent
#SBATCH --output=outputs/out/jromp-mpi-cuda-multiplication_%A.out          # Output filename
#SBATCH --error=outputs/err/jromp-mpi-cuda-multiplication_%A.err           # Error filename

# ---------------------------------------------------- #
# ------------------ Job execution ------------------- #
# ---------------------------------------------------- #
# Parameters:
# $1 = Optimization level

# Load the CUDA module if the hostname starts with frontend
if [[ $(hostname) == frontend* ]]; then
    module load CUDA/12.1.1
fi

# Set the CUDA compiler and flags
CC="nvcc"
CFLAGS="-Wall"
CUDA_LINKS="-lcublas"

# Compile the code
$CC $CFLAGS -o big_multiplication.cu.o big_multiplication.cu -O"$1" $CUDA_LINKS

# Hide warnings
export PRTE_MCA_plm_slurm_disable_warning=true
export PRTE_MCA_plm_slurm_ignore_args=true

# Use mlx5_0:1 interface for Infiniband
export UCX_NET_DEVICES=mlx5_0:1

matrix_size=

if [[ $(hostname) == frontend* ]]; then
    matrix_size=$((30 * 32 * 30))
else
    matrix_size=8192
fi

# Execute the code
srun ./big_multiplication.cu.o $matrix_size
