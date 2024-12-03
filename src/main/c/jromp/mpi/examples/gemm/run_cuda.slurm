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
#SBATCH --ntasks=1                                     # Number of tasks (processes)
#SBATCH --ntasks-per-node=1                            # Number of task per node
#SBATCH --cpus-per-task=32                             # Number of cpus per task
#SBATCH --nodes=1                                      # Number of nodes
#SBATCH --nodelist=cn6024                              # Node where the job is sent
#SBATCH --gres=gpu:1                                   # Number of GPUs required
#SBATCH --constraint="gpu_a100"                        # GPU model required

# ---------------------------------------------------- #
# --------------- Properties to adjust --------------- #
# ---------------------------------------------------- #
#SBATCH --job-name=jromp-mpi-cuda-gemm                           # Name of the job to be sent
#SBATCH --output=outputs/out/jromp-mpi-cuda-gemm_%A.out          # Output filename
#SBATCH --error=outputs/err/jromp-mpi-cuda-gemm_%A.err           # Error filename

# ---------------------------------------------------- #
# ------------------ Job execution ------------------- #
# ---------------------------------------------------- #
# Parameters:
# No parameters

# Set the CUDA compiler and flags
CC="nvcc"
CUDA_LINKS="-lcublas"
RUN_COMMAND=
matrix_size=
optimization_level=0

if [[ $(hostname) == cn* ]]; then
    cuda_module="CUDA/12.1.1"

    if ! module is-loaded "$cuda_module"; then
      module load $cuda_module
    fi

    RUN_COMMAND="srun"
    matrix_size=$((30 * 32 * 30))
else
    RUN_COMMAND=
    matrix_size=8000
fi

echo "Compiling CUDA..."

# Compile the code
$CC -o gemm.cu.o gemm.cu -O$optimization_level $CUDA_LINKS

# Hide warnings
export PRTE_MCA_plm_slurm_disable_warning=true
export PRTE_MCA_plm_slurm_ignore_args=true

# Use mlx5_0:1 interface for Infiniband
export UCX_NET_DEVICES=mlx5_0:1

echo "Executing CUDA..."

# Execute the code
$RUN_COMMAND ./gemm.cu.o $matrix_size
