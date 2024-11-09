#!/bin/bash

# ---------------------------------------------------- #
# --------------- Immutable properties --------------- #
# ---------------------------------------------------- #
#SBATCH --partition=formacion                          # Partition where the Jobs are sent
#SBATCH --account=ule_formacion_9                      # Account to charge the job
#SBATCH --qos=formacion                                # QOS to launch this partition
#SBATCH --time=01:00:00                                # Estimated maximum time for the job
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
# --nodelist         # Node where the job is sent

# ---------------------------------------------------- #
# --------------- Properties to adjust --------------- #
# ---------------------------------------------------- #
#SBATCH --job-name=genome                              # Name of the job to be sent
#SBATCH --output=outputs/out/genome_%A.out             # Output filename
#SBATCH --error=outputs/err/genome_%A.err              # Error filename

# ---------------------------------------------------- #
# ------------------ Job execution ------------------- #
# ---------------------------------------------------- #
# Parameters:
# None

function print_slurm_info() {
  echo ""
  echo ""
  echo "##################################################"
  echo "Date: $(date +'%d/%m/%Y %H:%M:%S')"
  echo "Job ID: $SLURM_JOB_ID"
  echo "Job Nodes: $SLURM_NNODES ($SLURM_JOB_NODELIST)"
  echo "Tasks: $SLURM_NTASKS ($SLURM_NTASKS_PER_NODE task(s)/node)"
  echo "CPUs: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK)) ($SLURM_CPUS_PER_TASK CPU(s)/task)"
  echo "##################################################"
  echo ""
  echo ""
}

DATA_DIR="$HOME/data/dna/huge"
PROJECT_BASE_PATH=../../../../../../..
MPI_PATH=$PROJECT_BASE_PATH/libs/ompi
MPI_RUN_EXECUTABLE=$MPI_PATH/bin/mpirun

# Compile the program
make clean all

# Hide warnings
export PRTE_MCA_plm_slurm_disable_warning=true
export PRTE_MCA_plm_slurm_ignore_args=true

# Print the SLURM info
print_slurm_info

# Execute the job
$MPI_RUN_EXECUTABLE --mca pml ob1 --bind-to none -np "$SLURM_NTASKS" ./genome.out "$DATA_DIR"
