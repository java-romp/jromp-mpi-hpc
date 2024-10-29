#!/bin/bash

# ---------------------------------------------------- #
# --------------- Immutable properties --------------- #
# ---------------------------------------------------- #
#SBATCH --partition=formacion                          # Partition where the Jobs are sent
#SBATCH --account=ule_formacion_9                      # Account to charge the job
#SBATCH --qos=formacion                                # QOS to launch this partition
#SBATCH --time=00:25:00                                # Estimated maximum time for the job
#SBATCH --mail-user=scastd00@estudiantes.unileon.es    # Mail to receive info (start/cancel/end)
#SBATCH --mail-type=FAIL                               # We want to receive all the alerts
#SBATCH --mail-type=END                                # We want to receive all the alerts
#SBATCH --mem=0                                        # Memory required (0 = no limit) Node for me

# ---------------------------------------------------- #
# --------------- Properties to adjust --------------- #
# ---------------------------------------------------- #
# --ntasks=4                                    # Number of tasks (processes)
#SBATCH --ntasks-per-node=8                           # Number of task per node
#SBATCH --cpus-per-task=4                              # Number of cpus per task
#SBATCH --nodes=1                                      # Number of nodes
#SBATCH --nodelist=cn6001                              # Node where the job is sent
#SBATCH --job-name=jromp-mpi                          # Name of the job to be sent
#SBATCH --output=outputs/out/jromp-mpi_%A.out # Output filename
#SBATCH --error=outputs/err/jromp-mpi_%A.err  # Error filename

# ---------------------------------------------------- #
# ------------------ Job execution ------------------- #
# ---------------------------------------------------- #
# Parameters:

export PRTE_MCA_plm_slurm_disable_warning=true
export PRTE_MCA_plm_slurm_ignore_args=true

srun ./gradlew runHelloMPI
