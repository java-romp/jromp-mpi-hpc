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
#SBATCH --mem=3000                                     # Memory required (0 = no limit) Node for me

# ---------------------------------------------------- #
# ------- Properties set on run_calendula.bash ------- #
# ---------------------------------------------------- #
#SBATCH --ntasks=1           # Number of tasks (processes)
#SBATCH --ntasks-per-node=1  # Number of task per node
#SBATCH --cpus-per-task=1    # Number of cpus per task
#SBATCH --nodes=1            # Number of nodes
#SBATCH --nodelist=cn6009         # Node where the job is sent

# ---------------------------------------------------- #
# --------------- Properties to adjust --------------- #
# ---------------------------------------------------- #
#SBATCH --job-name=prepare_data                              # Name of the job to be sent
#SBATCH --output=outputs/out/prepare_data_%A.out             # Output filename
#SBATCH --error=outputs/err/prepare_data_%A.err              # Error filename

DATA_DIR="$HOME/data"
cd "$DATA_DIR"/dna || exit

echo "Removing existing files"
rm -rf huge
mkdir huge

input_file="$DATA_DIR/dna/AllNucleotide.fa"
output_dir="huge"

echo "Extracting DNA sequences from $input_file"
awk -v output_dir="$output_dir" '
    BEGIN {n_seq=0;subdirectory_id=-1}
    # If the line starts with ">", it means a new sequence is starting
    /^>/ {
        # Close the previous file, if any
        if (out) close(out)

        # Create the new file name based on the header ignoring the > symbol
        header = substr($1, 2)

        # Create a new subdirectory every 5000 sequences
        if (n_seq % 5000 == 0) {
            subdirectory_id++
            subdirectory = sprintf("%s/%05d", output_dir, subdirectory_id)
            system("mkdir -p " subdirectory)
            print "Created subdirectory " subdirectory
        }

        # Update the sequence counter
        n_seq++
    }

    # Append the line to the current file
    {
        out = sprintf("%s/%05d/%s.fa", output_dir, subdirectory_id, header)
        print $0 > out
    }

    # Print a message every 10 million lines
    {
        if (NR % 10000000 == 0) {
            print "Processed " NR / 10000000 "0 million lines"
        }
    }
    END {print "Processed a total of " n_seq " sequences and " NR " lines"}
' "$input_file"
