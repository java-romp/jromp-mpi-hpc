#!/bin/bash

cd data/huge || exit

echo "Removing existing files"
rm -rf dna
mkdir dna

input_file="AllNucleotide.fa"
output_dir="dna"

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
