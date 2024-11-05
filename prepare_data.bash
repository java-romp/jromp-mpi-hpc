#!/bin/bash

cd data || exit

echo "Removing existing files"
rm -rf dna
mkdir dna

echo "Extracting all.fna.tar.gz"
tar -xzf all.fna.tar.gz -C dna

cd dna || exit
directories=$(ls -d */)
total_fna_files=0

echo "Extracting all .tgz files"
for dir in $directories; do
  cd "$dir" || exit
  tgz_files=$(find . -name "*.tgz")

  for file in $tgz_files; do
    echo "Extracting $dir${file:2}"
    tar -xzf "$file"
    rm "$file"
  done

  # Count the number of .fna files
  fna_files=$(find . -name "*.fna" | wc -l)
  total_fna_files=$((total_fna_files + fna_files))

  cd ..
done

echo "Total number of .fna files: $total_fna_files"
