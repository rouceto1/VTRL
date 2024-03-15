#!/bin/bash
# Initialize variables
suffix="experiments"

pwd=$(pwd)
dir_path=$suffix
dirs=($pwd"/../VTRL/"$suffix $pwd"/../VTRL2/"$suffix $pwd/"../VTRL3/"$suffix $pwd/"../VTRL4/"$suffix)
file_count=$(ls -1 "$dir_path" | wc -l)
files_per_dir=$((file_count / 4))

# Loop through all files in the specified directory
i=0
for f in "$dir_path"/*; do
    # Determine which directory to move the file to
    dir_index=$((i / files_per_dir))
    if [ $dir_index -ge ${#dirs[@]} ]; then
        dir_index=$((dir_index % ${#dirs[@]}))
    fi
    outdir="${dirs[$dir_index]}"
    echo "Moving $f to $outdir"
    mkdir -p "$outdir"
    # Move the file to the new subdirectory
    mv "$f" "$outdir"
    # Increment the file index
    ((i++))
done