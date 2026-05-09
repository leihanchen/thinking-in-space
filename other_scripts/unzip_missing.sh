#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=0-00:20:00
#SBATCH --output=out/%x-%N-%j.out

# Check if a directory was provided as an argument, otherwise use the current directory
TARGET_DIR="${1:-.}"

# Build the list of missing folders serially first, then unzip in parallel.
mapfile -d '' -t missing_zips < <(
    find "$TARGET_DIR" -type f -name "*.zip" -print0 | while IFS= read -r -d '' zip_file; do
        # Get the directory containing the zip file
        dirpath=$(dirname "$zip_file")

        # Get the name of the zip file without the extension (e.g., 'X' from 'X.zip')
        foldername=$(basename "$zip_file" .zip)

        # Check if a directory with that name does not exist in the same location
        if [ ! -d "$dirpath/$foldername" ]; then
            printf '%s\0' "$zip_file"
        else
            echo "Skipping '$zip_file', folder '$foldername' already exists." >&2
        fi
    done
)

if [ ${#missing_zips[@]} -eq 0 ]; then
    echo "No missing folders found."
    exit 0
fi

cpu_count="${SLURM_CPUS_PER_TASK:-$(nproc)}"
printf '%s\0' "${missing_zips[@]}" | xargs -0 -n1 -P "$cpu_count" -I {} bash -c '
    zip_file="$1"
    dirpath=$(dirname "$zip_file")
    foldername=$(basename "$zip_file" .zip)
    echo "Folder '\''$foldername'\'' not found. Unzipping '\''$zip_file'\''..."

    # Unzip the file into its containing directory
    # (If you want to force extraction into a new folder named 'X', change it to: -d "$dirpath/$foldername")
    unzip -q "$zip_file" -d "$dirpath"
' _ {}