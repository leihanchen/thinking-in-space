#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=0-00:15:00
#SBATCH --output=out/%x-%N-%j.out

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

SCRIPT_DIR="$(pwd)"
ENV_DIR="${ENV_DIR:-${SCRIPT_DIR}/h5py/ENV}"

if [ ! -d "$ENV_DIR" ]; then
    echo "ENV not found at $ENV_DIR" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

ZIP_ROOT="${1:-.}"
DATASET_ROOT="${2:-$ZIP_ROOT}"
if [ $# -ge 2 ]; then
    shift 2
elif [ $# -eq 1 ]; then
    shift 1
fi
EXTRA_ARGS=("$@")

PY_SCRIPT="${SCRIPT_DIR}/verify_packed_scenes.py"

mapfile -d '' -t zip_files < <(find "$ZIP_ROOT" -type f -name "*.zip" -print0)

if [ ${#zip_files[@]} -eq 0 ]; then
    echo "No zip files found under $ZIP_ROOT"
    exit 0
fi

cpu_count="${VERIFY_WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"

printf '%s\0' "${zip_files[@]}" | xargs -0 -n1 -P "$cpu_count" -I {} bash -c '
    zip_file="$1"
    py_script="$2"
    dataset_root="$3"
    shift 3
    python "$py_script" --zip-file "$zip_file" --dataset-root "$dataset_root" "$@"
' _ {} "$PY_SCRIPT" "$DATASET_ROOT" "${EXTRA_ARGS[@]}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
    deactivate
fi
