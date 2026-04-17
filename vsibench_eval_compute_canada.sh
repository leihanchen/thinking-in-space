#!/bin/bash
#SBATCH --account=def-wangcs
#SBATCH --job-name=vsibench_eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --time=7:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=christopher.indris@torontomu.ca

set -euo pipefail

printf "[%s] Starting VSI-Bench evaluation on host: %s\n" "$(date --iso-8601=seconds)" "${HOSTNAME}"

# Configure module environment. Adjust if your Compute Canada site uses different module names.
SCIPY_STACK_MODULE="${SCIPY_STACK_MODULE:-scipy-stack/2024a}"
module load StdEnv/2023 "${SCIPY_STACK_MODULE}"

# Allow callers to override the exact compiler/OpenCV module pair expected by Compute Canada.
GCC_MODULE="${GCC_MODULE:-gcc/12.3}"
OPENCV_MODULE="${OPENCV_MODULE:-opencv/4.9.0}"

module load "${GCC_MODULE}"
module load "${OPENCV_MODULE}"

CUDA_MODULE="${CUDA_MODULE:-cuda/12.2}"
module load "${CUDA_MODULE}"

# CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"
# if ! module load "${CUDA_MODULE}"; then
#   printf "[%s] Requested CUDA module '%s' not found. Falling back to cuda/11.8 for compatibility.\n" \
#     "$(date --iso-8601=seconds)" "${CUDA_MODULE}" >&2
#   module load cuda/11.8
# fi
# module load cuda/11.8
module load python/3.10
module load arrow/18.1.0

if [[ -z "${CUDA_HOME:-}" && -n "${EBROOTCUDA:-}" ]]; then
  export CUDA_HOME="${EBROOTCUDA}"
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_ROOT}"

SKIP_DEP_INSTALL="${SKIP_DEP_INSTALL:-1}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/containers/vsibench_eval.sif}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer}"

if [[ "${OFFLINE_MODE}" == "1" ]]; then
  printf "[%s] Running in OFFLINE mode\n" "$(date --iso-8601=seconds)"
fi

if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
  module load "${APPTAINER_MODULE}" 2>/dev/null || module load singularity 2>/dev/null || true
fi

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="singularity"
else
  printf "[%s] ERROR: apptainer/singularity runtime not found. Load the module or set APPTAINER_MODULE.\n" "$(date --iso-8601=seconds)" >&2
  exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
  printf "[%s] ERROR: SIF image not found at %s\n" "$(date --iso-8601=seconds)" "${SIF_PATH}" >&2
  printf "       Build it first with: apptainer build --fakeroot %s vsibench_eval.def\n" "${SIF_PATH}" >&2
  exit 1
fi

printf "[%s] Using container runtime: %s\n" "$(date --iso-8601=seconds)" "${APPTAINER_BIN}"
printf "[%s] Using SIF image: %s\n" "$(date --iso-8601=seconds)" "${SIF_PATH}"
printf "[%s] SKIP_DEP_INSTALL=%s (ignored when using SIF-based runtime)\n" "$(date --iso-8601=seconds)" "${SKIP_DEP_INSTALL}"

# Configure runtime defaults. Override via environment variables when submitting.
MODEL_LIST="${MODEL_LIST:-cvis-tmu/qwen2_5vl-7b-lora-sft-Scene30k_traineval_852steps_merged}"

if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
  SLURM_GPU_COUNT=$(echo "${SLURM_GPUS_PER_NODE}" | awk -F: '{print $NF}')
else
  SLURM_GPU_COUNT=1
fi

NUM_PROCESSES="${NUM_PROCESSES:-${SLURM_GPU_COUNT}}"
BENCHMARK="${BENCHMARK:-vsibench}"
EVAL_SCRIPT="${EVAL_SCRIPT:-evaluate_all_in_one.sh}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PROJECT_ROOT}/.cache/transformers}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

# Only set HF_TOKEN if in online mode or if user explicitly provides it
if [[ "${OFFLINE_MODE}" != "1" ]]; then
  if [ -f "/scratch/indrisch/TOKENS/huggingface/cvis-tmu-organization-token.txt" ]; then
    HF_TOKEN=$(<"/scratch/indrisch/TOKENS/huggingface/cvis-tmu-organization-token.txt")
    export HF_TOKEN
    printf "[%s] Loaded HF_TOKEN from file\n" "$(date --iso-8601=seconds)"
  elif [[ -n "${HF_TOKEN:-}" ]]; then
    printf "[%s] Using HF_TOKEN from environment variable\n" "$(date --iso-8601=seconds)"
  else
    export HF_TOKEN="${HF_TOKEN:-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}"
  fi
else
  # In offline mode, use pre-cached models only
  export HF_OFFLINE_MODE=1
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  printf "[%s] HF_OFFLINE_MODE enabled - using pre-cached models only\n" "$(date --iso-8601=seconds)"
fi

export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

export APPTAINERENV_OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export APPTAINERENV_TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export APPTAINERENV_HF_HOME="${HF_HOME}"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export APPTAINERENV_MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}"
export SINGULARITYENV_OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export SINGULARITYENV_TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export SINGULARITYENV_HF_HOME="${HF_HOME}"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export SINGULARITYENV_MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
  export SINGULARITYENV_HF_TOKEN="${HF_TOKEN}"
fi
if [[ "${OFFLINE_MODE}" == "1" ]]; then
  export APPTAINERENV_HF_OFFLINE_MODE=1
  export APPTAINERENV_HF_HUB_OFFLINE=1
  export APPTAINERENV_TRANSFORMERS_OFFLINE=1
  export SINGULARITYENV_HF_OFFLINE_MODE=1
  export SINGULARITYENV_HF_HUB_OFFLINE=1
  export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
fi

# Validate cache availability in offline mode
if [[ "${OFFLINE_MODE}" == "1" ]]; then
  if [[ ! -d "${TRANSFORMERS_CACHE}" ]] || [[ -z "$(find "${TRANSFORMERS_CACHE}" -type f -name "*.json" 2>/dev/null | head -1)" ]]; then
    printf "[%s] WARNING: Transformer cache directory may be empty: %s\n" "$(date --iso-8601=seconds)" "${TRANSFORMERS_CACHE}" >&2
    printf "       Make sure models are pre-cached before running in offline mode.\n" >&2
  fi
fi

printf "[%s] Launch command: %s --model %s --num_processes %s --benchmark %s\n" \
  "$(date --iso-8601=seconds)" "${EVAL_SCRIPT}" "${MODEL_LIST}" "${NUM_PROCESSES}" "${BENCHMARK}"

start_time=$(date +%s)

srun "${APPTAINER_BIN}" exec --nv \
  --bind "${PROJECT_ROOT}:${CONTAINER_WORKDIR}" \
  --bind "${TRANSFORMERS_CACHE}:${TRANSFORMERS_CACHE}" \
  --bind "${HF_HOME}:${HF_HOME}" \
  --bind "${HUGGINGFACE_HUB_CACHE}:${HUGGINGFACE_HUB_CACHE}" \
  --pwd "${CONTAINER_WORKDIR}" \
  "${SIF_PATH}" \
  bash "${CONTAINER_WORKDIR}/${EVAL_SCRIPT}" \
  --model "${MODEL_LIST}" \
  --num_processes "${NUM_PROCESSES}" \
  --benchmark "${BENCHMARK}"

end_time=$(date +%s)
duration=$((end_time - start_time))
printf "Inference time cost: %d seconds\n" "${duration}"
printf "[%s] Evaluation completed successfully\n" "$(date --iso-8601=seconds)"
