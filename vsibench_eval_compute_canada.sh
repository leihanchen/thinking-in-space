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

ENV_NAME="${ENV_NAME:-vsibench}"
SKIP_DEP_INSTALL="${SKIP_DEP_INSTALL:-0}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
VENV_BASE="${VENV_BASE:-${PROJECT_ROOT}/.venv}"
VENV_DIR="${VENV_DIR:-${VENV_BASE}/${ENV_NAME}}"

mkdir -p "${VENV_BASE}"

# Validate offline mode requirements
if [[ "${OFFLINE_MODE}" == "1" ]]; then
  printf "[%s] Running in OFFLINE mode\n" "$(date --iso-8601=seconds)"
  
  # Check venv exists in offline mode
  if [[ ! -d "${VENV_DIR}" ]]; then
    printf "[%s] ERROR: Virtual environment not found at %s\n" "$(date --iso-8601=seconds)" "${VENV_DIR}" >&2
    printf "       In offline mode, venv must be pre-built. Please prepare it first:\n" >&2
    printf "       1. Run on online machine: python -m venv %s\n" "${VENV_DIR}" >&2
    printf "       2. Install dependencies: source %s/bin/activate && pip install -r requirements.txt\n" "${VENV_DIR}" >&2
    exit 1
  fi
  
  # Enforce skip deps in offline mode
  SKIP_DEP_INSTALL="1"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  if [[ "${OFFLINE_MODE}" == "1" ]]; then
    printf "[%s] ERROR: Cannot create venv in offline mode\n" "$(date --iso-8601=seconds)" >&2
    exit 1
  fi
  printf "[%s] Creating python venv at %s\n" "$(date --iso-8601=seconds)" "${VENV_DIR}"
  python -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

cd "${PROJECT_ROOT}"

if [[ "${SKIP_DEP_INSTALL}" != "1" ]]; then
  if [[ "${OFFLINE_MODE}" == "1" ]]; then
    printf "[%s] ERROR: Cannot install dependencies in offline mode. Set SKIP_DEP_INSTALL=1 or run in online mode.\n" "$(date --iso-8601=seconds)" >&2
    exit 1
  fi
  printf "[%s] Installing Python dependencies\n" "$(date --iso-8601=seconds)"
  python -m pip install --upgrade pip setuptools wheel
  pushd transformers >/dev/null
  python -m pip install -e .
  popd >/dev/null
  python -m pip install -e .
  python -m pip install "s2wrapper@git+https://github.com/bfshi/scaling_on_scales"
  python -m pip install deepspeed
  python -m pip install qwen-vl-utils
else
  printf "[%s] Skipping dependency installation because SKIP_DEP_INSTALL=%s\n" "$(date --iso-8601=seconds)" "${SKIP_DEP_INSTALL}"
fi

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
  export HF_TOKEN="${HF_TOKEN:-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}"
else
  # In offline mode, use pre-cached models only
  export HF_OFFLINE_MODE=1
  printf "[%s] HF_OFFLINE_MODE enabled - using pre-cached models only\n" "$(date --iso-8601=seconds)"
fi

export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

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

srun bash "${EVAL_SCRIPT}" \
  --model "${MODEL_LIST}" \
  --num_processes "${NUM_PROCESSES}" \
  --benchmark "${BENCHMARK}"

end_time=$(date +%s)
duration=$((end_time - start_time))
printf "Inference time cost: %d seconds\n" "${duration}"
printf "[%s] Evaluation completed successfully\n" "$(date --iso-8601=seconds)"
