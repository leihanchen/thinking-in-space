#!/bin/bash
#SBATCH --account=def-wangcs
#SBATCH --job-name=vsibench_eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:2
#SBATCH --mem=512G
#SBATCH --time=12:00:00

set -euo pipefail

printf "[%s] Starting VSI-Bench evaluation on host: %s\n" "$(date --iso-8601=seconds)" "${HOSTNAME}"

# Configure module environment. Adjust if your Compute Canada site uses different module names.
module load StdEnv/2023 scipy-stack

# Allow callers to override the exact compiler/OpenCV module pair expected by Compute Canada.
GCC_MODULE="${GCC_MODULE:-gcc/12.3}"
OPENCV_MODULE="${OPENCV_MODULE:-opencv/4.9.0}"

module load "${GCC_MODULE}"
module load "${OPENCV_MODULE}"

# CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"
# if ! module load "${CUDA_MODULE}"; then
#   printf "[%s] Requested CUDA module '%s' not found. Falling back to cuda/11.8 for compatibility.\n" \
#     "$(date --iso-8601=seconds)" "${CUDA_MODULE}" >&2
#   module load cuda/11.8
# fi
# module load cuda/11.8
module load python/3.10
module load arrow/18.1.0

ENV_NAME="${ENV_NAME:-vsibench}"
SKIP_DEP_INSTALL="${SKIP_DEP_INSTALL:-0}"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
VENV_BASE="${VENV_BASE:-${PROJECT_ROOT}/.venv}"
VENV_DIR="${VENV_DIR:-${VENV_BASE}/${ENV_NAME}}"

mkdir -p "${VENV_BASE}"

if [[ ! -d "${VENV_DIR}" ]]; then
  printf "[%s] Creating python venv at %s\n" "$(date --iso-8601=seconds)" "${VENV_DIR}"
  python -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

cd "${PROJECT_ROOT}"

if [[ "${SKIP_DEP_INSTALL}" != "1" ]]; then
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
MODEL_LIST="${MODEL_LIST:-qwen_2p5_7b_instruct,qwen_2p5_7b_instruct_spar234k}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
BENCHMARK="${BENCHMARK:-vsibench}"
EVAL_SCRIPT="${EVAL_SCRIPT:-evaluate_all_in_one.sh}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SLURM_TMPDIR:-${HOME}/.cache/transformers}}"
export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-${HOME}/.cache/huggingface}}"
export HF_TOKEN="${HF_TOKEN:-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}"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_HOME}"

printf "[%s] Launch command: %s --model %s --num_processes %s --benchmark %s\n" \
  "$(date --iso-8601=seconds)" "${EVAL_SCRIPT}" "${MODEL_LIST}" "${NUM_PROCESSES}" "${BENCHMARK}"

srun bash "${EVAL_SCRIPT}" \
  --model "${MODEL_LIST}" \
  --num_processes "${NUM_PROCESSES}" \
  --benchmark "${BENCHMARK}"

printf "[%s] Evaluation completed successfully\n" "$(date --iso-8601=seconds)"
