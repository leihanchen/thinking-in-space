#!/bin/bash
#SBATCH --account=def-wangcs_gpu
#SBATCH --job-name=vsibench_eval
#SBATCH --output=%x-%N-%j.out
#SBATCH --error=%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h100:2
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=christopher.indris@torontomu.ca


# Settings for full run:
# #SBATCH --cpus-per-task=2
# #SBATCH --gpus-per-node=h100:2
# #SBATCH --mem=64G
# #SBATCH --time=7:00:00

set -euo pipefail

printf "[%s] Starting VSI-Bench evaluation on host: %s\n" "$(date --iso-8601=seconds)" "${HOSTNAME}"

USER_ACCOUNT=$(whoami)
echo "Running as user: ${USER_ACCOUNT}"

# --- Project root ---

if [[ "$USER_ACCOUNT" == "indrisch" ]] && [[ "$PWD" == *thinking-in-space* ]]; then
      PROJECT_ROOT="${PWD%%thinking-in-space*}/thinking-in-space"
else
    PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
    cd "${PROJECT_ROOT}"
fi

SYSCONFIG_DIR_PATH="$PROJECT_ROOT/scripts"
export PYTHONPATH="$PYTHONPATH:$SYSCONFIG_DIR_PATH"


# --- setting environment ---

# Detect cluster based on terminal prompt or hostname
if [[ "$HOSTNAME" == *"rg"* ]]; then
    CLUSTER="RORQUAL"
elif [[ "$HOSTNAME" == *"trig"* ]]; then
    CLUSTER="TRILLIUM"
elif [[ "$HOSTNAME" == *"kn"* ]]; then
    CLUSTER="KILLARNEY"
elif [[ "$HOSTNAME" == *"g"* ]]; then
    CLUSTER="NIBI"
    OFFLINE_MODE=0  # Nibi has internet access, so disable offline mode
else
    echo "Warning: Could not detect cluster from PS1 or HOSTNAME. Defaulting to RORQUAL."
    CLUSTER="RORQUAL"
fi

echo cluster detected: "${CLUSTER}"


if [[ "$USER_ACCOUNT" == "indrisch" ]]; then
    # Custom settings for user 'indrisch'
    TRANSFORMERS_CACHE="/scratch/indrisch/.cache/transformers"
    export HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HOME'))")" && echo "HF_HOME: $HF_HOME"
    export HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HUB_CACHE'))")" && echo "HF_HUB_CACHE: $HF_HUB_CACHE"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
    export TRITON_CACHE_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'TRITON_CACHE_DIR'))")" && echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
    export FLASHINFER_WORKSPACE_BASE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'FLASHINFER_WORKSPACE_BASE'))")" && echo "FLASHINFER_WORKSPACE_BASE: $FLASHINFER_WORKSPACE_BASE"
    export BEST_GPU="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'BEST_GPU'))")" && echo "BEST_GPU: $BEST_GPU"
    export TORCH_EXTENSIONS_DIR="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'TORCH_EXTENSIONS_DIR'))")" && echo "TORCH_EXTENSIONS_DIR: $TORCH_EXTENSIONS_DIR"
    export SIF_FILE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'SIF_FILE'))")" && echo "SIF_FILE: $SIF_FILE"
    export SIF_PATH="$SIF_FILE"

    export WANDB_DIR="${PROJECT_ROOT}/wandb/"
    if [[ "$BEST_GPU" == "h100" ]]; then
        export TORCH_CUDA_ARCH_LIST="9.0"
    else
        export TORCH_CUDA_ARCH_LIST="8.0"
    fi
else
    echo "Using default environment settings for ${USER_ACCOUNT}"
    SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/containers/vsibench_eval.sif}"
fi

# ----


# Configure module environment. The container already carries its own CUDA stack,
# so avoid exporting the host CUDA toolchain into the Apptainer runtime.
SCIPY_STACK_MODULE="${SCIPY_STACK_MODULE:-scipy-stack/2024a}"
module load StdEnv/2023 "${SCIPY_STACK_MODULE}"

# Allow callers to override the exact compiler/OpenCV module pair expected by Compute Canada.
GCC_MODULE="${GCC_MODULE:-gcc/12.3}"
OPENCV_MODULE="${OPENCV_MODULE:-opencv/4.9.0}"

module load "${GCC_MODULE}"
module load "${OPENCV_MODULE}"

module load python/3.10
module load arrow/18.1.0

SKIP_DEP_INSTALL="${SKIP_DEP_INSTALL:-1}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"

HOST_GCC_BIN="$(command -v gcc || true)"
HOST_GXX_BIN="$(command -v g++ || true)"

CUDA_MODULE="${CUDA_MODULE:-cuda/12.2}"
module load "${CUDA_MODULE}"

if [[ -z "${CUDA_HOME:-}" && -n "${EBROOTCUDA:-}" ]]; then
  export CUDA_HOME="${EBROOTCUDA}"
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
cd "${PROJECT_ROOT}"

SKIP_DEP_INSTALL="${SKIP_DEP_INSTALL:-1}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/vsibench_eval.sif}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
APPTAINER_MODULE="${APPTAINER_MODULE:-apptainer}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache/huggingface}"

if [[ "${OFFLINE_MODE}" == "1" ]]; then
  printf "[%s] Running in OFFLINE mode\n" "$(date --iso-8601=seconds)"
fi

if ! command -v apptainer >/dev/null 2>&1; then
  module load "${APPTAINER_MODULE}" 2>/dev/null || true
fi

if ! command -v apptainer >/dev/null 2>&1; then
  printf "[%s] ERROR: apptainer runtime not found. Load module '%s'.\n" "$(date --iso-8601=seconds)" "${APPTAINER_MODULE}" >&2
  exit 1
fi

APPTAINER_BIN="apptainer"

if [[ ! -f "${SIF_PATH}" ]]; then
  printf "[%s] ERROR: SIF image not found at %s\n" "$(date --iso-8601=seconds)" "${SIF_PATH}" >&2
  printf "       Build it first with: apptainer build --fakeroot %s vsibench_eval.def\n" "${SIF_PATH}" >&2
  exit 1
fi

printf "[%s] Using container runtime: %s\n" "$(date --iso-8601=seconds)" "${APPTAINER_BIN}"
printf "[%s] Using SIF image: %s\n" "$(date --iso-8601=seconds)" "${SIF_PATH}"
printf "[%s] SKIP_DEP_INSTALL=%s (ignored when using SIF-based runtime)\n" "$(date --iso-8601=seconds)" "${SKIP_DEP_INSTALL}"

# Configure runtime defaults. Override via environment variables when submitting.
# MODEL_LIST="${MODEL_LIST:-cvis-tmu/qwen2_5vl-7b-lora-sft-Scene30k_traineval_2130steps_merged,cvis-tmu/qwen2_5vl-7b-lora-sft-Scene30k_traineval_852steps_merged}"
MODEL_LIST="${MODEL_LIST:-cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_426steps,cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_852steps,cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_5epochs,cvis-tmu/videor1-lora-sft-Scene30k_traineval_426steps,cvis-tmu/videor1-lora-sft-Scene30k_traineval_852steps,cvis-tmu/videor1-lora-sft-Scene30k_traineval_5epochs}"

if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
  SLURM_GPU_COUNT=$(echo "${SLURM_GPUS_PER_NODE}" | awk -F: '{print $NF}')
else
  SLURM_GPU_COUNT=1
fi

NUM_PROCESSES="${NUM_PROCESSES:-${SLURM_GPU_COUNT}}"
BENCHMARK="${BENCHMARK:-vsibench}"
EVAL_SCRIPT="${EVAL_SCRIPT:-evaluate_all_in_one.sh}"

SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"


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
  # export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  printf "[%s] HF_OFFLINE_MODE enabled - using pre-cached models only\n" "$(date --iso-8601=seconds)"
fi

export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

export APPTAINERENV_OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export APPTAINERENV_HF_HOME="${HF_HOME}"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export APPTAINERENV_HF_HUB_CACHE="${HF_HUB_CACHE}"
export APPTAINERENV_HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export APPTAINERENV_HF_MODULES_CACHE="${HF_MODULES_CACHE}"
export APPTAINERENV_MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}"
export APPTAINERENV_PYTHONPATH="${CONTAINER_WORKDIR}:${CONTAINER_WORKDIR}/scripts"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export SINGULARITYENV_OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export SINGULARITYENV_TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export SINGULARITYENV_HF_HOME="${HF_HOME}"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export SINGULARITYENV_MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}"
export SINGULARITYENV_PYTHONPATH="${CONTAINER_WORKDIR}:${CONTAINER_WORKDIR}/scripts"
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export DECORD_EOF_RETRY_MAX=20480
# # Force container CUDA toolchain paths so DeepSpeed/Triton do not resolve host module paths.
# export APPTAINERENV_CUDA_HOME="/usr/local/cuda"
# export APPTAINERENV_CUDA_PATH="/usr/local/cuda"
# export APPTAINERENV_PATH="/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
fi
if [[ "${OFFLINE_MODE}" == "1" ]]; then
  export APPTAINERENV_HF_OFFLINE_MODE=1
  export APPTAINERENV_HF_HUB_OFFLINE=1
  export APPTAINERENV_TRANSFORMERS_OFFLINE=1
  export APPTAINERENV_HF_DATASETS_OFFLINE=1
fi

# Validate cache availability in offline mode
if [[ "${OFFLINE_MODE}" == "1" ]]; then
  if [[ ! -d "${HUGGINGFACE_HUB_CACHE}" ]]; then
    printf "[%s] WARNING: Hugging Face cache may be empty: %s\n" "$(date --iso-8601=seconds)" "${HUGGINGFACE_HUB_CACHE}" >&2
    printf "       Make sure models are pre-cached before running in offline mode.\n" >&2
  fi
fi

# --- final module check ---

module list

# --- launch ---

printf "[%s] Launch command: %s --model %s --num_processes %s --benchmark %s\n" \
  "$(date --iso-8601=seconds)" "${EVAL_SCRIPT}" "${MODEL_LIST}" "${NUM_PROCESSES}" "${BENCHMARK}"

# if [[ "${OFFLINE_MODE}" == "1" ]]; then
#   printf "[%s] Preflight: verifying container can see Hugging Face caches\n" "$(date --iso-8601=seconds)"
#   if ! "${APPTAINER_BIN}" exec --nv --cleanenv \
#     --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
#     --bind "${HF_HOME}:${HF_HOME}" \
#     --bind "${HF_DATASETS_CACHE}:${HF_DATASETS_CACHE}" \
#     --pwd "${PROJECT_ROOT}" \
#     "${SIF_PATH}" \
#     python3 - <<'PY'
# import os
# from pathlib import Path

# paths = {
#     "HF_HOME": os.environ.get("HF_HOME", ""),
#     "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
#     "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", ""),
#     "HF_MODULES_CACHE": os.environ.get("HF_MODULES_CACHE", ""),
# }

# for key, value in paths.items():
#     print(f"{key}={value}")
#     if not value or not Path(value).exists():
#         raise SystemExit(f"missing cache path: {key}={value}")

# hub_cache = Path(paths["HUGGINGFACE_HUB_CACHE"])
# dataset_snapshot = hub_cache / "datasets--nyu-visionx--VSI-Bench" / "snapshots"
# if not dataset_snapshot.exists():
#   raise SystemExit(f"dataset snapshot directory not found under {dataset_snapshot}")

# snapshot_entries = list(dataset_snapshot.glob("*/test.jsonl"))
# if not snapshot_entries:
#   raise SystemExit(f"VSI-Bench snapshot exists but test.jsonl is missing under {dataset_snapshot}")

# modules_cache = Path(paths["HF_MODULES_CACHE"])
# if not modules_cache.exists():
#   raise SystemExit(f"HF modules cache path missing: {modules_cache}")

# print("preflight ok")
# PY
#   then
#     printf "[%s] ERROR: container preflight failed; cache is not visible where datasets expects it\n" "$(date --iso-8601=seconds)" >&2
#     exit 1
#   fi
# fi

# printf "[%s] Preflight: verifying container compiler toolchain for Triton/DeepSpeed\n" "$(date --iso-8601=seconds)"
# if ! "${APPTAINER_BIN}" exec --nv --cleanenv \
#   --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
#   --bind "${HF_HOME}:${HF_HOME}" \
#   --pwd "${PROJECT_ROOT}" \
#   "${SIF_PATH}" \
#   bash -lc '
# set -e
# if command -v gcc >/dev/null 2>&1; then
#   gcc --version | head -n 1
#   exit 0
# fi
# if [[ -n "${CC:-}" ]] && [[ -x "${CC}" ]]; then
#   "${CC}" --version | head -n 1
#   exit 0
# fi
# echo "No C compiler visible in container PATH and CC is not executable" >&2
# exit 1
# '; then
#   printf "[%s] ERROR: container cannot find a usable C compiler (gcc/CC).\n" "$(date --iso-8601=seconds)" >&2
#   printf "       Host gcc: %s\n" "${HOST_GCC_BIN:-<not found>}" >&2
#   printf "       Rebuild SIF with gcc installed, or ensure module gcc path is visible inside apptainer.\n" >&2
#   exit 1
# fi

# printf "[%s] Preflight: verifying container nvcc path for DeepSpeed\n" "$(date --iso-8601=seconds)"
# if ! "${APPTAINER_BIN}" exec --nv --cleanenv \
#   --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
#   --bind "${HF_HOME}:${HF_HOME}" \
#   --pwd "${PROJECT_ROOT}" \
#   "${SIF_PATH}" \
#   bash -lc '
# set -e
# echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
# if [[ -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
#   "${CUDA_HOME}/bin/nvcc" --version | head -n 1
#   exit 0
# fi
# if command -v nvcc >/dev/null 2>&1; then
#   nvcc --version | head -n 1
#   exit 0
# fi
# echo "nvcc is not visible in container" >&2
# exit 1
# '; then
#   printf "[%s] ERROR: container cannot find nvcc. Check CUDA_HOME and CUDA toolkit installation in SIF.\n" "$(date --iso-8601=seconds)" >&2
#   exit 1
# fi

# Validate model cache from inside the container before launching full evaluation.
# if [[ "${OFFLINE_MODE}" == "1" ]]; then
#   if ! "${APPTAINER_BIN}" exec --nv \
#     "${CONTAINER_ENV_ARGS[@]}" \
#     --bind "${PROJECT_ROOT}:${CONTAINER_WORKDIR}" \
#     --bind "${HF_HOME}:${HF_HOME}" \
#     --pwd "${CONTAINER_WORKDIR}" \
#     "${SIF_PATH}" \
#     python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${MODEL_LIST}', filename='config.json', local_files_only=True)"; then
#     printf "[%s] ERROR: Model cache is not reachable inside container for model %s\n" "$(date --iso-8601=seconds)" "${MODEL_LIST}" >&2
#     printf "       Ensure HF cache contains this model and paths are correctly bound.\n" >&2
#     exit 1
#   fi
# fi

start_time=$(date +%s)
printf "[%s] Streaming container output directly to stdout/stderr.\n" "$(date --iso-8601=seconds)"

srun "${APPTAINER_BIN}" exec --fakeroot --nv --overlay /scratch/indrisch/thinking-in-space/containers/apptainer-overlay.img -C \
  --bind /etc/pki/tls/certs/ca-bundle.crt \
  --bind "${PROJECT_ROOT}:${CONTAINER_WORKDIR}" \
  --bind "${TRANSFORMERS_CACHE}:${TRANSFORMERS_CACHE}" \
  --bind "${HF_HOME}:${HF_HOME}" \
  --bind "${HUGGINGFACE_HUB_CACHE}:${HUGGINGFACE_HUB_CACHE}" \
  --env DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX} \
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
