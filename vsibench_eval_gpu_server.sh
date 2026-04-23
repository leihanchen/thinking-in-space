#!/bin/bash

set -euo pipefail

printf "[%s] Starting VSI-Bench evaluation on host: %s\n" "$(date --iso-8601=seconds)" "${HOSTNAME}"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

SKIP_DEP_INSTALL="${SKIP_DEP_INSTALL:-1}"
OFFLINE_MODE="${OFFLINE_MODE:-0}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache/huggingface}"

if [[ "${OFFLINE_MODE}" == "1" ]]; then
  printf "[%s] Running in OFFLINE mode\n" "$(date --iso-8601=seconds)"
fi

printf "[%s] SKIP_DEP_INSTALL=%s (environment assumed ready)\n" "$(date --iso-8601=seconds)" "${SKIP_DEP_INSTALL}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf "[%s] ERROR: nvidia-smi not found; cannot detect local GPU count.\n" "$(date --iso-8601=seconds)" >&2
  exit 1
fi

GPU_COUNT_DETECTED="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^\s*$/d' | wc -l | tr -d ' ')"
if [[ -z "${GPU_COUNT_DETECTED}" || "${GPU_COUNT_DETECTED}" -lt 1 ]]; then
  printf "[%s] ERROR: no GPUs detected via nvidia-smi.\n" "$(date --iso-8601=seconds)" >&2
  exit 1
fi

# Configure runtime defaults. Override via environment variables when launching.
MODEL_LIST="${MODEL_LIST:-cvis-tmu/qwen2_5vl-7b-lora-sft-Scene30k_traineval_852steps_merged}"
GPU_COUNT="${GPU_COUNT:-${GPU_COUNT_DETECTED}}"
NUM_PROCESSES="${NUM_PROCESSES:-${GPU_COUNT}}"
BENCHMARK="${BENCHMARK:-vsibench}"
EVAL_SCRIPT="${EVAL_SCRIPT:-evaluate_all_in_one.sh}"

CPU_THREADS="${CPU_THREADS:-2}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"

# Only set HF_TOKEN if in online mode or if user explicitly provides it.
if [[ "${OFFLINE_MODE}" != "1" ]]; then
  export HF_TOKEN="${HF_TOKEN:-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}"
else
  export HF_OFFLINE_MODE=1
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  printf "[%s] HF_OFFLINE_MODE enabled - using pre-cached models only\n" "$(date --iso-8601=seconds)"
fi

export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

# Validate cache availability in offline mode.
if [[ "${OFFLINE_MODE}" == "1" ]]; then
  if [[ ! -d "${HUGGINGFACE_HUB_CACHE}" ]]; then
    printf "[%s] WARNING: Hugging Face cache may be empty: %s\n" "$(date --iso-8601=seconds)" "${HUGGINGFACE_HUB_CACHE}" >&2
    printf "       Make sure models are pre-cached before running in offline mode.\n" >&2
  fi
fi

printf "[%s] Launch command: %s --model %s --num_processes %s --benchmark %s\n" \
  "$(date --iso-8601=seconds)" "${EVAL_SCRIPT}" "${MODEL_LIST}" "${NUM_PROCESSES}" "${BENCHMARK}"
printf "[%s] GPU count from nvidia-smi: %s\n" "$(date --iso-8601=seconds)" "${GPU_COUNT_DETECTED}"

if [[ "${OFFLINE_MODE}" == "1" ]]; then
  printf "[%s] Preflight: verifying local Hugging Face cache paths\n" "$(date --iso-8601=seconds)"
  if ! python3 - <<'PY'
import os
from pathlib import Path

paths = {
    "HF_HOME": os.environ.get("HF_HOME", ""),
    "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
    "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", ""),
    "HF_MODULES_CACHE": os.environ.get("HF_MODULES_CACHE", ""),
}

for key, value in paths.items():
    print(f"{key}={value}")
    if not value or not Path(value).exists():
        raise SystemExit(f"missing cache path: {key}={value}")

hub_cache = Path(paths["HUGGINGFACE_HUB_CACHE"])
dataset_snapshot = hub_cache / "datasets--nyu-visionx--VSI-Bench" / "snapshots"
if not dataset_snapshot.exists():
    raise SystemExit(f"dataset snapshot directory not found under {dataset_snapshot}")

snapshot_entries = list(dataset_snapshot.glob("*/test.jsonl"))
if not snapshot_entries:
    raise SystemExit(f"VSI-Bench snapshot exists but test.jsonl is missing under {dataset_snapshot}")

modules_cache = Path(paths["HF_MODULES_CACHE"])
if not modules_cache.exists():
    raise SystemExit(f"HF modules cache path missing: {modules_cache}")

print("preflight ok")
PY
  then
    printf "[%s] ERROR: local preflight failed; cache is not visible where datasets expects it\n" "$(date --iso-8601=seconds)" >&2
    exit 1
  fi
fi

printf "[%s] Preflight: verifying local compiler toolchain for Triton/DeepSpeed\n" "$(date --iso-8601=seconds)"
if ! bash -lc '
if command -v gcc >/dev/null 2>&1; then
  gcc --version | head -n 1
  exit 0
fi
if [[ -n "${CC:-}" ]] && [[ -x "${CC}" ]]; then
  "${CC}" --version | head -n 1
  exit 0
fi
echo "No C compiler visible in container PATH and CC is not executable" >&2
exit 1
'; then
  printf "[%s] ERROR: local environment cannot find a usable C compiler (gcc/CC).\n" "$(date --iso-8601=seconds)" >&2
  exit 1
fi

printf "[%s] Preflight: verifying local nvcc path for DeepSpeed\n" "$(date --iso-8601=seconds)"
if ! bash -lc '
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
if [[ -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
  "${CUDA_HOME}/bin/nvcc" --version | head -n 1
  exit 0
fi
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | head -n 1
  exit 0
fi
echo "nvcc is not visible in container" >&2
exit 1
'; then
  printf "[%s] ERROR: local environment cannot find nvcc. Check CUDA_HOME and CUDA toolkit installation.\n" "$(date --iso-8601=seconds)" >&2
  exit 1
fi

start_time=$(date +%s)
printf "[%s] Running evaluation directly in local environment.\n" "$(date --iso-8601=seconds)"

# Optional GPU pinning example:
#   CUDA_VISIBLE_DEVICES=0,1 ./vsibench_eval_gpu_server.sh
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  printf "[%s] ERROR: evaluation script not found: %s\n" "$(date --iso-8601=seconds)" "${EVAL_SCRIPT}" >&2
  exit 1
fi

bash "${EVAL_SCRIPT}" \
  --model "${MODEL_LIST}" \
  --num_processes "${NUM_PROCESSES}" \
  --benchmark "${BENCHMARK}"

end_time=$(date +%s)
duration=$((end_time - start_time))
printf "Inference time cost: %d seconds\n" "${duration}"
printf "[%s] Evaluation completed successfully\n" "$(date --iso-8601=seconds)"
