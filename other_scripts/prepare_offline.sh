#!/bin/bash
set -euo pipefail

printf "[%s] Starting offline preparation\n" "$(date --iso-8601=seconds)"

USER_ACCOUNT=$(whoami)

# Detect cluster based on terminal prompt or hostname
if [[ "$HOSTNAME" == *"rg"* ]]; then
    CLUSTER="RORQUAL"
elif [[ "$HOSTNAME" == *"trig"* ]]; then
    CLUSTER="TRILLIUM"
elif [[ "$HOSTNAME" == *"kn"* ]]; then
    CLUSTER="KILLARNEY"
elif [[ "$HOSTNAME" == *"g"* ]]; then
    CLUSTER="NIBI"
else
    CLUSTER="RORQUAL"
fi

if [[ "$USER_ACCOUNT" == "indrisch" ]]; then
    SYSCONFIG_DIR_PATH="${PWD%%thinking-in-space*}/thinking-in-space/scripts"
    export PYTHONPATH="${PYTHONPATH:-}:$SYSCONFIG_DIR_PATH"
    export HF_HOME="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HOME'))" 2>/dev/null || echo '')"
    export HF_HUB_CACHE="$(python3 -c "import sysconfigtool; print(sysconfigtool.read('${CLUSTER}', 'HF_HUB_CACHE'))" 2>/dev/null || echo '')"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
fi

PROJECT_ROOT="${PWD%%thinking-in-space*}/thinking-in-space"
CACHE_ROOT="${PROJECT_ROOT}/.cache/huggingface"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

if [ -f "/scratch/indrisch/TOKENS/huggingface/cvis-tmu-organization-token.txt" ]; then
    export HF_TOKEN=$(<"/scratch/indrisch/TOKENS/huggingface/cvis-tmu-organization-token.txt")
    printf "[%s] Loaded HF_TOKEN from file\n" "$(date --iso-8601=seconds)"
fi

MODEL_LIST="cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_426steps,cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_852steps,cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_5epochs,cvis-tmu/videor1-lora-sft-Scene30k_traineval_426steps,cvis-tmu/videor1-lora-sft-Scene30k_traineval_852steps,cvis-tmu/videor1-lora-sft-Scene30k_traineval_5epochs"
DATASET="nyu-visionx/VSI-Bench"

# Use python and huggingface_hub to thoroughly snapshot download
python3 -c "
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print('huggingface_hub is required. Try loading your python environment or module.')
    sys.exit(1)
import os

token = os.environ.get('HF_TOKEN')
dataset = '${DATASET}'
models = '${MODEL_LIST}'.split(',')

print(f'Downloading dataset: {dataset}')
snapshot_download(repo_id=dataset, repo_type='dataset', token=token, local_files_only=False)

for model in models:
    if model.strip():
        print(f'Downloading model: {model}')
        snapshot_download(repo_id=model, token=token, local_files_only=False)
"

printf "[%s] Offline preparation complete!\n" "$(date --iso-8601=seconds)"
