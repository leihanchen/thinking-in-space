#!/bin/bash

# run this script on the login node

_MODEL_LIST=(
  cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_5epochs_merged
  cvis-tmu/videor1-lora-sft-Scene30k_traineval_5epochs_merged
  cvis-tmu/Qwen2.5-VL-7B-COT-SFT
  cvis-tmu/Video-R1-7B
)
_ML=''
for i in "${_MODEL_LIST[@]}"; do 
  _ML="${_ML}${i}," 
done

MODEL_LIST=${_ML%,}  # remove trailing comma
export MODEL_LIST
export OVERLAY_IMG="/scratch/indrisch/thinking-in-space/containers/apptainer-overlay_2.img"

sbatch vsibench_eval_compute_canada.sh