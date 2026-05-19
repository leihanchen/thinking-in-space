#!/bin/bash

# run this script on the login node

# --- RUN 2 ---

# _MODEL_LIST=(
#   cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_5epochs_merged
#   cvis-tmu/videor1-lora-sft-Scene30k_traineval_5epochs_merged
#   cvis-tmu/Qwen2.5-VL-7B-COT-SFT
#   cvis-tmu/Video-R1-7B
# )
# _ML=''
# for i in "${_MODEL_LIST[@]}"; do 
#   _ML="${_ML}${i}," 
# done

# MODEL_LIST=${_ML%,}  # remove trailing comma
# export MODEL_LIST
# export OVERLAY_IMG="/scratch/indrisch/thinking-in-space/containers/apptainer-overlay_2.img"

# sbatch vsibench_eval_compute_canada.sh

# # --- RUN 3 ---

# _MODEL_LIST=(
#   cvis-tmu/videor1-lora-sft-Scene30k_traineval_426steps_merged
#   cvis-tmu/videor1-lora-sft-Scene30k_traineval_852steps_merged
# )
# _ML=''
# for i in "${_MODEL_LIST[@]}"; do 
#   _ML="${_ML}${i}," 
# done

# MODEL_LIST=${_ML%,}  # remove trailing comma
# export MODEL_LIST
# export OVERLAY_IMG="/scratch/indrisch/thinking-in-space/containers/apptainer-overlay_3.img"

# sbatch vsibench_eval_compute_canada.sh


# # --- RUN 4 ---

# _MODEL_LIST=(
#   cvis-tmu/Qwen2.5-VL-7B-COT-SFT
#   cvis-tmu/Video-R1-7B
# )
# _ML=''
# for i in "${_MODEL_LIST[@]}"; do 
#   _ML="${_ML}${i}," 
# done

# MODEL_LIST=${_ML%,}  # remove trailing comma
# export MODEL_LIST
# export OVERLAY_IMG="/scratch/indrisch/thinking-in-space/containers/apptainer-overlay_4.img"

# sbatch vsibench_eval_compute_canada.sh

# # --- RUN 5 ---

# _MODEL_LIST=(
#   cvis-tmu/videor1sft-lora-sft-Scene30k_traineval_852steps_merged
# )
# _ML=''
# for i in "${_MODEL_LIST[@]}"; do 
#   _ML="${_ML}${i}," 
# done

# MODEL_LIST=${_ML%,}  # remove trailing comma
# export MODEL_LIST
# export OVERLAY_IMG="/scratch/indrisch/thinking-in-space/containers/apptainer-overlay_5.img"

# sbatch vsibench_eval_compute_canada.sh


# # --- RUN 6 ---

# _MODEL_LIST=(
#   cvis-tmu/videor1-lora-sft-Scene30k_traineval_5epochs_merged
# )
# _ML=''
# for i in "${_MODEL_LIST[@]}"; do 
#   _ML="${_ML}${i}," 
# done

# MODEL_LIST=${_ML%,}  # remove trailing comma
# export MODEL_LIST
# export OVERLAY_IMG="/scratch/indrisch/thinking-in-space/containers/apptainer-overlay_6.img"

# sbatch vsibench_eval_compute_canada.sh

# # --- RUN 7 ---

# _MODEL_LIST=(
#   cvis-tmu/videor1-lora-sft-Scene30k_traineval_852steps_merged
# )
# _ML=''
# for i in "${_MODEL_LIST[@]}"; do 
#   _ML="${_ML}${i}," 
# done

# MODEL_LIST=${_ML%,}  # remove trailing comma
# export MODEL_LIST
# export OVERLAY_IMG="/scratch/indrisch/thinking-in-space/containers/apptainer-overlay_7.img"

# sbatch vsibench_eval_compute_canada.sh


# --- RUN 8 ---

# _MODEL_LIST=(
#   cvis-tmu/Video-R1-7B
# )
# _ML=''
# for i in "${_MODEL_LIST[@]}"; do 
#   _ML="${_ML}${i}," 
# done

# MODEL_LIST=${_ML%,}  # remove trailing comma
# export MODEL_LIST
# export OVERLAY_IMG="/scratch/indrisch/thinking-in-space/containers/apptainer-overlay_8.img"

# sbatch vsibench_eval_compute_canada.sh

# --- RUN 9 ---

_MODEL_LIST=(
  cvis-tmu/Qwen2.5-VL-7B-COT-SFT
)
_ML=''
for i in "${_MODEL_LIST[@]}"; do 
  _ML="${_ML}${i}," 
done

MODEL_LIST=${_ML%,}  # remove trailing comma
export MODEL_LIST
export OVERLAY_IMG="/scratch/indrisch/thinking-in-space/containers/apptainer-overlay_9.img"

sbatch vsibench_eval_compute_canada.sh