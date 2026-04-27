#!/bin/bash
# run on a login node

module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
module load apptainer

virtualenv temp_env
source temp_env/bin/activate
pip install --upgrade pip

mkdir -p /scratch/indrisch/apptainer_cachedir
mkdir -p /scratch/indrisch/apptainer_tmpdir
# export APPTAINER_CACHEDIR=/scratch/indrisch/apptainer_cachedir
# export APPTAINER_TMPDIR=/scratch/indrisch/apptainer_tmpdir

mkdir -p /scratch/indrisch/thinking-in-space/containers/
apptainer build --fakeroot /scratch/indrisch/thinking-in-space/containers/vsibench_eval.sif vsibench_eval.def

rm -r temp_env