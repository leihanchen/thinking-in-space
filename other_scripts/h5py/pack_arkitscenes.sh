#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=0-00:20:00
#SBATCH --output=out/%x-%N-%j.out

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

#virtualenv --no-download ENV
source ENV/bin/activate
#pip install --no-index --upgrade pip

#pip install --no-index h5py numpy torch tqdm

python pack_arkitscenes.py --verify --delete --workers 192 --dataset "/scratch/indrisch/ARKitScenes_data/3dod/Validation/"

deactivate
#rm -rf ENV
