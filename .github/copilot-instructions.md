# Copilot Instructions

# Creating a Virtual Environment:

To create a virtual environment on AllianceCan systems, you must first load relevant modules, and then create an environment and install packages using the existing wheels.

For example, use the following series of bash commands:

```bash
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
virtualenv --no-download ENV
source ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index h5py
```

# General Repository Environment Instructions

We are using AllianceCan (Compute Canada).
You currently have terminal access to the login node, whereas any SLURM script will be run on a compute node.
The login node does not have GPU access, and does not have a SLURM_TMPDIR (temporary directory created only for the SLURM run).
Computer nodes do not have internet access.

# Using the apptainer on the login node

To explore the container we are using, including checking its environment:

`apptainer exec --fakeroot --nv --overlay /scratch/indrisch/thinking-in-space/containers/apptainer-overlay.img -C /scratch/indrisch/thinking-in-space/containers/vsibench_eval.sif bash`