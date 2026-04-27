# Copilot Instructions

# General Repository Environment Instructions

We are using AllianceCan (Compute Canada).
You currently have terminal access to the login node, whereas any SLURM script will be run on a compute node.
The login node does not have GPU access, and does not have a SLURM_TMPDIR (temporary directory created only for the SLURM run).
Computer nodes do not have internet access.

# Using the apptainer on the login node

To explore the container we are using:

`apptainer exec --nv /scratch/indrisch/thinking-in-space/containers/vsibench_eval.sif bash`