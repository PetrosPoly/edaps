#!/bin/bash

#SBATCH --output=sbatch_log/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # Requesting 1 gpu
#SBATCH --constraint='titan_xp'
#SBATCH --mem=50G

cd /scratch_net/biwidl202/ppolydorou/project_edaps/edaps
source /scratch_net/biwidl202/ppolydorou/conda/etc/profile.d/conda.sh
conda activate edaps
PYTHONPATH="</scratch_net/biwidl202/ppolydorou/project_edaps/edaps>:$PYTHONPATH" && export PYTHONPATH

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi
python run_experiments.py --exp 1