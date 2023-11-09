#!/bin/bash

#SBATCH --output=sbatch_log/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # Requesting 1 gpu
#SBATCH --constraint='geforce_gtx_titan_x'

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

## nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1
nvidia-smi