#!/bin/bash

#SBATCH --output=sbatch_log/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # Requesting 1 GPU
#SBATCH --constraint='geforce_gtx_titan_x|titan_xp'
#SBATCH --mem=50G
#SBATCH --time=01:00:00 # Set a time limit if required

# Load necessary modules and activate conda environment
cd /scratch_net/biwidl202/ppolydorou/project_edaps/edaps
source /scratch_net/biwidl202/ppolydorou/conda/etc/profile.d/conda.sh
conda activate edaps

# Set PYTHONPATH environment variable
PYTHONPATH="/scratch_net/biwidl202/ppolydorou/project_edaps/edaps:$PYTHONPATH"

# Log noteworthy information
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Start GPU monitoring in a background process
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5 > gpu_usage_${SLURM_JOB_ID}.csv &

# Store the background job's PID
GPU_MONITOR_PID=$!

# Run the main Python script
python run_experiments.py --exp 1

# Kill the GPU monitoring process after the main script is done
kill $GPU_MONITOR_PID
