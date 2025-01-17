#!/bin/bash

#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:geforce_gtx_titan_x:1
#SBATCH  --mem=30G

cd /scratch_net/biwidl202/ppolydorou/project_edaps/edaps
source /scratch_net/biwidl202/ppolydorou/conda/etc/profile.d/conda.sh
conda activate edaps
PYTHONPATH="</scratch_net/biwidl202/ppolydorou/project_edaps/edaps>:$PYTHONPATH" && export PYTHONPATH

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

python run_experiments.py --config configs/edaps/syn2cs_uda_warm_dfthings_rcs_croppl_a999_edaps_s0_debug.py
