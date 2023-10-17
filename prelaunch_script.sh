#!/bin/bash

cd /scratch_net/biwidl202/ppolydorou/project_edaps/edaps
source /scratch_net/biwidl202/ppolydorou/conda/etc/profile.d/conda.sh
conda activate edaps
PYTHONPATH="</scratch_net/biwidl202/ppolydorou/project_edaps/edaps>:$PYTHONPATH" && export PYTHONPATH

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"