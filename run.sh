#!/bin/bash

SESSION=$1
SCRIPT=$2
GPU_ID=$3

screen -S "$SESSION" -dm bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate py3
jupyter nbconvert --to script \"$SCRIPT.ipynb\"
CUDA_VISIBLE_DEVICES=$GPU_ID python \"$SCRIPT.py\"
exec bash
"