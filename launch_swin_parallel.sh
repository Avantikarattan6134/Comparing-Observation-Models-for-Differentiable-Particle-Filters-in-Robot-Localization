#!/bin/bash
# Submit Swin training (3 tasks in parallel) then queue eval to run after all finish.
# Usage:  bash launch_swin_parallel.sh

set -e

mkdir -p logs

TRAIN_JOB=$(sbatch --parsable run_swin_train_array.sh)
echo "Training array submitted  — job ID: $TRAIN_JOB  (tasks: nav01, nav02, nav03 in parallel)"

EVAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB run_swin_eval_array.sh)
echo "Eval array queued         — job ID: $EVAL_JOB  (starts after all training jobs finish)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs:          logs/swin_train_${TRAIN_JOB}_[1-3].out"
echo "               logs/swin_eval_${EVAL_JOB}_[1-3].out"
