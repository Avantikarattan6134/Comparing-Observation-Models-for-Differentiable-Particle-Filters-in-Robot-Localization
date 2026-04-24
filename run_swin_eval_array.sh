#!/bin/bash
#SBATCH --job-name=dpf_swin_eval
#SBATCH --array=1-3
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/swin_eval_%A_%a.out
#SBATCH --error=logs/swin_eval_%A_%a.err

module load cuda/10.2.89
module load python3.10-anaconda/2023.03
source ~/.bashrc
conda activate dpf

cd /home/savedula/DeepRob/differentiable-particle-filters

TASKS=("" "nav01" "nav02" "nav03")
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "Evaluating Swin DPF on task: $TASK"

python -c "
import sys, numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from evaluate_swin import evaluate, compute_rmse, compute_ate

task      = '${TASK}'
data_path = 'data/100s'
model_path = 'models/' + task + '_swin'

r = evaluate(model_path, task, data_path)
print(f'  {task}  RMSE={r[\"rmse\"]:.4f} +- {r[\"rmse_std\"]:.4f}   ATE={r[\"ate\"]:.4f} +- {r[\"ate_std\"]:.4f}')
np.save(f'results_swin_{task}.npy', r)
print(f'  Saved to results_swin_{task}.npy')
"
