#!/bin/bash
#SBATCH --job-name=dpf_swin_train
#SBATCH --array=1-3
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/swin_train_%A_%a.out
#SBATCH --error=logs/swin_train_%A_%a.err

module load cuda/10.2.89
module load python3.10-anaconda/2023.03
source ~/.bashrc
conda activate dpf

cd /home/savedula/DeepRob/differentiable-particle-filters

TASKS=("" "nav01" "nav02" "nav03")
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "Training Swin DPF on task: $TASK"
cp methods/dpf_swin.py methods/dpf.py

python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from simple import train_dpf
task = '${TASK}'
train_dpf(task=task, data_path='data/100s', model_path='models/' + task + '_swin', plot=False)
"
