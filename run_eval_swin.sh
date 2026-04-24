#!/bin/bash
#SBATCH --job-name=eval_swin
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_swin_%j.out
#SBATCH --error=logs/eval_swin_%j.err

module load cuda/10.2.89
module load python3.10-anaconda/2023.03
source ~/.bashrc
conda activate dpf

cd /home/savedula/DeepRob/differentiable-particle-filters

python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from evaluate_swin import main
main()
"
