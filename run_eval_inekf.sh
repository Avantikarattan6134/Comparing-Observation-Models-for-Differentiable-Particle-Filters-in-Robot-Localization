#!/bin/bash
#SBATCH --job-name=eval_inekf
#SBATCH --account=engin1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_inekf_%j.out
#SBATCH --error=logs/eval_inekf_%j.err

module load cuda/10.2.89
export LD_LIBRARY_PATH=/home/arattan/cudnn7:$LD_LIBRARY_PATH
module load python3.10-anaconda/2023.03
source ~/.bashrc
conda activate dpf37

cd /scratch/engin_root/engin1/arattan/differentiable-particle-filters

python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from models.inekf_resnet_tf1 import test_inekf
print('=== nav01 ===')
test_inekf(task='nav01', data_path='data/100s', model_path='models/nav01_inekf')
print('=== nav02 ===')
test_inekf(task='nav02', data_path='data/100s', model_path='models/nav02_inekf')
print('=== nav03 ===')
test_inekf(task='nav03', data_path='data/100s', model_path='models/nav03_inekf')
"
