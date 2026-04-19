#!/bin/bash
#SBATCH --job-name=eval_cnn
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_cnn_%j.out
#SBATCH --error=logs/eval_cnn_%j.err

module load cuda/10.2.89
export LD_LIBRARY_PATH=/home/arattan/cudnn7:$LD_LIBRARY_PATH
module load python3.10-anaconda/2023.03
source ~/.bashrc
conda activate dpf37

cd /scratch/engin_root/engin1/arattan/differentiable-particle-filters
cp methods/dpf_original.py methods/dpf.py

python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from simple import test_dpf
print('=== nav01 ===')
test_dpf(task='nav01', data_path='data/100s', model_path='models/nav01_cnn')
print('=== nav02 ===')
test_dpf(task='nav02', data_path='data/100s', model_path='models/nav02_cnn')
print('=== nav03 ===')
test_dpf(task='nav03', data_path='data/100s', model_path='models/nav03_cnn')
"
