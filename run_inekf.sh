#!/bin/bash
#SBATCH --job-name=dpf_inekf
#SBATCH --account=engin1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/inekf_%j.out
#SBATCH --error=logs/inekf_%j.err

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
from models.inekf_resnet_tf1 import train_inekf
print('=== nav01 ===')
train_inekf(task='nav01', data_path='data/100s', model_path='models/nav01_inekf')
print('=== nav02 ===')
train_inekf(task='nav02', data_path='data/100s', model_path='models/nav02_inekf')
print('=== nav03 ===')
train_inekf(task='nav03', data_path='data/100s', model_path='models/nav03_inekf')
"
