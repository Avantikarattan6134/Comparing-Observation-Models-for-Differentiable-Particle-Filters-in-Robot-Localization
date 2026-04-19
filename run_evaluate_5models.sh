#!/bin/bash
#SBATCH --job-name=eval_5models
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_5models_%j.out
#SBATCH --error=logs/eval_5models_%j.err

module load cuda/10.2.89
export LD_LIBRARY_PATH=/home/arattan/cudnn7:$LD_LIBRARY_PATH
module load python3.10-anaconda/2023.03
source ~/.bashrc
conda activate dpf37

cd /scratch/engin_root/engin1/arattan/differentiable-particle-filters
python evaluate_5models.py
