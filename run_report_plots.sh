#!/bin/bash
#SBATCH --job-name=report_plots
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/report_plots_%j.out
#SBATCH --error=logs/report_plots_%j.err

module load cuda/10.2.89
module load python3.10-anaconda/2023.03
source ~/.bashrc
conda activate dpf

cd /home/savedula/DeepRob/differentiable-particle-filters

python generate_report_plots.py
