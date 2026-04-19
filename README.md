# Differentiable Particle Filters - Observation Model Comparison

ROB599 Final Project - University of Michigan

## Overview
Comparison of 5 observation models in the Differentiable Particle Filter (DPF) framework for robot localization in maze environments.

## Models Compared
| Model | Filter Type | Encoder |
|---|---|---|
| DPF + CNN | Particle Filter | Shallow CNN (baseline) |
| DPF + ViT | Particle Filter | Vision Transformer |
| DPF + ResNet | Particle Filter | ResNet-18 style |
| DPF + ResNet+LSTM | Particle Filter | ResNet + LSTM |
| InEKF + ResNet | Kalman Filter | ResNet |

## Key Results
ResNet encoder achieves best performance across all tasks.
InEKF underperforms DPF variants due to multimodal posterior distributions in maze navigation.

## Tasks
- nav01: Simple maze (easiest)
- nav02: Medium maze
- nav03: Complex maze (hardest)

## Environment
- TensorFlow 1.15, Sonnet 1.36, Python 3.7
- Trained on Great Lakes HPC (V100 GPUs)
