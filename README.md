# Comparing Observation Models for Differentiable Particle Filters in Robot Localization

**ROB599 Final Project — University of Michigan**  
Avantika Rattan, Karthik Vairavan, Saumit Vedula

---

## Overview

Systematic comparison of **6 observation model architectures** within the Differentiable Particle Filter (DPF) framework for robot localization in maze environments. We evaluate CNN, ViT, ResNet, ResNet+LSTM, Swin Transformer, and InEKF+ResNet across three navigation tasks of increasing complexity.

---

## Models Compared

| Model | Filter Type | Encoder | Description |
|---|---|---|---|
| DPF + CNN | Particle Filter | Shallow CNN | Original DPF baseline |
| DPF + ViT | Particle Filter | Vision Transformer | 4-block ViT, 4×4 patches |
| DPF + ResNet | Particle Filter | ResNet-18 style | 3-stage residual encoder |
| DPF + ResNet+LSTM | Particle Filter | ResNet + LSTM | ResNet + 2-layer LSTM |
| DPF + Swin Transformer | Particle Filter | Swin Transformer | 2-stage hierarchical, W=3 |
| InEKF + ResNet | Kalman Filter | ResNet | SE(2) Lie group filter |

---

## Key Results (RMSE in pixels, lower is better)

| Model | Filter | nav01 (Easy) | nav02 (Medium) | nav03 (Hard) |
|---|---|---|---|---|
| CNN | Particle | 85.41 | 141.61 | 240.57 |
| ViT | Particle | 83.65 | 155.28 | 308.98 |
| **ResNet** | **Particle** | **67.54** | **104.27** | **184.43** |
| ResNet+LSTM | Particle | 76.68 | 138.81 | 263.08 |
| Swin Transformer | Particle | 77.83 | 149.58 | 269.95 |
| InEKF+ResNet | Kalman | 326.47 | 497.89 | 595.14 |

---

## Key Findings

1. **ResNet achieves best accuracy** — reduces RMSE by up to 26% vs CNN baseline. Skip connections enable better gradient flow and feature extraction.

2. **ViT underperforms on small images** — global attention is less effective than local convolutional features at 24×24 resolution.

3. **Swin Transformer is competitive on easy tasks but degrades on harder ones** — hierarchical multi-scale design requires sufficient input resolution. At 24×24, only two patch merging stages are feasible, limiting its advantages.

4. **LSTM does not improve over ResNet alone** — DPF's particle propagation already captures sufficient temporal context, making LSTM redundant.

5. **Particle filters outperform Kalman filter by 4–5×** — InEKF's Gaussian assumption fails for multimodal posteriors in visually ambiguous maze navigation.

---

## Tasks

| Task | Difficulty | Description |
|---|---|---|
| nav01 | Easy | Simple maze with limited branching |
| nav02 | Medium | Medium complexity maze |
| nav03 | Hard | Complex maze with many similar-looking corridors |

Each task: 1000 training episodes × 99 timesteps, 24×24×3 RGB observations.

---

## Repository Structure

```
├── methods/
│   ├── dpf_original.py       # CNN baseline DPF
│   ├── dpf_vit.py            # ViT DPF
│   ├── dpf_resnet.py         # ResNet DPF
│   ├── dpf_resnet_lstm.py    # ResNet+LSTM DPF
│   └── dpf_swin.py           # Swin Transformer DPF
├── models/
│   ├── vit_encoder_tf1.py        # ViT encoder (TF1)
│   ├── resnet_encoder_tf1.py     # ResNet encoder (TF1)
│   ├── resnet_lstm_encoder_tf1.py # ResNet+LSTM encoder (TF1)
│   └── inekf_resnet_tf1.py       # InEKF+ResNet (TF1)
├── evaluate_4models.py       # Evaluate CNN/ViT/ResNet/ResNet+LSTM
├── evaluate_5models.py       # Evaluate all 5 DPF models + InEKF
├── generate_plots.py         # Generate trajectory visualization plots
├── results_summary.txt       # Full numerical results
└── report_plots/             # All figures used in the report
```

---

## Environment

- Python 3.7, TensorFlow 1.15, Sonnet 1.36
- CUDA 10.2, cuDNN 7.6.5
- Trained on **Great Lakes HPC** (University of Michigan ARC)
  - DPF models: NVIDIA V100 (gpu partition)
  - InEKF: NVIDIA RTX PRO 6000 (gpu-rtx6000 partition)

---

## Training

```bash
# Activate environment
module load cuda/10.2.89
export LD_LIBRARY_PATH=/home/arattan/cudnn7:$LD_LIBRARY_PATH
module load python3.10-anaconda/2023.03
conda activate dpf37

# Train each model (example: ResNet)
cp methods/dpf_resnet.py methods/dpf.py
python -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'experiments')
from simple import train_dpf
train_dpf(task='nav01', data_path='data/100s', model_path='models/nav01_resnet')
"
```

---

## Citation

```bibtex
@article{jonschkowski18,
  title={Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors},
  author={Jonschkowski, Rico and Rastogi, Divyam and Brock, Oliver},
  booktitle={Proceedings of Robotics: Science and Systems (RSS)},
  year={2018}
}
```
