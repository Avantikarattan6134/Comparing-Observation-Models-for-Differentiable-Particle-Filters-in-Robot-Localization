"""
generate_report_plots.py
------------------------
Regenerates all report_plots/ images, including the Swin encoder.

Step 1 – collect trajectories: calls collect_one_traj.py as a subprocess for
each (model, task) pair so TF resets cleanly between runs.  Results are cached
in report_plots/traj_data/<model>_<task>.npz and reused on re-runs.

Step 2 – plot: reads only from cache; no TF dependency in this file.

Outputs:
  report_plots/{model}_nav{01,02,03}_trajectory.png  — individual plots
  report_plots/comparison_nav{01,02,03}.png           — 5-column comparison

Run from repo root with dpf37 activated:
    python generate_report_plots.py
"""

import sys, os, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')

from utils.plotting_utils import plot_maze

os.makedirs('report_plots/traj_data', exist_ok=True)
os.makedirs('report_plots', exist_ok=True)

TASKS  = ['nav01', 'nav02', 'nav03']
MODELS = ['CNN', 'ViT', 'ResNet', 'ResNet+LSTM', 'Swin']

# ── Step 1: collect trajectories ────────────────────────────────────────────

for model_name in MODELS:
    for task in TASKS:
        cache = f'report_plots/traj_data/{model_name}_{task}.npz'
        if os.path.exists(cache):
            print(f'  CACHED  {model_name}/{task}', flush=True)
            continue
        print(f'  Collecting {model_name}/{task} ...', flush=True)
        ret = subprocess.run(
            [sys.executable, 'collect_one_traj.py', model_name, task],
            capture_output=True, text=True
        )
        if ret.returncode != 0:
            print(f'    ERROR:\n{ret.stderr[-800:]}')
        else:
            print(f'    {ret.stdout.strip()}')

# ── Step 2: load cache ───────────────────────────────────────────────────────

all_trajs = {}
for model_name in MODELS:
    all_trajs[model_name] = {}
    for task in TASKS:
        cache = f'report_plots/traj_data/{model_name}_{task}.npz'
        if os.path.exists(cache):
            d = np.load(cache)
            all_trajs[model_name][task] = (d['pred'], d['true'])
        else:
            all_trajs[model_name][task] = None

# ── helpers ──────────────────────────────────────────────────────────────────

def pos_error(pred, true):
    diff = pred[:, :, :2] - true[:, :, :2]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist.mean(axis=0), dist.std(axis=0)


def draw_trajectory_ax(ax, pred, true, task):
    plot_maze(maze=task, ax=ax, linewidth=1.0)
    for i in range(pred.shape[0]):
        ax.plot(true[i, :, 0], true[i, :, 1], 'r-', linewidth=0.8, alpha=0.7)
        ax.plot(pred[i, :, 0], pred[i, :, 1], 'g-', linewidth=0.8, alpha=0.7)
    ax.plot([], [], 'r-', label='True')
    ax.plot([], [], 'g-', label='Pred')
    ax.legend(fontsize=6, loc='upper right')
    ax.set_aspect('equal')
    ax.axis('off')


def draw_error_ax(ax, pred, true):
    mean_err, std_err = pos_error(pred, true)
    T = np.arange(len(mean_err))
    avg_rmse = np.sqrt(np.mean(np.sum((pred[:, :, :2] - true[:, :, :2]) ** 2, axis=-1)))
    ax.fill_between(T, mean_err - std_err, mean_err + std_err, alpha=0.3, color='#5C85D6')
    ax.plot(T, mean_err, color='#2b4fa0', linewidth=1.5)
    ax.set_xlabel('Timestep', fontsize=7)
    ax.set_ylabel('Error (pixels)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3)
    return avg_rmse


# ── Step 3: individual trajectory plots ─────────────────────────────────────

for model_name in MODELS:
    for task in TASKS:
        entry = all_trajs[model_name].get(task)
        if entry is None:
            print(f'  SKIP (no data): {model_name}/{task}')
            continue
        pred, true = entry

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{model_name} — {task}', fontweight='bold')

        draw_trajectory_ax(axes[0], pred, true, task)
        axes[0].set_title('Trajectory comparison', fontsize=9)

        avg_rmse = draw_error_ax(axes[1], pred, true)
        axes[1].set_title('Localization error over time', fontsize=9)
        axes[1].axhline(avg_rmse, color='red', linestyle='--', linewidth=1,
                        label=f'Avg RMSE: {avg_rmse:.1f}px')
        axes[1].legend(fontsize=7)

        plt.tight_layout()
        out = f'report_plots/{model_name}_{task}_trajectory.png'
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'  Saved {out}')

# ── Step 4: comparison plots (one per task, one column per model) ────────────

for task in TASKS:
    n = len(MODELS)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle(f'DPF Model Comparison — {task}', fontweight='bold', fontsize=13)

    for col, model_name in enumerate(MODELS):
        entry = all_trajs[model_name].get(task)

        ax_traj = axes[0, col]
        if entry is not None:
            pred, true = entry
            draw_trajectory_ax(ax_traj, pred, true, task)
            avg_rmse = np.sqrt(np.mean(
                np.sum((pred[:, :, :2] - true[:, :, :2]) ** 2, axis=-1)))
            ax_traj.set_title(f'{model_name}\nRMSE={avg_rmse:.1f}px', fontsize=8)
        else:
            ax_traj.set_title(f'{model_name}\nN/A', fontsize=8)
            ax_traj.axis('off')

        ax_err = axes[1, col]
        if entry is not None:
            avg_rmse = draw_error_ax(ax_err, pred, true)
            ax_err.set_title(f'Mean error: {avg_rmse:.1f}px', fontsize=8)
        else:
            ax_err.axis('off')

    plt.tight_layout()
    out = f'report_plots/comparison_{task}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved {out}')

print('\nDone.')
