"""
evaluate_models.py
------------------
Computes RMSE and ATE for CNN, ViT, ResNet on nav01, nav02, nav03.
Run from repo root with dpf37 activated:
    python evaluate_models.py
"""

import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')

from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state
from utils.exp_utils import get_default_hyperparams


def compute_rmse(predicted_states, true_states):
    """
    Root Mean Square Error of position (x, y).
    Ignores theta (index 2).

    Args:
        predicted_states : (T, 2) or (T, 3) predicted x,y,theta
        true_states      : (T, 2) or (T, 3) ground truth
    Returns:
        rmse : scalar
    """
    diff = predicted_states[:, :2] - true_states[:, :2]
    mse  = np.mean(np.sum(diff ** 2, axis=-1))
    return np.sqrt(mse)


def compute_ate(predicted_states, true_states):
    """
    Absolute Trajectory Error — mean Euclidean distance
    between predicted and true positions over the trajectory.

    Args:
        predicted_states : (T, 2+) array
        true_states      : (T, 2+) array
    Returns:
        ate : scalar
    """
    diff = predicted_states[:, :2] - true_states[:, :2]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    return np.mean(distances)


def evaluate(model_path, dpf_module, task, data_path, num_trials=10):
    """
    Run evaluation and compute RMSE + ATE.

    Args:
        model_path  : path to saved model
        dpf_module  : DPF class to use
        task        : 'nav01', 'nav02', 'nav03'
        data_path   : path to data folder
        num_trials  : number of test batches to average over
    Returns:
        dict with rmse, ate, rmse_std, ate_std
    """
    # Load test data
    test_data = load_data(data_path=data_path, filename=task + '_test')
    noisy_test_data = noisyfy_data(test_data)
    test_batch_iterator = make_batch_iterator(noisy_test_data, seq_len=50)

    hyperparams = get_default_hyperparams()

    tf.reset_default_graph()
    method = dpf_module(**hyperparams['global'])

    rmse_list = []
    ate_list  = []

    with tf.Session() as session:
        method.load(session, model_path)

        for i in range(num_trials):
            test_batch       = next(test_batch_iterator)
            true_states      = test_batch['s']           # (B, T, 3)
            test_batch_input = remove_state(test_batch, provide_initial_state=False)

            # predict returns estimated states (B, T, 3)
            result = method.predict(
                session, test_batch_input, **hyperparams['test']
            )

            # result shape: (B, T, 3) — weighted mean of particles
            predicted = np.array(result)    # (B, T, 3)
            true      = np.array(true_states)  # (B, T, 3)

            # Compute metrics per sequence then average
            B, T, _ = predicted.shape
            for b in range(B):
                rmse = compute_rmse(predicted[b], true[b])
                ate  = compute_ate(predicted[b], true[b])
                rmse_list.append(rmse)
                ate_list.append(ate)

    return {
        'rmse':     np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'ate':      np.mean(ate_list),
        'ate_std':  np.std(ate_list),
    }


def main():
    tasks     = ['nav01', 'nav02', 'nav03']
    data_path = 'data/100s'

    # ── CNN ──────────────────────────────────────────────────────────────────
    print("\nEvaluating CNN...")
    import importlib, methods.dpf as dpf_mod
    # Use original CNN dpf
    import shutil
    shutil.copy('methods/dpf_original.py', 'methods/dpf.py')
    importlib.reload(dpf_mod)
    from methods.dpf import DPF as DPF_CNN

    cnn_results = {}
    for task in tasks:
        print(f"  {task}...", end=' ', flush=True)
        try:
            res = evaluate(f'models/{task}_cnn', DPF_CNN, task, data_path)
            cnn_results[task] = res
            print(f"RMSE={res['rmse']:.4f}, ATE={res['ate']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            cnn_results[task] = None

    # ── ViT ──────────────────────────────────────────────────────────────────
    print("\nEvaluating ViT...")
    shutil.copy('methods/dpf_vit.py', 'methods/dpf.py')
    importlib.reload(dpf_mod)
    from methods.dpf import DPF as DPF_ViT

    vit_results = {}
    for task in tasks:
        print(f"  {task}...", end=' ', flush=True)
        try:
            res = evaluate(f'models/{task}_vit', DPF_ViT, task, data_path)
            vit_results[task] = res
            print(f"RMSE={res['rmse']:.4f}, ATE={res['ate']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            vit_results[task] = None

    # ── ResNet ────────────────────────────────────────────────────────────────
    print("\nEvaluating ResNet...")
    shutil.copy('methods/dpf_resnet.py', 'methods/dpf.py')
    importlib.reload(dpf_mod)
    from methods.dpf import DPF as DPF_ResNet

    resnet_results = {}
    for task in tasks:
        print(f"  {task}...", end=' ', flush=True)
        try:
            res = evaluate(f'models/{task}_resnet', DPF_ResNet, task, data_path)
            resnet_results[task] = res
            print(f"RMSE={res['rmse']:.4f}, ATE={res['ate']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            resnet_results[task] = None

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTS COMPARISON")
    print("="*70)
    print(f"{'Model':<10} {'Task':<8} {'RMSE':<12} {'RMSE±std':<14} {'ATE':<12} {'ATE±std':<12}")
    print("-"*70)

    all_results = {'CNN': cnn_results, 'ViT': vit_results, 'ResNet': resnet_results}

    for model_name, results in all_results.items():
        for task in tasks:
            r = results.get(task)
            if r:
                print(f"{model_name:<10} {task:<8} "
                      f"{r['rmse']:<12.4f} {r['rmse_std']:<14.4f} "
                      f"{r['ate']:<12.4f} {r['ate_std']:<12.4f}")
            else:
                print(f"{model_name:<10} {task:<8} {'N/A':<12} {'N/A':<14} {'N/A':<12} {'N/A':<12}")
        print()

    # Save results to file
    np.save('results_comparison.npy', {
        'CNN':    cnn_results,
        'ViT':    vit_results,
        'ResNet': resnet_results,
    })
    print("\nResults saved to results_comparison.npy")


if __name__ == '__main__':
    main()
