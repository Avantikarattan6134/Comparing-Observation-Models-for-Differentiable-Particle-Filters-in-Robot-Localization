"""
evaluate_swin.py
----------------
Computes RMSE and ATE for DPF + Swin Transformer on nav01, nav02, nav03.
Run from repo root with dpf37 activated:
    python evaluate_swin.py
"""

import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')

from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state
from utils.exp_utils import get_default_hyperparams
from methods.dpf_swin import DPF


def compute_rmse(predicted_states, true_states):
    diff = predicted_states[:, :2] - true_states[:, :2]
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))


def compute_ate(predicted_states, true_states):
    diff = predicted_states[:, :2] - true_states[:, :2]
    return np.mean(np.sqrt(np.sum(diff ** 2, axis=-1)))


def evaluate(model_path, task, data_path, num_trials=10):
    test_data         = load_data(data_path=data_path, filename=task + '_test')
    noisy_test_data   = noisyfy_data(test_data)
    test_batch_iterator = make_batch_iterator(noisy_test_data, seq_len=50)

    hyperparams = get_default_hyperparams()

    tf.reset_default_graph()
    method = DPF(**hyperparams['global'])

    rmse_list, ate_list = [], []

    with tf.Session() as session:
        method.load(session, model_path)

        for i in range(num_trials):
            test_batch       = next(test_batch_iterator)
            true_states      = test_batch['s']
            test_batch_input = remove_state(test_batch, provide_initial_state=False)
            result           = method.predict(session, test_batch_input, **hyperparams['test'])

            predicted = np.array(result)
            true      = np.array(true_states)
            B, T, _   = predicted.shape
            for b in range(B):
                rmse_list.append(compute_rmse(predicted[b], true[b]))
                ate_list.append(compute_ate(predicted[b], true[b]))

    return {
        'rmse':     np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'ate':      np.mean(ate_list),
        'ate_std':  np.std(ate_list),
    }


def main():
    tasks     = ['nav01', 'nav02', 'nav03']
    data_path = 'data/100s'

    print("\nEvaluating DPF + Swin Transformer...")
    print("=" * 65)

    results = {}
    for task in tasks:
        model_path = f'models/{task}_swin'
        print(f"  {task}...", end=' ', flush=True)
        try:
            r = evaluate(model_path, task, data_path)
            results[task] = r
            print(f"RMSE={r['rmse']:.4f} ± {r['rmse_std']:.4f},  "
                  f"ATE={r['ate']:.4f} ± {r['ate_std']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[task] = None

    print("\n" + "=" * 65)
    print(f"  {'Task':<8} {'RMSE':<12} {'RMSE±std':<14} {'ATE':<12} {'ATE±std'}")
    print("-" * 65)
    for task in tasks:
        r = results.get(task)
        if r:
            print(f"  {task:<8} {r['rmse']:<12.4f} {r['rmse_std']:<14.4f} "
                  f"{r['ate']:<12.4f} {r['ate_std']:.4f}")
        else:
            print(f"  {task:<8} {'N/A'}")

    np.save('results_swin.npy', results)
    print("\nResults saved to results_swin.npy")


if __name__ == '__main__':
    main()
