"""
evaluate_5models.py
-------------------
Computes RMSE and ATE for all 5 models:
  CNN, ViT, ResNet, ResNet+LSTM, InEKF+ResNet
Run from repo root:  python evaluate_5models.py
"""
import sys, numpy as np, tensorflow as tf, shutil
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state
from utils.exp_utils import get_default_hyperparams


def compute_rmse(p, t):
    return np.sqrt(np.mean(np.sum((p[:, :2] - t[:, :2]) ** 2, axis=-1)))


def compute_ate(p, t):
    return np.mean(np.sqrt(np.sum((p[:, :2] - t[:, :2]) ** 2, axis=-1)))


def evaluate_dpf_model(dpf_file, model_path, task, data_path, num_trials=10):
    """Evaluate DPF-based models (CNN, ViT, ResNet, ResNet+LSTM)."""
    shutil.copy(dpf_file, 'methods/dpf.py')
    if 'methods.dpf' in sys.modules:
        del sys.modules['methods.dpf']
    from methods.dpf import DPF

    test_data = load_data(data_path=data_path, filename=task + '_test')
    noisy     = noisyfy_data(test_data)
    iterator  = make_batch_iterator(noisy, seq_len=50)
    hp        = get_default_hyperparams()

    tf.reset_default_graph()
    method = DPF(**hp['global'])

    rmse_list, ate_list = [], []
    with tf.Session() as sess:
        method.load(sess, model_path)
        for i in range(num_trials):
            batch    = next(iterator)
            true_s   = batch['s']
            batch_in = remove_state(batch, provide_initial_state=False)
            result   = method.predict(sess, batch_in, **hp['test'])
            pred     = np.array(result)
            true     = np.array(true_s)
            B, T, _  = pred.shape
            for b in range(B):
                rmse_list.append(compute_rmse(pred[b], true[b]))
                ate_list.append(compute_ate(pred[b], true[b]))

    return {'rmse':     np.mean(rmse_list),
            'rmse_std': np.std(rmse_list),
            'ate':      np.mean(ate_list),
            'ate_std':  np.std(ate_list)}


def evaluate_inekf_model(model_path, task, data_path, num_trials=10):
    """Evaluate InEKF+ResNet model."""
    from models.inekf_resnet_tf1 import InEKFTrainer

    test_data = load_data(data_path=data_path, filename=task + '_test')
    noisy     = noisyfy_data(test_data)
    iterator  = make_batch_iterator(noisy, seq_len=50)

    tf.reset_default_graph()
    trainer = InEKFTrainer()

    rmse_list, ate_list = [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        stats = np.load(f'{model_path}/statistics.npz', allow_pickle=True)
        means = stats['means'].item()
        stds  = stats['stds'].item()
        trainer.saver.restore(sess, f'{model_path}/best_validation')

        for i in range(num_trials):
            batch = next(iterator)
            obs   = (batch['o'] - means['o']) / (stds['o'] + 1e-8)
            act   = batch['a'] / (stds['a'] + 1e-8)

            estimates = sess.run(
                trainer.state_estimates,
                feed_dict={
                    trainer.obs_ph:    obs,
                    trainer.action_ph: act,
                    trainer.state_ph:  np.zeros_like(batch['s']),
                    trainer.lr_ph:     0.0,
                }
            )
            pred    = np.array(estimates)
            true    = np.array(batch['s'])
            B, T, _ = pred.shape
            for b in range(B):
                rmse_list.append(compute_rmse(pred[b], true[b]))
                ate_list.append(compute_ate(pred[b], true[b]))

    return {'rmse':     np.mean(rmse_list),
            'rmse_std': np.std(rmse_list),
            'ate':      np.mean(ate_list),
            'ate_std':  np.std(ate_list)}


def main():
    tasks     = ['nav01', 'nav02', 'nav03']
    data_path = 'data/100s'

    # DPF-based models
    dpf_models = {
        'CNN':         ('methods/dpf_original.py',    'models/{}_cnn'),
        'ViT':         ('methods/dpf_vit.py',         'models/{}_vit'),
        'ResNet':      ('methods/dpf_resnet.py',      'models/{}_resnet'),
        'ResNet+LSTM': ('methods/dpf_resnet_lstm.py', 'models/{}_resnet_lstm'),
    }

    all_results = {}

    # Evaluate DPF models
    for model_name, (dpf_file, path_tmpl) in dpf_models.items():
        print(f"\nEvaluating {model_name}...")
        all_results[model_name] = {}
        for task in tasks:
            print(f"  {task}...", end=' ', flush=True)
            try:
                r = evaluate_dpf_model(
                    dpf_file, path_tmpl.format(task), task, data_path)
                all_results[model_name][task] = r
                print(f"RMSE={r['rmse']:.4f}, ATE={r['ate']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                all_results[model_name][task] = None

    # Evaluate InEKF model
    print(f"\nEvaluating InEKF+ResNet...")
    all_results['InEKF+ResNet'] = {}
    for task in tasks:
        print(f"  {task}...", end=' ', flush=True)
        try:
            r = evaluate_inekf_model(
                f'models/{task}_inekf', task, data_path)
            all_results['InEKF+ResNet'][task] = r
            print(f"RMSE={r['rmse']:.4f}, ATE={r['ate']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            all_results['InEKF+ResNet'][task] = None

    # Print comparison table
    print("\n" + "=" * 80)
    print("  FINAL RESULTS — 5 Model Comparison")
    print("=" * 80)
    print(f"{'Model':<14} {'Task':<8} {'RMSE':<10} {'RMSE±std':<12} "
          f"{'ATE':<10} {'ATE±std':<10}")
    print("-" * 80)

    for model_name, results in all_results.items():
        for task in tasks:
            r = results.get(task)
            if r:
                print(f"{model_name:<14} {task:<8} {r['rmse']:<10.4f} "
                      f"{r['rmse_std']:<12.4f} {r['ate']:<10.4f} "
                      f"{r['ate_std']:<10.4f}")
            else:
                print(f"{model_name:<14} {task:<8} {'N/A':<10} "
                      f"{'N/A':<12} {'N/A':<10} {'N/A':<10}")
        print()

    # Best model per task
    print("=" * 80)
    print("  BEST MODEL PER TASK (lowest RMSE)")
    print("=" * 80)
    for task in tasks:
        best_model, best_rmse = None, float('inf')
        for model_name, results in all_results.items():
            r = results.get(task)
            if r and r['rmse'] < best_rmse:
                best_rmse, best_model = r['rmse'], model_name
        print(f"  {task}: {best_model} (RMSE={best_rmse:.4f})")

    # Filter type comparison
    print("\n" + "=" * 80)
    print("  PARTICLE FILTER vs KALMAN FILTER (avg RMSE across tasks)")
    print("=" * 80)
    particle_models  = ['CNN', 'ViT', 'ResNet', 'ResNet+LSTM']
    kalman_models    = ['InEKF+ResNet']

    for group_name, group_models in [('Particle filters', particle_models),
                                      ('Kalman filters',   kalman_models)]:
        rmse_vals = []
        for m in group_models:
            for task in tasks:
                r = all_results.get(m, {}).get(task)
                if r:
                    rmse_vals.append(r['rmse'])
        if rmse_vals:
            print(f"  {group_name}: avg RMSE = {np.mean(rmse_vals):.4f}")

    np.save('results_5models.npy', all_results)
    print("\nResults saved to results_5models.npy")


if __name__ == '__main__':
    main()
