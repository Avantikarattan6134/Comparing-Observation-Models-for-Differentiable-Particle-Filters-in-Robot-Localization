"""
evaluate_4models.py
-------------------
Computes RMSE and ATE for CNN, ViT, ResNet, ResNet+LSTM on nav01/02/03.
Run from repo root:  python evaluate_4models.py
"""
import sys, numpy as np, tensorflow as tf, shutil
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state
from utils.exp_utils import get_default_hyperparams

def compute_rmse(p, t):
    return np.sqrt(np.mean(np.sum((p[:,:2]-t[:,:2])**2, axis=-1)))

def compute_ate(p, t):
    return np.mean(np.sqrt(np.sum((p[:,:2]-t[:,:2])**2, axis=-1)))

def evaluate_model(dpf_file, model_path, task, data_path, num_trials=10):
    shutil.copy(dpf_file, 'methods/dpf.py')
    if 'methods.dpf' in sys.modules:
        del sys.modules['methods.dpf']
    from methods.dpf import DPF
    test_data = load_data(data_path=data_path, filename=task+'_test')
    noisy     = noisyfy_data(test_data)
    iterator  = make_batch_iterator(noisy, seq_len=50)
    hp        = get_default_hyperparams()
    tf.reset_default_graph()
    method    = DPF(**hp['global'])
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
    return {'rmse': np.mean(rmse_list), 'rmse_std': np.std(rmse_list),
            'ate':  np.mean(ate_list),  'ate_std':  np.std(ate_list)}

def main():
    tasks     = ['nav01', 'nav02', 'nav03']
    data_path = 'data/100s'
    models = {
        'CNN':         ('methods/dpf_original.py',    'models/{}_cnn'),
        'ViT':         ('methods/dpf_vit.py',         'models/{}_vit'),
        'ResNet':      ('methods/dpf_resnet.py',      'models/{}_resnet'),
        'ResNet+LSTM': ('methods/dpf_resnet_lstm.py', 'models/{}_resnet_lstm'),
    }
    all_results = {}
    for model_name, (dpf_file, path_tmpl) in models.items():
        print(f"\nEvaluating {model_name}...")
        all_results[model_name] = {}
        for task in tasks:
            print(f"  {task}...", end=' ', flush=True)
            try:
                r = evaluate_model(dpf_file, path_tmpl.format(task), task, data_path)
                all_results[model_name][task] = r
                print(f"RMSE={r['rmse']:.4f}, ATE={r['ate']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                all_results[model_name][task] = None

    print("\n" + "="*78)
    print("  RESULTS — CNN vs ViT vs ResNet vs ResNet+LSTM")
    print("="*78)
    print(f"{'Model':<14} {'Task':<8} {'RMSE':<10} {'RMSE±std':<12} {'ATE':<10} {'ATE±std':<10}")
    print("-"*78)
    for model_name, results in all_results.items():
        for task in tasks:
            r = results.get(task)
            if r:
                print(f"{model_name:<14} {task:<8} {r['rmse']:<10.4f} "
                      f"{r['rmse_std']:<12.4f} {r['ate']:<10.4f} {r['ate_std']:<10.4f}")
            else:
                print(f"{model_name:<14} {task:<8} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
        print()

    print("="*78)
    print("  BEST MODEL PER TASK (lowest RMSE)")
    print("="*78)
    for task in tasks:
        best_model, best_rmse = None, float('inf')
        for model_name, results in all_results.items():
            r = results.get(task)
            if r and r['rmse'] < best_rmse:
                best_rmse, best_model = r['rmse'], model_name
        print(f"  {task}: {best_model} (RMSE={best_rmse:.4f})")

    np.save('results_comparison_4models.npy', all_results)
    print("\nResults saved to results_comparison_4models.npy")

if __name__ == '__main__':
    main()
