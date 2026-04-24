"""
collect_one_traj.py  <model_name> <task>
Collects trajectory data for one model/task and saves to report_plots/traj_data/.
Called once per model/task so TF resets between runs.
"""
import sys, os, shutil
import numpy as np
import tensorflow as tf

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state
from utils.exp_utils  import get_default_hyperparams

MODELS = {
    'CNN':         ('methods/dpf_original.py',   'models/{task}_cnn'),
    'ViT':         ('methods/dpf_vit.py',          'models/{task}_vit'),
    'ResNet':      ('methods/dpf_resnet.py',       'models/{task}_resnet'),
    'ResNet+LSTM': ('methods/dpf_resnet_lstm.py',  'models/{task}_resnet_lstm'),
    'Swin':        ('methods/dpf_swin.py',         'models/{task}_swin'),
}

model_name = sys.argv[1]
task       = sys.argv[2]
out_path   = f'report_plots/traj_data/{model_name}_{task}.npz'

if os.path.exists(out_path):
    print(f'EXISTS {out_path}')
    sys.exit(0)

dpf_file, path_tmpl = MODELS[model_name]
model_path = path_tmpl.format(task=task)

shutil.copy(dpf_file, 'methods/dpf.py')
from methods.dpf import DPF

test_data = load_data(data_path='data/100s', filename=task + '_test')
noisy     = noisyfy_data(test_data)
iterator  = make_batch_iterator(noisy, seq_len=50)
hp        = get_default_hyperparams()

tf.reset_default_graph()
method = DPF(**hp['global'])

with tf.Session() as sess:
    method.load(sess, model_path)
    batch    = next(iterator)
    true_s   = np.array(batch['s'])
    batch_in = remove_state(batch, provide_initial_state=False)
    result   = method.predict(sess, batch_in, **hp['test'])
    pred_s   = np.array(result)

pred_s = pred_s[:5]
true_s = true_s[:5]
np.savez(out_path, pred=pred_s, true=true_s)
rmse = float(np.sqrt(np.mean(np.sum((pred_s[:,:,:2] - true_s[:,:,:2])**2, axis=-1))))
print(f'SAVED {out_path}  RMSE={rmse:.1f}')
