"""
patch_resnet_lstm.py — run from repo root with dpf37 activated:
  conda activate dpf37
  python patch_resnet_lstm.py
"""
import shutil

def patch():
    
    shutil.copy('methods/dpf_original.py', 'methods/dpf_resnet_lstm.py')

    with open('methods/dpf_resnet_lstm.py', 'r', encoding='utf-8') as f:
        content = f.read()

    changes = 0

    # ── Add import 
    marker = 'from utils.method_utils import atan2, compute_sq_distance'
    imp = 'from models.resnet_lstm_encoder_tf1 import ResNetLSTMEncoderTF'
    if imp not in content:
        content = content.replace(marker, marker + '\n' + imp)
        print("  [1] ResNetLSTM import added")
        changes += 1

    # ── Replace CNN encoder 
    old_encoder = (
        "        self.encoder = snt.Sequential([\n"
        "            snt.nets.ConvNet2D([16, 32, 64], [[3, 3]], [2], [snt.SAME], activate_final=True, name='encoder/convnet'),\n"
        "            snt.BatchFlatten(),\n"
        "            lambda x: tf.nn.dropout(x,  self.placeholders['keep_prob']),\n"
        "            snt.Linear(128, name='encoder/linear'),\n"
        "            tf.nn.relu\n"
        "        ])"
    )
    new_encoder = (
        "        # ResNet2D + LSTM encoder\n"
        "        self.encoder = ResNetLSTMEncoderTF(\n"
        "            lstm_hidden=256, lstm_layers=2,\n"
        "            lstm_dropout=0.1, head_dropout=0.1,\n"
        "            use_mean_pool=True, out_dim=128,\n"
        "            seq_len=1, name='resnet_lstm_encoder'\n"
        "        )"
    )
    if old_encoder in content:
        content = content.replace(old_encoder, new_encoder)
        print("  [2] CNN encoder replaced with ResNet+LSTM")
        changes += 1
    else:
        print("  [2] WARNING: Could not match encoder block")

    # ── Replace encoder call 
    old_call = ("        encodings = snt.BatchApply(self.encoder)"
                "((self.placeholders['o'] - means['o']) / stds['o'])")
    new_call = (
        "        # ResNet+LSTM encoder\n"
        "        _obs   = (self.placeholders['o'] - means['o']) / stds['o']\n"
        "        _shape = tf.shape(_obs)\n"
        "        _flat  = tf.reshape(_obs, [-1, 24, 24, 3])\n"
        "        _enc   = self.encoder(_flat, is_training=False)\n"
        "        encodings = tf.reshape(_enc, [_shape[0], _shape[1], 128])"
    )
    if old_call in content:
        content = content.replace(old_call, new_call)
        print("  [3] Encoder call updated")
        changes += 1
    else:
        print("  [3] WARNING: Could not match encoder call")

    # ── Fix allow_pickle 
    content = content.replace(
        'statistics = dict(np.load(os.path.join(model_path, statistics_file)))',
        'statistics = dict(np.load(os.path.join(model_path, statistics_file), allow_pickle=True))'
    )

    with open('methods/dpf_resnet_lstm.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n  {changes}/3 changes applied to methods/dpf_resnet_lstm.py")
    if changes == 3:
        print("\n  Now run training:")
        print("  sbatch run_resnet_lstm.sh")


if __name__ == '__main__':
    print("=" * 55)
    print("  DPF + ResNet2D+LSTM patch")
    print("=" * 55)
    patch()
