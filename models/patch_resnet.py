"""
patch_resnet.py---this needs to be run using the dpf37 activated
then run python patch_resnet.py
"""
def patch():
    path = 'methods/dpf.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    changes = 0
    marker = 'from utils.method_utils import atan2, compute_sq_distance'
    imp = 'from models.resnet_encoder_tf1 import ResNetEncoderTF'
    if imp not in content:
        content = content.replace(marker, marker + '\n' + imp)
        print("  [1] ResNet import added")
        changes += 1
    else:
        print("  [1] Import already present")
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
        "        # ResNet-18 encoder — replaces original CNN\n"
        "        self.encoder = ResNetEncoderTF(out_dim=128, name='resnet_encoder')"
    )
    if old_encoder in content:
        content = content.replace(old_encoder, new_encoder)
        print("  [2] CNN encoder replaced with ResNet")
        changes += 1
    else:
        print("  [2] WARNING: Could not match encoder block")
    old_call = ("        encodings = snt.BatchApply(self.encoder)"
                "((self.placeholders['o'] - means['o']) / stds['o'])")
    new_call = (
        "        # ResNet encoder: flatten time dim, encode, reshape back\n"
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
 
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
 
    print(f"\n  {changes}/3 changes applied")
    if changes == 3:
        print("\n  All good! Now run:")
        print('  python -c "import sys; sys.path.insert(0,\'.\'); sys.path.insert(0,\'experiments\'); from simple import train_dpf; train_dpf(task=\'nav01\', data_path=\'data/100s\', model_path=\'models/nav01_resnet\', plot=False)"')
 
 
if __name__ == '__main__':
    print("=" * 55)
    print("  DPF + ResNet patch")
    print("=" * 55)
    patch()