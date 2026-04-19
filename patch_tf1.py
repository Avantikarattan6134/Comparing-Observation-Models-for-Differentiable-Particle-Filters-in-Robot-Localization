"""
patch_tf1.py — run from repo root with dpf37 activated:
  conda activate dpf37
  python patch_tf1.py
"""

def patch_dpf():
    path = 'methods/dpf.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    changes = 0

    # ── Fix 1: Add ViT import ─────────────────────────────────────────────────
    marker = 'from utils.method_utils import atan2, compute_sq_distance'
    vit_import = 'from models.vit_encoder_tf1 import ViTEncoderTF'
    if vit_import not in content:
        content = content.replace(marker, marker + '\n' + vit_import)
        print("  [1] ViT import added")
        changes += 1
    else:
        print("  [1] ViT import already present")

    # ── Fix 2: Replace CNN encoder block ─────────────────────────────────────
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
        "        # ViT encoder — replaces original CNN\n"
        "        self.encoder = ViTEncoderTF(\n"
        "            img_size=24, patch_size=4, embed_dim=128,\n"
        "            depth=4, num_heads=4, mlp_ratio=4,\n"
        "            dropout_rate=0.1, out_dim=128,\n"
        "            name='vit_encoder'\n"
        "        )"
    )
    if old_encoder in content:
        content = content.replace(old_encoder, new_encoder)
        print("  [2] CNN encoder replaced with ViT")
        changes += 1
    else:
        print("  [2] WARNING: Could not match encoder block exactly")
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'ConvNet2D' in line:
                print("  Found ConvNet2D at line", i+1)
                print('\n'.join(f"    {i-2+j}: {repr(lines[i-2+j])}"
                                for j in range(8)))
                break

    # ── Fix 3: Replace encoder call ──────────────────────────────────────────
    # The encoder call in TF1 Sonnet is: snt.BatchApply(self.encoder)(...)
    # ViT is called directly since it handles batch dims internally
    old_call = ("        encodings = snt.BatchApply(self.encoder)"
                "((self.placeholders['o'] - means['o']) / stds['o'])")
    new_call = (
        "        # ViT encoder handles (B*T, H, W, C) internally\n"
        "        _obs     = (self.placeholders['o'] - means['o']) / stds['o']\n"
        "        _shape   = tf.shape(_obs)\n"
        "        _flat    = tf.reshape(_obs, [-1, 24, 24, 3])\n"
        "        _enc     = self.encoder(_flat, is_training=False)\n"
        "        encodings = tf.reshape(_enc, [_shape[0], _shape[1], 128])"
    )
    if old_call in content:
        content = content.replace(old_call, new_call)
        print("  [3] Encoder call updated")
        changes += 1
    else:
        print("  [3] WARNING: Could not match encoder call")
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'BatchApply(self.encoder)' in line:
                print("  Found at line", i+1, ":", repr(line))
                break

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

    return changes


if __name__ == '__main__':
    print("=" * 55)
    print("  DPF + ViT TF1 patch")
    print("=" * 55)
    c = patch_dpf()
    print(f"\n  {c}/3 changes applied")
    if c == 3:
        print("\n  All good! Now run:")
        print('  python -c "import sys; sys.path.insert(0,\'.\'); '
              'sys.path.insert(0,\'experiments\'); from simple import '
              'train_dpf; train_dpf(task=\'nav01\', '
              'data_path=\'data/100s\', model_path=\'models/tmp\', plot=False)"')
    else:
        print("\n  Fix warnings above, then re-run")
