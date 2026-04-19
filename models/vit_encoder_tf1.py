"""
vit_encoder_tf1.py
------------------
Vision Transformer encoder in TensorFlow 1.x / Sonnet v1 style.
Drop-in replacement for the CNN encoder in methods/dpf.py.

Input  : (batch, 24, 24, 3)
Output : (batch, 128)
"""

import numpy as np
import tensorflow as tf
import sonnet as snt


class ViTEncoderTF(snt.AbstractModule):
    """
    Vision Transformer encoder as a Sonnet v1 AbstractModule.

    Replaces the snt.Sequential CNN encoder in DPF's build_modules().
    Input  : (B, 24, 24, 3)
    Output : (B, 128)

    Architecture:
      patch_size=4  ->  (24/4)^2 = 36 patches
      embed_dim=128, depth=4, num_heads=4, mlp_ratio=4
    """

    def __init__(self, img_size=24, patch_size=4, embed_dim=128,
                 depth=4, num_heads=4, mlp_ratio=4, dropout_rate=0.1,
                 out_dim=128, name='vit_encoder'):
        super(ViTEncoderTF, self).__init__(name=name)
        assert img_size % patch_size == 0
        self.img_size     = img_size
        self.patch_size   = patch_size
        self.embed_dim    = embed_dim
        self.depth        = depth
        self.num_heads    = num_heads
        self.head_dim     = embed_dim // num_heads
        self.mlp_ratio    = mlp_ratio
        self.dropout_rate = dropout_rate
        self.out_dim      = out_dim
        self.num_patches  = (img_size // patch_size) ** 2   # 36

    def _build(self, x, is_training=False):
        """
        Args:
            x           : (B, 24, 24, 3) float32
            is_training : bool — controls dropout
        Returns:
            (B, out_dim) float32
        """
        B = tf.shape(x)[0]

        # ── Step 1: Patch Embedding ───────────────────────────────────────────
        # Conv2D with kernel=stride=patch_size  ≡  linear projection per patch
        x = tf.layers.conv2d(
            x, filters=self.embed_dim,
            kernel_size=self.patch_size, strides=self.patch_size,
            padding='valid', name='patch_proj'
        )                                               # (B, 6, 6, embed_dim)
        x = tf.reshape(x, [B, self.num_patches, self.embed_dim])  # (B, 36, 128)

        # ── Step 2: CLS Token + Positional Embedding ─────────────────────────
        cls_token = tf.get_variable(
            'cls_token',
            shape=[1, 1, self.embed_dim],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        cls_tokens = tf.tile(cls_token, [B, 1, 1])     # (B, 1, 128)
        x = tf.concat([cls_tokens, x], axis=1)         # (B, 37, 128)

        pos_embed = tf.get_variable(
            'pos_embed',
            shape=[1, self.num_patches + 1, self.embed_dim],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        x = x + pos_embed                              # (B, 37, 128)
        x = tf.layers.dropout(x, rate=self.dropout_rate, training=is_training)

        # ── Steps 3-5: Transformer Encoder Blocks ────────────────────────────
        for i in range(self.depth):
            x = self._transformer_block(x, i, is_training)

        # ── Final LayerNorm ───────────────────────────────────────────────────
        x = self._layer_norm(x, name='final_norm')     # (B, 37, 128)

        # ── Extract CLS token ─────────────────────────────────────────────────
        cls_out = x[:, 0, :]                           # (B, 128)

        # ── Head: project to out_dim ─────────────────────────────────────────
        out = tf.layers.dense(cls_out, self.out_dim,
                              activation=tf.nn.relu, name='head')
        return out                                     # (B, 128)

    def _transformer_block(self, x, block_idx, is_training):
        """Pre-norm transformer block: LN -> MHSA -> residual -> LN -> MLP -> residual"""
        prefix = f'block_{block_idx}'

        # Attention branch
        residual = x
        x = self._layer_norm(x, name=f'{prefix}/norm1')
        x = self._mhsa(x, prefix, is_training)
        x = tf.layers.dropout(x, rate=self.dropout_rate, training=is_training)
        x = residual + x

        # MLP branch
        residual = x
        x = self._layer_norm(x, name=f'{prefix}/norm2')
        x = self._mlp(x, prefix, is_training)
        x = tf.layers.dropout(x, rate=self.dropout_rate, training=is_training)
        x = residual + x

        return x

    def _mhsa(self, x, prefix, is_training):
        """Multi-Head Self-Attention"""
        B  = tf.shape(x)[0]
        N  = tf.shape(x)[1]
        D  = self.embed_dim
        H  = self.num_heads
        dk = self.head_dim
        scale = dk ** -0.5

        # Project to Q, K, V
        qkv = tf.layers.dense(x, D * 3, use_bias=False,
                               name=f'{prefix}/qkv')    # (B, N, 3D)
        qkv = tf.reshape(qkv, [B, N, 3, H, dk])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])       # (3, B, H, N, dk)
        Q = qkv[0]; K = qkv[1]; V = qkv[2]             # each (B, H, N, dk)

        # Scaled dot-product attention
        attn = tf.matmul(Q, K, transpose_b=True) * scale  # (B, H, N, N)
        attn = tf.nn.softmax(attn, axis=-1)

        # Weighted sum
        out = tf.matmul(attn, V)                        # (B, H, N, dk)
        out = tf.transpose(out, [0, 2, 1, 3])           # (B, N, H, dk)
        out = tf.reshape(out, [B, N, D])                # (B, N, D)
        out = tf.layers.dense(out, D, name=f'{prefix}/out_proj')
        return out

    def _mlp(self, x, prefix, is_training):
        """Position-wise MLP: D -> 4D -> D with GELU"""
        hidden = int(self.embed_dim * self.mlp_ratio)
        x = tf.layers.dense(x, hidden, name=f'{prefix}/mlp_fc1')
        x = self._gelu(x)
        x = tf.layers.dropout(x, rate=self.dropout_rate, training=is_training)
        x = tf.layers.dense(x, self.embed_dim, name=f'{prefix}/mlp_fc2')
        return x

    def _layer_norm(self, x, name='ln'):
        """Layer normalization over last axis"""
        return tf.contrib.layers.layer_norm(
            x, begin_norm_axis=-1, begin_params_axis=-1, scope=name
        )

    def _gelu(self, x):
        """GELU activation: x * Phi(x)"""
        return x * tf.nn.sigmoid(1.702 * x)


# ==============================================================================
# SANITY CHECK
# ==============================================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  ViT TF1 Encoder — Sanity Check")
    print("=" * 55)

    tf.reset_default_graph()

    # Build model
    vit = ViTEncoderTF(img_size=24, patch_size=4, embed_dim=128,
                       depth=4, num_heads=4, out_dim=128)

    # Dummy input
    x   = tf.placeholder(tf.float32, [None, 24, 24, 3])
    out = vit(x, is_training=False)

    print(f"\n  Input  : {x.shape}")
    print(f"  Output : {out.shape}")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dummy = np.zeros([4, 24, 24, 3], dtype=np.float32)
        result = sess.run(out, feed_dict={x: dummy})
        print(f"  Result shape : {result.shape}")
        total = sum(v.shape.num_elements()
                    for v in tf.trainable_variables()
                    if 'vit_encoder' in v.name)
        print(f"  Parameters   : {total:,}")

    print("\n  ViT TF1 encoder ready!\n")
