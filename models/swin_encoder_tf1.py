"""
swin_encoder_tf1.py
-------------------
Swin Transformer encoder in TensorFlow 1.x / Sonnet v1 style.
Drop-in replacement for the CNN/ViT/ResNet encoder in methods/dpf.py.

Input  : (batch, 24, 24, 3)
Output : (batch, 128)

Architecture (Swin-Tiny scaled for 24x24 input):
  patch_size=4  ->  H=W=6, 36 patch tokens
  Stage 1: C=64,  2 Swin blocks, window_size=3  (W-MSA then SW-MSA)
  Patch Merging: 6x6 -> 3x3, C: 64->128
  Stage 2: C=128, 2 Swin blocks, window_size=3  (global, since H=W=ws)
  Global Average Pool -> Linear(128)
"""

import numpy as np
import tensorflow as tf
import sonnet as snt


# ── Pre-computation helpers (numpy, called at __init__ time) ──────────────────

def _compute_sw_mask(H, W, window_size, shift):
    """
    Additive attention mask for shifted-window attention.
    Entries are -100 where tokens from different original windows must not attend,
    0 otherwise.
    Returns ndarray (num_windows, ws*ws, ws*ws).
    """
    img_mask = np.zeros((1, H, W, 1), dtype=np.float32)
    h_slices = [slice(0, H - window_size),
                slice(H - window_size, H - shift),
                slice(H - shift, H)]
    w_slices = [slice(0, W - window_size),
                slice(W - window_size, W - shift),
                slice(W - shift, W)]
    cnt = 0
    for hs in h_slices:
        for ws_s in w_slices:
            img_mask[0, hs, ws_s, 0] = cnt
            cnt += 1
    nH = H // window_size
    nW = W // window_size
    img_mask = img_mask.reshape(1, nH, window_size, nW, window_size, 1)
    img_mask = img_mask.transpose(0, 1, 3, 2, 4, 5)              # (1,nH,nW,ws,ws,1)
    img_mask = img_mask.reshape(-1, window_size * window_size)     # (nH*nW, ws*ws)
    diff = img_mask[:, :, None] - img_mask[:, None, :]            # (nH*nW, ws*ws, ws*ws)
    return np.where(diff != 0, -100.0, 0.0).astype(np.float32)


def _build_rel_pos_index(window_size):
    """
    Static index array mapping each (i,j) token pair to a row in the bias table.
    Returns int32 ndarray of shape (ws*ws * ws*ws,).
    """
    coords = np.arange(window_size)
    gy, gx = np.meshgrid(coords, coords, indexing='ij')
    flat   = np.stack([gy.ravel(), gx.ravel()], axis=0)   # (2, ws*ws)
    rel    = flat[:, :, None] - flat[:, None, :]           # (2, L, L)
    rel[0] += window_size - 1
    rel[1] += window_size - 1
    rel[1] *= 2 * window_size - 1
    idx    = (rel[0] + rel[1]).ravel().astype(np.int32)    # (L*L,)
    return idx


# ── Window partition / reverse (TF ops) ──────────────────────────────────────

def _window_partition(x, H, W, ws, B):
    """(B, H*W, C) -> (B*nH*nW, ws*ws, C)"""
    C  = x.get_shape().as_list()[-1]
    nH = H // ws
    nW = W // ws
    x  = tf.reshape(x, [B, nH, ws, nW, ws, C])
    x  = tf.transpose(x, [0, 1, 3, 2, 4, 5])   # (B, nH, nW, ws, ws, C)
    x  = tf.reshape(x, [-1, ws * ws, C])
    return x


def _window_reverse(x, H, W, ws, B):
    """(B*nH*nW, ws*ws, C) -> (B, H*W, C)"""
    C  = x.get_shape().as_list()[-1]
    nH = H // ws
    nW = W // ws
    x  = tf.reshape(x, [B, nH, nW, ws, ws, C])
    x  = tf.transpose(x, [0, 1, 3, 2, 4, 5])   # (B, nH, ws, nW, ws, C)
    x  = tf.reshape(x, [B, H * W, C])
    return x


# ── Main encoder class ────────────────────────────────────────────────────────

class SwinEncoderTF(snt.AbstractModule):
    """
    Swin Transformer encoder as a Sonnet v1 AbstractModule.

    Replaces the CNN encoder in DPF's build_modules().
    Input  : (B, 24, 24, 3)
    Output : (B, out_dim=128)
    """

    def __init__(self, img_size=24, patch_size=4, embed_dim=64,
                 window_size=3, num_heads_s1=4, num_heads_s2=4,
                 mlp_ratio=4, dropout_rate=0.1, out_dim=128,
                 name='swin_encoder'):
        super(SwinEncoderTF, self).__init__(name=name)
        assert img_size % patch_size == 0
        self.patch_size   = patch_size
        self.embed_dim    = embed_dim           # Stage-1 channels
        self.embed_dim2   = embed_dim * 2       # Stage-2 channels (post-merge)
        self.window_size  = window_size
        self.shift        = window_size // 2    # 1 for ws=3
        self.num_heads_s1 = num_heads_s1
        self.num_heads_s2 = num_heads_s2
        self.mlp_ratio    = mlp_ratio
        self.dropout_rate = dropout_rate
        self.out_dim      = out_dim
        self.H1 = img_size // patch_size        # 6
        self.W1 = img_size // patch_size        # 6
        self.H2 = self.H1 // 2                  # 3
        self.W2 = self.W1 // 2                  # 3

        # Pre-compute masks and index arrays at construction time (numpy only)
        self._sw_mask_np = _compute_sw_mask(
            self.H1, self.W1, window_size, self.shift
        )  # (4, 9, 9)
        self._rpb_idx_s1 = _build_rel_pos_index(window_size)  # (81,)
        self._rpb_idx_s2 = _build_rel_pos_index(window_size)  # same ws

    # ── _build ─────────────────────────────────────────────────────────────────

    def _build(self, x, is_training=False):
        """
        Args:
            x           : (B, 24, 24, 3) float32
            is_training : bool
        Returns:
            (B, out_dim) float32
        """
        B  = tf.shape(x)[0]
        H1, W1 = self.H1, self.W1
        H2, W2 = self.H2, self.W2

        # Make the SW mask a TF constant (cheap; just a 4x9x9 array)
        sw_mask = tf.constant(self._sw_mask_np, dtype=tf.float32)

        # ── Patch Embedding ──────────────────────────────────────────────────
        x = tf.layers.conv2d(
            x, filters=self.embed_dim,
            kernel_size=self.patch_size, strides=self.patch_size,
            padding='valid', name='patch_embed'
        )                                              # (B, 6, 6, embed_dim)
        x = tf.reshape(x, [B, H1 * W1, self.embed_dim])
        x = tf.contrib.layers.layer_norm(
            x, begin_norm_axis=-1, begin_params_axis=-1, scope='patch_norm'
        )

        # ── Stage 1 (H=W=6, C=64) ────────────────────────────────────────────
        x = self._swin_block(
            x, H1, W1, self.embed_dim, self.num_heads_s1,
            self._rpb_idx_s1, shift=False, sw_mask=None,
            B=B, is_training=is_training, prefix='s1b0'
        )
        x = self._swin_block(
            x, H1, W1, self.embed_dim, self.num_heads_s1,
            self._rpb_idx_s1, shift=True, sw_mask=sw_mask,
            B=B, is_training=is_training, prefix='s1b1'
        )

        # ── Patch Merging (6x6->3x3, C: 64->128) ─────────────────────────────
        x = self._patch_merging(x, H1, W1, self.embed_dim, B, prefix='merge1')

        # ── Stage 2 (H=W=3=ws → single window = global attention, C=128) ─────
        x = self._swin_block(
            x, H2, W2, self.embed_dim2, self.num_heads_s2,
            self._rpb_idx_s2, shift=False, sw_mask=None,
            B=B, is_training=is_training, prefix='s2b0'
        )
        x = self._swin_block(
            x, H2, W2, self.embed_dim2, self.num_heads_s2,
            self._rpb_idx_s2, shift=False, sw_mask=None,
            B=B, is_training=is_training, prefix='s2b1'
        )

        # ── Final LayerNorm + Global Average Pool + Head ──────────────────────
        x = tf.contrib.layers.layer_norm(
            x, begin_norm_axis=-1, begin_params_axis=-1, scope='final_norm'
        )
        x = tf.reduce_mean(x, axis=1)              # (B, embed_dim2=128)
        x = tf.layers.dense(x, self.out_dim, activation=tf.nn.relu, name='head')
        return x                                    # (B, 128)

    # ── Swin block ─────────────────────────────────────────────────────────────

    def _swin_block(self, x, H, W, C, num_heads, rpb_idx, shift, sw_mask,
                    B, is_training, prefix):
        """Pre-norm Swin block: LN->(S)W-MSA->residual->LN->MLP->residual"""
        # Attention branch
        residual = x
        x = tf.contrib.layers.layer_norm(
            x, begin_norm_axis=-1, begin_params_axis=-1, scope=f'{prefix}/norm1'
        )
        x = self._window_attn(
            x, H, W, C, num_heads, rpb_idx, shift, sw_mask, B,
            is_training, prefix=f'{prefix}/attn'
        )
        x = tf.layers.dropout(x, rate=self.dropout_rate, training=is_training)
        x = residual + x

        # MLP branch
        residual = x
        x = tf.contrib.layers.layer_norm(
            x, begin_norm_axis=-1, begin_params_axis=-1, scope=f'{prefix}/norm2'
        )
        x = self._mlp(x, C, is_training, prefix=f'{prefix}/mlp')
        x = tf.layers.dropout(x, rate=self.dropout_rate, training=is_training)
        x = residual + x
        return x

    # ── Window attention ───────────────────────────────────────────────────────

    def _window_attn(self, x, H, W, C, num_heads, rpb_idx, shift, sw_mask,
                     B, is_training, prefix):
        """(Shifted) window multi-head self-attention."""
        ws       = self.window_size
        head_dim = C // num_heads
        scale    = head_dim ** -0.5
        nW_int   = (H // ws) * (W // ws)   # static Python int
        L        = ws * ws                  # tokens per window (static)

        # Cyclic shift before partitioning
        if shift:
            x_2d = tf.reshape(x, [B, H, W, C])
            x_2d = tf.roll(x_2d, shift=[-self.shift, -self.shift], axis=[1, 2])
            x    = tf.reshape(x_2d, [B, H * W, C])

        # Partition -> (B*nW, L, C)
        x_win = _window_partition(x, H, W, ws, B)

        # QKV projection
        qkv = tf.layers.dense(x_win, C * 3, use_bias=False,
                               name=f'{prefix}/qkv')          # (B*nW, L, 3C)
        qkv = tf.reshape(qkv, [-1, L, 3, num_heads, head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])             # (3, B*nW, H, L, hd)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Attention + relative position bias
        attn = tf.matmul(Q, K, transpose_b=True) * scale     # (B*nW, H, L, L)
        attn = attn + self._rel_pos_bias(ws, num_heads, rpb_idx, prefix)

        # Shifted-window mask
        if shift and sw_mask is not None:
            mask = tf.reshape(sw_mask, [1, nW_int, 1, L, L])
            attn = tf.reshape(attn,    [B, nW_int, num_heads, L, L])
            attn = attn + mask
            attn = tf.reshape(attn,    [-1, num_heads, L, L])

        attn = tf.nn.softmax(attn, axis=-1)

        # Weighted sum + output projection
        out = tf.matmul(attn, V)                              # (B*nW, H, L, hd)
        out = tf.transpose(out, [0, 2, 1, 3])                # (B*nW, L, H, hd)
        out = tf.reshape(out, [-1, L, C])
        out = tf.layers.dense(out, C, name=f'{prefix}/proj')

        # Reverse partition
        out = _window_reverse(out, H, W, ws, B)               # (B, H*W, C)

        # Reverse cyclic shift
        if shift:
            out_2d = tf.reshape(out, [B, H, W, C])
            out_2d = tf.roll(out_2d, shift=[self.shift, self.shift], axis=[1, 2])
            out    = tf.reshape(out_2d, [B, H * W, C])

        return out

    # ── Relative position bias ─────────────────────────────────────────────────

    def _rel_pos_bias(self, window_size, num_heads, rpb_idx, prefix):
        """Returns (1, num_heads, L, L) learnable relative position bias."""
        table_size = (2 * window_size - 1) ** 2
        L          = window_size ** 2

        with tf.variable_scope(f'{prefix}/rpb'):
            bias_table = tf.get_variable(
                'bias_table',
                shape=[table_size, num_heads],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )

        idx  = rpb_idx                                         # (L*L,) numpy int32
        bias = tf.gather(bias_table, idx)                      # (L*L, num_heads)
        bias = tf.reshape(bias, [L, L, num_heads])
        bias = tf.transpose(bias, [2, 0, 1])                   # (num_heads, L, L)
        return bias[tf.newaxis]                                 # (1, num_heads, L, L)

    # ── Patch merging ──────────────────────────────────────────────────────────

    def _patch_merging(self, x, H, W, C, B, prefix):
        """(B, H*W, C) -> (B, H/2*W/2, 2C)"""
        x  = tf.reshape(x, [B, H, W, C])
        x0 = x[:, 0::2, 0::2, :]   # top-left
        x1 = x[:, 1::2, 0::2, :]   # bottom-left
        x2 = x[:, 0::2, 1::2, :]   # top-right
        x3 = x[:, 1::2, 1::2, :]   # bottom-right
        x  = tf.concat([x0, x1, x2, x3], axis=-1)            # (B, H/2, W/2, 4C)
        x  = tf.reshape(x, [B, (H // 2) * (W // 2), 4 * C])
        x  = tf.contrib.layers.layer_norm(
            x, begin_norm_axis=-1, begin_params_axis=-1, scope=f'{prefix}/norm'
        )
        x  = tf.layers.dense(x, 2 * C, use_bias=False, name=f'{prefix}/reduction')
        return x

    # ── MLP ────────────────────────────────────────────────────────────────────

    def _mlp(self, x, C, is_training, prefix):
        """Position-wise FFN: C -> 4C -> C with GELU."""
        hidden = int(C * self.mlp_ratio)
        x = tf.layers.dense(x, hidden, name=f'{prefix}/fc1')
        x = x * tf.nn.sigmoid(1.702 * x)                     # GELU approximation
        x = tf.layers.dropout(x, rate=self.dropout_rate, training=is_training)
        x = tf.layers.dense(x, C, name=f'{prefix}/fc2')
        return x


# ── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Swin TF1 Encoder — Sanity Check")
    print("=" * 55)

    tf.reset_default_graph()

    swin = SwinEncoderTF(
        img_size=24, patch_size=4, embed_dim=64,
        window_size=3, num_heads_s1=4, num_heads_s2=4,
        mlp_ratio=4, dropout_rate=0.0, out_dim=128,
        name='swin_encoder'
    )

    x   = tf.placeholder(tf.float32, [None, 24, 24, 3])
    out = swin(x, is_training=False)

    print(f"\n  Input  : {x.shape}")
    print(f"  Output : {out.shape}")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        import numpy as np
        dummy  = np.zeros([4, 24, 24, 3], dtype=np.float32)
        result = sess.run(out, feed_dict={x: dummy})
        print(f"  Result shape : {result.shape}")
        total = sum(
            int(np.prod(v.shape.as_list()))
            for v in tf.trainable_variables()
            if 'swin_encoder' in v.name
        )
        print(f"  Parameters   : {total:,}")

    print("\n  Swin TF1 encoder ready!\n")
