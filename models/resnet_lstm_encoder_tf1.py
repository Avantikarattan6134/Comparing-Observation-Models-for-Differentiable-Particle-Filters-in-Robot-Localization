"""


Architecture:
  Stem      : Conv2D 3x3 s=1, BN, ReLU
  Stage 1   : 2x ResBlock2D(64,  stride=1)  → (B, 24, 24, 64)
  Stage 2   : 2x ResBlock2D(128, stride=2)  → (B, 12, 12, 128)
  Stage 3   : 2x ResBlock2D(256, stride=2)  → (B,  6,  6, 256)
  GAP       : Global Average Pool           → (B, 256)
  Pre-LSTM  : LayerNorm                     → (B, T, 256)
  LSTM      : 2-layer LSTM(hidden=256)      → (B, T, 256)
  Post-LSTM : LayerNorm                     → (B, T, 256)
  Head      : Dense(128) + ReLU             → (B, 128)

Input  : (B*T, 24, 24, 3)  — flattened time dimension
Output : (B*T, 128)         — matches original DPF encoder output
"""

import numpy as np
import tensorflow as tf
import sonnet as snt



# 2D RESIDUAL BLOCK 

def residual_block_2d(x, filters, stride=1, is_training=False, name='rb'):
    """
    2D residual block: two 3x3 convolutions with a skip connection.
    Directly adapted from BasicBlock1D but uses Conv2D.

    Args:
        x           : input tensor (B, H, W, C)
        filters     : number of output filters
        stride      : stride for first conv (2 = downsample)
        is_training : bool for BatchNorm
        name        : variable scope name
    Returns:
        output tensor
    """
    with tf.variable_scope(name):
        residual = x

        # ── First conv: Conv → BN → ReLU 
        x = tf.layers.conv2d(
            x, filters=filters, kernel_size=3,
            strides=stride, padding='same',
            use_bias=False, name='conv1'
        )
        x = tf.layers.batch_normalization(
            x, training=is_training, name='bn1'
        )
        x = tf.nn.relu(x)

        # ── Second conv: Conv → BN (no ReLU before skip add) 
        x = tf.layers.conv2d(
            x, filters=filters, kernel_size=3,
            strides=1, padding='same',
            use_bias=False, name='conv2'
        )
        x = tf.layers.batch_normalization(
            x, training=is_training, name='bn2'
        )

        # ── Skip connection 
        # Project if spatial size or channel count changed
        if stride != 1 or residual.get_shape().as_list()[-1] != filters:
            residual = tf.layers.conv2d(
                residual, filters=filters, kernel_size=1,
                strides=stride, padding='same',
                use_bias=False, name='shortcut_conv'
            )
            residual = tf.layers.batch_normalization(
                residual, training=is_training, name='shortcut_bn'
            )

        # ── Add residual THEN activate 
        x = tf.nn.relu(x + residual)
        return x



# RESNET2D + LSTM ENCODER 

class ResNetLSTMEncoderTF(snt.AbstractModule):
    """
    ResNet2D spatial encoder + LSTM temporal encoder.

    Note: For DPF, we process (B*T) frames independently through ResNet,
    then reshape and pass through LSTM to capture temporal dynamics.
    The DPF training loop calls encode on individual frames, so we
    use a stateless approach — LSTM context within each batch sequence.

    Args:
        lstm_hidden   : LSTM hidden size (default 256, matches original)
        lstm_layers   : Number of LSTM layers (default 2, matches original)
        lstm_dropout  : Dropout between LSTM layers (default 0.1)
        head_dropout  : Dropout in head MLP (default 0.1)
        use_mean_pool : True=mean over time, False=last step (matches original)
        out_dim       : Output dimension (128 to match DPF encoder)
        seq_len       : Sequence length T for reshaping (default 1 for DPF)
    """

    def __init__(self, lstm_hidden=256, lstm_layers=2, lstm_dropout=0.1,
                 head_dropout=0.1, use_mean_pool=True, out_dim=128,
                 seq_len=1, name='resnet_lstm_encoder'):
        super(ResNetLSTMEncoderTF, self).__init__(name=name)
        self.lstm_hidden   = lstm_hidden
        self.lstm_layers   = lstm_layers
        self.lstm_dropout  = lstm_dropout
        self.head_dropout  = head_dropout
        self.use_mean_pool = use_mean_pool
        self.out_dim       = out_dim
        self.seq_len       = seq_len

    def _build(self, x, is_training=False):
        """
        Args:
            x           : (B, 24, 24, 3) or (B*T, 24, 24, 3) image frames
            is_training : bool for BatchNorm and Dropout
        Returns:
            (B, out_dim) or (B*T, out_dim) embeddings
        """
        B = tf.shape(x)[0]

        # ── SPATIAL ENCODER: ResNet2D 
        

        x = tf.layers.conv2d(
            x, filters=64, kernel_size=3,
            strides=1, padding='same',
            use_bias=False, name='stem_conv'
        )
        x = tf.layers.batch_normalization(
            x, training=is_training, name='stem_bn'
        )
        x = tf.nn.relu(x)
        # Shape: (B, 24, 24, 64)

        # Stage 1: group_sizes[0] = 2 blocks, stride=1 (matches 64-group)
        x = residual_block_2d(x, filters=64, stride=1,
                              is_training=is_training, name='s1b1')
        x = residual_block_2d(x, filters=64, stride=1,
                              is_training=is_training, name='s1b2')
        # Shape: (B, 24, 24, 64)

        # Stage 2: group_sizes[1] = 2 blocks, stride=2 (matches maxpool s=2)
        x = residual_block_2d(x, filters=128, stride=2,
                              is_training=is_training, name='s2b1')
        x = residual_block_2d(x, filters=128, stride=1,
                              is_training=is_training, name='s2b2')
        # Shape: (B, 12, 12, 128)

        # Stage 3: group_sizes[2] = 2 blocks, stride=2 (matches 512-group s=2)
        x = residual_block_2d(x, filters=256, stride=2,
                              is_training=is_training, name='s3b1')
        x = residual_block_2d(x, filters=256, stride=1,
                              is_training=is_training, name='s3b2')
        # Shape: (B, 6, 6, 256)

        # Global Average Pooling (matches use_mean_pool concept in original)
        feat_dim = 256
        x = tf.reduce_mean(x, axis=[1, 2], name='gap')
        # Shape: (B, 256)  ← feat_dim = 512*expansion in original

        # ── TEMPORAL ENCODER: LSTM 
        # Reshape to sequence: (B, T, feat_dim)
        # For DPF single-frame mode: T=1
        x = tf.reshape(x, [B, self.seq_len, feat_dim])
        # Shape: (B, T, 256)

        # Pre-LSTM LayerNorm (matches pre_lstm_ln in original)
        x = tf.contrib.layers.layer_norm(
            x, begin_norm_axis=-1, begin_params_axis=-1,
            scope='pre_lstm_ln'
        )

        # 2-layer LSTM 
        # Build stacked LSTM cells
        lstm_cells = []
        for i in range(self.lstm_layers):
            cell = tf.nn.rnn_cell.LSTMCell(
                self.lstm_hidden,
                name=f'lstm_cell_{i}'
            )
            if is_training and self.lstm_dropout > 0 and i < self.lstm_layers - 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell,
                    output_keep_prob=1.0 - self.lstm_dropout
                )
            lstm_cells.append(cell)

        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        # Run LSTM over sequence
        outputs, _ = tf.nn.dynamic_rnn(
            stacked_lstm, x,
            dtype=tf.float32,
            scope='lstm'
        )
        # outputs shape: (B, T, lstm_hidden)

        # Post-LSTM LayerNorm 
        outputs = tf.contrib.layers.layer_norm(
            outputs, begin_norm_axis=-1, begin_params_axis=-1,
            scope='post_lstm_ln'
        )

        # Pool over time (matches use_mean_pool in original)
        if self.use_mean_pool:
            h = tf.reduce_mean(outputs, axis=1)   # (B, lstm_hidden)
        else:
            h = outputs[:, -1, :]                  # (B, lstm_hidden) last step

        # ── HEAD 
        # Matches mean_head in original (single output, no logstd for DPF)
        if is_training and self.head_dropout > 0:
            h = tf.nn.dropout(h, rate=self.head_dropout)

        h = tf.layers.dense(h, self.lstm_hidden,
                            activation=tf.nn.relu, name='head_fc1')

        if is_training and self.head_dropout > 0:
            h = tf.nn.dropout(h, rate=self.head_dropout)

        h = tf.layers.dense(h, self.out_dim,
                            activation=tf.nn.relu, name='head_fc2')
        # Shape: (B, out_dim=128)

        return h

# SANITY CHECK
if __name__ == '__main__':
    print("=" * 60)
    print("  ResNet2D + LSTM TF1 Encoder — Sanity Check")
    print("=" * 60)

    tf.reset_default_graph()

    model = ResNetLSTMEncoderTF(
        lstm_hidden=256, lstm_layers=2,
        lstm_dropout=0.1, head_dropout=0.1,
        use_mean_pool=True, out_dim=128,
        seq_len=1, name='resnet_lstm_encoder'
    )

    x   = tf.placeholder(tf.float32, [None, 24, 24, 3], name='obs')
    out = model(x, is_training=False)

    print(f"\n  Input  : {x.shape}")
    print(f"  Output : {out.shape}")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dummy  = np.zeros([4, 24, 24, 3], dtype=np.float32)
        result = sess.run(out, feed_dict={x: dummy})
        print(f"  Result : {result.shape}")
        total = sum(
            int(np.prod(v.shape.as_list()))
            for v in tf.trainable_variables()
            if 'resnet_lstm_encoder' in v.name
        )
        print(f"  Params : {total:,}")

    print("\n  ResNet2D+LSTM TF1 encoder ready!\n")
    print("  Comparison with original model_resnet_2res_nodrop_lstm.py:")
    print("  - BasicBlock1D (Conv1D) → residual_block_2d (Conv2D)")
    print("  - ResNet1D groups      → ResNet2D stages with GAP")
    print("  - LSTM(256, 2 layers)  → same LSTM(256, 2 layers)")
    print("  - mean_head            → head_fc1 + head_fc2 → 128-d output")
    print("  - use_mean_pool=True   → mean over T timesteps")
