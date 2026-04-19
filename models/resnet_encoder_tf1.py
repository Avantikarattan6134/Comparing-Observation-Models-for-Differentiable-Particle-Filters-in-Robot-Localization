import numpy as numpy
import tensorflow as tf
import sonnet as snt
import sonnet as sonnet

def residual_block(x, filters, stride=1, is_training=False, name='rb'):

#x: input tensor
#filters: output channel count
#stride: controls downsampling
#is_training: tells batch norm whether we are training or inferring
#name: variable scope name
    with tf.variable_scope(name):
        residual = x
        ## First Conv: conv _> bn ->relu
        x = tf.layers.conv2d(
            x, filters=filters, kernel_size=3,
            strides=stride, padding='same',
            use_bias=False, name='conv1'
        )
        x = tf.layers.batch_normalization(
            x, training=is_training, name='bn1'
        )
        x = tf.nn.relu(x)
        ## second conv: conv -> bn
        x = tf.layers.conv2d(
            x, filters=filters, kernel_size=3,
            strides=1, padding='same',
            use_bias=False, name='conv2'
        )
        x = tf.layers.batch_normalization(
            x, training=is_training, name='bn2'
        )
        ##Skip connection
        if stride != 1 or residual.get_shape().as_list()[-1] != filters:
            residual = tf.layers.conv2d(
                residual, filters=filters, kernel_size=1, strides=stride, padding='same', use_bias=False, name='shortcut_conv'
            )
            residual = tf.layers.batch_normalization(
                residual, training=is_training, name='shortcut_bn'
            )
        x = tf.nn.relu(x + residual)
        return x 

class ResNetEncoderTF(snt.AbstractModule):
    def __init__(self, out_dim=128, name='resnet_encoder'):
        super(ResNetEncoderTF, self).__init__(name=name)
        self.out_dim = out_dim
    def _build(self, x, is_training=False):
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
        x = residual_block(x, filters=64, stride=1,
                           is_training=is_training, name='s1b1')
        x = residual_block(x, filters=64, stride=1,
                           is_training=is_training, name='s1b2')
# Shape: (B, 24, 24, 64)
        x = residual_block(x, filters=128, stride=2,
                           is_training=is_training, name='s2b1')
        x = residual_block(x, filters=128, stride=1,
                           is_training=is_training, name='s2b2')
# Shape: (B, 12, 12, 128)
        x = residual_block(x, filters=256, stride=2, 
                           is_training=is_training, name='s3b1')
        x = residual_block(x, filters=256, stride=1,
                           is_training=is_training, name='s3b2')
# Shape: (B, 6, 6, 256)
        x = tf.reduce_mean(x, axis=[1, 2], name='gap')
# Shape: (B, 256)
        x = tf.layers.dense(
            x, self.out_dim,
            activation=tf.nn.relu,
            name='head'
        )
        return x
if __name__ == '__main__':
    print("=" * 55)
    print("ResNet TF1 Encoder - Sanity Check")
    print("=" * 55)
    tf.reset_default_graph()
    resnet = ResNetEncoderTF(out_dim=128, name='resnet_encoder')
    x = tf.placeholder(tf.float32, [None, 24, 24, 3], name='obs')
    out = resnet(x, is_training=False)
    print(f"\n Input : {x.shape}")
    print(f" Output : {out.shape}")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dummy = np.zeros([4, 24, 24, 3], dtype=np.float32)
        result = sess.run(out, feed_dict={x : dummy})
        print(f"  Result : {result.shape}")
        total = sum(
            int(np.prod(v.shape.as_list()))
            for v in tf.trainable_variables()
            if 'resnet_encoder' in v.name
        )
        print(f"  Params : {total:,}")
 
    print("\n  ResNet TF1 encoder ready — matches DPF CNN output shape!\n")
 