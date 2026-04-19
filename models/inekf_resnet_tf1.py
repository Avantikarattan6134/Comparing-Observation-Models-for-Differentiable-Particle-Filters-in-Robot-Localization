import os, numpy as np, tensorflow as tf, sonnet as snt, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments')
from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, make_repeating_batch_iterator, split_data, compute_staticstics
from utils.exp_utils import get_default_hyperparams

def _res_block(x, filters, stride, is_training, name):
    with tf.variable_scope(name):
        r = x
        x = tf.layers.conv2d(x, filters, 3, strides=stride, padding='same', use_bias=False, name='c1')
        x = tf.layers.batch_normalization(x, training=is_training, name='b1')
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters, 3, strides=1, padding='same', use_bias=False, name='c2')
        x = tf.layers.batch_normalization(x, training=is_training, name='b2')
        if stride != 1 or r.get_shape().as_list()[-1] != filters:
            r = tf.layers.conv2d(r, filters, 1, strides=stride, padding='same', use_bias=False, name='sc')
            r = tf.layers.batch_normalization(r, training=is_training, name='sb')
        return tf.nn.relu(x + r)

def resnet_encode(x, is_training, embed_dim=128):
    with tf.variable_scope('resnet_enc', reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d(x, 64, 3, padding='same', use_bias=False, name='stem')
        x = tf.layers.batch_normalization(x, training=is_training, name='sbn')
        x = tf.nn.relu(x)
        x = _res_block(x, 64,  1, is_training, 's1b1')
        x = _res_block(x, 64,  1, is_training, 's1b2')
        x = _res_block(x, 128, 2, is_training, 's2b1')
        x = _res_block(x, 128, 1, is_training, 's2b2')
        x = _res_block(x, 256, 2, is_training, 's3b1')
        x = _res_block(x, 256, 1, is_training, 's3b2')
        x = tf.reduce_mean(x, axis=[1, 2])
        return tf.layers.dense(x, embed_dim, activation=tf.nn.relu, name='head')

class InEKFResNet(snt.AbstractModule):
    def __init__(self, name='inekf_resnet'):
        super(InEKFResNet, self).__init__(name=name)
    def _build(self, observations, actions, is_training=False):
        B = tf.shape(observations)[0]
        T = 20
        z_flat = resnet_encode(tf.reshape(observations, [-1, 24, 24, 3]), is_training)
        z_all  = tf.reshape(z_flat, [B, T, 128])
        with tf.variable_scope('obs_head', reuse=tf.AUTO_REUSE):
            h     = tf.layers.dense(tf.reshape(z_all, [-1, 128]), 128, activation=tf.nn.relu, name='fc1')
            h     = tf.layers.dense(h, 64, activation=tf.nn.relu, name='fc3')
            z_obs = tf.reshape(tf.layers.dense(h, 3, name='fc2'), [B, T, 3])  # predicted state
        log_Q  = tf.get_variable('log_Q', shape=[3], initializer=tf.constant_initializer(-2.0*np.ones(3)))
        log_R  = tf.get_variable('log_R', shape=[3], initializer=tf.constant_initializer(-1.0*np.ones(3)))
        Q_diag = tf.exp(log_Q)
        R_diag = tf.exp(log_R)
        mu     = z_obs[:, 0, :]  # initialize state from first observation
        sigma  = tf.ones([3])
        estimates = []
        for t in range(T):
            act       = actions[:, t, :]          # (B, 3) normalized local frame actions
            theta     = mu[:, 2:3]                # current heading
            sin_t     = tf.sin(theta)
            cos_t     = tf.cos(theta)
            # rotate local actions to global frame (same as DPF motion model)
            new_x     = mu[:, 0:1] + (act[:, 0:1] * cos_t + act[:, 1:2] * sin_t)
            new_y     = mu[:, 1:2] + (act[:, 0:1] * sin_t - act[:, 1:2] * cos_t)
            new_theta = tf.atan2(tf.sin(mu[:, 2:3] + act[:, 2:3]), tf.cos(mu[:, 2:3] + act[:, 2:3]))
            mu        = tf.concat([new_x, new_y, new_theta], axis=1)
            sigma     = sigma + Q_diag
            K     = sigma / (sigma + R_diag + 1e-8)
            innovation = z_obs[:, t, :] - mu  # proper innovation: observed - predicted
            mu    = mu + K * innovation
            mu    = tf.concat([mu[:, :2], tf.atan2(tf.sin(mu[:, 2:3]), tf.cos(mu[:, 2:3]))], axis=1)
            sigma = (1.0 - K) * sigma
            estimates.append(mu)
        return tf.stack(estimates, axis=1)

class InEKFTrainer:
    def __init__(self, learning_rate=3e-4):
        self.obs_ph    = tf.placeholder(tf.float32, [None, 20, 24, 24, 3], 'obs')
        self.action_ph = tf.placeholder(tf.float32, [None, 20, 3], 'actions')
        self.state_ph  = tf.placeholder(tf.float32, [None, 20, 3], 'states')
        self.lr_ph     = tf.placeholder(tf.float32, [], 'lr')
        self.model     = InEKFResNet()
        self.state_estimates = self.model(self.obs_ph, self.action_ph, is_training=True)
        pos_loss   = tf.reduce_mean(tf.square(self.state_estimates[:, :, :2] - self.state_ph[:, :, :2]))
        ad         = self.state_estimates[:, :, 2] - self.state_ph[:, :, 2]
        self.loss  = pos_loss + 0.1 * tf.reduce_mean(tf.square(tf.atan2(tf.sin(ad), tf.cos(ad))))
        self.train_op = tf.train.AdamOptimizer(self.lr_ph).minimize(self.loss)
        self.saver    = tf.train.Saver()

    def fit(self, sess, data, model_path, num_epochs=500, seq_len=20, batch_size=32,
            epoch_length=50, patience=50, learning_rate=3e-4, **kw):
        os.makedirs(model_path, exist_ok=True)
        sess.run(tf.global_variables_initializer())
        st = compute_staticstics(data); means, stds = st[0], st[1]
        sp = split_data(data, 0.9); train_data = sp['train']; val_data = sp['val']
        ti = make_repeating_batch_iterator(train_data, epoch_length, batch_size, seq_len)
        vi = make_batch_iterator(val_data, seq_len)
        best = float('inf'); pc = 0
        for ep in range(num_epochs):
            tl = []
            for _ in range(epoch_length):
                b = next(ti)
                obs = (b['o'] - means['o']) / (stds['o'] + 1e-8)
                act = b['a'] / (stds['a'] + 1e-8)
                stt = (b['s'] - means['s']) / (stds['s'] + 1e-8)
                l, _ = sess.run([self.loss, self.train_op],
                    feed_dict={self.obs_ph: obs, self.action_ph: act,
                               self.state_ph: stt, self.lr_ph: learning_rate})
                tl.append(l)
            vl = []
            for _ in range(10):
                b = next(vi)
                obs = np.transpose((b['o'] - means['o']) / (stds['o'] + 1e-8), (1,0,2,3,4))
                act = np.transpose(b['a'] / (stds['a'] + 1e-8), (1,0,2))
                stt = np.transpose((b['s'] - means['s']) / (stds['s'] + 1e-8), (1,0,2))
                l = sess.run(self.loss,
                    feed_dict={self.obs_ph: obs, self.action_ph: act,
                               self.state_ph: stt, self.lr_ph: 0.0})
                vl.append(l)
            tm = np.mean(tl); vm = np.mean(vl)
            print(f"epoch {ep+1} == train: {tm:.4f} -- val: {vm:.4f} --")
            if vm < best:
                best = vm; pc = 0
                self.saver.save(sess, os.path.join(model_path, 'best_validation'))
                np.savez(os.path.join(model_path, 'statistics.npz'), means=means, stds=stds)
            else:
                pc += 1
                if pc >= patience:
                    print(f"Early stop at epoch {ep+1}")
                    break
        print(f"Done. Best val: {best:.4f}")

def train_inekf(task='nav01', data_path='data/100s', model_path='models/nav01_inekf', plot=False):
    print(f"Training InEKF+ResNet on {task}...")
    tf.reset_default_graph()
    hp = get_default_hyperparams()
    trainer = InEKFTrainer(learning_rate=hp['train']['learning_rate'])
    with tf.Session() as sess:
        trainer.fit(sess, noisyfy_data(load_data(data_path=data_path, filename=task+'_train')), model_path,
                    num_epochs=hp['train']['num_epochs'], seq_len=hp['train']['seq_len'],
                    batch_size=hp['train']['batch_size'], epoch_length=hp['train']['epoch_length'],
                    patience=hp['train']['patience'], learning_rate=hp['train']['learning_rate'])

def test_inekf(task='nav01', data_path='data/100s', model_path='models/nav01_inekf'):
    print(f"Testing InEKF+ResNet on {task}...")
    tf.reset_default_graph()
    trainer = InEKFTrainer()
    rl, al = [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        st = np.load(os.path.join(model_path, 'statistics.npz'), allow_pickle=True)
        means = st['means'].item(); stds = st['stds'].item()
        trainer.saver.restore(sess, os.path.join(model_path, 'best_validation'))
        it = make_batch_iterator(noisyfy_data(load_data(data_path=data_path, filename=task+'_test')), seq_len=20)
        for _ in range(10):
            b = next(it)
            obs = (b['o'] - means['o']) / (stds['o'] + 1e-8)
            act = b['a'] / (stds['a'] + 1e-8)
            est = sess.run(trainer.state_estimates,
                feed_dict={trainer.obs_ph: obs, trainer.action_ph: act,
                           trainer.state_ph: np.zeros_like(b['s']), trainer.lr_ph: 0.0})
            pred = np.array(est); true = np.array(b['s'])
            # Denormalize predictions back to pixel space
            pred = pred * stds['s'] + means['s']
            for bb in range(pred.shape[0]):
                d = pred[bb, :, :2] - true[bb, :, :2]
                rl.append(np.sqrt(np.mean(np.sum(d**2, axis=-1))))
                al.append(np.mean(np.sqrt(np.sum(d**2, axis=-1))))
    print(f"  RMSE: {np.mean(rl):.4f}  ATE: {np.mean(al):.4f}")
    return np.mean(rl), np.mean(al)

if __name__ == '__main__':
    tf.reset_default_graph()
    o = tf.placeholder(tf.float32, [None, 20, 24, 24, 3])
    a = tf.placeholder(tf.float32, [None, 20, 3])
    m = InEKFResNet(); out = m(o, a, is_training=False)
    print(f"Output: {out.shape}")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r = sess.run(out, feed_dict={o: np.zeros([2,20,24,24,3]), a: np.zeros([2,20,3])})
        total = sum(int(np.prod(v.shape.as_list())) for v in tf.trainable_variables())
        print(f"Result: {r.shape}, Params: {total:,}")
    print("Ready!")
