import h5py
import numpy as np
import os
import tensorflow as tf

fw = tf.contrib.framework

pprint = lambda x, msg: tf.Print(x, [x], message=msg)

def build_vimco_loss(l, log_q_h, log_q_h_list):
  """Builds VIMCO baseline as in https://arxiv.org/abs/1602.06725

  Args:
    l: Per-sample learning signal. shape [k, b] or
        [number of samples, batch_size]
    log_q_h: Sum of log q(h^l) over layers

  Returns:
    baseline to subtract from l
  """
  k, b = l.get_shape().as_list()
  kf = tf.cast(k, tf.float32)
  l_logsumexp = tf.reduce_logsumexp(l, [0], keep_dims=True)
  L_hat = l_logsumexp - tf.log(kf)
  s = tf.reduce_sum(l, 0, keep_dims=True)
  diag_mask = tf.expand_dims(tf.diag(tf.ones([k], dtype=tf.float32)), -1)
  off_diag_mask = 1. - diag_mask
  diff = tf.expand_dims(s - l, 0)  # expand for proper broadcasting
  l_i_diag = 1. / (kf - 1.) * diff * diag_mask
  l_i_off_diag = off_diag_mask * tf.pack([l] * k)
  l_i = l_i_diag + l_i_off_diag
  L_hat_minus_i = tf.reduce_logsumexp(l_i, [1]) - tf.log(kf)
  w = tf.stop_gradient(tf.exp((l - l_logsumexp)))
  local_l = tf.stop_gradient(L_hat - L_hat_minus_i)
  loss = local_l * log_q_h + w * l
  return loss / float(b), L_hat[0, :]


def _logsubexp(a, b, eps=1e-6):
  """Stable log(exp(a) - exp(b))."""
  return a + tf.log(1. - tf.clip_by_value(tf.exp(b - a), eps, 1. - eps))


def fully_connected(inp, size, scope):
  w = tf.get_variable(name=scope + '/weights',
      dtype=tf.float32, shape=[inp.get_shape.as_list()[-1], size])
  b = tf.get_variable(name=scope + '/biases',
      dtype=tf.float32, shape=[size])
  return tf.nn.xw_plus_b(inp, w, b)


def provide_data(config):
  """Provides batches of MNIST digits.

  Args:
    config: configuration object

  Returns:
    data_iterator: an iterator that returns numpy arrays of size [batch_size, 28, 28, 1]
    data_mean: mean of the split
    data_std: std of the split
  """
  cfg = config
  local_path = os.path.join(cfg['data/dir'], 'binarized_mnist.hdf5')
  if not os.path.exists(local_path):
    raise ValueError('need: ', local_path)
  f = h5py.File(local_path, 'r')
  if cfg['data/split'] == 'train_and_valid':
    train = f['train'][:]
    valid = f['valid'][:]
    data = np.vstack([train, valid])
  else:
    data = f[cfg['data/split']][:]
  try:
    if cfg['data/fixed_idx'] is not None:
      data = data[cfg['data/fixed_idx']:cfg['data/fixed_idx'] + 1]
  except:
    pass
  data = data[0:cfg['data/n_examples']]
  data_mean = np.mean(data, axis=0)
  data_std = np.std(data, axis=0)
  reshape = lambda t: np.reshape(t, (28, 28, 1))
  data_mean = reshape(data_mean)
  data_std = reshape(data_std)

  # create indexes for the data points.
  indexed_data = zip(range(len(data)), np.split(data, len(data)))
  def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
      # shuffle data
      idxs = np.arange(0, len(data))
      np.random.shuffle(idxs)
      shuf_data = [indexed_data[idx] for idx in idxs]
      for batch_idx in range(0, len(data), cfg['batch_size']):
        indexed_images_batch = shuf_data[batch_idx:batch_idx+cfg['batch_size']]
        indexes, images_batch = zip(*indexed_images_batch)
        images_batch = np.vstack(images_batch)
        images_batch = images_batch.reshape(
              (cfg['batch_size'], 28, 28, 1))
        yield indexes, images_batch

  return data_iterator(), data_mean, data_std


def remove_dir(config):
  """Delete directory contents if it exists."""
  cfg = config
  for f in tf.gfile.ListDirectory(cfg['log/dir']):
    path = os.path.join(cfg['log/dir'], f)
    if os.path.isdir(path):
      tf.gfile.DeleteRecursively(path)
    else:
      tf.gfile.Remove(path)
