"""Functions for building models for sigmoid belief net."""

import tensorflow as tf
import numpy as np

layers = tf.contrib.layers
dist = tf.contrib.distributions
learn = tf.contrib.learn
fw = tf.contrib.framework


class Model:
  def __init__(self, config):
    self.config = config

  def prior_sample(self, n):
    return tf.cast(self.p_h.sample(n), self.config['dtype'])

  def log_prob(self, data, h):
    """Log joint of the model.
    log f(x, h) = log p(x | h) + \sum_{i} log p(h_i | h_{i + 1})
    """
    cfg = self.config
    init = tf.random_normal_initializer(mean=cfg['p/w_eps'], stddev=0.001)
    with tf.variable_scope('model') and fw.arg_scope([layers.fully_connected],
        weights_initializer=init, biases_initializer=init):
      p_h = dist.Bernoulli(
              p=np.ones(cfg['p/h_dim'], dtype=cfg['dtype']) * cfg['p/bernoulli_p'],
              name='p_h_%d' % cfg['p/n_layers'], validate_args=False)
      self.p_h = p_h
      log_p_h = 0.
      log_p_h += tf.reduce_sum(p_h.log_pmf(h[-1]), -1)
      for n in range(cfg['p/n_layers'] - 1, 0, -1):
        log_p_h += self.layer_log_p_h(n=n, h_layer=h[n - 1], h_above=h[n])
      p_x_given_h = self.likelihood(h[0])
      log_p_x_given_h = tf.reduce_sum(p_x_given_h.log_pmf(data), [2, 3, 4])
      posterior_predictive = tf.cast(
          p_x_given_h.sample(), tf.float32)
      tf.image_summary('posterior_predictive',
                       posterior_predictive[0, :, :, :, :],
                       max_images=cfg['batch_size'])
      self.log_likelihood_tensor = tf.reduce_sum(tf.reduce_mean(log_p_x_given_h, 0))
      log_p_x_h = log_p_x_given_h + log_p_h
      self.posterior_predictive = posterior_predictive
    return log_p_x_given_h

  def layer_log_p_h(self, n, h_layer, h_above):
    """Build a layer of the model.

    p(h_layer | h_above) = Bernoulli(h_below; sigmoid(w^T h_above + b))
    """
    cfg = self.config
    h_above = tf.reshape(
        h_above, [cfg['q/n_samples'] * cfg['batch_size'], cfg['p/h_dim']])
    logits = layers.fully_connected(h_above, cfg['p/h_dim'],
        activation_fn=None, scope='fc%d' % n)
    logits = tf.reshape(
            logits, [cfg['q/n_samples'], cfg['batch_size'], cfg['p/h_dim']])
    p_h_given_h = dist.Bernoulli(
            logits=logits, name='p_h_%d' % n, validate_args=False)
    log_p_h_given_h = tf.reduce_sum(p_h_given_h.log_pmf(h_layer), -1)
    return log_p_h_given_h

  def likelihood(self, h_0, reuse=False):
    """Log likelihood of the data."""
    cfg = self.config
    h_0 = tf.reshape(
        h_0, [cfg['q/n_samples'] * cfg['batch_size'], cfg['p/h_dim']])
    with tf.variable_scope('model', reuse=reuse):
      p_logits = layers.fully_connected(
              h_0, np.prod(cfg['data/shape']), activation_fn=None, scope='fc0')
    out_shape = ([cfg['q/n_samples'], cfg['batch_size']]
        + cfg['data/shape'])
    p_logits = tf.reshape(p_logits, out_shape)
    p_x_given_h = dist.Bernoulli(
        logits=p_logits, name='p_x_given_h_0', validate_args=False)
    sample = p_x_given_h.sample()
    self.p_x_given_h_p = tf.sigmoid(p_logits)
    return p_x_given_h


class Variational:
  """Build the variational or proposal for the sigmoid belief net.

  The architecture mirrors the model, but in reverse.
  """
  def __init__(self, config):
    self.config = config

  def sample(self, data):
    """Sample from the model."""
    cfg = self.config
    with tf.variable_scope('variational'):
      q_h, h = [], []
      q_h_0, h_0 = self.layer_q_and_h(0, data, cfg['q/n_samples'])
      q_h.append(q_h_0)
      h.append(h_0)
      for n in range(1, cfg['p/n_layers']):
        q_h_n, h_n = self.layer_q_and_h(n, h[n - 1], 1)
        q_h.append(q_h_n)
        h.append(h_n)
      self.q_h = q_h
      self.h = h
      return h

  def layer_q_and_h(self, n, layer_input, n_samples, reuse=False):
    """Build a layer of the variational / proposal distribution.

    q(h_0 | x) = Bernoulli(h_0; sigmoid(w^T x + b))
    q(h_above | h_below) = Bernoulli(h_above; sigmoid(w^T h_below + b))
    """
    cfg = self.config
    in_shape = layer_input.get_shape().as_list()
    if n == 0:
      flat_shape = [cfg['batch_size'], -1]
    else:
      flat_shape = [cfg['q/n_samples'] * cfg['batch_size'], -1]
    inp = tf.reshape(layer_input, flat_shape)
    q_h_above_logits = layers.fully_connected(
        inp, cfg['p/h_dim'], activation_fn=None, scope='fc%d' % n)
    if n > 0:
      q_h_above_logits = tf.reshape(q_h_above_logits,
          [cfg['q/n_samples'], cfg['batch_size'], cfg['p/h_dim']])
    q_h_above = dist.Bernoulli(
            logits=q_h_above_logits, name='q_h_%d' % n, validate_args=False)
    if n_samples == 1 and n != 0:
      sample = q_h_above.sample()
    else:
      sample = q_h_above.sample(n_samples)
    log_q_h = q_h_above.log_pmf(sample)
    q_h_above_sample = tf.cast(sample, cfg['dtype'])
    return (q_h_above, q_h_above_sample)

  def log_prob(self, h):
    """Evaluate log probability of samples."""
    cfg = self.config
    log_q_h = []
    for n in range(cfg['p/n_layers']):
      log_q_h.append(self.q_h[n].log_pmf(tf.stop_gradient(h[n])))
    self.log_q_h = log_q_h
    self.log_q_h_t = [tf.placeholder(shape=v.get_shape(), dtype=v.dtype) for v in log_q_h]
    return log_q_h
