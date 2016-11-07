"""Functions for building models for sigmoid belief net."""

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim
dist = tf.contrib.distributions


class Model:
  def __init__(self, config):
    self.config = config

  def log_prob(self, data, h, reuse=False):
    """Log joint of the model.
    log f(x, h) = log p(x | h) + \sum_{i} log p(h_i | h_{i + 1})
    """
    cfg = self.config
    with tf.variable_scope('model', reuse=reuse):
      p_h = dist.Bernoulli(p=0.5, name='p_h_%d' % (cfg['p/n_layers'] - 1), validate_args=False)
      log_p_h = 0.
      log_p_h += tf.reduce_sum(p_h.log_pmf(h[-1]), -1)
      for n in range(cfg['p/n_layers'] - 1, 0, -1):
        print 'building layer', n-1
        log_p_h += self.layer_log_p_h(n=n - 1, h_layer=h[n - 1], h_above=h[n])
      log_p_x_given_h = self.log_likelihood(data, h[0])
    log_p_x_h = log_p_h + log_p_x_given_h
    return log_p_x_h

  def layer_log_p_h(self, n, h_layer, h_above):
    """Build a layer of the model.

    p(h_layer | h_above) = Bernoulli(h_below; sigmoid(w^T h_above + b))
    """
    cfg = self.config
    h_above = tf.reshape(
        h_above, [cfg['q/n_samples'] * cfg['batch_size'], cfg['p/h_dim']])
    logits = slim.fully_connected(h_above, cfg['p/h_dim'],
        activation_fn=None, scope='fc%d' % n)
    logits = tf.reshape(logits, [cfg['q/n_samples'], cfg['batch_size'], cfg['p/h_dim']])
    p_h_given_h = dist.Bernoulli(logits=logits, name='p_h_%d' % n, validate_args=False)
    log_p_h_given_h = tf.reduce_sum(p_h_given_h.log_pmf(h_layer), -1)
    return log_p_h_given_h

  def log_likelihood(self, data, h_0):
    """Log likelihood of the data."""
    cfg = self.config
    p_logits = slim.fully_connected(h_0, np.prod(cfg['data/shape']), scope='fco')
    out_shape = ([cfg['q/n_samples'], cfg['batch_size']]
        + cfg['data/shape'])
    p_logits = tf.reshape(p_logits, out_shape)
    p_x_given_h = dist.Bernoulli(logits=p_logits, name='p_x_given_h', validate_args=False)
    posterior_predictive = p_x_given_h.sample()
    print posterior_predictive
    tf.image_summary('posterior_predictive', 
                     tf.cast(posterior_predictive[0, :, :, :, :], tf.uint8), 
                     max_images=cfg['batch_size'])
    log_likelihood = p_x_given_h.log_pmf(data)
    return tf.reduce_sum(log_likelihood, [2, 3, 4])


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
      q_h = []
      h = []
      q_h_0, h_0 = self.layer_q_and_h(0, data, cfg['q/n_samples'])
      q_h.append(q_h_0)
      h.append(h_0)
      for n in range(1, cfg['p/n_layers']):
        q_h_n, h_n = self.layer_q_and_h(n, h[n - 1], 1)
        q_h.append(q_h_n)
        h.append(h_n)
      self.q_h = q_h
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
    q_h_above_logits = slim.fully_connected(
        inp, cfg['p/h_dim'], activation_fn=None, scope='fco%d' % n)
    if n > 0:
      q_h_above_logits = tf.reshape(q_h_above_logits, 
          [cfg['q/n_samples'], cfg['batch_size'], cfg['p/h_dim']])
    q_h_above = dist.Bernoulli(logits=q_h_above_logits, name='q_h_%d' % n, validate_args=False)
    if n_samples == 1:
      sample = q_h_above.sample()
    else:
      sample = q_h_above.sample(n_samples)
    q_h_above_sample = tf.cast(sample, cfg['dtype'])
    return (q_h_above, q_h_above_sample)

  def log_prob(self, h):
    """Evaluate log probability of samples."""
    cfg = self.config
    log_q_h = []
    for n in range(cfg['p/n_layers']):
      log_q_h.append(self.q_h[n].log_pmf(tf.stop_gradient(h[n])))
    return log_q_h
