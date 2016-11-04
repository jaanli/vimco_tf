"""Train a sigmoid belief net with VIMCO on MNIST."""

import tensorflow as tf
import logging
import util
import experiment_config
import os

tf.logging.set_verbosity(tf.logging.INFO)

st = tf.contrib.bayesflow.stochastic_tensor
sg = tf.contrib.bayesflow.stochastic_graph
sge = tf.contrib.bayesflow.stochastic_gradient_estimators
dist = tf.contrib.distributions
fw = tf.contrib.framework
slim = tf.contrib.slim
learn = tf.contrib.learn

def train(config):
  """Train sigmoid belief net on MNIST."""
  cfg = config
  if cfg['log/clear_dir']:
    for f in tf.gfile.ListDirectory(cfg['log/dir']):
      tf.gfile.Remove(os.path.join(cfg['log/dir'], f))

  # data
  input_data = tf.placeholder(cfg['dtype'],
                              [cfg['optim/batch_size']] +  cfg['data/shape'])
  data_iterator, _, _ = util.provide_data(cfg)
  def feed_fn():
    """Return feed dict when called."""
    _, images = data_iterator.next()
    return {input_data: images}

  # model
  # model has Bernoulli latent variables with Bernoulli likelihood
  p_z = dist.Bernoulli(p=0.5, validate_args=False)
  def build_likelihood(z, reuse=False):
    z = tf.cast(z, cfg['dtype'])
    z = tf.reshape(z, [cfg['q/n_samples'] * cfg['optim/batch_size'], -1])
    net = slim.fully_connected(z, cfg['p/hidden_size'],
        activation_fn=tf.nn.tanh, scope='fc1')
    p_logits = slim.fully_connected(net, cfg['data/size'], scope='fco')
    out_shape = [cfg['q/n_samples'],
                 cfg['optim/batch_size']] + cfg['data/shape']
    p_logits = tf.reshape(p_logits, out_shape)
    p_x_given_z = dist.Bernoulli(logits=p_logits, validate_args=False)
    return p_x_given_z

  # variational
  with tf.variable_scope('variational'):
    net_input = slim.flatten(input_data)
    net = slim.fully_connected(
        net_input, cfg['q/hidden_size'], scope='fc1', activation_fn=tf.nn.tanh)
    q_logits = slim.fully_connected(
        net, cfg['p/z_dim'], activation_fn=None, scope='fco')
    with st.value_type(st.SampleValue(n=cfg['q/n_samples'])):
      q_z = st.StochasticTensor(
          dist.Bernoulli, logits=q_logits, validate_args=False)

  # inference
  # build the ELBO and optimize it with respect to model and variational params
  with tf.variable_scope('model'):
    p_x_given_z = build_likelihood(q_z).log_pmf(input_data)
    E_log_likelihood = tf.reduce_sum(p_x_given_z, [2, 3, 4])
    E_log_prior = tf.reduce_sum(p_z.log_pmf(q_z), -1)
  E_log_q = tf.reduce_sum(q_z.distribution.log_pmf(q_z), -1)
  # elbo is monte carlo estimate of log joint minus entropy
  elbo = E_log_likelihood + E_log_prior - E_log_q
  mean_elbo = tf.reduce_mean(tf.reduce_mean(elbo, 0), 0)
  loss_q, w = util.build_vimco_loss(elbo)
  # loss_q = elbo
  loss_p = elbo * w
  tf.scalar_summary('elbo', mean_elbo)

  pre_score_function = tf.reduce_sum(q_z.distribution.log_prob(q_z), -1)
  score_loss_q = (pre_score_function * tf.stop_gradient(loss_q))
  optimizer = tf.train.AdamOptimizer(cfg['q/learning_rate'],
                                      beta1=cfg['optim/beta1'],
                                      beta2=cfg['optim/beta2'])
  train_q = optimizer.minimize(-score_loss_q,
                               var_list=fw.get_variables('variational'))
  train_p = optimizer.minimize(-loss_p, var_list=fw.get_variables('model'))
  global_step = fw.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)
  with tf.control_dependencies([increment_global_step]):
    train_op = tf.group(train_q, train_p)

  # monitors = [learn.monitors.PrintTensor([mean_elbo], every_n=100)]

  learn.train(graph=tf.get_default_graph(),
              output_dir=cfg['log/dir'],
              train_op=train_op,
              loss_op=mean_elbo,
              feed_fn=feed_fn,
              log_every_steps=10000)
              # log_every_steps=100,
              # monitors=monitors
              # supervisor_save_summaries_steps=100)
  #todo add update funcition to train to be called every so often


def main(_):
  cfg = experiment_config.get_config()
  train(cfg)


if __name__ == '__main__':
  tf.app.run()
