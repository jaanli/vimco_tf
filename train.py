"""Train a sigmoid belief net with VIMCO on MNIST."""

import tensorflow as tf
import logging
import os
import numpy as np
import getpass
import models
import inferences
import util
import experiment_config
import time

tf.logging.set_verbosity(tf.logging.INFO)
fw = tf.contrib.framework
learn = tf.contrib.learn

def train(config):
  """Train sigmoid belief net on MNIST."""
  cfg = config

  # data
  data_iterator, np_data_mean, _ = util.provide_data(cfg)
  input_data = tf.placeholder(cfg['dtype'],
                              [cfg['batch_size']] +  cfg['data/shape'])
  data_mean = tf.placeholder(cfg['dtype'], cfg['data/shape'])
  input_data_centered = input_data - tf.expand_dims(data_mean, 0)
  tf.image_summary('data', input_data, max_images=cfg['batch_size'])
  def feed_fn():
    _, images = data_iterator.next()
    return {input_data: images, data_mean: np_data_mean}
    # return {input_data: np.random.binomial(1, 0.5, size=(cfg['batch_size'],28,28,1)), data_mean: 0. * np_data_mean + 0.5}

  model = models.Model(cfg)
  variational = models.Variational(cfg)
  inference = inferences.VariationalInference(
      cfg, model, variational, input_data, input_data_centered)
  model_vars = fw.get_model_variables('model')
  variational_vars = fw.get_model_variables('variational')
  summary_op = tf.merge_all_summaries()
  saver = tf.train.Saver()

  # train
  if not cfg['eval_only']:
    learn.train(graph=tf.get_default_graph(),
                output_dir=cfg['log/dir'],
                train_op=inference.train_op,
                loss_op=tf.reduce_mean(inference.vimco_elbo),
                feed_fn=feed_fn,
                supervisor_save_summaries_steps=1000,
                log_every_steps=1000,
                max_steps=cfg['optim/n_iterations'])

  # evaluate likelihood on validation set
  with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(cfg['log/dir']))
    np_log_x = 0.
    np_l = 0.
    cfg.update({'data': {'split': 'valid', 'n_examples': 10000}})
    data_iterator, np_data_mean, _ = util.provide_data(cfg)
    for i in range(cfg['data/n_examples'] / cfg['data/batch_size']):
      _, images = data_iterator.next()
      tmp_np_log_x, tmp_np_l = sess.run(
          [model.log_likelihood_tensor, inference.vimco_elbo],
          {input_data: images, data_mean: np_data_mean})
      np_log_x += tmp_np_log_x
      np_l += np.sum(tmp_np_l)
    print('for validation set -- elbo: %.3f\tlog_likelihood: %.3f' % (
        np_l / cfg['data/n_examples'], np_log_x / cfg['data/n_examples']))


def main(_):
  cfg = experiment_config.get_config()
  print cfg
  if getpass.getuser() == 'jaan':
    cfg.update({'data': {'dir':'/home/jaan/dat'},
        'log': {'dir': '/home/jaan/fit/vimco_tf'}})
  train(cfg)


if __name__ == '__main__':
  tf.app.run()
