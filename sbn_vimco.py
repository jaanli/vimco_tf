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

tf.logging.set_verbosity(tf.logging.INFO)

st = tf.contrib.bayesflow.stochastic_tensor
sg = tf.contrib.bayesflow.stochastic_graph
sge = tf.contrib.bayesflow.stochastic_gradient_estimators
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
  data_iterator, data_mean, data_std = util.provide_data(cfg)
  input_data = tf.placeholder(cfg['dtype'],
                              [cfg['batch_size']] +  cfg['data/shape'])
  tf.image_summary('data', input_data, max_images=cfg['batch_size'])

  #input_data = tf.ones(shape=[cfg['batch_size']] + cfg['data/shape'],
  #                              dtype=cfg['dtype'])
  # normalize the data
  input_data_centered = input_data - data_mean
  def feed_fn():
    _, images = data_iterator.next()
    return {input_data: images}

  model = models.Model(cfg)
  variational = models.Variational(cfg)
  inference = inferences.VariationalInference(
      cfg, model, variational, input_data, input_data_centered)

  # monitors = [learn.monitors.PrintTensor([mean_elbo], every_n=100)]

  learn.train(graph=tf.get_default_graph(),
              output_dir=cfg['log/dir'],
              train_op=inference.train_op,
              loss_op=inference.mean_elbo,
              feed_fn=feed_fn,
              supervisor_save_summaries_steps=1,
              log_every_steps=1)
              # monitors=monitors
  #todo add update funcition to train to be called every so often


def main(_):
  cfg = experiment_config.get_config()
  print cfg
  if getpass.getuser() == 'jaan':
    cfg.update({'data': {'dir':'/home/jaan/dat'},
        'log': {'dir': '/home/jaan/fit/vimco_tf'}})
  elif getpass.getuser() == 'alessandra':
    cfg.update({'data': {'dir': '/Users/alessandra/Downloads/BinarizedMNIST'},
        'log': {'dir': '/Users/alessandra/Downloads/log'}})
  train(cfg)


if __name__ == '__main__':
  tf.app.run()
