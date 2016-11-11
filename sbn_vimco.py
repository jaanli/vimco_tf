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
  if cfg['log/clear_dir']:
    util.remove_dir(cfg)
  np.random.seed(433423)
  tf.set_random_seed(435354)

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

  model = models.Model(cfg)
  variational = models.Variational(cfg)
  inference = inferences.VariationalInference(
      cfg, model, variational, input_data, input_data_centered)
  print [v.name for v in tf.trainable_variables()]
  prior_predictive = model.likelihood(
      model.prior_sample(n=cfg['batch_size'] * cfg['q/n_samples']), reuse=True)
  prior_predictive = tf.cast(prior_predictive.sample(), cfg['dtype'])
  tf.image_summary(
      'prior_predictive', prior_predictive[0, :, :, :, :], max_images=cfg['batch_size'])
  print prior_predictive
  model_vars = fw.get_model_variables('model')
  variational_vars = fw.get_model_variables('variational')
  summary_op = tf.merge_all_summaries()
  saver = tf.train.Saver(max_to_keep=None)

  # # train
  # learn.train(graph=tf.get_default_graph(),
  #             output_dir=cfg['log/dir'],
  #             train_op=inference.train_op,
  #             loss_op=inference.vimco_elbo,
  #             feed_fn=feed_fn,
  #             supervisor_save_summaries_steps=100,
  #             log_every_steps=100,
  #             # max_steps=1)
  #             max_steps=cfg['optim/n_iterations'])
  with tf.Session() as sess:
    writer = tf.train.SummaryWriter(cfg['log/dir'], sess.graph)
    if cfg['ckpt_to_restore'] is not None:
      ckpt = cfg['ckpt_to_restore']
    else:
      ckpt = tf.train.latest_checkpoint(cfg['log/dir'])
    if ckpt is not None:
      saver.restore(sess, ckpt)
    else:
      sess.run(tf.initialize_all_variables())
    t0 = time.time()
    _, images = data_iterator.next()
    posterior_feed = {input_data: images, data_mean: np_data_mean}
    for i in range(cfg['optim/n_iterations']):
      feed_dict = feed_fn()
      s = sess.run(inference.global_step)
      if (s < cfg['c/lag'] and s == 0) or s % cfg['c/lag'] == 0:
        old_entropy = sess.run(variational.log_q_h, feed_dict)
        new_dict = {ph: t for ph, t in zip(variational.log_q_h_t, old_entropy)}
      feed_dict.update(new_dict)
      # print [(k.name, v.shape) for k,v in feed_dict.items()]
      if s % cfg['log/print_every'] == 0:
        # time.sleep(1)
        np_elbo = sess.run(inference.vimco_elbo, feed_dict)
        print 'iter %d elbo: %.3f speed: %.3f' % (s, np_elbo, (time.time() - t0) / cfg['log/print_every'])
        summary_str = sess.run(summary_op, feed_dict)
        writer.add_summary(summary_str, global_step=s)
        saver.save(
            sess, os.path.join(cfg['log/dir'], 'all-params'), global_step=s)
        util.save_prior_posterior_predictives(cfg, sess, inference, prior_predictive, model.posterior_predictive, posterior_feed, images)
        t0 = time.time()
      sess.run(inference.train_op, feed_dict)


  # evaluate likelihood on validation set
  with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(cfg['log/dir']))
    np_log_x = 0.
    cfg.update({'data': {'split': 'valid', 'n_examples': 10000}})
    data_iterator, np_data_mean, _ = util.provide_data(cfg)
    for i in range(cfg['data/n_examples'] / cfg['data/batch_size']):
      _, images = data_iterator.next()
      feed_dict = {input_data: images, data_mean: np_data_mean}
      np_log_x += sess.run(model.log_likelihood_tensor, feed_dict)
    print ('log-likelihood on evaluation set is: ',
        np_log_x / cfg['data/n_examples'])


def main(_):
  cfg = experiment_config.get_config()
  print cfg
  cfg.update({'ckpt_to_restore': '/home/jaan/fit/vimco/vimco_tf/all-params-300000'})
  if getpass.getuser() == 'jaan':
    cfg.update({'data': {'dir':'/home/jaan/dat'},
        'log': {'dir': '/home/jaan/fit/vimco/vimco_tf_tmp'}})
        # 'log': {'dir': '/home/jaan/fit/vimco/vimco_tf_constraint'}})
  elif getpass.getuser() == 'alessandra':
    cfg.update({'data': {'dir': '/Users/alessandra/Downloads/BinarizedMNIST'},
        'log': {'dir': '/Users/alessandra/Downloads/log'}})
  train(cfg)


if __name__ == '__main__':
  tf.app.run()
