"""Variational inference."""
import tensorflow as tf
import util

fw = tf.contrib.framework

class VariationalInference:
  def __init__(self, config, model, variational, data, data_centered):
    cfg = config
    h = variational.sample(data_centered)
    log_q_h_list = variational.log_prob(h)
    log_q_h = tf.add_n([tf.reduce_sum(t, -1) for t in log_q_h_list])
    log_p_x_h = model.log_prob(data, h)
    summary = lambda tag, v: tf.scalar_summary(tag, tf.reduce_mean(tf.reduce_mean(v, 0), 0))
    summary('log_p_x_h', log_p_x_h)
    summary('neg_log_q_h', -log_q_h)
    elbo = log_p_x_h - log_q_h
    summary('elbo', elbo)
    loss, vimco_elbo = util.build_vimco_loss(elbo, log_q_h, log_q_h_list)
    optimizer = tf.train.AdamOptimizer(cfg['optim/learning_rate'],
                                        beta1=cfg['optim/beta1'],
                                        beta2=cfg['optim/beta2'],
                                        epsilon=cfg['optim/epsilon'])
    global_step = fw.get_or_create_global_step()
    train_op = optimizer.minimize(-loss, global_step=global_step)
    tf.scalar_summary('vimco_elbo', tf.reduce_mean(vimco_elbo))
    self.vimco_elbo = vimco_elbo
    # self.vimco_elbo = tf.reduce_mean(tf.reduce_mean(log_p_x_h, 0), 0)
    self.train_op = train_op

