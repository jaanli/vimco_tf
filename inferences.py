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
    elbo = log_p_x_h - log_q_h
    loss = util.build_vimco_loss(elbo, log_q_h)
    #score_loss_q = tf.add_n(
    #    [t * tf.stop_gradient(tf.expand_dims(loss_q, -1)) for t in log_q_h_list])
    optimizer = tf.train.AdamOptimizer(cfg['q/learning_rate'],
                                        beta1=cfg['optim/beta1'],
                                        beta2=cfg['optim/beta2'])
    global_step = fw.get_or_create_global_step()
    train_op = optimizer.minimize(-loss, global_step=global_step)
    mean_elbo = tf.reduce_mean(tf.reduce_mean(elbo, 0), 0)
    tf.scalar_summary('elbo', mean_elbo)
    self.mean_elbo = mean_elbo
    self.train_op = train_op

