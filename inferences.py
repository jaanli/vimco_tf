"""Variational inference."""
import tensorflow as tf
import util

fw = tf.contrib.framework

class VariationalInference:
  def __init__(self, config, model, variational, data, data_centered):
    cfg = config
    global_step = fw.get_or_create_global_step()
    h = variational.sample(data_centered)
    log_q_h_list = variational.log_prob(h)
    log_q_h = tf.add_n([tf.reduce_sum(t, -1) for t in log_q_h_list])
    log_p_x_h = model.log_prob(data, h)
    elbo = log_p_x_h - log_q_h
    summary = lambda tag, v: tf.scalar_summary(tag, tf.reduce_mean(tf.reduce_mean(v, 0), 0))
    summary('log_p_x_h', log_p_x_h)
    summary('neg_log_q_h', -log_q_h)
    summary('elbo', elbo)
    if cfg['use_constraint']:
      k = tf.train.exponential_decay(
            cfg['c/init'], global_step, cfg['optim/n_iterations'],
            cfg['c/decay_rate'], staircase=True)
      constraint = -tf.add_n(
          [k * tf.square(log_q_h - tf.reduce_sum(old, -1)) for old in variational.log_q_h_t])
      summary('constraint', constraint)
      elbo = elbo + constraint
      summary('elbo_plus_constraint', elbo)
    loss, vimco_elbo = util.build_vimco_loss(elbo, log_q_h, log_q_h_list)
    optimizer = tf.train.AdamOptimizer(cfg['optim/learning_rate'],
                                        beta1=cfg['optim/beta1'],
                                        beta2=cfg['optim/beta2'],
                                        epsilon=cfg['optim/epsilon'])
    train_op = optimizer.minimize(-loss, global_step=global_step)
    tf.scalar_summary('vimco_elbo', vimco_elbo)
    self.vimco_elbo =  vimco_elbo
    self.global_step = global_step
    self.train_op = train_op

