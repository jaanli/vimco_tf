"""Experiment configuration for SBN."""
import config

def get_config():
  """Define config and command line args."""
  cfg = config.Config()
  cfg.define_string('dtype', 'float32', 'dtype to use')
  # data
  cfg.define_integer('batch_size', 20, 'batch size')
  with cfg.scope('data'):
    cfg.define_string('split', 'train', 'which split to use')
    cfg.define_string('dir', '/Users/jaanaltosaar/dat', 'data directory')
    cfg.define_integer('n_examples', 50000, 'number of datapoints')
    cfg.define_list('shape', [28, 28, 1], 'shape of the data')
  # optimization
  with cfg.scope('optim'):
    cfg.define_float('beta1', 0.9, 'adam beta1 parameter')
    cfg.define_float('beta2', 0.999, 'adam beta2 parameter')
  # model
  with cfg.scope('p'):
    cfg.define_integer('h_dim', 100, 'latent dimensionality')
    cfg.define_integer('n_layers', 3, 'number of layers')
    cfg.define_float('learning_rate', 0.0001, 'learning rate for p')
  # variational
  with cfg.scope('q'):
    cfg.define_integer('n_samples', 5, 'number of samples')
    cfg.define_float('learning_rate', 0.0001, 'learning rate for q')
  # logging
  with cfg.scope('log'):
    cfg.define_string('dir', '/Users/jaanaltosaar/tmp', 'output directory')
    cfg.define_boolean('clear_dir', True, 'clear output directory')
  cfg.parse_args()
  return cfg
