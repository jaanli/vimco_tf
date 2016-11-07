"""Configuration class that includes command line arguments."""
import contextlib
import argparse
import json
import collections
import copy


_GLOBAL_PARSER = argparse.ArgumentParser()


"""One-line tree from https://gist.github.com/hrldcpr/2012250"""
tree = lambda: collections.defaultdict(tree)


def _flatten(d, parent_key='', sep='/'):
  """Flatten nested dicts.

  From: http://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
  """
  items = []
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.MutableMapping):
      items.extend(_flatten(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)

def _default_to_regular(d):
  """Convert defaultdict to regular dict to throw KeyErrors."""
  if isinstance(d, collections.defaultdict):
    d = {k: _default_to_regular(v) for k, v in d.iteritems()}
  return d


def _define_helper(arg_name, default_value, docstring, argtype):
  """Registers 'arg_name' with 'default_value' and 'docstring'."""
  _GLOBAL_PARSER.add_argument("--" + arg_name,
                              default=default_value,
                              help=docstring,
                              type=argtype)


class Config(object):
  """Stores and retrieves configuration.

  Light wrapper around argparse, mostly taken from:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/flags.py

  Custom lookup: if key doesn't exist at the lowest level, pop up a level.
  """
  def __init__(self, dct=None):
    self._scope = None
    self._path = lambda s: self._scope + '/' + s if self._scope else s
    if dct is not None:
      self._tree = dct
      self.parsed = True
    else:
      self._tree = tree()
      self.parsed = False

  def __getitem__(self, key):
    if not self.parsed:
      self.parse_args()
    path = self._path(key)
    path = path.split('/')
    leaf_key = path[-1]
    while True:
      path.pop(-1)
      tree = self._tree
      for node in path:
        tree = tree[node]
      if leaf_key in tree:
        return tree[leaf_key]
      if len(path) == 0:
        raise KeyError("Parameter %s Not Found" % key)

  def __str__(self):
    return json.dumps(self._tree, sort_keys=True, indent=2)

  @contextlib.contextmanager
  def scope(self, scope_name):
    self._scope = scope_name
    yield
    self._scope = None

  def update(self, dictionary):
    flat = _flatten(dictionary)
    for key, value in flat.items():
      self._set_value(key, value, update=True)

  def _set_value(self, arg_name, value, update=False):
    path = arg_name.split('/')
    leaf_key = path.pop()
    tree = self._tree
    for node in path:
      tree = tree[node]
    if update and leaf_key not in tree:
      raise KeyError(
          'Key: %s not in config so it cannot be updated.' % leaf_key)
    else:
      tree[leaf_key] = value

  def copy(self):
    return Config(copy.deepcopy(self._tree))

  def parse_args(self):
    result, unparsed = _GLOBAL_PARSER.parse_known_args()
    for arg_name, val in vars(result).items():
      self._set_value(arg_name, val)
    self.parsed = True
    self._tree = _default_to_regular(self._tree)
    if unparsed:
      raise Warning('Unparsed args: %s' % unparsed)

  def define_string(self, arg_name, default_value, docstring):
    """defines an arg of type 'string'.
    Args:
      arg_name: The name of the arg as a string.
      default_value: The default value the arg should take as a string.
      docstring: A helpful message explaining the use of the arg.
    """
    arg_name = self._path(arg_name)
    _define_helper(arg_name, default_value, docstring, str)


  def define_integer(self, arg_name, default_value, docstring):
    """defines an arg of type 'int'.
    Args:
      arg_name: The name of the arg as a string.
      default_value: The default value the arg should take as an int.
      docstring: A helpful message explaining the use of the arg.
    """
    arg_name = self._path(arg_name)
    _define_helper(arg_name, default_value, docstring, int)


  def define_boolean(self, arg_name, default_value, docstring):
    """defines an arg of type 'boolean'.
    Args:
      arg_name: The name of the arg as a string.
      default_value: The default value the arg should take as a boolean.
      docstring: A helpful message explaining the use of the arg.
    """
    arg_name = self._path(arg_name)
    # Register a custom function for 'bool' so --arg=True works.
    def str2bool(v):
      return v.lower() in ('true', 't', '1')
    _GLOBAL_PARSER.add_argument('--' + arg_name,
                                nargs='?',
                                const=True,
                                help=docstring,
                                default=default_value,
                                type=str2bool)

    # Add negated version, stay consistent with argparse with regard to
    # dashes in arg names.
    _GLOBAL_PARSER.add_argument('--no' + arg_name,
                                action='store_false',
                                dest=arg_name.replace('-', '_'))

  def define_float(self, arg_name, default_value, docstring):
    """defines an arg of type 'float'.
    Args:
      arg_name: The name of the arg as a string.
      default_value: The default value the arg should take as a float.
      docstring: A helpful message explaining the use of the arg.
    """
    arg_name = self._path(arg_name)
    _define_helper(arg_name, default_value, docstring, float)

  def define_list(self, arg_name, default_value, docstring):
    """defines an arg of type 'list'.
    Args:
      arg_name: The name of the arg as a string.
      default_value: The default value the arg should take as a float.
      docstring: A helpful message explaining the use of the arg.
    """
    arg_name = self._path(arg_name)
    self._set_value(arg_name, default_value)



def main():
  # run tests
  cfg = Config()
  cfg.define_float('learning_rate', 0.5, 'docstring')
  cfg.define_integer('hidden_size', 200, 'hidden')
  cfg.define_float('model/learning_rate', 0.8, 'docstr')
  with cfg.scope('variational'):
    cfg.define_float('learning_rate', 2., 'learning rate for variational')
  cfg.parse_args()
  # test getting
  print cfg
  print cfg['variational/hidden_size']
  print cfg['hidden_size']
  print cfg['variational']
  print 'variational learning rate', cfg['variational/learning_rate']
  with cfg.scope('variational'):
    print cfg['learning_rate']

  # test loading
  cfg['variational'].update({'learning_rate': 0.08})
  print cfg['variational']
  dct = {'learning_rate': 0.00001, 'variational': {'learning_rate': '0.01'}}
  cfg.update(dct)
  print cfg


if __name__ == '__main__':
  main()
