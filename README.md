# vimco_tf
Variational Inference for Monte Carlo Objective (VIMCO) in tensorflow. The paper is here: https://arxiv.org/abs/1602.06725

This gets to a log-likelihood of `-94.3` nats on the validation set of the binarized MNIST data.

## How to run

Important: needs to be run with a tensorflow version of [at least 0.11.0](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation).

```
# get the binarized MNIST dataset, save to /tmp/binarized_mnist.hdf5
python make_binarized_mnist_hdf5_file.py

# run sbn training with vimco. ideally on GPU (10x speedup)
python sbn_vimco.py

#  visualize logs
tensorboard --logdir /tmp
```

Summaries and posterior predictives can be viewed on tensorboard:

![tensorboard](http://i.imgur.com/h9L1ygN.png)

This is heavily based off of Joost's implementation at https://github.com/y0ast/VIMCO (thank you Joost!)
