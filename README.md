# vimco_tf
Variational Inference for Monte Carlo Objective (VIMCO) in tensorflow. The paper is here: https://arxiv.org/abs/1602.06725

This gets to a log-likelihood of `-94.3` nats on the validation set of the binarized MNIST data.

Summaries and posterior predictives can be viewed on tensorboard:

![tensorboard](http://i.imgur.com/h9L1ygN.png)

This is heavily based off of Joost's implementation at https://github.com/y0ast/VIMCO (thank you Joost!)
