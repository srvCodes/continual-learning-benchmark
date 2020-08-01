# Continual Learning Benchmark [![forthebadge made-with-python](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.svg)](https://pytorch.org/)

This repo contains the code for reproducing the results for the following papers:

1. [Continual Learning in Human Activity Recognition: An Emperical Analysis of Regularization](https://drive.google.com/file/d/1B-p_xzlA2j56LtzxQyUHA34QwxedJosJ/view) [Accepted at ICML 2020 workshop on Continual Learning]
2. Benchmarking Continual Learning in Sensor-based Human Activity Recognition: an Empirical Analysis [Submitted to the Pervasive and Mobile Computing Journal]

![Incremental learning](https://github.com/srvCodes/continual-learning-benchmark/blob/master/utils/img/incremental_learning.png)


## Running the code

For training, please execute the `runner.sh` script that creates all the directories required for logging the outputs. One can add further similar commands for running further experiments.

For instance, training on *ARUBA* dataset with FWSR-styled exemplar selection:

```python
>>> python runner.py --dataset 'aruba' --total_classes 11 --base_classes 2 --new_classes 2 --epochs 160 --method 'kd_kldiv_wa1' --exemplar 'fwsr' # e.g. for FWSR-styled exemplar selection

```
