# Continual Learning Benchmark [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repo contains the code for reproducing the results of the following papers (done as part of my Master's thesis at St Andrews):

1. [Benchmarking Continual Learning in Sensor-based Human Activity Recognition: an Empirical Analysis](http://arxiv.org/abs/2104.09396) [Accepted in the _Information Sciences_ (April 2021)]
2. [Continual Learning in Human Activity Recognition (HAR): An Emperical Analysis of Regularization](https://research-repository.st-andrews.ac.uk/handle/10023/20242) [ICML workshop on Continual Learning (July 2020)]

![Incremental learning](https://github.com/srvCodes/continual-learning-benchmark/blob/master/utils/img/incremental_learning.png)

A sub-total of 11 recent continual learning techniques have been implemented on a component-wise basis:

1. Maintaining Discrimination and Fairness in Class Incremental Learning (WA-MDF) [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Maintaining_Discrimination_and_Fairness_in_Class_Incremental_Learning_CVPR_2020_paper.pdf)]
2. Adjusting Decision Boundary for Class Imbalanced Learning (WA-ADB) [[Paper](https://ieeexplore.ieee.org/document/9081988)]
3. Large Scale Incremental Learning (BiC) [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf)]
4. Learning a Unified Classifier Incrementally via Rebalancing (LUCIR) [[Paper](http://dahualin.org/publications/dhl19_increclass.pdf)]
5. Incremental Learning in Online Scenario (ILOS) [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Incremental_Learning_in_Online_Scenario_CVPR_2020_paper.pdf)]
6. Gradient Episodic Memory for Continual Learning (GEM) [[Paper](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf)]
7. Efficient Lifelong Learning with A-GEM [[Paper](https://openreview.net/forum?id=Hkf2_sC5FX)]
8. Elastic Weight Consolidation (EWC) [[Paper](https://arxiv.org/pdf/1612.00796.pdf)]
9. Rotated Elastic Weight Consolidation (R-EWC) [[Paper](https://arxiv.org/abs/1802.02950)]
10. Learning without Forgetting (LwF) [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8107520)]
11. Memory Aware Synapses (MAS) [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-01219-9_9)]

Additionally, the following six exemplar-selection techniques are available (for memory-rehearsal):

1. Herding from ICaRL [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf)]
2. Frank-Wolfe Sparse Regression (FWSR) [[Paper](https://arxiv.org/abs/1811.02702)]
3. K-means sampling
4. DPP sampling 
5. Boundary-based sampling [[Paper](https://ieeexplore.ieee.org/document/8986833)]
6. Sensitivity-based sampling [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8949290)]

## Running the code

For training, please execute the `runner.sh` script that creates all the directories required for logging the outputs. One can add further similar commands for running further experiments.

For instance, training on *ARUBA* dataset with FWSR-styled exemplar selection:

```python
>>> python runner.py --dataset 'aruba' --total_classes 11 --base_classes 2 --new_classes 2 --epochs 160 --method 'kd_kldiv_wa1' --exemplar 'fwsr' # e.g. for FWSR-styled exemplar selection

```

## Proposed Forgetting Score

The existing forgetting measure metric [1] suffers from self-relativeness, i.e., the forgetting score will remain low throughout the training if the model did not learn much information about the class at the beginning. Class-imbalance scenarios (as in our case) further amplify its ramifications [2]. Code for our correction to the forgetting score can be found [here](https://github.com/srvCodes/continual-learning-benchmark/blob/master/train/result_analyser.py#L211).

## Datasets

The experiments were performed on 8 publicly available HAR datasets. These can downloaded from the drive link in `datasets/`.

## Experimental protocol

The experiments for each dataset and for each train set / exemplar size were performed on 30 random sequences of tasks. The logs in `output_reports/[dataname]` (created after executing the bash script) contain the performances of each individual task sequence as the incremental learning progresses. The final accuracy is then reported as the average over the 30 runs (see instructions below for evaluation).

## Evaluating the logs

For evaluation, please uncomment the lines per the instructions in `runner.py`. This can be used to measure forgetting scores [2], base-new-old accuracies, and average report by holdout sizes.

## Combination of techniques

The component-wise implementation of techniques nevertheless helps in playing with two or more techniques. This can be done by tweaking the `--method` argument. The table below details some of these combinations:

Technique | Argument for `--method`
------------ | -------------
Knowledge distillation with margin ranking loss (KD_MR) | kd_kldiv_mr
KD_MR with WA-MDF | kd_kldiv_mr_wa1
KD_MR with WA-ADB | kd_kldiv_mr_wa2
KD_MR with less forget constraint loss (KD_LFC_MR) | kd_kldiv_lfc_mr
KD_LFC_MR with WA-MDF | kd_kldiv_lfc_mr_wa1
KD_LFC_MR with WA-ADB | kd_kldiv_lfc_mr_wa2
Cosine normalisation with knowledge distillation | cn_kd_kldiv

Furthermore, the logits replacement tweak of ILOS and weight initialisation from LUCIR can be used with either of the above methods by simply setting the following arguments:

Technique | Argument 
------------ | ----------------
ILOS (with either of above) | `--replace_new_logits = True`
LUCIR-styled weight initialisation (with either of above) | `--wt_init = True`

Please feel free to play around with these. We would be interested in knowing if the combinations deliver better results for you!

## Notes on incremental classes

- All the experiments in our papers used number of base classes and incremental classes as 2. For replicating this, set `--base_classes = 2` and `--new_classes = 2`. 

- For offline learning (_i.e._, without incremental training), set `--base_classes` to the total number of classes in the dataset and `--new_classes = 0`.

- For experiments with permuted datasets, set `--base_classes = --new_classes` where `--base_classes` = the total number of classes in the dataset.

## Verification

The implementations have been verified through runs on Split-MNIST and Permumted-MNIST - also available for download in `datasets/`.


## Acknowledgement

Special thanks to [sairin1202](https://github.com/sairin1202)'s implementation of [BiC](https://github.com/sairin1202/BIC) and [Electronic Tomato](https://github.com/ElectronicTomato)'s implementation of [GEM/AGEM/EWC/MAS](https://github.com/ElectronicTomato/continue_leanrning_agem/tree/master/agents). 

## References

[1] Chaudhry, A., Dokania, P.K., Ajanthan, T., & Torr, P.H. (2018). Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence. ECCV.

[2] Kim, C. D., Jeong, J., & Kim, G. (2020). Imbalanced continual learning with partitioning reservoir sampling. ECCV.



## Cite

If you found this repo useful in your work, please feel free to cite us:

```bibtex
@article{jha2021continual,
  title={Continual Learning in Sensor-based Human Activity Recognition: an Empirical Benchmark Analysis},
  author={Jha, Saurav and Schiemer, Martin and Zambonelli, Franco and Ye, Juan},
  journal={Information Sciences},
  year={2021},
  publisher={Elsevier}
}
```

```bibtex
@article{jha2020continual,
  title={Continual learning in human activity recognition: an empirical analysis of regularization},
  author={Jha, Saurav and Schiemer, Martin and Ye, Juan},
  journal={Proceedings of Machine Learning Research},
  year={2020}
}
```
