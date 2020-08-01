#!/usr/bin/env python
# coding=utf-8
# !/usr/bin/env python
# coding=utf-8
import numpy as np
import torch


def compute_features(tg_feature_model, cls_idx, evalloader, num_samples, num_features, device=None):
    tg_feature_model.eval()

    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            outputs = tg_feature_model(inputs.float().cuda())
            # print(num_samples, num_features, inputs.shape, start_idx, len(evalloader.dataset), targets, outputs.shape)
            features[start_idx:start_idx + inputs.shape[0], :] = np.squeeze(outputs.cpu())
            start_idx = start_idx + inputs.shape[0]
    assert (start_idx == num_samples)
    return features
