"""
Created on Wed Jan 12 12:51:47 2020
Contains methods to extend pytorch neural network outputlayers
@author: Martin Schiemer
"""

import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


def create_proxy_outputs_from_onehot(model, task_pool, batch_y):
    current_output_neurons = list(model.children())[-1].in_features

    # inititalize array with 0s 
    proxy_outputs = torch.zeros([batch_y.shape[0], current_output_neurons])
    # convert one hot to class
    batch_classes = np.where(batch_y == 1)[1]

    # get index of the class in pool
    pool_ind = [task_pool.index(c) for c in batch_y]

    for i, j in enumerate(pool_ind):
        proxy_outputs[i, j] = 1

    return proxy_outputs


def create_proxy_outputs(model, task_pool, batch_y):
    current_output_neurons = list(model.children())[-1].in_features

    # inititalize array with 0s 
    proxy_outputs = torch.zeros(len(batch_y), dtype=torch.long)

    # get index of the class in pool
    pool_ind = [task_pool.index(c) for c in batch_y]

    for i, j in enumerate(pool_ind):
        proxy_outputs[i] = j

    return proxy_outputs


def batch_transform(batch, model, task_pool, device):
    batch_x, batch_y = batch
    batch_y = create_proxy_outputs(model, task_pool, batch_y)

    # calculate model outputs and loss and backprop
    if model.type == "fc":
        shape_length = len(batch_x.shape)
        batch_x = batch_x.view(-1, batch_x.shape[shape_length - 2] *
                               batch_x.shape[shape_length - 1] *
                               batch_x.shape[shape_length - 3])

    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    return batch_x, batch_y


def pad_random_weights(vec, pad_width, *_, **__):
    vec[vec.size - pad_width[1]:] = random.uniform(-1, 1)


def pad_normal_dist_weights(vec, pad_width, iaxis, kwargs, mean, var):
    vec[vec.size - pad_width[1]:] = random.uniform(-1, 1)


def update_weights(net, amnt_new_classes):
    layer_key = list(net.state_dict().keys())[-2]
    weights = net.state_dict()[layer_key].cpu().detach().numpy()
    # weights = net.state_dict()[f"fc{output_layer_nr}.weight"].cpu().detach().numpy()
    w_mean = np.mean(weights, axis=0)
    w_std = np.std(weights, axis=0)
    new_weights = np.pad(weights, ((0, amnt_new_classes), (0, 0)), mode="constant", constant_values=0)
    for i in reversed(range(amnt_new_classes)):
        for j in range(new_weights.shape[1]):
            new_weights[new_weights.shape[0] - 1 - i][j] = np.random.normal(w_mean[j], w_std[j])
    return new_weights


def update_bias(net, amnt_new_classes):
    bias_key = list(net.state_dict().keys())[-1]
    bias = net.state_dict()[bias_key].cpu().detach().numpy()
    b_mean = np.mean(bias)
    b_std = np.std(bias)
    new_bias = np.zeros(len(bias) + amnt_new_classes, dtype="f")
    new_bias[:len(bias)] = bias
    for i in range(amnt_new_classes):
        new_bias[-1 - i] = np.random.normal(b_mean, b_std) - np.log(amnt_new_classes)
    return new_bias


def transform_state_dic(state_dict, old_keys):
    new_dic = {key: data[1] for key, data in zip(old_keys, state_dict.items())}

    return new_dic


def add_output_neurons(net, amnt_old_classes, amnt_new_classes):
    newmodel = torch.nn.Sequential(*(list(net.modules())[:-1]),
                                   nn.Linear(list(net.modules())[-1].in_features,
                                             amnt_old_classes + amnt_new_classes))
    return newmodel


def kaiming_normal_init(m):
    # Source: https://github.com/ngailapdi/LWF/blob/baa07ee322d4b2f93a28eba092ad37379f565aca/model.py#L28
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


def update_model(NET_FUNCTION, net, data_shape, dataname, amnt_new_classes, device, lwf=False):
    amnt_old_classes = list(net.children())[-1].out_features
    if amnt_new_classes >= 1:
        if lwf:
            # Source: https://github.com/ngailapdi/LWF/blob/baa07ee322d4b2f93a28eba092ad37379f565aca/model.py#L73
            in_features = list(net.children())[-1].in_features
            out_features = amnt_old_classes
            weights = net.fc.weight.data
            new_out_features = out_features + amnt_new_classes
            net.fc = torch.nn.Linear(in_features,
                                     new_out_features, bias=False)
            kaiming_normal_init(net.fc.weight)
            net.fc.weight.data[:out_features] = weights
            net = net.to(device)
            return net
        else:
            new_output_layer_w = update_weights(net, amnt_new_classes)
            new_output_layer_bias = update_bias(net, amnt_new_classes)
            new_model = NET_FUNCTION(data_shape, amnt_old_classes + amnt_new_classes, dataname)

            for i, l in enumerate(zip(new_model.children(), net.children())):
                if i == len(list(new_model.children())) - 1:
                    l[0].weight = torch.nn.Parameter(torch.from_numpy(new_output_layer_w))
                    l[0].bias = torch.nn.Parameter(torch.from_numpy(new_output_layer_bias))
                else:
                    l[0].weight = l[1].weight
                    l[0].bias = l[1].bias
            del net
            new_model = new_model.to(device)
            return new_model
    else:
        return net


def check_model_integrity(old_model, new_model, verbose=False):
    for i in old_model.state_dict().keys():
        if (np.array_equal(old_model.state_dict()[i].cpu().numpy(), new_model.state_dict()[i].cpu().numpy())):
            if verbose:
                print(f"key {i} is the same for both nets")
        else:
            if verbose:
                print("\n", i, "\n")
            for h in range(len(old_model.state_dict()[i])):
                try:
                    if np.array_equal(old_model.state_dict()[i][h].numpy(), new_model.state_dict()[i][h].numpy()):
                        if verbose:
                            print(f"key {i} weights of neuron {h} are the same for both nets\n")
                    else:

                        print(f"key {i} weights of neuron {h} are different for both nets\n Differces at:")
                        print(old_model.state_dict()[i][h].numpy() - new_model.state_dict()[i][h].numpy())
                        print("\n")
                        return False
                except:
                    pass
    return True
