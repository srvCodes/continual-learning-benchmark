import random

import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor
from torch import from_numpy, ones, zeros
from torch.utils import data
from . import modified_linear

PATH_TO_SAVE_WEIGHTS = 'saved_weights/'

def get_layer_dims(dataname):
    res_ = [1,2,2,4] if dataname in ['dsads'] else [1,2,4] if dataname in ['opp'] else [0.5, 1, 2] \
        if dataname in ['hapt', 'milan', 'pamap', 'aruba'] else [500, 500] if dataname in ['cifar100'] else [100, 100, 100] \
        if dataname in  ['mnist', 'permuted_mnist'] else [1,2,2]
    return res_

class Net(nn.Module):
    def __init__(self, input_dim, n_classes, dataname, lwf=False, cosine_liner=False):
        super(Net, self).__init__()
        self.dataname = dataname
        layer_nums = get_layer_dims(self.dataname)
        self.layer_sizes = layer_nums if self.dataname in ['cifar100', 'mnist'] else\
            [int(input_dim / num) for num in layer_nums]
        self.fc0 = nn.Linear(input_dim, self.layer_sizes[0])
        if len(self.layer_sizes) == 2:
            self.fc_penultimate = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        elif len(self.layer_sizes) == 3:
            self.fc1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
            self.fc_penultimate = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        elif (len(self.layer_sizes) == 4):
            self.fc1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
            self.fc2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
            self.fc_penultimate = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        final_dim = self.fc_penultimate.out_features
        self.fc = modified_linear.CosineLinear(final_dim, n_classes) if cosine_liner \
            else nn.Linear(final_dim, n_classes, bias=lwf==False) # no biases for LwF

    def forward(self, x):
        x = F.relu(self.fc0(x))
        if len(self.layer_sizes) > 2:
            x = F.relu(self.fc1(x))
            if len(self.layer_sizes) > 3:
                x = F.relu(self.fc2(x))
        x = F.relu(self.fc_penultimate(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Dataset(data.Dataset):
    def __init__(self, features, labels):
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        X = from_numpy(self.features[idx])
        y = self.labels[idx]
        y = LongTensor([y])
        return X, y

    def get_sample(self, sample_size):
        return random.sample(self.features, sample_size)


class BiasLayer(nn.Module):
    def __init__(self, device):
        super(BiasLayer, self).__init__()
        self.beta = nn.Parameter(ones(1, requires_grad=True, device=device))
        self.gamma = nn.Parameter(zeros(1, requires_grad=True, device=device))

    def forward(self, x):
        return self.beta * x + self.gamma

    def printParam(self, i):
        print(i, self.beta.item(), self.gamma.item())

    def get_beta(self):
        return self.beta

    def get_gamma(self):
        return self.gamma

    def set_beta(self, new_beta):
        self.beta = new_beta

    def set_gamma(self, new_gamma):
        self.gamma = new_gamma

    def set_grad(self, bool_value):
        self.beta.requires_grad = bool_value
        self.gamma.requires_grad = bool_value
