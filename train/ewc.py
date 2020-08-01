
from copy import deepcopy

from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import functional as F


class EWC(object):
    """
    Class for defining the diagonal fisher matrix.
    Author: https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
    """

    def __init__(self, model: nn.Module, dataset: list, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = Variable(p.data.to(self.device))

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_() # sets all data to zeros
            precision_matrices[n] = Variable(p.data.to(self.device)) # fisher matrix whose diagonal = precision of posterior p(theta | data)

        self.model.eval()
        for input, label in self.dataset:
            self.model.zero_grad()
            input, label = Variable(input), Variable(label)
            input = input.to(self.device)
            label = label.view(-1).to(self.device)
            output = self.model(input.float())
            # label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data.pow(2) / len(self.dataset) # grad = first order derivatives; point (b) - EWC paper

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2 # F_i * (theta_i  - theta'_i) ** 2
            loss += _loss.sum()
        return loss