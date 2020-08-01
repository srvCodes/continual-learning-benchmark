import torch
from torch import nn
from torch.autograd import Variable
from .BinDevianceLoss import BinDevianceLoss
import numpy as np


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


class BinBranchLoss(nn.Module):
    def __init__(self, margin=0.5, slice=[0, 170, 341, 512]):
        super(BinBranchLoss, self).__init__()
        self.s = slice
        self.margin = margin

    def forward(self, inputs, targets):
        inputs = [inputs[:, self.s[i]:self.s[i+1]]
                  for i in range(len(self.s)-1)]
        loss_list, prec_list, pos_d_list, neg_d_list = [], [], [], []

        for input in inputs:
            loss, prec, pos_d, neg_d = BinDevianceLoss(margin=self.margin)(input, targets)
            loss_list.append(loss)
            prec_list.append(prec)
            pos_d_list.append(pos_d)
            neg_d_list.append(neg_d)

        loss = torch.mean(torch.cat(loss_list))
        prec = np.mean(prec_list)
        pos_d = np.mean((pos_d_list))
        neg_d = np.mean((neg_d_list))

        return loss, prec, pos_d, neg_d