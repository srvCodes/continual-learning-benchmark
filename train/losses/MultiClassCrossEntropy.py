import torch
from torch.autograd import Variable


def MultiClassCrossEntropy(logits, labels, T, device):
    """
    Source: https://github.com/ngailapdi/LWF/blob/baa07ee322d4b2f93a28eba092ad37379f565aca/model.py#L16
    :param logits: output logits of the model
    :param labels: ground truth labels
    :param T: temperature scaler
    :return: the loss value wrapped in torch.autograd.Variable
    """
    labels = Variable(labels.data, requires_grad=False).to(device)
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    labels = torch.softmax(labels / T, dim=1)
    # print('outputs: ', outputs)
    # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    # print('OUT: ', outputs)
    return Variable(outputs.data, requires_grad=True).to(device)
