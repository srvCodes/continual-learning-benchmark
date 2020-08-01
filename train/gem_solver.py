
import numpy as np
import quadprog
import torch


def project2cone2(gradient, memories, device, margin=0.5, eps=1e-3):
    """
        Source: https://github.com/ElectronicTomato/continue_leanrning_agem/blob/master/agents/exp_replay.py#L317
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
        Modified from: https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py#L70
    """
    memories_np = memories.cpu().contiguous().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    # print(memories_np.shape, gradient_np.shape)
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose())
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    P = P + G * eps
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    new_grad = torch.Tensor(x).view(-1)
    new_grad = new_grad.to(device)
    return new_grad