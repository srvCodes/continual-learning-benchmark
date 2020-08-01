import numpy as np

def herding_selection(x, m, mean_=None):
    """
        Source: https://github.com/PatrickZH/End-to-End-Incremental-Learning/blob/39d6f4e594e805a713aa7a1deedbcb03d1f2c9cc/utils.py#L176
        Parameters
        ----------
        x: the features, n * dimension
        m: the number of selected exemplars
        Returns
        ----------
        pos_s: the position of selected exemplars
    """

    pos_s = []
    comb = 0
    mu = np.mean(x, axis=0, keepdims=False) if mean_ is None else mean_
    for k in range(m):
        det = mu * (k + 1) - comb
        dist = np.zeros(shape=(np.shape(x)[0]))
        for i in range(np.shape(x)[0]):
            if i in pos_s:
                dist[i] = np.inf
            else:
                dist[i] = np.linalg.norm(det - x[i])
        pos = np.argmin(dist)
        pos_s.append(pos)
        comb += x[pos]

    return pos_s