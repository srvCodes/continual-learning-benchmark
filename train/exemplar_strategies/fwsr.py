import numpy as np
from tqdm import tqdm


def FWSR_identify_exemplars(beta, A, K, max_iterations, num_exemp, greedy=True, verbose=True, zeta=0, epsilon=0,
                            order=2, positive=False, term_thres=1e-8):
    """
    zeta, epsilon, order, beta correspond to the variables of the following problem
    minimize_X || A @ X - A ||_F**2 + ||zeta(X.T @ 1 - 1)||_2 ** 2 + epsilon ||X||_F^2
        s.t. ||X||_{1, order} = sum_{i} ||X^(i)||_order <= beta
    where X^(i) denotes the ith row of X
    order can be equal to 1, 2, or infinity (corresponding to the l1, l2, or l-infinity norm)
    K = A.T @ A
    if greedy == True, then the method will terminate if and when the number of non-zero rows of X is >= num_exemp
        otherwise, the method will run until either max_iterations is hit or termination condition is reached (dictated by term_thres)
    If postive == True, then method will optimize the obove objective with the added constraint: X >=0 elementwise
    see https://arxiv.org/abs/1811.02702 for more details about the algorithm itself
    """
    max_iterations += 1  # this is useful for consistency in testing
    if zeta != 0:
        K += zeta ** 2

    trK = np.trace(K)
    n, m = A.shape
    X = np.zeros((m, m))

    exemplar_index_lst = []
    cost_lst = []
    G_lst = []

    prev_KX = 0
    S = None
    step_size = None
    max_index = None
    trXTK = 0
    trXTKX = 0

    D = 0
    trSTKS = 0
    trSTKX = 0
    trSTK = 0
    step_size = 0

    row_norm_X = np.linalg.norm(X, axis=1, ord=2)
    exemplar_index_lst = np.where(row_norm_X != 0)[0]
    len_of_exemplar_index_lst = []

    pbar = tqdm(total=int(max_iterations), unit="iter", unit_scale=False, leave=False, disable=not verbose)

    for iteration in range(int(max_iterations)):
        if greedy and len(exemplar_index_lst) >= num_exemp:
            pbar.update(int(max_iterations) - iteration)
            break
        pbar.set_postfix(num_ex=len_of_exemplar_index_lst[-1] if len(len_of_exemplar_index_lst) > 0 else 0,
                         tol=G_lst[-1] if len(G_lst) > 0 else np.inf, refresh=False)
        pbar.update(1)

        X += step_size * D
        row_norm_X = np.linalg.norm(X, axis=1, ord=2)
        exemplar_index_lst = np.where(row_norm_X != 0)[0]
        len_of_exemplar_index_lst.append(len(exemplar_index_lst))

        if len(exemplar_index_lst) == 0:
            KX = np.zeros((m, m))
        elif step_size is None or prev_KX is None or S is None or max_index is None:
            # print("SHOULD NEVER GET HERE") # this case is just defensive programming
            KX = K[:, exemplar_index_lst].dot(X[exemplar_index_lst])
        elif max_index == -1:
            # print("max index is -1")
            KX = prev_KX * (1 - step_size)
            KX1 = K[:, exemplar_index_lst].dot(X[exemplar_index_lst])
            assert np.all(np.isclose(KX, KX1, atol=0))
        else:
            KX = prev_KX * (1 - step_size) + step_size * np.outer(K[:, max_index], S[max_index])

        prev_KX = KX
        trXTK = step_size * trSTK + (1 - step_size) * trXTK
        trXTKX = (1 - step_size) ** 2 * trXTKX + step_size ** 2 * trSTKS + 2 * step_size * (1 - step_size) * trSTKX
        cost_lst.append(trXTKX - 2 * trXTK + trK)

        if epsilon == 0:
            gradient = KX - K  # with respect to Z
        else:
            gradient = KX - K + epsilon * X  # with respect to Z

        max_index = get_max_index(gradient=gradient, order=order, positive=positive)  # next index to update
        if max_index == -1 and positive:
            S = np.zeros((m, m))
            D = -X
            numerator = - trXTK + trXTKX
            denominator = trXTKX
        else:
            gradient_max_row = gradient[max_index].flatten()
            S = np.zeros((m, m))
            S[max_index] = make_S_row(gradient_max_row=gradient_max_row, beta=beta, order=order, positive=positive)
            D = S - X

            trSTK = np.inner(S[max_index], K[max_index])
            trSTKS = K[max_index, max_index] * np.inner(S[max_index], S[max_index])
            trSTKX = np.inner(S[max_index], KX[max_index])

            numerator = trSTK - trXTK - trSTKX + trXTKX
            denominator = trSTKS - 2 * trSTKX + trXTKX

        G = -2 * np.einsum("ij, ij ->", gradient, D)
        G_lst.append(G)

        if G < term_thres:
            # myprint("EARLY TERMINATION", verbose)
            break

        step_size = max(0, min(1, numerator / denominator))

    pbar.close()
    if not greedy and num_exemp is None:
        exemplar_indices = exemplar_index_lst
    elif not greedy:
        exemplar_indices = make_exemplar_indices(X.T, num_exemp)
    else:
        exemplar_indices = exemplar_index_lst
        # if len(exemplar_indices) < num_exemp:
        #     myprint("ALERT: less than num_exemp were selected: " + str(len(exemplar_indices)))

    return exemplar_indices


def get_max_index(gradient, order, positive):
    if positive:
        if np.all(gradient >= 0):
            return -1
        gradient = np.where(gradient < 0, gradient, 0)

    if order == 2:
        return np.argmax(np.linalg.norm(gradient, axis=1, ord=2))
    elif np.isinf(order):
        return np.argmax(np.linalg.norm(gradient, axis=1, ord=1))
    elif order == 1:
        return np.argmax(np.linalg.norm(gradient, axis=1, ord=np.inf))
    raise Exception("Improper ord arguement; ord = " + str(ord))


def make_S_row(gradient_max_row, beta, order, positive):
    if positive:
        return make_S_row_positive(gradient_max_row, beta, order)

    if order == 2:
        if np.linalg.norm(gradient_max_row, ord=2) == 0:
            val = np.zeros_like(gradient_max_row)
            val[0] = beta
            return val
        return -1 * gradient_max_row / np.linalg.norm(gradient_max_row, ord=2) * beta + 0.
    if np.isinf(order):
        sign_vec = np.sign(gradient_max_row)
        sign_vec[sign_vec == 0] = 1  # this is just to make sure a vertex of the ball is selected
        return -1 * sign_vec * beta + 0.
    if order == 1:
        max_index = np.argmax(np.abs(gradient_max_row))
        max_sign = np.sign(gradient_max_row[max_index])

        if max_sign == 0:
            max_sign = 1  # this is just to make sure a vertex of the ball is selected

        return_vec = np.zeros_like(gradient_max_row)
        return_vec[max_index] = -1 * max_sign * beta
        return return_vec + 0.


def make_S_row_positive(gradient_max_row, beta, order):
    gradient_max_row = np.where(gradient_max_row < 0, gradient_max_row, 0.)
    if order == 2:
        return -1 * gradient_max_row / np.linalg.norm(gradient_max_row, ord=2) * beta + 0.
    if np.isinf(order):
        sign_vec = np.sign(gradient_max_row)
        return -1 * sign_vec * beta + 0.
    if order == 1:
        max_index = np.argmax(np.abs(gradient_max_row))
        max_sign = np.sign(gradient_max_row[max_index])

        return_vec = np.zeros_like(gradient_max_row)
        return_vec[max_index] = -1 * max_sign * beta
        return return_vec + 0.


def compute_inner_product_of_S_max_row(m, beta, order):
    """
    To compute the optimal step size, one of the trace terms need the inner product of s_max^T s_max
    This calculation depends on the order of the group lasso ball.
    """
    if order == 2 or order == 1:
        return beta ** 2
    elif np.isinf(order):
        return (beta ** 2) * m


def make_exemplar_indices(Z, num_exemp):
    """
    horizontal_norms refers to the horizontal norms of ZT which are the vertical norms of Z
    """
    horizontal_norms = np.linalg.norm(Z, ord=2, axis=0)
    total_norm_sum = np.sum(horizontal_norms)
    sorted_indices = np.flipud(np.argsort(horizontal_norms))[:num_exemp]

    m = Z.shape[0]

    # don't pick coefficients that aren't used at all
    last_index = num_exemp
    for idx in range(len(sorted_indices)):
        og_idx = sorted_indices[idx]
        if horizontal_norms[og_idx] == 0.0:
            last_index = idx
            myprint("ALERT: less than num_exemp were selected")
            break

    return sorted_indices[:last_index]


def fw_objective(AX, X):
    return np.linalg.norm(AX - A, ord="fro") ** 2


def myprint(s, to_print=True):
    if to_print:
        print(s)