import numpy as np


def probability_differences(network_output, correct_vector, loss):
    """
    Calculates the difference between the two most probable classes
    """
    detached_output = network_output.cpu().detach().numpy()

    # if logit probabilities are used we need to get normal probabilities
    if np.sum(detached_output) != 1:
        detached_output = np.array([np.exp(x) / np.sum(np.exp(detached_output))
                                    for x in detached_output])

    ind = np.argpartition(detached_output, -2, axis=1)[:, -2:]
    # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops
    maxs = detached_output[np.arange(network_output.shape[0])[:, None], ind]
    abs_diffs = np.abs(np.diff(maxs))
    # adjust differences such that only correct classification is pushed
    abs_diffs = abs_diffs * correct_vector
    differences = np.sum(abs_diffs)

    return -(differences)


def norm_probability_differences(detached_output, true_labels, loss):
    """
    Calculates the difference between the two most probable classes and normalises it with tanh
    such that the resulting difference is between 0 and 1. This is multiplied by the current loss
    to bound the additonal loss term to numbers between 0 and the current cross entropy loss.
    This alleviates the CE loss the bigger the difference is
    """
    # if logit probabilities are used we need to get normal probabilities
    if np.sum(detached_output) != 1:
        detached_output = np.array([np.exp(x) / np.sum(np.exp(detached_output))
                                    for x in detached_output])

    predicted_labels = np.argmax(detached_output, 1)
    correct_vector = true_labels == predicted_labels

    ind = np.argpartition(detached_output, -2, axis=1)[:, -2:]
    # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops
    maxs = detached_output[np.arange(detached_output.shape[0])[:, None], ind]
    abs_diffs = np.abs(np.diff(maxs))
    # adjust differences such that only correct classification is pushed
    abs_diffs = abs_diffs * correct_vector
    differences = np.sum(abs_diffs)

    # now we need to scale the difference such that it is between 0 and 1
    # to do so we use tanh scaled such that tanh(x) ~ 1 for the max distance
    # of all sample in the batch
    # since we know that the maximum distance between the first and second highest class
    # ~ 100 and tanh ~ 1 starting from 2 we can calculate a scaling factor
    batch_size = len(detached_output)
    scaling_factor = 2 / (batch_size)
    differences = np.tanh(differences * scaling_factor)

    return -(differences * loss)


def neg_norm_probability_differences(detached_output, true_labels, loss):
    """
    Calculates the difference between the two most probable classes and normalises it with tanh
    such that the resulting difference is between 0 and 1. The inverse of this difference is taken.
    This is multiplied by the current loss to bound the additonal loss term to numbers between 0 and
    the current cross entropy loss. This increases the loss the smaller the differences are.
    """

    # if logit probabilities are used we need to get normal probabilities
    if np.sum(detached_output) != 1:
        detached_output = np.array([np.exp(x) / np.sum(np.exp(detached_output))
                                    for x in detached_output])

    predicted_labels = np.argmax(detached_output, 1)
    correct_vector = true_labels == predicted_labels

    ind = np.argpartition(detached_output, -2, axis=1)[:, -2:]
    # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops
    maxs = detached_output[np.arange(detached_output.shape[0])[:, None], ind]
    abs_diffs = np.abs(np.diff(maxs))
    # adjust differences such that only correct classification is pushed
    abs_diffs = abs_diffs * correct_vector
    differences = np.sum(abs_diffs)

    # now we need to scale the difference such that it is between 0 and 1
    # to do so we use tanh scaled such that tanh(x) ~ 1 for the max distance
    # of all sample in the batch
    # since we know that the maximum distance between the first and second highest class
    # ~ 100 and tanh ~ 1 starting from 2 we can calculate a scaling factor
    # the result of tanh gets subtracted from 1 to get a factor that is higher if the differences are lower
    batch_size = len(detached_output)
    scaling_factor = 2 / (batch_size)
    differences = 1 - np.tanh(differences * scaling_factor)

    return differences * loss


def output_probability_entropy(network_output):
    network_output = network_output.cpu().detach().numpy()
    if np.sum(network_output[1]) < 0:
        network_output = network_output * -1

    entropy = np.sum(np.sum(np.log2(network_output) * network_output, axis=1))

    return entropy