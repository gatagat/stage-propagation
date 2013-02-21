import numpy as np
import scipy as sp
import scipy.linalg

from harmonic_function import harmonic_function

def _solve_binary(labels, confidences, weights, mu=1e-2, eps=1e-1, **kwargs):
    """
    Propagates binary labels in a graph specified by weights.

    Minimizes:
    .. math::
        \|S\hat{Y}-SY\|^2 + \mu\hat{Y}^TL\hat{Y} + \mu\eps\|\hat{Y}\|^2

    by solving:
    .. math::
        (S + \muL + \mu\eps{}I)\hat{Y} * SY = 0  

    where Y and Y hat are known and fitted labels, S is a diagonal matrix with
    S[i,i] = 1 if Y[i] is known and 0 otherwise, L is a graph Laplacian. For
    more information consult [1].

    [1]: Sec. 11.3.2 of "Label Propagation and Quadratic Criterion", 2006 by Y. Bengio et al.

    Parameters:
    ===========
    weights: 2D float ndarray
        Weights of the edges between samples.
    labels: list/array
        Labels for all samples. Labels should be -1, or 1 for the two classes, 0 for unknown.
    confidences: float list/array
        Confidences of the assigned labels, 0 for unknown, 1 for known.
    mu: float
        Smoothness weight.
    eps: float
        Regularization weight.

    Returns:
    ========
    probability
        Probability of the positive class
    """
    assert eps > 0

    n = len(weights)
    L = np.diag(weights.sum(axis=1)) - weights.astype(float)
    S = np.diag(confidences).astype(float)
    M = S + mu*L + mu*eps*np.eye(n, dtype=float)
    c, lower = sp.linalg.cho_factor(M)
    pred = sp.linalg.cho_solve((c, lower), np.dot(S, labels))
    return (pred + 1) / 2


def _solve_binary_harmonic_function(labels, confidences, weights, **kwargs):
    """
    Wrapper around harmonic_function.

    labels: list, 1D ndarray
        Soft labels of all the samples. They act as unaries on nodes and are represented as dongle nodes.
    confidences: list, 1D ndarray
        Weights of edges connecting the dongle nodes with sample nodes.
    weights: 2D ndarray
        Weights of edges connecting the samples. They act as pairwise terms.

    Returns:
    --------
    probability: list, 1D ndarray
        Recovered probability of the positive label for the sample nodes.
    """
    n = len(labels)

    #      |0|E| 
    # W' = |---|
    #      |E|W|
    weights_ext = np.zeros((2*n, 2*n), dtype=float)
    weights_ext[n:, n:] = weights
    E = np.diag(np.exp(confidences))
    weights_ext[n:, :n] = E
    weights_ext[:n, n:] = E
    return harmonic_function(weights_ext, labels, class_mass_normalization=False)


def propagate_labels(predictions, weights, method_name=None, labels=None, output_dir=None, **kwargs):
    assert labels != None
    assert method_name != None
    assert output_dir != None
    propagated = np.zeros(len(predictions), dtype=predictions.dtype)
    propagated['id'] = predictions['id']
    for class_num, class_label in labels.items():
        if method_name == 'general':
            propagate_fn = _solve_binary
            predicted_labels = (predictions['pred'] == class_num).astype(float)*2 - 1
        else:
            propagate_fn =_solve_binary_harmonic_function
            predicted_labels = (predictions['pred'] == class_num).astype(float)
        confidence_name = 'prob%d' % class_num
        propagated[confidence_name] = propagate_fn(predicted_labels, predictions[confidence_name], weights, **kwargs)
    confidence_names = [ 'prob%d' % n for n in labels.keys() ]
    pred = propagated[confidence_names].view(float).reshape(len(propagated), -1).argmax(axis=1)
    propagated['pred'] = np.array(labels.keys())[pred]
    return propagated
