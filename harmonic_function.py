import numpy as np
import scipy as sp
import scipy.linalg

def harmonic_function(W, fl, class_mass_normalization=False):
    '''
Semi-supervised learning with basic harmonic function. Implements
eq (5) in "Semi-Supervised Learning Using Gaussian Fields 
and Harmonic Functions".  Xiaojin Zhu, Zoubin Ghahramani, John Lafferty.  
The Twentieth International Conference on Machine Learning (ICML-2003).

Parameters:
-----------
W : n*n weight matrix
    The first L entries(row,col) are for labeled data, the rest for unlabeled
    data.  W has to be symmetric, and all entries has to be non-negative.  Also
    note the graph may be disconnected, but each connected subgraph has to have
    at least one labeled point.  This is to make sure the sub-Laplacian matrix
    is invertible.
fl : L*c label matrix
    Each line is for a labeled point, in one-against-all encoding (all zero but
    one 1).  For example in binary classification each line would be either "0
    1" or "1 0".
class_mass_normalization : bool
    Apply Class Mass Normalization (CMN) to the solution, as in eq (9) of the
    ICML paper.  The class proportions are the maximum likelihood (frequency)
    estimate from labeled data fl.  The CMN heuristic is known to sometimes
    improve classification accuracy.

Returns:
--------
fu : (n-L)*c label matrix for the unlabeled points
    The harmonic solution, i.e. eq (5) in the ICML paper.  Each row is for an
    unlabeled point, and each column for a class.  The class with the largest
    value is the predicted class of the unlabeled point.

Note:
  If the Laplacian restricted to unlabeled entries is close to singular, 
  and fu is all NaN, there are probably connected components in the graph
  without any label.

Original Matlab code by Xiaojin Zhu, zhuxj@cs.cmu.edu, 2004.
Rewritten to Python by Tomas Kazmar, tomash.kazmar@seznam.cz, 2012.
    '''

    l = fl.shape[0] # the number of labeled points
    n = W.shape[0] # total number of points

    # the graph Laplacian L=D-W
    L = np.diag(W.sum(axis=0)) - W

    # the harmonic function.
    Lu_factor = sp.linalg.cho_factor(L[l:, l:])
    fu = -sp.linalg.cho_solve(Lu_factor, L[l:, :l].dot(fl))

    if class_mass_normalization:
        q = fl.sum(axis=0)+1 # the unnormalized class proportion estimate from labeled data, with Laplace smoothing
        fu *= np.tile(q / fu.sum(axis=0), (n-l, 1))

    return fu
