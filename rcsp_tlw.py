"""
Implement Regularized Common Spatial Pattern based on Transfer Learning
with weighted sources (rCSP-tlw) according to:

M. Cheng, Z. Lu, H. Wang, Regularized common spatial patterns with
subject-to-subject transfer of EEG signals. Cogn. Neurodyn. 11, 173–181
(2017).
"""


import numpy as np
import scipy.linalg
import pdb


def calc_cov_dissim(c1, c2):
    """Calculate dissimilarity between 2 covariance matrices

    Calculate the dissimilarity between two covariance matrices using
    the Frobenius norm.
    This is according to formula 9 of the original article

    Parameters
    ----------
    c1, c2 : array_like
        first/second covariance matrix, must be positive (semi-)definite
        square matrices

    Returns
    -------
    sim : float
        dissimilarity of both covariance matrices
    """
    try:
        if not np.allclose(c1, np.asmatrix(c1).H):
            raise ValueError
    except ValueError:
        raise ValueError('c1 must be a hermitian square matrix')
    try:
        if not np.allclose(c2, np.asmatrix(c2).H):
            raise ValueError
    except ValueError:
        raise ValueError('c2 must be a hermitian square matrix')
    return np.sqrt(np.trace((c1 - c2) @ (c1 - c2).T))


def get_rcsp_tl_weights(c, source_covs):
    """Calculate the weights of rcsp_tlw

    Calculate the weights that determine the influence of a subject on
    the rcsptl_w penalty term (formula 10 of the publication)

    Parameters
    ----------
    c : array_like
        covariance matrix of the target subject, must be positive
        (semi-)definite square matrix
    source_covs : iterable
        an iterable of N elements containing the covariance matrices -
        one for each of N source subjects - that should be used to
        calculate the weights between the source and target subjects.
        The covariance matrices need to be positive (semi-)definite
        square matrices

    Returns
    -------
    b_st: ndarray
        an array of length N containing the weights between the target
        and each of theN  source subjects
    """
    # calculate the dissimilarity between the target subject and all source
    # subjects' covariance matrices
    dissim_st = np.asarray([calc_cov_dissim(c, source_cov_now)
                            for source_cov_now in source_covs])
    # get the normalization constant
    N_t = (1/dissim_st).sum(0)
    b_st = 1 / (N_t * dissim_st)
    return b_st


def _take_abs_eigvals(c):
    """Eigendecomposition reconstruction with all-positive Eigenvalues

    Transform a square symmetric matrix to a positive definite one by
    taking the absolute of the eigenvalues

    Parameters
    ----------
    c : array_like
        a real, square, symmetric matrix

    Returns
    -------
    ndarray
        the reconstruction of c with all-positive eigenvalues
    """
    try:
        if not np.allclose(c.imag, 0):
            raise ValueError
        if not np.allclose(c, np.asmatrix(c).T):
            raise ValueError
    except ValueError:
        raise ValueError('c must be a real symmetric matrix')
    # get eigenvalues and eigenvectors
    eigval, eigvect = scipy.linalg.eigh(c)
    # return reconstrution after taking absolute value
    return (eigvect * np.abs(eigval)[np.newaxis]) @ eigvect.T


def covariance_csp(c1, c2):
    """Common Spatial Pattern analysis based on covariance matrices

    Finds spatial filters w that maximize the ratio:
        (w.T @ c1 @ w) / (w.T @ c2 @ w)

    Parameters
    ----------
    c1, c2 : array_like
        covariance matrices of both conditions. Must be real square
        symmetric matrices of equal shape n x n

    Returns
    -------
    variance_ratios : ndarray
        ratio of variances between w.T @ c1 @ w and w.T @ c2 @ w
        with length m, where m is the rank of c2, in ascending order.
    csp_filters : ndarray
        a matrix of spatial filters of shape n x m where m is the matrix
        rank of c2
        The single filters, corresponding to the entries of
        `variance ratios` will be in the columns of `filters`, i.e., 
        `filters[:, 0]` will be the first filter and `filters[:, 1]` the
        second.
    """
    # calculate whitening matrix
    c2_rank = np.linalg.matrix_rank(c2)
    c2_eigenvalues, c2_eigenvectors = np.linalg.eigh(c2)
    whitening_filters = (c2_eigenvectors[:, -c2_rank:] /
                         np.sqrt(c2_eigenvalues[-c2_rank:])[np.newaxis])
    # project c1 through whitening filter and recalculate eigenvalue
    # decomposition
    variance_ratios, post_whitening_filters = np.linalg.eigh(
            whitening_filters.T @ c1 @ whitening_filters)
    # concatenate filters
    csp_filters = whitening_filters @ post_whitening_filters
    return variance_ratios, csp_filters


def rcsp_tlw(c1, c2, target_cov, source_covs, alpha=0,
             c1_vs_c2_only=False):
    """Calculate rCSP-tlw

    Calculate Regularized Common Spatial Patterns based on Transfer
    Learning with weighted sources (rCSP-tlw)

    rCSP-tlw optimizes spatial filters to to find components with a
    maximum ratio of variances from two covariance matrices c1 and c2
    while enforcing some degree of similarity between the extracted
    components and those from other subjects.

    Parameters
    ----------
    c1, c2 : array_like
        covariance matrices of both conditions (of target subject). Must
        be real square symmetric matrices of equal shape n x n.
    target_cov : array_like
        A covariance matrix of the target_subject, irrespective of
        conditions. Must be a real square summetric matrix of shape n x n
    source_covs : iterable
        An iterable of n covariance matrices from n source subjects.
        Each of those matrices need to be real, square, and symmetric and
        of equal shape n x n
    alpha : float, default=0
        a regularization parameter to control the effect of source
        subjects on the CSP calculation. If 0, standard CSP between
        c1 and c2 will be calculated. Larger values will lead to
        increased effect of the source subjects.
    c1_vs_c2_only : bool, default=False
        whether only spatial filters for the maximization of c1 over c2
        should be found or also, vice versa, for c2 over c1.

    Returns
    -------
    c1_vs_c2_ratios : ndarray
        the variance ratios for the spatial filters that are optimized
        to maximize the ratio of c1 over c2. Array of length m, where
        m is the rank of c2 plus the added penalty sorted in ascending
        order
    c1_vs_c2_filters : ndarray
        the corresponding spatial filters, array of shape n x m where
        m is the rank of (c2 + penalty). The spatial filters reside in
        the columns of this array such that `c1_vs_c2_filters[:,0]` is
        the first spatiel filter etc.
    c2_vs_c1_ratios : ndarray, optional if c1_vs_c2_only=False
        the variance ratios for the spatial filters that are optimized
        to maximize the ratio of c2 over c1. Array of length k, where
        k is the rank of c1 plus the added penalty sorted in ascending
        order
    c2_vs_c1_filters : ndarray, optional if c1_vs_c2_only=False
        the corresponding spatial filters, array of shape n x k where
        k is the rank of (c1 + penalty). The spatial filters reside in
        the columns of this array such that `c2_vs_c1_filters[:,0]` is
        the first spatiel filter etc.
    """
    # get the weights between the target subject and all source subjects
    b_st = get_rcsp_tl_weights(target_cov, source_covs)
    # calculate the penalty term
    penalty = alpha * np.sum([
        b_st_now * _take_abs_eigvals(target_cov - source_cov_now)
        for b_st_now, source_cov_now in zip(b_st, source_covs)], 0)
    # for the eigenvalue decomposition, the penalty will be added to the 
    # input covariance matrices
    penalized_c1 = c1 + penalty
    penalized_c2 = c2 + penalty
    # find csp for c1 vs c2
    c1_vs_c2_ratios, c1_vs_c2_filters = covariance_csp(c1, penalized_c2)
    if c1_vs_c2_only is True:
        return c1_vs_c2_ratios, c1_vs_c2_filters
    elif c1_vs_c2_only is False:
        c2_vs_c1_ratios, c2_vs_c1_filters = covariance_csp(
                c2, penalized_c1)
        return (c1_vs_c2_ratios, c1_vs_c2_filters,
                c2_vs_c1_ratios, c2_vs_c1_filters)
