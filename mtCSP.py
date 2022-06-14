"""
Implements multisebject CSP according to:

D. Devlaminck, B. Wyns, M. Grosse-Wentrup, G. Otte, P. Santens,
Multisubject learning for common spatial patterns in motor-imagery BCI.
Comput. Intell. Neurosci. 2011, 217987 (2011).
"""
import numpy as np
from scipy.optimize import minimize as _minimize
import pdb


def get_e_s(n_subjects, s, d):
    """
    Parameters
    ----------
    n_subjects : int
        number of subjects
    s : int
        index of current subject
    d : dimensionality of the filter

    Returns
    -------
    e_s : ndarray
        the E_s matrix (formula 5)
    """
    e_s = np.vstack([
        np.eye(d),
        np.zeros([s*d, d]),
        np.eye(d),
        np.zeros([(n_subjects - (s + 1)) * d, d])
        ])
    return e_s

def get_d_s(n_subjects, s, d):
    """
    Parameters
    ----------
    n_subjects : int
        number of subjects
    s : int
        index of current subject
    d : dimensionality of the filter

    Returns
    -------
    d_s : ndarray
        the D_s matrix (from formula 5)
    """
    d_s1 = np.vstack([
        np.zeros([(s + 1) * d, d]),
        np.eye(d),
        np.zeros([(n_subjects - (s + 1)) * d, d])
        ])
    d_s2 = np.hstack([
        np.zeros([d, (s + 1) * d]),
        np.eye(d),
        np.zeros([d, (n_subjects - (s + 1)) * d])
        ])
    return d_s1 @ d_s2

def get_d_0(n_subjects, d):
    """
    Parameters
    ----------
    n_subjects : int
        number of subjects
    d : dimensionality of the filter

    Returns
    -------
    d_0 : ndarray
        the D_0 matrix (from formula 5)
    """
    d_01 = np.vstack([
        np.eye(d),
        np.zeros([n_subjects * d, d])
        ])
    d_02 = np.hstack([
        np.eye(d),
        np.zeros([d, n_subjects * d])
        ])
    return d_01 @ d_02


def get_target_covs(covs):
    """
    Parameters
    ----------
    covs : iterable
        an iterable of N elements containing the covariance matrices from
        the target condition - one for each of N source subjects.
        The covariance matrices need to be positive (semi-)definite
        square matrices

    Returns
    -------
    target_covs : iterable
        the covariance matrices, estended in such a way that the optimization
        for multisubject learning can be done
    """
    n_subjects = len(covs)
    target_covs = []
    for i, c in enumerate(covs):
        e_s = get_e_s(n_subjects, i, c.shape[0])
        target_covs.append(e_s @ c @ e_s.T)
    return target_covs

def get_contrast_covs(covs, lam1, lam2):
    """
    Parameters
    ----------
    covs : iterable
        an iterable of N elements containing the covariance matrices from
        the target condition - one for each of N source subjects.
        The covariance matrices need to be positive (semi-)definite
        square matrices
    lam1 : float
        the first penalty - this penalizes the size of the global filter
    lam2 : float
        the first penalty - this penalizes the size of the specific filter

    Returns
    -------
    contrast_covs : iterable
        the covariance matrices, extended in such a way that the optimization
        for multisubject learning can be done PLUS the penalties
    """
    n_subjects = len(covs)
    d_0 = get_d_0(n_subjects, covs[0].shape[0])
    contrast_covs = []
    for i, c in enumerate(covs):
        e_s = get_e_s(n_subjects, i, c.shape[0])
        d_s = get_d_s(n_subjects, i, c.shape[0])
        contrast_covs.append(e_s @ c @ e_s.T + lam1 * d_0 + lam2 * d_s)
    return contrast_covs

def _fun_jac(w, target_covs, contrast_covs, factor=1):
    """
    Calculate the objective function and its derivative

    Parameters:
    -----------
    w : ndarray
        the current filter
    target_covs : iterable
        the covariance matrices from the target condition, extended in the
        style of the algorithm (c.f. function get_target_covs)
    contrast_covs : iterable
        the covariance matrices from the contrast condition, extended in the
        style of the algorithm (c.f. function get_contrast_covs)
    factor : float (default 1)
        the result is multiplied with this factor. Defaults to 1.

    Returns:
    --------
    obj: float
        the result of the objective function (multiplied by `factor`)
    obj_d : ndarray
        the derivative of the objective function with respect to w (multiplied
        by `factor`)
    """
    rs_num = [w @ c1 @ w for c1 in target_covs]  # the numerator of rs (.T not needed because 1D array)
    rs_denom = [w @ c2 @ w for c2 in contrast_covs]  # the denominator of rs
    rs = [num / denom for num, denom in zip(rs_num, rs_denom)]
    obj = sum(rs)  # the objective function
    obj_d = 2*sum([(c1 @ w - rs_now * c2 @ w) / (w @ c2 @ w)
                   for c1, c2, rs_now in zip(target_covs, contrast_covs, rs)])
    return factor * obj, factor * np.asarray(obj_d)

def constraint(w, n_subjects, old_W=None):
    """Enforce orthogonality of spatial filters for every subject
    """
    n_channels = len(w) // (n_subjects + 1)
    if old_W is None:
        W = w.reshape(len(w),-1)
    else:
        W = np.hstack([old_W.reshape(len(w), -1), w[:, np.newaxis]])
    return sum(
            [single_constraint(W[:n_channels] +
                               W[(i + 1) * n_channels:(i + 2) * n_channels])
             for i in range(n_subjects)])

def constraint_d(w, n_subjects, old_W=None):
    n_channels = len(w) // (n_subjects + 1)
    if old_W is None:
        W = w.reshape(-1,1)
    else:
        W = np.hstack([old_W.reshape(len(w), -1), w[:, np.newaxis]])
    deriv = [single_constraint_d(
                                 W[:n_channels] +
                                 W[(i + 1) * n_channels:(i + 2) * n_channels])
                                 for i in range(n_subjects)]
    return np.hstack([sum(deriv), np.hstack(deriv)])

def single_constraint(W):
    deviation = W.T @ W - np.eye(W.shape[-1])
    constraint = np.sum((deviation)**2)
    return constraint

def single_constraint_d(W):
    deviation = W.T @ W - np.eye(W.shape[-1])
    constraint_d = (4 * deviation[-1] * W).sum(-1)
    return constraint_d

def maximize_mtCSP(c1, c2, lam1, lam2, iterations=100, old_W=None):
    """ Calculate a single multisubject-CSP filter

    Parameters
    ----------
    c1, c2 : iterables
        iterables of N elements containing the covariance matrices from
        the target (c1) and contrast (c2) condition - one for each of N source
        subjects. The covariance matrices need to be positive (semi-)definite
        square matrices
    lam1 : float
        the first penalty - this penalizes the size of the global filter
    lam2 : float
        the first penalty - this penalizes the size of the specific filter
    iterations : int (defaults to 20)
        the number of iterations from random starting points

    Returns:
    """
    n_subjects = len(c1)
    n_channels = c1[0].shape[0]
    target_covs = get_target_covs(c1)
    contrast_covs = get_contrast_covs(c2, lam1, lam2)
    minimizer_results = []
    for _ in range(iterations):
        x0 = np.random.randn(n_channels * (n_subjects + 1))
        # normalize to a norm of one
        x0 /= np.sqrt(x0 @ x0.T)
        minimizer_results.append(
            _minimize(_fun_jac,
                      x0=x0,
                      args=(target_covs, contrast_covs, -1),
                      method="SLSQP",
                      jac=True,
                      constraints=dict(
                                      type='eq',
                                      fun=constraint,
                                      jac=constraint_d,
                                      args=(n_subjects, old_W)),
                      options=dict(disp=True, maxiter=10000)))
    best_idx = np.argmin([res.fun for res in minimizer_results])
    # for the best result, calculate te CSP quotient
    all_filters = minimizer_results[best_idx].x
    w = [all_filters[:n_channels] + all_filters[(i + 1) * n_channels : (i + 2) * n_channels]
         for i in range(n_subjects)]
    CSP_quot = np.mean([(w_now @ c1_now @ w_now) / (w_now @ c2_now @ w_now)
                        for (w_now, c1_now, c2_now) in zip(w, c1, c2)])
    return CSP_quot, all_filters


if __name__ == "__main__":
    n_subjects = 10
    d = 8
    # generate random covariance matrices
    c1 = [np.random.randn(d, d) for _ in range(n_subjects)]
    c1 = [c @ c.T for c in c1]
    c2 = [np.random.randn(d, d) for _ in range(n_subjects)]
    c2 = [c @ c.T for c in c2]
    ###
    lam1 = 0.1
    lam2 = 20
    ###
    quot1, w1 = maximize_mtCSP(c1, c2, lam1, lam2)
    quot2, w2 = maximize_mtCSP(c1, c2, lam1, lam2, old_W=w1)
    quot3, w3 = maximize_mtCSP(c1, c2, lam1, lam2, old_W=np.vstack([w1, w2]).T)
    W = np.vstack([w1, w2, w3]).T
