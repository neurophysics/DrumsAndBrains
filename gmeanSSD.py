"""This module calculates the spatio-spectral decomposition of a number of
target covariances (`targets`) agains a single contrasting covariance matrix
`contrast`. The objective function is the generalized mean of the ratios
target/contrast.
The generalized mean approaches the maximum for large p and the minimum for
small p.
The result is a spatial filter w
"""
import numpy as np
import scipy
import scipy.optimize

def get_power(w,c):
    power = w.T.dot(c).dot(w)
    power_d = c.dot(w) + c.T.dot(w)
    return power, power_d

def genmean(x, p):
    """Calculate the generalized mean across all x"""
    p = float(p)
    '''Calculate the generalized mean of x with p as exponent'''
    return np.mean(x**p)**(1/p)

def genmean_d(x, p):
    """the derivative of the gernalized mean"""
    p=float(p)
    return np.mean(x**p)**(1/p - 1)*x**(p-1)/len(x)

def obj(w, targets, contrast, p):
    # this function should be minimized
    target_power, target_power_d = zip(*[get_power(w, t) for t in targets])
    contrast_power, contrast_power_d = get_power(w, contrast)
    power_ratio = np.asarray([t/contrast_power for t in target_power])
    power_ratio_d = np.asarray([(t_d*contrast_power - t*contrast_power_d)/
        contrast_power**2 for t,t_d in zip(target_power, target_power_d)])
    obj = genmean(power_ratio, p)
    obj_d = genmean_d(power_ratio, p).dot(power_ratio_d)
    return -obj + (np.sum(w**2) - 1)**2, -obj_d + 2*(np.sum(w**2) - 1)*2*w

def gmeanSSD(targets, contrast, p=1, num=None, disp=False):
    """Calculate the spatio-spectral decomposition by maximizing the average
    ratio of veriances between a number of target covariance matrix and a
    single contrast covariance matrix
    The target function is the generalized mean of the variance ratios

    Args:
        targets (iterable): an iterable with n covariance matrices the power of
            which should be maximized relative to the contrast
        contrast (numpy array): a single covariance matrix. The shape should
            the same as every covariance matrix in targets
        p (float): calculate the p-mean across the ratios of variances
            The generalized mean approaches the maximum for large p and the
            minimum for small p. Defaults to 1 (arithmetic mean)
        num (int > 0): the number of spatial filters that should be obtained.
            If None (default), the maximum number of filters that can be
            extracted will be obtained (relates to the rank).
        disp (boot): whether the optimization should return status messages

    Returns:
        mean_ratio: the p-mean ratio of the variances
        filt: the spatial filters, such that filt[:,0] belongs to the 1st
            mean ratio and filt[:,-1] to the last mean ratio. The filters are
            scaled to unit variance of the contrast.
    """
    rank = np.linalg.matrix_rank(contrast)
    if num is None: num=rank
    num = min(num, rank)
    # get whitening matrix from contrast
    w, v = scipy.linalg.eigh(contrast)
    W = v[:,-rank:]/np.sqrt(w[np.newaxis,-rank:])
    # whiten
    targets = [W.T.dot(t).dot(W) for t in targets]
    for i in range(num):
        if i>0:
            # project the previous filters out
            wx = np.linalg.svd(np.array(w), full_matrices=True
                    )[2][i:].T
        else:
            wx = np.eye(rank)
        ttemp = [wx.T.dot(t).dot(wx) for t in targets]
        ctemp = np.eye(rank - i)
        while True:
            x0 = np.random.randn(rank - i)
            x0 /= np.sqrt(np.sum(x0**2))
            # minimize
            res = scipy.optimize.minimize(obj, x0=x0, args=(ttemp,ctemp, p),
                    method='BFGS', jac=True, options=dict(disp=disp))
            if res['success']: break
        try:
            opt.append(-res['fun'])
            w.append(wx.dot(res['x']))
        except NameError:
            opt = [-res['fun']]
            w = [wx.dot(res['x'])]
        if num == 1:
            opt = opt[0]
    w = W.dot(np.array(w).T)
    #w /= np.sqrt(np.sum(w**2,0))
    return opt,w
