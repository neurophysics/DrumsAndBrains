import numpy as np
from scipy.optimize import minimize as _minimize
from scipy.optimize import shgo as _shgo

############################################
# internal funtions and constructors       #
# the functional classes are at the bottom #
############################################
class _SPoC(object):
    def __init__(self, bestof=15):
        self.bestof = bestof
        self.method = 'L-BFGS-B'
        self.disp = False

    def __call__(self, *args, **kwargs):
        try:
            num = kwargs['num']
        except KeyError:
            num = None
        # the first arguments are covariance matrices
        covs = args[:-1]
        # get the number of channels
        try:
            if covs[0].shape[0] == covs[0].shape[1]:
                ch = covs[0].shape[0]
            elif covs[0][0].shape[0] == covs[0][0].shape[1]:
                ch = covs[0][0].shape[0]
            else:
                raise ValueError
        except AttributeError:
            if covs[0][0].shape[0] == covs[0][0].shape[1]:
                ch = covs[0][0].shape[0]
            else:
                raise ValueError
        # the last argument is a behavioural variable
        z = args[-1]
        if num != 1:
            W = self._white(covs[0])
            rank = W.shape[-1]
            if num is None: num = rank
            else: num = min([num, rank])
            covs = [self._filter(W,C) for C in covs]
        for i in xrange(num):
            if i>0:
                # project the previous filters out
                wx = np.linalg.svd(np.array(w), full_matrices=True
                        )[2][i:].T
                temp = [self._filter(wx,C) for C in covs]
            else:
                wx = np.eye(ch)
                temp = list(covs)
            x0 = np.random.randn(self.bestof, wx.shape[1])
            # normalize x0
            x0 /= np.sqrt(np.sum(x0**2, -1))[:,np.newaxis]
            # optimize one step
            res = [
                    _minimize(
                    fun = self.fun,
                    x0 = x0_now,
                    args = temp + [z],
                    method=self.method,
                    jac = True, options=dict(
                        disp=self.disp))
                for x0_now in x0]
            w_i = [res_now.x for res_now in res]
            opt_i = [res_now.fun for res_now in res]
            try:
                opt.append(-np.nanmin(opt_i))
                w.append(wx.dot(w_i[np.nanargmin(opt_i)]))
            except NameError:
                opt = [-np.nanmin(opt_i)]
                w = [wx.dot(w_i[np.nanargmin(opt_i)])]
        if num == 1:
            opt = opt[0]
            w = w[0]
        else:
            opt = np.r_[opt]
            w = W.dot(np.array(w).T)[:,np.argsort(opt)[::-1]]
            opt = np.sort(opt)[::-1]
        return opt, w

def _white_covs(X):
    mean_cov =  X.mean(-1)
    rank = np.linalg.matrix_rank(mean_cov)
    bval, bvec = np.linalg.eigh(mean_cov)
    W = bvec[:,-rank:]/np.sqrt(bval[-rank:])
    return W

def _white_covs_avg(X):
    mean_cov =  np.mean([X1.mean(-1) for X1 in X], 0)
    rank = np.linalg.matrix_rank(mean_cov)
    bval, bvec = np.linalg.eigh(mean_cov)
    W = bvec[:,-rank:]/np.sqrt(bval[-rank:])
    return W

def _filter_covs(W, X):
    return W.T.dot(W.T.dot(X))

def _filter_covs_avg(W, X):
    return [_filter_covs(W, X1) for X1 in X]

def _corr_2(w, X, z):
    """
    function and gradient to maximize the squared correlation
    coefficient between w.T*X*w and z.

    In order to enable maximization, the negative squared correlation is
    returned
    """
    corr, corr_grad = _corr_grad(w, X, z)
    corr_2 = corr**2
    corr_2_grad = 2*corr*corr_grad
    # invert the sign to do maximization
    return -corr_2, -corr_2_grad

def _corr_avg_2(w, X, z):
    """
    function and gradient to maximize the squared average correlation
    coefficient.
    X and z are assumed to be lists of individual covariance
    matrices and performances.
    For every subject, the correlation coefficient of w.T*X*w and z is
    computed.

    The correlation is averaged across subjects and squared.
    In order to enable maximization, the negative squared correlation is
    returned.
    """
    corr, corr_grad = zip(*[_corr_grad(w, X1, z1)
        for X1, z1 in zip(X, z)])
    corr = np.mean(corr, 0)
    corr_grad = np.mean(corr_grad, 0)
    corr_2 = corr**2
    corr_2_grad = 2*corr*corr_grad
    # invert the sign to do maximization
    return -corr_2, -corr_2_grad

def _partial_corr_2(w, X, Y, z):
    """
    function and gradient to maximize the squared partial correlation
    coefficient between w.T*X*w and z after regressing w.T*Y*w out.

    In order to enable maximization, the negative squared partial
    correlation is returned
    """
    pcorr, pcorr_grad = _partial_corr_grad(w, X, Y, z)
    pcorr_2 = pcorr**2
    pcorr_2_grad = 2*pcorr*pcorr_grad
    # invert the sign to do maximization
    return -pcorr_2, -pcorr_2_grad

def _partial_corr_avg_2(w, X, Y, z, return_grad=False):
    """
    function and gradient to maximize the squared average partial
    correlation coefficient.
    X, Y, and z are assumed to be lists of individual covariance
    matrices and performances.
    For every subject, the partial correlation coefficient of
    w.T*X*w and z after regressing w.T*Y*w out is computed.

    The partial correlation is averaged across subjects and squared.
    In order to enable maximization, the negative squared partial
    correlation is returned
    """
    pcorr, pcorr_grad = zip(*[_partial_corr_grad(w, X1, Y1, z1)
        for X1, Y1, z1 in zip(X, Y, z)])
    pcorr = np.mean(pcorr, 0)
    pcorr_grad = np.mean(pcorr_grad, 0)
    pcorr_2 = pcorr**2
    pcorr_2_grad = 2*pcorr*pcorr_grad
    # force the norm of the filter to be 1
    return -pcorr_2, -pcorr_2_grad

def _corr_grad(w, X, z):
    # normalize
    z = (z - z.mean())/z.std()
    # get filtered and standardized powers
    X_p, X_p_grad = _norm_log_power(w,X)
    # get sample correlation coefficients
    pXz = (X_p*z).mean()
    pXz_grad = np.mean(X_p_grad*z, -1)
    return pXz, pXz_grad

def _partial_corr_grad(w,X,Y,z):
    # normalize
    z = (z - z.mean())/z.std()
    # get filtered and standardized powers
    X_p, X_p_grad = _norm_log_power(w,X)
    Y_p, Y_p_grad = _norm_log_power(w,Y)
    # get sample correlation coefficients
    pXz = (X_p*z).mean()
    pXz_grad = np.mean(X_p_grad*z, -1)
    pYz = (Y_p*z).mean()
    pYz_grad = np.mean(Y_p_grad*z, -1)
    pXY = (X_p*Y_p).mean()
    pXY_grad = np.mean(X_p_grad*Y_p + X_p*Y_p_grad, -1)
    # calculate the numerator of the partial correlation
    pXzY_num = pXz - pXY*pYz
    pXzY_num_grad = pXz_grad - (pXY_grad*pYz + pXY*pYz_grad)
    # get the (squared) denominator
    pXY_2 = pXY**2
    pXY_2_grad = 2*pXY*pXY_grad
    pYz_2 = pYz**2
    pYz_2_grad = 2*pYz*pYz_grad
    pXzY_denom_2 = (1 - pXY_2)*(1 - pYz_2)
    pXzY_denom_2_grad = -pXY_2_grad*(1 - pYz_2) + (1 - pXY_2)*(-pYz_2_grad)
    # get the denominator (take square root)
    pXzY_denom = np.sqrt(pXzY_denom_2)
    pXzY_denom_grad = 0.5*pXzY_denom_2_grad/np.sqrt(pXzY_denom_2)
    # get the partial correlation coefficient
    pXzY = pXzY_num/pXzY_denom
    pXzY_grad = (pXzY_num_grad*pXzY_denom -
            pXzY_num*pXzY_denom_grad)/pXzY_denom**2
    return pXzY, pXzY_grad

def _norm_log_power(w,X):
    # calculate the numerator
    power = w.dot(w.dot(X))
    # take the log of power + 1 to avoid nans due to roundoff errors
    log_power = np.log(power + 1)
    ###
    power_grad = 2*np.dot(w, X)
    log_power_grad = power_grad/(power + 1)
    mean_log_power = log_power.mean()
    mean_log_power_grad = np.mean(log_power_grad, -1) 
    log_power_detrend = log_power - mean_log_power
    log_power_detrend_grad = (log_power_grad -
            mean_log_power_grad[:,np.newaxis])
    # calculate the denominator
    log_power_var = np.mean(log_power_detrend**2)
    log_power_var_grad = np.mean(
            2*log_power_detrend*log_power_detrend_grad, -1)
    log_power_std = np.sqrt(log_power_var)
    log_power_std_grad = 0.5/log_power_std*log_power_var_grad
    # calculate the normed power
    log_power_norm = log_power_detrend/log_power_std
    log_power_norm_grad = ((log_power_detrend_grad*log_power_std -
            log_power_detrend*log_power_std_grad[:,np.newaxis])/
            log_power_var)
    return log_power_norm, log_power_norm_grad

##############################################################
# define two classes for the optimization of single subjects #
# or the average across multiple subjects                    #
##############################################################
class _SPoC_single(_SPoC):
    def _white(self, *args):
        return _white_covs(*args)

    def _filter(self, *args):
        return _filter_covs(*args)

class _SPoC_avg(_SPoC):
    def _white(self, *args):
        return _white_covs_avg(*args)

    def _filter(self, *args):
        return _filter_covs_avg(*args)

#################################
# define the functional classes #
#################################

class SPoCr2(_SPoC_single):
    def fun(self, w, args):
        return _corr_2(w, *args)

class SPoCr2_avg(_SPoC_avg):
    def fun(self, w, args):
        return _corr_avg_2(w, *args)

class pSPoCr2(_SPoC_single):
    def fun(self, w, args):
        return _partial_corr_2(w, *args)

class pSPoCr2_avg(_SPoC_avg):
    def fun(self, w, args):
        return _partial_corr_avg_2(w, *args)
