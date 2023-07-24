import numpy as np
import scipy.linalg
import pdb

def _prepareRiemannianMap(X, P):
    # calculate exponential map of tangent vector Si at initial point p
    Peigval, Peigvect = scipy.linalg.eigh(P)
    Peigval = np.where(np.isclose(Peigval, 0), 0, Peigval)
    Pdet = scipy.linalg.det(Peigvect)
    # assure that the determinant is 1
    if not np.isclose(np.abs(Pdet), 1):
        raise AssertionError(
                'Determinant of eigenvector matrix must be -1 or 1')
    # swap a column if determinant is -1
    if Pdet < 0:
        Peigval[[0,1]] = Peigval[[1,0]]
        Peigvect[:,[0,1]] = Peigvect[:,[1,0]]
    g = Peigvect * np.sqrt(Peigval)[np.newaxis]
    try:
        ginv = scipy.linalg.inv(g)
    except scipy.linalg.LinAlgError:
        ginv = scipy.linalg.pinv(g)
    Y = ginv @ X @ ginv.T
    Yeigval, Yeigvect = scipy.linalg.eigh(Y)
    Ydet = scipy.linalg.det(Yeigvect)
    # assure that the determinant is 1
    if not np.isclose(np.abs(Ydet), 1):
        raise AssertionError(
                'Determinant of eigenvector matrix must be -1 or 1')
    # swap a column if determinant is -1
    if Ydet < 0:
        Yeigval[[0,1]] = Yeigval[[1,0]]
        Yeigvect[:,[0,1]] = Yeigvect[:,[1,0]]
    return g, Yeigvect, Yeigval

def RiemannianExpMap(Si, P):
    g, Yeigvect, Yeigval = _prepareRiemannianMap(Si, P)
    return ((g @ Yeigvect) * np.exp(Yeigval)[np.newaxis]) @ (g @ Yeigvect).T

def RiemannianLogMap(Pi, P):
    g, Yeigvect, Yeigval = _prepareRiemannianMap(Pi, P)
    #print((Yeigval<0).mean(), Yeigval.min(), Yeigval.max())
    np.clip(Yeigval, 1E-10, None, Yeigval)
    return ((g @ Yeigvect) * np.log(Yeigval)[np.newaxis]) @ (g @ Yeigvect).T

def RiemannCovMean(covs, tol=1E-7):
    """Calculate the geometric mean of covariance matrices in the
    riemannian sense

    Args:
        covs (iterable): list/iterable of (positive semi-definite
            covariance matrices)
    Returns:
        cov_mean: geometric mean of the supplied covariance matrices
    """
    # initialize by the arithmetic mean
    P = np.mean(covs, 0)
    while True:
        print('\n', np.abs(P).sum(), "\n")
        logmaps = [RiemannianLogMap(cov_now, P) for cov_now in covs]
        S = np.mean([RiemannianLogMap(cov_now, P) for cov_now in covs], 0)
        error = (np.abs(S)**2).sum()
        if error <= tol: break
        P = RiemannianExpMap(S, P)
    return P

def getSessionClocks(fname):
    """Read the timestamps of the clock as EEG sample numbers
    The result is split into a list of clock-timestamps for every individual
    session

    Args:
        fname (str): Filename of the file containing the markers without
            the filename extension. '.vmrk' is added to fname

    Returns:
        session_clocks (list): list with arrays of indices to the EEG
            signaling the occurence of a clock trigger input such that
            session_clocks[0] contains the timestamps of the clock for the
                1st session
            session_clocks[1] contains the timestamps of the clock for the
                2nd session
                ...
    """
    marker = np.loadtxt(fname, skiprows=12, usecols=2,
            delimiter=',', dtype=int)
    # split into sessions
    diff = np.diff(marker)
    session_start = np.r_[marker[0], marker[1:][diff > 1005]]
    session_clocks = [marker[np.all([marker>=s, marker<e], axis=0)]
            for s, e in
            zip(session_start, np.r_[session_start[1:], np.inf])]
    return session_clocks

def SyncMusicToEEG(eeg_session_clocks, nearest_session_clocks,
        session_devs, s_rate=1000):
    """Synchronize events of audio files to EEG

    Args:
        eeg_session_clocks (list of ndarrays): a list of array (one array
            for every session) containing the sample indices of clock ticks
        nearest_session_clocks (list of ndarrays): a list of array (one
            array per session) containing the index of the nearest clock
            of that session (in every session the clocks are counted
            starting from 0)
        session_devs (list of ndarrays): the relative latencies (in s) of
            the events in the audio files to the clocks given in 
            nearest_session_clocks

    Returns:
        event_marker (ndarray of ints): the sample indices relative to the
            EEG of all the events in session_dev
    """
    cue_pos = np.hstack([e[m] + np.round(d*s_rate).astype(int) for e,m,d in
        zip(eeg_session_clocks, nearest_session_clocks, session_devs)])
    return cue_pos

def mtcsd(x, windows, weights, nfft):
    """Multitaper Cross-Spectral density estimation

    While the most frequent method (Welch's average periodogram) to
    estimate the spectra of a signal is to calculate the Fourier transform
    of data segments (after windowing/tapering) and to average across the
    power spectra of the segments afterwards, the multitaper spectral
    estimation calculates the spectrum from a single segment after
    application of different window functions (tapers) and averages across
    the result. Typically, 'slepian sequences' used.

    Args:
        x - data (the fourier transform is calculated along the last axis)
        win - list of window functions
        weight - the weight every window gets when averaging across windows
        nfft - the number of Fourier terms to output. If nfft>x.shape[-1],
            only the last nfft elements of x are used. If nfft>x.shape[-1],
            x will be zero-padded prior to Fourier transformation

    Output:
        csd - cross spectral density matrix

    Note:
        The frequencies relating to the Fourier coefficients can be
        obtained from np.fft.rfftfreq
    """
    #make zero mean
    if x.shape[-1] > nfft:
        x = x[...,-nfft:]
    csd = np.zeros([x.shape[0], x.shape[0], nfft//2 + 1], np.complex)
    n = x.shape[-1]
    for win,w in zip(windows, weights):
        x_taper = win*x
        x_taper -= x_taper.mean(-1)[...,np.newaxis]
        temp = np.fft.rfft(x_taper, n=nfft, axis=-1)
        csd += w*np.einsum('ik,jk->ijk', np.conj(temp), temp)
    csd /= weights.sum()
    return csd

def eigh_rank(a, b):
    '''Calculate the generalizid eigenvalue problem for hermitian matrices
    if b is rank deficient

    Args:
        a: square matrix a
        b: square matrix b

    Output:
        w - the eigenvalues in descending order
        v - the eigenvectors corresponding to the eigenvalues

    Notes:
        if b is rank deficient, the components with zero variance are
        discarded

        The eigenvector corresponding to the eigenvalue w[i] is the column
        v[:,i].
    '''
    rank = np.linalg.matrix_rank(b)
    w, v = scipy.linalg.eigh(b)
    # get the whitening matrix
    v = v[:,-rank:]/np.sqrt(w[-rank:])
    eigval, temp = scipy.linalg.eigh(v.T.dot(a).dot(v))
    filt = v.dot(temp)
    return np.sort(eigval)[::-1], filt[:,np.argsort(eigval)[::-1]]

def power_quot_grad(w, target_cov, contrast_cov):
    target_power = w.dot(w.dot(target_cov))
    target_power_grad = 2*np.dot(w, target_cov)
    ###
    contrast_power = w.dot(w.dot(contrast_cov))
    contrast_power_grad = 2*np.dot(w, contrast_cov)
    ###
    quot = target_power/contrast_power
    quot_grad = (target_power_grad*contrast_power -
            target_power*contrast_power_grad)/contrast_power**2
    return -quot, -quot_grad

def avg_power_quot_grad(w, target_covs, contrast_covs):
    quot, quot_grad = list(zip(*[power_quot_grad(w, t, c)
        for t,c in zip(target_covs, contrast_covs)]))
    return np.mean(quot), np.mean(quot_grad, 0)

