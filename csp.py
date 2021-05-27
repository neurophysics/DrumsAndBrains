"""
executes common spatial pattern algorithm on ERD data
"""
import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
import scipy
import meet

data_folder = sys.argv[1]
result_folder = sys.argv[2]
N_subjects = 21
s_rate = 1000 # sampling rate of the EEG
fbands = [[1,4], [4,8], [8,12], [12,20], [20,40]]

# read ERD data
# erd data is referenced to the average eeg amplitude, bandpass filtered
# with order 6, normalized s.t. 2 sec pre responce are 100%, trial-averaged
snareInlier = [] # list of 20 subjects, each shape ≤75
wdBlkInlier = []
ERD = [] # lists of 20 subjects, each list of 5 bands, each shape (32,2500)
i=0
while True:
    try:
        with np.load(os.path.join(result_folder, 'inlier_response.npz'),
            'r') as fi:
            snareInlier.append(fi['snareInlier_response_{:02d}'.format(i)])
            wdBlkInlier.append(fi['wdBlkInlier_response_{:02d}'.format(i)])
        with np.load(os.path.join(result_folder, 'ERD.npz'), 'r') as f_erd:
            ERD.append(f_erd['ERD_{:02d}'.format(i)])
            #fbands = f_erd['fbands'] # need to rerun basic motor for this
        i+=1
    except KeyError:
        break

# only take alpha (8-12) band for Now
ERD_alpha = [e[2] for e in ERD]

# calculate covariance matrices for all subjects
contrast_cov = []
target_cov = []
for erd in ERD_alpha:
    X_baseline = erd[:, :900] # class 1: baseline activity (on plot seemed to be -2 to -1.1s)
    X_erd = erd[:, 900:2000] # class 2: pre-response activity (ERD)
    contrast_cov.append(np.cov(X_baseline))
    target_cov.append(np.cov(X_erd))

## average the covariance matrices across all subjects
#@gunnar: we did this for ssd, here as well? would make sense to me
for t, c in zip(target_cov, contrast_cov):
    # normalize by the trace of the contrast covariance matrix
    t_now = t/np.trace(c)
    c_now = c/np.trace(c)
    try:
        all_target_cov += t_now
        all_contrast_cov += c_now
    except: #init
        all_target_cov = t_now
        all_contrast_cov = c_now

# calculate CSP
## EV and filter
#@gunnar: should I use helper_function.eigh_rank as you did for ssd?
CSP_eigvals, CSP_filters = scipy.linalg.eigh(all_target_cov,
        all_target_cov + all_contrast_cov)

# patterns
CSP_patterns = scipy.linalg.solve(
        CSP_filters.T.dot(all_target_cov).dot(CSP_filters),
        CSP_filters.T.dot(all_target_cov))

### normalize the patterns such that Cz is always positive
#@gunnar: we did this for ssd, here as well? would make sense to me
channames = meet.sphere.getChannelNames('channels.txt')
CSP_patterns*=np.sign(CSP_patterns[:,np.asarray(channames)=='CZ'])


np.savez(os.path.join(result_folder, 'CSP.npz'),
        CSP_eigvals = CSP_eigvals,
        CSP_filters = CSP_filters, # matrix W in paper
        CSP_patterns = CSP_patterns # matrix A in paper
        )
