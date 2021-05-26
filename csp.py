"""
executes common spatial pattern algorithm on ERD data
"""
import numpy as np
import sys
import csv
import os.path
import matplotlib.pyplot as plt
import meet
import helper_functions
import scipy as sp

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

1/0
#from AAND exercise 5 solution:
def train_CSP(epo, mrk_class):
# Usage: W, D = trainCSP(epo, mrk_class)
    C = epo.shape[1]
    X1 = np.reshape(np.transpose(epo[:,:,mrk_class==0], (1,0,2)), (C, -1))
    S1 = np.cov(X1)
    X2 = np.reshape(np.transpose(epo[:,:,mrk_class==1], (1,0,2)), (C, -1))
    S2 = np.cov(X2)
    D, W = scipy.linalg.eigh(a=S1, b=S1+S2)
    return W, D # D are EV, W is spatial pattern matrix


np.savez(os.path.join(result_folder, 'CSP.npz'),
        CSP_eigvals = CSP_eigvals,
        CSP_filters = CSP_filters, # matrix W in paper
        CSP_patterns = CSP_patterns # matrix A in paper
        )
