"""
Create the design matrix of the Within-Between Random Effects model
"""
import numpy as np
import sys
import os.path
from scipy.stats import zscore

try:
    result_folder = sys.argv[1]
except:
    result_folder = './Results/'


# target frequencies
snareFreq = 7./6
wdBlkFreq = 7./4

# number of SSD_components to use
N_SSD = 2

# load the SSD results from all subjects into a list
F_SSD = []

with np.load(os.path.join(result_folder, 'F_SSD.npz'), 'r') as fi:
    # load the frequency array
    f = fi['f']
    # find the index of the frequency array refering to snare and woodblock
    # frequency
    snare_idx = np.argmin((f - snareFreq)**2)
    wdBlk_idx = np.argmin((f - wdBlkFreq)**2)
    # loop through all arrays
    i = 0
    while True:
        try:
            temp = fi['arr_{}'.format(i)][:N_SSD, (snare_idx, wdBlk_idx),:,0]
            F_SSD.append(temp.reshape((-1, temp.shape[-1]), order='F'))
            i += 1
        except KeyError:
            break


# get the total number of trials
N_trials = np.sum([SSD_now.shape[-1] for SSD_now in F_SSD])

# build the design matrix
X = []
labels = []

# add intercept
labels.append('intercept')
X.append(np.ones((1, N_trials)))

# add within-subjects coefficient
labels.extend(['W1Snare', 'W2Snare', 'W1WdBlk', 'W2WdBlk'])
X.append(np.hstack([zscore(np.abs(SSD_now), -1) for SSD_now in F_SSD]))

# add between subjects coefficient
labels.extend(['B1Snare', 'B2Snare', 'B1WdBlk', 'B2WdBlk'])
X.append(zscore(
    np.hstack([np.abs(SSD_now).mean(-1)[:,np.newaxis]*np.ones(SSD_now.shape)
        for SSD_now in F_SSD]), -1))

# TODO: add all other rows of the matrix
