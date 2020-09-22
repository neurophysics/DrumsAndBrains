"""
This script imports the single-trial cross-spectral densities - prepared
by prepareFFTSSD.py and calculates the SSD of stimulation frequencies (and)
harmonics vs the neighbouring frequencies.

As input it requests the result folder
"""

import numpy as np
import scipy
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os.path
import helper_functions
import meet
from tqdm import trange, tqdm # for a progress bar

mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 10

cmap = 'plasma'

color1 = '#1f78b4'.upper()
color2 = '#33a02c'.upper()
color3 = '#b2df8a'.upper()
color4 = '#a6cee3'.upper()

colors=[color1, color2, color3, color4, 'k']

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

s_rate = 1000 # sampling rate of the EEG

result_folder = sys.argv[1]
N_subjects = 21

snareFreq = 7./6
wdBlkFreq = 7./4

# read the channel names
channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
N_channels = len(channames)

target_cov = []
contrast_cov = []

for i in range(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(result_folder, 'S%02d' % i)
                + '/prepared_FFTSSD.npz', 'r') as fi:
            target_cov.append(fi['target_cov'])
            contrast_cov.append(fi['contrast_cov'])
    except:
        print(('Warning: Subject %02d could not be loaded!' %i))

# average the covariance matrices across all subjects
# normalize by the trace of the contrast covariance matrix

for t, c in zip(target_cov, contrast_cov):
    t_now = t.mean(-1)/np.trace(c.mean(-1))
    c_now = c.mean(-1)/np.trace(c.mean(-1))
    try:
        all_target_cov += t_now
        all_contrast_cov += c_now
    except:
        all_target_cov = t_now
        all_contrast_cov = c_now

SSD_eigvals, SSD_filters = helper_functions.eigh_rank(
        all_target_cov, all_contrast_cov)

SSD_patterns = scipy.linalg.solve(
        SSD_filters.T.dot(all_target_cov).dot(SSD_filters),
        SSD_filters.T.dot(all_target_cov))

#normalize the patterns such that Cz is always positive
SSD_patterns*=np.sign(SSD_patterns[:,np.asarray(channames)=='CZ'])

# save the results
np.savez(os.path.join(result_folder, 'FFTSSD.npz'),
        SSD_eigvals = SSD_eigvals,
        SSD_filters = SSD_filters,
        SSD_patterns = SSD_patterns)
