"""
This script imports the single-trial cross-spectral densities - prepared
by prepareFFTSSD.py - and calculates the SSD of stimulation frequencies
vs the neighbouring frequencies.

As input it requests the result folder
"""
import numpy as np
import scipy
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os.path
import csv
import helper_functions
import meet
from tqdm import trange, tqdm # for a progress bar


# set parameters
## input
result_folder = sys.argv[1]
N_subjects = 21

## sampling rate of the EEG
s_rate = 1000

## target frequencies
snareFreq = 7./6
wdBlkFreq = 7./4

## plot
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


# read data (from channels.txt and prepared_FFTSSD.npz)
## read the channel names
channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
N_channels = len(channames)

## read the data of the single subjects
f = [] #frequency bins
F = [] #discrete Fourier transform
F_listen = []
F_silence = []
target_cov = [] #covariance matrix of frequencies 1.16 and 1.75
contrast_cov = [] #cov matrix of other frequencies in [1,2]
snareInlier = [] # which trials are Inlier - this is needed to relate EEG to
                 # behaviour
wdBlkInlier = []

for i in range(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(result_folder, 'S%02d' % i)
                + '/prepared_FFTSSD.npz', 'r') as fi:
            target_cov.append(fi['target_cov'])
            contrast_cov.append(fi['contrast_cov'])
            F.append(fi['F'])
            F_listen.append(fi['F_listen'])
            F_silence.append(fi['F_silence'])
            f.append(fi['f'])
            snareInlier.append(fi['snareInlier'])
            wdBlkInlier.append(fi['wdBlkInlier'])
    except:
        print(('Warning: Subject %02d could not be loaded!' %i))

# data preprocessing
## the frequency array should be the same for all subjects
if np.all([np.all(f[0] == f_now) for f_now in f]):
    f = f[0]

## average the covariance matrices across all subjects
for t, c in zip(target_cov, contrast_cov):
    # normalize by the trace of the contrast covariance matrix
    t_now = t.mean(-1)/np.trace(c.mean(-1))
    c_now = c.mean(-1)/np.trace(c.mean(-1))
    #t_now = t.mean(-1)
    #c_now = c.mean(-1)    #averaged over trials => shape (32,32)
    try:
        all_target_cov += t_now
        all_contrast_cov += c_now
    except: #init
        all_target_cov = t_now
        all_contrast_cov = c_now

# calculate SSD
## EV and filter
SSD_eigvals, SSD_filters = helper_functions.eigh_rank(
        all_target_cov, all_contrast_cov)

## patterns
SSD_patterns = scipy.linalg.solve(
        SSD_filters.T.dot(all_target_cov).dot(SSD_filters),
        SSD_filters.T.dot(all_target_cov))

### normalize the patterns such that Cz is always positive
SSD_patterns*=np.sign(SSD_patterns[:,np.asarray(channames)=='CZ'])

# average and normalize to plot
## apply SSD to FFT
F_SSD_both = [np.tensordot(SSD_filters, F_now, axes=(0,0)) for F_now in F]
F_SSD_listen = [np.tensordot(SSD_filters, F_now, axes=(0,0)) for F_now in F_listen]
F_SSD_silence = [np.tensordot(SSD_filters, F_now, axes=(0,0)) for F_now in F_silence]
## average across trials
F_SSD_mean = [(np.abs(F_now)**2).mean(-1) for F_now in F_SSD]
F_mean = [(np.abs(F_now)**2).mean(-1) for F_now in F]

## average across subjects
F_SSD_subj_mean = np.mean(F_SSD_mean, axis=0)
F_subj_mean = np.mean(F_mean, axis=0)

## normalize by mean power of frequencies (except snare/wdblk)
## (divide to get SNR => want higher SNR at target frequence)
### compute target and contrast mask
contrast_freqwin = [1, 2]
contrast_mask = np.all([f>=contrast_freqwin[0], f<=contrast_freqwin[1]], 0)

target_mask = np.zeros(f.shape, bool)
target_mask[np.argmin((f-snareFreq)**2)] = True
target_mask[np.argmin((f-wdBlkFreq)**2)] = True

### divide by mean power of frequencies (except snare/wdblk)
F_SSD_subj_mean_norm = F_SSD_subj_mean/F_SSD_subj_mean[
        :,target_mask != contrast_mask].mean(-1)[:,np.newaxis]
F_subj_mean_norm = F_subj_mean/F_subj_mean[
        :,target_mask != contrast_mask].mean(-1)[:,np.newaxis]

## alternatively, normalize for each frequency by their neighboring frequencies
F_SSD_subj_mean_peak = F_SSD_subj_mean / scipy.ndimage.convolve1d(
        F_SSD_subj_mean, np.r_[[1]*2, 0, [1]*2]/4)
F_subj_mean_peak = F_subj_mean / scipy.ndimage.convolve1d(
        F_subj_mean, np.r_[[1]*2, 0, [1]*2]/4)

# plot the results
f_plot_mask = np.all([f>=0.5, f<=4], 0)
SSD_num = 4

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(f[f_plot_mask], 20*np.log10(F_subj_mean_norm[:,f_plot_mask].T),
        'k-', alpha=0.1)
ax.plot(f[f_plot_mask], 20*np.log10(F_SSD_subj_mean_norm[:SSD_num,
    f_plot_mask].T))

save_results = {}
for i, (snareInlier_now, wdBlkInlier_now,
    F_SSD_both_now, F_SSD_listen_now, F_SSD_silence_now) in enumerate(zip(
    snareInlier, wdBlkInlier, F_SSD_both, F_SSD_listen, F_SSD_silence)):
    save_results['snareInlier_{:02d}'.format(i)] = snareInlier_now
    save_results['wdBlkInlier_{:02d}'.format(i)] = wdBlkInlier_now
    save_results['F_SSD_both_{:02d}'.format(i)] = F_SSD_both_now
    save_results['F_SSD_listen_{:02d}'.format(i)] = F_SSD_listen_now
    save_results['F_SSD_silence_{:02d}'.format(i)] = F_SSD_silence_now


# save the results
## save sorted F_SSD: add 4th dim that contains subject number to sort by

# F_SSD sorted, arr, and F_SSD_zip are not used anywhere???
#F_SSD_sorted = [F_now[:,:,:,np.newaxis] for F_now in F_SSD]
#for i in range(len(F_SSD)):
#    arr = F_SSD_sorted[i]
#    arr[0,0,0,0]=i
#F_SSD_zip = zip(F_SSD,range(len(F_SSD)))

np.savez(os.path.join(result_folder, 'F_SSD.npz'), **save_results, f=f)

## save SSD eigenvalues, filters and patterns in a.npz
np.savez(os.path.join(result_folder, 'FFTSSD.npz'),
        SSD_eigvals = SSD_eigvals,
        SSD_filters = SSD_filters,
        SSD_patterns = SSD_patterns
        )
