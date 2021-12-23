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

colors=[color1, color2, color3, color4]

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
target_cov = [] #covariance matrix of frequencies 1.16 and 1.75
contrast_cov = [] #cov matrix of other frequencies in [1,2]
snareInlier = [] # which trials are Inlier - this is needed to relate EEG to
                 # behaviour
wdBlkInlier = []
snareInlier_listen = []
wdBlkInlier_listen = []
snareInlier_silence = []
wdBlkInlier_silence = []

for i in range(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(result_folder, 'S%02d' % i)
                + '/prepared_FFTSSD.npz', 'r') as fi:
            target_cov.append(fi['target_cov'])
            contrast_cov.append(fi['contrast_cov'])
            F.append(fi['F'])
            f.append(fi['f'])
            snareInlier.append(fi['snareInlier'])
            wdBlkInlier.append(fi['wdBlkInlier'])
            snareInlier_listen.append(fi['snareInlier_listen'])
            wdBlkInlier_listen.append(fi['wdBlkInlier_listen'])
            snareInlier_silence.append(fi['snareInlier_silence'])
            wdBlkInlier_silence.append(fi['wdBlkInlier_silence'])
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
    # averaged over trials => shape (32,32)
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

## average across trials
F_SSD_mean = [(np.abs(F_now)**2).mean(-1) for F_now in F_SSD_both]
F_mean = [(np.abs(F_now)**2).mean(-1) for F_now in F]

## average across subjects
F_SSD_subj_mean = np.mean(F_SSD_mean, axis=0)
F_subj_mean = np.mean(F_mean, axis=0)

## normalize by mean power of frequencies (except snare/wdblk)
## (divide to get SNR => want higher SNR at target frequence)
### compute target and contrast mask
contrast_freqwin = [1, 4]
contrast_mask = np.all([f>=contrast_freqwin[0], f<=contrast_freqwin[1]], 0)

target_mask = np.zeros(f.shape, bool)
target_mask[np.argmin((f-snareFreq)**2)] = True
target_mask[np.argmin((f-wdBlkFreq)**2)] = True

# save the results
save_results = {}
for i, (snareInlier_now, wdBlkInlier_now,
    snareInlier_listen_now, wdBlkInlier_listen_now,
    snareInlier_silence_now, wdBlkInlier_silence_now) in enumerate(zip(
    snareInlier, wdBlkInlier, snareInlier_listen,
    wdBlkInlier_listen, snareInlier_silence, wdBlkInlier_silence)):
    save_results['snareInlier_{:02d}'.format(i)] = snareInlier_now
    save_results['wdBlkInlier_{:02d}'.format(i)] = wdBlkInlier_now
    save_results['snareInlier_listen_{:02d}'.format(i)] = snareInlier_listen_now
    save_results['wdBlkInlier_listen_{:02d}'.format(i)] = wdBlkInlier_listen_now
    save_results['snareInlier_silence_{:02d}'.format(i)] = snareInlier_silence_now
    save_results['wdBlkInlier_silence_{:02d}'.format(i)] = wdBlkInlier_silence_now

np.savez(os.path.join(result_folder, 'F_SSD.npz'), **save_results, f=f)

## save SSD eigenvalues, filters and patterns in a.npz
np.savez(os.path.join(result_folder, 'FFTSSD.npz'),
        SSD_eigvals = SSD_eigvals,
        SSD_filters = SSD_filters,
        SSD_patterns = SSD_patterns
        )

######################################
# plot the resulting EV and patterns #
######################################

# prepare the SSD components scalp maps
potmaps = [meet.sphere.potMap(chancoords, pat_now,
    projection='stereographic') for pat_now in SSD_patterns]

# define the height ratios of the subplot rows
h1 = 1
h2 = 1.3
h3 = 1

fig = plt.figure(figsize=(5.512,5.512))
gs = mpl.gridspec.GridSpec(3,1, height_ratios = [h1,h2,h3])

SNNR_ax = fig.add_subplot(gs[0,:])
SNNR_ax.plot(range(1,len(SSD_eigvals) + 1), 10*np.log10(SSD_eigvals), 'ko-', lw=2,
        markersize=5)
SNNR_ax.scatter([1], 10*np.log10(SSD_eigvals[0]), c=color1, s=60, zorder=1000)
SNNR_ax.scatter([2], 10*np.log10(SSD_eigvals[1]), c=color2, s=60, zorder=1000)
SNNR_ax.scatter([3], 10*np.log10(SSD_eigvals[2]), c=color3, s=60, zorder=1000)
SNNR_ax.scatter([4], 10*np.log10(SSD_eigvals[3]), c=color4, s=60, zorder=1000)
#SNNR_ax.axhline(0, c='k', lw=1)
SNNR_ax.set_xlim([0.5, len(SSD_eigvals)])
SNNR_ax.set_xticks(np.r_[1,range(5, len(SSD_eigvals) + 1, 5)])
SNNR_ax.set_ylabel('SNR (dB)')
SNNR_ax.set_xlabel('component (index)')
SNNR_ax.set_title('resulting SNR after SSD')

# plot the four spatial patterns
gs2 = mpl.gridspec.GridSpecFromSubplotSpec(2,4, gs[1,:],
        height_ratios = [1,0.1], wspace=0, hspace=0.8)
head_ax = []
pc = []
for i, pat in enumerate(potmaps[:4]):
    head_ax.append(fig.add_subplot(gs2[0,i], frame_on=False,
        aspect='equal'))
    # delete all ticks and ticklabels
    head_ax[-1].tick_params(**blind_ax)
    head_ax[-1].sharex(head_ax[0])
    head_ax[-1].sharey(head_ax[0])
    # scale the color of the pattern
    Z = pat[2]/np.abs(pat[2]).max()
    pc.append(head_ax[-1].pcolormesh(
        *pat[:2], Z, rasterized=True,
        cmap='coolwarm', vmin=-1, vmax=1, shading='auto'))
    head_ax[-1].contour(*pat, levels=[0], colors='w')
    head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5, zorder=1001)
    head_ax[-1].set_xlabel(r'\textbf{%d}' % (i + 1) +'\n'+
            '($\mathrm{SNR=%.2f\ dB}$)' % (10*np.log10(SSD_eigvals[i])))
    meet.sphere.addHead(head_ax[-1], ec=colors[i], zorder=1000, lw=3)
head_ax[0].set_ylim([-1.2,1.2])
head_ax[0].set_xlim([-1.1,1.2])

# add a colorbar
cbar_ax = fig.add_subplot(gs2[1,:])
cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
        label='amplitude (a.u.)', ticks=[-1,0,1])
cbar.ax.set_xticklabels(['-', '0', '+'])
cbar.ax.axvline(0, c='w', lw=2)

spect_ax = fig.add_subplot(gs[2,:])

[spect_ax.plot(f,
    10*np.log10(np.mean([t/t[...,contrast_mask != target_mask].mean(
        -1)[:,np.newaxis]
        for t in F_SSD_mean], 0)[i]),
        c=colors[i], lw=2) for i in range(4)]
spect_ax.set_xlim([0.5, 8])
spect_ax.set_ylim([-6, 12])
spect_ax.axhline(0, c='k', lw=1)
spect_ax.set_xlabel('frequency (Hz)')
spect_ax.set_ylabel('SNR (dB)')
spect_ax.set_title('normalized spectrum')

spect_ax.axvline(snareFreq, color='b', zorder=0, lw=1)
spect_ax.axvline(2*snareFreq, color='b', zorder=0, lw=1)
spect_ax.axvline(wdBlkFreq, color='r', zorder=0, lw=1)
spect_ax.axvline(2*wdBlkFreq, color='k', zorder=0, lw=1)

spect_ax.axvline(4*snareFreq, color='b', zorder=0, lw=1, ls=':')
spect_ax.axvline(5*snareFreq, color='b', zorder=0, lw=1, ls=':')
spect_ax.axvline(3*wdBlkFreq, color='r', zorder=0, lw=1, ls=':')
spect_ax.axvline(4*wdBlkFreq, color='k', zorder=0, lw=1, ls=':')

gs.tight_layout(fig, pad=0.2, h_pad=0.8)#
fig.canvas.draw()

# make sure that the heads are round
head_extent = (head_ax[0].transData.transform((1,1)) -
        head_ax[0].transData.transform((0,0)))
if head_extent[0] < head_extent[1]:
    head_ax[0].set_ylim(np.r_[head_ax[0].get_ylim()] *
            head_extent[1] / head_extent[0])
else:
    head_ax[0].set_xlim(np.r_[head_ax[0].get_xlim()] /
            (head_extent[1] / head_extent[0]))

fig.savefig(os.path.join(result_folder, 'FFTSSD_patterns.pdf'))
