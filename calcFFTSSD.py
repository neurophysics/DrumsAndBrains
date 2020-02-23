"""
This script imports the single-trial cross-spectral densities - prepared
by prepareFFTCSP.py and calculates the CSP (prestim vs poststim) across a
wide range of frequencies.

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

colors=[color1, color2, color3, color4]

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

prestim_csd = []
poststim_csd = []
f = []

for i in range(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(result_folder, 'S%02d' % i)
                + '/prepared_FFTSSD.npz', 'r') as fi:
            prestim_csd.append(fi['poststim_csd'])
            poststim_csd.append(fi['poststim_csd'])
            f.append(fi['f'])
    except:
        print(('Warning: Subject %02d could not be loaded!' %i))

poststim_norm_csd = [post/np.trace(pre).real
        for pre, post in zip(prestim_csd, poststim_csd)]

if np.all([np.all(f_now == f[0]) for f_now in f]):
    f = f[0]
#####################################################
# calculate the SSD for all subjects simultaneously #
#####################################################
# get both frequencies and their harmonics
harmonics = np.sort(np.unique(
    np.r_[np.arange(1,4)*snareFreq, np.arange(1,3)*wdBlkFreq]))
harmonics_idx = np.array([np.argmin((f-h)**2) for h in harmonics])
harmonics_csd = [c[...,harmonics_idx] for c in poststim_norm_csd]

# contrast against the frequency range between 0.5 and 5 Hz
contrast_idx = np.flatnonzero(np.all([f>=0.5, f<5],0))
contrast_idx = contrast_idx[~np.isin(contrast_idx, harmonics_idx)]
contrast_csd = [(c[...,contrast_idx]).mean(-1) for c in poststim_norm_csd]

## normalize (for every subject individually) with the power in the contrast
## frequencies
#harmonics_csd, contrast_csd = zip(*[(h/np.trace(c).real, c/np.trace(c).real)
#    for h,c in zip(harmonics_csd, contrast_csd)])

# calculate the SSD of the chosen frequencies with the harmonic mean as target
# take p=-10 to emphasize that the minimum SNR of the frequencies is the target
import gmeanSSD
SSD_obj, SSD_filt = gmeanSSD.gmeanSSD(
        np.mean(harmonics_csd, 0).T.real, np.mean(contrast_csd, 0).real,
        p=-100, num=None)

# calculate the objective function for every subject individually
SSD_obj_per_subject = np.array([
    np.array([
        -gmeanSSD.obj(f/np.sqrt(np.sum(f**2)), h.T.real, c.real, p=-1)[0]
        for f in SSD_filt.T])
    for h, c in zip(harmonics_csd, contrast_csd)])

SSD_patterns = scipy.linalg.solve(
        SSD_filt.T.dot(np.mean(harmonics_csd, 0).mean(-1).T.real).dot(
            SSD_filt),
        SSD_filt.T.dot(np.mean(harmonics_csd, 0).mean(-1).T.real))
#normalize the patterns such that Cz is always positive
SSD_patterns*=np.sign(SSD_patterns[:,np.asarray(channames)=='CZ'])

# plot the SSD components scalp maps
potmaps = [meet.sphere.potMap(chancoords, pat_now,
    projection='stereographic') for pat_now in SSD_patterns]

h1 = 1
h2 = 1.3
h3 = 1

fig = plt.figure(figsize=(5.512,5.512))
gs = mpl.gridspec.GridSpec(3,1, height_ratios = [h1,h2,h3])

SNNR_ax = fig.add_subplot(gs[0,:])
SNNR_ax.plot(range(1,len(SSD_obj) + 1), 10*np.log10(SSD_obj), 'ko-', lw=2,
        markersize=5)
SNNR_ax.scatter([1], 10*np.log10(SSD_obj[0]), c=color1, s=60, zorder=1000)
SNNR_ax.scatter([2], 10*np.log10(SSD_obj[1]), c=color2, s=60, zorder=1000)
SNNR_ax.scatter([3], 10*np.log10(SSD_obj[2]), c=color3, s=60, zorder=1000)
SNNR_ax.scatter([4], 10*np.log10(SSD_obj[3]), c=color4, s=60, zorder=1000)
SNNR_ax.axhline(0, c='k', lw=1)
SNNR_ax.set_xlim([0.5, len(SSD_obj)])
SNNR_ax.set_xticks(np.r_[1,range(5, len(SSD_obj) + 1, 5)])
SNNR_ax.set_ylabel('SNR (dB)')
SNNR_ax.set_xlabel('component (index)')
SNNR_ax.set_title('resulting SNR after SSD')

# plot the four spatial patterns
gs2 = mpl.gridspec.GridSpecFromSubplotSpec(2,4, gs[1,:],
        height_ratios = [1,0.1], wspace=0, hspace=0.8)
head_ax = []
pc = []
for i, pat in enumerate(potmaps[:4]):
    try:
        head_ax.append(fig.add_subplot(gs2[0,i], sharex=head_ax[0],
            sharey=head_ax[0], frame_on=False, aspect='equal'))
    except IndexError:
        head_ax.append(fig.add_subplot(gs2[0,i], frame_on=False, aspect='equal'))
    Z = pat[2]/np.abs(pat[2]).max()
    pc.append(head_ax[-1].pcolormesh(
        *pat[:2], Z, rasterized=True, cmap='coolwarm', vmin=-1, vmax=1))
    head_ax[-1].contour(*pat, levels=[0], colors='w')
    head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5, zorder=1001)
    head_ax[-1].set_xlabel(r'\textbf{%d}' % (i + 1) +'\n'+
            '($\mathrm{SNR=%.2f\ dB}$)' % (10*np.log10(SSD_obj[i])))
    head_ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(head_ax[-1], ec=colors[i], zorder=1000, lw=3)
head_ax[0].set_ylim([-1.1,1.3])
head_ax[0].set_xlim([-1.6,1.6])

# add a colorbar
cbar_ax = fig.add_subplot(gs2[1,:])
cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
        label='amplitude (a.u.)', ticks=[-1,0,1])
cbar.ax.set_xticklabels(['-', '0', '+'])
cbar.ax.axvline(0, c='w', lw=2)

spect_ax = fig.add_subplot(gs[2,:])
[spect_ax.plot(f,
    10*np.log10(SSD_filt[:,i].dot(SSD_filt[:,i].dot(
        np.mean([t/np.trace(t[...,contrast_idx].mean(-1)).real
            for t in poststim_norm_csd], 0).real))),
        c=colors[i], lw=2) for i in range(4)]
spect_ax.set_xlim([0.5, 8])
spect_ax.set_ylim([-1.1, 2.1])
spect_ax.set_yticks([-1, 0, 1, 2])
spect_ax.axhline(0, c='k', lw=1)
spect_ax.set_xlabel('frequency (Hz)')
spect_ax.set_ylabel('SNR (dB)')
spect_ax.set_title('normalized spectrum')

spect_ax.axvline(snareFreq, color='b', zorder=0, lw=2)
spect_ax.axvline(2*snareFreq, color='b', zorder=0, lw=2)
spect_ax.axvline(4*snareFreq, color='b', zorder=0, lw=1, ls='--')
spect_ax.axvline(5*snareFreq, color='b', zorder=0, lw=1, ls='--')
spect_ax.axvline(wdBlkFreq, color='r', zorder=0, lw=2)
spect_ax.axvline(2*wdBlkFreq, color='k', zorder=0, lw=2)
spect_ax.axvline(3*wdBlkFreq, color='r', zorder=0, lw=1, ls='--')
spect_ax.axvline(4*wdBlkFreq, color='k', zorder=0, lw=1, ls='--')

gs.tight_layout(fig, pad=0.2, h_pad=0.8)

fig.savefig(os.path.join(result_folder, 'FFTSSD_patterns.pdf'))
fig.savefig(os.path.join(result_folder, 'FFTSSD_patterns.png'))

## save the results
np.savez(os.path.join(result_folder, 'FFTSSD.npz'),
        SSD_obj = SSD_obj,
        SSD_obj_per_subject = SSD_obj_per_subject,
        SSD_filt = SSD_filt,
        SSD_patterns = SSD_patterns)
