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
color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()

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

# contrast against the frequency range between 0.5 and 4 Hz
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
        p=-10, num=None)

# plot the results
[plt.loglog(f,
    SSD_filt[:,i].dot(SSD_filt[:,i].dot(
        np.mean([t/np.trace(t[...,contrast_idx].mean(-1)).real
            for t in poststim_norm_csd], 0).real))) for i in range(6)]
plt.gca().axhline(1, lw=1, c='k')
plt.gca().set_xlim([0.5, 10])
plt.gca().set_ylim([0.5, 1.5])
1/0
plt.loglog(f, np.mean([eigvect[:,0].dot(c/np.trace(c).real).T.dot(
    eigvect[:,0]).real for c in prestim_csd], 0))

def calcPrePostCSP(prestim_csd, poststim_csd, randomize=False):
    """A function to calculate the CSP between pre- and poststim matrices,
    possibly after randomization between prestim and poststim
    
    Args:
        prestim_csd: a list of channel x  channel x frequency matrices of
            csds obtained pre-stimulus
        poststim_csd: a list of channel x channel x frequency matrices of
            csds obtained post-simulus
        randomize (bool): whether the pre- and post-stimulus groups should
            be randomized - usful for permutation testing
    """
    if randomize:
        # mix between prestim and poststim
        N = len(prestim_csd)
        chooser = np.random.choice([True,False], N, replace=True)
        prestim_idx = range(N) + chooser*N
        poststim_idx = range(N) + ~chooser*N
        prestim_csd, poststim_csd = zip(*[(
            (prestim_csd + poststim_csd)[pre_idx],
            (prestim_csd + poststim_csd)[post_idx])
            for pre_idx, post_idx in
            zip(prestim_idx, poststim_idx)])
    ## normalize both matrices by the prestim period
    prestim_csd, poststim_csd = zip(*[
        (pre/np.trace(pre).real, post/np.trace(pre).real)
        for pre,post in zip(prestim_csd, poststim_csd)])
    # average across subjects
    prestim_avg_csd = np.mean(prestim_csd, 0)
    poststim_avg_csd = np.mean(poststim_csd, 0)
    # calculate the CSP as an eigenvalue decomposition
    eigval, eigvect = zip(*[helper_functions.eigh_rank(post.real, pre.real)
        for pre, post in zip(prestim_avg_csd.T, poststim_avg_csd.T)])
    return eigval, eigvect

eigval, eigvect = calcPrePostCSP(prestim_csd, listen_csd)

1/0

## normalize both matrices by the prestim period
prestim_csd, listen_csd = zip(*[
    (pre/np.trace(pre).real, post/np.trace(pre).real)
    for pre,post in zip(prestim_csd, listen_csd)])
# average across subjects
prestim_avg_csd = np.mean(prestim_csd, 0)
listen_avg_csd = np.mean(listen_csd, 0)

chosen_CSP = np.argmin((f-1.75)**2)

snare_quot = eigval[chosen_CSP]
snare_filt = eigvect[chosen_CSP]
wdBlk_quot = eigval[chosen_CSP]
wdBlk_filt = eigvect[chosen_CSP]

snare_pattern = scipy.linalg.solve(
        snare_filt.T.dot(listen_avg_csd[...,chosen_CSP].real).dot(
            snare_filt),
        snare_filt.T.dot(listen_avg_csd[...,chosen_CSP].real))
wdBlk_pattern = scipy.linalg.solve(
        wdBlk_filt.T.dot(listen_avg_csd[...,chosen_CSP].real).dot(
            snare_filt),
        wdBlk_filt.T.dot(listen_avg_csd[...,chosen_CSP].real))

quot = eigval

## save the results
np.savez(os.path.join(result_folder, 'FFTCSP.npz'),
        snare_filt = snare_filt,
        snare_quot = snare_quot,
        snare_pattern = snare_pattern,
        wdBlk_filt = wdBlk_filt,
        wdBlk_quot = wdBlk_quot,
        wdBlk_pattern = wdBlk_pattern)

# plot the patterns
# name the ssd channels
snare_channames = ['SSD-%02d' % (i+1) for i in range(len(snare_pattern))]
wdBlk_channames = ['SSD-%02d' % (i+1) for i in range(len(wdBlk_pattern))]

# plot the SSD components scalp maps
snare_potmaps = [meet.sphere.potMap(chancoords, ssd_c,
    projection='stereographic') for ssd_c in snare_pattern]
wdBlk_potmaps = [meet.sphere.potMap(chancoords, ssd_c,
    projection='stereographic') for ssd_c in wdBlk_pattern]

fig = plt.figure(figsize=(3.54,3.54))
# plot with 8 rows and 4 columns
#gs = mpl.gridspec.GridSpec(10,4, height_ratios = 8*[1]+[0.2]+[1])
gs = mpl.gridspec.GridSpec(4,4, height_ratios = [1]+[1]+[0.7]+[0.1])

SNNR_ax = fig.add_subplot(gs[0,:], frame_on=True)
SNNR_ax.plot(f, [10*np.log10(q[0]) for q in quot], 'k-')
SNNR_ax.set_xlabel('frequency (Hz)')
SNNR_ax.set_ylabel('SNNR (dB)')
SNNR_ax.set_ylim(bottom=0, top=2.3)

SNNR_ax.set_xlim(1,5)
trans = mpl.transforms.blended_transform_factory(
        SNNR_ax.transData, SNNR_ax.transAxes)

SNNR_ax.text(snareFreq, 0.96, r'\textbf{*}', ha='center', va='top',
        transform=trans, color='b', fontsize=12)
SNNR_ax.text(wdBlkFreq, 0.96, r'\textbf{*}', ha='center', va='top',
        transform=trans, color='r', fontsize=12)
SNNR_ax.text(3.5, 0.96, r'\textbf{*}', ha='center', va='top',
        transform=trans, color='k', fontsize=12)
#SNNR_ax.text(7, 0.96, r'\textbf{*}', ha='center', va='top',
#        transform=trans, color='k', fontsize=12)

#mark_start = 0.5*(f[use_idx[chosen_SSD] - 3] + f[use_idx[chosen_SSD] - 2])
#mark_stop = 0.5*(f[use_idx[chosen_SSD] + 3] + f[use_idx[chosen_SSD] + 2])
#SNNR_ax.axvspan(mark_start, mark_stop, fc='r', alpha=0.2)

SNNR_ax.plot([f[chosen_CSP], f[chosen_CSP]], [10*np.log10(quot[chosen_CSP][0]),0], 'k-', lw=0.5)

eigvals_ax = fig.add_subplot(gs[1,:], frame_on=True)
eigvals_ax.plot(np.arange(1, len(snare_quot) + 1, 1), 10*np.log10(quot[chosen_CSP]),'ko-', markersize=np.sqrt(20))
eigvals_ax.set_xlim([0, len(snare_quot) + 1])
eigvals_ax.axhline(0, ls='-', c='k', lw=0.5)
eigvals_ax.axvspan(0, 4.5, fc='r', alpha=0.2)
eigvals_ax.set_ylabel('SNNR (dB)')
eigvals_ax.set_xlabel('component index')
ax = []
for i, (X,Y,Z) in enumerate(snare_potmaps):
    if i==4: break
    if i == 0:
        ax.append(fig.add_subplot(gs[2,0], frame_on = False))
    else:
        ax.append(fig.add_subplot(gs[2 + i//4,i%4], sharex=ax[0],
            sharey=ax[0], frame_on = False))
    Z /= np.abs(Z).max()
    ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(ax[-1])
    pc = ax[-1].pcolormesh(X, Y, Z, vmin=-1, vmax=1, rasterized=True,
            cmap=cmap)
    ax[-1].contour(X, Y, Z, levels=[0], colors='w')
    ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5)
    ax[-1].set_xlabel(r'\textbf{%d}' % (i + 1) +'\n'+
            '($\mathrm{SNNR=%.2f\ dB}$)' % (10*np.log10(snare_quot[i])))
ax[0].set_ylim([-1.1,1.3])

pc_ax = fig.add_subplot(gs[3,:])
cbar = plt.colorbar(pc, cax=pc_ax, orientation='horizontal',
        label='amplitude', ticks=[-1,0,1])
cbar.ax.set_xticklabels(['-', '0', '+'])
cbar.ax.axvline(0.5, c='w')
pc_ax.plot([0.5,0.5], [0,1], c='w', zorder=1000,
        transform=pc_ax.transAxes)

gs.tight_layout(fig, pad=0.2)

fig.canvas.draw()
SNNR_ax.plot(
        [f[chosen_CSP], 1],
            [0,
                SNNR_ax.transData.inverted().transform(
                    eigvals_ax.transAxes.transform([0,1]))[1]],
                'k-', clip_on=False, alpha=0.5, lw=0.5)
SNNR_ax.plot(
        [f[chosen_CSP], 5],
            [0,
                SNNR_ax.transData.inverted().transform(
                    eigvals_ax.transAxes.transform([1,1]))[1]],
                'k-', clip_on=False, alpha=0.5, lw=0.5)

fig.align_ylabels([SNNR_ax, eigvals_ax])
fig.savefig(os.path.join(result_folder, 'FFTCSP_patterns.pdf'))

