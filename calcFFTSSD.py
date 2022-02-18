"""
This script imports the single-trial cross-spectral densities - prepared
by prepareFFTSSD.py - and calculates the SSD of stimulation frequencies
vs the neighbouring frequencies.

As input it requests the result folder
"""

# TODO:
# Inspect spatial patterns for similarity across subjects, need to be sorted???
# save filtered single-trial data
# check source models for each of the patterns
# re-run statistical model

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
import pdb

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
mpl.rcParams['axes.titlesize'] = 8

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
            snare_target_cov = fi['snare_target_cov']
            wdBlk_target_cov = fi['wdBlk_target_cov']
            snare_contrast_cov = fi['snare_contrast_cov']
            wdBlk_contrast_cov = fi['wdBlk_contrast_cov']
            # normalize the snare frequency peak and its neighbouring
            # 'contrast' frequencies and the wdBlk peak and its contrast
            # frequencies such that both peaks contribute equally in the
            # SSD optimization
            snare_norm = np.trace(snare_contrast_cov.mean(-1))
            wdBlk_norm = np.trace(wdBlk_contrast_cov.mean(-1))
            target_cov.append(
                    snare_target_cov/snare_norm +
                    wdBlk_target_cov/wdBlk_norm)
            contrast_cov.append(
                    snare_contrast_cov/snare_norm +
                    wdBlk_contrast_cov/wdBlk_norm)
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

## normalize by mean power of frequencies (except snare/wdblk)
## (divide to get SNR => want higher SNR at target frequence)
### compute target and contrast mask
contrast_freqwin = [1, 4]
contrast_mask = np.all([f>=contrast_freqwin[0], f<=contrast_freqwin[1]], 0)

target_mask = np.zeros(f.shape, bool)
target_mask[np.argmin((f-snareFreq)**2)] = True
target_mask[np.argmin((f-wdBlkFreq)**2)] = True

# run rcsp with transfer learning
import rcsp_tlw

# cross validate the regularization parameter
alpha = np.r_[0, 10**np.linspace(-5, 5, 200)]
N_folds = 30

def cv_rcsp_tlw(alpha, N_folds, subject, target_cov, contrast_cov):
    # get the nummber of trials
    N_trials = target_cov[subject].shape[-1]
    fold_start_stop = np.linspace(0, N_trials, N_folds + 1).astype(int)
    random_idx = np.argsort(np.random.randn(N_trials))
    folds = [random_idx[fold_start_stop[i]:fold_start_stop[i+1]]
            for i in range(N_folds)]
    all_target_covs = [np.mean(c, -1) for c in target_cov]
    # initialize empty results array
    result = np.zeros([len(folds), len(alpha), 32])
    for fold_now in range(N_folds):
        train_idx = np.hstack(
                [folds[j] for j in list(range(fold_now)) +
                                   list(range(fold_now + 1, N_folds))])
        test_idx = folds[fold_now]
        c1_train = np.mean(target_cov[subject][..., train_idx], -1)
        c2_train = np.mean(contrast_cov[subject][..., train_idx], -1)
        c1_test = np.mean(target_cov[subject][..., test_idx], -1)
        c2_test = np.mean(contrast_cov[subject][..., test_idx], -1)
        for alpha_it, alpha_now in enumerate(alpha):
            rcsp_tlw_ratios, rcsp_tlw_filters = rcsp_tlw.rcsp_tlw(
                    c1_train, c2_train,
                    target_cov=all_target_covs[subject],
                    source_covs=all_target_covs[:subject] + all_target_covs[
                        subject + 1:],
                    alpha=alpha_now,
                    subject_weights=False,
                    c1_vs_c2_only=True)
            # find the testing variance
            c1_test_var = np.diag(rcsp_tlw_filters.T @ c1_test @ rcsp_tlw_filters)
            c2_test_var = np.diag(rcsp_tlw_filters.T @ c2_test @ rcsp_tlw_filters)
            var_ratios = np.clip(c1_test_var[~np.isclose(c2_test_var, 0)] / c2_test_var[~np.isclose(c2_test_var,0)], 0, None)
            result[fold_now, alpha_it,-len(var_ratios):] = var_ratios
    return result

for subject in range(len(target_cov)):
    if subject >= 10:
        subject_name = subject + 1
    else:
        subject_name = subject
    print('Running cross-validation for subject {}'.format(subject_name))
    cv_result_all = cv_rcsp_tlw(alpha, N_folds, subject, target_cov,
            contrast_cov)
    cv_result = cv_result_all[..., -1]
    # find the largest alpha that is within the optimum cross-validation
    # result - 1 standard error
    threshold = np.max(cv_result.mean(0) - cv_result.std(0)/np.sqrt(N_folds))
    best_alpha_idx = np.argmax(alpha[cv_result.mean(0) >= threshold])
    best_alpha=alpha[best_alpha_idx]
    # plot the cross-validation curve 
    fig = plt.figure()
    plt.semilogx(alpha, cv_result.mean(0), 'k-', label='mean')
    plt.semilogx(alpha, cv_result.mean(0) + cv_result.std(0)/np.sqrt(N_folds),
            'r-', lw=0.5, label='s.e.')
    plt.semilogx(alpha, cv_result.mean(0) - cv_result.std(0)/np.sqrt(N_folds),
            'r-', lw=0.5)
    plt.gca().set_xlabel('alpha')
    plt.gca().set_ylabel('test variance ratio')
    plt.axhline(threshold, c='k')
    plt.axvline(best_alpha, c='r')
    fig.legend(loc='upper right')
    plt.gca().set_title('RCSP with transfer learning, subject {}'.format(
        subject_name + 1))
    fig.tight_layout()
    fig.savefig(os.path.join(result_folder, 'S{:02d}'.format(subject_name + 1),
        'cv_result.pdf'), format='pdf')
    # train final filters for that subject
    all_target_covs = [np.mean(c, -1) for c in target_cov]
    rcsp_tlw_ratios, rcsp_tlw_filters = rcsp_tlw.rcsp_tlw(
                        np.mean(target_cov[subject], -1),
                        np.mean(contrast_cov[subject], -1),
                        target_cov=all_target_covs[subject],
                        source_covs=all_target_covs[:subject] + all_target_covs[
                            subject + 1:],
                        alpha=best_alpha,
                        subject_weights=False,
                        c1_vs_c2_only=True)
    # take the cross-validation SNR as parameter to plot and sort the
    # components according to the cross-validation result
    c1 = np.mean(target_cov[subject], -1)
    c2 = np.mean(contrast_cov[subject], -1)
    # find the variance ratio (SNR) of the resulting spatial filters
    c1_var = np.diag(rcsp_tlw_filters.T @ c1 @ rcsp_tlw_filters)
    c2_var = np.diag(rcsp_tlw_filters.T @ c2 @ rcsp_tlw_filters)
    # remove component with zero variance
    rcsp_tlw_ratios = np.clip(c1_var[~np.isclose(c2_var, 0)] /
                              c2_var[~np.isclose(c2_var, 0)], 0, None)
    rcsp_tlw_filters = rcsp_tlw_filters[:, ~np.isclose(c2_var, 0)]
    component_order = np.argsort(rcsp_tlw_ratios)
    rcsp_tlw_filters = rcsp_tlw_filters[:, component_order]
    rcsp_tlw_ratios = rcsp_tlw_ratios[component_order]
    # get the spatial patterns
    rcsp_tlw_patterns = scipy.linalg.solve(
            rcsp_tlw_filters.T.dot(all_target_covs[subject]).dot(
                rcsp_tlw_filters),
            rcsp_tlw_filters.T.dot(all_target_covs[subject]))
    ### normalize the patterns such that Cz is always positive
    rcsp_tlw_patterns *= np.sign(rcsp_tlw_patterns[
        :,np.asarray(channames)=='CZ'])
    ## save SSD eigenvalues, filters and patterns in a.npz
    np.savez(os.path.join(result_folder, 'S{:02d}'.format(subject_name + 1),
        'rcsp_tlw.npz'),
            rcsp_tlw_ratios = rcsp_tlw_ratios,
            rcsp_tlw_filters = rcsp_tlw_filters,
            rcsp_tlw_patterns = rcsp_tlw_patterns,
            )
    ## apply SSD to FFT
    F_SSD = np.tensordot(rcsp_tlw_filters, F[subject], axes=(0,0))
    ## average across trials
    F_SSD_mean = (np.abs(F_SSD)**2).mean(-1)
    F_mean = (np.abs(F[subject])**2).mean(-1)
    ######################################
    # plot the resulting EV and patterns #
    ######################################
    # prepare the SSD components scalp maps
    potmaps = [meet.sphere.potMap(chancoords, pat_now,
        projection='stereographic') for pat_now in rcsp_tlw_patterns]
    # define the height ratios of the subplot rows
    h1 = 1
    h2 = 1.5
    h3 = 1
    #
    fig = plt.figure(figsize=(5.51181,5))
    gs = mpl.gridspec.GridSpec(3,1, height_ratios = [h1,h2,h3])
    SNNR_ax = fig.add_subplot(gs[0,:])
    SNNR_ax.plot(range(1,len(rcsp_tlw_ratios) + 1),
            (10*np.log10(rcsp_tlw_ratios[::-1])), 'ko-', lw=2, markersize=5)
    SNNR_ax.scatter([1], 10*np.log10(rcsp_tlw_ratios[-1]), c=color1,
            s=60, zorder=1000)
    SNNR_ax.scatter([2], 10*np.log10(rcsp_tlw_ratios[-2]), c=color2,
            s=60, zorder=1000)
    SNNR_ax.scatter([3], 10*np.log10(rcsp_tlw_ratios[-3]), c=color3,
            s=60, zorder=1000)
    SNNR_ax.set_xlim([0.5, len(rcsp_tlw_ratios)])
    SNNR_ax.set_xticks(np.r_[1,range(5, len(rcsp_tlw_ratios) + 1, 5)])
    SNNR_ax.set_ylabel('SNNR (dB)')
    SNNR_ax.set_xlabel('component (index)')
    SNNR_ax.set_title('SNNR of SSD components')
    # plot the four spatial patterns
    gs2 = mpl.gridspec.GridSpecFromSubplotSpec(2,3, gs[1,:],
            height_ratios = [1,0.1])
    head_ax = []
    pc = []
    for i, pat in enumerate(potmaps[::-1][:3]):
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
        head_ax[-1].set_title(r'\textbf{%d}' % (i + 1) +'\n'+
                '($\mathrm{%.2f dB}$)' % ((10*np.log10(
                    rcsp_tlw_ratios[-(i + 1)]))))
        meet.sphere.addHead(head_ax[-1], ec=colors[i], zorder=1000, lw=3)
    head_ax[0].set_ylim([-1.1,1.2])
    head_ax[0].set_xlim([-1.5,1.5])

    # add a colorbar
    cbar_ax = fig.add_subplot(gs2[1,:])
    cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
            label='amplitude (a.u.)', ticks=[-1,0,1])
    cbar.ax.set_xticklabels(['-', '0', '+'])
    cbar.ax.axvline(0, c='w', lw=2)

    spect_ax = fig.add_subplot(gs[2,:])

    [spect_ax.plot(f,
        10*np.log10(comp/comp[...,contrast_mask != target_mask].mean(-1)),
            c=colors[i], lw=2) for i, comp in enumerate(F_SSD_mean[::-1][:3])]
    [spect_ax.plot(f,
        10*np.log10(comp/comp[...,contrast_mask != target_mask].mean(-1)),
            c='k', alpha=0.1, lw=0.5) for i, comp in enumerate(F_mean[:32])]

    spect_ax.set_xlim([0.5, 4])
    spect_ax.set_ylim([-10, 15])
    spect_ax.axhline(0, c='k', lw=1)
    spect_ax.set_xlabel('frequency (Hz)')
    spect_ax.set_ylabel('SNNR (dB)')
    spect_ax.set_title('normalized spectrum')

    spect_ax.axvline(snareFreq, color='b', zorder=0, lw=1)
    spect_ax.axvline(2*snareFreq, color='b', zorder=0, lw=1)
    spect_ax.axvline(wdBlkFreq, color='r', zorder=0, lw=1)
    spect_ax.axvline(2*wdBlkFreq, color='k', zorder=0, lw=1)

    spect_ax.axvline(4*snareFreq, color='b', zorder=0, lw=1, ls=':')
    spect_ax.axvline(5*snareFreq, color='b', zorder=0, lw=1, ls=':')
    spect_ax.axvline(3*wdBlkFreq, color='r', zorder=0, lw=1, ls=':')
    spect_ax.axvline(4*wdBlkFreq, color='k', zorder=0, lw=1, ls=':')

    gs.tight_layout(fig, pad=0.2, h_pad=1.0)
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

    fig.align_ylabels([SNNR_ax, spect_ax])
    fig.savefig(os.path.join(result_folder, 'S{:02d}'.format(
        subject_name + 1), 'FFTSSD_patterns.pdf'))
    fig.savefig(os.path.join(result_folder, 'S{:02d}'.format(
        subject_name + 1), 'FFTSSD_patterns.png'))
1/0

######################################################################
### This is all some old stuff and needs to be updated ###############
######################################################################

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

"""
######################################
# plot the resulting EV and patterns #
######################################

# prepare the SSD components scalp maps
potmaps = [meet.sphere.potMap(chancoords, pat_now,
    projection='stereographic') for pat_now in SSD_patterns]

# define the height ratios of the subplot rows
h1 = 1
h2 = 1.5
h3 = 1

fig = plt.figure(figsize=(3.54331,5))
gs = mpl.gridspec.GridSpec(3,1, height_ratios = [h1,h2,h3])

SNNR_ax = fig.add_subplot(gs[0,:])
SNNR_ax.plot(range(1,len(SSD_eigvals) + 1), 10*np.log10(SSD_eigvals), 'ko-', lw=2,
        markersize=5)
SNNR_ax.scatter([1], 10*np.log10(SSD_eigvals[0]), c=color1, s=60, zorder=1000)
SNNR_ax.scatter([2], 10*np.log10(SSD_eigvals[1]), c=color2, s=60, zorder=1000)
#SNNR_ax.scatter([3], 10*np.log10(SSD_eigvals[2]), c=color3, s=60, zorder=1000)
#SNNR_ax.scatter([4], 10*np.log10(SSD_eigvals[3]), c=color4, s=60, zorder=1000)
#SNNR_ax.axhline(0, c='k', lw=1)
SNNR_ax.set_xlim([0.5, len(SSD_eigvals)])
SNNR_ax.set_xticks(np.r_[1,range(5, len(SSD_eigvals) + 1, 5)])
SNNR_ax.set_ylabel('SNNR (dB)')
SNNR_ax.set_xlabel('component (index)')
SNNR_ax.set_title('SNNR of SSD components')

# plot the four spatial patterns
gs2 = mpl.gridspec.GridSpecFromSubplotSpec(2,2, gs[1,:],
        height_ratios = [1,0.1])
head_ax = []
pc = []
for i, pat in enumerate(potmaps[:2]):
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
    head_ax[-1].set_title(r'\textbf{%d}' % (i + 1) +'\n'+
            r'($\mathrm{SNR=%.2f\ dB}$)' % (10*np.log10(SSD_eigvals[i])))
    meet.sphere.addHead(head_ax[-1], ec=colors[i], zorder=1000, lw=3)
head_ax[0].set_ylim([-1.1,1.2])
head_ax[0].set_xlim([-1.5,1.5])

# add a colorbar
cbar_ax = fig.add_subplot(gs2[1,:])
cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
        label='amplitude (a.u.)', ticks=[-1,0,1])
cbar.ax.set_xticklabels(['-', '0', '+'])
cbar.ax.axvline(0, c='w', lw=2)

spect_ax = fig.add_subplot(gs[2,:])

>>>>>>> b5166906b62121da9ddbd3e3103fe8655f9bc103
[spect_ax.plot(f,
    10*np.log10(np.mean([t/t[...,contrast_mask != target_mask].mean(
        -1)[:,np.newaxis]
        for t in F_SSD_mean[::-1]], 0)[i]),
        c=colors[i], lw=2) for i in range(3)]
[spect_ax.plot(f,
    10*np.log10(np.mean([t/t[...,contrast_mask != target_mask].mean(
        -1)[:,np.newaxis]
        for t in F_mean], 0)[i]),
        c='k', alpha=0.1, lw=0.5) for i in range(32)]
"""
