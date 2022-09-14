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
import pdb
from tqdm import tqdm, trange

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
            target_cov.append(0.5 * (snare_target_cov / snare_norm +
                                     wdBlk_target_cov / wdBlk_norm))
            contrast_cov.append(0.5 * (snare_contrast_cov / snare_norm +
                                       wdBlk_contrast_cov / wdBlk_norm))
            F.append(fi['F_silence'])
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
import mtCSP

# cross validate the regularization parameter
lam1 = 0
lam2_cand = [0.5]
SNNR_lam = []
v_norm = []
get_filters = 10

try:
    with np.load(os.path.join(result_folder, 'mtCSP.npz'), 'r') as fi:
             subject_filters=fi['subject_filters']
except:
    # get whitening matrix for the average target covariance matrix
    target_cov_avg = np.mean([c.mean(-1) for c in target_cov], 0)
    rank = np.linalg.matrix_rank(target_cov_avg)
    eigval, eigvect = scipy.linalg.eigh(target_cov_avg)
    W = eigvect[:,-rank:]/np.sqrt(eigval[-rank:])
    for lam2 in lam2_cand:
        print('Lambda 2 is ',lam2)
        for it in trange(get_filters):
            if it == 0:
                quot_now, filters_now = mtCSP.maximize_mtCSP(
                        [W.T @ c.mean(-1) @ W for c in target_cov],
                        [W.T @ c.mean(-1) @ W for c in contrast_cov],
                        lam1,
                        lam2,
                        iterations=20)
                quot = [quot_now]
                all_filters = filters_now.reshape(-1, 1)
            else: #force other filters to be orthogonal
                quot_now, filters_now = mtCSP.maximize_mtCSP(
                        [W.T @ c.mean(-1) @ W for c in target_cov],
                        [W.T @ c.mean(-1) @ W for c in contrast_cov],
                        lam1,
                        lam2,
                        old_W = all_filters,
                        iterations=20)
                quot.append(quot_now)
                all_filters = np.hstack([all_filters, filters_now.reshape(-1, 1)])
        # transform the filters
        all_filters = np.vstack([W @ all_filters[i * rank : (i + 1) * rank]
            for i in range(len(target_cov) + 1)])
        # calculate the composite filters for every subject
        subject_filters = [all_filters[:N_channels] +
                           all_filters[(i + 1) * N_channels:(i + 2) * N_channels]
                           for i in range(len(target_cov))] #(N_subjects,N_channels,get_filters)

        # calculate the SNNR for every component and subject
        SNNR_per_subject = np.array([
            np.diag((filt.T @ target_now.mean(-1) @ filt) /
                    (filt.T @ contrast_now.mean(-1) @ filt))
            for (filt, target_now, contrast_now) in
            zip(subject_filters, target_cov, contrast_cov)])

        SNNR = SNNR_per_subject.mean(0)
        #
        # # check for optimal lambda2
        # # calculate vnorm, append SNNR for l curve
        # v_norm.append(np.sum((all_filters[N_channels:,0])**2)) #only individual filter
        # SNNR_lam.append(SNNR)
        # #plot filters over channel (should be similar enough)
        # plt.figure()
        # for s in range(N_subjects-1):
        #     f_now = subject_filters[s][:,0] #checking first filter only
        #     plt.plot(channames, (f_now-np.mean(f_now))/np.std(f_now))
        # plt.savefig(os.path.join(result_folder,'mtCSP_filters_lam{}.pdf'.format(lam2)))

    # # plot SNNR and ||v||2 for first filter (ERD)
    # plt.figure()
    # l = np.log([s[0] for s in SNNR_lam])
    # plt.plot([l[0], l[len(l)-1]],[np.log(v_norm)[0],np.log(v_norm)[-1]], c='k', alpha=0.5) #straight helpline
    # plt.plot(l, np.log(v_norm), alpha=0.5) # connecting points
    # for i,k in enumerate(lam2_cand):
    #     plt.scatter(l[i], np.log(v_norm[i]), label=str(round(k,2))) #one point per lambda (for labels)
    # plt.legend()
    # plt.xlabel('log (SNNR)')
    # plt.ylabel('log (vnorm)')
    # plt.title('mtCSP, component 1') #steepest gives optimal value
    # plt.savefig(os.path.join(result_folder,'mtCSP_Lcurve_1.pdf'))
    # np.save(os.path.join(result_folder, 'mtCSP_vnorm.npy'), v_norm)
    # np.save(os.path.join(result_folder, 'mtCSP_SNNR.npy'), SNNR_lam)


np.savez(os.path.join(result_folder, 'mtCSP.npz'),
         SNNR_per_subject=SNNR_per_subject,
         subject_filters=subject_filters)

# calculate the indiuvidual F_SSD for listen, silence and both
F_SSD = [np.tensordot(subject_filter_now, F_now, axes=(0, 0))
         for subject_filter_now, F_now in zip(subject_filters, F)]
F_SSD_mean = [(np.abs(F_SSD_now)**2).mean(-1) for F_SSD_now in F_SSD]
F_mean = [(np.abs(F_now)**2).mean(-1) for F_now in F]

# plot the three best components #
######################################
# plot the resulting EV and patterns #
######################################
# calculate the spatial pattern from the global pattern (first 32 coefficients) and the average
# of the target covariance matrix

global_filters = all_filters[:N_channels] #this wont work if mtCSP.npz already existed before!
global_filters = subject_filters[0][:N_channels] #should be equivalent, for each subject also includes the global one
#global_filters = np.mean(subject_filters, 0)
global_target_covariance = np.mean([c.mean(-1) for c in target_cov], 0)

# get the spatial patterns
global_patterns = scipy.linalg.solve(
        global_filters.T.dot(global_target_covariance).dot(global_filters),
        global_filters.T.dot(global_target_covariance))

### normalize the patterns such that Cz is always positive and that the maximum absolute
# value is +1
global_patterns *= np.sign(global_patterns[:,np.asarray(channames)=='CZ'])
global_patterns /= np.abs(global_patterns).max(-1)[:,np.newaxis]

# prepare the SSD components scalp maps
potmaps = [meet.sphere.potMap(chancoords, pat_now,
    projection='stereographic') for pat_now in global_patterns]

# define the height ratios of the subplot rows
h1 = 0.5
h2 = 1.5
h3 = 1

fig = plt.figure(figsize=(3.54331,5))
gs = mpl.gridspec.GridSpec(3,1, height_ratios = [h1,h2,h3])

SNNR_ax = fig.add_subplot(gs[0,:])
SNNR_ax.plot(range(1,len(SNNR) + 1), 10*np.log10(SNNR), 'ko-', lw=2,
        markersize=5)
SNNR_ax.scatter([1], 10*np.log10(SNNR[0]), c=color1, s=60, zorder=1000)
SNNR_ax.scatter([2], 10*np.log10(SNNR[1]), c=color2, s=60, zorder=1000)
SNNR_ax.scatter([3], 10*np.log10(SNNR[2]), c=color3, s=60, zorder=1000)
#SNNR_ax.scatter([4], 10*np.log10(SNNR[3]), c=color4, s=60, zorder=1000)
#SNNR_ax.axhline(0, c='k', lw=1)
SNNR_ax.set_xlim([0.5, len(SNNR) + 0.5])
SNNR_ax.set_xticks(np.r_[1,range(2, len(SNNR) + 1, 1)])
SNNR_ax.set_ylabel('SNNR (dB)')
SNNR_ax.set_xlabel('component (index)')
SNNR_ax.set_title('SNNR of SSD components')

# plot the four spatial patterns
gs2 = mpl.gridspec.GridSpecFromSubplotSpec(2,3, gs[1,:],
        height_ratios = [1,0.1])
head_ax = []
pc = []
for i, pat in enumerate(potmaps[:3]):
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
            r'($\mathrm{SNR=%.2f\ dB}$)' % (10*np.log10(SNNR[i])))
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
[spect_ax.plot(f, 10*np.log10(
    np.mean([comp[i]/comp[i,contrast_mask != target_mask].mean(-1) for comp in F_SSD_mean], 0)),
    c=colors[i], lw=2) for i in range(3)]
[spect_ax.plot(f, 10*np.log10(
    np.mean([comp[i]/comp[i,contrast_mask != target_mask].mean(-1) for comp in F_mean], 0)),
    c='k', alpha=0.1, lw=0.5) for i in range(32)]

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
fig.savefig(os.path.join(result_folder, 'FFTSSD_patterns_mtCSP.pdf'))
fig.savefig(os.path.join(result_folder, 'FFTSSD_patterns_mtCSP.png'))

#"""
## The cross-validation is crazily slow, so - for now - we leave it out here
#
#
#lam2 = 10**np.linspace(-4, 4, 9)
#lam2 = [100]
#N_folds = 1
#
#def get_folds(N_trials, N_folds):
#    fold_start_stop = np.linspace(0, N_trials, N_folds + 1).astype(int)
#    random_idx = np.argsort(np.random.randn(N_trials))
#    folds = [random_idx[fold_start_stop[i]:fold_start_stop[i+1]]
#             for i in range(N_folds)]
#    return folds
#
#def get_train_test_cov(cov, folds, fold_now):
#        train_idx = np.hstack(
#                [folds[j] for j in list(range(fold_now)) +
#                                   list(range(fold_now + 1, N_folds))])
#        test_idx = folds[fold_now]
#        train_cov = np.mean(cov[..., train_idx], -1)
#        test_cov = np.mean(cov[..., test_idx], -1)
#        return train_cov, test_cov
#
#def cv_rcsp_tlw(lam1, lam2, N_folds, target_cov, contrast_cov):
#    n_subjects = len(target_cov)
#    n_channels = target_cov[0].shape[0]
#    # get the nummber of trials for every subject
#    folds = [get_folds(target_cov_now.shape[-1], N_folds) for target_cov_now in target_cov]
#    # initialize empty results array
#    result = np.zeros([len(folds), len(lam1), len(lam2)])
#    # initialize a progress bar
#    pbar = tqdm(total=N_folds * len(lam1) * len(lam2))
#    for fold_now in range(N_folds):
#        # get the covariance matrices for that fold
#        c1_train, c1_test = zip(*[get_train_test_cov(target_cov_subject, folds_subject, fold_now)
#            for target_cov_subject, folds_subject in zip(target_cov,  folds)])
#        c2_train, c2_test = zip(*[get_train_test_cov(contrast_cov_subject, folds_subject, fold_now)
#            for contrast_cov_subject, folds_subject in zip(contrast_cov,  folds)])
#        for lam1_it, lam1_now in enumerate(lam1):
#            for lam2_it, lam2_now in enumerate(lam2):
#                train_quot, all_filters = mtCSP.maximize_mtCSP(
#                        c1_train,
#                        c2_train,
#                        lam1_now,
#                        lam2_now,
#                        iterations=3)
#                w = [all_filters[:n_channels] +
#                     all_filters[(i + 1) * n_channels:(i + 2) * n_channels]
#                     for i in range(n_subjects)]
#                test_quot = np.mean([(w_now @ c1_now @ w_now) / (w_now @ c2_now @ w_now)
#                                    for (w_now, c1_now, c2_now) in zip(w, c1_test, c2_test)])
#                result[fold_now, lam1_it, lam2_it] = test_quot
#                # update the progress bar
#                pbar.update()
#    pbar.close()
#    return result
#
#print('Running cross-validation')
#cv_result1 = cv_rcsp_tlw(lam1, lam2, N_folds, target_cov, contrast_cov)
#"""
