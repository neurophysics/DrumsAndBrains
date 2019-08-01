import argparse
import numpy as np
import scipy
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os.path
import helper_functions
import meet
from tqdm import trange

parser = argparse.ArgumentParser(description='Calculate PCO')

parser.add_argument('N_SSD', type=int, nargs='?', default=6,
        help='the number of SSD filters to use')
parser.add_argument('result_folder', type=str, default='./Results/',
        help='the folder to store all results', nargs='?')
parser.add_argument('--normalize', type=int, default=1,
        help='whether individual subject data should be normalized')
parser.add_argument('--normSSD', type=int, default=1,
        help='whether the normalized SSD should be used')
parser.add_argument('--absdev', type=int, default=1,
        help='whether to use absolute errors')
parser.add_argument('--rank', type=int, default=0,
        help='whether data from individual subjects is rank-normalized')
args = parser.parse_args()

args.normSSD = bool(args.normSSD)
args.absdev = bool(args.absdev)
args.rank = bool(args.rank)

mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 10

cmap = 'plasma'
color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

s_rate = 1000 # sampling rate of the EEG

N_subjects = 21
cov_thresh = 2500

# calculate the SSD from all subjects
# read the channel names
channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

N_channels = len(channames)

# load the SSD results
with np.load(os.path.join(args.result_folder, 'SSD_norm_%s.npz' % (
    args.normSSD)),
        'r') as f:
    ssd_eigvals = f['ssd_eigvals']
    ssd_filter = f['ssd_filter']

snareInlier = []
wdBlkInlier = []
snareSilence_covs = []
wdBlkSilence_covs = []
snareSilence_rec_covs = []
wdBlkSilence_rec_covs = []
snareSilence_rec_ssd_covs = []
wdBlkSilence_rec_ssd_covs = []
snare_deviation = []
wdBlk_deviation = []

# read the oscillatory data from the silence period
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/prepared_filterdata.npz', 'r') as f:
            snareInlier.append(f['snareInlier'])
            wdBlkInlier.append(f['wdBlkInlier'])
            ##################
            # get snare data #
            ##################
            snareSilenceData = f['snareSilenceData']
            snareSilenceData_rec = f['snareSilenceData_rec']
            # filter with the SSD filter
            snareSilenceData_rec_ssd = np.tensordot(
                    ssd_filter[:,:args.N_SSD], snareSilenceData_rec,
                    axes=(0,0))
            # get covariance matrices of individual trials
            snareSilence_covs.append(
                    np.einsum('ijk, ljk->ilk',
                        snareSilenceData, snareSilenceData)/
                    snareSilenceData.shape[1])
            snareSilence_rec_covs.append(
                    np.einsum('ijk, ljk->ilk',
                        snareSilenceData_rec, snareSilenceData_rec)/
                    snareSilenceData.shape[1])
            snareSilence_rec_ssd_covs.append(
                    np.einsum('ijk, ljk->ilk',
                        snareSilenceData_rec_ssd, snareSilenceData_rec_ssd)/
                    snareSilenceData.shape[1])
            ##################
            # get wdBlk data #
            ##################
            wdBlkSilenceData = f['wdBlkSilenceData']
            wdBlkSilenceData_rec = f['wdBlkSilenceData_rec']
            # filter with the SSD filter
            wdBlkSilenceData_rec_ssd = np.tensordot(
                    ssd_filter[:,:args.N_SSD], wdBlkSilenceData_rec,
                    axes=(0,0))
            # get covariance matrices of individual trials
            wdBlkSilence_covs.append(
                    np.einsum('ijk, ljk->ilk',
                        wdBlkSilenceData, wdBlkSilenceData)/
                    wdBlkSilenceData.shape[1])
            wdBlkSilence_rec_covs.append(
                    np.einsum('ijk, ljk->ilk',
                        wdBlkSilenceData_rec, wdBlkSilenceData_rec)/
                    wdBlkSilenceData.shape[1])
            wdBlkSilence_rec_ssd_covs.append(
                    np.einsum('ijk, ljk->ilk',
                        wdBlkSilenceData_rec_ssd, wdBlkSilenceData_rec_ssd)/
                    wdBlkSilenceData.shape[1])
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

# read the behavioural data
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/behavioural_results.npz', 'r') as f:
            snare_deviation.append(f['snare_deviation'])
            wdBlk_deviation.append(f['wdBlk_deviation'])
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

# only keep those trials where both, behaviour and EEG, were measured
# correctly
snare_deviation = [d[i] for d,i in zip(snare_deviation, snareInlier)]
wdBlk_deviation = [d[i] for d,i in zip(wdBlk_deviation, wdBlkInlier)]

(snareSilence_covs, snareSilence_rec_covs, snareSilence_rec_ssd_covs,
        snare_deviation) = zip(*[
    (p[...,np.isfinite(d)], q[...,np.isfinite(d)], r[...,np.isfinite(d)],
        d[np.isfinite(d)])
    for p, q, r, d in zip(snareSilence_covs, snareSilence_rec_covs,
        snareSilence_rec_ssd_covs, snare_deviation)])

(wdBlkSilence_covs, wdBlkSilence_rec_covs, wdBlkSilence_rec_ssd_covs,
        wdBlk_deviation) = zip(*[
    (p[...,np.isfinite(d)], q[...,np.isfinite(d)], r[...,np.isfinite(d)],
        d[np.isfinite(d)])
    for p, q, r, d in zip(wdBlkSilence_covs, wdBlkSilence_rec_covs,
        wdBlkSilence_rec_ssd_covs, wdBlk_deviation)])

###################
# reject outliers #
###################
snare_Inlier  = [np.all([
    p[range(N_channels), range(N_channels)].sum(0) < cov_thresh,
    np.abs(d) < 0.4], axis = 0) for p,d in zip(
            snareSilence_covs, snare_deviation)]
(snareSilence_covs, snareSilence_rec_covs, snareSilence_rec_ssd_covs,
        snare_deviation) = zip(*[
    (p[...,I], q[...,I], r[...,I], d[I])
    for p, q, r, d, I in zip(snareSilence_covs,
        snareSilence_rec_covs, snareSilence_rec_ssd_covs,
        snare_deviation, snare_Inlier)])
wdBlk_Inlier  = [np.all([
    p[range(N_channels), range(N_channels)].sum(0) < cov_thresh,
    np.abs(d) < 0.6], axis = 0) for p,d in zip(
            wdBlkSilence_covs, wdBlk_deviation)]
(wdBlkSilence_covs, wdBlkSilence_rec_covs, wdBlkSilence_rec_ssd_covs,
        wdBlk_deviation) = zip(*[
    (p[...,I], q[...,I], r[...,I], d[I])
    for p, q, r, d, I in zip(wdBlkSilence_covs,
        wdBlkSilence_rec_covs, wdBlkSilence_rec_ssd_covs,
        wdBlk_deviation, wdBlk_Inlier)])

if args.normalize:
    (snareSilence_covs, snareSilence_rec_covs,
            snareSilence_rec_ssd_covs) = zip(*[(
        p/np.trace(p.mean(-1)),
        q/np.trace(p.mean(-1)),
        r/np.trace(p.mean(-1)))
        #p/p[range(N_channels), range(N_channels)].sum(0),
        #q/p[range(N_channels), range(N_channels)].sum(0),
        #r/p[range(N_channels), range(N_channels)].sum(0))
        for p,q,r in zip(snareSilence_covs, snareSilence_rec_covs,
            snareSilence_rec_ssd_covs)])
    (wdBlkSilence_covs, wdBlkSilence_rec_covs,
            wdBlkSilence_rec_ssd_covs) = zip(*[(
        p/np.trace(p.mean(-1)),
        q/np.trace(p.mean(-1)),
        r/np.trace(p.mean(-1)))
        #p/p[range(N_channels), range(N_channels)].sum(0),
        #q/p[range(N_channels), range(N_channels)].sum(0),
        #r/p[range(N_channels), range(N_channels)].sum(0))
        for p,q,r in zip(wdBlkSilence_covs, wdBlkSilence_rec_covs,
            wdBlkSilence_rec_ssd_covs)])

if args.absdev:
    snare_deviation = [np.abs(d) for d in snare_deviation]
    wdBlk_deviation = [np.abs(d) for d in wdBlk_deviation]

if args.rank:
    snare_deviation = [d.argsort().argsort()/(len(d) - 1.)
            for d in snare_deviation]
    wdBlk_deviation = [d.argsort().argsort()/(len(d) - 1.)
            for d in wdBlk_deviation]

snare_mean = np.hstack(snare_deviation).mean()
snare_std = np.hstack(snare_deviation).std()
snare_deviation = [(d - snare_mean)/snare_mean for d in snare_deviation]

wdBlk_mean = np.hstack(wdBlk_deviation).mean()
wdBlk_std = np.hstack(wdBlk_deviation).std()
wdBlk_deviation = [(d - wdBlk_mean)/wdBlk_mean for d in wdBlk_deviation]

# concatenate all the data
snareSilence_covs = np.dstack(snareSilence_covs)
wdBlkSilence_covs = np.dstack(wdBlkSilence_covs)
snareSilence_rec_covs = np.dstack(snareSilence_rec_covs)
wdBlkSilence_rec_covs = np.dstack(wdBlkSilence_rec_covs)
snareSilence_rec_ssd_covs = np.dstack(snareSilence_rec_ssd_covs)
wdBlkSilence_rec_ssd_covs = np.dstack(wdBlkSilence_rec_ssd_covs)

# filter snare_ and wdBlkSilence_covs
snareSilence_ssd_covs = np.tensordot(
        ssd_filter[:,:args.N_SSD], np.tensordot(
            ssd_filter[:,:args.N_SSD], snareSilence_covs, axes=(0,0)),
        axes=(0,1))
wdBlkSilence_ssd_covs = np.tensordot(
        ssd_filter[:,:args.N_SSD], np.tensordot(
            ssd_filter[:,:args.N_SSD], wdBlkSilence_covs, axes=(0,0)),
        axes=(0,1))

bothSilence_covs = np.dstack([snareSilence_covs,
    wdBlkSilence_covs])
bothSilence_rec_covs = np.dstack([snareSilence_rec_covs,
    wdBlkSilence_rec_covs])
bothSilence_rec_ssd_covs = np.dstack([snareSilence_rec_ssd_covs,
    wdBlkSilence_rec_ssd_covs])
bothSilence_ssd_covs = np.dstack([snareSilence_ssd_covs,
    wdBlkSilence_ssd_covs])
both_deviation = snare_deviation + wdBlk_deviation

bothSilence_ssd_covs = np.tensordot(
        ssd_filter[:,:args.N_SSD], np.tensordot(
            ssd_filter[:,:args.N_SSD], bothSilence_covs, axes=(0,0)),
        axes=(0,1))

def calculate_SPoC(covs, a):
    a = (a - a.mean())/a.std()
    target = (covs * a).mean(-1)
    contrast = covs.mean(-1)
    eigvals, eigvect = scipy.linalg.eigh(target, contrast)
    return eigvals, eigvect

def calculate_SPoCb(cova, covb, a):
    #a = (a - a.mean())/a.std()
    target = (cova*a).mean(-1)
    contrast = (covb*a).mean(-1)
    eigvals, eigvect = scipy.linalg.eigh(target, contrast)
    return eigvals, eigvect

snareSilence_ssd_covs = np.tensordot(
        np.tensordot(ssd_filter[:,:args.N_SSD],
            snareSilence_covs, axes=(0,0)), ssd_filter[:,:args.N_SSD],
        axes=(1,0)).swapaxes(1,2)
wdBlkSilence_ssd_covs = np.tensordot(
        np.tensordot(ssd_filter[:,:args.N_SSD],
            wdBlkSilence_covs, axes=(0,0)), ssd_filter[:,:args.N_SSD],
        axes=(1,0)).swapaxes(1,2)

# calculate the PCO filter for snare data
snare_eigvals, snare_SPoC_filt = calculate_SPoC(
        snareSilence_rec_ssd_covs,
        np.hstack(snare_deviation))
snare_eigvals_boot = np.array([calculate_SPoC(
    snareSilence_rec_ssd_covs,
    np.random.permutation(np.hstack(snare_deviation))
    #np.hstack([np.random.permutation(d) for d in snare_deviation])
    )[0]
    for _ in trange(10000)])

# calculate the PCO filter for wdBlk data
wdBlk_eigvals, wdBlk_SPoC_filt = calculate_SPoC(
        wdBlkSilence_rec_ssd_covs,
        np.hstack(wdBlk_deviation))
wdBlk_eigvals_boot = np.array([calculate_SPoC(
    wdBlkSilence_rec_ssd_covs,
    np.random.permutation(np.hstack(wdBlk_deviation))
    #np.hstack([np.random.permutation(d) for d in wdBlk_deviation])
    )[0]
    for _ in trange(10000)])

# calculate the PCO filter for both data
both_eigvals, both_SPoC_filt = calculate_SPoC(
        bothSilence_rec_ssd_covs,
        np.hstack(both_deviation))
both_eigvals_boot = np.array([calculate_SPoC(
    bothSilence_rec_ssd_covs,
    np.random.permutation(np.hstack(both_deviation))
    #np.hstack([np.random.permutation(d) for d in both_deviation])
    )[0]
    for _ in trange(10000)])

snare_corr = np.array([np.corrcoef(
    np.tensordot(np.tensordot(p, snareSilence_rec_ssd_covs,
        axes=(0,0)), p, axes=(0,0)), np.hstack(snare_deviation))[0,1]
    for p in snare_SPoC_filt.T])

wdBlk_corr = np.array([np.corrcoef(
    np.tensordot(np.tensordot(p, wdBlkSilence_rec_ssd_covs,
        axes=(0,0)), p, axes=(0,0)), np.hstack(wdBlk_deviation))[0,1]
    for p in wdBlk_SPoC_filt.T])

both_corr = np.array([np.corrcoef(
    np.tensordot(np.tensordot(p, wdBlkSilence_rec_ssd_covs,
        axes=(0,0)), p, axes=(0,0)), np.hstack(wdBlk_deviation))[0,1]
    for p in wdBlk_SPoC_filt.T])

snare_p = np.where(snare_corr<0,
        (snare_eigvals_boot[:,0][:,np.newaxis]<= snare_eigvals).mean(0),
        (snare_eigvals_boot[:,-1][:,np.newaxis]>= snare_eigvals).mean(0))
wdBlk_p = np.where(wdBlk_corr<0,
        (wdBlk_eigvals_boot[:,0][:,np.newaxis]<= wdBlk_eigvals).mean(0),
        (wdBlk_eigvals_boot[:,-1][:,np.newaxis]>= wdBlk_eigvals).mean(0))
both_p = np.where(both_corr<0,
        (both_eigvals_boot[:,0][:,np.newaxis]<= both_eigvals).mean(0),
        (both_eigvals_boot[:,-1][:,np.newaxis]>= both_eigvals).mean(0))

#get and calculate the spatial patterns
snare_filter = ssd_filter[:,:args.N_SSD].dot(snare_SPoC_filt)
wdBlk_filter = ssd_filter[:,:args.N_SSD].dot(wdBlk_SPoC_filt)
both_filter = ssd_filter[:,:args.N_SSD].dot(both_SPoC_filt)

snare_pattern = scipy.linalg.solve(
        snare_filter.T.dot(snareSilence_rec_covs.mean(-1)).dot(snare_filter),
        snare_filter.T.dot(snareSilence_rec_covs.mean(-1)))
wdBlk_pattern = scipy.linalg.solve(
        wdBlk_filter.T.dot(wdBlkSilence_rec_covs.mean(-1)).dot(wdBlk_filter),
        wdBlk_filter.T.dot(wdBlkSilence_rec_covs.mean(-1)))
both_pattern = scipy.linalg.solve(
        both_filter.T.dot(bothSilence_rec_covs.mean(-1)).dot(both_filter),
        both_filter.T.dot(bothSilence_rec_covs.mean(-1)))

# plot the patterns
# name the SPoC channels
SPoC_channames = ['SPoC%02d' % (i+1) for i in xrange(len(snare_pattern))]

# plot the SPoC components scalp maps
snare_potmaps = [meet.sphere.potMap(chancoords, SPoC_c,
    projection='stereographic') for SPoC_c in snare_pattern]
wdBlk_potmaps = [meet.sphere.potMap(chancoords, SPoC_c,
    projection='stereographic') for SPoC_c in wdBlk_pattern]
both_potmaps = [meet.sphere.potMap(chancoords, SPoC_c,
    projection='stereographic') for SPoC_c in both_pattern]

plot_rows = int(np.ceil(args.N_SSD/4.))

snare_fig = plt.figure(figsize=(4.5,plot_rows*1.2+1.2))
# plot with 8 rows and 4 columns
gs = mpl.gridspec.GridSpec(plot_rows + 2,4,
        height_ratios = plot_rows*[1]+[0.2]+[1])
ax = []
for i, (X,Y,Z) in enumerate(snare_potmaps):
    if i == 0:
        ax.append(snare_fig.add_subplot(gs[0,0], frame_on = False))
    else:
        ax.append(snare_fig.add_subplot(gs[i//4,i%4], sharex=ax[0],
            sharey=ax[0], frame_on = False))
    Z /= np.abs(Z).max()
    ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(ax[-1])
    pc = ax[-1].pcolormesh(X, Y, Z, vmin=-1, vmax=1, rasterized=True,
            cmap=cmap)
    ax[-1].contour(X, Y, Z, levels=[0], colors='w')
    ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5)
    ax[-1].set_title(SPoC_channames[i] + '\nr=%.2f, p=%.03f' % (
        snare_corr[i], snare_p[i]))
pc_ax = snare_fig.add_subplot(gs[-2,:])
plt.colorbar(pc, cax=pc_ax, orientation='horizontal',
        label='relative amplitude')
pc_ax.plot([0.5,0.5], [0,1], c='w', zorder=1000,
        transform=pc_ax.transAxes)
eigvals_ax = snare_fig.add_subplot(gs[-1,:], frame_on=False)
eigvals_ax.plot(np.arange(1, len(snare_eigvals) + 1, 1), snare_eigvals,
        'ko-',  markersize=5)
eigvals_ax.set_xlim([0, len(snare_eigvals) + 1])
eigvals_ax.set_title('SPoC eigenvalues')
snare_fig.suptitle('snare SPoC patterns', size=14)
gs.tight_layout(snare_fig, pad=0.3, rect=(0,0,1,0.95))
snare_fig.savefig(os.path.join(args.result_folder, 'snare_SPoC_patterns.pdf'))

wdBlk_fig = plt.figure(figsize=(4.5,plot_rows*1.2+1.2))
# plot with 8 rows and 4 columns
gs = mpl.gridspec.GridSpec(plot_rows + 2,4,
        height_ratios = plot_rows*[1]+[0.2]+[1])
ax = []
for i, (X,Y,Z) in enumerate(wdBlk_potmaps):
    if i == 0:
        ax.append(wdBlk_fig.add_subplot(gs[0,0], frame_on = False))
    else:
        ax.append(wdBlk_fig.add_subplot(gs[i//4,i%4], sharex=ax[0],
            sharey=ax[0], frame_on = False))
    Z /= np.abs(Z).max()
    ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(ax[-1])
    pc = ax[-1].pcolormesh(X, Y, Z, vmin=-1, vmax=1, rasterized=True,
            cmap=cmap)
    ax[-1].contour(X, Y, Z, levels=[0], colors='w')
    ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5)
    ax[-1].set_title(SPoC_channames[i] + '\nr=%.2f, p=%.03f' % (
        wdBlk_corr[i], wdBlk_p[i]))
pc_ax = wdBlk_fig.add_subplot(gs[-2,:])
plt.colorbar(pc, cax=pc_ax, orientation='horizontal',
        label='relative amplitude')
pc_ax.plot([0.5,0.5], [0,1], c='w', zorder=1000,
        transform=pc_ax.transAxes)
eigvals_ax = wdBlk_fig.add_subplot(gs[-1,:], frame_on=False)
eigvals_ax.plot(np.arange(1, len(wdBlk_eigvals) + 1, 1), wdBlk_eigvals,
        'ko-',  markersize=5)
eigvals_ax.set_xlim([0, len(wdBlk_eigvals) + 1])
eigvals_ax.set_title('SPoC eigenvalues')
wdBlk_fig.suptitle('wdBlk SPoC patterns', size=14)
gs.tight_layout(wdBlk_fig, pad=0.3, rect=(0,0,1,0.95))
wdBlk_fig.savefig(os.path.join(args.result_folder, 'wdBlk_SPoC_patterns.pdf'))

both_fig = plt.figure(figsize=(4.5,plot_rows*1.2+1.2))
# plot with 8 rows and 4 columns
gs = mpl.gridspec.GridSpec(plot_rows + 2,4,
        height_ratios = plot_rows*[1]+[0.2]+[1])
ax = []
for i, (X,Y,Z) in enumerate(both_potmaps):
    if i == 0:
        ax.append(both_fig.add_subplot(gs[0,0], frame_on = False))
    else:
        ax.append(both_fig.add_subplot(gs[i//4,i%4], sharex=ax[0],
            sharey=ax[0], frame_on = False))
    Z /= np.abs(Z).max()
    ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(ax[-1])
    pc = ax[-1].pcolormesh(X, Y, Z, vmin=-1, vmax=1, rasterized=True,
            cmap=cmap)
    ax[-1].contour(X, Y, Z, levels=[0], colors='w')
    ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5)
    ax[-1].set_title(SPoC_channames[i] + '\nr=%.2f, p=%.03f' % (
        both_corr[i], both_p[i]))
pc_ax = both_fig.add_subplot(gs[-2,:])
plt.colorbar(pc, cax=pc_ax, orientation='horizontal',
        label='relative amplitude')
pc_ax.plot([0.5,0.5], [0,1], c='w', zorder=1000,
        transform=pc_ax.transAxes)
eigvals_ax = both_fig.add_subplot(gs[-1,:], frame_on=False)
eigvals_ax.plot(np.arange(1, len(both_eigvals) + 1, 1), both_eigvals,
        'ko-',  markersize=5)
eigvals_ax.set_xlim([0, len(both_eigvals) + 1])
eigvals_ax.set_title('SPoC eigenvalues')
both_fig.suptitle('both SPoC patterns', size=14)
gs.tight_layout(both_fig, pad=0.3, rect=(0,0,1,0.95))
both_fig.savefig(os.path.join(args.result_folder, 'both_SPoC_patterns.pdf'))
