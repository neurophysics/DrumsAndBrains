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
import hashlib
import cPickle

from scipy.optimize import fmin_l_bfgs_b as _minimize

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
cov_thresh = 2000

# calculate the SSD from all subjects
# read the channel names
channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

N_channels = len(channames)

# load the SSD results
with np.load(os.path.join(args.result_folder, 'FFTSSD.npz'),
        'r') as f:
    ssd_eigvals = f['ssd_eigvals']
    ssd_filter = f['ssd_filter']
"""
# load the SSD results
with np.load(os.path.join(args.result_folder, 'SSD_norm_%s.npz' % (
    args.normSSD)),
        'r') as f:
    ssd_eigvals = f['ssd_eigvals']
    ssd_filter = f['ssd_filter']
"""
snareInlier = []
wdBlkInlier = []
snareFitSilence = []
wdBlkFitSilence = []
snare_deviation = []
wdBlk_deviation = []
snare_cov = []
wdBlk_cov = []
snareSilence_rec_covs = []
wdBlkSilence_rec_covs = []

# read the oscillatory data from the silence period
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/prepared_filterdata.npz', 'r') as f:
            snareInlier.append(f['snareInlier'])
            wdBlkInlier.append(f['wdBlkInlier'])
            snareFitSilence.append(f['snareFitSilence'])
            wdBlkFitSilence.append(f['wdBlkFitSilence'])
            snareListenData = f['snareListenData']
            wdBlkListenData = f['wdBlkListenData']
            snare_cov.append(np.einsum('ijk, ljk->ilk',
                snareListenData, snareListenData)/snareListenData.shape[1])
            wdBlk_cov.append(np.einsum('ijk, ljk->ilk',
                wdBlkListenData, wdBlkListenData)/wdBlkListenData.shape[1])
            snareSilenceData_rec = f['snareSilenceData_rec']
            snareSilence_rec_covs.append(
                    np.einsum('ijk, ljk->ilk',
                        snareSilenceData_rec, snareSilenceData_rec)/
                    snareSilenceData_rec.shape[1])
            wdBlkSilenceData_rec = f['wdBlkSilenceData_rec']
            wdBlkSilence_rec_covs.append(
                    np.einsum('ijk, ljk->ilk',
                        wdBlkSilenceData_rec, wdBlkSilenceData_rec)/
                    wdBlkSilenceData_rec.shape[1])
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

# apply the SSD and get the complex-valued fitted data
snarePower_snareCue = [np.tensordot(ssd_filter[:,:args.N_SSD], f,
        axes=(0,0))[:,2:4] for f in snareFitSilence]
snarePower_snareCue = [d[:,0] + 1j*d[:,1] for d in snarePower_snareCue]
wdBlkPower_snareCue = [np.tensordot(ssd_filter[:,:args.N_SSD], f,
        axes=(0,0))[:,4:6] for f in snareFitSilence]
wdBlkPower_snareCue = [d[:,0] + 1j*d[:,1] for d in wdBlkPower_snareCue]

snarePower_wdBlkCue = [np.tensordot(ssd_filter[:,:args.N_SSD], f,
        axes=(0,0))[:,2:4] for f in wdBlkFitSilence]
snarePower_wdBlkCue = [d[:,0] + 1j*d[:,1] for d in snarePower_wdBlkCue]
wdBlkPower_wdBlkCue = [np.tensordot(ssd_filter[:,:args.N_SSD], f,
        axes=(0,0))[:,4:6] for f in wdBlkFitSilence]
wdBlkPower_wdBlkCue = [d[:,0] + 1j*d[:,1] for d in wdBlkPower_wdBlkCue]

# only keep those trials where both, behaviour and EEG, were measured
# correctly
snare_deviation = [d[i] for d,i in zip(snare_deviation, snareInlier)]
wdBlk_deviation = [d[i] for d,i in zip(wdBlk_deviation, wdBlkInlier)]

###################
# reject outliers #
###################
snare_inlier = [np.all([
    p[range(N_channels), range(N_channels)].sum(0) < cov_thresh,
    np.isfinite(d),
    np.abs(d) < 0.4], axis=0)
    for p, d in zip(snare_cov, snare_deviation)]

wdBlk_inlier = [np.all([
    p[range(N_channels), range(N_channels)].sum(0) < cov_thresh,
    np.isfinite(d),
    np.abs(d) < 0.6], axis=0)
    for p, d in zip(wdBlk_cov, wdBlk_deviation)]

(snarePower_snareCue, wdBlkPower_snareCue, snareSilence_rec_covs,
        snare_deviation) = zip(*[
    (p[...,I], q[...,I], r[...,I], d[I])
    for p, q, r, d,I in zip(snarePower_snareCue, wdBlkPower_snareCue,
        snareSilence_rec_covs, snare_deviation, snare_inlier)])

(snarePower_wdBlkCue, wdBlkPower_wdBlkCue, wdBlkSilence_rec_covs,
         wdBlk_deviation) = zip(*[
    (p[...,I], q[...,I], r[...,I], d[I])
    for p, q, r, d,I in zip(snarePower_wdBlkCue, wdBlkPower_wdBlkCue,
        wdBlkSilence_rec_covs, wdBlk_deviation, wdBlk_inlier)])

if args.normalize:
    snarePower_snareCue, wdBlkPower_snareCue, snareSilence_rec_covs = zip(
            *[(
        p/np.sqrt(np.trace(s.mean(-1))),
        q/np.sqrt(np.trace(s.mean(-1))),
        r/np.sqrt(np.trace(s.mean(-1))))
        for p, q, r, s in zip(snarePower_snareCue, wdBlkPower_snareCue,
            snareSilence_rec_covs, snare_cov)])
    snarePower_wdBlkCue, wdBlkPower_wdBlkCue, wdBlkSilence_rec_covs = zip(
            *[(
        p/np.sqrt(np.trace(s.mean(-1))),
        q/np.sqrt(np.trace(s.mean(-1))),
        r/np.sqrt(np.trace(s.mean(-1))))
        for p, q, r, s in zip(snarePower_wdBlkCue, wdBlkPower_wdBlkCue,
            wdBlkSilence_rec_covs, wdBlk_cov)])

if args.absdev:
    snare_deviation = [np.abs(d) for d in snare_deviation]
    wdBlk_deviation = [np.abs(d) for d in wdBlk_deviation]

if args.rank:
    snare_deviation = [d.argsort().argsort()/(len(d) - 1.)
            for d in snare_deviation]
    wdBlk_deviation = [d.argsort().argsort()/(len(d) - 1.)
            for d in wdBlk_deviation]

# concatenate data of individual subjects
snarePower_snareCue = np.hstack(snarePower_snareCue)
wdBlkPower_snareCue = np.hstack(wdBlkPower_snareCue)
snarePower_wdBlkCue = np.hstack(snarePower_wdBlkCue)
wdBlkPower_wdBlkCue = np.hstack(wdBlkPower_wdBlkCue)
snare_deviation = np.hstack(snare_deviation)
wdBlk_deviation = np.hstack(wdBlk_deviation)
snareSilence_rec_covs = np.dstack(snareSilence_rec_covs)
wdBlkSilence_rec_covs = np.dstack(wdBlkSilence_rec_covs)

# calculate the hash
save_hash = hashlib.sha256(cPickle.dumps([
    snarePower_snareCue,
    wdBlkPower_snareCue,
    snarePower_wdBlkCue,
    wdBlkPower_wdBlkCue,
    snare_deviation,
    wdBlk_deviation,
    snareSilence_rec_covs,
    wdBlkSilence_rec_covs,
    ])).hexdigest()

# calculate a cSPoC
def cSPoC_obj_der(w, target, contrast, a, *args, **kwargs):
    target_corr, target_corr_der = meet._cSPoC._env_corr(
            np.r_[w,1], target, (a+1j*np.zeros_like(a))[np.newaxis],
            *args, **kwargs)
    contrast_corr, contrast_corr_der = meet._cSPoC._env_corr(
            np.r_[w,1], contrast, (a+1j*np.zeros_like(a))[np.newaxis],
            *args, **kwargs)
    corr = target_corr + np.abs(contrast_corr)
    if np.abs(contrast_corr) < 1E-10:
        corr_der = target_corr_der
    else:
        corr_der = target_corr_der + contrast_corr/np.abs(
                contrast_corr)*contrast_corr_der
    return corr, corr_der[:-1]

def cSPoC(target, contrast, a, num=1, bestof=15, opt='min', log=False):
    if opt == 'max': sign = -1
    elif opt == 'min': sign = 1
    else: raise ValueError('opt must be min or max')
    # whiten for the real part of the target
    if num != 1:
        Wt, st = scipy.linalg.svd(target.real, full_matrices=False)[:2]
        rt = np.linalg.matrix_rank(target.real)
        if num is None: num = rt
        W = Wt[:,:rt]/st[:rt]
        target = W.T.dot(target)
        contrast = W.T.dot(contrast)
    for i in xrange(num):
        if i>0:
            # project the previous filters out
            wx = scipy.linalg.svd(np.array(w), full_matrices=True)[2][i:].T
        else:
            wx = np.eye(target.shape[0])
        w_i, corr_i = zip(*[
            _minimize(func = cSPoC_obj_der, fprime = None,
                x0 = np.random.randn(wx.shape[1]),
                args = (wx.T.dot(target), wx.T.dot(contrast), a, sign, log),
                m=100, approx_grad=False, iprint=0)[:2]
            for k in xrange(bestof)])
        try:
            corr.append(sign*np.min(corr_i))
            w.append(wx.dot(w_i[np.argmin(corr_i)]))
        except NameError:
            corr = [sign*np.min(corr_i)]
            w = [wx.dot(w_i[np.argmin(corr_i)])]
    if num == 1:
        corr = corr[0]
        w = w[0]
    else:
        corr = np.r_[corr]
        w = W.dot(np.array(w).T)
    return corr, w

# load the results

try:
    with np.load(os.path.join(args.result_folder,
        'cSPoC_results_%s.npz' % save_hash), 'rb') as f:
        snare_min_corr = f['snare_min_corr']
        snare_min_filt = f['snare_min_filt']
        snare_min_corr_boot = f['snare_min_corr_boot']
        snare_min_corr_p = f['snare_min_corr_p']
        snare_max_corr = f['snare_max_corr']
        snare_max_filt = f['snare_max_filt']
        snare_max_corr_boot = f['snare_max_corr_boot']
        snare_max_corr_p = f['snare_max_corr_p']
        wdBlk_min_corr = f['wdBlk_min_corr']
        wdBlk_min_filt = f['wdBlk_min_filt']
        wdBlk_min_corr_boot = f['wdBlk_min_corr_boot']
        wdBlk_min_corr_p = f['wdBlk_min_corr_p']
        wdBlk_max_corr = f['wdBlk_max_corr']
        wdBlk_max_filt = f['wdBlk_max_filt']
        wdBlk_max_corr_boot = f['wdBlk_max_corr_boot']
        wdBlk_max_corr_p = f['wdBlk_max_corr_p']
    print('Loaded all the data')
except:
    # if loading previous results did not work, calculate new results
    snare_min_corr, snare_min_filt = cSPoC(snarePower_snareCue,
            wdBlkPower_snareCue, snare_deviation, opt='min', num=None)
    snare_min_corr_boot = np.array([cSPoC(snarePower_snareCue,
        wdBlkPower_snareCue, np.random.permutation(snare_deviation),
        opt='min')[0]
        for _ in trange(1000)])
    snare_min_corr_p = (
            (snare_min_corr_boot[np.newaxis]<=snare_min_corr[
                :,np.newaxis]).sum(-1) + 1)/(len(snare_min_corr_boot) + 1.)
    snare_max_corr, snare_max_filt = cSPoC(snarePower_snareCue,
            wdBlkPower_snareCue, snare_deviation, opt='max', num=None)
    snare_max_corr_boot = np.array([cSPoC(snarePower_snareCue,
        wdBlkPower_snareCue, np.random.permutation(snare_deviation),
        opt='max')[0]
        for _ in trange(1000)])
    snare_max_corr_p = (
            (snare_max_corr_boot[np.newaxis]>=snare_max_corr[
                :,np.newaxis]).sum(-1) + 1)/(len(snare_max_corr_boot) + 1.)
    wdBlk_min_corr, wdBlk_min_filt = cSPoC(wdBlkPower_wdBlkCue,
            snarePower_wdBlkCue, wdBlk_deviation, opt='min', num=None)
    wdBlk_min_corr_boot = np.array([cSPoC(wdBlkPower_wdBlkCue,
        snarePower_wdBlkCue, np.random.permutation(wdBlk_deviation),
        opt='min')[0]
        for _ in trange(1000)])
    wdBlk_min_corr_p = (
            (wdBlk_min_corr_boot[np.newaxis]<=wdBlk_min_corr[
                :,np.newaxis]).sum(-1) + 1)/(len(wdBlk_min_corr_boot) + 1.)
    wdBlk_max_corr, wdBlk_max_filt = cSPoC(wdBlkPower_wdBlkCue,
            snarePower_wdBlkCue, wdBlk_deviation, opt='max', num=None)
    wdBlk_max_corr_boot = np.array([cSPoC(wdBlkPower_wdBlkCue,
        snarePower_wdBlkCue, np.random.permutation(wdBlk_deviation),
        opt='max')[0]
        for _ in trange(1000)])
    wdBlk_max_corr_p = (
            (wdBlk_max_corr_boot[np.newaxis]>=wdBlk_max_corr[
                :,np.newaxis]).sum(-1) + 1)/(len(wdBlk_max_corr_boot) + 1.)
    # save the results
    np.savez(os.path.join(args.result_folder, 'cSPoC_results_%s.npz' % save_hash),
            snare_min_corr = snare_min_corr,
            snare_min_filt = snare_min_filt,
            snare_min_corr_boot = snare_min_corr_boot,
            snare_min_corr_p = snare_min_corr_p,
            snare_max_corr = snare_max_corr,
            snare_max_filt = snare_max_filt,
            snare_max_corr_boot = snare_max_corr_boot,
            snare_max_corr_p = snare_max_corr_p,
            wdBlk_min_corr = wdBlk_min_corr,
            wdBlk_min_filt = wdBlk_min_filt,
            wdBlk_min_corr_boot = wdBlk_min_corr_boot,
            wdBlk_min_corr_p = wdBlk_min_corr_p,
            wdBlk_max_corr = wdBlk_max_corr,
            wdBlk_max_filt = wdBlk_max_filt,
            wdBlk_max_corr_boot = wdBlk_max_corr_boot,
            wdBlk_max_corr_p = wdBlk_max_corr_p,
            )

# plot the results
# get the spatial patterns
full_snare_min_filt = ssd_filter[:,:args.N_SSD].dot(snare_min_filt)
full_snare_max_filt = ssd_filter[:,:args.N_SSD].dot(snare_max_filt)
full_wdBlk_min_filt = ssd_filter[:,:args.N_SSD].dot(wdBlk_min_filt)
full_wdBlk_max_filt = ssd_filter[:,:args.N_SSD].dot(wdBlk_max_filt)

snare_min_pattern = scipy.linalg.solve(
        full_snare_min_filt.T.dot(
            snareSilence_rec_covs.mean(-1)).dot(full_snare_min_filt),
        full_snare_min_filt.T.dot(snareSilence_rec_covs.mean(-1)))
snare_max_pattern = scipy.linalg.solve(
        full_snare_max_filt.T.dot(
            snareSilence_rec_covs.mean(-1)).dot(full_snare_max_filt),
        full_snare_max_filt.T.dot(snareSilence_rec_covs.mean(-1)))
wdBlk_min_pattern = scipy.linalg.solve(
        full_wdBlk_min_filt.T.dot(
            wdBlkSilence_rec_covs.mean(-1)).dot(full_wdBlk_min_filt),
        full_wdBlk_min_filt.T.dot(wdBlkSilence_rec_covs.mean(-1)))
wdBlk_max_pattern = scipy.linalg.solve(
        full_wdBlk_max_filt.T.dot(
            wdBlkSilence_rec_covs.mean(-1)).dot(full_wdBlk_max_filt),
        full_wdBlk_max_filt.T.dot(wdBlkSilence_rec_covs.mean(-1)))

# name the SPoC channels
cSPoC_channames = [
        'snare minCorr 1', 'snare maxCorr 1',
        'wdBlk minCorr 1', 'wdBlk maxcorr 1',
        'snare minCorr 2', 'snare maxCorr 2',
        'wdBlk minCorr 2', 'wdBlk maxcorr 2',
        ]

all_corr = [
        snare_min_corr[0], snare_max_corr[0],
        wdBlk_min_corr[0], wdBlk_max_corr[0],
        snare_min_corr[1], snare_max_corr[1],
        wdBlk_min_corr[1], wdBlk_max_corr[1],
        ]
all_corr_p = [
        snare_min_corr_p[0], snare_max_corr_p[0],
        wdBlk_min_corr_p[0], wdBlk_max_corr_p[0],
        snare_min_corr_p[1], snare_max_corr_p[1],
        wdBlk_min_corr_p[1], wdBlk_max_corr_p[1]]


# plot the SPoC components scalp maps
potmaps = [meet.sphere.potMap(chancoords, SPoC_c,
    projection='stereographic') for SPoC_c in
    [
        snare_min_pattern[0],
        snare_max_pattern[0],
        wdBlk_min_pattern[0],
        wdBlk_max_pattern[0],
        snare_min_pattern[1],
        snare_max_pattern[1],
        wdBlk_min_pattern[1],
        wdBlk_max_pattern[1],
        ]]

plot_rows = 2

fig = plt.figure(figsize=(5,plot_rows*2 + 0.5))
gs = mpl.gridspec.GridSpec(plot_rows + 1,4,
        height_ratios = plot_rows*[1]+[0.1])
ax = []
for i, (X,Y,Z) in enumerate(potmaps):
    if i == 0:
        ax.append(fig.add_subplot(gs[0,0], frame_on = False))
    else:
        ax.append(fig.add_subplot(gs[i//4,i%4], sharex=ax[0],
            sharey=ax[0], frame_on = False))
    Z /= np.abs(Z).max()
    ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(ax[-1])
    pc = ax[-1].pcolormesh(X, Y, Z, vmin=-1, vmax=1, rasterized=True,
            cmap='coolwarm')
    ax[-1].contour(X, Y, Z, levels=[0], colors='w')
    ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5)
    ax[-1].set_title(cSPoC_channames[i] + '\nr=%.2f, p=%.03f' % (
        all_corr[i], all_corr_p[i]))
pc_ax = fig.add_subplot(gs[-1,:])
plt.colorbar(pc, cax=pc_ax, orientation='horizontal',
        label='relative amplitude')
pc_ax.plot([0.5,0.5], [0,1], c='w', zorder=1000,
        transform=pc_ax.transAxes)
fig.suptitle('snare SPoC patterns', size=14)
gs.tight_layout(fig, pad=0.3, rect=(0,0,1,0.875))
fig.savefig(os.path.join(args.result_folder, 'cSPoC_patterns.pdf'))

# analyse the results of the output
snareData = []
wdBlkData = []

subject_list = range(1, N_subjects + 1, 1)
subject_list.remove(11)

# read the oscillatory data from the silence period
for i, sI, wI in zip(subject_list, snare_inlier, wdBlk_inlier):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/prepared_filterdata.npz', 'r') as f:
            tempListen = np.tensordot(full_snare_max_filt[:,0], f[
                'snareListenData'],axes=(0,0))[:,sI]
            tempSilence = np.tensordot(full_snare_max_filt[:,0], f[
                'snareSilenceData'],axes=(0,0))[:,sI]
            snareData.append(
                    np.vstack([
                        tempListen - tempListen[-1],
                        tempSilence - tempSilence[0]])
                        )
            tempListen = np.tensordot(full_snare_max_filt[:,0], f[
                'wdBlkListenData'],axes=(0,0))[:,wI]
            tempSilence = np.tensordot(full_snare_max_filt[:,0], f[
                'wdBlkSilenceData'],axes=(0,0))[:,wI]
            wdBlkData.append(
                    np.vstack([
                        tempListen - tempListen[-1],
                        tempSilence - tempSilence[0]])
                        )
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

if args.normalize:
    snareData, wdBlkData = zip(*[
        (s/np.hstack([s,w]).std(), w/np.hstack([s,w]).std())
        for s,w in zip(snareData, wdBlkData)])

snareData = np.hstack(snareData)
wdBlkData = np.hstack(wdBlkData)

# calculate the S transform
# create a custom sampling scheme for the S transform
def custom_sampling(N):
    S_frange = [0.1, 10]
    S_fnum = 100
    S_Nperperiod = 4
    wanted_freqs = np.linspace(S_frange[0], S_frange[1], S_fnum)
    fftfreqs = np.fft.fftfreq(N, d=1./s_rate)
    # find the nearest frequency indices
    y = np.unique([np.argmin((w - fftfreqs)**2)
        for w in wanted_freqs])
    x = np.ones_like(y)*120
    return x,y

# define some constants
QPM = 140 # quarter notes per minute
SPQ = 60./QPM # seconds per quarter note
bar_duration = SPQ*4 # every bar consists of 4 quarter notes

# get the frequencies of the snaredrum (duple) and woodblock (triple) beats
snareFreq = 2./bar_duration
wdBlkFreq = 3./bar_duration

#calculate the S-transforms
tf_coords, snare_tf = meet.tf.gft(scipy.signal.detrend(
    snareData, axis=0, type='constant'), axis=0, sampling=custom_sampling)
tf_coords, wdBlk_tf = meet.tf.gft(scipy.signal.detrend(
    wdBlkData, axis=0, type='constant'), axis=0, sampling=custom_sampling)
snare_tf = np.abs(snare_tf)
wdBlk_tf = np.abs(wdBlk_tf)
tf_avg = np.mean(np.vstack([snare_tf, wdBlk_tf]), 0).reshape(-1, 240)

from tqdm import tqdm 
snare_corr = np.array([np.corrcoef(tf_now, snare_deviation)[0,1]
    for tf_now in tqdm(snare_tf.T)]).reshape(-1, 240)
wdBlk_corr = np.array([np.corrcoef(tf_now, wdBlk_deviation)[0,1]
    for tf_now in tqdm(wdBlk_tf.T)]).reshape(-1, 240)

tf_f = np.fft.fftfreq(snareData.shape[0], d=1./s_rate)[np.unique(tf_coords[0]).astype(int)]
tf_t = np.linspace(0, 4*bar_duration, 240)

fig = plt.figure(figsize=(10,3))
ax1 = fig.add_subplot(131)
pc1 = ax1.pcolormesh(tf_t, tf_f, scipy.ndimage.convolve1d(
    20*np.log10(tf_avg),
    weights=[-0.25, -0.25, 1, -0.25, -0.25],
    axis=0, mode='reflect'), cmap=cmap, vmin=0, vmax=1.5, rasterized=True)
ax1.axvline(3*bar_duration, c='w', lw=0.5)
ax1.axhline(snareFreq, c='w', lw=0.5)
ax1.axhline(wdBlkFreq, c='w', lw=0.5)
ax1.set_title('average power')
plt.colorbar(pc1, ax=ax1, label='power (dB)')
ax2 = fig.add_subplot(132, sharex=ax1, sharey=ax1)
pc2 = ax2.pcolormesh(tf_t, tf_f, snare_corr, cmap='coolwarm',
        vmin=-0.15, vmax=0.15, rasterized=True)
ax2.axvline(3*bar_duration, c='w', lw=0.5)
ax2.axhline(snareFreq, c='w', lw=0.5)
ax2.axhline(wdBlkFreq, c='w', lw=0.5)
ax2.set_title('snare correlation')
plt.colorbar(pc2, ax=ax2, label='correlation coefficient')
ax3 = fig.add_subplot(133, sharex=ax1, sharey=ax1)
pc3 = ax3.pcolormesh(tf_t, tf_f, wdBlk_corr, cmap='coolwarm',
        vmin=-0.15, vmax=0.15, linewidth=0, rasterized=True)
ax3.axvline(3*bar_duration, c='w', lw=0.5)
ax3.axhline(snareFreq, c='w', lw=0.5)
ax3.axhline(wdBlkFreq, c='w', lw=0.5)
ax3.set_title('wdBlk correlation')
plt.colorbar(pc3, ax=ax3, label='correlation coefficient')
ax3.set_xlabel('time (ms)')
ax1.set_ylabel('frequency (Hz)')
ax2.set_ylabel('frequency (Hz)')
ax3.set_ylabel('frequency (Hz)')
fig.tight_layout()
fig.savefig(os.path.join(args.result_folder, 'cSPoC_snareminResults.pdf'))
