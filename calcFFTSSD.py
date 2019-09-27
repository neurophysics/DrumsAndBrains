import numpy as np
import scipy
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os.path
import helper_functions
import meet
from tqdm import tqdm

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

# calculate the SSD from all subjects
# read the channel names
channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

N_channels = len(channames)

csd_1 = []
f = []

for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(result_folder, 'S%02d' % i)
                + '/prepared_FFTSSD.npz', 'r') as fi:
            csd_1.append(fi['csd'])
            f.append(fi['f'])
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

if np.all([np.all(f_now == f[0]) for f_now in f]):
    f = f[0]

def FFTSSD(target_covs, contrast_covs, num=None, bestof=15):
    if num != 1:
        mean_cov = np.mean(target_covs,0)
        # get whitening transform
        rank = np.linalg.matrix_rank(mean_cov)
        if num is None: num = rank
        else:
            num = min([num, rank])
        bval, bvec = np.linalg.eigh(mean_cov)
        W = bvec[:,-rank:]/np.sqrt(bval[-rank:])
        # whiten the covariance matrices
        target_covs = [np.dot(W.T, covs_now).dot(W)
                for covs_now in target_covs]
        contrast_covs = [np.dot(W.T, covs_now).dot(W)
                for covs_now in contrast_covs]
    for i in xrange(num):
        if i>0:
            # project the previous filters out
            wx = scipy.linalg.svd(np.array(w), full_matrices=True
                    )[2][i:].T
        else:
            wx = np.eye(target_covs[0].shape[0])
        temp1 = [wx.T.dot(covs_now).dot(wx)
                for covs_now in target_covs]
        temp2 = [wx.T.dot(covs_now).dot(wx)
                for covs_now in contrast_covs]
        x0 = np.random.randn(bestof, wx.shape[1])
        res = [
            scipy.optimize.minimize(
                fun = avg_power_quot_grad,
                x0 = x0_now,
                args = (temp1, temp2),
                method='L-BFGS-B',
                jac = True, options=dict(disp=False))
            for x0_now in x0]
        w_i = [res_now.x for res_now in res]
        corr_i = [res_now.fun for res_now in res]
        try:
            corr.append(-np.nanmin(corr_i))
            w.append(wx.dot(w_i[np.nanargmin(corr_i)]))
        except NameError:
            corr = [-np.nanmin(corr_i)]
            w = [wx.dot(w_i[np.nanargmin(corr_i)])]
    if num == 1:
        corr = corr[0]
        w = w[0]
    else:
        corr = np.r_[corr]
        w = W.dot(np.array(w).T)[:,np.argsort(corr)[::-1]]
        corr = np.sort(corr)[::-1]
    return corr, w

def power_quot_grad(w, target_cov, contrast_cov):
    target_power = w.dot(w.dot(target_cov))
    target_power_grad = 2*np.dot(w, target_cov)
    ###
    contrast_power = w.dot(w.dot(contrast_cov))
    contrast_power_grad = 2*np.dot(w, contrast_cov)
    ###
    quot = target_power/contrast_power
    quot_grad = (target_power_grad*contrast_power -
            target_power*contrast_power_grad)/contrast_power**2
    return -quot, -quot_grad

def avg_power_quot_grad(w, target_covs, contrast_covs):
    quot, quot_grad = zip(*[power_quot_grad(w, t, c)
        for t,c in zip(target_covs, contrast_covs)])
    return np.mean(quot), np.mean(quot_grad, 0)

target_covs = [scipy.ndimage.convolve1d(c.real,
    np.r_[1]/1., axis=-1)
    for c in csd_1]
contrast_covs = [scipy.ndimage.convolve1d(c.real,
    np.r_[1,1,1,1,0,1,1,1,1]/8., axis=-1)
    for c in csd_1]

use_idx = np.arange(len(f))[np.all([f>=1, f<=20], 0)]

quot = []
filt = []

for i in tqdm(use_idx, desc='prelim SSD'):
    temp_quot, temp_filt = FFTSSD(
            [c[...,i].real for c in target_covs],
            [c[...,i].real for c in contrast_covs],
            num = None, bestof=100)
    quot.append(temp_quot)
    filt.append(temp_filt)

#target_covs = [scipy.ndimage.convolve1d(c.real,
#    np.r_[1,1,1,1,1]/5., axis=-1)
#    for c in csd_1]
#contrast_covs = [scipy.ndimage.convolve1d(c.real,
#    np.r_[1,1,1,1,1,0,0,0,0,0,1,1,1,1,1]/10., axis=-1)
#    for c in csd_1]
#
#quot_final = []
#filt_final = []
#
#for i in tqdm(use_idx, desc='final SSD'):
#    temp_quot, temp_filt = FFTSSD(
#            [c[...,i].real for c in target_covs],
#            [c[...,i].real for c in contrast_covs],
#            num = None)
#    quot_final.append(temp_quot)
#    filt_final.append(temp_filt)

#chosen_SSD = np.argmax([q[0] for q in quot])
chosen_SSD = np.argmin((f[use_idx]-3.5)**2)

snare_quot = quot[chosen_SSD]
snare_filt = filt[chosen_SSD]
wdBlk_quot = quot[chosen_SSD]
wdBlk_filt = filt[chosen_SSD]

snare_pattern = scipy.linalg.solve(
        snare_filt.T.dot(np.mean([
            c[...,use_idx[chosen_SSD]] for c in target_covs],
            0)).dot(snare_filt),
        snare_filt.T.dot(np.mean([
            c[...,use_idx[chosen_SSD]] for c in target_covs],
                0)))
wdBlk_pattern = scipy.linalg.solve(
        wdBlk_filt.T.dot(np.mean([
            c[...,use_idx[chosen_SSD]] for c in target_covs],
            0)).dot(wdBlk_filt),
        wdBlk_filt.T.dot(np.mean([
            c[...,use_idx[chosen_SSD]] for c in target_covs],
                0)))

# plot the patterns
# name the ssd channels
snare_channames = ['SSD-%02d' % (i+1) for i in xrange(len(snare_pattern))]
wdBlk_channames = ['SSD-%02d' % (i+1) for i in xrange(len(wdBlk_pattern))]

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
SNNR_ax.plot(f[use_idx], [10*np.log10(q[0]) for q in quot], 'k-')
SNNR_ax.set_xlabel('frequency (Hz)')
SNNR_ax.set_ylabel('SNNR (dB)')
SNNR_ax.set_ylim(bottom=0, top=1.0)

SNNR_ax.set_xlim(1,15)
trans = mpl.transforms.blended_transform_factory(
        SNNR_ax.transData, SNNR_ax.transAxes)

SNNR_ax.text(snareFreq, 0.96, r'\textbf{*}', ha='center', va='top',
        transform=trans, color='b', fontsize=12)
SNNR_ax.text(wdBlkFreq, 0.96, r'\textbf{*}', ha='center', va='top',
        transform=trans, color='r', fontsize=12)
SNNR_ax.text(3.5, 0.96, r'\textbf{*}', ha='center', va='top',
        transform=trans, color='k', fontsize=12)
SNNR_ax.text(7, 0.96, r'\textbf{*}', ha='center', va='top',
        transform=trans, color='k', fontsize=12)

#mark_start = 0.5*(f[use_idx[chosen_SSD] - 3] + f[use_idx[chosen_SSD] - 2])
#mark_stop = 0.5*(f[use_idx[chosen_SSD] + 3] + f[use_idx[chosen_SSD] + 2])
#SNNR_ax.axvspan(mark_start, mark_stop, fc='r', alpha=0.2)

SNNR_ax.plot([3.5, 3.5], [10*np.log10(snare_quot[0]),0], 'k-', lw=0.5)

eigvals_ax = fig.add_subplot(gs[1,:], frame_on=True)
eigvals_ax.plot(np.arange(1, len(snare_quot) + 1, 1), 10*np.log10(snare_quot),
        'ko-', markersize=np.sqrt(20))
eigvals_ax.set_xlim([0, len(snare_quot) + 1])
eigvals_ax.axhline(0, ls='-', c='k', lw=0.5)
eigvals_ax.axvspan(0, 6.5, fc='r', alpha=0.2)
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
        [3.5, 1],
            [0,
                SNNR_ax.transData.inverted().transform(
                    eigvals_ax.transAxes.transform([0,1]))[1]],
                'k-', clip_on=False, alpha=0.5, lw=0.5)
SNNR_ax.plot(
        [3.5, 15],
            [0,
                SNNR_ax.transData.inverted().transform(
                    eigvals_ax.transAxes.transform([1,1]))[1]],
                'k-', clip_on=False, alpha=0.5, lw=0.5)

fig.align_ylabels([SNNR_ax, eigvals_ax])
fig.savefig(os.path.join(result_folder, 'FFTSSD_patterns.pdf'))

# save the results
np.savez(os.path.join(result_folder, 'FFTSSD.npz'),
        snare_filt = snare_filt,
        snare_quot = snare_quot,
        wdBlk_filt = wdBlk_filt,
        wdBlk_quot = wdBlk_quot)
