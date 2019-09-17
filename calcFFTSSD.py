import numpy as np
import scipy
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os.path
import helper_functions
import meet

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

f_ind = np.r_[np.abs(f - 3*snareFreq).argmin(), np.abs(f - 2*wdBlkFreq).argmin()]
#f_con = np.arange(len(f))[np.all([f>=2, f<=5], axis=0)]
f_con = [fi + np.array([-4, -3, -2, -1, 1, 2, 3, 4]) for fi in f_ind]

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

snare_target_covs = [c[...,f_ind[0]].real for c in csd_1]
snare_contrast_covs = [c[...,f_con[0]].mean(-1).real for c in csd_1]
wdBlk_target_covs = [c[...,f_ind[1]].real for c in csd_1]
wdBlk_contrast_covs = [c[...,f_con[1]].mean(-1).real for c in csd_1]

snare_quot, snare_filt = FFTSSD(snare_target_covs, snare_contrast_covs,
        num=None)
wdBlk_quot, wdBlk_filt = FFTSSD(wdBlk_target_covs, wdBlk_contrast_covs,
        num=None)

snare_pattern = scipy.linalg.solve(
        snare_filt.T.dot(np.mean(snare_target_covs, 0)).dot(snare_filt),
        snare_filt.T.dot(np.mean(snare_target_covs, 0)))
wdBlk_pattern = scipy.linalg.solve(
        wdBlk_filt.T.dot(np.mean(wdBlk_target_covs, 0)).dot(wdBlk_filt),
        wdBlk_filt.T.dot(np.mean(wdBlk_target_covs, 0)))

# plot the patterns
# name the ssd channels
snare_channames = ['SSD-%02d' % (i+1) for i in xrange(len(snare_pattern))]
wdBlk_channames = ['SSD-%02d' % (i+1) for i in xrange(len(wdBlk_pattern))]

# plot the SSD components scalp maps
snare_potmaps = [meet.sphere.potMap(chancoords, ssd_c,
    projection='stereographic') for ssd_c in snare_pattern]
wdBlk_potmaps = [meet.sphere.potMap(chancoords, ssd_c,
    projection='stereographic') for ssd_c in wdBlk_pattern]

# get the filtered spectra
filt_csd = [snare_filt.T.dot(snare_filt.T.dot(c)) for c in csd_1]
filt_csd_avg = np.mean(filt_csd, 0)

SNNR_i = np.array([c[0,0, f_ind[0]].real/c[0,0,f_con[0]].real.mean(-1)
    for c in filt_csd])

fmask = np.all([f>=1.5, f<=5.5], 0)
lsd1 = scipy.signal.detrend(np.sqrt(filt_csd_avg[0,0, fmask].real),
        type='linear')
lsd2 = scipy.signal.detrend(np.sqrt(filt_csd_avg[-1,-1, fmask].real),
        type='linear')

fig = plt.figure(figsize=(3.54,3.54))
# plot with 8 rows and 4 columns
#gs = mpl.gridspec.GridSpec(10,4, height_ratios = 8*[1]+[0.2]+[1])
gs = mpl.gridspec.GridSpec(4,4, height_ratios = 2*[1]+[0.1]+[1])
eigvals_ax = fig.add_subplot(gs[0,:], frame_on=True)
eigvals_ax.plot(np.arange(1, len(snare_quot) + 1, 1), 10*np.log10(snare_quot),
        'ko-', markersize=5)
eigvals_ax.set_xlim([0, len(snare_quot) + 1])
eigvals_ax.set_title('SSD eigenvalues')
eigvals_ax.axhline(1, ls='-', c='k', lw=0.5)
eigvals_ax.axvspan(0, 6.5, fc='r', alpha=0.2)
eigvals_ax.set_ylabel('SNNR at 3.5 Hz (dB)')
eigvals_ax.set_xlabel('component index')
ax = []
for i, (X,Y,Z) in enumerate(snare_potmaps):
    if i==4: break
    if i == 0:
        ax.append(fig.add_subplot(gs[1,0], frame_on = False))
    else:
        ax.append(fig.add_subplot(gs[1 + i//4,i%4], sharex=ax[0], sharey=ax[0],
                frame_on = False))
    Z /= np.abs(Z).max()
    ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(ax[-1])
    pc = ax[-1].pcolormesh(X, Y, Z, vmin=-1, vmax=1, rasterized=True,
            cmap=cmap)
    ax[-1].contour(X, Y, Z, levels=[0], colors='w')
    ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5)
    ax[-1].set_xlabel(r'\textbf{%d},  ($\mathrm{SNNR=%.2f dB}$)' % (i+1,
        snare_quot[i]))
ax[0].set_ylim([-1,1.3])
pc_ax = fig.add_subplot(gs[2,:])
cbar = plt.colorbar(pc, cax=pc_ax, orientation='horizontal',
        label='amplitude', ticks=[-1,0,1])
cbar.ax.set_xticklabels(['-', '0', '+'])
cbar.ax.set_axvline(0.5, c='w')
pc_ax.plot([0.5,0.5], [0,1], c='w', zorder=1000,
        transform=pc_ax.transAxes)

psd_ax = fig.add_subplot(gs[-1,:], frame_on=True)
psd_ax.plot(f[fmask], lsd1, c='r', label='SSD-01')
psd_ax.plot(f[fmask], lsd2, c='b', label='SSD-31')
psd_ax.set_xlim([1.5,5.5])
psd_ax.set_xlabel('frequency (Hz)')
psd_ax.set_ylabel('detrended spectrum')
psd_ax.legend(loc='upper right', fontsize=7)

gs.tight_layout(fig, pad=0.2)
fig.savefig(os.path.join(result_folder, 'FFTSSD_patterns.pdf'))


# save the results
np.savez(os.path.join(result_folder, 'FFTSSD.npz'),
        snare_filt = snare_filt,
        snare_quot = snare_quot,
        wdBlk_filt = wdBlk_filt,
        wdBlk_quot = wdBlk_quot,
        SNNR_i = SNNR_i)
