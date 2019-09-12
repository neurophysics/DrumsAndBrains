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
import SPoC

state='silence'

from scipy.optimize import fmin_l_bfgs_b as _minimize

parser = argparse.ArgumentParser(description='Calculate PCO')

parser.add_argument('result_folder', type=str, default='./Results/',
        help='the folder to store all results', nargs='?')
args = parser.parse_args()

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
    snare_quot = f['snare_quot']
    snare_filt = f['snare_filt']
    wdBlk_quot = f['wdBlk_quot']
    wdBlk_filt = f['wdBlk_filt']

#snare_N_SSD = np.sum(snare_quot>=1)
#wdBlk_N_SSD = np.sum(wdBlk_quot>=1)

snare_N_SSD = 7
wdBlk_N_SSD = 7

snare_silence_csd = []
wdBlk_silence_csd = []
snare_silence_csd_con = []
wdBlk_silence_csd_con = []
snare_deviation = []
wdBlk_deviation = []

# read the oscillatory data from the silence period
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/prepare_FFTcSPoC.npz', 'r') as f:
            snare_silence_csd_con.append(f['snare_%s_csd_con' % state])
            wdBlk_silence_csd_con.append(f['wdBlk_%s_csd_con' % state])
            snare_silence_csd.append(f['snare_%s_csd' % state])
            wdBlk_silence_csd.append(f['wdBlk_%s_csd' % state])
            snare_deviation.append(f['snare_deviation'])
            wdBlk_deviation.append(f['wdBlk_deviation'])
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

snare_silence_csd = [np.dot(
    snare_filt[:,:snare_N_SSD].T, np.dot(
        snare_filt[:,:snare_N_SSD].T, csd_now.T))
    for csd_now in snare_silence_csd]
snare_silence_csd_con = [np.dot(
    snare_filt[:,:snare_N_SSD].T, np.dot(
        snare_filt[:,:snare_N_SSD].T, csd_now.T))
    for csd_now in snare_silence_csd_con]
wdBlk_silence_csd = [np.dot(
    wdBlk_filt[:,:wdBlk_N_SSD].T, np.dot(
        wdBlk_filt[:,:wdBlk_N_SSD].T, csd_now.T))
    for csd_now in wdBlk_silence_csd]
wdBlk_silence_csd_con = [np.dot(
    wdBlk_filt[:,:wdBlk_N_SSD].T, np.dot(
        wdBlk_filt[:,:wdBlk_N_SSD].T, csd_now.T))
    for csd_now in wdBlk_silence_csd_con]

N_bootstrap=1
from tqdm import trange,tqdm

"""
# This turned out to be non-significant
###
snare_z = [np.abs(z_now).argsort().argsort() for z_now in snare_deviation]

pSPoCr2_avg = SPoC.pSPoCr2_avg(bestof=200)
snare_corr_var = np.array([pSPoCr2_avg(
        [c.real for c in snare_silence_csd],
        [c.real for c in snare_silence_csd_con],
        snare_z, num=1)[0] for _ in trange(20)])
1/0

snare_corr, snare_w = pSPoCr2_avg(
        [c.real for c in snare_silence_csd],
        [c.real for c in snare_silence_csd_con],
        snare_z, num=1)
snare_corr_boot = np.array([pSPoCr2_avg(
    [c.real for c in snare_silence_csd],
    [c.real for c in snare_silence_csd_con],
    [np.random.choice(z_now, size=len(z_now), replace=False)
        for z_now in snare_z],num=1)[0]
    for _ in trange(N_bootstrap)])
###
wdBlk_z = [np.abs(z_now).argsort().argsort() for z_now in wdBlk_deviation]
wdBlk_corr, wdBlk_w = pSPoCr2_avg(
        [c.real for c in wdBlk_silence_csd],
        [c.real for c in wdBlk_silence_csd_con],
        wdBlk_z, num=1)
wdBlk_corr_boot = np.array([pSPoCr2_avg(
    [c.real for c in wdBlk_silence_csd],
    [c.real for c in wdBlk_silence_csd_con],
    [np.random.choice(z_now, size=len(z_now), replace=False)
        for z_now in wdBlk_z],num=1)[0]
    for _ in trange(10)])
###
"""

snare_covs_all = np.concatenate(snare_silence_csd, -1).real
snare_covs_con_all = np.concatenate(snare_silence_csd_con, -1).real
snare_deviation_all = np.abs(np.hstack(snare_deviation)).argsort().argsort()
wdBlk_covs_all = np.concatenate(wdBlk_silence_csd, -1).real
wdBlk_covs_con_all = np.concatenate(wdBlk_silence_csd_con, -1).real
wdBlk_deviation_all = np.abs(np.hstack(wdBlk_deviation)).argsort().argsort()

partial_SPoC = SPoC.pSPoCr2()

snare_corr_2, snare_w = partial_SPoC(
        snare_covs_all, snare_covs_con_all,
        snare_deviation_all, num=None)
wdBlk_corr_2, wdBlk_w = partial_SPoC(
        wdBlk_covs_all, wdBlk_covs_con_all,
        wdBlk_deviation_all, num=None)

snare_corr_2_boot = np.array([partial_SPoC(
    snare_covs_all, snare_covs_con_all,
    np.random.choice(snare_deviation_all,
        size=len(snare_deviation_all), replace=False),
    num=1)[0] for _ in trange(N_bootstrap)])
wdBlk_corr_2_boot = np.array([partial_SPoC(
    wdBlk_covs_all, wdBlk_covs_con_all,
    np.random.choice(wdBlk_deviation_all,
        size=len(wdBlk_deviation_all), replace=False),
    num=1)[0] for _ in trange(N_bootstrap)])

# calculate raw p values
snare_corr_p = ((snare_corr_2_boot>snare_corr_2[:,np.newaxis]).sum(
    -1) + 1)/float(N_bootstrap + 1)
wdBlk_corr_p = ((wdBlk_corr_2_boot>wdBlk_corr_2[:,np.newaxis]).sum(
    -1) + 1)/float(N_bootstrap + 1)

# calculate the partial correlation coefficients
snare_corr = np.array([
    SPoC._partial_corr_grad(w_now, snare_covs_all, snare_covs_con_all,
        snare_deviation_all)[0]
    for w_now in snare_w.T])
wdBlk_corr = np.array([
    SPoC._partial_corr_grad(w_now, wdBlk_covs_all, wdBlk_covs_con_all,
        wdBlk_deviation_all)[0]
    for w_now in wdBlk_w.T])

snare_silence_csd = []
wdBlk_silence_csd = []

# read the oscillatory data from the silence period
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/prepare_FFTcSPoC.npz', 'r') as f:
            snare_silence_csd.append(f['snare_%s_csd' % state])
            wdBlk_silence_csd.append(f['wdBlk_%s_csd' % state])
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

#calculate the spatial patterns
snare_filter = snare_filt[:,:snare_N_SSD].dot(snare_w)
snare_pattern = scipy.linalg.lstsq(
        snare_filter.T.dot(
            np.vstack(snare_silence_csd).mean(0).real).dot(snare_filter),
        snare_filter.T.dot(np.vstack(snare_silence_csd).mean(0).real))[0]
snare_pattern /= np.abs(snare_pattern).max(-1)[:,np.newaxis]
wdBlk_filter = wdBlk_filt[:,:wdBlk_N_SSD].dot(wdBlk_w)
wdBlk_pattern = scipy.linalg.lstsq(
        wdBlk_filter.T.dot(
            np.vstack(wdBlk_silence_csd).mean(0).real).dot(wdBlk_filter),
        wdBlk_filter.T.dot(np.vstack(wdBlk_silence_csd).mean(0).real))[0]
wdBlk_pattern /= np.abs(wdBlk_pattern).max(-1)[:,np.newaxis]

# get the trial data
best_snare_listen_trials = []
best_snare_silence_trials = []
best_wdBlk_listen_trials = []
best_wdBlk_silence_trials = []

# read the oscillatory data from the silence period
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/prepare_FFTcSPoC.npz', 'r') as f:
            best_snare_listen_trials.append(
                    np.tensordot(
                        snare_filt[:,:snare_N_SSD].dot(snare_w[:,0]),
                        f['snare_listen_trials'],
                        axes=(0,0)))
            best_snare_silence_trials.append(
                    np.tensordot(
                        snare_filt[:,:snare_N_SSD].dot(snare_w[:,0]),
                        f['snare_silence_trials'],
                        axes=(0,0)))
            best_wdBlk_listen_trials.append(
                    np.tensordot(
                        wdBlk_filt[:,:wdBlk_N_SSD].dot(wdBlk_w[:,0]),
                        f['wdBlk_listen_trials'],
                        axes=(0,0)))
            best_wdBlk_silence_trials.append(
                    np.tensordot(
                        wdBlk_filt[:,:wdBlk_N_SSD].dot(wdBlk_w[:,0]),
                        f['wdBlk_silence_trials'],
                        axes=(0,0)))
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

best_snare_listen_trials = np.hstack(best_snare_listen_trials)
best_snare_silence_trials = np.hstack(best_snare_silence_trials)
best_wdBlk_listen_trials = np.hstack(best_wdBlk_listen_trials)
best_wdBlk_silence_trials = np.hstack(best_wdBlk_silence_trials)

# stitch them together
snareData = np.vstack([best_snare_listen_trials, best_snare_silence_trials])
wdBlkData = np.vstack([best_wdBlk_listen_trials, best_wdBlk_silence_trials])

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
snare_tf_avg = np.mean(snare_tf, 0).reshape(-1, 240)
wdBlk_tf_avg = np.mean(wdBlk_tf, 0).reshape(-1, 240)

# calculate the partial correlation
def calc_pcorr(X,Y,z):
    X = (X-X.mean(0))/X.std(0)
    Y = (Y-Y.mean(0))/Y.std(0)
    z = (z - z.mean())/z.std()
    XY = (X*Y[:,np.newaxis]).mean(0)
    Xz = (X*z[:,np.newaxis]).mean(0)
    Yz = (Y*z).mean(0)
    pcorr = (Xz - XY*Yz)/np.sqrt((1-XY**2)*(1-Yz**2))
    return pcorr

def calc_corr(X,z):
    X = (X-X.mean(0))/X.std(0)
    z = (z - z.mean())/z.std()
    Xz = (X*z[:,np.newaxis]).mean(0)
    return Xz

snare_corr_tf = calc_corr(np.log(snare_tf**2),
        snare_deviation_all).reshape(-1, 240)
wdBlk_corr_tf = calc_corr(np.log(wdBlk_tf**2),
        wdBlk_deviation_all).reshape(-1, 240)

snare_con_power = np.log(snare_w[:,0].T.dot(snare_w[:,0].T.dot(
    snare_covs_con_all)))
snare_pcorr_tf = calc_pcorr(np.log(snare_tf**2),
        snare_con_power, snare_deviation_all).reshape(-1, 240)
# bootstrap the partial correlation tf-transform
snare_pcorr_tf_boot = np.array([calc_pcorr(np.log(snare_tf**2),
        snare_con_power, np.random.choice(
            snare_deviation_all, size=len(snare_deviation_all),
            replace=False)) for _ in trange(1000)])

wdBlk_con_power = np.log(wdBlk_w[:,0].T.dot(wdBlk_w[:,0].T.dot(
    wdBlk_covs_con_all)))
wdBlk_pcorr_tf = calc_pcorr(np.log(wdBlk_tf**2),
        wdBlk_con_power, wdBlk_deviation_all).reshape(-1, 240)
# bootstrap the partial correlation tf-transform
wdBlk_pcorr_tf_boot = np.array([calc_pcorr(np.log(wdBlk_tf**2),
        wdBlk_con_power, np.random.choice(
            wdBlk_deviation_all, size=len(wdBlk_deviation_all),
            replace=False)) for _ in trange(1000)])

import stepdown_p
snare_pcorr_tf_p =stepdown_p.stepdown_p(-np.ravel(snare_pcorr_tf),
        -snare_pcorr_tf_boot).reshape(-1,240)


snare_X, snare_Y, snare_Z = zip(*[meet.sphere.potMap(chancoords, p)
    for p in snare_pattern])
wdBlk_X, wdBlk_Y, wdBlk_Z = zip(*[meet.sphere.potMap(chancoords, p)
    for p in wdBlk_pattern])


tf_f = np.fft.fftfreq(snareData.shape[0], d=1./s_rate)[np.unique(tf_coords[0]).astype(int)]
tf_t = np.linspace(0, 4*bar_duration, 240)

fig = plt.figure(figsize=(3,7))
gs = mpl.gridspec.GridSpec(7,2, height_ratios=[0.8,0.1,1,0.1,1,1,0.1])

ax00 = fig.add_subplot(gs[0,0], frameon=False)
pc00 = ax00.pcolormesh(snare_X[0], snare_Y[0], snare_Z[0], cmap='coolwarm',
        vmin=-1, vmax=1, rasterized=True)
ax00.contour(snare_X[0], snare_Y[0], snare_Z[0], levels=[0], colors='w')
meet.sphere.addHead(ax00)
ax00.tick_params(**blind_ax)
ax00.set_title('snare ($p=%.3f$)' % snare_corr_p[0])

ax01 = fig.add_subplot(gs[0,1], frameon=False, sharex=ax00, sharey=ax00)
pc01 = ax01.pcolormesh(wdBlk_X[0], wdBlk_Y[0], wdBlk_Z[0], cmap='coolwarm',
        vmin=-1, vmax=1, rasterized=True)
ax01.contour(wdBlk_X[0], wdBlk_Y[0], wdBlk_Z[0], levels=[0], colors='w')
meet.sphere.addHead(ax01)
ax01.tick_params(**blind_ax)
ax01.set_title('wdBlk ($p=%.3f$)' % wdBlk_corr_p[0])

ax00.set_xlim([-1.2,1.2])
ax00.set_ylim([-1.1,1.2])

pc_ax0 = fig.add_subplot(gs[1,:])
plt.colorbar(pc00, cax=pc_ax0, label='amplitude', orientation='horizontal')

ax11 = fig.add_subplot(gs[2,0])
pc11 = ax11.pcolormesh(tf_t, tf_f, scipy.ndimage.convolve1d(
    20*np.log10(snare_tf_avg),
    weights=np.r_[4*[-0.125], 1, 4*[-0.125]],
    axis=0, mode='reflect'), cmap=cmap, vmin=0, vmax=0.5,
    rasterized=True)
ax11.axvline(3*bar_duration, c='w', lw=0.5)
ax11.axhline(snareFreq, c='w', lw=0.5)
ax11.axhline(wdBlkFreq, c='w', lw=0.5)
ax11.set_title('average power')

ax11.set_xlabel('time (s)')
ax11.set_ylabel('freq. (Hz)')

ax12 = fig.add_subplot(gs[2,1], sharex=ax11, sharey=ax11)
pc12 = ax12.pcolormesh(tf_t, tf_f, scipy.ndimage.convolve1d(
    20*np.log10(wdBlk_tf_avg),
    weights=np.r_[4*[-0.125], 1, 4*[-0.125]],
    axis=0, mode='reflect'), cmap=cmap, vmin=0, vmax=0.5,
    rasterized=True)
ax12.axvline(3*bar_duration, c='w', lw=0.5)
ax12.axhline(snareFreq, c='w', lw=0.5)
ax12.axhline(wdBlkFreq, c='w', lw=0.5)
ax12.set_title('average power')
ax12.set_xlabel('time (s)')

pc_ax1 = fig.add_subplot(gs[3,:])
plt.colorbar(pc11, cax=pc_ax1, label='power (dB)', orientation='horizontal')

ax21 = fig.add_subplot(gs[4,0], sharex=ax11, sharey=ax11)
pc21 = ax21.pcolormesh(tf_t, tf_f, snare_corr_tf, cmap='coolwarm',
        vmin=-0.2, vmax=0.2, rasterized=True)
ax21.axvline(3*bar_duration, c='w', lw=0.5)
ax21.axhline(snareFreq, c='w', lw=0.5)
ax21.axhline(wdBlkFreq, c='w', lw=0.5)
ax21.set_title('snare correlation')
ax21.set_ylabel('freq. (Hz)')

ax22 = fig.add_subplot(gs[4,1], sharex=ax11, sharey=ax11)
pc22 = ax22.pcolormesh(tf_t, tf_f, wdBlk_corr_tf, cmap='coolwarm',
        vmin=-0.2, vmax=0.2, rasterized=True)
ax22.axvline(3*bar_duration, c='w', lw=0.5)
ax22.axhline(snareFreq, c='w', lw=0.5)
ax22.axhline(wdBlkFreq, c='w', lw=0.5)
ax22.set_title('wdBlk correlation')


ax31 = fig.add_subplot(gs[5,0], sharex=ax11, sharey=ax11)
pc31 = ax31.pcolormesh(tf_t, tf_f, snare_pcorr_tf, cmap='coolwarm',
        vmin=-0.2, vmax=0.2, rasterized=True)
ax31.axvline(3*bar_duration, c='w', lw=0.5)
ax31.axhline(snareFreq, c='w', lw=0.5)
ax31.axhline(wdBlkFreq, c='w', lw=0.5)
ax31.set_title('snare partial correlation')

ax31.set_xlabel('time (s)')
ax31.set_ylabel('freq. (Hz)')

ax32 = fig.add_subplot(gs[5,1], sharex=ax11, sharey=ax11)
pc32 = ax32.pcolormesh(tf_t, tf_f, wdBlk_pcorr_tf, cmap='coolwarm',
        vmin=-0.2, vmax=0.2, rasterized=True)
ax32.axvline(3*bar_duration, c='w', lw=0.5)
ax32.axhline(snareFreq, c='w', lw=0.5)
ax32.axhline(wdBlkFreq, c='w', lw=0.5)
ax32.set_title('snare partial correlation')

ax32.set_xlabel('time (s)')

pc_ax2 = fig.add_subplot(gs[6,:])
plt.colorbar(pc21, cax=pc_ax2, label='correlation', orientation='horizontal')

fig.tight_layout(pad=0.3, h_pad=0, w_pad=0)
fig.savefig(os.path.join(args.result_folder, 'FFTcSPoC_Results_%s.pdf' % state))
fig.savefig(os.path.join(args.result_folder, 'FFTcSPoC_Results_%s.png' % state))
