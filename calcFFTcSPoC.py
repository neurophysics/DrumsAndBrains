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
            snare_silence_csd_con.append(f['snare_listen_csd_con'])
            wdBlk_silence_csd_con.append(f['wdBlk_listen_csd_con'])
            snare_silence_csd.append(f['snare_listen_csd'])
            wdBlk_silence_csd.append(f['wdBlk_listen_csd'])
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

N_bootstrap=50
from tqdm import trange,tqdm

# This turned out to be non-significant
###
pSPoCr2_avg = SPoC.pSPoCr2_avg()
snare_z = [np.abs(z_now).argsort().argsort() for z_now in snare_deviation]
snare_corr, snare_w = pSPoCr2_avg(
        [c.real for c in snare_silence_csd],
        [c.real for c in snare_silence_csd_con],
        snare_z, num=None)
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
        wdBlk_z, num=None)
wdBlk_corr_boot = np.array([pSPoCr2_avg(
    [c.real for c in wdBlk_silence_csd],
    [c.real for c in wdBlk_silence_csd_con],
    [np.random.choice(z_now, size=len(z_now), replace=False)
        for z_now in wdBlk_z],num=1)[0]
    for _ in trange(N_bootstrap)])
###

1/0

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

1/0

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
snare_tf_avg = np.mean(snare_tf, 0). reshape(-1, 240)
wdBlk_tf_avg = np.mean(wdBlk_tf, 0). reshape(-1, 240)

from tqdm import tqdm 
snare_corr_tf = np.array([np.corrcoef(np.log(tf_now**2),
    snare_deviation_all)[0,1]
    for tf_now in tqdm(snare_tf.T)]).reshape(-1, 240)
wdBlk_corr_tf = np.array([np.corrcoef(np.log(tf_now**2),
    wdBlk_deviation_all)[0,1]
    for tf_now in tqdm(wdBlk_tf.T)]).reshape(-1, 240)

tf_f = np.fft.fftfreq(snareData.shape[0], d=1./s_rate)[np.unique(tf_coords[0]).astype(int)]
tf_t = np.linspace(0, 4*bar_duration, 240)

fig = plt.figure(figsize=(12,3))
ax1 = fig.add_subplot(141)
pc1 = ax1.pcolormesh(tf_t, tf_f, scipy.ndimage.convolve1d(
    20*np.log10(snare_tf_avg),
    weights=np.r_[4*[-0.125], 1, 4*[-0.125]],
    axis=0, mode='reflect'), cmap=cmap, vmin=0, vmax=0.5, rasterized=True)
ax1.axvline(3*bar_duration, c='w', lw=0.5)
ax1.axhline(snareFreq, c='w', lw=0.5)
ax1.axhline(wdBlkFreq, c='w', lw=0.5)
ax1.set_title('snare trials: average power')
plt.colorbar(pc1, ax=ax1, label='power difference (dB)')

ax2 = fig.add_subplot(142, sharex=ax1, sharey=ax1)
pc2 = ax2.pcolormesh(tf_t, tf_f, snare_corr_tf, cmap='coolwarm',
        vmin=-0.15, vmax=0.15, rasterized=True)
ax2.axvline(3*bar_duration, c='w', lw=0.5)
ax2.axhline(snareFreq, c='w', lw=0.5)
ax2.axhline(wdBlkFreq, c='w', lw=0.5)
ax2.set_title('snare correlation (p=%.3f)' % snare_corr_p[0])
plt.colorbar(pc2, ax=ax2, label='correlation coefficient')
ax3 = fig.add_subplot(143, sharex=ax1, sharey=ax1)
pc3 = ax3.pcolormesh(tf_t, tf_f, scipy.ndimage.convolve1d(
    20*np.log10(wdBlk_tf_avg),
    weights=np.r_[4*[-0.125], 1, 4*[-0.125]],
    axis=0, mode='reflect'), cmap=cmap, vmin=0, vmax=0.5, rasterized=True)
ax3.axvline(3*bar_duration, c='w', lw=0.5)
ax3.axhline(snareFreq, c='w', lw=0.5)
ax3.axhline(wdBlkFreq, c='w', lw=0.5)
ax3.set_title('wdBlk trials: average power')
plt.colorbar(pc3, ax=ax3, label='power difference (dB)')

ax4 = fig.add_subplot(144, sharex=ax1, sharey=ax1)
pc4 = ax4.pcolormesh(tf_t, tf_f, wdBlk_corr_tf, cmap='coolwarm',
        vmin=-0.15, vmax=0.15, linewidth=0, rasterized=True)
ax4.axvline(3*bar_duration, c='w', lw=0.5)
ax4.axhline(snareFreq, c='w', lw=0.5)
ax4.axhline(wdBlkFreq, c='w', lw=0.5)
ax4.set_title('wdBlk correlation (p=%.3f)' % wdBlk_corr_p[0])
plt.colorbar(pc4, ax=ax4, label='correlation coefficient')
ax4.set_xlabel('time (ms)')
ax1.set_ylabel('frequency (Hz)')
ax2.set_ylabel('frequency (Hz)')
ax3.set_ylabel('frequency (Hz)')
ax4.set_ylabel('frequency (Hz)')
fig.tight_layout()
fig.savefig(os.path.join(args.result_folder, 'cSPoC_Results.pdf'))
