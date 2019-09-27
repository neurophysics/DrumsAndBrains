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
from tqdm import tqdm, trange
import SPoC

########## define the type of analysis here #########
corrtype = 'all' # 'single' or 'all'
N_bootstrap = 500
snare_N_SSD = 6
wdBlk_N_SSD = snare_N_SSD
#####################################################

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
        'r') as fl:
    snare_quot = fl['snare_quot']
    snare_filt = fl['snare_filt']
    wdBlk_quot = fl['wdBlk_quot']
    wdBlk_filt = fl['wdBlk_filt']

snare_filt /= np.sqrt(np.sum(snare_filt**2, 0))
wdBlk_filt /= np.sqrt(np.sum(wdBlk_filt**2, 0))

snare_all_csd = []
wdBlk_all_csd = []
snare_deviation = []
wdBlk_deviation = []

# read the oscillatory data from the silence period
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/prepare_FFTcSPoC.npz', 'r') as fl:
            snare_all_csd.append(fl['snare_all_csd'])
            wdBlk_all_csd.append(fl['wdBlk_all_csd'])
            snare_deviation.append(fl['snare_deviation'])
            wdBlk_deviation.append(fl['wdBlk_deviation'])
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

# apply the SSD filters ############################
snare_all_csd = [np.dot(
    snare_filt[:,:snare_N_SSD].T, np.dot(
        snare_filt[:,:snare_N_SSD].T, csd_now.T))
    for csd_now in snare_all_csd]
wdBlk_all_csd = [np.dot(
    wdBlk_filt[:,:wdBlk_N_SSD].T, np.dot(
        wdBlk_filt[:,:wdBlk_N_SSD].T, csd_now.T))
    for csd_now in wdBlk_all_csd]
####################################################

f = np.fft.rfftfreq(12*s_rate, d=1./s_rate)
f = f[np.all([f>=0, f<=10], 0)]

# define the frequencies to analyse
# define some constants
QPM = 140 # quarter notes per minute
SPQ = 60./QPM # seconds per quarter note
bar_duration = SPQ*4 # every bar consists of 4 quarter notes

# get the frequencies of the snaredrum (duple) and woodblock (triple) beats
snareFreq = 2./bar_duration
wdBlkFreq = 3./bar_duration
snareIdx = np.abs(f - snareFreq).argmin()
wdBlkIdx = np.abs(f - wdBlkFreq).argmin()
thetaIdx = np.all([f>=1, f<=2.5],0)
thetaIdx = np.all([thetaIdx, ~np.in1d(np.arange(len(f)),
    np.r_[snareIdx, wdBlkIdx]) ],0)
######################################

"""
#calculate cSPoC for every individual frequency
snare_corr_2_f = []
snare_corr_f = []
snare_w_f = []
wdBlk_corr_2_f = []
wdBlk_corr_f = []
wdBlk_w_f = []

if corrtype=='single':
    # calculate the single-trial correlation betweem power and behaviour
    snare_z = [np.abs(z_now - np.median(z_now)).argsort().argsort()
            for z_now in snare_deviation]
    wdBlk_z = [np.abs(z_now - np.median(z_now)).argsort().argsort()
            for z_now in wdBlk_deviation]
    SPoCf = SPoC.pSPoCr2_avg(bestof=100)
elif corrtype == 'all':
    # calculate the correlation between power and behaviour across all
    # trials
    snare_z = np.hstack([np.abs(z_now - np.median(z_now))
        for z_now in snare_deviation]).argsort().argsort()
    wdBlk_z = np.hstack([np.abs(z_now - np.median(z_now))
        for z_now in wdBlk_deviation]).argsort().argsort()
    SPoCf = SPoC.pSPoCr2(bestof=20)

snare_contrast_covs = [scipy.ndimage.convolve1d(c.real,
    np.r_[1,1,1,1,0,1,1,1,1]/8., axis=-1)
    for c in snare_all_csd]
wdBlk_contrast_covs = [scipy.ndimage.convolve1d(c.real,
    np.r_[1,1,1,1,0,1,1,1,1]/8., axis=-1)
    for c in wdBlk_all_csd]

try:
    with np.load('Results/SPoCt_small.npz', 'rb') as fl:
        snare_t_corr2 = fl['snare_t_corr2']
        snare_t_w = fl['snare_t_w']
        wdBlk_t_corr2 = fl['wdBlk_t_corr2']
        wdBlk_t_w = fl['wdBlk_t_w']
        random_t_corr2 = fl['random_t_corr2']
except:
    SPoCt2 = SPoC.SPoCt2() 
    snare_t_corr2 = []
    snare_t_w = []
    wdBlk_t_corr2 = []
    wdBlk_t_w = []
    ###
    use_idx = np.arange(4, len(f) - 4, 1)
    ###
    for i in tqdm(use_idx, desc='cSPoC per freq'):
        if corrtype=='all':
            corr2, w = SPoCt2(
                    np.concatenate(
                        [c[:,:,i].real for c in snare_all_csd], -1),
                    np.concatenate(
                        [c[:,:,np.arange(i-4, i+5, 1)[
                            np.arange(i-4, i+5,1)!=i]].real
                            for c in snare_all_csd], -1).swapaxes(0,2),
                snare_z, num = 1)
            snare_t_corr2.append(corr2)
            snare_t_w.append(w)
            corr2, w = SPoCt2(
                    np.concatenate(
                        [c[:,:,i].real for c in wdBlk_all_csd], -1),
                    np.concatenate(
                        [c[:,:,np.arange(i-4, i+5, 1)[
                            np.arange(i-4, i+5,1)!=i]].real
                            for c in wdBlk_all_csd], -1).swapaxes(0,2),
                wdBlk_z, num = 1)
            wdBlk_t_corr2.append(corr2)
            wdBlk_t_w.append(w)
    # get a random result to calculate the mean and standard deviation unde
    # random conditions
    random_t_corr2 = []
    for _ in trange(100, desc='bootstrapping std'): # calculate 100 iterations
        i = int(np.random.choice(use_idx, size=1))
        corr2, w = SPoCt2(
                np.concatenate(
                    [c[:,:,i].real for c in snare_all_csd], -1),
                np.concatenate(
                    [c[:,:,np.arange(i-4, i+5, 1)[
                        np.arange(i-4, i+5,1)!=i]].real
                        for c in snare_all_csd], -1).swapaxes(0,2),
            np.random.choice(snare_z, size=len(snare_z), replace=False), num = 1)
        random_t_corr2.append(corr2)
    # save the results
    np.savez('Results/SPoCt_small.npz',
            snare_t_corr2 = snare_t_corr2,
            snare_t_w = snare_t_w,
            wdBlk_t_corr2 = wdBlk_t_corr2,
            wdBlk_t_w = wdBlk_t_w,
            random_t_corr2 = random_t_corr2)

#tcorr2_mean = np.mean([snare_t_corr2, wdBlk_t_corr2])
#tcorr2_std = np.std([snare_t_corr2, wdBlk_t_corr2])
tcorr2_mean = np.mean(random_t_corr2)
tcorr2_std = np.std(random_t_corr2)

# plot the results
snare_z = (np.array(snare_t_corr2) - tcorr2_mean)/tcorr2_std
wdBlk_z = (np.array(wdBlk_t_corr2) - tcorr2_mean)/tcorr2_std

fig = plt.figure(figsize=(5.51181, 3))
ax1 = fig.add_subplot(121)
ax1.plot(f[use_idx], snare_z, 'b-')
ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
ax2.plot(f[use_idx], wdBlk_z, 'r-')
ax1.set_xlabel('frequency (Hz)')
ax2.set_xlabel('frequency (Hz)')
ax1.set_ylabel('z score')

z_thresh = -scipy.stats.norm.ppf(0.025/(2*len(use_idx)))
ax1.axhline(z_thresh, ls='--', lw=0.5, c='k')
ax2.axhline(z_thresh, ls='--', lw=0.5, c='k')

ax1.set_title('duple cue trials')
ax2.set_title('triple cue trials')

fig.tight_layout(pad=0.3)

# calculate through all frequencies
for i in trange(len(f), desc='cSPoC per freq'):
    if corrtype=='all':
        corr2, w = SPoCf(
                np.concatenate(
                    [c[:,:,i].real for c in snare_all_csd], -1),
                np.concatenate(
                    [c[:,:,i].real for c in snare_contrast_covs], -1),
            snare_z,
            num = 1)
        corr = SPoC._partial_corr_grad(w,
                np.concatenate([c[:,:,i].real for c in snare_all_csd], -1),
                np.concatenate([c[:,:,i].real for c in snare_contrast_covs], -1),
                snare_z)[0]
        snare_corr_2_f.append(corr2)
        snare_corr_f.append(corr)
        snare_w_f.append(w)
        corr2, w = SPoCf(
                np.concatenate(
                    [c[:,:,i].real for c in wdBlk_all_csd], -1),
                np.concatenate(
                    [c[:,:,i].real for c in wdBlk_contrast_covs], -1),
            wdBlk_z,
            num = 1)
        corr = SPoC._partial_corr_grad(w,
                np.concatenate([c[:,:,i].real for c in wdBlk_all_csd], -1),
                np.concatenate([c[:,:,i].real for c in wdBlk_contrast_covs], -1),
                wdBlk_z)[0]
        wdBlk_corr_2_f.append(corr2)
        wdBlk_corr_f.append(corr)
        wdBlk_w_f.append(w)
    elif corrtype=='single':
        corr2, w = SPoCf([c[:,:,i].real for c in snare_all_csd],
            snare_z,
            num = 1)
        corr = np.mean([SPoC._corr_grad(w, c[:,:,i].real, z)[0]
            for c,z in zip(snare_all_csd, snare_z)])
        snare_corr_2_f.append(corr2)
        snare_corr_f.append(corr)
        snare_w_f.append(w)
        corr2, w = SPoCf([c[:,:,i].real for c in wdBlk_all_csd],
            wdBlk_z,
            num = 1)
        corr = np.mean([SPoC._corr_grad(w, c[:,:,i].real, z)[0]
            for c,z in zip(wdBlk_all_csd, wdBlk_z)])
        wdBlk_corr_2_f.append(corr2)
        wdBlk_corr_f.append(corr)
        wdBlk_w_f.append(w)

# plot the correlation to behaviour <-> frequency relation
fig = plt.figure(figsize=(5.51, 3))
ax1 = fig.add_subplot(121)
ax1.plot(f, snare_corr_f, c='k')
ax1.axhline(0, lw=0.5)
ax1.axvline(snareFreq, c='b')
ax1.set_title('duple beat')
###
ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
ax2.plot(f, wdBlk_corr_f, c='k')
ax2.axhline(0, lw=0.5)
ax2.axvline(wdBlkFreq, c='r')
ax2.set_title('triple beat')
###
ax1.set_xlabel('frequency (Hz)')
ax1.set_xlabel('frequency (Hz)')
ax1.set_ylabel('correlation to deviation')
###
fig.tight_layout(pad=0.3)


1/0
"""
try:
    with np.load(os.path.join(args.result_folder,
        'FFTcSPoC_%s_%s.npz' % (state, corrtype)), 'rb') as fl:
            snare_corr_2 = fl['snare_corr_2']
            snare_w = fl['snare_w']
            snare_corr = fl['snare_corr'],
            snare_corr_2_boot = fl['snare_corr_2_boot']
            wdBlk_corr_2 = fl['wdBlk_corr_2']
            wdBlk_w = fl['wdBlk_w']
            wdBlk_corr = fl['wdBlk_corr'],
            wdBlk_corr_2_boot = fl['wdBlk_corr_2_boot']
except (IOError, KeyError):
    if corrtype=='single':
        # calculate the single-trial correlation betweem power and behaviour
        snare_z = [np.abs(z_now - np.median(z_now)).argsort().argsort()
                for z_now in snare_deviation]
        wdBlk_z = [np.abs(z_now - np.median(z_now)).argsort().argsort()
                for z_now in wdBlk_deviation]
        partial_SPoC = SPoC.pSPoCr2_avg(bestof=20)
        ###
        snare_corr_2, snare_w = partial_SPoC(
                [c[:,:,snareIdx].real for c in snare_silence_csd],
                [c[:,:,thetaIdx].mean(2).real for c in snare_silence_csd],
                snare_z, num=None)
        snare_corr = np.array([
            np.mean([SPoC._partial_corr_grad(
                w_now,
                c[:,:,snareIdx].real,
                c[:,:,thetaIdx].mean(2).real,
                z_now)[0]
                for c, z_now in zip(snare_silence_csd, snare_z)])
            for w_now in snare_w.T])
        snare_corr_2_boot = np.array([partial_SPoC(
                [c[:,:,snareIdx].real for c in snare_silence_csd],
                [c[:,:,thetaIdx].mean(2).real for c in snare_silence_csd],
            [np.random.choice(z_now, size=len(z_now), replace=False)
                for z_now in snare_z],num=1)[0]
            for _ in trange(N_bootstrap, desc='snarePCorrBoot')])
        wdBlk_corr_2, wdBlk_w = partial_SPoC(
                [c[:,:,wdBlkIdx].real for c in wdBlk_silence_csd],
                [c[:,:,thetaIdx].mean(2).real for c in wdBlk_silence_csd],
                wdBlk_z, num=None)
        wdBlk_corr = np.array([
            np.mean([SPoC._partial_corr_grad(
                w_now,
                c[:,:,wdBlkIdx].real,
                c[:,:,thetaIdx].mean(2).real,
                z_now)[0]
                for c, z_now in zip(wdBlk_silence_csd, wdBlk_z)])
            for w_now in wdBlk_w.T])
        wdBlk_corr_2_boot = np.array([partial_SPoC(
                [c[:,:,wdBlkIdx].real for c in wdBlk_silence_csd],
                [c[:,:,thetaIdx].mean(2).real for c in wdBlk_silence_csd],
            [np.random.choice(z_now, size=len(z_now), replace=False)
                for z_now in wdBlk_z],num=1)[0]
            for _ in trange(N_bootstrap, desc='wdBlkPCorrBoot')])
    elif corrtype == 'all':
        # calculate the correlation between power and behaviour across all
        # trials
        snare_z = np.hstack([np.abs(z_now - np.median(z_now))
            for z_now in snare_deviation]).argsort().argsort()
        wdBlk_z = np.hstack([np.abs(z_now - np.median(z_now))
            for z_now in wdBlk_deviation]).argsort().argsort()
        partial_SPoC = SPoC.pSPoCr2(bestof=20)
        ###
        snare_corr_2, snare_w = partial_SPoC(
                np.concatenate([c[:,:,snareIdx].real
                    for c in snare_silence_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in snare_silence_csd], -1),
                snare_z, num=None)
        snare_corr = np.array([
            SPoC._partial_corr_grad(
                w_now,
                np.concatenate([c[:,:,snareIdx].real
                    for c in snare_silence_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in snare_silence_csd], -1),
                snare_z)[0]
            for w_now in snare_w.T])
        snare_corr_2_boot = np.array([partial_SPoC(
                np.concatenate([c[:,:,snareIdx].real
                    for c in snare_silence_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in snare_silence_csd], -1),
                np.random.choice(snare_z, size=len(snare_z), replace=False),
                num=1)[0]
                for _ in trange(N_bootstrap, desc='snarePCorrBoot')])
        wdBlk_corr_2, wdBlk_w = partial_SPoC(
                np.concatenate([c[:,:,wdBlkIdx].real
                    for c in wdBlk_silence_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in wdBlk_silence_csd], -1),
                wdBlk_z, num=None)
        wdBlk_corr = np.array([
            SPoC._partial_corr_grad(
                w_now,
                np.concatenate([c[:,:,wdBlkIdx].real
                    for c in wdBlk_silence_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in wdBlk_silence_csd], -1),
                wdBlk_z)[0]
            for w_now in wdBlk_w.T])
        wdBlk_corr_2_boot = np.array([partial_SPoC(
                np.concatenate([c[:,:,wdBlkIdx].real
                    for c in wdBlk_silence_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in wdBlk_silence_csd], -1),
                np.random.choice(wdBlk_z, size=len(wdBlk_z), replace=False),
                num=1)[0]
                for _ in trange(N_bootstrap, desc='wdBlkPCorrBoot')])
    else:
        raise ValueError('corrtype must be \'single\' or \'all\'')
    np.savez(os.path.join(args.result_folder,
        'FFTcSPoC_%s_%s.npz' % (state, corrtype)),
            snare_corr_2 = snare_corr_2,
            snare_w = snare_w,
            snare_corr = snare_corr,
            snare_corr_2_boot = snare_corr_2_boot,
            wdBlk_corr_2 = wdBlk_corr_2,
            wdBlk_w = wdBlk_w,
            wdBlk_corr = wdBlk_corr,
            wdBlk_corr_2_boot = wdBlk_corr_2_boot
            )

# calculate raw p values (i.e., without correction for multiple comparisons)
snare_corr_p = ((snare_corr_2_boot>snare_corr_2[:,np.newaxis]).sum(
    -1) + 1)/float(N_bootstrap + 1)
wdBlk_corr_p = ((wdBlk_corr_2_boot>wdBlk_corr_2[:,np.newaxis]).sum(
    -1) + 1)/float(N_bootstrap + 1)

snare_silence_csd = []
wdBlk_silence_csd = []

# read the oscillatory data from the silence period
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/prepare_FFTcSPoC.npz', 'r') as fl:
            snare_silence_csd.append(fl['snare_%s_csd' % state])
            wdBlk_silence_csd.append(fl['wdBlk_%s_csd' % state])
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

#calculate the spatial patterns
snare_filter = snare_filt[:,:snare_N_SSD].dot(snare_w)
snare_pattern = scipy.linalg.lstsq(
        snare_filter.T.dot(
            np.vstack(snare_silence_csd)[...,snareIdx].mean(0).real).dot(
                snare_filter),
        snare_filter.T.dot(np.vstack(snare_silence_csd)[...,snareIdx].mean(
            0).real))[0]
snare_pattern /= np.abs(snare_pattern).max(-1)[:,np.newaxis]
#calculate the spatial patterns
wdBlk_filter = wdBlk_filt[:,:wdBlk_N_SSD].dot(wdBlk_w)
wdBlk_pattern = scipy.linalg.lstsq(
        wdBlk_filter.T.dot(
            np.vstack(wdBlk_silence_csd)[...,wdBlkIdx].mean(0).real).dot(
                wdBlk_filter),
        wdBlk_filter.T.dot(np.vstack(wdBlk_silence_csd)[...,wdBlkIdx].mean(
            0).real))[0]
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
                + '/prepare_FFTcSPoC.npz', 'r') as fl:
            best_snare_listen_trials.append(
                    np.tensordot(
                        snare_filt[:,:snare_N_SSD].dot(snare_w[:,0]),
                        fl['snare_listen_trials'],
                        axes=(0,0)))
            best_snare_silence_trials.append(
                    np.tensordot(
                        snare_filt[:,:snare_N_SSD].dot(snare_w[:,0]),
                        fl['snare_silence_trials'],
                        axes=(0,0)))
            best_wdBlk_listen_trials.append(
                    np.tensordot(
                        wdBlk_filt[:,:wdBlk_N_SSD].dot(wdBlk_w[:,0]),
                        fl['wdBlk_listen_trials'],
                        axes=(0,0)))
            best_wdBlk_silence_trials.append(
                    np.tensordot(
                        wdBlk_filt[:,:wdBlk_N_SSD].dot(wdBlk_w[:,0]),
                        fl['wdBlk_silence_trials'],
                        axes=(0,0)))
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

# stitch them together
snareData = [np.vstack([b,s])
        for b,s in zip(best_snare_listen_trials, best_snare_silence_trials)]
wdBlkData = [np.vstack([b,s])
        for b,s in zip(best_wdBlk_listen_trials, best_wdBlk_silence_trials)]

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

#calculate the S-transforms
tf_coords, snare_tf = zip(*[meet.tf.gft(scipy.signal.detrend(tr, axis=0,
    type='constant'), axis=0, sampling=custom_sampling)
    for tr in tqdm(snareData, desc='snareTF')])
tf_coords, wdBlk_tf = zip(*[meet.tf.gft(scipy.signal.detrend(tr, axis=0,
    type='constant'), axis=0, sampling=custom_sampling)
    for tr in tqdm(wdBlkData, desc='wdBlkTF')])
tf_coords = tf_coords[0]

# take the amplitude values only
snare_tf = [np.abs(tf) for tf in snare_tf]
wdBlk_tf = [np.abs(tf) for tf in wdBlk_tf]
# calculate the average
snare_tf_avg = np.mean(np.vstack(snare_tf), 0).reshape(-1, 240)
wdBlk_tf_avg = np.mean(np.vstack(wdBlk_tf), 0).reshape(-1, 240)

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

snare_con_power = [np.log(snare_filter[:,0].dot(snare_filter[:,0].dot(
    c[...,thetaIdx].mean(-1).T.real)))
    for c in snare_silence_csd]
wdBlk_con_power = [np.log(wdBlk_filter[:,0].dot(wdBlk_filter[:,0].dot(
    c[...,thetaIdx].mean(-1).T.real)))
    for c in wdBlk_silence_csd]

if corrtype == 'single':
    vmax = 0.05
    snare_corr_tf = np.mean([
        calc_corr(
            np.log(tf**2),
            np.abs(z_now - np.median(z_now)).argsort().argsort())
        for tf, z_now in zip(snare_tf, snare_deviation)], 0
        ).reshape(-1, 240)
    wdBlk_corr_tf = np.mean([
        calc_corr(
            np.log(tf**2),
            np.abs(z_now - np.median(z_now)).argsort().argsort())
        for tf, z_now in zip(wdBlk_tf, wdBlk_deviation)], 0
        ).reshape(-1, 240)
    snare_pcorr_tf = np.mean([
        calc_pcorr(
            np.log(tf**2),
            Y,
            np.abs(z_now - np.median(z_now)).argsort().argsort())
        for tf, Y, z_now in zip(snare_tf, snare_con_power, snare_deviation)], 0
        ).reshape(-1, 240)
    wdBlk_pcorr_tf = np.mean([
        calc_pcorr(
            np.log(tf**2),
            Y,
            np.abs(z_now - np.median(z_now)).argsort().argsort())
        for tf, Y, z_now in zip(wdBlk_tf, wdBlk_con_power, wdBlk_deviation)], 0
        ).reshape(-1, 240)
elif corrtype == 'all':
    vmax = 0.1
    snare_corr_tf = calc_corr(
            np.vstack([np.log(tf**2) for tf in snare_tf]),
            np.hstack([np.abs(z_now - np.median(z_now))
                for z_now in snare_deviation]).argsort().argsort()
            ).reshape(-1, 240)
    wdBlk_corr_tf = calc_corr(
            np.vstack([np.log(tf**2) for tf in wdBlk_tf]),
            np.hstack([np.abs(z_now - np.median(z_now))
                for z_now in wdBlk_deviation]).argsort().argsort()
            ).reshape(-1, 240)
    snare_pcorr_tf = calc_pcorr(
            np.vstack([np.log(tf**2) for tf in snare_tf]),
            np.hstack(snare_con_power),
            np.hstack([np.abs(z_now - np.median(z_now))
                for z_now in snare_deviation]).argsort().argsort()
            ).reshape(-1, 240)
    wdBlk_pcorr_tf = calc_pcorr(
            np.vstack([np.log(tf**2) for tf in wdBlk_tf]),
            np.hstack(wdBlk_con_power),
            np.hstack([np.abs(z_now - np.median(z_now))
                for z_now in wdBlk_deviation]).argsort().argsort()
            ).reshape(-1, 240)

snare_X, snare_Y, snare_Z = zip(*[meet.sphere.potMap(chancoords, p)
    for p in snare_pattern])
wdBlk_X, wdBlk_Y, wdBlk_Z = zip(*[meet.sphere.potMap(chancoords, p)
    for p in wdBlk_pattern])

tf_f = np.fft.fftfreq(snareData[0].shape[0], d=1./s_rate)[
        np.unique(tf_coords[0]).astype(int)]
tf_t = np.linspace(0, 4*bar_duration, 240)

fig = plt.figure(figsize=(3.54331,5))
gs = mpl.gridspec.GridSpec(3,2, height_ratios=[0.6,1.1,1.1])

gs00 = mpl.gridspec.GridSpecFromSubplotSpec(2,2, gs[0,:], height_ratios=[1,0.12],
        wspace=0.2, hspace=0)

ax00 = fig.add_subplot(gs00[0,0], frameon=False)
pc00 = ax00.pcolormesh(snare_X[0], snare_Y[0], snare_Z[0], cmap='coolwarm',
        vmin=-1, vmax=1, rasterized=True)
ax00.contour(snare_X[0], snare_Y[0], snare_Z[0], levels=[0], colors='w')
meet.sphere.addHead(ax00)
ax00.tick_params(**blind_ax)
ax00.set_title('duple beat ($p=%.3f$)' % snare_corr_p[0])

ax01 = fig.add_subplot(gs00[0,1], frameon=False, sharex=ax00, sharey=ax00)
pc01 = ax01.pcolormesh(wdBlk_X[0], wdBlk_Y[0], wdBlk_Z[0], cmap='coolwarm',
        vmin=-1, vmax=1, rasterized=True)
ax01.contour(wdBlk_X[0], wdBlk_Y[0], wdBlk_Z[0], levels=[0], colors='w')
meet.sphere.addHead(ax01)
ax01.tick_params(**blind_ax)
ax01.set_title('triple beat ($p=%.3f$)' % wdBlk_corr_p[0])

ax00.set_xlim([-3,3])
ax00.set_ylim([-1.3,1.5])

pc_ax0 = fig.add_subplot(gs00[1,:])
pat_cbar = plt.colorbar(pc00, cax=pc_ax0, label='amplitude',
        orientation='horizontal', ticks=[-1,0,1])
pat_cbar.ax.set_xticklabels(['$-$', '$0$', '$+$'])
pat_cbar.ax.axvline(0.5, c='w')

gs10 = mpl.gridspec.GridSpecFromSubplotSpec(2,2, gs[1,:],
        height_ratios=[1,0.1], hspace=1.0, wspace=0.2)
ax11 = fig.add_subplot(gs10[0,0])
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

ax12 = fig.add_subplot(gs10[0,1], sharex=ax11, sharey=ax11)
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

pc_ax1 = fig.add_subplot(gs10[1,:])
plt.colorbar(pc11, cax=pc_ax1, label='SNNR (dB)', orientation='horizontal',
        ticks=mpl.ticker.MaxNLocator(5))


gs20 = mpl.gridspec.GridSpecFromSubplotSpec(2,2, gs[2,:],
        height_ratios=[1,0.1], hspace=1.0, wspace=0.2)
ax21 = fig.add_subplot(gs20[0,0], sharex=ax11, sharey=ax11)
pc21 = ax21.pcolormesh(tf_t, tf_f, snare_pcorr_tf, cmap='coolwarm',
        vmin=-vmax, vmax=vmax, rasterized=True)
ax21.axvline(3*bar_duration, c='w', lw=0.5)
ax21.axhline(snareFreq, c='w', lw=0.5)
ax21.axhline(wdBlkFreq, c='w', lw=0.5)
ax21.set_title('duple beat vs deviation')

ax21.set_ylabel('freq. (Hz)')
ax21.set_xlabel('time (s)')

ax22 = fig.add_subplot(gs20[0,1], sharex=ax11, sharey=ax11)
pc22 = ax22.pcolormesh(tf_t, tf_f, wdBlk_pcorr_tf, cmap='coolwarm',
        vmin=-vmax, vmax=vmax, rasterized=True)
ax22.axvline(3*bar_duration, c='w', lw=0.5)
ax22.axhline(snareFreq, c='w', lw=0.5)
ax22.axhline(wdBlkFreq, c='w', lw=0.5)
ax22.set_title('triple beat vs deviation')

ax22.set_xlabel('time (s)')

pc_ax2 = fig.add_subplot(gs20[1,:])
plt.colorbar(pc21, cax=pc_ax2, label='partial correlation coefficient',
        orientation='horizontal',
        ticks=mpl.ticker.MaxNLocator(5))

ax21.set_ylim([0.5,5])

fig.tight_layout(pad=0.3, h_pad=1, w_pad=1)
fig.savefig(os.path.join(args.result_folder,
    'FFTcSPoC_Results_%s_%s.pdf' % (state, corrtype)))
fig.savefig(os.path.join(args.result_folder,
    'FFTcSPoC_Results_%s_%s.png' % (state, corrtype)))
