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
N_bootstrap = 1000
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
snare_listen_csd = []
wdBlk_listen_csd = []
snare_silence_csd = []
wdBlk_silence_csd = []
snare_deviation = []
wdBlk_deviation = []

# read the oscillatory data from the silence period
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/prepare_FFTcSPoC.npz', 'r') as fl:
            snare_all_csd.append(fl['snare_all_csd'])
            wdBlk_all_csd.append(fl['wdBlk_all_csd'])
            snare_listen_csd.append(fl['snare_listen_csd'])
            wdBlk_listen_csd.append(fl['wdBlk_listen_csd'])
            snare_silence_csd.append(fl['snare_silence_csd'])
            wdBlk_silence_csd.append(fl['wdBlk_silence_csd'])
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
snare_listen_csd = [np.dot(
    snare_filt[:,:snare_N_SSD].T, np.dot(
        snare_filt[:,:snare_N_SSD].T, csd_now.T))
    for csd_now in snare_listen_csd]
wdBlk_listen_csd = [np.dot(
    wdBlk_filt[:,:wdBlk_N_SSD].T, np.dot(
        wdBlk_filt[:,:wdBlk_N_SSD].T, csd_now.T))
    for csd_now in wdBlk_listen_csd]
snare_silence_csd = [np.dot(
    snare_filt[:,:snare_N_SSD].T, np.dot(
        snare_filt[:,:snare_N_SSD].T, csd_now.T))
    for csd_now in snare_silence_csd]
wdBlk_silence_csd = [np.dot(
    wdBlk_filt[:,:wdBlk_N_SSD].T, np.dot(
        wdBlk_filt[:,:wdBlk_N_SSD].T, csd_now.T))
    for csd_now in wdBlk_silence_csd]
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
thetaIdx = np.all([f>=1, f<=2],0)
thetaIdx = np.all([thetaIdx, ~np.in1d(np.arange(len(f)),
    np.r_[snareIdx, wdBlkIdx]) ],0)

######################################

try:
    with np.load(os.path.join(args.result_folder,
        'FFTcSPoCv2.npz'), 'rb') as fl:
            snare_all_corr2 = fl['snare_all_corr2']
            snare_all_corr = fl['snare_all_corr']
            snare_listen_corr = fl['snare_listen_corr']
            snare_silence_corr = fl['snare_silence_corr']
            snare_all_corr2_boot = fl['snare_all_corr2_boot']
            snare_all_corr_boot = fl['snare_all_corr_boot']
            snare_listen_corr_boot = fl['snare_listen_corr_boot']
            snare_silence_corr_boot = fl['snare_silence_corr_boot']
            wdBlk_all_corr2 = fl['wdBlk_all_corr2']
            wdBlk_all_corr = fl['wdBlk_all_corr']
            wdBlk_listen_corr = fl['wdBlk_listen_corr']
            wdBlk_silence_corr = fl['wdBlk_silence_corr']
            wdBlk_all_corr2_boot = fl['wdBlk_all_corr2_boot']
            wdBlk_all_corr_boot = fl['wdBlk_all_corr_boot']
            wdBlk_listen_corr_boot = fl['wdBlk_listen_corr_boot']
            wdBlk_silence_corr_boot = fl['wdBlk_silence_corr_boot']
            snare_w = fl['snare_w']
            wdBlk_w = fl['wdBlk_w']
except (IOError, KeyError):
    # calculate the correlation between power and behaviour across all
    # trials
    snare_z = np.hstack([np.abs(z_now - np.median(z_now))
        for z_now in snare_deviation]).argsort().argsort()
    wdBlk_z = np.hstack([np.abs(z_now - np.median(z_now))
        for z_now in wdBlk_deviation]).argsort().argsort()
    partial_SPoC = SPoC.pSPoCr2(bestof=20)
    ###################################################################
    ###################################################################
    snare_all_corr2, snare_w = partial_SPoC(
            np.concatenate([c[:,:,snareIdx].real
                for c in snare_all_csd], -1),
            np.concatenate([c[:,:,thetaIdx].mean(2).real
                for c in snare_all_csd], -1),
            snare_z, num=None)
    snare_all_corr = np.array([
        SPoC._partial_corr_grad(
            w_now,
            np.concatenate([c[:,:,snareIdx].real
                for c in snare_all_csd], -1),
            np.concatenate([c[:,:,thetaIdx].mean(2).real
                for c in snare_all_csd], -1),
            snare_z)[0]
        for w_now in snare_w.T])
    snare_listen_corr = np.array([
        SPoC._partial_corr_grad(
            w_now,
            np.concatenate([c[:,:,snareIdx].real
                for c in snare_listen_csd], -1),
            np.concatenate([c[:,:,thetaIdx].mean(2).real
                for c in snare_listen_csd], -1),
            snare_z)[0]
        for w_now in snare_w.T])
    snare_silence_corr = np.array([
        SPoC._partial_corr_grad(
            w_now,
            np.concatenate([c[:,:,snareIdx].real
                for c in snare_silence_csd], -1),
            np.concatenate([c[:,:,thetaIdx].mean(2).real
                for c in snare_silence_csd], -1),
            snare_z)[0]
        for w_now in snare_w.T])
    ################################
    # calculate a bootstrap result #
    ################################
    snare_all_corr2_boot = []
    snare_all_corr_boot = []
    snare_listen_corr_boot = []
    snare_silence_corr_boot = []
    for _ in trange(N_bootstrap, desc='snare bootstrap'):
        ztemp = np.random.choice(snare_z, size=len(snare_z),
                replace=False)
        temp_all_boot2, temp_w = partial_SPoC(
                np.concatenate([c[:,:,snareIdx].real
                    for c in snare_all_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in snare_all_csd], -1),
                ztemp, num=1)
        temp_all_boot = SPoC._partial_corr_grad(temp_w,
                np.concatenate([c[:,:,snareIdx].real
                    for c in snare_all_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in snare_all_csd], -1),
                ztemp)[0]
        temp_listen_boot = SPoC._partial_corr_grad(temp_w,
                np.concatenate([c[:,:,snareIdx].real
                    for c in snare_listen_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in snare_listen_csd], -1),
                ztemp)[0]
        temp_silence_boot = SPoC._partial_corr_grad(temp_w,
                np.concatenate([c[:,:,snareIdx].real
                    for c in snare_silence_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in snare_silence_csd], -1),
                ztemp)[0]
        snare_all_corr2_boot.append(temp_all_boot2)
        snare_all_corr_boot.append(temp_all_boot)
        snare_listen_corr_boot.append(temp_listen_boot)
        snare_silence_corr_boot.append(temp_silence_boot)
    snare_all_corr2_boot = np.array(snare_all_corr2_boot)
    snare_all_corr_boot = np.array(snare_all_corr_boot)
    snare_listen_corr_boot = np.array(snare_listen_corr_boot)
    snare_silence_corr_boot = np.array(snare_silence_corr_boot)
    ###################################################################
    ###################################################################
    wdBlk_all_corr2, wdBlk_w = partial_SPoC(
            np.concatenate([c[:,:,wdBlkIdx].real
                for c in wdBlk_all_csd], -1),
            np.concatenate([c[:,:,thetaIdx].mean(2).real
                for c in wdBlk_all_csd], -1),
            wdBlk_z, num=None)
    wdBlk_all_corr = np.array([
        SPoC._partial_corr_grad(
            w_now,
            np.concatenate([c[:,:,wdBlkIdx].real
                for c in wdBlk_all_csd], -1),
            np.concatenate([c[:,:,thetaIdx].mean(2).real
                for c in wdBlk_all_csd], -1),
            wdBlk_z)[0]
        for w_now in wdBlk_w.T])
    wdBlk_listen_corr = np.array([
        SPoC._partial_corr_grad(
            w_now,
            np.concatenate([c[:,:,wdBlkIdx].real
                for c in wdBlk_listen_csd], -1),
            np.concatenate([c[:,:,thetaIdx].mean(2).real
                for c in wdBlk_listen_csd], -1),
            wdBlk_z)[0]
        for w_now in wdBlk_w.T])
    wdBlk_silence_corr = np.array([
        SPoC._partial_corr_grad(
            w_now,
            np.concatenate([c[:,:,wdBlkIdx].real
                for c in wdBlk_silence_csd], -1),
            np.concatenate([c[:,:,thetaIdx].mean(2).real
                for c in wdBlk_silence_csd], -1),
            wdBlk_z)[0]
        for w_now in wdBlk_w.T])
    ################################
    # calculate a bootstrap result #
    ################################
    wdBlk_all_corr2_boot = []
    wdBlk_all_corr_boot = []
    wdBlk_listen_corr_boot = []
    wdBlk_silence_corr_boot = []
    for _ in trange(N_bootstrap, desc='wdBlk bootstrap'):
        ztemp = np.random.choice(wdBlk_z, size=len(wdBlk_z),
                replace=False)
        temp_all_boot2, temp_w = partial_SPoC(
                np.concatenate([c[:,:,wdBlkIdx].real
                    for c in wdBlk_all_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in wdBlk_all_csd], -1),
                ztemp, num=1)
        temp_all_boot = SPoC._partial_corr_grad(temp_w,
                np.concatenate([c[:,:,wdBlkIdx].real
                    for c in wdBlk_all_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in wdBlk_all_csd], -1),
                ztemp)[0]
        temp_listen_boot = SPoC._partial_corr_grad(temp_w,
                np.concatenate([c[:,:,wdBlkIdx].real
                    for c in wdBlk_listen_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in wdBlk_listen_csd], -1),
                ztemp)[0]
        temp_silence_boot = SPoC._partial_corr_grad(temp_w,
                np.concatenate([c[:,:,wdBlkIdx].real
                    for c in wdBlk_silence_csd], -1),
                np.concatenate([c[:,:,thetaIdx].mean(2).real
                    for c in wdBlk_silence_csd], -1),
                ztemp)[0]
        wdBlk_all_corr2_boot.append(temp_all_boot2)
        wdBlk_all_corr_boot.append(temp_all_boot)
        wdBlk_listen_corr_boot.append(temp_listen_boot)
        wdBlk_silence_corr_boot.append(temp_silence_boot)
    wdBlk_all_corr2_boot = np.array(wdBlk_all_corr2_boot)
    wdBlk_all_corr_boot = np.array(wdBlk_all_corr_boot)
    wdBlk_listen_corr_boot = np.array(wdBlk_listen_corr_boot)
    wdBlk_silence_corr_boot = np.array(wdBlk_silence_corr_boot)
    # store the results
    np.savez(os.path.join(args.result_folder,
        'FFTcSPoCv2.npz'),
            snare_all_corr2 = snare_all_corr2,
            snare_all_corr = snare_all_corr,
            snare_listen_corr = snare_listen_corr,
            snare_silence_corr = snare_silence_corr,
            snare_all_corr2_boot = snare_all_corr2_boot,
            snare_all_corr_boot = snare_all_corr_boot,
            snare_listen_corr_boot = snare_listen_corr_boot,
            snare_silence_corr_boot = snare_silence_corr_boot,
            wdBlk_all_corr2 = wdBlk_all_corr2,
            wdBlk_all_corr = wdBlk_all_corr,
            wdBlk_listen_corr = wdBlk_listen_corr,
            wdBlk_silence_corr = wdBlk_silence_corr,
            wdBlk_all_corr2_boot = wdBlk_all_corr2_boot,
            wdBlk_all_corr_boot = wdBlk_all_corr_boot,
            wdBlk_listen_corr_boot = wdBlk_listen_corr_boot,
            wdBlk_silence_corr_boot = wdBlk_silence_corr_boot,
            snare_w = snare_w,
            wdBlk_w = wdBlk_w
            )

# calculate raw p values (i.e., without correction for multiple comparisons)
snare_all_corr_p = ((snare_all_corr_boot<snare_all_corr[:,np.newaxis]).sum(
    -1) + 1)/float(N_bootstrap + 1)
snare_listen_corr_p = ((snare_listen_corr_boot<snare_listen_corr[
    :,np.newaxis]).sum(-1) + 1)/float(N_bootstrap + 1)
snare_silence_corr_p = ((snare_silence_corr_boot<=snare_silence_corr[
    :,np.newaxis]).sum(-1) + 1)/float(N_bootstrap + 1)
wdBlk_all_corr_p = ((wdBlk_all_corr_boot<wdBlk_all_corr[:,np.newaxis]).sum(
    -1) + 1)/float(N_bootstrap + 1)
wdBlk_listen_corr_p = ((wdBlk_listen_corr_boot<wdBlk_listen_corr[
    :,np.newaxis]).sum(-1) + 1)/float(N_bootstrap + 1)
wdBlk_silence_corr_p = ((wdBlk_silence_corr_boot<=wdBlk_silence_corr[
    :,np.newaxis]).sum(-1) + 1)/float(N_bootstrap + 1)

snare_all_csd = []
wdBlk_all_csd = []

# read the oscillatory data from the silence period
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/prepare_FFTcSPoC.npz', 'r') as fl:
            snare_all_csd.append(fl['snare_all_csd'])
            wdBlk_all_csd.append(fl['wdBlk_all_csd'])
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

#calculate the spatial patterns
snare_filter = snare_filt[:,:snare_N_SSD].dot(snare_w)
snare_pattern = scipy.linalg.lstsq(
        snare_filter.T.dot(
            np.vstack(snare_all_csd)[...,snareIdx].mean(0).real).dot(
                snare_filter),
        snare_filter.T.dot(np.vstack(snare_all_csd)[...,snareIdx].mean(
            0).real))[0]
snare_pattern /= np.abs(snare_pattern).max(-1)[:,np.newaxis]
#calculate the spatial patterns
wdBlk_filter = wdBlk_filt[:,:wdBlk_N_SSD].dot(wdBlk_w)
wdBlk_pattern = scipy.linalg.lstsq(
        wdBlk_filter.T.dot(
            np.vstack(wdBlk_all_csd)[...,wdBlkIdx].mean(0).real).dot(
                wdBlk_filter),
        wdBlk_filter.T.dot(np.vstack(wdBlk_all_csd)[...,wdBlkIdx].mean(
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
    for c in snare_all_csd]
wdBlk_con_power = [np.log(wdBlk_filter[:,0].dot(wdBlk_filter[:,0].dot(
    c[...,thetaIdx].mean(-1).T.real)))
    for c in wdBlk_all_csd]

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

#######################
fig = plt.figure(figsize=(3.54331,2))
gs = mpl.gridspec.GridSpec(1,2)

gs00 = mpl.gridspec.GridSpecFromSubplotSpec(2,2, gs[0,:], height_ratios=[1,0.12],
        wspace=0.2, hspace=0)

ax00 = fig.add_subplot(gs00[0,0], frameon=False)
pc00 = ax00.pcolormesh(snare_X[0], snare_Y[0], snare_Z[0], cmap='coolwarm',
        vmin=-1, vmax=1, rasterized=True)
ax00.contour(snare_X[0], snare_Y[0], snare_Z[0], levels=[0], colors='w')
meet.sphere.addHead(ax00)
ax00.tick_params(**blind_ax)
ax00.set_title('duple beat ($p=%.3f$)' % snare_all_corr_p[0])

ax01 = fig.add_subplot(gs00[0,1], frameon=False, sharex=ax00, sharey=ax00)
pc01 = ax01.pcolormesh(wdBlk_X[0], wdBlk_Y[0], wdBlk_Z[0], cmap='coolwarm',
        vmin=-1, vmax=1, rasterized=True)
ax01.contour(wdBlk_X[0], wdBlk_Y[0], wdBlk_Z[0], levels=[0], colors='w')
meet.sphere.addHead(ax01)
ax01.tick_params(**blind_ax)
ax01.set_title('triple beat ($p=%.3f$)' % wdBlk_all_corr_p[0])

ax00.set_xlim([-1.3,1.3])
ax00.set_ylim([-1.3,1.5])

pc_ax0 = fig.add_subplot(gs00[1,:])
pat_cbar = plt.colorbar(pc00, cax=pc_ax0, label='amplitude',
        orientation='horizontal', ticks=[-1,0,1])
pat_cbar.ax.set_xticklabels(['$-$', '$0$', '$+$'])
pat_cbar.ax.axvline(0.5, c='w')

fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(args.result_folder,
    'FFTcSPoC_patterns.pdf'))
fig.savefig(os.path.join(args.result_folder,
    'FFTcSPoC_patterns.png'))
###################################

fig = plt.figure(figsize=(3.54331,5))
gs = mpl.gridspec.GridSpec(2,1)

gs10 = mpl.gridspec.GridSpecFromSubplotSpec(2,1, gs[0,:],
        height_ratios=[1,0.075], hspace=0.75, wspace=0.2)
ax11 = fig.add_subplot(gs10[0,0])
pc11 = ax11.pcolormesh(tf_t, tf_f, scipy.ndimage.convolve1d(
    20*np.log10(snare_tf_avg),
    weights=np.r_[4*[-0.125], 1, 4*[-0.125]],
    axis=0, mode='reflect'), cmap=cmap, vmin=0, vmax=0.5,
    rasterized=True)
ax11.axvline(3*bar_duration, c='k', lw=2, ymax=1, zorder=1000)
ax11.axhline(snareFreq, c='k', lw=2, zorder=100, alpha=0.2)
ax11.axhline(snareFreq, c='w', lw=0.5, zorder=101, ls='--')
ax11.set_title('signal-to-noise ratio (grand average)')

ax11.set_xlabel('time (s)')
ax11.set_ylabel('freq. (Hz)')


trans11 = mpl.transforms.blended_transform_factory(
    ax11.transData, ax11.transAxes)
t11 = ax11.text(1.5*bar_duration, 0.9, r'musical stimulus', ha='center',
        va='top', ma='center', transform=trans11,
        color='w', fontsize=10)
t11.set_bbox(dict(facecolor='k', alpha=0.4, edgecolor='None'))
t12 = ax11.text(3.5*bar_duration, 0.9, r'silence', ha='center',
        va='top', ma='center', transform=trans11,
        color='w', fontsize=10)
t12.set_bbox(dict(facecolor='k', alpha=0.4, edgecolor='None'))
pc_ax1 = fig.add_subplot(gs10[1,:])
plt.colorbar(pc11, cax=pc_ax1, label='SNNR (dB)', orientation='horizontal',
        ticks=mpl.ticker.MaxNLocator(5))

gs20 = mpl.gridspec.GridSpecFromSubplotSpec(2,1, gs[1,:],
        height_ratios=[1,0.075], hspace=0.75, wspace=0.2)
ax21 = fig.add_subplot(gs20[0,0], sharex=ax11, sharey=ax11)
pc21 = ax21.pcolormesh(tf_t, tf_f, snare_corr_tf, cmap='coolwarm',
        vmin=-vmax, vmax=vmax, rasterized=True)
ax21.axvline(3*bar_duration, c='k', lw=2, ymax=1, zorder=1000)
ax21.axhline(snareFreq, c='k', lw=2, zorder=100, alpha=0.2)
ax21.axhline(snareFreq, c='w', lw=0.5, zorder=101, ls='--')
ax21.set_title('correlation to deviation of performance'+'\n'+
        '(duple beat trials)')

ax21.set_ylabel('freq. (Hz)')
ax21.set_xlabel('time (s)')

trans21 = mpl.transforms.blended_transform_factory(
    ax21.transData, ax21.transAxes)
t21 = ax21.text(1.5*bar_duration, 0.9,
        r'musical stimulus' + '\n' + r'($p=%.3f$)' % snare_listen_corr_p[0],
        ha='center',
        va='top', ma='center', transform=trans21,
        color='w', fontsize=10)
t21.set_bbox(dict(facecolor='k', alpha=0.4, edgecolor='None'))
t22 = ax21.text(3.5*bar_duration, 0.9,
        r'silence' + '\n' + r'($p=%.3f$)' % snare_silence_corr_p[0],
        ha='center',
        va='top', ma='center', transform=trans21,
        color='w', fontsize=10)
t22.set_bbox(dict(facecolor='k', alpha=0.4, edgecolor='None'))

pc_ax2 = fig.add_subplot(gs20[1,:])
plt.colorbar(pc21, cax=pc_ax2, label='correlation coefficient',
        orientation='horizontal',
        ticks=mpl.ticker.MaxNLocator(5))

ax21.set_ylim([0.5,5])

fig.tight_layout(pad=0.3, h_pad=2.0)
fig.savefig(os.path.join(args.result_folder,
    'FFTcSPoC_TF_Results.pdf'))
fig.savefig(os.path.join(args.result_folder,
    'FFTcSPoC_TF_Results.png'))
