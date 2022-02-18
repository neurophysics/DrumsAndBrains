"""
prepare the SSD for a single subject

This script reads the EEG and behavioural data.

The Fourier transform is applied to the listen trials (first three bars
of the stimulus) after padding to 12 s to result in a frequency resolution
of 1/6 Hz.

SSD is prepared by:
    isolating snare and woodblock frequency from the FFT and applying the
    inverse transform, then calculation of covariance matrices for every
    trial = 'target'

    isolationg range of 1-2 Hz, applying inverse transform and calculation of
    covariance matrices for every single trial = 'contrast'

Results are stored for the subject in prepared_FFTSSD.npz

After running the script for every subject, proceed with calcFFTSSD.npz

The inputs are:
1. the EEG data of the subject (after artifact cleaning)
2. the behavioural data of the subject (in order to find the listening
    period)
"""

import numpy as np
import scipy.signal
import sys
import os.path
import helper_functions
import meet
import scipy.ndimage

data_folder = sys.argv[1]
subject = int(sys.argv[2])
result_folder = sys.argv[3]

s_rate = 1000 # sampling rate of the EEG

data_folder = os.path.join(data_folder, 'S%02d' % subject)
save_folder = os.path.join(result_folder, 'S%02d' % subject)

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# read data (from clean_data.npz, channels.txt, S{:02d}_eeg_all_files.vmrk,
# and behavioural_results.npz)
## read the cleaned EEG
eeg_fname = os.path.join(data_folder, 'clean_data.npz')
with np.load(eeg_fname) as npzfile:
    EEG = npzfile['clean_data']
    artifact_mask = npzfile['artifact_mask']

## read the channel names
channames = meet.sphere.getChannelNames(os.path.join(data_folder,
    '../channels.txt'))
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

## get session clocks
if os.path.exists(os.path.join(
    data_folder, 'S{:02d}_eeg_all_files.vmrk'.format(subject))):
    marker_fname = os.path.join(
            data_folder, 'S{:02d}_eeg_all_files.vmrk'.format(subject))
else:
    marker_fname = os.path.join(data_folder, 'S%02d_eeg.vmrk' % subject)

eeg_clocks = helper_functions.getSessionClocks(marker_fname)
eeg_clocks = [c for c in eeg_clocks if len(c) > 100]

assert len(eeg_clocks) == 6, '6 sessions expected'


## read behavioral data
with np.load(os.path.join(save_folder, 'behavioural_results.npz'),
        'r', allow_pickle=True, encoding='latin1') as f:
    snareCue_nearestClock = f['snareCue_nearestClock']
    snareCue_DevToClock = f['snareCue_DevToClock']
    wdBlkCue_nearestClock = f['wdBlkCue_nearestClock']
    wdBlkCue_DevToClock = f['wdBlkCue_DevToClock']
    snareCue_times = f['snareCue_times']
    wdBlkCue_times = f['wdBlkCue_times']
    bar_duration = f['bar_duration']
    snare_deviation = f['snare_deviation']
    wdBlk_deviation = f['wdBlk_deviation']

# data preprocesing
## find the sample of each Cue
snareCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
        snareCue_nearestClock, snareCue_DevToClock)
wdBlkCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
        wdBlkCue_nearestClock, wdBlkCue_DevToClock)

## get the sample indices at the start of the 3 'listening bars'
snareListenMarker = snareCue_pos - int(4*bar_duration*s_rate)
wdBlkListenMarker = wdBlkCue_pos - int(4*bar_duration*s_rate)

## get the temporal window
all_win = [0, int(4*bar_duration*s_rate)]
listen_win = [0, int(3*bar_duration*s_rate)]
silence_win = [int(3*bar_duration*s_rate), int(4*bar_duration*s_rate)]

## reject trials that contain rejected data segments
snareInlier = np.all(meet.epochEEG(artifact_mask, snareListenMarker,
    all_win), 0)
wdBlkInlier = np.all(meet.epochEEG(artifact_mask, wdBlkListenMarker,
    all_win), 0)
snareInlier_listen = np.all(meet.epochEEG(artifact_mask, snareListenMarker,
    listen_win), 0)
wdBlkInlier_listen = np.all(meet.epochEEG(artifact_mask, wdBlkListenMarker,
    listen_win), 0)
snareInlier_silence = np.all(meet.epochEEG(artifact_mask, snareListenMarker,
    silence_win), 0)
wdBlkInlier_silence = np.all(meet.epochEEG(artifact_mask, wdBlkListenMarker,
    silence_win), 0)

## get the frequencies of the snaredrum (duple) and woodblock (triple) beats
snareFreq = 2./bar_duration
wdBlkFreq = 3./bar_duration

## get a time index for the 3 listening bars and the silence bar
#t_listen = np.arange(listen_win[0], listen_win[1], 1)/float(s_rate) #unneaded?

## rereference to the average EEG amplitude
EEG -= EEG.mean(0)

## calculate the epoched data for the listen period
### mix both for they should have the same source
all_trials = meet.epochEEG(EEG,
        np.r_[snareListenMarker[snareInlier],
            wdBlkListenMarker[wdBlkInlier]],
        all_win)
listen_trials = meet.epochEEG(EEG,
        np.r_[snareListenMarker[snareInlier_listen],
            wdBlkListenMarker[wdBlkInlier_listen]],
        listen_win)
silence_trials = meet.epochEEG(EEG,
        np.r_[snareListenMarker[snareInlier_silence],
            wdBlkListenMarker[wdBlkInlier_silence]],
        silence_win)
# DFFT
## get frequency resolution of 1/6 Hz
nperseg = 12*s_rate
f = np.fft.rfftfreq(nperseg, d=1./s_rate)

def slepianFFT(x, nperseg, axis):
    """
    calculate FFT after multiplication with slepian window with highest
    frequency resolution
    """
    N = x.shape[axis]
    if N > nperseg:
        # clip data > nperseg
        sliceobj = [[None, None, None]] * x.ndim
        sliceobj[axis] = [None, nperseg, None]
        sliceobj = [slice(*sl) for sl in sliceobj]
        x = x[sliceobj]
    elif N < nperseg:
        after = nperseg - N
        padwidth = [[0, 0]] * x.ndim
        padwidth[axis] = [0, after]
        # zeropad
        x = np.pad(x, pad_width=padwidth, constant_values=0)
    # calculate slepian window which maximizes spectral concentration
    taper = scipy.signal.windows.dpss(nperseg, NW=1, Kmax=1, sym=False)[0]
    taper_shape = np.ones(x.ndim, int)
    taper_shape[axis] = nperseg
    taper.resize(taper_shape)
    return np.fft.rfft(taper*x, axis=axis)

## calculate the fourier transform of all trials
F = slepianFFT(all_trials, nperseg, axis=1)
F_listen = slepianFFT(listen_trials, nperseg, axis=1)
F_silence = slepianFFT(silence_trials, nperseg, axis=1)

## apply a filter in the Fourier domain to extract only the frequencies
## of interest
# choose the time window (listen, silence, both) here!
use_F = F

# get frequency indices
snare_idx = np.argmin((f-snareFreq)**2)
wdBlk_idx = np.argmin((f-wdBlkFreq)**2)

# weight Fourier transform with this to get target data
target_pattern = [1/9, 2/9, 3/9, 2/9, 1/9]
# weight Fourier transform with this array to get contrast data
contrast_pattern = [2/6, 1/6, 0, 1/6, 2/6]

snare_target_mask = np.zeros_like(f)
snare_target_mask[snare_idx - len(target_pattern)//2 :
                  snare_idx + len(target_pattern)//2 + 1] = target_pattern
wdBlk_target_mask = np.zeros_like(f)
wdBlk_target_mask[wdBlk_idx - len(target_pattern)//2 :
                  wdBlk_idx + len(target_pattern)//2 + 1] = target_pattern
snare_contrast_mask = np.zeros_like(f)
snare_contrast_mask[snare_idx - len(contrast_pattern)//2 :
                  snare_idx + len(contrast_pattern)//2 + 1] = contrast_pattern
wdBlk_contrast_mask = np.zeros_like(f)
wdBlk_contrast_mask[wdBlk_idx - len(contrast_pattern)//2 :
                  wdBlk_idx + len(contrast_pattern)//2 + 1] = contrast_pattern

# inverse FFT
## signal not periodic but good filter for magnitude response
snare_target_trials = np.fft.irfft(
        use_F*snare_target_mask[...,np.newaxis], n=nperseg, axis=1)
wdBlk_target_trials = np.fft.irfft(
        use_F*wdBlk_target_mask[...,np.newaxis], n=nperseg, axis=1)
snare_contrast_trials = np.fft.irfft(
        use_F*snare_contrast_mask[...,np.newaxis], n=nperseg, axis=1)
wdBlk_contrast_trials = np.fft.irfft(
        use_F*wdBlk_contrast_mask[...,np.newaxis], n=nperseg, axis=1)

# calculate the covariance matrix of every single trial (shape (32,32,147))
snare_target_cov = np.einsum('ijk, ljk -> ilk',
        snare_target_trials, snare_target_trials)
wdBlk_target_cov = np.einsum('ijk, ljk -> ilk',
        wdBlk_target_trials, wdBlk_target_trials)
snare_contrast_cov = np.einsum('ijk, ljk -> ilk',
        snare_contrast_trials, snare_contrast_trials)
wdBlk_contrast_cov = np.einsum('ijk, ljk -> ilk',
        wdBlk_contrast_trials, wdBlk_contrast_trials)

#save the eeg results
np.savez(os.path.join(save_folder, 'prepared_FFTSSD.npz'),
        snare_target_cov = snare_target_cov,
        wdBlk_target_cov = wdBlk_target_cov,
        snare_contrast_cov = snare_contrast_cov,
        wdBlk_contrast_cov = wdBlk_contrast_cov,
        snareInlier = snareInlier,
        wdBlkInlier = wdBlkInlier,
        snareInlier_listen = snareInlier_listen,
        wdBlkInlier_listen = wdBlkInlier_listen,
        snareInlier_silence = snareInlier_silence,
        wdBlkInlier_silence = wdBlkInlier_silence,
        F = F,
        F_listen = F_listen,
        F_silence = F_silence,
        f = f)
