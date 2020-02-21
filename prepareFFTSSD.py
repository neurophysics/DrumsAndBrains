"""
This script calculates cross-spectra across channels for every
single trial of a subject from the listening+silence period of the experiment,
averages the results across single trial (the result is the average
single-trial cross-spectral density matrix)
and stores the result as 'prepared_FFTSSD.npz' in the Result folder
of that subject

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

data_folder = sys.argv[1]
subject = int(sys.argv[2])
result_folder = sys.argv[3]

s_rate = 1000 # sampling rate of the EEG

data_folder = os.path.join(data_folder, 'S%02d' % subject)
save_folder = os.path.join(result_folder, 'S%02d' % subject)

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

eeg_fname = os.path.join(data_folder, 'clean_data.npz')

if os.path.exists(os.path.join(
    data_folder, 'S{:02d}_eeg_all_files.vmrk'.format(subject))):
    marker_fname = os.path.join(
            data_folder, 'S{:02d}_eeg_all_files.vmrk'.format(subject))
else:
    marker_fname = os.path.join(data_folder, 'S%02d_eeg.vmrk' % subject)

eeg_clocks = helper_functions.getSessionClocks(marker_fname)
eeg_clocks = [c for c in eeg_clocks if len(c) > 100]

assert len(eeg_clocks) == 6, '6 sessions expected'

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

# now, find the sample of each Cue
snareCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
        snareCue_nearestClock, snareCue_DevToClock)
wdBlkCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
        wdBlkCue_nearestClock, wdBlkCue_DevToClock)

# read the cleaned EEG and the artifact segment mask
with np.load(eeg_fname) as npzfile:
    EEG = npzfile['clean_data']
    artifact_mask = npzfile['artifact_mask']

# read the channel names
channames = meet.sphere.getChannelNames(os.path.join(data_folder,
    '../channels.txt'))
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

# get the sample indices at the start of the 3 'listening bars'
snareListenMarker = snareCue_pos - int(4*bar_duration*s_rate)
wdBlkListenMarker = wdBlkCue_pos - int(4*bar_duration*s_rate)

# get the temporal windows of the listening and silence bars and of both
all_win = [int(-2*bar_duration*s_rate), int(4*bar_duration*s_rate)]
listen_win = [0, int(3*bar_duration*s_rate)]
silence_win = [int(3*bar_duration*s_rate), int(4*bar_duration*s_rate)]

# reject trials that contain rejected data segments
snareInlier = np.all(meet.epochEEG(artifact_mask, snareListenMarker,
    all_win), 0)
wdBlkInlier = np.all(meet.epochEEG(artifact_mask, wdBlkListenMarker,
    all_win), 0)

# get the frequencies of the snaredrum (duple) and woodblock (triple) beats
snareFreq = 2./bar_duration
wdBlkFreq = 3./bar_duration

# get a time index for the 3 listening bars and the silence bar
t_listen = np.arange(listen_win[0], listen_win[1], 1)/float(s_rate)
t_silence = np.arange(silence_win[0], silence_win[1], 1)/float(s_rate)
t_all = np.arange(all_win[0], all_win[1], 1)/float(s_rate)

# rereference to the average EEG amplitude
EEG -= EEG.mean(0)

# calculate the epoched data in the listen period (for snare AND woodblock)
poststim_trials = meet.epochEEG(EEG,
        np.r_[snareListenMarker[snareInlier],
            wdBlkListenMarker[wdBlkInlier]],
        all_win)

nperseg = 12*s_rate
f = np.fft.rfftfreq(nperseg, d=1./s_rate)

# calculate slepian windows for multitaper spectral estimation
poststim_win, poststim_ratios = scipy.signal.windows.dpss(
        min([nperseg, poststim_trials.shape[1]]), NW=1.5,
        Kmax=2, sym=False, norm='subsample', return_ratios=True)

from tqdm import tqdm # for progress bar

# loop through all the trials and calculate the average csd across all
# trials
# before, make the trials zero mean and normalize the total variance of
# every trial to 0
poststim_csd = np.zeros([poststim_trials.shape[0], poststim_trials.shape[0],
    len(f)], np.complex)

for p in tqdm(poststim_trials, desc='Calc. csd',
        total=poststim_trials.shape[-1]):
    p_csd = np.zeros_like(poststim_csd)
    p_csd = helper_functions.mtcsd(l.T,
            poststim_win, poststim_ratios, nfft=nperseg)
    poststim_csd += p_csd/poststim_trials.shape[-1]

#save the eeg results
np.savez(os.path.join(save_folder, 'prepared_FFTSSD.npz'),
        poststim_csd=poststim_csd,
        f=f)
