import numpy as np
import scipy.signal
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

data_folder = sys.argv[1]
subject = int(sys.argv[2])
result_folder = sys.argv[3]

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

# apply a 0.5 - 20 Hz band-pass filter
EEG_hp = meet.iir.butterworth(EEG, fs=(0.1, 70), fp=(2, 40),
        s_rate=s_rate)

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
all_win = [0, int(4*bar_duration*s_rate)]
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
t_all = np.arange(listen_win[0], silence_win[1], 1)/float(s_rate)

# rereference to the average EEG amplitude
EEG -= EEG.mean(0)

# calculate the evoked potential to the listening and silence bars
listen_trials = meet.epochEEG(EEG,
        np.r_[snareListenMarker[snareInlier],
            wdBlkListenMarker[wdBlkInlier]],
        listen_win)

# calculate the average
listen_trials_avg = listen_trials.mean(-1)

def mtcsd(x, win, ratios, nfft=12*s_rate):
    #make zero mean
    if x.shape[-1] > nfft:
        x = x[...,-nfft:]
    x = x - x.mean(-1)[...,np.newaxis]
    f = np.fft.rfftfreq(nfft, d=1./s_rate)
    csd = np.zeros([x.shape[0], x.shape[0], len(f)], np.complex)
    n = x.shape[-1]
    for w,r in zip(win, ratios):
        temp = np.fft.rfft(w*x, n=nfft)
        csd += r*np.einsum('ik,jk->ijk', np.conj(temp), temp)
    csd /= ratios.sum()
    return f, csd

# use slepian windows
listen_win, listen_ratios = scipy.signal.windows.dpss(
        min([12*s_rate, len(t_listen)]), NW=1.5,
        Kmax=2, sym=False, norm='subsample', return_ratios=True)

## use a hanning window
#listen_win, listen_ratios = [scipy.signal.windows.hann(
#    min([12*s_rate, len(t_listen)]), sym=False)], np.array([1])

f = np.fft.rfftfreq(12*s_rate, d=1./s_rate)
# calculate the spectrum of all the single trials
csd = np.zeros([listen_trials.shape[0], listen_trials.shape[0], len(f)],
        np.complex)
from tqdm import tqdm
for t in tqdm(listen_trials.T):
    csd += mtcsd(t.T, listen_win, listen_ratios)[1]
csd /= listen_trials.shape[-1]

#save the eeg results
np.savez(os.path.join(save_folder, 'prepared_FFTSSD.npz'),
        csd=csd,
        f=f)