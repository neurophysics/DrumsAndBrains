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

print('preparing sPoC for subject {:02d}'.format(subject))

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
all_win = [0, int(4*bar_duration*s_rate)]
listen_win = [0, int(3*bar_duration*s_rate)]
silence_win = [int(3*bar_duration*s_rate), int(4*bar_duration*s_rate)]

# reject trials that contain rejected data segments
snareInlier = np.all([
    np.isfinite(snare_deviation),
    np.all(meet.epochEEG(artifact_mask, snareListenMarker,
    all_win), 0)],0)
wdBlkInlier = np.all([
    np.isfinite(wdBlk_deviation),
    np.all(meet.epochEEG(artifact_mask, wdBlkListenMarker,
    all_win), 0)],0)

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
snare_listen_trials = meet.epochEEG(EEG,
        snareListenMarker[snareInlier],
        listen_win)
snare_silence_trials = meet.epochEEG(EEG,
        snareListenMarker[snareInlier],
        silence_win)
snare_all_trials = meet.epochEEG(EEG,
        snareListenMarker[snareInlier],
        all_win)
# calculate the evoked potential to the listening and silence bars
wdBlk_listen_trials = meet.epochEEG(EEG,
        wdBlkListenMarker[wdBlkInlier],
        listen_win)
# calculate the evoked potential to the silenceing and silence bars
wdBlk_silence_trials = meet.epochEEG(EEG,
        wdBlkListenMarker[wdBlkInlier],
        silence_win)
wdBlk_all_trials = meet.epochEEG(EEG,
        wdBlkListenMarker[wdBlkInlier],
        all_win)

# use slepian windows for mutitaper spectral estimation of single trials
nperseg = 12*s_rate
f = np.fft.rfftfreq(nperseg, d=1./s_rate)

listen_Ntaper = min([nperseg, listen_win[1] - listen_win[0]])
silence_Ntaper = min([nperseg, silence_win[1] - silence_win[0]])
all_Ntaper = min([nperseg, all_win[1] - all_win[0]])

#BW = (wdBlkFreq-snareFreq)
BW = 1./6
listen_NW = listen_Ntaper*0.5*BW/s_rate
silence_NW = silence_Ntaper*0.5*BW/s_rate
all_NW = all_Ntaper*0.5*BW/s_rate

# calculate slepian windows for multitaper spectral estimation
listen_win, listen_ratios = scipy.signal.windows.dpss(
        listen_Ntaper, NW=listen_NW,
        Kmax=max(1, int(np.round(2*listen_NW - 1))),
        sym=False, norm='subsample', return_ratios=True)
silence_win, silence_ratios = scipy.signal.windows.dpss(
        silence_Ntaper, NW=silence_NW,
        Kmax=max(1, int(np.round(2*silence_NW - 1))),
        sym=False, norm='subsample', return_ratios=True)
all_win, all_ratios = scipy.signal.windows.dpss(
        all_Ntaper, NW=all_NW,
        Kmax=max(1, int(np.round(2*all_NW - 1))),
        sym=False, norm='subsample', return_ratios=True)

f = np.fft.rfftfreq(nperseg, d=1./s_rate)
# due to memory restrictions, keep only the frequencies < 50 Hz
f_keep = np.all([f>=0, f<=10], 0)
f = f[f_keep]

# calculate the multitaper spectrum of all the single trials
snare_listen_csd = np.array([
    helper_functions.mtcsd(t.T, listen_win, listen_ratios, nfft=nperseg)[
        ...,f_keep] for t in snare_listen_trials.T])
snare_silence_csd = np.array([
    helper_functions.mtcsd(t.T, silence_win, silence_ratios, nfft=nperseg)[
        ...,f_keep] for t in snare_silence_trials.T])
snare_all_csd = np.array([
    helper_functions.mtcsd(t.T, all_win, all_ratios, nfft=nperseg)[
        ...,f_keep] for t in snare_all_trials.T])
wdBlk_listen_csd = np.array([
    helper_functions.mtcsd(t.T, listen_win, listen_ratios, nfft=nperseg)[
        ...,f_keep] for t in wdBlk_listen_trials.T])
wdBlk_silence_csd = np.array([
    helper_functions.mtcsd(t.T, silence_win, silence_ratios, nfft=nperseg)[
        ...,f_keep] for t in wdBlk_silence_trials.T])
wdBlk_all_csd = np.array([
    helper_functions.mtcsd(t.T, all_win, all_ratios, nfft=nperseg)[
        ...,f_keep] for t in wdBlk_all_trials.T])

#save the eeg results
np.savez(os.path.join(save_folder, 'prepared_FFTcSPoC.npz'),
        snare_listen_csd = snare_listen_csd,
        snare_silence_csd = snare_silence_csd,
        snare_all_csd = snare_all_csd,
        wdBlk_listen_csd = wdBlk_listen_csd,
        wdBlk_silence_csd = wdBlk_silence_csd,
        wdBlk_all_csd = wdBlk_all_csd,
        snare_listen_trials = snare_listen_trials,
        snare_silence_trials = snare_silence_trials,
        wdBlk_listen_trials = wdBlk_listen_trials,
        wdBlk_silence_trials = wdBlk_silence_trials,
        snare_deviation = snare_deviation[snareInlier],
        wdBlk_deviation = wdBlk_deviation[wdBlkInlier],
        t_listen = t_listen,
        t_silence = t_silence,
        t_all = t_all,
        snareFreq = snareFreq,
        wdBlkFreq = wdBlkFreq,
        f=f
        )
