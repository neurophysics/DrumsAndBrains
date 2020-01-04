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
listen_trials = meet.epochEEG(EEG_hp,
        np.r_[snareListenMarker[snareInlier],
            wdBlkListenMarker[wdBlkInlier]],
        listen_win)
# calculate the average
listen_trials_avg = listen_trials.mean(-1)


###########################################
# construct a FIR filter as 3 Hz low-pass #
###########################################
# assert, that the number of taps is not longer than 1/2
# the interval between
# the woodblock beats (this avoids an influence of a previous erp on the
# silence bar)
numtaps = int(s_rate/wdBlkFreq)//2
# make numtaps odd
if (numtaps%2) == 0: numtaps -= 1
h = scipy.signal.firwin(numtaps, 3, width=3, window='sinc', pass_zero=True,
        fs=1000)
f, amp = scipy.signal.freqz(h, worN=32000, fs=1000)
# plot some filter analysis
impulse = np.zeros(2000)
impulse[1000] = 1
fimpulse = scipy.signal.lfilter(h,1, impulse)
timpulse = np.arange(-1000,1000,1)
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(timpulse, impulse, 'b-')
ax11 = plt.twinx(ax1)
ax11.plot(timpulse - numtaps//2, fimpulse, 'r-')
ax2 = fig.add_subplot(312)
ax2.plot(f, 20*np.log10(np.abs(amp)))
ax3 = fig.add_subplot(313, sharey=ax2)
ax3.plot(f, 20*np.log10(np.abs(amp)))
ax3.set_xlim([0,10])
ax1.set_title('impulse response (Ntaps: %d)' % numtaps)
ax2.set_title('frequency response')
ax3.set_title('frequency response - zoomed')
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('amplitude (a.u.)')
ax11.set_ylabel('amplitude (a.u.)')
ax2.set_xlabel('frequency (Hz)')
ax2.set_ylabel('gain (dB)')
ax3.set_xlabel('frequency (Hz)')
ax3.set_ylabel('gain (dB)')
fig.tight_layout()
fig.savefig(os.path.join(
    os.path.join(result_folder), 'S%02d' % subject,
    'FIR_filter.pdf'))

# fit a sloping line + cosine and sine of the snare and woodblock
# frequencies to the listening bars
# make up a design matrix having ones (intercept), t (slope), snare_cos, snare_sin, wdBlk_cos, wdBlk_sin as columns
listen_fit_matrix = np.array([
    np.ones_like(t_listen),
    t_listen,
    np.cos(2*np.pi*snareFreq*t_listen),
    np.sin(2*np.pi*snareFreq*t_listen),
    np.cos(2*np.pi*wdBlkFreq*t_listen),
    np.sin(2*np.pi*wdBlkFreq*t_listen)
    ]).T

# filter the EEG (3 Hz lowpass) and compensate for phase shift,
# get the listening trials and fit the sines and cosines
EEG_lowfreq = scipy.signal.lfilter(h, 1, EEG, axis=-1)
snareListenData = meet.epochEEG(EEG_lowfreq, snareListenMarker[snareInlier]
        + numtaps//2, listen_win)
wdBlkListenData = meet.epochEEG(EEG_lowfreq, wdBlkListenMarker[wdBlkInlier]
        + numtaps//2, listen_win)
# make the trials zero mean
snareListenData -= snareListenData.mean(axis=1)[:,np.newaxis]
wdBlkListenData -= wdBlkListenData.mean(axis=1)[:,np.newaxis]

snareFit = np.array([np.linalg.lstsq(listen_fit_matrix, c)[0]
    for c in snareListenData])
wdBlkFit = np.array([np.linalg.lstsq(listen_fit_matrix, c)[0]
    for c in wdBlkListenData])

# reconstruct the trials using only the cosine and sine information
# this means setting the constant and slope to 0
snareFit[:,:2] = 0
wdBlkFit[:,:2] = 0
snareListenData_rec = listen_fit_matrix.dot(snareFit).swapaxes(0,1)
wdBlkListenData_rec = listen_fit_matrix.dot(wdBlkFit).swapaxes(0,1)

# get the silence data
snareSilenceData = meet.epochEEG(EEG_lowfreq,
        snareListenMarker[snareInlier] + numtaps//2, silence_win)
wdBlkSilenceData = meet.epochEEG(EEG_lowfreq,
        wdBlkListenMarker[wdBlkInlier] + numtaps//2, silence_win)

silence_fit_matrix = np.array([
    np.ones_like(t_silence),
    t_silence,
    np.cos(2*np.pi*snareFreq*t_silence),
    np.sin(2*np.pi*snareFreq*t_silence),
    np.cos(2*np.pi*wdBlkFreq*t_silence),
    np.sin(2*np.pi*wdBlkFreq*t_silence)
    ]).T

snareFitSilence = np.array([np.linalg.lstsq(silence_fit_matrix, c)[0]
    for c in snareSilenceData])
wdBlkFitSilence = np.array([np.linalg.lstsq(silence_fit_matrix, c)[0]
    for c in wdBlkSilenceData])
# make the trials zero mean
snareSilenceData -= snareSilenceData.mean(axis=1)[:,np.newaxis]
wdBlkSilenceData -= wdBlkSilenceData.mean(axis=1)[:,np.newaxis]
# reconstruct the trials using only the cosine and sine information
# this means setting the constant and slope to 0
snareFitSilence[:,:2] = 0
wdBlkFitSilence[:,:2] = 0
snareSilenceData_rec = listen_fit_matrix.dot(snareFitSilence).swapaxes(
        0,1)
wdBlkSilenceData_rec = listen_fit_matrix.dot(wdBlkFitSilence).swapaxes(
        0,1)

#save the eeg results
np.savez(os.path.join(save_folder, 'prepared_filterdata.npz'),
    lowpass = h,
    listen_trials = listen_trials,
    listen_trials_avg = listen_trials_avg,
    snareListenData = snareListenData,
    wdBlkListenData = wdBlkListenData,
    snareListenData_rec = snareListenData_rec,
    wdBlkListenData_rec = wdBlkListenData_rec,
    snareSilenceData = snareSilenceData,
    wdBlkSilenceData = wdBlkSilenceData,
    snareSilenceData_rec = snareSilenceData_rec,
    wdBlkSilenceData_rec = wdBlkSilenceData_rec,
    snareInlier = snareInlier,
    wdBlkInlier = wdBlkInlier,
    snareFitListen = snareFit,
    wdBlkFitListen = wdBlkFit,
    snareFitSilence = snareFitSilence,
    wdBlkFitSilence = wdBlkFitSilence
    )
