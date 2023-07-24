# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import subprocess
import csv
import scipy
import scipy.stats
import helper_functions
import meet

'''Look at spectra in both periods before and after ssd if the two
oscillation frequencies are actually there (avg spectrum across all
trials per subject and also across all subjects)'''

data_folder = sys.argv[1]
result_folder = sys.argv[2]
N_subjects = 21
s_rate = 1000

if not os.path.exists(result_folder):
    os.mkdir(result_folder)

#as in eeg_outlier 88f
def plotSpectra(F, psd_pre,title,saveLocation,figsize=(10,10),grid=[8,4]):
    fig = plt.figure(figsize=figsize)
    # plot with 8 rows and 4 columns
    gs = mpl.gridspec.GridSpec(grid[0],grid[1], height_ratios = grid[0]*[1])
    ax = []
    for i, (psd_chan_now) in enumerate(psd_pre):
        if i == 0:
            ax.append(fig.add_subplot(gs[0,0]))
        else:
            ax.append(fig.add_subplot(gs[i//4,i%4], sharex=ax[0], sharey=ax[0]))
        ax[-1].plot(F, np.sqrt(psd_chan_now)*1000, c='k')
        ax[-1].grid(ls=':', alpha=0.8)
        ax[-1].set_xlabel('frequency (Hz)')
        ax[-1].set_ylabel('linear spectral density')
        ax[-1].set_title(channames[i])
    ax[-1].set_yscale('log')
    ax[-1].set_xscale('log')
    ax[-1].set_xticks([1,5,10,20,50,100])
    ax[-1].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax[-1].set_ylim([1E1, 1E5])
    fig.suptitle(title, size=14)
    gs.tight_layout(fig, pad=0.3, rect=(0,0,1,0.95))
    fig.savefig(saveLocation)
    plt.close(fig)

###########################################
# before applying SSD #
###########################################
#create empty arrays to store eeg data across subjects
allEEG_listening = np.empty((32,0))
allEEG_silence = np.empty((32,0))
for subject in range(1,1+N_subjects):
    print('calculating channel spectra subject '+str(subject))
    if subject==11:
        continue #no eeg for subject 11
    current_data_folder = os.path.join(data_folder, 'S%02d' % subject)
    current_result_folder = os.path.join(result_folder, 'S%02d' % subject)
    ###### Load EEG data ######
    # read the cleaned EEG and the artifact segment mask
    eeg_fname = os.path.join(current_data_folder, 'clean_data.npz')
    with np.load(eeg_fname) as npzfile:
        EEG = npzfile['clean_data']
        artifact_mask = npzfile['artifact_mask']
    # read the channel names
    channames = meet.sphere.getChannelNames(os.path.join(current_data_folder,
        '../channels.txt'))
    chancoords = meet.sphere.getStandardCoordinates(channames)
    chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
    chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
            projection='stereographic')

    ###### get frequencies of interest, divide into listen/silence ######
    with np.load(os.path.join(current_result_folder, 'behavioural_results.npz'),
            'r', allow_pickle=True) as f:
        snareCue_nearestClock = f['snareCue_nearestClock']
        snareCue_DevToClock = f['snareCue_DevToClock']
        wdBlkCue_nearestClock = f['wdBlkCue_nearestClock']
        wdBlkCue_DevToClock = f['wdBlkCue_DevToClock']
        snareCue_times = f['snareCue_times']
        wdBlkCue_times = f['wdBlkCue_times']
        bar_duration = f['bar_duration']
        snare_deviation = f['snare_deviation']
        wdBlk_deviation = f['wdBlk_deviation']
    snareFreq = 2./bar_duration
    wdBlkFreq = 3./bar_duration

    # now, find the sample of each Cue
    if os.path.exists(os.path.join(
        current_data_folder, 'S{:02d}_eeg_all_files.vmrk'.format(subject))):
        marker_fname = os.path.join(current_data_folder,
                'S{:02d}_eeg_all_files.vmrk'.format(subject))
    else:
        marker_fname = os.path.join(current_data_folder,
                'S%02d_eeg.vmrk' % subject)

    eeg_clocks = helper_functions.getSessionClocks(marker_fname)
    eeg_clocks = [c for c in eeg_clocks if len(c) > 100]

    assert len(eeg_clocks) == 6, '6 sessions expected'
    snareCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
            snareCue_nearestClock, snareCue_DevToClock)
    wdBlkCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
            wdBlkCue_nearestClock, wdBlkCue_DevToClock)

    # get the sample indices at the start of the 3 'listening bars'
    snareListenMarker = snareCue_pos - int(4*bar_duration*s_rate)
    wdBlkListenMarker = wdBlkCue_pos - int(4*bar_duration*s_rate)
    # get the sample indices at the start of the 'silence bars'
    snareSilenceMarker = snareCue_pos - int(bar_duration*s_rate)
    wdBlkSilenceMarker = wdBlkCue_pos - int(bar_duration*s_rate)
    # get the temporal windows of the listening and silence bars and of both
    all_win = [0, int(4*bar_duration*s_rate)]
    listen_win = [0, int(3*bar_duration*s_rate)]
    silence_win = [int(3*bar_duration*s_rate), int(4*bar_duration*s_rate)]
    # reject trials that contain rejected data segments
    snareInlier = np.all(meet.epochEEG(artifact_mask, snareListenMarker,
        all_win), 0)
    wdBlkInlier = np.all(meet.epochEEG(artifact_mask, wdBlkListenMarker,
        all_win), 0)

    # get a time index for the 3 listening bars and the silence bar
    t_listen = np.arange(listen_win[0], listen_win[1], 1)/float(s_rate)
    t_silence = np.arange(silence_win[0], silence_win[1], 1)/float(s_rate)
    t_all = np.arange(listen_win[0], silence_win[1], 1)/float(s_rate)

    # rereference to the average EEG amplitude
    EEG -= EEG.mean(0)


    ### Listening
    ListenMarker = np.concatenate((snareListenMarker,wdBlkListenMarker)) #no distinction in snare and wdBlk for now
    EEG_listening = np.zeros((32,ListenMarker.shape[0]*listen_win[1]))
    i = 0
    for start in ListenMarker:
        EEG_listening[:,i:i+listen_win[1]] = EEG[:,start:start+listen_win[1]]
        i=i+listen_win[1]
    # apply a 0.1 Hz high-pass filter
    data = meet.iir.butterworth(EEG_listening, fp=0.1, fs=0.08, s_rate=s_rate, axis=-1)
    F, psd_pre =  scipy.signal.welch(
            data, fs=s_rate, nperseg=1024, scaling='density')
    plotSpectra(F, psd_pre,
        title='Channel spectra, Listening Period, Subject S%02d' % subject,
        saveLocation=os.path.join(
            current_result_folder, 'Channel_spectra_Listening.pdf'))

    allEEG_listening = np.concatenate((allEEG_listening,EEG_listening),axis=1)

    ### Silence
    SilenceMarker = np.concatenate((snareSilenceMarker,wdBlkSilenceMarker)) #no distinction in snare and wdBlk for now
    EEG_silence = np.zeros((32,SilenceMarker.shape[0]*silence_win[1]))
    i = 0
    for start in SilenceMarker:
        EEG_silence[:,i:i+silence_win[1]] = EEG[:,start:start+silence_win[1]]
        i=i+silence_win[1]
    # apply a 0.1 Hz high-pass filter
    data = meet.iir.butterworth(EEG_silence, fp=0.1, fs=0.08, s_rate=s_rate, axis=-1)
    F, psd_pre = scipy.signal.welch(
            data, fs=s_rate, nperseg=1024, scaling='density')
    plotSpectra(F, psd_pre,
        title='Channel spectra, Silence Period, Subject S%02d' % subject,
        saveLocation=os.path.join(
                    current_result_folder, 'Channel_spectra_Silence.pdf'))
    allEEG_silence = np.concatenate((allEEG_silence,EEG_silence),axis=1)

#np.savez(os.path.join(result_folder, 'allEEG_silence.npz'), allEEG_silence=allEEG_silence)
#np.savez(os.path.join(result_folder, 'allEEG_listening.npz'), allEEG_listening=allEEG_listening)

#### Across subjects (takes a while)
#Listening
data = meet.iir.butterworth(allEEG_listening, fp=0.1, fs=0.08, s_rate=s_rate, axis=-1)
F, psd_pre = scipy.signal.welch(
            data, fs=s_rate, nperseg=1024, scaling='density')
plotSpectra(F, psd_pre,
    title='Channel spectra across subjects, Listening Period',
    saveLocation=os.path.join(
                result_folder, 'Channel_spectra_Listening.pdf'))

#Silence
data = meet.iir.butterworth(allEEG_silence, fp=0.1, fs=0.08, s_rate=s_rate, axis=-1)
F, psd_pre = scipy.signal.welch(
            data, fs=s_rate, nperseg=1024, scaling='density')
plotSpectra(F, psd_pre,
    title='Channel spectra across subjects, Silence Period',
    saveLocation=os.path.join(
                result_folder, 'Channel_spectra_Silence.pdf'))
