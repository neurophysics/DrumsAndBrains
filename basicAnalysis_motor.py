import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
import meet
import helper_functions
import scipy as sp

data_folder = sys.argv[1]
result_folder = sys.argv[2]
N_subjects = 21 #21 later, for now bc of speed
s_rate = 1000 # sampling rate of the EEG

# target frequencies
snareFreq = 7./6
wdBlkFreq = 7./4

# read raw EEG data
eegs = []
artifact_masks = []
for i in range(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(data_folder, 'S%02d' % i)
                + '/clean_data.npz', 'r') as fi:
            EEG = fi['clean_data'] # shape (32, 2901860)
            EEG -= EEG.mean(0) # rereference to the average EEG amplitude
            eegs.append(EEG)
            artifact_masks.append(fi['artifact_mask'])
    except:
        print(('Warning: Subject %02d could not be loaded!' %i))

# read the channel names
channames = meet.sphere.getChannelNames(os.path.join(data_folder,
    '../channels.txt'))
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

##### plot 2 sec preresponse for each channel #####
idx = 0 #index to asses eeg (0 to 19)
subj = 1 #index for subject number (1 to 10, 12 to 21)
while(subj <= N_subjects):
    # skip subject without eeg data
    if not os.path.exists(os.path.join(
        result_folder, 'S{:02d}'.format(subj), 'prepared_FFTSSD.npz')):
        subj += 1
        continue
    #print(idx, subj)
    eeg = eegs[idx]
    artifact_mask = artifact_masks[idx]
    save_folder = os.path.join(result_folder, 'S{:02d}'.format(subj))
    data_folder_subj = os.path.join(data_folder, 'S{:02d}'.format(subj))
    ## get session clocks
    if os.path.exists(os.path.join(
        data_folder_subj, 'S{:02d}_eeg_all_files.vmrk'.format(subj))):
        marker_fname = os.path.join(
                data_folder_subj, 'S{:02d}_eeg_all_files.vmrk'.format(subj))
    else:
        marker_fname = os.path.join(data_folder_subj, 'S%02d_eeg.vmrk' % subj)
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

    snareCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
            snareCue_nearestClock, snareCue_DevToClock)
    wdBlkCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
            wdBlkCue_nearestClock, wdBlkCue_DevToClock)

    snare_resp = snareCue_pos + int(0.5 * bar_duration * s_rate)
    wdBlk_resp = wdBlkCue_pos + int(2./3 * bar_duration * s_rate)

    #plot 2000ms pre response for each channel
    win = [-2000, 500]
    snareInlier = np.all(meet.epochEEG(artifact_mask, snare_resp,
        win), 0)
    wdBlkInlier = np.all(meet.epochEEG(artifact_mask, wdBlk_resp,
        win), 0)
    all_trials = meet.epochEEG(eeg,
            np.r_[snare_resp[snareInlier],
                wdBlk_resp[wdBlkInlier]],
            win)
    Nc = len(channames)
    fig, axs = plt.subplots(Nc, figsize=(4,8), sharex=True)
    fig.subplots_adjust(top=0.95, bottom=0.05)
    axs[0].set_title('2000 ms preresponse')
    for c in range(Nc):
        axs[c].plot(range(*win), all_trials.mean(-1)[c])
        axs[c].set_ylabel(channames[c], fontsize=8)
        axs[c].set_yticks([])
    #plt.show()
    fig.savefig(os.path.join(save_folder, 'motor_2000mspreresponse'))

    # ERD in frequency bands 1-4, 4-8, 8-12, 12-20, 20-40
    fbands = [[1,4], [4,8], [8,12], [12,20], [20,40]]
    i=1 #later loop over all frequency bands
    # 1. band-pass filters with order 10
    Wn = np.array(fbands[i]) / s_rate * 2
    b, a = sp.signal.butter(5, Wn, btype='bandpass')
    eeg_filt = sp.signal.lfilter(b, a, eeg)
    #2. Hilbert-Transform, absolute value
    eeg_filtHil = np.abs(sp.signal.hilbert(eeg_filt))
    #3. Normalisieren, so dass 2 sek prä-stimulus 100% sind und dann averagen
    win = [-2000, 0] # window is smaller than abvove => all that are Inlier above, will now be as well
    all_trials_filt = meet.epochEEG(eeg_filtHil,
            np.r_[snare_resp[snareInlier],
                wdBlk_resp[wdBlkInlier]],
            win)
    all_trials_filt = (all_trials_filt - all_trials_filt.mean(-1)[:,:,np.newaxis]
        ) / np.std(all_trials_filt,-1)[:,:,np.newaxis]
    # plot
    fig, axs = plt.subplots(Nc, figsize=(4,8), sharex=True)
    fig.subplots_adjust(top=0.95, bottom=0.05)
    axs[0].set_title('ERD band' + str(fbands[i][0]) + '-' + str(fbands[i][1])))
    for c in range(Nc):
        axs[c].plot(range(*win), all_trials_filt.mean(-1)[c])
        axs[c].set_ylabel(channames[c], fontsize=8)
        axs[c].set_yticks([])
    #plt.show()
    fig.savefig(os.path.join(save_folder, 'motor_ERD_band'
        + str(fbands[i][0]) + '-' + str(fbands[i][1])))


    idx += 1
    subj += 1
plt.close('all')
