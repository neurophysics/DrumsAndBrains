import numpy as np
import sys
import csv
import os.path
import matplotlib.pyplot as plt
import meet
import helper_functions
import scipy as sp

data_folder = sys.argv[1]
result_folder = sys.argv[2]
N_subjects = 21 #21 later, for now bc of speed (10min per subject)
s_rate = 1000 # sampling rate of the EEG

# color map
cmap = 'plasma'
color4 = '#1f78b4'.upper()
color3 = '#33a02c'.upper()
color2 = '#b2df8a'.upper()
color1 = '#a6cee3'.upper()
colors = [color1, color2, color3, color4, 'grey']

# target frequencies
snareFreq = 7./6
wdBlkFreq = 7./4

# read the channel names
channames = meet.sphere.getChannelNames(os.path.join(data_folder,'channels.txt'))
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

# get LQR data
LQ = []
with open(os.path.join(data_folder,'additionalSubjectInfo.csv'),'r') as infile:
    reader = csv.DictReader(infile, fieldnames=None, delimiter=';')
    for row in reader:
        LQ.append(int(row['LQ']))
# True if channels need to be reversed
left_handed = [True if i<0 else False for i in LQ]

##### plot 2 sec preresponse for each channel #####
all_BP = [] #avg over all subjects
all_ERD = [] #avg over all subjects
fbands = [[1,4], [4,8], [8,12], [12,20], [20,40]]
win = [-2000, 500]
cueHit_diff = []
idx = 0 #index to asses eeg (0 to 19)
subj = 1 #index for subject number (1 to 10, 12 to 21)
while(subj <= N_subjects):
    print(subj)
    # skip subject without eeg data
    if not os.path.exists(os.path.join(
        result_folder, 'S{:02d}'.format(subj), 'prepared_FFTSSD.npz')):
        subj += 1
        continue
    #print(idx, subj)

    # read raw EEG data
    with np.load(os.path.join(data_folder, 'S%02d' % subj)
            + '/clean_data.npz', 'r') as fi:
        eeg = fi['clean_data'] # shape (32, 2901860)
        artifact_mask = fi['artifact_mask']
    eeg -= eeg.mean(0) # rereference to the average EEG amplitude
    # for lefthanded subjects, switch electrodes
    if left_handed[idx]:
        #print('subject '+str(subj)+' is left-handed. Switching electrodes...')
        # see list(enumerate(channames))
        eeg = np.vstack([eeg[1,:], eeg[0,:], eeg[6,:], eeg[5,:], eeg[4,:],
            eeg[3,:], eeg[2,:], eeg[10,:], eeg[9,:], eeg[8,:], eeg[7,:],
            eeg[15,:], eeg[14,:], eeg[13,:], eeg[12,:], eeg[11,:], eeg[21,:],
            eeg[20,:], eeg[19,:], eeg[18,:], eeg[17,:], eeg[16,:], eeg[26,:],
            eeg[25,:], eeg[24,:], eeg[23,:], eeg[22,:], eeg[28,:], eeg[27,:],
            eeg[31,:], eeg[30,:], eeg[29,:]])
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

    # store times of Cue, Hit(response) and their time difference in sample
    snareCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
            snareCue_nearestClock, snareCue_DevToClock)
    wdBlkCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
            wdBlkCue_nearestClock, wdBlkCue_DevToClock)
    snare_cueHitdiff = ((0.5 * bar_duration + snare_deviation) * s_rate)
    wdBlk_cueHitdiff = ((2./3 * bar_duration + wdBlk_deviation) * s_rate)
    snareHit_times = snareCue_pos + snare_cueHitdiff
    wdBlkHit_times = wdBlkCue_pos + wdBlk_cueHitdiff
    # store time difference between Cues and response for plot
    cueHit_diff.append(np.hstack([snare_cueHitdiff, wdBlk_cueHitdiff]))

    snareHit_times_outlier = np.isnan(snareHit_times)
    wdBlkHit_times_outlier = np.isnan(wdBlkHit_times)
    snareHit_times = snareHit_times[~snareHit_times_outlier].astype(int)
    wdBlkHit_times = wdBlkHit_times[~wdBlkHit_times_outlier].astype(int)



    #plot 2000ms pre response for each channel
    snareInlier = np.all(meet.epochEEG(artifact_mask, snareHit_times,
        win), 0)
    wdBlkInlier = np.all(meet.epochEEG(artifact_mask, wdBlkHit_times,
        win), 0)
    all_trials = meet.epochEEG(eeg,
            np.r_[snareHit_times[snareInlier],
                wdBlkHit_times[wdBlkInlier]],
            win)
    all_trials -= all_trials[:,-win[0]][:,np.newaxis]
    if subj==2:
        all_trials = np.concatenate([
            all_trials[:,:,:139], all_trials[:,:,141:]], axis=2)
    if subj==12:
        all_trials = np.concatenate([
            all_trials[:,:,:47], all_trials[:,:,71:]], axis=2)
    Nc = len(channames)
    fig, axs = plt.subplots(int(np.ceil(Nc/4)), 4, figsize=(8,12),
            sharex=True, sharey=True)
    fig.subplots_adjust(top=0.95, bottom=0.05)
    fig.suptitle('BP: 2000 ms preresponse')
    for c in range(Nc):
        axs[c//4, c%4].plot(range(*win), all_trials.mean(-1)[c], linewidth=1, c='k')
        axs[c//4, c%4].set_ylabel(channames[c], fontsize=8)
        axs[c//4, c%4].axvline(0, lw=0.5, c='k')
        axs[c//4, c%4].axhline(0, lw=0.5, c='r', ls=':')
        axs[c//4, c%4].tick_params(axis='x', labelsize=8)
        axs[c//4, c%4].tick_params(axis='y', labelsize=8)
    #plt.show()
    fig.savefig(os.path.join(save_folder, 'motor_BP_2000mspreresponse.pdf'))
    all_BP.append(all_trials.mean(-1))

    # ERD in frequency bands 1-4, 4-8, 8-12, 12-20, 20-40
    ERDs = []
    for band in fbands:#later loop over all frequency bands
    # 1. band-pass filters with order 6 (3 into each direction)
        Wn = np.array(band) / s_rate * 2
        b, a = sp.signal.butter(3, Wn, btype='bandpass')
        eeg_filt = sp.signal.filtfilt(b, a, eeg)
        #2. Hilbert-Transform, absolute value
        eeg_filtHil = np.abs(sp.signal.hilbert(eeg_filt, axis=-1))
        #3. Normalisieren, so dass 2 sek prä-stimulus 100% sind und dann averagen
        all_trials_filt = meet.epochEEG(eeg_filtHil,
                np.r_[snareHit_times[snareInlier],
                    wdBlkHit_times[wdBlkInlier]],
                win)
        if subj==2:
            all_trials_filt = np.concatenate([
                all_trials_filt[:,:,:139], all_trials_filt[:,:,141:]], axis=2)
        if subj==12:
            all_trials_filt = np.concatenate([
                all_trials_filt[:,:,:47], all_trials_filt[:,:,71:]], axis=2)
        ERD = all_trials_filt.mean(-1)
        ERD /= ERD[:,0][:,np.newaxis]
        ERD *= 100
        ERDs.append(ERD)
    cueHit_diff_mean = np.nanmean(cueHit_diff[-1])
    cueHit_diff_sd = np.nanstd(cueHit_diff[-1])
    fig, axs = plt.subplots(int(np.ceil(Nc/3)), 3, figsize=(7,7),
            sharex=True, sharey=True)
    fig.subplots_adjust(top=0.95, bottom=0.05)
    fig.suptitle('ERD: 2000 ms preresponse')
    for c in range(Nc):
        handels = [] # for legend
        for i,ERD in enumerate(ERDs):
            h, = axs[c//3, c%3].plot(range(*win), ERD[c], linewidth=1,
                c=colors[i])
            handels.append(h)
        axs[c//3, c%3].set_ylabel(channames[c], fontsize=8)
        axs[c//3, c%3].axvline(0, lw=0.5, c='k')
        axs[c//3, c%3].axhline(100, lw=0.5, c='r', ls=':')
        axs[c//3, c%3].axvspan(-(cueHit_diff_mean-cueHit_diff_sd),
            -(cueHit_diff_mean+cueHit_diff_sd),
            alpha=0.3, color='red')
        axs[c//3, c%3].tick_params(axis='x', labelsize=8)
        axs[c//3, c%3].tick_params(axis='y', labelsize=8)
    fig.delaxes(axs[c//3, 2])
    plt.legend(handels,['frequency band '+str(i[0])+'-'+str(i[1]) +' Hz'
        for i in fbands], bbox_to_anchor=(1.7, 1.2), loc='upper center',
        borderaxespad=1, fontsize=5)
    #plt.show()
    fig.savefig(os.path.join(save_folder, 'motor_ERD_2000mspreresponse.pdf'))
    all_ERD.append(ERDs)
    idx += 1
    subj += 1
    plt.close('all')


# plot BP for all subjects
all_BP_avg = np.mean(all_BP, axis=0)
fig, axs = plt.subplots(int(np.ceil(Nc/4)), 4, figsize=(8,7),
        sharex=True, sharey=True)
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.05)
fig.suptitle('BP: 2000 ms preresponse, subject average')
for c in range(Nc):
    axs[c//4, c%4].plot(range(*win), all_BP_avg[c], linewidth=1, c='k')
    axs[c//4, c%4].set_ylabel(channames[c], fontsize=8)
    axs[c//4, c%4].axhline(0, lw=0.5, c='r', ls=':')
    axs[c//4, c%4].axvline(0, lw=0.5, c='k')
#plt.show()
fig.savefig(os.path.join(result_folder, 'motor_BP_2000mspreresponse.pdf'))

# plot ERD for all subjects
all_ERD_avg = [ np.mean([i[j] for i in all_ERD], axis=0)
    for j in range(len(fbands))] # for each band, average over subjects
cueHit_diff_mean = np.mean(np.hstack(cueHit_diff))
cueHit_diff_sd = np.std(np.hstack(cueHit_diff))
#[for i in all_ERD]
fig, axs = plt.subplots(int(np.ceil(Nc/3)), 3, figsize=(7,7),
        sharex=True, sharey=True)
fig.subplots_adjust(top=0.95, bottom=0.05)
fig.suptitle('ERD: 2000 ms preresponse')
for c in range(Nc):
    handels = [] # for legend
    for i,ERD in enumerate(all_ERD_avg):
        h, = axs[c//3, c%3].plot(range(*win), ERD[c], linewidth=1, c=colors[i])
        handels.append(h)
    axs[c//3, c%3].set_ylabel(channames[c], fontsize=8)
    #axs[c//3, c%3].set_ylim([50,200])
    axs[c//3, c%3].axvline(0, lw=0.5, c='k')
    axs[c//3, c%3].axhline(100, lw=0.5, c='r', ls=':')
    axs[c//3, c%3].axvspan(-(cueHit_diff_mean-cueHit_diff_sd),
        -(cueHit_diff_mean+cueHit_diff_sd),
        alpha=0.3, color='red')
    axs[c//3, c%3].tick_params(axis='x', labelsize=8)
    axs[c//3, c%3].tick_params(axis='y', labelsize=8)
fig.delaxes(axs[c//3, 2])
plt.legend(handels,['frequency band '+str(i[0])+'-'+str(i[1]) +' Hz'
    for i in fbands], bbox_to_anchor=(1.7, 1.2), loc='upper center',
    borderaxespad=1, fontsize=5)
#plt.show()
fig.savefig(os.path.join(result_folder, 'motor_ERD_2000mspreresponse.pdf'))
