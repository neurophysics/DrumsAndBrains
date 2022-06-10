"""
3b. (DR) basicAnalysis_motor.py
    plots: 'motor/BP_2000mspreresponse.pdf',
      'motor/ERD_2000mspreresponse.pdf' (+ for each subject),
      'motor/channelSpectra0-30.pdf'
      'motor/channelSpectra_ERD0-30.pdf'
    data: 'motor/ERD_P.npy',
        'motor/BP.npz',
        'motor/ERD.npz',
        'motor/covmat.npz',
        'motor/inlier.npz'
"""
import numpy as np
import sys
import csv
import os.path
import matplotlib.pyplot as plt
import matplotlib as mpl
import meet
import helper_functions
import scipy as sp
import random
from tqdm import tqdm


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
colors5 = [color1, color2, color3, color4, 'grey']
colors2 = ['k', 'crimson']

# target frequencies
snareFreq = 7./6
wdBlkFreq = 7./4

# read the channel names
channames = meet.sphere.getChannelNames(os.path.join(data_folder,'channels.txt'))
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
Nc = len(channames)
channames_topo4x8 = [
    'F7','FP1','FP2','F8',
    'F3','AFZ','FZ','F4',
    'FC5','FC1','FC2','FC6',
    'T7','C3','C4','T8',
    'CP1','CZ','PZ','CP2',
    'TP9','CP5','CP6','TP10',
    'P7','P3','P4','P8',
    'PO3','O1','O2','PO4']
topo_idx = np.array([channames.index(elem) for elem in channames_topo4x8])
# np.array(channames)[topo_idx]==np.array(channames_topo4x8) is True

# get LQR data
LQ = []
with open(os.path.join(data_folder,'additionalSubjectInfo.csv'),'r') as infile:
    reader = csv.DictReader(infile, fieldnames=None, delimiter=';')
    for row in reader:
        LQ.append(int(row['LQ']))
# True if channels need to be reversed
left_handed = [True if i<0 else False for i in LQ]

##### plot 2 sec preresponse for each channel #####
all_snareHit_inlier = []
all_wdBlkHit_inlier = []
all_snareHit_times = []
all_wdBlkHit_times = []
all_snareInlier = []
all_wdBlkInlier = []
all_BP = [] #avg over all subjects
all_BP_trials = [] #BP per trial (32,2500,150)
all_ERD = [] #avg over all subjects
cueHit_diff = []
contrast_cov = []
target_cov = []
fbands = [[7,12], [15,25]] #chosen after looking at spectra
win = [-2000, 500]
base_idx = range(750) #corresponds to -2000 to -1250ms
act_idx = range(1500,2000) #corresponds to -750 to 0ms
act_idx_lda = range(1400,1900) #-600 to -100ms

idx = 0 #index to asses eeg (0 to 19)
subj = 1 #index for subject number (1 to 10, 12 to 21)
if len(fbands)==2:
    colors = colors2
else:
    colors=colors5

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
    tp10_idx = channames.index('TP10')
    tp9_idx = channames.index('TP9')
    eeg -= (eeg[tp10_idx]+eeg[tp9_idx])/2 # reference to average of TP10 and TP9

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
    snareHit_times_unmasked = snareCue_pos + snare_cueHitdiff
    wdBlkHit_times_unmasked = wdBlkCue_pos + wdBlk_cueHitdiff

    # store time difference between Cues and response for plot
    cueHit_diff.append(np.hstack([snare_cueHitdiff, wdBlk_cueHitdiff]))

    snareHit_inlier = ~np.isnan(snareHit_times_unmasked)
    wdBlkHit_inlier = ~np.isnan(wdBlkHit_times_unmasked)
    all_snareHit_inlier.append(snareHit_inlier)
    all_wdBlkHit_inlier.append(wdBlkHit_inlier)
    snareHit_times = snareHit_times_unmasked[snareHit_inlier].astype(int)
    wdBlkHit_times = wdBlkHit_times_unmasked[wdBlkHit_inlier].astype(int)
    all_snareHit_times.append(snareHit_times)
    all_wdBlkHit_times.append(wdBlkHit_times)

    cueHit_diff_mean = np.nanmean(cueHit_diff[-1])
    cueHit_diff_sd = np.nanstd(cueHit_diff[-1])

    #plot 2000ms pre response for each channel
    snareInlier = np.all(meet.epochEEG(artifact_mask, snareHit_times,
        win), 0)
    wdBlkInlier = np.all(meet.epochEEG(artifact_mask, wdBlkHit_times,
        win), 0)
    all_snareInlier.append(snareInlier)
    all_wdBlkInlier.append(wdBlkInlier)

    try: # to read BP (32,2500)
        with np.load(os.path.join(result_folder, 'motor/BP.npz'),
            'r') as f_BP:
            BP = f_BP['BP_{:02d}'.format(idx)]
            all_trials = f_BP['BP_trials_{:02d}'.format(idx)]
    except FileNotFoundError:
        all_trials = meet.epochEEG(eeg,
                np.r_[snareHit_times[snareInlier],
                    wdBlkHit_times[wdBlkInlier]],
                win)
        BP = all_trials.mean(-1) # trial average, now shape (channels, time)
        BP -= np.mean(BP[:,base_idx], axis=1)[:,np.newaxis] #avg of base window is 0 for each channel
    all_BP_trials.append(all_trials)
    all_BP.append(BP)

    BP_topo = np.array(BP)[topo_idx] #reorder for topological plot
    fig, axs = plt.subplots(int(np.ceil(Nc/4)), 4, figsize=(8,12),
            sharex=True, sharey=True)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.11, right=0.95, hspace=0.2)
    fig.suptitle('BP: 2000 ms preresponse, trial-avg.', fontsize=14)
    for c in range(Nc):
        axs[c//4, c%4].plot(range(*win), BP_topo[c], linewidth=1, c='k')
        axs[c//4, c%4].set_title(channames_topo4x8[c], fontsize=8, pad=2)
        axs[c//4, c%4].axvline(0, lw=0.5, c='r')
        axs[c//4, c%4].axhline(0, lw=0.5, c='k', ls=':')
        axs[c//4, c%4].axvspan(-(cueHit_diff_mean-cueHit_diff_sd),
            -(cueHit_diff_mean+cueHit_diff_sd),
            alpha=1, color='bisque', label='mean cue time ± sd')
        axs[c//4, c%4].axvspan(act_idx[0]+win[0], act_idx[-1]+win[0],
            alpha=0.5, color='lightsalmon', label='pre-movement window') #_ gets ignored as label# y lim+label
    # y label
    axs[0,0].text(0.04, 0.5, s='amplitude [$\mu$V]',
        transform = fig.transFigure, rotation='vertical',
        ha='left', va='center', clip_on=False)
    # x label
    axs[0,0].text(0.5, 0.04, s='time around response [ms]',
        transform = fig.transFigure,
        ha='center', va='bottom', clip_on=False)
    # legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) #delete doubles
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=[0.5, 0.01],
        bbox_transform = fig.transFigure, loc='lower center', ncol=2)
    fig.savefig(os.path.join(save_folder, 'motor_BP_2000mspreresponse.pdf'))


    # ERD in frequency bands
    try: # to read ERD [N_fbands*(32,2500)]
        with np.load(os.path.join(result_folder, 'motor/ERD.npz'),
            'r') as f_ERD:
            ERDs = list(f_ERD['ERD_{:02d}'.format(idx)])
    except FileNotFoundError: #calculate ERD and covmats
        ERDs = []
        contrast_cov_subj = []
        target_cov_subj = []
        for band in fbands:#later loop over all frequency bands
        # 1. band-pass filters with order 6 (3 into each direction)
            Wn = np.array(band) / s_rate * 2
            b, a = sp.signal.butter(3, Wn, btype='bandpass')
            eeg_filtbp = sp.signal.filtfilt(b, a, eeg)
            # calculate covariance matrices for csp
            all_trials_filtbp = meet.epochEEG(eeg_filtbp,
                    np.r_[snareHit_times[snareInlier],
                        wdBlkHit_times[wdBlkInlier]],
                    win)
            contrast_trials = all_trials_filtbp[:, base_idx,:] # baseline activity
            target_trials = all_trials_filtbp[:, act_idx,:] # pre-response activity (ERD/S)
            target_cov_now = np.einsum(
                'ijk, ljk -> ilk', target_trials, target_trials)
            contrast_cov_now = np.einsum(
                'ijk, ljk -> ilk', contrast_trials, contrast_trials)
            contrast_cov_subj.append(contrast_cov_now)
            target_cov_subj.append(target_cov_now)

            #2. Hilbert-Transform, absolute value
            eeg_filtHil = np.abs(sp.signal.hilbert(eeg_filtbp, axis=-1))
            #3. Normalisieren, so dass 1-2 sek prä-response 100% sind und dann averagen
            all_trials_filt = meet.epochEEG(eeg_filtHil,
                    np.r_[snareHit_times[snareInlier],
                        wdBlkHit_times[wdBlkInlier]],
                    win)
            # calculate ERD
            ERD = all_trials_filt.mean(-1) #trial avg
            ERD /= ERD[:,base_idx].mean(-1)[:,np.newaxis]
            #ERD *= 100
            #ERD = 20*np.log10(ERD) #in db: ERD_db.npz
            ERDs.append(ERD)
        contrast_cov.append(contrast_cov_subj)
        target_cov.append(target_cov_subj)

    Nw = 4 #width i.e. number of columns
    fig, axs = plt.subplots(int(np.ceil(Nc/Nw)), Nw, figsize=(7,10), #+1 for legend
            sharex=True, sharey=True)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.11, right=0.95, hspace=0.3)
    fig.suptitle('ERD: 2000 ms preresponse, trial-avg.', fontsize=12)
    for c in range(Nc):
        for i,ERD in enumerate(ERDs): #for each band
            ERD_topo = np.array(ERD)[topo_idx]
            axs[c//Nw, c%Nw].plot(range(*win), ERD_topo[c]*100, linewidth=1, c=colors[i],
                label = '_'*c + str(fbands[i][0])+'-'+str(fbands[i][1]) +' Hz')
        axs[c//Nw, c%Nw].set_title(channames_topo4x8[c], fontsize=8, pad=2)
        axs[c//Nw, c%Nw].axvline(0, lw=0.5, c='r')
        axs[c//Nw, c%Nw].axhline(100, lw=0.5, c='k', ls=':')
        axs[c//Nw, c%Nw].axvspan(-(cueHit_diff_mean-cueHit_diff_sd),
            -(cueHit_diff_mean+cueHit_diff_sd),
            alpha=1, color='bisque',
            label='_'*c + 'mean cue time ± sd') #_ gets ignored as label
        axs[c//Nw, c%Nw].axvspan(act_idx[0]+win[0], act_idx[-1]+win[0],
            alpha=0.5, color='lightsalmon',
            label='_'*c + 'pre-movement window') #_ gets ignored as label
    # y lim+label
    #axs[0,0].set_ylim([-3.5,3.5])
    axs[0,0].text(0.035, 0.5, s='ERD [%]',
        transform = fig.transFigure, rotation='vertical',
        ha='left', va='center', clip_on=False)
    # x label
    axs[0,0].text(0.5, 0.04, s='time around response [ms]',
        transform = fig.transFigure,
        ha='center', va='bottom', clip_on=False)
    # legend in last (empty) axes
    # [axs[c//Nw+1,i].axis('off') for i in range(Nw)]
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # axs[c//Nw+1,1].legend(lines[:int(np.ceil(len(fbands)/2))],
    #     labels[:int(np.ceil(len(fbands)/2))], bbox_to_anchor=[0.5,1.1],
    #     loc='upper center', fontsize=8)
    # axs[c//Nw+1,2].legend(lines[int(np.ceil(len(fbands)/2)):],
    #     labels[int(np.ceil(len(fbands)/2)):], bbox_to_anchor=[0.5,1.1],
    #     loc='upper center', fontsize=8)
    plt.legend(lines, labels, bbox_to_anchor=[0.5, -0.0],
        bbox_transform = fig.transFigure, loc='lower center', ncol=len(lines))
    fig.savefig(os.path.join(save_folder, 'motor_ERD_2000mspreresponse_new.pdf'))
    all_ERD.append(ERDs)

    idx += 1
    subj += 1
    plt.close('all')

################################################################################
##### save data #####
save_BP = {}
for i, (bp, bpt) in enumerate(zip(all_BP, all_BP_trials)):
    save_BP['BP_{:02d}'.format(i)] = bp
    save_BP['BP_trials_{:02d}'.format(i)] = bpt
np.savez(os.path.join(result_folder, 'motor/BP.npz'), **save_BP)

save_ERD = {}
for i, erd in enumerate(all_ERD):
    save_ERD['ERD_{:02d}'.format(i)] = erd
np.savez(os.path.join(result_folder, 'motor/ERD.npz'), **save_ERD,
    fbands=fbands)

save_covmat = {}
for i, (c,t,sn,wb) in enumerate(zip(contrast_cov, target_cov,
    all_snareHit_times, all_wdBlkHit_times)):
    save_covmat['contrast_cov_{:02d}'.format(i)] = c
    save_covmat['target_cov_{:02d}'.format(i)] = t
    save_covmat['snareHit_times_{:02d}'.format(i)] = sn
    save_covmat['wdBlkHit_times_{:02d}'.format(i)] = wb
np.savez(os.path.join(result_folder, 'motor/covmat.npz'),
    **save_covmat,
    fbands=fbands,
    left_handed=left_handed,
    base_idx = base_idx, #corresponds to -2000 to -1250ms
    act_idx = act_idx, #corresponds to -750 to 0ms)
    act_idx_lda = act_idx_lda) #-600 to -100ms

# store the inlier of the hit responses
save_inlier = {}
for i, (shI, whI, sI, wI) in enumerate(zip(all_snareHit_inlier,
        all_wdBlkHit_inlier,all_snareInlier, all_wdBlkInlier)):
    save_inlier['snareHit_inlier_{:02d}'.format(i)] = shI
    save_inlier['wdBlkHit_inlier_{:02d}'.format(i)] = whI
    save_inlier['snareInlier_response_{:02d}'.format(i)] = sI
    save_inlier['wdBlkInlier_response_{:02d}'.format(i)] = wI
    save_inlier['win'] = win
np.savez(os.path.join(result_folder, 'motor/inlier.npz'), **save_inlier)

################################################################################
##### plot for all subjects #####
cueHit_diff_mean = np.nanmean(np.hstack(cueHit_diff))
cueHit_diff_sd = np.nanstd(np.hstack(cueHit_diff))

##### motor/ERD_2000mspreresponse.pdf #####
# plot BP for all subjects
all_BP_avg = np.mean(all_BP, axis=0)[topo_idx]
fig, axs = plt.subplots(int(np.ceil(Nc/4)), 4, figsize=(7,10),
        sharex=True, sharey=True)
fig.tight_layout()
fig.subplots_adjust(top=0.94, bottom=0.07, left=0.11, right=0.95, hspace=0.3)
fig.suptitle('BP: 2000 ms preresponse, subj.- and trial-avg.', fontsize=14)
for c in range(Nc):
    axs[c//4, c%4].plot(range(*win), all_BP_avg[c], linewidth=1, c='k')
    axs[c//4, c%4].set_title(channames_topo4x8[c], fontsize=8, pad=2)
    axs[c//4, c%4].axhline(0, lw=0.5, c='k', ls=':')
    axs[c//4, c%4].axvline(0, lw=0.5, c='r')
    axs[c//4, c%4].axvspan(-(cueHit_diff_mean-cueHit_diff_sd),
        -(cueHit_diff_mean+cueHit_diff_sd),
        alpha=1, color='bisque', label='mean cue time ± sd')
    axs[c//4, c%4].axvspan(act_idx[0]+win[0], act_idx[-1]+win[0],
        alpha=0.5, color='lightsalmon', label='pre-movement window')
# y label
axs[0,0].text(0.035, 0.5, s='amplitude [$\mu$V]',
    transform = fig.transFigure, rotation='vertical',
    ha='left', va='center', clip_on=False)
# x label
axs[0,0].text(0.5, 0.035, s='time around response [ms]',
    transform = fig.transFigure,
    ha='center', va='bottom', clip_on=False)
# legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles)) #delete doubles
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=[0.5, 0.0],
        bbox_transform = fig.transFigure, loc='lower center', ncol=2)
fig.savefig(os.path.join(result_folder, 'motor/BP_2000mspreresponse.pdf'))

##### motor/ERD_2000mspreresponse.pdf #####
# plot ERD for all subjects
Nw = 4 #width i.e. number of columns
all_ERD_avg = [np.mean([i[j][topo_idx] for i in all_ERD], axis=0)
    for j in range(len(fbands))] # for each band, average over subjects

fig, axs = plt.subplots(int(np.ceil(Nc/Nw)), Nw, figsize=(7,10), #+1 for legend
        sharex=True, sharey=True)
fig.subplots_adjust(top=0.94, bottom=0.07, left=0.11, right=0.95, hspace=0.3)
fig.suptitle('ERD: 2000 ms preresponse, trial-avg.', fontsize=14)
for c in range(Nc):
    for i,ERD in enumerate(all_ERD_avg):
        axs[c//Nw, c%Nw].plot(range(*win), ERD[c]*100, linewidth=1, c=colors[i],
            label = '_'*c + str(fbands[i][0])+'-'+str(fbands[i][1]) +' Hz')
    axs[c//Nw, c%Nw].set_title(channames_topo4x8[c], fontsize=8, pad=2)
    axs[c//Nw, c%Nw].axvline(0, lw=0.5, c='r')
    axs[c//Nw, c%Nw].axhline(100, lw=0.5, c='k', ls=':')
    axs[c//Nw, c%Nw].axvspan(-(cueHit_diff_mean-cueHit_diff_sd),
        -(cueHit_diff_mean+cueHit_diff_sd),
        alpha=1, color='bisque',
        label='_'*c + 'mean cue time ± sd') #_ gets ignored as label
    axs[c//Nw, c%Nw].axvspan(act_idx[0]+win[0], act_idx[-1]+win[0],
        alpha=0.5, color='lightsalmon',
            label='_'*c + 'pre-movement window') #_ gets ignored as label
#axs[0,0].set_ylim([-3.5,3.5])
axs[0,0].text(0.035, 0.5, s='ERD [%]', #change back to amplitude?
    transform = fig.transFigure, rotation='vertical',
    ha='left', va='center', clip_on=False)
# x label
axs[0,0].text(0.5, 0.035, s='time around response [ms]',
    transform = fig.transFigure,
    ha='center', va='bottom', clip_on=False)
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
plt.legend(lines, labels, bbox_to_anchor=[0.5, -0.0],
    bbox_transform = fig.transFigure, loc='lower center', ncol=len(lines))
fig.savefig(os.path.join(result_folder, 'motor/ERD_2000mspreresponse.pdf'))

################################################################################
# calculate p-Values and plot channel spectra to determine frequency bands
data = np.concatenate(all_BP_trials,axis=-1)[topo_idx]
F, psd_pre_act = sp.signal.welch(
        data[:,act_idx], fs=s_rate, nperseg=2000, nfft=2000, scaling='density', axis=1)
F, psd_pre_base = sp.signal.welch(
        data[:,base_idx], fs=s_rate, nperseg=2000, nfft=2000, scaling='density', axis=1)

uHz = 30 #upper Hz bound

##### motor/ERD_p.npy #####
# bootstrap to calculate p-value for act vs base
try:
    ERD_p = np.load(os.path.join(result_folder,'motor/ERD_p.npy'))
except FileNotFoundError:
    N_bootstrap = 10000
    # ERD for each channel and frequency:
    print('ERD_p.npy not found. Calculating p-Values with N_boot=',N_bootstrap)
    ERD_cf = psd_pre_act.mean(-1) / psd_pre_base.mean(-1)
    N_trials = psd_pre_act.shape[-1]
    # same for bootstrap ERD, sampling N_trials with replacement
    data_bootstrap = np.concatenate([psd_pre_act, psd_pre_base], -1)
    ERD_bootstrap = np.array(
            [data_bootstrap
                    [...,np.random.choice(N_trials, N_trials)].mean(-1
            )/data_bootstrap
                    [...,np.random.choice(N_trials, N_trials)].mean(-1)
        for _ in tqdm(range(N_bootstrap))])
    #p-value: how often is bootstrap stronger, two-tailed (use log)
    ERD_p = np.sum(np.abs(np.log(ERD_bootstrap)
        ) >= np.abs(np.log(ERD_cf)), axis=0)/(N_bootstrap + 1)
    ERD_p = ERD_p * (uHz/F[1]) #bonferroni correction
    np.save(os.path.join(result_folder, 'motor/ERD_p.npy'), ERD_p)


grid = [8,4] # plot with 8 rows and 4 columns

##### motor/channelSpectra0-30.pdf #####
fig = plt.figure(figsize=(10,10))
gs = mpl.gridspec.GridSpec(grid[0], grid[1], height_ratios = grid[0]*[1])
ax = []
for i, (psd_act_now,psd_base_now) in enumerate(
    zip(psd_pre_act.mean(-1),psd_pre_base.mean(-1))):
    if i == 0:
        ax.append(fig.add_subplot(gs[0,0]))
    else:
        ax.append(fig.add_subplot(gs[i//4,i%4], sharex=ax[0], sharey=ax[0]))
    ax[-1].plot(F, np.sqrt(psd_act_now)*1000, c='lightsalmon', label='pre-movement')
    ax[-1].plot(F, np.sqrt(psd_base_now)*1000, c='k', label='baseline')
    ax[-1].grid(ls=':', alpha=0.8)
    ax[-1].set_title(channames_topo4x8[i])
    ax[-1].fill_between(F, 0, 1, where=ERD_p[i]<0.05, alpha=0.2, color='k',
        transform=ax[-1].get_xaxis_transform(), label='p$<$0.05')
    if i>grid[0]*grid[1] - (grid[1]+1): #last row
        ax[-1].set_xlabel('frequency (Hz)')
    if i%(grid[1]*2) == 0: #first column, every other row
        ax[-1].set_ylabel('linear spectral density')
#ax[-1].set_yscale('log')
ax[-1].set_xticks([1,5,10,20,50,100])
ax[-1].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
plt.legend(lines, labels, bbox_to_anchor=[0.5, -0.0],
    bbox_transform = fig.transFigure, loc='upper center', ncol=len(lines))
plt.legend(loc='upper right')
#ax[-1].set_ylim([1E1, 1E5])
ax[-1].set_xticks(range(0,uHz,2))
ax[-1].set_xlim(xmin=0,xmax=uHz)
fig.suptitle('Channel Spectra, subj.-avg.', size=14)
gs.tight_layout(fig, pad=0.3, rect=(0,0,1,0.95))
fig.savefig(os.path.join(result_folder,'motor/channelSpectra0-30.pdf'))
plt.close(fig)

##### motor/channelSpectra_ERD0-30.pdf #####
fig = plt.figure(figsize=(10,10))
gs = mpl.gridspec.GridSpec(grid[0],grid[1], height_ratios = grid[0]*[1])
ax = []
for i, (psd_act_now,psd_base_now) in enumerate(
    zip(psd_pre_act.mean(-1),psd_pre_base.mean(-1))):
    if i == 0:
        ax.append(fig.add_subplot(gs[0,0]))
    else:
        ax.append(fig.add_subplot(gs[i//4,i%4], sharex=ax[0], sharey=ax[0]))
    ax[-1].plot(F, psd_act_now / psd_base_now *100, c='r', label='act')
    #ax[-1].plot(F, ERD_p[i], c='r', label='act')
    #ax[-1].axhline(0.05)
    ax[-1].grid(ls=':', alpha=0.8)
    ax[-1].set_title(channames_topo4x8[i])
    if i>grid[0]*grid[1] - (grid[1]+1): #last row
        ax[-1].set_xlabel('frequency (Hz)')
    if i%grid[1] == 0: #first column
        ax[-1].set_ylabel('ERD/S [\%]')
    ax[-1].fill_between(F, 0, 1, where=ERD_p[i]<0.05, alpha=0.2, color='k',
        transform=ax[-1].get_xaxis_transform(), label='p$<$0.05')
ax[-1].set_xticks(range(0,30,2))
ax[-1].set_xlim(xmin=0,xmax=30)
ax[-1].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
fig.suptitle('Channel Spectra, subj.-avg., is (TP9+TP10)/2', size=14)
gs.tight_layout(fig, pad=0.3, rect=(0,0,1,0.95))
fig.savefig(os.path.join(result_folder,'motor/channelSpectra_ERD0-30.pdf'))
plt.close(fig)
