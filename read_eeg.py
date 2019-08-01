import numpy as np
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

# now, find the sample of each Cue
snareCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
        snareCue_nearestClock, snareCue_DevToClock)
wdBlkCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
        wdBlkCue_nearestClock, wdBlkCue_DevToClock)

# read the cleaned EEG and the artifact segment mask
with np.load(eeg_fname, allow_pickle=True) as npzfile:
    EEG = npzfile['clean_data']
    artifact_mask = npzfile['artifact_mask']

# apply a 0.5 Hz high-pass filter
EEG_hp = meet.iir.butterworth(EEG, fs=(0.4, 30), fp=(0.5, 20), s_rate=s_rate)

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
all_trials = meet.epochEEG(EEG_hp,
        np.r_[snareListenMarker[snareInlier],
            wdBlkListenMarker[wdBlkInlier]],
        all_win)
# calculate the average
all_trials_avg = all_trials.mean(-1)

fig = plt.figure(figsize=(12,10))
gs = mpl.gridspec.GridSpec(EEG.shape[0], 1)
axes = []
for i, avg in enumerate(all_trials_avg):
    if i == 0:
        axes.append(fig.add_subplot(gs[i], frame_on=False))
    else:
        axes.append(fig.add_subplot(gs[i], sharex=axes[0], sharey=axes[0],
                frame_on=False))
    axes[i].plot(t_all, avg, 'k-')
    axes[i].set_ylabel(channames[i])
    axes[i].tick_params(**blind_ax)
axes[-1].tick_params(bottom=True, labelbottom=True)

complete_ax = fig.add_subplot(gs[:], frame_on=False, sharex=axes[0])
complete_ax.tick_params(**blind_ax)

[complete_ax.axvline(t, c=color1, lw=4, alpha=0.8)
        for t in np.arange(0,3*bar_duration,
    1./snareFreq)]
[complete_ax.axvline(t, c=color2, lw=4, alpha=0.8)
        for t in np.arange(0,3*bar_duration,
    1./wdBlkFreq)]
[complete_ax.axvline(t, c='k', lw=1, alpha=1)
        for t in np.arange(0,5*bar_duration,
    bar_duration)]

axes[-1].set_xlabel('time after 1st beat (s)')
axes[0].set_title('event related potential', size=12)

fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(save_folder, 'ERP_all_channels.pdf'))
fig.savefig(os.path.join(save_folder, 'ERP_all_channels.png'))

ERP_st_filter, ERP_avg_filter, ERP_corr = meet.spatfilt.CCAvReg(all_trials)
ERP_pattern = np.linalg.pinv(ERP_st_filter)
ERP_pattern /= np.abs(ERP_pattern).max(1)[:,np.newaxis]
ERP_X, ERP_Y, ERP_Z = zip(*[meet.sphere.potMap(chancoords, pat)
    for pat in ERP_pattern[:5]])

CCA_trials_avg = ERP_st_filter.T.dot(all_trials_avg)

# plot the 5 best components
fig = plt.figure(figsize=(10,6))
head_axes = []
ERP_axes = []
gs = mpl.gridspec.GridSpec(5, 2, width_ratios=(1,8))
for i in xrange(5):
    if i == 0:
        head_axes.append(fig.add_subplot(gs[i,0], frame_on=False))
    else:
        head_axes.append(fig.add_subplot(gs[i,0], frame_on=False,
                sharex=head_axes[0], sharey=head_axes[0]))
    head_axes[i].tick_params(**blind_ax)
    head_axes[i].pcolormesh(ERP_X[i], ERP_Y[i], ERP_Z[i],
            cmap=cmap, vmin=-1, vmax=1, rasterized=True)
    head_axes[i].contour(ERP_X[i], ERP_Y[i], ERP_Z[i],
            levels=[0], colors='w')
    head_axes[i].set_xlabel(r'$\rho_%d=%.2f$' % (i+1, ERP_corr[i]))
    meet.sphere.addHead(head_axes[i])
    if i == 0:
        ERP_axes.append(fig.add_subplot(gs[i,1], frame_on=True))
    else:
        ERP_axes.append(fig.add_subplot(gs[i,1], frame_on=True,
                sharex=ERP_axes[0], sharey=ERP_axes[0]))
    ERP_axes[-1].plot(t_all, CCA_trials_avg[i], c='k')
    ERP_axes[-1].tick_params(**blind_ax)

complete_ax = fig.add_subplot(gs[:,1], frame_on=False, sharex=ERP_axes[0])
complete_ax.tick_params(**blind_ax)

[complete_ax.axvline(t, c=color1, lw=2, alpha=0.8)
        for t in np.arange(0,3*bar_duration,
    1./snareFreq)]
[complete_ax.axvline(t, c=color2, lw=2, alpha=0.8)
        for t in np.arange(0,3*bar_duration,
    1./wdBlkFreq)]
[complete_ax.axvline(t, c='k', lw=1, alpha=1)
        for t in np.arange(0,5*bar_duration,
    bar_duration)]

ERP_axes[-1].tick_params(bottom=True, labelbottom=True)
ERP_axes[-1].set_xlabel('time after 1st beat (s)')
ERP_axes[0].set_xlim([t_all[0], t_all[-1]])

title_ax = fig.add_subplot(gs[:,:], frame_on=False)
title_ax.tick_params(**blind_ax)
title_ax.set_title('CCA averages')
fig.tight_layout(pad=0.3)

fig.savefig(os.path.join(save_folder, 'ERP_CCA_channels.pdf'))
fig.savefig(os.path.join(save_folder, 'ERP_CCA_channels.png'))


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

# filter the EEG between 0.5 and 2.5 Hz, get the listening trials and fit
# the sines and cosines
EEG_lowfreq = meet.iir.butterworth(EEG, fp=(0.5, 2.5), fs=(0.25, 3),
        s_rate=s_rate, axis=-1)
snareListenData = meet.epochEEG(EEG_lowfreq, snareListenMarker[snareInlier],
        listen_win)
wdBlkListenData = meet.epochEEG(EEG_lowfreq, wdBlkListenMarker[wdBlkInlier],
        listen_win)

snareListenData_broadband = meet.epochEEG(EEG,
        snareListenMarker[snareInlier], listen_win)
wdBlkListenData_broadband = meet.epochEEG(EEG,
        wdBlkListenMarker[wdBlkInlier], listen_win)

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

# calculate spatial filters enhancing the reconstructed data with only the
# two oscillations included - this is in the end
# a sort of SSD (spatial spectral decomposition)
# enhance the frequencies were searching for (douple/tripple beat)
# and suppress other frequencies
ssd_filter, ssd_eigvals = meet.spatfilt.CSP(
        np.concatenate((snareListenData_rec, wdBlkListenData_rec),
            axis=2).reshape(
                EEG.shape[0], -1),
        np.concatenate((snareListenData, wdBlkListenData),
            axis=2).reshape(
                EEG.shape[0], -1)
            )

ssd_pattern = np.linalg.inv(ssd_filter)

# plot the patterns
# name the ssd channels
ssd_channames = ['SSD%02d' % i for i in xrange(len(ssd_pattern))]

# plot the ICA  components scalp maps
ssd_potmaps = [meet.sphere.potMap(chancoords, ssd_c,
    projection='stereographic') for ssd_c in ssd_pattern]

fig = plt.figure(figsize=(4.5,10))
# plot with 8 rows and 4 columns
gs = mpl.gridspec.GridSpec(10,4, height_ratios = 8*[1]+[0.2]+[1])
ax = []
for i, (X,Y,Z) in enumerate(ssd_potmaps):
    if i == 0:
        ax.append(fig.add_subplot(gs[0,0], frame_on = False))
    else:
        ax.append(fig.add_subplot(gs[i//4,i%4], sharex=ax[0], sharey=ax[0],
                frame_on = False))
    Z /= np.abs(Z).max()
    ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(ax[-1])
    pc = ax[-1].pcolormesh(X, Y, Z, vmin=-1, vmax=1, rasterized=True,
            cmap=cmap)
    ax[-1].contour(X, Y, Z, levels=[0], colors='w')
    ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5)
    ax[-1].set_title(ssd_channames[i])
pc_ax = fig.add_subplot(gs[-2,:])
plt.colorbar(pc, cax=pc_ax, orientation='horizontal',
        label='relative amplitude')
pc_ax.plot([0.5,0.5], [0,1], c='w', zorder=1000,
        transform=pc_ax.transAxes)
eigvals_ax = fig.add_subplot(gs[-1,:])
eigvals_ax.plot(ssd_eigvals)
eigvals_ax.set_title('SSD eigenvalues')
fig.suptitle('SSD patterns, Subject S%02d' % subject, size=14)
gs.tight_layout(fig, pad=0.3, rect=(0,0,1,0.95))
fig.savefig(os.path.join(save_folder, 'SSD_patterns.pdf'))

snareListenBestFit = np.tensordot(ssd_filter[:,0],
        snareFit, axes=(0,0))[2:,:]
wdBlkListenBestFit = np.tensordot(ssd_filter[:,0],
        wdBlkFit, axes=(0,0))[2:,:]

snareListenBestAmp = np.array([
    np.sqrt(np.sum(snareListenBestFit[:2]**2, 0)),
    np.sqrt(np.sum(snareListenBestFit[2:]**2, 0)),
    ])
wdBlkListenBestAmp = np.array([
    np.sqrt(np.sum(wdBlkListenBestFit[:2]**2, 0)),
    np.sqrt(np.sum(wdBlkListenBestFit[2:]**2, 0)),
    ])

snareListenBestPhase = np.array([
    np.arctan2(*snareListenBestFit[:2]),
    np.arctan2(*snareListenBestFit[2:])
    ])
wdBlkListenBestPhase = np.array([
    np.arctan2(*wdBlkListenBestFit[:2]),
    np.arctan2(*wdBlkListenBestFit[2:])
    ])

# now, with the best-filtered data, fit the frequencies to the silence
# period
filt_EEG_lowfreq = np.dot(ssd_filter[:,0], EEG_lowfreq)
snareSilenceData = meet.epochEEG(filt_EEG_lowfreq,
        snareListenMarker[snareInlier], silence_win)
wdBlkSilenceData = meet.epochEEG(filt_EEG_lowfreq,
        wdBlkListenMarker[wdBlkInlier], silence_win)

silence_fit_matrix = np.array([
    np.ones_like(t_silence),
    t_silence,
    np.cos(2*np.pi*snareFreq*t_silence),
    np.sin(2*np.pi*snareFreq*t_silence),
    np.cos(2*np.pi*wdBlkFreq*t_silence),
    np.sin(2*np.pi*wdBlkFreq*t_silence)
    ]).T

snareFit_silence = np.linalg.lstsq(silence_fit_matrix, snareSilenceData)[0]
wdBlkFit_silence = np.linalg.lstsq(silence_fit_matrix, wdBlkSilenceData)[0]

snareSilenceBestAmp = np.array([
    np.sqrt(np.sum(snareFit_silence[2:4]**2, 0)),
    np.sqrt(np.sum(snareFit_silence[4:6]**2, 0))
    ])
wdBlkSilenceBestAmp = np.array([
    np.sqrt(np.sum(wdBlkFit_silence[2:4]**2, 0)),
    np.sqrt(np.sum(wdBlkFit_silence[4:6]**2, 0))
    ])

snareSilenceBestPhase = np.array([
    np.arctan2(*snareFit_silence[2:4]),
    np.arctan2(*snareFit_silence[4:6])
    ])
wdBlkSilenceBestPhase = np.array([
    np.arctan2(*wdBlkFit_silence[2:4]),
    np.arctan2(*wdBlkFit_silence[4:6])
    ])

#save the eeg results
np.savez(os.path.join(save_folder, 'eeg_results.npz'),
    snareListenBestAmp = snareListenBestAmp,
    wdBlkListenBestAmp = wdBlkListenBestAmp,
    snareListenBestPhase = snareListenBestPhase,
    wdBlkListenBestPhase = wdBlkListenBestPhase,
    snareSilenceBestAmp = snareSilenceBestAmp,
    wdBlkSilenceBestAmp = wdBlkSilenceBestAmp,
    snareSilenceBestPhase = snareSilenceBestPhase,
    wdBlkSilenceBestPhase = wdBlkSilenceBestPhase,
    snareInlier = snareInlier,
    wdBlkInlier = wdBlkInlier
    )


#plot oscillation amplitude/phase vs performance
# 1a Listen, Amplitude
r_snareListenAmp = np.corrcoef(snareListenBestAmp[0],
        np.abs(np.concatenate(snareCue_DevToClock))[snareInlier])[0][1]
r_wdBlkListenAmp = np.corrcoef(wdBlkListenBestAmp[0],
        np.abs(np.concatenate(wdBlkCue_DevToClock))[wdBlkInlier])[0][1]
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax1.set_title('oscillation amplitude in listening period vs snare performance, r = %02f'
        % r_snareListenAmp)
ax1.plot(snareListenBestAmp[0],
        np.abs(np.concatenate(snareCue_DevToClock))[snareInlier], 'ro')
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.set_title('oscillation amplitude in listening period vs wdBlk performance, r = %02f'
        % r_wdBlkListenAmp)
ax2.plot(wdBlkListenBestAmp[1],
        np.abs(np.concatenate(wdBlkCue_DevToClock))[wdBlkInlier], 'bo')
plt.xlabel('Oscillation Amplitude')
plt.ylabel('Absolute Deviation')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(save_folder, 'OszillationAmp_Performance_Listen.png'))
fig.savefig(os.path.join(save_folder, 'OszillationAmp_Performance_Listen.pdf'))

# 1b Listen, Phase
r_snareListenPhase = np.corrcoef(snareListenBestPhase[0],
        np.abs(np.concatenate(snareCue_DevToClock))[snareInlier])[0][1]
r_wdBlkListenPhase = np.corrcoef(wdBlkListenBestPhase[0],
        np.abs(np.concatenate(wdBlkCue_DevToClock))[wdBlkInlier])[0][1]
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax1.set_title('oscillation phase in listening period vs snare performance, r = %02f'
        % r_snareListenPhase)
ax1.plot(snareListenBestPhase[0],
        np.abs(np.concatenate(snareCue_DevToClock))[snareInlier], 'ro')
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.set_title('oscillation phase in listening period vs wdBlk performance, r = %02f'
        % r_wdBlkListenPhase)
ax2.plot(wdBlkListenBestPhase[1],
        np.abs(np.concatenate(wdBlkCue_DevToClock))[wdBlkInlier], 'bo')
plt.xlabel('Oscillation Phase')
plt.ylabel('Absolute Deviation')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(save_folder, 'OszillationPhase_Performance_Listen.png'))
fig.savefig(os.path.join(save_folder, 'OszillationPhase_Performance_Listen.pdf'))


# 2a. Silence, Amplitude
r_snareSilenceAmp = np.corrcoef(snareSilenceBestAmp[0],
        np.abs(np.concatenate(snareCue_DevToClock))[snareInlier])[0][1]
r_wdBlkSilenceAmp = np.corrcoef(wdBlkSilenceBestAmp[0],
        np.abs(np.concatenate(wdBlkCue_DevToClock))[wdBlkInlier])[0][1]
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax1.set_title('oscillation amplitude in silence period vs snare performance, r = %02f'
        % r_snareSilenceAmp)
ax1.plot(snareSilenceBestAmp[0],
        np.abs(np.concatenate(snareCue_DevToClock))[snareInlier], 'ro')
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.set_title('oscillation amplitude in silence period vs wdBlk performance, r = %02f'
        % r_wdBlkSilenceAmp)
ax2.plot(wdBlkSilenceBestAmp[0],
        np.abs(np.concatenate(wdBlkCue_DevToClock))[wdBlkInlier], 'bo')
plt.xlabel('Oscillation Amplitude')
plt.ylabel('Absolute Deviation')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(save_folder, 'OszillationAmp_Performance_Silence.png'))
fig.savefig(os.path.join(save_folder, 'OszillationAmp_Performance_Silence.pdf'))

# 2b. Silence, Phase
r_snareSilencePhase = np.corrcoef(snareSilenceBestPhase[0],
        np.abs(np.concatenate(snareCue_DevToClock))[snareInlier])[0][1]
r_wdBlkSilencePhase = np.corrcoef(wdBlkSilenceBestPhase[0],
        np.abs(np.concatenate(wdBlkCue_DevToClock))[wdBlkInlier])[0][1]
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax1.set_title('oscillation phase in silence period vs snare performance, r = %02f'
        % r_snareSilencePhase)
ax1.plot(snareSilenceBestPhase[0],
        np.abs(np.concatenate(snareCue_DevToClock))[snareInlier], 'ro')
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.set_title('oscillation phase in silence period vs wdBlk performance, r = %02f'
        % r_wdBlkSilencePhase)
ax2.plot(wdBlkSilenceBestPhase[0],
        np.abs(np.concatenate(wdBlkCue_DevToClock))[wdBlkInlier], 'bo')
plt.xlabel('Oscillation Phase')
plt.ylabel('Absolute Deviation')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(save_folder, 'OszillationPhase_Performance_Silence.png'))
fig.savefig(os.path.join(save_folder, 'OszillationPhase_Performance_Silence.pdf'))
