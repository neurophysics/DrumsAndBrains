import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path

data_folder = sys.argv[1]
result_folder = sys.argv[2]
N_subjects = 21

if not os.path.exists(result_folder):
    os.mkdir(result_folder)

self_perf = np.zeros((N_subjects,6))
self_vigil = np.zeros((N_subjects,6))
meandevs_snare = np.zeros((N_subjects,6))
meandevs_wdBlk = np.zeros((N_subjects,6))
snareCue_DevToClock = np.zeros((N_subjects, 75))
wdBlkCue_DevToClock = np.zeros((N_subjects, 75))

for subject in range(1,N_subjects+1):
    current_data_folder = os.path.join(data_folder, 'S%02d' % subject)

    # Collect self assessment of performance and vigilance data
    p = 0
    v = 0
    with open(os.path.join(current_data_folder,
            'S%02d_self-assessment.txt' % subject)) as f:
        for line in f:
            l = line.split(' ')
            if line[0] == 'P':
                self_perf[subject-1][p] = float(l[1])
                p+=1
            if line[0] == 'V':
                self_vigil[subject-1][v] = float(l[1])
                v+=1

    # collect real performance data
    with np.load(os.path.join(result_folder, 'S%02d' % subject,
            'behavioural_results.npz'), 'r', allow_pickle=True) as f:
        snareDev = np.abs(f['snareCue_DevToClock'])
        wdBlkDev = np.abs(f['wdBlkCue_DevToClock'])
        snareCue_DevToClock[subject-1] = np.concatenate(snareDev)
        wdBlkCue_DevToClock[subject-1] = np.concatenate(wdBlkDev)
        for i in range(0,6):
            meandevs_snare[subject-1][i] = [np.mean(s)
                    for s in np.abs(snareDev)][i] #mean dev per session
            meandevs_wdBlk[subject-1][i] = [np.mean(s)
                    for s in np.abs(wdBlkDev)][i]

# normalize
self_perf = (self_perf - np.mean(self_perf,0))/self_perf.std(0)
self_vigil = (self_vigil - np.mean(self_vigil,0))/self_vigil.std(0)
#calculate correlation coefficients (see plot title)
r_snarePerf = np.corrcoef(np.concatenate(meandevs_snare),
        np.concatenate(self_perf))[0][1]
r_snareVigil = np.corrcoef(np.concatenate(meandevs_snare),
        np.concatenate(self_vigil))[0][1]
r_wdBlkPerf = np.corrcoef(np.concatenate(meandevs_wdBlk),
        np.concatenate(self_perf))[0][1]
r_wdBlkVigil = np.corrcoef(np.concatenate(meandevs_wdBlk),
        np.concatenate(self_vigil))[0][1]

# plot - one for snare, one for wdBlk
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax1.set_title('self assessment vs real snare performance (r = %02f / %02f)'
        % (r_snarePerf, r_snareVigil))
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.set_title('self assessment vs real wdBlk performance (r = %02f / %02f)'
        % (r_wdBlkPerf, r_wdBlkVigil))
for subject in range(0,N_subjects):
    #ax1.set_ylim(0.,0.8)
    if subject== 0:
        ax1.plot(self_perf[subject],
                meandevs_snare[subject], 'ro',
                label = 'Performance Assessment')
        ax1.plot(self_vigil[subject],
                meandevs_snare[subject], 'bo',
                label = 'Vigilance Assessment')
        ax2.plot(self_perf[subject],
                meandevs_wdBlk[subject], 'ro',
                label = 'Performance Assessment')
        ax2.plot(self_vigil[subject],
                meandevs_wdBlk[subject], 'bo',
                label = 'Vigilance Assessment')
    else:
        ax1.plot(self_perf[subject],
                meandevs_snare[subject], 'ro')
        ax1.plot(self_vigil[subject],
                meandevs_snare[subject], 'bo')
        ax2.plot(self_perf[subject],
                meandevs_wdBlk[subject], 'ro')
        ax2.plot(self_vigil[subject],
                meandevs_wdBlk[subject], 'bo')
ax1.legend()
ax2.legend()
plt.xlabel('self assessment')
plt.ylabel('absolute deviation')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(result_folder, 'SelfAssessmentAndResponse.png'))
fig.savefig(os.path.join(result_folder, 'SelfAssessmentAndResponse.pdf'))


#plot oscillation amplitude vs performance
# read eeg data:
snareInlier = np.zeros((N_subjects,75))
wdBlkInlier = np.zeros((N_subjects,75))
snareListenBestAmp = np.zeros((N_subjects,75))
wdBlkListenBestAmp = np.zeros((N_subjects,75))
snareListenBestPhase = np.zeros((N_subjects,75))
wdBlkListenBestPhase = np.zeros((N_subjects,75))
snareSilenceBestAmp = np.zeros((N_subjects,75))
wdBlkSilenceBestAmp = np.zeros((N_subjects,75))
snareSilenceBestPhase = np.zeros((N_subjects,75))
wdBlkSilenceBestPhase = np.zeros((N_subjects,75))
for subject in range(1,N_subjects+1):
    if subject==11:
        continue #no eeg for subject 11
    with np.load(os.path.join(result_folder, 'S%02d' % subject,
            'eeg_results.npz'), 'r', allow_pickle=True) as f:
        snareInlier[subject-1] = f['snareInlier']
        snarePos = [x-i for i,x in enumerate(
                np.where([not s for s in snareInlier[subject-1]])[0])]
        #print snarePos
        wdBlkInlier[subject-1] = f['wdBlkInlier']
        wdBlkPos = [x-i for i,x in enumerate(
                np.where([not s for s in wdBlkInlier[subject-1]])[0])]
        snareListenBestAmp[subject-1] = np.insert(
                f['snareListenBestAmp'][0], snarePos, np.nan, axis=0)
        wdBlkListenBestAmp[subject-1] = np.insert(
                f['wdBlkListenBestAmp'][0], wdBlkPos, np.nan, axis=0)
        snareListenBestPhase[subject-1] = np.insert(
                f['snareListenBestPhase'][0], snarePos, np.nan, axis=0)
        wdBlkListenBestPhase[subject-1] = np.insert(
                f['wdBlkListenBestPhase'][0], wdBlkPos, np.nan, axis=0)
        snareSilenceBestAmp[subject-1] = np.insert(
                f['snareSilenceBestAmp'][0], snarePos, np.nan, axis=0)
        wdBlkSilenceBestAmp[subject-1] = np.insert(
                f['wdBlkSilenceBestAmp'][0], wdBlkPos, np.nan, axis=0)
        snareSilenceBestPhase[subject-1] = np.insert(
                f['snareSilenceBestPhase'][0], snarePos, np.nan, axis=0)
        wdBlkSilenceBestPhase[subject-1] = np.insert(
                f['wdBlkSilenceBestPhase'][0], wdBlkPos, np.nan, axis=0)

# average over all subjects
mean_snareCue_DevToClock = []
mean_wdBlkCue_DevToClock = []
mean_snareListenBestAmp = []
mean_wdBlkListenBestAmp = []
mean_snareListenBestPhase = []
mean_wdBlkListenBestPhase = []
mean_snareSilenceBestAmp = []
mean_wdBlkSilenceBestAmp = []
mean_snareSilenceBestPhase = []
mean_wdBlkSilenceBestPhase = []
for trial in range(0, 75):
    snareDevs = []
    wdBlkDevs = []
    snareListenAmps = []
    wdBlkListenAmps = []
    snareListenPhases = []
    wdBlkListenPhases = []
    snareSilenceAmps = []
    wdBlkSilenceAmps = []
    snareSilencePhases = []
    wdBlkSilencePhases = []
    for subject in range(0, N_subjects):
        if snareInlier[subject][trial]: #only append trials of subjects that are Inlier
            #print subject, trial, snareInlier[subject][trial]
            snareDevs.append(snareCue_DevToClock[subject][trial])
            snareListenAmps.append(snareListenBestAmp[subject][trial])
            snareListenPhases.append(snareListenBestPhase[subject][trial])
            snareSilenceAmps.append(snareSilenceBestAmp[subject][trial])
            snareSilencePhases.append(snareSilenceBestPhase[subject][trial])
        if wdBlkInlier[subject][trial]: #only append trials of subjects that are Inlier
            wdBlkDevs.append(wdBlkCue_DevToClock[subject][trial])
            wdBlkListenAmps.append(wdBlkListenBestAmp[subject][trial])
            wdBlkListenPhases.append(wdBlkListenBestPhase[subject][trial])
            wdBlkSilenceAmps.append(wdBlkSilenceBestAmp[subject][trial])
            wdBlkSilencePhases.append(wdBlkSilenceBestPhase[subject][trial])
    mean_snareCue_DevToClock.append(np.mean(snareDevs))
    mean_wdBlkCue_DevToClock.append(np.mean(wdBlkDevs))
    mean_snareListenBestAmp.append(np.mean(snareListenAmps))
    mean_wdBlkListenBestAmp.append(np.mean(wdBlkListenAmps))
    mean_snareListenBestPhase.append(np.mean(snareListenPhases))
    mean_wdBlkListenBestPhase.append(np.mean(wdBlkListenPhases))
    mean_snareSilenceBestAmp.append(np.mean(snareSilenceAmps))
    mean_wdBlkSilenceBestAmp.append(np.mean(wdBlkSilenceAmps))
    mean_snareSilenceBestPhase.append(np.mean(snareSilencePhases))
    mean_wdBlkSilenceBestPhase.append(np.mean(wdBlkSilencePhases))

# 1a. Listen, Amplitude
r_snareListenAmp = np.corrcoef(mean_snareListenBestAmp,
        mean_snareCue_DevToClock)[0][1]
r_wdBlkListenAmp = np.corrcoef(mean_wdBlkListenBestAmp,
        mean_wdBlkCue_DevToClock)[0][1]
fig = plt.figure(figsize=(10,10))
plt.title('oscillation amplitude in listening period vs snare performance')
plt.plot(mean_snareListenBestAmp, mean_snareCue_DevToClock, 'ro',
        label = 'Snare, r = %02f' % r_snareListenAmp)
plt.plot(mean_wdBlkListenBestAmp, mean_wdBlkCue_DevToClock, 'bo',
        label = 'Wood Block, r = %02f' % r_wdBlkListenAmp)
plt.legend()
plt.xlabel('Oscillation Amplitude')
plt.ylabel('Absolute Deviation')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(result_folder, 'ListeningAmp_Performance.png'))
fig.savefig(os.path.join(result_folder, 'ListeningAmp_Performance.pdf'))

# 1b. Listen, Phase
r_snareListenPhase = np.corrcoef(mean_snareListenBestPhase,
        mean_snareCue_DevToClock)[0][1]
r_wdBlkListenPhase = np.corrcoef(mean_wdBlkListenBestPhase,
        mean_wdBlkCue_DevToClock)[0][1]
fig = plt.figure(figsize=(10,10))
plt.title('oscillation phase in listening period vs snare performance')
plt.plot(mean_snareListenBestPhase, mean_snareCue_DevToClock, 'ro',
        label = 'Snare, r = %02f' % r_snareListenPhase)
plt.plot(mean_wdBlkListenBestPhase, mean_wdBlkCue_DevToClock, 'bo',
        label = 'Wood Block, r = %02f' % r_wdBlkListenPhase)
plt.legend()
plt.xlabel('Oscillation Phase')
plt.ylabel('Absolute Deviation')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(result_folder, 'ListeningPhase_Performance.png'))
fig.savefig(os.path.join(result_folder, 'ListeningPhase_Performance.pdf'))

# 2a. Silence, Amplitude
r_snareSilenceAmp = np.corrcoef(mean_snareSilenceBestAmp,
        mean_snareCue_DevToClock)[0][1]
r_wdBlkSilenceAmp = np.corrcoef(mean_wdBlkSilenceBestAmp,
        mean_wdBlkCue_DevToClock)[0][1]
fig = plt.figure(figsize=(10,10))
plt.title('oscillation amplitude in silence period vs snare performance')
plt.plot(mean_snareSilenceBestAmp, mean_snareCue_DevToClock, 'ro',
        label = 'Snare, r = %02f' % r_snareSilenceAmp)
plt.plot(mean_wdBlkSilenceBestAmp, mean_wdBlkCue_DevToClock, 'bo',
        label = 'Wood Block, r = %02f' % r_wdBlkSilenceAmp)
plt.legend()
plt.xlabel('Oscillation Amplitude')
plt.ylabel('Absolute Deviation')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(result_folder, 'SilenceAmp_Performance.png'))
fig.savefig(os.path.join(result_folder, 'SilenceAmp_Performance.pdf'))

# 1b. Silence, Phase
r_snareSilencePhase = np.corrcoef(mean_snareSilenceBestPhase,
        mean_snareCue_DevToClock)[0][1]
r_wdBlkSilencePhase = np.corrcoef(mean_wdBlkSilenceBestPhase,
        mean_wdBlkCue_DevToClock)[0][1]
fig = plt.figure(figsize=(10,10))
plt.title('oscillation phase in silence period vs snare performance')
plt.plot(mean_snareSilenceBestPhase, mean_snareCue_DevToClock, 'ro',
        label = 'Snare, r = %02f' % r_snareSilencePhase)
plt.plot(mean_wdBlkSilenceBestPhase, mean_wdBlkCue_DevToClock, 'bo',
        label = 'Wood Block, r = %02f' % r_wdBlkSilencePhase)
plt.legend()
plt.xlabel('Oscillation Phase')
plt.ylabel('Absolute Deviation')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(result_folder, 'SilencePhase_Performance.png'))
fig.savefig(os.path.join(result_folder, 'SilencePhase_Performance.pdf'))
