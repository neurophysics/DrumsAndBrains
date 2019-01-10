import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path

data_folder = sys.argv[1]
subjectnr = int(sys.argv[2]) #here: total number of subjects
result_folder = sys.argv[3]

save_folder = os.path.join(result_folder, 'all%02dsubjects' % subjectnr)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

self_perf = np.zeros((subjectnr,6))
self_vigil = np.zeros((subjectnr,6))
meandevs_snare = np.zeros((subjectnr,6))
meandevs_wdBlk = np.zeros((subjectnr,6))
snareCue_DevToClock = np.zeros((subjectnr, 75))
wdBlkCue_DevToClock = np.zeros((subjectnr, 75))

for subject in range(1,subjectnr+1):
    current_data_folder = os.path.join(data_folder, 'S%02d' % subject)

    # Collect self assessment of performance and vigilance data
    p = 0
    v = 0
    with open(os.path.join(current_data_folder,
            'S%02d_self-assessment.txt' % subject)) as f:
        for line in f:
            l = line.split(' ')
            if line[0] == 'P':
                self_perf[subject-1][p] = int(l[1])
                p+=1
            if line[0] == 'V':
                self_vigil[subject-1][v] = int(l[1])
                v+=1
    # normalize
    self_perf[subject-1] = [(i-np.mean(self_perf[subject-1])) /
            np.std(self_perf[subject-1])
            for i in self_perf[subject-1]]
    self_vigil[subject-1] = [(i-np.mean(self_vigil[subject-1])) /
            np.std(self_vigil[subject-1])
            for i in self_vigil[subject-1]]

    # collect real performance data
    with np.load(os.path.join(result_folder, 'S%02d' % subject,
            'behavioural_results.npz'), 'r') as f:
        snareDev = np.abs(f['snareCue_DevToClock'])
        wdBlkDev = np.abs(f['wdBlkCue_DevToClock'])
        snareCue_DevToClock[subject-1] = np.concatenate(snareDev)
        wdBlkCue_DevToClock[subject-1] = np.concatenate(wdBlkDev)
        for i in range(0,6):
            meandevs_snare[subject-1][i] = [np.mean(s)
                    for s in np.abs(snareDev)][i]
            meandevs_wdBlk[subject-1][i] = [np.mean(s)
                    for s in np.abs(wdBlkDev)][i]


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
ax1.set_title('self assessment performance/vigilance vs real snare performance (r = %02f / %02f)'% (r_snarePerf, r_snareVigil))
ax2 = fig.add_subplot(212, sharex=ax1)
ax2.set_title('self assessment performance/vigilance vs real wdBlk performance (r = %02f / %02f)'% (r_wdBlkPerf, r_wdBlkVigil))
for subject in range(0,subjectnr):
    #ax1.set_ylim(0.,0.8)
    if subject== 0:
        ax1.plot(np.sort(self_perf[subject]),meandevs_snare[subject][np.argsort(self_perf[subject])], 'ro-',
                label = 'Performance Assessment')
        ax1.plot(np.sort(self_vigil[subject]),meandevs_snare[subject][np.argsort(self_vigil[subject])], 'bo-',
                label = 'Vigilance Assessment')
        ax2.plot(np.sort(self_perf[subject]),meandevs_wdBlk[subject][np.argsort(self_perf[subject])], 'ro-',
                label = 'Performance Assessment')
        ax2.plot(np.sort(self_vigil[subject]),meandevs_wdBlk[subject][np.argsort(self_vigil[subject])], 'bo-',
                label = 'Vigilance Assessment')
    else:
        ax1.plot(np.sort(self_perf[subject]),meandevs_snare[subject][np.argsort(self_perf[subject])], 'ro-')
        ax1.plot(np.sort(self_vigil[subject]),meandevs_snare[subject][np.argsort(self_vigil[subject])], 'bo-')
        ax2.plot(np.sort(self_perf[subject]),meandevs_wdBlk[subject][np.argsort(self_perf[subject])], 'ro-')
        ax2.plot(np.sort(self_vigil[subject]),meandevs_wdBlk[subject][np.argsort(self_vigil[subject])], 'bo-')
ax1.legend()
ax2.legend()
plt.xlabel('self assessment')
plt.ylabel('absolute deviation')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(save_folder, 'SelfAssessmentAndResponse.png'))
fig.savefig(os.path.join(save_folder, 'SelfAssessmentAndResponse.pdf'))


#plot oscillation amplitude vs performance
# read eeg data:
snareInlier = np.zeros((subjectnr,75))
wdBlkInlier = np.zeros((subjectnr,75))
snareListenBestAmp = np.zeros((subjectnr,75))
wdBlkListenBestAmp = np.zeros((subjectnr,75))
snareListenBestPhase = np.zeros((subjectnr,75))
wdBlkListenBestPhase = np.zeros((subjectnr,75))
snareSilenceBestAmp = np.zeros((subjectnr,75))
wdBlkSilenceBestAmp = np.zeros((subjectnr,75))
snareSilenceBestPhase = np.zeros((subjectnr,75))
wdBlkSilenceBestPhase = np.zeros((subjectnr,75))
for subject in range(1,subjectnr+1):
    with np.load(os.path.join(result_folder, 'S%02d' % subject,
            'eeg_results.npz'), 'r') as f:
        snareInlier[subject-1] = f['snareInlier']
        snarePos = np.where([not s for s in snareInlier[subject-1]])
        wdBlkInlier[subject-1] = f['wdBlkInlier']
        wdBlkPos = np.where([not s for s in wdBlkInlier[subject-1]])
        snareListenBestAmp[subject-1] = np.insert(
                f['snareListenBestAmp'][0], snarePos[0], np.nan, axis=0)
        wdBlkListenBestAmp[subject-1] = np.insert(
                f['wdBlkListenBestAmp'][0], wdBlkPos[0], np.nan, axis=0)
        snareListenBestPhase[subject-1] = np.insert(
                f['snareListenBestPhase'][0], snarePos[0], np.nan, axis=0)
        wdBlkListenBestPhase[subject-1] = np.insert(
                f['wdBlkListenBestPhase'][0], wdBlkPos[0], np.nan, axis=0)
        snareSilenceBestAmp[subject-1] = np.insert(
                f['snareSilenceBestAmp'][0], snarePos[0], np.nan, axis=0)
        wdBlkSilenceBestAmp[subject-1] = np.insert(
                f['wdBlkSilenceBestAmp'][0], wdBlkPos[0], np.nan, axis=0)
        snareSilenceBestPhase[subject-1] = np.insert(
                f['snareSilenceBestPhase'][0], snarePos[0], np.nan, axis=0)
        wdBlkSilenceBestPhase[subject-1] = np.insert(
                f['wdBlkSilenceBestPhase'][0], wdBlkPos[0], np.nan, axis=0)

# average over all subjects
mean_snareCue_DevToClock = []
mean_wdBlkCue_DevToClock = []
mean_snareListenBestAmp = []
mean_wdBlkListenBestAmp = []
mean_snareListenBestPhase = []
mean_wdBlkListenBestPhase = []
for trial in range(0, 75):
    snareDevs = []
    wdBlkDevs = []
    snareListenAmps = []
    wdBlkListenAmps = []
    snareListenPhases = []
    wdBlkListenPhases = []
    for subject in range(0, subjectnr):
        if snareInlier[subject][trial]: #only append trials of subjects that are Inlier
            snareDevs.append(snareCue_DevToClock[subject][trial])
            snareListenAmps.append(snareListenBestAmp[subject][trial])
            snareListenPhases.append(snareListenBestPhase[subject][trial])
        if wdBlkInlier[subject][trial]: #only append trials of subjects that are Inlier
            wdBlkDevs.append(wdBlkCue_DevToClock[subject][trial])
            wdBlkListenAmps.append(wdBlkListenBestAmp[subject][trial])
            wdBlkListenPhases.append(wdBlkListenBestPhase[subject][trial])
    mean_snareCue_DevToClock.append(np.mean(snareDevs))
    mean_wdBlkCue_DevToClock.append(np.mean(wdBlkDevs))
    mean_snareListenBestAmp.append(np.mean(snareListenAmps))
    mean_wdBlkListenBestAmp.append(np.mean(wdBlkListenAmps))
    mean_snareListenBestPhase.append(np.mean(snareDevs))
    mean_wdBlkListenBestPhase.append(np.mean(snareDevs))
# 1. for listening period (3 bars)
# plot - one for snare, one for wdBlk
fig = plt.figure(figsize=(10,10))
plt.title('oscillation amplitude in listening period vs snare performance')
plt.plot(mean_snareListenBestAmp, mean_snareCue_DevToClock, 'ro',
        label = 'Snare')
plt.plot(mean_wdBlkListenBestAmp, mean_wdBlkCue_DevToClock, 'bo',
        label = 'Wood Block')
plt.legend()
plt.xlabel('Oscillation Amplitude')
plt.ylabel('Absolute Deviation')
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(save_folder, 'ListeningAmp_Performance.png'))
fig.savefig(os.path.join(save_folder, 'ListeningAmp_Performance.pdf'))
