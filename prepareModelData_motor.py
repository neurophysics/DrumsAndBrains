"""
stores csv containing all relevant data for motor analysis
"""
import numpy as np
import sys
import os.path
import scipy
from scipy.stats import zscore, iqr
import csv
import matplotlib.pyplot as plt
import random
import meet
import pdb
from tqdm import tqdm

data_folder = sys.argv[1]
result_folder = sys.argv[2]

N_subjects = 21

# reject behavioral outlier
iqr_rejection = True

# target frequencies
# snareFreq = 7./6
# wdBlkFreq = 7./4

###############################################################################
# load/prepare EEG data (ERD/SCSP and BPLDA)
###############################################################################
# load bp and erd data: we only take strongest component for each
#ERD_CSP = [] # stores trial averaged ERD/S_CSP per subject, each with shape (Nband, CSPcomp,time)
ERDCSP_trial = [] #stores ERD_CSP of best CSPcomp per subject,each shape (Nband, Ntrial)
ERSCSP_trial = [] # same for ERS
all_BP = [] # len 20, each (32,2500,1xx)=(channels, time in ms, trials)
# load the frequency array and inlier
snareHit_inlier = [] #(75,) True if we have a valid hit in the trial
wdBlkHit_inlier = []
snareInlier = [] #(72,) corresponds to snareHit_times that are already filtered by snareHit_inlier and artifact mask
wdBlkInlier = []
try:
    i=0
    while True:
        try:
            with np.load(os.path.join(result_folder,'motor/ERDCSP.npz')) as f:
                #ERD_CSP.append(f['ERDCSP_{:02d}'.format(i)]) #might not need this here
                ERDCSP_trial.append(f['ERDCSP_trial_{:02d}'.format(i)][:])
                ERSCSP_trial.append(f['ERSCSP_trial_{:02d}'.format(i)])
            with np.load(os.path.join(result_folder, 'motor/BP.npz'),
                'r') as fb:
                all_BP.append(fb['BP_trials_{:02d}'.format(i)])
            with np.load(os.path.join(result_folder, 'motor/inlier.npz'),
                'r') as fi:
                snareHit_inlier.append(fi['snareHit_inlier_{:02d}'.format(i)])
                wdBlkHit_inlier.append(fi['wdBlkHit_inlier_{:02d}'.format(i)])
                snareInlier.append(fi['snareInlier_response_{:02d}'.format(i)])
                wdBlkInlier.append(fi['wdBlkInlier_response_{:02d}'.format(i)])
                win = fi['win']
            i+=1
        except KeyError:
            break
    print('ERDCSP, BP and inlier succesfully read.')
except FileNotFoundError: # read ERD data and calculate CSP
    print('erdcsp.npz or BP.npz not found. Please run csp.py first.')

try:
    cfilt = np.load(os.path.join(result_folder,'motor/lda.npy'))
    # base_idx_lda is range(500) ie -2 to -1.5s
    # act_idx_lda is range(1400:1900) ie -600 to -100ms
    [base_idx_lda,act_idx_lda] = np.load(
        os.path.join(result_folder,'motor/lda_idx.npy'))
except KeyError:
    print('lda.npy or lda_idx.npy not found. Please run lda.py first.')

# for each label we will have on value per subject and trial (20,1xx)
N_bands = ERDCSP_trial[0].shape[0]
EEG_labels = (['BP'] +
    ['ERD{}'.format(i+1) for i in range(N_bands)] +
    ['ERS{}'.format(i+1) for i in range(N_bands)])

#apply lda to bp
BPlda = [np.tensordot(cfilt, b, axes=(0,0)) for b in all_BP] #each shape (2500, 143) now
#convert BP to decrease/increase value i.e. activation avg - baseline avg
BPLDA = [a[act_idx_lda].mean(0) - a[base_idx_lda].mean(0) for a in BPlda] #(143,) each

###############################################################################
# load/prepare behavioral data
# get performance session and trial indices and separate trials into snare
# and woodblock trials
###############################################################################

# read the musicality scores of all subjects
background = {}
with open(os.path.join(data_folder,'additionalSubjectInfo.csv'),'r') as infile:
    reader = csv.DictReader(infile, fieldnames=None, delimiter=';')
    for row in reader:
        key = row['Subjectnr']
        value = [int(row['LQ']),int(row['MusicQualification']),
            int(row['MusicianshipLevel']),int(row['TrainingYears'])]
        background[key] = value

raw_musicscores = np.array([background['%s' % i]
                            for i in list(range(1, 11, 1)) +
                            list(range(12, 22, 1))])  # excluded subject 11

z_musicscores = (raw_musicscores - np.mean(raw_musicscores, 0)
                 )/raw_musicscores.std(0)
musicscore = z_musicscores[:, 1:].mean(1)  # do not include the LQ


snare_deviation = []
snare_trial_idx = []
snare_session_idx = []
wdBlk_deviation = []
wdBlk_trial_idx = []
wdBlk_session_idx = []
# need the eeg to be full size so we can mask all invalid trials at once afterwards
ERDCSP_trial_150 = []
ERSCSP_trial_150 = []
BPLDA_150 = []

subj = 0
idx = 0
while True:
    subj += 1
    if not os.path.exists(os.path.join(
        result_folder, 'S{:02d}'.format(subj), 'behavioural_results.npz')):
        break
    # one subject does not have EEG data - Check and skip that subject
    elif not os.path.exists(os.path.join(
        result_folder, 'S{:02d}'.format(subj), 'prepared_FFTSSD.npz')):
        continue
    else:
        #########################
        # read behavioural data #
        #########################
        snareHit_inlier_now = snareHit_inlier[idx] # invalid hits
        wdBlkHit_inlier_now = wdBlkHit_inlier[idx]
        snareInlier_now = snareInlier[idx] # eeg artifacts
        wdBlkInlier_now = wdBlkInlier[idx]
        # todo: in the end also take inlier of eeg data
        with np.load(os.path.join(result_folder,'S{:02d}'.format(
            subj), 'behavioural_results.npz'), allow_pickle=True,
            encoding='bytes') as fi:
            # only take trials with valid hit times and no eeg artifacts
            snare_deviation_now = fi['snare_deviation'][
                    snareHit_inlier_now][snareInlier_now]
            wdBlk_deviation_now = fi['wdBlk_deviation'][
                    wdBlkHit_inlier_now][wdBlkInlier_now]
            # take only the trials where performance is not nan
            snare_finite = np.isfinite(snare_deviation_now)
            wdBlk_finite = np.isfinite(wdBlk_deviation_now)
            # take only the trials in range median ± 1.5*IQR
            if iqr_rejection:
                lb_snare = np.median(snare_deviation_now[snare_finite]
                    ) - 1.5*iqr(snare_deviation_now[snare_finite])
                ub_snare = np.median(snare_deviation_now[snare_finite]
                    ) + 1.5*iqr(snare_deviation_now[snare_finite])
                snare_inlier_now = ((snare_deviation_now > lb_snare) &
                                    (snare_deviation_now < ub_snare) &
                                     snare_finite)
                lb_wdBlk = np.median(wdBlk_deviation_now[wdBlk_finite]
                    ) - 1.5*iqr(wdBlk_deviation_now[wdBlk_finite])
                ub_wdBlk = np.median(wdBlk_deviation_now[wdBlk_finite]
                    ) + 1.5*iqr(wdBlk_deviation_now[wdBlk_finite])
                wdBlk_inlier_now = ((wdBlk_deviation_now > lb_wdBlk) &
                                    (wdBlk_deviation_now < ub_wdBlk) &
                                     wdBlk_finite)
            snare_deviation.append(
                snare_deviation_now[snare_inlier_now])
            wdBlk_deviation.append(
                wdBlk_deviation_now[wdBlk_inlier_now])

            # get the trial indices
            snare_times = fi['snareCue_times']
            wdBlk_times = fi['wdBlkCue_times']
            all_trial_idx = np.argsort(np.argsort(
                np.r_[snare_times, wdBlk_times]))
            snare_trial_idx.append(
                    all_trial_idx[:len(snare_times)][
                        #valid hits, no artifacts, iqr:
                        snareHit_inlier_now][snareInlier_now][snare_inlier_now])
            wdBlk_trial_idx.append(
                    all_trial_idx[len(snare_times):][
                        wdBlkHit_inlier_now][wdBlkInlier_now][wdBlk_inlier_now])

            # snare_trial_idx relates to 150 trials => need 150 trial version
            # First, combine Hit inlier and eeg inlier
            a = list(snareInlier_now)
            snareInlier_combined = [a.pop(0)
                if snareHit_inlier_now[i]==True else False
                for i in range(len(snareHit_inlier_now))]
            b = list(wdBlkInlier_now)
            wdBlkInlier_combined = [b.pop(0)
                if wdBlkHit_inlier_now[i]==True else False
                for i in range(len(wdBlkHit_inlier_now))]
            # Then, combine snare and wdblk inlier in teh right order
            snarewdBlkHit_inlier = np.hstack(
                [snareInlier_combined,wdBlkInlier_combined])
            allHit_inlier = snarewdBlkHit_inlier[np.argsort(all_trial_idx)] #argsort gives order of trials
            # add nan at invalid trials to get to shape (5,150)
            res = [] #erd
            res2 = [] #ers
            for b in range(N_bands):
                val = list(ERDCSP_trial[idx][b,:])
                res.append([val.pop(0) if allHit_inlier[i]==True else np.nan
                    for i in range(len(allHit_inlier))])
                val2 = list(ERSCSP_trial[idx][b,:])
                res2.append([val2.pop(0) if allHit_inlier[i]==True else np.nan
                    for i in range(len(allHit_inlier))])
            ERDCSP_trial_150.append(np.vstack(res))
            ERSCSP_trial_150.append(np.vstack(res2))
            val3 = list(BPLDA[idx])
            res3 = [val3.pop(0) if allHit_inlier[i]==True else np.nan
                for i in range(len(allHit_inlier))]
            BPLDA_150.append(np.array(res3))

            # get the session indices
            snare_session_idx.append(
                np.hstack(
                    [i*np.ones_like(session)
                        for i, session in enumerate(fi['snareCue_nearestClock'])
                        ]
                    )[snareHit_inlier_now][snareInlier_now][snare_inlier_now]
                )
            wdBlk_session_idx.append(
                np.hstack(
                    [i*np.ones_like(session)
                        for i, session in enumerate(fi['wdBlkCue_nearestClock'])
                        ]
                    )[wdBlkHit_inlier_now][wdBlkInlier_now][wdBlk_inlier_now]
                    )
            idx += 1



# wdBlk_subject = np.hstack([np.ones(F_SSD_now.shape[-1], int)*i
#     for i, F_SSD_now in enumerate(wdBlk_F_SSD)])

#########################################################################
# create data dictionary for storing as csv and subsequent R processing #
#########################################################################

def getEntryFromMultipleArrays(arrays):
    """Get a list of single entry of multiple 1d-arrays

    Args: arrays: list of N 1d arrays. These mustall have the same
                length

    Returns: item_generator: a generator that returns a list of single entries
             from all arrays
    """
    N = [len(a) for a in arrays]
    if not all([N_now == N[0] for N_now in N]):
        raise ValueError('length of arrays must be the same')
    N = N[0]
    for i in range(N):
        yield [a[i] for a in arrays]


def writeDictionaryToCSV(datadict, trial_type):
    with open(os.path.join(result_folder, '{}_data_motor.csv'.format(
        trial_type)), 'w') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC,
                quotechar='\"')
        writer.writerow(datadict.keys())
        # loop through the items
        for row in getEntryFromMultipleArrays(datadict.values()):
            writer.writerow(row)

# divide into snare and wdblk trials (also, only take valid behavioral trials)
ERDCSP_snare = [ERDCSP_trial_150[i][:,snare_trial_idx[i]]
    for i in range(N_subjects-1)]
ERDCSP_wdBlk = [ERDCSP_trial_150[i][:,wdBlk_trial_idx[i]]
    for i in range(N_subjects-1)]
ERSCSP_snare = [ERSCSP_trial_150[i][:,snare_trial_idx[i]]
    for i in range(N_subjects-1)]
ERSCSP_wdBlk = [ERSCSP_trial_150[i][:,wdBlk_trial_idx[i]]
    for i in range(N_subjects-1)]
BPLDA_snare = [BPLDA_150[i][snare_trial_idx[i]]
    for i in range(N_subjects-1)]
BPLDA_wdBlk = [BPLDA_150[i][wdBlk_trial_idx[i]]
    for i in range(N_subjects-1)]
subject_idx_snare = np.hstack([np.ones(erd.shape[-1], int)*i
    for i, erd in enumerate(ERDCSP_snare)])
subject_idx_wdBlk = np.hstack([np.ones(erd.shape[-1], int)*i
    for i, erd in enumerate(ERDCSP_wdBlk)])



# for snare #
#############
data = {} #2745 trials
# have [20*(5,143)] need 5 lists each 20*xx=1275 trials
erd_data = np.vstack( #(5,20)
    [np.hstack([erd[i] for erd in ERDCSP_snare]) for i in range(N_bands)])
ers_data = np.vstack(
    [np.hstack([ers[i] for ers in ERSCSP_snare]) for i in range(N_bands)])

# add EEG to dictionary
all_EEG = np.vstack([np.hstack(BPLDA_snare), erd_data, ers_data]) #(11,20)
for i, l, in enumerate(EEG_labels):
    data[l] = all_EEG[i,:]
# add subject index
data['subject'] = np.array([s+1 if s<10 else s+2 for s in subject_idx_snare])
# add musicality
data['musicality'] = musicscore[subject_idx_snare]
# add trial index (no log)
data['trial'] = np.hstack(snare_trial_idx)
# add session index (no log) and precision
data['session'] = np.hstack(snare_session_idx)
data['deviation'] = np.hstack(snare_deviation)

writeDictionaryToCSV(data, 'snare')

# for woodblock #
#################
wdBlk_data = {}
# have [20*(5,143)] need 5 lists each 20*xx=1275 trials
erd_data = np.vstack( #(5,20)
    [np.hstack([erd[i] for erd in ERDCSP_wdBlk]) for i in range(N_bands)])
ers_data = np.vstack(
    [np.hstack([ers[i] for ers in ERSCSP_wdBlk]) for i in range(N_bands)])

# add EEG to dictionary
all_EEG = np.vstack([np.hstack(BPLDA_wdBlk), erd_data, ers_data]) #(11,20)
for i, l, in enumerate(EEG_labels):
    data[l] = all_EEG[i,:]
# add subject index
data['subject'] = np.array([s+1 if s<10 else s+2 for s in subject_idx_wdBlk])
# add musicality
data['musicality'] = musicscore[subject_idx_wdBlk]
# add trial index (no log)
data['trial'] = np.hstack(wdBlk_trial_idx)
# add session index (no log) and precision
data['session'] = np.hstack(wdBlk_session_idx)
data['deviation'] = np.hstack(wdBlk_deviation)

writeDictionaryToCSV(data, 'wdBlk')
