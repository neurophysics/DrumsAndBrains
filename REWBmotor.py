"""
calculates best erd/s and feeds it into REWB model:
correlation of ERD/S with performance?
later also correlation of BP and performance
"""
import numpy as np
import sys
import os.path
import scipy
from scipy.stats import zscore, iqr
import csv
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm
import meet

data_folder = sys.argv[1]
result_folder = sys.argv[2]
N_subjects = 21
s_rate = 1000 # sampling rate of the EEG

iqr_rejection = True

'''channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)'''

##### read data #####
# read ERD/S per subject and trial (CSP applied)
snareInlier = [] # list of 20 subjects, each shape ≤75
wdBlkInlier = [] # -""-, one trial is 2000ms preresponse to 500 postresponse
ERDCSP_trial = [] #stores ERD_CSP of best CSPcomp per subject, each shape (Ntrial,)
ERSCSP_trial = [] # same for ERS
i=0
while True:
    try:
        with np.load(os.path.join(result_folder, 'motor/inlier.npz'),
            'r') as fi:
            snareInlier.append(fi['snareInlier_response_{:02d}'.format(i)])
            wdBlkInlier.append(fi['wdBlkInlier_response_{:02d}'.format(i)])
            win = fi['win']
        with np.load(os.path.join(result_folder, 'motor/erdcsp.npz'),
            'r') as f_covmat:
            ERDCSP_trial.append(f_covmat['ERDCSP_trial_{:02d}'.format(i)])
            ERSCSP_trial.append(f_covmat['ERSCSP_trial_{:02d}'.format(i)])
        i+=1
    except KeyError:
        break

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
    for i in list(range(1,11,1)) + list(range(12, 22, 1))]) #exclude subject 11

z_musicscores = (raw_musicscores - np.mean(raw_musicscores,0)
        )/raw_musicscores.std(0)
musicscore = z_musicscores[:,1:].mean(1) # do not include the LQ

snare_ERDCSP = []
snare_ERSCSP = []
wdBlk_ERDCSP = []
wdBlk_ERSCSP = []
snare_deviation = []
snare_trial_idx = []
snare_session_idx = []
wdBlk_deviation = []
wdBlk_trial_idx = []
wdBlk_session_idx = []
snareInlier_final=[]
subj = 0
idx = 0
while True:
    subj += 1
    print('subject ',subj)
    if not os.path.exists(os.path.join(
        result_folder, 'S{:02d}'.format(subj), 'behavioural_results.npz')):
        break
    # one subject does not have EEG data - Check and skip that subject
    elif not os.path.exists(os.path.join(
        result_folder, 'S{:02d}'.format(subj), 'prepared_FFTSSD.npz')):
        continue
    else:
        erdcsp = ERDCSP_trial[idx]
        erscsp = ERSCSP_trial[idx]
        snareInlier_now = snareInlier[idx]
        wdBlkInlier_now = wdBlkInlier[idx]
        # trials sorted???
        snare_ERDCSP.append(erdcsp[:snareInlier_now.sum()])
        snare_ERSCSP.append(erscsp[:snareInlier_now.sum()])
        wdBlk_ERDCSP.append(erdcsp[snareInlier_now.sum():])
        wdBlk_ERSCSP.append(erscsp[snareInlier_now.sum():])

        with np.load(os.path.join(result_folder,'S{:02d}'.format(subj),
            'behavioural_results.npz'), allow_pickle=True,
            encoding='bytes') as fi: #delete nans
            snare_deviation_now = fi['snare_deviation']
            snare_finite = [np.isfinite(snare_deviation_now)] #delete nan
            snare_deviation_now = snare_deviation_now[snare_finite]
            wdBlk_deviation_now = fi['wdBlk_deviation']
            wdBlk_finite = [np.isfinite(wdBlk_deviation_now)]
            wdBlk_deviation_now = wdBlk_deviation_now[wdBlk_finite]

            # take only the trials in range median ± 1.5*IQR
            if iqr_rejection:
                lb_snare = np.median(snare_deviation_now
                    ) - 1.5*iqr(snare_deviation_now)
                ub_snare = np.median(snare_deviation_now
                    ) + 1.5*iqr(snare_deviation_now)
                idx_iqr_snare = np.logical_and(
                    snare_deviation_now>lb_snare, snare_deviation_now<ub_snare)
                snareInlier_now = np.logical_and(
                    snareInlier_now, idx_iqr_snare)
                lb_wdBlk = np.median(wdBlk_deviation_now
                    ) - 1.5*iqr(wdBlk_deviation_now)
                ub_wdBlk = np.median(wdBlk_deviation_now
                    ) + 1.5*iqr(wdBlk_deviation_now)
                idx_iqr_wdBlk = np.logical_and(
                    wdBlk_deviation_now>lb_wdBlk, wdBlk_deviation_now<ub_wdBlk)
                wdBlkInlier_now = np.logical_and(
                    wdBlkInlier_now, idx_iqr_wdBlk)

            snare_deviation.append(
                snare_deviation_now[snareInlier_now])
            wdBlk_deviation.append(
                wdBlk_deviation_now[wdBlkInlier_now])

            snareInlier_final.append(snareInlier_now)

            # get the trial indices
            snare_times = fi['snareCue_times']
            wdBlk_times = fi['wdBlkCue_times']
            all_trial_idx = np.argsort(np.argsort(
                np.r_[snare_times, wdBlk_times]))
            snare_trial_idx_now = all_trial_idx[:len(
                snare_times)][snare_finite][snareInlier_now]
            snare_trial_idx.append(snare_trial_idx_now)
            wdBlk_trial_idx_now = all_trial_idx[len(
                snare_times):][wdBlk_finite][wdBlkInlier_now]
            wdBlk_trial_idx.append(wdBlk_trial_idx_now)

            # get the session indices
            snare_session_idx.append(
                np.hstack([i*np.ones_like(session)
                    for i, session in enumerate(
                        fi['snareCue_nearestClock'])])
                    [snare_finite][snareInlier_now])
            wdBlk_session_idx.append(
                np.hstack([i*np.ones_like(session)
                    for i, session in enumerate(
                        fi['wdBlkCue_nearestClock'])])
                    [wdBlk_finite][wdBlkInlier_now])
            idx += 1

##### calculate models #####
# start interface to R
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

base = importr('base')
stats = importr('stats')
parameters = importr('parameters')
lme4 = importr('lme4')
sjPlot = importr('sjPlot')
effectsize = importr('effectsize')

snare_subject = np.hstack([np.ones(x.shape[0], int)*(i + 1)
    for i, x in enumerate(snare_ERDCSP)])

1/0 
snare_SubjToTrials = np.unique(snare_subject, return_inverse=True)[1]
EEG_labels = (['Snare{}'.format(i+1) for i in range(N_SSD)] +
              ['WdBlk{}'.format(i+1) for i in range(N_SSD)])

wdBlk_subject = np.hstack([np.ones(F_SSD_now.shape[-1], int)*(i + 1)
    for i, F_SSD_now in enumerate(wdBlk_ERDCSP)])

wdBlk_SubjToTrials = np.unique(wdBlk_subject, return_inverse=True)[1]

###########################################
# load all the data into rpy2 R interface #
###########################################
snare_data = {}
wdBlk_data = {}

# add EEG
for i,l in enumerate(EEG_labels):
    snare_data[l] = robjects.vectors.FloatVector(
            np.hstack([F_SSD_now[i] for F_SSD_now in snare_F_SSD]))
    wdBlk_data[l] = robjects.vectors.FloatVector(
            np.hstack([F_SSD_now[i] for F_SSD_now in wdBlk_F_SSD]))
# add subject index
snare_data['subject'] = robjects.vectors.IntVector(
        np.arange(len(snare_F_SSD))[snare_SubjToTrials])
wdBlk_data['subject'] = robjects.vectors.IntVector(
        np.arange(len(wdBlk_F_SSD))[wdBlk_SubjToTrials])
# add musicality
snare_data['musicality'] = robjects.vectors.FloatVector(
        musicscore[snare_SubjToTrials])
wdBlk_data['musicality'] = robjects.vectors.FloatVector(
        musicscore[wdBlk_SubjToTrials])
# add trial index (no log)
snare_data['trial'] = robjects.vectors.FloatVector(
    np.hstack(snare_trial_idx) + 1)
wdBlk_data['trial'] = robjects.vectors.FloatVector(
    np.hstack(wdBlk_trial_idx) + 1)
# add session index (no log) and precision
snare_data['session'] = robjects.vectors.FloatVector(
    np.hstack(snare_session_idx) + 1)
snare_data['precision'] = robjects.vectors.FloatVector(
    np.log(np.abs(np.hstack(snare_deviation))))
wdBlk_data['session'] = robjects.vectors.FloatVector(
    np.hstack(wdBlk_session_idx) + 1)
wdBlk_data['precision'] = robjects.vectors.FloatVector(
    np.log(np.abs(np.hstack(wdBlk_deviation))))

Rsnare_data = base.data_frame(**snare_data)
RwdBlk_data = base.data_frame(**wdBlk_data)

# add within and between variables to the data frame
Rsnare_data = base.cbind(
        Rsnare_data,
        parameters.demean(
                Rsnare_data,
                select = base.c(*(EEG_labels + ['precision'])),
                group = 'subject'))
RwdBlk_data = base.cbind(
        RwdBlk_data,
        parameters.demean(
                RwdBlk_data,
                select = base.c(*(EEG_labels + ['precision'])),
                group = 'subject'))
# standardize, this took some googling, since R's scale function from rpy2
# returnd just a Matrix and not a data frame
Rsnare_data = effectsize.standardize(Rsnare_data)
RwdBlk_data = effectsize.standardize(RwdBlk_data)
