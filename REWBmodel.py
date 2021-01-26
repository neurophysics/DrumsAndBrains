"""
Create the design matrix of the Within-Between Random Effects model
"""
import numpy as np
import sys
import os.path
from scipy.stats import zscore
import csv
import sklearn

data_folder = sys.argv[1]
result_folder = sys.argv[2]

# target frequencies
snareFreq = 7./6
wdBlkFreq = 7./4

# number of SSD_components to use
N_SSD = 2

# load the SSD results from all subjects into a list
snare_F_SSD = []
wdBlk_F_SSD = []
snareInlier = []
wdBlkInlier = []

with np.load(os.path.join(result_folder, 'F_SSD.npz'), 'r') as fi:
    # load the frequency array
    f = fi['f']
    # find the index of the frequency array refering to snare and woodblock
    # frequency
    snare_idx = np.argmin((f - snareFreq)**2)
    wdBlk_idx = np.argmin((f - wdBlkFreq)**2)
    # loop through all arrays
    i = 0
    while True:
        try:
            F_SSD = fi['F_SSD_{:02d}'.format(i)][:N_SSD,
                    (snare_idx, wdBlk_idx)]
            snareInlier.append(fi['snareInlier_{:02d}'.format(i)])
            wdBlkInlier.append(fi['wdBlkInlier_{:02d}'.format(i)])
            snare_temp = F_SSD[...,:snareInlier[-1].sum()]
            wdBlk_temp = F_SSD[...,snareInlier[-1].sum():]
            snare_F_SSD.append(snare_temp.reshape((-1, snare_temp.shape[-1]),
                order='F'))
            wdBlk_F_SSD.append(wdBlk_temp.reshape((-1, wdBlk_temp.shape[-1]),
                order='F'))
            i += 1
        except KeyError:
            break

N_subjects = len(snare_F_SSD)

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
    for i in list(range(1,11,1)) + list(range(12, 22, 1))])

z_musicscores = (raw_musicscores - np.mean(raw_musicscores,0)
        )/raw_musicscores.std(0)
musicscore = z_musicscores[:,1:].mean(1) # do not include the LQ

# get the performance numbers
snare_deviation = []
snare_trial_idx = []
snare_session_idx = []
wdBlk_deviation = []
wdBlk_trial_idx = []
wdBlk_session_idx = []

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
        with np.load(os.path.join(result_folder,'S{:02d}'.format(subj),
            'behavioural_results.npz'), allow_pickle=True,
            encoding='bytes') as fi:
            # take only the trials where performance is not nan
            snare_finite = np.isfinite(fi['snare_deviation'])
            wdBlk_finite = np.isfinite(fi['wdBlk_deviation'])
            snare_F_SSD[idx] = snare_F_SSD[idx][:,
                    snare_finite[snareInlier[idx]]]
            wdBlk_F_SSD[idx] = wdBlk_F_SSD[idx][:,
                    wdBlk_finite[wdBlkInlier[idx]]]
            snare_inlier_now = np.all([snare_finite, snareInlier[idx]], 0)
            wdBlk_inlier_now = np.all([wdBlk_finite, wdBlkInlier[idx]], 0)
            # load data
            snare_deviation.append(
                    fi['snare_deviation'][snare_inlier_now])
            wdBlk_deviation.append(
                    fi['wdBlk_deviation'][wdBlk_inlier_now])
            # get the trial indices
            snare_times = fi['snareCue_times']
            wdBlk_times = fi['wdBlkCue_times']
            all_trial_idx = zscore(np.argsort(np.argsort(
                np.r_[snare_times, wdBlk_times])))
            snare_trial_idx.append(
                    all_trial_idx[:len(snare_times)][snare_inlier_now])
            wdBlk_trial_idx.append(
                    all_trial_idx[len(snare_times):][wdBlk_inlier_now])
            # get the session indices
            snare_session_idx.append(
                    zscore(np.hstack([i*np.ones_like(session)
                        for i, session in enumerate(
                            fi['snareCue_nearestClock'])])
                        )[snare_inlier_now])
            wdBlk_session_idx.append(
                    zscore(np.hstack([i*np.ones_like(session)
                        for i, session in enumerate(
                            fi['wdBlkCue_nearestClock'])])
                        )[wdBlk_inlier_now])
            idx += 1

def design_matrix(F_SSD, musicscore, trial_idx, session_idx):
    N_trials = sum([SSD_now.shape[-1] for SSD_now in F_SSD])
    N_subjects = len(F_SSD)
    X = []
    labels = []
    # add intercept
    labels.append('intercept')
    X.append(np.ones((1, N_trials)))
    # add within-subjects coefficient
    labels.extend(['WSnare1', 'WSnare2', 'WWdBlk1', 'WWdBlk2'])
    X.append(np.hstack([zscore(np.abs(SSD_now), -1) for SSD_now in F_SSD]))
    # get mean and std of between subjects effect
    B_mean = np.mean([np.abs(SSD_now).mean(-1) for SSD_now in F_SSD], 0)
    B_std = np.std([np.abs(SSD_now).mean(-1) for SSD_now in F_SSD], 0)
    # add between subjects coefficient
    labels.extend(['BSnare1', 'BSnare2', 'BWdBlk1', 'BWdBlk2'])
    X.append(
            np.hstack([((np.abs(SSD_now).mean(-1) - B_mean)/
                B_std)[:, np.newaxis] * np.ones(SSD_now.shape)
                for SSD_now in F_SSD]))
    # add musicality score
    labels.append('musicality')
    X.append(zscore(np.hstack([m*np.ones(SSD_now.shape[-1])
        for m, SSD_now in zip(musicscore, F_SSD)])))
    # add trial_idx
    labels.append('trial_idx')
    X.append(np.hstack(trial_idx))
    # add session_idx
    labels.append('session_idx')
    X.append(np.hstack(session_idx))
    # add random effect for the intercept
    labels.extend(['RE0_{:02d}'.format(i) for i in range(N_subjects)])
    RE0 = np.zeros([N_subjects, N_trials])
    subj = 0
    tr = 0
    for SSD_now in F_SSD:
        RE0[subj, tr:tr + SSD_now.shape[-1]] = 1
        tr += SSD_now.shape[-1]
        subj += 1
    X.append(RE0)
    # add random effect for the within-subjects (REW) effect
    [labels.extend(
        ['REWSnare1_{:02d}'.format(i),
        'REWSnare2_{:02d}'.format(i),
        'REWWdBlk1_{:02d}'.format(i),
        'REWWdBlk2_{:02d}'.format(i)])
        for i in range(N_subjects)]
    REW = np.zeros([N_SSD*2*N_subjects, N_trials])
    subj = 0
    tr = 0
    for SSD_now in F_SSD:
        REW[subj*N_SSD*2:(subj + 1)*N_SSD*2, tr:tr + SSD_now.shape[-1]] =(
                zscore(np.abs(SSD_now) - np.abs(SSD_now).mean(-1)[
                    :,np.newaxis], axis=-1))
        tr += SSD_now.shape[-1]
        subj += 1
    X.append(REW)
    return np.vstack(X), labels

# finally, get the design matrices
snare_design, snare_labels = design_matrix(
        snare_F_SSD, musicscore, snare_trial_idx, snare_session_idx)
snareY = np.hstack(snare_deviation)

wdBlk_design, wdBlk_labels = design_matrix(
        wdBlk_F_SSD, musicscore, wdBlk_trial_idx, wdBlk_session_idx)
wdBlkY = np.hstack(wdBlk_deviation)

# start interface to R
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# activate automatic conversion of numpy arrays to R
from rpy2.robjects import numpy2ri
numpy2ri.activate()

# import R's "glmnet"
glmnet = importr('glmnet')
from rpy2.robjects.functions import SignatureTranslatedFunction as STM

glmnet.cv_glmnet = STM(glmnet.cv_glmnet,
        init_prm_translate = {'penalty_factor': 'penalty.factor'})

# TODO
# check for the validity of the model in the original literature

snare_cv_model = glmnet.cv_glmnet(
        snare_design[1:].T, np.abs(snareY.reshape(-1,1)),
        alpha = 0.9,
        family = 'gaussian',
        intercept=True,
        standardize=False,
        nlambda = 500,
        nfolds = 20
        )
