"""
Create the design matrix of the Within-Between Random Effects model
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

# reject behavioral outlier
iqr_rejection = True

# target frequencies
snareFreq = 7./6
wdBlkFreq = 7./4

# number of SSD_components to use
N_SSD = 1

# load the SSD results from all subjects into a list
F_SSDs = []
snareInlier = []
wdBlkInlier = []

with np.load(os.path.join(result_folder, 'F_SSD.npz'), 'r') as fi:
    # load the frequency array
    f = fi['f']
    # find the index of the frequency array refering to snare and woodblock
    # frequency
    snare_idx = np.argmin((f - snareFreq)**2)
    wdBlk_idx = np.argmin((f - wdBlkFreq)**2)
    harmo_idx = np.argmin((f - 2*wdBlkFreq)**2)
    # loop through all arrays
    i = 0
    while True:
        try:
            F_SSD = fi['F_SSD_{:02d}'.format(i)]
            F_SSD = np.abs(F_SSD)
            ## subtract mean of neighbouring frequencies
            F_SSD = scipy.ndimage.convolve1d(
                    np.abs(F_SSD)**2,
                    np.r_[[-0.25]*2, 1, [-0.25]*2], axis=1)
            F_SSD = F_SSD[:N_SSD, (snare_idx,wdBlk_idx)]
            F_SSDs.append(F_SSD)
            snareInlier.append(fi['snareInlier_{:02d}'.format(i)])
            wdBlkInlier.append(fi['wdBlkInlier_{:02d}'.format(i)])
            i += 1
        except KeyError:
            break

# take power
#F_SSDs = [np.abs(F_SSD_now) for F_SSD_now in F_SSDs]

# scale F_SSD: concatenate along trial axis, calculate mean and sd, scale
F_SSD_mean = np.mean(np.concatenate(F_SSDs, -1), 2)
F_SSD_sd = np.std(np.concatenate(F_SSDs, -1), 2)
F_SSDs = [(i - F_SSD_mean[:,:,np.newaxis]) / F_SSD_sd[:,:,np.newaxis] for i in F_SSDs]

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

# get the performance numbers
snare_F_SSD = []
wdBlk_F_SSD = []
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
    # divide scaled F_SSD in wdBlk and snare
    if not os.path.exists(os.path.join(
        result_folder, 'S{:02d}'.format(subj), 'behavioural_results.npz')):
        break
    # one subject does not have EEG data - Check and skip that subject
    elif not os.path.exists(os.path.join(
        result_folder, 'S{:02d}'.format(subj), 'prepared_FFTSSD.npz')):
        continue
    else:
        F_SSD = F_SSDs[idx]
        snare_temp = F_SSD[...,:snareInlier[idx].sum()]
        wdBlk_temp = F_SSD[...,snareInlier[idx].sum():]
        snare_F_SSD.append(snare_temp.reshape((-1, snare_temp.shape[-1]),
            order='F'))
        wdBlk_F_SSD.append(wdBlk_temp.reshape((-1, wdBlk_temp.shape[-1]),
            order='F'))
        with np.load(os.path.join(result_folder,'S{:02d}'.format(subj),
            'behavioural_results.npz'), allow_pickle=True,
            encoding='bytes') as fi:
            snare_deviation_now = fi['snare_deviation'][snareInlier[idx]]
            wdBlk_deviation_now = fi['wdBlk_deviation'][wdBlkInlier[idx]]

            # take only the trials where performance is not nan
            snare_finite = np.isfinite(snare_deviation_now)
            wdBlk_finite = np.isfinite(wdBlk_deviation_now)
            snare_inlier_now = snare_finite #already filtered for snareInlier in line 41 and 96
            wdBlk_inlier_now = wdBlk_finite

            # take only the trials in range median ± 1.5*IQR
            if iqr_rejection:
                lb_snare = np.median(snare_deviation_now[snare_finite]
                    ) - 1.5*iqr(snare_deviation_now[snare_finite])
                ub_snare = np.median(snare_deviation_now[snare_finite]
                    ) + 1.5*iqr(snare_deviation_now[snare_finite])
                idx_iqr_snare = np.logical_and(
                    snare_deviation_now>lb_snare, snare_deviation_now<ub_snare)
                snare_inlier_now = np.logical_and(
                    snare_finite, idx_iqr_snare)
                lb_wdBlk = np.median(wdBlk_deviation_now[wdBlk_finite]
                    ) - 1.5*iqr(wdBlk_deviation_now[wdBlk_finite])
                ub_wdBlk = np.median(wdBlk_deviation_now[wdBlk_finite]
                    ) + 1.5*iqr(wdBlk_deviation_now[wdBlk_finite])
                idx_iqr_wdBlk = np.logical_and(
                    wdBlk_deviation_now>lb_wdBlk, wdBlk_deviation_now<ub_wdBlk)
                wdBlk_inlier_now = np.logical_and(
                    wdBlk_finite, idx_iqr_wdBlk)

            # (normalize subjects' deviation to have zero mean each) and append
            # w/o mean for now, want to induce precision not subjects consistency
            dev_mean = np.mean(np.hstack([
                snare_deviation_now[snare_inlier_now],
                wdBlk_deviation_now[wdBlk_inlier_now]]))
            snare_deviation.append(
                snare_deviation_now[snare_inlier_now])#-dev_mean)
            wdBlk_deviation.append(
                wdBlk_deviation_now[wdBlk_inlier_now])#-dev_mean)
            snare_F_SSD[idx] = snare_F_SSD[idx][:, snare_inlier_now]
            wdBlk_F_SSD[idx] = wdBlk_F_SSD[idx][:, wdBlk_inlier_now]

            # get the trial indices
            snare_times = fi['snareCue_times']
            wdBlk_times = fi['wdBlkCue_times']
            all_trial_idx = np.argsort(np.argsort(
                np.r_[snare_times, wdBlk_times]))
            snare_trial_idx_now = zscore(all_trial_idx[:len(
                snare_times)][snareInlier[idx]][snare_inlier_now])
            snare_trial_idx.append(snare_trial_idx_now)
            wdBlk_trial_idx_now = zscore(all_trial_idx[len(
                snare_times):][wdBlkInlier[idx]][wdBlk_inlier_now])
            wdBlk_trial_idx.append(wdBlk_trial_idx_now)

            # get the session indices
            snare_session_idx.append(
                zscore(np.hstack([i*np.ones_like(session)
                    for i, session in enumerate(
                        fi['snareCue_nearestClock'])])
                    )[snareInlier[idx]][snare_inlier_now])
            wdBlk_session_idx.append(
                zscore(np.hstack([i*np.ones_like(session)
                    for i, session in enumerate(
                        fi['wdBlkCue_nearestClock'])])
                    )[wdBlkInlier[idx]][wdBlk_inlier_now])
            idx += 1

# start interface to R
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

base = importr('base')
stats = importr('stats')
parameters = importr('parameters')
lme4 = importr('lme4')

snare_subject = np.hstack([np.ones(F_SSD_now.shape[-1], int)*(i + 1)
    for i, F_SSD_now in enumerate(snare_F_SSD)])

snare_SubjToTrials = np.unique(snare_subject, return_inverse=True)[1]
EEG_labels = (['Snare{}'.format(i+1) for i in range(N_SSD)] + 
              ['WdBlk{}'.format(i+1) for i in range(N_SSD)])

###########################################
# load all the data into rpy2 R interface #
###########################################
snare_data = {}
# add EEG
for i,l in enumerate(EEG_labels):
    snare_data[l] = robjects.vectors.FloatVector(
            np.hstack([F_SSD_now[i] for F_SSD_now in snare_F_SSD]))
# add subject index
snare_data['subject'] = robjects.vectors.IntVector(
        np.arange(len(snare_F_SSD))[snare_SubjToTrials])
# add musicality
snare_data['musicality'] = robjects.vectors.FloatVector(
        musicscore[snare_SubjToTrials])
# add trial index
snare_data['trial'] = robjects.vectors.FloatVector(np.hstack(snare_trial_idx))
# add session index
snare_data['session'] = robjects.vectors.FloatVector(np.hstack(snare_session_idx))
snare_data['precision'] = robjects.vectors.FloatVector(np.abs(np.hstack(snare_deviation)))

Rsnare_data = base.data_frame(**snare_data)

# add within and between variables to the data frame
Rsnare_data = robjects.r.cbind(
        Rsnare_data,
        parameters.demean(
                Rsnare_data,
                select = robjects.r.c(*(EEG_labels + ['precision'])),
                group = 'subject'))

#################################
# generate the necessary models #
#################################
snare_models = {}

snare_models['fe_b'] = stats.lm(
        'precision ~ ' +
        ' + '.join([l + '_between' for l in EEG_labels]) +
        ' + musicality + trial + session',
        data = Rsnare_data)

snare_models['fe_w'] = stats.lm(
        'precision ~ ' +
        ' + '.join([l + '_within' for l in EEG_labels]) + 
        ' + musicality + trial + session',
        data = Rsnare_data)

snare_models['fe_wb'] = stats.lm(
        'precision ~ ' +
        ' + '.join([l + '_within' for l in EEG_labels]) + ' + ' + 
        ' + '.join([l + '_between' for l in EEG_labels]) +
        ' + musicality + trial + session',
        data = Rsnare_data)

snare_models['lme_b_i'] = lme4.lmer(
        'precision ~ ' +
        ' + '.join([l + '_between' for l in EEG_labels]) +
        ' + musicality + trial + session + ' +
        '(1 | subject)',
        data = Rsnare_data, REML=False)

snare_models['lme_w_i'] = lme4.lmer(
        'precision ~ ' +
        ' + '.join([l + '_within' for l in EEG_labels]) +
        ' + musicality + trial + session + ' +
        '(1 | subject)',
        data = Rsnare_data, REML=False)

snare_models['lme_wb_i'] = lme4.lmer(
        'precision ~ ' +
        ' + '.join([l + '_within' for l in EEG_labels]) + ' + ' + 
        ' + '.join([l + '_between' for l in EEG_labels]) +
        ' + musicality + trial + session + ' +
        '(1  | subject)',
        data = Rsnare_data, REML=False)

snare_models['lme_wb_is'] = lme4.lmer(
        'precision ~ ' +
        ' + '.join([l + '_within' for l in EEG_labels]) + ' + ' + 
        ' + '.join([l + '_between' for l in EEG_labels]) +
        ' + musicality + trial + session + ' +
        '(1  + ' + 
        ' + '.join([l + '_within' for l in EEG_labels]) + 
        '| subject)',
        data = Rsnare_data, REML=False)

AIC = {}
for key, value in snare_models.items():
    AIC[key] = stats.AIC(value)[0]

# get the best model using the AIC
best_snare_model = min(AIC, key=AIC.get)
#######################################################################
# tabulating the results from rpy2 does not seem to work, so we need  #
# to import the model names to the  R environment and save the models #
# to make the last step in R itself
#######################################################################
for key, value in snare_models.items():
    robjects.r.assign(key, value)

robjects.r("save({}, file='snare_models.rds')".format(
    ', '.join(snare_models.keys())))


#Now, in R this can be read and tabulated as:
"""
library(sjPlot)
load(file='snare_models.rds')
tab_model(fe_b, fe_w, fe_wb, lme_b_i, lme_w_i, lme_wb_i, lme_wb_is,
    show.aic=TRUE, show.re.var=FALSE, show.ci=FALSE,
    show.icc=FALSE,
    dv.labels=c("FE between", "FE within", "FE within between", "RE between",
                "RE within", "RE within between",
                "RE within between, random slopes"))
"""
