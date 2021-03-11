"""
Create the design matrix of the Within-Between Random Effects model
"""
import numpy as np
import sys
import os.path
from scipy.stats import zscore, iqr
import csv
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm
import pandas as pd

data_folder = sys.argv[1]
result_folder = sys.argv[2]

# reject behavioral outlier
iqr_rejection = True

# target frequencies
snareFreq = 7./6
wdBlkFreq = 7./4

# number of SSD_components to use
N_SSD = 2

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
    # loop through all arrays
    i = 0
    while True:
        try:
            F_SSD = fi['F_SSD_{:02d}'.format(i)][:N_SSD,
                    (snare_idx, wdBlk_idx)]
            F_SSDs.append(F_SSD)
            snareInlier.append(fi['snareInlier_{:02d}'.format(i)])
            wdBlkInlier.append(fi['wdBlkInlier_{:02d}'.format(i)])
            i += 1
        except KeyError:
            break

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

def design_matrix(F_SSD, musicscore, trial_idx, session_idx, subject_idx):
    F_SSD = [F_SSD[i] for i in subject_idx]
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
        for m, SSD_now in zip(musicscore[subject_idx], F_SSD)])))
    # add trial_idx
    labels.append('trial_idx')
    X.append(np.hstack([trial_idx[i] for i in subject_idx]))
    # add session_idx
    labels.append('session_idx')
    X.append(np.hstack([session_idx[i] for i in subject_idx]))
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
# data splitting using N_SPLIT subjects for selection
N_SPLIT = 15
random.seed(42) # actually makes a difference for the chosen coefs
select_idx = sorted(random.sample(range(len(snare_F_SSD)), N_SPLIT))
infer_idx = [i for i in range(len(snare_F_SSD)) if i not in select_idx]

# model selection
snare_design_select, snare_labels = design_matrix(
        snare_F_SSD, musicscore, snare_trial_idx, snare_session_idx, select_idx)
snareY_select = np.hstack([snare_deviation[i] for i in select_idx])

wdBlk_design_select, wdBlk_labels = design_matrix(
        wdBlk_F_SSD, musicscore, wdBlk_trial_idx, wdBlk_session_idx, select_idx)
wdBlkY_select = np.hstack([wdBlk_deviation[i] for i in select_idx])

# start interface to R
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# activate automatic conversion of numpy arrays to R
from rpy2.robjects import numpy2ri
coef = robjects.r.coef
numpy2ri.activate()

###########################################################
# fit the models with R's glmnet and extract coefficients #
###########################################################
# import R's "glmnet"
glmnet = importr('glmnet')
from rpy2.robjects.functions import SignatureTranslatedFunction as STM

glmnet.cv_glmnet = STM(glmnet.cv_glmnet,
        init_prm_translate = {'penalty_factor': 'penalty.factor'})

snare_cv_model = glmnet.cv_glmnet(
        snare_design_select[1:].T, np.abs(snareY_select.reshape(-1,1)),
        alpha = 1,
        family = 'gaussian',
        intercept = True,
        standardize = False,
        nlambda = 500,
        nfolds = 20
        #penalty_factor = np.concatenate( #0 = no shrinkage, default 1
        #    (np.ones(11), np.zeros(snare_design_select.shape[0]-11))) #11 bc. np intercept
        )

snare_coefs = np.ravel(robjects.r['as'](coef(snare_cv_model.rx2['glmnet.fit'],
        s=snare_cv_model.rx2['lambda.min']), 'matrix')) #change to lambda1se w/o penalty factor

wdBlk_cv_model = glmnet.cv_glmnet(
        wdBlk_design_select[1:].T, np.abs(wdBlkY_select.reshape(-1,1)),
        alpha = 1,
        family = 'gaussian',
        intercept = True,
        standardize = False,
        nlambda = 500,
        nfolds = 20,
        penalty_factor = np.concatenate(
            (np.ones(11), np.zeros(snare_design_select.shape[0]-11))) #11 bc. np intercept
        )

wdBlk_coefs = np.ravel(robjects.r['as'](coef(wdBlk_cv_model.rx2['glmnet.fit'],
        s=wdBlk_cv_model.rx2['lambda.min']), 'matrix'))

# plot the select model coefficients
fig, ax = plt.subplots(ncols=2)

ax[0].barh(range(12), snare_coefs[:12],
           color=np.where(snare_coefs[:12]>0, 'r', 'b'))
ax[0].set_xlim([-0.12, 0.12])
ax[0].axvline(0, c='k')
ax[0].set_yticks(range(12))
ax[0].set_yticklabels([s.replace('_', '\_') for s in snare_labels[:12]])
ax[0].set_xlabel('coefficient')
ax[0].set_title('snare vs. absolute deviation')

ax[1].barh(range(12), wdBlk_coefs[:12],
          color=np.where(wdBlk_coefs[:12]>0, 'r', 'b'))
ax[1].set_xlim([-0.12, 0.12])
ax[1].axvline(0, c='k')
ax[1].set_yticks(range(12))
ax[1].set_yticklabels([s.replace('_', '\_') for s in wdBlk_labels[:12]])
ax[1].set_xlabel('coefficient')
ax[1].set_title('wdBlk vs. absolute deviation')

fig.tight_layout()
fig.savefig(os.path.join(result_folder, 'glmnet_result.pdf'))

# model inference: use selected model and other subjects to get p value
snare_design_infer, snare_labels = design_matrix(
        snare_F_SSD, musicscore, snare_trial_idx, snare_session_idx, infer_idx)
snareY_infer = np.hstack([snare_deviation[i] for i in infer_idx])

wdBlk_design_infer, wdBlk_labels = design_matrix(
        wdBlk_F_SSD, musicscore, wdBlk_trial_idx, wdBlk_session_idx, infer_idx)
wdBlkY_infer = np.hstack([wdBlk_deviation[i] for i in infer_idx])

'''snare_ols = LinearRegression(fit_intercept=False).fit(
    snare_design[np.where(snare_coefs)].T, snareY_infer)
snare_ols.coef_'''
# sklearn package doesnt give p value, use statsmodels package
# take coefficients from selection model, take all v0 and v1
snare_coef_list = [i for i in np.where(snare_coefs)[0] if i<12] + list(range(
    12, snare_design_infer.shape[0])) #snare_labels[12] = 'RE0_00'
# store in data frame to keep coef names
snare_df = pd.DataFrame(snare_design_infer[snare_coef_list].T,
    columns = [snare_labels[i] for i in snare_coef_list])
# statsmodels does not automatically generate intercept (keep it)
snare_ols = sm.OLS(snareY_infer, snare_df)
snare_ols_fit = snare_ols.fit()
print(snare_ols_fit.summary())
