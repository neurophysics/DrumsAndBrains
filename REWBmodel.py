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

def design_matrix(F_SSD, musicscore, trial_idx, session_idx, subject_idx):
    F_SSD = [F_SSD[i] for i in subject_idx]
    N_trials = sum([SSD_now.shape[-1] for SSD_now in F_SSD])
    N_subjects = len(F_SSD)
    X = []
    labels_X = []
    # add intercept
    labels_X.append('intercept')
    X.append(np.ones((1, N_trials)))
    # add within-subjects coefficient
    labels_X.extend(['WSnare{}'.format(i + 1) for i in range(N_SSD)] +
                    ['WWdblk{}'.format(i + 1) for i in range(N_SSD)]) # +
                    #['WHarmo{}'.format(i + 1) for i in range(N_SSD)])
    X.append(np.hstack([zscore(np.abs(SSD_now), -1) for SSD_now in F_SSD]))
    # get mean and std of between subjects effect
    B_mean = np.mean([np.abs(SSD_now).mean(-1) for SSD_now in F_SSD], 0)
    B_std = np.std([np.abs(SSD_now).mean(-1) for SSD_now in F_SSD], 0)
    # add between subjects coefficient
    labels_X.extend(['BSnare{}'.format(i + 1) for i in range(N_SSD)] +
                    ['BWdblk{}'.format(i + 1) for i in range(N_SSD)]) # +
                    #['BHarmo{}'.format(i + 1) for i in range(N_SSD)])
    X.append(
            np.hstack([((np.abs(SSD_now).mean(-1) - B_mean)/
                B_std)[:, np.newaxis] * np.ones(SSD_now.shape)
                for SSD_now in F_SSD]))
    # add musicality score
    labels_X.append('musicality')
    X.append(zscore(np.hstack([m*np.ones(SSD_now.shape[-1])
        for m, SSD_now in zip(musicscore[subject_idx], F_SSD)])))
    # add trial_idx
    labels_X.append('trial_idx')
    X.append(np.hstack([trial_idx[i] for i in subject_idx]))
    # add session_idx
    labels_X.append('session_idx')
    X.append(np.hstack([session_idx[i] for i in subject_idx]))
    return np.vstack(X), labels_X

def single_REMatrix(F_SSD_now, idx=0):
    """calculate REMatrix for a single subject"""
    Z = []
    N_trials = F_SSD_now.shape[-1]
    # add random effect for the intercept
    RE0 = np.ones([1, N_trials])
    Z.append(RE0)
    # add random effect for the within-subjects (REW) effect
    REW = zscore((np.abs(F_SSD_now) - np.abs(F_SSD_now).mean(-1)[
        :,np.newaxis]), axis=-1)
    Z.append(REW)
    return np.vstack(Z)

def REMatrix(F_SSD, subject_idx):
    labels = []
    F_SSD = [F_SSD[i] for i in subject_idx]
    Z = np.hstack([single_REMatrix(F_SSD_now, i)
        for i,F_SSD_now in enumerate(F_SSD)])
    labels.append('RE0')
    labels.extend(
        ['REWSnare{:d}'.format(i + 1) for i in range(N_SSD)] +
        ['REWWdblk{:d}'.format(i + 1) for i in range(N_SSD)])
    #    ['REWHarmo{:d}'.format(i + 1) for i in range(N_SSD)])
    subject = np.hstack([np.ones(F_SSD_now.shape[-1], int)*(i + 1)
        for i, F_SSD_now in enumerate(F_SSD)])
    return Z, labels, subject

# finally, get the design matrices
# data splitting using N_SPLIT subjects out of 20 for selection
N_SPLIT = 20
random.seed(42) # actually makes a difference for the chosen coefs
select_idx = sorted(random.sample(range(len(snare_F_SSD)), N_SPLIT))
infer_idx = [i for i in range(len(snare_F_SSD)) if i not in select_idx]


# model selection
snareX_select, snare_labelsX = design_matrix(snare_F_SSD,
                                             musicscore,
                                             snare_trial_idx,
                                             snare_session_idx,
                                             select_idx)
snareY_select = np.abs(np.hstack([snare_deviation[i] for i in select_idx]))
snareZ_select, snare_labelsZ, snare_subject = REMatrix(snare_F_SSD, select_idx)

wdBlkX_select, wdBlk_labelsX = design_matrix(wdBlk_F_SSD,
                                             musicscore,
                                             wdBlk_trial_idx,
                                             wdBlk_session_idx,
                                             select_idx)
wdBlkY_select = np.abs(np.hstack([wdBlk_deviation[i] for i in select_idx]))
wdBlkZ_select, wdBlk_labelsZ, wdBlk_subject = REMatrix(wdBlk_F_SSD, select_idx)

N_trials_snare = [SSD_now.shape[-1] for SSD_now in snare_F_SSD]
N_trials_wdBlk = [SSD_now.shape[-1] for SSD_now in wdBlk_F_SSD]

# start interface to R
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# activate automatic conversion of numpy arrays to R
from rpy2.robjects import numpy2ri, IntVector, Formula
coef = robjects.r.coef
confint = robjects.r.confint
residuals = robjects.r.residuals
quantile = robjects.r.quantile
predict = robjects.r.predict
lm = robjects.r.lm

numpy2ri.activate()
lme4 = importr('lme4')
fixef = robjects.r.fixef
base = importr('base')
stats = importr('stats')

# define a function for the residual sum of squares
RSS = lambda model: np.sum(np.array(robjects.r.resid(model))**2)

nfolds = 10
snare_folds = meet.elm.ssk_cv(snareX_select.T, labels=snare_subject, folds=nfolds)
wdBlk_folds = meet.elm.ssk_cv(wdBlkX_select.T, labels=wdBlk_subject, folds=nfolds)

# residual sum of squares of linear model
snare_lm_RSS = 0
wdBlk_lm_RSS = 0
# residual sum of squares of linear mixed effects models
snare_lme_RSS1 = 0
snare_lme_RSS2 = 0
wdBlk_lme_RSS1 = 0
wdBlk_lme_RSS2 = 0
# fit lmer model for snare
# this needs to be this unelegant because of how the R constructs the data.frames
x_string = (' + ').join(['X'+str(i) for i in range(1,snareX_select.shape[0])])
z_string = (' + ').join(['X'+str(i)+'.1' for i in range(1,snareZ_select.shape[0])])

#g_string
lm_fmla = Formula('y ~ 1 + ' + x_string)
lme_fmla1 = Formula('y ~ 1 + ' + x_string + ' + (1 | g)')
lme_fmla2 = Formula('y ~ 1 + ' + x_string + ' + (1 + ' + z_string + ' | g)')

''' this works in r:
x <- array(rnorm(40), dim=c(10,4))
y <- 5 + rnorm(10)
model <- lm(y ~ 1 + X1 + X2 + X3 + X4, data=data.frame(x))
x_test <- array(seq(-3, 3, 0.25), dim=c(7,4))
predict(model, newdata=data.frame(x_test))
'''
# loop through the folds
for i in range(nfolds):
    # snare
    snare_test_idx = snare_folds[i]
    snare_train_idx =  np.hstack([snare_folds[j]
        for j in range(nfolds) if j != i])
    # fit models
    robjects.globalenv['y'] = np.abs(snareY_select[snare_train_idx]
            ).reshape(-1,1) #(1132, 1)
    y_test = np.abs(snareY_select[snare_test_idx])
    robjects.globalenv['x_train'] = snareX_select[:,snare_train_idx][1:].T #(1132, 11)
    robjects.globalenv['g'] = snare_subject[snare_train_idx] #(1132,)
    robjects.globalenv['z_train'] = snareZ_select[:,snare_train_idx][1:].T #(1132, 4)
    snare_lm_model = lm(lm_fmla,
        data = robjects.r('data.frame(x_train)'))
    snare_lme_model1 = lme4.lmer(lme_fmla1,
        data = robjects.r('data.frame(x_train,g)'))
    snare_lme_model2 = lme4.lmer(lme_fmla2,
        data = robjects.r('data.frame(x_train,z_train,g)'))

    # test models
    robjects.globalenv['x_test'] = snareX_select[:,snare_test_idx][1:].T #(127, 11)
    robjects.globalenv['g'] = snare_subject[snare_test_idx]
    robjects.globalenv['z_test'] = snareZ_select[:,snare_test_idx][1:].T
    snare_lm_RSS += np.sum(y_test - np.array(predict(snare_lm_model,
        newdata = robjects.r('data.frame(x_test)')))**2)
    snare_lme_RSS1 += np.sum(y_test - np.array(predict(snare_lme_model1,
        newdata = robjects.r('data.frame(x_test,g)')))**2)
    snare_lme_RSS2 += np.sum(y_test - np.array(predict(snare_lme_model2,
        newdata = robjects.r('data.frame(x_test,z_test,g)')))**2)

    # woodblock
    wdBlk_test_idx = wdBlk_folds[i]
    wdBlk_train_idx =  np.hstack([wdBlk_folds[j]
        for j in range(nfolds) if j != i])
    # fit models
    robjects.globalenv['y'] = np.abs(wdBlkY_select[wdBlk_train_idx]
            ).reshape(-1,1) #(1132, 1)
    y_test = np.abs(wdBlkY_select[wdBlk_test_idx])
    robjects.globalenv['x_train'] = wdBlkX_select[:,wdBlk_train_idx][1:].T #(1132, 11)
    robjects.globalenv['g'] = wdBlk_subject[wdBlk_train_idx] #(1132,)
    robjects.globalenv['z_train'] = wdBlkZ_select[:,wdBlk_train_idx][1:].T #(1132, 4)
    wdBlk_lm_model = lm(lm_fmla,
        data = robjects.r('data.frame(x_train)'))
    wdBlk_lme_model1 = lme4.lmer(lme_fmla1,
        data = robjects.r('data.frame(x_train,g)'))
    wdBlk_lme_model2 = lme4.lmer(lme_fmla2,
        data = robjects.r('data.frame(x_train,z_train,g)'))

    # test models
    robjects.globalenv['x_test'] = wdBlkX_select[:,wdBlk_test_idx][1:].T #(127, 11)
    robjects.globalenv['g'] = wdBlk_subject[wdBlk_test_idx]
    robjects.globalenv['z_test'] = wdBlkZ_select[:,wdBlk_test_idx][1:].T
    wdBlk_lm_RSS += np.sum(y_test - np.array(predict(wdBlk_lm_model,
        newdata = robjects.r('data.frame(x_test)')))**2)
    wdBlk_lme_RSS1 += np.sum(y_test - np.array(predict(wdBlk_lme_model1,
        newdata = robjects.r('data.frame(x_test,g)')))**2)
    wdBlk_lme_RSS2 += np.sum(y_test - np.array(predict(wdBlk_lme_model2,
        newdata = robjects.r('data.frame(x_test,z_test,g)')))**2)

print('RSS after CV\nSnare\nlm: \t', snare_lm_RSS, '\nlme1: \t',
    snare_lme_RSS1, '\nlme2:\t', snare_lme_RSS2,'\nWoodblock\nlm: \t',
    wdBlk_lm_RSS, '\nlme1: \t', wdBlk_lme_RSS1, '\nlme2:\t', wdBlk_lme_RSS2)

# get coefficients
snare_lm_FEcoef = np.ravel(coef(snare_lm_model))
snare_lme1_FEcoef = np.ravel(fixef(snare_lme_model1))
snare_lme2_FEcoef = np.ravel(fixef(snare_lme_model2))
wdBlk_lm_FEcoef = np.ravel(coef(wdBlk_lm_model))
wdBlk_lme1_FEcoef = np.ravel(fixef(wdBlk_lme_model1))
wdBlk_lme2_FEcoef = np.ravel(fixef(wdBlk_lme_model2))

1/0
nsim = 500
# get confidence intervals for FE with bootstrap (this takes about 1,5*nsim sec)
# see also: https://rdrr.io/cran/lme4/man/confint.merMod.html
# alternatively use wald and afterwards delete nans:
#snare_confint_wald = np.ravel(confint(snare_lme_model,level=0.95,method='Wald'))
snare_lm_confint = np.ravel(confint(snare_lme_model, parm = 'beta_', #beta means only FE
    level=0.95, method='boot', nsim=nsim)) #[0.25 for parm1, 0.75 fpr parm2, 0.25 for parm1,...]
# transform to error bars: first line + values, second line - values for coef
snare_confint = [(snare_confint[i], snare_confint[i+1])
    for i in range(0, len(snare_confint), 2)]




snare_errbar = np.transpose(np.array([[np.abs(l - snare_FEcoef[int(i/2)]),
    np.abs(u - snare_FEcoef[int(i/2)])] for l,u in snare_confint]))
# I still dont get the error why profiling dpoes not work:
# error: Profiling over both the residual variance and
# fixed effects is not numerically consistent with
# profiling over the fixed effects only

#calculate p-value from CI
se = [(u-l)/(2*1.96) for l,u in snare_confint]
z_values = [e/s for e,s in zip(snare_FEcoef, se)]
snare_FEp = [np.exp(-0.717*z - 0.416*z**2) for z in z_values]
snare_FEp_str = [str(round(p,3)) if p>0.0001 else '<0.0001' for p in snare_FEp]
# not skewed, good sign that we have all relevant variables (and possibly more):
# plt.hist(residuals(snare_lme_model, 'pearson', scaled=True), 100)

# if package doesnt give p value:
# permute Y 1000x => get coef distribution under H0 (no relation between input and output),
# compute p by prob of coefficients we got in permutations

1/0


# computer for wdBlk
wdBlk_fmla = Formula('y ~ 0 + x + (0 + z | g)')
wdBlk_fmla.environment['y'] = np.abs(wdBlkY_select.reshape(-1,1))
wdBlk_fmla.environment[x_train] = wdBlkX_select.T
wdBlk_fmla.environment['z'] = wdBlkZ_select[wdBlkZ_select!=0].reshape(
    5,sum(N_trials_wdBlk)).T
wdBlk_fmla.environment['g'] = wdBlk_subjects.T[:,np.newaxis]
wdBlk_lme_model = lme4.lmer(wdBlk_fmla)

wdBlk_FEcoef = np.ravel(fixef(wdBlk_lme_model)) # stronger wdBlk power, musicality => better accuracy
# get confidence intervals for FE with bootstrap (this takes about 1,5*nsim sec)
wdBlk_confint = np.ravel(confint(wdBlk_lme_model, parm = 'beta_',
    level=0.95, method='boot', nsim=nsim))
wdBlk_confint = [(wdBlk_confint[i], wdBlk_confint[i+1])
    for i in range(0, len(wdBlk_confint), 2)]
# transform to error bars: first line + values, second line - values for coef
wdBlk_errbar = np.transpose(np.array([
    [np.abs(wdBlk_confint[i] - wdBlk_FEcoef[int(i/2)]),
    np.abs(wdBlk_confint[i+1] - wdBlk_FEcoef[int(i/2)])]
    for i in range(0, len(wdBlk_confint), 2)]))

#calculate p-value from CI
se = [(u-l)/(2*1.96) for l,u in wdBlk_confint]
z_values = [e/s for e,s in zip(wdBlk_FEcoef, se)]
wdBlk_FEp = [np.exp(-0.717*z - 0.416*z**2) for z in z_values]
wdBlk_FEp_str = [str(round(p,3)) if p>0.0001 else '<0.0001' for p in wdBlk_FEp]

# plot FE coefficients
fig, ax = plt.subplots(ncols=2)
ax[0].barh(range(12), snare_FEcoef, xerr=snare_errbar,
           color=np.where(snare_FEcoef>0, 'r', 'b'))
ax[0].set_xlim([-0.2, 0.2])
ax[0].axvline(0, c='k')
ax[0].set_yticks(range(12))
ax[0].set_yticklabels([s.replace('_', ' ') for s in snare_labelsX])
ax[0].set_xlabel('coefficient')
ax[0].set_title('snare vs. absolute deviation')

ax[1].barh(range(12), wdBlk_FEcoef, xerr=wdBlk_errbar,
          color=np.where(wdBlk_FEcoef>0, 'r', 'b'))
ax[1].set_xlim([-0.2, 0.2])
ax[1].axvline(0, c='k')
ax[1].set_yticks(range(12))
ax[1].set_yticklabels([s.replace('_', ' ') for s in wdBlk_labelsX])
ax[1].set_xlabel('coefficient')
ax[1].set_title('wdBlk vs. absolute deviation')

fig.tight_layout()
fig.savefig(os.path.join(result_folder, 'lme_FEcoef.pdf'))
1/0
# to run glmnet, X and Z need to be united
# (unclear whether this makes sense)
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
ax[0].set_yticklabels([s.replace('_', ' ') for s in snare_labels[:12]])
ax[0].set_xlabel('coefficient')
ax[0].set_title('snare vs. absolute deviation')

ax[1].barh(range(12), wdBlk_coefs[:12],
          color=np.where(wdBlk_coefs[:12]>0, 'r', 'b'))
ax[1].set_xlim([-0.12, 0.12])
ax[1].axvline(0, c='k')
ax[1].set_yticks(range(12))
ax[1].set_yticklabels([s.replace('_', ' ') for s in wdBlk_labels[:12]])
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
