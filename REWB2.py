"""
Calculate and compare different Within-Between Random Effects models
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

# reject behavioral outlier
iqr_rejection = True

# target frequencies
snareFreq = 7./6
wdBlkFreq = 7./4

# number of SSD_components to use
N_SSD = 2

# load the SSD results
with np.load(os.path.join(result_folder, 'FFTSSD.npz')) as f:
    SSD_eigvals = f['SSD_eigvals']
    SSD_filters = f['SSD_filters']
    SSD_patterns = f['SSD_patterns']

# load the frequency array and inlier
snareInlier = []
wdBlkInlier = []
snareInlier_listen = []
wdBlkInlier_listen = []
snareInlier_silence = []
wdBlkInlier_silence = []
i=0
while True:
    try:
        with np.load(os.path.join(result_folder, 'F_SSD.npz'), 'r') as fi:
            f = fi['f']
            snareInlier.append(fi['snareInlier_{:02d}'.format(i)])
            wdBlkInlier.append(fi['wdBlkInlier_{:02d}'.format(i)])
            snareInlier_listen.append(fi['snareInlier_listen_{:02d}'.format(i)])
            wdBlkInlier_listen.append(fi['wdBlkInlier_listen_{:02d}'.format(i)])
            snareInlier_silence.append(fi['snareInlier_silence_{:02d}'.format(i)])
            wdBlkInlier_silence.append(fi['wdBlkInlier_silence_{:02d}'.format(i)])
        # find the index of the frequency array refering to snare and woodblock
        # frequency
        snare_idx = np.argmin((f - snareFreq)**2)
        wdBlk_idx = np.argmin((f - wdBlkFreq)**2)
        harmo_idx = np.argmin((f - 2*wdBlkFreq)**2)
        delta_idx1 = np.argmin((f - 1)**2)
        delta_idx4 = np.argmin((f - 4)**2)
        i+=1
    except KeyError:
        break

# loop through subjects and calculate different SSDs
F_SSDs = []
F_SSDs_listen = []
F_SSDs_silence = []
for i in range(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(result_folder, 'S%02d' % i)
                + '/prepared_FFTSSD.npz', 'r') as fi:
            # calculate and append SSD for both listening and silence
            F_SSD = np.tensordot(SSD_filters, fi['F'], axes=(0,0))
            delta_F_SSD = np.mean(np.abs(F_SSD[:,delta_idx1:delta_idx4]),
                axis=1)
            F_SSD = np.hstack([F_SSD[:N_SSD, (snare_idx,wdBlk_idx)],
                delta_F_SSD[:N_SSD, np.newaxis]])
            F_SSDs.append(F_SSD)
            # calculate and append SSD for listening window
            F_SSD_listen = np.tensordot(SSD_filters, fi['F_listen'], axes=(0,0))
            delta_F_SSD = np.mean(np.abs(F_SSD_listen[:,delta_idx1:delta_idx4]),
                axis=1)
            F_SSD_listen = np.hstack([F_SSD_listen[:N_SSD, (snare_idx,wdBlk_idx)],
                delta_F_SSD[:N_SSD, np.newaxis]])
            F_SSDs_listen.append(F_SSD_listen)

            F_SSD_silence = np.tensordot(SSD_filters, fi['F_silence'], axes=(0,0))
            delta_F_SSD = np.mean(np.abs(F_SSD_silence[:,delta_idx1:delta_idx4]),
                axis=1)
            F_SSD_silence = np.hstack([F_SSD_silence[:N_SSD, (snare_idx,wdBlk_idx)],
                delta_F_SSD[:N_SSD, np.newaxis]])
            F_SSDs_silence.append(F_SSD_silence)

    except:
        print(('Warning: Subject %02d could not be loaded!' %i))

# take absolute value to get EEG amplitude and log to transform
# to a linear scale
F_SSDs = [np.log(np.abs(F_SSD_now)) for F_SSD_now in F_SSDs]
F_SSDs_listen = [np.log(np.abs(F_SSD_now)) for F_SSD_now in F_SSDs_listen]
F_SSDs_silence = [np.log(np.abs(F_SSD_now)) for F_SSD_now in F_SSDs_silence]

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

for ssd_type in ['both', 'listen', 'silence']:
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
            if ssd_type=='both':
                F_SSD = F_SSDs[idx]
                snareInlier_now = snareInlier[idx]
                wdBlkInlier_now = wdBlkInlier[idx]
            if ssd_type=='listen':
                F_SSD = F_SSDs_listen[idx]
                snareInlier_now = snareInlier_listen[idx]
                wdBlkInlier_now = wdBlkInlier_listen[idx]
            if ssd_type=='silence':
                F_SSD = F_SSDs_silence[idx]
                snareInlier_now = snareInlier_silence[idx]
                wdBlkInlier_now = wdBlkInlier_silence[idx]

            snare_temp = F_SSD[...,:snareInlier_now.sum()]
            wdBlk_temp = F_SSD[...,snareInlier_now.sum():]
            snare_F_SSD.append(snare_temp.reshape((-1, snare_temp.shape[-1]),
                order='F'))
            wdBlk_F_SSD.append(wdBlk_temp.reshape((-1, wdBlk_temp.shape[-1]),
                order='F'))
            with np.load(os.path.join(result_folder,'S{:02d}'.format(subj),
                'behavioural_results.npz'), allow_pickle=True,
                encoding='bytes') as fi:
                snare_deviation_now = fi['snare_deviation'][snareInlier_now]
                wdBlk_deviation_now = fi['wdBlk_deviation'][wdBlkInlier_now]

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

                snare_deviation.append(
                    snare_deviation_now[snare_inlier_now])
                wdBlk_deviation.append(
                    wdBlk_deviation_now[wdBlk_inlier_now])
                snare_F_SSD[idx] = snare_F_SSD[idx][:, snare_inlier_now]
                wdBlk_F_SSD[idx] = wdBlk_F_SSD[idx][:, wdBlk_inlier_now]

                # get the trial indices
                snare_times = fi['snareCue_times']
                wdBlk_times = fi['wdBlkCue_times']
                all_trial_idx = np.argsort(np.argsort(
                    np.r_[snare_times, wdBlk_times]))
                snare_trial_idx_now = all_trial_idx[:len(
                    snare_times)][snareInlier_now][snare_inlier_now]
                snare_trial_idx.append(snare_trial_idx_now)
                wdBlk_trial_idx_now = all_trial_idx[len(
                    snare_times):][wdBlkInlier_now][wdBlk_inlier_now]
                wdBlk_trial_idx.append(wdBlk_trial_idx_now)

                # get the session indices
                snare_session_idx.append(
                    np.hstack([i*np.ones_like(session)
                        for i, session in enumerate(
                            fi['snareCue_nearestClock'])])
                        [snareInlier_now][snare_inlier_now])
                wdBlk_session_idx.append(
                    np.hstack([i*np.ones_like(session)
                        for i, session in enumerate(
                            fi['wdBlkCue_nearestClock'])])
                        [wdBlkInlier_now][wdBlk_inlier_now])
                idx += 1

    # start interface to R
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    base = importr('base')
    stats = importr('stats')
    parameters = importr('parameters')
    lme4 = importr('lme4')
    sjPlot = importr('sjPlot')
    effectsize = importr('effectsize')

    snare_subject = np.hstack([np.ones(F_SSD_now.shape[-1], int)*(i + 1)
        for i, F_SSD_now in enumerate(snare_F_SSD)])

    snare_SubjToTrials = np.unique(snare_subject, return_inverse=True)[1]
    EEG_labels = (['Snare{}'.format(i+1) for i in range(N_SSD)] +
                  ['WdBlk{}'.format(i+1) for i in range(N_SSD)] +
                  ['Delta{}'.format(i+1) for i in range(N_SSD)])

    wdBlk_subject = np.hstack([np.ones(F_SSD_now.shape[-1], int)*(i + 1)
        for i, F_SSD_now in enumerate(wdBlk_F_SSD)])

    wdBlk_SubjToTrials = np.unique(wdBlk_subject, return_inverse=True)[1]

    ###########################################
    # load all the data into rpy2 R interface #
    ###########################################
    snare_data = {}
    wdBlk_data = {}

    # add EEG
    # absolute value and log have already been take above!
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
    # add trial index
    snare_data['trial'] = robjects.vectors.FloatVector(np.log(np.hstack(snare_trial_idx) + 1))
    wdBlk_data['trial'] = robjects.vectors.FloatVector(np.log(np.hstack(wdBlk_trial_idx) + 1))
    # add session index
    snare_data['session'] = robjects.vectors.FloatVector(np.log(np.hstack(snare_session_idx) + 1))
    snare_data['precision'] = robjects.vectors.FloatVector(np.log(np.abs(np.hstack(snare_deviation))))
    wdBlk_data['session'] = robjects.vectors.FloatVector(np.log(np.hstack(wdBlk_session_idx) + 1))
    wdBlk_data['precision'] = robjects.vectors.FloatVector(np.log(np.abs(np.hstack(wdBlk_deviation))))

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

    #################################
    # generate the necessary models #
    #################################
    ###### snare ######
    snare_models = {}

    snare_models['fe_b'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_between' for l in EEG_labels]) +
            ' + musicality + trial + session',
            data = Rsnare_data)

    snare_models['fe_b_onlysnare'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_between' for l in EEG_labels
                if l.startswith('Snare')]) +
            ' + musicality + trial + session',
            data = Rsnare_data)

    snare_models['fe_w'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) +
            ' + musicality + trial + session',
            data = Rsnare_data)

    snare_models['fe_w_onlysnare'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels
                if l.startswith('Snare')]) +
            ' + musicality + trial + session',
            data = Rsnare_data)

    snare_models['fe_wb'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels]) +
            ' + musicality + trial + session',
            data = Rsnare_data)

    snare_models['fe_wb_onlysnare'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels
                if l.startswith('Snare')]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels
                if l.startswith('Snare')]) +
            ' + musicality + trial + session',
            data = Rsnare_data)

    snare_models['lme_b_i'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_between' for l in EEG_labels]) +
            ' + musicality + trial + session + ' +
            '(1 | subject)',
            data = Rsnare_data, REML=False)

    snare_models['lme_b_i_onlysnare'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_between' for l in EEG_labels
                if l.startswith('Snare')]) +
            ' + musicality + trial + session + ' +
            '(1 | subject)',
            data = Rsnare_data, REML=False)

    snare_models['lme_w_i'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) +
            ' + musicality + trial + session + ' +
            '(1 | subject)',
            data = Rsnare_data, REML=False)

    snare_models['lme_w_i_onlysnare'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels
                if l.startswith('Snare')]) +
            ' + musicality + trial + session + ' +
            '(1 | subject)',
            data = Rsnare_data, REML=False)

    snare_models['lme_wb_i'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels]) +
            ' + musicality + trial + session + ' +
            '(1  | subject)',
            data = Rsnare_data, REML=False)

    snare_models['lme_wb_i_onlysnare'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels
                if l.startswith('Snare')]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels
                if l.startswith('Snare')]) +
            ' + musicality + trial + session + ' +
            '(1  | subject)',
            data = Rsnare_data, REML=False)

    snare_models['lme_wb_is'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels]) +
            ' + musicality + trial + session + ' +
            '(1  + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) +
            '| subject)',
            data = Rsnare_data, REML=False)

    snare_models['lme_wb_is_onlysnare'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels
                if l.startswith('Snare')]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels
                if l.startswith('Snare')]) +
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

    ###### woodblock ######
    wdBlk_models = {}

    wdBlk_models['fe_b'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_between' for l in EEG_labels]) +
            ' + musicality + trial + session',
            data = RwdBlk_data)

    wdBlk_models['fe_b_onlywdBlk'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_between' for l in EEG_labels
                if l.startswith('WdBlk')]) +
            ' + musicality + trial + session',
            data = RwdBlk_data)

    wdBlk_models['fe_w'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) +
            ' + musicality + trial + session',
            data = RwdBlk_data)

    wdBlk_models['fe_w_onlywdBlk'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels
                if l.startswith('WdBlk')]) +
            ' + musicality + trial + session',
            data = RwdBlk_data)

    wdBlk_models['fe_wb'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels]) +
            ' + musicality + trial + session',
            data = RwdBlk_data)

    wdBlk_models['fe_wb_onlywdBlk'] = stats.lm(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels
                if l.startswith('WdBlk')]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels
                if l.startswith('WdBlk')]) +
            ' + musicality + trial + session',
            data = RwdBlk_data)

    wdBlk_models['lme_b_i'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_between' for l in EEG_labels]) +
            ' + musicality + trial + session + ' +
            '(1 | subject)',
            data = RwdBlk_data, REML=False)

    wdBlk_models['lme_b_i_onlywdBlk'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_between' for l in EEG_labels
                if l.startswith('WdBlk')]) +
            ' + musicality + trial + session + ' +
            '(1 | subject)',
            data = RwdBlk_data, REML=False)

    wdBlk_models['lme_w_i'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) +
            ' + musicality + trial + session + ' +
            '(1 | subject)',
            data = RwdBlk_data, REML=False)

    wdBlk_models['lme_w_i_onlywdBlk'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels
                if l.startswith('WdBlk')]) +
            ' + musicality + trial + session + ' +
            '(1 | subject)',
            data = RwdBlk_data, REML=False)

    wdBlk_models['lme_wb_i'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels]) +
            ' + musicality + trial + session + ' +
            '(1  | subject)',
            data = RwdBlk_data, REML=False)

    wdBlk_models['lme_wb_i_onlywdBlk'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels
                if l.startswith('WdBlk')]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels
                if l.startswith('WdBlk')]) +
            ' + musicality + trial + session + ' +
            '(1  | subject)',
            data = RwdBlk_data, REML=False)

    wdBlk_models['lme_wb_is'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels]) +
            ' + musicality + trial + session + ' +
            '(1  + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) +
            '| subject)',
            data = RwdBlk_data, REML=False)

    wdBlk_models['lme_wb_is_onlywdBlk'] = lme4.lmer(
            'precision ~ ' + '0 + ' +
            ' + '.join([l + '_within' for l in EEG_labels
                if l.startswith('WdBlk')]) + ' + ' +
            ' + '.join([l + '_between' for l in EEG_labels
                if l.startswith('WdBlk')]) +
            ' + musicality + trial + session + ' +
            '(1  + ' +
            ' + '.join([l + '_within' for l in EEG_labels]) +
            '| subject)',
            data = RwdBlk_data, REML=False)

    AIC = {}
    for key, value in wdBlk_models.items():
        AIC[key] = stats.AIC(value)[0]

    # get the best model using the AIC
    best_wdBlk_model = min(AIC, key=AIC.get)




    #######################################################################
    # tabulating the results from rpy2 does not seem to work, so we need  #
    # to import the model names to the  R environment and save the models #
    # to make the last step in R itself
    #######################################################################
    for key, value in snare_models.items():
        base.assign(key, value)

    robjects.r("save({}, file='snare_models.rds')".format(
        ', '.join(snare_models.keys())))

    #Now, the data can be opend and tabulated in R
    with open('tabulate_snare_models.r', 'w') as f:
        f.writelines("library(sjPlot)" + "\n")
        f.writelines("load(file='snare_models.rds')" + "\n")
        f.writelines("tab_model({}, show.aic=TRUE, show.re.var=FALSE, show.ci=FALSE, show.icc=FALSE, dv.labels=c('{}'), file='Results/snare_models_{:s}.html')".format(
            ", ".join(snare_models.keys()),
            "', '".join(snare_models.keys()),
            ssd_type)
            )

    os.system('Rscript tabulate_snare_models.r')

    for key, value in wdBlk_models.items():
        base.assign(key, value)

    robjects.r("save({}, file='wdBlk_models.rds')".format(
        ', '.join(wdBlk_models.keys())))

    #Now, the data can be opend and tabulated in R
    with open('tabulate_wdBlk_models.r', 'w') as f:
        f.writelines("library(sjPlot)" + "\n")
        f.writelines("load(file='wdBlk_models.rds')" + "\n")
        f.writelines("tab_model({}, show.aic=TRUE, show.re.var=FALSE, show.ci=FALSE, show.icc=FALSE, dv.labels=c('{}'), file='Results/wdBlk_models_{:s}.html')".format(
            ", ".join(wdBlk_models.keys()),
            "', '".join(wdBlk_models.keys()),
            ssd_type)
            )

    os.system('Rscript tabulate_wdBlk_models.r')
