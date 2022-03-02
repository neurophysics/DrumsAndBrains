"""
Calculate and compare different Within-Between Random Effects models
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

# the argument condition can be 'both', 'listen', or 'silence'
condition = sys.argv[3]
if condition not in ['both', 'listen', 'silence']:
    raise ValueError('condition must be both, listen, or silence')

N_subjects = 21

# reject behavioral outlier
iqr_rejection = True

# target frequencies
snareFreq = 7./6
wdBlkFreq = 7./4

# delta frequency range
delta_range = [1, 4]

# number of SSD_components to use
N_SSD = 3

EEG_labels = (['Snare{}'.format(i+1) for i in range(N_SSD)] +
              ['WdBlk{}'.format(i+1) for i in range(N_SSD)])

convolve_EEG_labels = (['SnareConvolve{}'.format(i+1) for i in range(N_SSD)] +
              ['WdBlkConvolve{}'.format(i+1) for i in range(N_SSD)])

delta_EEG_labels = (['Delta{}'.format(i+1) for i in range(N_SSD)])

# load the frequency array and inlier
snareInlier = []
wdBlkInlier = []

# loop through subjects and calculate different SSDs
F_SSDs = []
convolve_F_SSDs = [] # contrast power to neighbouring frequencies
delta_F_SSDs = [] # include total delta power

for i in range(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(result_folder, 'S%02d' % i)
                + '/rcsp_tlw.npz', 'r') as fi:
            SSD_eigvals = fi['rcsp_tlw_ratios']
            SSD_filters = fi['rcsp_tlw_filters']
            SSD_patterns = fi['rcsp_tlw_patterns']
        # standardize the SSD filters to make power comparable .......
        1/0
        with np.load(os.path.join(result_folder, 'S%02d' % i)
                + '/prepared_FFTSSD.npz', 'r') as fi:
            f = fi['f']
            #######################################################
            # calculate and append power at requested frequencies #
            #######################################################
            if condition == 'both':
                F_SSD = np.abs(np.tensordot(SSD_filters, fi['F'], axes=(0,0)))
                snareInlier.append(fi['snareInlier_{:02d}'.format(i)])
                wdBlkInlier.append(fi['wdBlkInlier_{:02d}'.format(i)])
            else:
                F_SSD = np.abs(np.tensordot(SSD_filters, fi['F_{}'.format(
                    condition)], axes=(0,0)))
                snareInlier.append(fi['snareInlier_{}_{:02d}'.format(
                    condition, i)])
                wdBlkInlier.append(fi['wdBlkInlier_{}_{:02d}'.format(
                    condition, i)])
            convolve_F_SSD = scipy.ndimage.convolve1d(
                F_SSD, np.array([-0.25, -0.25, 1, -0.25, -0.25]), axis=1)
            delta_F_SSD = np.mean(np.abs(F_SSD[:,delta_idx[0]:delta_idx[1]]),
                axis=1)
            # append to data lists
            F_SSDs.append(F_SSD[:N_SSD, (snare_idx, wdBlk_idx)])
            convolve_F_SSDs.append(convolve_F_SSD[:N_SSD,
                (snare_idx, wdBlk_idx)])
            delta_F_SSDs.append(delta_F_SSD[:N_SSD])
    except ValueError:
        print(('Warning: Subject %02d could not be loaded!' %i))

snare_idx = np.argmin((f - snareFreq)**2)
wdBlk_idx = np.argmin((f - wdBlkFreq)**2)
delta_idx = [np.argmin((f - delta_range[0])**2),
             np.argmin((f - delta_range[1])**2)]

# take log to transform to a linear scale (not needed for convolution)
F_SSDs = [np.log(F_SSD_now) for F_SSD_now in F_SSDs]
delta_F_SSDs = [np.log(F_SSD_now) for F_SSD_now in delta_F_SSDs]

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

###############################################################################
# get performance session and trial indices and separate trials into snare
# and woodblock trials
###############################################################################
snare_F_SSD = []
wdBlk_F_SSD = []
delta_snare_F_SSD = []
delta_wdBlk_F_SSD = []
convolve_snare_F_SSD = []
convolve_wdBlk_F_SSD = []

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
        #############################################
        # divide F_SSDs into wdBlk and snare trials #
        #############################################
        F_SSD_now = F_SSDs[idx]
        delta_F_SSD_now = delta_F_SSDs[idx]
        convolve_F_SSD_now = convolve_F_SSDs[idx]
        snareInlier_now = snareInlier[idx]
        wdBlkInlier_now = wdBlkInlier[idx]
        # append data to snare and woodblock SSD
        snare_temp = F_SSD_now[...,:snareInlier_now.sum()]
        wdBlk_temp = F_SSD_now[...,snareInlier_now.sum():]
        snare_F_SSD.append(
                snare_temp.reshape((-1, snare_temp.shape[-1]),
                    order='F'))
        wdBlk_F_SSD.append(
                wdBlk_temp.reshape((-1, wdBlk_temp.shape[-1]),
                    order='F'))
        delta_snare_temp = delta_F_SSD_now[...,:snareInlier_now.sum()]
        delta_wdBlk_temp = delta_F_SSD_now[...,snareInlier_now.sum():]
        delta_snare_F_SSD.append(
                delta_snare_temp.reshape((-1, delta_snare_temp.shape[-1]),
                    order='F'))
        delta_wdBlk_F_SSD.append(
                delta_wdBlk_temp.reshape((-1, delta_wdBlk_temp.shape[-1]),
                    order='F'))
        convolve_snare_temp = convolve_F_SSD_now[...,
                :snareInlier_now.sum()]
        convolve_wdBlk_temp = convolve_F_SSD_now[...,
                snareInlier_now.sum():]
        convolve_snare_F_SSD.append(
                convolve_snare_temp.reshape((-1,
                    convolve_snare_temp.shape[-1]), order='F'))
        convolve_wdBlk_F_SSD.append(
                convolve_wdBlk_temp.reshape((-1,
                    convolve_wdBlk_temp.shape[-1]), order='F'))
        #########################
        # read behavioural data #
        #########################
        with np.load(os.path.join(result_folder,'S{:02d}'.format(
            subj), 'behavioural_results.npz'), allow_pickle=True,
            encoding='bytes') as fi:
            snare_deviation_now = fi['snare_deviation'][
                    snareInlier_now]
            wdBlk_deviation_now = fi['wdBlk_deviation'][
                    wdBlkInlier_now]
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
            snare_F_SSD[idx] = snare_F_SSD[idx][:, snare_inlier_now]
            wdBlk_F_SSD[idx] = wdBlk_F_SSD[idx][:, wdBlk_inlier_now]
            delta_snare_F_SSD[idx] = delta_snare_F_SSD[idx][:, snare_inlier_now]
            delta_wdBlk_F_SSD[idx] = delta_wdBlk_F_SSD[idx][:, wdBlk_inlier_now]
            convolve_snare_F_SSD[idx] = convolve_snare_F_SSD[idx][:,
                    snare_inlier_now]
            convolve_wdBlk_F_SSD[idx] = convolve_wdBlk_F_SSD[idx][:,
                    wdBlk_inlier_now]
            # get the trial indices
            snare_times = fi['snareCue_times']
            wdBlk_times = fi['wdBlkCue_times']
            all_trial_idx = np.argsort(np.argsort(
                np.r_[snare_times, wdBlk_times]))
            snare_trial_idx.append(
                    all_trial_idx[:len(snare_times)][snareInlier_now][
                        snare_inlier_now])
            wdBlk_trial_idx.append(
                    all_trial_idx[len(snare_times):][wdBlkInlier_now][
                        wdBlk_inlier_now])
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

snare_subject = np.hstack([np.ones(F_SSD_now.shape[-1], int)*i
    for i, F_SSD_now in enumerate(snare_F_SSD)])

# I am not sure, why this line might be necessary ...
# the result remains unchanged ...
snare_SubjToTrials = np.unique(snare_subject, return_inverse=True)[1]

wdBlk_subject = np.hstack([np.ones(F_SSD_now.shape[-1], int)*i
    for i, F_SSD_now in enumerate(wdBlk_F_SSD)])

# I am not sure, why this line might be necessary ...
# the result remains unchanged ...
wdBlk_SubjToTrials = np.unique(wdBlk_subject, return_inverse=True)[1]

#########################################################################
# create data dictionary for storing as csv and subsequent R processing #
#########################################################################

def addEEGtoDict(datadict, labels, data):
    for i, l, in enumerate(labels):
        datadict[l] = np.hstack([data_now[i] for data_now in data])
        # add within data
        datadict[l + '_within'] = np.hstack([
            data_now[i] - np.mean(data_now[i])
            for data_now in data])
        # standardize for each subject
        datadict[l + '_within_standard'] = np.hstack([
            (data_now[i] - np.mean(data_now[i]))/np.std(data_now[i])
            for data_now in data])
        # add between data
        datadict[l + '_between'] = np.hstack([
            np.ones_like(data_now[i]) * np.mean(data_now[i])
            for data_now in data])

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
    with open(os.path.join(result_folder, '{}_data_{}.csv'.format(
        trial_type, condition)), 'w') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC,
                quotechar='\"')
        writer.writerow(datadict.keys())
        # loop through the items
        for row in getEntryFromMultipleArrays(datadict.values()):
            writer.writerow(row)
    
# for snare #
#############
snare_data = {}
# add EEG to dictionary
addEEGtoDict(snare_data, EEG_labels, snare_F_SSD)
addEEGtoDict(snare_data, delta_EEG_labels, delta_snare_F_SSD)
addEEGtoDict(snare_data, convolve_EEG_labels, convolve_snare_F_SSD)
# add subject index
snare_data['subject'] = (
        np.arange(len(snare_F_SSD))[snare_SubjToTrials])
# add musicality
snare_data['musicality'] = musicscore[snare_SubjToTrials]
# add trial index (no log)
snare_data['trial'] = np.hstack(snare_trial_idx)
# add session index (no log) and precision
snare_data['session'] = np.hstack(snare_session_idx)
snare_data['deviation'] = np.hstack(snare_deviation)

writeDictionaryToCSV(snare_data, 'snare')

# for woodblock #
#################
wdBlk_data = {}
# add EEG to dictionary
addEEGtoDict(wdBlk_data, EEG_labels, wdBlk_F_SSD)
addEEGtoDict(wdBlk_data, delta_EEG_labels, delta_wdBlk_F_SSD)
addEEGtoDict(wdBlk_data, convolve_EEG_labels, convolve_wdBlk_F_SSD)
# add subject index
wdBlk_data['subject'] = (
        np.arange(len(wdBlk_F_SSD))[wdBlk_SubjToTrials])
# add musicality
wdBlk_data['musicality'] = musicscore[wdBlk_SubjToTrials]
# add trial index (no log)
wdBlk_data['trial'] = np.hstack(wdBlk_trial_idx)
# add session index (no log) and precision
wdBlk_data['session'] = np.hstack(wdBlk_session_idx)
wdBlk_data['deviation'] = np.hstack(wdBlk_deviation)

writeDictionaryToCSV(wdBlk_data, 'wdBlk')

