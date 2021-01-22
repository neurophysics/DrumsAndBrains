"""
Create the design matrix of the Within-Between Random Effects model
"""
import numpy as np
import sys
import os.path
from scipy.stats import zscore
import csv

data_folder = sys.argv[1]
result_folder = sys.argv[2]

# target frequencies
snareFreq = 7./6
wdBlkFreq = 7./4

# number of SSD_components to use
N_SSD = 2

# load the SSD results from all subjects into a list
F_SSD = []

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
            temp = fi['arr_{}'.format(i)][:N_SSD, (snare_idx, wdBlk_idx),:,0]
            F_SSD.append(temp.reshape((-1, temp.shape[-1]), order='F'))
            i += 1
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
    for i in list(range(1,11,1)) + list(range(12, 22, 1))])

z_musicscores = (raw_musicscores - np.mean(raw_musicscores,0)
        )/raw_musicscores.std(0)
musicscore = z_musicscores[:,1:].mean(1) # do not include the LQ

# get the total number of trials
N_trials = np.sum([SSD_now.shape[-1] for SSD_now in F_SSD])
N_subjects = len(F_SSD)

# build the design matrix
X = []
labels = []

# add intercept
labels.append('intercept')
X.append(np.ones((1, N_trials)))

# add within-subjects coefficient
labels.extend(['W1Snare', 'W2Snare', 'W1WdBlk', 'W2WdBlk'])
X.append(np.hstack([zscore(np.abs(SSD_now), -1) for SSD_now in F_SSD]))

# add between subjects coefficient
labels.extend(['B1Snare', 'B2Snare', 'B1WdBlk', 'B2WdBlk'])
X.append(zscore(
    np.hstack([np.abs(SSD_now).mean(-1)[:, np.newaxis]*np.ones(
        SSD_now.shape) for SSD_now in F_SSD]), -1))

# add musicality score
labels.extend(['musicality'])
X.append(zscore(np.hstack([m*np.ones(SSD_now.shape[-1])
    for m, SSD_now in zip(musicscore, F_SSD)])))

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

# TODO: add all other rows of the matrix
