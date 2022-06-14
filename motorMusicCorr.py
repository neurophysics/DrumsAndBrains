"""
calculates and plots (equivalently to SSDMusicCorr.py)
the correlation of musicality vs. motor potential strength
"""
import numpy as np
import scipy
from scipy import linalg, stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os.path
import helper_functions
import meet
import csv

data_folder = sys.argv[1]
result_folder = sys.argv[2]

# load musicality value
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

ERDCSP_trial = [] # ERDCSP per subject, each shape (Nband, CSPcomp,trial)
BPLDA = [] # lda applied to BP, len 20, each (1xx,)=N_trials
i=0
while True:
    try:
        with np.load(os.path.join(result_folder,'motor/ERDCSP.npz')) as f:
            ERDCSP_trial.append(f['ERDCSP_trial_{:02d}'.format(i)]) #might not need this here
        with np.load(os.path.join(result_folder,'motor/BPLDA.npz')) as f:
            BPLDA.append(f['BPLDA_{:02d}'.format(i)])
        i+=1
    except KeyError:
        break

ERD1_alpha = np.array([e[0].mean(-1)[0] for e in ERDCSP_trial])
ERD2_alpha = np.array([e[0].mean(-1)[1] for e in ERDCSP_trial])
ERD1_beta = np.array([e[1].mean(-1)[0] for e in ERDCSP_trial])
ERD2_beta = np.array([e[1].mean(-1)[1] for e in ERDCSP_trial])
# average over CSP components so we have one for alpha and one for beta?
#lieber nur die ersten beiden
# bp auf separaten plots
# erd in 2x2 grid

BPLDA_avg = np.array([b.mean() for b in BPLDA])

# calculate bootstrap coefficient
def boot(data, N_bootstrap = 50000):
    corr = np.corrcoef(musicscore.argsort().argsort(),
        data.argsort().argsort())[0,1] #spearman rank reduces outlier influence
    corr_boot = np.array([
        np.corrcoef(musicscore.argsort().argsort(),
            np.random.choice(data.argsort().argsort(), size=len(data), replace=False))[0,1]
        for _ in range(N_bootstrap)])
    corr_p = (np.sum(corr_boot>=corr) + 1)/(N_bootstrap + 1)
    return corr_p,corr

BP_corr_p, BP_corr = boot(BPLDA_avg)
ERD1_alpha_corr_p, ERD1_alpha_corr = boot(ERD1_alpha)
ERD2_alpha_corr_p, ERD2_alpha_corr = boot(ERD2_alpha)
ERD1_beta_corr_p, ERD1_beta_corr = boot(ERD1_beta)
ERD2_beta_corr_p, ERD2_beta_corr = boot(ERD2_beta)

x = np.arange(-1.5,3,1)

# plot the result
### BP
fig = plt.figure(figsize=(4,3.5))
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.92, bottom=0.13, left=0.15, right=0.98,
    hspace=0.1, wspace=0.1)
slope, intercept = scipy.stats.linregress(musicscore, BPLDA_avg)[:2] #regression line
ax.scatter(musicscore, BPLDA_avg, c='k', s=20)
ax.plot(x, slope*x + intercept, 'k-')
ax.set_ylabel('BPLDA')
ax.set_xlabel('musicality (z-score)')
ax.text(0.95, 0.5,
        r'''Spearman's $R^2=%.2f$''' % BP_corr**2 + '\n' + r'$p=%.3f$' % BP_corr_p,
        ha='right', va='bottom', ma='right', transform=ax.transAxes, fontsize=7)
fig.suptitle('Correlation of BP and musicality')
fig.savefig(os.path.join(result_folder, 'motor/motorMusicCorr_BP.pdf'))
plt.close()
### ERD
# one dot per subject
fig, axs = plt.subplots(2, 2, figsize=(7,6),
        sharex=True, sharey=False)
fig.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.98,
    hspace=0.1, wspace=0.2)
#ERD alpha
slope, intercept = scipy.stats.linregress(musicscore, ERD1_alpha)[:2] #regression line
axs[0,0].scatter(musicscore, ERD1_alpha, c='k', s=20)
axs[0,0].plot(x, slope*x + intercept, 'k-')
axs[0,0].set_ylabel('alpha ERD 1')
axs[0,0].text(0.95, 0.5,
        r'''Spearman's $R^2=%.2f$''' % ERD1_alpha_corr**2 + '\n'
        + r'$p=%.3f$' % ERD1_alpha_corr_p,
        ha='right', va='bottom', ma='right', transform=axs[0,0].transAxes, fontsize=7)

slope, intercept = scipy.stats.linregress(musicscore, ERD2_alpha)[:2] #regression line
axs[0,1].scatter(musicscore, ERD2_alpha, c='k', s=20)
axs[0,1].plot(x, slope*x + intercept, 'k-')
axs[0,1].set_ylabel('alpha ERD 2')
axs[0,1].text(0.95, 0.5,
        r'''Spearman's $R^2=%.2f$''' % ERD2_alpha_corr**2 + '\n'
        + r'$p=%.3f$' % ERD2_alpha_corr_p,
        ha='right', va='bottom', ma='right', transform=axs[0,1].transAxes, fontsize=7)
#ERD beta
slope, intercept = scipy.stats.linregress(musicscore, ERD1_beta)[:2] #regression line
axs[1,0].scatter(musicscore, ERD1_beta, c='k', s=20)
axs[1,0].plot(x, slope*x + intercept, 'k-')
axs[1,0].set_xlabel('musical experience (z-score)')
axs[1,0].set_ylabel('beta ERD 1')
axs[1,0].text(0.95, 0.5,
        r'''Spearman's $R^2=%.2f$''' % ERD1_beta_corr**2 + '\n'
        + r'$p=%.3f$' % ERD1_beta_corr_p,
        ha='right', va='bottom', ma='right', transform=axs[1,0].transAxes, fontsize=7)

slope, intercept = scipy.stats.linregress(musicscore, ERD2_beta)[:2] #regression line
axs[1,1].scatter(musicscore, ERD2_beta, c='k', s=20)
axs[1,1].plot(x, slope*x + intercept, 'k-')
axs[1,1].set_xlabel('musical experience (z-score)')
axs[1,1].set_ylabel('beta ERD 2')
axs[1,1].text(0.95, 0.5,
        r'''Spearman's $R^2=%.2f$''' % ERD2_beta_corr**2 + '\n'
        + r'$p=%.3f$' % ERD2_beta_corr_p,
        ha='right', va='bottom', ma='right', transform=axs[1,1].transAxes, fontsize=7)
fig.suptitle('Correlation of ERD and musicality')
fig.savefig(os.path.join(result_folder, 'motor/motorMusicCorr_ERD.pdf'))
plt.close()
