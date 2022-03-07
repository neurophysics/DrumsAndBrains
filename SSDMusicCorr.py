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

mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 10

cmap = 'plasma'
color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

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

#load the SSD results
#SNNR_i = []
#for i in (list(range(1,11,1)) + list(range(12, 22, 1))):
#    with np.load(os.path.join(result_folder, 'S%02d' % i, 'rcsp_tlw.npz'), 'r') as fi:
#        SNNR_i.append(np.mean(fi['rcsp_tlw_ratios'][-1:]))
with np.load(os.path.join(result_folder, 'mtCSP.npz'), 'r') as fi:
    SNNR_i = fi['SNNR_per_subject'][:,:].mean(-1)

# convert to dB
SNNR_i = 10*np.log10(SNNR_i)

# calculate bootstrap coefficient
N_bootstrap = 50000
corr = np.corrcoef(musicscore.argsort().argsort(),
    SNNR_i.argsort().argsort())[0,1] #spearman rank reduces outlier influence
corr_boot = np.array([
    np.corrcoef(musicscore.argsort().argsort(),
        np.random.choice(SNNR_i.argsort().argsort(), size=len(SNNR_i), replace=False))[0,1]
    for _ in range(N_bootstrap)])
corr_p = (np.sum(corr_boot>=corr) + 1)/(N_bootstrap + 1)

# calculate a regression line
slope, intercept = scipy.stats.linregress(musicscore, SNNR_i)[:2]
x= np.arange(-1.5,3,1)

# plot the result
fig = plt.figure(figsize=(3.54, 2.5))
ax = fig.add_subplot(111)
ax.scatter(musicscore, SNNR_i, c='k', s=20)
ax.plot(x, slope*x + intercept, 'k-')
ax.set_xlabel('musical experience (z-score)')
ax.set_ylabel('SNR at polyrhythm frequecies (dB)')
ax.text(0.95, 0.05,
        r'''Spearman's $R^2=%.2f$''' % corr**2 + '\n' + r'$p=%.3f$' % corr_p,
        ha='right', va='bottom', ma='right', transform=ax.transAxes, fontsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(result_folder, 'SNNR_exp.pdf'))
fig.savefig(os.path.join(result_folder, 'SNNR_exp.png'))
