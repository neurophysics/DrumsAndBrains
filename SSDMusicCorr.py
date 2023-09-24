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
try:
    if sys.argv[3] == 'harmonic':
        harmonic= True
    else:
        harmonic = False
except IndexError:
    harmonic = False


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

# indices = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,17,18,19]# for checking how much the three muscial geniuses carry
# raw_musicscores = raw_musicscores[indices] # for checking how much the three muscial geniuses carry


z_musicscores = (raw_musicscores - np.mean(raw_musicscores,0)
        )/raw_musicscores.std(0)
musicscore = z_musicscores[:,1:].mean(1) # do not include the LQ


#load the SSD results
#SNNR_i = []
#for i in (list(range(1,11,1)) + list(range(12, 22, 1))):
#    with np.load(os.path.join(result_folder, 'S%02d' % i, 'rcsp_tlw.npz'), 'r') as fi:
#        SNNR_i.append(np.mean(fi['rcsp_tlw_ratios'][-1:]))
with np.load(os.path.join(result_folder, 'mtCSP.npz'), 'r') as fi:
    SNNR_i = fi['SNNR_per_subject'][:,:1].mean(-1) #look at first filter
#    SNNR_i = fi['SNNR_per_subject'][:,:].mean(-1) #before 13.01.23 we looked at avg
    subject_filters = fi['subject_filters'] #[20x(32,10)]

if harmonic:
    ### calculate plot with 3.5 Hz data to see if musicality correlates with entrainment of harmonic frequency
    N_subjects = len(subject_filters)+1
    target_cov_harmonic = []
    contrast_cov_harmonic = []
    for i in range(1, N_subjects + 1, 1):
        try:
            with np.load(os.path.join(result_folder, 'S%02d' % i)
                    + '/prepared_FFTSSD.npz', 'r') as fi:
                target_cov_harmonic.append(fi['harmonic_target_cov'])
                contrast_cov_harmonic.append(fi['harmonic_contrast_cov'])
        except:
            print(('Warning: Subject %02d could not be loaded!' %i))

    SNNR_per_subject_harmonic = np.array([
        np.diag((filt.T @ target_now.mean(-1) @ filt) /
                (filt.T @ contrast_now.mean(-1) @ filt))
        for (filt, target_now, contrast_now) in
        zip(subject_filters, target_cov_harmonic, contrast_cov_harmonic)])

    SNNR_i = SNNR_per_subject_harmonic[:,:1].mean(-1) #shape is (20,10) so avg of all subjects but only filter 0 and 1


# convert to dB
SNNR_i = 10*np.log10(SNNR_i)


#SNNR_i=SNNR_i[indices] # for checking how much the three muscial geniuses carry

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
ax.set_ylabel('SNR at polyrhythm freq. (dB), avg. Filter 1-2')
ax.text(0.95, 0.05,
        r'''Spearman's $R^2=%.2f$''' % corr**2 + '\n' + r'$p=%.3f$' % corr_p,
        ha='right', va='bottom', ma='right', transform=ax.transAxes, fontsize=7)
fig.tight_layout(pad=0.3)
plot_name = 'SNNR_exp_filt01avg_woMusicGeniuses'
if harmonic:
    plot_name = plot_name + '_harmonic.pdf'
else:
    plot_name = plot_name + '.pdf'
fig.savefig(os.path.join(result_folder, plot_name))
#fig.savefig(os.path.join(result_folder, 'SNNR_exp.png'))
print('Plot ' + plot_name + ' was saved in '+ result_folder)
