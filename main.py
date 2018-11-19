# -*- coding: utf-8 -*-
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from collections import defaultdict
from csv import DictReader
import scipy
import scipy.stats

home_folder = os.path.expanduser('~')

# make standard folders depending on the current user
if 'hansmitdampf' in home_folder:
    data_default = os.path.join(home_folder,
            'Neuro/Data/2018-Polyrhythm')
    result_default = os.path.join(home_folder,
            'Neuro/Python-Reps/Scripts/2018-Polyrhythm/Results')
    python_exec = 'python2'
elif 'Carola' in home_folder:
    data_default = os.path.join(home_folder,
            'Documents/Arbeit/Charite/DrumsAndBrains/Data')
    result_default = os.path.join(home_folder,
            'Documents/Arbeit/Charite/DrumsAndBrains/Results')
    python_exec = 'python'
# add more users by inserting elif lines
else:
    data_default = home_folder
    result_default = home_folder
    python_exec = 'python'

# parse all the arguments
parser= argparse.ArgumentParser(
        description='Package for combined classifier - welcome!')
parser.add_argument('-p', '--python', default=python_exec,
        help='python executable')
parser.add_argument('-d', '--data_folder', default=data_default,
        help='Path to data folder.')
parser.add_argument('-r', '--result_folder', default=result_default,
        help='Path to result folder.')
parser.add_argument('-n', '--nrofsubjects', default=10, type=int,
        help='Number of subjects.')
args=parser.parse_args()

if not os.path.exists(args.result_folder):
    os.mkdir(args.result_folder)

#calculate and read behavioural results into behaviouraldict:
#{'S01': {'snareCue_times': [46.28689342,...], ...}, 'S02': {...} }
behaviouraldict = {}
for i in range(1,args.nrofsubjects+1):
    # load the results into a dictionary
    try:
        with np.load(os.path.join(args.result_folder,'S%02d' % i,
            'behavioural_results.npz')) as behave_file:
            behaviouraldict['S%02d' % i] = dict(behave_file)
    except:
        # run the script to analyze the behavioural data
        subprocess.call("%s read_aif.py %s %d %s" % (
            args.python, args.data_folder, i, args.result_folder),
            shell=True)
        with np.load(os.path.join(args.result_folder,'S%02d' % i,
            'behavioural_results.npz')) as behave_file:
            behaviouraldict['S%02d' % i] = dict(behave_file)

"""
####1. Plot self assessment vs real performance:
#read self assessment:


sadict = defaultdict(list) #(s01:[(1,5,8),(2,6,7),..], )
for i in range(1, args.nrofsubjects+1):
    subject_sa = np.loadtxt(os.path.join(args.data,'S%02d' % i, 'S%02d_self-assessment.txt' % i), delimiter=' ', usecols=[1], dtype=int)
    session = subject_sa[::3]
    assert np.all(session == np.arange(1, len(session) + 1, 1))
    performance = subject_sa[1::3]
    vigilance = subject_sa[2::3]
    sadict['S%02d' %i] = zip(session, performance, vigilance)

#plot list of snare_precision, woodblk_prescision, self assessment vs
#trials (x-axis = trial 1-150, y is deviation and self assessment)
#print behaviouraldict['s1'].keys()
fig = plt.figure()
ax = fig.add_subplot(111)
trials = range(1,151)

wdBlkCue_nearestClock = []
for elem in behaviouraldict['s1']['wdBlkCue_nearestClock']:
    wdBlkCue_nearestClock = wdBlkCue_nearestClock + list(elem)
wdBlk_deviation = behaviouraldict['s1']['wdBlk_deviation']

snareCue_nearestClock = []
for elem in behaviouraldict['s1']['snareCue_nearestClock']:
    snareCue_nearestClock = snareCue_nearestClock + list(elem)
snare_deviation = behaviouraldict['s1']['snare_deviation']
len(snareCue_nearestClock), len(snare_deviation)
'''
wdBlkzip = []
snarezip = []
posw = 0
poss = 0
for i in range(0,6): #for every of the 6 parts
    wdBlkzip.append(zip(wdBlkCue_nearestClock[i],
        wdBlk_deviation[posw:posw+len(wdBlkCue_nearestClock[i])]))
    posw = posw + len(wdBlkCue_nearestClock[i])
    snarezip.append(zip(snareCue_nearestClock[i],
        snare_deviation[posw:posw+len(snareCue_nearestClock[i])]))
    posw = posw + len(snareCue_nearestClock[i])
# gives us: two zips [([1,2,3,4,5,6],[1,2,3,4,5,6]), (Part 2),...]
#total_deviation = [i for k,i in sorted(zipSnare+zipWdBlk)]
#make 5 vertikel lines for each of the 6 parts

'''
for i in range(0,5):
    x = min(behaviouraldict['s1']['snareCue_nearestClock'][i][0],
        behaviouraldict['s1']['wdBlkCue_nearestClock'][i][0])
    plt.axvline(x=25*(i+1)) #is it really 25 trials/session? - rather divide into 6 subplots?
total_deviation = [a for i,a in sorted(
    zip(snareCue_nearestClock,snare_deviation)+
    zip(wdBlkCue_nearestClock,wdBlk_deviation)
    )]
colorzip = sorted(zip(snare_deviation,['b']*75) + zip(wdBlk_deviation,['r']*75))
cd = {}
for k,v in colorzip:
    cd[k] = v
plt.axhline(y=0)
#ax.plot(trials, total_deviation, color = [e for i,e in colorzip], linewidth = 0.1, marker='o', label='')
#ax.plot(trials, wdBlk_deviation, snare_deviationlinewidth = 0.1, marker='o', label='')
#ax.plot(trials, selfassessment, linewidth = 1, marker='o', label='')
plt.xlabel('Trial number')
legend = ax.legend(prop={'size': 8})
plt.savefig(os.path.join(args.result_folder, 'performance_plot.pdf'))

"""
###2. plot performance and musical background:
#read subject background (LQ and music qualification)
#background is a dict {"subjectnr":[LQ, Quali, Level, years]}
filename = os.path.join(args.data_folder,'additionalSubjectInfo.csv')
background = {}
with open(filename) as infile:
    reader = DictReader(infile, fieldnames=None, delimiter=';')
    for row in reader:
        key = row['Subjectnr']
        value = [int(row['LQ']),int(row['MusicQualification']),
            int(row['MusicianshipLevel']),int(row['TrainingYears'])]
        background[key] = value

raw_musicscores = np.array(background.values())
z_musicscores = (raw_musicscores - np.mean(raw_musicscores,0)
        )/raw_musicscores.std(0)
musicscore = z_musicscores[:,1:].mean(1) # do not include the LQ

snare_abs_performance = np.zeros(args.nrofsubjects)
wb_abs_performance = np.zeros(args.nrofsubjects)
snare_rel_performance = np.zeros(args.nrofsubjects)
wb_rel_performance = np.zeros(args.nrofsubjects)
for i,v in enumerate(behaviouraldict.values()):
    snaredev = v['snare_deviation']
    snaredev = snaredev[np.isfinite(snaredev)]
    wbdev = v['wdBlk_deviation']
    wbdev = wbdev[np.isfinite(wbdev)]
    snare_abs_performance[i] = scipy.stats.trim1(np.abs(snaredev), 0.25
            ).mean()
    wb_abs_performance[i] = scipy.stats.trim1(np.abs(wbdev), 0.25).mean()
    snare_rel_performance[i] = scipy.stats.trim1(np.abs(
        snaredev - scipy.stats.trim_mean(snaredev, 0.125)), 0.25).mean()
    wb_rel_performance[i] = scipy.stats.trim1(np.abs(
        wbdev - scipy.stats.trim_mean(wbdev, 0.125)), 0.25).mean()

snare_abs_expregress = scipy.stats.linregress(
        musicscore[0:args.nrofsubjects],
        np.log(snare_abs_performance))
snare_rel_expregress = scipy.stats.linregress(
        musicscore[0:args.nrofsubjects],
        np.log(snare_rel_performance))
wb_abs_expregress = scipy.stats.linregress(
        musicscore[0:args.nrofsubjects],
        np.log(wb_abs_performance))
wb_rel_expregress = scipy.stats.linregress(
        musicscore[0:args.nrofsubjects],
        np.log(wb_rel_performance))

N_permute = 10000
snare_abs_slope_permute = np.array([
    scipy.stats.linregress(musicscore[0:args.nrofsubjects],
        np.log(snare_abs_performance)[
            np.random.randn(args.nrofsubjects).argsort()
            ]).slope for _ in xrange(N_permute)])
snare_rel_slope_permute = np.array([
    scipy.stats.linregress(musicscore[0:args.nrofsubjects],
        np.log(snare_rel_performance)[
            np.random.randn(args.nrofsubjects).argsort()
            ]).slope for _ in xrange(N_permute)])
wb_abs_slope_permute = np.array([
    scipy.stats.linregress(musicscore[0:args.nrofsubjects],
        np.log(wb_abs_performance)[
            np.random.randn(args.nrofsubjects).argsort()
            ]).slope for _ in xrange(N_permute)])
wb_rel_slope_permute = np.array([
    scipy.stats.linregress(musicscore[0:args.nrofsubjects],
        np.log(wb_rel_performance)[
            np.random.randn(args.nrofsubjects).argsort()
            ]).slope for _ in xrange(N_permute)])

snare_abs_slope_p = np.mean(
        snare_abs_slope_permute <= snare_abs_expregress.slope)
snare_rel_slope_p = np.mean(
        snare_rel_slope_permute <= snare_rel_expregress.slope)
wb_abs_slope_p = np.mean(
        wb_abs_slope_permute <= wb_abs_expregress.slope)
wb_rel_slope_p = np.mean(
        wb_rel_slope_permute <= wb_rel_expregress.slope)

x = np.linspace(-2, 2, 100)

# plot musicscore vs behaviour
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
ax1.scatter(musicscore[0:args.nrofsubjects], snare_abs_performance,
    marker = 'o', label='Snare, p=%.3f' % snare_abs_slope_p,
    color='b')
ax1.plot(x, np.exp(snare_abs_expregress[1]+snare_abs_expregress[0]*x),
        'b-')
ax1.scatter(musicscore[0:args.nrofsubjects], wb_abs_performance,
    marker = 'o', label='Woodblock, p=%.3f' % wb_abs_slope_p,
    color='r')
ax1.plot(x, np.exp(wb_abs_expregress[1]+wb_abs_expregress[0]*x),
        'r-')
ax2.scatter(musicscore[0:args.nrofsubjects], snare_rel_performance,
    marker = 'o', label='Snare, p=%.3f' % snare_rel_slope_p,
    color='b')
ax2.plot(x, np.exp(snare_rel_expregress[1]+snare_rel_expregress[0]*x),
        'b-')
ax2.scatter(musicscore[0:args.nrofsubjects], wb_rel_performance,
    marker = 'o', label='Woodblock, p=%.3f' % wb_rel_slope_p,
    color='r')
ax2.plot(x, np.exp(wb_rel_expregress[1]+wb_rel_expregress[0]*x), 'r-')
ax1.legend(prop={'size': 8})
ax2.legend(prop={'size': 8})
ax1.set_title('Absolute deviatios')
ax2.set_title('Relative deviations')
ax1.set_xlabel('musical experience (z-score)')
ax2.set_xlabel('musical experience (z-score)')
ax1.set_ylabel('absolute deviation (ms)')
ax2.set_ylabel('relative deviation (ms)')

fig.savefig(os.path.join(args.result_folder,
    'performance_background_plot.pdf'))
