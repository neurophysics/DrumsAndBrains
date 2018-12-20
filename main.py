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
            'Documents/Uni/Berufspraktikum/data')
    result_default = os.path.join(home_folder,
            'Documents/Uni/Berufspraktikum/results')
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

###1. plot performance vs musical background:
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
