# -*- coding: utf-8 -*-
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
from collections import defaultdict
import csv
import scipy
import scipy.stats

data_folder = sys.argv[1]
subjectnr = int(sys.argv[2]) #here: total number of subjects
result_folder = sys.argv[3]

save_folder = os.path.join(result_folder, 'all%02dsubjects' % subjectnr)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

save_folder = os.path.join(result_folder, 'all%02dsubjects' % subjectnr)

#calculate and read behavioural results into behaviouraldict:
#{'S01': {'snareCue_times': [46.28689342,...], ...}, 'S02': {...} }
behaviouraldict = {}
for i in range(1,subjectnr+1):
    # load the results into a dictionary
    try:
        with np.load(os.path.join(result_folder,'S%02d' % i,
            'behavioural_results.npz'), allow_pickle=True) as behave_file:
            behaviouraldict['S%02d' % i] = dict(behave_file)
    except:
        # run the script to analyze the behavioural data
        subprocess.call("%s read_aif.py %s %d %s" % (
            'python2', data_folder, i, result_folder),
            shell=True)
        with np.load(os.path.join(result_folder,'S%02d' % i,
            'behavioural_results.npz'), allow_pickle=True) as behave_file:
            behaviouraldict['S%02d' % i] = dict(behave_file)

###1. plot performance vs musical background:
#read subject background (LQ and music qualification)
#background is a dict {"subjectnr":[LQ, Quali, Level, years]}
background = {}
with open(os.path.join(data_folder,'additionalSubjectInfo.csv'),'rU') as infile:
    reader = csv.DictReader(infile, fieldnames=None, delimiter=';')
    for row in reader:
        key = row['Subjectnr']
        value = [int(row['LQ']),int(row['MusicQualification']),
            int(row['MusicianshipLevel']),int(row['TrainingYears'])]
        background[key] = value

raw_musicscores = np.array(background.values())
z_musicscores = (raw_musicscores - np.mean(raw_musicscores,0)
        )/raw_musicscores.std(0)
musicscore = z_musicscores[:,1:].mean(1) # do not include the LQ

snare_abs_performance = np.zeros(subjectnr)
wb_abs_performance = np.zeros(subjectnr)
snare_rel_performance = np.zeros(subjectnr)
wb_rel_performance = np.zeros(subjectnr)
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
        musicscore[0:subjectnr],
        np.log(snare_abs_performance))
snare_rel_expregress = scipy.stats.linregress(
        musicscore[0:subjectnr],
        np.log(snare_rel_performance))
wb_abs_expregress = scipy.stats.linregress(
        musicscore[0:subjectnr],
        np.log(wb_abs_performance))
wb_rel_expregress = scipy.stats.linregress(
        musicscore[0:subjectnr],
        np.log(wb_rel_performance))

N_permute = 10000
snare_abs_slope_permute = np.array([
    scipy.stats.linregress(musicscore[0:subjectnr],
        np.log(snare_abs_performance)[
            np.random.randn(subjectnr).argsort()
            ]).slope for _ in xrange(N_permute)])
snare_rel_slope_permute = np.array([
    scipy.stats.linregress(musicscore[0:subjectnr],
        np.log(snare_rel_performance)[
            np.random.randn(subjectnr).argsort()
            ]).slope for _ in xrange(N_permute)])
wb_abs_slope_permute = np.array([
    scipy.stats.linregress(musicscore[0:subjectnr],
        np.log(wb_abs_performance)[
            np.random.randn(subjectnr).argsort()
            ]).slope for _ in xrange(N_permute)])
wb_rel_slope_permute = np.array([
    scipy.stats.linregress(musicscore[0:subjectnr],
        np.log(wb_rel_performance)[
            np.random.randn(subjectnr).argsort()
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
fig.subplots_adjust(wspace=.5)
ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
ax1.scatter(musicscore[0:subjectnr], snare_abs_performance,
    marker = 'o', label='Snare, p=%.3f' % snare_abs_slope_p,
    color='b')
ax1.plot(x, np.exp(snare_abs_expregress[1]+snare_abs_expregress[0]*x),
        'b-')
ax1.scatter(musicscore[0:subjectnr], wb_abs_performance,
    marker = 'o', label='Woodblock, p=%.3f' % wb_abs_slope_p,
    color='r')
ax1.plot(x, np.exp(wb_abs_expregress[1]+wb_abs_expregress[0]*x),
        'r-')
ax2.scatter(musicscore[0:subjectnr], snare_rel_performance,
    marker = 'o', label='Snare, p=%.3f' % snare_rel_slope_p,
    color='b')
ax2.plot(x, np.exp(snare_rel_expregress[1]+snare_rel_expregress[0]*x),
        'b-')
ax2.scatter(musicscore[0:subjectnr], wb_rel_performance,
    marker = 'o', label='Woodblock, p=%.3f' % wb_rel_slope_p,
    color='r')
ax2.plot(x, np.exp(wb_rel_expregress[1]+wb_rel_expregress[0]*x), 'r-')
ax1.legend(prop={'size': 8})
ax2.legend(prop={'size': 8})
ax1.set_title('Absolute deviations')
ax2.set_title('Relative deviations')
ax1.set_xlabel('musical experience (z-score)')
ax2.set_xlabel('musical experience (z-score)')
ax1.set_ylabel('absolute deviation (ms)')
ax2.set_ylabel('relative deviation (ms)')

fig.savefig(os.path.join(save_folder,
    'performance_background_plot.pdf'))
