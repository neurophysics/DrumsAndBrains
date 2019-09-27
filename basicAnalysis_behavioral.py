# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import csv
import scipy
import scipy.stats

import read_aif

data_folder = sys.argv[1]
subjectnr = int(sys.argv[2]) #here: total number of subjects
result_folder = sys.argv[3]

save_folder = os.path.join(result_folder, 'all%02dsubjects' % subjectnr)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

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
        key = "S%02d" % int(row['Subjectnr']) #same format as behaviouraldict
        value = [int(row['LQ']),int(row['MusicQualification']),
            int(row['MusicianshipLevel']),int(row['TrainingYears'])]
        background[key] = value

raw_musicscores = np.array([v for k,v in sorted(background.items())])
z_musicscores = (raw_musicscores - np.mean(raw_musicscores,0)
        )/raw_musicscores.std(0)
musicscore = z_musicscores[:,1:].mean(1) # do not include the LQ

snare_abs_performance = np.zeros(subjectnr)
wb_abs_performance = np.zeros(subjectnr)
snare_rel_performance = np.zeros(subjectnr)
wb_rel_performance = np.zeros(subjectnr)
for k,v in sorted(behaviouraldict.items()):
    i = int(k[1:])-1 #'S01'=> entry 0
    snaredev = v['snare_deviation']
    snaredev = snaredev[np.isfinite(snaredev)]
    wbdev = v['wdBlk_deviation']
    wbdev = wbdev[np.isfinite(wbdev)]
    snare_abs_performance[i] = scipy.stats.trim1(np.abs(snaredev), 0.0
            ).mean()
    wb_abs_performance[i] = scipy.stats.trim1(np.abs(wbdev), 0.0).mean()
    snare_rel_performance[i] = scipy.stats.trim1(np.abs(
        snaredev - scipy.stats.trim_mean(snaredev, 0.0)), 0.0).mean()
    wb_rel_performance[i] = scipy.stats.trim1(np.abs(
        wbdev - scipy.stats.trim_mean(wbdev, 0.0)), 0.0).mean()

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

snare_abs_slope_p = (np.sum(snare_abs_slope_permute <= snare_abs_expregress.slope) + 1)/float(N_permute + 1)
snare_rel_slope_p = (np.sum(snare_rel_slope_permute <= snare_rel_expregress.slope) + 1)/float(N_permute + 1)
wb_abs_slope_p = (np.sum(wb_abs_slope_permute <= wb_abs_expregress.slope) + 1)/float(N_permute + 1)
wb_rel_slope_p = (np.sum(wb_rel_slope_permute <= wb_rel_expregress.slope) + 1)/float(N_permute + 1)

x = np.linspace(-1.5, 3, 100)

# plot musicscore vs behaviour
fig = plt.figure(figsize=(5.51, 3))
ax1 = fig.add_subplot(121)
fig.subplots_adjust(wspace=.5)
ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
ax1.scatter(musicscore[0:subjectnr], snare_abs_performance,
    marker = 'o', label='duple beat, p=%.4f' % snare_abs_slope_p,
    color='b')
ax1.plot(x, np.exp(snare_abs_expregress[1]+snare_abs_expregress[0]*x),
        'b-')
ax1.scatter(musicscore[0:subjectnr], wb_abs_performance,
    marker = 'o', label='triple beat, p=%.4f' % wb_abs_slope_p,
    color='r')
ax1.plot(x, np.exp(wb_abs_expregress[1]+wb_abs_expregress[0]*x),
        'r-')
ax2.scatter(musicscore[0:subjectnr], snare_rel_performance,
    marker = 'o', label='duple beat, p=%.4f' % snare_rel_slope_p,
    color='b')
ax2.plot(x, np.exp(snare_rel_expregress[1]+snare_rel_expregress[0]*x),
        'b-')
ax2.scatter(musicscore[0:subjectnr], wb_rel_performance,
    marker = 'o', label='triple beat, p=%.4f' % wb_rel_slope_p,
    color='r')
ax2.plot(x, np.exp(wb_rel_expregress[1]+wb_rel_expregress[0]*x), 'r-')
ax1.legend(prop={'size': 8})
ax2.legend(prop={'size': 8})
ax1.set_title('absolute deviations')
ax2.set_title('relative deviations')
ax1.set_xlabel('musical experience (z-score)')
ax2.set_xlabel('musical experience (z-score)')
ax1.set_ylabel('absolute deviation (ms)')
ax2.set_ylabel('relative deviation (ms)')

fig.tight_layout(pad=0.3)

fig.savefig(os.path.join(save_folder,
    'performance_background_plot.pdf'))
fig.savefig(os.path.join(save_folder,
    'performance_background_plot.png'))

###2. plot performance
## NeuralEntrl_Ssxxresponse (read_aif.py) for all subjects
# calculate the latency between cue and response
#constants from read_aif.py:
QPM = 140 # quarter notes per minute
SPQ = 60./QPM # seconds per quarter note
bar_duration = SPQ*4 # every bar consists of 4 quarter notes

allSnare_latencies = np.zeros((subjectnr,75))
allWdBlk_latencies = np.zeros((subjectnr,75))
for k,v in sorted(behaviouraldict.items()):
    i = int(k[1:]) #'S01' => 1
    drumIn_fname =  os.path.join(data_folder,
            'S%02d/NeuralEntrStimLive8_Ss%02d_DrumIn.aif' % (i,i))
    drum_times = read_aif.get_ClickTime(drumIn_fname)
    snareCue_times = v['snareCue_times']
    wdBlkCue_times = v['wdBlkCue_times']
    snare_latencies = read_aif.get_ResponseLatency(snareCue_times, drum_times,
        2*bar_duration)
    wdBlk_latencies = read_aif.get_ResponseLatency(wdBlkCue_times, drum_times,
        2*bar_duration)
    allSnare_latencies[i-1] = snare_latencies
    allWdBlk_latencies[i-1] = wdBlk_latencies

# plot the results
hist_bins = np.arange(0.5, 1.5 + 0.025, 0.025)
fig = plt.figure(figsize=(5.51, 3))
ax1 = fig.add_subplot(111)
ax1.hist(np.concatenate(allSnare_latencies), bins=hist_bins, color='b', label='duple beat trials',
        edgecolor='w', alpha=0.6)
ax1.axvline(bar_duration/2., color='b', label='correct duple lat.')
ax1.hist(np.concatenate(allWdBlk_latencies), bins=hist_bins, color='r', label='triple beat trials',
        edgecolor='w', alpha=0.6)
ax1.axvline(2*bar_duration/3., color='r', label='correct triple lat.')
ax1.set_xlabel('latency to cue (s)')
ax1.set_ylabel('number of trials')
ax1.legend(loc='upper left')
#ax1.set_ylim([0,30])
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(save_folder,
    'NeuralEntrl_response.png'))
fig.savefig(os.path.join(save_folder,
    'NeuralEntrl_response.pdf'))
