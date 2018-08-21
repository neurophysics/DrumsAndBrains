# -*- coding: utf-8 -*-
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from collections import defaultdict
from csv import DictReader


def main():
    parser=argparse.ArgumentParser( description='Package for combined classifier - welcome!')
    parser.add_argument('-d', '--data', default='/Users/Carola/Documents/Uni/Berufspraktikum/data/',
    help='Path to data folder.')
    parser.add_argument('-n', '--nrofsubjects', default=1, type=int,
    help='Number of subjects.')
    args=parser.parse_args()

    #calculate and read behavioural results into behaviouraldict:
    #{'s1': {'snareCue_times': [46.28689342,...], ...}, 's2': {...} }
    behaviouraldict = defaultdict(dict)
    for i in range(1,args.nrofsubjects+1):
        subprocess.call("python read_aif.py "+str(i), shell = True)
        results = np.load(os.path.join('results','Ss0'+str(i),'behavioural_results.npz'))
        d = defaultdict(list)
        for r in results.files:
            d[r] = results[r]
        behaviouraldict['s'+str(i)] = d

####1. Plot self assessment vs real performance:

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
    plt.savefig(os.path.join(os.getcwd(),'data',
        'performance_plot.pdf'))


###2. plot performance and musical background:
    #read subject background (LQ and music qualification)
    #background is a dict {"subjectnr":[LQ, Quali, Level, years]}
    filename = os.path.join(args.data,'additionalSubjectInfo.csv')
    background = defaultdict(list)
    with open(filename) as infile:
        reader = DictReader(infile, fieldnames=None, delimiter=';')
        for row in reader:
            key = row['Subjectnr']
            value = [row['LQ'],row['MusicQualification'],
                row['MusicianshipLevel'],row['TrainingYears']]
            background[key] = value

    musicscore = []
    for s,v in background.items():
        score = int(v[1])+int(v[2])+int(v[3]) #musicscore is adding up values for now
        musicscore.append(score)

    performance_sd = []
    performance_mean = []
    for s,v in behaviouraldict.items():
        snaredev = [e for e in v['snare_deviation']]
        wbdev = [e for e in v['wdBlk_deviation']]
        performance = np.abs(snaredev)+np.abs(wbdev)
        performance = [p for p in performance if str(p) != 'nan']
        performance_sd.append(np.std(performance))
        performance_mean.append(sum(performance)/len(performance))

    #print behaviouraldict
    #print performance_mean, performance_sd, musicscore
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    # may include yerr=performance_sd
    ax2.errorbar(musicscore[0:args.nrofsubjects], performance_mean,
        marker = 'o', label='Performance vs musical experience')
    legend = ax.legend(prop={'size': 8})
    r = np.corrcoef(musicscore[0:args.nrofsubjects], performance_mean)
    plt.title('n = '+str(args.nrofsubjects)+', r = '+str(r[0][1]))
    plt.savefig(os.path.join(os.getcwd(),'results',
        'performance_background_plot.pdf'))


    return

if __name__ == '__main__':
  main()
