import argparse
import numpy as np
import scipy
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc('text.latex',
    preamble=r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{xcolor}')
import sys
import os.path
import helper_functions
import meet
from tqdm import tqdm, trange
import SPoC

from scipy.optimize import fmin_l_bfgs_b as _minimize

parser = argparse.ArgumentParser(description='Calculate PCO')
parser.add_argument('result_folder', type=str, default='./Results/',
        help='the folder to store all results', nargs='?')
parser.add_argument('data_folder', type=str, default='./Data/',
        help='the folder to store all data', nargs='?')
args = parser.parse_args()

mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 10

cmap = 'plasma'
color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

s_rate = 1000 # sampling rate of the EEG

N_subjects = 21

# calculate the SSD from all subjects
# read the channel names
channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

# define the frequencies to analyse
# define some constants
QPM = 140 # quarter notes per minute
SPQ = 60./QPM # seconds per quarter note
bar_duration = SPQ*4 # every bar consists of 4 quarter notes

# get the frequencies of the snaredrum (duple) and woodblock (triple) beats
snareFreq = 2./bar_duration
wdBlkFreq = 3./bar_duration

N_channels = len(channames)
# read the oscillatory data from the silence period

# for i in range(1, N_subjects + 1, 1):
#     try:
#         with np.load(os.path.join(args.result_folder, 'S%02d' % i)
#                 + '/prepare_FFTcSPoC.npz', 'r') as fl:
#             try:
#                 snare_listen_trials += fl['snare_listen_trials'].sum(-1)
#                 snare_silence_trials += fl['snare_silence_trials'].sum(-1)
#                 wdBlk_listen_trials += fl['wdBlk_listen_trials'].sum(-1)
#                 wdBlk_silence_trials += fl['wdBlk_silence_trials'].sum(-1)
#                 snare_N += fl['snare_listen_trials'].shape[-1]
#                 wdBlk_N += fl['wdBlk_listen_trials'].shape[-1]
#             except NameError:
#                 snare_listen_trials = fl['snare_listen_trials'].sum(-1)
#                 snare_silence_trials = fl['snare_silence_trials'].sum(-1)
#                 wdBlk_listen_trials = fl['wdBlk_listen_trials'].sum(-1)
#                 wdBlk_silence_trials = fl['wdBlk_silence_trials'].sum(-1)
#                 snare_N = fl['snare_listen_trials'].shape[-1]
#                 wdBlk_N = fl['wdBlk_listen_trials'].shape[-1]
#     except IOError:
#         print('Warning: Subject %02d could not be loaded!' %i)
#
#
# listen_avg = (snare_listen_trials + wdBlk_listen_trials)/(
#         snare_N + wdBlk_N)
# silence_avg = (snare_silence_trials + wdBlk_silence_trials)/(
#         snare_N + wdBlk_N)

all_listen = []
all_silence = []
snare_N = 0
wdBlk_N = 0
subject_list = []
for i in range(1, N_subjects + 1, 1):
    try:
        with np.load(
            os.path.join(args.result_folder, 'S%02d' % i, 'eeg_results.npz'),
            ) as f:
            snareInlier = f['snareInlier']
            wdBlkInlier = f['wdBlkInlier']

        with np.load(
            os.path.join(args.result_folder, 'S%02d' % i, 'behavioural_results.npz'),
                'r', allow_pickle = True, encoding = 'latin1') as f:
            snareCue_nearestClock = f['snareCue_nearestClock']
            snareCue_DevToClock = f['snareCue_DevToClock']
            wdBlkCue_nearestClock = f['wdBlkCue_nearestClock']
            wdBlkCue_DevToClock = f['wdBlkCue_DevToClock']
            bar_duration = f['bar_duration']

        with np.load(
            os.path.join(args.data_folder, 'S%02d' % i, 'clean_data.npz'),
            allow_pickle = True, encoding = 'latin1') as npzfile:
                EEG = npzfile['clean_data']
                artifact_mask = npzfile['artifact_mask']
        if os.path.exists(os.path.join(
            args.data_folder, 'S%02d' % i, 'S{:02d}_eeg_all_files.vmrk'.format(i))):
            marker_fname = os.path.join(
                    args.data_folder, 'S%02d' % i, 'S{:02d}_eeg_all_files.vmrk'.format(i))
        else:
            marker_fname = os.path.join(args.data_folder, 'S%02d' % i, 'S%02d_eeg.vmrk' % i)

        eeg_clocks = helper_functions.getSessionClocks(marker_fname)
        # now, find the sample of each
        eeg_clocks = [c for c in eeg_clocks if len(c) > 100]
        snareCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
                snareCue_nearestClock, snareCue_DevToClock)
        wdBlkCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
                wdBlkCue_nearestClock, wdBlkCue_DevToClock)
        # get each begin of listen and silence period
        snareListenMarker = snareCue_pos - int(4*bar_duration*s_rate)
        wdBlkListenMarker = wdBlkCue_pos - int(4*bar_duration*s_rate)
        snareSilenceMarker = snareCue_pos - int(bar_duration*s_rate)
        wdBlkSilenceMarker = wdBlkCue_pos - int(bar_duration*s_rate)
        # rereference to the average EEG amplitude
        EEG -= EEG.mean(0)
        # epoch to listening
        all_win = [0, int(4*bar_duration*s_rate)]
        # !! takes snare and wdblk together so not weighted by number of trials
        all_l = meet.epochEEG(EEG,
                np.r_[snareListenMarker[snareInlier],
                        wdBlkListenMarker[wdBlkInlier]],
                all_win)
        all_listen.append(all_l.mean(-1)) #trial avg
        all_s = meet.epochEEG(EEG,
                np.r_[snareSilenceMarker[snareInlier],
                        wdBlkSilenceMarker[wdBlkInlier]],
                all_win)
        all_silence.append(all_s.mean(-1))
        wdBlk_N += snareInlier.sum()
        snare_N += wdBlkInlier.sum()
        subject_list.append(i)
    except IOError:
        print('Warning: Subject %02d could not be loaded!' %i)

def plot_evokedComponents(eeg_data):
        t = np.arange(eeg_data.shape[-1])/float(s_rate)
        t1_mask = np.all([t>=3.5, t<=3.7], 0)
        t2_mask = np.all([t>=4.1, t<=4.3], 0)
        t3_mask = np.all([t>=4.4, t<=4.6], 0)
        plot_avg = eeg_data[np.array(channames)=='CZ'][0]

        t1_idx = np.arange(len(t))[t1_mask][
                np.argmax(plot_avg[t1_mask])]
        t2_idx = np.arange(len(t))[t2_mask][
                np.argmax(plot_avg[t2_mask])]
        t3_idx = np.arange(len(t))[t3_mask][
                np.argmax(plot_avg[t3_mask])]

        X1, Y1, Z1 = meet.sphere.potMap(
                chancoords, eeg_data[:,t1_idx])
        X2, Y2, Z2 = meet.sphere.potMap(
                chancoords, eeg_data[:,t2_idx])
        X3, Y3, Z3 = meet.sphere.potMap(
                chancoords, eeg_data[:,t3_idx])
        vmax = np.max([np.abs(Z1), np.abs(Z2), np.abs(Z3)])

        fig = plt.figure(figsize=(7.48031, 3))
        gs = mpl.gridspec.GridSpec(2,1, height_ratios=[1.75,1])
        gs1 = mpl.gridspec.GridSpecFromSubplotSpec(2,1, gs[1,:], hspace=0,
                height_ratios=[2,1])
        ax1 = fig.add_subplot(gs1[0,:])
        ax1.tick_params(right=False, top=False, labelright=False, labeltop=False,
                bottom=True, labelbottom=False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.plot(t, plot_avg, 'k-', label='Cz')

        ax1.axvline(3*bar_duration, c='k', lw=2)
        ax1.axvline(1*bar_duration, c='k', lw=0.5)
        ax1.axvline(2*bar_duration, c='k', lw=0.5)

        #ax1.axhline(0, c='k', lw=0.5)


        rhythm_ax = fig.add_subplot(gs1[1,:], sharex=ax1, frame_on=True)
        rhythm_ax.tick_params(**blind_ax)
        rhythm_ax.tick_params(bottom=True, labelbottom=True)
        rhythm_trans = mpl.transforms.blended_transform_factory(
        rhythm_ax.transData, rhythm_ax.transAxes)
        ax1_trans = mpl.transforms.blended_transform_factory(
        ax1.transData, ax1.transAxes)
        [rhythm_ax.text(t_now, 0, r'$\blacksquare$',
                ha='left', va='bottom', ma='center', transform=rhythm_trans,
                color='b', fontsize=13)
                for t_now in np.arange(0,3*bar_duration, bar_duration/2.)]
        [rhythm_ax.text(t_now, 1, r'$\blacksquare$',
                ha='left', va='top', ma='center', transform=rhythm_trans,
                color='r', fontsize=13)
                for t_now in np.arange(0,3*bar_duration, bar_duration/3.)]

        t11 = rhythm_ax.text(1.5*bar_duration, 1.0, r'\textbf{musical stimulus}',
                ha='center', va='top', ma='center', transform=ax1_trans,
                color='k', fontsize=10)
        t12 = rhythm_ax.text(3.5*bar_duration, 1.0, r'\textbf{silence}',
                ha='center', va='top', ma='center', transform=ax1_trans,
                color='k', fontsize=10)

        rhythm_ax.spines['left'].set_visible(True)
        rhythm_ax.spines['right'].set_visible(False)
        rhythm_ax.spines['top'].set_visible(False)
        rhythm_ax.spines['bottom'].set_visible(True)
        rhythm_ax.axvline(3*bar_duration, c='k', lw=2)
        rhythm_ax.axvline(1*bar_duration, c='k', lw=0.5)
        rhythm_ax.axvline(2*bar_duration, c='k', lw=0.5)

        ax1.set_xlim(0, 4*bar_duration)

        ax1.set_ylabel(r'ampl. ($\mathrm{\mu V}$)')
        rhythm_ax.set_xlabel('time (s)')

        #ax1.text(0.5*bar_duration, 1.0, r'channel: Cz', ha='center', va='top',
        #        transform=ax1_trans, fontsize=10, clip_on=False)

        gs2 = mpl.gridspec.GridSpecFromSubplotSpec(2,3, gs[0,:], hspace=0.6,
                height_ratios=[0.1,1])
        pax1 = fig.add_subplot(gs2[1,0], frame_on=False)
        pax1.tick_params(**blind_ax)
        pc1 = pax1.pcolormesh(X1, Y1, Z1, rasterized=True, vmin=-vmax, vmax=vmax,
                cmap='coolwarm')
        pax1.contour(X1, Y1, Z1, levels=[0], colors='w')
        meet.sphere.addHead(pax1)
        pax1.text(0.5, 0, r'$t=%.2f$ s' % t[t1_idx], ha='center', va='bottom',
                fontsize=10, transform=pax1.transAxes)

        pax2 = fig.add_subplot(gs2[1,1], frame_on=False, sharex=pax1, sharey=pax1)
        pax2.tick_params(**blind_ax)
        pc2 = pax2.pcolormesh(X2, Y2, Z2, rasterized=True, vmin=-vmax, vmax=vmax,
                cmap='coolwarm')
        pax2.contour(X2, Y2, Z2, levels=[0], colors='w')
        meet.sphere.addHead(pax2)
        pax2.text(0.5, 0, r'$t=%.2f$ s' % t[t2_idx], ha='center', va='bottom',
                fontsize=10, transform=pax2.transAxes)

        pax3 = fig.add_subplot(gs2[1,2], frame_on=False, sharex=pax1, sharey=pax1)
        pax3.tick_params(**blind_ax)
        pc3 = pax3.pcolormesh(X3, Y3, Z3, rasterized=True, vmin=-vmax, vmax=vmax,
                cmap='coolwarm')
        pax3.contour(X3, Y3, Z3, levels=[0], colors='w')
        meet.sphere.addHead(pax3)
        pax3.text(0.5, 0, r'$t=%.2f$ s' % t[t3_idx], ha='center', va='bottom',
                fontsize=10, transform=pax3.transAxes)

        pax1.set_xlim([-2.3, 2.3])
        pax1.set_ylim([-1.6, 1.3])

        cb_ax = fig.add_subplot(gs2[0,:])
        cbar = plt.colorbar(pc1, cax=cb_ax, label='amplitude ($\mathrm{\mu V}$)',
                orientation='horizontal')
        #cbar.ax.set_xticklabels(['$-$', '$0$', '$+$'])
        cbar.ax.axvline(0., c='w')#, transform=cbar.ax.transData) #0 in repect to absolute value not axis units

        fig.tight_layout(pad=0.3, h_pad=0.5)
        fig.canvas.draw()
        ax1.autoscale(False)
        ax1_min = ax1.get_ylim()[1]

        ax1.plot([t[t1_idx], t[t1_idx]], [plot_avg[t1_idx], ax1_min],
                'k-', lw=0.5)
        ax1.plot([
        t[t1_idx],
        ax1.transData.inverted().transform(
                pax1.transAxes.transform([0.5,0]))[0]],
        [ax1_min,
        ax1.transData.inverted().transform(
                pax1.transAxes.transform([0.5,0]))[1]],
        'k-', lw=0.5, clip_on=False,
        transform=ax1.transData, alpha=1.0)

        ax1.plot([t[t2_idx], t[t2_idx]], [plot_avg[t2_idx], ax1_min],
                'r-', lw=0.5)
        ax1.plot([
        t[t2_idx],
        ax1.transData.inverted().transform(
                pax2.transAxes.transform([0.5,0]))[0]],
        [ax1_min,
        ax1.transData.inverted().transform(
                pax2.transAxes.transform([0.5,0]))[1]],
        'r-', lw=0.5, clip_on=False,
        transform=ax1.transData, alpha=1.0)

        ax1.plot([t[t3_idx], t[t3_idx]], [plot_avg[t3_idx], ax1_min],
                'b-', lw=0.5)
        ax1.plot([
        t[t3_idx],
        ax1.transData.inverted().transform(
                pax3.transAxes.transform([0.5,0]))[0]],
        [ax1_min,
        ax1.transData.inverted().transform(
                pax3.transAxes.transform([0.5,0]))[1]],
        'b-', lw=0.5, clip_on=False,
        transform=ax1.transData, alpha=1.0)

        return fig,plt

1/0
##### check for each subject if it shows entrainment
for i in subject_list:
        eeg_subj = np.hstack([all_listen[i-1], all_silence[i-1]])
        fig,plt = plot_evokedComponents(eeg_subj)
        fig.savefig(os.path.join(args.result_folder, 'S%02d' % i,
    'EvokedComponent.pdf'))
plt.close('all')

##### store for average subject #####
# get the spatial patterns
listen_avg = sum(all_listen) / len(all_listen)
silence_avg = sum(all_silence) / len(all_silence)
eeg_avg = np.hstack([listen_avg, silence_avg])
plot_evokedComponents(eeg_avg)
fig.savefig(os.path.join(args.result_folder,
    'EvokedComponent.pdf'))
fig.savefig(os.path.join(args.result_folder,
    'EvokedComponent.png'))
