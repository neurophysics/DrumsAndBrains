import argparse
import mne
import numpy as np
import scipy
import sys
import os.path
import matplotlib.pyplot as plt
import meet
import helper_functions

# define some constants
QPM = 140 # quarter notes per minute
SPQ = 60./QPM # seconds per quarter note
bar_duration = SPQ*4 # every bar consists of 4 quarter notes
s_rate = 1000 # sampling rate of the EEG (1000 samples per sec)

parser = argparse.ArgumentParser(
        description='Calculate source from cSpoC pattern')
parser.add_argument('result_folder', type=str, default='./Results/',
        help='the folder to store all results', nargs='?')
args = parser.parse_args()

# read cSpoC pattern
try:
    with np.load(os.path.join(args.result_folder,
            'calcFFTcSpoCv2_spatial_pattern.npz'), 'r') as f:
        wdBlk_pattern = f['wdBlk_pattern'].reshape(32,6) #shape (6,32) polarities for all
        snare_pattern = f['snare_pattern'].reshape(32,6) # channels and per session
except:
    print('Warning: Could not load cSpoC pattern')

# read channel names and positions
channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
n_channels = wdBlk_pattern.shape[0]

montage = mne.channels.make_dig_montage(ch_pos=dict(zip(channames, chancoords)),
                                        coord_frame='head')
info = mne.create_info(ch_names=channames, montage=montage,
        sfreq=s_rate, ch_types=np.repeat('eeg',n_channels))

# both have the same epochs: one value per session
tmin = -0.0005  # start of each epoch: half a sample before (1 sample is 0.001s)
tmax = 0.0005  # end of each epoch: half a sample after

####### snare #######
events_snare = np.zeros((6,3),dtype=int) #every point is important
events_snare[:,0] = range(6)

raw_snare = mne.io.RawArray(snare_pattern.reshape(32,6), info)
event_id = {"snare": 0}
epochs_snare = mne.Epochs(raw_snare, events_snare, event_id, tmin, tmax,
        baseline=None, preload=True)

evoked_snare = epochs_snare.average()
# fucntion does bs:
# shape = (32,2) but we want (32,6)
# snare_pattern[0] =array([ 0.4977339 ,  0.60976539,  0.27984362, -0.73780303, -0.27334276,0.13660771])
# evoked_snare.data[0] = array([0.07523942, 0.00301419]), = [mean(first five), mean(last five)]
# => need evoked data structure but overwrite what function did:
evoked_snare.data = snare_pattern #need shape (32,6)
evoked_snare.times = np.linspace(0, 0.005, 6)

fig = evoked_snare.plot_topomap(show=False,times=evoked_snare.times)
# todo: change suptitle of plot_topomap to session 1-6 isntead of 0.001s ...
fig.suptitle('Bars 1-3 (Listening), snare trials\n')
plt.show()

####### wdBlk #######
events_wdBlk = np.zeros((6,3),dtype=int) #every point is important
events_wdBlk[:,0] = range(6)
events_wdBlk[:,2] = 1 #wdBlk has event ID 1

raw_wdBlk = mne.io.RawArray(wdBlk_pattern.reshape(32,6), info)
event_id = {"wdBlk": 1}
epochs_wdBlk = mne.Epochs(raw_wdBlk, events_wdBlk, event_id, tmin, tmax,
        baseline=None, proj=True)

evoked_wdBlk = epochs_wdBlk.average()
# need evoked data structure but overwrite what function did:
evoked_wdBlk.data = snare_pattern #need shape (32,6)
evoked_wdBlk.times = np.linspace(0, 0.005, 6)

fig = evoked_wdBlk.plot_topomap(show=False, times=evoked_wdBlk.times)
fig.suptitle('Bars 1-3 (Listening), wdBlk trials\n')
plt.show()
