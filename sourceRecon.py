import mne
import numpy as np
import scipy
import sys
import os.path
import matplotlib.pyplot as plt
import meet
import helper_functions


s_rate = 1000 # sampling rate of the EEG (1000 samples per sec)
data_folder = sys.argv[1]
result_folder = sys.argv[2]

if not os.path.exists(result_folder):
    os.mkdir(result_folder)

#for subject in range(1, N_subjects + 1, 1):
#for now only for first subject
subject = 1
try:
    with np.load(os.path.join(data_folder, 'S%02d' % subject,
            'clean_data.npz'), 'r') as f:
        artifact_mask = f['artifact_mask']
        clean_data = f['clean_data']
        #clean_data = f['clean_data'][:, artifact_mask] #DONT exclude artifacts, otherwise sample numbers wont match later on
except:
    print('Warning: Subject %02d could not be loaded!' %i)

if os.path.exists(os.path.join(
    data_folder, 'S%02d' % subject,
    'S{:02d}_eeg_all_files.vmrk'.format(subject)
    )):
    marker_fname = os.path.join(
            data_folder, 'S%02d' % subject,
            'S{:02d}_eeg_all_files.vmrk'.format(subject))
else:
    marker_fname = os.path.join(data_folder, 'S%02d' % subject,
        'S%02d_eeg.vmrk' % subject)

eeg_clocks = helper_functions.getSessionClocks(marker_fname)
eeg_clocks = [c for c in eeg_clocks if len(c) > 100]


# read the channel names
channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
n_channels = clean_data.shape[0]

# convert our data to raw format (io: https://mne.tools/stable/auto_examples/io/plot_objects_from_arrays.html#sphx-glr-auto-examples-io-plot-objects-from-arrays-py)
montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(channames, chancoords)),
        coord_frame='head')
info = mne.create_info(
        ch_names=channames,
        sfreq=s_rate,
        ch_types=np.repeat('eeg',n_channels))
info.set_montage(montage)
raw = mne.io.RawArray(clean_data, info)

#plot to check
scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
#raw.plot(n_channels=n_channels, scalings=scalings,
#        title='Auto-scaled Data from arrays',show=True, block=True)
#raw.plot_psd(fmin=1., fmax=45., tmax=60., average=False);
raw.info

#https://mne.tools/stable/auto_tutorials/source-modeling/plot_mne_dspm_source_localization.html#sphx-glr-auto-tutorials-source-modeling-plot-mne-dspm-source-localization-py
# triggers are all duple and triple beats, see read_aif.py
# müssen aus eeg rohdaten tick von sync clock (siehe vmrk datei wie in zb FFtcSpoc)
# siehe zb prepareFFtcSpoc.py: eeg_clocks enthält sync ticks
# syncMusicToEeg gibt uns drum schläge in eeg daten [in sample nummer]
#--> brauchen read_aif, helper funtions

#a) events are only Cues in bar 4
with np.load(os.path.join(result_folder, 'S%02d' % subject,
    'behavioural_results.npz'), 'r', allow_pickle=True, encoding='latin1') as f:
    snareCue_nearestClock = f['snareCue_nearestClock']
    snareCue_DevToClock = f['snareCue_DevToClock']
    wdBlkCue_nearestClock = f['wdBlkCue_nearestClock']
    wdBlkCue_DevToClock = f['wdBlkCue_DevToClock']
    bar_duration = f['bar_duration']

snareCue_events = helper_functions.SyncMusicToEEG(eeg_clocks,
        snareCue_nearestClock,
        snareCue_DevToClock, s_rate=s_rate)
snareCue_events = [np.array([e, 0, 0]) for e in snareCue_events]

wdBlkCue_events = helper_functions.SyncMusicToEEG(eeg_clocks,
        wdBlkCue_nearestClock,
        wdBlkCue_DevToClock, s_rate=s_rate)
wdBlkCue_events = [np.array([e, 0, 1]) for e in wdBlkCue_events]

allCues_events = np.concatenate((
        wdBlkCue_events,snareCue_events))
allCues_events = allCues_events[np.argsort(allCues_events[:,0])]
allCues_events[:, 0] = allCues_events[:, 0]-allCues_events[0,0]
# shows when which event occured over time
#mne.viz.plot_events(allCues_events, raw.info['sfreq'], raw.first_samp);

event_id = {"snare": 0, "wdBlk": 1}
tmin = -0.0  # start of each epoch (a) takt 4
tmax = bar_duration  # end of each epoch a) one bar takes around 1.71 s
epochs = mne.Epochs(raw, allCues_events, event_id, tmin, tmax,
        baseline=None, proj=True)

# volt of different electrodes over time and their location on sculp
evoked = epochs.average()
evoked.plot(show=False,spatial_colors=True);
# plot scalp polarities over times
evoked.plot_topomap(show=False,times=np.linspace(0., 0.3, 6));
#both together
evoked.plot_joint(show=False,times=[0.105, 0.130, 0.180]); #TODO give times of all beats

#where does it come from in the brain?
evoked_faces = epochs['snare'].average()
evoked_faces.plot_topomap(show=False,times=[0.095], size=2)
evoked_faces = epochs['wdBlk'].average()
evoked_faces.plot_topomap(show=False,times=[0.095], size=2)


#b) track listening (bars 1-3): event is first beat of each trial

# get events. differentietae between snare and wdblk?
# first beat of trial is cue - 3 bars (in sample time)
bar_samples = int(s_rate * bar_duration) # number of samples per bar
allStarts_events = allCues_events
allStarts_events[:,0] = np.array([allCues_events[:,0]- 3*bar_samples])

tmin_b = -0.1  # start of each epoch (0.1 seconds before trial)
tmax_b = bar_duration*3  # listening period
epochs_b = mne.Epochs(raw, allStarts_events, event_id, tmin_b, tmax_b,
        baseline=None, proj=True)

# volt of different electrodes over time and their location on sculp
evoked_b = epochs_b.average()
evoked_b.plot(show=False,spatial_colors=True);
# plot scalp polarities over times -.-
fig, axes = plt.subplots(ncols=6, nrows=3)
for ax,i in zip(axes,range(3)):
    evoked_b.plot_topomap(axes=ax,show=False, colorbar=False,
                times=np.linspace(i*bar_duration, (i+1)*bar_duration, 6))
    fig.suptitle('Bars 1-3 (Listening)')
plt.show()
