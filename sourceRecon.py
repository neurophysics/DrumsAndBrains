import mne
import numpy as np
import scipy
import sys
import os.path
import matplotlib.pyplot as plt
import meet
import helper_functions


s_rate = 1000 # sampling rate of the EEG
data_folder = sys.argv[1]
N_subjects = int(sys.argv[2]) #here: total number of subjects
result_folder = sys.argv[3]

save_folder = os.path.join(result_folder, 'all%02dsubjects' % N_subjects)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

#for subject in range(1, N_subjects + 1, 1):
#for now only for first subject
subject = 1
try:
    with np.load(os.path.join(data_folder, 'S%02d' % subject,
            'clean_data.npz'), 'r') as f:
        artifact_mask = f['artifact_mask']
        clean = f['clean_data']
        clean_data = f['clean_data'][:, artifact_mask] #exclude artifacts
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
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(channames, chancoords)),
                                        coord_frame='head')
info = mne.create_info(ch_names=channames, montage=montage,
        sfreq=s_rate, ch_types=np.repeat('eeg',n_channels))
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

#a) only Cues
with np.load(os.path.join(result_folder, 'S%02d' % subject,
    'behavioural_results.npz'), 'r', allow_pickle=True, encoding='latin1') as f:
    snareCue_nearestClock = f['snareCue_nearestClock']
    snareCue_DevToClock = f['snareCue_DevToClock']
    wdBlkCue_nearestClock = f['wdBlkCue_nearestClock']
    wdBlkCue_DevToClock = f['wdBlkCue_DevToClock']

snareCue_events = helper_functions.SyncMusicToEEG(eeg_clocks,
        snareCue_nearestClock,
        snareCue_DevToClock, s_rate=s_rate)
snareCue_events = [np.array([e, 0, 0]) for e in snareCue_events]

wdBlkCue_events = helper_functions.SyncMusicToEEG(eeg_clocks,
        wdBlkCue_nearestClock,
        wdBlkCue_DevToClock, s_rate=s_rate)
wdBlkCue_events = [np.array([e, 0, 1]) for e in wdBlkCue_events]

allCues_events = np.concatenate((
        wdBlkCue_events,snareCue_events)) #ich krieg es nicht sortiert :(
# shows when which event occured over time
#mne.viz.plot_events(allCues_events, raw.info['sfreq'], raw.first_samp);

event_id = {"snare": 0, "wdBlk": 1}
tmin = -0.0  # start of each epoch (a) takt 4 und b) takt 1-3
tmax = 1.72  # end of each epoch a) one bar takes around 1.71 s
epochs = mne.Epochs(raw, allCues_events, event_id, tmin, tmax,
        baseline=None, proj=True)

data = epochs.get_data() #26 bad epochs ???

# volt of different electrodes over time and their location on sculp
evoked = epochs.average()
evoked.plot(spatial_colors=True);
# plot scalp polarities over times
evoked.plot_topomap(times=np.linspace(0., 0.3, 6));
#both together
evoked.plot_joint(times=[0.105, 0.130, 0.180]); #TODO give times of all beats

#where does it come from in teh brain?
evoked_faces = epochs['snare'].average()
evoked_faces.plot_topomap(times=[0.095], size=2)
evoked_faces = epochs['wdBlk'].average()
evoked_faces.plot_topomap(times=[0.095], size=2)
