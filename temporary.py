# this is only to manually reject some trials in subject 2 and 12
# only run once!
import sys
import numpy as np
import os
import helper_functions
data_folder = sys.argv[1]
result_folder = sys.argv[2]
s_rate = 1000

# subject 2: trial 139-140
# load current segments
artifact_segments2 = np.load(os.path.join(data_folder,
    'S02/artifact_segments.npy'))
# get positions where trial 139 begins and 140 ends
if os.path.exists(os.path.join(
    data_folder, 'S02/S02_eeg_all_files.vmrk')):
    marker_fname = os.path.join(
            data_folder, 'S02/S02_eeg_all_files.vmrk')
else:
    marker_fname = os.path.join(data_folder, 'S02/S02_eeg.vmrk')
eeg_clocks = helper_functions.getSessionClocks(marker_fname)
eeg_clocks = [c for c in eeg_clocks if len(c) > 100]
assert len(eeg_clocks) == 6, '6 sessions expected'

with np.load(os.path.join(result_folder,'S02/behavioural_results.npz'),
    'r', allow_pickle=True, encoding='latin1') as f:
    snareCue_nearestClock = f['snareCue_nearestClock']
    snareCue_DevToClock = f['snareCue_DevToClock']
    wdBlkCue_nearestClock = f['wdBlkCue_nearestClock']
    wdBlkCue_DevToClock = f['wdBlkCue_DevToClock']
    snareCue_times = f['snareCue_times']
    wdBlkCue_times = f['wdBlkCue_times']
snareCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
        snareCue_nearestClock, snareCue_DevToClock)
wdBlkCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
        wdBlkCue_nearestClock, wdBlkCue_DevToClock)
all_pos = np.sort(np.hstack([snareCue_pos, wdBlkCue_pos]))
ex_pos = np.array([all_pos[139]/s_rate, (all_pos[141]-1)/s_rate])
artifact_segments2 = np.vstack([artifact_segments2, ex_pos])
np.save(os.path.join(data_folder, 'S02/artifact_segments.npy'),
        artifact_segments2)

#subject 12: trial 47-70
# load current segments
artifact_segments12 = np.load(os.path.join(data_folder,
    'S12/artifact_segments.npy'))
# get positions where trial 139 begins and 140 ends
if os.path.exists(os.path.join(
    data_folder, 'S12/S12_eeg_all_files.vmrk')):
    marker_fname = os.path.join(
            data_folder, 'S12/S12_eeg_all_files.vmrk')
else:
    marker_fname = os.path.join(data_folder, 'S12/S12_eeg.vmrk')
eeg_clocks = helper_functions.getSessionClocks(marker_fname)
eeg_clocks = [c for c in eeg_clocks if len(c) > 100]
assert len(eeg_clocks) == 6, '6 sessions expected'

with np.load(os.path.join(result_folder,'S12/behavioural_results.npz'),
    'r', allow_pickle=True, encoding='latin1') as f:
    snareCue_nearestClock = f['snareCue_nearestClock']
    snareCue_DevToClock = f['snareCue_DevToClock']
    wdBlkCue_nearestClock = f['wdBlkCue_nearestClock']
    wdBlkCue_DevToClock = f['wdBlkCue_DevToClock']
    snareCue_times = f['snareCue_times']
    wdBlkCue_times = f['wdBlkCue_times']
snareCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
        snareCue_nearestClock, snareCue_DevToClock)
wdBlkCue_pos = helper_functions.SyncMusicToEEG(eeg_clocks,
        wdBlkCue_nearestClock, wdBlkCue_DevToClock)
all_pos = np.sort(np.hstack([snareCue_pos, wdBlkCue_pos]))
ex_pos12 = np.array([all_pos[47]/s_rate, (all_pos[71]-1)/s_rate])
artifact_segments12 = np.vstack([artifact_segments12, ex_pos12])
np.save(os.path.join(data_folder, 'S12/artifact_segments.npy'),
        artifact_segments12)

#afterwards run eeg_outlier.py again, so the updated artifact mask gets stored
