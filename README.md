# DrumsAndBrains
The DrumsAndBrain experiment aims at revealing the relation between neuronal oscillations and polyrhythmic listening experience and performance.

## Description of the experiment.
Sujects were listening to a drum beat (snare and woodblock) playing a duple (two notes per bar - snare drum) vs. a triple (3 beats per bar - woodblock) rhythm. This polyrhythm was played for three bars to the subjects followed by a bar of silence. An auditory cue beat (snare or woodblock) at the start of the subsequent bar indicated whether the subjects should tap the last note of that bar of either the duple or triple rhythm on an electronic drumpad.
This paradigm forces the subjects to concentrate on both rhythms and keep the rhythms active internally during the break bar - since only after the break bar the cue indicates which rhythm should be performed. 
The tempo of the rhythm had been 150 QPM (quarter notes per minute), i.e. a complete bar took around 1.71 s.
32-channel EEG with BrainAmp EEG amplifiers had been recorded from the subjects during the experiments. Subjects were requested to relax, fixate a cross at the wall and perform as precisely as possible. Synchronisation between the EEG recordings and the cues and behavioural responses was achieved by feeding a 1/s trigger (referred to as 'clock') simultaneously into the audio recordings (cues and behavioural responses) and EEG.
Handedness was recorded using the Edinburgh Handedness Inventory and questions regarding the musical experience of the subjects were asked an their responses recorded.

## Analysis of behavioural data
Run the script read_aif.py with
1. 1st argument: data folder
2. 2nd argument: subject number
3. 3rd argument: result folder

e.g., the line

    python read_aif.py ~/Data 01 Results

looks for subject "S01" in folder ~/Data and will put the results into folder ./Results 

The script will store a file 'behavioural_results.npz' in the result forder of that subject. It contains the following fields:

* snareCue_nearestClock: for each snare drum cue, the index of the closest trigger 'clock'
* snareCue_DevToClock: for each snare drum cue, the deviation (in s) to the nearest clock
* snareCuetimes: the times (in s of the aif file) of snare drum cues
* snare_deviation: the deviation of the performed snare drum rhythm beats to the optimal timing (in s)
* bar_duration

All 'snare_' fields equally exist for 'wdBlk_'

Additionally some plots are saved in the Results folder of that subject.

## Outlier Rejection
Run the script eeg_outlier.py with
1. 1st argument: data folder
2. 2nd argument: subject number
3. 3rd argument: result folder

e.g., the line

    python eeg_outlier.py ~/Data 01 Results

looks for subject "S01" in folder ~/Data and will put the results into folder ./Results 

### Outlier Rejection Process
1. A window with the EEG data will open. Change gain with +/- and scroll with 'Home'/'End', LeftArrow, RightArrow, PgUp, PgDown Keys. Select artifactual segments by drawing a rectangle (only temporal information will be recorded, the selected channels are irrelavant) and save by pressing 'r'. After you think all relevant artifact segments were selected, close the window. The plotted EEG data had been downsampled to 250 Hz, high-pass filtered above 2 Hz for subsequent ICA, and re-referenced to the average potential.
2. Repeat the process in a new window to be sure you selected everything important.
3. (optional) "Bad" channels can be rejected (and interpolated by a spherical spline algorithm) by creating a file 'interpolate_channels.txt' in the data folder of that subject and containing the channel names to be rejected (uppercase letters), one per line.
4. ICA (FastICA after a whitening step using PCA) will be performed and the independent components, their scalp projections and powers will be plotted. The variance of the independent components will reflect the contribution to the surface EEG. Create a file 'reject_ICs.txt' in the data folder of the subject containing the names of the components to be rejected ('IC00', 'IC01' etc - one per line).
5. The final 2 windows show the data before (Figure 1) and after (Figure 2) ICA-based artifact rejection. You may make a screenshot for documentation purposes.
6. The 'cleaned' data is stored in the data folder of the subject as 'clean_data.npz' with fields 'clean data' containing the data (high-pass filtered at 0.1 Hz, 1000 Hz sampling rate, average reference), and 'artifact_mask', a boolean array of the same length as the data with Trues at non-selected timepoints and Falses at all timepoints selected as artifacts (in Step 1).

The ICA results are stored in the data folder of every single subject as a file "ICA_result.joblib". If the ICA should be re-computed, delete that file.

Additionally some plots are saved in the Results folder of that subject.
