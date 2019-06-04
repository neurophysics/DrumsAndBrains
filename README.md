# DrumsAndBrains
The DrumsAndBrain experiment aims at revealing the relation between neuronal oscillations and polyrhythmic listening experience and performance.

## Description of the experiment.
Sujects were listening to a drum beat (snare and woodblock) playing a duple (two notes per bar - snare drum) vs. a triple (3 beats per bar - woodblock) rhythm. This polyrhythm was played for three bars to the subjects followed by a bar of silence. An auditory cue beat (snare or woodblock) at the start of the subsequent bar indicated whether the subjects should tap the last note of that bar of either the duple or triple rhythm on an electronic drumpad.
This paradigm forces the subjects to concentrate on both rhythms and keep the rhythms active internally during the break bar - since only after the break bar the cue indicates which rhythm should be performed.
75 trials of every condition (duple/triple) were recorded (=150 trials in total) in every subject in (pseudo)-randomized order. Recordings were split into 6 sessions with breaks of self-determined duration. After every session subjects were asked to rate their performance and vigilance. Recordings took in total about 45 min per subject.
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

## Calculation of Spatial filters
We are using spatial filters to obtain the best signal-to-noise ratio available.

Four different types of spatial filters are to be used:

1. Canonical correlation analysis (CCA): Filters are trained from the
listening period to maximize the auditory evoked potentials.
Algorithmically, this works by maximizing the correlation between the
single trials and the average across single trials.

2. Spatio-spectral decomposition (SSD): Filters are trained from the
listening period to maximize the power of oscillations at the frequencies
of the polyrhythm the subjects were listening to.
Algorithmically, this works by (1) low-frequency band-pass filtering
between 0.5 and 2.5 Hz (lf-data), (2) extracting the requested frequencies
from the data by fitting sines and cosines at those frequencies (= explicite
Fourier transform) and reconstructing with only those frequencies included
(entrained-data), (3) common spatial pattern (CSP) of lf-data vs. entrained
data.

3. Phase coupling optimization (PCO): Filters are trained from the silence
period to maximize the dependence between the phases of neuronal
oscillations and performance. Algorithmically, this works by maximizing the
mean vector length. Typically, a smaller number of SSD filters are used as
pre-processing to safe-guard against overfitting.

4. Source power comodulation (SPoC) optimization. Filters are trained from
the silence period to maximize the dependence between the power of neuronal
oscillations and performance. Typically, a smaller number of SSD filters are
used as pre-processing to safe-guard against overfitting.

Note, that there is an important difference between methods 1+2 and 3+4.
Methods 3+4 directly maximize the relation between oscillations and
behaviour and thus might be prone to overfitting (towards that specific
relation).

Here, we chose an approach to train the spatial filters across subjects,
which (1) increases the amount of data and hereby decreases the risk of
overfitting and (2) should lead to a generalized result - approximately
valid for an 'average' subject.

### Prepare Filters
The script `prepare_filters.py` needs to be run for every subject.
It requires 3 arguments:

1. `data_folder`
2. subject number
3. `result_folder` (used to store the results)

In the `result_folder` of that subject, a file `prepared_filterdata.npz`
will be stored with the following entries:

- `listen_trials`, a channesl x samples x trials array of EEG data during
    listening period
- `listen_trials_avg`, a channels x samples array of averaged EEG data
    (across trials) during listening period
- `snareListenData`, a channels x samples x trials array of EEG data
    after low-frequency band-pass filtering (0.5-2.5 Hz) for the listening
    period of trials cued with a snare drum (duple beat).
- `snareListenData_rec`, a channels x samples x trials array of EEG data
    contaning only the duple and triple beat frequencies.
- `snareInlier`, a 1d boolean array indication which trials are valid (for
    trials cued with a snare drum)
- 'snareFitListen', a trials x 6 matrix, fitted by the explicit Fourier
    transform with the following entries:
    - `snareFitListen[:,0]`: the mean values of every trial
    - `snareFitListen[:,1]`: the slope values of every trial
    - `snareFitListen[:,2]`: the scale of cosine at the snare (duple)
        frequency (= real part)
    - `snareFitListen[:,3]`: the scale of sine at the snare (duple)
        frequency (= imaginary part)
    - `snareFitListen[:,4]`: the scale of cosine at the wdBlk (triple)
        frequency (= real part)
    - `snareFitListen[:,5]`: the scale of sine at the wdBlk (triple)
        frequency (= imaginary part)
- all `Listen` and `snare` fields also exist as `Silence` and `wdBlk`
    fields.

### Calculate SSD
`calcSSD.py` calculates the SSD across all subjects.
It requires 2 arguments:

1. `result_folder` (used to store the results)
2. `normalize`: `0` or `1` to indicate whether the data should be
    normalized for every individual subject prior to averaging across
    subjects.

SSD is calculated by (1) reading `snareListenData`, `snareListenData_rec`,
`wdBlkListenData`, and `wdBlkListenData_rec` from individual subjects.
(2) concatenating the snare and woodblock triakls (during the listening
period, these are equal) and calculating the covariance matrix for 
the `Data` and `Data_rec` arrays. (3) If normalization is requested, the
covariance matrices are divided by the trace of the covariance matrix of
the`Data` array. (4) Averaging across subjects, (5) generalized eigenvalue
decomposition of `Data_rec_cov` vs. `(Data_rec_cov + Data_cov)`

The script stores the results as an image (`SSD_patterns_norm_True.pdf` or
`SSD_patterns_norm_Fals.pdf`, depending on wheter normalization had been
requested) and as a file (`SSD_norm_True.npz` or `SSD_norm_False.npz`)
containing:

1. `ssd_eigvals`: the eigenvalues (bound between 0 and 1) signal the ratio
    of variances between the extracted frequencies and the low-frequency
    band-pass for the respective component.
2. `ssd_filter`: the spatial filters of the respective components (in the
    columns of that array).
