# DrumsAndBrains
The DrumsAndBrain experiment aims at revealing the relation between neuronal oscillations and polyrhythmic listening experience and performance.

## Description of the experiment.
Subjects were listening to a drum beat (snare and woodblock) playing a duple (two notes per bar - snare drum) vs. a triple (3 beats per bar - woodblock) rhythm. This polyrhythm was played for three bars to the subjects followed by a bar of silence. An auditory cue beat (snare or woodblock) at the start of the subsequent bar indicated whether the subjects should tap the last note of that bar of either the duple or triple rhythm on an electronic drumpad.
This paradigm forces the subjects to concentrate on both rhythms and keep the rhythms active internally during the break bar - since only after the break bar the cue indicates which rhythm should be performed.
75 trials of every condition (duple/triple) were recorded (=150 trials in total) in every subject in (pseudo)-randomized order. Recordings were split into 6 sessions with breaks of self-determined duration. After every session subjects were asked to rate their performance and vigilance. Recordings took in total about 45 min per subject.
The tempo of the rhythm had been 150 QPM (quarter notes per minute), i.e. a complete bar took around 1.71 s.
32-channel EEG with BrainAmp EEG amplifiers had been recorded from the subjects during the experiments. Subjects were requested to relax, fixate a cross at the wall and perform as precisely as possible. Synchronisation between the EEG recordings and the cues and behavioural responses was achieved by feeding a 1/s trigger (referred to as 'clock') simultaneously into the audio recordings (cues and behavioural responses) and EEG.
Handedness was recorded using the Edinburgh Handedness Inventory and questions regarding the musical experience of the subjects were asked an their responses recorded.

## Analysis of behavioural data
Run the script `read_aif.py` with
1. 1st argument: data folder
2. 2nd argument: subject number
3. 3rd argument: result folder

e.g., the line

    python read_aif.py ~/Data 01 Results

looks for subject `S01` in folder `~/Data` and will put the results into folder `./Results`

The script will store a file `behavioural_results.npz` in the result forder of that subject. It contains the following fields:

- `snareCue_nearestClock`: for each snare drum cue, the index of the closest trigger 'clock'
- `snareCue_DevToClock`: for each snare drum cue, the deviation (in s) to the nearest clock
- `snareCuetimes`: the times (in s of the aif file) of snare drum cues
- `snare_deviation`: the deviation of the performed snare drum rhythm beats to the optimal timing (in s)
- `bar_duration`

All `snare_` fields equally exist for `wdBlk_`

Additionally some plots are saved in the Results folder of that subject.

## Outlier Rejection
Run the script `eeg_outlier.py` with
1. 1st argument: data folder
2. 2nd argument: subject number
3. 3rd argument: result folder

e.g., the line

    python eeg_outlier.py ~/Data 01 Results

looks for subject `S01` in folder `~/Data` and will put the results into folder `./Results`

### Outlier Rejection Process
1. A window with the EEG data will open. Change gain with +/- and scroll with 'Home'/'End', LeftArrow, RightArrow, PgUp, PgDown Keys. Select artifactual segments by drawing a rectangle (only temporal information will be recorded, the selected channels are irrelevant) and save by pressing 'r'. After you think all relevant artifact segments were selected, close the window. The plotted EEG data had been downsampled to 250 Hz, high-pass filtered above 2 Hz for subsequent ICA, and re-referenced to the average potential.
2. Repeat the process in a new window to be sure you selected everything important.
3. (optional) "Bad" channels can be rejected (and interpolated by a spherical spline algorithm) by creating a file `interpolate_channels.txt` in the data folder of that subject and containing the channel names to be rejected (uppercase letters), one per line.
4. ICA (FastICA after a whitening step using PCA) will be performed and the independent components, their scalp projections and powers will be plotted. The variance of the independent components will reflect the contribution to the surface EEG. Create a file `reject_ICs.txt` in the data folder of the subject containing the names of the components to be rejected ('IC00', 'IC01' etc - one per line).
5. The final 2 windows show the data before (Figure 1) and after (Figure 2) ICA-based artifact rejection. You may make a screenshot for documentation purposes.
6. The 'cleaned' data is stored in the data folder of the subject as `clean_data.npz` with fields `clean data` containing the data (high-pass filtered at 0.1 Hz, 1000 Hz sampling rate, average reference), and `artifact_mask`, a boolean array of the same length as the data with Trues at non-selected timepoints and Falses at all timepoints selected as artifacts (in Step 1).

The ICA results are stored in the data folder of every single subject as a file `ICA_result.joblib`. If the ICA should be re-computed, delete that file.

Additionally some plots are saved in the Results folder of that subject.

## Calculation of Spatial filters

We are using spatial filters to obtain the best signal-to-noise ratio available. We chose an approach to train the spatial filters across subjects,
which (1) increases the amount of data and hereby decreases the risk of
overfitting and (2) should lead to a generalized result - approximately
valid for an 'average' subject.

Spatio-spectral decomposition (SSD): Filters are trained from the
listening and silence period to maximize the power of oscillations at the frequencies
of the polyrhythm the subjects were listening to.
Algorithmically, this works by
(1) isolating snare and woodblock frequency from the FFT and applying the inverse transform, then calculation of covariance matrices for every trial = 'target',
(2) isolating range of 1-2 Hz, applying inverse transform and calculation of covariance matrices for every single trial 'contrast' (1+2 is done in `prepareFFTSSD.py`) and
(3) calculating the SSD of stimulation frequencies vs. the neighboring frequencies (in `calcFFTSSD.py`).


### Prepare SSD calculation

The script `prepareFFTSSD.py` needs to be run for every subject.
It requires 3 arguments:

1. `data_folder`
2. subject number
3. `result_folder` (used to store the results)

In the `result_folder` of that subject, a file `prepared_FFTSSD.npz`
will be stored containing as fields:

1. `target_cov`: the covariance matrix (cross spectral density, csd) for the target frequencies (7/6 and 7/4) of all single trials of that subject in the listening period (the first 3 bars)
2. `contrast_cov`: as `target_cov` but for the contrast frequencies, i.e. a 1-2 Hz window but without the target frequencies
3. `F`: the Fast Fourier Transform (FFT) of all listen trials (padded to 12 sec)
4. `f`: an array of frequencies (in Hz) use in the FFT

### Calculate SSD
The script `calcFFTSSD.py` calculates the across-subject SSD.

As input argument, it requires the `result_folder`. Number of subjects is set to be 21.

It first loads the results from `prepareFFTSSD.py` and normalizes the
target and contrast covariance matrices of every subject by the trace of their contrast covariance matrix. This already normalizes the power across subjects.

Then, SSD eigenvalues and filters are obtained using eigh_rank in from `helper_functions.py`. The corresponding SSD patterns are calculated and normalized s.t. channel Cz is always positive.

Finally, the FFT with and without applied SSD is normalized and averaged and plotted to verify the SSD (use `plot.show()` to show the plot).

`calcFFTSSD.py` stores its results in  the file `FFTSSD.npz` containing the
fields:
1. `F_SSD`: the FFT with applied SSD
2. `SSD_eigvals`: the found eigenvalues corresponding to the SSD filters
3. `SSD_filters`: the found SSD filters
4. `SSD_patterns`: the spatial patterns corresponding to these filters


### Correlation of SSD result with musical experience
Run the script `SSDMusicCorr.py` with `data_folder` and `result_folder` as
arguments.
Significance testing is done as one-tailed permutation testing of the
correlation with `N=1000` permutations.
Result is plotted as `SNNR_exp.pdf` and `png`.

### Within-between random effects (REWB) model
A regression model is used to link EEG power at different frequencies/components, musical experience, session/trial index etc. to drumming performance.
If responses of subject i in trial t are given as `y_it` and EEG power of component/frequency `k` in that subject and trial as `x_kit`, the regression is discribed by the formula:

<img src="https://render.githubusercontent.com/render/math?math=y_{it} = \beta_0 %2B \sum_{k=0}^{K-1}\beta_{1Wk}(x_{kit} - \overline{x}_{ki}) %2B \sum_{k=0}^{K-1}\beta_{2Bk}\overline{x}_{ki} %2B \beta_3 z_i %2B \upsilon_{i0} %2B \sum_{k=0}^{K-1}\upsilon_{ki1}(x_{kit} - \overline{x}_{ki}) %2B \epsilon_{kit0}">

where `z_i` is the musical performance of that subject, all betas are coefficients and all upsilons are random effects. Subscripts W denote within-subjects and subscripts B denote between-subjects effects. Epsilon is an error term.
Additionally, trial-index and session-index are fed as independent variables into the design matrix of the regression problem.

## GAMLSS
Best two SSD Filters applied to EEG, additional subject information and behavioral data is read and divided into snare and wdBlk trials. Behavioral data is normalized by rejecting data outside median ± 1.5 IQR, resulting histograms and qqplots over all subjects are stored in the result folder under `gamlss_NormalityDev.pdf`.
OLS is performed for snare and wdBlk individually.
explained variable: Behavioral data (deviation)
explanatory variables: (intercept,) SSD filtered EEG, musical score, random effect (RE) within and between subjects
...
