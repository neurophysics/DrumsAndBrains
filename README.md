# DrumsAndBrains

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

