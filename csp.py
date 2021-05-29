"""
executes common spatial pattern algorithm on ERD data
"""
import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
import scipy
import meet
import helper_functions

data_folder = sys.argv[1]
result_folder = sys.argv[2]
N_subjects = 21
s_rate = 1000 # sampling rate of the EEG

# read ERD data
# erd data is referenced to the average eeg amplitude, bandpass filtered
# with order 6, normalized s.t. 2 sec pre responce are 100%, trial-averaged
snareInlier = [] # list of 20 subjects, each shape ≤75
wdBlkInlier = []
target_covs = [] # stores pre-movemenet data, lists of 20 subjects, each list of 5 bands, each shape (32,2500)
contrast_covs = [] # stores baseline data, lists of 20 subjects, each list of 5 bands, each shape (32,2500)
all_snareHit_times = []
all_wdBlkHit_times = []
i=0
while True:
    try:
        with np.load(os.path.join(result_folder, 'motor_inlier.npz'),
            'r') as fi:
            snareInlier.append(fi['snareInlier_response_{:02d}'.format(i)])
            wdBlkInlier.append(fi['wdBlkInlier_response_{:02d}'.format(i)])
            win = fi['win']
        with np.load(os.path.join(result_folder, 'motor_covmat.npz'), 'r') as f_covmat:
            target_covs.append(f_covmat['target_cov_{:02d}'.format(i)])
            contrast_covs.append(f_covmat['contrast_cov_{:02d}'.format(i)])
            all_snareHit_times.append(f_covmat['snareHit_times_{:02d}'.format(i)])
            all_wdBlkHit_times.append(f_covmat['wdBlkHit_times_{:02d}'.format(i)])
            fbands = f_covmat['fbands']
            left_handed = f_covmat['left_handed']
        i+=1
    except KeyError:
        break

# for now, choose alpha band
target_cov_a = [t[2] for t in target_covs]
contrast_cov_a = [t[2] for t in contrast_covs]

## average the covariance matrices across all subjects
for t, c in zip(target_cov_a, contrast_cov_a):
    # normalize by the trace of the contrast covariance matrix
    t_now = t.mean(-1)/np.trace(c.mean(-1))
    c_now = c.mean(-1)/np.trace(c.mean(-1))
    try:
        all_target_cov += t_now
        all_contrast_cov += c_now
    except: #init
        all_target_cov = t_now
        all_contrast_cov = c_now

# calculate CSP
## EV and filter
CSP_eigvals, CSP_filters = helper_functions.eigh_rank(all_target_cov,
        all_target_cov + all_contrast_cov) #interestingly only gives 31 EV

# patterns
CSP_patterns = scipy.linalg.solve(
        CSP_filters.T.dot(all_target_cov).dot(CSP_filters),
        CSP_filters.T.dot(all_target_cov))

### normalize the patterns such that Cz is always positive
#@gunnar: we did this for ssd, here as well? would make sense to me
channames = meet.sphere.getChannelNames('channels.txt')
CSP_patterns*=np.sign(CSP_patterns[:,np.asarray(channames)=='CZ'])


np.savez(os.path.join(result_folder, 'CSP.npz'),
        CSP_eigvals = CSP_eigvals,
        CSP_filters = CSP_filters, # matrix W in paper
        CSP_patterns = CSP_patterns # matrix A in paper
        )

# plot eigenvalues
plt.plot(CSP_eigvals, 'o')
plt.title('CSP EV, small ERD, large ERS')
# first argument is pre movement so
# small EV for ERD: here 1 or maybe 3
# large EV for ERS: here 2 or 4
CSP_ERDnum = 1
CSP_ERSnum = 2

# apply CSP to EEG
ERD_CSP = []
idx = 0 #index to asses eeg (0 to 19)
subj = 1 #index for subject number (1 to 10, 12 to 21)
while(subj <= N_subjects):
    print(subj)
    # skip subject without eeg data
    if not os.path.exists(os.path.join(
        result_folder, 'S{:02d}'.format(subj), 'prepared_FFTSSD.npz')):
        subj += 1
        continue
    with np.load(os.path.join(data_folder, 'S%02d' % subj)
            + '/clean_data.npz', 'r') as fi:
        eeg = fi['clean_data'] # shape (32, 2901860)
        artifact_mask = fi['artifact_mask']
    eeg -= eeg.mean(0) # rereference to the average EEG amplitude
    # for lefthanded subjects, switch electrodes
    if left_handed[idx]:
        #print('subject '+str(subj)+' is left-handed. Switching electrodes...')
        # see list(enumerate(channames))
        eeg = np.vstack([eeg[1,:], eeg[0,:], eeg[6,:], eeg[5,:], eeg[4,:],
            eeg[3,:], eeg[2,:], eeg[10,:], eeg[9,:], eeg[8,:], eeg[7,:],
            eeg[15,:], eeg[14,:], eeg[13,:], eeg[12,:], eeg[11,:], eeg[21,:],
            eeg[20,:], eeg[19,:], eeg[18,:], eeg[17,:], eeg[16,:], eeg[26,:],
            eeg[25,:], eeg[24,:], eeg[23,:], eeg[22,:], eeg[28,:], eeg[27,:],
            eeg[31,:], eeg[30,:], eeg[29,:]])

    band = [8,12] #choose alpha for now
    # band-pass filters with order 6 (3 into each direction)
    Wn = np.array(band) / s_rate * 2
    b, a = scipy.signal.butter(3, Wn, btype='bandpass')
    eeg_filtbp = scipy.signal.filtfilt(b, a, eeg)
    # apply CSP to eeg data
    EEG_CSP_subj = np.tensordot(CSP_filters, eeg_filtbp, axes=(0,0))
    # Hilbert-Transform, absolute value
    eeg_filtHil = np.abs(scipy.signal.hilbert(EEG_CSP_subj, axis=-1))
    # normalize s.t.2000ms preresponse are 100%
    snareHit_times = all_snareHit_times[idx]
    wdBlkHit_times = all_wdBlkHit_times[idx]
    all_trials_filt = meet.epochEEG(eeg_filtHil,
            np.r_[snareHit_times[snareInlier[idx]],
                wdBlkHit_times[wdBlkInlier[idx]]],
            win)
    # calculate ERD
    ERD_CSP_subj = all_trials_filt.mean(-1) # trial average
    ERD_CSP_subj /= ERD_CSP_subj[:,0][:,np.newaxis] #want to start at 0%
    ERD_CSP_subj *= 100 # ERD in percent
    ERD_CSP.append(ERD_CSP_subj)

    idx += 1
    subj += 1

# average over subjects
ERD_CSP_subjmean = np.mean(ERD_CSP, axis=0)

# plot CSP components
erd_t = range(ERD_CSP[0].shape[1])
plt.plot(erd_t, ERD_CSP_subjmean[-CSP_ERDnum:,:].T, label='ERD')
plt.plot(erd_t, ERD_CSP_subjmean[:CSP_ERSnum,:].T, label='ERS')
plt.legend()
plt.show()
