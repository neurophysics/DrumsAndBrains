"""
executes common spatial pattern algorithm on ERD data
"""
import numpy as np
import sys
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import meet
import helper_functions

data_folder = sys.argv[1]
result_folder = sys.argv[2]
N_subjects = 21
s_rate = 1000 # sampling rate of the EEG

channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)


##### read ERD data ####
# ERD (ERS) = percentage of power decrease (increase):
# ERD% = (A-R)/R*100
# A event period power 1250-2000 (i.e.(-750 to 0 because we want pre stimulus)))
# R reference period power 0-750 (i.e. -2000:-1250 samples)
# erd data is already referenced to the average eeg amplitude, bandpass filtered
# with order 6, normalized s.t. 2 sec pre responce are 100%, trial-averaged
snareInlier = [] # list of 20 subjects, each shape ≤75
wdBlkInlier = []
target_covs = [] # stores pre-movemenet data, lists of 20 subjects, each list of 5 bands, each shape (32,143)
contrast_covs = [] # stores baseline data, lists of 20 subjects, each list of 5 bands, each shape (32,143)
all_snareHit_times = []
all_wdBlkHit_times = []
i=0
while True:
    try:
        with np.load(os.path.join(result_folder, 'motor/inlier.npz'),
            'r') as fi:
            snareInlier.append(fi['snareInlier_response_{:02d}'.format(i)])
            wdBlkInlier.append(fi['wdBlkInlier_response_{:02d}'.format(i)])
            win = fi['win']
        with np.load(os.path.join(result_folder, 'motor/covmat.npz'), 'r') as f_covmat:
            target_covs.append(f_covmat['target_cov_{:02d}'.format(i)])
            contrast_covs.append(f_covmat['contrast_cov_{:02d}'.format(i)])
            all_snareHit_times.append(f_covmat['snareHit_times_{:02d}'.format(i)])
            all_wdBlkHit_times.append(f_covmat['wdBlkHit_times_{:02d}'.format(i)])
            fbands = f_covmat['fbands']
            left_handed = f_covmat['left_handed']
            base_idx = f_covmat['base_idx'] #corresponds to -2000 to -1250ms
            act_idx = f_covmat['act_idx'] #corresponds to -750 to 0ms
        i+=1
    except KeyError:
        break
band_names = ['[' + str(b[0]) + '-' + str(b[1]) + ']' for b in fbands]

##### try reading CSP, if not calculate it #####
CSP_eigvals = []
CSP_filters = []
CSP_patterns = []
try:
    i=0
    while True:
        try:
            with np.load(os.path.join(result_folder,'motor/CSP.npz')) as f:
                band_names = band_names
                CSP_eigvals.append(f['CSP_eigvals{:s}'.format(band_names[i])])
                CSP_filters.append(f['CSP_filters{:s}'.format(band_names[i])])
                CSP_patterns.append(f['CSP_patterns{:s}'.format(band_names[i])])
            i+=1
        except IndexError:
            break
    print('CSP succesfully read.')
except FileNotFoundError: # read ERD data and calculate CSP
    for band_idx, band_name in enumerate(band_names):
        target_cov_a = [t[band_idx] for t in target_covs]
        contrast_cov_a = [t[band_idx] for t in contrast_covs]

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

        ##### calculate CSP #####
        ## EV and filter
        CSP_eigval, CSP_filter = helper_functions.eigh_rank(all_target_cov,
                all_contrast_cov)
        CSP_eigvals.append(CSP_eigval)
        CSP_filters.append(CSP_filter)

        # patterns
        CSP_pattern = scipy.linalg.solve(
                CSP_filter.T.dot(all_target_cov).dot(CSP_filter),
                CSP_filter.T.dot(all_target_cov))

        ### normalize the patterns such that Cz is always positive
        CSP_pattern*=np.sign(CSP_pattern[:,np.asarray(channames)=='CZ'])
        CSP_patterns.append(CSP_pattern)

    save_results = {}
    for (band_now, eigvals_now, filters_now, patterns_now) in zip(
        band_names, CSP_eigvals, CSP_filters, CSP_patterns):
        save_results['CSP_eigvals{:s}'.format(band_now)] = eigvals_now
        save_results['CSP_filters{:s}'.format(band_now)] = filters_now
        save_results['CSP_patterns{:s}'.format(band_now)] = patterns_now
    np.savez(os.path.join(result_folder, 'motor', 'CSP.npz'),
        band_names=band_names, **save_results)
    print('CSP succesfully calculated.')


##### try reading applied ERDCSP, if not apply CSP to EEG (takes a while) #####
ERD_CSP = [] # stores trial averaged ERD/S_CSP per subject, each with shape (Nband, CSPcomp,time)
ERDCSP_trial = [] #stores ERD_CSP of best CSPcomp per subject,each shape (Nband, Ntrial)
ERSCSP_trial = [] # same for ERS
try:
    i=0
    while True:
        try:
            with np.load(os.path.join(result_folder,'motor/erdcsp2.npz')) as f:
                ERD_CSP.append(f['ERDCSP_{:02d}'.format(i)])
                ERDCSP_trial.append(f['ERDCSP_trial_{:02d}'.format(i)])
                ERSCSP_trial.append(f['ERSCSP_trial_{:02d}'.format(i)])
            i+=1
        except KeyError:
            break
    print('ERDCSP succesfully read.')
except FileNotFoundError: # read ERD data and calculate CSP
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

        ERD_CSP_subj = []
        ERDCSP_trial_band = []
        ERSCSP_trial_band = []
        for band_idx, band in enumerate(fbands):
            # band-pass filters with order 6 (3 into each direction)
            Wn = np.array(band) / s_rate * 2
            b, a = scipy.signal.butter(3, Wn, btype='bandpass')
            eeg_filtbp = scipy.signal.filtfilt(b, a, eeg)
            # apply CSP to eeg data
            EEG_CSP_subj = np.tensordot(CSP_filters[band_idx], eeg_filtbp,
                axes=(0,0))
            # Hilbert-Transform, absolute value (could also be abs**2)
            eeg_filtHil = np.abs(scipy.signal.hilbert(EEG_CSP_subj, axis=-1))**2
            # normalize s.t.2000ms preresponse are 100%
            snareHit_times = all_snareHit_times[idx]
            wdBlkHit_times = all_wdBlkHit_times[idx]
            all_trials_filt = meet.epochEEG(eeg_filtHil,
                    np.r_[snareHit_times[snareInlier[idx]],
                        wdBlkHit_times[wdBlkInlier[idx]]],
                    win) # (ERDcomp, time, trials) = (31, 2500, 143)
            # calculate trial averaged ERDCSP
            ERD_CSP_subj_band = all_trials_filt.mean(-1) # trial average

            #ERD_CSP_subj_band /= ERD_CSP_subj_band[:,0:750].mean(1)[:,np.newaxis] #baseline avg should be 100%
            #ERD_CSP_subj_band *= 100 # ERD in percent
            ERD_CSP_subj.append(ERD_CSP_subj_band)

            # calculate ERD in percent per subject and trial
            ERDCSP_allCSP_trial = (np.mean(all_trials_filt[:,act_idx], axis=1
                    ) - np.mean(all_trials_filt[:,base_idx], axis=1)
                    ) / np.mean(all_trials_filt[:,base_idx], axis=1) * 100
            # only keep min/max value i.e. best component
            ERDCSP_trial_band.append(
                ERDCSP_allCSP_trial[np.argmin(np.min(ERDCSP_allCSP_trial, axis=1))])
            ERSCSP_trial_band.append(
                ERDCSP_allCSP_trial[np.argmax(np.max(ERDCSP_allCSP_trial, axis=1))])
            # end for each band
        ERD_CSP.append(ERD_CSP_subj)
        ERDCSP_trial.append(ERDCSP_trial_band)
        ERSCSP_trial.append(ERSCSP_trial_band)
        idx += 1
        subj += 1

    save_ERDCSP = {}
    for i, (e, d, s) in enumerate(zip(ERD_CSP, ERDCSP_trial, ERSCSP_trial)):
        save_ERDCSP['ERDCSP_{:02d}'.format(i)] = e
        save_ERDCSP['ERDCSP_trial_{:02d}'.format(i)] = d
        save_ERDCSP['ERSCSP_trial_{:02d}'.format(i)] = s
    np.savez(os.path.join(result_folder, 'motor/erdcsp.npz'), **save_ERDCSP)
    print('ERDCSP succesfully calculated and stored.')
