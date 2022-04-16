"""
executes multisubject common spatial pattern algorithm on ERD data
"""
import numpy as np
import sys
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import helper_functions
import meet
from tqdm import tqdm, trange
import mtCSP


data_folder = sys.argv[1]
result_folder = sys.argv[2]
N_subjects = 21
s_rate = 1000 # sampling rate of the EEG

channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
N_channels = len(channames)

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
#regularization parameters for mtCSP
lam1 = 0 # penalizes the size of the global filter
lam2 = 0.5 # penalizes the size of the specific filter
lam2_cand = np.logspace(-5,5,num=11)
get_filters = 5

CSP_eigvals = [ ] #5[(10,)], (get_filters,) for each band
CSP_filters = [] #5[20[(32,10)]], [(N_channels,get_filters) for each subject] for each band
CSP_patterns = [] #5[21[(10,32)]], [(get_filters,N_channels) for each subject(+1 global in front)] for each band
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
    for band_idx, band_name in enumerate(band_names[0]):
        contrast_cov_a = [t[band_idx] for t in contrast_covs]
        target_cov_a = [t[band_idx] for t in target_covs]
        target_cov_avg = np.mean([c.mean(-1) for c in target_cov_a], 0) #trial and subject avg
        # get whitening matrix for the average target covariance matrix
        rank = np.linalg.matrix_rank(target_cov_avg)
        eigval, eigvect = scipy.linalg.eigh(target_cov_avg)
        W = eigvect[:,-rank:]/np.sqrt(eigval[-rank:])
        v_norm = []
        for lam2 in lam2_cand:
            for it in trange(get_filters): #last 5 filters are ERD (contrast, target)
                if it == 0:
                    quot_now, filters_now = mtCSP.maximize_mtCSP(
                            [W.T @ c.mean(-1) @ W for c in contrast_cov_a],
                            [W.T @ c.mean(-1) @ W for c in target_cov_a],
                            lam1,
                            lam2,
                            iterations=20)
                    quot = [quot_now]
                    all_filters = filters_now.reshape(-1, 1)
                else: #force other filters to be orthogonal
                    quot_now, filters_now = mtCSP.maximize_mtCSP(
                            [W.T @ c.mean(-1) @ W for c in contrast_cov_a],
                            [W.T @ c.mean(-1) @ W for c in target_cov_a],
                            lam1,
                            lam2,
                            old_W = all_filters,
                            iterations=20)
                    quot.append(quot_now)
                    all_filters = np.hstack([all_filters, filters_now.reshape(-1, 1)])
            #erd und ers also orthogonal to each other to reduce later covariability
            # for it in trange(get_filters): #first 5 filters are ERS (target, contrast)
            #     if it == 0:
            #         quot_now, filters_now = mtCSP.maximize_mtCSP(
            #                 [W.T @ c.mean(-1) @ W for c in target_cov_a],
            #                 [W.T @ c.mean(-1) @ W for c in contrast_cov_a],
            #                 lam1,
            #                 lam2,
            #                 iterations=20)
            #         quot = [quot_now]
            #         all_filters = filters_now.reshape(-1, 1)
            #     else: #force other filters to be orthogonal
            #         quot_now, filters_now = mtCSP.maximize_mtCSP(
            #                 [W.T @ c.mean(-1) @ W for c in target_cov_a],
            #                 [W.T @ c.mean(-1) @ W for c in contrast_cov_a],
            #                 lam1,
            #                 lam2,
            #                 old_W = all_filters,
            #                 iterations=20)
            #         quot.append(quot_now)
            #         all_filters = np.hstack([all_filters, filters_now.reshape(-1, 1)])

            # transform the filters
            all_filters = np.vstack([W @ all_filters[i * rank : (i + 1) * rank]
                for i in range(len(target_cov_a) + 1)]) #(21*32,5)
            # calculate the composite filters for every subject
            subject_filters = [all_filters[:N_channels] +
                               all_filters[(i + 1) * N_channels:(i + 2) * N_channels]
                               for i in range(len(target_cov_a))]
            CSP_filters.append(subject_filters) #(20,32,10)

            # calculate the SNNR (EV equivalent)for every component and subject
            SNNR_per_subject = np.array([
                np.diag((filt.T @ target_now.mean(-1) @ filt) /
                        (filt.T @ contrast_now.mean(-1) @ filt))
                for (filt, target_now, contrast_now) in
                zip(subject_filters, target_cov_a, contrast_cov_a)])
            SNNR = SNNR_per_subject.mean(0)
            CSP_eigvals.append(SNNR) #(10,)

            # patterns
            CSP_pattern_subjects = [scipy.linalg.solve(
                    CSP_filter.T.dot(target_subj.mean(-1)).dot(CSP_filter),
                    CSP_filter.T.dot(target_subj.mean(-1)))
                    for CSP_filter, target_subj in zip(subject_filters, target_cov_a)]
            global_filters = all_filters[:N_channels]
            CSP_pattern_global = scipy.linalg.solve(
                    global_filters.T.dot(target_cov_avg).dot(global_filters),
                    global_filters.T.dot(target_cov_avg))
            CSP_pattern = np.vstack([[CSP_pattern_global],CSP_pattern_subjects])
            ### normalize the patterns such that Cz is always positive
            CSP_pattern = [p*np.sign(p[:,np.asarray(channames)=='CZ'])
                    for p in CSP_pattern]
            CSP_patterns.append(CSP_pattern) #(21, 5, 32)

            #plot filters over channel (should be similar enough)
            for s in range(N_subjects-1):
                f_now = subject_filters[s][:,0]
                plt.plot(channames, (f_now-np.mean(f_now))/np.std(f_now))
            plt.savefig(os.path.join(result_folder,'motor/CSP_filters{}_lam{}.pdf'.format(band_names[0],lam2)))
            # looking at first filter of ERD for now
            # villeicht all_filters stattdessen? Gunnar meinte, nur die individuellen filter!
            v_norm.append(
                np.sqrt(np.sum(np.square(np.array(subject_filters)[:,:,0]))))

        # plot SNNR and ||v||2 for first filter (ERD)
        plt.figure()
        plt.scatter(np.log([x[0] for x in CSP_eigvals]), np.log(v_norm), label=str(lam2)) #one point per lambda
        plt.legend()
        plt.savefig(os.path.join(result_folder,'motor/CSP_Lcurve.pdf'))
    1/0
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
            with np.load(os.path.join(result_folder,'motor/erdcsp.npz')) as f:
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
            EEG_CSP_subj = np.tensordot(CSP_filters[band_idx][idx], eeg_filtbp,
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
