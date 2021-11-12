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

## plot
color0 = '#543005'.upper()#dark brown
color1 = '#8c510a'.upper()
color2 = '#bf812d'.upper()
color3 = '#dfc27d'.upper()
color4 = '#f6e8c3'.upper() #light brown
color5 = '#c7eae5'.upper() #light blue
color6 = '#80cdc1'.upper()
color7 = '#35978f'.upper()
color8 = '#01665e'.upper()
color9 = '#003c30'.upper() #dark blue
colors = [color0, color1, color2, color3, color4, color5, color6, color7, color8, color9]

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)


##### read ERD data ####
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
            with np.load('Results/motor/CSP.npz') as f:
                CSP_eigvals.append(f['CSP_eigvals{:s}'.format(band_names[i])])
                CSP_filters.append(f['CSP_filters{:s}'.format(band_names[i])])
                CSP_patterns.append(f['CSP_patterns{:s}'.format(band_names[i])])
            i+=1
        except IndexError:
            break
    print('CSP succesfully read.')
except FileNotFoundError: # read ERD data and calculate CSP
    print('calculating CSP...')
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
    np.savez(os.path.join(result_folder, 'motor', 'CSP.npz'), **save_results)


##### apply CSP to EEG (takes a while) #####
ERD_CSP = [] # stores trial averaged ERD/S_CSP per subject, each with shape (Nband, CSPcomp,time)
ERDCSP_trial = [] #stores ERD_CSP of best CSPcomp per subject,each shape (Nband, Ntrial)
ERSCSP_trial = [] # same for ERS
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
        EEG_CSP_subj = np.tensordot(CSP_filters[band_idx], eeg_filtbp, axes=(0,0))
        # Hilbert-Transform, absolute value (could also be abs**2)
        eeg_filtHil = np.abs(scipy.signal.hilbert(EEG_CSP_subj, axis=-1))
        # normalize s.t.2000ms preresponse are 100%
        snareHit_times = all_snareHit_times[idx]
        wdBlkHit_times = all_wdBlkHit_times[idx]
        all_trials_filt = meet.epochEEG(eeg_filtHil,
                np.r_[snareHit_times[snareInlier[idx]],
                    wdBlkHit_times[wdBlkInlier[idx]]],
                win) # (ERDcomp, time, trials) = (31, 2500, 143)
        # calculate trial averaged ERDCSP
        ERD_CSP_subj_band = all_trials_filt.mean(-1) # trial average
        ERD_CSP_subj_band /= ERD_CSP_subj_band[:,0:750].mean(1)[:,np.newaxis] #baseline avg should be 100%
        ERD_CSP_subj_band *= 100 # ERD in percent
        ERD_CSP_subj.append(ERD_CSP_subj_band)

        # calculate ERD in percent per subject and trial
        # ERD (ERS) = percentage of power decrease (increase):
        # ERD% = (A-R)/R*100
        # A event period power 1250-2000 (i.e.(-750 to 0 because we want pre stimulus)))
        # R reference period power 0-750 (i.e. -2000:-1250 samples),
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


##### plot ERDCSP components #####
# look at plot to determine number of components
for i,ev in enumerate(CSP_eigvals):
    plt.plot(ev, 'o')
    plt.title('CSP EV band {}, small ERD, large ERS'.format(band_names[1]))
    plt.show()
# first argument is pre movement so
CSP_ERDnums = [2,5,5,5,5] # e.g. [:3] # small EV 2 or 5 mostly both work
CSP_ERSnums = [4,4,4,4,5] #e.g. [-4:] # large EV

for band_idx, band_name in enumerate(band_names):
    # average over subjects
    ERD_CSP_subjmean = np.mean([e[band_idx] for e in ERD_CSP], axis=0)
    ev = CSP_eigvals[band_idx]
    CSP_ERDnum = CSP_ERDnums[band_idx]
    CSP_ERSnum = CSP_ERSnums[band_idx]

    # plot CSP components
    erd_t = range(win[0], win[1])
    plt.figure()
    # plot in order from top to bottom
    for s in range(CSP_ERSnum): #0,1,2,3
        plt.plot(erd_t, ERD_CSP_subjmean[s,:].T,
            label='ERS %d' % (s+1) + ' ($\mathrm{EV=%.2fdB}$)' % round(
            10*np.log10(ev[s]),2), color=colors[s])
    for d in range(-CSP_ERDnum,0,1): #-3,-2,-1
        plt.plot(erd_t, ERD_CSP_subjmean[d,:].T,
            label='ERD %d' % (-d) + ' ($\mathrm{EV=%.2fdB}$)' % round(
            10*np.log10(ev[d]), 2), color=colors[d])
    plt.plot(erd_t, ERD_CSP_subjmean[CSP_ERSnum:-CSP_ERDnum,:].T,
        c='black', alpha=0.1)
    plt.legend(fontsize=8)
    plt.xlabel('time around response [ms]', fontsize=10)
    plt.ylabel('CSP filtered EEG, relative amplitude [%]', fontsize=10)
    plt.title('subj.-avg. and eeg-applied CSP components {} Hz]'.format(
        band_name[:-1]), fontsize=12)
    plt.savefig(os.path.join(result_folder,
        'motor/erdcsp_comp{}.pdf'.format(band_name)))

##### plot EV and CSP patterns #####
for band_idx, band_name in enumerate(band_names):
    ev = CSP_eigvals[band_idx]

    potmaps = [meet.sphere.potMap(chancoords, pat_now,
        projection='stereographic') for pat_now in CSP_patterns[band_idx]]
    h1 = 1 #ev
    h2 = 1.3 #ERS
    h3 = 1.3 #ERD
    h4 = 0.1 #colorbar

    fig = plt.figure(figsize = (5.512,5.512))
    gs = mpl.gridspec.GridSpec(4,1, height_ratios = [h1,h2,h3,h4])

    SNNR_ax = fig.add_subplot(gs[0,:])
    SNNR_ax.plot(range(1,len(ev) + 1), 10*np.log10(ev), 'ko-', lw=2,
            markersize=5)
    for d in range(4): # small EV for ERD
        SNNR_ax.scatter([d+1], 10*np.log10(ev[d]),
        c=colors[d], s=60, zorder=1000)
    for s in range(-4,0):#[-4, -3, -2, -1]
        SNNR_ax.scatter([len(ev)+s+1], 10*np.log10(ev[s]),
        c=colors[s], s=60, zorder=1000)
    SNNR_ax.axhline(0, c='k', lw=1)
    SNNR_ax.set_xlim([0.5, len(ev)+0.5])
    SNNR_ax.set_xticks(np.r_[1,range(5, len(ev) + 1, 5)])
    SNNR_ax.set_ylabel('SNR (dB)')
    SNNR_ax.set_xlabel('component (index)')
    SNNR_ax.set_title('resulting SNR after CSP for band ' + band_name)

    # plot the four spatial patterns for ERD
    gs3 = mpl.gridspec.GridSpecFromSubplotSpec(2,4, gs[2,:],
            height_ratios = [1,0.1], wspace=0, hspace=0.8)
    head_ax = []
    pc = []
    for d, pat in enumerate(reversed(potmaps[-4:])): # take last 4, reverse, then enumerate
        try:
            head_ax.append(fig.add_subplot(gs3[0,d], sharex=head_ax[0],
                sharey=head_ax[0], frame_on=False, aspect='equal'))
        except IndexError:
            head_ax.append(fig.add_subplot(gs3[0,d], frame_on=False, aspect='equal'))
        Z = pat[2]/np.abs(pat[2]).max()
        pc.append(head_ax[-1].pcolormesh(
            *pat[:2], Z, rasterized=True, cmap='coolwarm',
            vmin=-1, vmax=1, shading='auto'))
        head_ax[-1].contour(*pat, levels=[0], colors='w')
        head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
                alpha=0.5, zorder=1001)
        head_ax[-1].set_xlabel('ERD %d' % (d + 1) +'\n'+
                '($\mathrm{SNR=%.2fdB}$)' % (10*np.log10(ev[-(d+1)])),
                fontsize=8)
        head_ax[-1].tick_params(**blind_ax)
        meet.sphere.addHead(head_ax[-1], ec=colors[d], zorder=1000, lw=3)
    head_ax[0].set_ylim([-1.1,1.3])
    head_ax[0].set_xlim([-1.6,1.6])

    # plot the four spatial patterns for ERS
    gs2 = mpl.gridspec.GridSpecFromSubplotSpec(2,4, gs[1,:],
            height_ratios = [1,0.1], wspace=0, hspace=0.8)
    head_ax = []
    pc = []
    for s, pat in enumerate(potmaps[:4]):
        try:
            head_ax.append(fig.add_subplot(gs2[0,s], sharex=head_ax[0],
                sharey=head_ax[0], frame_on=False, aspect='equal'))
        except IndexError:
            head_ax.append(fig.add_subplot(gs2[0,s], frame_on=False, aspect='equal'))
        Z = pat[2]/np.abs(pat[2]).max()
        pc.append(head_ax[-1].pcolormesh(
            *pat[:2], Z, rasterized=True, cmap='coolwarm',
            vmin=-1, vmax=1, shading='auto'))
        head_ax[-1].contour(*pat, levels=[0], colors='w')
        head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
                alpha=0.5, zorder=1001)
        head_ax[-1].set_xlabel('ERS %d' % (s + 1) +'\n'+
                '($\mathrm{SNR=%.2fdB}$)' % (10*np.log10(ev[s])),
                fontsize=8)
        head_ax[-1].tick_params(**blind_ax)
        meet.sphere.addHead(head_ax[-1], ec=colors[-(s+1)], zorder=1000, lw=3)
    head_ax[0].set_ylim([-1.1,1.3])
    head_ax[0].set_xlim([-1.6,1.6])

    # add a colorbar
    cbar_ax = fig.add_subplot(gs[3,:])
    cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
            label='amplitude (a.u.)', ticks=[-1,0,1])
    cbar.ax.set_xticklabels(['-', '0', '+'])
    cbar.ax.axvline(0, c='w', lw=2)

    '''spect_ax = fig.add_subplot(gs[2,:])
    [spect_ax.plot(f,
        10*np.log10(CSP_filters[:,i].dot(CSP_filters[:,i].dot(
            np.mean([t/np.trace(t[...,contrast_idx].mean(-1)).real
                for t in poststim_norm_csd], 0).real))),
            c=colors[i], lw=2) for i in range(4)]
    spect_ax.set_xlim([0.5, 8])
    spect_ax.set_ylim([-1.1, 1.1])
    spect_ax.axhline(0, c='k', lw=1)
    spect_ax.set_xlabel('frequency (Hz)')
    spect_ax.set_ylabel('SNR (dB)')
    spect_ax.set_title('normalized spectrum')

    spect_ax.axvline(snareFreq, color='b', zorder=0, lw=1)
    spect_ax.axvline(2*snareFreq, color='b', zorder=0, lw=1)
    spect_ax.axvline(wdBlkFreq, color='r', zorder=0, lw=1)
    spect_ax.axvline(2*wdBlkFreq, color='k', zorder=0, lw=1)
    spect_ax.axvline(4*wdBlkFreq, color='k', zorder=0, lw=1)'''

    gs.tight_layout(fig, pad=0.5)#, pad=0.2, h_pad=0.8

    fig.savefig(os.path.join(result_folder,
        'motor/CSP_patterns{}.pdf'.format(band_name)))
