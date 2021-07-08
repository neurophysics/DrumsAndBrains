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
## plot
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 10
cmap = 'plasma'
color1 = '#1f78b4'.upper()
color2 = '#33a02c'.upper()
color3 = '#b2df8a'.upper()
color4 = '#a6cee3'.upper()
colors=[color1, color2, color3, color4]

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)

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
        all_contrast_cov)

# patterns
CSP_patterns = scipy.linalg.solve(
        CSP_filters.T.dot(all_target_cov).dot(CSP_filters),
        CSP_filters.T.dot(all_target_cov))

### normalize the patterns such that Cz is always positive
#@gunnar: we did this for ssd, here as well? would make sense to me
CSP_patterns*=np.sign(CSP_patterns[:,np.asarray(channames)=='CZ'])


np.savez(os.path.join(result_folder, 'motor/CSP.npz'),
        CSP_eigvals = CSP_eigvals,
        CSP_filters = CSP_filters, # matrix W in paper
        CSP_patterns = CSP_patterns # matrix A in paper
        )

# plot eigenvalues

'''with np.load('Results/motor/CSP.npz') as f:
    CSP_eigvals = f['CSP_eigvals']
    CSP_filters = f['CSP_filters']
    CSP_patterns = f['CSP_patterns']'''


plt.plot(CSP_eigvals, 'o')
plt.title('CSP EV, small ERD, large ERS')
# first argument is pre movement so
# small EV for ERD: here 1 or maybe 3
# large EV for ERS: here 2 or 4
CSP_ERDnum = 3
CSP_ERSnum = 4

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

save_ERDCSP = {}
for i, e in enumerate(ERD_CSP):
    save_ERDCSP['ERDCSP_{:02d}'.format(i)] = e
np.savez(os.path.join(result_folder, 'motor/erdcsp.npz'), **save_ERDCSP)

# plot CSP components
erd_t = range(ERD_CSP[0].shape[1])
plt.figure()
for d in range(CSP_ERDnum):
    plt.plot(erd_t, ERD_CSP_subjmean[-(d+1),:].T,
        label='ERD {}, EV={}dB'.format(d+1, round(CSP_eigvals[-d], 2)))
for s in range(CSP_ERSnum):
    plt.plot(erd_t, ERD_CSP_subjmean[s,:].T,
        label='ERS {}, EV={}dB'.format(s+1, round(CSP_eigvals[s],2)))
plt.plot(erd_t, ERD_CSP_subjmean[CSP_ERSnum:-CSP_ERDnum,:].T, c='black', alpha=0.1)
plt.legend()
plt.title('subj.-avg. and eeg-applied CSP components')
plt.savefig(os.path.join(result_folder, 'motor/erdcsp.pdf'))

# plot ev and spatial patterns
potmaps = [meet.sphere.potMap(chancoords, pat_now,
    projection='stereographic') for pat_now in CSP_patterns]
h1 = 1 #ev
h2 = 1.3 #ERS
h3 = 1.3 #ERD
h4 = 0.1 #colorbar
fig = plt.figure(figsize = (5.512,5.512))
gs = mpl.gridspec.GridSpec(4,1, height_ratios = [h1,h2,h3,h4])

SNNR_ax = fig.add_subplot(gs[0,:])
SNNR_ax.plot(range(1,len(CSP_eigvals) + 1), 10*np.log10(CSP_eigvals), 'ko-', lw=2,
        markersize=5)
SNNR_ax.scatter([1], 10*np.log10(CSP_eigvals[0]), c=color1, s=60, zorder=1000)
SNNR_ax.scatter([2], 10*np.log10(CSP_eigvals[1]), c=color2, s=60, zorder=1000)
SNNR_ax.scatter([3], 10*np.log10(CSP_eigvals[2]), c=color3, s=60, zorder=1000)
SNNR_ax.scatter([4], 10*np.log10(CSP_eigvals[3]), c=color4, s=60, zorder=1000)
SNNR_ax.scatter([len(CSP_eigvals)], 10*np.log10(CSP_eigvals[-1]), c=color1, s=60, zorder=1000)
SNNR_ax.scatter([len(CSP_eigvals)-1], 10*np.log10(CSP_eigvals[-2]), c=color2, s=60, zorder=1000)
SNNR_ax.scatter([len(CSP_eigvals)-2], 10*np.log10(CSP_eigvals[-3]), c=color3, s=60, zorder=1000)
SNNR_ax.scatter([len(CSP_eigvals)-3], 10*np.log10(CSP_eigvals[-4]), c=color4, s=60, zorder=1000)
SNNR_ax.axhline(0, c='k', lw=1)
SNNR_ax.set_xlim([0.5, len(CSP_eigvals)+0.5])
SNNR_ax.set_xticks(np.r_[1,range(5, len(CSP_eigvals) + 1, 5)])
SNNR_ax.set_ylabel('SNR (dB)')
SNNR_ax.set_xlabel('component (index)')
SNNR_ax.set_title('resulting SNR after CSP')

# plot the four spatial patterns for ERS
gs2 = mpl.gridspec.GridSpecFromSubplotSpec(2,4, gs[1,:],
        height_ratios = [1,0.1], wspace=0, hspace=0.8)
head_ax = []
pc = []
for i, pat in enumerate(potmaps[:4]):
    try:
        head_ax.append(fig.add_subplot(gs2[0,i], sharex=head_ax[0],
            sharey=head_ax[0], frame_on=False, aspect='equal'))
    except IndexError:
        head_ax.append(fig.add_subplot(gs2[0,i], frame_on=False, aspect='equal'))
    Z = pat[2]/np.abs(pat[2]).max()
    pc.append(head_ax[-1].pcolormesh(
        *pat[:2], Z, rasterized=True, cmap='coolwarm',
        vmin=-1, vmax=1, shading='auto'))
    head_ax[-1].contour(*pat, levels=[0], colors='w')
    head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5, zorder=1001)
    head_ax[-1].set_xlabel('ERS %d' % (i + 1) +'\n'+
            '($\mathrm{SNR=%.2f\ dB}$)' % (10*np.log10(CSP_eigvals[i])))
    head_ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(head_ax[-1], ec=colors[i], zorder=1000, lw=3)
head_ax[0].set_ylim([-1.1,1.3])
head_ax[0].set_xlim([-1.6,1.6])

# plot the four spatial patterns for ERD
gs3 = mpl.gridspec.GridSpecFromSubplotSpec(2,4, gs[2,:],
        height_ratios = [1,0.1], wspace=0, hspace=0.8)
head_ax = []
pc = []
for i, pat in enumerate(reversed(potmaps[-4:])): # take last 4, reverse, then enumerate
    try:
        head_ax.append(fig.add_subplot(gs3[0,i], sharex=head_ax[0],
            sharey=head_ax[0], frame_on=False, aspect='equal'))
    except IndexError:
        head_ax.append(fig.add_subplot(gs3[0,i], frame_on=False, aspect='equal'))
    Z = pat[2]/np.abs(pat[2]).max()
    pc.append(head_ax[-1].pcolormesh(
        *pat[:2], Z, rasterized=True, cmap='coolwarm',
        vmin=-1, vmax=1, shading='auto'))
    head_ax[-1].contour(*pat, levels=[0], colors='w')
    head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5, zorder=1001)
    head_ax[-1].set_xlabel('ERD %d' % (i + 1) +'\n'+
            '($\mathrm{SNR=%.2f\ dB}$)' % (10*np.log10(CSP_eigvals[-(i+1)])))
    head_ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(head_ax[-1], ec=colors[i], zorder=1000, lw=3)
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

fig.savefig(os.path.join(result_folder, 'motor/CSP_patterns.pdf'))
