"""
plots:
BPLDA.pdf
mtCSP_filter_optLam2.pdf
erdmtcsp_comp[7-12].pdf, erdmtcsp_comp[15-25].pdf
mtCSP_patterns[7-12].pdf, mtCSP_patterns[15-25].pdf
mtCSP[7-12].pdf, mtCSP[15-25].pdf
patternsByTime.pdf
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

## plot
plt.rcParams.update({'font.size': 8})
color0 = '#543005'.upper() #dark brown, ERS1
color1 = '#8c510a'.upper()
color2 = '#bf812d'.upper()
color3 = '#dfc27d'.upper()
color4 = '#f6e8c3'.upper() #light brown, ERS5
color5 = '#c7eae5'.upper() #light blue, ERD5
color6 = '#80cdc1'.upper()
color7 = '#35978f'.upper()
color8 = '#01665e'.upper()
color9 = '#003c30'.upper() #dark blue, ERD1
colors = [color0, color1, color2, color3, color4, color5, color6, color7, color8, color9]

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)

##### read motor inlier and win #####
#snareInlier = [] # list of 20 subjects, each shape ≤75
#wdBlkInlier = [] #same for wdBlk
# i=0
# while True:
#     try:
with np.load(os.path.join(result_folder, 'motor/inlier.npz'),
    'r') as fi:
    #snareInlier.append(fi['snareInlier_response_{:02d}'.format(i)])
    #wdBlkInlier.append(fi['wdBlkInlier_response_{:02d}'.format(i)])
    win = fi['win']
with np.load(os.path.join(result_folder, 'motor/covmat.npz'),
    'r') as f_covmat:
    fbands = f_covmat['fbands']
    base_idx = f_covmat['base_idx'] #corresponds to -2000 to -1250ms
    act_idx = f_covmat['act_idx'] #corresponds to -750 to 0ms
    act_idx_lda = f_covmat['act_idx_lda'] #corresponds to -600 to -1000ms
    #     i+=1
    # except FileNotFoundError:
    #     break

erd_t = range(win[0], win[1])
base_ms = [t+win[0] for t in base_idx] #-2000 to 500 milliseconds
act_ms = [t+win[0] for t in act_idx] #-2000 to 500

##### read unfiltered BP adn ERD #####
BPs = [] #stores BP per subject, each shape (N_channel, time)=(32,2500)
ERDs = [] #stores ERD per subject, each shape (band,Ncomp, time)=(5,32,2500)
#all_BP_trials = []
i=0
while True: #loop over subjects
    try:
        with np.load(os.path.join(result_folder, 'motor/BP.npz'),
            'r') as f_BP:
            BPs.append(f_BP['BP_{:02d}'.format(i)])
            #all_BP_trials.append(f_BP['BP_trials_{:02d}'.format(i)])
        with np.load(os.path.join(result_folder, 'motor/ERD.npz'),
            'r') as f_ERD:
            ERDs.append(f_ERD['ERD_{:02d}'.format(i)])
        i+=1
    except KeyError:
        break

##### read CSP #####
# always first ERD, then ERS filter
CSP_eigvals = [] #stores EV per band, each shape (N_filters*2,)=(10,)
CSP_filters = [] #stores filters per band, each shape (N_subjects,;N_channels,N_filters*2)=(20,32,10)
CSP_patterns = [] #store spatterns per band, each shape (N_subjects+1global,N_filters*2,N_channels)=(21,10,32)
try:
    i=0 #loop over bands
    while True:
        try:
            with np.load(os.path.join(result_folder,'motor/mtCSP.npz')) as f:
                band_names = f['band_names']
                CSP_eigvals.append(f['CSP_eigvals{:s}'.format(band_names[i])])
                CSP_filters.append(f['CSP_filters{:s}'.format(band_names[i])])
                CSP_patterns.append(f['CSP_patterns{:s}'.format(band_names[i])])
            i+=1
        except IndexError:
            break
    print('CSP and BP succesfully read.')
except FileNotFoundError: # read ERD data and calculate CSP
    print('Please run basicAnalysis_motor.py and csp.py first.')

cfilt = np.load(os.path.join(result_folder,'motor/lda.npy'))

N_filters = int(CSP_filters[0].shape[-1]/2) # have erd and ers
CSP_ERDnum = N_filters
CSP_ERSnum = N_filters
N_bands = len(band_names)

##### read BP and ERD_CSP ######
ERD_CSP = [] # stores trial averaged ERD/S_CSP per subject, each with shape (Nband, N_filters*2,time)
ERDCSP_trial = [] #stores ERD_CSP of best CSPcomp per subject,each shape (Nband, Ntrial)
ERSCSP_trial = [] # same for ERS
try:
    i=0 #loop over subjects
    while True:
        try:
            with np.load(os.path.join(result_folder,'motor/ERDCSP.npz')) as f:
                ERD_CSP.append(f['ERDCSP_{:02d}'.format(i)])
                ERDCSP_trial.append(f['ERDCSP_trial_{:02d}'.format(i)])
                ERSCSP_trial.append(f['ERSCSP_trial_{:02d}'.format(i)])
            i+=1
        except KeyError:
            break
    print('ERDCSP succesfully read.')
except FileNotFoundError: # read ERD data and calculate CSP
    print('Please run csp.py first.')


##### BPLDA.pdf: plot filter for lam2=3  #####
BPlda = [np.tensordot(cfilt, b, axes=(0,0)) for b in BPs] #each shape (2500,) now
pat = meet.sphere.potMap(chancoords, cfilt, projection='stereographic')
# plot for each subject
s_rate = 1000
vmax = np.max(np.abs(cfilt)) #0 is in the middle, scale for plotting potmaps

all_cueHit_diff_mean = []
all_cueHit_diff_sd = []
for subj_idx in range(N_subjects-1): #only have 20 entries for subject 11 is missing
    if subj_idx < 10: #subject idx 10 already belongs to subject 12
        subj = subj_idx+1
    else:
        subj = subj_idx+2
    # find hit area
    with np.load(os.path.join(result_folder,'S%02d'%subj, 'behavioural_results.npz'),
            'r', allow_pickle=True, encoding='latin1') as f:
        bar_duration = f['bar_duration']
        snare_deviation = f['snare_deviation']
        wdBlk_deviation = f['wdBlk_deviation']
    snare_cueHitdiff = ((0.5 * bar_duration + snare_deviation) * s_rate)
    wdBlk_cueHitdiff = ((2./3 * bar_duration + wdBlk_deviation) * s_rate)
    cueHit_diff = np.hstack([snare_cueHitdiff, wdBlk_cueHitdiff])
    cueHit_diff_mean = np.nanmean(cueHit_diff)
    all_cueHit_diff_mean.append(cueHit_diff_mean)
    cueHit_diff_sd = np.nanstd(cueHit_diff)
    all_cueHit_diff_sd.append(cueHit_diff_sd)

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(erd_t, BPlda[subj_idx], color='k')
    plt.title('LDA filtered BP, Subject %02d'%subj)
    plt.axvspan(-(cueHit_diff_mean-cueHit_diff_sd),
        -(cueHit_diff_mean+cueHit_diff_sd),
        alpha=1, color='bisque', label='mean cue time ± sd')
    plt.axvspan(act_idx_lda[0]+win[0], act_idx_lda[-1]+win[0],
        alpha=0.5, color='lightsalmon',
            label='activation window')
    plt.axvline(0, lw=0.5, c='r')
    plt.axhline(0, lw=0.5, c='k', ls=':')
    plt.legend(loc='upper center')
    plt.xlabel('time around response [ms]')
    plt.ylabel('amplitude [a.u.]')
    # insert axis for pattern
    head_ax = ax.inset_axes([0.,0.15,0.4,0.4]) #[x0, y0, width, height]
    head_ax.tick_params(**blind_ax)
    pc = []
    pc.append(head_ax.pcolormesh(
        *pat, rasterized=True, cmap='coolwarm',
        vmin=-vmax, vmax=vmax, shading='auto'))
    head_ax.contour(*pat, levels=[0], colors='w')
    head_ax.scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5, zorder=1001)
    meet.sphere.addHead(head_ax, ec='black', zorder=1000, lw=2)
    head_ax.set_ylim([-1.1,1.3])
    head_ax.set_xlim([-1.6,1.6])
    head_ax.set_title('LDA Pattern')
    # insert axis for colorbar
    bar_ax = ax.inset_axes([0.,0.13,0.4,0.02]) #[x0, y0, width, height]
    fig.colorbar(pc[-1], cax=bar_ax, orientation='horizontal',
            label='amplitude (a.u.)')
    plt.savefig(os.path.join(result_folder, 'S%02d'% subj, 'motor_BPLDA.pdf'))

# plot subject averaged
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(erd_t, np.mean(BPlda,axis=0),color='k')
plt.title('LDA filtered BP, subj.-avg.')
plt.axvline(0, lw=0.5, c='r')
plt.axhline(0, lw=0.5, c='k', ls=':')
plt.axvspan(-(np.mean(all_cueHit_diff_mean)-np.mean(cueHit_diff_sd)),
    -(np.mean(cueHit_diff_mean)+np.mean(cueHit_diff_sd)),
    alpha=1, color='bisque', label='mean cue time ± sd')
plt.axvspan(act_idx_lda[0]+win[0], act_idx_lda[-1]+win[0],
    alpha=0.5, color='lightsalmon',
        label='activation window')
plt.xlabel('time around response [ms]')
plt.ylabel('amplitude [a.u.]')
plt.legend(loc='upper center')
# insert axis for pattern
head_ax = ax.inset_axes([0.,0.15,0.4,0.4]) #[x0, y0, width, height]
head_ax.tick_params(**blind_ax)
pc = []
pc.append(head_ax.pcolormesh(
    *pat, rasterized=True, cmap='coolwarm',
    vmin=-vmax, vmax=vmax, shading='auto'))
head_ax.contour(*pat, levels=[0], colors='w')
head_ax.scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
        alpha=0.5, zorder=1001)
meet.sphere.addHead(head_ax, ec='black', zorder=1000, lw=2)
head_ax.set_ylim([-1.1,1.3])
head_ax.set_xlim([-1.6,1.6])
head_ax.set_title('LDA Pattern')
# insert axis for colorbar
bar_ax = ax.inset_axes([0.,0.13,0.4,0.02]) #[x0, y0, width, height]
fig.colorbar(pc[-1], cax=bar_ax, orientation='horizontal',
        label='amplitude (a.u.)')
plt.savefig(os.path.join(result_folder, 'motor/BPLDA.pdf'))

plt.close('all')


##### mtCSP_filter_optLam2.pdf: plot filter for lam2=3  #####
# currently only for ERD, add for ERS?
fig, axs = plt.subplots(N_bands, 2, figsize=(10,10),
        sharex=True, sharey=False)
fig.subplots_adjust(top=0.94, bottom=0.05, left=0.05, right=0.99,
    hspace=0.3, wspace=0.1)
fig.suptitle(r'Best mtCSP filter for every subject with lam2=3', fontsize=12)
for band_idx, band_name in enumerate(band_names):
    axs[band_idx,0].set_title('ERD, {} Hz'.format(band_name[1:-1]))
    for s in range(N_subjects-1):
        f_now = CSP_filters[band_idx][s,:,0]
        axs[band_idx,0].plot(channames, (f_now-np.mean(f_now))/np.std(f_now))
axs[band_idx,0].set_xlabel('channel')
for band_idx, band_name in enumerate(band_names):
    axs[band_idx,1].set_title('ERS, {} Hz'.format(band_name[1:-1]))
    for s in range(N_subjects-1):
        f_now = CSP_filters[band_idx][s,:,N_filters] #ERS filter after ERD filter
        axs[band_idx,1].plot(channames, (f_now-np.mean(f_now))/np.std(f_now))
plt.xticks(ticks=range(len(channames)),labels=[])
axs[band_idx,1].set_xlabel('channel')
fig.savefig(os.path.join(result_folder, 'motor/mtCSP_filter_optLam2.pdf'))


##### erdmtcsp_comp[1-4].pdf: plot ERDCSP components #####
# look at plot to determine number of components
# for i,ev in enumerate(CSP_eigvals):
#     plt.plot(ev, 'o')
#     plt.title('CSP EV band {}, small ERD, large ERS'.format(band_names[1]))
#     plt.show()
# first argument is pre movement so

for band_idx, band_name in enumerate(band_names):
    ev = CSP_eigvals[band_idx]

    # average over subjects and normalize
    ERD_CSP_subjmean = np.mean([e[band_idx] for e in ERD_CSP], axis=0)
    ERD_CSP_subjmean /= ERD_CSP_subjmean[:,base_idx].mean(1)[:,np.newaxis] #baseline avg should be 100%
    ERD_CSP_subjmean *= 100 # ERD in percent

    # plot CSP components
    plt.figure()
    # plot in order from top to bottom
    for s in range(CSP_ERSnum): #0,1,2,3,4
        plt.plot(erd_t, ERD_CSP_subjmean[N_filters+s,:].T,
            label='ERS %d' % (s+1) + ' ({}\%)'.format(round(ev[N_filters+s]*100)),
            color=colors[s])
    for d in range(CSP_ERDnum-1,-1,-1): #4,3,2,1,0
        plt.plot(erd_t, ERD_CSP_subjmean[d,:].T,
            label='ERD %d' % (d+1) + ' ({}\%)'.format(round(ev[d]*100)),
            color=colors[-(d+1)])
    plt.plot(erd_t, ERD_CSP_subjmean[CSP_ERSnum:-CSP_ERDnum,:].T,
        c='black', alpha=0.1)
    plt.axvspan(base_ms[0], base_ms[-1], alpha=0.3, color=colors[4],
        label='contrast period') #_ gets ignored as label
    plt.axvspan(act_ms[0], act_ms[-1], alpha=0.3, color=colors[5],
        label='target period')
    plt.legend(fontsize=8)
    plt.xlabel('time around response [ms]', fontsize=10)
    plt.ylabel('CSP filtered EEG, relative amplitude [\%]', fontsize=10)
    plt.title('subj.-avg. and eeg-applied CSP filter {} Hz]'.format(
        band_name[:-1]), fontsize=12)
    plt.savefig(os.path.join(result_folder,
        'motor/erdmtcsp_comp{}.pdf'.format(band_name)))


##### mtCSP_patterns[1-4].pdf:  plot EV and global CSP patterns #####
# calculating the potmaps, takes a while
potmaps_csp = []
for band_idx, band_name in enumerate(band_names):
    potmaps = [meet.sphere.potMap(chancoords, pat_now,
        projection='stereographic') for pat_now in CSP_patterns[band_idx][0]] #at 0 because that is the global one
    potmaps_csp.append(potmaps)

for band_idx, band_name in enumerate(band_names):
    ev = CSP_eigvals[band_idx] #order: ERD1..ERD5,ERS1..ERS5
    #ev_reordered = np.hstack([ev[N_filters:],ev[::-1][N_filters:]]) #order as colors: ERS1..ERS5,ERD5..ERD1
    potmaps = potmaps_csp[band_idx]
    h1 = 1 #ev
    h2 = 1.3 #ERS
    h3 = 1.3 #ERD
    h4 = 0.1 #colorbar

    fig = plt.figure(figsize = (5.512,5.512))
    gs = mpl.gridspec.GridSpec(4,1, height_ratios = [h1,h2,h3,h4])

    colors_reordered = np.hstack([colors[N_filters:][::-1],colors[:N_filters]])
    SNNR_ax = fig.add_subplot(gs[0,:])
    SNNR_ax.plot(range(1,len(ev) + 1), 10*np.log10(ev), 'ko-', lw=2,
            markersize=5)
    for d,e in enumerate(ev):
        SNNR_ax.scatter([d+1], 10*np.log10(e),
        c=colors_reordered[d], s=60, zorder=1000)
    SNNR_ax.axhline(0, c='k', lw=1)
    SNNR_ax.set_xlim([0.5, len(ev)+0.5])
    SNNR_ax.set_xticks(np.r_[1,range(5, len(ev) + 1, 5)])
    SNNR_ax.set_ylabel('SNR (dB)')
    SNNR_ax.set_xlabel('component (index)')
    SNNR_ax.set_title('resulting SNR after CSP for band ' + band_name)

    # plot the five spatial patterns for ERD
    gs2 = mpl.gridspec.GridSpecFromSubplotSpec(1,5, gs[1,:],
            wspace=0, hspace=0.8)
    head_ax = []
    pc = []
    for d, pat in enumerate(potmaps[:N_filters]): # first half are ERD
        try:
            head_ax.append(fig.add_subplot(gs2[0,d], sharex=head_ax[0],
                sharey=head_ax[0], frame_on=False, aspect='equal'))
        except IndexError:
            head_ax.append(fig.add_subplot(gs2[0,d], frame_on=False, aspect='equal'))
        Z = pat[2]/np.abs(pat[2]).max()
        pc.append(head_ax[-1].pcolormesh(
            *pat[:2], Z, rasterized=True, cmap='coolwarm',
            vmin=-1, vmax=1, shading='auto'))
        head_ax[-1].contour(*pat, levels=[0], colors='w')
        head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
                alpha=0.5, zorder=1001)
        head_ax[-1].set_xlabel('ERD %d' % (d + 1) +'\n'+
                ' ({}\%)'.format(round(ev[d]*100)),
                fontsize=8)
        head_ax[-1].tick_params(**blind_ax)
        meet.sphere.addHead(head_ax[-1], ec=colors[-(d+1)], zorder=1000, lw=3)
    head_ax[0].set_ylim([-1.1,1.3])
    head_ax[0].set_xlim([-1.6,1.6])

    # plot the five spatial patterns for ERS
    gs3 = mpl.gridspec.GridSpecFromSubplotSpec(1,5, gs[2,:],
        wspace=0, hspace=0.8)
    head_ax = []
    pc = []
    for s, pat in enumerate(potmaps[N_filters:]): #last half are ERS
        try:
            head_ax.append(fig.add_subplot(gs3[0,s], sharex=head_ax[0],
                sharey=head_ax[0], frame_on=False, aspect='equal'))
        except IndexError:
            head_ax.append(fig.add_subplot(gs3[0,s], frame_on=False, aspect='equal'))
        Z = pat[2]/np.abs(pat[2]).max()
        pc.append(head_ax[-1].pcolormesh(
            *pat[:2], Z, rasterized=True, cmap='coolwarm',
            vmin=-1, vmax=1, shading='auto'))
        head_ax[-1].contour(*pat, levels=[0], colors='w')
        head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
                alpha=0.5, zorder=1001)
        head_ax[-1].set_xlabel('ERS %d' % (s + 1) +'\n'+
                ' ({}\%)'.format(round(ev[N_filters+s]*100)),
                fontsize=8)
        head_ax[-1].tick_params(**blind_ax)
        meet.sphere.addHead(head_ax[-1], ec=colors[s], zorder=1000, lw=3)
    head_ax[0].set_ylim([-1.1,1.3])
    head_ax[0].set_xlim([-1.6,1.6])

    # add a colorbar
    cbar_ax = fig.add_subplot(gs[3,:])
    cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
            label='amplitude (a.u.)', ticks=[-1,0,1])
    cbar.ax.set_xticklabels(['-', '0', '+'])
    cbar.ax.axvline(0, c='w', lw=2)

    # spect_ax = fig.add_subplot(gs[2,:])
    # [spect_ax.plot(f,
    #     10*np.log10(CSP_filters[:,i].dot(CSP_filters[:,i].dot(
    #         np.mean([t/np.trace(t[...,contrast_idx].mean(-1)).real
    #             for t in poststim_norm_csd], 0).real))),
    #         c=colors[i], lw=2) for i in range(4)]
    # spect_ax.set_xlim([0.5, 8])
    # spect_ax.set_ylim([-1.1, 1.1])
    # spect_ax.axhline(0, c='k', lw=1)
    # spect_ax.set_xlabel('frequency (Hz)')
    # spect_ax.set_ylabel('SNR (dB)')
    # spect_ax.set_title('normalized spectrum')
    #
    # spect_ax.axvline(snareFreq, color='b', zorder=0, lw=1)
    # spect_ax.axvline(2*snareFreq, color='b', zorder=0, lw=1)
    # spect_ax.axvline(wdBlkFreq, color='r', zorder=0, lw=1)
    # spect_ax.axvline(2*wdBlkFreq, color='k', zorder=0, lw=1)
    # spect_ax.axvline(4*wdBlkFreq, color='k', zorder=0, lw=1)

    gs.tight_layout(fig, pad=0.5)#, pad=0.2, h_pad=0.8

    fig.savefig(os.path.join(result_folder,
        'motor/mtCSP_patterns{}.pdf'.format(band_name)))


##### plot mtCSP{}.pdf: combination of above plots, components with patterns above and below ####
for band_idx, band_name in enumerate(band_names):
    ev = CSP_eigvals[band_idx]

    potmaps = potmaps_csp[band_idx]
    h1 = 2. #ERS
    h2 = 5. #component plot
    h3 = 2. #ERD
    h4 = 0.08 #colorbar

    fig = plt.figure(figsize = (5.5,7))
    gs = mpl.gridspec.GridSpec(4,1, height_ratios = [h1,h2,h3,h4], top=0.99,
        bottom = 0.1, hspace=0.0)

    # plot the five spatial patterns for ERS
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(2,5, gs[0,:], #two rows to have some space below
            height_ratios=[1,0.1], wspace=0, hspace=0.7)
    head_ax = []
    pc = []
    for s, pat in enumerate(potmaps[N_filters:]): #last half are ERS
        try:
            head_ax.append(fig.add_subplot(gs1[0,s], sharex=head_ax[0],
                sharey=head_ax[0], frame_on=False, aspect='equal'))
        except IndexError:
            head_ax.append(fig.add_subplot(gs1[0,s], frame_on=False, aspect='equal'))
        Z = pat[2]/np.abs(pat[2]).max()
        pc.append(head_ax[-1].pcolormesh(
            *pat[:2], Z, rasterized=True, cmap='coolwarm',
            vmin=-1, vmax=1, shading='auto'))
        head_ax[-1].contour(*pat, levels=[0], colors='w')
        head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
                alpha=0.5, zorder=1001)
        head_ax[-1].set_xlabel('ERS %d' % (s + 1) +'\n({}\%)'.format(
                round(ev[N_filters+s]*100)),fontsize=8)
        head_ax[-1].tick_params(**blind_ax)
        meet.sphere.addHead(head_ax[-1], ec=colors[s], zorder=1000, lw=3)
    head_ax[0].set_ylim([-1.1,1.3])
    head_ax[0].set_xlim([-1.6,1.6])

    # plot the components over time
    gs2 = mpl.gridspec.GridSpecFromSubplotSpec(1,1, gs[1,:],
        wspace=0, hspace=0.7)
    comp_ax = fig.add_subplot(gs2[0,0])

    # take subject mean and normalize
    ERD_CSP_subjmean = np.mean([e[band_idx] for e in ERD_CSP], axis=0)
    ERD_CSP_subjmean /= ERD_CSP_subjmean[:,base_idx].mean(1)[:,np.newaxis] #baseline avg should be 100%
    ERD_CSP_subjmean *= 100 # ERD in percent

    for s in range(CSP_ERSnum): #0,1,2,3,4
        comp_ax.plot(erd_t, ERD_CSP_subjmean[N_filters+s,:].T,
            label='ERS %d' % (s+1), color=colors[s])
    for d in range(CSP_ERDnum-1,-1,-1): #4,3,2,1,0
        comp_ax.plot(erd_t, ERD_CSP_subjmean[d,:].T,
            label='ERD %d' % (d+1),
            color=colors[-(d+1)])
    comp_ax.plot(erd_t, ERD_CSP_subjmean[CSP_ERSnum:-CSP_ERDnum,:].T,
        c='black', alpha=0.1)
    comp_ax.axvspan(base_ms[0], base_ms[-1], alpha=0.2, color='yellowgreen',
        label='contrast period')
    comp_ax.axvspan(act_ms[0], act_ms[-1], alpha=0.3, color='lightsalmon',
        label='target period')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,1,2,3,4,10,5,6,7,8,9,11]
    comp_ax.legend([handles[idx] for idx in order],
        [labels[idx] for idx in order], loc='upper left', fontsize=8, ncol=2)
    comp_ax.set_xlabel('time around response [ms]', fontsize=10)
    comp_ax.set_ylabel('CSP filtered EEG, relative amplitude [\%]', fontsize=10)
    comp_ax.set_title('subj.-avg. and eeg-applied CSP components {} Hz]'.format(
        band_name[:-1]), fontsize=12)

    # plot the five spatial patterns for ERD
    gs3 = mpl.gridspec.GridSpecFromSubplotSpec(1,5, gs[2,:],
            wspace=0, hspace=0.7)
    head_ax = []
    pc = []
    for d, pat in enumerate(potmaps[:N_filters]): # first half are ERD
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
        head_ax[-1].set_xlabel('ERD %d' % (d + 1) +'\n({}\%)'.format(
                round(ev[d]*100)),fontsize=8)
        head_ax[-1].tick_params(**blind_ax)
        meet.sphere.addHead(head_ax[-1], ec=colors[-(d+1)], zorder=1000, lw=3)
    head_ax[0].set_ylim([-1.1,1.3])
    head_ax[0].set_xlim([-1.6,1.6])

    # add a colorbar
    cbar_ax = fig.add_subplot(gs[3,:])
    cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
            label='amplitude (a.u.)', ticks=[-1,0,1])
    cbar.ax.set_xticklabels(['-', '0', '+'])
    cbar.ax.axvline(0, c='w', lw=2)

    gs.tight_layout(fig, pad=0.5, h_pad=0.8)#, pad=0.2, h_pad=0.8
    fig.savefig(os.path.join(result_folder,
        'motor/mtCSP{}.pdf'.format(band_name)))

##### plot mtCSP{}.pdf for each subject: combination of above plots, components with patterns above and below ####
for subj_idx in range(N_subjects-1): #only have 20 entries for subject 11 is missing
    if subj_idx < 10: #subject idx 10 already belongs to subject 12
        subj = subj_idx+1
    else:
        subj = subj_idx+2
    print('Plotting mtCSP_band.py for subject ', subj)
    for band_idx, band_name in enumerate(band_names):
        ev = CSP_eigvals[band_idx]
        potmaps = [meet.sphere.potMap(chancoords, pat_now,
            projection='stereographic') for pat_now in CSP_patterns[band_idx][subj_idx+1]] #+1 because first one is global
        h1 = 2. #ERS
        h2 = 5. #component plot
        h3 = 2. #ERD
        h4 = 0.08 #colorbar

        fig = plt.figure(figsize = (5.5,7))
        gs = mpl.gridspec.GridSpec(4,1, height_ratios = [h1,h2,h3,h4], top=0.99,
            bottom = 0.1, hspace=0.0)

        # plot the five spatial patterns for ERS
        gs1 = mpl.gridspec.GridSpecFromSubplotSpec(2,5, gs[0,:], #two rows to have some space below
                height_ratios=[1,0.1], wspace=0, hspace=0.7)
        head_ax = []
        pc = []
        for s, pat in enumerate(potmaps[N_filters:]): #last half are ERS
            try:
                head_ax.append(fig.add_subplot(gs1[0,s], sharex=head_ax[0],
                    sharey=head_ax[0], frame_on=False, aspect='equal'))
            except IndexError:
                head_ax.append(fig.add_subplot(gs1[0,s], frame_on=False, aspect='equal'))
            Z = pat[2]/np.abs(pat[2]).max()
            pc.append(head_ax[-1].pcolormesh(
                *pat[:2], Z, rasterized=True, cmap='coolwarm',
                vmin=-1, vmax=1, shading='auto'))
            head_ax[-1].contour(*pat, levels=[0], colors='w')
            head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
                    alpha=0.5, zorder=1001)
            head_ax[-1].set_xlabel('ERS %d' % (s + 1) +'\n({}\%)'.format(
                    round(ev[N_filters+s]*100)),fontsize=8)
            head_ax[-1].tick_params(**blind_ax)
            meet.sphere.addHead(head_ax[-1], ec=colors[s], zorder=1000, lw=3)
        head_ax[0].set_ylim([-1.1,1.3])
        head_ax[0].set_xlim([-1.6,1.6])

        # plot the components over time
        gs2 = mpl.gridspec.GridSpecFromSubplotSpec(1,1, gs[1,:],
            wspace=0, hspace=0.7)
        comp_ax = fig.add_subplot(gs2[0,0])

        # take subject mean and normalize
        ERD_CSP_subj = ERD_CSP[subj_idx][band_idx]
        ERD_CSP_subj /= ERD_CSP_subj[:,base_idx].mean(1)[:,np.newaxis] #baseline avg should be 100%
        ERD_CSP_subj *= 100 # ERD in percent

        for s in range(CSP_ERSnum): #0,1,2,3,4
            comp_ax.plot(erd_t, ERD_CSP_subj[N_filters+s,:].T,
                label='ERS %d' % (s+1), color=colors[s])
        for d in range(CSP_ERDnum-1,-1,-1): #4,3,2,1,0
            comp_ax.plot(erd_t, ERD_CSP_subj[d,:].T,
                label='ERD %d' % (d+1), color=colors[-(d+1)])
        comp_ax.plot(erd_t, ERD_CSP_subj[CSP_ERSnum:-CSP_ERDnum,:].T,
            c='black', alpha=0.1)
        comp_ax.axvspan(base_ms[0], base_ms[-1], alpha=0.2, color='yellowgreen',
            label='contrast period')
        comp_ax.axvspan(act_ms[0], act_ms[-1], alpha=0.3, color='lightsalmon',
            label='target period')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,1,2,3,4,10,5,6,7,8,9,11]
        comp_ax.legend([handles[idx] for idx in order],
            [labels[idx] for idx in order], loc='upper left', fontsize=8, ncol=2)
        comp_ax.set_xlabel('time around response [ms]', fontsize=10)
        comp_ax.set_ylabel('CSP filtered EEG, relative amplitude [dB]', fontsize=10)
        comp_ax.set_title('subj.-avg. and eeg-applied CSP components {} Hz]'.format(
            band_name[:-1]), fontsize=12)

        # plot the five spatial patterns for ERD
        gs3 = mpl.gridspec.GridSpecFromSubplotSpec(1,5, gs[2,:],
                wspace=0, hspace=0.7)
        head_ax = []
        pc = []
        for d, pat in enumerate(potmaps[:N_filters]): # first half are ERD
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
            head_ax[-1].set_xlabel('ERD %d' % (d + 1) +'\n({}\%)'.format(
                    round(ev[d]*100)), fontsize=8)
            head_ax[-1].tick_params(**blind_ax)
            meet.sphere.addHead(head_ax[-1], ec=colors[-(d+1)], zorder=1000, lw=3)
        head_ax[0].set_ylim([-1.1,1.3])
        head_ax[0].set_xlim([-1.6,1.6])

        # add a colorbar
        cbar_ax = fig.add_subplot(gs[3,:])
        cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
                label='amplitude (a.u.)', ticks=[-1,0,1])
        cbar.ax.set_xticklabels(['-', '0', '+'])
        cbar.ax.axvline(0, c='w', lw=2)

        gs.tight_layout(fig, pad=0.5, h_pad=0.8)#, pad=0.2, h_pad=0.8
        fig.savefig(os.path.join(result_folder,'S%02d'% subj,
            'motor_mtCSP{}.pdf'.format(band_name)))
    plt.close('all')


##### patternsByTime.pdf: plot BP and ERP patterns over time for avg and each subject #####
times = [-1000,-750,-500,-250,0] #ms relative to response
time_idx = [t-win[0]-1 for t in times] #win is -2000 to 500

BP_now = np.mean(BPs,axis=0) #subject avg.
ERD_now = np.mean(ERDs, axis=0) #subject avg.
potmaps_BP = [meet.sphere.potMap(chancoords, BP_now[:,t],
    projection='stereographic') for t in time_idx]
vmax_BP = np.max(np.abs(BP_now[:,999:1999])) #0 is in the middle, scale for plotting potmaps

fig = plt.figure(figsize = (5.512,4.8))
h1 = 2 #BP, ERD
h28 = 0.1 #colorbars
gs = mpl.gridspec.GridSpec(5,1, height_ratios = [h1,h28,h1,h1,h28])

# first line BP
gs1 = mpl.gridspec.GridSpecFromSubplotSpec(1,5, gs[0,:], wspace=0, hspace=0.2)
head_ax = []
pc = [] #for color bar
for i, pat in enumerate(potmaps_BP):
    try:
        head_ax.append(fig.add_subplot(gs1[0,i], sharex=head_ax[0],
            sharey=head_ax[0], frame_on=False, aspect='equal'))
    except IndexError:
        head_ax.append(fig.add_subplot(gs1[0,i], frame_on=False, aspect='equal'))
    pc.append(head_ax[-1].pcolormesh(
        *pat, rasterized=True, cmap='coolwarm', #*pat gives him all pats one by one
        vmin=-vmax_BP, vmax=vmax_BP, shading='auto'))
    head_ax[-1].contour(*pat, levels=[0], colors='w')
    head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5, zorder=1001)
    head_ax[0].set_ylabel('BP', fontsize=8)
    head_ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(head_ax[-1], ec='black', zorder=1000, lw=2)
head_ax[0].set_ylim([-1.1,1.3])
head_ax[0].set_xlim([-1.6,1.6])
head_ax[2].set_title('BP and ERD patterns for average subject')

# line 2: add a colorbar for BP
cbar.ax.tick_params(labelsize=8)
cbar_ax = fig.add_subplot(gs[1,:])
cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
        label=r'amplitude [$\mu V$]')#, ticks=[-1,0,1])
#cbar.ax.set_xticklabels(['-', '0', '+'])
cbar.ax.axvline(0, c='w', lw=2)

# line 3-7: ERD per band (only ERD => last component)
vmax = np.max(np.abs(ERD_now[0,:,999:1999])) #alpha; 0 is in the middle, scale for plotting potmaps
for band_idx, band_name in enumerate(band_names):
    potmaps_ERD = [meet.sphere.potMap(chancoords, ERD_now[band_idx,:,t],
        projection='stereographic') for t in time_idx]

    gsi = mpl.gridspec.GridSpecFromSubplotSpec(1,5, gs[band_idx+2,:],
        wspace=0, hspace=0.2)
    head_ax = []
    for i, pat in enumerate(potmaps_ERD):
        try:
            head_ax.append(fig.add_subplot(gsi[0,i], sharex=head_ax[0],
                sharey=head_ax[0], frame_on=False, aspect='equal'))
        except IndexError: #first head
            head_ax.append(fig.add_subplot(gsi[0,i], frame_on=False, aspect='equal'))
        pc.append(head_ax[-1].pcolormesh(
            *pat, rasterized=True, cmap='coolwarm',
            vmin=-vmax, vmax=vmax, shading='auto'))
        head_ax[-1].contour(*pat, levels=[0], colors='w')
        head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
                alpha=0.5, zorder=1001)
        head_ax[0].set_ylabel('ERD\n' + band_name + ' Hz', fontsize=8)
        head_ax[-1].tick_params(**blind_ax)
        meet.sphere.addHead(head_ax[-1], ec='black', zorder=1000, lw=2)
        head_ax[-1].set_ylim([-1.1,1.3])
        head_ax[-1].set_xlim([-1.6,1.6])
        if band_idx == len(band_names)-1:
            head_ax[-1].set_xlabel(str(times[i]) + 'ms', fontsize=8)

# line 8: add a colorbar
cbar.ax.tick_params(labelsize=8)
cbar_ax = fig.add_subplot(gs[4,:])
cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
        label='amplitude [dB]')#, ticks=[-1,0,1])
#cbar.ax.set_xticklabels(['-', '0', '+'])
cbar.ax.axvline(0, c='w', lw=2)

gs.tight_layout(fig, pad=0.5, h_pad=0.2)
plt.savefig(os.path.join(result_folder,
    'motor/patternsByTime.pdf'))


for subj_idx in range(N_subjects-1): #only have 20 entries for subject 11 is missing
    if subj_idx < 10: #subject idx 10 already belongs to subject 12
        subj = subj_idx+1
    else:
        subj = subj_idx+2

    print('calculating and plotting patterns for subject '+str(subj)+'...')
    BP_now = BPs[subj_idx]
    ERD_now = ERDs[subj_idx]

    potmaps_BP = [meet.sphere.potMap(chancoords, BP_now[:,t],
        projection='stereographic') for t in time_idx]
    vmax_BP = np.max(np.abs(BP_now[:,999:1999])) #0 is in the middle, scale for plotting potmaps


    fig = plt.figure(figsize = (5.512,4.8))
    h1 = 2 #BP, ERD
    h28 = 0.1 #colorbars
    gs = mpl.gridspec.GridSpec(5,1, height_ratios = [h1,h28,h1,h1,h28])

    # first line BP
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(1,5, gs[0,:], wspace=0, hspace=0.2)
    head_ax = []
    pc = [] #for color bar
    for i, pat in enumerate(potmaps_BP):
        try:
            head_ax.append(fig.add_subplot(gs1[0,i], sharex=head_ax[0],
                sharey=head_ax[0], frame_on=False, aspect='equal'))
        except IndexError:
            head_ax.append(fig.add_subplot(gs1[0,i], frame_on=False, aspect='equal'))
        pc.append(head_ax[-1].pcolormesh(
            *pat, rasterized=True, cmap='coolwarm', #*pat gives him all pats one by one
            vmin=-vmax_BP, vmax=vmax_BP, shading='auto'))
        head_ax[-1].contour(*pat, levels=[0], colors='w')
        head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
                alpha=0.5, zorder=1001)
        head_ax[0].set_ylabel('BP', fontsize=8)
        head_ax[-1].tick_params(**blind_ax)
        meet.sphere.addHead(head_ax[-1], ec='black', zorder=1000, lw=2)
    head_ax[0].set_ylim([-1.1,1.3])
    head_ax[0].set_xlim([-1.6,1.6])
    head_ax[2].set_title('BP and ERD patterns for subject ' + str(subj))

    # line 2: add a colorbar for BP
    cbar.ax.tick_params(labelsize=8)
    cbar_ax = fig.add_subplot(gs[1,:])
    cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
            label=r'amplitude [$\mu V$]')#, ticks=[-1,0,1])
    #cbar.ax.set_xticklabels(['-', '0', '+'])
    cbar.ax.axvline(0, c='w', lw=2)

    # line 3-7: ERD per band (only ERD => last component)
    vmax = np.max(np.abs(ERD_now[0,:,999:1999])) #0 is in the middle, scale for plotting potmaps
    for band_idx, band_name in enumerate(band_names):
        potmaps_ERD = [meet.sphere.potMap(chancoords, ERD_now[band_idx,:,t],
            projection='stereographic') for t in time_idx]

        gsi = mpl.gridspec.GridSpecFromSubplotSpec(1,5, gs[band_idx+2,:],
            wspace=0, hspace=0.2)
        head_ax = []
        for i, pat in enumerate(potmaps_ERD):
            try:
                head_ax.append(fig.add_subplot(gsi[0,i], sharex=head_ax[0],
                    sharey=head_ax[0], frame_on=False, aspect='equal'))
            except IndexError: #first head
                head_ax.append(fig.add_subplot(gsi[0,i], frame_on=False, aspect='equal'))
            pc.append(head_ax[-1].pcolormesh(
                *pat, rasterized=True, cmap='coolwarm',
                vmin=-vmax, vmax=vmax, shading='auto'))
            head_ax[-1].contour(*pat, levels=[0], colors='w')
            head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
                    alpha=0.5, zorder=1001)
            head_ax[0].set_ylabel('ERD\n' + band_name + ' Hz', fontsize=8)
            head_ax[-1].tick_params(**blind_ax)
            meet.sphere.addHead(head_ax[-1], ec='black', zorder=1000, lw=2)
            head_ax[-1].set_ylim([-1.1,1.3])
            head_ax[-1].set_xlim([-1.6,1.6])
            if band_idx == len(band_names)-1:
                head_ax[-1].set_xlabel(str(times[i]) + 'ms', fontsize=8)

    # line 8: add a colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar_ax = fig.add_subplot(gs[4,:])
    cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
            label='amplitude [dB]')#, ticks=[-1,0,1])
    #cbar.ax.set_xticklabels(['-', '0', '+'])
    cbar.ax.axvline(0, c='w', lw=2)

    gs.tight_layout(fig, pad=0.5, h_pad=0.2)
    plt.savefig(os.path.join(result_folder,'S%02d'% subj,
        'motor_patternsByTime.pdf'))
plt.close('all')
