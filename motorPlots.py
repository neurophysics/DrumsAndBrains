"""
plots patternsByTime.pdf
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
color0 = '#543005'.upper() #dark brown
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

##### read motor inlier and win #####
snareInlier = [] # list of 20 subjects, each shape ≤75
wdBlkInlier = [] #same for wdBlk
i=0
while True:
    try:
        with np.load(os.path.join(result_folder, 'motor/inlier.npz'),
            'r') as fi:
            snareInlier.append(fi['snareInlier_response_{:02d}'.format(i)])
            wdBlkInlier.append(fi['wdBlkInlier_response_{:02d}'.format(i)])
            win = fi['win']
        with np.load(os.path.join(result_folder, 'motor/covmat.npz'),
            'r') as f_covmat:
            fbands = f_covmat['fbands']
            base_idx = f_covmat['base_idx'] #corresponds to -2000 to -1250ms
            act_idx = f_covmat['act_idx'] #corresponds to -750 to 0ms
        i+=1
    except KeyError:
        break

base_ms = [t+win[0] for t in base_idx] #-2000 to 500 milliseconds
act_ms = [t+win[0] for t in act_idx] #-2000 to 500

##### read BP #####
BPs = [] #stores BP per subject, each shape (Ncomp, time)=(32,2500)
ERDs = [] #stores ERD per subject, each shape (band,Ncomp, time)=(5,32,2500)
i=0
while True: #loop over subjects
    try:
        with np.load(os.path.join(result_folder, 'motor/BP.npz'),
            'r') as f_BP:
            BPs.append(f_BP['BP_{:02d}'.format(i)])
        with np.load(os.path.join(result_folder, 'motor/ERD.npz'),
            'r') as f_ERD:
            ERDs.append(f_ERD['ERD_{:02d}'.format(i)])
        i+=1
    except KeyError:
        break

##### read CSP #####
CSP_eigvals = [] #stores EV per band, each shape (CSPcomp,)=(32,)
CSP_filters = [] #stores filters per band, each shape (CSPcomp,CSPcomp)=(32,32)
CSP_patterns = [] #store spatterns per band, each shape (CSPcomp,CSPcomp)=(32,32)
try:
    i=0 #loop over bands
    while True:
        try:
            with np.load(os.path.join(result_folder,'motor/CSP.npz')) as f:
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

##### read BP and ERD_CSP ######
ERD_CSP = [] # stores trial averaged ERD/S_CSP per subject, each with shape (Nband, CSPcomp,time)
ERDCSP_trial = [] #stores ERD_CSP of best CSPcomp per subject,each shape (Nband, Ntrial)
ERSCSP_trial = [] # same for ERS
try:
    i=0 #loop over subjects
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
    print('Please run csp.py first.')

##### plot ERDCSP components #####
# look at plot to determine number of components
for i,ev in enumerate(CSP_eigvals):
    plt.plot(ev, 'o')
    plt.title('CSP EV band {}, small ERD, large ERS'.format(band_names[1]))
    #plt.show()
# first argument is pre movement so
CSP_ERDnums = [2,5,5,5,5] # e.g. [:3] # small EV 2 or 5 mostly both work
CSP_ERSnums = [4,4,4,4,5] #e.g. [-4:] # large EV

for band_idx, band_name in enumerate(band_names):
    # average over subjects
    ERD_CSP_subjmean = np.mean([e[band_idx] for e in ERD_CSP], axis=0)
    ev = CSP_eigvals[band_idx]
    CSP_ERDnum = CSP_ERDnums[band_idx]
    CSP_ERSnum = CSP_ERSnums[band_idx]
    # normalize
    ERD_CSP_subjmean /= ERD_CSP_subjmean[:,0:750].mean(1)[:,np.newaxis] #baseline avg should be 100%
    ERD_CSP_subjmean *= 100 # ERD in percent

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
    plt.axvspan(base_ms[0], base_ms[-1], alpha=0.3, color=colors[4],
        label='contrast period') #_ gets ignored as label
    plt.axvspan(act_ms[0], act_ms[-1], alpha=0.3, color=colors[5],
        label='target period')
    plt.legend(fontsize=8)
    plt.xlabel('time around response [ms]', fontsize=10)
    plt.ylabel('CSP filtered EEG, relative amplitude [%]', fontsize=10)
    plt.title('subj.-avg. and eeg-applied CSP components {} Hz]'.format(
        band_name[:-1]), fontsize=12)
    plt.savefig(os.path.join(result_folder,
        'motor/erdcsp_comp{}.pdf'.format(band_name)))


##### plot EV and CSP patterns #####
# this, especially calculating the potmaps, takes a while
potmaps_csp = []
for band_idx, band_name in enumerate(band_names):
    ev = CSP_eigvals[band_idx]

    potmaps = [meet.sphere.potMap(chancoords, pat_now,
        projection='stereographic') for pat_now in CSP_patterns[band_idx]]
    potmaps_csp.append(potmaps)
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
        meet.sphere.addHead(head_ax[-1], ec=colors[s], zorder=1000, lw=3)
    head_ax[0].set_ylim([-1.1,1.3])
    head_ax[0].set_xlim([-1.6,1.6])

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
        meet.sphere.addHead(head_ax[-1], ec=colors[-(d+1)], zorder=1000, lw=3)
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
        'motor/CSP_patterns{}.pdf'.format(band_name)))

##### plot combination of above plots: components with patterns above and below ####
for band_idx, band_name in enumerate(band_names):
    ev = CSP_eigvals[band_idx]

    potmaps = potmaps_csp[band_idx]
    h1 = 2. #ERS
    h2 = 5. #component plot
    h3 = 2. #ERD
    h4 = 0.08 #colorbar

    fig = plt.figure(figsize = (5.5,7))
    gs = mpl.gridspec.GridSpec(4,1, height_ratios = [h1,h2,h3,h4], top=0.99,
        bottom = -1, hspace=0.0)

    # plot the five spatial patterns for ERS
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(2,5, gs[0,:],
            height_ratios = [1,0.1], wspace=0, hspace=0.7)
    head_ax = []
    pc = []
    for s, pat in enumerate(potmaps[:5]):
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
        head_ax[-1].set_xlabel('ERS %d' % (s + 1),
                fontsize=8)
        head_ax[-1].tick_params(**blind_ax)
        meet.sphere.addHead(head_ax[-1], ec=colors[s], zorder=1000, lw=3)
    head_ax[0].set_ylim([-1.1,1.3])
    head_ax[0].set_xlim([-1.6,1.6])

    # plot the four spatial patterns for ERD
    comp_ax = fig.add_subplot(gs[1,:])
    ERD_CSP_subjmean = np.mean([e[band_idx] for e in ERD_CSP], axis=0)
    CSP_ERDnum = CSP_ERDnums[band_idx]
    CSP_ERSnum = CSP_ERSnums[band_idx]
    # normalize
    ERD_CSP_subjmean /= ERD_CSP_subjmean[:,0:750].mean(1)[:,np.newaxis] #baseline avg should be 100%
    ERD_CSP_subjmean *= 100 # ERD in percent
    for s in range(CSP_ERSnum): #0,1,2,3
        comp_ax.plot(erd_t, ERD_CSP_subjmean[s,:].T,
            label='ERS %d' % (s+1) + ' ($\mathrm{EV=%.2fdB}$)' % round(
            10*np.log10(ev[s]),2), color=colors[s])
    for d in range(-CSP_ERDnum,0,1): #-3,-2,-1
        comp_ax.plot(erd_t, ERD_CSP_subjmean[d,:].T,
            label='ERD %d' % (-d) + ' ($\mathrm{EV=%.2fdB}$)' % round(
            10*np.log10(ev[d]), 2), color=colors[d])
    comp_ax.plot(erd_t, ERD_CSP_subjmean[CSP_ERSnum:-CSP_ERDnum,:].T,
        c='black', alpha=0.1)
    comp_ax.axvspan(base_ms[0], base_ms[-1], alpha=0.3, color=colors[4],
        label='contrast period') #_ gets ignored as label
    comp_ax.axvspan(act_ms[0], act_ms[-1], alpha=0.3, color=colors[5],
        label='target period')
    comp_ax.legend(fontsize=8)
    comp_ax.set_xlabel('time around response [ms]', fontsize=10)
    comp_ax.set_ylabel('CSP filtered EEG, relative amplitude [%]', fontsize=10)
    comp_ax.set_title('subj.-avg. and eeg-applied CSP components {} Hz]'.format(
        band_name[:-1]), fontsize=12)

    # plot the five spatial patterns for ERD
    gs3 = mpl.gridspec.GridSpecFromSubplotSpec(2,5, gs[2,:],
            height_ratios = [1,0.1], wspace=0, hspace=0.7)
    head_ax = []
    pc = []
    for d, pat in enumerate(reversed(potmaps[-5:])): # take last 5, reverse, then enumerate
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
        head_ax[-1].set_xlabel('ERD %d' % (d + 1),
                fontsize=8)
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
        'motor/CSP{}.pdf'.format(band_name)))
plt.close('all')

##### plot BP and ERP patterns over time fro each subject #####
times = [-1000,-750,-500,-250,0] #ms relative to response
time_idx = [t-win[0]-1 for t in times] #win is -2000 to 500
for subj_idx in range(N_subjects-1): #only have 20 entries for subject 11 is missing
    if subj_idx < 10: #subject idx 10 already belongs to subject 12
        subj = subj_idx+1
    else:
        subj = subj_idx+2
    
    print('calculating and plotting patterns for subject '+str(subj)+'...')
    BP_subj = BPs[subj_idx]
    ERD_subj = ERDs[subj_idx]

    potmaps_BP = [meet.sphere.potMap(chancoords, BP_subj[:,t],
        projection='stereographic') for t in time_idx]
    vmax_BP = np.max(np.abs(BP_subj[:,999:1999])) #0 is in the middle, scale for plotting potmaps


    fig = plt.figure(figsize = (5.512,9))
    h1 = 2 #BP, ERD
    h28 = 0.1 #colorbars
    gs = mpl.gridspec.GridSpec(8,1, height_ratios = [h1,h28,h1,h1,h1,h1,h1,h28])

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
            label=r'amplitude [$\mu$ V]')#, ticks=[-1,0,1])
    #cbar.ax.set_xticklabels(['-', '0', '+'])
    cbar.ax.axvline(0, c='w', lw=2)

    # line 3-7: ERD per band (only ERD => last component)
    vmax = np.max(np.abs(ERD_subj[2,:,999:1999])) #0 is in the middle, scale for plotting potmaps
    for band_idx, band_name in enumerate(band_names):
        potmaps_ERD = [meet.sphere.potMap(chancoords, ERD_subj[band_idx,:,t],
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
    cbar_ax = fig.add_subplot(gs[7,:])
    cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
            label='amplitude [dB]')#, ticks=[-1,0,1])
    #cbar.ax.set_xticklabels(['-', '0', '+'])
    cbar.ax.axvline(0, c='w', lw=2)

    gs.tight_layout(fig, pad=0.5, h_pad=0.2)
    plt.savefig(os.path.join(result_folder,'S%02d'% subj,
        'motor_patternsByTime.pdf'))
