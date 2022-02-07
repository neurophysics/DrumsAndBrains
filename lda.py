'uses LDA to transform into seperable BP and contrast group'

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

###### read BP ######
# unprocessed, just epoched eeg by hittimes
all_BP = [] # len 20, each (32,2500,1xx)=(channels, time in ms, trials)
i=0
while True:
    try:
        with np.load(os.path.join(result_folder, 'motor/BP.npz'),
            'r') as fi:
            all_BP.append(fi['BP_trials_{:02d}'.format(i)])
        i+=1
    except KeyError:
        break
# read window -2000 to 500 ms
with np.load(os.path.join(result_folder, 'motor/inlier.npz'), 'r') as f:
    win = f['win']

# stitch all subjects together so we have many trials
all_BP = np.concatenate(all_BP, axis=-1) #now shape (channels,trials) = (32,2500,20*1xx)
all_BP = all_BP - all_BP[:,:1400,:].mean(1)[:,np.newaxis,:]
# divide into classes, both shape (channels, trials)
BP = all_BP[:,1400:1900,:].mean(1) #-800 to -100ms
contrast = all_BP[:,:500,:].mean(1) #-2 to -1.2

# center classwise, estimate cov on all features at once
Xpool = np.hstack([BP-BP.mean(-1)[:,np.newaxis],
    contrast-contrast.mean(-1)[:,np.newaxis]])
C = np.cov(Xpool) #(32,32)
cfilt = np.linalg.pinv(C).dot(contrast.mean(-1) - BP.mean(-1))

# alternatively, use gunnars LDA:
#cfilt,diff = g.LDA(all_BP[:,:,:1900]), smooth=500)ull
np.save(os.path.join(result_folder, 'motor/lda.npy'), cfilt)

##### plots #####
#check component
mean = cfilt.dot(all_BP.mean(-1)).T
sd = np.tensordot(cfilt, all_BP, axes=[0,0]).std(-1).T
plt.figure()
plt.plot(mean)
plt.plot(mean+sd, c='b', alpha = 0.5)
plt.plot(mean-sd, c='b', alpha = 0.5)
plt.show()

# plot pattern
blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)

fig = plt.figure()
h1 = 2 #patterns
h2 = 0.1 #colorbar
gs = mpl.gridspec.GridSpec(2,1, height_ratios = [h1,h2])
pat = meet.sphere.potMap(chancoords, cfilt, projection='stereographic')
vmax = np.max(np.abs(cfilt)) #0 is in the middle, scale for plotting potmaps
gsi = mpl.gridspec.GridSpecFromSubplotSpec(1,1, gs[0,:],
    wspace=0, hspace=0.2)
head_ax = []
pc = []
head_ax.append(fig.add_subplot(gsi[0,0], frame_on=False, aspect='equal'))
pc.append(head_ax[-1].pcolormesh(
    *pat, rasterized=True, cmap='coolwarm',
    vmin=-vmax, vmax=vmax, shading='auto'))
head_ax[-1].contour(*pat, levels=[0], colors='w')
head_ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
        alpha=0.5, zorder=1001)
head_ax[-1].tick_params(**blind_ax)
meet.sphere.addHead(head_ax[-1], ec='black', zorder=1000, lw=2)
head_ax[-1].set_ylim([-1.1,1.3])
head_ax[-1].set_xlim([-1.6,1.6])
head_ax[0].set_title('LDA Pattern')

# add a colorbar
cbar_ax = fig.add_subplot(gs[1,:])
cbar_ax.tick_params(labelsize=8)
cbar = plt.colorbar(pc[-1], cax=cbar_ax, orientation='horizontal',
        label='amplitude (a.u.)')#, ticks=[-1,0,1])
#cbar.ax.set_xticklabels(['-', '0', '+'])
cbar_ax.axvline(0, c='w', lw=2)

gs.tight_layout(fig, pad=0.5, h_pad=0.2)
plt.savefig(os.path.join(result_folder,'motor/LDApattern.pdf'))
