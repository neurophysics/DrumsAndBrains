import numpy as np
import scipy
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os.path
import helper_functions
import meet

mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 10

cmap = 'plasma'
color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

s_rate = 1000 # sampling rate of the EEG

result_folder = sys.argv[1]
normalize = bool(int(sys.argv[2]))
N_subjects = 21

# calculate the SSD from all subjects
# read the channel names
channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

N_channels = len(channames)

listen_cov = np.zeros((N_channels, N_channels), float)
listen_rec_cov = np.zeros_like(listen_cov)

k = 1
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(result_folder, 'S%02d' % i)
                + '/prepared_filterdata.npz', 'r') as f:
            snareListenData = f['snareListenData']
            wdBlkListenData = f['wdBlkListenData']
            snareListenData_rec = f['snareListenData_rec']
            wdBlkListenData_rec = f['wdBlkListenData_rec']
        org_cov = np.cov(np.dstack([snareListenData, wdBlkListenData]).reshape(
            N_channels, -1, order='F'))
        rec_cov = np.cov(np.dstack([snareListenData_rec, wdBlkListenData_rec]
            ).reshape(N_channels, -1, order='F'))
        if normalize:
            # if requested, normalize
            rec_cov /= np.trace(org_cov)
            org_cov /= np.trace(org_cov)
        listen_cov = listen_cov + (org_cov - listen_cov)/k
        listen_rec_cov = listen_rec_cov + (rec_cov - listen_rec_cov)/k
        k += 1
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

del snareListenData
del wdBlkListenData
del snareListenData_rec
del wdBlkListenData_rec

# calculate the actual SSD filters enhancing the reconstructed data with only
# the two oscillations included - this is in the end
# a sort of SSD (spatial spectral decomposition)
# enhance the frequencies we're searching for (double/triple beat)
# and suppress other frequencies
ssd_eigvals, ssd_filter = scipy.linalg.eigh(listen_rec_cov,
        listen_cov + listen_rec_cov)
ssd_filter = ssd_filter[:,::-1]
ssd_eigvals = ssd_eigvals[::-1]

ssd_pattern = np.linalg.inv(ssd_filter)

# plot the patterns
# name the ssd channels
ssd_channames = ['SSD%02d' % (i+1) for i in xrange(len(ssd_pattern))]

# plot the ICA  components scalp maps
ssd_potmaps = [meet.sphere.potMap(chancoords, ssd_c,
    projection='stereographic') for ssd_c in ssd_pattern]

fig = plt.figure(figsize=(4.5,10))
# plot with 8 rows and 4 columns
gs = mpl.gridspec.GridSpec(10,4, height_ratios = 8*[1]+[0.2]+[1])
ax = []
for i, (X,Y,Z) in enumerate(ssd_potmaps):
    if i == 0:
        ax.append(fig.add_subplot(gs[0,0], frame_on = False))
    else:
        ax.append(fig.add_subplot(gs[i//4,i%4], sharex=ax[0], sharey=ax[0],
                frame_on = False))
    Z /= np.abs(Z).max()
    ax[-1].tick_params(**blind_ax)
    meet.sphere.addHead(ax[-1])
    pc = ax[-1].pcolormesh(X, Y, Z, vmin=-1, vmax=1, rasterized=True,
            cmap=cmap)
    ax[-1].contour(X, Y, Z, levels=[0], colors='w')
    ax[-1].scatter(chancoords_2d[:,0], chancoords_2d[:,1], c='k', s=2,
            alpha=0.5)
    ax[-1].set_title(ssd_channames[i] + ' (%.2f)' % ssd_eigvals[i])
pc_ax = fig.add_subplot(gs[-2,:])
plt.colorbar(pc, cax=pc_ax, orientation='horizontal',
        label='relative amplitude')
pc_ax.plot([0.5,0.5], [0,1], c='w', zorder=1000,
        transform=pc_ax.transAxes)
eigvals_ax = fig.add_subplot(gs[-1,:], frame_on=False)
eigvals_ax.plot(np.arange(1, N_channels + 1, 1), ssd_eigvals, 'ko-',
        markersize=5)
eigvals_ax.set_xlim([0, N_channels + 1])
eigvals_ax.set_title('SSD eigenvalues')
fig.suptitle('SSD patterns', size=14)
gs.tight_layout(fig, pad=0.3, rect=(0,0,1,0.95))
fig.savefig(os.path.join(result_folder, 'SSD_patterns_norm_%s.pdf' % normalize))

# save the results
np.savez(os.path.join(result_folder, 'SSD_norm_%s.npz' % normalize),
        ssd_eigvals = ssd_eigvals,
        ssd_filter = ssd_filter)
