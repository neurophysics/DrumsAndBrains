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
cov_thresh = 2000

# calculate the SSD from all subjects
# read the channel names
channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

N_channels = len(channames)

for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(result_folder, 'S%02d' % i)
                + '/prepared_filterdata.npz', 'r') as f:
            snareListenData = f['snareListenData']
            wdBlkListenData = f['wdBlkListenData']
            snareListenData_rec = f['snareListenData_rec']
            wdBlkListenData_rec = f['wdBlkListenData_rec']
        temp = np.dstack([snareListenData, wdBlkListenData])
        org_cov.append(np.einsum('ijk, ljk->ilk', temp, temp)/temp.shape[1])
        temp = np.dstack([snareListenData_rec, wdBlkListenData_rec])
        rec_cov.append(np.einsum('ijk, ljk->ilk', temp, temp)/temp.shape[1])
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

del snareListenData
del wdBlkListenData
del snareListenData_rec
del wdBlkListenData_rec
del temp

# threshold for outliers
inlier = [p[range(N_channels), range(N_channels)].sum(0) < cov_thresh
        for p in org_cov]

org_cov, rec_cov = zip(*[(p[...,I], q[...,I])
    for p,q,I in zip(org_cov, rec_cov, inlier)])

if normalize:
    # if requested, normalize the subjects
    org_cov, rec_cov = zip(*[(
        p/np.trace(p.mean(-1)),
        q/np.trace(p.mean(-1)))
        for p,q in zip(org_cov, rec_cov)])
    org_cov, rec_cov = zip(*[(
        p/np.trace(p.mean(-1)),
        q/np.trace(p.mean(-1)))
        for p,q in zip(org_cov, rec_cov)])

# calculate the actual SSD filters enhancing the reconstructed data with only
# the two oscillations included - this is in the end
# a sort of SSD (spatial spectral decomposition)
# enhance the frequencies we're searching for (double/triple beat)
# and suppress other frequencies
listen_cov = np.dstack(org_cov).mean(-1)
listen_rec_cov = np.dstack(rec_cov).mean(-1)

try:
    ssd_eigvals, ssd_filter = scipy.linalg.eigh(listen_rec_cov,
        listen_cov + listen_rec_cov)
except scipy.linalg.LinAlgError:
    # if rank deficient
    rank = np.linalg.matrix_rank(listen_cov + listen_rec_cov)
    W_vals, W_vect = scipy.linalg.eigh(listen_cov + listen_rec_cov)
    W = W_vect[:,-rank:]/np.sqrt(W_vals[-rank:])
    ssd_eigvals, ssd_filter = scipy.linalg.eigh(
            W.T.dot(listen_rec_cov).dot(W))
    ssd_filter = W.dot(ssd_filter)

ssd_filter = ssd_filter[:,::-1]
ssd_eigvals = ssd_eigvals[::-1]

ssd_pattern = scipy.linalg.solve(
        ssd_filter.T.dot(listen_rec_cov).dot(ssd_filter),
        ssd_filter.T.dot(listen_rec_cov))

# plot the patterns
# name the ssd channels
ssd_channames = ['SSD%02d' % (i+1) for i in xrange(len(ssd_pattern))]

# plot the SSD components scalp maps
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
eigvals_ax.plot(np.arange(1, len(ssd_eigvals) + 1, 1), ssd_eigvals, 'ko-',
        markersize=5)
eigvals_ax.set_xlim([0, len(ssd_eigvals) + 1])
eigvals_ax.set_title('SSD eigenvalues')
fig.suptitle('SSD patterns', size=14)
gs.tight_layout(fig, pad=0.3, rect=(0,0,1,0.95))
fig.savefig(os.path.join(result_folder, 'SSD_patterns_norm_%s.pdf' % normalize))

# save the results
np.savez(os.path.join(result_folder, 'SSD_norm_%s.npz' % normalize),
        ssd_eigvals = ssd_eigvals,
        ssd_filter = ssd_filter)
