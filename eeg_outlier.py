import numpy as np
import aifc
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os.path
import helper_functions
import meet
import scipy.signal
from sklearn.decomposition import FastICA
import pickle

mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 10

cmap = 'plasma'
color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

s_rate = 1000
s_rate_ds = s_rate//4

data_folder = sys.argv[1]
subject = int(sys.argv[2])
result_folder = sys.argv[3]

data_folder = os.path.join(data_folder, 'S%02d' % subject)
save_folder = os.path.join(result_folder, 'S%02d' % subject)

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# get the channel names
channames = meet.sphere.getChannelNames(os.path.join(data_folder,
    '../channels.txt'))
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

eeg_fname = os.path.join(data_folder, 'S%02d_eeg.eeg' % subject)

# read the eeg
data = np.fromfile(eeg_fname, '<i2').reshape(len(channames), -1,
        order='F')/10.

data -= data.mean(0)

# calculate spectrum of all channels before outlier/artifact rejection
F, psd_pre =  scipy.signal.welch(
        data, fs=s_rate, nperseg=1024, scaling='density')

fig = plt.figure(figsize=(10,10))
# plot with 8 rows and 4 columns
gs = mpl.gridspec.GridSpec(8,4, height_ratios = 8*[1])
ax = []
for i, (psd_chan_now) in enumerate(psd_pre):
    if i == 0:
        ax.append(fig.add_subplot(gs[0,0]))
    else:
        ax.append(fig.add_subplot(gs[i//4,i%4], sharex=ax[0], sharey=ax[0]))
    ax[-1].plot(F, np.sqrt(psd_chan_now)*1000, c='k')
    ax[-1].grid(ls=':', alpha=0.8)
    ax[-1].set_xlabel('frequency (Hz)')
    ax[-1].set_ylabel('linear spectral density')
    ax[-1].set_title(channames[i])
ax[-1].set_yscale('log')
ax[-1].set_xscale('log')
ax[-1].set_xticks([1,5,10,20,50,100])
ax[-1].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax[-1].set_ylim([1E1, 1E5])
fig.suptitle('Channel spectra, pre-cleaning, Subject S%02d' % subject,
        size=14)
gs.tight_layout(fig, pad=0.3, rect=(0,0,1,0.95))
fig.savefig(os.path.join(save_folder, 'Channel_spectra_precleaning.pdf'))
plt.close(fig)

# downsample the data
data_ds = meet.iir.butterworth(data, s_rate=s_rate,
        fs=0.5*s_rate_ds, fp=0.4*s_rate_ds)[:,::4]
t = np.arange(data_ds.shape[-1])/float(s_rate_ds)

try:
    artifact_segments = np.load(os.path.join(data_folder,
        'artifact_segments.npy'))
except:
    # manual outlier rejection
    eeg_viewer = meet.eeg_viewer.plotEEG(data_ds, channames, t)
    eeg_viewer.show()
    artifact_segments = np.asarray(eeg_viewer.select)
    np.save(os.path.join(data_folder, 'artifact_segments.npy'),
            artifact_segments)

try:
    interpolate_channels = np.atleast_1d(
            np.loadtxt(os.path.join(data_folder,
                'interpolate_channels.txt'), dtype='str'))
    interpolate_mask = np.any([np.array(channames) == ch.upper()
        for ch in interpolate_channels], 0)
    G_hh = meet.sphere._getGH(chancoords[~interpolate_mask],
            chancoords[~interpolate_mask], m=4, n=7, which='G')
    G_hw = meet.sphere._getGH(chancoords[~interpolate_mask],
            chancoords, m=4, n=7, which='G')
    data = meet.sphere._sphereSpline(data[~interpolate_mask], G_hh=G_hh,
            G_hw=G_hw, smooth=0, type='Interpolation')
    data_ds = meet.sphere._sphereSpline(data_ds[~interpolate_mask],
            G_hh=G_hh, G_hw=G_hw, smooth=0, type='Interpolation')
except:
    pass

# make a mask from the selected intervals
artifact_samples = (artifact_segments*s_rate).astype(int)
artifact_samples_ds = (artifact_segments*s_rate_ds).astype(int)
artifact_mask = np.ones(data.shape[-1], dtype=bool)
artifact_mask_ds = np.ones(data_ds.shape[-1], dtype=bool)
for start, end in artifact_samples:
    artifact_mask[start:end] = False
for start, end in artifact_samples_ds:
    artifact_mask_ds[start:end] = False

try:
    with np.load(os.path.join(data_folder, 'ICA_result.npz'), 'rb') as f:
        mixing_matrix = f['mixing_matrix']
        unmixing_matrix = f['unmixing_matrix']
except:
    # apply an ICA on the rest
    ica = FastICA(max_iter=1000, random_state=0)
    ica.fit(data_ds[:,artifact_mask_ds].T)
    mixing_matrix = ica.mixing_.T
    unmixing_matrix = ica.components_.T
    np.savez(os.path.join(data_folder, 'ICA_result.npz'),
            mixing_matrix = mixing_matrix,
            unmixing_matrix = unmixing_matrix)

# apply the ICA
sources_ds = np.dot(unmixing_matrix.T, data_ds)
sources = np.dot(unmixing_matrix.T, data)

# name the ica channels
ica_channames = ['IC%02d' % i for i in xrange(len(sources_ds))]

# plot the ICA  components scalp maps
ic_potmaps = [meet.sphere.potMap(chancoords, ic,
    projection='stereographic') for ic in mixing_matrix]

fig = plt.figure(figsize=(5,10))
# plot with 8 rows and 4 columns
gs = mpl.gridspec.GridSpec(9,4, height_ratios = 8*[1]+[0.2])
ax = []
for i, (X,Y,Z) in enumerate(ic_potmaps):
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
    ax[-1].set_title(ica_channames[i])
pc_ax = fig.add_subplot(gs[-1,:])
plt.colorbar(pc, cax=pc_ax, orientation='horizontal',
        label='relative amplitude')
pc_ax.plot([0.5,0.5], [0,1], c='w', zorder=1000,
        transform=pc_ax.transAxes)
fig.suptitle('ICA patterns, Subject S%02d' % subject, size=14)
gs.tight_layout(fig, pad=0.3, rect=(0,0,1,0.95))
fig.savefig(os.path.join(save_folder, 'ICA_patterns.pdf'))

# calculate spectrum of all ICs before outlier/artifact rejection
F_ds, psd_ica = scipy.signal.welch(
        sources_ds, fs=s_rate_ds, nperseg=1024//4, scaling='density')

fig = plt.figure(figsize=(10,10))
# plot with 8 rows and 4 columns
gs = mpl.gridspec.GridSpec(8,4, height_ratios = 8*[1])
ax = []
for i, (psd_ica_now) in enumerate(psd_ica):
    if i == 0:
        ax.append(fig.add_subplot(gs[0,0]))
    else:
        ax.append(fig.add_subplot(gs[i//4,i%4], sharex=ax[0], sharey=ax[0]))
    ax[-1].plot(F_ds, np.sqrt(psd_ica_now)*1000, c='k')
    ax[-1].grid(ls=':', alpha=0.8)
    ax[-1].set_xlabel('frequency (Hz)')
    ax[-1].set_ylabel('linear spectral density')
    ax[-1].set_title(ica_channames[i])
ax[-1].set_yscale('log')
ax[-1].set_xscale('log')
ax[-1].set_xticks([1,5,10,20,50,100])
ax[-1].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
fig.suptitle('ICA spectra, Subject S%02d' % subject, size=14)
gs.tight_layout(fig, pad=0.3, rect=(0,0,1,0.95))
fig.savefig(os.path.join(save_folder, 'ICA_spectra.pdf'))

# have a look at the ica components
ica_viewer = meet.eeg_viewer.plotEEG(sources_ds, ica_channames,
        t[artifact_mask_ds])
ica_viewer.show()
reject_ICs = np.atleast_1d(
        np.loadtxt(os.path.join(data_folder,
            'reject_ICs.txt'), dtype='str'))
ica_reject_mask = np.any([np.array(ica_channames) == ch.upper()
    for ch in reject_ICs], 0)

# get the 'clean' downsampled data
sources_ds[ica_reject_mask] = 0
unmixed_data_ds = data_ds.copy()
unmixed_data_ds = np.dot(mixing_matrix.T, sources_ds)

# get the 'clean' data at full sampling rate
sources[ica_reject_mask] = 0
unmixed_data = data.copy()
unmixed_data = np.dot(mixing_matrix.T, sources)

# plot the data before and after artifact rejection
eeg_viewer = meet.eeg_viewer.plotEEG(data_ds, channames, t)
clean_eeg_viewer = meet.eeg_viewer.plotEEG(unmixed_data_ds, channames, t)
eeg_viewer.show()

# calculate spectrum of all channels after outlier/artifact rejection
F, psd_post =  scipy.signal.welch(
        unmixed_data, fs=s_rate, nperseg=1024, scaling='density')

fig = plt.figure(figsize=(10,10))
# plot with 8 rows and 4 columns
gs = mpl.gridspec.GridSpec(8,4, height_ratios = 8*[1])
ax = []
for i, (psd_chan_now) in enumerate(psd_post):
    if i == 0:
        ax.append(fig.add_subplot(gs[0,0]))
    else:
        ax.append(fig.add_subplot(gs[i//4,i%4], sharex=ax[0], sharey=ax[0]))
    ax[-1].plot(F, np.sqrt(psd_chan_now)*1000, c='k')
    ax[-1].grid(ls=':', alpha=0.8)
    ax[-1].set_xlabel('frequency (Hz)')
    ax[-1].set_ylabel('linear spectral density')
    ax[-1].set_title(channames[i])
ax[-1].set_yscale('log')
ax[-1].set_xscale('log')
ax[-1].set_xticks([1,5,10,20,50,100])
ax[-1].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax[-1].set_ylim([1E1, 1E5])
fig.suptitle('Channel spectra, post-cleaning, Subject S%02d' % subject,
        size=14)
gs.tight_layout(fig, pad=0.3, rect=(0,0,1,0.95))
fig.savefig(os.path.join(save_folder, 'Channel_spectra_postcleaning.pdf'))
plt.close(fig)

# save the clean data
np.savez(os.path.join(data_folder, 'clean_data.npz'),
        clean_data=unmixed_data,
        artifact_mask = artifact_mask)
