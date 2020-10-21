import numpy as np
import tables
import meet
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats

# load spoc pattern
parser = argparse.ArgumentParser(
        description='Calculate source from cSpoC pattern')
parser.add_argument('result_folder', type=str, default='./Results/',
        help='the folder to store all results', nargs='?')
args = parser.parse_args()


scatter_cmap = 'OrRd'
scatter_cmap_inst = mpl.cm.get_cmap(scatter_cmap)

try:
    with np.load(os.path.join(args.result_folder,
            'FFTSSD.npz'), 'r') as f:
        wdBlk_pattern = f['SSD_patterns'][0] # shape (32,) polarities for all
        snare_pattern = f['SSD_patterns'][0] # channels
except:
    print('Warning: Could not load cSpoC pattern')

# load ny head
with tables.open_file('sa_nyhead.mat', 'r') as f:
    mni2mri_matrix = f.root.sa.mni2mri.read()
    large_coord = f.root.sa.cortex75K.vc.read()
    large_leadfield = f.root.sa.cortex75K.V_fem_normal.read() #assume axons oriented perpendiculary to cortical surface
    sulcimap = f.root.sa.cortex75K.sulcimap.read().astype(bool).ravel()
    vertex_indices = f.root.sa.cortex2K.in_from_cortex75K.read().ravel().astype(int) - 1 # -1 because this gives matlab indices
    electrode_labels = f.root.sa.clab_electrodes.read()
    # indices 0 to 1001 left, 1002 to 2003 right:
    #inLeft = f.root.sa.cortex2K.in_left.read().astype(int)-1
    #inRight = f.root.sa.cortex2K.in_right.read().astype(int)-1
    mri = f.root.sa.mri.data.read()

#reduce to 2k voxels
#vertex_indices = np.intersect1d(np.where(sulcimap)[0], vertex_indices)
coord = large_coord[:,vertex_indices]
leadfield = large_leadfield[vertex_indices] # reduce voxels and channels

# reduce EEG electrodes to ours (30 electrodes, after excluding TP9 and 10)
ch_labels = []
for elem in electrode_labels:
    ch_labels.append(''.join([chr(int(c)) for c in elem[0]]))
channames = meet.sphere.getChannelNames('channels.txt')
channames = [x.lower() for x in channames]
ch_indices = [i for i, x in enumerate(ch_labels) if x.lower() in channames]

leadfield = leadfield[:, ch_indices]

# remove TP9 and TP10 from the patterns (they are not included in the NY head)
index = [i for i,x in enumerate(channames)
    if x.lower()=='tp9' or x.lower()=='tp10']
snare_pattern = (np.delete(snare_pattern,index).reshape(1,-1)
    - snare_pattern.mean())
wdBlk_pattern = (np.delete(wdBlk_pattern,index).reshape(1,-1)
    - snare_pattern.mean())

# Calculate similarity between sPoC-Pattern and leadfield projections
# choose single best voxel with max cosine similarity
snare_sim = cosine_similarity(leadfield, snare_pattern)
snare_singleFit = np.argmax(abs(snare_sim))
wdBlk_sim = cosine_similarity(leadfield, wdBlk_pattern)
wdBlk_singleFit = np.argmax(abs(wdBlk_sim))

# pairwise best voxels
# check if pairs (i,i+1002) are correct (left,right) voxel pairs using coordinates:
# sum([((coord[0,l])*(-1)==coord[0,l+1002]) for l in range(1002)]) # only differ in first coord
leadfield_paired = np.zeros([leadfield.shape[0]//2, leadfield.shape[1]])
for i in range(leadfield.shape[0]//2):
    leadfield_paired[i] = leadfield[i] + leadfield[i + leadfield.shape[0]//2]

snare_pairedSim = cosine_similarity(leadfield_paired, snare_pattern)
snare_pairedFit = (np.argmax(abs(snare_pairedSim)),
    np.argmax(abs(snare_pairedSim))+leadfield.shape[0]//2)
wdBlk_pairedSim = cosine_similarity(leadfield_paired, wdBlk_pattern)
wdBlk_pairedFit = (np.argmax(abs(wdBlk_pairedSim)),
    np.argmax(abs(wdBlk_pairedSim))+leadfield.shape[0]//2)

def mni2mri(inpoint, mat=mni2mri_matrix):
    '''transform mni coordinates to mri coordinates using given
    transformation matrix'''
    inpoint = np.r_[inpoint, 1]
    return np.dot(inpoint,mat)[:3]

# Visualize
snare_singleMNI = coord[:,snare_singleFit]
snare_pairedMNI = (coord[:,snare_pairedFit[0]] ,coord[:,snare_pairedFit[1]])
snare_singleMRI = mni2mri(snare_singleMNI)
snare_pairedMRI = np.array([
    mni2mri(snare_pairedMNI[0]),mni2mri(snare_pairedMNI[1])])

mri = mri.swapaxes(0,2) #X should be left to right, Y back front, Z down up
#mri.shape = (394, 466, 378)
l = snare_pairedMRI[0].astype(int)
#l = (l[0],l[2],l[1]) #swap axes as in mri
r = snare_pairedMRI[1].astype(int)
#r = (r[0],r[2],r[1]) #swap axes as in mri

# define colormap
cmap = 'bone'
# define keyword argument dict for axes without any labels
blank_ax = dict(top=False, bottom=False, left=False, right=False,
        labeltop=False, labelbottom=False,
        labelleft=False, labelright=False)

snare_pairedSim_norm = np.abs(snare_pairedSim)
snare_pairedSim_norm /= snare_pairedSim_norm.max()

snare_singleSim_norm = np.abs(snare_sim)
snare_singleSim_norm /= snare_singleSim_norm.max()

##########################
# plot paired similarity #
##########################
fig1, ax1 = plt.subplots(3,1,figsize=(4,10))
ax1[0].imshow(mri[mri.shape[0]//2 + 5,:,:].T, cmap=cmap, origin='lower',
        aspect='equal') #lim: 466x378
ax1[0].set_title('Sagittal')

for i, sim in enumerate(snare_pairedSim_norm.ravel()):
    l_now,r_now = mni2mri(coord[:,i]), mni2mri(np.r_[-1,1,1]*coord[:,i]) 
    ax1[0].scatter(l_now[1],l_now[2], s=75, color=scatter_cmap_inst(sim),
            alpha=0.9*sim, edgecolors='none', zorder=sim)
    ax1[0].scatter(r_now[1],r_now[2], s=75, color=scatter_cmap_inst(sim),
            alpha=0.9*sim, edgecolors='none', zorder=sim)

ax1[1].imshow(mri[:,mri.shape[1]//2,:].T, cmap=cmap, origin='lower',
        aspect='equal') #lim: 394x378
ax1[1].set_title('Coronal')

for i, sim in enumerate(snare_pairedSim_norm.ravel()):
    l_now,r_now = mni2mri(coord[:,i]), mni2mri(np.r_[-1,1,1]*coord[:,i]) 
    ax1[1].scatter(l_now[0],l_now[2], s=75, color=scatter_cmap_inst(sim),
            alpha=0.9*sim, edgecolors='none', zorder=sim)
    ax1[1].scatter(r_now[0],r_now[2], s=75, color=scatter_cmap_inst(sim),
            alpha=0.9*sim, edgecolors='none', zorder=sim)

ax1[2].imshow(mri[:,:,mri.shape[2]//2].T, cmap=cmap, origin='lower',
        aspect='equal') #lim: 394x466
ax1[2].set_title('Horizontal')

for i, sim in enumerate(snare_pairedSim_norm.ravel()):
    l_now,r_now = mni2mri(coord[:,i]), mni2mri(np.r_[-1,1,1]*coord[:,i]) 
    ax1[2].scatter(l_now[0],l_now[1], s=75, color=scatter_cmap_inst(sim),
            alpha=0.9*sim, edgecolors='none', zorder=sim)
    ax1[2].scatter(r_now[0],r_now[1], s=75, color=scatter_cmap_inst(sim),
            alpha=0.9*sim, edgecolors='none', zorder=sim)

# remove frame and labels
for ax_now in ax1:
    ax_now.tick_params(**blank_ax)
    ax_now.set_frame_on(False)

fig1.tight_layout()

##########################
# plot single similarity #
##########################
fig2, ax2 = plt.subplots(3,1,figsize=(4,10))
ax2[0].imshow(mri[mri.shape[0]//2 + 5,:,:].T, cmap=cmap, origin='lower',
        aspect='equal') #lim: 466x378
ax2[0].set_title('Sagittal')

for i, sim in enumerate(snare_singleSim_norm.ravel()):
    l_now = mni2mri(coord[:,i])
    ax2[0].scatter(l_now[1],l_now[2], s=75, color=scatter_cmap_inst(sim),
            alpha=0.9*sim, edgecolors='none', zorder=sim)

ax2[1].imshow(mri[:,mri.shape[1]//2,:].T, cmap=cmap, origin='lower',
        aspect='equal') #lim: 394x378
ax2[1].set_title('Coronal')

for i, sim in enumerate(snare_singleSim_norm.ravel()):
    l_now = mni2mri(coord[:,i])
    ax2[1].scatter(l_now[0],l_now[2], s=75, color=scatter_cmap_inst(sim),
            alpha=0.9*sim, edgecolors='none', zorder=sim)

ax2[2].imshow(mri[:,:,mri.shape[2]//2].T, cmap=cmap, origin='lower',
        aspect='equal') #lim: 394x466
ax2[2].set_title('Horizontal')

for i, sim in enumerate(snare_singleSim_norm.ravel()):
    l_now = mni2mri(coord[:,i])
    ax2[2].scatter(l_now[0],l_now[1], s=75, color=scatter_cmap_inst(sim),
            alpha=0.9*sim, edgecolors='none', zorder=sim)

# remove frame and labels
for ax_now in ax2:
    ax_now.tick_params(**blank_ax)
    ax_now.set_frame_on(False)

fig2.tight_layout()

plt.show()


def plotMNI(coordinates, figname):
    x,y,z = coordinates
    fig,ax = plt.subplots(3,1,figsize=(4,10))
    plt.subplots_adjust(hspace=0.4)
    ax[0].imshow(mri[x,:,:].T) #lim: 466x378
    ax[0].set_title('Sagittal')
    ax[0].invert_yaxis() #otherwise the picture is turned upside down
    ax[0].scatter(y,z,s=5, c='red')

    ax[1].imshow(mri[:,y,:].T) #lim: 394x378
    ax[1].invert_yaxis()
    ax[1].scatter(x,z,s=5, c='red')
    ax[1].set_title('Coronal')

    ax[2].imshow(mri[:,:,z].T) #lim: 394x466
    ax[2].invert_yaxis()
    ax[2].scatter(x,y,s=5, c='red')
    ax[2].set_title('Horizontal')
    plt.savefig(figname)

occ_coordMRI = (-8,-76,-8)
occ = mni2mri(test_coordMRI).astype(int)
plotMNI(occ,'test_occipital.png')
plotMNI(mni2mri((32,-4,-50)).astype(int),'test_temporal.png')
plotMNI(mni2mri((28,-4,-26)).astype(int),'test_amygdala.png')
plotMNI(mni2mri((50,28,34)).astype(int),'test_PFC.png')
plotMNI(mni2mri((10,26,44)).astype(int),'test_dACC.png')
plotMNI(mni2mri((4,-10,4)).astype(int),'test_thalamus.png')
