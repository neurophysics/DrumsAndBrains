import numpy as np
import tables
import meet
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# load spoc pattern
parser = argparse.ArgumentParser(
        description='Calculate source from cSpoC pattern')
parser.add_argument('result_folder', type=str, default='./Results/',
        help='the folder to store all results', nargs='?')
args = parser.parse_args()
try:
    with np.load(os.path.join(args.result_folder,
            'calcFFTcSpoCv2_spatial_pattern.npz'), 'r') as f:
        wdBlk_pattern = f['wdBlk_pattern'][0] #shape (6,32) polarities for all
        snare_pattern = f['snare_pattern'][0] # channels and per session
except:
    print('Warning: Could not load cSpoC pattern')

# load ny head
with tables.open_file('sa_nyhead.mat', 'r') as f:
    cortex = f.root.sa.cortex2K.in_from_cortex75K.read()
    mni2mri = f.root.sa.mni2mri.read()
    large_leadfield = f.root.sa.cortex75K.V_fem_normal.read() #assume orthogonal axons
    vertex_indices = f.root.sa.cortex2K.in_from_cortex75K.read()
    vertex_indices = vertex_indices.astype(int)[0]-1 #-1 bc matlab indices!!!
    ascii_labels = f.root.sa.clab_electrodes.read() #array von ascii, mit chr(c) konvertieren
    # indices 0 to 1001 left, 1002 to 2003 right:
    #inLeft = f.root.sa.cortex2K.in_left.read().astype(int)-1
    #inRight = f.root.sa.cortex2K.in_right.read().astype(int)-1
    coord = f.root.sa.cortex75K.vc_smooth.read()[:,vertex_indices]
    mri = f.root.sa.mri.data.read()

# map channels to ours
ch_labels = []
for elem in ascii_labels:
    ch_labels.append(''.join([chr(c) for c in elem[0]]))
channames = meet.sphere.getChannelNames('channels.txt')
channames = [x.lower() for x in channames]
ch_indices = [i for i, x in enumerate(ch_labels) if x.lower() in channames]

# reduce to 2k voxel x 30(exclude TP9 and 10) channel leadfield matrix
# and 30 channel patterns
leadfield2k = large_leadfield[vertex_indices,:]
leadfield = leadfield2k[:, ch_indices] # reduce #channels
index = [i for i,x in enumerate(channames)
    if x.lower()=='tp9' or x.lower()=='tp10']
snare_pattern = np.delete(snare_pattern,index).reshape(1,-1)
wdBlk_pattern = np.delete(wdBlk_pattern,index).reshape(1,-1)

# choose single best voxel with max cosine similarity
snare_sim = cosine_similarity(leadfield, snare_pattern)
snare_singleFit = np.argmax(abs(snare_sim))
wdBlk_sim = cosine_similarity(leadfield, wdBlk_pattern)
wdBlk_singleFit = np.argmax(abs(wdBlk_sim))

# pairwise best voxels
# check if pairs (i,i+1002) are correct (left,right) voxel pairs using coordinates:
# sum([((coord[0,l])*(-1)==coord[0,l+1002]) for l in range(1002)]) # only differ in first coord
leadfield_paired = np.zeros((1002,30))
for i in range(1002):
    leadfield_paired[i,:] = leadfield[i,:]+leadfield[i+1002,:]

snare_pairedSim = cosine_similarity(leadfield_paired, snare_pattern)
snare_pairedFit = (np.argmax(abs(snare_pairedSim)),
    np.argmax(abs(snare_pairedSim))+1002)
wdBlk_pairedSim = cosine_similarity(leadfield_paired, wdBlk_pattern)
wdBlk_pairedFit = (np.argmax(abs(wdBlk_pairedSim)),
    np.argmax(abs(wdBlk_pairedSim))+1002)
print('Best single Fits: snare '+str(snare_singleFit)+
    ', wood block '+str(wdBlk_singleFit))
print('Best paired Fits: snare '+str(snare_pairedFit)+
    ', wood block '+str(wdBlk_pairedFit))

# Visualize
snare_singleCoords = coord[:,snare_singleFit]
snare_pairedCoords = (coord[:,snare_pairedFit[0]],coord[:,snare_pairedFit[1]])
wdBlk_singleCoords = coord[:,wdBlk_singleFit]
wdBlk_pairedCoords = (coord[:,wdBlk_pairedFit[0]],coord[:,wdBlk_pairedFit[1]])
