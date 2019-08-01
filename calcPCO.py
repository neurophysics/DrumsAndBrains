import argparse
import numpy as np
import scipy
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os.path
import helper_functions
import meet
from tqdm import trange

parser = argparse.ArgumentParser(description='Calculate PCO')

parser.add_argument('N_SSD', type=int, nargs='?', default=6,
        help='the number of SSD filters to use')
parser.add_argument('result_folder', type=str, default='./Results/',
        help='the folder to store all results', nargs='?')
parser.add_argument('--normalize', type=int, default=1,
        help='whether individual subject data should be normalized')
parser.add_argument('--normSSD', type=int, default=1,
        help='whether the normalized SSD should be used')
parser.add_argument('--absdev', type=int, default=0,
        help='whether to use absolute errors')
parser.add_argument('--rank', type=int, default=1,
        help='whether data from individual subjects is rank-normalized')
args = parser.parse_args()

args.normSSD = bool(args.normSSD)
args.absdev = bool(args.absdev)
args.rank = bool(args.rank)

mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 10

cmap = 'plasma'
color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False,
        labelbottom=False)

s_rate = 1000 # sampling rate of the EEG

N_subjects = 21

# calculate the SSD from all subjects
# read the channel names
channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')

N_channels = len(channames)

# load the SSD results
with np.load(os.path.join(args.result_folder, 'SSD_norm_%s.npz' % (
    args.normSSD)),
        'r') as f:
    ssd_eigvals = f['ssd_eigvals']
    ssd_filter = f['ssd_filter']

snareInlier = []
wdBlkInlier = []
snareFitSilence = []
wdBlkFitSilence = []
snare_deviation = []
wdBlk_deviation = []

# read the oscillatory data from the silence period
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/prepared_filterdata.npz', 'r') as f:
            snareInlier.append(f['snareInlier'])
            wdBlkInlier.append(f['wdBlkInlier'])
            snareFitSilence.append(f['snareFitSilence'])
            wdBlkFitSilence.append(f['wdBlkFitSilence'])
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

# read the behavioural data
for i in xrange(1, N_subjects + 1, 1):
    try:
        with np.load(os.path.join(args.result_folder, 'S%02d' % i)
                + '/behavioural_results.npz', 'r') as f:
            snare_deviation.append(f['snare_deviation'])
            wdBlk_deviation.append(f['wdBlk_deviation'])
    except:
        print('Warning: Subject %02d could not be loaded!' %i)

# get the complex-valued fitted data
snare_data = [np.tensordot(ssd_filter[:,:args.N_SSD], f,
        axes=(0,0))[:,2:4] for f in snareFitSilence]
snare_data = [d[:,0] + 1j*d[:,1] for d in snare_data]
wdBlk_data = [np.tensordot(ssd_filter[:,:args.N_SSD], f,
        axes=(0,0))[:,4:6] for f in wdBlkFitSilence]
wdBlk_data = [d[:,0] + 1j*d[:,1] for d in wdBlk_data]

if args.normalize:
    snare_data = [d/np.sqrt(np.trace(np.cov(d.real))) for d in snare_data]
    wdBlk_data = [d/np.sqrt(np.trace(np.cov(d.real))) for d in wdBlk_data]

# only keep those trials where both, behaviour and EEG, were measured
# correctly
snare_deviation = [d[i] for d,i in zip(snare_deviation, snareInlier)]
wdBlk_deviation = [d[i] for d,i in zip(wdBlk_deviation, wdBlkInlier)]

snare_data, snare_deviation = zip(*[
    (p[:,np.isfinite(d)], d[np.isfinite(d)])
    for p, d in zip(snare_data, snare_deviation)])
wdBlk_data, wdBlk_deviation = zip(*[
    (p[:,np.isfinite(d)], d[np.isfinite(d)])
    for p, d in zip(wdBlk_data, wdBlk_deviation)])

if args.absdev:
    snare_deviation = [np.abs(d) for d in snare_deviation]
    wdBlk_deviation = [np.abs(d) for d in wdBlk_deviation]

if args.rank:
    snare_deviation = [d.argsort().argsort()/(len(d) - 1.)
            for d in snare_deviation]
    wdBlk_deviation = [d.argsort().argsort()/(len(d) - 1.)
            for d in wdBlk_deviation]

# concatenate all the data
snare_data = np.hstack(snare_data)
wdBlk_data = np.hstack(wdBlk_data)
snare_deviation = np.hstack(snare_deviation)
wdBlk_deviation = np.hstack(wdBlk_deviation)

# calculate the PCO filter for snare data
snare_vlen, snare_PCO_filt = meet.spatfilt.PCOa(snare_deviation, snare_data)
snare_vlen_boot = np.hstack([meet.spatfilt.PCOa(
    snare_deviation[np.random.randn(len(snare_deviation)).argsort()],
    snare_data)[0][0] for _ in trange(1000)])
snare_p = (snare_vlen_boot >= snare_vlen).mean()

# calculate the PCO filter for woodblock data
wdBlk_vlen, wdBlk_PCO_filt = meet.spatfilt.PCOa(wdBlk_deviation, wdBlk_data)
wdBlk_vlen_boot = np.hstack([meet.spatfilt.PCOa(
    wdBlk_deviation[np.random.randn(len(wdBlk_deviation)).argsort()],
    wdBlk_data)[0][0] for _ in trange(1000)])
wdBlk_p = (wdBlk_vlen_boot >= wdBlk_vlen).mean()
