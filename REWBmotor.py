"""
calculates best erd/s and feeds it into REWB model:
correlation of ERD/S with performance?
later also correlation of BP and performance
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

'''channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)'''

snareInlier = [] # list of 20 subjects, each shape ≤75
wdBlkInlier = []
ERDCSP_trial = [] #stores ERD_CSP of best CSPcomp per subject, each shape (Ntrial,)
ERSCSP_trial = [] # same for ERS
i=0
while True:
    try:
        with np.load(os.path.join(result_folder, 'motor/inlier.npz'),
            'r') as fi:
            snareInlier.append(fi['snareInlier_response_{:02d}'.format(i)])
            wdBlkInlier.append(fi['wdBlkInlier_response_{:02d}'.format(i)])
            win = fi['win']
        with np.load(os.path.join(result_folder, 'motor/erdcsp.npz'),
            'r') as f_covmat:
            ERDCSP_trial.append(f_covmat['ERDCSP_trial_{:02d}'.format(i)])
            ERSCSP_trial.append(f_covmat['ERSCSP_trial_{:02d}'.format(i)])
        i+=1
    except KeyError:
        break
