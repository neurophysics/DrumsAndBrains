"""
calculates best erd/s and feeds it into REWB model:
correlation of ERD/S with performance?
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

channames = meet.sphere.getChannelNames('channels.txt')
chancoords = meet.sphere.getStandardCoordinates(channames)
chancoords_2d = meet.sphere.projectSphereOnCircle(chancoords,
        projection='stereographic')
chancoords = meet.sphere.projectCoordsOnSphere(chancoords)

snareInlier = [] # list of 20 subjects, each shape ≤75
wdBlkInlier = []
ERDCSPs = [] #ERD with shape (channels=32, time pts in ms=2500) for each subject (only alpha band for now)
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
            ERDCSPs.append(f_covmat['ERDCSP_{:02d}'.format(i)])
        i+=1
    except KeyError:
        break

# compute ERD/S (in %)
# ERD (ERS) = percentage of power decrease (increase):
# ERD% = (A-R)/R*100 (R reference period power, A event period power)
# R=0:900, A=900:2000
modelERDs = []
modelERSs = []
for i in range(20): #change to N_subjects later
    erdcsp_subj = ERDCSPs[i]
    erdcsp_p = (np.mean(erdcsp_subj[:,900:2000], axis=1) - np.mean(
        erdcsp_subj[:,:900], axis=1)
        ) / np.mean(erdcsp_subj[:,:900], axis=1) * 100
        # should be in % so -100 to 100

    # choose channel with highest ERD/S for model
    erd_idx = np.argmin(erdcsp_p) #ERD
    ers_idx = np.argmax(erdcsp_p) #ERS
    print('Subject', i+1)
    print('ERD:', round(min(erdcsp_p),2), '% at channel', channames[erd_idx])
    # fits to plot
    print('ERS: ', round(max(erdcsp_p),2), '% at channel', channames[ers_idx])
    # hmmm looks like erd as well in plot, mabe 900 is too early?

    modelERDs.append(erdcsp_subj[erd_idx])
    modelERSs.append(erdcsp_subj[ers_idx])

    #REWB2 model gets number of trials but we averaged over them?
