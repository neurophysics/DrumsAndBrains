'uses LDA to transform into separable BP and contrast group'

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
with np.load(os.path.join(result_folder, 'motor/covmat.npz'), 'r') as f:
    act_idx = f['act_idx']  #-500 to 0ms
    base_idx = f['base_idx'] #corresponds to -2000 to -1250ms

# stitch all subjects together so we have many trials
all_BP = np.concatenate(all_BP, axis=-1) #now shape (channels,trials) = (32,2500,20*1xx)
all_BP = all_BP - all_BP[:,:1400,:].mean(1)[:,np.newaxis,:] #baseline here up to 1400, only changes plot not filter
# divide into classes, both shape (channels, trials)
BP = all_BP[:,act_idx,:].mean(1)
contrast = all_BP[:,base_idx,:].mean(1)

# center classwise, estimate cov on all features at once
Xpool = np.hstack([BP-BP.mean(-1)[:,np.newaxis],
    contrast-contrast.mean(-1)[:,np.newaxis]])
C = np.cov(Xpool) #(32,32)
cfilt = np.linalg.pinv(C).dot(contrast.mean(-1) - BP.mean(-1))

# alternatively, use gunnars LDA:
#cfilt,diff = g.LDA(all_BP[:,:,:1900]), smooth=500)ull
np.save(os.path.join(result_folder, 'motor/lda.npy'), cfilt)


# apply lda to bp
all_BP = []
i=0
while True:
    try:
        with np.load(os.path.join(result_folder, 'motor/BP.npz'),
            'r') as fb:
            all_BP.append(fb['BP_{:02d}'.format(i)])
        i+=1
    except KeyError:
        break
BPlda = [np.tensordot(cfilt, b, axes=(0,0)) for b in all_BP] #each shape (2500, 143) now
#convert BP to decrease/increase value i.e. activation avg - baseline avg
BPLDA = [a[act_idx].mean(0) - a[base_idx].mean(0) for a in BPlda] #(143,) each

save_results = {}
for i,b in enumerate(BPLDA):
    save_results['BPLDA_{:02d}'.format(i)] = b
np.savez(os.path.join(result_folder, 'motor', 'BPLDA.npz'), **save_results)

##### plots #####
#check component
# mean = cfilt.dot(np.array(all_BP).mean(0)).T
# sd = np.tensordot(cfilt, np.array(all_BP).mean(0), axes=[0,0]).std(-1).T
# plt.figure()
# plt.plot(mean)
# plt.plot(mean+sd, c='b', alpha = 0.5)
# plt.plot(mean-sd, c='b', alpha = 0.5)
# plt.show()
