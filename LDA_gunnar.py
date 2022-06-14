from __future__ import division
import numpy as np
import scipy.ndimage

def LDA(data, smooth=10):
    '''
    Calculate a linear discriminant analysis between the first and last datapoint
    in data, after smoothing

    data - 3d array
        1st dim: channels
        2nd dim: samples
        3rd dim: number of trials
    '''
    # extract data
    if smooth > 1:
        data = scipy.ndimage.filters.convolve1d(data,
                np.ones(smooth, float)/smooth, axis=1)[:,
                        int(np.ceil(smooth/2.))  - 1:
                        -int(np.floor(smooth/2.)) + 1]
    ###############
    # whiten data #
    ###############
    Why, sy = np.linalg.svd(data.reshape(data.shape[0],-1),
            full_matrices=False)[:2]
    # get rank
    py = np.linalg.matrix_rank(data.reshape(data.shape[0],-1))
    Why = Why[:,:py] / sy[:py][np.newaxis]
    # whiten and calculate means
    means = np.tensordot(Why, data, axes = (0,0)).mean(-1)[:,[0,-1]]
    ######################
    # start optimization #
    ######################
    for i in range(2):
        if i == 0:
            # get first filter
            # the covariance of data is now, after whitening, the identity
            # matrix
            # smooth, calculate the mean and take first and last point
            cfilt = means[:,0] - means[:,1]
        else:
            # get consecutive pairs of filters
            # project data into null space of previous filters
            # this is done by getting the right eigenvectors of the filter
            # maxtrix corresponding to vanishing eigenvalues
            #################################################################
            # There might be an error in this section, Needs to be checked ##
            #################################################################
            By = np.linalg.svd(np.atleast_2d(cfilt.T),
                             full_matrices=True)[2][i:].T
            #means_b = np.dot(By.T, means)
            ## get best filter for each run
            #cfilt = np.column_stack([
            #    cfilt,
            #    By.dot(means_b[:,0] - means_b[:,1])])
            cfilt = np.column_stack([
                cfilt,
                By])
    # project filters back into original (un-whitened) channel space
    diff = -1*np.diff(np.dot(cfilt.T, means), axis=-1)
    cfilt = Why.dot(cfilt)
    #normalize filters to have unit length
    cfilt = cfilt / np.sqrt(np.sum(cfilt**2, 0))
    return cfilt, diff
