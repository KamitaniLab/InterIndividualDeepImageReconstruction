from __future__ import print_function

import os
import warnings
import yaml
from itertools import product
from time import time

import hdf5storage
import numpy as np
import scipy.io as sio

import bdpy
from bdpy.distcomp import DistComp
from bdpy.util import makedir_ifnot
from bdpy.dataform import save_array

from fastl2lir import FastL2LiR

import json
import pandas as pd

from pyhyperalignment_revise import Hyperalignment

# Main #######################################################################

def main():
    # Data settings ----------------------------------------------------

    # Brain data
    converter_param = './params/converter_params_1conversion.csv'
    # To train converter for all subject pairs and with varying training samples, uncommented the line below.
    # converter_param = './params/converter_params.csv'
    df_param = pd.read_csv(converter_param)
    brain_dir = '../data/fmri'
    subjects_list = {'sub-01': 'sub-01_NaturalImageTraining.h5',
                     'sub-02': 'sub-02_NaturalImageTraining.h5', 
                     'sub-03': 'sub-03_NaturalImageTraining.h5',
                     'sub-04': 'sub-04_NaturalImageTraining.h5',
                     'sub-05': 'sub-05_NaturalImageTraining.h5'}

    data_brain = {subject: bdpy.BData(os.path.join(brain_dir, dat_file))
                for subject, dat_file in subjects_list.items()}

    label_name = 'image_index'

    rois_list = {
        'VC'  : 'ROI_VC = 1'
    }

    methods = {'wholeVC': ['ROI_LH_VC = 1', 'ROI_RH_VC = 1'],
               'subareawise': ['ROI_LH_V1 = 1', 'ROI_LH_V2 = 1', 'ROI_LH_V3 = 1', 'ROI_LH_hV4 = 1', 'ROI_LH_HVC = 1',
                               'ROI_RH_V1 = 1', 'ROI_RH_V2 = 1', 'ROI_RH_V3 = 1', 'ROI_RH_hV4 = 1', 'ROI_RH_HVC = 1']}

    # Results directory
    results_dir_root = './template_pairwise_results'
    tmp_dir = './tmp'

    analysis_basename = os.path.splitext(os.path.basename(__file__))[0]
    for index, row in df_param.iterrows():
        src = str(row['Source'])
        trg = str(row['Target'])
        roi = str(row['ROI'])
        method = str(row['Method'])
        num_samples = int(row['Number of samples'])
        print('--------------------')
        print('Source: %s' % src)
        print('Target: %s' % trg)
        print('ROI: %s' % roi)
        print('Number of samples: %s' % num_samples)
        print('Method: %s' % method)


        # Setup
        # -----
        conversion = src+'2'+trg
        analysis_id = analysis_basename + '-' + conversion + '-' + roi + '-' + method +'-' + str(num_samples) 
        results_dir = os.path.join(results_dir_root, analysis_basename, conversion, roi, method, 
                                   str(num_samples), 'model')
        makedir_ifnot(results_dir)
        makedir_ifnot(tmp_dir)

        # Check whether the analysis has been done or not.
        result_model = os.path.join(results_dir, 'transformation.mat')
        if os.path.exists(result_model):
            print('%s already exists and skipped' % result_model)
            continue

        dist = DistComp(lockdir=tmp_dir, comp_id=analysis_id)
        if dist.islocked():
            print('%s is already running. Skipped.' % analysis_id)
            continue
        dist.lock()

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        idxs_merge_src = compute_roi_index(data_brain[src], rois_list[roi], methods[method])
        idxs_merge_trg = compute_roi_index(data_brain[trg], rois_list[roi], methods[method])

        # Brain data
        x = data_brain[src].select(rois_list[roi])        # Brain data
        x_labels = data_brain[src].select(label_name)  # Image labels in the brain data

        y = data_brain[trg].select(rois_list[roi])
        y_labels = data_brain[trg].select(label_name)


        print('Total elapsed time (data preparation): %f' % (time() - start_time))

        # Model training
        # --------------
        print('Model training')
        start_time = time()
        train_NCconverter(x, y,
                          x_labels, y_labels, idxs_merge_src, idxs_merge_trg, num_samples,
                          output=results_dir, save_chunk=True,
                          axis_chunk=1, tmp_dir='tmp',
                          comp_id=analysis_id)
        dist.unlock()
        print('Total elapsed time (model training): %f' % (time() - start_time))

    print('%s finished.' % analysis_basename)


# Functions ##################################################################
def train_NCconverter(x, y, x_labels, y_labels, idxs_merge_src, idxs_merge_trg, num_sample,
                      output='./NCconverter_results.mat', save_chunk=False,
                      axis_chunk=1, tmp_dir='./tmp',
                      comp_id=None):
    

    if y.ndim == 4:
        # The Y input to the NCconveter has to be strictly number of samples x number of features
        y = y.reshape((y.shape[0], -1))
    elif y.ndim == 2:
        pass
    else:
        raise ValueError('Unsupported feature array shape')

    # Sample selection ------------------------------------------------------
    # The dataset contains 6000 samples, here we choose the needed sample size.
    # Sort the X and Y data, such that they are aligned and is easy to select the data.
    # after sorting the label becomes [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,...]
    x_index = np.argsort(x_labels.flatten())
    x_labels = x_labels[x_index]
    x = x[x_index,:]

    y_index = np.argsort(y_labels.flatten())
    y_labels = y_labels[y_index]
    y = y[y_index,:]

    # If we only need a sample size smaller than 1200, we choose from the first repetition.
    if num_sample < 1200:
        rep = 1
    else:
        rep = int(num_sample/1200)

    # select the needed repetitions.
    tmp = np.zeros(5,dtype=bool)
    tmp[:rep] = True
    sel = np.tile(tmp, 1200)
    x_labels = x_labels[sel]
    x = x[sel]
    y_labels = y_labels[sel]
    y = y[sel]

    # If we only need a sample size smaller than 1200, samples belongs to different categories are chosen to avoid any bias.
    # Here we have 150 image categories, 8 images per category 
    if num_sample==300:
        # 2 images per category
        x = x[0::4]
        y = y[0::4]
        x_labels = x_labels[0::4]
        y_labels = y_labels[0::4]    
    
    elif num_sample==600:
        # 4 images per category
        x = np.vstack((x[0::4], x[1::4]))
        y = np.vstack((y[0::4], y[1::4]))
        x_labels = np.vstack((x_labels[0::4], x_labels[1::4]))
        y_labels = np.vstack((y_labels[0::4], y_labels[1::4]))

    elif num_sample==900:   
        # 6 images per category
        x = np.vstack((x[0::4], x[1::4], x[2::4]))
        y = np.vstack((y[0::4], y[1::4], y[2::4]))
        x_labels = np.vstack((x_labels[0::4], x_labels[1::4], x_labels[2::4]))
        y_labels = np.vstack((y_labels[0::4], y_labels[1::4], y_labels[2::4]))


    # Preprocessing ----------------------------------------------------------
    print('Preprocessing')
    start_time = time()

    # Normalize X (source fMRI data)
    x_mean = np.mean(x, axis=0)[np.newaxis, :] # np.newaxis was added to match Matlab outputs
    x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]
    x_normalized = (x - x_mean) / x_norm


    # Normalize Y (target fMRI data)
    y_mean = np.mean(y, axis=0)[np.newaxis, :]
    y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]
    y_normalized = (y - y_mean) / y_norm

    print('Elapsed time: %f' % (time() - start_time))

    # Model training loop ----------------------------------------------------
    start_time = time()
    print('Training')
    # Model training
    M = np.zeros((x_normalized.shape[1], y_normalized.shape[1]))
    for k in idxs_merge_src.keys():
        idxs_sub_src = idxs_merge_src[k]
        idxs_sub_trg = idxs_merge_trg[k]
        
        model = Hyperalignment()
        Ws = model.train([x_normalized[:,idxs_sub_src],  y_normalized[:, idxs_sub_trg]])
        shared_space_dim = model.tm
        W_src = Ws[0]
        W_trg = Ws[1]
        print(W_src.shape)
        print(W_trg.shape)

        if len(idxs_sub_src) > shared_space_dim:
            W_src = W_src[:,:shared_space_dim]
        elif len(idxs_sub_src) < shared_space_dim:
            W_src = W_src[:len(idxs_sub_src),:]

        if len(idxs_sub_trg) > shared_space_dim:
            W_trg = W_trg[:,:shared_space_dim]
        elif len(idxs_sub_trg) < shared_space_dim:
            W_trg = W_trg[:len(idxs_sub_trg),:]

        print(W_src.shape)
        print(W_trg.shape)

        W = np.matmul(W_src, W_trg.T)


        grid = tuple(np.meshgrid(idxs_sub_src, idxs_sub_trg, indexing='ij'))
        M_grid = M[grid]
        W[M_grid != 0.0] = 0.0

        M[grid] += W
        #M[-1, idxs_sub_trg] += b

    # Save chunk results
    result_model = os.path.join(output, 'transformation.mat')
    save_array(result_model, M, 'M', dtype=np.float32, sparse=False) 
    print('Saved %s' % result_model)

    etime = time() - start_time
    print('Elapsed time: %f' % etime)

    # Save results -----------------------------------------------------------
    print('Saving normalization parameters.')
    norm_param = {'x_mean' : x_mean, 'y_mean' : y_mean,
                    'x_norm' : x_norm, 'y_norm' : y_norm}
    save_targets = [u'x_mean', u'y_mean', u'x_norm', u'y_norm']
    for sv in save_targets:
        save_file = os.path.join(output, sv + '.mat')
        if not os.path.exists(save_file):
            try:
                save_array(save_file, norm_param[sv], key=sv, dtype=np.float32, sparse=False)
                print('Saved %s' % save_file)
            except IOError:
                warnings.warn('Failed to save %s. Possibly double running.' % save_file)


    if not save_chunk:
        # Merge results into 'model'mat'
        raise NotImplementedError('Result merging is not implemented yet.')

    return None
    

def compute_roi_index(data, embeded_roi, roi_mapping):
    '''
    To aggregate the conversion matrices for each brain area,
    the embedded indices for each brain area in the VC is necessary.
    '''
    _, base_idx = data.select(embeded_roi, return_index=True)
    del _

    idx_loc = np.where(base_idx)[0]
    idx_mapper = dict(zip(idx_loc, range(len(idx_loc))))

    idxs = {}
    for roi in roi_mapping:
        
        _, idx = data.select(roi, return_index=True)
        
        loc = np.where(idx)[0]
        idxs[roi] = [idx_mapper[l] for l in loc]

    return idxs

# Entry point ################################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()