'''Feature prediction: prediction (test) script'''


from __future__ import print_function

import glob
import os
from itertools import product
from time import time

import numpy as np
import scipy.io as sio
import hdf5storage

import bdpy
from bdpy.util import makedir_ifnot, get_refdata
from bdpy.distcomp import DistComp

from bdpy.dataform import Features, load_array, save_array

from fastl2lir import FastL2LiR
import pandas as pd

# Main #######################################################################

def main():
    # Read settings ----------------------------------------------------

    converter_param = './params/converter_params.csv'
    df_param = pd.read_csv(converter_param)
    # Brain data
    brain_dir = '../data/fmri'
    subjects_list = {'sub-01': 'sub-01_NaturalImageTest.h5',
                     'sub-02': 'sub-02_NaturalImageTest.h5', 
                     'sub-03': 'sub-03_NaturalImageTest.h5',
                     'sub-04': 'sub-04_NaturalImageTest.h5',
                     'sub-05': 'sub-05_NaturalImageTest.h5'}

    label_name = 'image_index'

    rois_list = {
        'VC'  : 'ROI_VC = 1'
    }


    # Image features
    features_dir = '/home/share/data/contents_shared/ImageNetTest/derivatives/features'
    network = 'caffe/VGG_ILSVRC_19_layers'
    features_list = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                     'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                     'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                     'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                     'fc6', 'fc7', 'fc8'][::-1]


    # results directory
    results_dir_root = './feat_results'

    # Converter models
    nc_models_dir_root = os.path.join('./Procrustes_results', 'Procrustes_training')


    # Misc settings
    analysis_basename = os.path.splitext(os.path.basename(__file__))[0]

    # Pretrained model: modify if models are placed in other directory.
    pre_results_dir_root = '../pretrained_models'
    pre_analysis_basename = 'featdec_deeprecon_500voxel_vgg19_allunits_fastl2lir_alpha100_predict'
    pre_models_dir_root =os.path.join(pre_results_dir_root, pre_analysis_basename)

    # if the target subject have different name, please change it here.
    trg_model_name = '{}'

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: bdpy.BData(os.path.join(brain_dir, dat_file))
                  for sbj, dat_file in subjects_list.items()}
    data_features = Features(os.path.join(features_dir, network))

    # Initialize directories -------------------------------------------
    makedir_ifnot(results_dir_root)
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

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

        for feat in features_list:
            print('----------------------------------------')
            print('Layer: %s' % feat)

            # Distributed computation setup
            # -----------------------------
            conversion = src+'2'+trg
            analysis_id = analysis_basename + '-' + conversion + '-' + roi + '-' + method +'-' + str(num_samples) + '-' + feat + str(shape)
            results_dir_prediction = os.path.join(results_dir_root, analysis_basename, 'decoded_features', network, feat, conversion, roi, method, 
                                       str(num_samples))
            results_dir_accuracy = os.path.join(results_dir_root, analysis_basename, 'prediction_accuracy', network, feat, conversion, roi, method, 
                                       str(num_samples))

            if os.path.exists(results_dir_prediction):
                print('%s is already done. Skipped.' % analysis_id)
                continue

            dist = DistComp(lockdir='tmp', comp_id=analysis_id)
            if dist.islocked_lock():
                print('%s is already running. Skipped.' % analysis_id)
                continue



            # Preparing data
            # --------------
            print('Preparing data')

            start_time = time()

            # Brain data
            x = data_brain[src].select(rois_list[roi])        # Brain data
            x_labels = data_brain[src].select(label_name)  # Image labels in the brain data

            # Target features and image labels (file names)
            y = data_features.get_features(feat)
            y_labels = data_features.index
            image_names = data_features.labels

            # Get test data
            x_test = x
            x_test_labels = x_labels

            y_test = y
            y_test_labels = y_labels

            # Averaging brain data
            x_test_labels_unique = np.unique(x_test_labels)
            x_test_averaged = np.vstack([np.mean(x_test[(x_test_labels == lb).flatten(), :], axis=0) for lb in x_test_labels_unique])

            print('Total elapsed time (data preparation): %f' % (time() - start_time))

            # Convert x_test_averaged
            nc_models_dir = os.path.join(nc_models_dir_root, conversion, roi, method, 
                                   str(num_samples), 'model')
            x_test_averaged = test_ncconverter(nc_models_dir, x_test_averaged)

            # Prediction
            # ----------
            print('Prediction')

            start_time = time()
            y_pred = test_fastl2lir_div(os.path.join(pre_models_dir_root, network, feat, trg_model_name.format(trg), roi, 'model'), x_test_averaged)
            print('Total elapsed time (prediction): %f' % (time() - start_time))

            # Calculate prediction accuracy
            # -----------------------------
            print('Prediction accuracy')

            start_time = time()

            y_pred_2d = y_pred.reshape([y_pred.shape[0], -1])
            y_true_2d = y.reshape([y.shape[0], -1])

            y_true_2d = get_refdata(y_true_2d, y_labels, x_test_labels_unique)

            n_units = y_true_2d.shape[1]

            accuracy = np.array([np.corrcoef(y_pred_2d[:, i].flatten(), y_true_2d[:, i].flatten())[0, 1]
                                 for i in range(n_units)])
            accuracy = accuracy.reshape((1,) + y_pred.shape[1:])

            print('Mean prediction accuracy: {}'.format(np.nanmean(accuracy)))

            print('Total elapsed time (prediction accuracy): %f' % (time() - start_time))

            # Save results
            # ------------
            print('Saving results')

            makedir_ifnot(results_dir_prediction)
            makedir_ifnot(results_dir_accuracy)

            start_time = time()

            # Predicted features
            for i, lb in enumerate(x_test_labels_unique):
                # Predicted features
                feat = np.array([y_pred[i,]])  # To make feat shape 1 x M x N x ...

                image_filename = image_names[int(lb) - 1]  # Image labels are one-based image indexes

                # Save file name
                save_file = os.path.join(results_dir_prediction, '%s.mat' % image_filename)

                # Save
                save_array(save_file, feat, 'feat', dtype=np.float32, sparse=False) 

            print('Saved %s' % results_dir_prediction)

            # Prediction accuracy
            save_file = os.path.join(results_dir_accuracy, 'accuracy.mat')
            save_array(save_file, accuracy, 'accuracy', dtype=np.float32, sparse=False) 
            print('Saved %s' % save_file)

            print('Elapsed time (saving results): %f' % (time() - start_time))

            dist.unlock()

        print('%s finished.' % analysis_basename)


# Functions ##################################################################
def test_ncconverter(model_store, x):
    # Load NC converter
    print('Load NC converter')
    NCconverter = hdf5storage.loadmat(os.path.join(model_store, 'transformation.mat'))
    M = NCconverter['M']

    x_mean = hdf5storage.loadmat(os.path.join(model_store, 'x_mean.mat'))['x_mean']  # shape = (1, n_voxels)
    x_norm = hdf5storage.loadmat(os.path.join(model_store, 'x_norm.mat'))['x_norm']  # shape = (1, n_voxels)
    y_mean = hdf5storage.loadmat(os.path.join(model_store, 'y_mean.mat'))['y_mean']  # shape = (1, shape_features)
    y_norm = hdf5storage.loadmat(os.path.join(model_store, 'y_norm.mat'))['y_norm']  # shape = (1, shape_features)

    # Normalize X
    x = (x - x_mean) / x_norm
    # add bias term
    #x = np.hstack((x, np.ones((x.shape[0], 1))))
    converted_x = np.matmul(x, M)

    converted_x = converted_x * y_norm + y_mean
    
    return converted_x



def test_fastl2lir_div(model_store, x, chunk_axis=1):
    # W: shape = (n_voxels, shape_features)
    if os.path.isdir(os.path.join(model_store, 'W')):
        W_files = sorted(glob.glob(os.path.join(model_store, 'W', '*.mat')))
    elif os.path.isfile(os.path.join(model_store, 'W.mat')):
        W_files = [os.path.join(model_store, 'W.mat')]
    else:
        raise RuntimeError('W not found.')

    # b: shape = (1, shape_features)
    if os.path.isdir(os.path.join(model_store, 'b')):
        b_files = sorted(glob.glob(os.path.join(model_store, 'b', '*.mat')))
    elif os.path.isfile(os.path.join(model_store, 'b.mat')):
        b_files = [os.path.join(model_store, 'b.mat')]
    else:
        raise RuntimeError('b not found.')

    x_mean = hdf5storage.loadmat(os.path.join(model_store, 'x_mean.mat'))['x_mean']  # shape = (1, n_voxels)
    x_norm = hdf5storage.loadmat(os.path.join(model_store, 'x_norm.mat'))['x_norm']  # shape = (1, n_voxels)
    y_mean = hdf5storage.loadmat(os.path.join(model_store, 'y_mean.mat'))['y_mean']  # shape = (1, shape_features)
    y_norm = hdf5storage.loadmat(os.path.join(model_store, 'y_norm.mat'))['y_norm']  # shape = (1, shape_features)


    x = (x - x_mean) / x_norm

    # Prediction
    y_pred_list = []
    for i, (Wf, bf) in enumerate(zip(W_files, b_files)):
        print('Chunk %d' % i)

        start_time = time()
        W_tmp = load_array(Wf, key='W')
        b_tmp = load_array(bf, key='b')

        model = FastL2LiR(W=W_tmp, b=b_tmp)
        y_pred_tmp = model.predict(x)

        # Denormalize Y
        if y_mean.ndim == 2:
            y_pred_tmp = y_pred_tmp * y_norm + y_mean
        else:
            y_pred_tmp = y_pred_tmp * y_norm[:, [i], :] + y_mean[:, [i], :]

        y_pred_list.append(y_pred_tmp)

        print('Elapsed time: %f s' % (time() - start_time))

    return np.concatenate(y_pred_list, axis=chunk_axis)


# Entry point ################################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()
