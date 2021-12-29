'''Script for Deep Image Reconstruction.
Modified from: https://github.com/KamitaniLab/DeepImageReconstruction

This is mainly used for the reconstruction of artificial images.

- Reconstruction algorithm: L-BFGS (without deep generator network)
- DNN: Caffe VGG19
- Layers: All conv and fc layers
- Images: ImageNetTest
'''


import os
import pickle
from datetime import datetime
from itertools import product

import caffe
import h5py
import numpy as np
import PIL.Image
import scipy.io as sio

from icnn.icnn_lbfgs import reconstruct_image
from icnn.utils import clip_extreme_value, estimate_cnn_feat_std, normalise_img

import pandas as pd

from bdpy.distcomp import DistComp
from bdpy.util import makedir_ifnot


# Settings ###################################################################

# GPU usage settings
caffe.set_mode_gpu()
caffe.set_device(0)

converter_param = '../params/converter_params.csv'

df_param = pd.read_csv(converter_param)

# Decoded features settings
decoded_features_result_dir = '../feat_results/ncc_predict_feat'
decoded_features_dir = os.path.join(decoded_features_result_dir, 'decoded_features')

#decode_feature_filename = lambda net, layer, subject, roi, image_label: os.path.join(decoded_features_dir, net, layer, subject, roi, image_label + '.mat')
def decode_feature_filename(decoded_features_dir, net, layer, conversion, roi, method, num_samples, alpha, image_label):
    return os.path.join(decoded_features_dir, net, layer, conversion, roi, method, str(num_samples), str(alpha), image_label + '.mat')

# Data settings
results_dir = './recon_img_results'
network = 'caffe/VGG_ILSVRC_19_layers'

image_label_list = ['Img0001',
                    'Img0002',
                    'Img0003',
                    'Img0004',
                    'Img0005',
                    'Img0006',
                    'Img0007',
                    'Img0008',
                    'Img0009',
                    'Img0010',
                    'Img0011',
                    'Img0012',
                    'Img0013',
                    'Img0014',
                    'Img0015',
                    'Img0016',
                    'Img0017',
                    'Img0018',
                    'Img0019',
                    'Img0020',
                    'Img0021',
                    'Img0022',
                    'Img0023',
                    'Img0024',
                    'Img0025',
                    'Img0026',
                    'Img0027',
                    'Img0028',
                    'Img0029',
                    'Img0030',
                    'Img0031',
                    'Img0032',
                    'Img0033',
                    'Img0034',
                    'Img0035',
                    'Img0036',
                    'Img0037',
                    'Img0038',
                    'Img0039',
                    'Img0040']

max_iteration = 200

# Main #######################################################################

# Initialize CNN -------------------------------------------------------------

# Average image of ImageNet
img_mean_file = './data/ilsvrc_2012_mean.npy'
img_mean = np.load(img_mean_file)
img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

# CNN model
model_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel'
prototxt_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.prototxt'
channel_swap = (2, 1, 0)
net = caffe.Classifier(prototxt_file, model_file, mean=img_mean, channel_swap=channel_swap)
h, w = net.blobs['data'].data.shape[-2:]
net.blobs['data'].reshape(1, 3, h, w)

# Initial image for the optimization (here we use the mean of ilsvrc_2012_mean.npy as RGB values)
initial_image = np.zeros((h, w, 3), dtype='float32')
initial_image[:, :, 0] = img_mean[2].copy()
initial_image[:, :, 1] = img_mean[1].copy()
initial_image[:, :, 2] = img_mean[0].copy()

# Feature SD estimated from true CNN features of 10000 images
feat_std_file = './data/estimated_vgg19_cnn_feat_std.mat'
feat_std0 = sio.loadmat(feat_std_file)

# CNN Layers (all conv and fc layers)
layers = [layer for layer in net.blobs.keys() if 'conv' in layer or 'fc' in layer]

# Setup results directory ----------------------------------------------------

save_dir_root = os.path.join(results_dir, os.path.splitext(__file__)[0])
if not os.path.exists(save_dir_root):
    os.makedirs(save_dir_root)

# Set reconstruction options -------------------------------------------------

opts = {
    # The loss function type: {'l2','l1','inner','gram'}
    'loss_type': 'l2',

    # The maximum number of iterations
    'maxiter': max_iteration,

    # The initial image for the optimization (setting to None will use random noise as initial image)
    'initial_image': initial_image,

    # A python dictionary consists of channels to be selected, arranged in
    # pairs of layer name (key) and channel numbers (value); the channel
    # numbers of each layer are the channels to be used in the loss function;
    # use all the channels if some layer not in the dictionary; setting to None
    # for using all channels for all layers;
    'channel': None,

    # A python dictionary consists of masks for the traget CNN features,
    # arranged in pairs of layer name (key) and mask (value); the mask selects
    # units for each layer to be used in the loss function (1: using the uint;
    # 0: excluding the unit); mask can be 3D or 2D numpy array; use all the
    # units if some layer not in the dictionary; setting to None for using all
    #units for all layers;
    'mask': None,

    # Display the information on the terminal or not
    'disp': True
}

# Save the optional parameters
with open(os.path.join(save_dir_root, 'options.pkl'), 'w') as f:
    pickle.dump(opts, f)

# Reconstrucion --------------------------------------------------------------

for index, row in df_param.iterrows():
    src = str(row['Source'])
    trg = str(row['Target'])
    roi = str(row['ROI'])
    method = str(row['Method'])
    alpha = int(row['Alpha'])
    num_samples = int(row['Number of samples'])
    print('--------------------')
    print('Source: %s' % src)
    print('Target: %s' % trg)
    print('ROI: %s' % roi)
    print('alpha: %s' % alpha)
    print('Number of samples: %s' % num_samples)
    print('Method: %s' % method)
    conversion = src+'2'+trg

    save_dir = os.path.join(save_dir_root, conversion, roi, method, str(num_samples), str(alpha))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print('%s already exists and skipped' % save_dir)
        continue

    makedir_ifnot('tmp_recon')
    recon_id = '-'.join(['shape',conversion, roi, method, str(num_samples), str(alpha)])
    dist = DistComp(lockdir='tmp_recon', comp_id=recon_id)
    if dist.islocked():
        print('%s is already running. Skipped.' % recon_id)
        continue
    dist.lock()

    for image_label in image_label_list:

        print('')
        print('Subject:     ' + conversion)
        print('ROI:         ' + roi)
        print('Image label: ' + image_label)
        print('')



        # Load the decoded CNN features
        features = {}
        for layer in layers:
            # The file full name depends on the data structure for decoded CNN features
            file_name = decode_feature_filename(decoded_features_dir, network, layer, conversion, roi, method, num_samples, alpha, image_label)
            
            feat = h5py.File(file_name)['feat'].value.transpose()[0, :]
            if 'fc' in layer:
                feat = feat.reshape(feat.size)

            # Correct the norm of the decoded CNN features
            feat_std = estimate_cnn_feat_std(feat)
            feat = (feat / feat_std) * feat_std0[layer]

            features.update({layer: feat})

        # Weight of each layer in the total loss function

        # Norm of the CNN features for each layer
        feat_norm = np.array([np.linalg.norm(features[layer]) for layer in layers], dtype='float32')

        # Use the inverse of the squared norm of the CNN features as the weight for each layer
        weights = 1. / (feat_norm ** 2)

        # Normalise the weights such that the sum of the weights = 1
        weights = weights / weights.sum()
        layer_weight = dict(zip(layers, weights))

        opts.update({'layer_weight': layer_weight})

        # Reconstruction
        snapshots_dir = os.path.join(save_dir, 'snapshots', 'image-%s' % image_label)
        recon_img, loss_list = reconstruct_image(features, net,
                                                 save_intermediate=False,
                                                 save_intermediate_path=snapshots_dir,
                                                 **opts)

        # Save the results

        # Save the raw reconstructed image
        save_name = 'recon_img' + '-' + image_label + '.mat'
        sio.savemat(os.path.join(save_dir, save_name), {'recon_img': recon_img})

        # To better display the image, clip pixels with extreme values (0.02% of
        # pixels with extreme low values and 0.02% of the pixels with extreme high
        # values). And then normalise the image by mapping the pixel value to be
        # within [0,255].
        save_name = 'recon_img_normalized' + '-' + image_label + '.jpg'
        PIL.Image.fromarray(normalise_img(clip_extreme_value(recon_img, pct=4))).save(os.path.join(save_dir, save_name))
