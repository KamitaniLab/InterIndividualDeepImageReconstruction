'''Script for Deep Image Reconstruction for natural images
Modified from: https://github.com/KamitaniLab/DeepImageReconstruction

This is mainly used for the reconstruction of natural images.

- Reconstruction algorithm: Deep generator network + Gradient descent
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

from icnn.icnn_dgn_gd import reconstruct_image
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
                    'Img0040',
                    'Img0041',
                    'Img0042',
                    'Img0043',
                    'Img0044',
                    'Img0045',
                    'Img0046',
                    'Img0047',
                    'Img0048',
                    'Img0049',
                    'Img0050']

n_iteration = 200

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

# Generator network
model_file = './net/generator_for_inverting_fc7/generator.caffemodel'
prototxt_file = './net/generator_for_inverting_fc7/generator.prototxt'
net_gen = caffe.Net(prototxt_file, model_file, caffe.TEST)
input_layer_gen = 'feat'        # Input layer for generator net
output_layer_gen = 'generated'  # Output layer for generator net

# Feature size for input layer of the generator net
feat_size_gen = net_gen.blobs[input_layer_gen].data.shape[1:]
num_of_unit = net_gen.blobs[input_layer_gen].data[0].size

# Upper bound for input layer of the generator net
bound_file = './data/act_range/3x/fc7.txt'
upper_bound = np.loadtxt(bound_file, delimiter=' ', usecols=np.arange(0, num_of_unit), unpack=True)
upper_bound = upper_bound.reshape(feat_size_gen)

# Initial features for the input layer of the generator (we use a 0 vector as initial features)
initial_gen_feat = np.zeros_like(net_gen.blobs[input_layer_gen].data[0])

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

    # The total number of iterations for gradient descend
    'iter_n': n_iteration,

    # Learning rate
    'lr_start': 2.,
    'lr_end': 1e-10,

    # Gradient with momentum
    'momentum_start': 0.9,
    'momentum_end': 0.9,

    # Decay for the features of the input layer of the generator after each
    # iteration
    'decay_start': 0.01,
    'decay_end': 0.01,

    # Name of the input layer of the generator (str)
    'input_layer_gen': input_layer_gen,

    # Name of the output layer of the generator (str)
    'output_layer_gen': output_layer_gen,

    # Upper and lower boundary for the input layer of the generator
    'feat_upper_bound': upper_bound,
    'feat_lower_bound': 0.,

    # The initial features of the input layer of the generator (setting to
    # None will use random noise as initial features)
    'initial_gen_feat': initial_gen_feat,

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

    # Display the information on the terminal for every n iterations
    'disp_every': 1
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
    recon_id = '-'.join([conversion, roi, method, str(num_samples), str(alpha)])
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
        norm_and_clip = lambda x: normalise_img(clip_extreme_value(x, pct=4))
        recon_img, loss_list = reconstruct_image(features, net, net_gen,
                                                 save_intermediate=False,
                                                 save_intermediate_path=snapshots_dir,
                                                 save_intermediate_postprocess=norm_and_clip,
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
    dist.unlock()