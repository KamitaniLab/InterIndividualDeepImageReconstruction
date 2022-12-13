import hdf5storage
import numpy as np
import bdpy
import os
import bdpy
import pandas as pd
from itertools import product, combinations, combinations_with_replacement

def calculate_profile_noise_ceiling(data, label):
    '''
    The noise ceiling is simply the average of the cross-correlations between 
    fMRI responses to an identical stimulus.
    '''

    sort_idx = np.argsort(label.flatten())
    data = data[sort_idx]
    rep = 24 # trials per image
    
    nc = []
    for vox in range(data.shape[1]):
        tmp = data[:,vox]
        tmp = tmp.reshape(rep,-1,order='F')
        
        corrs = np.corrcoef(tmp)
        nc.append((np.sum(corrs)-rep)/(rep*(rep-1)))
            
    nc = np.array(nc)

    # correlation below zero is set to be zero noise ceiling.
    nc[np.where(nc < 0)] = 0

    return nc

subjects = ['sub-01']

f = 'path/to/dataset in bdata format'
roi_dict = {'VC': 'ROI_VC'}

result_data = []
for sbj in subjects:
    a = bdpy.BData(f.format(sbj))
    label = a.select('image_index')
    for roi, roi_str in roi_dict.items():
        dat = a.select(roi_str)
        nc = calculate_profile_noise_ceiling(dat,label)

        for i, corr in enumerate(nc):
            result_data.append({'Subject': sbj, 'Correlation': corr, 'ROI':roi, 'Vox_idx': i})

df = pd.DataFrame(result_data, index=None)
df.to_csv('profile_noise_ceiling_single_trial.csv', index=None)