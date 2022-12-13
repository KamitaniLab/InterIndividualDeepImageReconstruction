import hdf5storage
import numpy as np
import bdpy
import os
import bdpy
import pandas as pd
from itertools import product, combinations, combinations_with_replacement
import time

def calculate_noise_ceiling_thr(data, label):
    '''
    To calculate the 99th percentile threshold for profile correlation of single trial samples.
    If a noise ceiling is below this threshold, the noise ceiling is considered caused by the random noise.
    In other words, the measurement is not realiable and the signals contain no information.
    '''
    sort_idx = np.argsort(label.flatten())
    data = data[sort_idx]
    rep = 24 # trials per image
    
    nc_thd1 = []
    nc_thd5 = []
    for vox in range(data.shape[1]):
        print('Voxel: {}'.format(vox))
        tmp = data[:,vox]
        tmp = tmp.reshape(rep,-1,order='F')
        
        samplings = []
        
        for it in range(10000):
        
            tmp_permute = np.zeros_like(tmp)
            for i in range(tmp.shape[0]):
                tmp_permute[i,:] = np.random.permutation(tmp[i,:])

            corrs = np.corrcoef(tmp, tmp_permute)[rep:,:rep]
            samplings.append((np.sum(corrs)-np.sum(np.tril(corrs)))/(rep*(rep-1)/2))
        
        samplings = np.sort(samplings)
        nc_thd1.append(samplings[-100])
        nc_thd5.append(samplings[-500])
        
    nc_thd1 = np.array(nc_thd1) # 99th percentile is used in the paper
    nc_thd5 = np.array(nc_thd5)
    
    return nc_thd1, nc_thd5


subjects = ['sub-01']

f = 'path/to/dataset in bdata format'
roi_dict = {'VC': 'ROI_VC'}

result_data = []
for sbj in subjects:
    print('Subject {}'.format(sbj))
    a = bdpy.BData(f.format(sbj))
    label = a.select('image_index')
    
    for roi, roi_str in roi_dict.items():
        dat = a.select(roi_str)
        s = time.time()
        thd1, thd5 = calculate_noise_ceiling_thr(dat,label)
        t = time.time()
        print(t-s)

        for i, corr in enumerate(thd1):
            result_data.append({'Subject': sbj, 'Threshold 1': corr, 'Threshold 5': thd5[i], 'ROI':roi, 'Vox_idx': i})
            
df = pd.DataFrame(result_data, index=None)
df.to_csv('profile_nc_threshold_single_trial.csv', index=None)