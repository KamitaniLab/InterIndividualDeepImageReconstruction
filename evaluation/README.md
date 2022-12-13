To obtain the normalized profile correlation by noise ceiling.  

1. Run the `calculate_fmri_profile_corr.py` to obtain the raw profile correlation.  
2. Run the `calculate_profile_noise_ceiling.py` to compute the noise ceiling for each voxel.  
3. Run the `calculate_profile_nc_threshold.py` to compute the threshold for noise ceiling.  

The normalized profile correlation can obtained by dividing the raw correlation by noise ceiling (note that voxels whose noise ceiling below 99th percentile threshold should be excluded). See `normalized_profile_correlation.ipynb` as an example.  

More to add for codes of other evaluations.