# Inter-individual deep image reconstruction
Demo code for Ho, Horikawa, Majima, and Kamitani (2022), [Inter-individual deep image reconstruction](https://www.biorxiv.org/content/10.1101/2021.12.31.474501v1). 

## Requirements
- Python 2 or 3 (Python 2 is required for image reconstruction)
- Numpy
- Scipy
- Pandas
- bdpy: https://github.com/KamitaniLab/bdpy
- FastL2LiR: https://github.com/KamitaniLab/PyFastL2LiR


## Usage

### Training for neural code converter
Run the `ncc_training.py` to train the neural code converters for a pair of source and target subjects.

### DNN feature decoding from converted brain activities
Run the  `ncc_predict_feat.py` to predict the DNN features from the converted brain activities with the trained neural code converters. Pre-trained DNN feature decoders of the target subjects are necessary to run this script. We used the same methodology in the previous study for DNN feature decoding ([Horikawa & Kamitani, 2017, Generic decoding of seen and imagined objects using hierarchical visual features, Nat Commun.](https://www.nature.com/articles/ncomms15037)). Python code for the DNN feature decoding is available at https://github.com/KamitaniLab/dnn-feature-decoding.

### Image reconstruction from decoded CNN features
We used the same methodology in the previous study for image reconstruction ([Deep image reconstruction from human brain activity](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633)). Please follow its instruction to setup the environment.

Run the `image_reconstruction/recon_image_naturalImage_VGG19_DGN_GD.py` to reconstruct the natural images shown in the original paper.  
Run the `image_reconstruction/recon_image_artificialImage_VGG19_NoDGN_LBFGS.py` to reconstruct the artificial images shown in the original paper.

## Data
The data in h5 format could be downloaded from:  

https://figshare.com/articles/dataset/Inter-individual_deep_image_reconstruction/17985578

The Raw fMRI data could be downloaded from:   

https://openneuro.org/datasets/ds001506/versions/1.3.1  
https://openneuro.org/datasets/ds003430/versions/1.1.1  
https://openneuro.org/datasets/ds003993

## References
- Yamada K, Miyawaki Y, Kamitani Y. Inter-subject neural code converter for visual image representation. NeuroImage. 2015; 113: 289–297. https://doi.org/10.1016/j.neuroimage.2015.03.059 
- Haxby JV, Guntupalli JS, Connolly AC, Halchenko YO, Conroy BR, Gobbini MI, et al. A Common, High-Dimensional Model of the Representational Space in Human Ventral Temporal Cortex. Neuron. 2011; 72: 404–416. https://doi.org/10.1016/j.neuron.2011.08.026 
- Horikawa and Kamitani (2017) Generic decoding of seen and imagined objects using hierarchical visual features. Nature Communications 8:15037. https://www.nature.com/articles/ncomms15037
- Shen, Horikawa, Majima, and Kamitani (2019) Deep image reconstruction from human brain activity. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1006633