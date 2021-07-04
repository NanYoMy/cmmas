# Cross-Modality Multi-Atlas Segmentation UsingDeep Neural Networks
This repository contains the original source code for the paper "Cross-Modality Multi-Atlas Segmentation UsingDeep Neural Networks" in MICCAI2020, which proposes multi-atlas segmentation framework based on deep neural network from cross-modality medical images.
At present, we have improve the MAS method for a journal paper, we recommend to use the new version codes. 
## Overview
This repository contains script to train and test cross moldaity-MMAS method, one can check these script to run the method
- [cross_modality_registration_network](./slurm/run_train_test.slurm): Performs the registration of cross-modality medical images.
- [patch_based_network](./slurm/run_patch_embedding.slurm): Performs the patch-based label fusion method

## Reference
`Ding, Wangbin, et al. "Cross-modality multi-atlas segmentation using deep neural networks." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020.`

## Acknowledgement
This project is largely based on the "labreg", "simpleitk", "antspy" repositories.

