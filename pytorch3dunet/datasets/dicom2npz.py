#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:11:16 2021

@author: shkim
"""

import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
import os

def _load_files(dir):
    assert os.path.isdir(dir), 'Cannot find the dataset directory'
    # logger.info(f'Loading data from {dir}')
    reader = sitk.ImageSeriesReader()
    dicomFiles = reader.GetGDCMSeriesFileNames(dir)
    reader.SetFileNames(dicomFiles)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    img3d = sitk.GetArrayFromImage(image)
    # img3d = img3d.transpose((1,2,0))
    return img3d

def prepare_dataset(root_dir, save_dir, mode):
    patients = os.listdir(os.path.join(root_dir,mode))
    for p in patients:
        print(p)
        dicom_array = _load_files(os.path.join(root_dir,mode,p))
        mask_array = _load_files(os.path.join(root_dir,mode+'_masks',p))
        
        if dicom_array.shape[0] >= 600 and mode == 'train':
            dicom_array = dicom_array[100:-100, :, :]
            mask_array = mask_array[100:-100, :, :]
        
        dicom_array = resize(dicom_array.astype(np.float64),(296,296,296)).astype(np.float64)
        mask_array = resize(mask_array.astype(np.float32),(296,296,296)).astype(np.int8)
        
        np.savez_compressed(os.path.join(save_dir, mode, p)+'.npz', ct=dicom_array, mask=mask_array)


if __name__ == '__main__':
    root_dir = '/media/shkim/7cf97277-d9cd-454f-994f-59734f6775d0/skin_data/'
    save_dir = '/home/shkim/Libraries/pytorch-3dunet/datasets/skin/'
    
    mode = ['val']
    
    for m in mode:
        prepare_dataset(root_dir, save_dir, m)