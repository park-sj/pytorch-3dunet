#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:11:16 2021

@author: shkim

dicom 영상을 patch단위로 훈련돌리고 싶을 때 data를 준비하기 위한 코드
결과는 npz 파일 형식으로 저장함
"""

import os
import numpy as np
from random import randint
from skimage.transform import resize

from pytorch3dunet.datasets.dicom import load_dicom_series as _load_files

''' Configuration '''
PATCH_PER_CT = 64
PATCH_SIZE = (128,128,128)


def prepare_dataset(root_dir, save_dir, mode):
    patients = os.listdir(os.path.join(root_dir,mode))
    for p in patients:
        print(p)
        dicom_array = _load_files(os.path.join(root_dir,mode,p))
        mask_array = _load_files(os.path.join(root_dir,mode+'_masks',p))
        
        if dicom_array.shape[0] >= 600:
            dicom_array = dicom_array[100:-100, :, :]
            mask_array = mask_array[100:-100, :, :]
        
        # dicom_array = resize(dicom_array.astype(np.float64),(296,296,296)).astype(np.float64)
        # mask_array = resize(mask_array.astype(np.float32),(296,296,296)).astype(np.int8)
        
        array_shape = dicom_array.shape
        for i in range(PATCH_PER_CT):
            
            flag_patch = True
            while flag_patch:
                x = randint(0, array_shape[0]-PATCH_SIZE[0])
                y = randint(0, array_shape[1]-PATCH_SIZE[1])
                z = randint(0, array_shape[2]-PATCH_SIZE[2])
                
                dicom_patch = dicom_array[x:x+PATCH_SIZE[0],y:y+PATCH_SIZE[1],z:z+PATCH_SIZE[2]]
                mask_patch = mask_array[x:x+PATCH_SIZE[0],y:y+PATCH_SIZE[1],z:z+PATCH_SIZE[2]]
            
                if np.count_nonzero(mask_patch) > (mask_patch.size * 0.01):
                    flag_patch = False
        
            np.savez_compressed(os.path.join(save_dir, mode, p)+'_'+str(i).zfill(3)+'.npz', ct=dicom_patch, mask=mask_patch)


if __name__ == '__main__':
    root_dir = '/media/shkim/7cf97277-d9cd-454f-994f-59734f6775d0/skin_data/'
    save_dir = '/home/shkim/Libraries/pytorch-3dunet/datasets/skin/'
    
    mode = ['train', 'val']
    
    for m in mode:
        prepare_dataset(root_dir, save_dir, m)