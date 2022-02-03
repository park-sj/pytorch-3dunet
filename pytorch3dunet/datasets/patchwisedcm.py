#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:12:17 2021

@author: shkim

While dicom.py doesn't crop or divide images into patches, skin.py does so.

Use NpzDataset in npz.py for training and SkinDcmDataset for prediction.
"""

import os
import numpy as np
import math
import SimpleITK as sitk

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.datasets.dicom import load_dicom_series

logger = get_logger('PatchwiseDcmDataset')

class PatchwiseDcmDataset(ConfigDataset):
    def __init__(self, file_path, phase, slice_builder_config, transformer_config, mirror_padding=(0, 32, 32)):
        """
        :param file_path: path to dicom root directory
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        """
        assert os.path.isdir(file_path), 'Incorrect dataset directory'
        assert phase in ['test'], 'Use NpzDataset for training and validating'
        

        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"
        
        self.mirror_padding = mirror_padding
        self.slice_builder_config = slice_builder_config
        self.phase = phase
        self.file_path = file_path
        self.patients = os.listdir(os.path.join(file_path, phase))
        self.transformer_config = transformer_config

        self.weight_maps = None
        
        self.count = -1
        self.getImage(0)
    
    def getImage(self, count):
        if count >= len(self.patients):
            raise StopIteration
        if self.phase == 'test':
            logger.info(f'Loading dcm files from {os.path.join(self.file_path, self.patients[count])}')
        self.cur_image = load_dicom_series(os.path.join(self.file_path, self.patients[count]))

        ''' padding '''
        patch_shape = self.slice_builder_config['patch_shape']
        stride_shape = self.slice_builder_config['stride_shape']
        input_shape = self.cur_image.shape
        self.input_shape = input_shape
        target_size = list(input_shape)
        for i in range(len(input_shape)):
            if input_shape[i] < patch_shape[i]:
                target_size[i] = patch_shape[i]
            else:
                n = math.ceil((input_shape[i]-patch_shape[i])/stride_shape[i])
                target_size[i] = patch_shape[i] + n*stride_shape[i]
        padding_shape = ((math.ceil((target_size[0]-input_shape[0])/2), (math.floor((target_size[0]-input_shape[0])/2))),
                         (math.ceil((target_size[1]-input_shape[1])/2), (math.floor((target_size[1]-input_shape[1])/2))),
                         (math.ceil((target_size[2]-input_shape[2])/2), (math.floor((target_size[2]-input_shape[2])/2))))
        self.target_size = target_size
        self.cur_image = np.pad(self.cur_image, padding_shape, mode="constant", constant_values=self.transformer_config['raw']['Normalize']['min_value'])
        self.cur_image = np.expand_dims(self.cur_image, 0)  
        
        slice_builder = get_slice_builder(self.cur_image, None, self.weight_maps, self.slice_builder_config)
        
        # stats are dummy value
        transformer = transforms.get_transformer(self.transformer_config, min_value=0, max_value=0,
                                                 mean=0, std=0)
        self.raw_transform = transformer.raw_transform()
              
        self.image_slices = slice_builder.raw_slices
        self.patch_per_image = len(self.image_slices)

        
    def __getitem__(self, idx):
        idx = idx % self.patch_per_image
        image = self.image_slices[idx]
        
        image = self._transform_patches(self.cur_image, image, self.raw_transform)
        
        raw_idx = self.image_slices[idx]
        if len(raw_idx) == 4:
            raw_idx = raw_idx[1:]
        return image, raw_idx
        
    def __len__(self):
        return len(self.patients) * self.patch_per_image
    
    @staticmethod
    def _transform_patches(datasets, label_idx, transformer):
        transformed_patches = []
        for dataset in datasets:
            # get the label data and apply the label transformer
            transformed_patch = transformer(dataset[label_idx])
            transformed_patches.append(transformed_patch)

        # if transformed_patches is a singleton list return the first element only
        if len(transformed_patches) == 1:
            return transformed_patches[0]
        else:
            return transformed_patches
        
    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # mirror padding conf
        mirror_padding = dataset_config.get('mirror_padding', None)

        return [cls(file_paths[0], phase, slice_builder_config, transformer_config, mirror_padding)]
    