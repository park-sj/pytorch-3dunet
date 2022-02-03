#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:53:30 2020

@author: Junwon Son

Load .dcm dataset
"""

import os
import numpy as np
import SimpleITK as sitk

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('DicomDataset')

class DicomDataset(ConfigDataset):
    def __init__(self, file_path, mask_path, phase, slice_builder_config, transformer_config, mirror_padding=(0, 32, 32)):
        """
        :param file_path: path to dicom root directory
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        """
        assert os.path.isdir(file_path), 'Incorrect dataset directory'
        assert phase in ['train', 'val', 'test']
        
        if phase in ['train', 'val']:
            mirror_padding = None
            
        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"
        
        self.mirror_padding = mirror_padding
        self.slice_builder_config = slice_builder_config
        self.phase = phase
        self.file_path = file_path
        self.mask_path = mask_path
        self.patients = os.listdir(file_path)
        self.transformer_config = transformer_config
        
        self.patch_per_image = 1
    
    def getImage(self, count):
        if count >= len(self.patients):
            raise StopIteration
        if self.phase == 'test':
            logger.info(f'Loading dcm files from {os.path.join(self.file_path, self.patients[count])}')
        self.cur_image = load_dicom_series(os.path.join(self.file_path, self.patients[count]))
        
        # stats are dummy value
        transformer = transforms.get_transformer(self.transformer_config, min_value=0, max_value=0,
                                                 mean=0, std=0)
        self.raw_transform = transformer.raw_transform()
        if self.phase != 'test':
            self.masks_transform = transformer.label_transform()       
        if self.phase != 'test':
            self.cur_mask = load_dicom_series(os.path.join(self.mask_path, self.patients[count]))
        else:
            self.cur_mask = None
        
    def __getitem__(self, idx):
        self.getImage(int(idx / self.patch_per_image))
        name = self.patients[int(idx / self.patch_per_image)]
        
        image = self.raw_transform(self.cur_image)
        
        if self.phase != 'test':
            mask = self.masks_transform(self.cur_mask)
            return image, mask, name
        else:
            return image
        
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
        # load masks to process
        mask_paths = phase_config.get('mask_paths', None)
        if mask_paths is not None:
            mask_paths = mask_paths[0]
        # mirror padding conf
        mirror_padding = dataset_config.get('mirror_padding', None)

        return [cls(file_paths[0], mask_paths, phase, slice_builder_config, transformer_config, mirror_padding)]


def load_dicom_series(dir):
    assert os.path.isdir(dir), 'Cannot find the dataset directory'
    reader = sitk.ImageSeriesReader()
    dicomFiles = reader.GetGDCMSeriesFileNames(dir)
    reader.SetFileNames(dicomFiles)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    img3d = sitk.GetArrayFromImage(image)
    return img3d