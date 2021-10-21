#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2021.10.18

@author: Junwon Son

This dataset is for predicting airway or bone
Use npz dataset for training
"""

import os
import numpy as np
import glob
import SimpleITK as sitk

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('DicomDataset')

class ABDataset(ConfigDataset):
    def __init__(self, file_path, phase, slice_builder_config, transformer_config, mirror_padding=(0, 32, 32)):
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
        self.patients = os.listdir(os.path.join(file_path, phase))
        self.transformer_config = transformer_config

    
    def getImage(self, count):
        if count >= len(self.patients):
            raise StopIteration
        if self.phase == 'test':
            logger.info(f'Loading dcm files from {os.path.join(self.file_path, self.phase, self.patients[count])}')
        self.cur_image = self._load_files(os.path.join(self.file_path, self.phase, self.patients[count]))
        
        self.slice_builder_config['patch_shape'] = list(self.cur_image.shape)
        slice_builder = get_slice_builder(np.expand_dims(self.cur_image, 0), None, None, self.slice_builder_config)
        self.image_slices = slice_builder.raw_slices
        
        # min_value, max_value, mean, std = calculate_stats(self.cur_image.astype(np.float32))
        self.min_value = -500
        self.max_value = 2000
        self.cur_image[self.cur_image>self.max_value] = self.max_value
        self.cur_image[self.cur_image<self.min_value] = self.min_value
        mean = (self.min_value + self.max_value)/2
        std = (self.max_value - self.min_value)/2
        transformer = transforms.get_transformer(self.transformer_config, min_value=self.min_value, max_value=self.max_value,
                                                 mean=mean, std=std)
        self.raw_transform = transformer.raw_transform()
        if self.phase != 'test':
            self.masks_transform = transformer.label_transform()
        self.cur_image = np.expand_dims(self.cur_image, 0)        
        if self.phase != 'test':
            self.cur_mask = self._load_files(os.path.join(self.file_path, self.phase + '_masks', self.patients[count]))
            self.cur_mask = np.expand_dims(self.cur_mask, 0)
        else:
            self.cur_mask = None
        
    def __getitem__(self, idx):
        self.getImage(idx) 
        image = self.image_slices[idx]
        image = self._transform_patches(self.cur_image, image, self.raw_transform)

        
        if self.phase != 'test':
            mask = self.masks_transform(self.cur_mask)
            return image, mask
        else:
            raw_idx = self.image_slices[idx]
            if len(raw_idx) == 4:
                raw_idx = raw_idx[1:]
            return image, raw_idx
        
    def __len__(self):
        return len(self.patients)

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
    
    @staticmethod
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
