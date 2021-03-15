#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:53:30 2020

@author: Junwon Son

Load .dcm dataset
"""

import os
import numpy as np
import glob
import pydicom

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('DicomDataset')

class DicomDataset(ConfigDataset):
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
        
        image_sample = self._load_files('/home/shkim/Libraries/pytorch-3dunet/datasets/JW/train/ParkHyunA/')
        min_value, max_value, mean, std = calculate_stats(image_sample)
        transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                 mean=mean, std=std)
        self.raw_transform = transformer.raw_transform()
        
        if phase != 'test':
            self.masks_transform = transformer.label_transform()
        
        # slice
        self.weight_maps = None
        image_sample = image_sample[:400, 100:500, 100:400]        
        image_sample = np.expand_dims(image_sample, 0)

        slice_builder = get_slice_builder(image_sample, None, self.weight_maps, self.slice_builder_config)
        image_slices = slice_builder.raw_slices
        self.patch_per_image = len(image_slices)
        
        self.count = -1
    
    def getImage(self, count):
        if count >= len(self.patients):
            raise StopIteration
        self.cur_image = self._load_files(os.path.join(self.file_path, self.phase, self.patients[count]))
        self.cur_image = self.cur_image[:400, 100:500, 100:400]
        if self.phase != 'test':
            self.cur_mask = self._load_files(os.path.join(self.file_path, self.phase + '_masks', self.patients[count]))
            self.cur_mask = np.flip(self.cur_mask, 2)
            self.cur_mask = self.cur_mask[:400, 100:500, 100:400]
        else:
            self.cur_mask = None
        self.cur_image = np.expand_dims(self.cur_image, 0)
        self.cur_mask = np.expand_dims(self.cur_mask, 0)
        slice_builder = get_slice_builder(self.cur_image, self.cur_mask, self.weight_maps, self.slice_builder_config)
        self.image_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        
    def __getitem__(self, idx):
        self.count += 1
        idx = self.count
        if idx % self.patch_per_image == 0:
            self.getImage(int(idx / self.patch_per_image))
            
        idx = idx % self.patch_per_image
        image = self.image_slices[idx]
        # print(image.dtype)
        # image = np.expand_dims(image, 0)
        # print(self.cur_image.dtype)
        # print(image.dtype)
        image = self._transform_patches(self.cur_image, image, self.raw_transform)
        
        if self.phase != 'test':
            mask = self.label_slices[idx]
            # mask = np.expand_dims(mask, 0)
            mask = self._transform_patches(self.cur_mask, mask, self.masks_transform)
            return image, mask
        else:
            # image = np.expand_dims(image, 0)
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
        # mirror padding conf
        mirror_padding = dataset_config.get('mirror_padding', None)

        return [cls(file_paths[0], phase, slice_builder_config, transformer_config, mirror_padding)]
    
    @staticmethod
    def _load_files(dir):
        """ dir is a root directory that contains folders account for each patient
            files_data is a list of 3d dicom data """

        dir = os.path.join(dir, '*')
        
        # load the DICOM files
        files = []
        for fname in glob.glob(dir, recursive=False):
    #        print("loading: {}".format(fname))
            files.append(pydicom.dcmread(fname))
        
        # logger.info(f"file path : {dir}, file count: {len(files)}")
        
        # skip files with no SliceLocation (eg scout views)
        slices = []
        skipcount = 0
        for f in files:
            if hasattr(f, 'InstanceNumber'):
                slices.append(f)
            else:
                skipcount = skipcount + 1
        
        # ensure they are in the correct order
    #    slices = sorted(slices, key=lambda s: s.InstanceNumber)
        slices, files = zip(*sorted(zip(slices,files), key = lambda s: s[0].InstanceNumber))
    #    [file for _, file in sorted(zip(slices,files), key = lambda s: s[0].InstanceNumber)]
    #    sorted(files, key=lambda s: slices.InstanceNumber)
        
        # create 3D array
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        img3d = np.zeros(img_shape, dtype = np.int16)
        
        # fill 3D array with the images from the files
        for i, s in enumerate(slices):
            img2d = s.pixel_array.astype(dtype = np.int16)
            img3d[:, :, i] = img2d

        return img3d
