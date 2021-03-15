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
from pytorch3dunet.datasets.utils import ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('DicomDataset')

class DicomDataset(ConfigDataset):
    def __init__(self, file_path, phase, transformer_config, mirror_padding=(0, 32, 32)):
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
        self.phase = phase
        
        self.images = self._load_files(os.path.join(file_path, phase))
        
        min_value, max_value, mean, std = calculate_stats(self.images[0])
                
        transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                 mean=mean, std=std)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()
        
        if phase != 'test':
            # load labeled images
            masks_dir = os.path.join(file_path, phase + '_masks')
            assert os.path.isdir(masks_dir)
            self.masks = self._load_files(masks_dir)
            assert len(self.images) == len(self.masks)
            # load label images transformer
            self.masks_transform = transformer.label_transform()
        else:
            self.masks = None
            self.masks_transform = None

            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                padded_imgs = []
                for img in self.images:
                    padded_img = np.pad(img, pad_width=pad_width, mode='reflect')
                    padded_imgs.append(padded_img)

                self.images = padded_imgs
        
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        
        image = self.images[idx]
        if self.phase != 'test':
            mask = self.masks[idx]
            return self.raw_transform(image), self.masks_transform(mask)
        else:
            return self.raw_transform(image)
        
    def __len__(self):
        return len(self.images)
    
    
    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load files to process
        file_paths = phase_config['file_paths']
        # mirror padding conf
        mirror_padding = dataset_config.get('mirror_padding', None)

        return [cls(file_paths[0], phase, transformer_config, mirror_padding)]
    
    @staticmethod
    def _load_files(dir):
        """ dir is a root directory that contains folders account for each patient
            files_data is a list of 3d dicom data """
        files_data = []
        for patient in os.listdir(dir):
            filepath = os.path.join(dir, patient)
            
            filepath += '/*'
            # if filepath[-3:-1] != 'dcm':
            #     if filepath[-1] == '/':
            #         filepath += '*.dcm'
            #     else:
            #         filepath += '/*.dcm'
            
            # load the DICOM files
            files = []
            for fname in glob.glob(filepath, recursive=False):
        #        print("loading: {}".format(fname))
                files.append(pydicom.dcmread(fname))
            
            logger.info(f"file path : {filepath}, file count: {len(files)}")
            
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

            files_data.append(img3d)

        return files_data
