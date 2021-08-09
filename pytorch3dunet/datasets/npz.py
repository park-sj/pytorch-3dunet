#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 8 2021

@author: Junwon Son

Load .npz dataset

뼈, 비강 분할 모델 만들 때, CT 풀 이미지를 모두 인풋으로 받는게 아니라
ROI 일부만 잘라서 인풋으로 받도록 만들 생각이다. 잘라낸 인풋은 numpy
compressed 파일인 npz 형태로 저장해서 사용한다. 훈련 데이터는 한 개의
npz 파일 안에 ct, mask를 모두 포함하고 있다. numpy.load 함수로 data를
load 한 후 dict 방식으로 불러오면 된다.
"""

import os
import numpy as np
import random
from skimage.transform import resize

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('NpzDataset')

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
        self.transformer_config = transformer_config
        
        
        # x = [self._load_files(os.path.join(self.file_path, self.phase, y)) for y in self.patients]
        # image_sample = self._load_files('/home/shkim/Libraries/pytorch-3dunet/datasets/JW/train/ParkHyunA/')
#        test_patient = os.listdir(os.path.join(file_path, 'test'))
        image_sample = np.zeros((296,296,296))
        # min_value, max_value, mean, std = calculate_stats(image_sample.astype(np.float64))
        # transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
        #                                          mean=mean, std=std)
        # self.raw_transform = transformer.raw_transform()
        # if phase != 'test':
        #     self.masks_transform = transformer.label_transform()
        
        # slice
        self.weight_maps = None
        # image_sample = image_sample[:400, 100:500, 100:500]
        # image_sample = image_sample[20:580, 20:580, 20:580]
        # image_sample = image_sample[4:596, 4:596, 4:596]
        # image_sample = downscale_local_mean(image_sample, (2,2,2))
        # image_sample = np.expand_dims(image_sample, 0)
        
        image_sample = resize(image_sample.astype(np.float32), (296,296,296), anti_aliasing = False)
        image_sample = np.expand_dims(image_sample, 0)

        slice_builder = get_slice_builder(image_sample, None, self.weight_maps, self.slice_builder_config)
        image_slices = slice_builder.raw_slices
        self.patch_per_image = len(image_slices)
        
        self.count = -1
    
    def getImage(self, count):
        if count >= len(self.patients):
            raise StopIteration
        if self.phase == 'test':
            logger.info(f'Loading dcm files from {os.path.join(self.file_path, self.phase, self.patients[count])}')
        data = np.load(os.path.join(self.file_path, self.phase, self.patients[count]))
        self.cur_image = data['ct']
        
        # min_value, max_value, mean, std = calculate_stats(self.cur_image.astype(np.float32))
        self.min_value = -750
        self.max_value = 1250
        self.cur_image[self.cur_image>self.max_value] = self.max_value
        self.cur_image[self.cur_image<self.min_value] = self.min_value
        self.cur_image = resize(self.cur_image.astype(np.float32), (296, 296, 296), anti_aliasing = False)
        mean = (self.min_value + self.max_value)/2
        std = (self.max_value - self.min_value)/2
        transformer = transforms.get_transformer(self.transformer_config, min_value=self.min_value, max_value=self.max_value,
                                                 mean=mean, std=std)
        self.raw_transform = transformer.raw_transform()
        if self.phase != 'test':
            self.masks_transform = transformer.label_transform()
        self.cur_image = np.expand_dims(self.cur_image, 0)        
        if self.phase != 'test':
            self.cur_mask = data['mask']
            if random.getrandbits(1):
                self.cur_image.flip(axis=2)
                self.cur_mask.flip(axis=2)
#            if self.cur_mask.shape[0] >= 600:
#                self.cur_mask = self.cur_mask[100:-100, :, :]
            self.cur_mask = resize(self.cur_mask.astype(np.float32), (296, 296, 296), anti_aliasing = False)
            self.cur_mask = np.expand_dims(self.cur_mask, 0)
        else:
            self.cur_mask = None
        slice_builder = get_slice_builder(self.cur_image, self.cur_mask, self.weight_maps, self.slice_builder_config)
        self.image_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices

        
    def __getitem__(self, idx):
        # self.count += 1
        # idx = self.count
        # if idx % self.patch_per_image == 0:
        #     self.getImage(int(idx / self.patch_per_image))
        
        # self.getImage(int(idx % len(self.patients)))
        # logger.info(f'getting item number {idx}')
        self.getImage(int(idx / self.patch_per_image))
        name = self.patients[int(idx / self.patch_per_image)]
        
        idx = idx % self.patch_per_image
        image = self.image_slices[idx]
        
        # image = np.expand_dims(image, 0)
        
        image = self._transform_patches(self.cur_image, image, self.raw_transform)
        
        if self.phase != 'test':
            mask = self.label_slices[idx]
            # mask = np.expand_dims(mask, 0)
            mask = self._transform_patches(self.cur_mask, mask, self.masks_transform)
            if self.phase == 'train':
                image += np.random.normal(0,0.2, image.shape).astype(np.float32) # Gaussian noise
                # noise = generate_perlin_noise_3d((296,296,296), (8,8,8))*0.5
                # noise[noise<0] = 0
                # image += noise # Perlin noise
            return image, mask, name
        else:

            # image = np.expand_dims(image, 0)
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
    